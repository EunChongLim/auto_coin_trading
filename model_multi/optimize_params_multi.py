"""
멀티 타임프레임 파라미터 최적화
Walk-Forward 방식으로 최적 파라미터 탐색
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from itertools import product
import pickle

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_multi.Common.data_loader import load_multi_timeframe_data
from model_multi.Common.features_multi import calculate_multi_timeframe_features


def backtest_with_params(df_1m, df_1h, df_4h, model_path, buy_threshold, stop_loss, take_profit, time_limit):
    """특정 파라미터로 백테스트"""
    
    # 모델 로드
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    window_size_1m = model_data['window_size_1m']
    window_size_1h = model_data['window_size_1h']
    window_size_4h = model_data['window_size_4h']
    
    balance = 1000000
    holding = False
    buy_price = 0
    buy_time_idx = 0
    buy_amount = 0
    
    trades = []
    fee_rate = 0.0005
    
    start_idx = max(window_size_1m, window_size_1h * 60, window_size_4h * 240)
    
    for i in range(start_idx, len(df_1m)):
        current_time = df_1m.iloc[i]['datetime']
        current_price = df_1m.iloc[i]['close']
        
        # 보유 중 매도 확인
        if holding:
            profit_pct = (current_price - buy_price) / buy_price * 100
            hold_minutes = i - buy_time_idx
            
            sell = False
            reason = ""
            
            if profit_pct >= take_profit:
                sell = True
                reason = "익절"
            elif profit_pct <= -stop_loss:
                sell = True
                reason = "손절"
            elif hold_minutes >= time_limit:
                sell = True
                reason = "시간초과"
            
            if sell:
                sell_price = current_price * (1 - fee_rate)
                balance = buy_amount * sell_price
                
                trades.append({
                    'profit': profit_pct,
                    'reason': reason
                })
                
                holding = False
                buy_price = 0
                buy_time_idx = 0
                buy_amount = 0
        
        # 미보유 시 매수 확인
        else:
            window_1m = df_1m.iloc[i - window_size_1m:i]
            window_1h = df_1h[df_1h['datetime'] <= current_time].tail(window_size_1h)
            window_4h = df_4h[df_4h['datetime'] <= current_time].tail(window_size_4h)
            
            if len(window_1m) < window_size_1m or len(window_1h) < 30 or len(window_4h) < 30:
                continue
            
            features = calculate_multi_timeframe_features(window_1m, window_1h, window_4h)
            
            if features is None:
                continue
            
            X = pd.DataFrame([features])[feature_cols]
            X = X.fillna(0)
            
            predicted_profit = model.predict(X, num_iteration=model.best_iteration)[0]
            
            if predicted_profit >= buy_threshold:
                if features.get('1m_rsi', 50) > 75:
                    continue
                
                buy_price = current_price * (1 + fee_rate)
                buy_amount = balance / buy_price
                balance = 0
                holding = True
                buy_time_idx = i
    
    # 최종 청산
    if holding:
        final_price = df_1m.iloc[-1]['close']
        sell_price = final_price * (1 - fee_rate)
        balance = buy_amount * sell_price
        profit_pct = (final_price - buy_price) / buy_price * 100
        
        trades.append({
            'profit': profit_pct,
            'reason': '강제청산'
        })
    
    # 결과
    total_return = (balance - 1000000) / 1000000 * 100
    
    if len(trades) == 0:
        return {
            'total_return': total_return,
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0
        }
    
    wins = [t for t in trades if t['profit'] > 0]
    win_rate = len(wins) / len(trades) * 100
    avg_profit = np.mean([t['profit'] for t in trades])
    
    return {
        'total_return': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_profit': avg_profit
    }


def optimize_walk_forward():
    """Walk-Forward 방식 파라미터 최적화"""
    
    print("="*80)
    print("멀티 타임프레임 파라미터 최적화")
    print("="*80)
    
    # 2024년을 3개 구간으로 분할
    periods = [
        ("20240101", "20240331", "20240401", "20240430"),  # Q1 → 4월
        ("20240401", "20240630", "20240701", "20240731"),  # Q1+Q2 → 7월
        ("20240701", "20240930", "20241001", "20241031"),  # Q3 → 10월
    ]
    
    # 파라미터 그리드
    param_grid = {
        'buy_threshold': [0.05, 0.1, 0.15, 0.2],
        'stop_loss': [0.2, 0.3, 0.4, 0.5],
        'take_profit': [0.4, 0.6, 0.8, 1.0],
        'time_limit': [5, 8, 10, 15]
    }
    
    model_path = "model/lgb_multi_timeframe.pkl"
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 모델 파일 없음: {model_path}")
        print("먼저 train_multi.py를 실행하세요!")
        return
    
    all_results = []
    
    # 각 기간마다 테스트
    for period_idx, (train_start, train_end, valid_start, valid_end) in enumerate(periods, 1):
        print(f"\n{'='*80}")
        print(f"Period {period_idx}: 검증 {valid_start}~{valid_end}")
        print(f"{'='*80}")
        
        # 검증 데이터 로드 (각 타임프레임을 API에서 직접)
        df_1m, df_1h, df_4h = load_multi_timeframe_data(valid_start, valid_end)
        
        if df_1m is None or len(df_1m) == 0:
            print("[WARNING] 검증 데이터 없음, 스킵")
            continue
        
        # 모든 파라미터 조합 테스트
        param_combinations = list(product(
            param_grid['buy_threshold'],
            param_grid['stop_loss'],
            param_grid['take_profit'],
            param_grid['time_limit']
        ))
        
        total_combos = len(param_combinations)
        print(f"\n총 {total_combos}개 조합 테스트...\n")
        
        period_results = []
        
        for idx, (buy_th, stop, take, time) in enumerate(param_combinations, 1):
            result = backtest_with_params(
                df_1m, df_1h, df_4h, model_path,
                buy_threshold=buy_th,
                stop_loss=stop,
                take_profit=take,
                time_limit=time
            )
            
            result['params'] = {
                'buy_threshold': buy_th,
                'stop_loss': stop,
                'take_profit': take,
                'time_limit': time
            }
            result['period'] = period_idx
            
            period_results.append(result)
            
            if idx % 50 == 0:
                print(f"  진행: {idx}/{total_combos} ({idx/total_combos*100:.1f}%)")
        
        all_results.extend(period_results)
        
        # 해당 기간 베스트
        best = max(period_results, key=lambda x: x['total_return'])
        print(f"\n[Period {period_idx} Best]")
        print(f"  파라미터: {best['params']}")
        print(f"  수익률: {best['total_return']:+.2f}%")
        print(f"  거래: {best['num_trades']}회")
        print(f"  승률: {best['win_rate']:.1f}%")
    
    # 전체 결과 분석
    print(f"\n{'='*80}")
    print("전체 기간 최적 파라미터 분석")
    print(f"{'='*80}\n")
    
    # 파라미터별 평균 성과
    param_performance = {}
    
    for result in all_results:
        param_key = tuple(result['params'].items())
        
        if param_key not in param_performance:
            param_performance[param_key] = []
        
        param_performance[param_key].append(result['total_return'])
    
    # 평균 수익률 기준 정렬
    param_avg = {
        k: {
            'avg_return': np.mean(v),
            'std_return': np.std(v),
            'min_return': np.min(v),
            'max_return': np.max(v),
            'periods': len(v)
        }
        for k, v in param_performance.items()
    }
    
    sorted_params = sorted(param_avg.items(), key=lambda x: x[1]['avg_return'], reverse=True)
    
    print("Top 10 최적 파라미터:\n")
    for rank, (param_tuple, perf) in enumerate(sorted_params[:10], 1):
        params = dict(param_tuple)
        print(f"{rank}. {params}")
        print(f"   평균: {perf['avg_return']:+.2f}% | 표준편차: {perf['std_return']:.2f}% | "
              f"최소: {perf['min_return']:+.2f}% | 최대: {perf['max_return']:+.2f}%")
        print()
    
    # 최종 추천
    best_param_tuple = sorted_params[0][0]
    best_params = dict(best_param_tuple)
    best_perf = sorted_params[0][1]
    
    print(f"{'='*80}")
    print("최종 추천 파라미터")
    print(f"{'='*80}")
    print(f"\nbuy_threshold = {best_params['buy_threshold']}")
    print(f"stop_loss = {best_params['stop_loss']}")
    print(f"take_profit = {best_params['take_profit']}")
    print(f"time_limit = {best_params['time_limit']}")
    print(f"\n평균 수익률: {best_perf['avg_return']:+.2f}%")
    print(f"수익률 범위: {best_perf['min_return']:+.2f}% ~ {best_perf['max_return']:+.2f}%")
    print(f"표준편차: {best_perf['std_return']:.2f}%")
    print(f"검증 기간: {best_perf['periods']}개")
    
    # 결과 저장
    result_data = {
        'best_params': best_params,
        'best_performance': best_perf,
        'all_results': all_results,
        'top_10': [(dict(p), perf) for p, perf in sorted_params[:10]]
    }
    
    with open('optimization_result_multi.pkl', 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\n[저장] optimization_result_multi.pkl")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    optimize_walk_forward()

