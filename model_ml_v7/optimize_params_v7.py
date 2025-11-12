"""
v7 모델 파라미터 최적화 (Walk-Forward 방식)

원칙:
1. 2024년 데이터를 여러 구간으로 분할
2. 각 구간마다 walk-forward 방식으로 최적화
3. 과적합 방지를 위해 검증 구간에서 평가
4. 최종적으로 가장 안정적인 파라미터 선택
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import pickle

# v7 모듈 임포트
from model_ml_v7.Common.tick_loader import download_tick_data
from model_ml_v7.Common.tick_aggregator import aggregate_ticks_to_minute
from model_ml_v7.Common.features_v7 import calculate_combined_features


def load_combined_data(start_date, end_date):
    """1분봉 + 틱 데이터 로드 및 병합"""
    from Common.auto_download_data import download_1m_data
    
    # 1분봉 데이터
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_candles = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        candles = download_1m_data(date_str, output_dir="../data/daily_1m")
        
        if candles is not None and not candles.empty:
            all_candles.append(candles)
        
        current_dt += timedelta(days=1)
    
    if not all_candles:
        return pd.DataFrame()
    
    df_candles_raw = pd.concat(all_candles, ignore_index=True)
    
    # 컬럼명 유연하게 처리
    df_candles = pd.DataFrame()
    
    if 'date_time_kst' in df_candles_raw.columns:
        df_candles['timestamp'] = pd.to_datetime(df_candles_raw['date_time_kst'])
    elif 'date_time_utc' in df_candles_raw.columns:
        df_candles['timestamp'] = pd.to_datetime(df_candles_raw['date_time_utc'])
    elif 'candle_date_time_kst' in df_candles_raw.columns:
        df_candles['timestamp'] = pd.to_datetime(df_candles_raw['candle_date_time_kst'])
    
    df_candles['trade_price'] = df_candles_raw.get('trade_price', df_candles_raw.get('close'))
    df_candles['opening_price'] = df_candles_raw.get('opening_price', df_candles_raw.get('open'))
    df_candles['high_price'] = df_candles_raw.get('high_price', df_candles_raw.get('high'))
    df_candles['low_price'] = df_candles_raw.get('low_price', df_candles_raw.get('low'))
    df_candles['candle_acc_trade_volume'] = df_candles_raw.get('candle_acc_trade_volume', df_candles_raw.get('volume'))
    
    df_candles = df_candles.sort_values('timestamp').reset_index(drop=True)
    
    # 틱 데이터
    all_ticks = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        
        try:
            df_tick = download_tick_data(date_str)
            if df_tick is not None and not df_tick.empty:
                all_ticks.append(df_tick)
        except:
            pass
        
        current_dt += timedelta(days=1)
    
    if not all_ticks:
        return df_candles
    
    # 틱 데이터 병합 및 1분 집계
    df_ticks_raw = pd.concat(all_ticks, ignore_index=True)
    df_ticks = aggregate_ticks_to_minute(df_ticks_raw)
    
    # datetime 컬럼명을 timestamp로 변경
    if 'datetime' in df_ticks.columns:
        df_ticks = df_ticks.rename(columns={'datetime': 'timestamp'})
    
    df_ticks = df_ticks.sort_values('timestamp').reset_index(drop=True)
    
    # 병합
    df = pd.merge(df_candles, df_ticks, on='timestamp', how='left', suffixes=('', '_tick'))
    
    # 틱 feature 없는 경우 0으로 채우기
    tick_cols = ['tick_count', 'buy_volume', 'sell_volume', 'buy_pressure', 
                 'large_buy_count', 'large_sell_count', 'vwap']
    for col in tick_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def backtest_with_params(df, model_path, buy_threshold, stop_loss, take_profit, time_limit):
    """특정 파라미터로 백테스트"""
    import joblib
    
    # 모델 로드
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    balance = 1000000
    coin = 0
    buy_price = 0
    buy_time_idx = 0
    
    trades = []
    window_size = 100
    
    for i in range(window_size, len(df)):
        current_price = df.loc[i, 'trade_price']
        current_time = df.loc[i, 'timestamp']
        
        # 보유 중이면 매도 확인
        if coin > 0:
            profit_pct = (current_price - buy_price) / buy_price * 100
            hold_minutes = i - buy_time_idx
            
            sell_reason = None
            
            if profit_pct >= take_profit:
                sell_reason = 'take_profit'
            elif profit_pct <= -stop_loss:
                sell_reason = 'stop_loss'
            elif hold_minutes >= time_limit:
                sell_reason = 'time_limit'
            
            if sell_reason:
                revenue = coin * current_price * 0.9995
                balance += revenue
                
                trades.append({
                    'exit_reason': sell_reason,
                    'profit_pct': profit_pct
                })
                
                coin = 0
                buy_price = 0
                buy_time_idx = 0
        
        # 미보유 시 매수 확인
        else:
            # 캔들 윈도우 (필수 컬럼만)
            required_cols = ['timestamp', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']
            candle_window = df.iloc[i-window_size:i][required_cols].copy()
            candle_window.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            }, inplace=True)
            candle_window['datetime'] = candle_window['timestamp']
            candle_window = candle_window[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # NaN 채우기
            candle_window = candle_window.ffill().bfill().fillna(0)
            
            # 틱 윈도우 (틱 feature 컬럼들)
            tick_window = None
            if 'buy_pressure' in df.columns:
                tick_window = df.iloc[i-window_size:i]
            
            features = calculate_combined_features(candle_window, tick_window)
            
            if features is None:
                continue
            
            # Feature DataFrame 생성 및 필요한 컬럼만 선택
            feature_df = pd.DataFrame([features])
            feature_values = feature_df[feature_cols].values
            
            # NaN 처리
            if np.isnan(feature_values).any():
                feature_values = np.nan_to_num(feature_values, nan=0.0)
            
            prediction = model.predict(feature_values)[0]
            
            if prediction >= buy_threshold:
                fee = current_price * 0.0005
                coin = balance / (current_price + fee)
                balance = 0
                buy_price = current_price
                buy_time_idx = i
    
    # 최종 청산
    if coin > 0:
        final_revenue = coin * df.iloc[-1]['trade_price'] * 0.9995
        balance += final_revenue
        
        profit_pct = (df.iloc[-1]['trade_price'] - buy_price) / buy_price * 100
        trades.append({
            'exit_reason': 'final',
            'profit_pct': profit_pct
        })
    
    total_return = (balance - 1000000) / 1000000 * 100
    
    if len(trades) == 0:
        return {
            'total_return': total_return,
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0
        }
    
    wins = [t for t in trades if t['profit_pct'] > 0]
    win_rate = len(wins) / len(trades) * 100
    avg_profit = np.mean([t['profit_pct'] for t in trades])
    
    return {
        'total_return': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'trades': trades
    }


def optimize_walk_forward():
    """Walk-Forward 방식으로 파라미터 최적화"""
    
    print("=" * 80)
    print("v7 파라미터 최적화 (Walk-Forward 방식)")
    print("=" * 80)
    
    # 2024년을 4개 분기로 나눔
    periods = [
        ("20240101", "20240331", "20240401", "20240430"),  # Q1 학습 → Q2 검증
        ("20240401", "20240630", "20240701", "20240731"),  # Q1+Q2 학습 → Q3 검증
        ("20240701", "20240930", "20241001", "20241031"),  # Q3 학습 → Q4 검증
    ]
    
    # 파라미터 조합
    param_grid = {
        'buy_threshold': [0.05, 0.1, 0.15, 0.2],
        'stop_loss': [0.2, 0.3, 0.4, 0.5],
        'take_profit': [0.4, 0.6, 0.8, 1.0],
        'time_limit': [5, 8, 10, 15]
    }
    
    model_path = "model/lgb_v7_tick.pkl"
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 모델 파일 없음: {model_path}")
        print("먼저 train_v7_walk_forward.py를 실행하세요!")
        return
    
    all_results = []
    
    # 각 기간마다 모든 파라미터 조합 테스트
    for period_idx, (train_start, train_end, valid_start, valid_end) in enumerate(periods, 1):
        print(f"\n{'=' * 80}")
        print(f"Period {period_idx}: 학습 {train_start}~{train_end} → 검증 {valid_start}~{valid_end}")
        print(f"{'=' * 80}")
        
        # 검증 데이터 로드
        print(f"\n[Loading] 검증 데이터 로드 중...")
        df_valid = load_combined_data(valid_start, valid_end)
        
        if df_valid.empty:
            print("[WARNING] 검증 데이터 없음, 스킵")
            continue
        
        print(f"[OK] {len(df_valid)} 분 데이터 로드됨\n")
        
        # 모든 파라미터 조합 테스트
        period_results = []
        
        param_combinations = list(product(
            param_grid['buy_threshold'],
            param_grid['stop_loss'],
            param_grid['take_profit'],
            param_grid['time_limit']
        ))
        
        total_combos = len(param_combinations)
        print(f"총 {total_combos}개 파라미터 조합 테스트 중...\n")
        
        for idx, (buy_th, stop, take, time) in enumerate(param_combinations, 1):
            result = backtest_with_params(
                df_valid, model_path,
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
        
        # 해당 기간 베스트 파라미터
        best = max(period_results, key=lambda x: x['total_return'])
        print(f"\n[Period {period_idx} Best]")
        print(f"  파라미터: {best['params']}")
        print(f"  수익률: {best['total_return']:.2f}%")
        print(f"  거래: {best['num_trades']}회")
        print(f"  승률: {best['win_rate']:.1f}%")
    
    # 전체 결과 분석
    print(f"\n{'=' * 80}")
    print("전체 기간 최적 파라미터 분석")
    print(f"{'=' * 80}\n")
    
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
    
    # 최종 추천 파라미터
    best_param_tuple = sorted_params[0][0]
    best_params = dict(best_param_tuple)
    best_perf = sorted_params[0][1]
    
    print(f"{'=' * 80}")
    print("최종 추천 파라미터")
    print(f"{'=' * 80}")
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
    
    with open('optimization_result_v7.pkl', 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\n[저장] optimization_result_v7.pkl")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    optimize_walk_forward()

