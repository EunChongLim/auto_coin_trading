"""
모델 v2.0 임계값 최적화
buy_threshold, sell_threshold, stop_loss, take_profit 최적화
"""

import pandas as pd
import numpy as np
import joblib
from download_data import load_daily_csv
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
import random
from datetime import datetime, timedelta
import itertools


def run_backtest_v2_optimized(date_str, model_data, initial_balance=1_000_000, fee_rate=0.0005, 
                               buy_threshold=0.5, sell_threshold=0.5, stop_loss_pct=0.5, take_profit_pct=1.0):
    """
    3-Class 모델 기반 백테스팅 (최적화용)
    """
    # 1. 데이터 로드
    df = load_daily_csv(date_str, "data/daily_1m", "1m")
    if df is None or len(df) == 0:
        return None
    
    # 컬럼 매핑
    df = df.rename(columns={
        'date_time_utc': 'timestamp',
        'acc_trade_volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 2. 특징 생성
    df = add_all_indicators(df)
    df = add_multi_timeframe_features(df)
    df = df.dropna()
    
    if len(df) < 100:
        return None
    
    # 3. 예측
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # 특징 존재 여부 확인
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        return None
    
    X = df[feature_cols]
    predictions = model.predict(X, num_iteration=model.best_iteration)
    
    # 예측 확률: [하락(0), 횡보(1), 상승(2)]
    df['prob_down'] = predictions[:, 0]
    df['prob_sideways'] = predictions[:, 1]
    df['prob_up'] = predictions[:, 2]
    
    # 4. 백테스팅
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    
    trades = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        price = row['close']
        
        # 보유 중
        if coin_holding > 0:
            profit_rate = (price - buy_price) / buy_price * 100
            
            sell_reason = None
            
            # 손절
            if profit_rate <= -stop_loss_pct:
                sell_reason = "손절"
            # 익절
            elif profit_rate >= take_profit_pct:
                sell_reason = "익절"
            # 하락 예측 매도
            elif row['prob_down'] >= sell_threshold:
                sell_reason = "하락예측"
            
            if sell_reason:
                # 매도
                balance = coin_holding * price * (1 - fee_rate)
                profit = balance - buy_balance
                
                trades.append({
                    'type': 'SELL',
                    'reason': sell_reason,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'balance_after': balance
                })
                
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        
        # 미보유 중
        else:
            # 상승 예측 매수
            if row['prob_up'] >= buy_threshold and balance > 10000:
                # 매수
                buy_balance = balance
                coin_holding = (balance * (1 - fee_rate)) / price
                buy_price = price
                
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'balance_before': balance
                })
                
                balance = 0
    
    # 마지막 포지션 청산
    if coin_holding > 0:
        final_price = df.iloc[-1]['close']
        balance = coin_holding * final_price * (1 - fee_rate)
        profit = balance - buy_balance
        profit_rate = (final_price - buy_price) / buy_price * 100
        
        trades.append({
            'type': 'SELL',
            'reason': '종료',
            'profit': profit,
            'profit_rate': profit_rate,
            'balance_after': balance
        })
    
    # 5. 결과 계산
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    
    if num_trades > 0:
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        lose_trades = [t for t in sell_trades if t['profit'] <= 0]
        win_rate = len(win_trades) / num_trades * 100
        
        avg_profit = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([abs(t['profit']) for t in lose_trades]) if lose_trades else 0
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
    
    return {
        'return': total_return,
        'final_balance': final_balance,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def optimize_thresholds():
    """
    임계값 Grid Search 최적화
    """
    print("=" * 80)
    print("Model v2.0 Threshold Optimization")
    print("=" * 80)
    
    # 모델 로드
    print("\n[Load Model]")
    model_data = joblib.load("model/lgb_model_v2.pkl")
    print(f"   - Version: {model_data['version']}")
    print(f"   - Features: {len(model_data['feature_cols'])}")
    
    # 테스트 날짜 준비
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(10, len(all_days))))
    
    print(f"\n[Test Period] 10 days")
    print(f"   {', '.join(test_days[:5])}...")
    
    # 파라미터 그리드
    buy_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    sell_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    stop_losses = [0.4, 0.5, 0.6]
    take_profits = [0.8, 1.0, 1.2, 1.5]
    
    total_combinations = len(buy_thresholds) * len(sell_thresholds) * len(stop_losses) * len(take_profits)
    
    print(f"\n[Parameter Grid]")
    print(f"   - buy_threshold: {buy_thresholds}")
    print(f"   - sell_threshold: {sell_thresholds}")
    print(f"   - stop_loss: {stop_losses}")
    print(f"   - take_profit: {take_profits}")
    print(f"   - Total: {total_combinations} combinations")
    
    # Grid Search
    print(f"\n" + "=" * 80)
    print(f"Grid Search Start")
    print("=" * 80)
    
    all_results = []
    combo_idx = 0
    
    for buy_th, sell_th, stop_loss, take_profit in itertools.product(
        buy_thresholds, sell_thresholds, stop_losses, take_profits
    ):
        combo_idx += 1
        
        # 매수 임계값이 매도 임계값보다 낮아야 함
        if buy_th >= sell_th:
            continue
        
        print(f"\n[{combo_idx}/{total_combinations}] 매수={buy_th:.1f} | 매도={sell_th:.1f} | 손절={stop_loss:.1f}% | 익절={take_profit:.1f}%")
        
        day_results = []
        
        for date_str in test_days:
            result = run_backtest_v2_optimized(
                date_str, model_data,
                buy_threshold=buy_th,
                sell_threshold=sell_th,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit
            )
            
            if result:
                day_results.append(result)
        
        if day_results:
            avg_return = np.mean([r['return'] for r in day_results])
            avg_trades = np.mean([r['num_trades'] for r in day_results])
            avg_win_rate = np.mean([r['win_rate'] for r in day_results])
            
            print(f"   수익률: {avg_return:+.2f}% | 거래: {avg_trades:.1f}회/일 | 승률: {avg_win_rate:.1f}%")
            
            all_results.append({
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'avg_win_rate': avg_win_rate,
                'score': avg_return  # 정렬 기준
            })
    
    # 결과 정렬
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 상위 10개 출력
    print("\n" + "=" * 80)
    print("Optimization Results (Top 10)")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:10], 1):
        print(f"\n[Rank {i}]")
        print(f"   buy_threshold: {result['buy_threshold']:.1f}")
        print(f"   sell_threshold: {result['sell_threshold']:.1f}")
        print(f"   stop_loss: {result['stop_loss']:.1f}%")
        print(f"   take_profit: {result['take_profit']:.1f}%")
        print(f"   ---")
        print(f"   avg_return: {result['avg_return']:+.2f}%")
        print(f"   avg_trades: {result['avg_trades']:.1f}/day")
        print(f"   avg_win_rate: {result['avg_win_rate']:.1f}%")
    
    # 최적 설정
    best = all_results[0]
    
    print("\n" + "=" * 80)
    print("Best Configuration")
    print("=" * 80)
    
    print(f"\nParameters:")
    print(f"   buy_threshold = {best['buy_threshold']:.1f}  # Buy if prob_up >= {best['buy_threshold']*100:.0f}%")
    print(f"   sell_threshold = {best['sell_threshold']:.1f}  # Sell if prob_down >= {best['sell_threshold']*100:.0f}%")
    print(f"   stop_loss_pct = {best['stop_loss']:.1f}  # Stop loss at {best['stop_loss']}%")
    print(f"   take_profit_pct = {best['take_profit']:.1f}  # Take profit at {best['take_profit']}%")
    
    print(f"\nPerformance:")
    print(f"   avg_return: {best['avg_return']:+.2f}%")
    print(f"   avg_trades: {best['avg_trades']:.1f}/day")
    print(f"   avg_win_rate: {best['avg_win_rate']:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    random.seed(42)
    optimize_thresholds()


