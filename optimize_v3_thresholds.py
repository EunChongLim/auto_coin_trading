"""
v3.0 모델 임계값 최적화
"""

import pandas as pd
import numpy as np
import joblib
from download_data import load_daily_csv
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
from datetime import datetime, timedelta
import random
import itertools


def backtest_v3_optimized(date_str, buy_threshold, sell_threshold, stop_loss_pct, take_profit_pct):
    """v3.0 백테스팅 (파라미터화)"""
    model_data = joblib.load("model/lgb_model_v3.pkl")
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    df = load_daily_csv(date_str, "data/daily_1m", "1m")
    if df is None or len(df) < 100:
        return None
    
    df = df.rename(columns={'date_time_utc': 'timestamp', 'acc_trade_volume': 'volume'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    df = add_all_indicators(df)
    df = add_multi_timeframe_features(df)
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    X = df[feature_cols]
    predictions = model.predict(X, num_iteration=model.best_iteration)
    df['prob_up'] = predictions[:, 2]
    df['prob_down'] = predictions[:, 0]
    
    balance = 1_000_000
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trades = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        price = row['close']
        
        if coin_holding > 0:
            profit_rate = (price - buy_price) / buy_price * 100
            
            sell_reason = None
            if profit_rate <= -stop_loss_pct:
                sell_reason = "손절"
            elif profit_rate >= take_profit_pct:
                sell_reason = "익절"
            elif row['prob_down'] >= sell_threshold:
                sell_reason = "ML하락"
            
            if sell_reason:
                balance = coin_holding * price * 0.9995
                profit = balance - buy_balance
                trades.append({'type': 'SELL', 'profit': profit})
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        else:
            if row['prob_up'] >= buy_threshold and balance > 10000:
                buy_balance = balance
                coin_holding = (balance * 0.9995) / price
                buy_price = price
                balance = 0
                trades.append({'type': 'BUY'})
    
    if coin_holding > 0:
        balance = coin_holding * df.iloc[-1]['close'] * 0.9995
        profit = balance - buy_balance
        trades.append({'type': 'SELL', 'profit': profit})
    
    total_return = (balance - 1_000_000) / 1_000_000 * 100
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    win_rate = len([t for t in sell_trades if t['profit'] > 0]) / num_trades * 100 if num_trades > 0 else 0
    
    return {'return': total_return, 'num_trades': num_trades, 'win_rate': win_rate}


def optimize_v3():
    """v3.0 임계값 최적화"""
    print("=" * 80)
    print("Model v3.0 Threshold Optimization")
    print("=" * 80)
    
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(10, len(all_days))))
    
    print(f"\nTest: 10 days")
    print(f"   {', '.join(test_days[:5])}...")
    
    # 파라미터 그리드
    buy_thresholds = [0.05, 0.08, 0.1, 0.12, 0.15]
    sell_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    stop_losses = [0.5, 0.6, 0.7]
    take_profits = [1.2, 1.5, 1.8, 2.0]
    
    total = len(buy_thresholds) * len(sell_thresholds) * len(stop_losses) * len(take_profits)
    
    print(f"\n[Grid] {total} combinations")
    print(f"  buy: {buy_thresholds}")
    print(f"  sell: {sell_thresholds}")
    print(f"  stop: {stop_losses}")
    print(f"  take: {take_profits}")
    
    print("\n" + "=" * 80)
    print("Testing...")
    print("=" * 80)
    
    all_results = []
    combo_idx = 0
    
    for buy_th, sell_th, stop, take in itertools.product(buy_thresholds, sell_thresholds, stop_losses, take_profits):
        combo_idx += 1
        
        if buy_th >= sell_th:
            continue
        
        print(f"\n[{combo_idx}/{total}] buy={buy_th:.2f} | sell={sell_th:.2f} | stop={stop:.1f}% | take={take:.1f}%")
        
        day_results = []
        for date_str in test_days:
            result = backtest_v3_optimized(date_str, buy_th, sell_th, stop, take)
            if result:
                day_results.append(result)
        
        if day_results:
            avg_return = np.mean([r['return'] for r in day_results])
            avg_trades = np.mean([r['num_trades'] for r in day_results])
            avg_win_rate = np.mean([r['win_rate'] for r in day_results])
            
            print(f"   {avg_return:+.2f}% | {avg_trades:.1f} trades/day | {avg_win_rate:.1f}% win")
            
            all_results.append({
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'stop_loss': stop,
                'take_profit': take,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'avg_win_rate': avg_win_rate
            })
    
    all_results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    print("\n" + "=" * 80)
    print("Top 10 Results")
    print("=" * 80)
    
    for i, r in enumerate(all_results[:10], 1):
        print(f"\n[{i}]")
        print(f"   buy={r['buy_threshold']:.2f} | sell={r['sell_threshold']:.2f} | stop={r['stop_loss']:.1f}% | take={r['take_profit']:.1f}%")
        print(f"   Return: {r['avg_return']:+.2f}% | Trades: {r['avg_trades']:.1f}/day | Win: {r['avg_win_rate']:.1f}%")
    
    best = all_results[0]
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    
    print(f"\nbuy_threshold = {best['buy_threshold']}")
    print(f"sell_threshold = {best['sell_threshold']}")
    print(f"stop_loss_pct = {best['stop_loss']}")
    print(f"take_profit_pct = {best['take_profit']}")
    
    print(f"\nPerformance:")
    print(f"  Return: {best['avg_return']:+.2f}%")
    print(f"  Trades: {best['avg_trades']:.1f}/day")
    print(f"  Win Rate: {best['avg_win_rate']:.1f}%")
    
    print("\n" + "=" * 80)
    
    return best


if __name__ == "__main__":
    random.seed(42)
    best_config = optimize_v3()

