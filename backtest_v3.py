"""
v3.0 모델 백테스팅
"""

import pandas as pd
import numpy as np
import joblib
from download_data import load_daily_csv
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
from datetime import datetime, timedelta
import random


def backtest_v3(date_str, threshold=0.1):
    """v3.0 백테스팅"""
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
            if profit_rate <= -0.6:
                sell_reason = "손절"
            elif profit_rate >= 1.5:
                sell_reason = "익절"
            elif row['prob_down'] >= 0.4:
                sell_reason = "ML하락"
            
            if sell_reason:
                balance = coin_holding * price * 0.9995
                profit = balance - buy_balance
                trades.append({'type': 'SELL', 'profit': profit, 'profit_rate': profit_rate})
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        else:
            if row['prob_up'] >= threshold and balance > 10000:
                buy_balance = balance
                coin_holding = (balance * 0.9995) / price
                buy_price = price
                balance = 0
                trades.append({'type': 'BUY'})
    
    if coin_holding > 0:
        balance = coin_holding * df.iloc[-1]['close'] * 0.9995
        profit = balance - buy_balance
        trades.append({'type': 'SELL', 'profit': profit, 'profit_rate': (df.iloc[-1]['close'] - buy_price) / buy_price * 100})
    
    total_return = (balance - 1_000_000) / 1_000_000 * 100
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    win_rate = len([t for t in sell_trades if t['profit'] > 0]) / num_trades * 100 if num_trades > 0 else 0
    
    return {'date': date_str, 'return': total_return, 'num_trades': num_trades, 'win_rate': win_rate}


def test_v3(num_days=10):
    """v3.0 테스트"""
    print("=" * 80)
    print("Model v3.0 Testing")
    print("=" * 80)
    
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(num_days, len(all_days))))
    
    print(f"\nTest: {num_days} days\n")
    
    results = []
    for i, date_str in enumerate(test_days, 1):
        print(f"[{i}/{num_days}] {date_str}...")
        result = backtest_v3(date_str, threshold=0.1)
        if result:
            results.append(result)
            print(f"   {result['return']:+.2f}% | {result['num_trades']} trades | {result['win_rate']:.1f}% win")
    
    if results:
        avg_return = np.mean([r['return'] for r in results])
        avg_trades = np.mean([r['num_trades'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
        
        print("\n" + "=" * 80)
        print("v3.0 Results")
        print("=" * 80)
        print(f"\nAvg Return: {avg_return:+.2f}%")
        print(f"Avg Trades: {avg_trades:.1f}/day")
        print(f"Avg Win Rate: {avg_win_rate:.1f}%")
        
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        print(f"\nML v2.0: +1.46% (3.0 trades/day, 77.4% win)")
        print(f"ML v3.0: {avg_return:+.2f}% ({avg_trades:.1f} trades/day, {avg_win_rate:.1f}% win)")
        
        if avg_return > 1.46:
            print("\n*** v3.0 WINS! ***")
            return True
        else:
            print("\nv2.0 still better")
            return False


if __name__ == "__main__":
    random.seed(42)
    test_v3(10)

