"""
v3.0 모델 백테스팅 - 6월 데이터 전용 (학습 데이터 제외)

학습: 2025년 1~5월 (80일)
테스트: 2025년 6월 (완전히 미래 데이터)
"""

import pandas as pd
import numpy as np
import joblib
from download_data import load_daily_csv
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
from datetime import datetime, timedelta
import random


def backtest_v3_continuous_june(start_date_str, num_days=10, buy_threshold=0.15, sell_threshold=0.4, 
                                stop_loss=0.6, take_profit=1.8, verbose=True):
    """
    연속된 N일 백테스팅 (6월 데이터 전용)
    """
    model_data = joblib.load("model/lgb_model_v3.pkl")
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # 연속된 N일 데이터 로드
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    all_data = []
    date_boundaries = []
    
    if verbose:
        print(f"\n[Loading {num_days} continuous days from {start_date_str}...]")
    
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        
        df_day = load_daily_csv(date_str, "data/daily_1m", "1m")
        if df_day is None or len(df_day) < 50:
            continue
        
        df_day = df_day.rename(columns={'date_time_utc': 'timestamp', 'acc_trade_volume': 'volume'})
        df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])
        df_day = df_day.set_index('timestamp')
        df_day = df_day.sort_index()
        
        all_data.append(df_day)
        date_boundaries.append((date_str, len(df_day)))
    
    if len(all_data) < num_days // 2:
        if verbose:
            print(f"[ERROR] Only {len(all_data)}/{num_days} days loaded")
        return None
    
    df = pd.concat(all_data, axis=0).sort_index()
    
    if verbose:
        print(f"[OK] Loaded {len(all_data)} days, {len(df)} candles total")
        print(f"[Processing indicators...]")
    
    df = add_all_indicators(df)
    df = add_multi_timeframe_features(df)
    df = df.dropna()
    
    if len(df) < 100:
        if verbose:
            print(f"[ERROR] Only {len(df)} valid candles after indicators")
        return None
    
    X = df[feature_cols]
    predictions = model.predict(X, num_iteration=model.best_iteration)
    df['prob_up'] = predictions[:, 2]
    df['prob_down'] = predictions[:, 0]
    
    if verbose:
        print(f"[Trading simulation started...]\n")
    
    # 거래 시뮬레이션
    initial_balance = 1_000_000
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trades = []
    
    daily_stats = []
    cumulative_rows = 0
    
    for date_str, day_rows in date_boundaries:
        day_start_idx = cumulative_rows
        day_end_idx = cumulative_rows + day_rows
        cumulative_rows = day_end_idx
        
        if day_end_idx > len(df):
            day_end_idx = len(df)
        
        day_start_balance = balance if coin_holding == 0 else coin_holding * df.iloc[day_start_idx]['close']
        day_start_trades = len([t for t in trades if t['type'] == 'SELL'])
        
        for i in range(day_start_idx, day_end_idx):
            if i >= len(df):
                break
            
            row = df.iloc[i]
            price = row['close']
            
            if coin_holding > 0:
                profit_rate = (price - buy_price) / buy_price * 100
                
                sell_reason = None
                if profit_rate <= -stop_loss:
                    sell_reason = "손절"
                elif profit_rate >= take_profit:
                    sell_reason = "익절"
                elif row['prob_down'] >= sell_threshold:
                    sell_reason = "ML하락"
                
                if sell_reason:
                    balance = coin_holding * price * 0.9995
                    profit = balance - buy_balance
                    trades.append({'type': 'SELL', 'profit': profit, 'profit_rate': profit_rate, 
                                 'date': date_str, 'reason': sell_reason})
                    coin_holding = 0
                    buy_price = 0
                    buy_balance = 0
            else:
                if row['prob_up'] >= buy_threshold and balance > 10000:
                    buy_balance = balance
                    coin_holding = (balance * 0.9995) / price
                    buy_price = price
                    balance = 0
                    trades.append({'type': 'BUY', 'date': date_str})
        
        if day_end_idx < len(df):
            day_end_price = df.iloc[day_end_idx - 1]['close']
        else:
            day_end_price = df.iloc[-1]['close']
        
        day_end_balance = balance if coin_holding == 0 else coin_holding * day_end_price
        day_return = (day_end_balance - day_start_balance) / day_start_balance * 100
        day_total_return = (day_end_balance - initial_balance) / initial_balance * 100
        day_trades = len([t for t in trades if t['type'] == 'SELL']) - day_start_trades
        
        daily_stats.append({
            'date': date_str,
            'day_return': day_return,
            'total_return': day_total_return,
            'day_trades': day_trades,
            'balance': day_end_balance,
            'holding': coin_holding > 0
        })
        
        if verbose:
            holding_str = f"보유중 (Entry: {buy_price:,.0f})" if coin_holding > 0 else "대기"
            print(f"  [{date_str}] 일수익: {day_return:+.2f}% | 누적: {day_total_return:+.2f}% | 거래: {day_trades}회 | {holding_str}")
    
    if coin_holding > 0:
        final_price = df.iloc[-1]['close']
        balance = coin_holding * final_price * 0.9995
        profit = balance - buy_balance
        profit_rate = (final_price - buy_price) / buy_price * 100
        trades.append({'type': 'SELL', 'profit': profit, 'profit_rate': profit_rate, 
                     'date': 'FINAL', 'reason': '종료'})
        coin_holding = 0
    
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    win_rate = (len([t for t in sell_trades if t['profit'] > 0]) / num_trades * 100) if num_trades > 0 else 0
    
    return {
        'start_date': start_date_str,
        'num_days': len(daily_stats),
        'total_return': total_return,
        'total_trades': num_trades,
        'avg_trades_per_day': num_trades / len(daily_stats) if daily_stats else 0,
        'win_rate': win_rate,
        'final_balance': final_balance,
        'daily_stats': daily_stats
    }


def test_v3_june():
    """6월 데이터로 테스트 (학습 데이터 제외)"""
    print("=" * 80)
    print("Model v3.0 Testing - JUNE 2025 (Unseen Future Data)")
    print("=" * 80)
    print("\n[IMPORTANT]")
    print("Training: Jan-May 2025 (80 days)")
    print("Testing: June 2025 (UNSEEN future data)")
    print("This is the TRUE performance test!")
    print("=" * 80)
    
    # 6월에서 연속된 10일 3번 테스트
    start_dates = ["20250601", "20250611", "20250621"]  # 6월 초, 중, 말
    
    all_results = []
    for i, start_date in enumerate(start_dates, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/3: Starting from {start_date}")
        print(f"{'='*80}")
        
        result = backtest_v3_continuous_june(
            start_date, 
            num_days=10, 
            buy_threshold=0.15, 
            sell_threshold=0.4, 
            stop_loss=0.6, 
            take_profit=1.8,
            verbose=True
        )
        
        if result:
            all_results.append(result)
            print(f"\n[Period Summary]")
            print(f"  Total Return: {result['total_return']:+.2f}%")
            print(f"  Total Trades: {result['total_trades']} ({result['avg_trades_per_day']:.1f}/day)")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Final Balance: {result['final_balance']:,.0f} KRW")
        else:
            print(f"\n[ERROR] Failed to test period from {start_date}")
        
        print()
    
    if all_results:
        print("\n" + "=" * 80)
        print("Overall Results (JUNE 2025 - UNSEEN DATA)")
        print("=" * 80)
        
        avg_total_return = np.mean([r['total_return'] for r in all_results])
        avg_trades_per_day = np.mean([r['avg_trades_per_day'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        
        all_daily_returns = []
        for r in all_results:
            for day in r['daily_stats']:
                all_daily_returns.append(day['day_return'])
        
        avg_daily_return = np.mean(all_daily_returns) if all_daily_returns else 0
        
        print(f"\nAvg 10-Day Return: {avg_total_return:+.2f}%")
        print(f"Avg Daily Return: {avg_daily_return:+.2f}%")
        print(f"Avg Trades/Day: {avg_trades_per_day:.1f}")
        print(f"Avg Win Rate: {avg_win_rate:.1f}%")
        
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        print(f"\nTraining Data (Jan-May): +59.75% (14.9 trades/day, 81.7% win) <- OVERFITTING!")
        print(f"Test Data (June):        {avg_total_return:+.2f}% ({avg_trades_per_day:.1f} trades/day, {avg_win_rate:.1f}% win) <- REAL")
        
        print("\n" + "=" * 80)
        print("Expected Real Trading Performance")
        print("=" * 80)
        print(f"Daily Return: {avg_daily_return:+.2f}%")
        print(f"Monthly Return (30 days): ~{avg_daily_return * 30:+.1f}%")
        print(f"Trades/Day: {avg_trades_per_day:.1f}")
        print(f"Win Rate: {avg_win_rate:.1f}%")
        
        if avg_total_return > 2.0:
            print("\n*** PROFITABLE on unseen data! ***")
        elif avg_total_return > 0:
            print("\n*** Slightly profitable ***")
        else:
            print("\n*** LOSS on unseen data - needs improvement ***")


if __name__ == "__main__":
    test_v3_june()

