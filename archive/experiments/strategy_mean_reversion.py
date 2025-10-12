"""
전략 1: Mean Reversion + Breakout
학술적 근거: 비트코인의 평균 회귀 성향 + 지지/저항 돌파
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from datetime import datetime, timedelta
import random


def calculate_indicators(df):
    """
    기술적 지표 계산
    """
    # Bollinger Bands (20, 2)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume MA (20)
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    
    # ATR (14) - 변동성 측정
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # 60분 고점/저점 (지지/저항)
    df['high_60'] = df['high'].rolling(window=60).max()
    df['low_60'] = df['low'].rolling(window=60).min()
    
    return df


def backtest_mean_reversion(date_str, initial_balance=1_000_000, fee_rate=0.0005):
    """
    Mean Reversion + Breakout 백테스팅
    
    매수 조건:
    1. 가격이 하단 밴드 근처 (BB_Lower * 1.01 이하)
    2. 거래량 급증 (평균의 2배 이상)
    3. RSI 과매도 (35 이하)
    4. 60분 저점 위 (바닥 확인)
    
    매도 조건:
    1. 상단 밴드 도달 (BB_Upper * 0.99 이상)
    2. RSI 과열 (75 이상)
    3. 변동성 기반 익절 (ATR * 3)
    4. 변동성 기반 손절 (-ATR * 1.5)
    """
    # 데이터 로드
    df = load_daily_csv(date_str, "data/daily_1m", "1m")
    if df is None or len(df) < 100:
        return None
    
    # 컬럼 매핑
    df = df.rename(columns={
        'date_time_utc': 'timestamp',
        'acc_trade_volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 지표 계산
    df = calculate_indicators(df)
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    # 백테스팅
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        
        # 보유 중
        if coin_holding > 0:
            profit_rate = (price - buy_price) / buy_price * 100
            
            # ATR 기반 동적 손익
            atr_pct = (row['atr'] / price) * 100
            take_profit_pct = atr_pct * 3  # ATR의 3배
            stop_loss_pct = atr_pct * 1.5  # ATR의 1.5배
            
            sell_reason = None
            
            # 변동성 기반 익절
            if profit_rate >= take_profit_pct:
                sell_reason = "ATR익절"
            # 변동성 기반 손절
            elif profit_rate <= -stop_loss_pct:
                sell_reason = "ATR손절"
            # 상단 밴드 도달
            elif pd.notna(row['bb_upper']) and price >= row['bb_upper'] * 0.99:
                sell_reason = "상단밴드"
            # RSI 과열
            elif row['rsi'] > 75:
                sell_reason = "RSI과열"
            
            if sell_reason:
                # 매도
                balance = coin_holding * price * (1 - fee_rate)
                profit = balance - buy_balance
                
                trades.append({
                    'type': 'SELL',
                    'reason': sell_reason,
                    'price': price,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'balance_after': balance
                })
                
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        
        # 미보유 중
        else:
            if pd.notna(row['rsi']) and pd.notna(row['bb_lower']) and pd.notna(row['volume_ma']):
                # Mean Reversion 매수 조건
                near_lower_band = price <= row['bb_lower'] * 1.01  # 하단 근처
                volume_surge = row['volume'] > row['volume_ma'] * 2.0  # 거래량 급증
                rsi_oversold = row['rsi'] < 35  # 과매도
                above_support = price > row['low_60']  # 60분 저점 위 (바닥 확인)
                
                buy_signal = near_lower_band and volume_surge and rsi_oversold and above_support
                
                if buy_signal and balance > 10000:
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
            'price': final_price,
            'profit': profit,
            'profit_rate': profit_rate,
            'balance_after': balance
        })
    
    # 결과 계산
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    
    if num_trades > 0:
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = len(win_trades) / num_trades * 100
    else:
        win_rate = 0
    
    return {
        'date': date_str,
        'return': total_return,
        'final_balance': final_balance,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'trades': trades
    }


def run_multi_day_test(num_days=10):
    """
    여러 날짜 테스트
    """
    print("=" * 80)
    print("Strategy 1: Mean Reversion + Breakout")
    print("=" * 80)
    
    # 랜덤 날짜 선택
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(num_days, len(all_days))))
    
    print(f"\nTest Period: {num_days} days")
    print(f"   {', '.join(test_days[:5])}...\n")
    
    results = []
    
    for i, date_str in enumerate(test_days, 1):
        print(f"[{i}/{num_days}] {date_str} testing...")
        
        result = backtest_mean_reversion(date_str)
        
        if result:
            results.append(result)
            print(f"   Return: {result['return']:+.2f}% | Trades: {result['num_trades']} | Win Rate: {result['win_rate']:.1f}%")
    
    # 집계
    if not results:
        print("\nNo results")
        return
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    avg_return = np.mean([r['return'] for r in results])
    avg_trades = np.mean([r['num_trades'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
    
    print(f"\nAvg Return: {avg_return:+.2f}%")
    print(f"Avg Trades: {avg_trades:.1f}/day")
    print(f"Avg Win Rate: {avg_win_rate:.1f}%")
    
    print("\nDaily Results:")
    for r in results:
        print(f"   {r['date']}: {r['return']:+.2f}% ({r['num_trades']} trades)")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    random.seed(42)
    run_multi_day_test(num_days=10)

