"""
전략 5: Heikin-Ashi + Supertrend
학술적 근거: 노이즈 제거 기법 + 추세 추종
핵심: 일반 캔들의 노이즈를 제거하고 진짜 추세만 포착
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from datetime import datetime, timedelta
import random


def convert_to_heikin_ashi(df):
    """
    일반 캔들을 Heikin-Ashi 캔들로 변환
    
    Heikin-Ashi 계산:
    - HA_Close = (Open + High + Low + Close) / 4
    - HA_Open = (Previous HA_Open + Previous HA_Close) / 2
    - HA_High = Max(High, HA_Open, HA_Close)
    - HA_Low = Min(Low, HA_Open, HA_Close)
    """
    ha_df = df.copy()
    
    # HA Close
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # HA Open (첫 번째는 일반 캔들 평균)
    ha_df['ha_open'] = 0.0
    ha_df.iloc[0, ha_df.columns.get_loc('ha_open')] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2
    
    for i in range(1, len(ha_df)):
        ha_df.iloc[i, ha_df.columns.get_loc('ha_open')] = (
            ha_df.iloc[i-1]['ha_open'] + ha_df.iloc[i-1]['ha_close']
        ) / 2
    
    # HA High
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    
    # HA Low
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return ha_df


def calculate_supertrend(df, period=10, multiplier=3):
    """
    Supertrend 지표 계산
    
    핵심:
    - ATR 기반으로 동적 지지/저항선 생성
    - 가격이 Supertrend 위: 상승 추세
    - 가격이 Supertrend 아래: 하락 추세
    """
    # ATR 계산
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    # Basic Bands
    hl_avg = (df['high'] + df['low']) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Supertrend 계산
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            # 이전 추세가 상승
            if direction.iloc[i-1] == 1:
                if df['close'].iloc[i] <= supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                    direction.iloc[i] = 1
            # 이전 추세가 하락
            else:
                if df['close'].iloc[i] >= supertrend.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                    direction.iloc[i] = -1
    
    df['supertrend'] = supertrend
    df['trend_direction'] = direction
    
    return df


def calculate_indicators(df):
    """
    기술적 지표 계산
    """
    # RSI
    delta = df['ha_close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # EMA (빠른 추세)
    df['ema12'] = df['ha_close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['ha_close'].ewm(span=26, adjust=False).mean()
    
    return df


def backtest_heikin_ashi(date_str, initial_balance=1_000_000, fee_rate=0.0005):
    """
    Heikin-Ashi + Supertrend 백테스팅
    
    매수 조건 (명확한 상승 추세):
    1. Supertrend 방향 = 1 (상승)
    2. 가격 > Supertrend (지지선 위)
    3. HA 캔들 양봉 (HA_Close > HA_Open)
    4. EMA12 > EMA26 (단기 추세 상승)
    5. RSI 40~70 (과열 아님)
    6. Volume > avg * 1.1
    
    매도 조건:
    1. Supertrend 방향 = -1 (하락 전환)
    2. HA 캔들 음봉 연속 2개
    3. RSI > 75
    4. 익절: +1.5%, 손절: -0.7%
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
    
    # Heikin-Ashi 변환
    df = convert_to_heikin_ashi(df)
    
    # Supertrend 계산
    df = calculate_supertrend(df, period=10, multiplier=3)
    
    # 기타 지표 계산
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
    
    stop_loss_pct = 0.7
    take_profit_pct = 1.5
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
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
            # Supertrend 하락 전환
            elif pd.notna(row['trend_direction']) and row['trend_direction'] == -1:
                sell_reason = "추세반전"
            # HA 음봉 연속 2개
            elif row['ha_close'] < row['ha_open'] and prev_row['ha_close'] < prev_row['ha_open']:
                sell_reason = "HA음봉"
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
            if all(pd.notna(row[col]) for col in ['trend_direction', 'supertrend', 'ha_close', 'ha_open', 'ema12', 'ema26', 'rsi', 'volume_ratio']):
                # Heikin-Ashi + Supertrend 매수 조건
                uptrend = row['trend_direction'] == 1  # 상승 추세
                price_above_supertrend = price > row['supertrend']  # 지지선 위
                ha_bullish = row['ha_close'] > row['ha_open']  # HA 양봉
                ema_bullish = row['ema12'] > row['ema26']  # EMA 상승
                rsi_ok = 40 < row['rsi'] < 70  # 과열 아님
                volume_ok = row['volume_ratio'] > 1.1  # 거래량 충분
                
                buy_signal = (
                    uptrend and 
                    price_above_supertrend and 
                    ha_bullish and 
                    ema_bullish and 
                    rsi_ok and 
                    volume_ok
                )
                
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
    print("Strategy 5: Heikin-Ashi + Supertrend")
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
        
        result = backtest_heikin_ashi(date_str)
        
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
    
    # ML v2.0과 비교
    print("\n" + "=" * 80)
    print("Comparison with ML v2.0")
    print("=" * 80)
    print(f"\nML v2.0      : +1.46% (3.0 trades/day, 77.4% win)")
    print(f"Heikin-Ashi : {avg_return:+.2f}% ({avg_trades:.1f} trades/day, {avg_win_rate:.1f}% win)")
    
    if avg_return > 1.46:
        print(f"\n*** NEW WINNER! ***")
        print(f"Heikin-Ashi beats ML v2.0 by {avg_return - 1.46:+.2f}%")
    elif avg_return > 0:
        print(f"\nPositive but not better than ML v2.0")
        print(f"ML v2.0 leads by {1.46 - avg_return:+.2f}%")
    else:
        print(f"\nNegative returns - ML v2.0 is better by {1.46 - avg_return:+.2f}%")
    
    print("\nDaily Results:")
    for r in results:
        print(f"   {r['date']}: {r['return']:+.2f}% ({r['num_trades']} trades)")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    random.seed(42)
    run_multi_day_test(num_days=10)

