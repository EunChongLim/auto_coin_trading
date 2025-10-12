"""
전략 2: Momentum + Trend Following
학술적 근거: "Momentum Effect in Cryptocurrency Returns" (2019)
비트코인의 강한 모멘텀 효과 활용
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
    # Moving Averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # ADX (Average Directional Index) - 추세 강도
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr14 = true_range.rolling(window=14).sum()
    plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / tr14)
    
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['adx'] = dx.rolling(window=14).mean()
    
    # Price Rate of Change (ROC) - 모멘텀
    df['roc'] = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100
    
    return df


def backtest_momentum(date_str, initial_balance=1_000_000, fee_rate=0.0005):
    """
    Momentum + Trend Following 백테스팅
    
    매수 조건 (강한 추세 + 모멘텀):
    1. Golden Cross (MA5 > MA20 > MA60) - 상승 추세
    2. MACD > 0 AND MACD > Signal - 모멘텀 상승
    3. RSI 50~70 - 과열 아님
    4. ADX > 25 - 추세 강함
    5. ROC > 0 - 가격 모멘텀 양수
    6. Volume > avg * 1.2 - 거래량 뒷받침
    
    매도 조건:
    1. Dead Cross (MA5 < MA20) - 추세 반전
    2. RSI > 80 - 과열
    3. MACD < Signal - 모멘텀 약화
    4. 익절: +2.0%
    5. 손절: -1.0%
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
    
    # 고정 손익
    stop_loss_pct = 1.0
    take_profit_pct = 2.0
    
    for i in range(len(df)):
        row = df.iloc[i]
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
            # Dead Cross (추세 반전)
            elif pd.notna(row['ma5']) and pd.notna(row['ma20']) and row['ma5'] < row['ma20']:
                sell_reason = "DeadCross"
            # RSI 과열
            elif row['rsi'] > 80:
                sell_reason = "RSI과열"
            # MACD 약화
            elif pd.notna(row['macd']) and pd.notna(row['macd_signal']) and row['macd'] < row['macd_signal']:
                sell_reason = "MACD약화"
            
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
            if all(pd.notna(row[col]) for col in ['ma5', 'ma20', 'ma60', 'macd', 'macd_signal', 'rsi', 'adx', 'roc', 'volume_ratio']):
                # Momentum + Trend Following 매수 조건
                golden_cross = row['ma5'] > row['ma20'] > row['ma60']  # 상승 추세
                macd_positive = row['macd'] > 0 and row['macd'] > row['macd_signal']  # 모멘텀 상승
                rsi_good = 50 < row['rsi'] < 70  # 과열 아님
                strong_trend = row['adx'] > 25  # 추세 강함
                positive_momentum = row['roc'] > 0  # 가격 모멘텀 양수
                volume_support = row['volume_ratio'] > 1.2  # 거래량 뒷받침
                
                buy_signal = (
                    golden_cross and 
                    macd_positive and 
                    rsi_good and 
                    strong_trend and 
                    positive_momentum and 
                    volume_support
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
    print("Strategy 2: Momentum + Trend Following")
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
        
        result = backtest_momentum(date_str)
        
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

