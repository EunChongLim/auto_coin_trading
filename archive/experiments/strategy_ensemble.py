"""
전략 6: 앙상블 투표 시스템
학술적 근거: Ensemble Methods (다중 전략 조합으로 안정성 향상)
핵심: 여러 전략의 신호를 투표로 결정
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from datetime import datetime, timedelta
import random


def calculate_all_indicators(df):
    """
    모든 지표 한번에 계산
    """
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Moving Averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    return df


def get_strategy_signals(df, i):
    """
    각 전략별 매수/매도 신호 생성
    
    Returns:
        dict: {'strategy_name': (buy_signal, sell_signal)}
    """
    row = df.iloc[i]
    prev_row = df.iloc[i-1] if i > 0 else row
    
    signals = {}
    
    # 전략 1: Mean Reversion (Bollinger Bands)
    mean_rev_buy = (
        row['close'] <= row['bb_lower'] * 1.02 and
        row['volume_ratio'] > 1.5 and
        row['rsi'] < 35
    )
    mean_rev_sell = (
        row['close'] >= row['bb_upper'] * 0.98 or
        row['rsi'] > 75
    )
    signals['MeanReversion'] = (mean_rev_buy, mean_rev_sell)
    
    # 전략 2: Momentum (MA Cross + MACD)
    momentum_buy = (
        row['ma5'] > row['ma20'] > row['ma60'] and
        row['macd'] > row['macd_signal'] and
        50 < row['rsi'] < 70 and
        row['volume_ratio'] > 1.2
    )
    momentum_sell = (
        row['ma5'] < row['ma20'] or
        row['macd'] < row['macd_signal'] or
        row['rsi'] > 80
    )
    signals['Momentum'] = (momentum_buy, momentum_sell)
    
    # 전략 3: Stochastic (과매수/과매도)
    stoch_buy = (
        row['stoch_k'] < 20 and
        row['stoch_k'] > prev_row['stoch_k'] and  # 상승 중
        row['rsi'] < 45 and
        row['volume_ratio'] > 1.3
    )
    stoch_sell = (
        row['stoch_k'] > 80 or
        row['rsi'] > 75
    )
    signals['Stochastic'] = (stoch_buy, stoch_sell)
    
    # 전략 4: Volume Breakout
    volume_buy = (
        row['volume_ratio'] > 2.0 and
        row['close'] > prev_row['close'] and
        row['close'] > row['ma20'] and
        row['rsi'] < 60
    )
    volume_sell = (
        row['volume_ratio'] < 0.8 or
        row['rsi'] > 75
    )
    signals['VolumeBreakout'] = (volume_buy, volume_sell)
    
    return signals


def backtest_ensemble(date_str, initial_balance=1_000_000, fee_rate=0.0005, 
                      buy_votes_needed=2, sell_votes_needed=2):
    """
    앙상블 전략 백테스팅
    
    투표 시스템:
    - 4개 전략이 각각 매수/매도 신호 생성
    - N개 이상 동의하면 실행 (기본: 2개)
    
    매수: 2개 이상 전략이 매수 신호
    매도: 2개 이상 전략이 매도 신호 OR 손익 조건
    
    손절: -0.8%, 익절: +1.5%
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
    df = calculate_all_indicators(df)
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    # 백테스팅
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trades = []
    
    stop_loss_pct = 0.8
    take_profit_pct = 1.5
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        
        # 각 전략 신호 수집
        try:
            strategy_signals = get_strategy_signals(df, i)
        except:
            continue
        
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
            else:
                # 매도 투표 집계
                sell_votes = sum(1 for name, (buy, sell) in strategy_signals.items() if sell)
                
                if sell_votes >= sell_votes_needed:
                    sell_reason = f"투표{sell_votes}/4"
            
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
            # 매수 투표 집계
            buy_votes = sum(1 for name, (buy, sell) in strategy_signals.items() if buy)
            
            if buy_votes >= buy_votes_needed and balance > 10000:
                # 매수
                buy_balance = balance
                coin_holding = (balance * (1 - fee_rate)) / price
                buy_price = price
                
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'balance_before': balance,
                    'votes': buy_votes
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
    print("Strategy 6: Ensemble Voting System")
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
        
        result = backtest_ensemble(date_str, buy_votes_needed=2, sell_votes_needed=2)
        
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
    print(f"\nML v2.0   : +1.46% (3.0 trades/day, 77.4% win)")
    print(f"Ensemble  : {avg_return:+.2f}% ({avg_trades:.1f} trades/day, {avg_win_rate:.1f}% win)")
    
    if avg_return > 1.46:
        print(f"\n*** NEW WINNER! ***")
        print(f"Ensemble beats ML v2.0 by {avg_return - 1.46:+.2f}%")
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

