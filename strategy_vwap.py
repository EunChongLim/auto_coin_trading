"""
전략 3: VWAP + Volume Profile
학술적 근거: 기관 투자자 매매 기법 (VWAP Trading)
핵심: VWAP를 중심으로 평균 회귀 + Volume Profile로 지지/저항 확인
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from datetime import datetime, timedelta
import random


def calculate_vwap_indicators(df):
    """
    VWAP 및 관련 지표 계산
    """
    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # VWAP 표준편차 밴드
    df['price_vol'] = df['close'] * df['volume']
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_price_vol'] = df['price_vol'].cumsum()
    
    # Rolling VWAP (60분)
    window = 60
    df['rolling_vwap'] = df['price_vol'].rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    
    # VWAP 대비 가격 위치
    df['vwap_diff'] = ((df['close'] - df['rolling_vwap']) / df['rolling_vwap']) * 100
    
    # Volume Profile - 가격대별 거래량 (간단한 버전)
    # 최근 60분간의 평균 거래량
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Volume-weighted 변동성
    df['volume_std'] = df['close'].rolling(window=20).std()
    
    # RSI (필터용)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Average (추세 확인용)
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    return df


def backtest_vwap(date_str, initial_balance=1_000_000, fee_rate=0.0005,
                  vwap_range=(-1.2, 0.2), volume_multiplier=1.2, rsi_max=60):
    """
    VWAP + Volume Profile 백테스팅 (파라미터 조정 가능)
    
    매수 조건 (완화):
    1. 가격이 VWAP 근처 (기본: -1.2% ~ +0.2%) - 더 넓은 범위
    2. 거래량 증가 (기본: 평균의 1.2배) - 조건 완화
    3. RSI < 60 (기본) - 과열만 아니면 OK
    4. 가격 > MA20 제거 (너무 엄격)
    
    매도 조건:
    1. 가격이 VWAP 위 (+0.5% 이상) - 목표 상향
    2. RSI > 70 (과열)
    3. 거래량 감소 (평균 이하)
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
    
    # 지표 계산
    df = calculate_vwap_indicators(df)
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
            # VWAP 위로 올라감 (목표 달성)
            elif pd.notna(row['vwap_diff']) and row['vwap_diff'] > 0.5:
                sell_reason = "VWAP상향"
            # RSI 과열
            elif row['rsi'] > 70:
                sell_reason = "RSI과열"
            # 거래량 감소
            elif row['volume_ratio'] < 0.8:
                sell_reason = "거래량감소"
            
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
            if pd.notna(row['vwap_diff']) and pd.notna(row['rsi']) and pd.notna(row['volume_ratio']):
                # VWAP 기반 매수 조건 (완화됨)
                near_vwap = vwap_range[0] < row['vwap_diff'] < vwap_range[1]  # VWAP 근처
                volume_ok = row['volume_ratio'] > volume_multiplier  # 거래량 조건 완화
                rsi_ok = row['rsi'] < rsi_max  # RSI 조건 완화
                
                buy_signal = near_vwap and volume_ok and rsi_ok
                
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
    print("Strategy 3: VWAP + Volume Profile")
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
        
        result = backtest_vwap(date_str)
        
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

