"""
전략 4: Ichimoku Cloud (일목균형표)
학술적 근거: 70년 이상 검증된 일본 기술적 분석
핵심: 5개 라인으로 추세, 지지/저항, 모멘텀을 한번에 판단
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from datetime import datetime, timedelta
import random


def calculate_ichimoku(df):
    """
    Ichimoku Cloud 지표 계산
    
    5개 라인:
    1. Tenkan-sen (전환선): (9일 최고+최저)/2
    2. Kijun-sen (기준선): (26일 최고+최저)/2
    3. Senkou Span A (선행스팬A): (전환선+기준선)/2, 26일 선행
    4. Senkou Span B (선행스팬B): (52일 최고+최저)/2, 26일 선행
    5. Chikou Span (후행스팬): 현재 종가, 26일 후행
    """
    # Tenkan-sen (전환선) - 9일
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    
    # Kijun-sen (기준선) - 26일
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    
    # Senkou Span A (선행스팬 A) - 26일 선행
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    # Senkou Span B (선행스팬 B) - 52일 최고최저, 26일 선행
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    # Chikou Span (후행스팬) - 26일 후행
    df['chikou_span'] = df['close'].shift(-26)
    
    # 구름 두께 (변동성 측정)
    df['cloud_thickness'] = np.abs(df['senkou_span_a'] - df['senkou_span_b'])
    df['cloud_color'] = np.where(df['senkou_span_a'] > df['senkou_span_b'], 1, -1)  # 1: 상승, -1: 하락
    
    # 추가 지표
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    
    return df


def calculate_rsi(series, period=14):
    """RSI 계산"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_ichimoku(date_str, initial_balance=1_000_000, fee_rate=0.0005):
    """
    Ichimoku Cloud 백테스팅
    
    강력한 매수 신호 (TK Cross + Cloud):
    1. 가격이 구름 위 (상승 추세)
    2. 전환선 > 기준선 (단기 모멘텀 상승)
    3. 구름이 녹색 (Span A > Span B)
    4. Chikou Span이 과거 가격 위
    5. RSI 40~70 (과열 아님)
    6. 거래량 > 평균
    
    매도 조건:
    1. 가격이 구름 아래로 하락
    2. 전환선 < 기준선 (Dead Cross)
    3. RSI > 75
    4. 익절: +1.5%, 손절: -0.8%
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
    
    # Ichimoku 계산
    df = calculate_ichimoku(df)
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
            # 가격이 구름 아래로 하락 (추세 반전)
            elif pd.notna(row['senkou_span_a']) and pd.notna(row['senkou_span_b']):
                cloud_top = max(row['senkou_span_a'], row['senkou_span_b'])
                if price < cloud_top:
                    sell_reason = "구름하락"
            # Dead Cross (전환선 < 기준선)
            elif pd.notna(row['tenkan_sen']) and pd.notna(row['kijun_sen']) and row['tenkan_sen'] < row['kijun_sen']:
                sell_reason = "DeadCross"
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
            if all(pd.notna(row[col]) for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'rsi', 'volume_ma']):
                # Ichimoku 매수 조건
                cloud_top = max(row['senkou_span_a'], row['senkou_span_b'])
                cloud_bottom = min(row['senkou_span_a'], row['senkou_span_b'])
                
                above_cloud = price > cloud_top  # 가격이 구름 위 (상승 추세)
                tk_cross = row['tenkan_sen'] > row['kijun_sen']  # Golden Cross
                green_cloud = row['cloud_color'] > 0  # 구름 녹색
                
                # Chikou Span 확인 (26일 전 가격과 비교)
                if i >= 26:
                    past_price = df.iloc[i-26]['close']
                    chikou_above = row['chikou_span'] > past_price
                else:
                    chikou_above = True  # 데이터 부족 시 무시
                
                rsi_ok = 40 < row['rsi'] < 70  # 과열 아님
                volume_ok = row['volume'] > row['volume_ma']  # 거래량 충분
                
                # 강력한 매수 신호 (모든 조건 만족)
                buy_signal = (
                    above_cloud and 
                    tk_cross and 
                    green_cloud and 
                    chikou_above and 
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
    print("Strategy 4: Ichimoku Cloud")
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
        
        result = backtest_ichimoku(date_str)
        
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

