"""
규칙 기반 전략 (볼린저밴드) 손익 비율 최적화
이미 검증된 파라미터로 손익 비율만 최적화
"""

import random
import pandas as pd
import numpy as np
from itertools import product
from download_data import load_daily_csv
from main import compute_rsi


def run_rule_backtest(date_str, bb_tolerance=1.001, volume_multiplier=3.5, rsi_threshold=25,
                      stop_loss_pct=0.5, take_profit_pct=0.5, 
                      data_dir="data/daily_1m", timeframe="1m"):
    """
    규칙 기반 백테스팅 (볼린저밴드 + 거래량 + RSI)
    """
    # CSV 파일 로드
    df_full = load_daily_csv(date_str, data_dir, timeframe)
    
    if df_full is None or len(df_full) == 0:
        return None
    
    # 컬럼 매핑
    df_full = df_full.rename(columns={
        'date_time_utc': 'timestamp',
        'acc_trade_volume': 'volume'
    })
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    df_full = df_full.set_index('timestamp')
    df_full = df_full.sort_index()
    
    # 윈도우 크기 (1분봉 60개 = 1시간)
    window_size = 60
    
    if len(df_full) < window_size + 20:
        return None
    
    # 지표 계산
    df_full['rsi'] = compute_rsi(df_full['close'], 14)
    df_full['volume_ma'] = df_full['volume'].rolling(window=20).mean()
    df_full['bb_middle'] = df_full['close'].rolling(window=20).mean()
    bb_std = df_full['close'].rolling(window=20).std()
    df_full['bb_upper'] = df_full['bb_middle'] + (bb_std * 2)
    df_full['bb_lower'] = df_full['bb_middle'] - (bb_std * 2)
    
    # 백테스팅
    initial_balance = 1_000_000
    balance = initial_balance
    coin_holding = 0
    buy_price = 0
    buy_balance = 0  # 매수 시점 잔고 기록
    fee_rate = 0.0005
    
    trades = []
    
    for i in range(window_size, len(df_full)):
        row = df_full.iloc[i]
        price = row['close']
        volume = row['volume']
        
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
            # 상단밴드 매도
            elif pd.notna(row['bb_upper']) and price >= row['bb_upper'] * 0.999:
                sell_reason = "상단밴드"
            
            if sell_reason:
                # 매도
                balance = coin_holding * price * (1 - fee_rate)
                profit = balance - buy_balance  # 매수 시점 잔고와 비교
                
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
                # 매수 조건
                bb_touch = price <= row['bb_lower'] * bb_tolerance
                volume_surge = volume > row['volume_ma'] * volume_multiplier
                rsi_oversold = row['rsi'] < rsi_threshold
                
                buy_signal = bb_touch and volume_surge and rsi_oversold
                
                if buy_signal and balance > 10000:
                    # 매수
                    buy_balance = balance  # 매수 시점 잔고 기록
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
        final_price = df_full.iloc[-1]['close']
        balance = coin_holding * final_price * (1 - fee_rate)
        profit = balance - buy_balance  # 매수 시점 잔고와 비교
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
    
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    win_trades = [t for t in sell_trades if t['profit'] > 0]
    loss_trades = [t for t in sell_trades if t['profit'] <= 0]
    
    win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
    
    return {
        'date': date_str,
        'total_return': total_return,
        'trade_count': len(buy_trades),
        'win_count': len(win_trades),
        'loss_count': len(loss_trades),
        'win_rate': win_rate,
        'final_balance': final_balance
    }


def optimize_rule_based_stop_take():
    """
    규칙 기반 전략의 손익 비율 최적화
    """
    print("=" * 80)
    print("🎯 규칙 기반 전략 (볼린저밴드) 손익 비율 최적화")
    print("=" * 80)
    
    # 이미 검증된 최적 파라미터 사용
    bb_tolerance = 1.001      # 하단밴드 0.1% 허용
    volume_multiplier = 3.5   # 거래량 3.5배
    rsi_threshold = 25        # RSI < 25
    
    print(f"\n📊 고정 파라미터 (이미 검증됨):")
    print(f"   하단밴드 허용: 0.1% (BB_Lower × {bb_tolerance})")
    print(f"   거래량 배수: {volume_multiplier}배")
    print(f"   RSI 과매도: < {rsi_threshold}")
    
    # 손익 비율 그리드
    stop_losses = [0.2, 0.3, 0.4, 0.5, 0.6]  # 0.2% ~ 0.6%
    take_profits = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]  # 0.4% ~ 1.2%
    
    print(f"\n🔍 손익 비율 테스트:")
    print(f"   손절: {[f'{x}%' for x in stop_losses]}")
    print(f"   익절: {[f'{x}%' for x in take_profits]}")
    print(f"   총 조합: {len(stop_losses)} × {len(take_profits)} = {len(stop_losses) * len(take_profits)}개")
    
    # 테스트 날짜 (랜덤 10일)
    from datetime import datetime, timedelta
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(10, len(all_days))))
    
    print(f"\n📅 테스트 기간: {len(test_days)}일")
    print(f"   {', '.join(test_days[:5])}{'...' if len(test_days) > 5 else ''}")
    
    # 결과 저장
    results = []
    
    # 그리드 서치
    combinations = list(product(stop_losses, take_profits))
    
    for i, (stop_loss, take_profit) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(combinations)}] 손절 {stop_loss}% | 익절 {take_profit}%")
        print(f"{'='*80}")
        
        daily_results = []
        
        for date_str in test_days:
            result = run_rule_backtest(
                date_str=date_str,
                bb_tolerance=bb_tolerance,
                volume_multiplier=volume_multiplier,
                rsi_threshold=rsi_threshold,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                data_dir="data/daily_1m",
                timeframe="1m"
            )
            
            if result:
                daily_results.append(result)
        
        if daily_results:
            avg_return = np.mean([r['total_return'] for r in daily_results])
            avg_trades = np.mean([r['trade_count'] for r in daily_results])
            avg_win_rate = np.mean([r['win_rate'] for r in daily_results])
            
            result_summary = {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'ratio': take_profit / stop_loss,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'avg_win_rate': avg_win_rate,
                'test_days': len(daily_results)
            }
            
            results.append(result_summary)
            
            print(f"   수익률: {avg_return:+.2f}% | 거래: {avg_trades:.1f}회/일 | 승률: {avg_win_rate:.1f}%")
    
    # 결과 정렬 (수익률 기준)
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    # 상위 10개 출력
    print("\n" + "=" * 80)
    print("📊 규칙 기반 전략 손익 비율 최적화 결과 (상위 10개)")
    print("=" * 80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n[{i}위]")
        print(f"   손절: {result['stop_loss']}% | 익절: {result['take_profit']}% | 손익비: {result['ratio']:.1f}:1")
        print(f"   수익률: {result['avg_return']:+.2f}%")
        print(f"   거래: {result['avg_trades']:.1f}회/일 | 승률: {result['avg_win_rate']:.1f}%")
    
    # 최적 파라미터
    if results:
        best = results[0]
        
        print("\n" + "=" * 80)
        print("🏆 최종 추천 설정")
        print("=" * 80)
        print(f"\n전략 파라미터:")
        print(f"   하단밴드 허용: 0.1% (BB_Lower × {bb_tolerance})")
        print(f"   거래량 배수: {volume_multiplier}배")
        print(f"   RSI 과매도: < {rsi_threshold}")
        print(f"\n손익 비율:")
        print(f"   손절: {best['stop_loss']}%")
        print(f"   익절: {best['take_profit']}%")
        print(f"   손익비: {best['ratio']:.1f}:1")
        print(f"\n성능:")
        print(f"   평균 수익률: {best['avg_return']:+.2f}%")
        print(f"   평균 거래: {best['avg_trades']:.1f}회/일")
        print(f"   평균 승률: {best['avg_win_rate']:.1f}%")
        print("=" * 80)
        
        # 양수 수익률 조합
        positive_results = [r for r in results if r['avg_return'] > 0]
        
        if positive_results:
            print(f"\n✅ 수익 창출 조합: {len(positive_results)}개")
            for r in positive_results[:5]:
                print(f"   손절 {r['stop_loss']}% / 익절 {r['take_profit']}%: {r['avg_return']:+.2f}%")
        else:
            print(f"\n⚠️  모든 조합이 손실입니다. 전략 재검토 필요")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    optimize_rule_based_stop_take()

