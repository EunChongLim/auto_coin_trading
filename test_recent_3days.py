"""
현재 모델로 최근 3일 백테스트
2025-10-22 ~ 2025-10-24 (3일)
"""

from backtest_v3 import backtest_v3_continuous
import shutil
import os
from datetime import datetime, timedelta

print("="*80)
print("현재 모델 최근 3일 백테스트")
print("="*80)

# 현재 날짜 기준 3일 전
end_date = datetime(2025, 10, 24)
start_date = end_date - timedelta(days=2)

print(f"\n테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} (3일)")
print(f"모델: lgb_model_v3.pkl (현재 RandomForest 적용)")
print(f"파라미터:")
print(f"  - buy_threshold: 0.20")
print(f"  - sell_threshold: 0.35")
print(f"  - stop_loss: 1.5%")
print(f"  - take_profit: 1.2%")
print("="*80)

# 모델이 이미 RandomForest로 교체되어 있으므로 백업/교체 불필요

try:
    result = backtest_v3_continuous(
        start_date_str=start_date.strftime("%Y%m%d"),
        num_days=3,
        buy_threshold=0.20,
        sell_threshold=0.35,
        stop_loss=1.5,
        take_profit=1.2,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("백테스트 결과 요약")
    print("="*80)
    print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} (3일)")
    print(f"수익률: {result['total_return']:+.2f}%")
    print(f"총 거래: {result['total_trades']}회")
    print(f"승률: {result['win_rate']:.1f}%")
    
    if result['total_trades'] > 0:
        wins = int(result['total_trades'] * result['win_rate'] / 100)
        losses = result['total_trades'] - wins
        print(f"승/패: {wins}승 {losses}패")
        print(f"평균 거래/일: {result['total_trades']/3:.1f}회")
    
    print("="*80)
    
    # 성능 평가
    if result['total_return'] > 0:
        print("\n[SUCCESS] 수익 발생!")
        print(f"  - 3일간 +{result['total_return']:.2f}% 수익")
        print(f"  - 승률 {result['win_rate']:.1f}%")
        if result['win_rate'] >= 50:
            print("  - 높은 승률 유지")
    else:
        print("\n[WARNING] 손실 발생")
        print(f"  - 3일간 {result['total_return']:.2f}% 손실")
        print(f"  - 승률 {result['win_rate']:.1f}%")
        print("  - 최근 시장 상황이 모델에 불리할 수 있음")
    
    print("="*80)

except Exception as e:
    print(f"\n[ERROR] 백테스트 실패: {e}")
    import traceback
    traceback.print_exc()

