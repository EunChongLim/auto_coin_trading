"""
v7 파라미터 최적화 진행 상황 체크
"""
import os
import pickle
from datetime import datetime

result_path = "optimization_result_v7.pkl"

print("=" * 60)
print("v7 파라미터 최적화 진행 상황")
print("=" * 60)

if os.path.exists(result_path):
    print(f"\n[완료] 최적화 완료!")
    
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    
    best_params = data['best_params']
    best_perf = data['best_performance']
    
    print(f"\n[최종 추천 파라미터]")
    print(f"  buy_threshold = {best_params['buy_threshold']}")
    print(f"  stop_loss = {best_params['stop_loss']}")
    print(f"  take_profit = {best_params['take_profit']}")
    print(f"  time_limit = {best_params['time_limit']}")
    
    print(f"\n[성과]")
    print(f"  평균 수익률: {best_perf['avg_return']:+.2f}%")
    print(f"  수익률 범위: {best_perf['min_return']:+.2f}% ~ {best_perf['max_return']:+.2f}%")
    print(f"  표준편차: {best_perf['std_return']:.2f}%")
    print(f"  검증 기간: {best_perf['periods']}개")
    
    print(f"\n[Top 3 파라미터]")
    for rank, (params, perf) in enumerate(data['top_10'][:3], 1):
        print(f"\n{rank}. {params}")
        print(f"   평균: {perf['avg_return']:+.2f}% | 표준편차: {perf['std_return']:.2f}%")
else:
    print(f"\n[진행중] 최적화 진행 중...")
    print(f"\n256개 파라미터 조합 × 3개 기간 = 768회 백테스트")
    print(f"예상 소요 시간: 30~60분")
    print(f"\n이 스크립트를 다시 실행하여 진행 상황 확인:")
    print(f"  python check_optimization.py")

print("\n" + "=" * 60)

