"""
model_v3 파라미터 최적화
buy_threshold, sell_threshold, stop_loss, take_profit 조합 테스트
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from backtest_v3 import backtest_v3_sliding_window

import itertools
from multiprocessing import Pool, cpu_count


def test_single_params(args):
    """단일 파라미터 조합 테스트"""
    start_date, num_days, buy_th, sell_th, stop_loss, take_profit = args
    
    result = backtest_v3_sliding_window(
        start_date_str=start_date,
        num_days=num_days,
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        stop_loss=stop_loss,
        take_profit=take_profit,
        verbose=False
    )
    
    if result is None:
        return None
    
    return {
        'buy_threshold': buy_th,
        'sell_threshold': sell_th,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'return': result['total_return'],
        'win_rate': result['win_rate'],
        'num_trades': result['num_trades']
    }


def optimize_params(start_date, num_days=10, max_workers=None):
    """
    파라미터 최적화
    
    Args:
        start_date: 시작일 (YYYYMMDD)
        num_days: 테스트 일수
        max_workers: 병렬 처리 워커 수 (None=CPU 코어 수)
    """
    print(f"\n{'='*80}")
    print(f"model_v3 파라미터 최적화")
    print(f"기간: {start_date} ~ {num_days}일")
    print(f"{'='*80}\n")
    
    # 테스트할 파라미터 조합
    buy_thresholds = [0.20, 0.25, 0.30]
    sell_thresholds = [0.30, 0.35, 0.40]
    stop_losses = [1.0, 1.2, 1.5, 2.0]
    take_profits = [1.0, 1.2, 1.5, 2.0]
    
    # 모든 조합 생성
    all_combinations = list(itertools.product(
        [start_date],
        [num_days],
        buy_thresholds,
        sell_thresholds,
        stop_losses,
        take_profits
    ))
    
    total_combinations = len(all_combinations)
    print(f"총 {total_combinations}개 조합 테스트 시작...\n")
    
    # 병렬 처리
    if max_workers is None:
        max_workers = cpu_count()
    
    print(f"병렬 처리: {max_workers} workers\n")
    
    with Pool(max_workers) as pool:
        results = pool.map(test_single_params, all_combinations)
    
    # None 제거 및 정렬
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"최적화 완료! 상위 10개 결과:")
    print(f"{'='*80}\n")
    
    for i, r in enumerate(results[:10], 1):
        print(f"[{i}] Return: {r['return']:+.2f}% | Win: {r['win_rate']:.1f}% | Trades: {r['num_trades']}")
        print(f"    buy={r['buy_threshold']:.2f}, sell={r['sell_threshold']:.2f}, "
              f"stop={r['stop_loss']:.1f}%, take={r['take_profit']:.1f}%\n")
    
    # 최적 파라미터
    best = results[0]
    print(f"{'='*80}")
    print(f"최적 파라미터:")
    print(f"  - buy_threshold: {best['buy_threshold']:.2f}")
    print(f"  - sell_threshold: {best['sell_threshold']:.2f}")
    print(f"  - stop_loss: {best['stop_loss']:.1f}%")
    print(f"  - take_profit: {best['take_profit']:.1f}%")
    print(f"  - 수익률: {best['return']:+.2f}%")
    print(f"  - 승률: {best['win_rate']:.1f}%")
    print(f"  - 거래 횟수: {best['num_trades']}")
    print(f"{'='*80}\n")
    
    return best


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_params.py YYYYMMDD [num_days] [max_workers]")
        print("Example: python optimize_params.py 20250328 10 4")
        sys.exit(1)
    
    start_date = sys.argv[1]
    num_days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    optimize_params(start_date, num_days, max_workers)

