"""
VWAP 전략 파라미터 최적화
"""

import numpy as np
from strategy_vwap import backtest_vwap
from datetime import datetime, timedelta
import random
import itertools


def optimize_vwap_params(num_days=10):
    """
    VWAP 전략 파라미터 Grid Search
    """
    print("=" * 80)
    print("VWAP Strategy Parameter Optimization")
    print("=" * 80)
    
    # 테스트 날짜 준비
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(num_days, len(all_days))))
    
    print(f"\nTest Period: {num_days} days")
    print(f"   {', '.join(test_days[:5])}...")
    
    # 파라미터 그리드
    vwap_ranges = [
        (-1.5, 0.3),  # 매우 넓음
        (-1.2, 0.2),  # 넓음
        (-1.0, 0.0),  # 중간
        (-0.8, -0.2), # 좁음 (VWAP 아래만)
    ]
    
    volume_multipliers = [1.0, 1.2, 1.5]
    rsi_maxs = [50, 55, 60, 65]
    
    total_combos = len(vwap_ranges) * len(volume_multipliers) * len(rsi_maxs)
    
    print(f"\n[Parameter Grid]")
    print(f"  - VWAP ranges: {len(vwap_ranges)} options")
    print(f"  - Volume multipliers: {volume_multipliers}")
    print(f"  - RSI max: {rsi_maxs}")
    print(f"  - Total combinations: {total_combos}")
    
    print("\n" + "=" * 80)
    print("Testing...")
    print("=" * 80)
    
    all_results = []
    combo_idx = 0
    
    for vwap_range, vol_mult, rsi_max in itertools.product(vwap_ranges, volume_multipliers, rsi_maxs):
        combo_idx += 1
        
        print(f"\n[{combo_idx}/{total_combos}] VWAP={vwap_range}, Vol={vol_mult}x, RSI<{rsi_max}")
        
        daily_results = []
        
        for date_str in test_days:
            result = backtest_vwap(
                date_str,
                vwap_range=vwap_range,
                volume_multiplier=vol_mult,
                rsi_max=rsi_max
            )
            
            if result:
                daily_results.append(result)
        
        if daily_results:
            avg_return = np.mean([r['return'] for r in daily_results])
            avg_trades = np.mean([r['num_trades'] for r in daily_results])
            avg_win_rate = np.mean([r['win_rate'] for r in daily_results if r['num_trades'] > 0])
            
            print(f"   Return: {avg_return:+.2f}% | Trades: {avg_trades:.1f}/day | Win Rate: {avg_win_rate:.1f}%")
            
            all_results.append({
                'vwap_range': vwap_range,
                'volume_multiplier': vol_mult,
                'rsi_max': rsi_max,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'avg_win_rate': avg_win_rate,
                'score': avg_return  # 정렬 기준
            })
        else:
            print(f"   No valid results")
    
    # 결과 정렬
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 상위 10개 출력
    print("\n" + "=" * 80)
    print("Top 10 Results")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:10], 1):
        print(f"\n[Rank {i}]")
        print(f"   VWAP Range: {result['vwap_range']}")
        print(f"   Volume Multiplier: {result['volume_multiplier']}x")
        print(f"   RSI Max: {result['rsi_max']}")
        print(f"   ---")
        print(f"   Avg Return: {result['avg_return']:+.2f}%")
        print(f"   Avg Trades: {result['avg_trades']:.1f}/day")
        print(f"   Avg Win Rate: {result['avg_win_rate']:.1f}%")
    
    # 최적 설정
    if all_results:
        best = all_results[0]
        
        print("\n" + "=" * 80)
        print("Best Configuration")
        print("=" * 80)
        
        print(f"\nParameters:")
        print(f"   vwap_range = {best['vwap_range']}")
        print(f"   volume_multiplier = {best['volume_multiplier']}")
        print(f"   rsi_max = {best['rsi_max']}")
        
        print(f"\nPerformance:")
        print(f"   Avg Return: {best['avg_return']:+.2f}%")
        print(f"   Avg Trades: {best['avg_trades']:.1f}/day")
        print(f"   Avg Win Rate: {best['avg_win_rate']:.1f}%")
        
        # ML v2.0과 비교
        print("\n" + "=" * 80)
        print("Comparison with ML v2.0")
        print("=" * 80)
        print("\nML v2.0: +1.46% (3.0 trades/day, 77.4% win rate)")
        print(f"VWAP  : {best['avg_return']:+.2f}% ({best['avg_trades']:.1f} trades/day, {best['avg_win_rate']:.1f}% win rate)")
        
        if best['avg_return'] > 1.46:
            print("\n*** VWAP WINS! ***")
            print(f"VWAP beats ML v2.0 by {best['avg_return'] - 1.46:+.2f}%")
        elif best['avg_return'] > 0:
            print("\nPositive but not better than ML v2.0")
        else:
            print("\nNegative returns - ML v2.0 is better")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    random.seed(42)
    optimize_vwap_params(num_days=10)

