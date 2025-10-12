"""
새로운 전략 2개 동시 테스트
Strategy 5: Heikin-Ashi + Supertrend
Strategy 6: Ensemble Voting System
"""

import numpy as np
from strategy_heikin_ashi import backtest_heikin_ashi
from strategy_ensemble import backtest_ensemble
from datetime import datetime, timedelta
import random


def test_both_strategies(num_days=10):
    """
    두 새로운 전략을 동시 테스트
    """
    print("=" * 80)
    print("NEW STRATEGIES TEST")
    print("=" * 80)
    print("\nStrategy 5: Heikin-Ashi + Supertrend")
    print("  - Noise reduction via HA candles")
    print("  - Clear trend signals via Supertrend")
    print("\nStrategy 6: Ensemble Voting (4 strategies)")
    print("  - Mean Reversion + Momentum + Stochastic + Volume")
    print("  - 2+ votes required for action")
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
    print(f"   {', '.join(test_days[:5])}...")
    print("\n" + "=" * 80)
    
    results_ha = []
    results_ensemble = []
    
    for i, date_str in enumerate(test_days, 1):
        print(f"\n[{i}/{num_days}] Testing {date_str}...")
        
        # Strategy 5: Heikin-Ashi
        result_ha = backtest_heikin_ashi(date_str)
        if result_ha:
            results_ha.append(result_ha)
            print(f"  Heikin-Ashi: {result_ha['return']:+6.2f}% | {result_ha['num_trades']:2d} trades | {result_ha['win_rate']:5.1f}% win")
        else:
            print(f"  Heikin-Ashi: No data")
        
        # Strategy 6: Ensemble
        result_ensemble = backtest_ensemble(date_str)
        if result_ensemble:
            results_ensemble.append(result_ensemble)
            print(f"  Ensemble   : {result_ensemble['return']:+6.2f}% | {result_ensemble['num_trades']:2d} trades | {result_ensemble['win_rate']:5.1f}% win")
        else:
            print(f"  Ensemble   : No data")
    
    # 집계
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if results_ha:
        avg_return_ha = np.mean([r['return'] for r in results_ha])
        avg_trades_ha = np.mean([r['num_trades'] for r in results_ha])
        avg_win_rate_ha = np.mean([r['win_rate'] for r in results_ha if r['num_trades'] > 0])
        
        print("\n[Strategy 5: Heikin-Ashi + Supertrend]")
        print(f"  Avg Return  : {avg_return_ha:+.2f}%")
        print(f"  Avg Trades  : {avg_trades_ha:.1f}/day")
        print(f"  Avg Win Rate: {avg_win_rate_ha:.1f}%")
    else:
        avg_return_ha = -999
        avg_trades_ha = 0
        avg_win_rate_ha = 0
    
    if results_ensemble:
        avg_return_ensemble = np.mean([r['return'] for r in results_ensemble])
        avg_trades_ensemble = np.mean([r['num_trades'] for r in results_ensemble])
        avg_win_rate_ensemble = np.mean([r['win_rate'] for r in results_ensemble if r['num_trades'] > 0])
        
        print("\n[Strategy 6: Ensemble Voting]")
        print(f"  Avg Return  : {avg_return_ensemble:+.2f}%")
        print(f"  Avg Trades  : {avg_trades_ensemble:.1f}/day")
        print(f"  Avg Win Rate: {avg_win_rate_ensemble:.1f}%")
    else:
        avg_return_ensemble = -999
        avg_trades_ensemble = 0
        avg_win_rate_ensemble = 0
    
    # 비교
    print("\n" + "=" * 80)
    print("COMPARISON WITH ALL STRATEGIES")
    print("=" * 80)
    
    all_strategies = [
        ("ML v2.0", 1.46, 3.0, 77.4),
        ("Heikin-Ashi", avg_return_ha, avg_trades_ha, avg_win_rate_ha),
        ("Ensemble", avg_return_ensemble, avg_trades_ensemble, avg_win_rate_ensemble),
        ("Ichimoku", -0.90, 8.1, 13.6),
        ("VWAP", -1.82, 26.9, 22.6),
        ("Momentum", -2.19, 16.5, 8.2),
        ("Mean Reversion", -3.16, 33.0, 13.9),
    ]
    
    # 수익률 기준 정렬
    all_strategies.sort(key=lambda x: x[1], reverse=True)
    
    print("\nRanking by Return:")
    for rank, (name, ret, trades, win_rate) in enumerate(all_strategies, 1):
        status = "***" if ret > 0 else ""
        print(f"  {rank}. {name:20s}: {ret:+6.2f}% | {trades:4.1f} trades/day | {win_rate:5.1f}% win {status}")
    
    # 최종 추천
    best = all_strategies[0]
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    print(f"\nBest Strategy: {best[0]}")
    print(f"  - Return   : {best[1]:+.2f}%")
    print(f"  - Trades   : {best[2]:.1f}/day")
    print(f"  - Win Rate : {best[3]:.1f}%")
    
    if best[0] == "ML v2.0":
        print("\nML v2.0 remains the champion!")
        print("Continue running main.py")
    else:
        print(f"\n*** NEW CHAMPION: {best[0]} ***")
        print(f"Consider updating main.py with this strategy")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    random.seed(42)
    test_both_strategies(num_days=10)

