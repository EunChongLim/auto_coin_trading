"""
새로운 두 전략 비교 테스트
Strategy 3: VWAP + Volume Profile
Strategy 4: Ichimoku Cloud
"""

import numpy as np
from strategy_vwap import backtest_vwap
from strategy_ichimoku import backtest_ichimoku
from datetime import datetime, timedelta
import random


def compare_new_strategies(num_days=10):
    """
    두 새로운 전략을 동일한 날짜로 비교 테스트
    """
    print("=" * 80)
    print("NEW STRATEGY COMPARISON TEST")
    print("=" * 80)
    print("\nStrategy 3: VWAP + Volume Profile")
    print("  - Institutional trading method")
    print("  - Buy below VWAP, sell above")
    print("  - Volume confirmation")
    print("\nStrategy 4: Ichimoku Cloud")
    print("  - 70+ years proven Japanese method")
    print("  - 5-line comprehensive analysis")
    print("  - Trend + Support/Resistance + Momentum")
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
    
    results_vwap = []
    results_ichimoku = []
    
    for i, date_str in enumerate(test_days, 1):
        print(f"\n[{i}/{num_days}] Testing {date_str}...")
        
        # Strategy 3: VWAP
        result_vwap = backtest_vwap(date_str)
        if result_vwap:
            results_vwap.append(result_vwap)
            print(f"  VWAP       : {result_vwap['return']:+.2f}% | {result_vwap['num_trades']:2d} trades | {result_vwap['win_rate']:5.1f}% win")
        else:
            print(f"  VWAP       : No data")
        
        # Strategy 4: Ichimoku
        result_ichimoku = backtest_ichimoku(date_str)
        if result_ichimoku:
            results_ichimoku.append(result_ichimoku)
            print(f"  Ichimoku   : {result_ichimoku['return']:+.2f}% | {result_ichimoku['num_trades']:2d} trades | {result_ichimoku['win_rate']:5.1f}% win")
        else:
            print(f"  Ichimoku   : No data")
    
    # 집계 및 비교
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    if not results_vwap and not results_ichimoku:
        print("\nNo results for both strategies")
        return
    
    # Strategy 3 (VWAP) 통계
    if results_vwap:
        avg_return_vwap = np.mean([r['return'] for r in results_vwap])
        avg_trades_vwap = np.mean([r['num_trades'] for r in results_vwap])
        avg_win_rate_vwap = np.mean([r['win_rate'] for r in results_vwap if r['num_trades'] > 0])
        
        print("\n[Strategy 3: VWAP + Volume Profile]")
        print(f"  Avg Return  : {avg_return_vwap:+.2f}%")
        print(f"  Avg Trades  : {avg_trades_vwap:.1f}/day")
        print(f"  Avg Win Rate: {avg_win_rate_vwap:.1f}%")
        print(f"  Test Days   : {len(results_vwap)}")
    else:
        print("\n[Strategy 3: VWAP + Volume Profile]")
        print("  No valid results")
        avg_return_vwap = 0
        avg_trades_vwap = 0
        avg_win_rate_vwap = 0
    
    # Strategy 4 (Ichimoku) 통계
    if results_ichimoku:
        avg_return_ichimoku = np.mean([r['return'] for r in results_ichimoku])
        avg_trades_ichimoku = np.mean([r['num_trades'] for r in results_ichimoku])
        avg_win_rate_ichimoku = np.mean([r['win_rate'] for r in results_ichimoku if r['num_trades'] > 0])
        
        print("\n[Strategy 4: Ichimoku Cloud]")
        print(f"  Avg Return  : {avg_return_ichimoku:+.2f}%")
        print(f"  Avg Trades  : {avg_trades_ichimoku:.1f}/day")
        print(f"  Avg Win Rate: {avg_win_rate_ichimoku:.1f}%")
        print(f"  Test Days   : {len(results_ichimoku)}")
    else:
        print("\n[Strategy 4: Ichimoku Cloud]")
        print("  No valid results")
        avg_return_ichimoku = 0
        avg_trades_ichimoku = 0
        avg_win_rate_ichimoku = 0
    
    # 비교
    print("\n" + "=" * 80)
    print("WINNER")
    print("=" * 80)
    
    if results_vwap and results_ichimoku:
        print(f"\nReturn   : {'VWAP' if avg_return_vwap > avg_return_ichimoku else 'Ichimoku':9s} wins ({max(avg_return_vwap, avg_return_ichimoku):+.2f}% vs {min(avg_return_vwap, avg_return_ichimoku):+.2f}%)")
        print(f"Win Rate : {'VWAP' if avg_win_rate_vwap > avg_win_rate_ichimoku else 'Ichimoku':9s} wins ({max(avg_win_rate_vwap, avg_win_rate_ichimoku):.1f}% vs {min(avg_win_rate_vwap, avg_win_rate_ichimoku):.1f}%)")
        print(f"Trades   : {'VWAP' if avg_trades_vwap < avg_trades_ichimoku else 'Ichimoku':9s} (fewer = more selective)")
        
        # 종합 점수
        score_vwap = (avg_return_vwap * 0.5) + (avg_win_rate_vwap * 0.3) + (10 if avg_trades_vwap < avg_trades_ichimoku else 0)
        score_ichimoku = (avg_return_ichimoku * 0.5) + (avg_win_rate_ichimoku * 0.3) + (10 if avg_trades_ichimoku < avg_trades_vwap else 0)
        
        print(f"\nOverall Winner: {'VWAP' if score_vwap > score_ichimoku else 'Ichimoku'}")
        
        # 기존 ML v2.0과 비교
        print("\n" + "=" * 80)
        print("COMPARISON WITH ML v2.0")
        print("=" * 80)
        print("\nML v2.0 Performance (baseline):")
        print("  Avg Return  : +1.46%")
        print("  Avg Trades  : 3.0/day")
        print("  Avg Win Rate: 77.4%")
        
        best_return = max(avg_return_vwap, avg_return_ichimoku)
        best_strategy = "VWAP" if avg_return_vwap > avg_return_ichimoku else "Ichimoku"
        
        print(f"\nBest New Strategy: {best_strategy}")
        print(f"  Avg Return  : {best_return:+.2f}%")
        
        if best_return > 1.46:
            print(f"\n*** NEW WINNER! ***")
            print(f"{best_strategy} beats ML v2.0 by {best_return - 1.46:+.2f}%")
            print(f"Recommendation: Replace main.py with {best_strategy} strategy")
        elif best_return > 0:
            print(f"\nPositive returns but not better than ML v2.0")
            print(f"ML v2.0 still leads by {1.46 - best_return:+.2f}%")
            print(f"Recommendation: Keep ML v2.0 in main.py")
        else:
            print(f"\nNegative returns")
            print(f"ML v2.0 significantly better by {1.46 - best_return:+.2f}%")
            print(f"Recommendation: Definitely keep ML v2.0")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    random.seed(42)
    compare_new_strategies(num_days=10)

