"""
두 전략 비교 테스트
Strategy 1: Mean Reversion + Breakout
Strategy 2: Momentum + Trend Following
"""

import numpy as np
from strategy_mean_reversion import backtest_mean_reversion
from strategy_momentum import backtest_momentum
from datetime import datetime, timedelta
import random


def compare_strategies(num_days=10):
    """
    두 전략을 동일한 날짜로 비교 테스트
    """
    print("=" * 80)
    print("STRATEGY COMPARISON TEST")
    print("=" * 80)
    print("\nStrategy 1: Mean Reversion + Breakout")
    print("  - Bollinger Bands + Volume + RSI")
    print("  - Dynamic stop/take (ATR-based)")
    print("\nStrategy 2: Momentum + Trend Following")
    print("  - Moving Averages + MACD + ADX")
    print("  - Fixed stop/take (1.0% / 2.0%)")
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
    
    results_strategy1 = []
    results_strategy2 = []
    
    for i, date_str in enumerate(test_days, 1):
        print(f"\n[{i}/{num_days}] Testing {date_str}...")
        
        # Strategy 1
        result1 = backtest_mean_reversion(date_str)
        if result1:
            results_strategy1.append(result1)
            print(f"  Strategy 1: {result1['return']:+.2f}% | {result1['num_trades']} trades | {result1['win_rate']:.1f}% win")
        else:
            print(f"  Strategy 1: No data")
        
        # Strategy 2
        result2 = backtest_momentum(date_str)
        if result2:
            results_strategy2.append(result2)
            print(f"  Strategy 2: {result2['return']:+.2f}% | {result2['num_trades']} trades | {result2['win_rate']:.1f}% win")
        else:
            print(f"  Strategy 2: No data")
    
    # 집계 및 비교
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    if not results_strategy1 and not results_strategy2:
        print("\nNo results for both strategies")
        return
    
    # Strategy 1 통계
    if results_strategy1:
        avg_return1 = np.mean([r['return'] for r in results_strategy1])
        avg_trades1 = np.mean([r['num_trades'] for r in results_strategy1])
        avg_win_rate1 = np.mean([r['win_rate'] for r in results_strategy1 if r['num_trades'] > 0])
        
        print("\n[Strategy 1: Mean Reversion + Breakout]")
        print(f"  Avg Return: {avg_return1:+.2f}%")
        print(f"  Avg Trades: {avg_trades1:.1f}/day")
        print(f"  Avg Win Rate: {avg_win_rate1:.1f}%")
        print(f"  Test Days: {len(results_strategy1)}")
    else:
        print("\n[Strategy 1: Mean Reversion + Breakout]")
        print("  No valid results")
        avg_return1 = 0
        avg_trades1 = 0
        avg_win_rate1 = 0
    
    # Strategy 2 통계
    if results_strategy2:
        avg_return2 = np.mean([r['return'] for r in results_strategy2])
        avg_trades2 = np.mean([r['num_trades'] for r in results_strategy2])
        avg_win_rate2 = np.mean([r['win_rate'] for r in results_strategy2 if r['num_trades'] > 0])
        
        print("\n[Strategy 2: Momentum + Trend Following]")
        print(f"  Avg Return: {avg_return2:+.2f}%")
        print(f"  Avg Trades: {avg_trades2:.1f}/day")
        print(f"  Avg Win Rate: {avg_win_rate2:.1f}%")
        print(f"  Test Days: {len(results_strategy2)}")
    else:
        print("\n[Strategy 2: Momentum + Trend Following]")
        print("  No valid results")
        avg_return2 = 0
        avg_trades2 = 0
        avg_win_rate2 = 0
    
    # 비교
    print("\n" + "=" * 80)
    print("WINNER")
    print("=" * 80)
    
    if results_strategy1 and results_strategy2:
        print(f"\nReturn: Strategy {'1' if avg_return1 > avg_return2 else '2'} wins ({max(avg_return1, avg_return2):+.2f}% vs {min(avg_return1, avg_return2):+.2f}%)")
        print(f"Win Rate: Strategy {'1' if avg_win_rate1 > avg_win_rate2 else '2'} wins ({max(avg_win_rate1, avg_win_rate2):.1f}% vs {min(avg_win_rate1, avg_win_rate2):.1f}%)")
        print(f"Stability: Strategy {'1' if avg_trades1 < avg_trades2 else '2'} (fewer trades = more selective)")
        
        # 종합 점수
        score1 = (avg_return1 * 0.5) + (avg_win_rate1 * 0.3) + (10 if avg_trades1 < avg_trades2 else 0)
        score2 = (avg_return2 * 0.5) + (avg_win_rate2 * 0.3) + (10 if avg_trades2 < avg_trades1 else 0)
        
        print(f"\nOverall Winner: Strategy {'1' if score1 > score2 else '2'}")
        
        if avg_return1 > avg_return2:
            print("\nRecommendation: Use Strategy 1 (Mean Reversion + Breakout)")
            print("  - Better returns")
            print("  - Adaptive to volatility (ATR-based)")
        else:
            print("\nRecommendation: Use Strategy 2 (Momentum + Trend Following)")
            print("  - Better returns")
            print("  - Captures strong trends")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    random.seed(42)
    compare_strategies(num_days=10)

