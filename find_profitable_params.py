"""
현재 RandomForest 모델로 다양한 파라미터 조합 테스트
최근 3일 데이터에서 수익이 나는 조합 찾기
"""

from backtest_v3 import backtest_v3_continuous
import itertools

print("="*80)
print("파라미터 최적화 - 최근 3일 백테스트 (2025-10-22~24)")
print("="*80)
print("\n모델: lgb_model_v3.pkl (RandomForest)")
print("목표: 수익이 나는 파라미터 조합 찾기")
print("="*80)

# 파라미터 그리드
buy_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
sell_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]
stop_losses = [1.0, 1.5, 2.0]
take_profits = [0.8, 1.0, 1.2, 1.5]

print(f"\n테스트 범위:")
print(f"  buy_threshold: {buy_thresholds}")
print(f"  sell_threshold: {sell_thresholds}")
print(f"  stop_loss: {stop_losses}")
print(f"  take_profit: {take_profits}")

total_combinations = len(buy_thresholds) * len(sell_thresholds) * len(stop_losses) * len(take_profits)
print(f"\n총 조합: {total_combinations}개")
print("="*80)

results = []
profitable_combos = []

current = 0
for buy, sell, stop, take in itertools.product(buy_thresholds, sell_thresholds, stop_losses, take_profits):
    current += 1
    
    # 무의미한 조합 스킵
    if buy >= sell:
        continue
    if stop >= take:
        continue
    
    print(f"\n[{current}/{total_combinations}] buy={buy}, sell={sell}, stop={stop}%, take={take}%")
    
    try:
        result = backtest_v3_continuous(
            start_date_str="20251022",
            num_days=3,
            buy_threshold=buy,
            sell_threshold=sell,
            stop_loss=stop,
            take_profit=take,
            verbose=False
        )
        
        result_data = {
            'buy': buy,
            'sell': sell,
            'stop': stop,
            'take': take,
            'return': result['total_return'],
            'trades': result['total_trades'],
            'win_rate': result['win_rate']
        }
        results.append(result_data)
        
        profit_marker = " *** PROFIT ***" if result['total_return'] > 0 else ""
        print(f"   Return: {result['total_return']:+.2f}% | Trades: {result['total_trades']} | WinRate: {result['win_rate']:.1f}%{profit_marker}")
        
        if result['total_return'] > 0:
            profitable_combos.append(result_data)
            print(f"   >>> 수익 조합 발견! <<<")
    
    except Exception as e:
        print(f"   ERROR: {e}")

# 결과 분석
print("\n\n" + "="*80)
print("최종 결과")
print("="*80)

# 수익률 순 정렬
results_sorted = sorted(results, key=lambda x: x['return'], reverse=True)

print(f"\n[TOP 20] 수익률 순위")
print("-"*80)
print(f"{'Rank':<6} {'Buy':<8} {'Sell':<8} {'Stop':<8} {'Take':<8} {'Return':<12} {'Trades':<10} {'WinRate':<10}")
print("-"*80)

for i, r in enumerate(results_sorted[:20], 1):
    marker = " PROFIT" if r['return'] > 0 else ""
    print(f"{i:<6} {r['buy']:<8.2f} {r['sell']:<8.2f} {r['stop']:<8.1f} {r['take']:<8.1f} {r['return']:>+9.2f}% {r['trades']:>6} {r['win_rate']:>8.1f}%{marker}")

# 수익 조합 요약
print("\n" + "="*80)
if profitable_combos:
    print(f"수익 조합 발견: {len(profitable_combos)}개")
    print("="*80)
    
    for i, c in enumerate(profitable_combos[:10], 1):
        print(f"\n{i}. buy={c['buy']}, sell={c['sell']}, stop={c['stop']}%, take={c['take']}%")
        print(f"   Return: {c['return']:+.2f}% | Trades: {c['trades']} | Win Rate: {c['win_rate']:.1f}%")
    
    # 최고 수익 조합
    best = profitable_combos[0]
    print("\n" + "="*80)
    print("최고 수익 조합")
    print("="*80)
    print(f"buy_threshold: {best['buy']}")
    print(f"sell_threshold: {best['sell']}")
    print(f"stop_loss: {best['stop']}%")
    print(f"take_profit: {best['take']}%")
    print(f"\n성과:")
    print(f"  수익률: {best['return']:+.2f}%")
    print(f"  승률: {best['win_rate']:.1f}%")
    print(f"  거래: {best['trades']}회")
    
else:
    print("수익 조합을 찾지 못했습니다.")
    print("="*80)
    print("\n최선의 조합 (최소 손실):")
    best = results_sorted[0]
    print(f"  buy={best['buy']}, sell={best['sell']}, stop={best['stop']}%, take={best['take']}%")
    print(f"  수익률: {best['return']:+.2f}%")
    print(f"  승률: {best['win_rate']:.1f}%")
    print(f"  거래: {best['trades']}회")
    
    print("\n분석:")
    print("  최근 3일 시장이 모든 파라미터 조합에서 불리합니다.")
    print("  권장사항:")
    print("  1. 더 긴 기간으로 테스트 (7일, 10일)")
    print("  2. 다른 기간으로 테스트 (다른 3일)")
    print("  3. 시장 상황 변화 대기")

print("="*80)

