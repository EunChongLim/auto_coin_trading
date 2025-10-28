"""
model_v4 파라미터 최적화 (B규칙 전략)
ml_threshold, atr_multiplier, risk% 등 최적화
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from backtest_v4 import backtest_v4_with_rules

import itertools
from multiprocessing import Pool, cpu_count


def test_single_params(args):
    """단일 파라미터 조합 테스트"""
    start_date, num_days, ml_buy, ml_sell, atr_mult, risk, use_ml, use_partial = args
    
    result = backtest_v4_with_rules(
        start_date_str=start_date,
        num_days=num_days,
        ml_buy_threshold=ml_buy,
        ml_sell_threshold=ml_sell,
        atr_stop_multiplier=atr_mult,
        risk_pct=risk,
        use_ml=use_ml,
        use_partial_exit=use_partial,
        verbose=False
    )
    
    if result is None:
        return None
    
    return {
        'ml_buy_threshold': ml_buy,
        'ml_sell_threshold': ml_sell,
        'atr_stop_multiplier': atr_mult,
        'risk_pct': risk,
        'use_ml': use_ml,
        'use_partial_exit': use_partial,
        'return': result['total_return'],
        'win_rate': result['win_rate'],
        'num_trades': result['num_trades'],
        'partial_exits': result.get('partial_exits', 0)
    }


def optimize_params_v4(start_date, num_days=10, max_workers=None):
    """
    model_v4 파라미터 최적화 (B규칙 전략)
    """
    print(f"\n{'='*80}")
    print(f"model_v4 파라미터 최적화 (B규칙 전략)")
    print(f"기간: {start_date} ~ {num_days}일")
    print(f"{'='*80}\n")
    
    # 테스트할 파라미터 조합
    ml_buy_thresholds = [0.20, 0.25, 0.30]
    ml_sell_thresholds = [0.30, 0.35, 0.40]
    atr_multipliers = [1.0, 1.2, 1.5]
    risk_pcts = [0.5, 1.0]
    use_ml_options = [True, False]  # ML 보조 ON/OFF
    use_partial_options = [True, False]  # 부분청산 ON/OFF
    
    # 모든 조합 생성
    all_combinations = list(itertools.product(
        [start_date],
        [num_days],
        ml_buy_thresholds,
        ml_sell_thresholds,
        atr_multipliers,
        risk_pcts,
        use_ml_options,
        use_partial_options
    ))
    
    total_combinations = len(all_combinations)
    print(f"총 {total_combinations}개 조합 테스트 시작...\n")
    
    # 병렬 처리
    if max_workers is None:
        max_workers = min(cpu_count(), 4)  # 최대 4개 워커
    
    print(f"병렬 처리: {max_workers} workers\n")
    
    with Pool(max_workers) as pool:
        results = pool.map(test_single_params, all_combinations)
    
    # None 제거 및 정렬
    results = [r for r in results if r is not None and r['num_trades'] > 0]
    
    if len(results) == 0:
        print("\n[WARN] 거래가 발생한 조합이 없습니다!")
        print("  - B규칙 조건이 너무 엄격할 수 있습니다")
        print("  - ML 없이 (use_ml=False) 테스트 권장")
        print("  - 또는 strategy_rules.py에서 조건 완화 필요\n")
        return None
    
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"최적화 완료! 상위 10개 결과:")
    print(f"{'='*80}\n")
    
    for i, r in enumerate(results[:10], 1):
        print(f"[{i}] Return: {r['return']:+.2f}% | Win: {r['win_rate']:.1f}% | Trades: {r['num_trades']} | Partial: {r['partial_exits']}")
        print(f"    ML buy={r['ml_buy_threshold']:.2f}, sell={r['ml_sell_threshold']:.2f}, "
              f"ATR={r['atr_stop_multiplier']:.1f}x, risk={r['risk_pct']:.1f}%")
        print(f"    ML보조: {'ON' if r['use_ml'] else 'OFF'}, 부분청산: {'ON' if r['use_partial_exit'] else 'OFF'}\n")
    
    # 최적 파라미터
    best = results[0]
    print(f"{'='*80}")
    print(f"최적 파라미터:")
    print(f"  - ml_buy_threshold: {best['ml_buy_threshold']:.2f}")
    print(f"  - ml_sell_threshold: {best['ml_sell_threshold']:.2f}")
    print(f"  - atr_stop_multiplier: {best['atr_stop_multiplier']:.1f}x")
    print(f"  - risk_pct: {best['risk_pct']:.1f}%")
    print(f"  - use_ml: {best['use_ml']}")
    print(f"  - use_partial_exit: {best['use_partial_exit']}")
    print(f"  - 수익률: {best['return']:+.2f}%")
    print(f"  - 승률: {best['win_rate']:.1f}%")
    print(f"  - 거래 횟수: {best['num_trades']}")
    print(f"  - 부분청산: {best['partial_exits']}회")
    print(f"{'='*80}\n")
    
    return best


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_params_v4.py YYYYMMDD [num_days] [max_workers]")
        print("Example: python optimize_params_v4.py 20250328 10 4")
        sys.exit(1)
    
    start_date = sys.argv[1]
    num_days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    optimize_params_v4(start_date, num_days, max_workers)
