"""
ML 임계값 동적 최적화 및 Rule Engine 우선순위 재조정
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

from ml_model import MLSignalModel
from rule_engine import RuleEngine
from risk_manager import RiskManager
from hybrid_backtest import run_multi_day_backtest


def test_dynamic_threshold():
    """
    동적 임계값 테스트: quantile 기반으로 상위 N% 확률만 매수 신호로 사용
    """
    print("=" * 80)
    print("🔍 동적 임계값 최적화")
    print("=" * 80)
    
    # 모델 로드
    ml_model = MLSignalModel("model/lgb_model.pkl")
    
    # Rule Engine은 필터로만 사용 (ML이 우선)
    rule_engine = RuleEngine(strategy='aggressive')
    risk_manager = RiskManager(stop_loss_pct=0.003, take_profit_pct=0.008)
    
    # Quantile 기반 임계값 테스트
    # 상위 5%, 10%, 15%, 20% 확률만 사용
    quantiles = [0.90, 0.85, 0.80, 0.75]  # 상위 10%, 15%, 20%, 25%
    
    print("\n📊 테스트 설정:")
    print("   - 전략: ML 우선 → Rule 필터")
    print("   - Quantile 기반 임계값 (동적)")
    print("   - 손절: 0.3%, 익절: 0.8%")
    
    results = []
    
    for q in quantiles:
        pct = (1-q) * 100
        print(f"\n{'='*80}")
        print(f"🎯 상위 {pct:.0f}% 확률만 매수 신호로 사용")
        print(f"{'='*80}")
        
        # 실제로는 백테스팅 중에 동적으로 계산해야 하므로
        # 여기서는 근사치로 fixed threshold 사용
        # quantile 0.90 ≈ threshold 0.15 (경험적)
        # quantile 0.85 ≈ threshold 0.10
        # quantile 0.80 ≈ threshold 0.07
        # quantile 0.75 ≈ threshold 0.05
        
        if q == 0.90:
            threshold = 0.25
        elif q == 0.85:
            threshold = 0.20
        elif q == 0.80:
            threshold = 0.15
        else:
            threshold = 0.10
        
        print(f"   근사 임계값: {threshold:.2f}")
        
        result = run_multi_day_backtest(
            start_date="20250101",
            end_date="20250530",
            ml_model=ml_model,
            rule_engine=rule_engine,
            risk_manager=risk_manager,
            test_days=10,
            ml_threshold=threshold,
            data_dir="data/daily_1m",
            timeframe="1m"
        )
        
        if result:
            result['quantile'] = q
            result['threshold_approx'] = threshold
            results.append(result)
            
            print(f"\n   ✅ 평균 수익률: {result['avg_return']:+.2f}%")
            print(f"   📈 평균 거래: {result['avg_trades']:.1f}회/일")
            print(f"   🎲 평균 승률: {result['avg_win_rate']:.1f}%")
    
    # 결과 정렬
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📊 동적 임계값 최적화 결과")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        pct = (1-result['quantile']) * 100
        print(f"\n[{i}위] 상위 {pct:.0f}% (임계값 ≈ {result['threshold_approx']:.2f})")
        print(f"   수익률: {result['avg_return']:+.2f}%")
        print(f"   거래: {result['avg_trades']:.1f}회/일")
        print(f"   승률: {result['avg_win_rate']:.1f}%")
    
    # 최적 설정
    if results:
        best = results[0]
        pct = (1-best['quantile']) * 100
        
        print("\n" + "=" * 80)
        print("🏆 최적 설정")
        print("=" * 80)
        print(f"상위 {pct:.0f}% 확률 사용 (임계값 ≈ {best['threshold_approx']:.2f})")
        print(f"평균 수익률: {best['avg_return']:+.2f}%")
        print(f"평균 거래: {best['avg_trades']:.1f}회/일")
        print(f"평균 승률: {best['avg_win_rate']:.1f}%")
        print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    test_dynamic_threshold()

