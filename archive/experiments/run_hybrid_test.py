"""
하이브리드 백테스팅 실행 스크립트
"""

import random
import numpy as np

from ml_model import MLSignalModel
from rule_engine import RuleEngine
from risk_manager import RiskManager
from hybrid_backtest import run_multi_day_backtest


def main():
    print("=" * 80)
    print("🚀 하이브리드 전략 백테스팅")
    print("=" * 80)
    
    # 1. 모델 로드
    ml_model = MLSignalModel("model/lgb_model.pkl")
    
    # 2. 규칙 엔진 초기화
    rule_engine = RuleEngine(strategy='aggressive')  # normal → aggressive
    
    # 3. 리스크 관리자 초기화
    risk_manager = RiskManager(
        stop_loss_pct=0.005,    # 0.5% 손절
        take_profit_pct=0.008,  # 0.8% 익절
        fee_rate=0.0005
    )
    
    # 4. 백테스팅 실행
    print("\n" + "=" * 80)
    print("📊 백테스팅 설정")
    print("=" * 80)
    print(f"전략: 규칙(RSI/MA-Aggressive) + ML(LightGBM)")
    print(f"데이터: 1초봉 (초단위 실시간 대응)")
    print(f"ML 임계값: 0.05 (매수), 0.02 (매도)")
    print(f"손절: 0.5%, 익절: 0.8%")
    print(f"테스트 기간: 2025-01-01 ~ 2025-05-30")
    print(f"테스트 날짜: 랜덤 10일")
    
    result = run_multi_day_backtest(
        start_date="20250101",
        end_date="20250530",
        ml_model=ml_model,
        rule_engine=rule_engine,
        risk_manager=risk_manager,
        test_days=10,
        ml_threshold=0.05,  # 라벨 불균형 대응
        data_dir="data/daily",
        timeframe="1s"  # 1초봉 사용
    )
    
    if result:
        print("\n" + "=" * 80)
        print("🎯 최종 결과")
        print("=" * 80)
        
        if result['avg_return'] > 0:
            print(f"✅ 평균 수익률: +{result['avg_return']:.2f}%")
        else:
            print(f"❌ 평균 수익률: {result['avg_return']:.2f}%")
        
        print(f"📈 평균 거래 횟수: {result['avg_trades']:.1f}회/일")
        print(f"🎲 평균 승률: {result['avg_win_rate']:.1f}%")
        
        print("\n💡 해석:")
        if result['avg_return'] > 0.5:
            print("   ✅ 수익성 있는 전략입니다!")
        elif result['avg_return'] > 0:
            print("   ⚠️  소폭 수익, 수수료 고려 시 실익 미미")
        else:
            print("   ❌ 손실 전략, 파라미터 튜닝 필요")
        
        print("\n📝 개선 방향:")
        print("   1. ML 임계값 조정 (0.65 → 0.7~0.8로 상향)")
        print("   2. 라벨 생성 시 profit_threshold 조정")
        print("   3. 규칙 엔진 전략 변경 (conservative/aggressive)")
        print("   4. 손익 비율 조정 (stop_loss, take_profit)")
        
        print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

