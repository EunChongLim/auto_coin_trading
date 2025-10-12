"""
손익 비율 최적화 (Stop Loss & Take Profit)
개선된 ML 모델 + 다양한 손익 비율 조합 테스트
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from itertools import product

from ml_model import MLSignalModel
from rule_engine import RuleEngine
from risk_manager import RiskManager
from hybrid_backtest import run_multi_day_backtest


def optimize_stop_take_profit():
    """
    손익 비율 그리드 서치
    """
    print("=" * 80)
    print("🎯 손익 비율 최적화 (Grid Search)")
    print("=" * 80)
    
    # 모델 로드
    ml_model = MLSignalModel("model/lgb_model.pkl")
    rule_engine = RuleEngine(strategy='aggressive')
    
    # 손익 비율 그리드 정의
    stop_losses = [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
    take_profits = [0.005, 0.008, 0.010, 0.015, 0.020]  # 0.5%, 0.8%, 1.0%, 1.5%, 2.0%
    
    # ML 임계값 (이전 최적값)
    ml_threshold = 0.15
    
    print(f"\n📊 테스트 설정:")
    print(f"   ML 임계값: {ml_threshold:.2f} (고정)")
    print(f"   규칙 전략: aggressive")
    print(f"   손절 옵션: {[f'{x*100:.1f}%' for x in stop_losses]}")
    print(f"   익절 옵션: {[f'{x*100:.1f}%' for x in take_profits]}")
    print(f"   총 조합: {len(stop_losses)} × {len(take_profits)} = {len(stop_losses) * len(take_profits)}개")
    
    # 결과 저장
    results = []
    
    # 그리드 서치
    combinations = list(product(stop_losses, take_profits))
    
    for i, (stop_loss, take_profit) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(combinations)}] 손절 {stop_loss*100:.1f}% | 익절 {take_profit*100:.1f}%")
        print(f"{'='*80}")
        
        # Risk Manager 생성
        risk_manager = RiskManager(
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            fee_rate=0.0005
        )
        
        try:
            # 백테스팅 (10일)
            result = run_multi_day_backtest(
                start_date="20250101",
                end_date="20250530",
                ml_model=ml_model,
                rule_engine=rule_engine,
                risk_manager=risk_manager,
                test_days=10,
                ml_threshold=ml_threshold,
                data_dir="data/daily_1m",
                timeframe="1m"
            )
            
            if result:
                result['stop_loss'] = stop_loss
                result['take_profit'] = take_profit
                result['ratio'] = take_profit / stop_loss  # 손익비
                results.append(result)
                
                print(f"   수익률: {result['avg_return']:+.2f}% | 거래: {result['avg_trades']:.1f}회/일 | 승률: {result['avg_win_rate']:.1f}%")
        
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            continue
    
    # 결과 정렬 (수익률 기준)
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    # 상위 10개 출력
    print("\n" + "=" * 80)
    print("📊 손익 비율 최적화 결과 (상위 10개)")
    print("=" * 80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n[{i}위]")
        print(f"   손절: {result['stop_loss']*100:.1f}% | 익절: {result['take_profit']*100:.1f}% | 손익비: {result['ratio']:.1f}:1")
        print(f"   수익률: {result['avg_return']:+.2f}%")
        print(f"   거래: {result['avg_trades']:.1f}회/일 | 승률: {result['avg_win_rate']:.1f}%")
    
    # 최적 파라미터
    if results:
        best = results[0]
        
        print("\n" + "=" * 80)
        print("🏆 최적 손익 비율")
        print("=" * 80)
        print(f"손절: {best['stop_loss']*100:.1f}%")
        print(f"익절: {best['take_profit']*100:.1f}%")
        print(f"손익비: {best['ratio']:.1f}:1")
        print(f"\n평균 수익률: {best['avg_return']:+.2f}%")
        print(f"평균 거래: {best['avg_trades']:.1f}회/일")
        print(f"평균 승률: {best['avg_win_rate']:.1f}%")
        print("=" * 80)
        
        # 승률별 분석
        print("\n" + "=" * 80)
        print("📈 승률별 분석")
        print("=" * 80)
        
        high_winrate = [r for r in results if r['avg_win_rate'] >= 30]
        if high_winrate:
            high_winrate.sort(key=lambda x: x['avg_return'], reverse=True)
            best_high = high_winrate[0]
            print(f"\n높은 승률 (≥30%) 중 최고 수익:")
            print(f"   손절: {best_high['stop_loss']*100:.1f}% | 익절: {best_high['take_profit']*100:.1f}%")
            print(f"   수익률: {best_high['avg_return']:+.2f}% | 승률: {best_high['avg_win_rate']:.1f}%")
        
        # 손익비별 분석
        print("\n" + "=" * 80)
        print("💰 손익비별 분석")
        print("=" * 80)
        
        high_ratio = [r for r in results if r['ratio'] >= 3.0]
        if high_ratio:
            high_ratio.sort(key=lambda x: x['avg_return'], reverse=True)
            best_ratio = high_ratio[0]
            print(f"\n높은 손익비 (≥3:1) 중 최고 수익:")
            print(f"   손절: {best_ratio['stop_loss']*100:.1f}% | 익절: {best_ratio['take_profit']*100:.1f}%")
            print(f"   수익률: {best_ratio['avg_return']:+.2f}% | 손익비: {best_ratio['ratio']:.1f}:1")
        
        # JSON 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"stop_take_optimization_{timestamp}.json"
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {output_file}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    optimize_stop_take_profit()

