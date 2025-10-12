"""
하이브리드 전략 파라미터 최적화
Grid Search를 통해 최적 조합 탐색
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from itertools import product
import json

from ml_model import MLSignalModel
from rule_engine import RuleEngine
from risk_manager import RiskManager
from hybrid_backtest import run_multi_day_backtest


def grid_search_hybrid_params():
    """
    하이브리드 전략 파라미터 그리드 서치
    """
    print("=" * 80)
    print("🔍 하이브리드 전략 파라미터 최적화")
    print("=" * 80)
    
    # 모델 로드
    ml_model = MLSignalModel("model/lgb_model.pkl")
    
    # 파라미터 그리드 정의
    param_grid = {
        'ml_buy_threshold': [0.01, 0.03, 0.05, 0.10],  # ML 매수 임계값
        'rule_strategy': ['conservative', 'normal', 'aggressive'],  # 규칙 전략
        'stop_loss_pct': [0.003, 0.005, 0.008],  # 손절 (0.3%, 0.5%, 0.8%)
        'take_profit_pct': [0.005, 0.008, 0.010],  # 익절 (0.5%, 0.8%, 1.0%)
    }
    
    # 전체 조합 수
    total_combinations = (
        len(param_grid['ml_buy_threshold']) * 
        len(param_grid['rule_strategy']) * 
        len(param_grid['stop_loss_pct']) * 
        len(param_grid['take_profit_pct'])
    )
    
    print(f"\n📊 파라미터 그리드:")
    print(f"   ML 매수 임계값: {param_grid['ml_buy_threshold']}")
    print(f"   규칙 전략: {param_grid['rule_strategy']}")
    print(f"   손절: {[f'{x*100:.1f}%' for x in param_grid['stop_loss_pct']]}")
    print(f"   익절: {[f'{x*100:.1f}%' for x in param_grid['take_profit_pct']]}")
    print(f"\n🎯 총 조합: {total_combinations}개")
    
    # 2단계 백테스팅 (학습 + 검증)
    train_start = "20250101"
    train_end = "20250331"
    val_start = "20250401"
    val_end = "20250530"
    
    print(f"\n📅 학습 기간: {train_start} ~ {train_end} (랜덤 5일)")
    print(f"📅 검증 기간: {val_start} ~ {val_end} (랜덤 5일)")
    
    # 결과 저장
    results = []
    
    # 그리드 서치
    print("\n" + "=" * 80)
    print("🔄 그리드 서치 시작")
    print("=" * 80)
    
    combinations = list(product(
        param_grid['ml_buy_threshold'],
        param_grid['rule_strategy'],
        param_grid['stop_loss_pct'],
        param_grid['take_profit_pct']
    ))
    
    for i, (ml_threshold, strategy, stop_loss, take_profit) in enumerate(combinations, 1):
        print(f"\n[{i}/{total_combinations}] 테스트 중...")
        print(f"   ML={ml_threshold:.2f} | 전략={strategy} | 손절={stop_loss*100:.1f}% | 익절={take_profit*100:.1f}%")
        
        try:
            # 규칙 엔진 & 리스크 관리자 생성
            rule_engine = RuleEngine(strategy=strategy)
            risk_manager = RiskManager(
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                fee_rate=0.0005
            )
            
            # 학습 백테스팅
            train_result = run_multi_day_backtest(
                start_date=train_start,
                end_date=train_end,
                ml_model=ml_model,
                rule_engine=rule_engine,
                risk_manager=risk_manager,
                test_days=5,
                ml_threshold=ml_threshold,
                data_dir="data/daily_1m",
                timeframe="1m"
            )
            
            # 검증 백테스팅
            val_result = run_multi_day_backtest(
                start_date=val_start,
                end_date=val_end,
                ml_model=ml_model,
                rule_engine=rule_engine,
                risk_manager=risk_manager,
                test_days=5,
                ml_threshold=ml_threshold,
                data_dir="data/daily_1m",
                timeframe="1m"
            )
            
            if train_result and val_result:
                result = {
                    'ml_threshold': ml_threshold,
                    'rule_strategy': strategy,
                    'stop_loss_pct': stop_loss,
                    'take_profit_pct': take_profit,
                    'train_return': train_result['avg_return'],
                    'train_trades': train_result['avg_trades'],
                    'train_win_rate': train_result['avg_win_rate'],
                    'val_return': val_result['avg_return'],
                    'val_trades': val_result['avg_trades'],
                    'val_win_rate': val_result['avg_win_rate'],
                    'avg_return': (train_result['avg_return'] + val_result['avg_return']) / 2
                }
                
                results.append(result)
                
                print(f"   학습: {train_result['avg_return']:+.2f}% | 검증: {val_result['avg_return']:+.2f}%")
        
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            continue
    
    # 결과 정렬 (검증 수익률 기준)
    results.sort(key=lambda x: x['val_return'], reverse=True)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("📊 최적화 결과 (상위 10개)")
    print("=" * 80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n[{i}위]")
        print(f"   ML 임계값: {result['ml_threshold']:.2f}")
        print(f"   규칙 전략: {result['rule_strategy']}")
        print(f"   손절/익절: {result['stop_loss_pct']*100:.1f}% / {result['take_profit_pct']*100:.1f}%")
        print(f"   학습 수익률: {result['train_return']:+.2f}% (거래 {result['train_trades']:.1f}회/일, 승률 {result['train_win_rate']:.1f}%)")
        print(f"   검증 수익률: {result['val_return']:+.2f}% (거래 {result['val_trades']:.1f}회/일, 승률 {result['val_win_rate']:.1f}%)")
        print(f"   평균 수익률: {result['avg_return']:+.2f}%")
    
    # JSON 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"optimization_result_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    # 최적 파라미터
    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print("🏆 최적 파라미터")
        print("=" * 80)
        print(f"ML 임계값: {best['ml_threshold']:.2f}")
        print(f"규칙 전략: {best['rule_strategy']}")
        print(f"손절: {best['stop_loss_pct']*100:.1f}%")
        print(f"익절: {best['take_profit_pct']*100:.1f}%")
        print(f"\n검증 수익률: {best['val_return']:+.2f}%")
        print(f"평균 수익률: {best['avg_return']:+.2f}%")
        print("=" * 80)
        
        # 최적 파라미터로 full 테스트
        print("\n" + "=" * 80)
        print("🧪 최적 파라미터로 전체 기간 테스트 (랜덤 10일)")
        print("=" * 80)
        
        rule_engine = RuleEngine(strategy=best['rule_strategy'])
        risk_manager = RiskManager(
            stop_loss_pct=best['stop_loss_pct'],
            take_profit_pct=best['take_profit_pct'],
            fee_rate=0.0005
        )
        
        full_result = run_multi_day_backtest(
            start_date="20250101",
            end_date="20250530",
            ml_model=ml_model,
            rule_engine=rule_engine,
            risk_manager=risk_manager,
            test_days=10,
            ml_threshold=best['ml_threshold'],
            data_dir="data/daily_1m",
            timeframe="1m"
        )
        
        if full_result:
            print(f"\n✅ 전체 기간 수익률: {full_result['avg_return']:+.2f}%")
            print(f"   거래 횟수: {full_result['avg_trades']:.1f}회/일")
            print(f"   승률: {full_result['avg_win_rate']:.1f}%")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    grid_search_hybrid_params()

