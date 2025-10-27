#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 모델 파라미터 최적화 스크립트
- 현재 RandomForest 모델로 다양한 파라미터 조합 테스트
- 최근 3일 데이터로 백테스트
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 기존 모듈 import
from backtest_v3 import backtest_v3_continuous

def test_parameter_combinations():
    """다양한 파라미터 조합 테스트"""
    print("=" * 80)
    print("기존 모델 파라미터 최적화 시작")
    print("=" * 80)
    
    # 테스트할 파라미터 조합
    buy_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    sell_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]
    stop_losses = [1.0, 1.2, 1.5, 1.8, 2.0]
    take_profits = [0.8, 1.0, 1.2, 1.5, 1.8]
    
    # 백테스트 설정
    start_date_str = "20251022"  # 최근 3일
    num_days = 3
    model_path = "model/lgb_model_v3.pkl"  # 현재 모델
    
    print(f"백테스트 설정:")
    print(f"   기간: {start_date_str} ({num_days}일)")
    print(f"   모델: {model_path}")
    print(f"   테스트 조합: {len(buy_thresholds)} x {len(sell_thresholds)} x {len(stop_losses)} x {len(take_profits)} = {len(buy_thresholds) * len(sell_thresholds) * len(stop_losses) * len(take_profits)}개")
    
    results = []
    total_combinations = len(buy_thresholds) * len(sell_thresholds) * len(stop_losses) * len(take_profits)
    current = 0
    
    # 모든 조합 테스트
    for buy_threshold, sell_threshold, stop_loss, take_profit in product(
        buy_thresholds, sell_thresholds, stop_losses, take_profits
    ):
        current += 1
        print(f"\n[{current}/{total_combinations}] 테스트 중...")
        print(f"   Buy: {buy_threshold}, Sell: {sell_threshold}, Stop: {stop_loss}%, Take: {take_profit}%")
        
        try:
            # 백테스트 실행
            backtest_result = backtest_v3_continuous(
                start_date_str=start_date_str,
                num_days=num_days,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                stop_loss=stop_loss,
                take_profit=take_profit,
                verbose=False
            )
            
            # 결과 저장
            result = {
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'total_return': backtest_result['total_return'],
                'win_rate': backtest_result['win_rate'],
                'total_trades': backtest_result['total_trades'],
                'max_drawdown': backtest_result.get('max_drawdown', 0),
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0)
            }
            
            results.append(result)
            
            print(f"   결과: {backtest_result['total_return']:.2f}% (승률: {backtest_result['win_rate']:.1f}%, 거래: {backtest_result['total_trades']}회)")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
    
    # 결과 분석 및 출력
    if results:
        print("\n" + "=" * 80)
        print("최종 결과 요약")
        print("=" * 80)
        
        # 수익률 기준 정렬
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        print(f"{'순위':<4} {'Buy':<6} {'Sell':<6} {'Stop':<6} {'Take':<6} {'수익률':<8} {'승률':<8} {'거래수':<6} {'MDD':<6}")
        print("-" * 80)
        
        for i, result in enumerate(results[:20], 1):  # 상위 20개만 표시
            print(f"{i:<4} {result['buy_threshold']:<6} {result['sell_threshold']:<6} "
                  f"{result['stop_loss']:<6} {result['take_profit']:<6} "
                  f"{result['total_return']:<8.2f}% {result['win_rate']:<8.1f}% "
                  f"{result['total_trades']:<6} {result['max_drawdown']:<6.2f}%")
        
        # 최고 성능 파라미터
        best_result = results[0]
        print(f"\n최고 성능 파라미터:")
        print(f"   Buy Threshold: {best_result['buy_threshold']}")
        print(f"   Sell Threshold: {best_result['sell_threshold']}")
        print(f"   Stop Loss: {best_result['stop_loss']}%")
        print(f"   Take Profit: {best_result['take_profit']}%")
        print(f"   수익률: {best_result['total_return']:.2f}%")
        print(f"   승률: {best_result['win_rate']:.1f}%")
        print(f"   거래 수: {best_result['total_trades']}회")
        
        # 수익 모델이 있는지 확인
        profitable_results = [r for r in results if r['total_return'] > 0]
        if profitable_results:
            print(f"\nSUCCESS: 수익 파라미터 발견: {len(profitable_results)}개")
            for result in profitable_results[:5]:  # 상위 5개만 표시
                print(f"   - Buy:{result['buy_threshold']}, Sell:{result['sell_threshold']}, "
                      f"Stop:{result['stop_loss']}%, Take:{result['take_profit']}% "
                      f"=> {result['total_return']:.2f}%")
        else:
            print(f"\nERROR: 수익 파라미터 없음. 모든 조합이 손실을 보였습니다.")
        
        # 결과를 파일로 저장
        results_df = pd.DataFrame(results)
        results_df.to_csv("parameter_optimization_results.csv", index=False)
        print(f"\n결과가 'parameter_optimization_results.csv'에 저장되었습니다.")
        
    else:
        print("ERROR: 모든 파라미터 테스트 실패!")

if __name__ == "__main__":
    test_parameter_combinations()
