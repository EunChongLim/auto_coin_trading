#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3분 예측 모델 백테스트
"""

from backtest_v3 import backtest_v3_continuous

if __name__ == "__main__":
    # 3분 예측 모델 백테스트
    print("=" * 80)
    print("3분 예측 모델 백테스트")
    print("=" * 80)
    
    result = backtest_v3_continuous(
        start_date_str="20251022",
        num_days=3,
        buy_threshold=0.25,
        sell_threshold=0.35,
        stop_loss=1.5,
        take_profit=1.2,
        verbose=True
    )
    
    if result:
        print("\n" + "=" * 80)
        print("백테스트 결과:")
        print(f"  수익률: {result['total_return']:.2f}%")
        print(f"  승률: {result['win_rate']:.1f}%")
        print(f"  거래수: {result['total_trades']}회")
        print(f"  최대 손실: {result.get('max_drawdown', 0):.2f}%")
        print("=" * 80)
