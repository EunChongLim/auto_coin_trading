"""
거래 내역 디버깅 - 승률 100%인데 손실 원인 파악
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from indicators import add_all_indicators
from feature_engineer import create_features
from ml_model import MLSignalModel
from rule_engine import RuleEngine
from risk_manager import RiskManager
from hybrid_backtest import run_hybrid_backtest


def main():
    print("=" * 80)
    print("🔍 거래 내역 상세 분석")
    print("=" * 80)
    
    # 모델 로드
    ml_model = MLSignalModel("model/lgb_model.pkl")
    rule_engine = RuleEngine(strategy='aggressive')
    risk_manager = RiskManager(stop_loss_pct=0.005, take_profit_pct=0.008, fee_rate=0.0005)
    
    # 테스트 데이터 (1초봉)
    date_str = "20250107"
    print(f"\n📅 분석 날짜: {date_str}")
    
    df = load_daily_csv(date_str, "data/daily", "1s")
    df = df.rename(columns={'date_time_utc': 'timestamp', 'acc_trade_volume': 'volume'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 백테스팅 (verbose=False로 실행)
    result = run_hybrid_backtest(df, ml_model, rule_engine, risk_manager, 
                                ml_threshold=0.05, verbose=False)
    
    trades = result['trades']
    
    print(f"\n📊 전체 결과:")
    print(f"   수익률: {result['total_return']:+.2f}%")
    print(f"   거래 횟수: {result['trade_count']}회")
    print(f"   승률: {result['win_rate']:.1f}%")
    print(f"   승: {result['win_count']}회, 패: {result['loss_count']}회")
    
    # 매도 거래만 추출
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    print(f"\n📋 매도 거래 상세 (처음 10개):")
    print("=" * 80)
    
    for i, trade in enumerate(sell_trades[:10], 1):
        print(f"\n[{i}] {trade['timestamp']}")
        print(f"   매도가: {trade['price']:,.0f}원")
        print(f"   수익: {trade['profit']:+,.0f}원 ({trade['profit_rate']:+.2%})")
        print(f"   이유: {trade['reason']}")
        print(f"   잔고: {trade['balance_after']:,.0f}원")
    
    # 승/패 분포
    print("\n" + "=" * 80)
    print("💰 수익 분포:")
    
    profits = [t['profit'] for t in sell_trades]
    profit_rates = [t['profit_rate'] for t in sell_trades]
    
    print(f"   평균 수익: {np.mean(profits):+,.0f}원 ({np.mean(profit_rates):+.2%})")
    print(f"   최대 수익: {np.max(profits):+,.0f}원 ({np.max(profit_rates):+.2%})")
    print(f"   최소 수익: {np.min(profits):+,.0f}원 ({np.min(profit_rates):+.2%})")
    
    # profit > 0인 거래 vs profit <= 0인 거래
    positive_profits = [p for p in profits if p > 0]
    negative_profits = [p for p in profits if p <= 0]
    
    print(f"\n   profit > 0: {len(positive_profits)}개 (평균 {np.mean(positive_profits):+,.0f}원)")
    print(f"   profit <= 0: {len(negative_profits)}개 (평균 {np.mean(negative_profits):+,.0f}원)")
    
    print("\n" + "=" * 80)
    print("💡 결론:")
    if len(positive_profits) == len(sell_trades):
        print("   ⚠️  모든 거래가 profit > 0으로 계산됨 → 버그 가능성!")
    else:
        print("   ✅ 승률 계산이 정상입니다.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

