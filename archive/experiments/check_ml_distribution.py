"""
ML 모델의 예측 확률 분포 확인
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from indicators import add_all_indicators
from feature_engineer import create_features
from ml_model import MLSignalModel


def main():
    print("=" * 80)
    print("🔍 ML 모델 예측 확률 분포 분석")
    print("=" * 80)
    
    # 모델 로드
    ml_model = MLSignalModel("model/lgb_model.pkl")
    
    # 테스트 데이터 로드 (하루치)
    date_str = "20250107"
    print(f"\n📅 테스트 데이터: {date_str}")
    
    df = load_daily_csv(date_str, "data/daily_1m")
    if df is None:
        print("❌ 데이터 로드 실패")
        return
    
    # 컬럼 매핑
    df = df.rename(columns={
        'date_time_utc': 'timestamp',
        'acc_trade_volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    print(f"   데이터 크기: {len(df):,}개")
    
    # 지표 추가
    df = add_all_indicators(df)
    
    # 특징 생성
    df, feature_cols = create_features(df)
    
    # NaN 제거
    df_clean = df[feature_cols].dropna()
    
    print(f"   유효 데이터: {len(df_clean):,}개")
    
    # 예측
    print("\n🤖 예측 중...")
    predictions = []
    
    for i in range(len(df_clean)):
        features = df_clean.iloc[i]
        prob = ml_model.predict_proba(features)
        predictions.append(prob)
    
    predictions = np.array(predictions)
    
    # 분포 분석
    print("\n" + "=" * 80)
    print("📊 예측 확률 분포")
    print("=" * 80)
    
    print(f"평균: {predictions.mean():.4f}")
    print(f"표준편차: {predictions.std():.4f}")
    print(f"최소: {predictions.min():.4f}")
    print(f"최대: {predictions.max():.4f}")
    
    print(f"\n📈 백분위수:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(predictions, p)
        print(f"   {p:2d}%: {val:.4f}")
    
    print(f"\n🎯 임계값별 매수 기회:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (predictions >= threshold).sum()
        pct = count / len(predictions) * 100
        print(f"   {threshold:.1f} 이상: {count:4d}개 ({pct:5.2f}%)")
    
    print("\n💡 추천 임계값: 0.1 ~ 0.3 (거래 기회 충분)")
    print("=" * 80)


if __name__ == "__main__":
    main()

