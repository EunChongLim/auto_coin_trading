"""
모델 v2.0: 멀티 타임프레임 + 3-Class Classification
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from download_data import load_daily_csv
from multi_timeframe_features import prepare_multi_timeframe_data
import random
from datetime import datetime, timedelta


def load_multiple_days_v2(start_date, end_date, data_dir="data/daily_1m", timeframe="1m", max_days=30):
    """
    여러 날짜의 1분봉 데이터 로드
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    # 랜덤 샘플링
    if len(all_days) > max_days:
        all_days = sorted(random.sample(all_days, max_days))
    
    print(f"\n📅 {len(all_days)}일치 1분봉 데이터 로드 중...")
    
    dfs = []
    for i, date_str in enumerate(all_days, 1):
        df = load_daily_csv(date_str, data_dir, timeframe)
        if df is not None and len(df) > 0:
            # 컬럼 매핑
            df = df.rename(columns={
                'date_time_utc': 'timestamp',
                'acc_trade_volume': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            dfs.append(df)
            
            if i % 10 == 0:
                print(f"  [{i}/{len(all_days)}] 로드 완료...")
    
    if not dfs:
        raise ValueError("데이터를 로드할 수 없습니다.")
    
    merged_df = pd.concat(dfs, axis=0)
    merged_df = merged_df.sort_index()
    
    print(f"✅ 총 {len(merged_df):,}개 1분봉 데이터 로드 완료")
    
    return merged_df


def train_3class_model(X_train, y_train, X_val, y_val, use_smote=False):
    """
    3-Class LightGBM 모델 학습
    """
    print("\n🤖 LightGBM 3-Class 모델 학습 시작...")
    
    # SMOTE 오버샘플링 (옵션)
    if use_smote:
        print(f"   ⚙️  SMOTE 오버샘플링 중...")
        print(f"   - 원본 라벨 분포:")
        for label in [0, 1, 2]:
            count = (y_train == label).sum()
            print(f"      {label}: {count:,}개 ({count/len(y_train)*100:.1f}%)")
        
        smote = SMOTE(sampling_strategy='not majority', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"   - 샘플링 후 라벨 분포:")
        for label in [0, 1, 2]:
            count = (y_train_resampled == label).sum()
            print(f"      {label}: {count:,}개 ({count/len(y_train_resampled)*100:.1f}%)")
        
        X_train = X_train_resampled
        y_train = y_train_resampled
    
    # 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 파라미터 설정 (3-class multiclass)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # 학습
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=20)
        ]
    )
    
    return model


def evaluate_3class_model(model, X, y, dataset_name="Test"):
    """
    3-Class 모델 평가
    """
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y, y_pred_class)
    
    print(f"\n📊 {dataset_name} 데이터 평가 결과:")
    print(f"   - Accuracy: {accuracy:.4f}")
    
    print(f"\n분류 리포트:")
    target_names = ['하락(0)', '횡보(1)', '상승(2)']
    print(classification_report(y, y_pred_class, target_names=target_names, zero_division=0))
    
    print(f"\n혼동 행렬:")
    cm = confusion_matrix(y, y_pred_class)
    print(f"   실제\\예측 |  하락  |  횡보  |  상승")
    print(f"   ---------|--------|--------|--------")
    print(f"   하락(0)  | {cm[0][0]:6d} | {cm[0][1]:6d} | {cm[0][2]:6d}")
    print(f"   횡보(1)  | {cm[1][0]:6d} | {cm[1][1]:6d} | {cm[1][2]:6d}")
    print(f"   상승(2)  | {cm[2][0]:6d} | {cm[2][1]:6d} | {cm[2][2]:6d}")


def main():
    """
    메인 실행 함수
    """
    print("=" * 80)
    print("🚀 비트코인 ML 모델 v2.0 학습 시작")
    print("   - 멀티 타임프레임 (1m, 5m, 15m, 60m)")
    print("   - 3-Class Classification (하락, 횡보, 상승)")
    print("=" * 80)
    
    # 1. 데이터 로드 (1분봉)
    df = load_multiple_days_v2("20250101", "20250530", data_dir="data/daily_1m", timeframe="1m", max_days=30)
    
    # 2. 멀티 타임프레임 특징 & 3-Class 라벨 생성
    print("\n📊 멀티 타임프레임 특징 & 라벨 생성 중...")
    X, y, feature_cols, df_with_features = prepare_multi_timeframe_data(
        df,
        future_minutes=20,      # 20분 후 예측
        down_threshold=-0.003,  # -0.3% 하락
        up_threshold=0.005      # +0.5% 상승
    )
    
    # 3. 학습/검증/테스트 분할
    print("\n✂️  데이터 분할 중...")
    
    # 시계열 순서 유지
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]
    
    print(f"   - 학습: {len(X_train):,}개")
    print(f"   - 검증: {len(X_val):,}개")
    print(f"   - 테스트: {len(X_test):,}개")
    
    # 4. 모델 학습
    model = train_3class_model(X_train, y_train, X_val, y_val, use_smote=False)
    
    # 5. 모델 평가
    print("\n" + "=" * 80)
    print("📈 모델 평가")
    print("=" * 80)
    
    evaluate_3class_model(model, X_train, y_train, "학습")
    evaluate_3class_model(model, X_val, y_val, "검증")
    evaluate_3class_model(model, X_test, y_test, "테스트")
    
    # 6. 특징 중요도
    print("\n" + "=" * 80)
    print("🔍 특징 중요도 (상위 15개)")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(15).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:10.0f}")
    
    # 7. 모델 저장
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'version': '2.0',
        'type': '3-class',
        'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(model_data, "model/lgb_model_v2.pkl")
    print(f"\n💾 모델 저장 완료: model/lgb_model_v2.pkl")
    
    print("\n" + "=" * 80)
    print("✅ 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

