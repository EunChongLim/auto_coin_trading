"""
멀티 타임프레임 Walk-Forward 학습
1분봉 + 1시간봉 + 4시간봉 통합 모델
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_multi.Common.data_loader import load_multi_timeframe_data
from model_multi.Common.features_multi import calculate_multi_timeframe_features, get_all_feature_columns


def train_multi_timeframe_model(
    start_date="20240101",
    end_date="20241231",
    window_size_1m=100,
    window_size_1h=100,
    window_size_4h=100,
    prediction_minutes=5,
    train_days=30,
    test_days=7
):
    """
    멀티 타임프레임 Walk-Forward 학습
    
    Args:
        start_date, end_date: 학습 기간
        window_size_1m, 1h, 4h: 각 타임프레임 윈도우 크기
        prediction_minutes: 예측 목표 (N분 후 수익률)
        train_days, test_days: Walk-Forward 구간
    """
    print("="*80)
    print("멀티 타임프레임 Walk-Forward 학습")
    print("="*80)
    print(f"Window: 1m={window_size_1m}, 1h={window_size_1h}, 4h={window_size_4h}")
    print(f"Prediction: {prediction_minutes}분 후 수익률")
    print(f"Train/Test: {train_days}일 / {test_days}일")
    
    # === 1. 데이터 로드 (각 타임프레임을 API에서 직접) ===
    df_1m, df_1h, df_4h = load_multi_timeframe_data(start_date, end_date)
    
    # === 2. Feature 컬럼 ===
    feature_cols = get_all_feature_columns()
    print(f"\n[Features] {len(feature_cols)} features")
    
    # === 3. Walk-Forward 학습 ===
    print(f"\n[Walk-Forward Training]")
    
    train_minutes = train_days * 1440
    test_minutes = test_days * 1440
    total_minutes = len(df_1m)
    
    # 최소 시작 인덱스 (충분한 윈도우 확보)
    min_start_idx = max(window_size_1m, window_size_1h * 60, window_size_4h * 240)
    
    all_X = []
    all_y = []
    
    fold_count = 0
    current_start = min_start_idx
    
    while current_start + train_minutes + prediction_minutes < total_minutes:
        fold_count += 1
        
        train_end = current_start + train_minutes
        train_data_1m = df_1m.iloc[current_start:train_end]
        
        # 슬라이딩 윈도우로 Feature 생성
        fold_X = []
        fold_y = []
        
        for i in range(window_size_1m, len(train_data_1m) - prediction_minutes):
            actual_idx = current_start + i
            current_time = df_1m.iloc[actual_idx]['datetime']
            
            # 1분봉 윈도우
            window_1m = df_1m.iloc[actual_idx - window_size_1m:actual_idx]
            
            # 1시간봉 윈도우 (현재 시각 이전)
            window_1h = df_1h[df_1h['datetime'] <= current_time].tail(window_size_1h)
            
            # 4시간봉 윈도우 (현재 시각 이전)
            window_4h = df_4h[df_4h['datetime'] <= current_time].tail(window_size_4h)
            
            # 윈도우 크기 확인
            if len(window_1m) < window_size_1m or len(window_1h) < 30 or len(window_4h) < 30:
                continue
            
            # Feature 계산
            features = calculate_multi_timeframe_features(window_1m, window_1h, window_4h)
            
            if features is None:
                continue
            
            # Target: N분 후 수익률
            current_price = df_1m.iloc[actual_idx]['close']
            future_price = df_1m.iloc[actual_idx + prediction_minutes]['close']
            target = (future_price - current_price) / current_price * 100
            
            fold_X.append(features)
            fold_y.append(target)
        
        if fold_X:
            all_X.extend(fold_X)
            all_y.extend(fold_y)
            print(f"  Fold {fold_count}: {len(fold_X):,} samples")
        
        # 다음 구간으로 이동
        current_start += test_minutes
    
    print(f"\n[Total Training Samples] {len(all_X):,}")
    
    if len(all_X) == 0:
        print("[ERROR] No training samples generated!")
        return None
    
    # === 4. DataFrame 변환 ===
    X = pd.DataFrame(all_X)
    y = np.array(all_y)
    
    # NaN 처리
    print(f"\n[Checking NaN...]")
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  Found {nan_count} NaN in features, filling with 0")
        X = X.fillna(0)
    
    y_nan_count = np.isnan(y).sum()
    if y_nan_count > 0:
        print(f"  Found {y_nan_count} NaN in targets, removing...")
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"  Remaining: {len(X):,}")
    
    # Feature 컬럼 정렬
    missing_cols = [col for col in feature_cols if col not in X.columns]
    if missing_cols:
        print(f"\n[Warning] Missing columns: {missing_cols[:10]}... ({len(missing_cols)} total)")
        for col in missing_cols:
            X[col] = 0
    
    X = X[feature_cols]
    
    # === 5. Train/Test 분할 ===
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\n[Data Split]")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # === 6. LightGBM 학습 ===
    print(f"\n[Training LightGBM...]")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 100,
        'max_depth': 7
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # === 7. 성능 평가 ===
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n[Performance]")
    print(f"  Train MAE: {train_mae:.4f}%")
    print(f"  Test MAE:  {test_mae:.4f}%")
    print(f"  Train R²:  {train_r2:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")
    
    # === 8. Feature Importance ===
    print(f"\n[Top 20 Important Features]")
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n  Rank  Feature                      Importance  Timeframe")
    print("  " + "-"*65)
    for idx, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        feat_name = row['feature']
        
        if feat_name.startswith('1m_'):
            tf_type = "1분봉  "
        elif feat_name.startswith('1h_'):
            tf_type = "1시간봉"
        elif feat_name.startswith('4h_'):
            tf_type = "4시간봉"
        else:
            tf_type = "관계   "
        
        print(f"  {idx:2d}.  {feat_name:27s}  {row['importance']:10.0f}  ({tf_type})")
    
    # === 9. 모델 저장 ===
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'window_size_1m': window_size_1m,
        'window_size_1h': window_size_1h,
        'window_size_4h': window_size_4h,
        'prediction_minutes': prediction_minutes,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_date_range': f"{start_date} ~ {end_date}",
        'version': 'multi_timeframe_v1'
    }
    
    save_path = "model/lgb_multi_timeframe.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_data, save_path)
    print(f"\n[Saved] Model: {save_path}")
    
    print("\n" + "="*80)
    print("멀티 타임프레임 학습 완료!")
    print("="*80)
    
    return model_data


if __name__ == "__main__":
    # 2024년 전체 학습
    train_multi_timeframe_model(
        start_date="20240101",
        end_date="20241231",
        window_size_1m=100,
        window_size_1h=100,
        window_size_4h=100,
        prediction_minutes=5,
        train_days=30,
        test_days=7
    )

