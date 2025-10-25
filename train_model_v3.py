"""
모델 v3.0: 라벨 임계값 최적화 + 더 많은 데이터
상승 기준을 낮춰서 더 많은 상승 샘플 확보
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from download_data import load_daily_csv
from multi_timeframe_features import prepare_multi_timeframe_data
from datetime import datetime, timedelta
import random


def load_data_v3(max_days=80):
    """더 많은 데이터 로드 (2024년 학습용)"""
    start = datetime.strptime("20240101", "%Y%m%d")
    end = datetime.strptime("20241231", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    if len(all_days) > max_days:
        all_days = sorted(random.sample(all_days, max_days))
    
    print(f"\n[Loading] {len(all_days)} days...")
    
    dfs = []
    for i, date_str in enumerate(all_days, 1):
        df = load_daily_csv(date_str, "data/daily_1m", "1m")
        if df is not None and len(df) > 0:
            df = df.rename(columns={'date_time_utc': 'timestamp', 'acc_trade_volume': 'volume'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            dfs.append(df)
            if i % 20 == 0:
                print(f"  [{i}/{len(all_days)}]...")
    
    merged = pd.concat(dfs).sort_index()
    print(f"[OK] {len(merged):,} candles")
    return merged


def train_v3():
    """v3.0 학습"""
    print("=" * 80)
    print("Model v3.0 Training")
    print("2024 Data (Train) -> 2025 Data (Test)")
    print("Optimized label thresholds + More data")
    print("=" * 80)
    
    # 더 많은 데이터 로드
    df = load_data_v3(max_days=80)
    
    # 3-Class 라벨 (임계값 조정)
    print("\n[Preparing Data]")
    X, y, feature_cols, _ = prepare_multi_timeframe_data(
        df,
        future_minutes=15,        # 20분 → 15분 (빠른 반응)
        down_threshold=-0.002,    # -0.3% → -0.2% (더 민감)
        up_threshold=0.003        # +0.5% → +0.3% (더 많은 상승 샘플)
    )
    
    # 분할
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
    
    print(f"\n[Split] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # LightGBM 학습
    print("\n[Training LightGBM v3.0]")
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
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
    
    model = lgb.train(
        params, train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    
    # 평가
    print("\n[Evaluation]")
    for dataset_name, X_set, y_set in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = model.predict(X_set)
        pred_class = np.argmax(pred, axis=1)
        acc = (pred_class == y_set).mean()
        
        # 상승(2) 예측 개수
        up_predictions = (pred_class == 2).sum()
        up_actual = (y_set == 2).sum()
        
        print(f"{dataset_name}: Acc={acc:.4f} | Predicted Up: {up_predictions} | Actual Up: {up_actual}")
    
    # 저장
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'version': '3.0',
        'type': '3-class-optimized'
    }
    
    joblib.dump(model_data, "model/lgb_model_v3.pkl")
    print("\n[Saved] model/lgb_model_v3.pkl")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    train_v3()

