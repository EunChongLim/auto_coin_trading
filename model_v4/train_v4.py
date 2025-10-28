"""
model_v4: Enhanced 지표 모델 학습
EMA, ADX, Pivot Points 등 추가 지표 포함
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import sys
import os

# 루트 Common (데이터 로드)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.download_data import load_daily_csv

# model_v4 Common (지표 계산)
sys.path.insert(0, os.path.dirname(__file__))
from Common.multi_timeframe_features import prepare_multi_timeframe_data

from datetime import datetime, timedelta
import random


def load_data_v4(max_days=80):
    """2024년 데이터 로드"""
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
        df = load_daily_csv(date_str, "../data/daily_1m", "1m")
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


def train_v4():
    """v4.0 Enhanced 모델 학습"""
    print("=" * 80)
    print("Model v4.0 Training - Enhanced Indicators")
    print("2024 Data (Train) -> 2025 Data (Test)")
    print("EMA, ADX, Pivot Points 추가")
    print("=" * 80)
    
    df = load_data_v4(max_days=80)
    
    # 특징 및 라벨 생성
    print("[Preparing Data]")
    future_minutes = 3
    down_threshold = -0.001
    up_threshold = 0.002
    
    X, y = prepare_multi_timeframe_data(
        df,
        future_minutes=future_minutes,
        down_threshold=down_threshold,
        up_threshold=up_threshold
    )
    
    print("[Data Prepared]")
    print(f"   - Samples: {len(X):,}")
    print(f"   - Features: {len(X.columns)}")
    print(f"   - Label Distribution:")
    for label, count in y.value_counts().sort_index().items():
        pct = count / len(y) * 100
        label_name = ['Down', 'Sideways', 'Up'][label]
        print(f"      {label} ({label_name}): {count:,} ({pct:.1f}%)")
    
    # 데이터 분할
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]
    
    print(f"[Split] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # LightGBM 학습
    print("[Training LightGBM v4.0 Enhanced]")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
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
        'verbose': -1
    }
    
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
    
    # 평가
    print("[Evaluation]")
    
    def evaluate_set(X_set, y_set, name):
        pred = model.predict(X_set, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1)
        acc = (pred_class == y_set).mean()
        pred_up = (pred_class == 2).sum()
        actual_up = (y_set == 2).sum()
        print(f"{name}: Acc={acc:.4f} | Predicted Up: {pred_up} | Actual Up: {actual_up}")
    
    evaluate_set(X_train, y_train, "Train")
    evaluate_set(X_val, y_val, "Val")
    evaluate_set(X_test, y_test, "Test")
    
    # 모델 저장
    model_data = {
        'model': model,
        'feature_cols': X.columns.tolist(),
        'version': 'v4.0-enhanced',
        'type': '3-class-enhanced',
        'future_minutes': future_minutes,
        'down_threshold': down_threshold,
        'up_threshold': up_threshold
    }
    
    joblib.dump(model_data, "model/lgb_model_v4_enhanced.pkl")
    print("[Saved] model/lgb_model_v4_enhanced.pkl")
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    train_v4()

