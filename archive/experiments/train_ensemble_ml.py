"""
ML 앙상블: LightGBM + XGBoost + CatBoost
3개 모델의 투표로 최종 결정
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import random
from download_data import load_daily_csv
from multi_timeframe_features import prepare_multi_timeframe_data

# XGBoost, CatBoost 체크
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed")


def load_data(max_days=40):
    """데이터 로드"""
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
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
            if i % 10 == 0:
                print(f"  [{i}/{len(all_days)}]...")
    
    merged = pd.concat(dfs).sort_index()
    print(f"[OK] {len(merged):,} candles")
    return merged


def train_ensemble():
    """앙상블 모델 학습"""
    print("=" * 80)
    print("ML Ensemble Training")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data(max_days=40)
    
    # 특징 생성 (2-Class로 단순화)
    print("\n[Preparing Data]")
    X, y, feature_cols, _ = prepare_multi_timeframe_data(
        df, future_minutes=20, down_threshold=-0.999, up_threshold=0.003  # 0.3% 이상만 상승
    )
    
    # 2-Class로 변환
    y_binary = (y == 2).astype(int)  # 상승(2)만 1, 나머지 0
    
    print(f"\n[2-Class Labels]")
    print(f"  Up(1): {y_binary.sum():,} ({y_binary.sum()/len(y_binary)*100:.1f}%)")
    print(f"  Down/Sideways(0): {len(y_binary)-y_binary.sum():,} ({(len(y_binary)-y_binary.sum())/len(y_binary)*100:.1f}%)")
    
    # 분할
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X.iloc[:train_size], y_binary.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y_binary.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y_binary.iloc[train_size+val_size:]
    
    print(f"\n[Split] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # LightGBM
    print("\n[Training LightGBM]")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
        'verbose': -1
    }
    
    model_lgb = lgb.train(lgb_params, lgb_train, num_boost_round=100,
                          valid_sets=[lgb_val], callbacks=[lgb.early_stopping(20)])
    
    # XGBoost (설치되어 있다면)
    model_xgb = None
    if XGBOOST_AVAILABLE:
        print("[Training XGBoost]")
        model_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            random_state=42
        )
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # CatBoost (설치되어 있다면)
    model_cat = None
    if CATBOOST_AVAILABLE:
        print("[Training CatBoost]")
        model_cat = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            verbose=False
        )
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # 평가
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    for name, model, available in [("LightGBM", model_lgb, True), 
                                    ("XGBoost", model_xgb, XGBOOST_AVAILABLE),
                                    ("CatBoost", model_cat, CATBOOST_AVAILABLE)]:
        if available and model is not None:
            if name == "LightGBM":
                pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                pred = model.predict(X_test)
            
            acc = (pred == y_test).mean()
            precision = ((pred == 1) & (y_test == 1)).sum() / max((pred == 1).sum(), 1)
            recall = ((pred == 1) & (y_test == 1)).sum() / max(y_test.sum(), 1)
            
            print(f"\n{name}: Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # 저장
    ensemble_data = {
        'lgb': model_lgb,
        'xgb': model_xgb if XGBOOST_AVAILABLE else None,
        'cat': model_cat if CATBOOST_AVAILABLE else None,
        'feature_cols': feature_cols,
        'version': 'Ensemble-v1.0'
    }
    
    joblib.dump(ensemble_data, "model/ensemble_model.pkl")
    print("\n[Saved] model/ensemble_model.pkl")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    train_ensemble()

