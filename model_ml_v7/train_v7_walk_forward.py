"""
v7 Walk-Forward 학습
1분봉 + 틱 데이터 통합
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from Common.tick_loader import download_tick_data
from Common.tick_aggregator import aggregate_ticks_to_minute, add_tick_features
from Common.features_v7 import calculate_combined_features, get_all_feature_columns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.auto_download_data import download_1m_data


def load_combined_data(start_date, end_date):
    """
    1분봉 + 틱 데이터 통합 로드
    
    Returns:
        DataFrame: datetime, open, high, low, close, volume, 
                   + 틱 집계 (buy_pressure, tick_count 등)
    """
    print(f"\n[Loading Data] {start_date} ~ {end_date}")
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_candles = []
    all_ticks = []
    
    current_dt = start_dt
    total_days = (end_dt - start_dt).days + 1
    day_count = 0
    
    while current_dt <= end_dt:
        day_count += 1
        date_str = current_dt.strftime("%Y%m%d")
        
        # 1분봉 로드
        candle_path = os.path.join("../data/daily_1m", f"KRW-BTC_candle-1m_{date_str}.csv")
        if not os.path.exists(candle_path):
            candle_df = download_1m_data(date_str, output_dir="../data/daily_1m")
            if candle_df is not None:
                all_candles.append(candle_df)
        else:
            candle_df = pd.read_csv(candle_path)
            all_candles.append(candle_df)
        
        # 틱 데이터 로드
        tick_df = download_tick_data(date_str, output_dir="../data/ticks")
        if tick_df is not None:
            all_ticks.append(tick_df)
        
        if day_count % 30 == 0:
            print(f"  Progress: {day_count}/{total_days} days")
        
        current_dt += timedelta(days=1)
    
    print(f"\n[Processing...]")
    
    # 1분봉 병합
    candle_combined = pd.concat(all_candles, ignore_index=True)
    
    # 컬럼 정리
    candle_clean = pd.DataFrame()
    
    if 'date_time_kst' in candle_combined.columns:
        candle_clean['datetime'] = pd.to_datetime(candle_combined['date_time_kst'])
    elif 'date_time_utc' in candle_combined.columns:
        candle_clean['datetime'] = pd.to_datetime(candle_combined['date_time_utc'])
    
    if 'open' in candle_combined.columns:
        candle_clean['open'] = candle_combined['open']
        candle_clean['high'] = candle_combined['high']
        candle_clean['low'] = candle_combined['low']
        candle_clean['close'] = candle_combined['close']
    elif 'opening_price' in candle_combined.columns:
        candle_clean['open'] = candle_combined['opening_price']
        candle_clean['high'] = candle_combined['high_price']
        candle_clean['low'] = candle_combined['low_price']
        candle_clean['close'] = candle_combined['trade_price']
    
    if 'volume' in candle_combined.columns:
        candle_clean['volume'] = candle_combined['volume']
    elif 'acc_trade_volume' in candle_combined.columns:
        candle_clean['volume'] = candle_combined['acc_trade_volume']
    
    candle_clean = candle_clean.sort_values('datetime').reset_index(drop=True)
    candle_clean['datetime'] = candle_clean['datetime'].dt.floor('min')
    
    # 틱 데이터 병합 및 집계
    if all_ticks:
        tick_combined = pd.concat(all_ticks, ignore_index=True)
        tick_agg = aggregate_ticks_to_minute(tick_combined)
        tick_agg = add_tick_features(tick_agg)
        
        # 1분봉 + 틱 병합
        result = pd.merge(
            candle_clean,
            tick_agg,
            on='datetime',
            how='left',
            suffixes=('', '_tick')
        )
        
        # volume 충돌 처리 (1분봉 volume 우선)
        if 'volume_tick' in result.columns:
            result = result.drop('volume_tick', axis=1)
        
        # 틱 데이터 없는 행은 기본값으로 채우기
        tick_cols = ['tick_count', 'buy_pressure', 'large_buy_count', 'large_sell_count',
                     'buy_pressure_ma5', 'buy_pressure_ma20', 'buy_pressure_change',
                     'tick_speed_ma5', 'tick_speed_ma20', 'tick_speed_ratio',
                     'buy_pressure_std', 'large_trade_imbalance', 'volume_imbalance', 'price_vs_vwap']
        
        for col in tick_cols:
            if col in result.columns:
                if col == 'buy_pressure' or 'ma' in col:
                    result[col] = result[col].fillna(0.5)
                else:
                    result[col] = result[col].fillna(0)
    else:
        result = candle_clean
        print("  [Warning] No tick data available")
    
    print(f"[OK] {len(result):,} minutes loaded")
    return result


def train_v7_walk_forward(
    start_date="20250101",
    end_date="20250731",
    window_size=100,
    prediction_minutes=5,
    train_days=30,
    test_days=7
):
    """
    v7 Walk-Forward 학습 (1분봉 + 틱)
    """
    print("="*80)
    print("v7 Walk-Forward Training - Candles + Tick Data")
    print("="*80)
    print(f"Window Size: {window_size} 분")
    print(f"Prediction: {prediction_minutes} 분 후 수익률")
    print(f"Train/Test: {train_days}일 / {test_days}일")
    
    # 데이터 로드
    df = load_combined_data(start_date, end_date)
    
    # Feature 컬럼
    feature_cols = get_all_feature_columns()
    
    print(f"\n[Features]")
    print(f"  Total: {len(feature_cols)} features")
    print(f"  - v6 (candle): 27")
    print(f"  - Tick: {len(feature_cols) - 27}")
    
    # Walk-Forward 학습
    print(f"\n[Walk-Forward Training]")
    
    train_minutes = train_days * 1440
    test_minutes = test_days * 1440
    total_minutes = len(df)
    
    all_X = []
    all_y = []
    
    fold_count = 0
    current_start = 0
    
    while current_start + train_minutes + window_size + prediction_minutes < total_minutes:
        fold_count += 1
        
        train_end = current_start + train_minutes
        train_data = df.iloc[current_start:train_end]
        
        # 슬라이딩 윈도우로 Feature 생성
        fold_X = []
        fold_y = []
        
        for i in range(window_size, len(train_data) - prediction_minutes):
            # 1분봉 윈도우
            candle_window = train_data.iloc[i-window_size:i][['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # 틱 집계 윈도우 (있으면)
            tick_window = None
            if 'buy_pressure' in train_data.columns:
                tick_window = train_data.iloc[i-window_size:i]
            
            # Feature 계산
            features = calculate_combined_features(candle_window, tick_window)
            
            # Target 계산 (N분 후 수익률)
            current_price = train_data.iloc[i]['close']
            future_price = train_data.iloc[i + prediction_minutes]['close']
            target = ((future_price - current_price) / current_price) * 100
            
            if features is not None:
                fold_X.append(features)
                fold_y.append(target)
        
        if fold_X:
            all_X.extend(fold_X)
            all_y.extend(fold_y)
            print(f"  Fold {fold_count}: {len(fold_X):,} samples")
        
        current_start += test_minutes
    
    print(f"\n[Total Training Samples] {len(all_X):,}")
    
    # DataFrame 변환
    X = pd.DataFrame(all_X)
    y = np.array(all_y)
    
    # NaN 제거
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
    
    # Feature 컬럼 확인
    missing_cols = [col for col in feature_cols if col not in X.columns]
    if missing_cols:
        print(f"\n[Warning] Missing columns: {missing_cols}")
        for col in missing_cols:
            X[col] = 0
    
    X = X[feature_cols]
    
    # Train/Test 분할
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\n[Data Split]")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # LightGBM 학습
    print(f"\n[Training LightGBM with Tick Features...]")
    
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
    
    # 성능 평가
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n[Performance]")
    print(f"  Train MAE:  {train_mae:.4f}%")
    print(f"  Test MAE:   {test_mae:.4f}%")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    
    # v6와 비교
    v6_test_r2 = 0.0056
    improvement = (test_r2 - v6_test_r2) / v6_test_r2 * 100
    print(f"\n[vs v6]")
    print(f"  v6 Test R²: {v6_test_r2:.4f}")
    print(f"  v7 Test R²: {test_r2:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    # 예측 분포
    print(f"\n[Prediction Distribution]")
    print(f"  Actual: mean={y_test.mean():.4f}%, std={y_test.std():.4f}%")
    print(f"  Predicted: mean={y_pred_test.mean():.4f}%, std={y_pred_test.std():.4f}%")
    
    # Feature Importance (상위 20개)
    print(f"\n[Top 20 Important Features]")
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n  Rank  Feature                 Importance  Type")
    print("  " + "-"*60)
    for idx, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        feat_name = row['feature']
        feat_type = "TICK" if feat_name in ['tick_count', 'buy_pressure', 'large_buy_count', 'large_sell_count',
                                             'buy_pressure_ma5', 'buy_pressure_ma20', 'buy_pressure_change',
                                             'tick_speed_ma5', 'tick_speed_ma20', 'tick_speed_ratio',
                                             'buy_pressure_std', 'large_trade_imbalance', 'volume_imbalance', 
                                             'price_vs_vwap'] else "v6  "
        print(f"  {idx:2d}.  {feat_name:22s}  {row['importance']:8.0f}  ({feat_type})")
    
    # 모델 저장
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'window_size': window_size,
        'prediction_minutes': prediction_minutes,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_date_range': f"{start_date} ~ {end_date}",
        'version': 'v7_tick_enhanced'
    }
    
    save_path = "model/lgb_v7_tick.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_data, save_path)
    print(f"\n[Saved] Model: {save_path}")
    
    print("\n" + "="*80)
    print("v7 Training Complete!")
    print("="*80)
    
    return model_data


if __name__ == "__main__":
    # 2024년 1~12월 학습 (전체 1년)
    train_v7_walk_forward(
        start_date="20240101",
        end_date="20241231",
        window_size=100,
        prediction_minutes=5,
        train_days=30,
        test_days=7
    )

