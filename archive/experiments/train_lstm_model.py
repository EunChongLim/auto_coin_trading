"""
LSTM 딥러닝 모델 학습
시계열 특화 모델로 패턴 학습
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from download_data import load_daily_csv
from indicators import add_all_indicators
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras 체크
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Installing required packages...")


def load_multiple_days(start_date, end_date, max_days=60):
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
    
    if len(all_days) > max_days:
        all_days = sorted(random.sample(all_days, max_days))
    
    print(f"\n[Loading Data] {len(all_days)} days...")
    
    dfs = []
    for i, date_str in enumerate(all_days, 1):
        df = load_daily_csv(date_str, "data/daily_1m", "1m")
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                'date_time_utc': 'timestamp',
                'acc_trade_volume': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            dfs.append(df)
            
            if i % 20 == 0:
                print(f"  [{i}/{len(all_days)}] loaded...")
    
    if not dfs:
        raise ValueError("No data loaded")
    
    merged_df = pd.concat(dfs, axis=0)
    merged_df = merged_df.sort_index()
    
    print(f"[OK] Total {len(merged_df):,} candles loaded")
    
    return merged_df


def create_lstm_sequences(df, lookback=60, future_minutes=20, profit_threshold=0.005):
    """
    LSTM용 시퀀스 데이터 생성
    
    Args:
        df: OHLCV + 지표 데이터
        lookback: 과거 N분 참조
        future_minutes: 미래 N분 예측
        profit_threshold: 상승 기준
    
    Returns:
        X: (samples, lookback, features)
        y: (samples,) - 0 or 1
    """
    # 특징 선택
    feature_cols = ['close', 'volume', 'rsi', 'ma_fast', 'ma_slow', 
                    'bb_upper', 'bb_lower', 'macd', 'macd_signal']
    
    # 존재하는 컬럼만 사용
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 라벨 생성
    future_price = df['close'].shift(-future_minutes)
    price_change = (future_price - df['close']) / df['close']
    labels = (price_change >= profit_threshold).astype(int)
    
    # 시퀀스 생성
    X_list = []
    y_list = []
    
    for i in range(lookback, len(df) - future_minutes):
        # 과거 lookback개 데이터
        sequence = df[feature_cols].iloc[i-lookback:i].values
        
        # NaN 체크
        if np.isnan(sequence).any():
            continue
        
        X_list.append(sequence)
        y_list.append(labels.iloc[i])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n[Sequences Created]")
    print(f"  - Samples: {len(X):,}")
    print(f"  - Shape: {X.shape}")
    print(f"  - Label 1 (Up): {y.sum():,} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  - Label 0 (Down/Sideways): {len(y)-y.sum():,} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def build_lstm_model(input_shape, learning_rate=0.001):
    """
    LSTM 모델 구축
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def train_lstm():
    """
    LSTM 모델 학습
    """
    if not KERAS_AVAILABLE:
        print("\n[ERROR] TensorFlow not installed!")
        print("Please run: pip install tensorflow")
        return
    
    print("=" * 80)
    print("LSTM Model Training")
    print("=" * 80)
    
    # 1. 데이터 로드
    df = load_multiple_days("20250101", "20250530", max_days=60)
    
    # 2. 지표 계산
    print("\n[Computing Indicators]")
    df = add_all_indicators(df)
    df = df.dropna()
    
    # 3. 시퀀스 생성
    print("\n[Creating LSTM Sequences]")
    X, y, feature_cols = create_lstm_sequences(
        df, 
        lookback=60,        # 60분 (1시간) 참조
        future_minutes=20,  # 20분 후 예측
        profit_threshold=0.003  # 0.3% 상승
    )
    
    # 4. 정규화
    print("\n[Normalizing Data]")
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    X_normalized = X_normalized.reshape(n_samples, n_timesteps, n_features)
    
    # 5. 학습/검증/테스트 분할
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X_normalized[:train_size]
    y_train = y[:train_size]
    
    X_val = X_normalized[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X_normalized[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\n[Data Split]")
    print(f"  - Train: {len(X_train):,}")
    print(f"  - Val  : {len(X_val):,}")
    print(f"  - Test : {len(X_test):,}")
    
    # 6. 모델 학습
    print("\n[Building LSTM Model]")
    model = build_lstm_model(input_shape=(n_timesteps, n_features))
    
    print("\n[Model Summary]")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    print("\n[Training Started]")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 7. 평가
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    train_loss, train_acc, train_precision, train_recall = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTrain: Acc={train_acc:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}")
    print(f"Val  : Acc={val_acc:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}")
    print(f"Test : Acc={test_acc:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")
    
    # 8. 모델 저장
    model.save("model/lstm_model.h5")
    
    # Scaler 저장
    import joblib
    model_data = {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'lookback': 60,
        'version': 'LSTM-v1.0'
    }
    joblib.dump(model_data, "model/lstm_model_data.pkl")
    
    print("\n[Model Saved]")
    print("  - model/lstm_model.h5")
    print("  - model/lstm_model_data.pkl")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    if KERAS_AVAILABLE:
        tf.random.set_seed(42)
    train_lstm()

