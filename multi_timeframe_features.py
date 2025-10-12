"""
멀티 타임프레임 특징 생성
1분, 5분, 15분, 60분 데이터를 통합하여 특징 생성
"""

import pandas as pd
import numpy as np
from indicators import add_all_indicators


def resample_to_timeframe(df_1m, timeframe='5min'):
    """
    1분봉을 다른 시간봉으로 리샘플링
    
    Args:
        df_1m: 1분봉 데이터 (timestamp index)
        timeframe: '5min'=5분, '15min'=15분, '60min'=60분
    
    Returns:
        리샘플링된 DataFrame
    """
    df_resampled = df_1m.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_resampled


def add_multi_timeframe_features(df_1m):
    """
    멀티 타임프레임 특징 추가
    
    Args:
        df_1m: 1분봉 데이터 (timestamp index, OHLCV)
    
    Returns:
        특징이 추가된 DataFrame
    """
    df = df_1m.copy()
    
    print("[Multi Timeframe Features]")
    
    # 1. 5분봉 특징
    print("   - 5min...")
    df_5m = resample_to_timeframe(df_1m, '5min')
    df_5m = add_all_indicators(df_5m)
    
    # 5분봉 특징을 1분봉에 병합 (forward fill)
    if 'rsi' in df_5m.columns:
        df['rsi_5m'] = df_5m['rsi'].reindex(df.index, method='ffill')
    if 'ma_fast' in df_5m.columns:
        df['ma_fast_5m'] = df_5m['ma_fast'].reindex(df.index, method='ffill')
    if 'ma_slow' in df_5m.columns:
        df['ma_slow_5m'] = df_5m['ma_slow'].reindex(df.index, method='ffill')
    if 'bb_width' in df_5m.columns:
        df['bb_width_5m'] = df_5m['bb_width'].reindex(df.index, method='ffill')
    if 'volume_ma' in df_5m.columns:
        df['volume_ma_5m'] = df_5m['volume_ma'].reindex(df.index, method='ffill')
    
    # 2. 15분봉 특징
    print("   - 15min...")
    df_15m = resample_to_timeframe(df_1m, '15min')
    df_15m = add_all_indicators(df_15m)
    
    if 'rsi' in df_15m.columns:
        df['rsi_15m'] = df_15m['rsi'].reindex(df.index, method='ffill')
    if 'ma_fast' in df_15m.columns:
        df['ma_fast_15m'] = df_15m['ma_fast'].reindex(df.index, method='ffill')
    if 'ma_slow' in df_15m.columns:
        df['ma_slow_15m'] = df_15m['ma_slow'].reindex(df.index, method='ffill')
    if 'macd' in df_15m.columns:
        df['macd_15m'] = df_15m['macd'].reindex(df.index, method='ffill')
    
    # 3. 60분봉 특징
    print("   - 60min...")
    df_60m = resample_to_timeframe(df_1m, '60min')
    df_60m = add_all_indicators(df_60m)
    
    if 'rsi' in df_60m.columns:
        df['rsi_60m'] = df_60m['rsi'].reindex(df.index, method='ffill')
    if 'ma_fast' in df_60m.columns:
        df['ma_fast_60m'] = df_60m['ma_fast'].reindex(df.index, method='ffill')
    if 'ma_slow' in df_60m.columns:
        df['ma_slow_60m'] = df_60m['ma_slow'].reindex(df.index, method='ffill')
    if 'bb_width' in df_60m.columns:
        df['bb_width_60m'] = df_60m['bb_width'].reindex(df.index, method='ffill')
    
    # 4. 멀티 타임프레임 조합 특징
    print("   - Combined...")
    
    # 추세 일치도 (모든 시간대 이동평균 방향 일치)
    if 'ma_fast' in df.columns and 'ma_fast_5m' in df.columns and 'ma_fast_15m' in df.columns:
        df['trend_alignment'] = (
            ((df['close'] > df['ma_fast']) & (df['close'] > df['ma_fast_5m']) & (df['close'] > df['ma_fast_15m'])).astype(int) -
            ((df['close'] < df['ma_fast']) & (df['close'] < df['ma_fast_5m']) & (df['close'] < df['ma_fast_15m'])).astype(int)
        )
    
    # RSI 다이버전스 (1분 vs 60분)
    if 'rsi' in df.columns and 'rsi_60m' in df.columns:
        df['rsi_divergence'] = df['rsi'] - df['rsi_60m']
    
    # 가격 위치 (60분 고저 대비)
    df_60m_high = df_60m['high'].reindex(df.index, method='ffill')
    df_60m_low = df_60m['low'].reindex(df.index, method='ffill')
    df['price_position_60m'] = (df['close'] - df_60m_low) / (df_60m_high - df_60m_low + 1e-8)  # 0으로 나누기 방지
    
    # 5. 고급 시장 구조 특징
    print("   - Market structure...")
    
    # 최근 N분간 고점/저점 개수
    df['swing_highs_10'] = df['high'].rolling(10).apply(
        lambda x: sum(x == x.max()), raw=True
    )
    df['swing_lows_10'] = df['low'].rolling(10).apply(
        lambda x: sum(x == x.min()), raw=True
    )
    
    # 가격 가속도 (2차 미분)
    df['price_acceleration'] = df['close'].diff().diff()
    
    # ATR (Average True Range) - 변동성
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']  # 정규화
    
    # 6. 캔들 패턴 (간단한 패턴만)
    print("   - Candle patterns...")
    
    body = df['close'] - df['open']
    body_pct = body / df['open']
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    
    # Doji (몸통이 매우 작음)
    df['is_doji'] = (np.abs(body_pct) < 0.001).astype(int)
    
    # 망치형 (아래 그림자 긴 양봉)
    df['is_hammer'] = ((lower_shadow > body * 2) & (body > 0)).astype(int)
    
    # 교수형 (위 그림자 긴 음봉)
    df['is_hanging_man'] = ((upper_shadow > np.abs(body) * 2) & (body < 0)).astype(int)
    
    print("[DONE] Total features: {}".format(len(df.columns)))
    
    return df


def create_3class_label(df, future_minutes=20, down_threshold=-0.003, up_threshold=0.005):
    """
    3-Class 라벨 생성
    
    Args:
        df: OHLCV 데이터
        future_minutes: 미래 N분
        down_threshold: 하락 기준 (-0.003 = -0.3%)
        up_threshold: 상승 기준 (0.005 = 0.5%)
    
    Returns:
        pandas Series: 라벨 (0=하락, 1=횡보, 2=상승)
    """
    future_price = df['close'].shift(-future_minutes)
    current_price = df['close']
    
    price_change = (future_price - current_price) / current_price
    
    # 3-Class 라벨
    label = pd.Series(1, index=df.index)  # 기본: 횡보
    label[price_change <= down_threshold] = 0  # 하락
    label[price_change >= up_threshold] = 2  # 상승
    
    return label


def prepare_multi_timeframe_data(df, future_minutes=20, down_threshold=-0.003, up_threshold=0.005):
    """
    멀티 타임프레임 ML 데이터 준비
    
    Args:
        df: 원본 1분봉 OHLCV 데이터
        future_minutes: 미래 N분
        down_threshold: 하락 기준
        up_threshold: 상승 기준
    
    Returns:
        tuple: (X, y, feature_cols, df_with_features)
    """
    # 1. 1분봉 지표 추가
    df = add_all_indicators(df)
    
    # 2. 멀티 타임프레임 특징 추가
    df = add_multi_timeframe_features(df)
    
    # 3. 라벨 생성 (3-Class)
    df['label'] = create_3class_label(df, future_minutes, down_threshold, up_threshold)
    
    # 4. NaN 제거
    df_clean = df.dropna()
    
    # 5. 특징 선택 (가격, 라벨 제외)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'acc_trade_price', 'label']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # 6. X, y 분리
    X = df_clean[feature_cols]
    y = df_clean['label']
    
    # 라벨 분포 출력
    label_counts = y.value_counts().sort_index()
    label_names = {0: 'Down', 1: 'Sideways', 2: 'Up'}
    
    print(f"\n[Data Prepared]")
    print(f"   - Samples: {len(X):,}")
    print(f"   - Features: {len(feature_cols)}")
    print(f"   - Label Distribution:")
    for label_val, count in label_counts.items():
        pct = count / len(y) * 100
        print(f"      {label_val} ({label_names[label_val]}): {count:,} ({pct:.1f}%)")
    
    return X, y, feature_cols, df_clean


if __name__ == "__main__":
    print("[Module] multi_timeframe_features.py loaded")
    print("Available functions:")
    print("   - add_multi_timeframe_features()")
    print("   - create_3class_label()")
    print("   - prepare_multi_timeframe_data()")

