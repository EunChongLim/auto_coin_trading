"""
ML 모델 학습을 위한 특징(Feature) 생성 모듈
"""

import pandas as pd
import numpy as np
from indicators import add_all_indicators


def create_label(df, future_minutes=5, profit_threshold=0.005, use_rank=True, rank_percentile=0.8):
    """
    라벨 생성: 절대 수익률 또는 상대 랭크 기반
    
    Args:
        df: OHLCV 데이터프레임
        future_minutes: 미래 N분
        profit_threshold: 수익률 임계값 (use_rank=False일 때)
        use_rank: True면 상대 랭크 기반, False면 절대 수익률 기반
        rank_percentile: 상위 몇 %를 상승으로 볼지 (0.8 = 상위 20%)
    
    Returns:
        pandas Series: 라벨 (0 or 1)
    """
    future_price = df['close'].shift(-future_minutes)
    current_price = df['close']
    
    price_change = (future_price - current_price) / current_price
    
    if use_rank:
        # 상대 랭크 기반: 상위 N% 구간을 1로 라벨링
        rank_pct = price_change.rank(pct=True)
        label = (rank_pct >= rank_percentile).astype(int)
    else:
        # 절대 수익률 기반 (기존 방식)
        label = (price_change >= profit_threshold).astype(int)
    
    return label


def create_features(df):
    """
    ML 모델용 특징 생성
    
    Args:
        df: 지표가 포함된 OHLCV 데이터프레임
    
    Returns:
        list: 특징 컬럼 이름 리스트
    """
    df = df.copy()
    
    # 1. RSI 관련 특징
    df['rsi_normalized'] = df['rsi'] / 100  # 0~1로 정규화
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # 2. MA 관련 특징
    df['ma_diff'] = df['ma_fast'] - df['ma_slow']
    df['ma_diff_pct'] = df['ma_diff'] / df['close']
    df['price_above_ma_fast'] = (df['close'] > df['ma_fast']).astype(int)
    df['price_above_ma_slow'] = (df['close'] > df['ma_slow']).astype(int)
    
    # 3. 볼린저 밴드 관련 특징
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 4. MACD 관련 특징
    df['macd_normalized'] = df['macd'] / df['close']
    df['macd_hist_normalized'] = df['macd_hist'] / df['close']
    df['macd_cross'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    
    # 5. 거래량 관련 특징
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_surge'] = (df['volume'] > df['volume_ma'] * 2).astype(int)
    
    # 6. 가격 변화 특징
    df['price_momentum_1'] = df['price_change']
    df['price_momentum_5'] = df['price_change_5']
    
    # 7. 시간 특징 (1분봉 기준)
    if 'timestamp' in df.columns or isinstance(df.index, pd.DatetimeIndex):
        if isinstance(df.index, pd.DatetimeIndex):
            time_index = df.index
        else:
            time_index = pd.to_datetime(df['timestamp'])
        
        df['hour'] = time_index.hour
        df['minute'] = time_index.minute
        df['is_morning'] = (df['hour'] >= 9) & (df['hour'] < 12)
        df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 18)
        df['is_night'] = (df['hour'] >= 18) | (df['hour'] < 9)
    
    # 특징 컬럼 리스트
    feature_cols = [
        'rsi_normalized', 'rsi_oversold', 'rsi_overbought',
        'ma_diff_pct', 'price_above_ma_fast', 'price_above_ma_slow',
        'bb_position', 'bb_width',
        'macd_normalized', 'macd_hist_normalized', 'macd_cross',
        'volume_ratio', 'volume_surge',
        'price_momentum_1', 'price_momentum_5',
    ]
    
    # 시간 특징 추가 (있는 경우)
    if 'hour' in df.columns:
        feature_cols.extend(['hour', 'minute', 'is_morning', 'is_afternoon', 'is_night'])
    
    return df, feature_cols


def prepare_ml_data(df, future_minutes=5, profit_threshold=0.005, use_rank=True, rank_percentile=0.8):
    """
    ML 학습용 데이터 준비 (전체 파이프라인)
    
    Args:
        df: 원본 OHLCV 데이터프레임
        future_minutes: 미래 N분
        profit_threshold: 수익률 임계값 (use_rank=False일 때)
        use_rank: True면 상대 랭크 기반, False면 절대 수익률 기반
        rank_percentile: 상위 몇 %를 상승으로 볼지
    
    Returns:
        tuple: (X, y, feature_cols, df_with_features)
    """
    # 1. 지표 추가
    df = add_all_indicators(df)
    
    # 2. 특징 생성
    df, feature_cols = create_features(df)
    
    # 3. 라벨 생성
    df['label'] = create_label(df, future_minutes, profit_threshold, use_rank, rank_percentile)
    
    # 4. NaN 제거
    df_clean = df.dropna()
    
    # 5. X, y 분리
    X = df_clean[feature_cols]
    y = df_clean['label']
    
    label_method = f"상대 랭크 (상위 {(1-rank_percentile)*100:.0f}%)" if use_rank else f"절대 수익률 (>={profit_threshold*100:.1f}%)"
    print(f"✅ ML 데이터 준비 완료")
    print(f"   - 샘플 수: {len(X):,}개")
    print(f"   - 특징 수: {len(feature_cols)}개")
    print(f"   - 라벨 방법: {label_method}")
    print(f"   - 라벨 분포: 상승={y.sum():,}개 ({y.mean()*100:.1f}%), 하락/유지={len(y)-y.sum():,}개")
    
    return X, y, feature_cols, df_clean


if __name__ == "__main__":
    print("✅ feature_engineer.py 모듈 로드 완료")
    print("📊 사용 가능한 함수:")
    print("  - create_label()")
    print("  - create_features()")
    print("  - prepare_ml_data()")

