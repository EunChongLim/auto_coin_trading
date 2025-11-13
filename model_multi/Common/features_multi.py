"""
멀티 타임프레임 Feature 계산
1분봉 + 1시간봉 + 4시간봉 통합 Feature
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_rsi(prices, period=14):
    """RSI 계산"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    if len(prices) < slow:
        return 0.0, 0.0, 0.0
    
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().iloc[-1]
    
    macd = ema_fast - ema_slow
    
    # Signal line (단순화: 최근 값 사용)
    macd_signal = macd  # 실제로는 MACD의 EMA
    macd_hist = macd - macd_signal
    
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """볼린저 밴드 계산"""
    if len(prices) < period:
        return 0.5
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    current_price = prices[-1]
    
    if upper == lower:
        return 0.5
    
    # 0 (하단) ~ 1 (상단)
    position = (current_price - lower) / (upper - lower)
    return max(0, min(1, position))


def calculate_atr(high, low, close, period=14):
    """ATR (Average True Range) 계산"""
    if len(high) < period + 1:
        return 0.0
    
    tr_list = []
    for i in range(1, len(high)):
        h = high[i]
        l = low[i]
        c_prev = close[i-1]
        
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    
    atr = np.mean(tr_list[-period:]) if len(tr_list) >= period else 0.0
    return atr


def calculate_single_timeframe_features(window_df, prefix=''):
    """
    단일 타임프레임의 Feature 계산
    
    Args:
        window_df: DataFrame [datetime, open, high, low, close, volume]
        prefix: Feature 이름 접두사 (예: '1m_', '1h_', '4h_')
    
    Returns:
        dict: Feature 딕셔너리
    """
    if len(window_df) < 30:
        # 최소 데이터 부족
        return None
    
    df = window_df.copy()
    
    # 가격 및 거래량 배열
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    
    current_close = closes[-1]
    current_volume = volumes[-1]
    
    features = {}
    
    # === 가격 변화 ===
    features[f'{prefix}price_change_1'] = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
    features[f'{prefix}price_change_3'] = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
    features[f'{prefix}price_change_5'] = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
    features[f'{prefix}price_change_10'] = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0
    
    # === 모멘텀 ===
    features[f'{prefix}momentum_5'] = np.mean(np.diff(closes[-6:])) if len(closes) >= 6 else 0
    features[f'{prefix}momentum_10'] = np.mean(np.diff(closes[-11:])) if len(closes) >= 11 else 0
    
    # === 거래량 ===
    volume_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
    features[f'{prefix}volume_change_1'] = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if len(volumes) >= 2 else 0
    features[f'{prefix}volume_change_5'] = (np.mean(volumes[-5:]) - np.mean(volumes[-10:-5])) / np.mean(volumes[-10:-5]) * 100 if len(volumes) >= 10 else 0
    features[f'{prefix}volume_ratio'] = current_volume / volume_ma if volume_ma > 0 else 1.0
    features[f'{prefix}volume_spike'] = 1 if current_volume > volume_ma * 1.5 else 0
    
    # === RSI ===
    rsi = calculate_rsi(closes, period=14)
    features[f'{prefix}rsi'] = rsi
    features[f'{prefix}rsi_oversold'] = 1 if rsi < 30 else 0
    features[f'{prefix}rsi_overbought'] = 1 if rsi > 70 else 0
    
    # === MACD ===
    macd, macd_signal, macd_hist = calculate_macd(closes)
    features[f'{prefix}macd'] = macd
    features[f'{prefix}macd_signal'] = macd_signal
    features[f'{prefix}macd_hist'] = macd_hist
    
    # === 볼린저 밴드 ===
    features[f'{prefix}bb_position'] = calculate_bollinger_bands(closes)
    
    # === 이동평균 ===
    ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_close
    ma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_close
    ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_close
    
    features[f'{prefix}ma_cross_5_10'] = 1 if ma_5 > ma_10 else -1
    features[f'{prefix}price_above_ma20'] = 1 if current_close > ma_20 else 0
    
    # === 변동성 ===
    features[f'{prefix}atr'] = calculate_atr(highs, lows, closes, period=14)
    features[f'{prefix}volatility_5'] = np.std(closes[-5:]) if len(closes) >= 5 else 0
    features[f'{prefix}volatility_10'] = np.std(closes[-10:]) if len(closes) >= 10 else 0
    features[f'{prefix}volatility_ratio'] = features[f'{prefix}volatility_5'] / (features[f'{prefix}volatility_10'] + 1e-10)
    features[f'{prefix}high_low_range'] = (highs[-1] - lows[-1]) / current_close * 100
    
    # === 시간 (1분봉만) ===
    if prefix == '1m_':
        current_hour = df['datetime'].iloc[-1].hour if 'datetime' in df.columns else 12
        features[f'{prefix}hour'] = current_hour
        features[f'{prefix}active_hours'] = 1 if 9 <= current_hour <= 23 else 0
    
    return features


def calculate_4h_simple_features(window_df, prefix='4h_'):
    """
    4시간봉의 단순화된 Feature (큰 추세만)
    
    Args:
        window_df: 4시간봉 DataFrame
        prefix: Feature 이름 접두사
    
    Returns:
        dict: 단순화된 Feature
    """
    if len(window_df) < 30:
        return None
    
    df = window_df.copy()
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    features = {}
    
    # EMA 추세 (50 vs 200)
    ema_50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1] if len(closes) >= 50 else closes[-1]
    ema_200 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1] if len(closes) >= 50 else closes[-1]  # 데이터 부족 시 단순화
    
    features[f'{prefix}ema_trend'] = 1 if ema_50 > ema_200 else -1
    
    # 추세 강도
    recent_high = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
    recent_low = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]
    current_price = closes[-1]
    
    if recent_high == recent_low:
        trend_strength = 0
    else:
        # 0 (하단) ~ 1 (상단)
        trend_strength = (current_price - recent_low) / (recent_high - recent_low)
        trend_strength = (trend_strength - 0.5) * 2  # -1 ~ +1로 변환
    
    features[f'{prefix}trend_strength'] = trend_strength
    
    # 주요 지지/저항 거리
    resistance = recent_high
    support = recent_low
    
    features[f'{prefix}resistance_distance'] = (resistance - current_price) / current_price * 100
    features[f'{prefix}support_distance'] = (current_price - support) / current_price * 100
    
    # 장기 모멘텀
    features[f'{prefix}momentum_10'] = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0
    features[f'{prefix}momentum_20'] = (closes[-1] - closes[-21]) / closes[-21] * 100 if len(closes) >= 21 else 0
    
    return features


def calculate_cross_timeframe_features(features_1m, features_1h, features_4h, window_1m, window_1h, window_4h):
    """
    타임프레임 간 관계 Feature
    
    Args:
        features_1m, features_1h, features_4h: 각 타임프레임의 Feature dict
        window_1m, window_1h, window_4h: 각 타임프레임의 DataFrame
    
    Returns:
        dict: 타임프레임 간 관계 Feature
    """
    features = {}
    
    # === 추세 일치도 ===
    trend_1m = 1 if features_1m['1m_macd'] > 0 else -1
    trend_1h = 1 if features_1h['1h_macd'] > 0 else -1
    trend_4h = features_4h['4h_ema_trend']
    
    # 세 추세가 모두 같으면 +1, 모두 다르면 -1
    if trend_1m == trend_1h == trend_4h:
        features['trend_alignment'] = 1
    elif trend_1m == trend_1h or trend_1h == trend_4h or trend_1m == trend_4h:
        features['trend_alignment'] = 0
    else:
        features['trend_alignment'] = -1
    
    # === RSI 다이버전스 ===
    features['rsi_divergence_1m_1h'] = features_1m['1m_rsi'] - features_1h['1h_rsi']
    features['rsi_divergence_1h_4h'] = features_1h['1h_rsi'] - 50  # 4h는 RSI 없으므로 중립값 대비
    
    # === 가격 위치 ===
    price_1m = window_1m['close'].iloc[-1]
    price_1h = window_1h['close'].iloc[-1]
    price_4h = window_4h['close'].iloc[-1]
    
    features['price_position_vs_1h'] = (price_1m - price_1h) / price_1h * 100
    features['price_position_vs_4h'] = (price_1m - price_4h) / price_4h * 100
    
    # === 거래량 비율 ===
    volume_1m = window_1m['volume'].iloc[-1]
    volume_1h = window_1h['volume'].iloc[-1] / 60 if window_1h['volume'].iloc[-1] > 0 else 1  # 1시간 = 60분
    features['volume_ratio_1m_vs_1h'] = volume_1m / volume_1h if volume_1h > 0 else 1.0
    
    # === 변동성 비율 ===
    features['volatility_ratio_1m_1h'] = features_1m['1m_atr'] / (features_1h['1h_atr'] + 1e-10)
    
    # === 추세 강도 차이 ===
    features['trend_strength_diff'] = features_4h['4h_trend_strength'] - (1 if features_1m['1m_macd'] > 0 else -1)
    
    # === 볼린저 밴드 위치 일치 ===
    bb_1m = features_1m['1m_bb_position']
    bb_1h = features_1h['1h_bb_position']
    
    if bb_1m > 0.5 and bb_1h > 0.5:
        features['bb_alignment'] = 1
    elif bb_1m < 0.5 and bb_1h < 0.5:
        features['bb_alignment'] = -1
    else:
        features['bb_alignment'] = 0
    
    # === 4시간 필터 강도 ===
    features['4h_filter_strength'] = features_4h['4h_ema_trend'] * abs(features_4h['4h_trend_strength'])
    
    return features


def calculate_multi_timeframe_features(window_1m, window_1h, window_4h):
    """
    멀티 타임프레임 통합 Feature 계산
    
    Args:
        window_1m: 1분봉 DataFrame (100개)
        window_1h: 1시간봉 DataFrame (100개)
        window_4h: 4시간봉 DataFrame (100개)
    
    Returns:
        dict: 통합 Feature (약 80개)
    """
    # 각 타임프레임 Feature
    features_1m = calculate_single_timeframe_features(window_1m, prefix='1m_')
    features_1h = calculate_single_timeframe_features(window_1h, prefix='1h_')
    features_4h = calculate_4h_simple_features(window_4h, prefix='4h_')
    
    if features_1m is None or features_1h is None or features_4h is None:
        return None
    
    # 타임프레임 간 관계 Feature
    cross_features = calculate_cross_timeframe_features(
        features_1m, features_1h, features_4h,
        window_1m, window_1h, window_4h
    )
    
    # 통합
    all_features = {**features_1m, **features_1h, **features_4h, **cross_features}
    
    # NaN 체크
    for key, value in all_features.items():
        if pd.isna(value) or (isinstance(value, (int, float)) and np.isnan(value)):
            all_features[key] = 0
    
    return all_features


def get_all_feature_columns():
    """전체 Feature 컬럼 리스트 반환"""
    
    # 1분봉 (27개)
    features_1m = [
        '1m_price_change_1', '1m_price_change_3', '1m_price_change_5', '1m_price_change_10',
        '1m_momentum_5', '1m_momentum_10',
        '1m_volume_change_1', '1m_volume_change_5', '1m_volume_ratio', '1m_volume_spike',
        '1m_rsi', '1m_rsi_oversold', '1m_rsi_overbought',
        '1m_macd', '1m_macd_signal', '1m_macd_hist',
        '1m_bb_position',
        '1m_ma_cross_5_10', '1m_price_above_ma20',
        '1m_atr', '1m_volatility_5', '1m_volatility_10', '1m_volatility_ratio', '1m_high_low_range',
        '1m_hour', '1m_active_hours'
    ]
    
    # 1시간봉 (25개, 시간 제외)
    features_1h = [
        '1h_price_change_1', '1h_price_change_3', '1h_price_change_5', '1h_price_change_10',
        '1h_momentum_5', '1h_momentum_10',
        '1h_volume_change_1', '1h_volume_change_5', '1h_volume_ratio', '1h_volume_spike',
        '1h_rsi', '1h_rsi_oversold', '1h_rsi_overbought',
        '1h_macd', '1h_macd_signal', '1h_macd_hist',
        '1h_bb_position',
        '1h_ma_cross_5_10', '1h_price_above_ma20',
        '1h_atr', '1h_volatility_5', '1h_volatility_10', '1h_volatility_ratio', '1h_high_low_range'
    ]
    
    # 4시간봉 (6개, 단순화)
    features_4h = [
        '4h_ema_trend',
        '4h_trend_strength',
        '4h_resistance_distance',
        '4h_support_distance',
        '4h_momentum_10',
        '4h_momentum_20'
    ]
    
    # 타임프레임 간 관계 (10개)
    cross_features = [
        'trend_alignment',
        'rsi_divergence_1m_1h',
        'rsi_divergence_1h_4h',
        'price_position_vs_1h',
        'price_position_vs_4h',
        'volume_ratio_1m_vs_1h',
        'volatility_ratio_1m_1h',
        'trend_strength_diff',
        'bb_alignment',
        '4h_filter_strength'
    ]
    
    return features_1m + features_1h + features_4h + cross_features


if __name__ == "__main__":
    print("="*80)
    print("멀티 타임프레임 Feature 테스트")
    print("="*80)
    
    feature_cols = get_all_feature_columns()
    print(f"\n총 Feature 개수: {len(feature_cols)}")
    print(f"\n- 1분봉: 27개")
    print(f"- 1시간봉: 25개")
    print(f"- 4시간봉: 6개 (단순화)")
    print(f"- 타임프레임 관계: 10개")
    print(f"\n전체 Feature 리스트:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

