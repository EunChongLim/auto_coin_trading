"""
A-E 규칙 기반 지표 계산 모듈 (model_v4 전용)
1-5분 스캘핑 최적화
"""

import pandas as pd
import numpy as np


def compute_rsi(series, period=14):
    """RSI (Relative Strength Index) 계산"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_ema(series, span, min_periods=None):
    """지수이동평균 (Exponential Moving Average) 계산"""
    if min_periods is None:
        min_periods = span
    return series.ewm(span=span, adjust=False, min_periods=min_periods).mean()


def compute_bollinger_bands(series, window=20, num_std=2):
    """볼린저 밴드 계산"""
    middle = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width_pct = ((upper - lower) / middle) * 100  # BB 폭 퍼센트
    
    return middle, upper, lower, width_pct


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence) 계산"""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def compute_atr(df, period=14):
    """
    ATR (Average True Range) 계산
    변동성 측정 및 스탑로스 계산에 사용
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return atr


def compute_vwap_session(df):
    """
    세션 VWAP (Volume Weighted Average Price) 계산
    금일 시작부터의 누적 VWAP
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # 날짜별로 그룹화하여 세션 VWAP 계산
    if isinstance(df.index, pd.DatetimeIndex):
        df_temp = df.copy()
        df_temp['date'] = df_temp.index.date
        df_temp['tp_volume'] = typical_price * df['volume']
        
        # 각 날짜별 누적 계산
        df_temp['cum_tp_vol'] = df_temp.groupby('date')['tp_volume'].cumsum()
        df_temp['cum_vol'] = df_temp.groupby('date')['volume'].cumsum()
        
        vwap = df_temp['cum_tp_vol'] / df_temp['cum_vol']
        return vwap
    else:
        # DatetimeIndex가 아닌 경우 전체 기간 VWAP
        cum_tp_vol = (typical_price * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        return cum_tp_vol / cum_vol


def add_all_indicators(df):
    """
    A-E 규칙에 필요한 모든 지표 추가
    
    Args:
        df: OHLCV 데이터프레임 (컬럼: open, high, low, close, volume)
    
    Returns:
        DataFrame: 지표가 추가된 데이터프레임
    """
    df = df.copy()
    
    # === 1. EMA (1분봉 기준) ===
    df['ema5'] = compute_ema(df['close'], 5)
    df['ema20'] = compute_ema(df['close'], 20)
    df['ema50'] = compute_ema(df['close'], 50)
    df['ema200'] = compute_ema(df['close'], 200)
    
    # === 2. RSI (모멘텀) ===
    df['rsi'] = compute_rsi(df['close'], 14)
    
    # === 3. MACD (모멘텀 확인) ===
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['close'])
    
    # === 4. Bollinger Bands (변동성, 브레이크아웃) ===
    df['bb_middle'], df['bb_upper'], df['bb_lower'], df['bb_width_pct'] = compute_bollinger_bands(df['close'], 20)
    
    # === 5. ATR (변동성, 스탑/사이징) ===
    df['atr'] = compute_atr(df, 14)
    df['atr_pct'] = (df['atr'] / df['close']) * 100  # ATR 퍼센트
    
    # === 6. 거래량 (유동성) ===
    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # === 7. VWAP (세션 기준선) ===
    df['vwap_session'] = compute_vwap_session(df)
    
    # === 8. 가격 변화율 ===
    df['price_change'] = df['close'].pct_change()
    
    # === 9. 캔들 패턴 (보조) ===
    df['body'] = abs(df['close'] - df['open'])
    df['body_pct'] = (df['body'] / df['close']) * 100
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    return df


if __name__ == "__main__":
    print("✅ indicators.py (A-E 규칙, model_v4) 모듈 로드 완료")
    print("📊 구현된 지표:")
    print("  - EMA: 5, 20, 50, 200")
    print("  - RSI: 14")
    print("  - MACD: 12-26-9 (histogram)")
    print("  - Bollinger Bands: width_pct")
    print("  - ATR: 14 (스탑/사이징)")
    print("  - Volume: ma20, ratio")
    print("  - VWAP: session")
    print("  - Candle Patterns: body, wicks")

