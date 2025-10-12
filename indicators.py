"""
기술적 지표 계산 모듈
RSI, MA, Bollinger Bands 등
"""

import pandas as pd
import numpy as np


def compute_rsi(series, period=14):
    """
    RSI (Relative Strength Index) 계산
    
    Args:
        series: 가격 시리즈 (pandas Series)
        period: RSI 기간 (기본 14)
    
    Returns:
        pandas Series: RSI 값 (0~100)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_ma(series, window):
    """
    이동평균 (Moving Average) 계산
    
    Args:
        series: 가격 시리즈
        window: 윈도우 크기
    
    Returns:
        pandas Series: MA 값
    """
    return series.rolling(window=window).mean()


def compute_ema(series, span):
    """
    지수이동평균 (Exponential Moving Average) 계산
    
    Args:
        series: 가격 시리즈
        span: EMA 기간
    
    Returns:
        pandas Series: EMA 값
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_bollinger_bands(series, window=20, num_std=2):
    """
    볼린저 밴드 계산
    
    Args:
        series: 가격 시리즈
        window: 윈도우 크기
        num_std: 표준편차 배수
    
    Returns:
        tuple: (middle, upper, lower)
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return middle, upper, lower


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence) 계산
    
    Args:
        series: 가격 시리즈
        fast: 빠른 EMA 기간
        slow: 느린 EMA 기간
        signal: 시그널 라인 기간
    
    Returns:
        tuple: (macd, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def compute_volume_ma(volume_series, window=20):
    """
    거래량 이동평균 계산
    
    Args:
        volume_series: 거래량 시리즈
        window: 윈도우 크기
    
    Returns:
        pandas Series: 거래량 MA
    """
    return volume_series.rolling(window=window).mean()


def add_all_indicators(df):
    """
    DataFrame에 모든 지표 추가
    
    Args:
        df: OHLCV 데이터프레임 (컬럼: open, high, low, close, volume)
    
    Returns:
        DataFrame: 지표가 추가된 데이터프레임
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = compute_rsi(df['close'], 14)
    
    # 이동평균
    df['ma_fast'] = compute_ma(df['close'], 5)
    df['ma_slow'] = compute_ma(df['close'], 20)
    
    # 볼린저 밴드
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = compute_bollinger_bands(df['close'], 20)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['close'])
    
    # 거래량 지표
    df['volume_ma'] = compute_volume_ma(df['volume'], 20)
    
    # 가격 변화율
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(periods=5)
    
    return df


if __name__ == "__main__":
    # 테스트
    print("✅ indicators.py 모듈 로드 완료")
    print("📊 사용 가능한 함수:")
    print("  - compute_rsi()")
    print("  - compute_ma()")
    print("  - compute_ema()")
    print("  - compute_bollinger_bands()")
    print("  - compute_macd()")
    print("  - compute_volume_ma()")
    print("  - add_all_indicators()")

