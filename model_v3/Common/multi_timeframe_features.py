"""
A-E 규칙 기반 멀티타임프레임 특징 생성
1분/5분/15분/60분 통합
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from indicators import add_all_indicators, compute_ema


def add_multi_timeframe_features(df_1m):
    """
    A-E 규칙에 따른 멀티타임프레임 특징 추가
    
    핵심 특징:
    1. 레짐(추세): ema50_15m, ema200_15m
    2. 위치(구조): pos60 = (close - low_60m) / (high_60m - low_60m)
    3. 유동성: vol_1m > 1.3 * vol_1m_ma20
    4. 변동성(스탑): atr14_1m
    5. 변동성(브레이크아웃): bb_width_pct_20_1m
    6. 모멘텀: rsi14_1m, macd_hist_1m
    7. MTF 추세 일치도: trend_score = Σ sign(ema20-ema50) for 1/5/15/60m
    8. 세션 기준선: vwap_session
    
    Args:
        df_1m: 1분봉 OHLCV 데이터 (indicators 추가된 상태)
    
    Returns:
        DataFrame: 멀티타임프레임 특징이 추가된 데이터프레임
    """
    df = df_1m.copy()
    
    # === 1. 5분봉 리샘플링 ===
    df_5m = df.resample('5T', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_5m['ema20'] = compute_ema(df_5m['close'], 20)
    df_5m['ema50'] = compute_ema(df_5m['close'], 50)
    
    # === 2. 15분봉 리샘플링 ===
    df_15m = df.resample('15T', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_15m['ema20'] = compute_ema(df_15m['close'], 20)
    df_15m['ema50'] = compute_ema(df_15m['close'], 50)
    df_15m['ema200'] = compute_ema(df_15m['close'], 200)
    
    # === 3. 60분봉 리샘플링 ===
    df_60m = df.resample('60T', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_60m['ema20'] = compute_ema(df_60m['close'], 20)
    df_60m['ema50'] = compute_ema(df_60m['close'], 50)
    df_60m['high_60m'] = df_60m['high']
    df_60m['low_60m'] = df_60m['low']
    
    # === 4. 1분봉으로 Forward Fill (미마감 봉 금지) ===
    # 5분봉 특징
    df['ema20_5m'] = df_5m['ema20'].reindex(df.index, method='ffill')
    df['ema50_5m'] = df_5m['ema50'].reindex(df.index, method='ffill')
    
    # 15분봉 특징
    df['ema20_15m'] = df_15m['ema20'].reindex(df.index, method='ffill')
    df['ema50_15m'] = df_15m['ema50'].reindex(df.index, method='ffill')
    df['ema200_15m'] = df_15m['ema200'].reindex(df.index, method='ffill')
    
    # 60분봉 특징
    df['ema20_60m'] = df_60m['ema20'].reindex(df.index, method='ffill')
    df['ema50_60m'] = df_60m['ema50'].reindex(df.index, method='ffill')
    df['high_60m'] = df_60m['high_60m'].reindex(df.index, method='ffill')
    df['low_60m'] = df_60m['low_60m'].reindex(df.index, method='ffill')
    
    # === 5. A규칙: 핵심 특징 계산 ===
    
    # (1) 레짐 필터: ema50_15m vs ema200_15m
    df['regime_bull'] = (df['ema50_15m'] > df['ema200_15m']).astype(int)
    
    # (2) 위치(구조): pos60
    df['pos60'] = (df['close'] - df['low_60m']) / (df['high_60m'] - df['low_60m'] + 1e-10)
    
    # (3) 유동성 필터 (이미 indicators.py에서 계산됨)
    # volume_ratio = volume / volume_ma20
    
    # (4) 변동성 지표 (이미 indicators.py에서 계산됨)
    # atr, atr_pct, bb_width_pct
    
    # (5) 모멘텀 지표 (이미 indicators.py에서 계산됨)
    # rsi, macd_hist
    
    # (6) MTF 추세 일치도: trend_score
    sign_1m = np.sign(df['ema20'] - df['ema50'])
    sign_5m = np.sign(df['ema20_5m'] - df['ema50_5m'])
    sign_15m = np.sign(df['ema50_15m'] - df['ema200_15m'])  # 15분은 50 vs 200
    sign_60m = np.sign(df['ema20_60m'] - df['ema50_60m'])
    
    df['trend_score'] = sign_1m + sign_5m + sign_15m + sign_60m  # -4 ~ +4
    
    # (7) VWAP 관련 (이미 indicators.py에서 계산됨)
    # vwap_session
    
    # (8) 가격 vs VWAP
    df['price_vs_vwap'] = (df['close'] - df['vwap_session']) / (df['vwap_session'] + 1e-10)
    
    # === 6. 추가 보조 특징 ===
    
    # EMA 크로스
    df['ema_cross_20_50'] = (df['ema20'] > df['ema50']).astype(int)
    
    # 가격 포지션 (현재가 vs EMA20)
    df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
    
    # RSI 레벨
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # MACD 기울기 (t-1 vs t)
    df['macd_hist_rising'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)
    
    return df


def prepare_multi_timeframe_data(df_1m, future_minutes=3, down_threshold=-0.001, up_threshold=0.002):
    """
    학습용 데이터 준비 (특징 + 라벨)
    
    Args:
        df_1m: 1분봉 OHLCV 데이터
        future_minutes: 미래 N분 후 가격으로 라벨링
        down_threshold: 하락 기준 (%)
        up_threshold: 상승 기준 (%)
    
    Returns:
        X (features), y (labels)
    """
    # 1단계: 지표 계산
    df = add_all_indicators(df_1m)
    
    # 2단계: 멀티타임프레임 특징
    df = add_multi_timeframe_features(df)
    
    # 3단계: 라벨 생성 (N분 후 가격 변화율)
    df['future_return'] = df['close'].shift(-future_minutes) / df['close'] - 1
    
    # 라벨링: 0=Down, 1=Sideways, 2=Up
    df['label'] = 1  # 기본값 Sideways
    df.loc[df['future_return'] <= down_threshold, 'label'] = 0  # Down
    df.loc[df['future_return'] >= up_threshold, 'label'] = 2  # Up
    
    # 4단계: NaN 제거 및 특징/라벨 분리
    df = df.dropna()
    
    # 특징 선택 (라벨링용 컬럼 제외)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_return', 'label',
                    'body', 'upper_wick', 'lower_wick',  # 캔들 패턴은 학습에서 제외
                    'bb_middle', 'bb_upper', 'bb_lower', 'macd', 'macd_signal']  # 중복/파생 컬럼
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    return X, y


if __name__ == "__main__":
    print("✅ multi_timeframe_features.py (A-E 규칙) 모듈 로드 완료")
    print("📊 구현된 멀티타임프레임 특징:")
    print("  - 레짐: ema50_15m, ema200_15m, regime_bull")
    print("  - 위치: pos60 (60분 high-low 기준)")
    print("  - 유동성: volume_ratio (> 1.3 필터)")
    print("  - 변동성: atr, bb_width_pct")
    print("  - 모멘텀: rsi, macd_hist, macd_hist_rising")
    print("  - MTF 추세: trend_score (-4 ~ +4)")
    print("  - 세션: vwap_session, price_vs_vwap")
