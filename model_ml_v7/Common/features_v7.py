"""
v7 Feature Engineering
v6 (1분봉 Feature) + 틱 Feature 통합
"""

import pandas as pd
import numpy as np
import sys
import os

# v6 features_window 가져오기
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model_ml_v6'))
from Common.features_window import calculate_features_from_window as calc_v6_features


def calculate_combined_features(candle_window, tick_window=None):
    """
    1분봉 + 틱 통합 Feature 계산
    
    Args:
        candle_window: DataFrame (최근 100개 1분봉)
                       컬럼: open, high, low, close, volume, datetime
        tick_window: DataFrame (최근 100개 1분의 틱 집계) - Optional
                     컬럼: buy_pressure, tick_count, large_buy_count 등
    
    Returns:
        dict: v6 Feature + 틱 Feature
    """
    # === 1. v6 기본 Feature (1분봉만) ===
    v6_features = calc_v6_features(candle_window, window_size=len(candle_window))
    
    if v6_features is None:
        return None
    
    # tick_window가 없으면 v6 Feature만 반환
    if tick_window is None or len(tick_window) == 0:
        return v6_features
    
    # === 2. 틱 Feature 추가 ===
    latest_tick = tick_window.iloc[-1]
    
    # 최신 1분 틱 정보
    v6_features['tick_count'] = latest_tick.get('tick_count', 0)
    v6_features['buy_pressure'] = latest_tick.get('buy_pressure', 0.5)
    v6_features['large_buy_count'] = latest_tick.get('large_buy_count', 0)
    v6_features['large_sell_count'] = latest_tick.get('large_sell_count', 0)
    v6_features['vwap'] = latest_tick.get('vwap', candle_window['close'].iloc[-1])
    
    # 틱 추세 정보 (최근 N분)
    if len(tick_window) >= 5:
        v6_features['buy_pressure_ma5'] = tick_window['buy_pressure'].tail(5).mean()
        v6_features['buy_pressure_change'] = tick_window['buy_pressure'].iloc[-1] - tick_window['buy_pressure'].iloc[-2]
        v6_features['tick_speed_ma5'] = tick_window['tick_count'].tail(5).mean()
    else:
        v6_features['buy_pressure_ma5'] = latest_tick.get('buy_pressure', 0.5)
        v6_features['buy_pressure_change'] = 0
        v6_features['tick_speed_ma5'] = latest_tick.get('tick_count', 0)
    
    if len(tick_window) >= 20:
        v6_features['buy_pressure_ma20'] = tick_window['buy_pressure'].tail(20).mean()
        v6_features['tick_speed_ma20'] = tick_window['tick_count'].tail(20).mean()
        v6_features['buy_pressure_std'] = tick_window['buy_pressure'].tail(20).std()
    else:
        v6_features['buy_pressure_ma20'] = latest_tick.get('buy_pressure', 0.5)
        v6_features['tick_speed_ma20'] = latest_tick.get('tick_count', 0)
        v6_features['buy_pressure_std'] = 0
    
    # 체결 속도 비율
    tick_speed_ma20 = v6_features['tick_speed_ma20']
    v6_features['tick_speed_ratio'] = latest_tick.get('tick_count', 0) / (tick_speed_ma20 + 1e-10)
    
    # 불균형 지표
    v6_features['large_trade_imbalance'] = latest_tick.get('large_buy_count', 0) - latest_tick.get('large_sell_count', 0)
    
    buy_vol = latest_tick.get('buy_volume', 0)
    sell_vol = latest_tick.get('sell_volume', 0)
    total_vol = buy_vol + sell_vol
    v6_features['volume_imbalance'] = (buy_vol - sell_vol) / (total_vol + 1e-10)
    
    # VWAP 대비 가격
    vwap = latest_tick.get('vwap', candle_window['close'].iloc[-1])
    current_price = candle_window['close'].iloc[-1]
    v6_features['price_vs_vwap'] = ((current_price - vwap) / vwap) * 100
    
    # NaN 체크
    for key, value in v6_features.items():
        if pd.isna(value) or (isinstance(value, (int, float)) and np.isnan(value)):
            v6_features[key] = 0
    
    return v6_features


def get_all_feature_columns():
    """전체 Feature 컬럼 리스트 (v6 + 틱)"""
    
    # v6 기본 Feature
    v6_features = [
        # 가격 변화
        'price_change_1', 'price_change_3', 'price_change_5', 'price_change_10',
        'momentum_5', 'momentum_10',
        
        # 거래량
        'volume_change_1', 'volume_change_5', 'volume_ratio', 'volume_spike',
        
        # RSI
        'rsi', 'rsi_oversold', 'rsi_overbought',
        
        # MACD
        'macd', 'macd_signal', 'macd_hist',
        
        # 볼린저 밴드
        'bb_position',
        
        # 이동평균
        'ma_cross_5_10', 'price_above_ma20',
        
        # 변동성
        'atr', 'volatility_5', 'volatility_10', 'volatility_ratio', 'high_low_range',
        
        # 시간
        'hour', 'active_hours'
    ]
    
    # 틱 Feature
    tick_features = [
        # 기본 틱 정보
        'tick_count',
        'buy_pressure',
        'large_buy_count',
        'large_sell_count',
        
        # 매수 압력 추세
        'buy_pressure_ma5',
        'buy_pressure_ma20',
        'buy_pressure_change',
        'buy_pressure_std',
        
        # 체결 속도
        'tick_speed_ma5',
        'tick_speed_ma20',
        'tick_speed_ratio',
        
        # 불균형
        'large_trade_imbalance',
        'volume_imbalance',
        'price_vs_vwap'
    ]
    
    return v6_features + tick_features


if __name__ == "__main__":
    print("="*80)
    print("v7 Feature 테스트")
    print("="*80)
    
    feature_cols = get_all_feature_columns()
    print(f"\n총 Feature 개수: {len(feature_cols)}")
    print(f"\nv6 Feature (1분봉): {27}개")
    print(f"틱 Feature: {len(feature_cols) - 27}개")
    print(f"\n전체 Feature 리스트:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

