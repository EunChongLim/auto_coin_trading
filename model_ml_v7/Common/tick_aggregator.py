"""
틱 데이터를 1분 단위로 집계
"""

import pandas as pd
import numpy as np


def aggregate_ticks_to_minute(tick_df):
    """
    틱 데이터 → 1분 단위 집계
    
    Args:
        tick_df: 틱 데이터 DataFrame
                 컬럼: ['seq', 'timestamp', 'volume', 'price', 'ask_bid', 'datetime']
    
    Returns:
        DataFrame: 1분 단위 집계
    """
    # datetime이 없으면 생성
    if 'datetime' not in tick_df.columns:
        tick_df['datetime'] = pd.to_datetime(tick_df['timestamp'], unit='ms')
    
    # 1분 단위로 floor
    tick_df = tick_df.copy()
    tick_df['minute'] = tick_df['datetime'].dt.floor('min')
    
    # 기본 OHLCV
    ohlcv = tick_df.groupby('minute').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
        'seq': 'count'
    })
    
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume', 'tick_count']
    
    # 매수/매도 분리
    buy_df = tick_df[tick_df['ask_bid'] == 'BID'].groupby('minute')['volume'].sum()
    sell_df = tick_df[tick_df['ask_bid'] == 'ASK'].groupby('minute')['volume'].sum()
    
    ohlcv['buy_volume'] = buy_df.fillna(0)
    ohlcv['sell_volume'] = sell_df.fillna(0)
    
    # 매수 압력
    total_volume = ohlcv['buy_volume'] + ohlcv['sell_volume']
    ohlcv['buy_pressure'] = ohlcv['buy_volume'] / (total_volume + 1e-10)
    
    # 대량 거래 감지
    large_threshold = tick_df['volume'].quantile(0.95)
    
    large_buys = tick_df[
        (tick_df['volume'] > large_threshold) & 
        (tick_df['ask_bid'] == 'BID')
    ].groupby('minute').size()
    
    large_sells = tick_df[
        (tick_df['volume'] > large_threshold) & 
        (tick_df['ask_bid'] == 'ASK')
    ].groupby('minute').size()
    
    ohlcv['large_buy_count'] = large_buys.fillna(0).astype(int)
    ohlcv['large_sell_count'] = large_sells.fillna(0).astype(int)
    
    # VWAP (Volume Weighted Average Price)
    tick_df['value'] = tick_df['price'] * tick_df['volume']
    vwap = tick_df.groupby('minute').apply(
        lambda x: x['value'].sum() / (x['volume'].sum() + 1e-10),
        include_groups=False
    )
    ohlcv['vwap'] = vwap
    
    # 인덱스를 컬럼으로
    ohlcv = ohlcv.reset_index()
    ohlcv.rename(columns={'minute': 'datetime'}, inplace=True)
    
    return ohlcv


def add_tick_features(tick_agg_df):
    """
    틱 집계 DataFrame에 추가 Feature 계산
    
    Args:
        tick_agg_df: aggregate_ticks_to_minute의 결과
    
    Returns:
        DataFrame: Feature 추가된 DataFrame
    """
    df = tick_agg_df.copy()
    
    # 매수 압력 이동평균
    df['buy_pressure_ma5'] = df['buy_pressure'].rolling(5, min_periods=1).mean()
    df['buy_pressure_ma20'] = df['buy_pressure'].rolling(20, min_periods=1).mean()
    
    # 매수 압력 변화
    df['buy_pressure_change'] = df['buy_pressure'].diff()
    
    # 체결 속도 이동평균
    df['tick_speed_ma5'] = df['tick_count'].rolling(5, min_periods=1).mean()
    df['tick_speed_ma20'] = df['tick_count'].rolling(20, min_periods=1).mean()
    
    # 체결 속도 비율 (현재 / 평균)
    df['tick_speed_ratio'] = df['tick_count'] / (df['tick_speed_ma20'] + 1e-10)
    
    # 매수 압력 표준편차 (변동성)
    df['buy_pressure_std'] = df['buy_pressure'].rolling(20, min_periods=1).std()
    
    # 대량 거래 불균형
    df['large_trade_imbalance'] = df['large_buy_count'] - df['large_sell_count']
    
    # 거래량 불균형
    df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['volume'] + 1e-10)
    
    # VWAP 대비 가격
    df['price_vs_vwap'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
    
    return df


def get_tick_feature_columns():
    """틱 기반 Feature 컬럼 리스트"""
    return [
        # 기본 틱 정보
        'tick_count',
        'buy_pressure',
        'large_buy_count',
        'large_sell_count',
        'vwap',
        
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


if __name__ == "__main__":
    # 테스트
    import sys
    sys.path.insert(0, '..')
    from Common.tick_loader import download_tick_data
    
    print("="*80)
    print("틱 집계 테스트 - 2024-01-01")
    print("="*80)
    
    # 틱 데이터 로드
    tick_df = download_tick_data("20240101")
    
    if tick_df is not None:
        # 1분 집계
        minute_df = aggregate_ticks_to_minute(tick_df)
        
        print(f"\n[집계 결과]")
        print(f"틱 건수: {len(tick_df):,}")
        print(f"1분봉 건수: {len(minute_df):,}")
        print(f"\n컬럼: {list(minute_df.columns)}")
        
        print(f"\n[샘플 데이터 (처음 10분)]")
        print(minute_df.head(10))
        
        # Feature 추가
        minute_df = add_tick_features(minute_df)
        
        print(f"\n[Feature 추가 후 컬럼]")
        print(list(minute_df.columns))
        
        print(f"\n[틱 Feature (처음 10분)]")
        tick_cols = get_tick_feature_columns()
        print(minute_df[tick_cols].head(10))

