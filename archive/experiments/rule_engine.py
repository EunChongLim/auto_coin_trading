"""
규칙 기반 신호 생성 모듈
RSI, MA 등 기술적 지표를 기반으로 매수/매도 신호 생성
"""

import pandas as pd
import numpy as np


def generate_rule_signal(df):
    """
    RSI & MA 기반 기본 매매 조건
    
    Args:
        df: 지표가 포함된 OHLCV 데이터프레임
    
    Returns:
        tuple: (buy_signal, sell_signal) - pandas Series (Boolean)
    """
    # 매수 신호: RSI 적정 범위 + 상승 추세 + 볼륨 확인
    buy_signal = (
        (df['rsi'] > 30) & (df['rsi'] < 60) &
        (df['close'] > df['ma_fast']) &
        (df['ma_fast'] > df['ma_slow']) &
        (df['volume'] > df['volume_ma'] * 1.1)
    )
    
    # 매도 신호: RSI 과매수 또는 하락 추세
    sell_signal = (
        (df['rsi'] > 70) |
        (df['close'] < df['ma_slow']) |
        (df['close'] < df['bb_lower'])
    )
    
    return buy_signal, sell_signal


def generate_conservative_signal(df):
    """
    보수적인 매매 신호 (더 엄격한 조건)
    
    Args:
        df: 지표가 포함된 OHLCV 데이터프레임
    
    Returns:
        tuple: (buy_signal, sell_signal)
    """
    # 매수: 매우 과매도 + 강한 상승 추세
    buy_signal = (
        (df['rsi'] > 25) & (df['rsi'] < 35) &
        (df['close'] > df['ma_fast']) &
        (df['ma_fast'] > df['ma_slow']) &
        (df['volume'] > df['volume_ma'] * 2.0) &
        (df['macd'] > df['macd_signal'])
    )
    
    # 매도: 과매수 또는 강한 하락 신호
    sell_signal = (
        (df['rsi'] > 80) |
        ((df['close'] < df['ma_fast']) & (df['ma_fast'] < df['ma_slow'])) |
        (df['macd'] < df['macd_signal'])
    )
    
    return buy_signal, sell_signal


def generate_aggressive_signal(df):
    """
    공격적인 매매 신호 (더 느슨한 조건)
    
    Args:
        df: 지표가 포함된 OHLCV 데이터프레임
    
    Returns:
        tuple: (buy_signal, sell_signal)
    """
    # 매수: 넓은 RSI 범위 + 기본 조건
    buy_signal = (
        (df['rsi'] > 35) & (df['rsi'] < 65) &
        (df['close'] > df['ma_fast']) &
        (df['volume'] > df['volume_ma'])
    )
    
    # 매도: RSI 과매수만 체크
    sell_signal = (
        (df['rsi'] > 75)
    )
    
    return buy_signal, sell_signal


class RuleEngine:
    """
    규칙 기반 신호 생성 엔진
    """
    
    def __init__(self, strategy='normal'):
        """
        Args:
            strategy: 'conservative', 'normal', 'aggressive'
        """
        self.strategy = strategy
        
        if strategy == 'conservative':
            self.signal_func = generate_conservative_signal
        elif strategy == 'aggressive':
            self.signal_func = generate_aggressive_signal
        else:
            self.signal_func = generate_rule_signal
        
        print(f"📋 규칙 엔진 초기화 (전략: {strategy})")
    
    def get_signals(self, df):
        """
        매수/매도 신호 반환
        
        Args:
            df: 지표가 포함된 DataFrame
        
        Returns:
            tuple: (buy_signal, sell_signal)
        """
        return self.signal_func(df)


if __name__ == "__main__":
    print("✅ rule_engine.py 모듈 로드 완료")
    print("📊 사용 가능한 함수:")
    print("  - generate_rule_signal() - 기본 전략")
    print("  - generate_conservative_signal() - 보수적 전략")
    print("  - generate_aggressive_signal() - 공격적 전략")
    print("  - RuleEngine 클래스")

