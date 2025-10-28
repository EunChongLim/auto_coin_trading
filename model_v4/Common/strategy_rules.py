"""
B규칙: 룰 기반 매수/매도 전략
A-E 규칙에 따른 시그널 생성
"""

import pandas as pd
import numpy as np


class RuleBasedStrategy:
    """
    A-E 규칙 기반 전략
    
    매수/매도 조건을 룰로 정의하고,
    ML 예측은 보조 신호로 사용
    """
    
    def __init__(self, 
                 ml_buy_threshold=0.25,
                 ml_sell_threshold=0.35,
                 atr_stop_multiplier=1.2,
                 risk_pct=1.0):
        """
        Args:
            ml_buy_threshold: ML 매수 확률 임계값
            ml_sell_threshold: ML 매도 확률 임계값
            atr_stop_multiplier: ATR 스탑 배수 (k)
            risk_pct: 위험 퍼센트 (0.5~1.0%)
        """
        self.ml_buy_threshold = ml_buy_threshold
        self.ml_sell_threshold = ml_sell_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_pct = risk_pct
    
    
    def check_long_signal(self, row, ml_prob_up=None):
        """
        롱(매수) 신호 체크
        
        B규칙 롱 시나리오:
        1. 레짐: ema50_15m > ema200_15m AND trend_score >= +2
        2. 유동성: volume_ratio > 1.3
        3. 위치: pos60 > 0.35
        4. 브레이크아웃: bb_width_pct >= p70 or VWAP 상향 리클레임
        5. 트리거: rsi > 50 AND macd_hist_rising
        6. (보조) ML: prob_up >= threshold
        
        Args:
            row: DataFrame의 한 행 (모든 지표 포함)
            ml_prob_up: ML 예측 상승 확률 (선택)
        
        Returns:
            bool: 매수 신호 여부
        """
        try:
            # 필수 조건 체크
            if pd.isna(row.get('ema50_15m')) or pd.isna(row.get('ema200_15m')):
                return False
            
            # 1. 레짐 필터 (상승 추세)
            regime_bull = row.get('ema50_15m', 0) > row.get('ema200_15m', 0)
            trend_aligned = row.get('trend_score', 0) >= 2
            
            if not (regime_bull and trend_aligned):
                return False
            
            # 2. 유동성 필터
            liquidity_ok = row.get('volume_ratio', 0) > 1.3
            
            if not liquidity_ok:
                return False
            
            # 3. 위치 필터 (하단이 아닌 곳에서)
            position_ok = row.get('pos60', 0) > 0.35
            
            if not position_ok:
                return False
            
            # 4. 브레이크아웃 컨텍스트 (변동성 확장 or VWAP 돌파)
            # bb_width_pct의 70th percentile 계산은 rolling 필요 → 단순화: > 2.5%
            volatility_expansion = row.get('bb_width_pct', 0) > 2.5
            vwap_reclaim = row.get('price_vs_vwap', -1) > 0  # 가격 > VWAP
            
            breakout_context = volatility_expansion or vwap_reclaim
            
            if not breakout_context:
                return False
            
            # 5. 모멘텀 트리거
            rsi_ok = row.get('rsi', 0) > 50
            macd_rising = row.get('macd_hist_rising', 0) == 1
            
            if not (rsi_ok and macd_rising):
                return False
            
            # 6. (보조) ML 확률
            if ml_prob_up is not None:
                ml_ok = ml_prob_up >= self.ml_buy_threshold
                if not ml_ok:
                    return False
            
            return True
            
        except Exception as e:
            return False
    
    
    def check_short_signal(self, row, ml_prob_down=None):
        """
        숏(매도) 신호 체크
        
        B규칙 숏 시나리오 (롱의 반대):
        1. 레짐: ema50_15m < ema200_15m AND trend_score <= -2
        2. 유동성: volume_ratio > 1.3
        3. 위치: pos60 < 0.65
        4. 브레이크아웃: bb_width_pct >= p70 or VWAP 하향 리젝션
        5. 트리거: rsi < 50 AND macd_hist_falling
        6. (보조) ML: prob_down >= threshold
        
        Args:
            row: DataFrame의 한 행
            ml_prob_down: ML 예측 하락 확률 (선택)
        
        Returns:
            bool: 매도 신호 여부
        """
        try:
            if pd.isna(row.get('ema50_15m')) or pd.isna(row.get('ema200_15m')):
                return False
            
            # 1. 레짐 필터 (하락 추세)
            regime_bear = row.get('ema50_15m', 0) < row.get('ema200_15m', 0)
            trend_aligned = row.get('trend_score', 0) <= -2
            
            if not (regime_bear and trend_aligned):
                return False
            
            # 2. 유동성 필터
            liquidity_ok = row.get('volume_ratio', 0) > 1.3
            
            if not liquidity_ok:
                return False
            
            # 3. 위치 필터 (상단이 아닌 곳에서)
            position_ok = row.get('pos60', 1) < 0.65
            
            if not position_ok:
                return False
            
            # 4. 브레이크아웃 컨텍스트
            volatility_expansion = row.get('bb_width_pct', 0) > 2.5
            vwap_reject = row.get('price_vs_vwap', 1) < 0  # 가격 < VWAP
            
            breakout_context = volatility_expansion or vwap_reject
            
            if not breakout_context:
                return False
            
            # 5. 모멘텀 트리거
            rsi_ok = row.get('rsi', 100) < 50
            macd_falling = row.get('macd_hist_rising', 1) == 0  # not rising = falling
            
            if not (rsi_ok and macd_falling):
                return False
            
            # 6. (보조) ML 확률
            if ml_prob_down is not None:
                ml_ok = ml_prob_down >= self.ml_sell_threshold
                if not ml_ok:
                    return False
            
            return True
            
        except Exception as e:
            return False
    
    
    def calculate_position_size(self, equity, entry_price, atr):
        """
        ATR 기반 포지션 사이징
        
        B규칙 사이징:
        size = (equity * risk_pct) / (k * atr)
        
        Args:
            equity: 현재 자본
            entry_price: 진입 가격
            atr: 현재 ATR
        
        Returns:
            float: 포지션 크기 (코인 개수)
        """
        if atr <= 0:
            # ATR이 0이면 고정 비율 사용
            return equity * 0.995 / entry_price
        
        # 위험 금액
        risk_amount = equity * (self.risk_pct / 100)
        
        # 1R = k * ATR
        one_r = self.atr_stop_multiplier * atr
        
        # 포지션 사이즈 = 위험 금액 / 1R
        position_size = risk_amount / one_r
        
        # 최대 99.5% 자본 사용
        max_position = equity * 0.995 / entry_price
        
        return min(position_size, max_position)
    
    
    def calculate_stop_loss(self, entry_price, atr, direction='long'):
        """
        ATR 기반 스탑로스 계산
        
        B규칙 스탑:
        - 롱: entry - k * atr
        - 숏: entry + k * atr
        
        Args:
            entry_price: 진입 가격
            atr: 현재 ATR
            direction: 'long' or 'short'
        
        Returns:
            float: 스탑로스 가격
        """
        if direction == 'long':
            return entry_price - (self.atr_stop_multiplier * atr)
        else:
            return entry_price + (self.atr_stop_multiplier * atr)
    
    
    def check_partial_exit(self, entry_price, current_price, atr, direction='long'):
        """
        부분 청산 조건 체크
        
        B규칙 청산:
        +1R 도달 시 50% 부분 청산
        
        Args:
            entry_price: 진입 가격
            current_price: 현재 가격
            atr: ATR
            direction: 'long' or 'short'
        
        Returns:
            dict: {'should_exit': bool, 'exit_ratio': float, 'reason': str}
        """
        one_r = self.atr_stop_multiplier * atr
        
        if direction == 'long':
            profit = current_price - entry_price
            if profit >= one_r:
                return {
                    'should_exit': True,
                    'exit_ratio': 0.5,
                    'reason': f'+1R 부분청산 (profit={profit:.0f}, 1R={one_r:.0f})'
                }
        else:
            profit = entry_price - current_price
            if profit >= one_r:
                return {
                    'should_exit': True,
                    'exit_ratio': 0.5,
                    'reason': f'+1R 부분청산 (profit={profit:.0f}, 1R={one_r:.0f})'
                }
        
        return {'should_exit': False, 'exit_ratio': 0, 'reason': ''}


if __name__ == "__main__":
    print("✅ strategy_rules.py (B규칙 전략 엔진) 로드 완료")
    print("📊 구현된 기능:")
    print("  - check_long_signal(): 롱 진입 조건")
    print("  - check_short_signal(): 숏 진입 조건")
    print("  - calculate_position_size(): ATR 기반 사이징")
    print("  - calculate_stop_loss(): ATR 기반 스탑")
    print("  - check_partial_exit(): +1R 부분청산")

