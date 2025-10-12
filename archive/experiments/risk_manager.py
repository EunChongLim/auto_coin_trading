"""
리스크 관리 모듈
손절/익절, 포지션 크기 관리
"""


class RiskManager:
    """
    리스크 관리 클래스
    """
    
    def __init__(self, stop_loss_pct=0.005, take_profit_pct=0.005, fee_rate=0.0005):
        """
        Args:
            stop_loss_pct: 손절 비율 (0.005 = 0.5%)
            take_profit_pct: 익절 비율 (0.005 = 0.5%)
            fee_rate: 수수료 비율 (0.0005 = 0.05%)
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.fee_rate = fee_rate
        
        print(f"🛡️  리스크 관리 초기화")
        print(f"   - 손절: {stop_loss_pct*100:.2f}%")
        print(f"   - 익절: {take_profit_pct*100:.2f}%")
        print(f"   - 수수료: {fee_rate*100:.3f}%")
    
    def should_close_position(self, entry_price, current_price):
        """
        포지션을 청산해야 하는지 판단
        
        Args:
            entry_price: 진입 가격
            current_price: 현재 가격
        
        Returns:
            str or None: 'STOP_LOSS', 'TAKE_PROFIT', None
        """
        if entry_price == 0 or current_price == 0:
            return None
        
        profit_rate = (current_price - entry_price) / entry_price
        
        if profit_rate <= -self.stop_loss_pct:
            return "STOP_LOSS"
        elif profit_rate >= self.take_profit_pct:
            return "TAKE_PROFIT"
        
        return None
    
    def calculate_position_size(self, balance, price, max_position_ratio=1.0):
        """
        포지션 크기 계산
        
        Args:
            balance: 현재 잔고
            price: 현재 가격
            max_position_ratio: 최대 포지션 비율 (1.0 = 전액)
        
        Returns:
            float: 매수할 코인 수량
        """
        available_balance = balance * max_position_ratio * (1 - self.fee_rate)
        coin_amount = available_balance / price
        
        return coin_amount
    
    def calculate_profit(self, entry_price, exit_price, coin_amount):
        """
        수익 계산 (수수료 포함)
        
        Args:
            entry_price: 진입 가격
            exit_price: 청산 가격
            coin_amount: 코인 수량
        
        Returns:
            float: 실현 수익
        """
        buy_cost = entry_price * coin_amount * (1 + self.fee_rate)
        sell_revenue = exit_price * coin_amount * (1 - self.fee_rate)
        profit = sell_revenue - buy_cost
        
        return profit
    
    def get_profit_rate(self, entry_price, current_price):
        """
        수익률 계산
        
        Args:
            entry_price: 진입 가격
            current_price: 현재 가격
        
        Returns:
            float: 수익률 (소수)
        """
        if entry_price == 0:
            return 0.0
        
        return (current_price - entry_price) / entry_price


class AdaptiveRiskManager(RiskManager):
    """
    적응형 리스크 관리 (변동성 기반)
    """
    
    def __init__(self, base_stop_loss=0.005, base_take_profit=0.005, fee_rate=0.0005):
        super().__init__(base_stop_loss, base_take_profit, fee_rate)
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.volatility = None
        
        print(f"🔄 적응형 리스크 관리 활성화")
    
    def update_volatility(self, price_series, window=20):
        """
        변동성 업데이트
        
        Args:
            price_series: 가격 시리즈
            window: 윈도우 크기
        """
        returns = price_series.pct_change()
        self.volatility = returns.rolling(window=window).std().iloc[-1]
    
    def adjust_thresholds(self):
        """
        변동성에 따라 손익 임계값 조정
        """
        if self.volatility is None:
            return
        
        # 변동성이 높으면 손익 범위를 넓힘
        volatility_factor = max(1.0, self.volatility * 100)  # 변동성을 %로 변환
        
        self.stop_loss_pct = min(self.base_stop_loss * volatility_factor, 0.02)  # 최대 2%
        self.take_profit_pct = min(self.base_take_profit * volatility_factor, 0.03)  # 최대 3%


if __name__ == "__main__":
    print("✅ risk_manager.py 모듈 로드 완료")
    print("🛡️  사용 가능한 클래스:")
    print("  - RiskManager - 기본 리스크 관리")
    print("  - AdaptiveRiskManager - 적응형 리스크 관리")

