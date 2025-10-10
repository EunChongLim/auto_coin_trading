import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import time
import datetime

def compute_rsi(series, period=14):
    """RSI(상대강도지수) 계산"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    avg_gain = up.rolling(window=period, min_periods=1).mean()
    avg_loss = down.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_simulation(ticker="KRW-BTC", stop_loss_pct=2.0, take_profit_pct=3.0, fee_rate=0.0005):
    """
    스캘핑 자동매매 시뮬레이션 (비트코인 최적화)
    
    Args:
        ticker: 거래할 코인 티커
        stop_loss_pct: 손절 퍼센트 (비트코인: 2.0% 권장)
        take_profit_pct: 익절 퍼센트 (비트코인: 3.0% 권장)
        fee_rate: 거래 수수료율 (기본 0.05%)
    
    비트코인 특성 고려사항:
    - 일일 변동성: 3-5% (손절/익절 여유 필요)
    - 강한 트렌드: 과매수/과매도 지속 가능
    - 높은 유동성: 빠른 체결 가능
    """
    print("=" * 60)
    print("⚡ 스캘핑 자동매매 시작 ⚡")
    print(f"📊 손절: -{stop_loss_pct}% | 익절: +{take_profit_pct}% | 수수료: {fee_rate*100}%")
    print("=" * 60)

    # 초기 자금 및 상태 변수
    initial_balance = 1_000_000
    balance = initial_balance
    coin_holding = 0
    buy_price = 0  # 매수 가격 추적
    trade_count = 0  # 거래 횟수
    win_count = 0  # 성공 거래 횟수
    total_profit = 0  # 총 수익

    while True:
        try:
            # 최신 200개 1분봉 데이터 조회
            df = pyupbit.get_ohlcv(ticker, interval="minute1", count=200)
            if df is None or len(df) < 50:
                print("⚠️ 데이터 조회 실패, 5초 후 재시도...")
                time.sleep(5)
                continue

            # 기술적 지표 계산
            df['rsi'] = compute_rsi(df['close'], 14)
            df['ma_fast'] = df['close'].rolling(window=5).mean()  # 초단기 이동평균
            df['ma_slow'] = df['close'].rolling(window=20).mean()  # 단기 이동평균
            df['volume_ma'] = df['volume'].rolling(window=20).mean()  # 거래량 이동평균
            
            # 볼린저 밴드 계산 (변동성 체크)
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            price = latest['close']
            volume = latest['volume']
            
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # === 보유 중일 때: 손절/익절 체크 (최우선) ===
            if coin_holding > 0:
                profit_rate = ((price - buy_price) / buy_price) * 100
                current_value = coin_holding * price * (1 - fee_rate)  # 수수료 반영
                
                # 손절 조건
                if profit_rate <= -stop_loss_pct:
                    balance = current_value
                    trade_profit = balance - (initial_balance if trade_count == 0 else balance)
                    total_profit += trade_profit
                    trade_count += 1
                    
                    print(f"\n🔴 [{now}] 손절 실행!")
                    print(f"   매수가: {buy_price:,.0f}원 → 현재가: {price:,.0f}원")
                    print(f"   수익률: {profit_rate:.2f}% | 손실액: {trade_profit:,.0f}원")
                    
                    coin_holding = 0
                    buy_price = 0
                
                # 익절 조건
                elif profit_rate >= take_profit_pct:
                    balance = current_value
                    trade_profit = current_value - (buy_price * coin_holding)
                    total_profit += trade_profit
                    trade_count += 1
                    win_count += 1
                    
                    print(f"\n🟢 [{now}] 익절 실행!")
                    print(f"   매수가: {buy_price:,.0f}원 → 현재가: {price:,.0f}원")
                    print(f"   수익률: {profit_rate:.2f}% | 수익액: {trade_profit:,.0f}원")
                    
                    coin_holding = 0
                    buy_price = 0
                
                # RSI 과매수 신호 매도 (비트코인: 기준 상향 75→80)
                elif latest['rsi'] > 80:
                    balance = current_value
                    trade_profit = current_value - (buy_price * coin_holding)
                    total_profit += trade_profit
                    trade_count += 1
                    if trade_profit > 0:
                        win_count += 1
                    
                    print(f"\n🟡 [{now}] RSI 과매수 매도! (비트코인 강세 지속)")
                    print(f"   매수가: {buy_price:,.0f}원 → 현재가: {price:,.0f}원")
                    print(f"   수익률: {profit_rate:.2f}% | 손익: {trade_profit:,.0f}원 | RSI: {latest['rsi']:.1f}")
                    
                    coin_holding = 0
                    buy_price = 0
                
                # 보유 중 상태 출력 (10초마다)
                else:
                    print(f"[{now}] 💎 보유중 | 수익률: {profit_rate:+.2f}% | 현재가: {price:,.0f}원 | RSI: {latest['rsi']:.1f}")

            # === 미보유 중일 때: 매수 시그널 체크 ===
            else:
                # 스캘핑 매수 조건 (비트코인 최적화)
                rsi_oversold = 35 < latest['rsi'] < 55  # RSI 과매도 구간 탈출 (비트코인: 범위 확대)
                rsi_rising = latest['rsi'] > prev['rsi']  # RSI 상승 중
                volume_surge = volume > latest['volume_ma'] * 1.2  # 거래량 급증 (비트코인: 기준 완화 1.3→1.2)
                price_above_ma = price > latest['ma_fast']  # 가격이 초단기 이평선 위
                bullish_candle = latest['close'] > latest['open']  # 양봉
                near_bb_lower = price < latest['bb_middle']  # 볼린저밴드 중심선 아래 (저가 구간)
                
                buy_signal = (
                    rsi_oversold and 
                    rsi_rising and 
                    volume_surge and 
                    price_above_ma and 
                    bullish_candle
                )
                
                if buy_signal and balance > 10000:
                    # 수수료 반영하여 매수
                    coin_holding = (balance * (1 - fee_rate)) / price
                    buy_price = price
                    balance = 0
                    trade_count += 1
                    
                    print(f"\n💹 [{now}] 매수 체결!")
                    print(f"   매수가: {buy_price:,.0f}원 | 수량: {coin_holding:.6f}")
                    print(f"   RSI: {latest['rsi']:.1f} | 거래량비: {(volume/latest['volume_ma']):.2f}x")
                    print(f"   목표: +{take_profit_pct}% | 손절: -{stop_loss_pct}%")
                else:
                    # 대기 중 상태 (매 사이클마다 출력 - 10초)
                    volume_ratio = volume / latest['volume_ma'] if latest['volume_ma'] > 0 else 0
                    rsi_status = "🔴과매수" if latest['rsi'] > 75 else "🟢과매도" if latest['rsi'] < 35 else "⚪중립"
                    
                    # 매수 조건 체크 상태 표시
                    conditions_met = sum([rsi_oversold, rsi_rising, volume_surge, price_above_ma, bullish_candle])
                    
                    print(f"\n{'='*60}")
                    print(f"[{now}] ⏳ 대기중 - 매수 시그널 감지 중... (BTC 최적화)")
                    print(f"   현재가: {price:,.0f}원 | RSI: {latest['rsi']:.1f} {rsi_status}")
                    print(f"   거래량비: {volume_ratio:.2f}x | 매수조건 충족: {conditions_met}/5개")
                    print(f"   [{'✓' if rsi_oversold else '✗'}] RSI 35-55 구간 | [{'✓' if rsi_rising else '✗'}] RSI 상승중")
                    print(f"   [{'✓' if volume_surge else '✗'}] 거래량 1.2배+  | [{'✓' if price_above_ma else '✗'}] 가격>5일선")
                    print(f"   [{'✓' if bullish_candle else '✗'}] 양봉 발생")
                    print(f"{'='*60}")

            # 통계 출력 (5분마다)
            if trade_count > 0 and int(time.time()) % 300 == 0:
                win_rate = (win_count / trade_count) * 100
                total_value = balance if coin_holding == 0 else coin_holding * price
                total_return = ((total_value - initial_balance) / initial_balance) * 100
                
                print("\n" + "=" * 60)
                print(f"📈 거래 통계 | 총 거래: {trade_count}회 | 승률: {win_rate:.1f}%")
                print(f"💰 총 수익: {total_profit:,.0f}원 | 수익률: {total_return:+.2f}%")
                print("=" * 60 + "\n")

            time.sleep(10)  # 10초마다 체크 (스캘핑은 빠른 체크 필요)

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            time.sleep(5)

if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv()

    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

    # Upbit 객체 생성 (실제 거래용 - 사용 시 주석 해제)
    # upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    # print("API 연결 성공 ✅")

    # 거래 설정 (비트코인 최적화)
    ticker = "KRW-BTC"  # 비트코인
    stop_loss = 2.0     # 손절 2.0% (비트코인 변동성 고려)
    take_profit = 3.0   # 익절 3.0% (비트코인 수익 목표)
    
    print("\n🎯 스캘핑 전략 설정 (비트코인 최적화)")
    print(f"   티커: {ticker}")
    print(f"   손절: -{stop_loss}% (변동성 고려)")
    print(f"   익절: +{take_profit}% (트렌드 활용)")
    print(f"   RSI: 35-55 매수, >80 매도")
    print(f"   거래량: 평균 1.2배 이상")
    print(f"   초기 자금: 1,000,000원")
    print("\n⚠️  주의: 이것은 모의 거래입니다. 실제 거래는 신중하게 결정하세요.\n")
    
    # 모의 거래 시작
    run_simulation(ticker, stop_loss_pct=stop_loss, take_profit_pct=take_profit)
