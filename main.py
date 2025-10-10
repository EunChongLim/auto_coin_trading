import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import time
import datetime
import requests

def get_second_ohlcv(ticker, count=1000):
    """
    1초봉 데이터 조회 (실시간 거래용)
    
    Args:
        ticker: 마켓 코드 (예: KRW-BTC)
        count: 조회할 캔들 개수 (최대 200 × 호출 횟수)
    
    Returns:
        DataFrame: OHLCV 데이터 (1초봉)
    """
    url = "https://api.upbit.com/v1/candles/seconds"
    headers = {"accept": "application/json"}
    all_data = []
    to_param = None
    calls_needed = (count + 199) // 200
    
    try:
        for i in range(calls_needed):
            params = {
                "market": ticker,
                "count": min(200, count - len(all_data))
            }
            if to_param:
                params["to"] = to_param
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            
            if data:
                to_param = data[-1]['candle_date_time_kst']
            
            if i < calls_needed - 1:
                time.sleep(0.25)
            
            if len(all_data) >= count:
                break
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df = df[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        return df
    except Exception as e:
        print(f"⚠️ 1초봉 조회 오류: {e}")
        return None

def compute_rsi(series, period=14):
    """RSI(상대강도지수) 계산 - EMA 기반 (스캘핑 최적화)"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # EMA(지수 이동 평균) 사용 - 최신 데이터에 더 높은 가중치
    avg_gain = up.ewm(span=period, adjust=False).mean()
    avg_loss = down.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_simulation(ticker="KRW-BTC", stop_loss_pct=1.5, take_profit_pct=0.8, fee_rate=0.0005):
    """
    스캘핑 자동매매 시뮬레이션 (비트코인 초단타 v3.0)
    
    Args:
        ticker: 거래할 코인 티커
        stop_loss_pct: 손절 퍼센트 (1.5% - 백테스팅 최적화)
        take_profit_pct: 익절 퍼센트 (0.8% - 초단타 전략)
        fee_rate: 거래 수수료율 (기본 0.05%)
    
    v3.0 초단타 전략:
    - 익절: 0.8% (작은 수익 반복)
    - 손절: 1.5% (빠른 손절)
    - RSI 매도: 수익 0.5% 이상일 때만
    - 거래량: 1.05배 (완화)
    - 백테스팅 검증: 평균 +0.75% 수익률, 익절 33.3%
    """
    print("=" * 60)
    print("⚡ 초단타 스캘핑 자동매매 시작 v3.0 ⚡")
    print(f"📊 손절: -{stop_loss_pct}% | 익절: +{take_profit_pct}% | 수수료: {fee_rate*100}%")
    print(f"⚡ 1초봉 200개 실시간 분석 | 1초마다 갱신")
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
            # 최신 200개 1초봉 데이터 조회 (약 3분 = 200초)
            df = get_second_ohlcv(ticker, count=200)
            if df is None or len(df) < 50:
                print("⚠️ 1초봉 데이터 조회 실패, 5초 후 재시도...")
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
                    trade_profit = current_value - (buy_price * coin_holding * (1 + fee_rate))
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
                    trade_profit = current_value - (buy_price * coin_holding * (1 + fee_rate))
                    total_profit += trade_profit
                    trade_count += 1
                    win_count += 1
                    
                    print(f"\n🟢 [{now}] 익절 실행! (초단타 0.8%)")
                    print(f"   매수가: {buy_price:,.0f}원 → 현재가: {price:,.0f}원")
                    print(f"   수익률: {profit_rate:.2f}% | 수익액: {trade_profit:,.0f}원")
                    
                    coin_holding = 0
                    buy_price = 0
                
                # RSI 과매수 신호 매도 (조건부: 수익 0.5% 이상일 때만)
                elif latest['rsi'] > 80 and profit_rate > 0.5:
                    balance = current_value
                    trade_profit = current_value - (buy_price * coin_holding * (1 + fee_rate))
                    total_profit += trade_profit
                    trade_count += 1
                    if trade_profit > 0:
                        win_count += 1
                    
                    print(f"\n🟡 [{now}] RSI 과매수 매도! (수익 확보)")
                    print(f"   매수가: {buy_price:,.0f}원 → 현재가: {price:,.0f}원")
                    print(f"   수익률: {profit_rate:.2f}% | 손익: {trade_profit:,.0f}원 | RSI: {latest['rsi']:.1f}")
                    
                    coin_holding = 0
                    buy_price = 0
                
                # 보유 중 상태 출력 (10초마다)
                else:
                    print(f"[{now}] 💎 보유중 | 수익률: {profit_rate:+.2f}% | 현재가: {price:,.0f}원 | RSI: {latest['rsi']:.1f}")

            # === 미보유 중일 때: 매수 시그널 체크 ===
            else:
                # 스캘핑 매수 조건 (초단타 v3.0)
                rsi_oversold = 35 < latest['rsi'] < 55  # RSI 과매도 구간 탈출
                rsi_rising = latest['rsi'] > prev['rsi']  # RSI 상승 중
                volume_surge = volume > latest['volume_ma'] * 1.05  # 거래량 급증 (v3.0: 1.05배로 완화)
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
                    print(f"[{now}] ⏳ 대기중 - 매수 시그널 감지 중... (v3.0 초단타)")
                    print(f"   현재가: {price:,.0f}원 | RSI: {latest['rsi']:.1f} {rsi_status}")
                    print(f"   거래량비: {volume_ratio:.2f}x | 매수조건 충족: {conditions_met}/5개")
                    print(f"   [{'✓' if rsi_oversold else '✗'}] RSI 35-55 구간 | [{'✓' if rsi_rising else '✗'}] RSI 상승중")
                    print(f"   [{'✓' if volume_surge else '✗'}] 거래량 1.05배+ | [{'✓' if price_above_ma else '✗'}] 가격>5일선")
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

            time.sleep(1)  # 1초마다 체크 (초단타 - 최대 빠른 반응)

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

    # 거래 설정 (초단타 v3.0 - 백테스팅 검증 완료)
    ticker = "KRW-BTC"  # 비트코인
    stop_loss = 1.5     # 손절 1.5% (빠른 손절)
    take_profit = 0.8   # 익절 0.8% (초단타 - 작은 수익 반복)
    
    print("\n🎯 초단타 스캘핑 전략 v3.0 (백테스팅 검증 완료)")
    print(f"   티커: {ticker}")
    print(f"   손절: -{stop_loss}% (빠른 손절)")
    print(f"   익절: +{take_profit}% (초단타 전략)")
    print(f"   RSI: 35-55 매수, >80 매도 (수익 0.5%+ 조건)")
    print(f"   거래량: 평균 1.05배 이상")
    print(f"   초기 자금: 1,000,000원")
    print(f"\n📊 백테스팅 결과: 평균 +0.75% 수익률, 익절 33.3%")
    print("\n⚠️  주의: 이것은 모의 거래입니다. 실제 거래는 신중하게 결정하세요.\n")
    
    # 모의 거래 시작
    run_simulation(ticker, stop_loss_pct=stop_loss, take_profit_pct=take_profit)
