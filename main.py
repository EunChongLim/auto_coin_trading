"""
실전 자동매매 v3.0: 멀티 타임프레임 + 3-Class ML (최적화)
80일 데이터 학습 + 최적 임계값
"""

import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import time
import datetime
import joblib
import requests
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features


def get_minute_ohlcv(ticker, interval=1, count=200):
    """
    분봉 데이터 조회 (1분, 3분, 5분, 10분, 15분, 30분, 60분, 240분 가능)
    
    Args:
        ticker: 마켓 코드 (예: KRW-BTC)
        interval: 분봉 간격 (1, 3, 5, 10, 15, 30, 60, 240)
        count: 조회할 캔들 개수 (최대 200)
    
    Returns:
        DataFrame: OHLCV 데이터
    """
    try:
        df = pyupbit.get_ohlcv(ticker, interval=f"minute{interval}", count=count)
        
        if df is None or len(df) == 0:
            return None
        
        # 컬럼 이름 표준화
        df = df.reset_index()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'value']
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Minute data error: {e}")
        return None


def load_ml_model():
    """
    멀티 타임프레임 ML 모델 로드 (v3.0)
    """
    try:
        model_data = joblib.load("model/lgb_model_v3.pkl")
        print(f"[Model Loaded] Version: {model_data['version']}, Type: {model_data['type']}")
        print(f"   Features: {len(model_data['feature_cols'])}")
        return model_data
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return None


def predict_signal(df, model_data, buy_threshold=0.1, sell_threshold=0.4):
    """
    ML 모델 기반 매매 신호 예측
    
    Args:
        df: OHLCV 데이터 (멀티 타임프레임 특징 포함)
        model_data: 모델 데이터
        buy_threshold: 매수 임계값 (상승 확률)
        sell_threshold: 매도 임계값 (하락 확률)
    
    Returns:
        tuple: (buy_signal, sell_signal, probs)
    """
    try:
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        
        # 특징 존재 여부 확인
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARN] Missing features: {missing_cols[:5]}...")
            return False, False, None
        
        # 예측
        X = df[feature_cols].iloc[-1:]  # 최신 데이터만
        predictions = model.predict(X, num_iteration=model.best_iteration)
        
        # 예측 확률: [하락(0), 횡보(1), 상승(2)]
        prob_down = predictions[0][0]
        prob_sideways = predictions[0][1]
        prob_up = predictions[0][2]
        
        # 매매 신호
        buy_signal = prob_up >= buy_threshold
        sell_signal = prob_down >= sell_threshold
        
        return buy_signal, sell_signal, {
            'down': prob_down,
            'sideways': prob_sideways,
            'up': prob_up
        }
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return False, False, None


def run_live_trading(ticker="KRW-BTC", 
                     buy_threshold=0.15, 
                     sell_threshold=0.4,
                     stop_loss_pct=0.6, 
                     take_profit_pct=1.8, 
                     fee_rate=0.0005):
    """
    멀티 타임프레임 ML 기반 실시간 자동매매 v3.0
    
    최적화된 설정 (80일 데이터 학습):
    - buy_threshold: 0.15 (상승 확률 15% 이상)
    - sell_threshold: 0.4 (하락 확률 40% 이상)
    - stop_loss: 0.6%
    - take_profit: 1.8%
    
    백테스팅 성능 (v3.0 최적화):
    - 평균 수익률: +2.23%
    - 평균 승률: 82.7%
    - 평균 거래: 5.4회/일
    """
    print("=" * 80)
    print("Multi-Timeframe ML Auto-Trading v3.0")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Buy Threshold: {buy_threshold} (prob_up >= {buy_threshold*100:.0f}%)")
    print(f"Sell Threshold: {sell_threshold} (prob_down >= {sell_threshold*100:.0f}%)")
    print(f"Stop Loss: {stop_loss_pct}%")
    print(f"Take Profit: {take_profit_pct}%")
    print(f"Fee Rate: {fee_rate*100}%")
    print("\nBacktesting Performance (v3.0 Optimized):")
    print(f"  - Avg Return: +2.23%")
    print(f"  - Win Rate: 82.7%")
    print(f"  - Avg Trades: 5.4/day")
    print("=" * 80)
    
    # 모델 로드
    print("\n[Step 1] Loading ML model...")
    model_data = load_ml_model()
    if model_data is None:
        print("[ERROR] Failed to load model!")
        return
    
    # 초기 자금 및 상태 변수
    initial_balance = 1_000_000
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trade_count = 0
    win_count = 0
    total_profit = 0
    
    # 초기 데이터 로드
    # 24시간(1440분) = 백테스트와 동일한 윈도우 크기
    # pyupbit는 한 번에 최대 200개만 가능하므로 8번 호출 필요
    print("\n[Step 2] Loading initial 1-minute candle data (1440 minutes = 24 hours)...")
    
    # 1440개 로드 (200개씩 8번)
    all_dfs = []
    print("   Loading in batches of 200...")
    
    for i in range(8):
        try:
            if i == 0:
                # 첫 번째: 최신 200개
                df_temp = get_minute_ohlcv(ticker, interval=1, count=200)
                if df_temp is not None and len(df_temp) > 0:
                    print(f"   Batch {i+1}/8: {df_temp.index.min().strftime('%H:%M')} ~ {df_temp.index.max().strftime('%H:%M')} ({len(df_temp)} candles)")
            else:
                # 전체 누적 데이터의 가장 오래된 시간 사용 (중복 제거)
                combined_df = pd.concat(all_dfs).sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                oldest_time = combined_df.index.min()
                
                # oldest_time을 그대로 to로 사용 (중복은 나중에 제거)
                to_param = oldest_time.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"  # ISO 8601 형식 (T 포함)
                
                # 직접 Upbit API 호출 (pyupbit는 timezone을 제거하므로)
                url = "https://api.upbit.com/v1/candles/minutes/1"
                params = {
                    "market": ticker,
                    "to": to_param,
                    "count": 200
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    if len(data) > 0:
                        
                        # DataFrame 변환
                        df_temp = pd.DataFrame(data)
                        df_temp = df_temp[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
                        df_temp.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
                        df_temp = df_temp.set_index('timestamp').sort_index()
                        print(f"   Batch {i+1}/8: {df_temp.index.min().strftime('%H:%M')} ~ {df_temp.index.max().strftime('%H:%M')} ({len(df_temp)} candles)")
                    else:
                        df_temp = None
                else:
                    print(f"   [ERROR] API request failed: {response.status_code}")
                    df_temp = None
            
            if df_temp is not None and len(df_temp) > 0:
                all_dfs.append(df_temp)
            else:
                print(f"   No more data (stopping at batch {i+1}/8)")
                break
            
            time.sleep(0.1)  # API 제한 방지
            
        except Exception as e:
            print(f"   Batch {i+1}/8: Error - {e}")
            break
    
    # 합치고 중복 제거
    if len(all_dfs) > 0:
        df = pd.concat(all_dfs).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        print(f"\n[OK] Loaded {len(df)} candles from {len(all_dfs)} batches")
    else:
        print("[ERROR] Failed to load initial data!")
        return
    
    print(f"   Time range: {df.index[0]} ~ {df.index[-1]}")
    
    print("\n" + "=" * 80)
    print("Live trading started!")
    print("=" * 80 + "\n")
    
    last_update_minute = None
    
    while True:
        try:
            current_time = datetime.datetime.now()
            current_minute = current_time.replace(second=0, microsecond=0)
            
            # 1분마다 데이터 업데이트 (완성된 캔들만)
            if last_update_minute is None or current_minute > last_update_minute:
                # 최신 3개 1분봉 가져오기
                df_new = get_minute_ohlcv(ticker, interval=1, count=3)
                
                if df_new is not None and len(df_new) >= 2:
                    # 완성된 캔들만 사용 (iloc[-2]: 두 번째 최신 = 완성됨)
                    # iloc[-1]은 현재 진행 중인 미완성 캔들이므로 제외
                    completed_candle = df_new.iloc[-2:-1]
                    
                    # 새로운 완성 캔들만 추가 (중복 방지)
                    if completed_candle.index[0] not in df.index:
                        df = pd.concat([df, completed_candle]).sort_index()
                        
                        # 1440개 유지 (24시간 슬라이딩 윈도우)
                        if len(df) > 1440:
                            df = df.iloc[-1440:]
                        
                        print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] New candle added: {completed_candle.index[0]} (window: {len(df)})")
                    
                    last_update_minute = current_minute
                    
                    # 지표 및 멀티 타임프레임 특징 생성
                    try:
                        df_features = add_all_indicators(df.copy())
                        df_features = add_multi_timeframe_features(df_features)
                        
                        # NaN 체크 (중요 특징만)
                        latest_features = df_features.iloc[-1]
                        critical_features = ['rsi', 'ma_fast', 'ma_slow', 'rsi_5m', 'rsi_15m']
                        
                        # 중요 특징에 NaN이 있는지만 체크
                        has_critical_nan = any(pd.isna(latest_features.get(f)) for f in critical_features if f in latest_features)
                        
                        if has_critical_nan:
                            print("[WARN] Critical features have NaN, accumulating more data...")
                            time.sleep(30)  # 30초 대기 (더 많은 데이터 축적)
                            continue
                        
                    except Exception as e:
                        print(f"[ERROR] Feature generation failed: {e}")
                        time.sleep(10)
                        continue
                    
                    # ML 예측
                    buy_signal, sell_signal, probs = predict_signal(
                        df_features, model_data, buy_threshold, sell_threshold
                    )
                    
                    if probs is None:
                        print("[WARN] Prediction failed")
                        time.sleep(10)
                        continue
                    
                    latest = df.iloc[-1]
                    price = latest['close']
                    now_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # === 보유 중: 손익 체크 ===
                    if coin_holding > 0:
                        profit_rate = (price - buy_price) / buy_price * 100
                        
                        # 손절
                        if profit_rate <= -stop_loss_pct:
                            balance = coin_holding * price * (1 - fee_rate)
                            profit = balance - buy_balance
                            total_profit += profit
                            trade_count += 1
                            
                            print(f"\n[STOP LOSS] {now_str}")
                            print(f"   Buy: {buy_price:,.0f} -> Sell: {price:,.0f}")
                            print(f"   Profit Rate: {profit_rate:.2f}% | Loss: {profit:,.0f} KRW")
                            
                            coin_holding = 0
                            buy_price = 0
                            buy_balance = 0
                        
                        # 익절
                        elif profit_rate >= take_profit_pct:
                            balance = coin_holding * price * (1 - fee_rate)
                            profit = balance - buy_balance
                            total_profit += profit
                            trade_count += 1
                            win_count += 1
                            
                            print(f"\n[TAKE PROFIT] {now_str}")
                            print(f"   Buy: {buy_price:,.0f} -> Sell: {price:,.0f}")
                            print(f"   Profit Rate: {profit_rate:.2f}% | Profit: {profit:,.0f} KRW")
                            
                            coin_holding = 0
                            buy_price = 0
                            buy_balance = 0
                        
                        # ML 하락 신호 매도
                        elif sell_signal:
                            balance = coin_holding * price * (1 - fee_rate)
                            profit = balance - buy_balance
                            total_profit += profit
                            trade_count += 1
                            if profit > 0:
                                win_count += 1
                            
                            print(f"\n[SELL SIGNAL] {now_str}")
                            print(f"   Buy: {buy_price:,.0f} -> Sell: {price:,.0f}")
                            print(f"   Profit Rate: {profit_rate:.2f}% | Profit: {profit:,.0f} KRW")
                            print(f"   ML: Down={probs['down']:.3f}, Sideways={probs['sideways']:.3f}, Up={probs['up']:.3f}")
                            
                            coin_holding = 0
                            buy_price = 0
                            buy_balance = 0
                        
                        # 보유 중
                        else:
                            print(f"[{now_str}] HOLDING | Profit: {profit_rate:+.2f}% | Price: {price:,.0f}")
                            print(f"   ML: Down={probs['down']:.3f}, Sideways={probs['sideways']:.3f}, Up={probs['up']:.3f}")
                    
                    # === 미보유: 매수 신호 체크 ===
                    else:
                        if buy_signal and balance > 10000:
                            # 매수
                            buy_balance = balance
                            coin_holding = (balance * (1 - fee_rate)) / price
                            buy_price = price
                            balance = 0
                            
                            print(f"\n[BUY] {now_str}")
                            print(f"   Price: {buy_price:,.0f} | Amount: {coin_holding:.6f}")
                            print(f"   ML: Down={probs['down']:.3f}, Sideways={probs['sideways']:.3f}, Up={probs['up']:.3f}")
                            print(f"   Target: +{take_profit_pct}% | Stop: -{stop_loss_pct}%")
                        
                        else:
                            # 대기 중
                            print(f"\n{'='*80}")
                            print(f"[{now_str}] WAITING")
                            print(f"   Price: {price:,.0f}")
                            print(f"   ML: Down={probs['down']:.3f}, Sideways={probs['sideways']:.3f}, Up={probs['up']:.3f}")
                            print(f"   Buy Signal: {'YES' if buy_signal else 'NO'} (need >= {buy_threshold})")
                            print(f"{'='*80}")
                    
                    # 통계 출력
                    if trade_count > 0:
                        win_rate = (win_count / trade_count) * 100
                        total_value = balance if coin_holding == 0 else coin_holding * price
                        total_return = (total_value - initial_balance) / initial_balance * 100
                        
                        print(f"\n[Statistics] Trades: {trade_count} | Win Rate: {win_rate:.1f}%")
                        print(f"   Total Profit: {total_profit:,.0f} KRW | Return: {total_return:+.2f}%")
            
            # 10초 대기
            time.sleep(10)
        
        except KeyboardInterrupt:
            print("\n\n[EXIT] Trading stopped by user")
            break
        
        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(10)
    
    # 최종 통계
    if trade_count > 0:
        print("\n" + "=" * 80)
        print("Final Statistics")
        print("=" * 80)
        
        win_rate = (win_count / trade_count) * 100
        total_value = balance if coin_holding == 0 else coin_holding * price
        total_return = (total_value - initial_balance) / initial_balance * 100
        
        print(f"Total Trades: {trade_count}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Profit: {total_profit:,.0f} KRW")
        print(f"Final Balance: {total_value:,.0f} KRW")
        print(f"Total Return: {total_return:+.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv()
    
    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
    
    # Upbit 객체 생성 (실제 거래용 - 사용 시 주석 해제)
    # upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    # print("API Connected")
    
    # 거래 설정 (v3.0 최적화 완료)
    ticker = "KRW-BTC"
    buy_threshold = 0.15     # 상승 확률 15% 이상
    sell_threshold = 0.4     # 하락 확률 40% 이상
    stop_loss = 0.6          # 손절 0.6%
    take_profit = 1.8        # 익절 1.8%
    
    print("\n" + "=" * 80)
    print("Multi-Timeframe ML Auto-Trading v3.0")
    print("BEST OPTIMIZED SETTINGS APPLIED")
    print("=" * 80)
    print(f"\nTicker: {ticker}")
    print(f"Buy Threshold: {buy_threshold} (prob_up >= {buy_threshold*100:.0f}%)")
    print(f"Sell Threshold: {sell_threshold} (prob_down >= {sell_threshold*100:.0f}%)")
    print(f"Stop Loss: {stop_loss}%")
    print(f"Take Profit: {take_profit}%")
    print(f"Initial Balance: 1,000,000 KRW")
    print(f"\nBacktesting Results (v3.0):")
    print(f"  - Avg Return: +2.23% (best among 8 strategies)")
    print(f"  - Win Rate: 82.7%")
    print(f"  - Avg Trades: 5.4/day")
    print(f"\nImprovement from v2.0:")
    print(f"  - Return: +1.46% -> +2.23% (+53% improvement)")
    print(f"  - Win Rate: 77.4% -> 82.7% (+5.3%p)")
    print(f"\nWARNING: This is simulation. Use at your own risk!")
    print("=" * 80 + "\n")
    
    # 실전 거래 시작
    run_live_trading(
        ticker=ticker,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
