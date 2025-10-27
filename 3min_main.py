"""
실전 자동매매 v3.3: 3분 예측 모델 (개선)
- 모델: lgb_model_v3.pkl (LightGBM, 3분 예측)
- 전략: 스캘핑
- 최적 파라미터: buy=0.25, sell=0.35, stop=1.5%, take=1.2%
- 백테스트 성능: +0.31% (3일), 71.4% 승률, 7회 거래
"""

import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import time
import datetime
import joblib
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features


# 로그 파일 경로
LOG_FILE = "3min_trading_log.txt"


def write_log(message, print_also=True):
    """로그 파일에 메시지 기록 (이어쓰기)"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    # 파일에 기록 (append mode)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    
    # 콘솔에도 출력
    if print_also:
        print(log_message)


def log_program_start(ticker, buy_threshold, sell_threshold, stop_loss_pct, take_profit_pct):
    """프로그램 시작 로그"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 100 + '\n')
        f.write(f'프로그램 시작: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 100 + '\n')
        f.write(f'티커: {ticker}\n')
        f.write(f'매수 임계값: {buy_threshold} (prob_up >= {buy_threshold*100:.0f}%)\n')
        f.write(f'매도 임계값: {sell_threshold} (prob_down >= {sell_threshold*100:.0f}%)\n')
        f.write(f'손절: {stop_loss_pct}%\n')
        f.write(f'익절: {take_profit_pct}%\n')
        f.write('=' * 100 + '\n\n')
    
    print('\n' + '=' * 100)
    print(f'프로그램 시작: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 100)


def get_minute_ohlcv(ticker, interval=1, count=200):
    """분봉 데이터 조회"""
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
    """멀티 타임프레임 ML 모델 로드 (v3.3 - 3분 예측)"""
    try:
        model_data = joblib.load("model/lgb_model_v3.pkl")
        version_info = model_data.get('version', 'v3.3')
        model_type = model_data.get('type', '3-class-optimized')
        
        print(f"[Model Loaded] Version: {version_info}, Type: {model_type}")
        print(f"   Features: {len(model_data['feature_cols'])}")
        print(f"   Prediction: 3 minutes")
        return model_data
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return None


def predict_signal(df, model_data, buy_threshold=0.25, sell_threshold=0.35):
    """ML 모델 기반 매매 신호 예측"""
    try:
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        
        # 특징 존재 여부 확인
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARN] Missing features: {missing_cols[:5]}...")
            return False, False, None
        
        # 예측
        X = df[feature_cols].iloc[-1:]
        
        # LightGBM vs sklearn 모델 구분
        if hasattr(model, 'best_iteration'):
            predictions = model.predict(X, num_iteration=model.best_iteration)
            prob_down = predictions[0][0]
            prob_sideways = predictions[0][1]
            prob_up = predictions[0][2]
        else:
            predictions = model.predict_proba(X)[0]
            prob_down = predictions[0]
            prob_sideways = predictions[1]
            prob_up = predictions[2]
        
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
                    buy_threshold=0.25, 
                    sell_threshold=0.35,
                    stop_loss_pct=1.5, 
                    take_profit_pct=1.2, 
                    fee_rate=0.0005):
    """3분 예측 모델 기반 실시간 자동매매 v3.3"""
    
    # 프로그램 시작 로그
    log_program_start(ticker, buy_threshold, sell_threshold, stop_loss_pct, take_profit_pct)
    
    # ML 모델 로드
    print("\n[Step 1] Loading ML model...")
    model_data = load_ml_model()
    if model_data is None:
        print("[ERROR] Model load failed. Exiting.")
        return
    
    # 초기 데이터 로드
    print("\n[Step 2] Loading initial 1-minute candle data (3000 minutes)...")
    df = get_minute_ohlcv(ticker, interval=1, count=3000)
    
    if df is None or len(df) == 0:
        print("[ERROR] Failed to load initial data.")
        return
    
    # 초기 지표 및 특징 계산
    print("\n[Step 3] Calculating indicators and removing NaN...")
    try:
        df = add_all_indicators(df)
        df = add_multi_timeframe_features(df)
        df = df.dropna()
        
        if len(df) < 100:
            print(f"[WARN] Not enough data ({len(df)} candles)")
            return
        
        print(f"[OK] Initial data ready: {len(df)} candles")
    except Exception as e:
        print(f"[ERROR] Feature calculation failed: {e}")
        return
    
    # 초기 설정
    initial_balance = 1000000
    balance = initial_balance
    coin_holding = 0
    buy_price = 0
    buy_balance = 0
    trade_count = 0
    win_count = 0
    total_profit = 0
    
    print("\n[Step 4] Starting live trading...\n")
    
    last_update_minute = None
    
    while True:
        try:
            current_time = datetime.datetime.now()
            current_minute = current_time.replace(second=0, microsecond=0)
            
            # 현재 가격 조회
            price = pyupbit.get_current_price(ticker)
            
            if price is None:
                print("[ERROR] Failed to get current price")
                time.sleep(10)
                continue
            
            # 1분마다 데이터 업데이트 (완성된 캔들만)
            if last_update_minute is None or current_minute > last_update_minute:
                # 최신 3개 1분봉 가져오기
                df_new = get_minute_ohlcv(ticker, interval=1, count=3)
                
                if df_new is not None and len(df_new) >= 2:
                    # 완성된 캔들만 사용 (iloc[-2]: 두 번째 최신 = 완성됨)
                    completed_candle = df_new.iloc[-2:-1]
                    
                    # 새로운 완성 캔들만 추가 (중복 방지)
                    if completed_candle.index[0] not in df.index:
                        df = pd.concat([df, completed_candle]).sort_index()
                        
                        # 3000개 유지
                        if len(df) > 3000:
                            df = df.iloc[-3000:]
                        
                        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] New candle added: {completed_candle.index[0]} (window: {len(df)})")
                    
                    last_update_minute = current_minute
                    
                    # 지표 및 멀티 타임프레임 특징 생성
                    try:
                        df_features = add_all_indicators(df.copy())
                        df_features = add_multi_timeframe_features(df_features)
                        df_features = df_features.dropna()
                        
                        if len(df_features) < 100:
                            print(f"[WARN] Not enough data after dropna ({len(df_features)} candles), skipping...")
                            time.sleep(30)
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
                            
                            msg = f"[STOP LOSS] 매도 | Buy: {buy_price:,.0f} -> Sell: {price:,.0f} | 수익률: {profit_rate:.2f}% | 손실: {profit:,.0f} KRW"
                            write_log(msg)
                            
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
                            
                            msg = f"[TAKE PROFIT] 매도 | Buy: {buy_price:,.0f} -> Sell: {price:,.0f} | 수익률: {profit_rate:.2f}% | 수익: {profit:,.0f} KRW"
                            write_log(msg)
                            
                            coin_holding = 0
                            buy_price = 0
                            buy_balance = 0
                        
                        # ML 하락 신호 매도
                        elif sell_signal:
                            balance = coin_holding * price * (1 - fee_rate)
                            profit = balance - buy_balance
                            profit_pct = (profit / buy_balance) * 100
                            total_profit += profit
                            trade_count += 1
                            
                            if profit > 0:
                                win_count += 1
                            
                            msg = f"[ML SELL] 매도 | Buy: {buy_price:,.0f} -> Sell: {price:,.0f} | 수익률: {profit_pct:.2f}% | 수익: {profit:,.0f} KRW | Down={probs['down']:.3f}"
                            write_log(msg)
                            
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
                            
                            msg = f"[BUY] 매수 | Price: {buy_price:,.0f} | Amount: {coin_holding:.6f} | Up={probs['up']:.3f} | Target: +{take_profit_pct}% / Stop: -{stop_loss_pct}%"
                            write_log(msg)
                        
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
            
            # 30초 대기
            time.sleep(30)
        
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
    load_dotenv()
    
    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
    
    # 거래 설정 (v3.3 - 3분 예측 모델)
    ticker = "KRW-BTC"
    buy_threshold = 0.25
    sell_threshold = 0.35
    stop_loss = 1.5
    take_profit = 1.2
    
    print("\n" + "=" * 80)
    print("Multi-Timeframe ML Auto-Trading v3.3 [3-MINUTE PREDICTION]")
    print("3분 예측 스캘핑 전략")
    print("=" * 80)
    print(f"\n[Model Information]")
    print(f"  - File: lgb_model_v3.pkl")
    print(f"  - Algorithm: LightGBM Classifier")
    print(f"  - Prediction: 3분 예측")
    print(f"  - Strategy: 스캘핑")
    print(f"\n[Trading Parameters]")
    print(f"  - Ticker: {ticker}")
    print(f"  - Buy Threshold: {buy_threshold}")
    print(f"  - Sell Threshold: {sell_threshold}")
    print(f"  - Stop Loss: {stop_loss}%")
    print(f"  - Take Profit: {take_profit}%")
    print(f"  - Initial Balance: 1,000,000 KRW")
    print(f"\n[Backtesting Results - 2025-10-22~24, 3일]")
    print(f"  - Return: +0.31% (3 days)")
    print(f"  - Win Rate: 71.4%")
    print(f"  - Trades: 7회 (5승 2패)")
    print(f"\n[WARNING] This is simulation. Use at your own risk!")
    print("=" * 80 + "\n")
    
    run_live_trading(
        ticker=ticker,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )