"""
model_v3 실전 자동매매
- 모델: extreme_RF_fm10_d4_u5.pkl (RandomForest, 10분 예측)
- 최적 파라미터: buy=0.25, sell=0.35, stop=1.5%, take=1.2%
"""

import os
import sys

# model_v3 Common (지표 계산)
sys.path.insert(0, os.path.dirname(__file__))
from Common.indicators import add_all_indicators
from Common.multi_timeframe_features import add_multi_timeframe_features

from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import time
import datetime
import joblib
import requests


# 로그 파일 경로
LOG_FILE = "model_v3_trading_log.txt"


def write_log(message, print_also=True):
    """로그 파일에 메시지 기록 (이어쓰기)"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    
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


def get_1min_ohlcv_batch(ticker, count=3000):
    """1분봉 데이터 가져오기 (200개씩 배치)"""
    all_data = []
    batch_size = 200
    num_batches = (count + batch_size - 1) // batch_size
    
    print(f"\n[Step 2] Loading initial 1-minute candle data ({count} minutes = {count//60} hours)...")
    print(f"   Loading in batches of {batch_size}...")
    
    for i in range(num_batches):
        try:
            if i == 0:
                df = pyupbit.get_ohlcv(ticker, interval="minute1", count=batch_size)
            else:
                to_timestamp = all_data[0].index[0] - pd.Timedelta(minutes=1)
                df = pyupbit.get_ohlcv(ticker, interval="minute1", count=batch_size, to=to_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            
            if df is None or len(df) == 0:
                break
                
            all_data.insert(0, df)
            oldest_time = df.index[0].strftime('%H:%M')
            newest_time = df.index[-1].strftime('%H:%M')
            print(f"   Batch {i+1}/{num_batches}: {oldest_time} ~ {newest_time} ({len(df)} candles)")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   [WARN] Batch {i+1} error: {e}")
            break
    
    if not all_data:
        return None
    
    result = pd.concat(all_data).drop_duplicates()
    result = result.sort_index()
    
    print(f"\n[OK] Loaded {len(result)} candles from {len(all_data)} batches")
    print(f"   Time range: {result.index[0]} ~ {result.index[-1]}")
    
    return result


def predict_signal(df, model, feature_cols):
    """ML 모델로 매수/매도 신호 예측"""
    if len(df) < 200:
        return None, None, None
    
    df_features = df.copy()
    df_features = add_all_indicators(df_features)
    df_features = add_multi_timeframe_features(df_features)
    df_features = df_features.dropna()
    
    if len(df_features) == 0:
        return None, None, None
    
    latest = df_features.iloc[-1]
    missing = [col for col in feature_cols if col not in latest.index]
    
    if missing:
        print(f"[WARN] Missing features: {missing[:5]}...")
        return None, None, None
    
    X = latest[feature_cols].values.reshape(1, -1)
    
    if hasattr(model, 'best_iteration'):
        probs = model.predict(X, num_iteration=model.best_iteration)[0]
    else:
        probs = model.predict_proba(X)[0]
    
    prob_down = probs[0]
    prob_sideways = probs[1]
    prob_up = probs[2]
    
    return prob_down, prob_sideways, prob_up


def run_live_trading(ticker="KRW-BTC", 
                     buy_threshold=0.25, 
                     sell_threshold=0.35, 
                     stop_loss_pct=1.5, 
                     take_profit_pct=1.2):
    """
    실전 자동매매 메인 로직
    """
    load_dotenv()
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)
    
    print("\n" + "="*84)
    print("Multi-Timeframe ML Auto-Trading v3.3 [model_v3]")
    print("="*84)
    
    # 모델 로드
    print("\n[Step 1] Loading ML model...")
    model_data = joblib.load("model/lgb_model_v3.pkl")
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    print(f"[Model Loaded] Features: {len(feature_cols)}")
    
    # 초기 데이터 로드
    df_1m = get_1min_ohlcv_batch(ticker, count=3000)
    if df_1m is None or len(df_1m) < 1800:
        print("[ERROR] Failed to load initial data")
        return
    
    print(f"\n[Step 3] Calculating indicators and removing NaN...")
    df_features = df_1m.copy()
    df_features = add_all_indicators(df_features)
    df_features = add_multi_timeframe_features(df_features)
    
    print(f"   Before dropna: {len(df_features)} candles")
    df_features = df_features.dropna()
    print(f"   After dropna: {len(df_features)} candles")
    print(f"   Final range: {df_features.index[0]} ~ {df_features.index[-1]}")
    
    # 프로그램 시작 로그
    log_program_start(ticker, buy_threshold, sell_threshold, stop_loss_pct, take_profit_pct)
    
    # 거래 상태
    holding = False
    buy_price = 0
    buy_amount = 0
    last_update_minute = None
    
    print("\n[Trading started. Press Ctrl+C to stop]\n")
    
    # 메인 루프
    while True:
        try:
            current_time = datetime.datetime.now()
            current_minute = current_time.strftime('%Y-%m-%d %H:%M')
            
            # 매 분마다 데이터 업데이트
            if last_update_minute != current_minute:
                last_update_minute = current_minute
                
                new_candle = pyupbit.get_ohlcv(ticker, interval="minute1", count=1)
                if new_candle is not None and len(new_candle) > 0:
                    df_1m = pd.concat([df_1m, new_candle]).drop_duplicates()
                    df_1m = df_1m.tail(3000)
                    df_1m = df_1m.sort_index()
                    
                    df_features = df_1m.copy()
                    df_features = add_all_indicators(df_features)
                    df_features = add_multi_timeframe_features(df_features)
                    df_features = df_features.dropna()
            
            # 현재가 조회
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                time.sleep(5)
                continue
            
            # ML 예측
            prob_down, prob_sideways, prob_up = predict_signal(df_features, model, feature_cols)
            
            if prob_down is None:
                time.sleep(10)
                continue
            
            # 매수 신호
            buy_signal = (not holding) and (prob_up >= buy_threshold)
            
            # 매도 신호
            if holding:
                profit_pct = ((current_price - buy_price) / buy_price) * 100
                sell_by_ml = (prob_down >= sell_threshold)
                sell_by_stop = (profit_pct <= -stop_loss_pct)
                sell_by_take = (profit_pct >= take_profit_pct)
                sell_signal = sell_by_ml or sell_by_stop or sell_by_take
            else:
                sell_signal = False
                profit_pct = 0
            
            # 상태 출력
            print("="*84)
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {'HOLDING' if holding else 'WAITING'}")
            print(f"   Price: {current_price:,.0f}")
            if holding:
                print(f"   Profit: {profit_pct:+.2f}%")
            print(f"   ML: Down={prob_down:.3f}, Sideways={prob_sideways:.3f}, Up={prob_up:.3f}")
            
            if buy_signal:
                print(f"   Buy Signal: YES (Up >= {buy_threshold:.2f})")
            elif not holding:
                print(f"   Buy Signal: NO (need >= {buy_threshold:.2f})")
            
            if holding:
                if sell_by_take:
                    print(f"   Sell Signal: TAKE PROFIT (+{profit_pct:.2f}% >= +{take_profit_pct}%)")
                elif sell_by_stop:
                    print(f"   Sell Signal: STOP LOSS ({profit_pct:.2f}% <= -{stop_loss_pct}%)")
                elif sell_by_ml:
                    print(f"   Sell Signal: ML (Down={prob_down:.3f} >= {sell_threshold:.2f})")
                else:
                    print(f"   Sell Signal: NO")
            
            print("="*84)
            
            # 매수 실행
            if buy_signal:
                krw_balance = upbit.get_balance("KRW")
                if krw_balance > 5500:
                    buy_amount_krw = krw_balance * 0.995
                    order = upbit.buy_market_order(ticker, buy_amount_krw)
                    
                    if order and 'uuid' in order:
                        time.sleep(0.5)
                        buy_price = current_price
                        buy_amount = buy_amount_krw / buy_price
                        holding = True
                        
                        log_msg = (f"[BUY] 매수 | Price: {buy_price:,.0f} | "
                                 f"Amount: {buy_amount:.6f} | Up={prob_up:.3f} | "
                                 f"Target: +{take_profit_pct}% / Stop: -{stop_loss_pct}%")
                        write_log(log_msg)
            
            # 매도 실행
            if sell_signal and holding:
                coin_balance = upbit.get_balance(ticker.split('-')[1])
                if coin_balance > 0:
                    order = upbit.sell_market_order(ticker, coin_balance)
                    
                    if order and 'uuid' in order:
                        time.sleep(0.5)
                        
                        reason = ""
                        if sell_by_take:
                            reason = f"익절 (+{profit_pct:.2f}%)"
                        elif sell_by_stop:
                            reason = f"손절 ({profit_pct:.2f}%)"
                        else:
                            reason = f"ML 매도 (Down={prob_down:.3f})"
                        
                        log_msg = (f"[SELL] 매도 | Price: {current_price:,.0f} | "
                                 f"Profit: {profit_pct:+.2f}% | {reason}")
                        write_log(log_msg)
                        
                        holding = False
                        buy_price = 0
                        buy_amount = 0
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n[Program stopped by user]")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_live_trading(
        ticker="KRW-BTC",
        buy_threshold=0.25,
        sell_threshold=0.35,
        stop_loss_pct=1.5,
        take_profit_pct=1.2
    )

