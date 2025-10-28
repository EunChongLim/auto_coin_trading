"""
model_v4 실전 자동매매 (A-E 규칙 + B규칙 전략)
- 룰 기반 매수/매도
- ATR 동적 스탑로스
- 부분청산 (+1R 50%)
"""

import os
import sys

# model_v4 Common (지표 계산 + 전략)
sys.path.insert(0, os.path.dirname(__file__))
from Common.indicators import add_all_indicators
from Common.multi_timeframe_features import add_multi_timeframe_features
from Common.strategy_rules import RuleBasedStrategy

from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import time
import datetime
import joblib
import requests


# 로그 파일 경로
LOG_FILE = "model_v4_trading_log.txt"


def write_log(message, print_also=True):
    """로그 파일에 메시지 기록"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    
    if print_also:
        print(log_message)


def log_program_start(params):
    """프로그램 시작 로그"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 100 + '\n')
        f.write(f'프로그램 시작: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 100 + '\n')
        f.write(f'티커: {params["ticker"]}\n')
        f.write(f'전략: B규칙 (룰 기반 + ML 보조 + ATR 동적 스탑)\n')
        f.write(f'ML 매수 임계값: {params["ml_buy_threshold"]:.2f}\n')
        f.write(f'ML 매도 임계값: {params["ml_sell_threshold"]:.2f}\n')
        f.write(f'ATR 스탑 배수: {params["atr_stop_multiplier"]}x\n')
        f.write(f'위험률: {params["risk_pct"]}%\n')
        f.write(f'부분청산: {"ON" if params["use_partial_exit"] else "OFF"}\n')
        f.write('=' * 100 + '\n\n')


def get_1min_ohlcv_batch(ticker, count=3000):
    """1분봉 데이터 가져오기 (200개씩 배치)"""
    all_data = []
    batch_size = 200
    num_batches = (count + batch_size - 1) // batch_size
    
    print(f"\n[Loading {count} minutes data in batches of {batch_size}...]")
    
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
            print(f"   Batch {i+1}/{num_batches}: {len(df)} candles")
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   [WARN] Batch {i+1} error: {e}")
            break
    
    if not all_data:
        return None
    
    result = pd.concat(all_data).drop_duplicates()
    result = result.sort_index()
    
    print(f"[OK] Loaded {len(result)} candles")
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


def run_live_trading_v4(ticker="KRW-BTC", 
                        ml_buy_threshold=0.25, 
                        ml_sell_threshold=0.35,
                        atr_stop_multiplier=1.2,
                        risk_pct=1.0,
                        use_ml=True,
                        use_partial_exit=True):
    """
    model_v4 실전 자동매매 (B규칙 전략)
    """
    load_dotenv()
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)
    
    print("\n" + "="*84)
    print("Multi-Timeframe ML Auto-Trading v4.0 [B규칙 전략]")
    print("룰 기반 + ML 보조 + ATR 동적 스탑 + 부분청산")
    print("="*84)
    
    # 모델 로드
    print("\n[Step 1] Loading ML model...")
    try:
        model_data = joblib.load("model/lgb_model_v4_enhanced.pkl")
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        print(f"[Model Loaded] Version: {model_data.get('version', 'unknown')}, Features: {len(feature_cols)}")
    except:
        print("[WARN] Model not found. Running without ML.")
        use_ml = False
        model = None
        feature_cols = []
    
    # 전략 엔진
    strategy = RuleBasedStrategy(
        ml_buy_threshold=ml_buy_threshold,
        ml_sell_threshold=ml_sell_threshold,
        atr_stop_multiplier=atr_stop_multiplier,
        risk_pct=risk_pct
    )
    
    # 초기 데이터 로드
    df_1m = get_1min_ohlcv_batch(ticker, count=3000)
    if df_1m is None or len(df_1m) < 1800:
        print("[ERROR] Failed to load initial data")
        return
    
    print(f"\n[Step 2] Calculating indicators...")
    df_features = df_1m.copy()
    df_features = add_all_indicators(df_features)
    df_features = add_multi_timeframe_features(df_features)
    df_features = df_features.dropna()
    
    print(f"[OK] {len(df_features)} candles ready")
    
    # 프로그램 시작 로그
    params = {
        'ticker': ticker,
        'ml_buy_threshold': ml_buy_threshold,
        'ml_sell_threshold': ml_sell_threshold,
        'atr_stop_multiplier': atr_stop_multiplier,
        'risk_pct': risk_pct,
        'use_partial_exit': use_partial_exit
    }
    log_program_start(params)
    
    # 거래 상태
    holding = False
    buy_price = 0
    buy_amount = 0
    entry_atr = 0
    partial_exited = False
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
            
            # 최신 특징
            if len(df_features) == 0:
                time.sleep(30)
                continue
            
            latest = df_features.iloc[-1]
            current_atr = latest.get('atr', current_price * 0.02)
            
            # ML 예측 (선택적)
            prob_down, prob_sideways, prob_up = None, None, None
            if use_ml and model is not None:
                prob_down, prob_sideways, prob_up = predict_signal(df_features, model, feature_cols)
            
            # === 매수 신호 ===
            if not holding:
                buy_signal = strategy.check_long_signal(latest, prob_up if use_ml else None)
                
                if buy_signal:
                    krw_balance = upbit.get_balance("KRW")
                    if krw_balance > 5500:
                        # ATR 기반 포지션 사이징
                        buy_amount = strategy.calculate_position_size(krw_balance, current_price, current_atr)
                        buy_amount_krw = buy_amount * current_price
                        
                        order = upbit.buy_market_order(ticker, buy_amount_krw)
                        
                        if order and 'uuid' in order:
                            time.sleep(0.5)
                            buy_price = current_price
                            entry_atr = current_atr
                            holding = True
                            partial_exited = False
                            
                            log_msg = (f"[BUY] 룰 매수 | Price: {buy_price:,.0f} | "
                                     f"Amount: {buy_amount:.6f} | ATR: {entry_atr:,.0f} | "
                                     f"Up={prob_up:.3f if prob_up else 'N/A'}")
                            write_log(log_msg)
                            
                            print(f"\n{'='*84}")
                            print(f"[BUY EXECUTED] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Price: {buy_price:,.0f}")
                            print(f"  Amount: {buy_amount:.6f}")
                            print(f"  Stop: {strategy.calculate_stop_loss(buy_price, entry_atr):,.0f} (-{atr_stop_multiplier}x ATR)")
                            print(f"{'='*84}\n")
            
            # === 매도 신호 ===
            if holding:
                profit_pct = ((current_price - buy_price) / buy_price) * 100
                stop_price = strategy.calculate_stop_loss(buy_price, entry_atr, direction='long')
                atr_stop_hit = current_price <= stop_price
                
                # 부분청산 체크
                if use_partial_exit and not partial_exited:
                    partial_info = strategy.check_partial_exit(buy_price, current_price, entry_atr, direction='long')
                    if partial_info['should_exit']:
                        coin_balance = upbit.get_balance(ticker.split('-')[1])
                        exit_amount = coin_balance * partial_info['exit_ratio']
                        
                        order = upbit.sell_market_order(ticker, exit_amount)
                        if order and 'uuid' in order:
                            time.sleep(0.5)
                            partial_exited = True
                            
                            log_msg = f"[PARTIAL SELL] 50% 청산 | {partial_info['reason']} | Profit: {profit_pct:+.2f}%"
                            write_log(log_msg)
                            
                            print(f"\n{'='*84}")
                            print(f"[PARTIAL EXIT] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Reason: {partial_info['reason']}")
                            print(f"  Profit: {profit_pct:+.2f}%")
                            print(f"{'='*84}\n")
                
                # 완전 청산 조건
                ml_sell_signal = strategy.check_short_signal(latest, prob_down if use_ml else None)
                full_exit = atr_stop_hit or ml_sell_signal
                
                if full_exit:
                    coin_balance = upbit.get_balance(ticker.split('-')[1])
                    if coin_balance > 0:
                        order = upbit.sell_market_order(ticker, coin_balance)
                        
                        if order and 'uuid' in order:
                            time.sleep(0.5)
                            
                            reason = "ATR 스탑" if atr_stop_hit else "룰 매도"
                            
                            log_msg = (f"[SELL] {reason} | Price: {current_price:,.0f} | "
                                     f"Profit: {profit_pct:+.2f}% | Down={prob_down:.3f if prob_down else 'N/A'}")
                            write_log(log_msg)
                            
                            print(f"\n{'='*84}")
                            print(f"[SELL EXECUTED] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Reason: {reason}")
                            print(f"  Profit: {profit_pct:+.2f}%")
                            print(f"{'='*84}\n")
                            
                            holding = False
                            buy_price = 0
                            buy_amount = 0
                            entry_atr = 0
                            partial_exited = False
            
            # 상태 출력
            print("="*84)
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {'HOLDING' if holding else 'WAITING'}")
            print(f"   Price: {current_price:,.0f} | ATR: {current_atr:,.0f}")
            if holding:
                print(f"   Entry: {buy_price:,.0f} | Profit: {profit_pct:+.2f}%")
                print(f"   Stop: {stop_price:,.0f} | Partial: {'DONE' if partial_exited else 'READY'}")
            if prob_up is not None:
                print(f"   ML: Down={prob_down:.3f}, Sideways={prob_sideways:.3f}, Up={prob_up:.3f}")
            print("="*84)
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n[Program stopped by user]")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(30)


if __name__ == "__main__":
    run_live_trading_v4(
        ticker="KRW-BTC",
        ml_buy_threshold=0.25,
        ml_sell_threshold=0.35,
        atr_stop_multiplier=1.2,
        risk_pct=1.0,
        use_ml=True,
        use_partial_exit=True
    )
