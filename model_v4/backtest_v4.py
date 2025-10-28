"""
model_v4 백테스팅 (A-E 규칙 + B규칙 전략)
룰 기반 + ML 보조 + ATR 동적 스탑 + 부분청산
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# 루트 Common (데이터 로드)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.download_data import load_daily_csv
from Common.auto_download_data import download_1m_data

# model_v4 Common (지표 계산 + 전략)
sys.path.insert(0, os.path.dirname(__file__))
from Common.indicators import add_all_indicators
from Common.multi_timeframe_features import add_multi_timeframe_features
from Common.strategy_rules import RuleBasedStrategy

from datetime import datetime, timedelta


def backtest_v4_with_rules(start_date_str, num_days=10, 
                           ml_buy_threshold=0.25, 
                           ml_sell_threshold=0.35,
                           atr_stop_multiplier=1.2,
                           risk_pct=1.0,
                           use_ml=True,
                           use_partial_exit=True,
                           verbose=True):
    """
    model_v4 B규칙 전략 백테스팅
    
    Args:
        start_date_str: 시작일 (YYYYMMDD)
        num_days: 테스트 일수
        ml_buy_threshold: ML 매수 임계값
        ml_sell_threshold: ML 매도 임계값
        atr_stop_multiplier: ATR 스탑 배수 (k)
        risk_pct: 위험 퍼센트 (0.5~1.0%)
        use_ml: ML 보조 신호 사용 여부
        use_partial_exit: 부분청산 사용 여부
        verbose: 상세 출력
    """
    # 모델 로드
    try:
        model_data = joblib.load("model/lgb_model_v4_enhanced.pkl")
        model = model_data['model']
        feature_cols = model_data['feature_cols']
    except:
        if verbose:
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
    
    fee_rate = 0.0005
    window_size = 1800
    
    # 데이터 로드
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    all_data = []
    
    if verbose:
        print(f"\n[Loading data: 2 days before + {num_days} test days from {start_date_str}...]")
    
    for i in range(-2, num_days):
        date = start_date + timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        
        df = load_daily_csv(date_str, data_dir="../data/daily_1m")
        if df is not None:
            all_data.append(df)
            if verbose and i >= 0:
                print(f"  [{i+1}/{num_days}] {date_str}: {len(df)} candles")
    
    if not all_data:
        if verbose:
            print("[ERROR] No data loaded")
        return None
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 컬럼명 통일 (백테스트 호환성)
    if 'acc_trade_volume' in df_all.columns:
        df_all = df_all.rename(columns={'acc_trade_volume': 'volume'})
    if 'date_time_utc' in df_all.columns:
        df_all = df_all.rename(columns={'date_time_utc': 'candle_date_time_kst'})
    
    # 중복 제거 및 정렬
    if 'candle_date_time_kst' in df_all.columns:
        df_all = df_all.drop_duplicates(subset=['candle_date_time_kst'])
        df_all = df_all.sort_values('candle_date_time_kst')
        time_col = 'candle_date_time_kst'
    elif 'timestamp' in df_all.columns:
        df_all = df_all.drop_duplicates(subset=['timestamp'])
        df_all = df_all.sort_values('timestamp')
        time_col = 'timestamp'
    else:
        df_all = df_all.drop_duplicates()
        time_col = df_all.columns[0]
    
    # trade_price가 없으면 close로 대체
    if 'trade_price' not in df_all.columns and 'close' in df_all.columns:
        df_all['trade_price'] = df_all['close']
    
    if verbose:
        print(f"[OK] Total {len(df_all)} candles loaded")
        print(f"\n[Strategy] B규칙 + ATR 동적 스탑")
        print(f"  - ML 보조: {'ON' if use_ml else 'OFF'}")
        print(f"  - 부분청산: {'ON' if use_partial_exit else 'OFF'}")
        print(f"  - ATR 배수: {atr_stop_multiplier}x")
        print(f"  - 위험률: {risk_pct}%\n")
    
    # 거래 상태
    balance = 1_000_000
    initial_balance = balance
    holding = False
    buy_price = 0
    buy_amount = 0
    entry_atr = 0
    partial_exited = False
    trades = []
    
    # 슬라이딩 윈도우 백테스트
    for i in range(window_size, len(df_all)):
        df_window = df_all.iloc[i-window_size:i].copy()
        df_window = df_window.reset_index(drop=True)
        
        # 특징 계산
        df_features = df_window.copy()
        
        # DatetimeIndex 설정 (resample을 위해 필수)
        if time_col in df_features.columns:
            df_features[time_col] = pd.to_datetime(df_features[time_col])
            df_features = df_features.set_index(time_col)
        
        df_features = add_all_indicators(df_features)
        df_features = add_multi_timeframe_features(df_features)
        df_features = df_features.dropna()
        
        if len(df_features) == 0:
            continue
        
        latest = df_features.iloc[-1]
        
        # ML 예측 (선택적)
        prob_down, prob_up = None, None
        if use_ml and model is not None:
            missing = [col for col in feature_cols if col not in latest.index]
            if not missing:
                X = latest[feature_cols].values.reshape(1, -1)
                
                if hasattr(model, 'best_iteration'):
                    probs = model.predict(X, num_iteration=model.best_iteration)[0]
                else:
                    probs = model.predict_proba(X)[0]
                
                prob_down = probs[0]
                prob_up = probs[2]
        
        current_candle = df_all.iloc[i]
        current_price = current_candle.get('trade_price', current_candle.get('close', 0))
        current_time = current_candle.get(time_col, df_all.index[i] if hasattr(df_all, 'index') else i)
        current_atr = latest.get('atr', current_price * 0.02)  # 기본 2%
        
        # === 매수 신호 (진입) ===
        if not holding:
            buy_signal = strategy.check_long_signal(latest, prob_up if use_ml else None)
            
            if buy_signal:
                # ATR 기반 포지션 사이징
                buy_amount = strategy.calculate_position_size(balance, current_price, current_atr)
                buy_price = current_price * (1 + fee_rate)
                entry_atr = current_atr
                balance = 0
                holding = True
                partial_exited = False
                
                trades.append({
                    'type': 'BUY',
                    'time': current_time,
                    'price': buy_price,
                    'amount': buy_amount,
                    'atr': entry_atr,
                    'prob_up': prob_up if use_ml else None,
                    'balance': 0
                })
        
        # === 매도 신호 (청산) ===
        if holding:
            profit_pct = ((current_price - buy_price) / buy_price) * 100
            
            # ATR 기반 스탑로스
            stop_price = strategy.calculate_stop_loss(buy_price, entry_atr, direction='long')
            atr_stop_hit = current_price <= stop_price
            
            # 부분청산 체크
            partial_exit_signal = False
            if use_partial_exit and not partial_exited:
                partial_info = strategy.check_partial_exit(buy_price, current_price, entry_atr, direction='long')
                if partial_info['should_exit']:
                    # 50% 부분 매도
                    exit_amount = buy_amount * partial_info['exit_ratio']
                    sell_price = current_price * (1 - fee_rate)
                    partial_balance = exit_amount * sell_price
                    balance += partial_balance
                    buy_amount -= exit_amount
                    partial_exited = True
                    
                    trades.append({
                        'type': 'PARTIAL_SELL',
                        'time': current_time,
                        'price': sell_price,
                        'amount': exit_amount,
                        'profit_pct': profit_pct,
                        'reason': partial_info['reason'],
                        'balance': balance
                    })
                    partial_exit_signal = True
            
            # 완전 청산 조건
            # 1. ATR 스탑
            # 2. ML + 룰 기반 매도 신호
            ml_sell_signal = strategy.check_short_signal(latest, prob_down if use_ml else None)
            
            full_exit = atr_stop_hit or ml_sell_signal
            
            if full_exit:
                sell_price = current_price * (1 - fee_rate)
                balance = buy_amount * sell_price
                
                reason = ""
                if atr_stop_hit:
                    reason = f"ATR 스탑 ({atr_stop_multiplier:.1f}x)"
                else:
                    reason = "룰 매도"
                
                trades.append({
                    'type': 'SELL',
                    'time': current_time,
                    'price': sell_price,
                    'profit_pct': profit_pct,
                    'prob_down': prob_down if use_ml else None,
                    'reason': reason,
                    'balance': balance
                })
                
                holding = False
                buy_price = 0
                buy_amount = 0
                entry_atr = 0
                partial_exited = False
    
    # 최종 정산
    if holding:
        final_candle = df_all.iloc[-1]
        final_price = final_candle.get('trade_price', final_candle.get('close', 0))
        final_time = final_candle.get(time_col, df_all.index[-1] if hasattr(df_all, 'index') else len(df_all)-1)
        sell_price = final_price * (1 - fee_rate)
        balance = buy_amount * sell_price
        profit_pct = ((sell_price - buy_price) / buy_price) * 100
        
        trades.append({
            'type': 'SELL',
            'time': final_time,
            'price': sell_price,
            'profit_pct': profit_pct,
            'prob_down': None,
            'reason': "종료",
            'balance': balance
        })
    
    # 결과 계산
    total_return = ((balance - initial_balance) / initial_balance) * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winning_trades = [t for t in sell_trades if t.get('profit_pct', 0) > 0]
    losing_trades = [t for t in sell_trades if t.get('profit_pct', 0) <= 0]
    
    win_rate = (len(winning_trades) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
    
    result = {
        'total_return': total_return,
        'win_rate': win_rate,
        'num_trades': len([t for t in trades if t['type'] == 'BUY']),
        'winning': len(winning_trades),
        'losing': len(losing_trades),
        'partial_exits': len([t for t in trades if t['type'] == 'PARTIAL_SELL']),
        'final_balance': balance,
        'trades': trades
    }
    
    if verbose:
        print(f"\n[Backtest Result - B규칙 전략]")
        print(f"  Return: {total_return:+.2f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Trades: {result['num_trades']} ({result['winning']}승 {result['losing']}패)")
        print(f"  Partial Exits: {result['partial_exits']}")
        print(f"  Final Balance: {balance:,.0f} KRW")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtest_v4.py YYYYMMDD [num_days]")
        print("Example: python backtest_v4.py 20250328 10")
        sys.exit(1)
    
    start_date = sys.argv[1]
    num_days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    backtest_v4_with_rules(
        start_date_str=start_date,
        num_days=num_days,
        ml_buy_threshold=0.25,
        ml_sell_threshold=0.35,
        atr_stop_multiplier=1.2,
        risk_pct=1.0,
        use_ml=True,
        use_partial_exit=True,
        verbose=True
    )
