"""
model_v3 백테스팅 (슬라이딩 윈도우 방식)
실거래와 동일한 방식으로 백테스트
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

# model_v3 Common (지표 계산)
sys.path.insert(0, os.path.dirname(__file__))
from Common.indicators import add_all_indicators
from Common.multi_timeframe_features import add_multi_timeframe_features

from datetime import datetime, timedelta


def backtest_v3_sliding_window(start_date_str, num_days=10, buy_threshold=0.25, sell_threshold=0.35, 
                               stop_loss=1.5, take_profit=1.2, verbose=True):
    """
    슬라이딩 윈도우 백테스팅 (실거래 방식)
    
    Args:
        start_date_str: 시작일 (YYYYMMDD)
        num_days: 테스트 일수
        buy_threshold: 매수 임계값 (Up 확률)
        sell_threshold: 매도 임계값 (Down 확률)
        stop_loss: 손절률 (%)
        take_profit: 익절률 (%)
        verbose: 상세 출력 여부
    """
    model_data = joblib.load("model/lgb_model_v3.pkl")
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    fee_rate = 0.0005
    window_size = 1800  # 실거래와 동일
    
    # 데이터 로드
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    all_data = []
    
    if verbose:
        print(f"\n[Loading data: 2 days before + {num_days} test days from {start_date_str}...]")
    
    # 이전 2일 + 테스트 기간 로드
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
    df_all = df_all.drop_duplicates(subset=['candle_date_time_kst'])
    df_all = df_all.sort_values('candle_date_time_kst')
    
    if verbose:
        print(f"[OK] Total {len(df_all)} candles loaded")
    
    # 거래 상태
    balance = 1_000_000
    initial_balance = balance
    holding = False
    buy_price = 0
    buy_amount = 0
    trades = []
    
    # 슬라이딩 윈도우 백테스트
    for i in range(window_size, len(df_all)):
        df_window = df_all.iloc[i-window_size:i].copy()
        df_window = df_window.reset_index(drop=True)
        
        # 특징 계산
        df_features = df_window.copy()
        df_features = add_all_indicators(df_features)
        df_features = add_multi_timeframe_features(df_features)
        df_features = df_features.dropna()
        
        if len(df_features) == 0:
            continue
        
        latest = df_features.iloc[-1]
        missing = [col for col in feature_cols if col not in latest.index]
        
        if missing:
            continue
        
        X = latest[feature_cols].values.reshape(1, -1)
        
        if hasattr(model, 'best_iteration'):
            probs = model.predict(X, num_iteration=model.best_iteration)[0]
        else:
            probs = model.predict_proba(X)[0]
        
        prob_down = probs[0]
        prob_up = probs[2]
        
        current_candle = df_all.iloc[i]
        current_price = current_candle['trade_price']
        current_time = current_candle['candle_date_time_kst']
        
        # 매수 신호
        if not holding and prob_up >= buy_threshold:
            buy_price = current_price * (1 + fee_rate)
            buy_amount = balance / buy_price
            balance = 0
            holding = True
            
            trades.append({
                'type': 'BUY',
                'time': current_time,
                'price': buy_price,
                'amount': buy_amount,
                'prob_up': prob_up,
                'balance': 0
            })
        
        # 매도 신호
        if holding:
            profit_pct = ((current_price - buy_price) / buy_price) * 100
            
            sell_by_ml = (prob_down >= sell_threshold)
            sell_by_stop = (profit_pct <= -stop_loss)
            sell_by_take = (profit_pct >= take_profit)
            
            if sell_by_ml or sell_by_stop or sell_by_take:
                sell_price = current_price * (1 - fee_rate)
                balance = buy_amount * sell_price
                
                reason = ""
                if sell_by_take:
                    reason = "익절"
                elif sell_by_stop:
                    reason = "손절"
                else:
                    reason = "ML 매도"
                
                trades.append({
                    'type': 'SELL',
                    'time': current_time,
                    'price': sell_price,
                    'profit_pct': profit_pct,
                    'prob_down': prob_down,
                    'reason': reason,
                    'balance': balance
                })
                
                holding = False
                buy_price = 0
                buy_amount = 0
    
    # 최종 정산
    if holding:
        final_candle = df_all.iloc[-1]
        sell_price = final_candle['trade_price'] * (1 - fee_rate)
        balance = buy_amount * sell_price
        profit_pct = ((sell_price - buy_price) / buy_price) * 100
        
        trades.append({
            'type': 'SELL',
            'time': final_candle['candle_date_time_kst'],
            'price': sell_price,
            'profit_pct': profit_pct,
            'prob_down': 0,
            'reason': "종료",
            'balance': balance
        })
    
    # 결과 계산
    total_return = ((balance - initial_balance) / initial_balance) * 100
    
    winning_trades = [t for t in trades if t['type'] == 'SELL' and t.get('profit_pct', 0) > 0]
    losing_trades = [t for t in trades if t['type'] == 'SELL' and t.get('profit_pct', 0) <= 0]
    
    win_rate = (len(winning_trades) / len([t for t in trades if t['type'] == 'SELL'])) * 100 if len([t for t in trades if t['type'] == 'SELL']) > 0 else 0
    
    result = {
        'total_return': total_return,
        'win_rate': win_rate,
        'num_trades': len([t for t in trades if t['type'] == 'BUY']),
        'winning': len(winning_trades),
        'losing': len(losing_trades),
        'final_balance': balance,
        'trades': trades
    }
    
    if verbose:
        print(f"\n[Backtest Result]")
        print(f"  Return: {total_return:+.2f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Trades: {result['num_trades']} ({result['winning']}승 {result['losing']}패)")
        print(f"  Final Balance: {balance:,.0f} KRW")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtest_v3.py YYYYMMDD [num_days]")
        print("Example: python backtest_v3.py 20250328 10")
        sys.exit(1)
    
    start_date = sys.argv[1]
    num_days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    backtest_v3_sliding_window(
        start_date_str=start_date,
        num_days=num_days,
        buy_threshold=0.25,
        sell_threshold=0.35,
        stop_loss=1.5,
        take_profit=1.2,
        verbose=True
    )

