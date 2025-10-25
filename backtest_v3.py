"""
v3.0 모델 백테스팅 (슬라이딩 윈도우 방식)
실거래와 동일한 방식으로 백테스트 → 신뢰할 수 있는 결과
"""

import pandas as pd
import numpy as np
import joblib
from download_data import load_daily_csv
from auto_download_data import download_1m_data
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
from datetime import datetime, timedelta


def backtest_v3_sliding_window(start_date_str, num_days=10, buy_threshold=0.20, sell_threshold=0.40, 
                               stop_loss=1.0, take_profit=1.5, verbose=True):
    """
    슬라이딩 윈도우 백테스팅 (실거래 방식)
    - 매 분마다 최신 1800개 윈도우 유지
    - 특징 재계산
    - 실거래 결과와 거의 동일
    
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
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        
        df_day = load_daily_csv(date_str, "data/daily_1m", "1m")
        if df_day is None:
            df_day = download_1m_data(date_str)
        
        if df_day is not None and len(df_day) >= 50:
            df_day = df_day.rename(columns={
                'date_time_utc': 'timestamp', 
                'acc_trade_volume': 'volume',
                'trade_price': 'close',
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low'
            })
            df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])
            df_day = df_day.set_index('timestamp')
            df_day = df_day.sort_index()
            all_data.append(df_day)
            
            if i < 0:
                if verbose:
                    print(f"  Pre-load: {date_str} ({len(df_day)} candles)")
            else:
                if verbose:
                    print(f"  Test day {i+1}: {date_str} ({len(df_day)} candles)")
    
    if len(all_data) < 2:
        if verbose:
            print("[ERROR] Not enough data")
        return None
    
    df_all = pd.concat(all_data).sort_index()
    
    if verbose:
        print(f"\n[OK] Total loaded: {len(df_all)} candles")
        print(f"   Time range: {df_all.index[0]} ~ {df_all.index[-1]}")
        print(f"\n[Starting sliding window backtest...]")
    
    # 거래 변수
    initial_balance = 1_000_000
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trades = []
    
    # 테스트 시작 인덱스 (이전 2일 이후)
    test_start_time = datetime.strptime(start_date_str, "%Y%m%d")
    test_start_idx = df_all.index.searchsorted(test_start_time)
    
    # 일별 통계
    daily_stats = []
    current_date = None
    day_start_balance = balance
    day_start_trades = 0
    
    # 슬라이딩 윈도우 시뮬레이션 (매 분마다)
    for i in range(test_start_idx, len(df_all)):
        # 윈도우 업데이트 (최신 1800개 유지)
        if i >= window_size:
            current_window = df_all.iloc[i-window_size+1:i+1].copy()
        else:
            current_window = df_all.iloc[:i+1].copy()
        
        # 특징 계산 (매 분마다 재계산)
        df_features = current_window.copy()
        df_features = add_all_indicators(df_features)
        df_features = add_multi_timeframe_features(df_features)
        df_features = df_features.dropna()
        
        if len(df_features) < 100:
            continue
        
        # ML 예측 (최신 시점만)
        X_latest = df_features[feature_cols].iloc[-1:]
        
        # LightGBM vs sklearn 모델 구분
        if hasattr(model, 'best_iteration'):
            # LightGBM
            pred = model.predict(X_latest, num_iteration=model.best_iteration)[0]
        else:
            # sklearn 모델 (RandomForest, ExtraTrees, HistGradientBoosting 등)
            pred = model.predict_proba(X_latest)[0]
        
        prob_up = pred[2]
        prob_down = pred[0]
        
        row = df_all.iloc[i]
        price = row['close']
        timestamp = df_all.index[i]
        
        # 일별 통계 업데이트
        check_date = timestamp.date()
        if current_date != check_date:
            if current_date is not None:
                # 이전 날짜 통계 저장
                day_end_balance = balance if coin_holding == 0 else coin_holding * price
                day_return = (day_end_balance - day_start_balance) / day_start_balance * 100
                day_total_return = (day_end_balance - initial_balance) / initial_balance * 100
                day_trades = len([t for t in trades if t['type'] == 'SELL']) - day_start_trades
                
                daily_stats.append({
                    'date': current_date.strftime("%Y%m%d"),
                    'day_return': day_return,
                    'total_return': day_total_return,
                    'day_trades': day_trades,
                    'balance': day_end_balance,
                    'holding': coin_holding > 0
                })
                
                if verbose:
                    holding_str = f"보유중" if coin_holding > 0 else "대기"
                    print(f"  [{current_date.strftime('%Y%m%d')}] 일수익: {day_return:+.2f}% | 누적: {day_total_return:+.2f}% | 거래: {day_trades}회 | {holding_str}")
            
            current_date = check_date
            day_start_balance = balance if coin_holding == 0 else coin_holding * price
            day_start_trades = len([t for t in trades if t['type'] == 'SELL'])
        
        # 거래 로직
        if coin_holding > 0:
            profit_rate = (price - buy_price) / buy_price * 100
            
            sell_reason = None
            if profit_rate <= -stop_loss:
                sell_reason = "손절"
            elif profit_rate >= take_profit:
                sell_reason = "익절"
            elif prob_down >= sell_threshold:
                sell_reason = "ML하락"
            
            if sell_reason:
                balance = coin_holding * price * (1 - fee_rate)
                profit = balance - buy_balance
                trades.append({
                    'type': 'SELL',
                    'timestamp': timestamp,
                    'buy_price': buy_price,
                    'sell_price': price,
                    'amount': coin_holding,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'date': timestamp.strftime("%Y%m%d"),
                    'reason': sell_reason,
                    'prob_down': prob_down
                })
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        else:
            if prob_up >= buy_threshold and balance > 10000:
                buy_balance = balance
                coin_holding = (balance * (1 - fee_rate)) / price
                buy_price = price
                balance = 0
                trades.append({
                    'type': 'BUY',
                    'timestamp': timestamp,
                    'price': price,
                    'amount': coin_holding,
                    'date': timestamp.strftime("%Y%m%d"),
                    'prob_up': prob_up
                })
    
    # 마지막 날 통계
    if current_date is not None:
        day_end_balance = balance if coin_holding == 0 else coin_holding * df_all.iloc[-1]['close']
        day_return = (day_end_balance - day_start_balance) / day_start_balance * 100
        day_total_return = (day_end_balance - initial_balance) / initial_balance * 100
        day_trades = len([t for t in trades if t['type'] == 'SELL']) - day_start_trades
        
        daily_stats.append({
            'date': current_date.strftime("%Y%m%d"),
            'day_return': day_return,
            'total_return': day_total_return,
            'day_trades': day_trades,
            'balance': day_end_balance,
            'holding': coin_holding > 0
        })
        
        if verbose:
            holding_str = f"보유중" if coin_holding > 0 else "대기"
            print(f"  [{current_date.strftime('%Y%m%d')}] 일수익: {day_return:+.2f}% | 누적: {day_total_return:+.2f}% | 거래: {day_trades}회 | {holding_str}")
    
    # 최종 정리
    if coin_holding > 0:
        final_price = df_all.iloc[-1]['close']
        balance = coin_holding * final_price * (1 - fee_rate)
        profit = balance - buy_balance
        profit_rate = (final_price - buy_price) / buy_price * 100
        trades.append({
            'type': 'SELL',
            'timestamp': df_all.index[-1],
            'buy_price': buy_price,
            'sell_price': final_price,
            'amount': coin_holding,
            'profit': profit,
            'profit_rate': profit_rate,
            'date': 'FINAL',
            'reason': '종료',
            'prob_down': 0
        })
        coin_holding = 0
    
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    win_rate = (len([t for t in sell_trades if t['profit'] > 0]) / num_trades * 100) if num_trades > 0 else 0
    
    return {
        'start_date': start_date_str,
        'num_days': len(daily_stats),
        'total_return': total_return,
        'total_trades': num_trades,
        'avg_trades_per_day': num_trades / len(daily_stats) if daily_stats else 0,
        'win_rate': win_rate,
        'final_balance': final_balance,
        'daily_stats': daily_stats,
        'trades': trades
    }


# 호환성을 위한 별칭
def backtest_v3_continuous(start_date_str, num_days=10, buy_threshold=0.20, sell_threshold=0.40, 
                           stop_loss=1.0, take_profit=1.5, verbose=True):
    """
    연속된 N일 백테스팅 (슬라이딩 윈도우 방식)
    backtest_v3_sliding_window의 별칭
    """
    return backtest_v3_sliding_window(start_date_str, num_days, buy_threshold, sell_threshold, 
                                      stop_loss, take_profit, verbose)


if __name__ == "__main__":
    import random
    random.seed(42)
    
    # 테스트
    print("=" * 100)
    print("슬라이딩 윈도우 백테스트 테스트")
    print("=" * 100)
    
    result = backtest_v3_continuous(
        start_date_str="20250401",
        num_days=7,
        buy_threshold=0.20,
        sell_threshold=0.40,
        stop_loss=1.0,
        take_profit=1.5,
        verbose=True
    )
    
    if result:
        print("\n" + "=" * 100)
        print("결과")
        print("=" * 100)
        print(f"수익률: {result['total_return']:+.2f}%")
        print(f"거래: {result['total_trades']}회")
        print(f"승률: {result['win_rate']:.1f}%")
        print(f"최종 잔고: {result['final_balance']:,.0f} KRW")