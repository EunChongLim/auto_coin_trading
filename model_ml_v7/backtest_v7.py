"""
v7 백테스트 - 1분봉 + 틱 데이터 통합
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from Common.tick_loader import download_tick_data
from Common.tick_aggregator import aggregate_ticks_to_minute, add_tick_features
from Common.features_v7 import calculate_combined_features, get_all_feature_columns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.auto_download_data import download_1m_data


def load_combined_data(start_date, end_date):
    """1분봉 + 틱 데이터 통합 로드"""
    print(f"\n[Loading Data] {start_date} ~ {end_date}")
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_candles = []
    all_ticks = []
    
    current_dt = start_dt
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        
        # 1분봉
        candle_path = os.path.join("../data/daily_1m", f"KRW-BTC_candle-1m_{date_str}.csv")
        if not os.path.exists(candle_path):
            candle_df = download_1m_data(date_str, output_dir="../data/daily_1m")
            if candle_df is not None:
                all_candles.append(candle_df)
        else:
            candle_df = pd.read_csv(candle_path)
            all_candles.append(candle_df)
        
        # 틱 데이터
        tick_df = download_tick_data(date_str, output_dir="../data/ticks")
        if tick_df is not None:
            all_ticks.append(tick_df)
        
        current_dt += timedelta(days=1)
    
    # 1분봉 병합
    candle_combined = pd.concat(all_candles, ignore_index=True)
    
    candle_clean = pd.DataFrame()
    
    if 'date_time_kst' in candle_combined.columns:
        candle_clean['datetime'] = pd.to_datetime(candle_combined['date_time_kst'])
    elif 'date_time_utc' in candle_combined.columns:
        candle_clean['datetime'] = pd.to_datetime(candle_combined['date_time_utc'])
    
    if 'open' in candle_combined.columns:
        candle_clean['open'] = candle_combined['open']
        candle_clean['high'] = candle_combined['high']
        candle_clean['low'] = candle_combined['low']
        candle_clean['close'] = candle_combined['close']
    elif 'opening_price' in candle_combined.columns:
        candle_clean['open'] = candle_combined['opening_price']
        candle_clean['high'] = candle_combined['high_price']
        candle_clean['low'] = candle_combined['low_price']
        candle_clean['close'] = candle_combined['trade_price']
    
    if 'volume' in candle_combined.columns:
        candle_clean['volume'] = candle_combined['volume']
    elif 'acc_trade_volume' in candle_combined.columns:
        candle_clean['volume'] = candle_combined['acc_trade_volume']
    
    candle_clean = candle_clean.sort_values('datetime').reset_index(drop=True)
    candle_clean['datetime'] = candle_clean['datetime'].dt.floor('min')
    
    # 틱 데이터 병합
    if all_ticks:
        tick_combined = pd.concat(all_ticks, ignore_index=True)
        tick_agg = aggregate_ticks_to_minute(tick_combined)
        tick_agg = add_tick_features(tick_agg)
        
        result = pd.merge(candle_clean, tick_agg, on='datetime', how='left', suffixes=('', '_tick'))
        
        if 'volume_tick' in result.columns:
            result = result.drop('volume_tick', axis=1)
        
        # 틱 컬럼 채우기
        tick_cols = ['tick_count', 'buy_pressure', 'large_buy_count', 'large_sell_count',
                     'buy_pressure_ma5', 'buy_pressure_ma20', 'buy_pressure_change',
                     'tick_speed_ma5', 'tick_speed_ma20', 'tick_speed_ratio',
                     'buy_pressure_std', 'large_trade_imbalance', 'volume_imbalance', 'price_vs_vwap']
        
        for col in tick_cols:
            if col in result.columns:
                if 'pressure' in col or 'ma' in col:
                    result[col] = result[col].fillna(0.5)
                else:
                    result[col] = result[col].fillna(0)
    else:
        result = candle_clean
    
    print(f"[OK] {len(result):,} minutes loaded")
    return result


def backtest_v7(
    model_path="model/lgb_v7_tick.pkl",
    start_date="20220101",
    end_date="20221231",
    buy_threshold=0.1,
    stop_loss=0.3,
    take_profit=0.6,
    time_limit=8,
    initial_balance=1000000
):
    """v7 백테스트"""
    print("="*80)
    print("v7 Backtest - Candles + Tick Data")
    print("="*80)
    
    # 모델 로드
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    window_size = model_data['window_size']
    
    print(f"\n[Model]")
    print(f"  Version: {model_data['version']}")
    print(f"  Train Date: {model_data['train_date_range']}")
    print(f"  Features: {len(feature_cols)} (v6: 27, Tick: {len(feature_cols)-27})")
    print(f"  Test R²: {model_data['test_r2']:.4f}")
    
    # 데이터 로드
    df = load_combined_data(start_date, end_date)
    
    print(f"\n[Strategy]")
    print(f"  Buy: Predicted >= +{buy_threshold}%")
    print(f"  Stop Loss: -{stop_loss}%")
    print(f"  Take Profit: +{take_profit}%")
    print(f"  Time Limit: {time_limit} 분")
    
    # 백테스트
    balance = initial_balance
    holding = False
    buy_price = 0
    buy_time_idx = 0
    trades = []
    fee_rate = 0.0005
    
    print(f"\n[Backtesting...]")
    
    for i in range(window_size, len(df)):
        # 윈도우 데이터
        candle_window = df.iloc[i-window_size:i][['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        tick_window = None
        if 'buy_pressure' in df.columns:
            tick_window = df.iloc[i-window_size:i]
        
        # Feature 계산
        features = calculate_combined_features(candle_window, tick_window)
        
        if features is None:
            continue
        
        # 예측
        X = pd.DataFrame([features])[feature_cols]
        predicted_profit = model.predict(X, num_iteration=model.best_iteration)[0]
        
        current_price = df.iloc[i]['close']
        
        # 매수
        if not holding and predicted_profit >= buy_threshold:
            if features.get('rsi', 50) > 70:
                continue
            
            buy_price = current_price * (1 + fee_rate)
            buy_amount = balance / buy_price
            balance = 0
            holding = True
            buy_time_idx = i
            
            trades.append({
                'type': 'BUY',
                'time': df.iloc[i]['datetime'],
                'price': buy_price,
                'predicted': predicted_profit
            })
        
        # 매도
        if holding:
            profit_pct = ((current_price - buy_price) / buy_price) * 100
            holding_minutes = i - buy_time_idx
            
            sell = False
            reason = ""
            
            if profit_pct >= take_profit:
                sell = True
                reason = "익절"
            elif profit_pct <= -stop_loss:
                sell = True
                reason = "손절"
            elif holding_minutes >= time_limit:
                sell = True
                reason = "시간초과"
            
            if sell:
                sell_price = current_price * (1 - fee_rate)
                balance = buy_amount * sell_price
                
                trades.append({
                    'type': 'SELL',
                    'time': df.iloc[i]['datetime'],
                    'price': sell_price,
                    'profit': profit_pct,
                    'reason': reason
                })
                
                holding = False
    
    # 최종 청산
    if holding:
        final_price = df.iloc[-1]['close']
        sell_price = final_price * (1 - fee_rate)
        balance = buy_amount * sell_price
        profit_pct = ((final_price - buy_price) / buy_price) * 100
        
        trades.append({
            'type': 'SELL',
            'time': df.iloc[-1]['datetime'],
            'price': sell_price,
            'profit': profit_pct,
            'reason': '강제청산'
        })
    
    # 결과 계산
    if balance == 0 or np.isnan(balance):
        balance = initial_balance
    
    total_return = ((balance - initial_balance) / initial_balance) * 100
    
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    if sell_trades:
        profits = [t['profit'] for t in sell_trades if not np.isnan(t['profit'])]
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = len(win_trades) / len(sell_trades) * 100
        avg_profit = np.mean(profits) if profits else 0
        avg_win = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
        loss_trades = [t['profit'] for t in sell_trades if t['profit'] <= 0]
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        
        reasons = {}
        for t in sell_trades:
            reason = t['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
    else:
        win_rate = 0
        avg_profit = 0
        avg_win = 0
        avg_loss = 0
        reasons = {}
    
    # 통계
    test_days = (datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")).days + 1
    trades_per_day = len(buy_trades) / test_days
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"백테스트 결과")
    print(f"{'='*80}")
    print(f"총 수익률: {total_return:+.2f}%")
    print(f"최종 잔고: {balance:,.0f}원")
    print(f"총 거래: {len(buy_trades)}회 (일평균 {trades_per_day:.1f}회)")
    print(f"승률: {win_rate:.1f}% ({len([t for t in sell_trades if t['profit'] > 0])}/{len(sell_trades)})")
    print(f"평균 수익: {avg_profit:+.3f}%")
    print(f"평균 익절: {avg_win:+.3f}%")
    print(f"평균 손절: {avg_loss:+.3f}%")
    
    if reasons:
        print(f"\n매도 이유:")
        for reason, count in reasons.items():
            pct = (count / len(sell_trades)) * 100
            print(f"  {reason}: {count}회 ({pct:.1f}%)")
    
    print(f"{'='*80}\n")
    
    return {
        'total_return': total_return,
        'num_trades': len(buy_trades),
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'final_balance': balance
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python backtest_v7.py YYYYMMDD YYYYMMDD")
        print("Example: python backtest_v7.py 20220101 20221231")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    
    backtest_v7(
        start_date=start_date,
        end_date=end_date,
        buy_threshold=0.1,
        stop_loss=0.3,
        take_profit=0.6,
        time_limit=8
    )

