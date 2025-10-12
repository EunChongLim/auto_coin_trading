"""
앙상블 ML 모델 백테스팅
"""

import pandas as pd
import numpy as np
from download_data import load_daily_csv
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
from datetime import datetime, timedelta
import random
import joblib


def backtest_ensemble_ml(date_str, initial_balance=1_000_000, fee_rate=0.0005, threshold=0.15):
    """
    앙상블 ML 백테스팅
    """
    # 모델 로드
    ensemble_data = joblib.load("model/ensemble_model.pkl")
    model_lgb = ensemble_data['lgb']
    model_xgb = ensemble_data['xgb']
    model_cat = ensemble_data['cat']
    feature_cols = ensemble_data['feature_cols']
    
    # 데이터 로드
    df = load_daily_csv(date_str, "data/daily_1m", "1m")
    if df is None or len(df) < 100:
        return None
    
    df = df.rename(columns={'date_time_utc': 'timestamp', 'acc_trade_volume': 'volume'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 특징 생성
    df = add_all_indicators(df)
    df = add_multi_timeframe_features(df)
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    # 예측
    X = df[feature_cols]
    
    # LightGBM 예측
    pred_lgb = model_lgb.predict(X)
    
    # XGBoost 예측 (있으면)
    if model_xgb:
        pred_xgb = model_xgb.predict_proba(X)[:, 1]
    else:
        pred_xgb = pred_lgb
    
    # CatBoost 예측 (있으면)
    if model_cat:
        pred_cat = model_cat.predict_proba(X)[:, 1]
    else:
        pred_cat = pred_lgb
    
    # 앙상블 (평균)
    df['pred_prob'] = (pred_lgb + pred_xgb + pred_cat) / 3
    
    # 백테스팅
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    trades = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        price = row['close']
        pred_prob = row['pred_prob']
        
        # 보유 중
        if coin_holding > 0:
            profit_rate = (price - buy_price) / buy_price * 100
            
            sell_reason = None
            if profit_rate <= -0.6:
                sell_reason = "손절"
            elif profit_rate >= 1.5:
                sell_reason = "익절"
            elif pred_prob < 0.05:
                sell_reason = "ML하락"
            
            if sell_reason:
                balance = coin_holding * price * (1 - fee_rate)
                profit = balance - buy_balance
                
                trades.append({
                    'type': 'SELL',
                    'reason': sell_reason,
                    'profit': profit,
                    'profit_rate': profit_rate
                })
                
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        
        # 미보유 중
        else:
            if pred_prob >= threshold and balance > 10000:
                buy_balance = balance
                coin_holding = (balance * (1 - fee_rate)) / price
                buy_price = price
                balance = 0
                
                trades.append({'type': 'BUY', 'price': price})
    
    # 청산
    if coin_holding > 0:
        final_price = df.iloc[-1]['close']
        balance = coin_holding * final_price * (1 - fee_rate)
        profit = balance - buy_balance
        
        trades.append({
            'type': 'SELL',
            'reason': '종료',
            'profit': profit,
            'profit_rate': (final_price - buy_price) / buy_price * 100
        })
    
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    win_rate = len([t for t in sell_trades if t['profit'] > 0]) / num_trades * 100 if num_trades > 0 else 0
    
    return {
        'date': date_str,
        'return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate
    }


def test_ensemble_ml(num_days=10):
    """앙상블 ML 테스트"""
    print("=" * 80)
    print("Ensemble ML Testing")
    print("=" * 80)
    
    start = datetime.strptime("20250101", "%Y%m%d")
    end = datetime.strptime("20250530", "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(num_days, len(all_days))))
    
    print(f"\nTest Period: {num_days} days\n")
    
    results = []
    for i, date_str in enumerate(test_days, 1):
        print(f"[{i}/{num_days}] {date_str}...")
        result = backtest_ensemble_ml(date_str, threshold=0.15)
        if result:
            results.append(result)
            print(f"   Return: {result['return']:+.2f}% | Trades: {result['num_trades']} | Win: {result['win_rate']:.1f}%")
    
    if results:
        avg_return = np.mean([r['return'] for r in results])
        avg_trades = np.mean([r['num_trades'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
        
        print("\n" + "=" * 80)
        print("Ensemble ML Results")
        print("=" * 80)
        print(f"\nAvg Return: {avg_return:+.2f}%")
        print(f"Avg Trades: {avg_trades:.1f}/day")
        print(f"Avg Win Rate: {avg_win_rate:.1f}%")
        
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        print(f"\nML v2.0    : +1.46% (3.0 trades/day, 77.4% win)")
        print(f"Ensemble ML: {avg_return:+.2f}% ({avg_trades:.1f} trades/day, {avg_win_rate:.1f}% win)")
        
        if avg_return > 1.46:
            print("\n*** ENSEMBLE ML WINS! ***")
            return True
        else:
            print("\nNot better than ML v2.0")
            return False


if __name__ == "__main__":
    random.seed(42)
    test_ensemble_ml(10)

