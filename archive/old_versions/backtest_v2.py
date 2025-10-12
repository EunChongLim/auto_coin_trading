"""
모델 v2.0 백테스팅: 3-Class Prediction 기반 매매
"""

import pandas as pd
import numpy as np
import joblib
from download_data import load_daily_csv
from indicators import add_all_indicators
from multi_timeframe_features import add_multi_timeframe_features
import random
from datetime import datetime, timedelta


def run_backtest_v2(date_str, model_data, initial_balance=1_000_000, fee_rate=0.0005, 
                     buy_threshold=0.5, sell_threshold=0.5, stop_loss_pct=0.5, take_profit_pct=1.0):
    """
    3-Class 모델 기반 백테스팅
    
    Args:
        date_str: 테스트 날짜
        model_data: 모델 데이터 (model, feature_cols)
        buy_threshold: 매수 신호 임계값 (상승 확률)
        sell_threshold: 매도 신호 임계값 (하락 확률)
        stop_loss_pct: 손절 비율
        take_profit_pct: 익절 비율
    """
    # 1. 데이터 로드
    df = load_daily_csv(date_str, "data/daily_1m", "1m")
    if df is None or len(df) == 0:
        return None
    
    # 컬럼 매핑
    df = df.rename(columns={
        'date_time_utc': 'timestamp',
        'acc_trade_volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 2. 특징 생성
    df = add_all_indicators(df)
    df = add_multi_timeframe_features(df)
    df = df.dropna()
    
    if len(df) < 100:
        return None
    
    # 3. 예측
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # 특징 존재 여부 확인
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"   ⚠️  누락된 특징: {missing_cols}")
        return None
    
    X = df[feature_cols]
    predictions = model.predict(X, num_iteration=model.best_iteration)
    
    # 예측 확률: [하락(0), 횡보(1), 상승(2)]
    df['prob_down'] = predictions[:, 0]
    df['prob_sideways'] = predictions[:, 1]
    df['prob_up'] = predictions[:, 2]
    df['pred_class'] = np.argmax(predictions, axis=1)
    
    # 4. 백테스팅
    balance = initial_balance
    buy_balance = 0
    coin_holding = 0
    buy_price = 0
    
    trades = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        price = row['close']
        
        # 보유 중
        if coin_holding > 0:
            profit_rate = (price - buy_price) / buy_price * 100
            
            sell_reason = None
            
            # 손절
            if profit_rate <= -stop_loss_pct:
                sell_reason = "손절"
            # 익절
            elif profit_rate >= take_profit_pct:
                sell_reason = "익절"
            # 하락 예측 매도
            elif row['prob_down'] >= sell_threshold:
                sell_reason = f"하락예측({row['prob_down']:.2f})"
            
            if sell_reason:
                # 매도
                balance = coin_holding * price * (1 - fee_rate)
                profit = balance - buy_balance
                
                trades.append({
                    'type': 'SELL',
                    'reason': sell_reason,
                    'time': idx,
                    'price': price,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'balance_after': balance
                })
                
                coin_holding = 0
                buy_price = 0
                buy_balance = 0
        
        # 미보유 중
        else:
            # 상승 예측 매수
            if row['prob_up'] >= buy_threshold and balance > 10000:
                # 매수
                buy_balance = balance
                coin_holding = (balance * (1 - fee_rate)) / price
                buy_price = price
                
                trades.append({
                    'type': 'BUY',
                    'reason': f"상승예측({row['prob_up']:.2f})",
                    'time': idx,
                    'price': price,
                    'balance_before': balance
                })
                
                balance = 0
    
    # 마지막 포지션 청산
    if coin_holding > 0:
        final_price = df.iloc[-1]['close']
        balance = coin_holding * final_price * (1 - fee_rate)
        profit = balance - buy_balance
        profit_rate = (final_price - buy_price) / buy_price * 100
        
        trades.append({
            'type': 'SELL',
            'reason': '종료',
            'time': df.index[-1],
            'price': final_price,
            'profit': profit,
            'profit_rate': profit_rate,
            'balance_after': balance
        })
    
    # 5. 결과 계산
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    num_trades = len(sell_trades)
    
    if num_trades > 0:
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        lose_trades = [t for t in sell_trades if t['profit'] <= 0]
        win_rate = len(win_trades) / num_trades * 100
        
        avg_profit = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([abs(t['profit']) for t in lose_trades]) if lose_trades else 0
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
    
    # 예측 분포 통계
    pred_dist = df['pred_class'].value_counts()
    
    return {
        'date': date_str,
        'return': total_return,
        'final_balance': final_balance,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'pred_down': pred_dist.get(0, 0),
        'pred_sideways': pred_dist.get(1, 0),
        'pred_up': pred_dist.get(2, 0),
        'trades': trades
    }


def run_multi_day_backtest_v2(start_date, end_date, num_days=10):
    """
    여러 날짜에 대해 백테스팅 수행
    """
    print("=" * 80)
    print("🚀 모델 v2.0 멀티 데이 백테스팅")
    print("=" * 80)
    
    # 모델 로드
    print("\n📦 모델 로드 중...")
    model_data = joblib.load("model/lgb_model_v2.pkl")
    print(f"   - 버전: {model_data['version']}")
    print(f"   - 타입: {model_data['type']}")
    print(f"   - 특징 수: {len(model_data['feature_cols'])}개")
    
    # 테스트 날짜 랜덤 샘플링
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    test_days = sorted(random.sample(all_days, min(num_days, len(all_days))))
    
    print(f"\n📅 테스트 기간: {num_days}일")
    print(f"   {', '.join(test_days[:5])}...")
    
    # 백테스팅
    results = []
    
    for i, date_str in enumerate(test_days, 1):
        print(f"\n[{i}/{num_days}] {date_str} 백테스팅 중...")
        
        result = run_backtest_v2(date_str, model_data)
        
        if result:
            results.append(result)
            print(f"   수익률: {result['return']:+.2f}% | 거래: {result['num_trades']}회 | 승률: {result['win_rate']:.1f}%")
            print(f"   예측: 하락={result['pred_down']}, 횡보={result['pred_sideways']}, 상승={result['pred_up']}")
    
    # 집계
    if not results:
        print("\n❌ 백테스팅 결과 없음")
        return
    
    print("\n" + "=" * 80)
    print("📊 종합 결과")
    print("=" * 80)
    
    avg_return = np.mean([r['return'] for r in results])
    avg_trades = np.mean([r['num_trades'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    
    print(f"\n💰 평균 수익률: {avg_return:+.2f}%")
    print(f"📈 평균 거래 횟수: {avg_trades:.1f}회/일")
    print(f"🎯 평균 승률: {avg_win_rate:.1f}%")
    
    # 날짜별 결과
    print(f"\n📅 날짜별 결과:")
    for r in results:
        print(f"   {r['date']}: {r['return']:+.2f}% ({r['num_trades']}회)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    random.seed(42)
    run_multi_day_backtest_v2("20250101", "20250530", num_days=10)

