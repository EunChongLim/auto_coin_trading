"""
멀티 타임프레임 백테스트
학습과 동일한 방식으로 과거 데이터 검증
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_multi.Common.data_loader import load_multi_timeframe_data
from model_multi.Common.features_multi import calculate_multi_timeframe_features


def backtest_multi_timeframe(
    model_path="model/lgb_multi_timeframe.pkl",
    start_date="20220101",
    end_date="20221231",
    buy_threshold=0.1,
    stop_loss=0.3,
    take_profit=0.6,
    time_limit=8,
    initial_balance=1000000
):
    """
    멀티 타임프레임 백테스트
    
    Args:
        model_path: 모델 파일 경로
        start_date, end_date: 백테스트 기간
        buy_threshold: 매수 임계값 (예측 수익률 %)
        stop_loss: 손절 %
        take_profit: 익절 %
        time_limit: 최대 보유 시간 (분)
        initial_balance: 초기 자본
    """
    print("="*80)
    print("멀티 타임프레임 백테스트")
    print("="*80)
    
    # === 1. 모델 로드 ===
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    window_size_1m = model_data['window_size_1m']
    window_size_1h = model_data['window_size_1h']
    window_size_4h = model_data['window_size_4h']
    
    print(f"\n[Model Info]")
    print(f"  Version: {model_data['version']}")
    print(f"  Train Date: {model_data['train_date_range']}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Test R²: {model_data['test_r2']:.4f}")
    print(f"  Window: 1m={window_size_1m}, 1h={window_size_1h}, 4h={window_size_4h}")
    
    # === 2. 데이터 로드 (각 타임프레임을 API에서 직접) ===
    df_1m, df_1h, df_4h = load_multi_timeframe_data(start_date, end_date)
    
    # === 3. 백테스트 전략 ===
    print(f"\n[Strategy]")
    print(f"  Buy: Predicted >= +{buy_threshold}%")
    print(f"  Stop Loss: -{stop_loss}%")
    print(f"  Take Profit: +{take_profit}%")
    print(f"  Time Limit: {time_limit} 분")
    
    # === 4. 백테스트 실행 ===
    balance = initial_balance
    holding = False
    buy_price = 0
    buy_time_idx = 0
    buy_amount = 0
    
    trades = []
    fee_rate = 0.0005
    
    # 시작 인덱스
    start_idx = max(window_size_1m, window_size_1h * 60, window_size_4h * 240)
    
    print(f"\n[Backtesting...] from idx {start_idx}")
    
    for i in range(start_idx, len(df_1m)):
        current_time = df_1m.iloc[i]['datetime']
        current_price = df_1m.iloc[i]['close']
        
        # === 보유 중이면 매도 확인 ===
        if holding:
            profit_pct = (current_price - buy_price) / buy_price * 100
            hold_minutes = i - buy_time_idx
            
            sell = False
            reason = ""
            
            if profit_pct >= take_profit:
                sell = True
                reason = "익절"
            elif profit_pct <= -stop_loss:
                sell = True
                reason = "손절"
            elif hold_minutes >= time_limit:
                sell = True
                reason = "시간초과"
            
            if sell:
                sell_price = current_price * (1 - fee_rate)
                balance = buy_amount * sell_price
                
                trades.append({
                    'type': 'SELL',
                    'time': current_time,
                    'price': sell_price,
                    'profit': profit_pct,
                    'reason': reason
                })
                
                holding = False
                buy_price = 0
                buy_time_idx = 0
                buy_amount = 0
        
        # === 미보유 시 매수 확인 ===
        else:
            # 윈도우 데이터 준비 (학습과 동일!)
            window_1m = df_1m.iloc[i - window_size_1m:i]
            window_1h = df_1h[df_1h['datetime'] <= current_time].tail(window_size_1h)
            window_4h = df_4h[df_4h['datetime'] <= current_time].tail(window_size_4h)
            
            # 윈도우 크기 확인
            if len(window_1m) < window_size_1m or len(window_1h) < 30 or len(window_4h) < 30:
                continue
            
            # Feature 계산 (학습과 100% 동일!)
            features = calculate_multi_timeframe_features(window_1m, window_1h, window_4h)
            
            if features is None:
                continue
            
            # 예측
            X = pd.DataFrame([features])[feature_cols]
            X = X.fillna(0)
            
            predicted_profit = model.predict(X, num_iteration=model.best_iteration)[0]
            
            # 매수 조건
            if predicted_profit >= buy_threshold:
                # 추가 필터: 1분봉 RSI 과매수 방지
                if features.get('1m_rsi', 50) > 75:
                    continue
                
                # 매수 실행
                buy_price = current_price * (1 + fee_rate)
                buy_amount = balance / buy_price
                balance = 0
                holding = True
                buy_time_idx = i
                
                trades.append({
                    'type': 'BUY',
                    'time': current_time,
                    'price': buy_price,
                    'predicted': predicted_profit
                })
        
        # 진행 상황 출력
        if i % 10000 == 0:
            progress = (i - start_idx) / (len(df_1m) - start_idx) * 100
            print(f"  Progress: {progress:.1f}% (idx {i}/{len(df_1m)})")
    
    # === 5. 최종 청산 ===
    if holding:
        final_price = df_1m.iloc[-1]['close']
        sell_price = final_price * (1 - fee_rate)
        balance = buy_amount * sell_price
        profit_pct = (final_price - buy_price) / buy_price * 100
        
        trades.append({
            'type': 'SELL',
            'time': df_1m.iloc[-1]['datetime'],
            'price': sell_price,
            'profit': profit_pct,
            'reason': '강제청산'
        })
    
    # === 6. 결과 분석 ===
    if balance == 0 or np.isnan(balance):
        balance = initial_balance
    
    total_return = (balance - initial_balance) / initial_balance * 100
    
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    if sell_trades:
        profits = [t['profit'] for t in sell_trades if not np.isnan(t['profit'])]
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        loss_trades = [t for t in sell_trades if t['profit'] <= 0]
        
        win_rate = len(win_trades) / len(sell_trades) * 100
        avg_profit = np.mean(profits) if profits else 0
        avg_win = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['profit'] for t in loss_trades]) if loss_trades else 0
        
        # 매도 이유별 통계
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
    trades_per_day = len(buy_trades) / test_days if test_days > 0 else 0
    
    # === 7. 결과 출력 ===
    print(f"\n{'='*80}")
    print(f"백테스트 결과")
    print(f"{'='*80}")
    print(f"총 수익률: {total_return:+.2f}%")
    print(f"최종 잔고: {balance:,.0f}원")
    print(f"총 거래: {len(buy_trades)}회 (일평균 {trades_per_day:.1f}회)")
    
    if sell_trades:
        print(f"승률: {win_rate:.1f}% ({len(win_trades)}/{len(sell_trades)})")
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
        'final_balance': balance,
        'trades': trades
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python backtest_multi.py YYYYMMDD YYYYMMDD")
        print("Example: python backtest_multi.py 20220101 20221231")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    
    backtest_multi_timeframe(
        start_date=start_date,
        end_date=end_date,
        buy_threshold=0.1,
        stop_loss=0.3,
        take_profit=0.6,
        time_limit=8
    )

