"""
하이브리드 백테스팅 (규칙 + ML)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

from download_data import load_daily_csv
from indicators import add_all_indicators
from feature_engineer import create_features
from ml_model import MLSignalModel
from rule_engine import RuleEngine
from risk_manager import RiskManager


def run_hybrid_backtest(df, ml_model, rule_engine, risk_manager, 
                        ml_threshold=0.65, verbose=True):
    """
    하이브리드 백테스팅 실행
    
    Args:
        df: OHLCV 데이터프레임
        ml_model: ML 모델
        rule_engine: 규칙 엔진
        risk_manager: 리스크 관리자
        ml_threshold: ML 매수 임계값
        verbose: 상세 출력 여부
    
    Returns:
        dict: 백테스팅 결과
    """
    # 초기 설정
    initial_balance = 1_000_000
    balance = initial_balance
    coin = 0
    entry_price = 0
    buy_balance = 0  # 매수 시점의 잔고 기록
    
    trades = []
    
    # 지표 추가
    df = add_all_indicators(df)
    
    # 특징 생성
    df, feature_cols = create_features(df)
    
    # 규칙 기반 신호 생성
    buy_signal, sell_signal = rule_engine.get_signals(df)
    
    # NaN 제거 후 시작
    start_idx = df[feature_cols].notna().all(axis=1).idxmax()
    start_pos = df.index.get_loc(start_idx)
    
    if verbose:
        print(f"\n🔄 하이브리드 백테스팅 시작 (데이터: {len(df):,}개)")
        print(f"   ML 임계값: {ml_threshold:.2f}")
    
    # 백테스팅 루프
    for i in range(start_pos, len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        price = row['close']
        
        # ML 예측
        features = df.iloc[i][feature_cols]
        ml_prob = ml_model.predict_proba(features)
        
        # 포지션 보유 중
        if coin > 0:
            # 손익 체크
            close_reason = risk_manager.should_close_position(entry_price, price)
            
            # 매도 조건: 손익 또는 규칙 기반 매도 신호 + ML 낮은 확률
            if close_reason or (sell_signal.iloc[i] and ml_prob < 0.02):
                # 매도
                balance = coin * price * (1 - risk_manager.fee_rate)
                profit = balance - buy_balance  # 매수 시점 잔고와 비교
                profit_rate = risk_manager.get_profit_rate(entry_price, price)
                
                trades.append({
                    'type': 'SELL',
                    'reason': close_reason or 'RULE',
                    'timestamp': timestamp,
                    'price': price,
                    'coin': coin,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'ml_prob': ml_prob,
                    'balance_after': balance
                })
                
                if verbose and len(trades) % 10 == 0:
                    print(f"   [{len(trades):3d}] SELL @ {price:>10,.0f} | {close_reason or 'RULE':>12s} | ML={ml_prob:.2f} | 수익={profit:+,.0f}원")
                
                coin = 0
                entry_price = 0
                buy_balance = 0
        
        # 포지션 미보유 중
        else:
            # 매수 조건: 규칙 기반 신호 + ML 높은 확률
            if buy_signal.iloc[i] and ml_prob > ml_threshold and balance > 10000:
                # 매수
                buy_balance = balance  # 매수 시점 잔고 기록
                coin = risk_manager.calculate_position_size(balance, price)
                entry_price = price
                
                trades.append({
                    'type': 'BUY',
                    'reason': 'HYBRID',
                    'timestamp': timestamp,
                    'price': price,
                    'coin': coin,
                    'profit': 0,
                    'profit_rate': 0,
                    'ml_prob': ml_prob,
                    'balance_after': 0
                })
                
                if verbose and len(trades) % 10 == 0:
                    print(f"   [{len(trades):3d}] BUY  @ {price:>10,.0f} | {'HYBRID':>12s} | ML={ml_prob:.2f}")
                
                balance = 0
    
    # 마지막 포지션 청산
    if coin > 0:
        final_price = df.iloc[-1]['close']
        balance = coin * final_price * (1 - risk_manager.fee_rate)
        profit_rate = risk_manager.get_profit_rate(entry_price, final_price)
        
        trades.append({
            'type': 'SELL',
            'reason': 'FINAL',
            'timestamp': df.index[-1],
            'price': final_price,
            'coin': coin,
            'profit': balance - buy_balance,  # 매수 시점 잔고와 비교
            'profit_rate': profit_rate,
            'ml_prob': 0,
            'balance_after': balance
        })
    
    # 결과 계산
    final_balance = balance
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    win_trades = [t for t in sell_trades if t['profit'] > 0]
    loss_trades = [t for t in sell_trades if t['profit'] <= 0]
    
    win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
    
    result = {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return': total_return,
        'trade_count': len(buy_trades),
        'win_count': len(win_trades),
        'loss_count': len(loss_trades),
        'win_rate': win_rate,
        'trades': trades
    }
    
    if verbose:
        print(f"\n📊 백테스팅 결과:")
        print(f"   - 최종 잔고: {final_balance:,.0f}원")
        print(f"   - 수익률: {total_return:+.2f}%")
        print(f"   - 거래 횟수: {len(buy_trades)}회")
        print(f"   - 승률: {win_rate:.1f}%")
        print(f"   - 승: {len(win_trades)}회, 패: {len(loss_trades)}회")
    
    return result


def run_multi_day_backtest(start_date, end_date, ml_model, rule_engine, risk_manager,
                           test_days=10, ml_threshold=0.65, data_dir="data/daily", timeframe="1s"):
    """
    여러 날짜에 대해 백테스팅 수행
    
    Args:
        start_date, end_date: 날짜 범위 (YYYYMMDD)
        ml_model: ML 모델
        rule_engine: 규칙 엔진
        risk_manager: 리스크 관리자
        test_days: 테스트할 날짜 수
        ml_threshold: ML 임계값
        data_dir: 데이터 디렉토리
        timeframe: 시간봉 ('1s' 또는 '1m')
    
    Returns:
        dict: 통합 결과
    """
    # 날짜 리스트 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    # 랜덤 샘플링
    test_days_list = sorted(random.sample(all_days, min(test_days, len(all_days))))
    
    print("\n" + "=" * 80)
    print(f"🧪 멀티 백테스팅 시작 ({len(test_days_list)}일)")
    print("=" * 80)
    print(f"테스트 날짜: {', '.join(test_days_list[:5])}{'...' if len(test_days_list) > 5 else ''}")
    
    results = []
    
    for i, date_str in enumerate(test_days_list, 1):
        print(f"\n[{i}/{len(test_days_list)}] {date_str} 백테스팅...")
        
        # 데이터 로드
        df = load_daily_csv(date_str, data_dir, timeframe)
        if df is None or len(df) < 100:
            print(f"   ⚠️  데이터 부족, 스킵")
            continue
        
        # 컬럼 매핑
        df = df.rename(columns={
            'date_time_utc': 'timestamp',
            'acc_trade_volume': 'volume'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # 백테스팅
        try:
            result = run_hybrid_backtest(df, ml_model, rule_engine, risk_manager,
                                        ml_threshold=ml_threshold, verbose=False)
            result['date'] = date_str
            results.append(result)
            
            print(f"   ✅ 수익률: {result['total_return']:+.2f}% | 거래: {result['trade_count']}회 | 승률: {result['win_rate']:.1f}%")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            continue
    
    # 통합 결과
    if not results:
        print("\n❌ 백테스팅 결과 없음")
        return None
    
    avg_return = np.mean([r['total_return'] for r in results])
    avg_trades = np.mean([r['trade_count'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    
    print("\n" + "=" * 80)
    print("📊 전체 결과 요약")
    print("=" * 80)
    print(f"평균 수익률: {avg_return:+.2f}%")
    print(f"평균 거래 횟수: {avg_trades:.1f}회/일")
    print(f"평균 승률: {avg_win_rate:.1f}%")
    print("=" * 80)
    
    return {
        'avg_return': avg_return,
        'avg_trades': avg_trades,
        'avg_win_rate': avg_win_rate,
        'results': results
    }


if __name__ == "__main__":
    print("✅ hybrid_backtest.py 모듈 로드 완료")
    print("🔄 사용 가능한 함수:")
    print("  - run_hybrid_backtest() - 단일 백테스팅")
    print("  - run_multi_day_backtest() - 멀티 백테스팅")

