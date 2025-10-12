"""
비트코인 자동매매 파라미터 최적화 도구 (1분봉 + 볼린저 밴드 전략)
Grid Search 방식으로 최적 파라미터 탐색 (CSV 기반 초고속)

전략:
    - 1분봉 데이터 사용 (1초봉 대비 노이즈 감소)
    - 하단 밴드 터치 + 거래량 급증 + RSI 과매도 → 매수
    - 손절/익절 또는 상단 밴드 터치 → 매도

실행 방법:
    python backtest_optimizer.py

장점:
    - CSV 파일 사용으로 API 호출 없음
    - 약 5-10분 소요 (1분봉은 1초봉보다 60배 적은 데이터)
    - 과적합 방지를 위한 2단계 검증
"""

import random
import datetime
import json
import pandas as pd
import sys
from download_data import load_daily_csv
from main import compute_rsi


class TeeOutput:
    """터미널과 파일에 동시 출력 (UTF-8)"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        # UTF-8 BOM 포함하여 Windows 메모장에서도 정상 표시
        self.log = open(filename, 'w', encoding='utf-8-sig')
    
    def write(self, message):
        self.terminal.write(message)
        # 파일에는 안전하게 쓰기 (에러 무시)
        try:
            self.log.write(message)
        except UnicodeEncodeError:
            self.log.write(message.encode('utf-8', errors='replace').decode('utf-8'))
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def run_csv_backtest(date_str, bb_tolerance=1.005, volume_multiplier=2.5, rsi_threshold=30,
                     stop_loss_pct=0.5, take_profit_pct=0.8, bb_upper_sell=True,
                     data_dir="data/daily_1m", progress_prefix=""):
    """
    CSV 파일 기반 단일 날짜 백테스팅 (볼린저 밴드 + 거래량 전략)
    
    Args:
        date_str: 날짜 (YYYYMMDD)
        bb_tolerance: 하단 밴드 터치 허용 범위 (1.001 = 0.1% 위까지 허용)
        volume_multiplier: 거래량 급증 기준 (평균의 N배)
        rsi_threshold: RSI 과매도 기준 (N 이하)
        stop_loss_pct: 손절 %
        take_profit_pct: 익절 %
        bb_upper_sell: 상단 밴드 터치 시 매도 여부
        data_dir: CSV 파일 디렉토리
        progress_prefix: 진행 상황 앞에 표시할 문자열
    
    Returns:
        dict: 백테스팅 결과 (없으면 None)
    """
    
    # CSV 파일 로드
    df_full = load_daily_csv(date_str, data_dir)
    
    if df_full is None or len(df_full) == 0:
        return None
    
    # 컬럼 이름 매핑 (Upbit CSV 형식 → 백테스팅 형식)
    df_full = df_full.rename(columns={
        'date_time_utc': 'timestamp',
        'acc_trade_volume': 'volume'
    })
    
    # open, high, low, close는 이미 있음
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    df_full = df_full.set_index('timestamp')
    df_full = df_full.sort_index()
    
    # 초기 설정 (1분봉)
    window_size = 60  # 1시간 (60분)
    
    if len(df_full) < window_size + 20:
        return None
    
    initial_balance = 1_000_000
    balance = initial_balance
    coin_holding = 0
    buy_price = 0
    buy_index = -1
    fee_rate = 0.0005
    
    trades = []
    trade_count = 0
    win_count = 0
    total_profit = 0
    
    # 🚀 최적화: 전체 데이터에 대해 지표를 한 번만 계산 (슬라이딩 윈도우 방식 유지)
    df_full['rsi'] = compute_rsi(df_full['close'], 14)
    df_full['volume_ma'] = df_full['volume'].rolling(window=20).mean()
    
    # 볼린저 밴드 계산
    df_full['bb_middle'] = df_full['close'].rolling(window=20).mean()
    bb_std = df_full['close'].rolling(window=20).std()
    df_full['bb_upper'] = df_full['bb_middle'] + (bb_std * 2)
    df_full['bb_lower'] = df_full['bb_middle'] - (bb_std * 2)
    
    # 슬라이딩 윈도우 방식 (지표는 이미 계산됨, 값만 참조)
    total_iterations = len(df_full) - window_size
    print_interval = max(1, total_iterations // 20)  # 5%마다 출력
    
    for i in range(window_size, len(df_full)):
        # 진행 상황 표시 (5%마다)
        if progress_prefix and (i - window_size) % print_interval == 0:
            progress = int(((i - window_size) / total_iterations) * 100)
            print(f"\r{progress_prefix} | 윈도우:{progress}%", end="", flush=True)
        
        # 🚀 최적화: 윈도우 전체가 아닌 현재 시점 값만 참조 (수십배 빠름)
        row = df_full.iloc[i]
        prev_row = df_full.iloc[i-1]
        price = row['close']
        volume = row['volume']
        timestamp = row.name
        
        # 보유 중: 손절/익절/밴드 터치 체크
        if coin_holding > 0:
            profit_rate = ((price - buy_price) / buy_price) * 100
            current_value = coin_holding * price * (1 - fee_rate)
            
            sell_reason = None
            
            if profit_rate <= -stop_loss_pct:
                sell_reason = "손절"
            elif profit_rate >= take_profit_pct:
                sell_reason = "익절"
            elif bb_upper_sell and pd.notna(row['bb_upper']) and price >= row['bb_upper'] * 0.999:
                sell_reason = "상단밴드"
            
            if sell_reason:
                balance = current_value
                trade_profit = current_value - (buy_price * coin_holding * (1 + fee_rate))
                total_profit += trade_profit
                trade_count += 1
                
                if trade_profit > 0:
                    win_count += 1
                
                trades.append({
                    'type': sell_reason,
                    'profit_rate': profit_rate,
                    'profit': trade_profit
                })
                
                coin_holding = 0
                buy_price = 0
                buy_index = -1
        
        # 미보유 중: 매수 시그널 체크 (볼린저 밴드 + 거래량 + RSI 전략)
        else:
            if pd.notna(row['rsi']) and pd.notna(row['bb_lower']) and pd.notna(row['volume_ma']):
                # 1. 하단 밴드 터치 (과매도)
                bb_touch = price <= row['bb_lower'] * bb_tolerance
                
                # 2. 거래량 급증 (큰 움직임 예상)
                volume_surge = volume > row['volume_ma'] * volume_multiplier
                
                # 3. RSI 과매도 (추가 확인)
                rsi_oversold = row['rsi'] < rsi_threshold
                
                buy_signal = bb_touch and volume_surge and rsi_oversold
                
                if buy_signal and balance > 10000:
                    coin_holding = (balance * (1 - fee_rate)) / price
                    buy_price = price
                    buy_index = i
                    balance = 0
    
    # 마지막 보유 중이면 강제 청산
    if coin_holding > 0:
        final_price = df_full.iloc[-1]['close']
        balance = coin_holding * final_price * (1 - fee_rate)
        trade_profit = balance - (buy_price * coin_holding * (1 + fee_rate))
        total_profit += trade_profit
        trade_count += 1
        if trade_profit > 0:
            win_count += 1
    
    # 진행률 100% 표시 (prefix가 있는 경우만)
    if progress_prefix:
        print(f"\r{progress_prefix} | 윈도우:100%", end="", flush=True)
    
    # 결과 반환
    final_balance = balance
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    
    profit_trades = len([t for t in trades if t['type'] == '익절'])
    loss_trades = len([t for t in trades if t['type'] == '손절'])
    bb_trades = len([t for t in trades if t['type'] == '상단밴드'])
    
    return {
        'date': date_str,
        'total_return': total_return,
        'trade_count': trade_count,
        'win_count': win_count,
        'win_rate': win_rate,
        'profit_trades': profit_trades,
        'loss_trades': loss_trades,
        'bb_trades': bb_trades
    }


def grid_search_parameters():
    """
    Grid Search로 최적 파라미터 탐색 (CSV 기반)
    
    단계 1: 학습 기간 (10일)에서 최적 파라미터 찾기
    단계 2: 다른 10일로 검증하여 과적합 방지
    """
    
    print("=" * 80)
    print("🔍 파라미터 최적화 - 1분봉 + 볼린저 밴드 전략 (CSV 기반)")
    print("=" * 80)
    print("\n📋 탐색 범위:")
    print("   - 하단밴드 허용: [1.001 (0.1%), 1.005 (0.5%), 1.01 (1%)]")
    print("   - 거래량 배수: [2.0, 2.5, 3.0, 3.5]")
    print("   - RSI 과매도: [25, 30, 35]")
    print("   - 손절: [0.3, 0.5, 0.8]")
    print("   - 익절: [0.5, 0.8, 1.0]")
    print(f"\n🎯 총 조합: 3 × 4 × 3 × 3 × 3 = 324개")
    print(f"⚡ 1분봉 CSV 사용 (노이즈 감소, 빠른 처리)")
    print(f"🎯 전략: 하단밴드 터치 + 거래량 급증 + RSI 과매도")
    
    # 파라미터 범위 정의 (새로운 전략)
    bb_tolerances = [
        1.001,  # 하단밴드의 0.1% 위까지 허용 (매우 엄격)
        1.005,  # 하단밴드의 0.5% 위까지 허용 (보통)
        1.01,   # 하단밴드의 1% 위까지 허용 (느슨)
    ]
    
    volume_multipliers = [2.0, 2.5, 3.0, 3.5]  # 거래량 급증 기준 (기존보다 높게)
    rsi_thresholds = [25, 30, 35]  # RSI 과매도 기준
    stop_losses = [0.3, 0.5, 0.8]  # 타이트한 손절
    take_profits = [0.5, 0.8, 1.0]  # 익절
    
    # 1단계: 학습 기간 (CSV 파일이 있는 날짜 중 랜덤 10일)
    print("\n" + "=" * 80)
    print("📚 1단계: 학습 기간 (CSV에서 랜덤 10일)")
    print("=" * 80)
    
    # 2025년 1월 ~ 5월 중 랜덤 10일 선택
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 5, 30)
    all_days = []
    current = start_date
    while current <= end_date:
        all_days.append(current.strftime("%Y%m%d"))
        current += datetime.timedelta(days=1)
    
    train_days = sorted(random.sample(all_days, 10))
    print(f"학습 날짜: {', '.join(train_days[:5])}... (10일)")
    
    results = []
    total_combinations = len(bb_tolerances) * len(volume_multipliers) * len(rsi_thresholds) * len(stop_losses) * len(take_profits)
    current = 0
    
    print(f"\n⏳ 총 {total_combinations}개 조합 테스트 중... (예상: 5-10분, 1분봉은 빠름!)\n")
    
    for bb_tol in bb_tolerances:
        for vol_mult in volume_multipliers:
            for rsi_th in rsi_thresholds:
                for sl in stop_losses:
                    for tp in take_profits:
                        current += 1
                        
                        try:
                            # 백테스팅 실행
                            day_results = []
                            for idx, date_str in enumerate(train_days, 1):
                                # 진행 상황 prefix 생성
                                bb_pct = (bb_tol - 1) * 100
                                prefix = (f"[{current}/{total_combinations}] 밴드:{bb_pct:.1f}%, "
                                         f"거래량:{vol_mult}배, RSI<{rsi_th}, 손절:{sl}%, 익절:{tp}% | 날짜:{idx}/{len(train_days)}일")
                                
                                result = run_csv_backtest(
                                    date_str=date_str,
                                    bb_tolerance=bb_tol,
                                    volume_multiplier=vol_mult,
                                    rsi_threshold=rsi_th,
                                    stop_loss_pct=sl,
                                    take_profit_pct=tp,
                                    bb_upper_sell=True,
                                    progress_prefix=prefix
                                )
                                
                                if result:
                                    day_results.append(result)
                            
                            if day_results:
                                avg_return = sum(r['total_return'] for r in day_results) / len(day_results)
                                avg_win_rate = sum(r['win_rate'] for r in day_results) / len(day_results)
                                total_trades = sum(r['trade_count'] for r in day_results)
                                
                                results.append({
                                    'bb_tolerance': bb_tol,
                                    'volume_multiplier': vol_mult,
                                    'rsi_threshold': rsi_th,
                                    'stop_loss': sl,
                                    'take_profit': tp,
                                    'avg_return': avg_return,
                                    'avg_win_rate': avg_win_rate,
                                    'total_trades': total_trades,
                                    'avg_trades_per_day': total_trades / len(day_results)
                                })
                                
                                # 완료 시 한 줄에 최종 결과 표시
                                print(f"\r[{current}/{total_combinations}] 밴드:{bb_pct:.1f}%, "
                                      f"거래량:{vol_mult}배, RSI<{rsi_th}, 손절:{sl}%, 익절:{tp}% "
                                      f"→ {avg_return:+.2f}% (승률:{avg_win_rate:.1f}%, 일거래:{total_trades/len(day_results):.1f}회)")
                            else:
                                print(f"\r[{current}/{total_combinations}] 밴드:{bb_pct:.1f}%, "
                                      f"거래량:{vol_mult}배, RSI<{rsi_th}, 손절:{sl}%, 익절:{tp}% → ⚠️ 데이터 부족" + " " * 20)
                        
                        except Exception as e:
                            print(f"\r[{current}/{total_combinations}] 밴드:{bb_pct:.1f}%, "
                                  f"거래량:{vol_mult}배, RSI<{rsi_th}, 손절:{sl}%, 익절:{tp}% → ❌ 오류: {str(e)[:30]}" + " " * 20)
                            continue
    
    if not results:
        print("\n❌ 모든 테스트 실패. 프로그램 종료.")
        return None
    
    # 결과 정렬 (평균 수익률 기준)
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    # 상위 5개 출력
    print("\n" + "=" * 80)
    print("🏆 학습 기간 상위 5개 조합")
    print("=" * 80)
    print(f"{'순위':<4} {'밴드':<6} {'거래량':<6} {'RSI<':<5} {'손절':<5} {'익절':<5} {'수익률':<8} {'승률':<6} {'일거래':<6}")
    print("-" * 80)
    
    for i, r in enumerate(results[:5], 1):
        bb_pct = (r['bb_tolerance'] - 1) * 100
        print(f"{i:<4} {bb_pct:<5.1f}% {r['volume_multiplier']:<6} "
              f"{r['rsi_threshold']:<5} {r['stop_loss']:<5} {r['take_profit']:<5} "
              f"{r['avg_return']:>+7.2f}% {r['avg_win_rate']:>5.1f}% {r['avg_trades_per_day']:>5.1f}")
    
    # 2단계: 상위 3개를 다른 10일로 검증
    print("\n" + "=" * 80)
    print("🔬 2단계: 검증 기간 (다른 10일)")
    print("=" * 80)
    
    # 학습 기간과 겹치지 않는 10일 선택
    available_days = [d for d in all_days if d not in train_days]
    validation_days = sorted(random.sample(available_days, 10))
    print(f"검증 날짜: {', '.join(validation_days[:5])}... (10일)")
    
    top_3 = results[:3]
    validation_results = []
    
    for i, params in enumerate(top_3, 1):
        try:
            day_results = []
            for idx, date_str in enumerate(validation_days, 1):
                # 검증 진행 상황 prefix 생성
                bb_pct = (params['bb_tolerance'] - 1) * 100
                prefix = (f"[{i}/3] 밴드:{bb_pct:.1f}%, 거래량:{params['volume_multiplier']}배, "
                         f"RSI<{params['rsi_threshold']}, 손절:{params['stop_loss']}%, 익절:{params['take_profit']}% | 검증:{idx}/{len(validation_days)}일")
                
                result = run_csv_backtest(
                    date_str=date_str,
                    bb_tolerance=params['bb_tolerance'],
                    volume_multiplier=params['volume_multiplier'],
                    rsi_threshold=params['rsi_threshold'],
                    stop_loss_pct=params['stop_loss'],
                    take_profit_pct=params['take_profit'],
                    bb_upper_sell=True,
                    progress_prefix=prefix
                )
                
                if result:
                    day_results.append(result)
            
            if day_results:
                val_avg_return = sum(r['total_return'] for r in day_results) / len(day_results)
                val_avg_win_rate = sum(r['win_rate'] for r in day_results) / len(day_results)
                val_total_trades = sum(r['trade_count'] for r in day_results)
                
                validation_results.append({
                    **params,
                    'train_return': params['avg_return'],
                    'val_return': val_avg_return,
                    'val_win_rate': val_avg_win_rate,
                    'val_avg_trades': val_total_trades / len(day_results),
                    'performance_drop': params['avg_return'] - val_avg_return
                })
                
                # 완료 시 한 줄에 최종 결과 표시
                print(f"\r[{i}/3] 밴드:{bb_pct:.1f}%, 거래량:{params['volume_multiplier']}배, "
                      f"RSI<{params['rsi_threshold']}, 손절:{params['stop_loss']}%, 익절:{params['take_profit']}% "
                      f"→ {val_avg_return:+.2f}% (학습:{params['avg_return']:+.2f}%, 차이:{params['avg_return'] - val_avg_return:+.2f}%p)")
            else:
                print(f"\r[{i}/3] 밴드:{bb_pct:.1f}%, 거래량:{params['volume_multiplier']}배, "
                      f"RSI<{params['rsi_threshold']}, 손절:{params['stop_loss']}%, 익절:{params['take_profit']}% → ⚠️ 검증 실패" + " " * 20)
        
        except Exception as e:
            print(f"\r[{i}/3] 밴드:{bb_pct:.1f}%, 거래량:{params['volume_multiplier']}배, "
                  f"RSI<{params['rsi_threshold']}, 손절:{params['stop_loss']}%, 익절:{params['take_profit']}% → ❌ 오류: {str(e)[:30]}" + " " * 20)
            continue
    
    if not validation_results:
        print("\n❌ 모든 검증 실패.")
        return None
    
    # 최종 결과 출력
    print("\n" + "=" * 80)
    print("🎯 최종 결과 - 과적합 방지 검증")
    print("=" * 80)
    print(f"{'순위':<4} {'밴드':<6} {'거래량':<6} {'RSI<':<5} {'손절':<5} {'익절':<5} "
          f"{'학습수익률':<10} {'검증수익률':<10} {'차이':<7}")
    print("-" * 80)
    
    # 검증 수익률 기준으로 재정렬
    validation_results.sort(key=lambda x: x['val_return'], reverse=True)
    
    for i, r in enumerate(validation_results, 1):
        bb_pct = (r['bb_tolerance'] - 1) * 100
        print(f"{i:<4} {bb_pct:<5.1f}% {r['volume_multiplier']:<6} "
              f"{r['rsi_threshold']:<5} {r['stop_loss']:<5} {r['take_profit']:<5} "
              f"{r['train_return']:>+9.2f}% {r['val_return']:>+9.2f}% {r['performance_drop']:>+6.2f}%p")
    
    # 최적 파라미터 선정
    best = validation_results[0]
    bb_pct_best = (best['bb_tolerance'] - 1) * 100
    
    print("\n" + "=" * 80)
    print("🏆 최적 파라미터 (검증 수익률 기준)")
    print("=" * 80)
    print(f"하단밴드 허용: {bb_pct_best:.1f}% (BB_Lower × {best['bb_tolerance']})")
    print(f"거래량 배수: {best['volume_multiplier']}배")
    print(f"RSI 과매도: < {best['rsi_threshold']}")
    print(f"손절: {best['stop_loss']}%")
    print(f"익절: {best['take_profit']}%")
    print(f"\n학습 수익률: {best['train_return']:+.2f}%")
    print(f"검증 수익률: {best['val_return']:+.2f}%")
    print(f"검증 승률: {best['val_win_rate']:.1f}%")
    print(f"일평균 거래: {best['val_avg_trades']:.1f}회")
    
    # 과적합 경고
    if best['performance_drop'] > 2.0:
        print(f"\n⚠️ 과적합 경고: 학습-검증 차이가 {best['performance_drop']:+.2f}%p로 큽니다!")
        print("   → 더 보수적인 파라미터 선택을 권장합니다.")
    elif best['performance_drop'] > 1.0:
        print(f"\n⚠️ 주의: 학습-검증 차이가 {best['performance_drop']:+.2f}%p입니다.")
        print("   → 실거래 전 추가 검증을 권장합니다.")
    else:
        print(f"\n✅ 양호: 학습-검증 차이가 {best['performance_drop']:+.2f}%p로 안정적입니다!")
    
    # 결과 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"optimization_result_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best,
            'all_validation_results': validation_results,
            'train_days': train_days,
            'validation_days': validation_days,
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {result_file}")
    
    print("\n" + "=" * 80)
    print("📝 다음 단계")
    print("=" * 80)
    print("1. ✅ 최적 파라미터 발견 (볼린저 밴드 전략)")
    print("2. 📡 backtest.py와 main.py에 새 전략 적용 필요")
    print(f"   - 하단밴드: {bb_pct_best:.1f}%, 거래량: {best['volume_multiplier']}배, RSI<{best['rsi_threshold']}")
    print(f"   - 손절: {best['stop_loss']}%, 익절: {best['take_profit']}%")
    print("3. 🎯 backtest.py로 최신 API 데이터 재검증")
    print("4. 💰 소액 실거래 테스트")
    print("=" * 80)
    
    return validation_results


if __name__ == "__main__":
    # 출력을 터미널과 파일에 동시 저장
    log_filename = "backtest_optimizer_log.txt"
    tee = TeeOutput(log_filename)
    sys.stdout = tee
    
    try:
        print("\n🚀 비트코인 자동매매 파라미터 최적화 (1분봉 + 볼린저 밴드)\n")
        print("✅ 장점:")
        print("   - 1분봉 사용 (1초봉 대비 노이즈 60배 감소)")
        print("   - CSV 파일 사용 (API 호출 없음)")
        print("   - 5-10분 소요 (매우 빠름!)")
        print("   - 과적합 방지 2단계 검증")
        print("\n🎯 전략:")
        print("   - 매수: 하단밴드 터치 + 거래량 급증 + RSI 과매도")
        print("   - 매도: 손절/익절 또는 상단밴드 터치")
        print(f"\n📄 로그 파일: {log_filename}")
        print("\n📦 시작...\n")
        
        results = grid_search_parameters()
        
        if results:
            print("\n✅ 최적화 완료!")
        else:
            print("\n❌ 최적화 실패")
    
    finally:
        # 출력 복원 및 파일 닫기
        tee.close()
