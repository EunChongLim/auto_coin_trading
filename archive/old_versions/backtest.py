import pyupbit
import pandas as pd
import datetime
import requests
import time as time_module
import random
from main import compute_rsi

def get_second_ohlcv(ticker, total_count=3600, to_date=None):
    """
    1초봉 데이터 조회 (Upbit API 직접 호출)
    API 제한(200개)을 우회하여 여러 번 호출
    
    Args:
        ticker: 마켓 코드 (예: KRW-BTC)
        total_count: 조회할 총 캔들 개수 (예: 3600 = 1시간)
        to_date: 조회 시작 시각 (선택, datetime 객체)
    
    Returns:
        DataFrame: OHLCV 데이터 (1초봉)
    """
    url = "https://api.upbit.com/v1/candles/seconds"
    headers = {"accept": "application/json"}
    
    all_data = []
    
    # to_date가 있으면 해당 시각부터 과거로 조회 (한국 시간대 포함)
    to_param = to_date.strftime("%Y-%m-%dT%H:%M:%S+09:00") if to_date else None
    
    # 200개씩 여러 번 호출
    calls_needed = (total_count + 199) // 200  # 올림
    
    try:
        for i in range(calls_needed):
            params = {
                "market": ticker,
                "count": min(200, total_count - len(all_data))
            }
            
            if to_param:
                params["to"] = to_param
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # 데이터가 없거나 200개 미만이면 더 이상 과거 데이터 없음
            if not data:
                break
            
            all_data.extend(data)
            
            # 진행 상황 표시
            if (i + 1) % 18 == 0:
                hours = (i + 1) // 18
                print(f".", end="", flush=True)
                if hours % 6 == 0:
                    print(f" {hours}h", end="", flush=True)
            
            # 다음 호출을 위한 to 파라미터 업데이트
            if len(data) == 200:
                # 가장 오래된 캔들의 시각 + 1초 (gap 방지, to는 해당 시각 미만 반환)
                # 예: to=23:55:31 → 23:55:30 이하만 반환, 23:55:31 제외
                # 따라서 to=23:55:32로 해야 23:55:31 포함
                oldest_time = pd.to_datetime(data[-1]['candle_date_time_kst'])
                to_param = (oldest_time + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
            else:
                # 200개 미만이면 더 이상 과거 데이터 없음
                break
            
            # API Rate Limit 대응 (초당 10회, 안전하게 초당 5회로 제한)
            if i < calls_needed - 1:
                time_module.sleep(0.3)  # 0.3초 대기 (안전)
            
            if len(all_data) >= total_count:
                break
        
        if not all_data:
            raise Exception("❌ API에서 데이터를 전혀 받지 못했습니다!")
        
        # DataFrame 변환
        print(f"\n✅ API 호출 완료: {len(all_data)}개 받음")
        df = pd.DataFrame(all_data)
        
        # 필요한 컬럼만 선택 및 이름 변경
        df = df[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 중복 제거
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        df = df.set_index('timestamp')
        df = df.sort_index()  # 시간순 정렬 (과거→최신)
        
        print(f"   최종: {len(df)}개 ({df.index[0].strftime('%H:%M:%S')} ~ {df.index[-1].strftime('%H:%M:%S')})")
        
        return df
        
    except Exception as e:
        print(f"\n❌ 초봉 조회 오류: {e}")
        print(f"⚠️  API Rate Limit 또는 네트워크 오류로 중단됩니다.")
        raise  # 오류 발생 시 즉시 종료

def run_single_backtest(ticker, days_ago, stop_loss_pct, take_profit_pct, fee_rate, rsi_sell_threshold, 
                       rsi_condition_profit=1.0, use_seconds=True, 
                       rsi_low=30, rsi_high=60, volume_multiplier=1.1):
    """단일 백테스트 실행 (슬라이딩 윈도우 방식 - main.py와 동일 로직)
    
    Args:
        rsi_low: RSI 하한값 (기본: 30)
        rsi_high: RSI 상한값 (기본: 60)
        volume_multiplier: 거래량 배수 (기본: 1.1)
    """
    
    # 과거 데이터 조회 (전체 가져오기)
    if use_seconds:
        # 1초봉: 86400개 = 24시간 데이터
        # days_ago일 전 날짜의 23:59:59부터 조회 (하루 전체)
        target_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).replace(hour=23, minute=59, second=59)
        print(f"(1초봉 24시간, 3600개 윈도우)", end=" ")
        df_full = get_second_ohlcv(ticker, total_count=86400, to_date=target_date)
    else:
        # 1분봉: 하루 = 1440개
        end_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        df_full = pyupbit.get_ohlcv(ticker, interval="minute1", count=1440, to=end_date.strftime("%Y%m%d%H%M%S"))
    
    if df_full is None or len(df_full) == 0:
        return None
    
    # 초기 설정
    initial_balance = 1_000_000
    balance = initial_balance
    coin_holding = 0
    buy_price = 0
    buy_index = -1
    
    trades = []
    trade_count = 0
    win_count = 0
    total_profit = 0
    
    # 슬라이딩 윈도우 방식 시뮬레이션 (main.py와 동일)
    window_size = 3600  # main.py와 동일: 3600개 = 1시간 (1초봉 기준)
    
    # 데이터가 윈도우 크기보다 작으면 스킵
    if len(df_full) < window_size + 100:
        print(f"⚠️ 데이터 부족 (필요: {window_size}개 이상, 실제: {len(df_full)}개)")
        return None
    
    # 🚀 최적화: 전체 데이터에 대해 지표를 한 번만 계산 (검증 완료)
    df_full['rsi'] = compute_rsi(df_full['close'], 14)
    df_full['ma_fast'] = df_full['close'].rolling(window=5).mean()
    df_full['ma_slow'] = df_full['close'].rolling(window=20).mean()
    df_full['volume_ma'] = df_full['volume'].rolling(window=20).mean()
    df_full['bb_middle'] = df_full['close'].rolling(window=20).mean()
    bb_std = df_full['close'].rolling(window=20).std()
    df_full['bb_upper'] = df_full['bb_middle'] + (bb_std * 2)
    df_full['bb_lower'] = df_full['bb_middle'] - (bb_std * 2)
    
    # 진행 상황 표시
    total_iterations = len(df_full) - window_size
    
    for i in range(window_size, len(df_full)):
        # 진행 상황 표시 (6시간마다 - 21600초 = 6시간)
        if (i - window_size) % 21600 == 0:
            hours = (i - window_size) // 3600
            print(f" {hours}h", end="", flush=True)
        
        # 🚀 최적화: 현재 시점 값만 참조 (계산 없음, 수십배 빠름)
        row = df_full.iloc[i]
        prev_row = df_full.iloc[i-1]
        price = row['close']
        volume = row['volume']
        timestamp = row.name
        
        # 보유 중: 손절/익절 체크
        if coin_holding > 0:
            profit_rate = ((price - buy_price) / buy_price) * 100
            current_value = coin_holding * price * (1 - fee_rate)
            
            sell_reason = None
            
            # 손절
            if profit_rate <= -stop_loss_pct:
                sell_reason = "손절"
            # 익절
            elif profit_rate >= take_profit_pct:
                sell_reason = "익절"
            # RSI 과매수 (조건부: 수익 1% 이상일 때만)
            elif row['rsi'] > rsi_sell_threshold and profit_rate > rsi_condition_profit:
                sell_reason = "RSI매도"
            
            if sell_reason:
                balance = current_value
                trade_profit = current_value - (buy_price * coin_holding * (1 + fee_rate))
                total_profit += trade_profit
                trade_count += 1
                
                if trade_profit > 0:
                    win_count += 1
                
                hold_time = i - buy_index
                trades.append({
                    'type': sell_reason,
                    'buy_time': df_full.index[buy_index],
                    'sell_time': timestamp,
                    'buy_price': buy_price,
                    'sell_price': price,
                    'profit_rate': profit_rate,
                    'profit': trade_profit,
                    'hold_minutes': hold_time
                })
                
                coin_holding = 0
                buy_price = 0
                buy_index = -1
        
        # 미보유 중: 매수 시그널 체크 (main.py와 동일)
        else:
            # 모든 지표가 유효한지 확인
            if pd.notna(row['rsi']) and pd.notna(row['ma_fast']) and pd.notna(row['volume_ma']):
                rsi_oversold = rsi_low < row['rsi'] < rsi_high  # 파라미터화된 RSI 범위
                rsi_rising = row['rsi'] > prev_row['rsi']
                volume_surge = volume > row['volume_ma'] * volume_multiplier  # 파라미터화된 거래량 배수
                price_above_ma = price > row['ma_fast']
                bullish_candle = row['close'] > row['open']
                
                buy_signal = (
                    rsi_oversold and 
                    rsi_rising and 
                    volume_surge and 
                    price_above_ma and 
                    bullish_candle
                )
                
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
    
    # 결과 반환
    final_balance = balance
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    
    profit_trades = len([t for t in trades if t['type'] == '익절'])
    loss_trades = len([t for t in trades if t['type'] == '손절'])
    rsi_trades = len([t for t in trades if t['type'] == 'RSI매도'])
    
    # 날짜 표시
    end_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    if use_seconds:
        date_str = f"{end_date.strftime('%Y-%m-%d')} (1초봉)"
    else:
        date_str = end_date.strftime('%Y-%m-%d')
    
    return {
        'date': date_str,
        'total_return': total_return,
        'win_rate': win_rate,
        'trade_count': trade_count,
        'win_count': win_count,
        'profit_trades': profit_trades,
        'loss_trades': loss_trades,
        'rsi_trades': rsi_trades,
        'trades': trades
    }

def run_multi_backtest(days_list=[1, 2, 3, 7, 14, 30], rsi_threshold=80, take_profit=0.8, rsi_condition_profit=0.5, 
                       use_seconds=False, rsi_low=30, rsi_high=60, volume_multiplier=1.1):
    """여러 날짜 백테스팅
    
    Args:
        rsi_low: RSI 하한값 (기본: 30)
        rsi_high: RSI 상한값 (기본: 60)
        volume_multiplier: 거래량 배수 (기본: 1.1)
    """
    
    print("=" * 80)
    print("⚡ 초단타 스캘핑 백테스팅 - 알고리즘 종합 검증 v3.1 (슬라이딩 윈도우)")
    print("=" * 80)
    
    if use_seconds:
        print(f"\n⚡ 1초봉 테스트 (각 날짜별 24시간 데이터, 3600개 슬라이딩 윈도우 = 1시간)")
        print(f"📊 손절: -1.5% | 익절: +{take_profit}% | RSI 매도: >{rsi_threshold} (수익 {rsi_condition_profit}% 이상)")
        print(f"📈 매수 조건: RSI {rsi_low}~{rsi_high}, 거래량 {volume_multiplier}배 이상 (BTC 적합형)")
        print(f"⏳ 1초봉 백테스팅 진행 중... (BTC 최적화 알고리즘)")
        print(f"   (각 날짜당 432회 API 호출, 약 2분씩 소요)\n")
        
        results = []
        
        for days_ago in days_list:
            print(f"  📍 {days_ago}일 전 데이터 분석 중... ", end="")
            result = run_single_backtest(
                ticker="KRW-BTC",
                days_ago=days_ago,
                stop_loss_pct=1.5,
                take_profit_pct=take_profit,
                fee_rate=0.0005,
                rsi_sell_threshold=rsi_threshold,
                rsi_condition_profit=rsi_condition_profit,
                use_seconds=True,
                rsi_low=rsi_low,
                rsi_high=rsi_high,
                volume_multiplier=volume_multiplier
            )
            
            if result:
                results.append(result)
                status = "✅" if result['total_return'] > 0 else "❌"
                print(f"{status} 수익률: {result['total_return']:+.2f}%")
            else:
                print("❌ 데이터 없음")
    else:
        print(f"\n📅 테스트 기간: {len(days_list)}일")
        print(f"📊 손절: -1.5% | 익절: +{take_profit}% | RSI 매도: >{rsi_threshold} (수익 {rsi_condition_profit}% 이상)")
        print(f"📈 매수 조건: RSI {rsi_low}~{rsi_high}, 거래량 {volume_multiplier}배 이상")
        print(f"🎯 테스트 날짜: {', '.join([f'{d}일 전' for d in days_list])}")
        print("\n⏳ 백테스팅 진행 중...\n")
        
        results = []
        
        for days_ago in days_list:
            print(f"  📍 {days_ago}일 전 데이터 분석 중...", end=" ")
            result = run_single_backtest(
                ticker="KRW-BTC",
                days_ago=days_ago,
                stop_loss_pct=1.5,  # 2.0 → 1.5 (손익비 1:1)
                take_profit_pct=take_profit,
                fee_rate=0.0005,
                rsi_sell_threshold=rsi_threshold,
                rsi_condition_profit=rsi_condition_profit,
                use_seconds=False,
                rsi_low=rsi_low,
                rsi_high=rsi_high,
                volume_multiplier=volume_multiplier
            )
            
            if result:
                results.append(result)
                status = "✅" if result['total_return'] > 0 else "❌"
                print(f"{status} 수익률: {result['total_return']:+.2f}%")
            else:
                print("❌ 데이터 없음")
    
    if not results:
        print("\n❌ 테스트 결과가 없습니다.")
        return None
    
    # 종합 통계
    print("\n" + "=" * 80)
    print("📊 종합 분석 결과")
    print("=" * 80)
    
    # 전체 평균
    avg_return = sum(r['total_return'] for r in results) / len(results)
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
    total_trades = sum(r['trade_count'] for r in results)
    positive_days = len([r for r in results if r['total_return'] > 0])
    
    total_profit_trades = sum(r['profit_trades'] for r in results)
    total_loss_trades = sum(r['loss_trades'] for r in results)
    total_rsi_trades = sum(r['rsi_trades'] for r in results)
    
    print(f"\n💰 수익 현황")
    print(f"   평균 수익률: {avg_return:+.2f}%")
    print(f"   수익일: {positive_days}일 / {len(results)}일 ({positive_days/len(results)*100:.1f}%)")
    print(f"   최고 수익: {max(r['total_return'] for r in results):+.2f}%")
    print(f"   최저 수익: {min(r['total_return'] for r in results):+.2f}%")
    
    print(f"\n📈 거래 통계")
    print(f"   평균 승률: {avg_win_rate:.1f}%")
    print(f"   총 거래: {total_trades}회")
    print(f"   일평균 거래: {total_trades/len(results):.1f}회")
    
    print(f"\n📋 거래 유형 분석")
    print(f"   익절: {total_profit_trades}회 ({total_profit_trades/total_trades*100:.1f}%)")
    print(f"   손절: {total_loss_trades}회 ({total_loss_trades/total_trades*100:.1f}%)")
    print(f"   RSI매도: {total_rsi_trades}회 ({total_rsi_trades/total_trades*100:.1f}%)")
    
    # 날짜별 상세
    print("\n" + "=" * 80)
    print("📅 날짜별 상세 결과")
    print("=" * 80)
    print(f"{'날짜':<12} {'수익률':>8} {'승률':>6} {'거래':>4} {'익절':>4} {'손절':>4} {'RSI':>4}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['date']:<12} {r['total_return']:>+7.2f}% {r['win_rate']:>5.1f}% "
              f"{r['trade_count']:>4} {r['profit_trades']:>4} {r['loss_trades']:>4} {r['rsi_trades']:>4}")
    
    print("=" * 80)
    
    # 평가 및 권장사항
    print("\n🎯 알고리즘 평가")
    print("-" * 80)
    
    if avg_return > 5:
        print("✅ 우수: 평균 수익률이 매우 높습니다!")
    elif avg_return > 2:
        print("✅ 양호: 안정적인 수익을 내고 있습니다.")
    elif avg_return > 0:
        print("⚠️ 보통: 수익은 나지만 개선이 필요합니다.")
    else:
        print("❌ 불량: 손실이 발생하고 있습니다. 알고리즘 수정 필요!")
    
    print("\n💡 개선 권장사항:")
    
    # RSI 매도 비율 분석
    rsi_ratio = total_rsi_trades / total_trades * 100 if total_trades > 0 else 0
    if rsi_ratio > 70:
        print(f"   ⚠️ RSI 매도가 {rsi_ratio:.0f}%로 너무 높습니다!")
        print(f"   → RSI 기준을 80에서 85로 상향 조정 권장")
    elif rsi_ratio > 50:
        print(f"   ⚠️ RSI 매도가 {rsi_ratio:.0f}%로 높은 편입니다.")
        print(f"   → RSI 기준 상향 또는 수익 중일 때만 RSI 매도 고려")
    
    # 승률 분석
    if avg_win_rate < 55:
        print(f"   ⚠️ 승률 {avg_win_rate:.1f}%로 낮습니다.")
        print(f"   → 매수 조건 강화 또는 손절폭 축소 고려")
    
    # 거래 빈도 분석
    avg_trades = total_trades / len(results)
    if avg_trades < 5:
        print(f"   ⚠️ 일평균 거래 {avg_trades:.1f}회로 적습니다.")
        print(f"   → 매수 조건 완화 (거래량 1.2배 → 1.1배 등)")
    elif avg_trades > 30:
        print(f"   ⚠️ 일평균 거래 {avg_trades:.1f}회로 많습니다.")
        print(f"   → 과도한 매매로 수수료 부담, 조건 강화 권장")
    
    print("\n" + "=" * 80)
    
    return results

if __name__ == "__main__":
    print("\n🎯 비트코인 스캘핑 백테스터 v3.1 - 슬라이딩 윈도우 전략\n")
    
    # 초단타 스캘핑 알고리즘 자동 실행
    days_list = sorted(random.sample(range(1, 31), 3))  # 1~30일 중 랜덤 3일
    rsi_threshold = 80
    take_profit = 0.8  # 초단타: 0.8% 익절
    rsi_condition = 0.5  # 수익 0.5% 이상일 때만 RSI 매도
    
    print("⚡ 초단타 스캘핑 v3.1 알고리즘 설정 (1초봉, BTC 최적화):")
    print(f"   - 테스트: {days_list}일 전 (랜덤 3일)")
    print("   - 데이터: 1초봉 (각 날짜별 24시간 = 86,400개 캔들)")
    print("   - 윈도우: 3600개 슬라이딩 윈도우 (1시간 = RSI 14분, MA 5/20분)")
    print("   - 익절: 0.8% (초단타)")
    print("   - 손절: 1.5% (빠른 손절)")
    print("   - RSI 매도: 수익 0.5% 이상일 때만 (조건부)")
    print("   - 매수 조건: RSI 30~60, 거래량 1.1배 이상")
    print("\n💡 전략: BTC 특성에 맞춘 매수 조건 최적화")
    print("⏰ 예상 소요 시간: 약 6-7분")
    print("⚠️  진행 상황: . = 1시간, 6h = 6시간 완료")
    print()
    
    results = run_multi_backtest(
        days_list=days_list, 
        rsi_threshold=rsi_threshold, 
        take_profit=take_profit,
        rsi_condition_profit=rsi_condition,
        use_seconds=True  # 1초봉 사용
    )
    
    if results:
        print("\n✅ 1초봉 백테스팅 완료!")
        print("\n💡 결과가 만족스러우면 main.py에 적용하세요.")

