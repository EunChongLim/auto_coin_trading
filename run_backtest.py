"""
백테스트 통합 실행 스크립트
- 단일 날짜 테스트
- 월별 연속 테스트
- 거래 로그 저장
"""

from backtest_v3 import backtest_v3_continuous
from datetime import datetime
import sys

def save_trade_log(result, params, log_file='backtest_log.txt'):
    """거래 로그 저장 (이유 상세)"""
    if not result or 'trades' not in result:
        return
    
    trades = result['trades']
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n' + '=' * 100 + '\n')
        f.write(f'백테스트: {result["start_date"]} ({result["num_days"]}일)\n')
        f.write(f'파라미터: buy={params["buy"]}, sell={params["sell"]}, stop={params["stop"]}%, take={params["take"]}%\n')
        f.write(f'수익률: {result["total_return"]:+.2f}% | 거래: {result["total_trades"]}회 | 승률: {result["win_rate"]:.1f}%\n')
        f.write('=' * 100 + '\n\n')
        
        buy_info = None
        for trade in trades:
            if trade['type'] == 'BUY':
                buy_info = trade
                ts = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{ts}] [BUY] 매수 | Price: {trade['price']:,.0f} | "
                       f"Amount: {trade['amount']:.6f} | "
                       f"이유: Up={trade['prob_up']:.3f} >= {params['buy']:.2f}\n")
            
            elif trade['type'] == 'SELL' and buy_info:
                ts = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # 보유 시간 계산
                hold_time = (trade['timestamp'] - buy_info['timestamp']).total_seconds() / 60
                
                # 매도 이유별 태그 및 상세 설명
                if trade['reason'] == '손절':
                    reason_tag = 'STOP LOSS'
                    reason_detail = f"손실 {trade['profit_rate']:.2f}% <= -{params['stop']}%"
                elif trade['reason'] == '익절':
                    reason_tag = 'TAKE PROFIT'
                    reason_detail = f"수익 {trade['profit_rate']:.2f}% >= +{params['take']}%"
                elif trade['reason'] == 'ML하락':
                    reason_tag = 'ML SELL'
                    reason_detail = f"Down={trade['prob_down']:.3f} >= {params['sell']:.2f}"
                else:
                    reason_tag = 'FINAL'
                    reason_detail = "테스트 종료"
                
                f.write(f"[{ts}] [{reason_tag}] 매도 | "
                       f"Buy: {trade['buy_price']:,.0f} -> Sell: {trade['sell_price']:,.0f} | "
                       f"수익률: {trade['profit_rate']:+.2f}% | "
                       f"수익: {trade['profit']:,.0f} KRW | "
                       f"보유: {hold_time:.1f}분 | "
                       f"이유: {reason_detail}\n")
                
                buy_info = None
        
        # 거래 요약 통계
        f.write('\n' + '=' * 100 + '\n')
        f.write('거래 통계\n')
        f.write('=' * 100 + '\n')
        
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        if sell_trades:
            # 매도 이유별 통계
            reasons = {}
            for t in sell_trades:
                reason = t['reason']
                if reason not in reasons:
                    reasons[reason] = {'count': 0, 'profit': 0, 'wins': 0}
                reasons[reason]['count'] += 1
                reasons[reason]['profit'] += t['profit']
                if t['profit'] > 0:
                    reasons[reason]['wins'] += 1
            
            f.write(f"\n[매도 이유별 통계]\n")
            for reason, stats in reasons.items():
                win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
                avg_profit = stats['profit'] / stats['count'] if stats['count'] > 0 else 0
                f.write(f"  {reason}: {stats['count']}회 | 승률 {win_rate:.1f}% | 평균 {avg_profit:,.0f} KRW\n")
            
            # 수익 분석
            profits = [t['profit'] for t in sell_trades]
            profit_rates = [t['profit_rate'] for t in sell_trades]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]
            
            f.write(f"\n[수익 분석]\n")
            f.write(f"  평균 수익: {sum(profits)/len(profits):,.0f} KRW\n")
            f.write(f"  평균 수익률: {sum(profit_rates)/len(profit_rates):.2f}%\n")
            
            if wins:
                f.write(f"\n  [승리 거래]\n")
                f.write(f"    횟수: {len(wins)}회\n")
                f.write(f"    평균 수익: {sum(wins)/len(wins):,.0f} KRW\n")
                f.write(f"    최대 수익: {max(wins):,.0f} KRW\n")
            
            if losses:
                f.write(f"\n  [손실 거래]\n")
                f.write(f"    횟수: {len(losses)}회\n")
                f.write(f"    평균 손실: {sum(losses)/len(losses):,.0f} KRW\n")
                f.write(f"    최대 손실: {min(losses):,.0f} KRW\n")
            
            # 보유시간 분석
            hold_times = []
            prev_buy = None
            for trade in trades:
                if trade['type'] == 'BUY':
                    prev_buy = trade
                elif trade['type'] == 'SELL' and prev_buy:
                    hold_time = (trade['timestamp'] - prev_buy['timestamp']).total_seconds() / 60
                    hold_times.append(hold_time)
                    prev_buy = None
            
            if hold_times:
                f.write(f"\n[보유시간 분석]\n")
                f.write(f"  평균: {sum(hold_times)/len(hold_times):.1f}분\n")
                f.write(f"  최소: {min(hold_times):.1f}분\n")
                f.write(f"  최대: {max(hold_times):.1f}분\n")
        
        f.write('\n' + '=' * 100 + '\n')
    
    print(f"로그 저장: {log_file}")


def test_single_month(year, month, params, save_log=False):
    """단일 월 테스트"""
    start_date = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    
    num_days = (next_month - start_date).days
    start_str = start_date.strftime("%Y%m%d")
    
    print(f"\n{'='*100}")
    print(f"{year}년 {month}월 백테스트 ({num_days}일)")
    print(f"파라미터: buy={params['buy']}, sell={params['sell']}, stop={params['stop']}%, take={params['take']}%")
    print(f"{'='*100}")
    
    result = backtest_v3_continuous(
        start_date_str=start_str,
        num_days=num_days,
        buy_threshold=params['buy'],
        sell_threshold=params['sell'],
        stop_loss=params['stop'],
        take_profit=params['take'],
        verbose=True
    )
    
    if result:
        print(f"\n[결과]")
        print(f"  수익률: {result['total_return']:+.2f}%")
        print(f"  거래: {result['total_trades']}회 ({result['avg_trades_per_day']:.1f}회/일)")
        print(f"  승률: {result['win_rate']:.1f}%")
        print(f"  최종 잔고: {result['final_balance']:,.0f} KRW")
        
        if save_log:
            log_file = f"backtest_log_{year}{month:02d}.txt"
            save_trade_log(result, params, log_file)
        
        return result
    else:
        print("\n[실패] 백테스트 실패")
        return None


def test_multiple_months(year, months, params):
    """여러 달 연속 테스트"""
    print(f"\n{'='*100}")
    print(f"{year}년 {len(months)}개월 백테스트")
    print(f"파라미터: buy={params['buy']}, sell={params['sell']}, stop={params['stop']}%, take={params['take']}%")
    print(f"{'='*100}")
    
    results = []
    for month in months:
        result = test_single_month(year, month, params, save_log=False)
        if result:
            results.append({
                'month': f"{year}-{month:02d}",
                'return': result['total_return'],
                'trades': result['total_trades'],
                'trades_per_day': result['avg_trades_per_day'],
                'win_rate': result['win_rate']
            })
    
    if results:
        print(f"\n{'='*100}")
        print("월별 요약")
        print(f"{'='*100}")
        print(f"{'월':<10} {'수익률':<12} {'거래':<10} {'승률':<10}")
        print(f"{'-'*100}")
        
        for r in results:
            print(f"{r['month']:<10} {r['return']:>+9.2f}% {r['trades']:>4}회 ({r['trades_per_day']:>4.1f}/일) {r['win_rate']:>6.1f}%")
        
        print(f"{'-'*100}")
        avg_return = sum(r['return'] for r in results) / len(results)
        avg_trades = sum(r['trades_per_day'] for r in results) / len(results)
        avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
        print(f"{'평균':<10} {avg_return:>+9.2f}% {'':>4}   ({avg_trades:>4.1f}/일) {avg_win_rate:>6.1f}%")


if __name__ == "__main__":
    print("=" * 100)
    print("백테스트 실행")
    print("=" * 100)
    
    # 기본 파라미터 (실전 권장)
    params = {
        'buy': 0.20,
        'sell': 0.50,
        'stop': 1.0,
        'take': 1.5
    }
    
    # 사용법 안내
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "month":
            # 단일 월 테스트: python run_backtest.py month 2025 1
            year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
            month = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            test_single_month(year, month, params, save_log=True)
        
        elif command == "multi":
            # 여러 달 테스트: python run_backtest.py multi 2025
            year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
            months = [1, 2, 3, 4]
            test_multiple_months(year, months, params)
        
        else:
            print("사용법:")
            print("  python run_backtest.py month [년] [월]    - 단일 월 테스트")
            print("  python run_backtest.py multi [년]         - 여러 달 테스트")
    
    else:
        # 기본: 2025년 1월
        print("\n기본 실행: 2025년 1월")
        print("다른 옵션: python run_backtest.py month 2025 4")
        test_single_month(2025, 1, params, save_log=True)

