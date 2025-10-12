"""
Upbit 1분봉 데이터 다운로드 및 병합 스크립트

공식 데이터 소스: https://crix-data.upbit.com/
- 날짜별 zip 파일 다운로드
- 압축 해제 및 CSV 병합
- data/daily_1m/ 폴더에 저장

사용법:
    python download_data.py
"""

import requests
import zipfile
import os
import pandas as pd
from datetime import datetime, timedelta
import time

def download_upbit_data(start_date, end_date, output_dir="data/daily_1m"):
    """
    Upbit 1분봉 데이터를 날짜별로 다운로드하고 압축 해제
    
    Args:
        start_date: 시작 날짜 (YYYYMMDD 문자열)
        end_date: 종료 날짜 (YYYYMMDD 문자열)
        output_dir: 저장 디렉토리 (기본: data/daily_1m)
    """
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 날짜 범위 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    print("=" * 80)
    print("📥 Upbit 1분봉 데이터 다운로드 (일자별 CSV)")
    print("=" * 80)
    print(f"기간: {start_date} ~ {end_date}")
    print(f"총 {len(date_list)}일")
    print(f"저장 위치: {output_dir}/")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    total_candles = 0
    
    for i, date_str in enumerate(date_list, 1):
        # URL 생성 (1분봉)
        year = date_str[:4]
        url = f"https://crix-data.upbit.com/candle/KRW-BTC/daily/1m/{year}/KRW-BTC_candle-1m_{date_str}.zip"
        zip_path = os.path.join(output_dir, f"{date_str}.zip")
        csv_filename = f"KRW-BTC_candle-1m_{date_str}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # 이미 존재하면 스킵
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[{i}/{len(date_list)}] {date_str} ⏭️  이미 존재 ({len(df):,}개)")
            success_count += 1
            total_candles += len(df)
            continue
        
        print(f"[{i}/{len(date_list)}] {date_str} 다운로드 중... ", end="", flush=True)
        
        try:
            # zip 파일 다운로드
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # zip 파일 저장
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # zip 압축 해제
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # CSV 파일 확인
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    success_count += 1
                    total_candles += len(df)
                    print(f"✅ {len(df):,}개 캔들")
                    
                    # zip 파일만 삭제 (CSV는 보관)
                    os.remove(zip_path)
                else:
                    print("❌ CSV 파일 없음")
                    fail_count += 1
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            
            elif response.status_code == 404:
                print("⚠️ 데이터 없음 (404)")
                fail_count += 1
            
            else:
                print(f"❌ 오류 ({response.status_code})")
                fail_count += 1
            
            # Rate Limit 방지
            time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ 예외: {e}")
            fail_count += 1
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue
    
    # 통계
    print("\n" + "=" * 80)
    print("📊 다운로드 완료")
    print("=" * 80)
    print(f"성공: {success_count}일")
    print(f"실패: {fail_count}일")
    print(f"성공률: {success_count/(success_count+fail_count)*100:.1f}%")
    print(f"총 캔들: {total_candles:,}개")
    print(f"저장 위치: {output_dir}/")
    print("=" * 80)
    
    return output_dir


def load_daily_csv(date_str, data_dir="data/daily_1m", timeframe="1m"):
    """
    특정 날짜의 CSV 파일 로드 (백테스팅용)
    
    Args:
        date_str: 날짜 (YYYYMMDD 문자열)
        data_dir: CSV 파일이 저장된 디렉토리
        timeframe: 시간봉 ('1s' 또는 '1m', 기본: '1m')
    
    Returns:
        DataFrame: OHLCV 데이터 (없으면 None)
        컬럼: date_time_utc, open, high, low, close, acc_trade_price, acc_trade_volume
    """
    csv_file = os.path.join(data_dir, f"KRW-BTC_candle-{timeframe}_{date_str}.csv")
    
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    df['date_time_utc'] = pd.to_datetime(df['date_time_utc'])
    
    return df


if __name__ == "__main__":
    print("\n🚀 Upbit 1분봉 데이터 다운로더\n")
    
    # 다운로드 기간 설정
    START_DATE = "20250101"
    END_DATE = "20250530"
    
    print(f"📅 다운로드 기간: {START_DATE} ~ {END_DATE}")
    total_days = (datetime.strptime(END_DATE, '%Y%m%d') - datetime.strptime(START_DATE, '%Y%m%d')).days + 1
    print(f"⏱️  예상 소요 시간: 약 {int(total_days * 0.1)} 분 (1분봉은 1초봉보다 빠름)")
    print(f"📦 다운로드 시작...\n")
    
    # 데이터 다운로드 (일자별 CSV로 저장)
    output_dir = download_upbit_data(START_DATE, END_DATE)
    
    if output_dir:
        print("\n✅ 모든 작업 완료!")
        print(f"\n📝 다음 단계:")
        print(f"   1. 일자별 CSV 확인: {output_dir}/")
        print(f"   2. backtest.py 수정 (CSV 파일 사용)")
        print(f"   3. backtest_optimizer.py 실행 (API 호출 없이 초고속!)")
        print(f"\n💡 백테스팅 예시:")
        print(f"   from download_data import load_daily_csv")
        print(f"   df = load_daily_csv('20250115')  # 1월 15일 데이터")

