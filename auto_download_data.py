"""
자동 데이터 다운로드 함수
백테스트 시 필요한 날짜의 1분봉 데이터를 자동으로 다운로드
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta


def download_1m_data(date_str, output_dir="data/daily_1m"):
    """
    특정 날짜의 1분봉 데이터를 Upbit API에서 다운로드
    
    Args:
        date_str: YYYYMMDD 형식 (예: "20251013")
        output_dir: 저장 디렉토리
    
    Returns:
        DataFrame or None
    """
    
    # 파일명 생성
    filename = f"KRW-BTC_candle-1m_{date_str}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 이미 파일이 있으면 로드만
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    print(f"  Downloading {date_str}...", end=" ", flush=True)
    
    try:
        # 날짜 파싱
        target_date = datetime.strptime(date_str, "%Y%m%d")
        end_time = target_date + timedelta(days=1)  # 다음날 00:00
        
        url = "https://api.upbit.com/v1/candles/minutes/1"
        all_data = []
        to_param = end_time.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"
        
        batch = 1
        while True:
            params = {
                "market": "KRW-BTC",
                "to": to_param,
                "count": 200
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"ERROR {response.status_code}")
                return None
            
            data = response.json()
            if len(data) == 0:
                break
            
            # 해당 날짜 범위만 필터링
            filtered = []
            for candle in data:
                candle_time = datetime.strptime(candle['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S")
                if candle_time.date() == target_date.date():
                    filtered.append(candle)
                elif candle_time < target_date:
                    # 목표 날짜보다 이전이면 중단
                    break
            
            if len(filtered) > 0:
                all_data.extend(filtered)
            else:
                # 더 이상 해당 날짜 데이터가 없으면 중단
                break
            
            # 가장 오래된 시간을 다음 to로 사용
            oldest_time = datetime.strptime(data[-1]['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S")
            
            # 목표 날짜보다 이전이면 중단
            if oldest_time.date() < target_date.date():
                break
            
            to_param = oldest_time.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"
            batch += 1
            time.sleep(0.1)  # API 제한 방지
            
            if batch > 20:  # 안전장치 (최대 20번 = 4000개)
                break
        
        if len(all_data) == 0:
            print("No data")
            return None
        
        # DataFrame 변환
        df = pd.DataFrame(all_data)
        df = df[['candle_date_time_kst', 'candle_date_time_utc', 'opening_price', 'high_price', 
                 'low_price', 'trade_price', 'candle_acc_trade_volume', 'candle_acc_trade_price']]
        
        # 컬럼명 변경 (기존 CSV 형식과 일치)
        df.columns = ['date_time_kst', 'date_time_utc', 'opening_price', 'high_price', 
                      'low_price', 'trade_price', 'acc_trade_volume', 'acc_trade_price']
        
        # 시간순 정렬
        df = df.sort_values('date_time_kst')
        
        # 저장
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(filepath, index=False)
        
        print(f"OK ({len(df)} candles)")
        return df
        
    except Exception as e:
        print(f"FAILED ({e})")
        return None


if __name__ == "__main__":
    # 테스트
    df = download_1m_data("20251013")
    if df is not None:
        print(f"\nDownloaded: {len(df)} candles")
        print(f"Range: {df['date_time_kst'].iloc[0]} ~ {df['date_time_kst'].iloc[-1]}")

