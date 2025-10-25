"""
실거래 기간 데이터 다운로드
2025-10-16 06:34:57 ~ 2025-10-17 21:35:01
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def download_live_period_data():
    """실거래와 동일한 기간 데이터 다운로드"""
    
    # 실거래 시작/종료 시간
    start_time = datetime(2025, 10, 16, 6, 34, 0)
    end_time = datetime(2025, 10, 17, 21, 36, 0)  # 1분 여유
    
    print("=" * 100)
    print("실거래 기간 데이터 다운로드")
    print("=" * 100)
    print(f"시작: {start_time}")
    print(f"종료: {end_time}")
    print(f"기간: {(end_time - start_time).total_seconds() / 3600:.1f}시간")
    
    # Upbit API 직접 호출 (200개씩)
    # 약 39시간 = 2340분 → 12번 호출 필요
    
    all_data = []
    current = end_time
    batch_count = 0
    
    url = "https://api.upbit.com/v1/candles/minutes/1"
    
    print(f"\n다운로드 중...")
    
    while current > start_time:
        batch_count += 1
        
        # ISO 8601 형식 + timezone
        to_param = current.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"
        
        params = {
            "market": "KRW-BTC",
            "to": to_param,
            "count": 200
        }
        
        try:
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"  Batch {batch_count}: API 실패 (status={response.status_code})")
                break
            
            data = response.json()
            
            if len(data) == 0:
                print(f"  Batch {batch_count}: 데이터 없음")
                break
            
            # DataFrame 변환
            df = pd.DataFrame(data)
            df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
            df = df.set_index('candle_date_time_kst')
            df = df.sort_index()
            
            all_data.append(df)
            
            oldest_time = df.index[0]
            print(f"  Batch {batch_count}: {oldest_time} ~ {df.index[-1]} ({len(df)}개)")
            
            # 시작 시간보다 이전이면 종료
            if oldest_time <= start_time:
                break
            
            current = oldest_time - timedelta(seconds=1)
            time.sleep(0.1)  # API 제한
            
        except Exception as e:
            print(f"  Batch {batch_count}: 오류 - {e}")
            break
    
    # 합치기
    if all_data:
        df_all = pd.concat(all_data).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        
        # 시간 범위 필터링
        df_all = df_all[(df_all.index >= start_time) & (df_all.index <= end_time)]
        
        print(f"\n[다운로드 완료]")
        print(f"  총 캔들: {len(df_all)}개")
        print(f"  시간 범위: {df_all.index[0]} ~ {df_all.index[-1]}")
        
        # CSV 저장 (기존 daily_1m 형식과 동일하게)
        df_save = df_all.reset_index()
        
        # 필요한 컬럼만 선택 및 이름 변경
        df_save = df_save[['candle_date_time_kst', 'opening_price', 'high_price', 
                          'low_price', 'trade_price', 'candle_acc_trade_volume']]
        
        df_save = df_save.rename(columns={
            'candle_date_time_kst': 'date_time_utc',
            'candle_acc_trade_volume': 'acc_trade_volume'
        })
        
        # 저장
        output_file = "data/daily_1m/KRW-BTC_live_period_20251016-17.csv"
        df_save.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n저장 완료: {output_file}")
        print(f"\n[저장된 데이터 샘플]")
        print(df_save.head(3))
        print("...")
        print(df_save.tail(3))
        
        return df_save
    else:
        print("다운로드 실패!")
        return None


if __name__ == "__main__":
    download_live_period_data()

