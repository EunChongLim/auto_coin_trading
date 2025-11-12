"""
틱 데이터 다운로드 및 로드
Upbit 일일 체결 데이터: https://crix-data.upbit.com/trade/KRW-BTC/daily/YYYY/KRW-BTC_trade_YYYYMMDD.zip
"""

import os
import requests
import pandas as pd
import zipfile
from datetime import datetime, timedelta


def download_tick_data(date_str, output_dir="../../data/ticks"):
    """
    Upbit 틱 데이터 다운로드
    
    Args:
        date_str: YYYYMMDD 형식
        output_dir: 저장 디렉토리
    
    Returns:
        DataFrame or None
    """
    year = date_str[:4]
    filename = f"KRW-BTC_trade_{date_str}"
    zip_path = os.path.join(output_dir, f"{filename}.zip")
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    
    # 이미 CSV가 있으면 로드
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # ZIP이 있으면 압축 해제
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [Warning] Unzip failed for {date_str}: {e}")
            return None
    
    # 다운로드
    url = f"https://crix-data.upbit.com/trade/KRW-BTC/daily/{year}/{filename}.zip"
    
    try:
        print(f"  Downloading {date_str}...", end=" ", flush=True)
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            
            # ZIP 저장
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # CSV 로드
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"OK ({len(df):,} ticks)")
                return df
            else:
                print("FAILED (no CSV)")
                return None
        else:
            print(f"FAILED (HTTP {response.status_code})")
            return None
    
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def load_tick_data_range(start_date, end_date, output_dir="../../data/ticks"):
    """
    기간별 틱 데이터 로드
    
    Args:
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        output_dir: 저장 디렉토리
    
    Returns:
        DataFrame (전체 틱 데이터)
    """
    print(f"\n[Loading Tick Data] {start_date} ~ {end_date}")
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_ticks = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        
        df = download_tick_data(date_str, output_dir)
        if df is not None:
            all_ticks.append(df)
        
        current_dt += timedelta(days=1)
    
    if not all_ticks:
        raise ValueError(f"No tick data found for {start_date} ~ {end_date}")
    
    # 합치기
    result = pd.concat(all_ticks, ignore_index=True)
    
    # datetime 변환
    result['datetime'] = pd.to_datetime(result['timestamp'], unit='ms')
    
    print(f"[OK] Total {len(result):,} ticks loaded")
    return result


if __name__ == "__main__":
    # 테스트: 2024-01-01 데이터 다운로드
    df = download_tick_data("20240101")
    if df is not None:
        print(f"\n[Test Result]")
        print(f"Ticks: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())

