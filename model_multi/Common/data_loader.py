"""
멀티 타임프레임 데이터 로더
각 타임프레임을 Upbit API에서 직접 다운로드
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta


def download_candle_data(date_str, timeframe='1m', output_dir="../data"):
    """
    특정 날짜의 캔들 데이터를 Upbit API에서 다운로드
    
    Args:
        date_str: YYYYMMDD 형식
        timeframe: '1m', '1h', '4h'
        output_dir: 저장 디렉토리
    
    Returns:
        DataFrame or None
    """
    
    # API 엔드포인트 및 파일명 설정
    timeframe_config = {
        '1m': {
            'url_minutes': 1,
            'dir': 'daily_1m',
            'filename': f'KRW-BTC_candle-1m_{date_str}.csv'
        },
        '1h': {
            'url_minutes': 60,
            'dir': 'daily_1h',
            'filename': f'KRW-BTC_candle-1h_{date_str}.csv'
        },
        '4h': {
            'url_minutes': 240,
            'dir': 'daily_4h',
            'filename': f'KRW-BTC_candle-4h_{date_str}.csv'
        }
    }
    
    if timeframe not in timeframe_config:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use '1m', '1h', or '4h'")
    
    config = timeframe_config[timeframe]
    url_minutes = config['url_minutes']
    save_dir = os.path.join(output_dir, config['dir'])
    filepath = os.path.join(save_dir, config['filename'])
    
    # 이미 파일이 있으면 로드만
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # 컬럼명 통일 처리
        if 'datetime_kst' not in df.columns and 'date_time_kst' in df.columns:
            df = df.rename(columns={'date_time_kst': 'datetime_kst', 'date_time_utc': 'datetime_utc'})
        if 'open' not in df.columns and 'opening_price' in df.columns:
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'acc_trade_volume': 'volume'
            })
        return df
    
    print(f"  Downloading {timeframe} {date_str}...", end=" ", flush=True)
    
    try:
        # 날짜 파싱
        target_date = datetime.strptime(date_str, "%Y%m%d")
        end_time = target_date + timedelta(days=1)  # 다음날 00:00
        
        url = f"https://api.upbit.com/v1/candles/minutes/{url_minutes}"
        all_data = []
        to_param = end_time.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"
        
        batch = 1
        max_batches = 50 if timeframe == '1m' else 10  # 1분봉은 많으니 더 많은 배치
        
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
                    break
            
            if len(filtered) > 0:
                all_data.extend(filtered)
            else:
                break
            
            # 가장 오래된 시간을 다음 to로 사용
            oldest_time = datetime.strptime(data[-1]['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S")
            
            if oldest_time.date() < target_date.date():
                break
            
            to_param = oldest_time.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"
            batch += 1
            time.sleep(0.1)  # API 제한 방지
            
            if batch > max_batches:
                break
        
        if len(all_data) == 0:
            print("No data")
            return None
        
        # DataFrame 변환
        df = pd.DataFrame(all_data)
        
        # 필요한 컬럼 선택 및 컬럼명 통일
        df_clean = pd.DataFrame()
        df_clean['datetime_kst'] = df['candle_date_time_kst']
        df_clean['datetime_utc'] = df['candle_date_time_utc']
        df_clean['open'] = df['opening_price']
        df_clean['high'] = df['high_price']
        df_clean['low'] = df['low_price']
        df_clean['close'] = df['trade_price']
        df_clean['volume'] = df['candle_acc_trade_volume']
        
        df = df_clean
        
        # 시간순 정렬
        df = df.sort_values('datetime_kst')
        
        # 저장
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(filepath, index=False)
        
        print(f"OK ({len(df)} candles)")
        return df
        
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def load_multi_timeframe_data(start_date, end_date, output_dir="../data"):
    """
    기간별 멀티 타임프레임 데이터 로드
    
    Args:
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        output_dir: 데이터 저장 디렉토리
    
    Returns:
        tuple: (df_1m, df_1h, df_4h)
    """
    print(f"\n[Loading Multi-Timeframe Data] {start_date} ~ {end_date}")
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_1m = []
    all_1h = []
    all_4h = []
    
    current_dt = start_dt
    day_count = 0
    total_days = (end_dt - start_dt).days + 1
    
    while current_dt <= end_dt:
        day_count += 1
        date_str = current_dt.strftime("%Y%m%d")
        
        # 1분봉 (항상 다운로드)
        df_1m = download_candle_data(date_str, timeframe='1m', output_dir=output_dir)
        if df_1m is not None and not df_1m.empty:
            all_1m.append(df_1m)
        
        # 1시간봉
        df_1h = download_candle_data(date_str, timeframe='1h', output_dir=output_dir)
        if df_1h is not None and not df_1h.empty:
            all_1h.append(df_1h)
        
        # 4시간봉
        df_4h = download_candle_data(date_str, timeframe='4h', output_dir=output_dir)
        if df_4h is not None and not df_4h.empty:
            all_4h.append(df_4h)
        
        if day_count % 30 == 0:
            print(f"  Progress: {day_count}/{total_days} days")
        
        current_dt += timedelta(days=1)
    
    # 병합 및 정리
    print(f"\n[Merging Data...]")
    
    # 1분봉
    if all_1m:
        df_1m_final = pd.concat(all_1m, ignore_index=True)
        
        # datetime 컬럼 생성 (유연하게 처리)
        if 'datetime_kst' in df_1m_final.columns:
            df_1m_final['datetime'] = pd.to_datetime(df_1m_final['datetime_kst'])
        elif 'date_time_kst' in df_1m_final.columns:
            df_1m_final['datetime'] = pd.to_datetime(df_1m_final['date_time_kst'])
        elif 'date_time_utc' in df_1m_final.columns:
            df_1m_final['datetime'] = pd.to_datetime(df_1m_final['date_time_utc'])
        elif 'candle_date_time_kst' in df_1m_final.columns:
            df_1m_final['datetime'] = pd.to_datetime(df_1m_final['candle_date_time_kst'])
        elif 'datetime' in df_1m_final.columns:
            df_1m_final['datetime'] = pd.to_datetime(df_1m_final['datetime'])
        else:
            print(f"  [WARNING] Available columns: {df_1m_final.columns.tolist()}")
            raise ValueError("Cannot find datetime column!")
        
        # volume 컬럼 처리
        if 'volume' not in df_1m_final.columns:
            if 'acc_trade_volume' in df_1m_final.columns:
                df_1m_final['volume'] = df_1m_final['acc_trade_volume']
        
        df_1m_final = df_1m_final.sort_values('datetime').reset_index(drop=True)
        # 필요한 컬럼만 선택
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df_1m_final = df_1m_final[required_cols]
        print(f"  1분봉: {len(df_1m_final):,} candles")
    else:
        df_1m_final = pd.DataFrame()
    
    # 1시간봉
    if all_1h:
        df_1h_final = pd.concat(all_1h, ignore_index=True)
        
        # datetime 컬럼 생성
        if 'datetime_kst' in df_1h_final.columns:
            df_1h_final['datetime'] = pd.to_datetime(df_1h_final['datetime_kst'])
        elif 'date_time_kst' in df_1h_final.columns:
            df_1h_final['datetime'] = pd.to_datetime(df_1h_final['date_time_kst'])
        elif 'date_time_utc' in df_1h_final.columns:
            df_1h_final['datetime'] = pd.to_datetime(df_1h_final['date_time_utc'])
        elif 'candle_date_time_kst' in df_1h_final.columns:
            df_1h_final['datetime'] = pd.to_datetime(df_1h_final['candle_date_time_kst'])
        elif 'datetime' in df_1h_final.columns:
            df_1h_final['datetime'] = pd.to_datetime(df_1h_final['datetime'])
        else:
            print(f"  [WARNING] Available columns: {df_1h_final.columns.tolist()}")
            raise ValueError("Cannot find datetime column in 1h data!")
        
        # volume 컬럼 처리
        if 'volume' not in df_1h_final.columns:
            if 'acc_trade_volume' in df_1h_final.columns:
                df_1h_final['volume'] = df_1h_final['acc_trade_volume']
        
        df_1h_final = df_1h_final.sort_values('datetime').reset_index(drop=True)
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df_1h_final = df_1h_final[required_cols]
        # 중복 제거 (날짜 경계에서 중복 가능)
        df_1h_final = df_1h_final.drop_duplicates(subset=['datetime'], keep='first')
        print(f"  1시간봉: {len(df_1h_final):,} candles")
    else:
        df_1h_final = pd.DataFrame()
    
    # 4시간봉
    if all_4h:
        df_4h_final = pd.concat(all_4h, ignore_index=True)
        
        # datetime 컬럼 생성
        if 'datetime_kst' in df_4h_final.columns:
            df_4h_final['datetime'] = pd.to_datetime(df_4h_final['datetime_kst'])
        elif 'date_time_kst' in df_4h_final.columns:
            df_4h_final['datetime'] = pd.to_datetime(df_4h_final['date_time_kst'])
        elif 'date_time_utc' in df_4h_final.columns:
            df_4h_final['datetime'] = pd.to_datetime(df_4h_final['date_time_utc'])
        elif 'candle_date_time_kst' in df_4h_final.columns:
            df_4h_final['datetime'] = pd.to_datetime(df_4h_final['candle_date_time_kst'])
        elif 'datetime' in df_4h_final.columns:
            df_4h_final['datetime'] = pd.to_datetime(df_4h_final['datetime'])
        else:
            print(f"  [WARNING] Available columns: {df_4h_final.columns.tolist()}")
            raise ValueError("Cannot find datetime column in 4h data!")
        
        # volume 컬럼 처리
        if 'volume' not in df_4h_final.columns:
            if 'acc_trade_volume' in df_4h_final.columns:
                df_4h_final['volume'] = df_4h_final['acc_trade_volume']
        
        df_4h_final = df_4h_final.sort_values('datetime').reset_index(drop=True)
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df_4h_final = df_4h_final[required_cols]
        # 중복 제거
        df_4h_final = df_4h_final.drop_duplicates(subset=['datetime'], keep='first')
        print(f"  4시간봉: {len(df_4h_final):,} candles")
    else:
        df_4h_final = pd.DataFrame()
    
    print(f"[OK] Multi-timeframe data loaded")
    
    return df_1m_final, df_1h_final, df_4h_final


def get_current_partial_candle(df_full, timeframe, current_time):
    """
    현재 진행 중인 (미완성) 캔들 계산
    
    Args:
        df_full: 해당 타임프레임의 완전한 DataFrame
        timeframe: '1m', '1h', '4h'
        current_time: 현재 시각 (datetime)
    
    Returns:
        dict: 미완성 캔들 데이터
    """
    df = df_full.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 현재 캔들의 시작 시각 계산
    if timeframe == '1m':
        candle_start = current_time.replace(second=0, microsecond=0)
    elif timeframe == '1h':
        candle_start = current_time.replace(minute=0, second=0, microsecond=0)
    elif timeframe == '4h':
        hour = (current_time.hour // 4) * 4
        candle_start = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    # candle_start ~ current_time 사이의 데이터
    # 실제로는 해당 타임프레임의 이전 완성 캔들까지만 사용
    # 미완성 캔들은 실시간에서 계산해야 함
    
    # 간단히 마지막 완성 캔들을 반환
    completed = df[df['datetime'] < current_time]
    
    if len(completed) == 0:
        return None
    
    return completed.iloc[-1].to_dict()


if __name__ == "__main__":
    # 테스트
    print("="*80)
    print("멀티 타임프레임 데이터 로더 테스트")
    print("="*80)
    
    # 단일 날짜 테스트
    print("\n[1일 테스트: 2024-01-01]")
    df_1m = download_candle_data("20240101", timeframe='1m')
    df_1h = download_candle_data("20240101", timeframe='1h')
    df_4h = download_candle_data("20240101", timeframe='4h')
    
    if df_1m is not None:
        print(f"\n1분봉: {len(df_1m)}개")
        print(f"시간 범위: {df_1m['datetime_kst'].iloc[0]} ~ {df_1m['datetime_kst'].iloc[-1]}")
    
    if df_1h is not None:
        print(f"\n1시간봉: {len(df_1h)}개")
        print(f"시간 범위: {df_1h['datetime_kst'].iloc[0]} ~ {df_1h['datetime_kst'].iloc[-1]}")
    
    if df_4h is not None:
        print(f"\n4시간봉: {len(df_4h)}개")
        print(f"시간 범위: {df_4h['datetime_kst'].iloc[0]} ~ {df_4h['datetime_kst'].iloc[-1]}")
    
    print("\n" + "="*80)

