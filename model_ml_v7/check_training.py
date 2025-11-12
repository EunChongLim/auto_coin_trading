"""
v7 학습 진행 상황 체크
"""
import os
from datetime import datetime

model_path = "model/lgb_v7_tick.pkl"

print("=" * 60)
print("v7 학습 진행 상황 체크")
print("=" * 60)

if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    print(f"\n[완료] 모델 파일 생성됨!")
    print(f"  - 파일 크기: {file_size / 1024:.2f} KB")
    print(f"  - 생성 시간: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n다음 명령으로 백테스트 실행:")
    print(f"  python backtest_v7.py --start_date 20220101 --end_date 20221231")
else:
    print(f"\n[진행중] 모델 파일 아직 생성 안됨")
    print(f"\n2024년 틱 데이터 366개를 다운로드하고 집계 중...")
    print(f"예상 소요 시간: 30~60분")
    print(f"\n이 스크립트를 다시 실행하여 진행 상황 확인:")
    print(f"  python check_training.py")

print("\n" + "=" * 60)

