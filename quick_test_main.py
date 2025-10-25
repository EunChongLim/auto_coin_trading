"""
main.py 빠른 동작 테스트
"""

import joblib

print("="*80)
print("main.py 동작 테스트")
print("="*80)

# 1. 모델 로드 테스트
print("\n[1] 모델 로드 테스트...")
try:
    model_data = joblib.load("model/lgb_model_v3.pkl")
    print(f"   - Model: {type(model_data['model']).__name__}")
    print(f"   - Features: {len(model_data['feature_cols'])}")
    print(f"   - Has best_iteration: {hasattr(model_data['model'], 'best_iteration')}")
    print("   [OK] Model load success!")
except Exception as e:
    print(f"   [ERROR] {e}")

# 2. 파라미터 확인
print("\n[2] main.py 파라미터 확인...")
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
    # sell_threshold 확인
    if 'sell_threshold = 0.40' in content:
        print("   [OK] sell_threshold = 0.40")
    else:
        print("   [WARN] sell_threshold is not 0.40")
    
    # v3.2 확인
    if 'v3.2' in content:
        print("   [OK] Version v3.2")
    else:
        print("   [WARN] Version not updated")
    
    # RandomForest 언급 확인
    if 'RandomForest' in content:
        print("   [OK] RandomForest model info included")
    else:
        print("   [WARN] RandomForest info missing")
    
    # predict_proba 확인 (sklearn 호환)
    if 'predict_proba' in content:
        print("   [OK] sklearn model compatible code included")
    else:
        print("   [ERROR] sklearn compatibility missing")

# 3. 로그 파일 위치
print("\n[3] Log file settings...")
if 'LOG_FILE = "trading_log.txt"' in content:
    print("   [OK] Log file: trading_log.txt")
else:
    print("   [ERROR] Log file not configured")

print("\n" + "="*80)
print("main.py 준비 완료!")
print("="*80)
print("\n실행 명령: python main.py")
print("="*80)

