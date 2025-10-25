"""
현재 적용된 모델 확인
"""
import joblib

model_data = joblib.load('model/lgb_model_v3.pkl')

print("="*80)
print("현재 model/lgb_model_v3.pkl 파일 정보")
print("="*80)
print(f"\n모델 타입: {type(model_data['model']).__name__}")
print(f"특징 개수: {len(model_data['feature_cols'])}")

if 'train_acc' in model_data:
    print(f"Train 정확도: {model_data['train_acc']:.4f}")
if 'test_acc' in model_data:
    print(f"Test 정확도: {model_data['test_acc']:.4f}")

if 'params' in model_data:
    params = model_data['params']
    print(f"\n학습 파라미터:")
    print(f"  - future_minutes: {params.get('future_minutes', 'N/A')}")
    print(f"  - down_threshold: {params.get('down_threshold', 'N/A')}")
    print(f"  - up_threshold: {params.get('up_threshold', 'N/A')}")

print("\n" + "="*80)
if type(model_data['model']).__name__ == 'RandomForestClassifier':
    print("✓ extreme_RF_fm10_d4_u5.pkl이 정상적으로 적용되었습니다!")
else:
    print("✗ 다른 모델이 로드되었습니다.")
print("="*80)

