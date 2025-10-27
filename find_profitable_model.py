"""
최근 3일 데이터에서 수익이 나는 모델 찾기
다양한 알고리즘과 설정 테스트
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import os
from datetime import datetime, timedelta
from backtest_v3 import backtest_v3_continuous

print("="*80)
print("수익 모델 탐색 - 최근 3일 백테스트 (2025-10-22~24)")
print("="*80)

# 학습 데이터 로드
print("\n[Step 1] 학습 데이터 로드 중...")
try:
    data_files = []
    data_dir = "data/daily"
    
    # 2024년 데이터 (200일)
    for year in [2024]:
        files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"KRW-BTC_candle_{year}")])[:200]
        data_files.extend([os.path.join(data_dir, f) for f in files])
    
    print(f"   Loading {len(data_files)} files...")
    df_list = [pd.read_csv(f) for f in data_files]
    df_all = pd.concat(df_list, ignore_index=True)
    
    print(f"   Total samples: {len(df_all):,}")
    
    # 특징과 타겟 분리
    feature_cols = [col for col in df_all.columns if col not in ['candle_date_time_kst', 'target', 'target_return']]
    X = df_all[feature_cols].copy()
    y = df_all['target'].copy()
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"   Features: {len(feature_cols)}")

except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

# 테스트할 모델 정의
models_to_test = []

print("\n[Step 2] 모델 정의 및 학습...")

# 1. RandomForest 변형
print("\n1. RandomForest 변형...")
for n_estimators in [100, 200]:
    for max_depth in [10, 20, None]:
        name = f"RF_n{n_estimators}_d{max_depth if max_depth else 'None'}"
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        models_to_test.append((name, model))
        print(f"   - {name}")

# 2. ExtraTrees
print("\n2. ExtraTrees...")
for n_estimators in [100, 200]:
    name = f"ET_n{n_estimators}"
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    models_to_test.append((name, model))
    print(f"   - {name}")

# 3. HistGradientBoosting
print("\n3. HistGradientBoosting...")
for learning_rate in [0.05, 0.1]:
    for max_depth in [10, 20]:
        name = f"HGB_lr{learning_rate}_d{max_depth}"
        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        models_to_test.append((name, model))
        print(f"   - {name}")

# 4. LightGBM
print("\n4. LightGBM...")
for num_leaves in [31, 50]:
    for learning_rate in [0.05, 0.1]:
        name = f"LGB_nl{num_leaves}_lr{learning_rate}"
        model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=100,
            random_state=42,
            verbose=-1
        )
        models_to_test.append((name, model))
        print(f"   - {name}")

# 5. GradientBoosting (sklearn)
print("\n5. GradientBoosting (sklearn)...")
for learning_rate in [0.05, 0.1]:
    name = f"GB_lr{learning_rate}"
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    models_to_test.append((name, model))
    print(f"   - {name}")

print(f"\n총 {len(models_to_test)}개 모델 테스트 예정")

# 모델 학습 및 테스트
print("\n[Step 3] 모델 학습 및 백테스트...")
print("="*80)

results = []
profitable_models = []

for i, (name, model) in enumerate(models_to_test, 1):
    print(f"\n[{i}/{len(models_to_test)}] {name}")
    print("-"*80)
    
    try:
        # 학습
        print("   Training...", end=" ")
        model.fit(X_train, y_train)
        print("Done")
        
        # 테스트 정확도
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"   Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # 모델 저장
        model_path = f"model/test_{name}.pkl"
        model_data = {
            'model': model,
            'feature_cols': feature_cols,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        joblib.dump(model_data, model_path)
        
        # lgb_model_v3.pkl로 임시 복사
        backup_path = "model/lgb_model_v3_temp_backup.pkl"
        if i == 1:
            os.rename("model/lgb_model_v3.pkl", backup_path)
        
        import shutil
        shutil.copy(model_path, "model/lgb_model_v3.pkl")
        
        # 백테스트 (최근 3일)
        print("   Backtesting (2025-10-22~24)...", end=" ")
        result = backtest_v3_continuous(
            start_date_str="20251022",
            num_days=3,
            buy_threshold=0.20,
            sell_threshold=0.35,
            stop_loss=1.5,
            take_profit=1.2,
            verbose=False
        )
        print("Done")
        
        # 결과 저장
        result_data = {
            'name': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'return': result['total_return'],
            'trades': result['total_trades'],
            'win_rate': result['win_rate']
        }
        results.append(result_data)
        
        # 결과 출력
        profit_marker = " *** PROFIT ***" if result['total_return'] > 0 else ""
        print(f"   Return: {result['total_return']:+.2f}% | Trades: {result['total_trades']} | WinRate: {result['win_rate']:.1f}%{profit_marker}")
        
        # 수익 모델 발견
        if result['total_return'] > 0:
            profitable_models.append(result_data)
            print(f"   >>> 수익 모델 발견! <<<")
        
        # 임시 파일 삭제
        os.remove(model_path)
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

# 원본 모델 복원
if os.path.exists("model/lgb_model_v3_temp_backup.pkl"):
    os.rename("model/lgb_model_v3_temp_backup.pkl", "model/lgb_model_v3.pkl")

# 결과 분석
print("\n\n" + "="*80)
print("최종 결과")
print("="*80)

# 수익률 순 정렬
results_sorted = sorted(results, key=lambda x: x['return'], reverse=True)

print(f"\n[TOP 10] 수익률 순위")
print("-"*80)
print(f"{'Rank':<6} {'Model':<30} {'Return':<12} {'Trades':<10} {'WinRate':<10}")
print("-"*80)

for i, r in enumerate(results_sorted[:10], 1):
    marker = " *** PROFIT" if r['return'] > 0 else ""
    print(f"{i:<6} {r['name']:<30} {r['return']:>+9.2f}% {r['trades']:>6} {r['win_rate']:>8.1f}%{marker}")

# 수익 모델 요약
print("\n" + "="*80)
if profitable_models:
    print(f"수익 모델 발견: {len(profitable_models)}개")
    print("="*80)
    
    for i, m in enumerate(profitable_models, 1):
        print(f"\n{i}. {m['name']}")
        print(f"   Return: {m['return']:+.2f}%")
        print(f"   Trades: {m['trades']}회")
        print(f"   Win Rate: {m['win_rate']:.1f}%")
        print(f"   Test Acc: {m['test_acc']:.4f}")
    
    # 최고 수익 모델
    best = profitable_models[0]
    print("\n" + "="*80)
    print("최고 수익 모델")
    print("="*80)
    print(f"모델: {best['name']}")
    print(f"수익률: {best['return']:+.2f}%")
    print(f"승률: {best['win_rate']:.1f}%")
    print(f"거래: {best['trades']}회")
    
else:
    print("수익 모델을 찾지 못했습니다.")
    print("="*80)
    print("\n최선의 모델 (최소 손실):")
    best = results_sorted[0]
    print(f"  모델: {best['name']}")
    print(f"  수익률: {best['return']:+.2f}%")
    print(f"  승률: {best['win_rate']:.1f}%")
    print(f"  거래: {best['trades']}회")
    
    print("\n권장사항:")
    print("  1. 다른 파라미터 조합 시도 (buy/sell threshold)")
    print("  2. 더 긴 기간으로 테스트 (7일, 10일)")
    print("  3. 시장 상황 변화 대기")

print("="*80)

