"""
LightGBM 모델 학습 스크립트
1분봉 데이터로 모델 학습 후 저장
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from download_data import load_daily_csv
from feature_engineer import prepare_ml_data
import random
from datetime import datetime, timedelta


def load_multiple_days(start_date, end_date, data_dir="data/daily", timeframe="1s", max_days=30):
    """
    여러 날짜의 데이터 로드
    
    Args:
        start_date: 시작 날짜 (YYYYMMDD)
        end_date: 종료 날짜 (YYYYMMDD)
        data_dir: 데이터 디렉토리
        timeframe: 시간봉 ('1s' 또는 '1m')
        max_days: 최대 로드 일수
    
    Returns:
        DataFrame: 병합된 데이터
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    all_days = []
    current = start
    while current <= end:
        all_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    # 랜덤 샘플링 (너무 많으면)
    if len(all_days) > max_days:
        all_days = sorted(random.sample(all_days, max_days))
    
    timeframe_name = "1초봉" if timeframe == "1s" else "1분봉"
    print(f"\n📅 {len(all_days)}일치 {timeframe_name} 데이터 로드 중...")
    
    dfs = []
    for i, date_str in enumerate(all_days, 1):
        df = load_daily_csv(date_str, data_dir, timeframe)
        if df is not None and len(df) > 0:
            # 컬럼 이름 매핑
            df = df.rename(columns={
                'date_time_utc': 'timestamp',
                'acc_trade_volume': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            dfs.append(df)
            
            if i % 10 == 0:
                print(f"  [{i}/{len(all_days)}] 로드 완료...")
    
    if not dfs:
        raise ValueError("데이터를 로드할 수 없습니다.")
    
    merged_df = pd.concat(dfs, axis=0)
    merged_df = merged_df.sort_index()
    
    print(f"✅ 총 {len(merged_df):,}개 {timeframe_name} 데이터 로드 완료")
    
    return merged_df


def train_lgb_model(X_train, y_train, X_val, y_val, use_smote=True, target_ratio=0.5):
    """
    LightGBM 모델 학습 (SMOTE 오버샘플링 + scale_pos_weight)
    
    Args:
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        use_smote: SMOTE 사용 여부
        target_ratio: SMOTE 목표 비율 (0.3 = 소수 클래스를 다수 클래스의 30%로)
    
    Returns:
        LightGBM model
    """
    print("\n🤖 LightGBM 모델 학습 시작...")
    
    # SMOTE 오버샘플링 (학습 데이터에만 적용)
    if use_smote:
        print(f"   ⚙️  SMOTE 오버샘플링 중 (목표 비율: {target_ratio:.1%})...")
        original_pos = y_train.sum()
        original_neg = len(y_train) - original_pos
        print(f"   - 원본: 양성={original_pos:,}, 음성={original_neg:,} (비율={original_pos/len(y_train):.2%})")
        
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        new_pos = y_train_resampled.sum()
        new_neg = len(y_train_resampled) - new_pos
        print(f"   - 샘플링 후: 양성={new_pos:,}, 음성={new_neg:,} (비율={new_pos/len(y_train_resampled):.2%})")
        
        X_train = X_train_resampled
        y_train = y_train_resampled
    
    # 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # scale_pos_weight 계산
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # 파라미터 설정
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,  # 불균형 가중치
        'verbose': -1,
        'seed': 42
    }
    
    print(f"   - scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 학습
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=20)
        ]
    )
    
    return model


def evaluate_model(model, X, y, dataset_name="Test"):
    """
    모델 평가
    """
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    print(f"\n📊 {dataset_name} 평가 결과:")
    print(f"   - ROC-AUC: {roc_auc_score(y, y_pred_proba):.4f}")
    print(f"\n분류 리포트:")
    print(classification_report(y, y_pred, target_names=['하락/유지', '상승']))
    
    print(f"\n혼동 행렬:")
    cm = confusion_matrix(y, y_pred)
    print(f"   [[TN={cm[0,0]:,}, FP={cm[0,1]:,}],")
    print(f"    [FN={cm[1,0]:,}, TP={cm[1,1]:,}]]")
    
    return y_pred_proba


def main():
    """
    메인 실행 함수
    """
    print("=" * 80)
    print("🚀 비트코인 ML 모델 학습 시작")
    print("=" * 80)
    
    # 1. 데이터 로드 (1분봉)
    df = load_multiple_days("20250101", "20250530", data_dir="data/daily_1m", timeframe="1m", max_days=30)
    
    # 2. 특징 & 라벨 생성 (상대 랭크 기반)
    print("\n📊 특징 & 라벨 생성 중...")
    X, y, feature_cols, df_with_features = prepare_ml_data(
        df,
        future_minutes=20,      # 20분 후 예측 (더 긴 시간으로 안정적 예측)
        use_rank=True,          # 상대 랭크 기반
        rank_percentile=0.8     # 상위 20%를 상승으로 라벨링
    )
    
    # 3. 학습/검증/테스트 분할 (시계열 순서 유지)
    print("\n✂️  데이터 분할 중...")
    
    # 시계열이므로 랜덤하게 섞지 않고 순서대로 분할
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]
    
    print(f"   - 학습: {len(X_train):,}개")
    print(f"   - 검증: {len(X_val):,}개")
    print(f"   - 테스트: {len(X_test):,}개")
    
    # 4. 모델 학습
    model = train_lgb_model(X_train, y_train, X_val, y_val)
    
    # 5. 평가
    print("\n" + "=" * 80)
    print("📈 모델 평가")
    print("=" * 80)
    
    evaluate_model(model, X_train, y_train, "학습 데이터")
    evaluate_model(model, X_val, y_val, "검증 데이터")
    evaluate_model(model, X_test, y_test, "테스트 데이터")
    
    # 6. 특징 중요도
    print("\n" + "=" * 80)
    print("🔍 특징 중요도 (상위 10개)")
    print("=" * 80)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"   {row['feature']:<25} {row['importance']:>10.0f}")
    
    # 7. 모델 저장
    model_path = "model/lgb_model.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'version': '1.0',
        'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, model_path)
    
    print(f"\n💾 모델 저장 완료: {model_path}")
    print("\n" + "=" * 80)
    print("✅ 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

