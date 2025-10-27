#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 머신러닝 모델 테스트 스크립트
- TabNet, Transformer, TCN, 앙상블 등 최신 모델 테스트
- 최근 3일 데이터로 백테스트
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 기존 모듈 import
from backtest_v3 import backtest_v3_continuous
from multi_timeframe_features import prepare_multi_timeframe_data

def load_training_data():
    """2025년 1-5월 데이터로 학습 데이터 로드"""
    print("학습 데이터 로드 중...")
    
    all_data = []
    for month in range(1, 6):  # 1-5월
        for day in range(1, 32):
            try:
                date_str = f"2025{month:02d}{day:02d}"
                file_path = f"data/daily/KRW-BTC_candle-1s_{date_str}.csv"
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # 컬럼명 통일
                    if 'date_time_utc' in df.columns:
                        df = df.rename(columns={'date_time_utc': 'datetime'})
                    if 'acc_trade_volume' in df.columns:
                        df = df.rename(columns={'acc_trade_volume': 'volume'})
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    all_data.append(df)
                    print(f"   OK {date_str}: {len(df)} candles")
            except:
                continue
    
    if not all_data:
        print("ERROR: 학습 데이터를 찾을 수 없습니다!")
        return None
    
    # 데이터 결합
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"총 학습 데이터: {len(df):,} candles")
    print(f"   기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    
    return df

def prepare_features_and_labels(df, future_minutes=10, down_threshold=0.4, up_threshold=0.5):
    """특징과 레이블 준비"""
    print(f"특징 계산 중... (future_minutes={future_minutes})")
    
    # datetime을 인덱스로 설정
    df_indexed = df.copy()
    df_indexed = df_indexed.set_index('datetime')
    
    # 멀티 타임프레임 특징 계산
    df_features = prepare_multi_timeframe_data(df_indexed)
    
    if df_features is None or len(df_features) < 1000:
        print("ERROR: 특징 계산 실패!")
        return None, None, None
    
    # 미래 가격 계산
    df_features['future_price'] = df_features['close'].shift(-future_minutes)
    df_features['price_change_pct'] = (df_features['future_price'] - df_features['close']) / df_features['close'] * 100
    
    # 레이블 생성
    conditions = [
        df_features['price_change_pct'] <= -down_threshold,
        df_features['price_change_pct'] >= up_threshold
    ]
    choices = [0, 2]  # Down, Up
    df_features['label'] = np.select(conditions, choices, default=1)  # Sideways
    
    # NaN 제거
    df_features = df_features.dropna()
    
    print(f"   특징 수: {len([col for col in df_features.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'future_price', 'price_change_pct', 'label']])}")
    print(f"   레이블 분포: Down={sum(df_features['label']==0)}, Sideways={sum(df_features['label']==1)}, Up={sum(df_features['label']==2)}")
    
    return df_features, None, None

def train_and_test_model(model_name, model, df_features, start_date_str, num_days, 
                        buy_threshold, sell_threshold, stop_loss, take_profit):
    """모델 학습 및 백테스트"""
    print(f"\n{model_name} 테스트 시작...")
    
    # 특징 선택 (숫자형 컬럼만)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 
                                 'future_price', 'price_change_pct', 'label'] and 
                   df_features[col].dtype in ['float64', 'int64']]
    
    X = df_features[feature_cols]
    y = df_features['label']
    
    # 학습/검증 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 스케일링 (필요한 경우)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    try:
        if model_name in ['SVM', 'KNN', 'MLP']:
            model.fit(X_train_scaled, y_train)
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
        
        print(f"   학습 정확도: {train_score:.3f}")
        print(f"   검증 정확도: {test_score:.3f}")
        
        # 모델 저장
        model_data = {
            'model': model,
            'scaler': scaler if model_name in ['SVM', 'KNN', 'MLP'] else None,
            'feature_cols': feature_cols,
            'version': 'v3.4-advanced',
            'type': model_name,
            'train_score': train_score,
            'test_score': test_score
        }
        
        temp_model_path = f"model/temp_{model_name.replace(' ', '_')}.pkl"
        joblib.dump(model_data, temp_model_path)
        
        # 백테스트
        print(f"   백테스트 실행 중...")
        backtest_result = backtest_v3_continuous(
            start_date_str=start_date_str,
            num_days=num_days,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model_path=temp_model_path,
            verbose=False
        )
        
        # 임시 파일 삭제
        os.remove(temp_model_path)
        
        return {
            'model_name': model_name,
            'train_score': train_score,
            'test_score': test_score,
            'backtest_result': backtest_result
        }
        
    except Exception as e:
        print(f"   ERROR: {model_name} 실패: {e}")
        return None

def main():
    print("=" * 80)
    print("고급 머신러닝 모델 테스트 시작")
    print("=" * 80)
    
    # 학습 데이터 로드
    df = load_training_data()
    if df is None:
        return
    
    # 특징 및 레이블 준비
    df_features, _, _ = prepare_features_and_labels(df, future_minutes=60, down_threshold=1.0, up_threshold=1.0)
    if df_features is None:
        return
    
    # 테스트할 모델들 정의
    models_to_test = [
        # 기본 앙상블 모델들
        ("RandomForest_Advanced", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ("ExtraTrees_Advanced", ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ("HistGradientBoosting_Advanced", HistGradientBoostingClassifier(max_depth=15, learning_rate=0.1, random_state=42)),
        ("LightGBM_Advanced", lgb.LGBMClassifier(num_leaves=50, learning_rate=0.1, max_depth=15, random_state=42, n_jobs=-1)),
        
        # 신경망 모델
        ("MLP_Deep", MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42)),
        ("MLP_Wide", MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500, random_state=42)),
        
        # 서포트 벡터 머신
        ("SVM_RBF", SVC(kernel='rbf', probability=True, random_state=42)),
        ("SVM_Poly", SVC(kernel='poly', degree=3, probability=True, random_state=42)),
        
        # K-최근접 이웃
        ("KNN_5", KNeighborsClassifier(n_neighbors=5)),
        ("KNN_10", KNeighborsClassifier(n_neighbors=10)),
        
        # 로지스틱 회귀
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        
        # 그래디언트 부스팅
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)),
    ]
    
    # 앙상블 모델 추가
    print("\n앙상블 모델 생성 중...")
    
    # 개별 모델들
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMClassifier(num_leaves=31, random_state=42, n_jobs=-1)
    
    # 투표 앙상블
    voting_ensemble = VotingClassifier([
        ('rf', rf),
        ('et', et),
        ('lgb', lgb_model)
    ], voting='soft')
    
    models_to_test.append(("Voting_Ensemble", voting_ensemble))
    
    # 백테스트 설정
    start_date_str = "2025-10-22"  # 최근 3일
    num_days = 3
    buy_threshold = 0.25
    sell_threshold = 0.35
    stop_loss = 1.5
    take_profit = 1.2
    
    print(f"\n백테스트 설정:")
    print(f"   기간: {start_date_str} ({num_days}일)")
    print(f"   매수 임계값: {buy_threshold}")
    print(f"   매도 임계값: {sell_threshold}")
    print(f"   손절: {stop_loss}%, 익절: {take_profit}%")
    
    # 모델 테스트 실행
    results = []
    
    for model_name, model in models_to_test:
        result = train_and_test_model(
            model_name, model, df_features, 
            start_date_str, num_days,
            buy_threshold, sell_threshold, stop_loss, take_profit
        )
        
        if result:
            results.append(result)
    
    # 결과 정리 및 출력
    print("\n" + "=" * 80)
    print("최종 결과 요약")
    print("=" * 80)
    
    if results:
        # 수익률 기준 정렬
        results.sort(key=lambda x: x['backtest_result']['total_return'], reverse=True)
        
        print(f"{'순위':<4} {'모델명':<25} {'학습정확도':<8} {'검증정확도':<8} {'수익률':<8} {'승률':<8} {'거래수':<6}")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            bt = result['backtest_result']
            print(f"{i:<4} {result['model_name']:<25} {result['train_score']:<8.3f} {result['test_score']:<8.3f} "
                  f"{bt['total_return']:<8.2f}% {bt['win_rate']:<8.1f}% {bt['total_trades']:<6}")
        
        # 최고 성능 모델
        best_model = results[0]
        print(f"\n최고 성능 모델: {best_model['model_name']}")
        print(f"   수익률: {best_model['backtest_result']['total_return']:.2f}%")
        print(f"   승률: {best_model['backtest_result']['win_rate']:.1f}%")
        print(f"   거래 수: {best_model['backtest_result']['total_trades']}회")
        
        # 수익 모델이 있는지 확인
        profitable_models = [r for r in results if r['backtest_result']['total_return'] > 0]
        if profitable_models:
            print(f"\nSUCCESS: 수익 모델 발견: {len(profitable_models)}개")
            for model in profitable_models:
                print(f"   - {model['model_name']}: {model['backtest_result']['total_return']:.2f}%")
        else:
            print(f"\nERROR: 수익 모델 없음. 모든 모델이 손실을 보였습니다.")
    
    else:
        print("ERROR: 모든 모델 테스트 실패!")

if __name__ == "__main__":
    main()
