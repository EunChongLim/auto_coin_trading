"""
ML 모델 로드 및 예측 모듈
"""

import joblib
import numpy as np
import pandas as pd


class MLSignalModel:
    """
    학습된 LightGBM 모델을 로드하여 예측하는 클래스
    """
    
    def __init__(self, model_path="model/lgb_model.pkl"):
        """
        모델 로드
        
        Args:
            model_path: 모델 파일 경로
        """
        print(f"📦 모델 로드 중: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.version = model_data.get('version', 'unknown')
        self.train_date = model_data.get('train_date', 'unknown')
        
        print(f"✅ 모델 로드 완료 (버전: {self.version}, 학습일: {self.train_date})")
        print(f"   특징 수: {len(self.feature_cols)}개")
    
    def predict_proba(self, features):
        """
        예측 확률 반환
        
        Args:
            features: 특징 벡터 (DataFrame, Series 또는 numpy array)
        
        Returns:
            float or np.array: 상승 확률 (0~1)
        """
        # Series를 numpy array로 변환
        if isinstance(features, pd.Series):
            features = features.values
        
        # DataFrame인 경우 feature_cols 순서대로 선택
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_cols].values
        
        # 단일 샘플 처리 (1D array)
        if features.ndim == 1:
            features_array = features.reshape(1, -1)
            prob = self.model.predict(features_array)[0]
        else:
            # 다중 샘플 처리 (2D array)
            prob = self.model.predict(features)
        
        return prob
    
    def predict(self, features, threshold=0.5):
        """
        예측 라벨 반환
        
        Args:
            features: 특징 벡터
            threshold: 분류 임계값
        
        Returns:
            int or np.array: 예측 라벨 (0 or 1)
        """
        prob = self.predict_proba(features)
        return (prob >= threshold).astype(int)


if __name__ == "__main__":
    print("✅ ml_model.py 모듈 로드 완료")
    print("🤖 MLSignalModel 클래스 사용 가능")

