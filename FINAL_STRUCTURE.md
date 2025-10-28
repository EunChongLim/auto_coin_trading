# 프로젝트 구조 정리 완료 (최종)

## 📁 디렉토리 구조

```
coin_trading/
├── Common/                          # 공유: 데이터 로드만
│   ├── download_data.py            # CSV 파일 로드
│   └── auto_download_data.py       # Upbit API 데이터 다운로드
│
├── model_v3/                       # 현재 사용 중인 모델
│   ├── Common/                     # v3 전용: 기존 지표
│   │   ├── indicators.py
│   │   └── multi_timeframe_features.py
│   ├── model/
│   │   └── lgb_model_v3.pkl       # RandomForest (10분 예측)
│   ├── main_v3.py                 # 실거래
│   ├── backtest_v3.py             # 백테스트
│   └── optimize_params.py         # 파라미터 최적화
│
└── model_v4/                       # A-E 규칙 모델
    ├── Common/                     # v4 전용: A-E 규칙 지표
    │   ├── indicators.py           # EMA, RSI, MACD, ATR, BB, VWAP
    │   └── multi_timeframe_features.py  # 레짐, pos60, trend_score 등
    ├── model/
    │   └── lgb_model_v4_enhanced.pkl  # LightGBM (3분 예측)
    ├── train_v4.py                # 모델 학습
    ├── main_v4.py                 # 실거래
    ├── backtest_v4.py             # 백테스트
    └── optimize_params_v4.py      # 파라미터 최적화
```

## 🔑 핵심 설계 원칙

### 1. 모델에 영향을 주는 것만 분리

- ✅ **분리됨**: `indicators.py`, `multi_timeframe_features.py`
  - 지표 계산 로직이 다르면 모델 예측 결과가 달라짐
  - 각 모델이 독립적인 버전 사용
- ✅ **공유됨**: `download_data.py`, `auto_download_data.py`
  - 데이터 로드만 수행, 모델에 영향 없음
  - 루트 `Common/`에서 공유

### 2. Import 구조

#### model_v3 (기존 지표)

```python
# 루트 Common (데이터 로드)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.download_data import load_daily_csv
from Common.auto_download_data import download_1m_data

# model_v3 Common (지표 계산)
sys.path.insert(0, os.path.dirname(__file__))
from Common.indicators import add_all_indicators
from Common.multi_timeframe_features import add_multi_timeframe_features
```

#### model_v4 (A-E 규칙)

```python
# 루트 Common (데이터 로드)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.download_data import load_daily_csv

# model_v4 Common (지표 계산)
sys.path.insert(0, os.path.dirname(__file__))
from Common.indicators import add_all_indicators
from Common.multi_timeframe_features import add_multi_timeframe_features
```

## 📊 model_v3 vs model_v4

### model_v3 (기존)

**지표:**

- SMA 기반
- RSI, MACD, Bollinger Bands
- 기본 멀티타임프레임

**전략:**

- ML 확률 기반 단순 매매
- 고정 손익비율

**모델:**

- RandomForest, 10분 예측
- 파라미터: buy=0.25, sell=0.35, stop=1.5%, take=1.2%

### model_v4 (A-E 규칙)

**지표 (A-C 규칙):**

1. **레짐(추세)**: `ema50_15m`, `ema200_15m`, `regime_bull`
2. **위치(구조)**: `pos60 = (close - low_60m) / (high_60m - low_60m)`
3. **유동성**: `volume_ratio > 1.3`
4. **변동성(스탑)**: `atr14_1m`, `atr_pct`
5. **변동성(브레이크아웃)**: `bb_width_pct` (p70/p30 기준)
6. **모멘텀**: `rsi`, `macd_hist`, `macd_hist_rising`
7. **MTF 추세 일치도**: `trend_score` (-4 ~ +4)
8. **세션 기준선**: `vwap_session`, `price_vs_vwap`

**전략 (B 규칙):**

- 룰 기반 매매 (구현 예정)
- ATR 동적 스탑/사이징
- 부분 청산 로직

**모델:**

- LightGBM, 3분 예측
- 학습: 2024년 데이터
- 테스트: 2025년 데이터

## 🚀 실행 방법

### model_v3

```bash
cd model_v3

# 실거래
python main_v3.py

# 백테스트
python backtest_v3.py 20250328 10

# 파라미터 최적화
python optimize_params.py 20250328 10 4
```

### model_v4

```bash
cd model_v4

# 모델 학습
python train_v4.py

# 실거래
python main_v4.py

# 백테스트
python backtest_v4.py 20250328 10

# 파라미터 최적화
python optimize_params_v4.py 20250328 10 4
```

## ⚠️ 주의사항

1. **독립성 보장**

   - v3와 v4는 서로 영향을 주지 않음
   - 각 모델 디렉토리 내에서 실행
   - 지표 계산 로직이 완전히 분리됨

2. **데이터 로드는 공유**

   - 루트 `Common/`의 데이터 로드 함수 사용
   - CSV 읽기/API 호출만 수행
   - 모델 결과에 영향 없음

3. **모델 파일 관리**

   - 각 모델의 `model/` 폴더에 위치
   - v3: `lgb_model_v3.pkl`
   - v4: `lgb_model_v4_enhanced.pkl`

4. **로그 파일 분리**
   - v3: `model_v3_trading_log.txt`
   - v4: `model_v4_trading_log.txt`

## 🎯 다음 단계 (model_v4 완성)

1. ✅ **A-C 규칙 지표 구현** (완료)

   - EMA, ATR, VWAP, pos60, trend_score 등

2. ⏳ **B 규칙 전략 구현** (예정)

   - 룰 기반 매수/매도 조건
   - ATR 동적 스탑로스
   - 부분 청산 로직

3. ⏳ **모델 학습 및 테스트** (예정)

   - 2024년 데이터로 학습
   - 2025년 데이터로 백테스트
   - 파라미터 최적화

4. ⏳ **E 규칙 검증** (예정)
   - 룩어헤드 방지 확인
   - 슬리피지/수수료 반영
   - 피처 기여도 분석

## 📝 변경 이력

- **2025-10-27**: 최종 구조 확정
  - 데이터 로드는 루트 Common 공유
  - 지표 계산은 각 모델 Common에 분리
  - v4에 A-E 규칙 지표 구현 완료
