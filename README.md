# 비트코인 자동매매 프로젝트

## 📁 프로젝트 구조

```
coin_trading/
├── Common/                    # 공유: 데이터 로드
│   ├── download_data.py
│   └── auto_download_data.py
│
├── model_v3/                 # 현재 사용 중 (기존 지표)
│   ├── Common/              # v3 전용 지표
│   ├── model/
│   ├── main_v3.py
│   ├── backtest_v3.py
│   └── optimize_params.py
│
├── model_v4/                 # A-E 규칙 (Enhanced)
│   ├── Common/              # v4 전용 지표
│   ├── model/
│   ├── train_v4.py
│   ├── main_v4.py
│   ├── backtest_v4.py
│   └── optimize_params_v4.py
│
└── data/
    ├── daily/               # 1초봉 데이터
    └── daily_1m/            # 1분봉 데이터
```

## 🎯 모델 비교

### model_v3 (현재 사용 중)

- **알고리즘:** RandomForest Classifier
- **예측:** 10분 후 가격
- **지표:** SMA, RSI, MACD, BB (기존)
- **전략:** ML 확률 기반 단순 매매
- **파라미터:** buy=0.25, sell=0.35, stop=1.5%, take=1.2%
- **성능:** +1.69% (10일), 60.0% 승률

### model_v4 (A-E 규칙 구현)

- **알고리즘:** LightGBM Classifier
- **예측:** 3분 후 가격 (스캘핑)
- **지표 (A-C 규칙):**
  1. 레짐: `ema50_15m`, `ema200_15m`
  2. 위치: `pos60` (60분 high-low)
  3. 유동성: `volume_ratio > 1.3`
  4. 변동성(스탑): `atr`, `atr_pct`
  5. 변동성(브레이크아웃): `bb_width_pct`
  6. 모멘텀: `rsi`, `macd_hist`
  7. MTF 추세: `trend_score` (-4 ~ +4)
  8. 세션: `vwap_session`
- **전략 (B 규칙):** 룰 기반 + ATR 동적 스탑 (구현 예정)
- **상태:** 지표 구현 완료, 전략 구현 중

## 🚀 사용 방법

### 1. model_v3 (현재 사용 중)

**실거래:**

```bash
cd model_v3
python main_v3.py
```

**백테스트:**

```bash
cd model_v3
python backtest_v3.py 20250328 10
# 사용법: python backtest_v3.py YYYYMMDD [num_days]
```

**파라미터 최적화:**

```bash
cd model_v3
python optimize_params.py 20250328 10 4
# 사용법: python optimize_params.py YYYYMMDD [num_days] [max_workers]
```

### 2. model_v4 (A-E 규칙)

**모델 학습:**

```bash
cd model_v4
python train_v4.py
```

**백테스트:**

```bash
cd model_v4
python backtest_v4.py 20250328 10
```

**파라미터 최적화:**

```bash
cd model_v4
python optimize_params_v4.py 20250328 10 4
```

**실거래:**

```bash
cd model_v4
python main_v4.py
```

## 🔑 핵심 설계 원칙

### 1. 완전한 독립성

- **각 모델이 독립적인 지표 계산 로직 사용**
  - `model_v3/Common/` - 기존 지표
  - `model_v4/Common/` - A-E 규칙 지표
- **데이터 로드만 공유** (`Common/`)
  - CSV 파일 읽기, API 호출
  - 모델 결과에 영향 없음

### 2. 데이터 분리

- **학습:** 2024년 데이터
- **테스트:** 2025년 데이터
- **Look-ahead Bias 방지**

### 3. 백테스팅

- **슬라이딩 윈도우 방식** (실거래와 동일)
- **Window size:** 1800개 (30시간)
- **수수료:** 0.05% 반영

## 📊 A-E 규칙 (model_v4)

### A. 최소 조합 (8개 핵심 지표)

1. **레짐(추세)**: `ema50_15m > ema200_15m` (롱), `<` (숏)
2. **위치(구조)**: `pos60 = (close - low_60m) / (high - low)60m`
3. **유동성**: `vol_1m > 1.3 * vol_1m_ma20`
4. **변동성(스탑)**: `atr14_1m` (포지션 사이징)
5. **변동성(브레이크아웃)**: `bb_width_pct` (p70/p30 기준)
6. **모멘텀**: `rsi14_1m`, `macd_hist_1m`
7. **MTF 추세 일치도**: `trend_score = Σ sign(ema20-ema50)` for 1/5/15/60m
8. **세션 기준선**: `vwap_session`

### B. 신호 규칙 (구현 예정)

**롱 시나리오:**

- 레짐: `ema50_15m > ema200_15m AND trend_score >= +2`
- 유동성: `vol_1m > 1.3 * vol_1m_ma20`
- 위치: `pos60 > 0.35`
- 브레이크아웃: `bb_width_pct >= p70` or VWAP 상향 리클레임
- 트리거: `rsi > 50` AND `macd_hist 기울기↑`

**리스크:**

- 스탑: `entry - 1.2 * atr14_1m`
- 사이징: `(equity * 0.5~1.0%) / (1.2 * atr)`
- 청산: +1R 50% 부분청산 → ATR 트레일

### C. 컬럼/파라미터

- EMA(1m): 5, 20, 50, 200
- EMA(MTF): 20/50 for 5m/15m/60m
- RSI/ATR/BB: 14/14/20
- 거래량: ma20
- 구조: high_60m, low_60m, pos60
- VWAP: session

### D. 유지/제거

- ✅ **유지**: RSI, MACD hist, ATR, BB폭, EMA, pos60, VWAP, trend_score
- ⚠️ **덜 사용**: 캔들 패턴 (1m 노이즈)
- ❌ **제거**: SMA/EMA 중복

### E. 검증 체크리스트

- ✅ 룩어헤드 방지 (상위 TF 마감봉만)
- ✅ 웜업 (min_periods=window)
- ✅ 백테스트 체결 (봉 마감가 기준)
- ✅ 슬리피지/수수료 (0.05%)
- ⏳ 피처 기여도 (Permutation Importance)

## 📦 환경 설정

```bash
pip install -r requirements.txt
```

`.env` 파일에 Upbit API 키 설정:

```
UPBIT_ACCESS_KEY=your_access_key
UPBIT_SECRET_KEY=your_secret_key
```

## ⚠️ 주의사항

1. **모델 독립성**

   - v3와 v4는 서로 영향을 주지 않음
   - 각 모델 디렉토리 내에서 실행 필수

2. **백테스트 필수**

   - 실거래 전 반드시 백테스트로 검증
   - 최소 7일 이상 테스트 권장

3. **과적합 방지**

   - 학습/테스트 데이터 분리 엄수
   - 파라미터 최적화 시 여러 기간 검증

4. **데이터 공유**
   - 루트 `Common/`은 데이터 로드만
   - 지표 계산은 각 모델 `Common/` 사용

## 📝 다음 단계

### model_v4 완성

1. ✅ A-C 규칙 지표 구현
2. ⏳ B 규칙 전략 구현 (룰 기반 매매)
3. ⏳ 모델 학습 및 백테스트
4. ⏳ E 규칙 검증 및 최적화

---

**면책:** 이 프로그램은 시뮬레이션 목적이며, 실거래 시 본인 책임 하에 사용하세요.
