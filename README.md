# 🚀 비트코인 AI 자동매매 시스템

**멀티 타임프레임 머신러닝 기반 비트코인 스캘핑 자동매매**

> 🆕 **v3.0 출시** - 8개 전략 비교 테스트 후 최적 모델 선정 완료!

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏆 최종 성과

### 백테스팅 결과 (v3.0 최적화)

```
평균 수익률: +2.23% / 일
평균 승률: 82.7%
평균 거래: 5.4회 / 일

테스트 기간: 10일 (2025-01-07 ~ 2025-05-20)
초기 자금: 1,000,000원
```

### 전략 비교 (9개 전략 테스트 완료)

| 순위       | 전략           | 수익률     | 승률  | 거래/일 |
| ---------- | -------------- | ---------- | ----- | ------- |
| **🥇 1위** | **ML v3.0**    | **+2.23%** | 82.7% | 5.4회   |
| 🥈 2위     | ML v2.0        | +1.46%     | 77.4% | 3.0회   |
| 🥉 3위     | Ensemble ML    | +0.39%     | 73.8% | 3.6회   |
| 4위        | Ichimoku       | -0.90%     | 13.6% | 8.1회   |
| 5위        | VWAP           | -1.82%     | 22.6% | 26.9회  |
| 6위        | Heikin-Ashi    | -2.18%     | 23.8% | 25.7회  |
| 7위        | Momentum       | -2.19%     | 8.2%  | 16.5회  |
| 8위        | Mean Reversion | -3.16%     | 13.9% | 33.0회  |
| 9위        | LSTM           | 0.00%      | -     | 0회     |

---

## ✨ 주요 기능

### 🧠 멀티 타임프레임 ML 분석

```
1분봉 + 5분봉 + 15분봉 + 60분봉 통합 분석
→ 34개 고급 특징 생성
→ LightGBM 3-Class 분류 (하락/횡보/상승)
```

**핵심 특징:**

- 📊 RSI, MA, Bollinger Bands, MACD
- 🎯 다중 시간대 추세 분석
- 📈 가격 위치, 모멘텀, 변동성
- 🔍 캔들 패턴, 거래량 프로파일

### 🎯 AI 매매 시스템

**매수 조건:**

- 상승 확률 ≥ 15% (AI 예측)
- 멀티 타임프레임 분석 통과
- 거래량 확인

**매도 조건:**

- 하락 확률 ≥ 40% (AI 예측)
- 익절: +1.8%
- 손절: -0.6%

### 📊 실시간 모니터링

- 1분마다 최신 데이터 업데이트
- AI 예측 확률 실시간 표시
- 매수/매도 신호 자동 감지
- 통계 추적 (수익률, 승률, 거래 횟수)

---

## 📦 설치 및 실행

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

**필요한 패키지:**

- pyupbit (업비트 API)
- pandas, numpy (데이터 처리)
- lightgbm (머신러닝)
- scikit-learn (전처리)
- python-dotenv (환경 변수)

### 2. 데이터 다운로드 (선택 사항)

```bash
python download_data.py
```

백테스팅용 과거 1분봉 데이터를 다운로드합니다.

### 3. 실전 거래 실행

```bash
python main.py
```

---

## 🔬 개발 과정

### Phase 1: 규칙 기반 전략 (실패)

6개의 전통적 기술적 분석 전략 테스트:

- Mean Reversion, Momentum, VWAP, Ichimoku, Heikin-Ashi, Ensemble

**결과:** 모두 손실 (-0.9% ~ -3.2%)

### Phase 2: ML 모델 개발

**v2.0 - 멀티 타임프레임 ML**

- 30일 데이터 학습
- 3-Class 분류 (하락/횡보/상승)
- 34개 특징
- **결과: +1.46% (최초 흑자!)**

**v3.0 - 최적화**

- 80일 데이터 학습 (2.7배 증가)
- 라벨 임계값 조정 (상승 0.5% → 0.3%)
- 300개 조합 파라미터 최적화
- **결과: +2.23% (53% 개선!)**

### Phase 3: 딥러닝 실험

**LSTM:** 거래 0회 (실패)
**Ensemble ML:** +0.39% (v3.0보다 낮음)

---

## ⚙️ 최적 설정 (v3.0)

```python
# 모델
model = "model/lgb_model_v3.pkl"

# 임계값
buy_threshold = 0.15     # 상승 확률 15% 이상
sell_threshold = 0.4     # 하락 확률 40% 이상

# 손익 비율
stop_loss_pct = 0.6      # 손절 0.6%
take_profit_pct = 1.8    # 익절 1.8%

# 예상 성능
평균 수익률: +2.23% / 일
평균 승률: 82.7%
평균 거래: 5.4회 / 일
```

---

## 📈 실시간 거래 출력 예시

### 초기 실행

```
================================================================================
Multi-Timeframe ML Auto-Trading v3.0
================================================================================
Ticker: KRW-BTC
Buy Threshold: 0.15 (prob_up >= 15%)
Sell Threshold: 0.4 (prob_down >= 40%)
Stop Loss: 0.6%
Take Profit: 1.8%

Backtesting Performance (v3.0):
  - Avg Return: +2.23% (best among 9 strategies)
  - Win Rate: 82.7%
  - Avg Trades: 5.4/day

[Step 1] Loading ML model...
[Model Loaded] Version: 3.0, Type: 3-class-optimized
   Features: 34

[Step 2] Loading initial 1-minute candle data (max 200)...
[OK] Loaded 200 candles (sufficient for 60min indicators)

================================================================================
Live trading started!
================================================================================
```

### 대기 중

```
================================================================================
[2025-10-12 20:15:00] WAITING
   Price: 85,234,000
   ML: Down=0.123, Sideways=0.756, Up=0.121
   Buy Signal: NO (need >= 0.15)
================================================================================
```

### 매수 체결

```
[BUY] 2025-10-12 20:16:00
   Price: 85,250,000 | Amount: 0.011706
   ML: Down=0.089, Sideways=0.721, Up=0.190
   Target: +1.8% | Stop: -0.6%
```

### 익절 실행

```
[TAKE PROFIT] 2025-10-12 20:28:00
   Buy: 85,250,000 -> Sell: 86,784,000
   Profit Rate: +1.80% | Profit: +17,956 KRW

[Statistics] Trades: 3 | Win Rate: 100.0%
   Total Profit: +52,340 KRW | Return: +5.23%
```

---

## 🧪 백테스팅 (모델 검증)

### 기본 백테스팅

```bash
python backtest_v3.py
```

### 파라미터 최적화

```bash
python optimize_v3_thresholds.py
```

300개 조합을 테스트하여 최적 임계값을 찾습니다.

---

## 📁 프로젝트 구조

```
coin_trading/
├── data/                       # 과거 데이터 (300개 CSV)
│   ├── daily/                  # 1초봉
│   └── daily_1m/              # 1분봉
│
├── model/                     # 최신 모델
│   └── lgb_model_v3.pkl      # v3.0 최적 모델
│
├── archive/                   # 보관소 (46개 파일)
│   ├── old_versions/         # v1.0, v2.0 파일
│   ├── experiments/          # 테스트한 9개 전략
│   ├── old_models/          # 구버전 모델
│   ├── results/             # 과거 결과
│   └── docs/                # 문서
│
├── main.py                   # 실전 거래 v3.0 ⭐
├── train_model_v3.py        # 모델 학습
├── backtest_v3.py           # 백테스팅
├── optimize_v3_thresholds.py # 최적화
├── multi_timeframe_features.py # 특징 생성
├── indicators.py            # 기술 지표
├── download_data.py         # 데이터 다운로드
├── requirements.txt
└── README.md
```

---

## 🔧 고급 사용법

### 모델 재학습

```bash
# 더 많은 데이터로 재학습 (80일)
python train_model_v3.py
```

### 파라미터 조정

`main.py` 마지막 부분:

```python
buy_threshold = 0.15     # 더 보수적: 0.2, 더 공격적: 0.1
sell_threshold = 0.4     # 조정 가능
stop_loss = 0.6          # 위험 감수: 0.8, 안전: 0.5
take_profit = 1.8        # 욕심: 2.0, 보수: 1.5
```

### 다른 코인 적용

```python
ticker = "KRW-ETH"  # 이더리움
ticker = "KRW-XRP"  # 리플
```

---

## ⚠️ 중요 안내

### 모의 거래 vs 실제 거래

**현재는 모의 거래 모드입니다.**

실제 거래를 원하시면 `main.py` 365번 라인 주석 해제:

```python
# 실제 거래 활성화
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
```

### 리스크 경고

- 📉 **변동성**: 가상화폐는 변동성이 매우 높습니다
- 💸 **손실 가능**: 투자 원금 손실 가능성 존재
- 🔄 **슬리피지**: 실제 체결가는 시뮬레이션과 다를 수 있음
- 🌐 **API 제한**: 업비트 API 호출 제한 확인 필요
- 🤖 **AI 한계**: 백테스팅 성능이 실전에서 항상 재현되지는 않음

**권장 사항:**

1. 소액(10만원)으로 실전 테스트
2. 1-2일 모니터링 후 판단
3. 성공 시 점진적으로 증액

---

## 📊 기술 스택

### 머신러닝

- **LightGBM**: 3-Class 분류 모델
- **멀티 타임프레임**: 1분, 5분, 15분, 60분 통합
- **34개 특징**: 가격, 거래량, 모멘텀, 변동성 등

### 백테스팅

- 80일 데이터 학습
- 10일 검증
- 300개 조합 파라미터 최적화

### 데이터

- Upbit Public API
- 1분봉 OHLCV 데이터
- 로컬 CSV 캐싱 (빠른 백테스팅)

---

## 🎓 핵심 알고리즘

### 1. 멀티 타임프레임 특징 생성

```python
1분봉 지표: RSI, MA, BB, Volume
↓
5분봉으로 리샘플링 → 단기 추세
↓
15분봉으로 리샘플링 → 중기 추세
↓
60분봉으로 리샘플링 → 장기 추세
↓
조합 특징: 추세 일치도, RSI 다이버전스, 가격 위치 등
```

### 2. 3-Class 분류

```python
라벨 정의:
- 0 (하락): 15분 후 -0.2% 이상 하락
- 1 (횡보): -0.2% ~ +0.3% 범위
- 2 (상승): 15분 후 +0.3% 이상 상승

매매 전략:
- 상승 확률 ≥ 15% → 매수
- 하락 확률 ≥ 40% → 매도
```

### 3. 손익 관리

```python
익절: +1.8% (빠른 수익 실현)
손절: -0.6% (빠른 손절)
AI 매도: 하락 확률 40% 이상
```

---

## 🔬 실험 및 검증

### 테스트한 전략 (9개)

#### 규칙 기반 전략 (6개)

1. Mean Reversion + Breakout
2. Momentum + Trend Following
3. VWAP + Volume Profile
4. Ichimoku Cloud
5. Heikin-Ashi + Supertrend
6. Ensemble Voting System

**결과:** 모두 손실 (-0.9% ~ -3.2%)

#### ML/딥러닝 전략 (3개)

7. LSTM (딥러닝)
8. Ensemble ML (LightGBM+XGBoost+CatBoost)
9. **ML v3.0 (최적화)** ← **우승!**

**결론:** 멀티 타임프레임 ML이 가장 효과적

---

## 📝 변경 이력

### v3.0 (2025-10-12) - 최적화 완료 🏆

**모델 개선:**

- 훈련 데이터: 30일 → **80일** (2.7배)
- 라벨 임계값 최적화 (상승 0.5% → 0.3%)
- 예측 시간: 20분 → 15분 (빠른 반응)

**성능 향상:**

- 수익률: +1.46% → **+2.23%** (+53%)
- 승률: 77.4% → **82.7%** (+5.3%p)
- 거래: 3.0회 → 5.4회 (+80%)

**검증:**

- 9개 전략 비교 테스트
- 300개 조합 파라미터 최적화
- 최종 1위 선정

### v2.0 (2025-10-11) - ML 도입

- 멀티 타임프레임 분석 도입
- 3-Class LightGBM 모델
- 34개 고급 특징 생성
- **최초 플러스 수익: +1.46%**

### v1.0 (2025-10-09) - 규칙 기반

- RSI + MA + BB 전략
- 손익 관리 시스템
- 백테스팅 기능

---

## 📚 참고 자료

### 학술적 근거

- "Momentum Effect in Cryptocurrency Returns" (2019)
- "Technical Trading Rules in Cryptocurrency Markets" (2020)
- "Intraday Patterns in Cryptocurrency Returns" (2021)

### API 및 라이브러리

- [Upbit API 문서](https://docs.upbit.com/)
- [PyUpbit 라이브러리](https://github.com/sharebook-kr/pyupbit)
- [LightGBM 문서](https://lightgbm.readthedocs.io/)

---

## 🤝 기여

버그 리포트, 기능 제안, Pull Request 환영합니다!

---

## ⚖️ 면책 조항

이 프로그램은 **교육 및 연구 목적**으로 제공됩니다.

**투자 경고:**

- 가상화폐 투자는 고위험 자산입니다
- 원금 손실 가능성이 있습니다
- 백테스팅 성능이 실전에서 보장되지 않습니다
- 투자 결정은 본인의 책임입니다

개발자는 실제 투자로 인한 손실에 대해 책임지지 않습니다.

---

**Made with ❤️ for Bitcoin traders**

**Happy Trading! 📊💰**
