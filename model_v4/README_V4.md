# model_v4: A-E 규칙 기반 스캘핑 전략

## 🎯 개요

**model_v4**는 1-5분 스캘핑에 최적화된 A-E 규칙 기반 자동매매 시스템입니다.

### 핵심 특징

- ✅ **A-C 규칙**: 8개 핵심 지표 (레짐, 위치, 유동성, 변동성, 모멘텀, MTF, 세션)
- ✅ **B 규칙**: 룰 기반 매수/매도 전략 (ML 보조)
- ✅ **ATR 동적 스탑**: 시장 변동성에 맞춘 스탑로스
- ✅ **부분청산**: +1R 도달 시 50% 청산
- ✅ **E 규칙**: 룩어헤드 방지, 봉 마감가 체결

## 📊 A규칙: 핵심 지표 (8개)

### 1. 레짐(추세)

- `ema50_15m > ema200_15m` → 상승 레짐
- `ema50_15m < ema200_15m` → 하락 레짐
- `regime_bull` = 1 or 0

### 2. 위치(구조)

- `pos60 = (close - low_60m) / (high_60m - low_60m)`
- 롱: `pos60 > 0.35` (하단 회피)
- 숏: `pos60 < 0.65` (상단 회피)

### 3. 유동성

- `volume_ratio = volume / volume_ma20`
- 필터: `volume_ratio > 1.3`

### 4. 변동성 (스탑/사이징)

- `atr14_1m` (ATR 14기간)
- `atr_pct` (ATR 퍼센트)
- 용도: 스탑로스, 포지션 사이징

### 5. 변동성 (브레이크아웃)

- `bb_width_pct` (볼린저 밴드 폭 퍼센트)
- 브레이크아웃: `bb_width_pct > 2.5%`
- 리버전: `bb_width_pct < 1.5%`

### 6. 모멘텀

- `rsi14_1m` (RSI 14기간)
- `macd_hist_1m` (MACD 히스토그램)
- `macd_hist_rising` (MACD 상승 여부)

### 7. MTF 추세 일치도

- `trend_score = Σ sign(ema20 - ema50)` for 1m/5m/15m/60m
- 범위: -4 ~ +4
- 롱: `trend_score >= +2`
- 숏: `trend_score <= -2`

### 8. 세션 기준선

- `vwap_session` (금일 세션 VWAP)
- `price_vs_vwap` (가격 vs VWAP 비율)
- 리클레임: `price_vs_vwap > 0`

## 🎲 B규칙: 전략 로직

### 롱(매수) 조건

```python
1. 레짐: ema50_15m > ema200_15m AND trend_score >= +2
2. 유동성: volume_ratio > 1.3
3. 위치: pos60 > 0.35
4. 브레이크아웃: bb_width_pct > 2.5% OR price_vs_vwap > 0
5. 트리거: rsi > 50 AND macd_hist_rising == 1
6. (보조) ML: prob_up >= 0.25
```

### 숏(매도) 조건

```python
1. 레짐: ema50_15m < ema200_15m AND trend_score <= -2
2. 유동성: volume_ratio > 1.3
3. 위치: pos60 < 0.65
4. 브레이크아웃: bb_width_pct > 2.5% OR price_vs_vwap < 0
5. 트리거: rsi < 50 AND macd_hist_rising == 0
6. (보조) ML: prob_down >= 0.35
```

### 리스크 관리

**ATR 기반 스탑로스:**

```python
stop_loss = entry_price - (1.2 * atr)  # 롱
stop_loss = entry_price + (1.2 * atr)  # 숏
```

**ATR 기반 포지션 사이징:**

```python
risk_amount = equity * 1.0%
one_r = 1.2 * atr
position_size = risk_amount / one_r
```

**부분청산:**

```python
if profit >= 1R:
    sell 50% of position
    trail remaining with ATR or 60m high/low
```

## 📁 파일 구조

```
model_v4/
├── Common/
│   ├── indicators.py              # A규칙 지표 계산
│   ├── multi_timeframe_features.py  # 멀티타임프레임 특징
│   └── strategy_rules.py          # B규칙 전략 엔진
├── model/
│   └── lgb_model_v4_enhanced.pkl  # LightGBM 모델 (3분 예측)
├── train_v4.py                    # 모델 학습
├── backtest_v4.py                 # 백테스트 (B규칙 적용)
├── main_v4.py                     # 실거래 (B규칙 적용)
└── optimize_params_v4.py          # 파라미터 최적화
```

## 🚀 사용 방법

### 1. 모델 학습 (2024년 데이터)

```bash
cd model_v4
python train_v4.py
```

- 입력: `data/daily_1m/` (1분봉)
- 기간: 2024년 1월 ~ 12월 (80일 샘플링)
- 라벨: 3분 후 가격 (Down: -0.1%, Up: +0.2%)
- 출력: `model/lgb_model_v4_enhanced.pkl`

### 2. 백테스트 (2025년 데이터)

```bash
cd model_v4
python backtest_v4.py 20250328 10
```

- B규칙 전략 적용
- ATR 동적 스탑
- 부분청산 포함
- ML 보조 신호

### 3. 파라미터 최적화

```bash
cd model_v4
python optimize_params_v4.py 20250328 10 4
```

- 최적화 대상:
  - `ml_buy_threshold` (0.20, 0.25, 0.30)
  - `ml_sell_threshold` (0.30, 0.35, 0.40)
  - `atr_stop_multiplier` (1.0, 1.2, 1.5, 2.0)
  - `risk_pct` (0.5, 1.0, 1.5%)

### 4. 실거래

```bash
cd model_v4
python main_v4.py
```

- B규칙 전략 실행
- ATR 동적 스탑 적용
- 부분청산 자동 실행
- 로그: `model_v4_trading_log.txt`

## ⚙️ 파라미터

### 기본값 (권장)

```python
ml_buy_threshold = 0.25      # ML 매수 확률
ml_sell_threshold = 0.35     # ML 매도 확률
atr_stop_multiplier = 1.2    # ATR 스탑 배수 (k)
risk_pct = 1.0               # 위험률 (%)
use_ml = True                # ML 보조 신호 사용
use_partial_exit = True      # 부분청산 사용
```

### 보수적 설정

```python
ml_buy_threshold = 0.30
ml_sell_threshold = 0.30
atr_stop_multiplier = 1.0
risk_pct = 0.5
```

### 공격적 설정

```python
ml_buy_threshold = 0.20
ml_sell_threshold = 0.40
atr_stop_multiplier = 1.5
risk_pct = 1.5
```

## 📈 백테스트 결과 (예정)

### 학습 데이터

- 기간: 2024년 (80일)
- 샘플: ~110,000개
- 라벨 분포: Down 12.9%, Sideways 83.1%, Up 4.0%

### 테스트 데이터

- 기간: 2025년 3월 28일 ~ 4월 6일 (10일)
- 전략: B규칙 (룰 기반 + ML 보조)
- 결과: TBD

## ⚠️ 주의사항

### 1. 룩어헤드 방지

- ✅ 상위 타임프레임은 마감봉만 사용
- ✅ Forward Fill로 1분봉에 매핑
- ✅ 미마감 봉 금지

### 2. 백테스트 체결

- ✅ 봉 마감가 기준 체결
- ✅ 수수료 0.05% 반영
- ✅ 슬라이딩 윈도우 방식 (window_size=1800)

### 3. 과적합 방지

- ✅ 2024년 학습, 2025년 테스트
- ✅ 80일 랜덤 샘플링
- ✅ 단순한 룰 기반 전략

### 4. 실거래 주의

- ⚠️ 백테스트 검증 필수
- ⚠️ 소액으로 시작
- ⚠️ ATR 스탑 신뢰
- ⚠️ 부분청산 활용

## 🔄 다음 단계

1. ✅ A-C 규칙 지표 구현
2. ✅ B 규칙 전략 구현
3. ✅ ATR 동적 스탑/사이징
4. ✅ 부분청산 로직
5. ⏳ 모델 학습 (진행 중)
6. ⏳ 백테스트 검증
7. ⏳ 파라미터 최적화
8. ⏳ E 규칙 검증

## 📝 변경 이력

- **2025-10-27**: model_v4 생성
  - A-E 규칙 전면 구현
  - B규칙 전략 엔진 추가
  - ATR 동적 스탑/사이징
  - 부분청산 로직
  - 독립 Common 모듈

---

**면책:** 이 프로그램은 시뮬레이션 목적이며, 실거래 시 본인 책임 하에 사용하세요.
