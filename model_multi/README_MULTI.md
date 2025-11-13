# 🚀 멀티 타임프레임 트레이딩 모델

## 📊 개요

**Model Multi**는 1분봉, 1시간봉, 4시간봉을 동시에 분석하여 매매 결정을 내리는 멀티 타임프레임 머신러닝 모델입니다.

### 핵심 아이디어

```
4시간봉: "큰 강의 흐름" 🌊 → 장기 필터 (상승장/하락장 판단)
    ↓
1시간봉: "물살의 세기" 💧 → 중기 컨텍스트 (조정 완료 여부)
    ↓
1분봉: "파도 타는 타이밍" 🏄 → 진입/청산 시그널 (정확한 타이밍)

→ 세 타임프레임이 일치할 때 강력한 매매 신호!
```

---

## 📁 프로젝트 구조

```
model_multi/
├── Common/
│   ├── data_loader.py              # 멀티 타임프레임 데이터 로드 (Upbit API)
│   ├── features_multi.py           # 멀티 타임프레임 Feature 계산
│   └── timeframe_resampler.py      # (미사용, 리샘플링 방식)
├── model/
│   └── lgb_multi_timeframe.pkl     # 학습된 LightGBM 모델
├── train_multi.py                  # Walk-Forward 학습
├── backtest_multi.py               # 백테스트
├── optimize_params_multi.py        # 파라미터 최적화
└── README_MULTI.md                 # 이 파일
```

---

## 🎯 각 타임프레임의 역할

| 타임프레임 | 역할 | 업데이트 빈도 | Feature 개수 | 중요도 |
|-----------|------|-------------|------------|--------|
| **1분봉** | 진입/청산 시그널 | 매 1분 (1,440회/일) | 27개 | 높음 ⭐⭐⭐ |
| **1시간봉** | 중기 컨텍스트 | 매 1시간 (24회/일) | 25개 | 중간 ⭐⭐ |
| **4시간봉** | 장기 필터 | 매 4시간 (6회/일) | 6개 (단순) | 낮음 ⭐ |
| **관계** | 타임프레임 간 정렬 | 매 1분 | 10개 | 중간 ⭐⭐ |

### **1분봉 (Trigger)**
- **역할**: 정확한 진입/청산 타이밍 포착
- **Feature**: RSI, MACD, 거래량 급증, 볼린저 밴드, 단기 변동성
- **예시**: "지금 당장 사야 하는가? 팔아야 하는가?"

### **1시간봉 (Context)**
- **역할**: 현재 시장 상황 파악
- **Feature**: 중기 추세, 이동평균 위치, 지지/저항선
- **예시**: "조정이 끝났는가? 건강한 상승인가?"

### **4시간봉 (Filter)**
- **역할**: 큰 방향성 제공
- **Feature**: EMA 추세, 주요 지지/저항, 장기 모멘텀
- **예시**: "지금 상승장인가, 하락장인가?"

---

## 📡 데이터 수집 방식

### **각 타임프레임을 Upbit API에서 직접 조회**

```python
# ✅ 올바른 방식 (현재 구현)
df_1m = load_from_api(timeframe='1m')    # /v1/candles/minutes/1
df_1h = load_from_api(timeframe='1h')    # /v1/candles/minutes/60
df_4h = load_from_api(timeframe='4h')    # /v1/candles/minutes/240

# ❌ 잘못된 방식 (사용 안함)
df_1m = load_from_api(timeframe='1m')
df_1h = resample(df_1m, '1h')  # 1분봉을 단순 집계
df_4h = resample(df_1m, '4h')  # 1분봉을 단순 집계
```

**왜 직접 조회?**
1. **정확성**: Upbit 공식 OHLCV 데이터
2. **신뢰성**: 리샘플링 오차 제거
3. **실시간 일치**: 실거래 시 동일한 데이터 사용

---

## 📈 Feature 구성 (총 68개)

### **1분봉 Feature (27개)**
```python
# 가격 변화
1m_price_change_1, 1m_price_change_3, 1m_price_change_5, 1m_price_change_10
1m_momentum_5, 1m_momentum_10

# 거래량
1m_volume_change_1, 1m_volume_change_5, 1m_volume_ratio, 1m_volume_spike

# RSI
1m_rsi, 1m_rsi_oversold, 1m_rsi_overbought

# MACD
1m_macd, 1m_macd_signal, 1m_macd_hist

# 볼린저 밴드
1m_bb_position

# 이동평균
1m_ma_cross_5_10, 1m_price_above_ma20

# 변동성
1m_atr, 1m_volatility_5, 1m_volatility_10, 1m_volatility_ratio, 1m_high_low_range

# 시간
1m_hour, 1m_active_hours
```

### **1시간봉 Feature (25개)**
- 1분봉과 동일한 지표, 시간 정보 제외
- `1h_` 접두사 사용

### **4시간봉 Feature (6개, 단순화)**
```python
4h_ema_trend              # EMA 50 vs 200 (상승=1, 하락=-1)
4h_trend_strength         # 추세 강도 (-1 ~ +1)
4h_resistance_distance    # 저항선까지 거리 (%)
4h_support_distance       # 지지선까지 거리 (%)
4h_momentum_10            # 10캔들 모멘텀
4h_momentum_20            # 20캔들 모멘텀
```

### **타임프레임 간 관계 Feature (10개)**
```python
trend_alignment           # 세 추세 일치 여부 (1=일치, -1=불일치)
rsi_divergence_1m_1h      # 1분 RSI - 1시간 RSI
rsi_divergence_1h_4h      # 1시간 RSI - 중립값
price_position_vs_1h      # 현재가 vs 1시간봉 가격 (%)
price_position_vs_4h      # 현재가 vs 4시간봉 가격 (%)
volume_ratio_1m_vs_1h     # 1분 거래량 / 1시간 평균
volatility_ratio_1m_1h    # 1분 변동성 / 1시간 변동성
trend_strength_diff       # 4시간 - 1분 추세 강도 차이
bb_alignment              # 볼린저 밴드 위치 일치
4h_filter_strength        # 4시간 필터 강도
```

---

## 🎓 학습 방법

### **Walk-Forward 학습**

```python
# 규칙 준수
학습: 2024년 전체 (20240101 ~ 20241231)
백테스트1: 2022년 전체 (하락장 검증)
백테스트2: 2025년 최근 (최신 데이터 검증)
```

### **핵심 원칙**
1. **슬라이딩 윈도우 방식**
   - 1분봉: 최근 100개
   - 1시간봉: 최근 100개
   - 4시간봉: 최근 100개

2. **미완성 캔들 포함**
   - 현재 진행 중인 캔들도 실시간 집계하여 사용
   - 학습/백테스트/실시간 모두 동일한 방식

3. **룩어헤드 바이어스 제거**
   - 현재 시점 이전 데이터만 사용
   - `close` 가격만 사용 (실시간 가능)

4. **로직 완전 일치**
   - 학습 = 백테스트 = 실시간
   - Feature 계산 방법 100% 동일

### **LightGBM 파라미터**
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_data_in_leaf': 100
}
```

---

## 🚀 사용 방법

### **1. 학습**

```bash
cd model_multi
python train_multi.py
```

- 2024년 전체 데이터로 학습
- `model/lgb_multi_timeframe.pkl` 생성

### **2. 백테스트**

```bash
# 2022년 하락장 테스트
python backtest_multi.py 20220101 20221231

# 2025년 최신 데이터 테스트
python backtest_multi.py 20250801 20251031
```

### **3. 파라미터 최적화**

```bash
python optimize_params_multi.py
```

- 2024년 3개 구간으로 Walk-Forward 최적화
- 총 256개 조합 테스트 (4×4×4×4)
- 결과: `optimization_result_multi.pkl`

---

## 📊 백테스트 전략

### **매수 조건**
```python
if predicted_profit >= buy_threshold:  # 예: 0.1%
    if 1m_rsi < 75:  # 과매수 방지
        매수
```

### **매도 조건**
```python
# 1. 익절
if profit >= take_profit:  # 예: +0.6%
    매도

# 2. 손절
if profit <= -stop_loss:  # 예: -0.3%
    매도

# 3. 시간 초과
if holding_minutes >= time_limit:  # 예: 8분
    매도
```

### **최적화할 파라미터**
```python
buy_threshold: [0.05, 0.1, 0.15, 0.2]      # 매수 임계값 (%)
stop_loss: [0.2, 0.3, 0.4, 0.5]            # 손절 (%)
take_profit: [0.4, 0.6, 0.8, 1.0]          # 익절 (%)
time_limit: [5, 8, 10, 15]                 # 보유 시간 (분)
```

---

## 💡 전략 시나리오

### **시나리오 1: 강력한 매수 (3개 일치)** 💪

```python
# 4시간봉: 상승 필터 ✅
4h_ema_trend = 1  # 상승
4h_trend_strength = 0.8  # 강한 상승

# 1시간봉: 조정 완료 ✅
1h_rsi = 45  # 과매도에서 회복
1h_price_above_ma20 = True  # 20MA 위

# 1분봉: 매수 시그널 ✅
1m_rsi = 35  # 단기 과매도
1m_volume_spike = True  # 거래량 급증

→ 매수 진입! 🚀
```

### **시나리오 2: 약한 매수 (불일치)** ⚠️

```python
# 4시간봉: 하락 필터 ❌
4h_ema_trend = -1  # 하락

# 1시간봉: 단기 반등
1h_macd = 양수 전환

# 1분봉: 매수 시그널
1m_rsi = 30
1m_volume_spike = True

→ 매수 보류! ⏸️
   (4시간봉 하락 = 큰 흐름은 하락)
```

### **시나리오 3: 청산** 🔔

```python
# 4시간봉: 여전히 상승 ✅
4h_ema_trend = 1

# 1시간봉: 과열 신호 ⚠️
1h_rsi = 75  # 과매수

# 1분봉: 매도 시그널 🔴
1m_rsi = 80  # 극도 과매수
1m_macd_hist = 음수 전환

→ 청산! 💰
```

---

## ⚠️ 주의사항

### **1. 데이터 일관성**
- ✅ 학습 / 백테스트 / 실시간 Feature 계산 방식 100% 일치
- ✅ 윈도우 크기 고정 (100개)
- ✅ 통계 범위 동일 (rolling mean, std 등)

### **2. 룩어헤드 바이어스 제거**
- ✅ 백테스트에서 `close` 가격만 사용
- ✅ 미래 정보 사용 금지
- ✅ 현재 시점 이전 데이터만

### **3. 미완성 캔들 처리**
- ✅ 학습/백테스트/실시간 모두 동일하게 미완성 캔들 포함
- ✅ 현재 진행 중인 캔들도 실시간 집계

---

## 📌 다음 단계

1. **학습 실행**
   ```bash
   python train_multi.py
   ```

2. **백테스트 검증**
   ```bash
   python backtest_multi.py 20220101 20221231  # 하락장
   python backtest_multi.py 20250801 20251031  # 최신
   ```

3. **파라미터 최적화**
   ```bash
   python optimize_params_multi.py
   ```

4. **3개 구간 모두 양수 수익 확인**
   - 2024년 (학습)
   - 2022년 (하락장 검증)
   - 2025년 (최신 검증)

5. **실전 배포** (별도 main_multi.py 구현 필요)

---

## 🔧 기술 스택

- **머신러닝**: LightGBM
- **데이터**: Upbit API (1분봉)
- **리샘플링**: pandas resample
- **백테스트**: 슬라이딩 윈도우 방식

---

## 📞 문의

프로젝트 관련 문의사항은 이슈로 남겨주세요.

---

**Created**: 2025-11-12  
**Version**: v1.0  
**Status**: 개발 완료, 학습 및 검증 대기 중

