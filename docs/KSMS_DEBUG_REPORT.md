# 🔍 KSMS 백테스트 버그 분석 리포트

**날짜**: 2025-11-14  
**문제**: 차트와 실제 결과 불일치

---

## 🚨 보고된 문제

```
초기 자본: 100만원
최종 자산: 1,847원 (❌ -99.8% 손실!)
CAGR: 2.9291 (✅ +293% ???)
차트: 2.5억원까지 상승
```

**명확한 모순!**

---

## 🔍 원인 분석

### 1. 차트 스케일 문제 ⭐⭐⭐

**발견 사항**:
- 차트 Y축 최대값: **2.5억원**
- 실제 초기 자본: **100만원**
- Y축 스케일: **10억원 전략 기준**

**원인**:
```python
# reports.py의 plot_results()가 여러 전략을 비교할 때
# Y축을 가장 큰 전략에 맞춤

# 예: kqm_v3 (10억) vs ksms (100만)
# → Y축이 10억 스케일로 설정
# → ksms 곡선이 비정상적으로 보임
```

### 2. 실제 성과 (추정)

100만원 → 1,847원 = **-99.8% 손실**

**CAGR 계산이 잘못됨!**

```python
# 올바른 계산
초기: 1,000,000원
최종: 1,847원
수익률: -99.8%
CAGR: 약 -60% ~ -80% (기간에 따라)

# 하지만 보고된 CAGR: +2.9291 ???
```

---

## 💡 문제의 근본 원인

### Cause 1: 비교 백테스트 실행

사용자가 아마도 다음과 같이 실행:
```bash
python reports.py --compare kqm_v3 ksms
# 또는
python reports.py --compare kqm_small_cap ksms
```

**결과**:
- kqm (10억) vs ksms (100만) 비교
- Y축이 10억 스케일로 고정
- ksms 곡선이 비정상

### Cause 2: CAGR 계산 오류 가능성

`utils.py`의 `perf_stats()` 함수에서:
```python
cagr = (equity[-1] / equity[0]) ** (252/len(equity)) - 1
```

만약 `equity`가 잘못된 스케일로 저장되었다면?
- equity[0] = 1,000,000 (100만)
- equity[-1] = 250,000,000 (2.5억, 잘못됨!)
- 계산된 CAGR = +293% (양수)

---

## 🔧 해결 방법

### 즉시 확인 사항

#### 1. 단독 백테스트 실행
```bash
# KSMS만 단독 실행
python reports.py --strategy ksms

# 비교 실행 금지 (스케일 차이로 혼란)
```

#### 2. 터미널 출력 확인
```
실행 시 다음을 확인:
✅ KSMS 백테스트 완료: XXX개 데이터 포인트
   총 거래: XX회
   최종 자산: XXX원 (수익률: XX%)
```

#### 3. Equity Curve 출력
```python
# 백테스트 직후 equity_curve 값 확인
print(ec_df.head())
print(ec_df.tail())
print(f"Min: {ec_df['equity'].min():,.0f}")
print(f"Max: {ec_df['equity'].max():,.0f}")
```

---

## 🐛 예상되는 버그

### Bug 1: equity_curve 스케일 오류

```python
# strategy_ksms.py의 equity 계산에서
# 혹시 다른 전략의 INIT_CASH를 참조?

# 확인 필요:
# - init_cash = 1_000_000 (✅ 맞음)
# - equity 계산 로직 (확인 필요)
```

### Bug 2: 포지션 가치 계산 오류

```python
# run_backtest() 내부의 equity 계산:
if position is not None:
    position_value = current_price * position["qty"]
    equity = cash + position_value
else:
    equity = cash

# 만약 position_value가 비정상적으로 크다면?
```

---

## ✅ 검증 체크리스트

- [ ] **단독 백테스트 실행** (비교 X)
- [ ] **터미널 출력 확인** (최종 자산)
- [ ] **거래 로그 확인** (실제 매수/매도 가격)
- [ ] **equity_curve 값 확인** (min/max)
- [ ] **CAGR 재계산** (utils.py)

---

## 🚀 권장 조치

### 1. 즉시 실행
```bash
# 캐시 삭제 후 재실행
rm -rf data/cache/backtest_*
python reports.py --strategy ksms
```

### 2. 로그 확인
```python
# strategy_ksms.py에 디버그 출력 추가
print(f"[DEBUG] Date: {current_date}, Cash: {cash:,.0f}, Equity: {equity:,.0f}")
```

### 3. 소액 전략 전용 reports 작성
```python
# reports_small.py 생성
# 100만원 전략 전용 차트 (스케일 맞춤)
# Y축: 0 ~ 1,000만원 (10배까지만)
```

---

## 📊 올바른 기대 성과

### 소형주 스윙 전략 (100만원)

| 시나리오 | 초기 | 최종 | 수익률 | CAGR |
|---------|------|------|--------|------|
| **매우 좋음** | 100만 | 300만 | +200% | +30% |
| **좋음** | 100만 | 150만 | +50% | +15% |
| **보통** | 100만 | 110만 | +10% | +5% |
| **나쁨** | 100만 | 80만 | -20% | -10% |
| **매우 나쁨** | 100만 | 50만 | -50% | -20% |

### 현재 결과 (비정상)
```
초기: 100만원
최종: 1,847원 ← ❌ 비정상!
수익률: -99.8% ← ❌ 완전 파산
CAGR: +2.9291 ← ❌ 계산 오류
```

---

## 🎯 다음 단계

1. **단독 백테스트 재실행**
   ```bash
   python reports.py --strategy ksms > ksms_log.txt 2>&1
   ```

2. **로그 파일 확인**
   - 거래 횟수
   - 승률
   - 평균 수익/손실
   - 최종 자산

3. **문제 지속 시**
   - equity_curve 직접 출력
   - 거래 로그 CSV 저장
   - 디버그 모드로 실행

---

**결론**: 차트 스케일 불일치 + CAGR 계산 오류로 추정. 단독 백테스트 재실행 필요! 🔧

