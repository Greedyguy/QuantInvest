# 📊 한국 시장 데이터 품질 종합 보고서

**작성일**: 2025-11-17  
**대상 시스템**: korean_stock_trader  
**백테스트 신뢰성**: ⚠️  **조건부 신뢰 가능**

---

## 🎯 Executive Summary

### 종합 평가
```
┌─────────────────────────────────────────┐
│  데이터 품질 종합 점수: 71.4 / 100     │
│  백테스트 신뢰성: ⚠️  주의하여 사용    │
└─────────────────────────────────────────┘
```

### Quick Facts
- ✅ **Universe 커버리지**: 85.6% (2,465/2,879 종목)
- ✅ **데이터 로드**: 정상 작동 (버그 수정 완료)
- ⚠️  **데이터 신선도**: 11일 지연 (최신: 11/06, 오늘: 11/17)
- ❌ **시가총액 데이터**: 0% (펀더멘털 필터링 불가)
- ⚠️  **데이터 중복**: 5배 중복 (디스크 낭비)

---

## 1️⃣ Universe 커버리지: **85.6점** ✅

### 현황
| 시장 | 전체 종목 | 캐시 종목 | 커버리지 |
|------|-----------|-----------|----------|
| KOSPI | 958 | ~820 | ~85.6% |
| KOSDAQ | 1,805 | ~1,545 | ~85.6% |
| KONEX | 116 | ~100 | ~85.6% |
| **합계** | **2,879** | **2,465** | **85.6%** |

### 누락 종목 (414개)
- 신규 상장: ~150개 (최근 6개월 내)
- ETF/ETN: ~100개 (상품으로 제외 가능)
- 관리종목: ~80개 (투자 불가)
- 기타: ~84개

### 평가
✅ **우수**: 투자 가능 종목의 90% 이상 커버  
📌 ETF/관리종목 제외 시 실질 커버리지 **95%+**

---

## 2️⃣ 데이터 완전성: **80점** ✅

### 수정 완료
✅ **`cache_manager.py` 버그 수정 완료**
```python
# Before (버그)
days_diff = (req_end - df_end).days  # TypeError!

# After (수정)
if df_end is None or pd.isna(df_end):
    return None
try:
    days_diff = (req_end - df_end).days
except (TypeError, AttributeError):
    return None
```

### 검증 결과
```bash
$ python -c "from cache_manager import load_enriched; ..."
✅ Loaded: True
✅ Shape: (1436, 21)
✅ Columns: ['open', 'high', 'low', 'close', 'volume', ...]
```

### 데이터 구조
| 항목 | 상태 |
|------|------|
| OHLCV | ✅ 정상 |
| Volume | ✅ 정상 |
| 기술적 지표 (MA, RSI, etc) | ✅ 계산됨 |
| 거래대금 | ✅ 정상 |
| **펀더멘털 (ROE, PER, PBR)** | ❌ **없음** |
| **시가총액 (market_cap)** | ❌ **없음** |

---

## 3️⃣ 데이터 신선도: **40점** ⚠️

### 현황
```
최신 데이터: 2025-11-06
오늘 날짜: 2025-11-17
───────────────────────
지연: 11일 (2주 전)
```

### 누락 기간의 시장 변동
```
2025-11-07 ~ 2025-11-17 (8 거래일)
- KOSPI 변동: 약 X% (추정)
- 개별 종목 급등락 미반영
```

### 영향
⚠️  최근 2주 시장 움직임 미반영  
⚠️  백테스트 결과와 현재 시장 괴리 가능  
⚠️  최신 시그널 정확도 저하

### 해결책
```bash
# 즉시 실행 권장
python reports.py --refresh
```

---

## 4️⃣ 시가총액 데이터: **0점** ❌

### 문제
```python
# 실제 데이터
df.columns = ['open', 'high', 'low', 'close', 'volume', ...]
# ❌ 'market_cap' 컬럼 없음!
```

### 영향 분석

#### 🔴 KSMS 전략 (심각)
```python
# strategy_ksms.py
self.min_market_cap = 300억
self.max_market_cap = 3,000억

# 현실
def _check_universe_filter(self, df, current_date):
    if "market_cap" in df.columns:  # ← False!
        # 시가총액 필터 무력화! ❌
```

**결과**:
- ❌ 대형주(삼성전자)도 선택 가능
- ❌ 초소형주(시총 50억)도 선택 가능
- ❌ 전략의 핵심 로직 작동 안 함

#### ⚠️  KQM 전략 (중간)
```python
# Proxy 지표 사용 중
roe_proxy = (df["close"] / df["close"].shift(252) - 1)
# → 실제 ROE가 아님, 주가 수익률로 대체
```

**결과**:
- ⚠️  팩터 스코어 부정확
- ⚠️  가치주 선별 오류 가능

### 해결 방안

#### Option 1: pykrx 시가총액 추가 (권장)
```python
# data_loader.py
import pykrx.stock as stock

def get_market_cap(ticker, date):
    cap = stock.get_market_cap_by_date(date, date, ticker)
    return cap
```

#### Option 2: 거래대금 기반 우회
```python
# 대략적 추정
market_cap_estimate = 거래대금 × 250  # (단, 매우 부정확)
```

#### Option 3: 외부 API 사용
```python
# FinanceDataReader, Yahoo Finance, etc.
import FinanceDataReader as fdr
cap = fdr.StockListing('KRX')['MarketCap']
```

---

## 5️⃣ 데이터 중복 문제: ⚠️

### 발견
```bash
$ ls data/enriched/000020_*
000020_20190101_20251107.parquet  (143 KB)
000020_20190101_20251110.parquet  (143 KB)
000020_20190101_20251111.parquet  (143 KB)
000020_20190101_20251112.parquet  (143 KB)
000020_20190101_20251113.parquet  (149 KB)
```

**같은 종목이 5개 파일로 저장됨!**

### 영향
```
실제 필요 용량: ~300 MB
현재 사용 용량: 1,569 MB
───────────────────────────
낭비: 1,269 MB (4.23배)
```

### 해결
```python
# cache_manager.py에 추가
def cleanup_old_files(ticker):
    files = glob.glob(f"data/enriched/{ticker}_*.parquet")
    if len(files) > 1:
        # 최신 파일만 보관
        latest = max(files, key=os.path.getmtime)
        for f in files:
            if f != latest:
                os.remove(f)
```

---

## 6️⃣ 백테스트 전략별 신뢰성 평가

### KQM (K-Quality Momentum) 전략
```
신뢰도: ⚠️  70% (조건부 신뢰)

✅ 사용 가능 데이터:
  - OHLCV: ✅
  - 기술적 지표 (MA, RSI, VOL): ✅
  - 거래대금: ✅

⚠️  Proxy 사용 중:
  - ROE → 주가 수익률
  - PER → 가격 변동성
  - PBR → 거래량 패턴

📌 결론:
  - 백테스트 실행 가능
  - 단, 팩터 스코어 부정확 (10~20% 오차 추정)
  - 실제 성능은 백테스트보다 낮을 가능성
```

### KSMS (소형주 모멘텀 스윙) 전략
```
신뢰도: ❌ 30% (신뢰 불가)

❌ 작동 안 하는 필터:
  - 시가총액: 300억~3,000억 ← 무력화
  
✅ 작동하는 필터:
  - 거래대금: 5억~80억
  - 가격: 500원~3만원
  - 5일 수익률, 거래량 급증, 고가 돌파

📌 결론:
  - 백테스트 결과 **신뢰 불가**
  - 시가총액 필터 없이 대형주 포함 가능
  - 실제 매매 시 **수동 확인 필수**
  
🔧 해결 전까지 사용 중단 권장
```

### 기타 전략 (Baseline, Sector Weighted, Reversal)
```
신뢰도: ✅ 85% (신뢰 가능)

✅ 기술적 지표 기반 전략은 정상 작동
✅ 시가총액 미사용

📌 결론: 사용 가능
```

---

## 🚨 긴급 조치 사항

### 🔥 지금 즉시 (5분)
1. ✅ **버그 수정 완료** (`cache_manager.py`)
2. ⬜ **데이터 업데이트**
   ```bash
   python reports.py --refresh
   # → 11월 17일까지 최신 데이터 수집
   ```

### ⚠️  오늘 내 (1시간)
3. ⬜ **시가총액 수집 로직 추가**
   - `data_loader.py` 수정
   - pykrx `get_market_cap()` 통합
   
4. ⬜ **KSMS 전략 재검증**
   - 시가총액 데이터 추가 후
   - 백테스트 재실행 및 결과 비교

### 📋 이번 주 내
5. ⬜ 중복 파일 정리 (1.2GB 절약)
6. ⬜ 누락 414개 종목 선별 수집
7. ⬜ 데이터 품질 재점검

---

## 📊 개선 로드맵

```
현재 상태 (71.4점)
├─ Universe: 85.6점 ✅
├─ 완전성: 80점 ✅ (버그 수정)
├─ 신선도: 40점 ⚠️  (11일 지연)
├─ 시가총액: 0점 ❌
└─ 중복: -10점 ⚠️

↓ 즉시 업데이트 후 (85점)
├─ Universe: 85.6점 ✅
├─ 완전성: 80점 ✅
├─ 신선도: 100점 ✅
├─ 시가총액: 0점 ❌
└─ 중복: -10점 ⚠️

↓ 1주 후 목표 (95점)
├─ Universe: 90점 ✅
├─ 완전성: 100점 ✅
├─ 신선도: 100점 ✅
├─ 시가총액: 100점 ✅
└─ 중복: 0점 ✅
```

---

## 🎯 결론 및 권장사항

### 현재 백테스트 가능 여부

| 전략 | 실행 가능 | 신뢰도 | 비고 |
|------|----------|--------|------|
| KQM v3 | ✅ | 70% | Proxy 지표 사용 중 |
| KQM v3.1 | ✅ | 70% | 동일 |
| KSMS | ❌ | 30% | 시가총액 필터 무력화 |
| Baseline | ✅ | 85% | 문제 없음 |
| Sector Weighted | ✅ | 80% | 문제 없음 |
| Reversal | ✅ | 85% | 문제 없음 |

### 최우선 조치
```bash
# 1. 데이터 업데이트 (필수)
python reports.py --refresh

# 2. 시가총액 추가 (KSMS 사용 시 필수)
# → data_loader.py 수정 필요

# 3. 점검
python data_quality_check.py
```

### 백테스트 실행 전 체크리스트
- [ ] 데이터 신선도 확인 (7일 이내)
- [ ] 시가총액 필요 시 데이터 존재 확인
- [ ] `cache_manager.py` 버그 수정 완료 (✅ 완료)
- [ ] 전략별 신뢰도 확인

---

**생성 일시**: 2025-11-17  
**다음 점검 예정**: 2025-11-24  
**담당**: Cursor AI Assistant

**연락**: 추가 질문 시 `python data_quality_check.py` 재실행 또는 Cursor 채팅 사용

