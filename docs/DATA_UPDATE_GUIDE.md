# 📊 데이터 업데이트 가이드

## 🔄 데이터 업데이트 방법

### 방법 1: 자동 업데이트 스크립트 (권장)
```bash
python update_data.py
```

**기능**:
- ✅ 중복 파일 자동 정리 (디스크 공간 절약)
- ✅ 캐시 초기화
- ✅ 최신 데이터 자동 다운로드
- ✅ **시가총액 데이터 포함**
- ⏱️  소요 시간: 5~10분

---

### 방법 2: 수동 업데이트
```bash
# 캐시 무시하고 데이터 로드
python reports.py --strategy baseline --no-cache
```

**특징**:
- 중복 파일 정리 없음
- 증분 업데이트 사용하지 않음
- 모든 데이터 새로 다운로드

---

## 🆕 시가총액 데이터 추가 완료!

### 변경 사항
`data_loader.py`에 시가총액 수집 기능 추가:

```python
# 새로운 함수
def get_market_cap(ticker, start, end):
    """pykrx에서 시가총액 조회"""
    df_cap = stock.get_market_cap_by_date(start, end, ticker)
    # → 'market_cap' 컬럼 반환

# 업데이트된 함수
def get_ohlcv_one(ticker, start, end, include_market_cap=True):
    """OHLCV + 시가총액 통합 수집"""
    # 기본적으로 시가총액 포함
```

### 사용 예시
```python
from data_loader import get_ohlcv_one

# 시가총액 포함 (기본)
df = get_ohlcv_one("005930", "2020-01-01", "2025-11-17")
print(df.columns)
# → ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap']

# 시가총액 제외
df = get_ohlcv_one("005930", "2020-01-01", "2025-11-17", include_market_cap=False)
```

---

## 📋 업데이트 후 확인사항

### 1. 데이터 품질 재점검
```bash
python data_quality_check.py
```

**목표 점수**: 85/100 이상

**확인 항목**:
- [ ] Universe 커버리지 > 85%
- [ ] 데이터 신선도 < 7일
- [ ] **시가총액 커버리지 > 80%** ← 새로 추가됨!
- [ ] 중복 파일 < 10%

---

### 2. KSMS 전략 재검증
```bash
# 시가총액 필터 작동 확인
python -c "
from cache_manager import load_enriched
from config import START, END
from datetime import date

end_str = date.today().strftime('%Y-%m-%d') if END is None else END
df = load_enriched('005930', START, end_str)

print(f'종목: 삼성전자 (005930)')
print(f'market_cap 컬럼: {\"market_cap\" in df.columns}')
if 'market_cap' in df.columns:
    print(f'시가총액 데이터: {df[\"market_cap\"].notna().sum()}/{len(df)} 일')
    print(f'평균 시총: {df[\"market_cap\"].mean() / 1e12:.2f}조원')
"
```

**예상 결과**:
```
종목: 삼성전자 (005930)
market_cap 컬럼: True
시가총액 데이터: 1436/1436 일
평균 시총: 450.23조원
```

---

### 3. KSMS 백테스트 재실행
```bash
# 시가총액 필터가 정상 작동하는지 확인
python reports.py --strategy ksms
```

**확인사항**:
- 선택된 종목들의 시가총액이 300억~3,000억 범위인지
- 대형주 (예: 삼성전자, SK하이닉스)가 제외되었는지
- 초소형주 (시총 100억 미만)가 제외되었는지

---

## 🔧 문제 해결

### Q1. "market_cap 컬럼이 여전히 없어요"
```bash
# 캐시 강제 초기화 후 재다운로드
rm -rf data/enriched/*
python update_data.py
```

---

### Q2. "시가총액 데이터가 일부만 수집됐어요"
**원인**: pykrx API의 일시적 오류 또는 상장폐지 종목

**해결**:
1. 잠시 후 재시도
2. 특정 종목만 재수집:
   ```python
   from data_loader import get_ohlcv_one
   df = get_ohlcv_one("종목코드", "2020-01-01", "2025-11-17", include_market_cap=True)
   ```

---

### Q3. "업데이트가 너무 오래 걸려요"
**정상입니다!**
- 2,400+ 종목 × 시가총액 API 호출
- 예상 소요 시간: **5~15분**

**빠르게 하려면**:
```bash
# 기존 OHLCV 캐시 유지, enriched만 재계산
python reports.py --strategy baseline --no-backtest-cache
```

---

### Q4. "pykrx API 오류가 자주 나요"
```python
# data_loader.py의 get_market_cap()에서
# 오류 시 조용히 무시하고 진행
# → OHLCV는 정상 수집, 시가총액만 누락
```

**권장**:
- 네트워크 안정적인 환경에서 실행
- 시간대: 장 마감 후 (오후 4시 이후)
- pykrx 서버 부하가 적은 시간

---

## 📊 업데이트 전/후 비교

### Before (시가총액 없음)
```python
df.columns
# ['open', 'high', 'low', 'close', 'volume', 'value', 'ticker', ...]

# KSMS 전략
if "market_cap" in df.columns:  # ← False!
    # 필터 작동 안 함 ❌
```

### After (시가총액 포함)
```python
df.columns
# ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'ticker', ...]

# KSMS 전략
if "market_cap" in df.columns:  # ← True! ✅
    mc = row["market_cap"]
    if mc < 300억 or mc > 3000억:
        return False  # 필터 작동! ✅
```

---

## 🚀 다음 단계

업데이트 완료 후:

1. ✅ **데이터 품질 점검**
   ```bash
   python data_quality_check.py
   ```

2. ✅ **KSMS 백테스트 재실행**
   ```bash
   python reports.py --strategy ksms
   ```

3. ✅ **결과 비교**
   - 이전 백테스트 (시총 필터 없음)
   - 업데이트 후 백테스트 (시총 필터 작동)
   - 성능 차이 확인

---

**생성일**: 2025-11-17  
**마지막 업데이트**: 시가총액 수집 기능 추가  
**다음 업데이트 예정**: 펀더멘털 지표 (ROE, PER, PBR) 추가

