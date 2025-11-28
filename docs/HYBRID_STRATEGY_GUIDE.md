# 🔥 Hybrid Portfolio Strategy 백테스팅 가이드

`hybrid_portfolio_trader.py`의 실시간 매매 로직을 백테스트용으로 변환한 전략입니다.

## 📊 전략 개요

### 구성
- **Korean Aggressive (70%)**: 중소형주 모멘텀 전략
- **Production Portfolio (30%)**: 대형주 안정성 전략

### 자본 배분
```
초기 자본: 100만원
├── Korean Aggressive: 70만원 (70%)
│   ├── 최대 포지션: 7개
│   └── 포지션 크기: 20% (약 14만원/종목)
└── Production Portfolio: 30만원 (30%)
    ├── 최대 포지션: 6개
    └── 포지션 크기: 50% (약 15만원/종목)
```

## 🎯 매매 전략

### Korean Aggressive (70% 자본)

**매수 조건:**
- ✅ RSI: 40 ~ 85
- ✅ 가격 > MA5
- ✅ GAP: 0.5% 이상 (전일 대비)
- ✅ 가격 범위: 1,000원 ~ 100,000원
- ✅ 거래량: 50,000주 이상

**특징:**
- 중소형주 대상
- 거래량 기반 선정
- 단기 모멘텀 포착

### Production Portfolio (30% 자본)

**매수 조건:**
- ✅ RSI: 35 ~ 75
- ✅ 가격 > MA20 × 98%
- ✅ 가격: 8,000원 이상
- ✅ 가격 범위: 3,000원 ~ 500,000원
- ✅ 거래량: 30,000주 이상

**특징:**
- 대형주 대상 (KOSPI/KOSDAQ)
- 안정성 중시
- 중장기 추세 추종

## 🛡️ 리스크 관리

### 공통 매도 조건
1. **손절**: -10% 도달
2. **익절**: +20% 도달
3. **기간만료 (수익 실현)**: 10일 이상 보유 + 5% 이상 수익
4. **기간만료**: 20일 이상 보유

### 종목 중복 관리
- 단일 종목 최대 비중: 20% (전체 포트폴리오 기준)
- Korean/Portfolio 간 종목 중복 방지
- 전체 포트폴리오 관점에서 리스크 분산

## 🚀 백테스트 실행

### 기본 실행
```bash
python reports.py --strategy hybrid_portfolio
```

### 비교 백테스트
```bash
python reports.py --compare baseline hybrid_portfolio
python reports.py --compare kqm_v3 hybrid_portfolio
```

### 캐시 무시하고 실행
```bash
python reports.py --strategy hybrid_portfolio --no-cache
```

## 📈 예상 성과

### 목표 지표
- **CAGR**: 15~25%
- **Sharpe Ratio**: 1.0 이상
- **MDD**: -20% 이내
- **Win Rate**: 55~65%

### 전략별 기대 역할

**Korean Aggressive (70%)**
- 높은 수익률 추구
- 단기 모멘텀 포착
- 승률: 50~60%
- 평균 수익: 5~10%

**Production Portfolio (30%)**
- 안정성 제공
- MDD 완화
- 승률: 60~70%
- 평균 수익: 3~7%

## ⚙️ 파라미터 조정

전략 파라미터를 수정하려면 `strategies/strategy_hybrid.py`에서 수정:

```python
strategy = HybridPortfolioStrategy(
    korean_aggressive_ratio=0.70,        # Korean 비율 (기본 70%)
    production_portfolio_ratio=0.30,     # Portfolio 비율 (기본 30%)
    korean_max_positions=7,              # Korean 최대 포지션 (기본 7개)
    portfolio_max_positions=6,           # Portfolio 최대 포지션 (기본 6개)
    korean_position_size=0.20,           # Korean 포지션 크기 (기본 20%)
    portfolio_position_size=0.50,        # Portfolio 포지션 크기 (기본 50%)
    max_single_stock_ratio=0.20          # 단일 종목 최대 비중 (기본 20%)
)
```

## 📊 실제 매매와 차이점

| 구분 | 실시간 매매 | 백테스트 |
|------|------------|----------|
| 데이터 | SQLite DB + yfinance | Parquet (enriched) |
| 실행 방식 | 비동기 (async/await) | 동기식 |
| 가격 조회 | 실시간 API | 과거 OHLCV |
| 주문 체결 | 한국투자증권 API | 시뮬레이션 |
| 포지션 관리 | 실제 계좌 동기화 | 메모리 관리 |

## 🔧 고급 설정

### 자본 배분 변경 예시

**보수적 (Korean 50% + Portfolio 50%)**
```python
strategy = HybridPortfolioStrategy(
    korean_aggressive_ratio=0.50,
    production_portfolio_ratio=0.50
)
```

**공격적 (Korean 80% + Portfolio 20%)**
```python
strategy = HybridPortfolioStrategy(
    korean_aggressive_ratio=0.80,
    production_portfolio_ratio=0.20
)
```

### 포지션 개수 조정

**소액 자본 (3개 + 3개)**
```python
strategy = HybridPortfolioStrategy(
    korean_max_positions=3,
    portfolio_max_positions=3
)
```

**대규모 자본 (10개 + 10개)**
```python
strategy = HybridPortfolioStrategy(
    korean_max_positions=10,
    portfolio_max_positions=10
)
```

## 📝 주의사항

1. **데이터 요구사항**
   - 최소 60일 이상의 enriched 데이터 필요
   - RSI, MA5, MA20 등 기술적 지표 필수

2. **백테스트 한계**
   - 슬리피지, 체결 지연 등 실제 매매와 차이 존재
   - 과거 성과가 미래 수익을 보장하지 않음

3. **리스크 관리**
   - 단일 종목 집중 리스크 (최대 20%)
   - 시장 급락 시 손절 연쇄 발생 가능
   - 충분한 분산투자 권장

## 🎓 전략 개선 아이디어

1. **Market Regime Filter 추가**
   - KOSPI/KOSDAQ 지수 추세 필터
   - 약세장에서 현금 비중 확대

2. **동적 자본 배분**
   - 전략별 성과에 따라 배분 비율 조정
   - 월간/분기별 리밸런싱

3. **섹터 분산**
   - 섹터별 최대 비중 제한
   - 섹터 로테이션 전략 통합

4. **변동성 기반 포지션 사이징**
   - ATR 기반 리스크 조정
   - Equal Risk Contribution (ERC)

## 📞 문의 및 개선

전략 개선 제안이나 버그 리포트는 이슈로 등록해주세요!

---

**Created**: 2024-11-19  
**Based on**: `hybrid_portfolio_trader.py` v1.0  
**Author**: Korean Stock Trader Team

