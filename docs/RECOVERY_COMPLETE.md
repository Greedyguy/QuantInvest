# ✅ 파일 복구 완료

**복구 완료일:** 2025-11-14 00:23

## 📁 복구된 파일 목록

### 1️⃣ 전략 파일들
- ✅ `strategies/base_strategy.py` - 전략 베이스 클래스
- ✅ `strategies/__init__.py` - 전략 레지스트리
- ✅ `strategies/strategy_sector_weighted.py` - 섹터 가중 전략
- ✅ `strategies/strategy_improved.py` - 개선된 전략
- ✅ `strategies/strategy_reversal.py` - 리버설 전략 v2
- ✅ `strategies/strategy_kqm.py` - KQM 전략 (기존 파일 유지)
- ✅ `strategies/strategy_kqm_v2.py` - KQM v2 전략
- ✅ `strategies/strategy_kqm_v3.py` - KQM v3 전략
- ✅ `strategies/strategy_kqm_v3_1.py` - **KQM v3.1 전략 (최신)**

### 2️⃣ 최적화 파일
- ✅ `optuna_optimizer.py` - **Optuna 하이퍼파라미터 최적화**

## 🎯 핵심 기능

### KQM v3.1 전략 (리스크 컨트롤 강화)
```python
class KQMStrategyV3_1(BaseStrategy):
    - 리밸런싱 주기: 10일 (조정 가능)
    - 보유 종목: 30개
    - 섹터 제한: 4개/섹터
    - 종목 비중 상한: 7%
    - Low-Vol 필터: 변동성 상위 30% 제거
    - ERC 포지션 사이징
```

### Optuna 최적화 (하이퍼파라미터)
```python
최적화 대상:
1. ✅ 팩터 가중치 (MOM6, MOM3, QUALITY, VOL, VAL)
2. ✅ 리밸런싱 주기 (5~20일)
3. ⚠️  거래대금 필터 (주석 처리됨, 필요시 활성화 가능)

목표 함수:
Score = 0.6 * (Sharpe_train + 0.5*MDD_train) + 
        0.4 * (Sharpe_valid + 0.5*MDD_valid)
```

## 🚀 실행 방법

### 1. 전략 백테스트
```bash
# KQM v3.1 실행
python reports.py --strategy kqm_v3_1

# 전략 비교
python reports.py --compare kqm_v2 kqm_v3_1

# 모든 KQM 버전 비교
python reports.py --strategy kqm kqm_v2 kqm_v3 kqm_v3_1
```

### 2. Optuna 최적화
```bash
# 빠른 테스트 (20회, ~10분)
python optuna_optimizer.py 20

# 표준 실행 (50회, ~30분) ⭐
python optuna_optimizer.py 50

# 정밀 실행 (100회, ~1시간)
python optuna_optimizer.py 100
```

### 3. 최적화 결과 확인
```bash
# 결과 파일 확인
cat data/meta/kqm_optuna_weights.json
```

## 📊 등록된 전략 (8개)

1. **baseline** - 기본 모멘텀 전략
2. **sector_weighted** - 섹터 모멘텀 가중 전략
3. **improved** - 개선된 통합 전략
4. **reversal** - 단기 리버설 전략 v2
5. **kqm** - K-Quality Momentum (기본)
6. **kqm_v2** - KQM v2 (Enhanced)
7. **kqm_v3** - KQM v3 (Risk Filter + ERC)
8. **kqm_v3_1** - **KQM v3.1 (Risk Control Enhanced)** ⭐

## 🔍 전략 확인
```bash
python -c "from strategies import list_strategies; [print(f'{n}: {d}') for n, d in list_strategies()]"
```

## 📈 예상 성과 (백테스트 필요)

| 전략 | CAGR | Sharpe | MDD | 특징 |
|------|------|--------|-----|------|
| KQM v2 | 18.3% | 0.77 | -36.4% | 기본 (10일, 30종목) |
| KQM v3 | ~16% | ~0.85 | ~-30% | Risk Filter + ERC |
| **KQM v3.1** | **목표 12~15%** | **≥0.8** | **≤-25%** | **리스크 컨트롤 강화** ⭐ |

## ⚠️ 주의사항

1. **데이터 품질 확인** (중요!)
   ```bash
   python data_validator.py
   ```

2. **캐시 관리**
   - 캐시 사용 (빠름): `python reports.py --strategy kqm_v3_1`
   - 캐시 제거 (최신 데이터): `python reports.py --strategy kqm_v3_1 --no-cache`

3. **pykrx API 에러**
   - 오늘이 거래일이 아니면 API 에러 발생 가능
   - 캐시를 사용하거나 다음 거래일에 재시도

## 💡 다음 단계

1. ✅ **KQM v3.1 백테스트 실행**
   ```bash
   python reports.py --strategy kqm_v3_1 --validate-quality --auto-refresh
   ```

2. ✅ **Optuna 최적화 실행**
   ```bash
   python optuna_optimizer.py 50
   ```

3. ⏳ **최적 가중치 적용 및 재테스트**
   - `data/meta/kqm_optuna_weights.json` 확인
   - 최적 가중치로 v3.1 재실행
   - 성과 비교 분석

4. ⏳ **거래대금 필터 추가 (선택사항)**
   - `optuna_optimizer.py`의 거래대금 필터 주석 해제
   - `strategy_kqm_v3_1.py`에 거래대금 파라미터 추가
   - 재최적화

## 🎉 복구 완료!

모든 삭제된 파일이 성공적으로 복구되었습니다!

### 복구된 파일 요약
- **전략 파일**: 9개
- **최적화 파일**: 1개
- **총 복구**: 10개

이제 백테스트와 최적화를 실행하실 수 있습니다! 🚀

