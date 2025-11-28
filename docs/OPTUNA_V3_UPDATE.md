# 🔧 Optuna Optimizer v3 기반 변경

**날짜**: 2025-11-14  
**변경 사항**: optuna_optimizer.py를 v3.1 → v3 기반으로 변경

---

## 📊 변경 전 vs 후

### Before (v3.1 기반)
```python
from strategies.strategy_kqm_v3_1 import KQMStrategyV3_1

strategy = KQMStrategyV3_1(
    factor_weights=factor_weights,
    rebal_days=rebal_days
)
```

**특징**:
- Low-vol filter (70% cutoff)
- Sector cap: 4개
- Position weight cap: 7%
- Negative momentum filter

### After (v3 기반)
```python
from strategies.strategy_kqm_v3 import KQMStrategyV3

strategy = KQMStrategyV3(
    rebal_days=rebal_days,
    factor_weights=factor_weights
)
```

**특징**:
- MA120 Risk Filter
- Sector Momentum weighting
- ERC position sizing
- Sector cap: 5개

---

## 🔄 strategy_kqm_v3.py 수정 사항

### 1. `__init__` 파라미터 추가

```python
# Before
def __init__(self):
    self.rebalance_days = 10
    self.holdings_count = 30
    self.sector_cap = 5

# After
def __init__(self, rebal_days=10, n_stocks=30, sector_cap=5, factor_weights=None):
    self.rebalance_days = rebal_days
    self.holdings_count = n_stocks
    self.sector_cap = sector_cap
    
    # 팩터 가중치 (Optuna 최적화 지원)
    if factor_weights is None:
        self.factor_weights = {
            'MOM6': 0.40,
            'MOM3': 0.10,
            'QUALITY': 0.20,
            'VOL': 0.20,
            'VAL': 0.10,
        }
    else:
        self.factor_weights = factor_weights
```

### 2. Factor Score 계산 수정

```python
# Before (하드코딩)
day["score"] = (
    day["mom6m_rank"] * 0.30 +
    day["mom3m_rank"] * 0.20 +
    day["roe_proxy_rank"] * 0.20 +
    day["inv_vol_smooth_rank"] * 0.20 +
    day["val_proxy_rank"] * 0.10
)

# After (동적 가중치)
day["score"] = (
    self.factor_weights.get('MOM6', 0.30) * day["mom6m_rank"] +
    self.factor_weights.get('MOM3', 0.20) * day["mom3m_rank"] +
    self.factor_weights.get('QUALITY', 0.20) * day["roe_proxy_rank"] +
    self.factor_weights.get('VOL', 0.20) * day["inv_vol_smooth_rank"] +
    self.factor_weights.get('VAL', 0.10) * day["val_proxy_rank"]
)
```

---

## 🎯 이제 가능한 작업

### 1. v3 기반 Optuna 최적화 실행
```bash
python optuna_optimizer.py 50
```

### 2. 기본 v3 백테스트
```python
from strategies.strategy_kqm_v3 import KQMStrategyV3

strategy = KQMStrategyV3()  # 기본 파라미터
```

### 3. 커스텀 팩터 가중치로 백테스트
```python
strategy = KQMStrategyV3(
    rebal_days=17,
    factor_weights={
        'MOM6': 0.35,
        'MOM3': 0.15,
        'QUALITY': 0.20,
        'VOL': 0.20,
        'VAL': 0.10,
    }
)
```

---

## ⚠️ 주의사항

### v3.1 vs v3 차이점

| 기능 | v3 | v3.1 |
|------|----|----- |
| **Low-vol Filter** | ❌ | ✅ (70% cutoff) |
| **Position Cap** | ❌ | ✅ (7%) |
| **Sector Cap** | 5개 | 4개 |
| **MA Filter** | MA120 | Simplified |
| **Sector Momentum** | ✅ | ❌ |
| **ERC Sizing** | ✅ | ✅ |

**권장**:
- **안정성 중시**: v3.1 사용
- **수익성 중시**: v3 사용
- **Optuna 최적화**: 이제 v3 기반으로 실행

---

## 🚀 다음 단계

1. **v3 기반 Optuna 최적화 실행**
   ```bash
   python optuna_optimizer.py 50
   ```

2. **최적화 결과를 v3_2에 반영**
   - 새로운 최적 파라미터 확인
   - v3_2 업데이트

3. **백테스트 비교**
   ```bash
   python reports.py --compare kqm_v3 kqm_v3_2
   ```

---

모든 준비 완료! 이제 v3 기반으로 Optuna 최적화를 실행할 수 있습니다! 🎉

