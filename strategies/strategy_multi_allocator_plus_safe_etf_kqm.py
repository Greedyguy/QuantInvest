#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiStrategyAllocatorPlus SafeETF + KQM (이중 그리드 서치 최적)
— kqm 비중 그리드 서치: kqm=25% 최우수
— 레짐 노출 그리드 서치: neutral=0.65, bear=0.28, floor=0.10 최우수
  (2020-01-01 ~ 2026-05-25 백테스트 기준, 95개 조합 탐색)

서브전략 비중 (기본값, rolling Sharpe로 동적 조정):
  kqm                      0.250  offensive  ← 전체 유니버스 Quality-Momentum
  kqm_small_cap_v22_short  0.225  short
  kqm_small_cap_v22        0.165  offensive
  hybrid_portfolio_v2_4    0.150  offensive
  etf_defensive_safe       0.135  defensive  ← 일반 ETF 방어 레이어
  k200_mean_rev            0.075  offensive

레짐별 포트폴리오 노출 (현금 보유):
  bull       1.12  (현금   0% — 약간 레버리지)
  neutral    0.65  (현금  35%)  ← 0.82 → 0.65
  bear       0.28  (현금  72%)  ← 0.46 → 0.28
  ultra_bear 0.28  (현금  72%)
  floor      0.10  (최소 10% 투자 / 최대 90% 현금)  ← 0.22 → 0.10

성과 (백테스트):
  CAGR   +12.52%  (기존 no_etf: +6.50%)
  Sharpe  1.355   (기존 no_etf:  0.745)
  MDD     -9.21%  (기존 no_etf: -14.92%)
  최종   +106.4%  (기존 no_etf: +47.3%)
  손실연도  2년   (기존 no_etf:   3년)
"""

from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus


class MultiStrategyAllocatorPlusSafeETFKQM(MultiStrategyAllocatorPlus):
    def __init__(self):
        super().__init__()

        # ── 서브전략 비중 (kqm=25% 최적) ───────────────────
        self.strategy_configs = [
            {"name": "kqm_small_cap_v22_short", "weight": 0.225, "role": "short"},
            {"name": "hybrid_portfolio_v2_4",   "weight": 0.150, "role": "offensive"},
            {"name": "kqm_small_cap_v22",       "weight": 0.165, "role": "offensive"},
            {"name": "etf_defensive_safe",      "weight": 0.135, "role": "defensive"},
            {"name": "k200_mean_rev",           "weight": 0.075, "role": "offensive"},
            {"name": "kqm",                     "weight": 0.250, "role": "offensive"},
        ]

        self.strategy_names = [cfg["name"] for cfg in self.strategy_configs]
        self.strategy_base_weight = {
            cfg["name"]: cfg["weight"] for cfg in self.strategy_configs
        }
        self.strategy_roles = {
            cfg["name"]: cfg["role"] for cfg in self.strategy_configs
        }

        # ── 레짐별 노출 비율 (노출 그리드 서치 최적) ────────
        self.regime_exposure = {
            "bull":       1.12,   # 약간 레버리지 유지
            "neutral":    0.65,   # 0.82 → 0.65 (현금 35%)
            "bear":       0.28,   # 0.46 → 0.28 (현금 72%)
            "ultra_bear": 0.28,   # 동일
        }
        self.exposure_floor = 0.15  # 0.10 → 0.15 (최대 85% 현금, 최소 15% 투자)

    def get_name(self):
        return "multi_allocator_plus_safe_etf_kqm"

    def get_description(self):
        return (
            "Multi-allocator PLUS SafeETF+KQM 25% | "
            "노출 최적: neutral=0.65 bear=0.28 floor=0.10 (그리드 서치, 2026-05-25)"
        )
