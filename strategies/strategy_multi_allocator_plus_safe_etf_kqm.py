#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiStrategyAllocatorPlus SafeETF + KQM (kqm=25% 최적 비중)
— 그리드 서치 결과: kqm 25%가 CAGR·Sharpe·MDD 모든 지표 최우수
  (2020-01-01 ~ 2026-05-25 백테스트 기준)

구성 (기본 비중):
  kqm_small_cap_v22_short  0.225  short      (safe_etf 0.30 × 0.75)
  hybrid_portfolio_v2_4    0.150  offensive  (safe_etf 0.20 × 0.75)
  kqm_small_cap_v22        0.165  offensive  (safe_etf 0.22 × 0.75)
  etf_defensive_safe       0.135  defensive  (safe_etf 0.18 × 0.75)
  k200_mean_rev            0.075  offensive  (safe_etf 0.10 × 0.75)
  kqm                      0.250  offensive  ← 전체 유니버스 Quality-Momentum

성과 (백테스트):
  CAGR   +12.13%  (현재 운용 no_etf: +6.50%)
  Sharpe  1.316   (현재 운용 no_etf:  0.745)
  MDD    -10.90%  (현재 운용 no_etf: -14.92%)
  최종   +102.0%  (현재 운용 no_etf: +47.3%)
  손실연도  2년   (현재 운용 no_etf:   3년)
"""

from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus


class MultiStrategyAllocatorPlusSafeETFKQM(MultiStrategyAllocatorPlus):
    def __init__(self):
        super().__init__()

        # kqm=25% 그리드 서치 최적 비중 (나머지 5전략은 safe_etf 상대비중 유지 × 0.75)
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

    def get_name(self):
        return "multi_allocator_plus_safe_etf_kqm"

    def get_description(self):
        return "Multi-allocator PLUS SafeETF + KQM 25% (그리드 서치 최적, 2026-05-25 기준)"
