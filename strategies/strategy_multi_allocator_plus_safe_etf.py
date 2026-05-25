#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiStrategyAllocatorPlus SafeETF
— multi_allocator_plus_no_etf 에서 etf_defensive_safe 를 복원한 버전
  (레버리지/인버스 ETF 제거, 일반 ETF만 사용)

구성:
  kqm_small_cap_v22_short  0.30  short
  hybrid_portfolio_v2_4    0.20  offensive
  kqm_small_cap_v22        0.22  offensive
  etf_defensive_safe       0.18  defensive  ← 핵심: 일반 ETF 방어 레이어 복원
  k200_mean_rev            0.10  offensive
"""

from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus


class MultiStrategyAllocatorPlusSafeETF(MultiStrategyAllocatorPlus):
    def __init__(self):
        super().__init__()

        self.strategy_configs = [
            {"name": "kqm_small_cap_v22_short", "weight": 0.30, "role": "short"},
            {"name": "hybrid_portfolio_v2_4",   "weight": 0.20, "role": "offensive"},
            {"name": "kqm_small_cap_v22",       "weight": 0.22, "role": "offensive"},
            {"name": "etf_defensive_safe",      "weight": 0.18, "role": "defensive"},
            {"name": "k200_mean_rev",           "weight": 0.10, "role": "offensive"},
        ]

        self.strategy_names = [cfg["name"] for cfg in self.strategy_configs]
        self.strategy_base_weight = {
            cfg["name"]: cfg["weight"] for cfg in self.strategy_configs
        }
        self.strategy_roles = {
            cfg["name"]: cfg["role"] for cfg in self.strategy_configs
        }

    def get_name(self):
        return "multi_allocator_plus_safe_etf"

    def get_description(self):
        return "Multi-allocator PLUS SafeETF (일반 ETF 방어 레이어 복원, 레버리지/인버스 제외)"
