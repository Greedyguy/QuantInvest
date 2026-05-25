#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiStrategyAllocator PLUS Midcap
- 기존 multi_allocator_plus 에서 소형주 비중을 줄이고
  kmr_midcap(중형주 mean-reversion)과 br_pullback(신고가 눌림매수)을 추가
- 소형주 집중도 완화: small-cap 52% → 37%
  + kmr_midcap 10% (중형주, 10,000~150,000원 ADV 20~300억)
  + br_pullback 7%  (신고가+눌림, 5,000~80,000원)
"""

from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus


class MultiStrategyAllocatorPlusMidcap(MultiStrategyAllocatorPlus):
    def __init__(self):
        super().__init__()

        self.strategy_configs = [
            {"name": "kqm_small_cap_v22_short", "weight": 0.25, "role": "short"},
            {"name": "hybrid_portfolio_v2_4",   "weight": 0.21, "role": "offensive"},
            {"name": "kqm_small_cap_v22",       "weight": 0.12, "role": "offensive"},
            {"name": "etf_defensive",           "weight": 0.15, "role": "defensive"},
            {"name": "k200_mean_rev",           "weight": 0.10, "role": "offensive"},
            {"name": "kmr_midcap",              "weight": 0.10, "role": "offensive"},
            {"name": "br_pullback_v1",          "weight": 0.07, "role": "offensive"},
        ]

        self.strategy_names = [cfg["name"] for cfg in self.strategy_configs]
        self.strategy_base_weight = {
            cfg["name"]: cfg.get("weight", 1.0) for cfg in self.strategy_configs
        }
        self.strategy_roles = {
            cfg["name"]: cfg.get("role", "offensive") for cfg in self.strategy_configs
        }

    def get_name(self):
        return "multi_allocator_plus_midcap"

    def get_description(self):
        return (
            "Multi-strategy allocator PLUS Midcap "
            "(소형주 52%→37%, kmr_midcap 10% + br_pullback 7% 추가)"
        )
