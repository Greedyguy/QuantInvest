#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiStrategyAllocator PLUS (no ETF strategy)
- 기존 multi_allocator_plus 에서 etf_defensive 자식전략만 제거한 버전
"""

from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus


class MultiStrategyAllocatorPlusNoETF(MultiStrategyAllocatorPlus):
    def __init__(self):
        # 먼저 기존 PLUS 설정으로 초기화
        super().__init__()

        # etf_defensive 를 제외한 자식 전략만 사용
        self.strategy_configs = [
            {"name": "kqm_small_cap_v22_short", "weight": 0.35, "role": "short"},
            {"name": "hybrid_portfolio_v2_4",   "weight": 0.25, "role": "offensive"},
            {"name": "kqm_small_cap_v22",       "weight": 0.25, "role": "offensive"},
            {"name": "k200_mean_rev",           "weight": 0.15, "role": "offensive"},
        ]

        # base 클래스가 사용하는 메타 정보도 함께 갱신
        self.strategy_names = [cfg["name"] for cfg in self.strategy_configs]
        self.strategy_base_weight = {
            cfg["name"]: cfg.get("weight", 1.0) for cfg in self.strategy_configs
        }
        self.strategy_roles = {
            cfg["name"]: cfg.get("role", "offensive") for cfg in self.strategy_configs
        }

    def get_name(self):
        return "multi_allocator_plus_no_etf"

    def get_description(self):
        return "Multi-strategy allocator PLUS (no etf_defensive child)"
