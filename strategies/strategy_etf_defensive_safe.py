#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETFDefensiveSafe — 파생상품 교육 없이 매수 가능한 일반 ETF만 사용하는 버전
  069500  KODEX 200       (beta)     — 일반 ETF, 매수 제한 없음
  305720  TIGER 미국채10년 (balanced) — 일반 ETF, 매수 제한 없음

레짐별 동작:
  bull        → KODEX 200 (주식 베타)
  neutral     → KODEX 200
  bear        → TIGER 미국채 (채권 방어)
  ultra_bear  → TIGER 미국채
"""

from strategies.strategy_etf_defensive import ETFRiskOverlayStrategy


class ETFDefensiveSafeStrategy(ETFRiskOverlayStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            etf_universe={
                "069500": {"role": "beta"},      # KODEX 200
                "305720": {"role": "balanced"},  # TIGER 미국채10년
            },
            **kwargs,
        )

    def get_name(self) -> str:
        return "etf_defensive_safe"

    def get_description(self) -> str:
        return "ETF Defensive Safe (KODEX200 + TIGER미국채, 파생교육 불필요)"
