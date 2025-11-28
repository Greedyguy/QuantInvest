#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.strategy_kqm_small_cap_v2_2_regime import (
    KQMSmallCapStrategyV22Regime,
)
from strategies.strategy_kqm_small_cap_v2_2_short import (
    KQMSmallCapStrategyV22Short,
)


class KQMSmallCapStrategyV22Blend(BaseStrategy):
    """
    KQM Small Cap v2.2 Blend
    - Regime(상승장 강점)과 Short(약세 방어) 전략을 레짐 신호 기반 가중 혼합
    - 일일 수익률 레벨에서 가중합 → 누적 에쿼티 산출
    """

    def __init__(
        self,
        weight_map=None,  # {'bull':1.0,'neutral':0.6,'bear':0.2,'ultra_bear':0.0}
        smooth_span=5,
    ):
        self.weight_map = weight_map or {
            "bull": 1.0,
            "neutral": 0.6,
            "bear": 0.2,
            "ultra_bear": 0.0,
        }
        self.smooth_span = smooth_span

        # 내부 하위 전략
        self.regime_strat = KQMSmallCapStrategyV22Regime()
        self.short_strat = KQMSmallCapStrategyV22Short()

    def get_name(self):
        return "kqm_small_cap_v22_blend"

    def get_description(self):
        return (
            "KQM Small Cap v2.2 Blend (Regime + Short 가중 혼합)"
        )

    # -------------------------------
    # 레짐 계산 (KOSDAQ 기준)
    # -------------------------------
    def _prepare_regime(self, market_index):
        if market_index is None or "close" not in market_index.columns:
            return None
        idx = market_index.copy().sort_index()
        idx = idx[["close"]].astype(float)
        idx["ma60"] = idx["close"].rolling(60).mean()
        idx["mom20"] = idx["close"].pct_change(20)
        idx["mom5"] = idx["close"].pct_change(5)
        return idx

    def _classify_regime(self, row):
        close, ma60, mom20, mom5 = row.close, row.ma60, row.mom20, row.mom5
        if np.isnan(ma60) or np.isnan(mom20):
            return "neutral"
        if close > ma60 and mom20 > 0:
            return "bull"
        if close < ma60 and (mom20 < -0.01) and (mom5 < 0):
            return "ultra_bear"
        if (close < ma60) or (mom20 < -0.02):
            return "bear"
        return "neutral"

    def _build_weights(self, dates, market_index):
        if market_index is None or len(market_index) == 0:
            w = pd.Series(0.5, index=pd.Index(dates, name="date"))
            return w

        reg = self._prepare_regime(market_index)
        if reg is None or reg.empty:
            return pd.Series(0.5, index=pd.Index(dates, name="date"))

        reg = reg.reindex(dates).ffill()
        regimes = reg.apply(self._classify_regime, axis=1)
        w = regimes.map(self.weight_map).astype(float)
        # lookahead 방지: 다음날 적용
        w = w.shift(1)
        # 스무딩 (EMA)
        if self.smooth_span and self.smooth_span > 1:
            w = w.ewm(span=self.smooth_span, adjust=False).mean()
        w = w.fillna(0.5).clip(0.0, 1.0)
        w.index.name = "date"
        return w

    # -------------------------------
    # 메인: 하위 전략 두 개 실행 후 가중합
    # -------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):
        ec_regime, _ = self.regime_strat.run_backtest(
            enriched, market_index=market_index, weights=weights, silent=True
        )
        ec_short, _ = self.short_strat.run_backtest(
            enriched, market_index=market_index, weights=weights, silent=True
        )

        # 날짜 정렬 및 수익률 계산
        all_dates = sorted(set(ec_regime.index) | set(ec_short.index))
        if not all_dates:
            return pd.DataFrame(), []

        ec_r = ec_regime.reindex(all_dates).ffill()
        ec_s = ec_short.reindex(all_dates).ffill()

        r_r = ec_r["equity"].pct_change().fillna(0.0)
        r_s = ec_s["equity"].pct_change().fillna(0.0)

        # 레짐 기반 가중
        w = self._build_weights(all_dates, market_index)
        r_c = w * r_r + (1.0 - w) * r_s

        # 에쿼티 누적
        init_cap = 1_000_000.0
        equity = (1.0 + r_c).cumprod() * init_cap
        ec = pd.DataFrame({"equity": equity})
        ec.index.name = "date"

        return ec, []

