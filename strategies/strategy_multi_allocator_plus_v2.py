#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental variant of MultiStrategyAllocatorPlus with tighter protection logic
"""

from strategies.strategy_multi_allocator import MultiStrategyAllocator
import pandas as pd
import numpy as np


class MultiStrategyAllocatorPlusV2(MultiStrategyAllocator):
    def __init__(self):
        super().__init__(
            strategy_configs=[
                {"name": "kqm_small_cap_v22_short", "weight": 0.30, "role": "short"},
                {"name": "hybrid_portfolio_v2_4", "weight": 0.20, "role": "offensive"},
                {"name": "kqm_small_cap_v22", "weight": 0.18, "role": "offensive"},
                {"name": "etf_defensive", "weight": 0.22, "role": "defensive"},
                {"name": "k200_mean_rev", "weight": 0.10, "role": "offensive"},
            ],
            regime_exposure={
                "bull": 1.06,
                "neutral": 0.78,
                "bear": 0.44,
                "ultra_bear": 0.26,
            },
            vol_target=None,
            vol_target_map={
                "bull": 0.20,
                "neutral": 0.155,
                "soft_bear": 0.12,
                "bear": 0.10,
                "ultra_bear": 0.085,
            },
            exposure_floor=0.2,
            performance_window=30,
            protection_window=60,
            role_target_blend=0.7,
            fast_momentum_window=12,
            role_floor_stress={
                1: {"short": 0.26, "defensive": 0.34},
                2: {"short": 0.36, "defensive": 0.42},
            },
            regime_role_targets={
                "bull": {"offensive": 0.60, "defensive": 0.18, "short": 0.22},
                "neutral": {"offensive": 0.48, "defensive": 0.32, "short": 0.20},
                "soft_bear": {"offensive": 0.32, "defensive": 0.38, "short": 0.30},
                "bear": {"offensive": 0.24, "defensive": 0.42, "short": 0.34},
                "ultra_bear": {"offensive": 0.18, "defensive": 0.43, "short": 0.39},
            },
        )

    def get_name(self):
        return "multi_allocator_plus_v2"

    def get_description(self):
        return "Multi-strategy allocator PLUS v2 (defensive tuned experimental)"

    def _dynamic_exposure(self, regime_df, dates):
        base_expo = super()._dynamic_exposure(regime_df, dates)
        if regime_df is None or regime_df.empty:
            return base_expo
        reg = regime_df.reindex(dates).ffill()
        adj = base_expo.copy()
        for date in dates:
            if date not in reg.index:
                continue
            row = reg.loc[date]
            mom5 = row.get("mom5", np.nan)
            mom20 = row.get("mom20", np.nan)
            mom1m = row.get("mom1m", np.nan)
            mom3m = row.get("mom3m", np.nan)
            close = row.get("close", np.nan)
            ma60 = row.get("ma60", np.nan)
            expo = adj.loc[date]
            if not np.isnan(mom5):
                if mom5 < -0.04:
                    expo = min(expo, 0.32)
                elif mom5 < -0.02:
                    expo = min(expo, 0.45)
                elif mom5 > 0.04 and (np.isnan(mom20) or mom20 > 0):
                    expo = max(expo, 0.85)
                elif mom5 > 0.02 and (np.isnan(mom20) or mom20 > -0.005):
                    expo = max(expo, 0.7)
            if not np.isnan(ma60) and not np.isnan(close):
                if close < ma60:
                    expo = min(expo, 0.5 if (np.isnan(mom20) or mom20 > -0.02) else 0.35)
                elif close > ma60 and (np.isnan(mom1m) or mom1m > 0.015) and (np.isnan(mom3m) or mom3m > 0.04):
                    expo = max(expo, 0.9)
            adj.loc[date] = np.clip(expo, self.exposure_floor, 1.2)
        return adj

    def _apply_recent_acceleration(self, weights, fast_signal):
        base_adjusted = super()._apply_recent_acceleration(weights, fast_signal)
        if base_adjusted.empty:
            return base_adjusted
        fast_signal = fast_signal.reindex(base_adjusted.index).fillna(0.0)
        adjusted = base_adjusted.copy()
        offensive = [k for k, v in self.strategy_roles.items() if v in ("offensive", "hybrid")]
        defensive = [k for k, v in self.strategy_roles.items() if v == "defensive"]
        short = [k for k, v in self.strategy_roles.items() if v == "short"]
        for date in adjusted.index:
            sig = fast_signal.loc[date]
            row = adjusted.loc[date]
            if sig > 0.02:
                row = self._scale_row(row, offensive, 1.18)
                row = self._scale_row(row, defensive, 0.92)
                row = self._scale_row(row, short, 0.8)
            if sig > 0.045:
                row = self._scale_row(row, offensive, 1.28)
                row = self._scale_row(row, defensive, 0.88)
                row = self._scale_row(row, short, 0.75)
            if sig < -0.015:
                row = self._scale_row(row, offensive, 0.78)
                row = self._scale_row(row, defensive, 1.1)
                row = self._scale_row(row, short, 1.18)
            if sig < -0.03:
                row = self._scale_row(row, offensive, 0.6)
                row = self._scale_row(row, defensive, 1.22)
                row = self._scale_row(row, short, 1.32)
            if sig < -0.05:
                row = self._scale_row(row, offensive, 0.5)
                row = self._scale_row(row, defensive, 1.30)
                row = self._scale_row(row, short, 1.40)
            adjusted.loc[date] = row
        row_sum = adjusted.sum(axis=1).replace(0, pd.NA)
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        adjusted = adjusted.div(row_sum, axis=0).fillna(base)
        return adjusted

    def _apply_momentum_exposure_boost(self, exposures, fast_signal):
        base_adj = super()._apply_momentum_exposure_boost(exposures, fast_signal)
        if base_adj.empty:
            return base_adj
        fast_signal = fast_signal.reindex(base_adj.index).fillna(0.0)
        adj = base_adj.copy()
        for date in adj.index:
            sig = fast_signal.loc[date]
            if sig < -0.025:
                adj.loc[date] = max(adj.loc[date] * (1 + sig * 2.6), self.exposure_floor + 0.01)
            elif sig < -0.012:
                adj.loc[date] = max(adj.loc[date] * (1 + sig * 2.0), self.exposure_floor + 0.02)
            elif sig > 0.03:
                adj.loc[date] = min(adj.loc[date] * (1 + min(sig * 2.3, 0.2)), 1.20)
            if sig > 0.045:
                adj.loc[date] = max(adj.loc[date], 0.78)
        return adj

    def _performance_stress(self, exposures, blended_ret):
        if exposures is None or exposures.empty:
            return super()._performance_stress(exposures, blended_ret)
        blended = blended_ret.reindex(exposures.index).fillna(0.0)
        eq = (1.0 + blended).cumprod()
        running_max = eq.cummax().replace(0, pd.NA)
        dd = eq / running_max - 1.0
        min_periods = max(15, self.protection_window // 2)
        roll = (1.0 + blended).rolling(self.protection_window, min_periods=min_periods).apply(lambda x: np.prod(x) - 1.0, raw=True)
        roll = roll.fillna(0.0)
        stress_levels = pd.Series(0, index=exposures.index)
        adj = exposures.copy()
        for date in exposures.index:
            expo = exposures.loc[date]
            f = 1.0
            level = 0
            r = roll.loc[date] if date in roll.index else 0.0
            d = dd.loc[date] if date in dd.index else 0.0
            if r < -0.01 or d < -0.015:
                f *= 0.88
                level = max(level, 1)
            if r < -0.025 or d < -0.03:
                f *= 0.75
                level = max(level, 1)
            if r < -0.045 or d < -0.05:
                f *= 0.6
                level = max(level, 2)
            if d < -0.08:
                f *= 0.45
                level = max(level, 2)
            if r > 0.045 and d > -0.025:
                f *= 1.14
            if r > 0.085 and d > -0.01:
                f *= 1.2
            boost = 1.0
            if d > -0.012 and r > 0.02:
                boost = 1.06
            if d > -0.006 and r > 0.035:
                boost = 1.12
            cap = 1.12
            if d <= -0.025:
                cap = min(cap, 0.65)
            if d <= -0.04:
                cap = min(cap, 0.55)
            if level >= 2:
                cap = min(cap, 0.45)
            min_cap = self.exposure_floor + 0.03
            if d > -0.018 and r > 0.02:
                min_cap = max(min_cap, 0.55)
            if d > -0.01 and r > 0.04:
                min_cap = max(min_cap, 0.65)
            val = expo * f * boost
            val = max(val, min_cap)
            adj.loc[date] = min(val, cap)
            stress_levels.loc[date] = level
        return adj.clip(lower=self.exposure_floor), stress_levels

    def compute_security_targets(self, enriched, market_index=None, weights_override=None, silent=True):
        """자식 전략의 시그널을 결합하여 실거래용 티커별 목표 비중을 계산"""
        child_results = self._run_child_strategies(enriched, market_index, weights_override=weights_override, silent=silent)
        if not child_results:
            return pd.DataFrame()

        child_returns = self._build_child_returns(child_results)
        ret_df = pd.concat(child_returns.values(), axis=1).fillna(0.0)
        ret_df.columns = list(child_returns.keys())
        shared_index = ret_df.index
        if shared_index.empty:
            return pd.DataFrame()

        regime_df = self._prepare_regime(market_index)
        expos = self._dynamic_exposure(regime_df, ret_df.index)
        expos = expos.shift(1).fillna(self.regime_exposure.get("neutral", 0.7))
        expos = expos.clip(lower=self.exposure_floor, upper=1.2)

        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        raw_weights = self._dynamic_strategy_weights(ret_df).reindex(shared_index).ffill().fillna(base)
        base_blended = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
        fast_signal = self._meta_fast_signal(base_blended)
        expos, stress_levels = self._performance_stress(expos.reindex(shared_index), base_blended)
        expos = self._apply_momentum_exposure_boost(expos, fast_signal)
        strategy_weights = self._apply_regime_bias(raw_weights, expos, stress_levels=stress_levels)
        strategy_weights = self._apply_performance_filter(strategy_weights, ret_df)
        strategy_weights = self._apply_fast_momentum_boost(strategy_weights, ret_df)
        strategy_weights = self._apply_recent_acceleration(strategy_weights, fast_signal)
        self.latest_target_weights = strategy_weights

        weight_frames = {}
        for strat, data in child_results.items():
            frame = data.get("weights")
            if frame is None or frame.empty:
                frame = self._convert_trades_to_weights(data.get("trades", []), data.get("equity", pd.DataFrame()), shared_index, enriched)
            weight_frames[strat] = frame

        security_targets = self._combine_strategy_targets(weight_frames, strategy_weights, expos.reindex(shared_index))
        self.latest_security_weights = security_targets
        return security_targets
