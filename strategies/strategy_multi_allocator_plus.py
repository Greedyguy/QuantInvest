#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiStrategyAllocator PLUS
- Offensive bias with regime-aware exposure controls
- Preserves the version that achieved >10% long-term CAGR
"""

from strategies.strategy_multi_allocator import MultiStrategyAllocator
import pandas as pd
import numpy as np


class MultiStrategyAllocatorPlus(MultiStrategyAllocator):
    def __init__(self):
        super().__init__(
            strategy_configs=[
                {"name": "kqm_small_cap_v22_short", "weight": 0.32, "role": "short"},
                {"name": "hybrid_portfolio_v2_4", "weight": 0.22, "role": "offensive"},
                {"name": "kqm_small_cap_v22", "weight": 0.20, "role": "offensive"},
                {"name": "etf_defensive", "weight": 0.18, "role": "defensive"},
                {"name": "k200_mean_rev", "weight": 0.12, "role": "offensive"},
            ],
            regime_exposure={
                "bull": 1.12,
                "neutral": 0.82,
                "bear": 0.46,
                "ultra_bear": 0.28,
            },
            vol_target=None,
            vol_target_map={
                "bull": 0.22,
                "neutral": 0.16,
                "soft_bear": 0.13,
                "bear": 0.11,
                "ultra_bear": 0.09,
            },
            exposure_floor=0.22,
            performance_window=30,
            protection_window=60,
            role_target_blend=0.68,
            fast_momentum_window=15,
            role_floor_stress={
                1: {"short": 0.23, "defensive": 0.32},
                2: {"short": 0.34, "defensive": 0.40},
            },
            regime_role_targets={
                "bull": {"offensive": 0.64, "defensive": 0.18, "short": 0.18},
                "neutral": {"offensive": 0.48, "defensive": 0.32, "short": 0.20},
                "soft_bear": {"offensive": 0.34, "defensive": 0.38, "short": 0.28},
                "bear": {"offensive": 0.27, "defensive": 0.40, "short": 0.33},
                "ultra_bear": {"offensive": 0.20, "defensive": 0.42, "short": 0.38},
            },
        )
        # Recovery is calculated on every run so it can be observed in shadow
        # reports, but it must be explicitly enabled before it changes orders.
        self.stress_recovery_enabled = False
        self.stress_recovery_step = 0.05
        # One day must satisfy four independent confirmations at once; further
        # rungs are separated by a cooldown so exposure still rises gradually.
        self.stress_recovery_confirm_days = 1
        self.stress_recovery_cooldown_days = 3
        self.latest_stress_baseline_exposure = None
        self.latest_recovery_candidate_exposure = None
        self.latest_recovery_context = None

    def get_name(self):
        return "multi_allocator_plus"

    def get_description(self):
        return "Multi-strategy allocator PLUS (regime-aware offensive blend)"

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
                row = self._scale_row(row, offensive, 0.8)
                row = self._scale_row(row, defensive, 1.08)
                row = self._scale_row(row, short, 1.12)
            if sig < -0.035:
                row = self._scale_row(row, offensive, 0.65)
                row = self._scale_row(row, defensive, 1.18)
                row = self._scale_row(row, short, 1.25)
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
            if sig < -0.02:
                adj.loc[date] = max(adj.loc[date] * (1 + sig * 2.2), self.exposure_floor + 0.02)
            elif sig > 0.035:
                adj.loc[date] = min(adj.loc[date] * (1 + min(sig * 2.0, 0.18)), 1.18)
            if sig > 0.05:
                adj.loc[date] = max(adj.loc[date], 0.72)
        return adj

    def _stress_recovery_candidate(self, exposures, stressed, stress_levels, blended_ret):
        """Build a slow recovery path without weakening the fast risk cut.

        A recovery rung is earned only after a rebound from a recent trough,
        positive short-term returns, and easing volatility are confirmed.  A
        fresh low or sharp loss removes every rung immediately.  The loop only
        uses data available at each date, keeping the result prefix invariant.
        """
        if stressed is None or stressed.empty:
            return stressed, pd.DataFrame(index=getattr(stressed, "index", None))

        index = stressed.index
        base = exposures.reindex(index).ffill().fillna(self.exposure_floor)
        levels = stress_levels.reindex(index).fillna(0).astype(int)
        blended = blended_ret.reindex(index).fillna(0.0)
        equity = (1.0 + blended).cumprod()
        prior_trough = equity.shift(1).rolling(20, min_periods=5).min()
        trough = equity.rolling(20, min_periods=1).min().replace(0, pd.NA)
        rebound = (equity / trough - 1.0).fillna(0.0)
        ret5 = equity.pct_change(5).fillna(0.0)
        ret10 = equity.pct_change(10).fillna(0.0)
        vol5 = blended.rolling(5, min_periods=5).std()
        vol20 = blended.rolling(20, min_periods=10).std()

        candidate = stressed.copy()
        rows = []
        rung = 0
        recovery_anchor = None
        confirm_window = []
        cooldown = 0
        prior_level = 0

        for date in index:
            daily_ret = float(blended.loc[date])
            five_day = float(ret5.loc[date])
            ten_day = float(ret10.loc[date])
            rebound_at_date = float(rebound.loc[date])
            short_vol = vol5.loc[date]
            long_vol = vol20.loc[date]
            vol_easing = bool(
                pd.notna(short_vol)
                and pd.notna(long_vol)
                and float(short_vol) <= float(long_vol) * 0.95
            )
            previous_low = prior_trough.loc[date]
            fresh_low = bool(
                pd.notna(previous_low)
                and float(equity.loc[date]) < float(previous_low) * 0.995
            )
            level = int(levels.loc[date])
            shock = (
                daily_ret <= -0.025
                or five_day <= -0.04
                or fresh_low
                or level > prior_level
            )
            confirmed = (
                level >= 1
                and rebound_at_date >= 0.025
                and five_day >= 0.012
                and ten_day > 0.0
                and vol_easing
            )

            if shock:
                rung = 0
                recovery_anchor = None
                confirm_window = []
                cooldown = 0
            else:
                confirm_window.append(bool(confirmed))
                confirm_window = confirm_window[
                    -max(self.stress_recovery_confirm_days + 1, 3):
                ]
                if cooldown > 0:
                    cooldown -= 1
                if (
                    sum(confirm_window) >= self.stress_recovery_confirm_days
                    and cooldown == 0
                ):
                    rung += 1
                    stressed_now = float(stressed.loc[date])
                    recovery_anchor = (
                        max(stressed_now, recovery_anchor or stressed_now)
                        + self.stress_recovery_step
                    )
                    confirm_window = []
                    cooldown = self.stress_recovery_cooldown_days
                elif rung > 0 and (
                    five_day <= 0.0
                    or (
                        pd.notna(short_vol)
                        and pd.notna(long_vol)
                        and float(short_vol) > float(long_vol) * 1.25
                    )
                ):
                    rung -= 1
                    if recovery_anchor is not None:
                        recovery_anchor = max(
                            float(stressed.loc[date]),
                            recovery_anchor - self.stress_recovery_step,
                        )
                    cooldown = max(cooldown, 1)

            if level <= 0:
                rung = max(rung - 1, 0)
                recovery_anchor = None

            recovery_cap = 0.45 if level >= 2 else (0.65 if level == 1 else 0.85)
            if level <= 0:
                recovered = float(stressed.loc[date])
            else:
                recovered_floor = min(
                    recovery_anchor
                    if recovery_anchor is not None
                    else float(stressed.loc[date]),
                    recovery_cap,
                    float(base.loc[date]),
                )
                # Recovery is an upside overlay only.  It must never turn into
                # an additional exposure cut when the baseline has recovered.
                recovered = max(float(stressed.loc[date]), recovered_floor)
            candidate.loc[date] = max(recovered, self.exposure_floor)
            rows.append({
                "recovery_rung": int(rung),
                "recovery_anchor": recovery_anchor,
                "confirmed": bool(confirmed),
                "confirmation_count": int(sum(confirm_window)),
                "shock": bool(shock),
                "rebound_20d": rebound_at_date,
                "return_5d": five_day,
                "return_10d": ten_day,
                "vol_easing": bool(vol_easing),
            })
            prior_level = level

        context = pd.DataFrame(rows, index=index)
        return candidate.clip(lower=self.exposure_floor), context

    def _performance_stress(self, exposures, blended_ret):
        if exposures is None or exposures.empty:
            return super()._performance_stress(exposures, blended_ret)
        blended = blended_ret.reindex(exposures.index).fillna(0.0)
        eq = (1.0 + blended).cumprod()
        # rolling 252일 고점 기준 — start_date 길이에 무관하게 일정한 stress 판단
        rolling_max = eq.rolling(252, min_periods=20).max().replace(0, pd.NA)
        running_max = rolling_max.fillna(eq.cummax().replace(0, pd.NA))
        dd = eq / running_max - 1.0
        min_periods = max(15, self.protection_window // 2)
        roll = (1.0 + blended).rolling(self.protection_window, min_periods=min_periods).apply(lambda x: np.prod(x) - 1.0, raw=True)
        roll = roll.fillna(0.0)
        # warmup 감쇠는 각 날짜까지 관측된 길이만 사용한다. 전체 입력 길이를 쓰면
        # 미래 데이터가 추가됐을 때 과거 stress 판정까지 바뀌는 look-ahead가 생긴다.
        warmup = pd.Series(
            np.minimum(np.arange(1, len(exposures) + 1) / 252.0, 1.0),
            index=exposures.index,
        )
        stress_levels = pd.Series(0, index=exposures.index)
        adj = exposures.copy()
        for date in exposures.index:
            expo = exposures.loc[date]
            f = 1.0
            level = 0
            warmup_at_date = float(warmup.loc[date])
            r = (roll.loc[date] if date in roll.index else 0.0) * warmup_at_date
            d = (dd.loc[date] if date in dd.index else 0.0) * warmup_at_date
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
            if r > 0.05 and d > -0.02:
                f *= 1.12
            if r > 0.09 and d > -0.01:
                f *= 1.18
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
        baseline = adj.clip(lower=self.exposure_floor)
        recovery, recovery_context = self._stress_recovery_candidate(
            exposures,
            baseline,
            stress_levels,
            blended,
        )
        self.latest_stress_baseline_exposure = baseline.copy()
        self.latest_recovery_candidate_exposure = recovery.copy()
        self.latest_recovery_context = recovery_context.copy()
        if self.stress_recovery_enabled:
            return recovery, stress_levels
        return baseline, stress_levels

    def compute_security_targets(
        self,
        enriched,
        market_index=None,
        secondary_index=None,
        weights_override=None,
        silent=True,
        target_policy="legacy",
    ):
        """자식 전략 결합 신호를 기반으로 티커별 목표 비중 계산.

        ``target_policy='preserve_child_cash'`` is reserved for audited
        comparisons.  The default intentionally preserves production output.
        """
        child_results = self._run_child_strategies(enriched, market_index, weights_override=weights_override, silent=silent)
        if not child_results:
            return pd.DataFrame()

        child_returns = self._build_child_returns(child_results)
        ret_df = pd.concat(child_returns.values(), axis=1).fillna(0.0)
        ret_df.columns = list(child_returns.keys())
        self.latest_child_results = child_results
        self.latest_child_returns = ret_df.copy()
        shared_index = ret_df.index
        if shared_index.empty:
            return pd.DataFrame()

        regime_df = self._prepare_regime(market_index, secondary_index=secondary_index)
        expos = self._dynamic_exposure(regime_df, ret_df.index)
        expos = expos.shift(1).fillna(self.regime_exposure.get("neutral", 0.7))
        expos = expos.clip(lower=self.exposure_floor, upper=1.2)
        self.latest_regime_context = regime_df
        self.latest_base_exposure = expos.copy()

        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        raw_weights = self._dynamic_strategy_weights(ret_df).reindex(shared_index).ffill().fillna(base)
        base_blended = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
        fast_signal = self._meta_fast_signal(base_blended)
        expos, stress_levels = self._performance_stress(expos.reindex(shared_index), base_blended)
        self.latest_stress_exposure = expos.copy()
        expos = self._apply_momentum_exposure_boost(expos, fast_signal)
        self.latest_final_exposure = expos.copy()
        self.latest_stress_levels = stress_levels.copy()
        self.latest_fast_signal = fast_signal.copy()
        strategy_weights = self._apply_regime_bias(raw_weights, expos, stress_levels=stress_levels)
        strategy_weights = self._apply_performance_filter(strategy_weights, ret_df)
        strategy_weights = self._apply_fast_momentum_boost(strategy_weights, ret_df)
        strategy_weights = self._apply_recent_acceleration(strategy_weights, fast_signal)
        style_context = self._prepare_style_context(market_index, secondary_index=secondary_index, dates=shared_index)
        strategy_weights = self._apply_style_rotation(strategy_weights, style_context)
        self.latest_style_context = style_context
        self.latest_target_weights = strategy_weights

        weight_frames = {}
        for strat, data in child_results.items():
            frame = data.get("weights")
            if frame is None or frame.empty:
                frame = self._convert_trades_to_weights(
                    data.get("trades", []),
                    data.get("equity", pd.DataFrame()),
                    shared_index,
                    enriched,
                )
            else:
                frame = frame.copy()
                frame.index = pd.to_datetime(frame.index)
                frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
            weight_frames[strat] = frame
        self.latest_child_weight_frames = weight_frames

        security_style_map = self._build_security_style_map(enriched)
        if target_policy == "legacy":
            combine_targets = self._combine_strategy_targets
        elif target_policy == "preserve_child_cash":
            combine_targets = self._combine_strategy_targets_preserving_cash
        else:
            raise ValueError(f"unknown target policy: {target_policy}")

        security_targets = combine_targets(
            weight_frames,
            strategy_weights,
            expos.reindex(shared_index),
            style_context=style_context,
            security_style_map=security_style_map,
        )
        self.latest_security_style_map = security_style_map
        self.latest_security_weights = security_targets
        return security_targets
