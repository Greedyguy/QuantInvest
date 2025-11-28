#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-strategy allocator
- Executes multiple child strategies (offensive/defensive/mean-reversion etc.)
- Converts their trade logs into daily target weights, nets overlapping tickers
- Applies regime-based total exposure slider and rolling Sharpe-based strategy weights
- Returns a single equity curve / trade log compatible with reports.py
"""

import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy
from collections import defaultdict

from utils import perf_stats
from config import FEE_PER_SIDE, TAX_RATE_SELL


class MultiStrategyAllocator(BaseStrategy):
    def __init__(
        self,
        strategy_configs=None,
        regime_exposure=None,
        sharpe_window=60,
        vol_target=None,
        vol_target_map=None,
        max_turnover=0.25,
        slippage=0.002,
        dd_penalty=0.5,
        exposure_floor=0.3,
        exposure_ema_span=5,
        momentum_window=60,
        performance_window=42,
        protection_window=84,
        role_target_blend=0.55,
        fast_momentum_window=21,
        role_floor_stress=None,
        regime_role_targets=None,
    ):
        """
        strategy_names: list of strategy identifiers to combine
        regime_exposure: {'bull':0.9, 'neutral':0.7, 'bear':0.4, 'ultra_bear':0.2}
        sharpe_window: rolling window for Sharpe-based weighting (days)
        vol_target: desired annualized volatility for the combined portfolio
        max_turnover: fraction of portfolio value allowed to change seats per rebalance
        """
        self.strategy_configs = strategy_configs or [
            {"name": "kqm_small_cap_v22_short", "weight": 0.45, "role": "short"},
            {"name": "hybrid_portfolio_v2_4", "weight": 0.25, "role": "offensive"},
            {"name": "kqm_small_cap_v22", "weight": 0.10, "role": "offensive"},
            {"name": "etf_defensive", "weight": 0.20, "role": "defensive"},
        ]
        self.strategy_names = [cfg["name"] for cfg in self.strategy_configs]
        self.strategy_base_weight = {cfg["name"]: cfg.get("weight", 1.0) for cfg in self.strategy_configs}
        self.strategy_roles = {cfg["name"]: cfg.get("role", "offensive") for cfg in self.strategy_configs}
        self.regime_exposure = regime_exposure or {
            "bull": 1.05,
            "neutral": 0.8,
            "bear": 0.45,
            "ultra_bear": 0.25,
        }
        self.sharpe_window = sharpe_window
        self.vol_target = vol_target
        self.vol_target_map = vol_target_map or {
            "bull": 0.18,
            "neutral": 0.13,
            "soft_bear": 0.11,
            "bear": 0.095,
            "ultra_bear": 0.08,
        }
        self.max_turnover = max_turnover
        self.slippage = slippage
        self.dd_penalty = dd_penalty
        self.exposure_floor = exposure_floor
        self.exposure_ema_span = exposure_ema_span
        self.momentum_window = momentum_window
        self.sharpe_window = sharpe_window
        self.latest_target_weights = None
        self.performance_window = performance_window
        self.protection_window = protection_window
        self.role_target_blend = role_target_blend
        self.fast_momentum_window = fast_momentum_window
        self.role_floor_stress = role_floor_stress or {
            1: {"short": 0.22, "defensive": 0.28},
            2: {"short": 0.32, "defensive": 0.35},
        }
        self.regime_role_targets = regime_role_targets or {
            "bull": {"offensive": 0.60, "defensive": 0.25, "short": 0.15},
            "neutral": {"offensive": 0.45, "defensive": 0.35, "short": 0.20},
            "soft_bear": {"offensive": 0.35, "defensive": 0.38, "short": 0.27},
            "bear": {"offensive": 0.28, "defensive": 0.40, "short": 0.32},
            "ultra_bear": {"offensive": 0.20, "defensive": 0.42, "short": 0.38},
        }

    def get_name(self):
        return "multi_allocator"

    def get_description(self):
        return "Multi-strategy allocator (regime-based exposure + rolling Sharpe weights)"

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------
    def _prepare_regime(self, market_index):
        if market_index is None or "close" not in market_index.columns:
            return None
        idx = market_index.copy().sort_index()
        idx = idx[["close"]].astype(float)
        idx["ma60"] = idx["close"].rolling(60).mean()
        close = idx["close"]
        idx["mom20"] = close.pct_change(20)
        idx["mom5"] = close.pct_change(5)
        idx["mom1m"] = close.pct_change(21)
        idx["mom3m"] = close.pct_change(63)
        idx["mom10d"] = close.pct_change(10)
        idx["vol20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        return idx

    def _classify_regime(self, row):
        close, ma60, mom20, mom5 = row.close, row.ma60, row.mom20, row.mom5
        if np.isnan(ma60) or np.isnan(mom20):
            return "neutral"
        if close > ma60 and mom20 > 0:
            return "bull"
        if close < ma60 and mom20 < -0.01 and mom5 < 0:
            return "ultra_bear"
        if close < ma60 or mom20 < -0.02:
            return "bear"
        return "neutral"

    def _dynamic_exposure(self, regime_df, dates):
        if regime_df is None or regime_df.empty:
            return pd.Series(self.regime_exposure.get("neutral", 0.7), index=dates)
        reg = regime_df.reindex(dates).ffill()
        expos = []
        for date in dates:
            row = reg.loc[date]
            mom1 = row.get("mom1m", np.nan)
            mom3 = row.get("mom3m", np.nan)
            mom10 = row.get("mom10d", np.nan)
            vol = row.get("vol20", np.nan)
            expo = self.regime_exposure.get("neutral", 0.7)
            if np.isnan(mom1) or np.isnan(mom3) or np.isnan(mom10):
                expos.append(expo)
                continue
            if mom1 > 0.03 and mom3 > 0.08 and mom10 > 0:
                expo = 0.95
            elif mom1 > 0.01 and mom3 > 0.03:
                expo = 0.80
            elif mom3 > 0:
                expo = 0.6
            elif mom1 > -0.03:
                expo = 0.4
            else:
                expo = 0.18
            if not np.isnan(vol) and vol > 0.45:
                expo = min(expo * 0.7, 0.4)
            expos.append(expo)
        return pd.Series(expos, index=dates)

    def _dynamic_strategy_weights(self, ret_df):
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        if ret_df.empty:
            return pd.DataFrame([base], index=ret_df.index)
        roll = (1.0 + ret_df).rolling(self.momentum_window, min_periods=20).apply(lambda x: np.prod(x) - 1.0, raw=True)
        roll = roll.shift(1).fillna(0.0)
        sharpe = ret_df.rolling(self.sharpe_window, min_periods=20).mean() / (ret_df.rolling(self.sharpe_window, min_periods=20).std() + 1e-9)
        sharpe = sharpe.shift(1).fillna(0.0)
        perf_adj = (1.0 + roll).clip(lower=0.1, upper=2.0)
        sharpe_adj = (1.0 + sharpe).clip(lower=0.2, upper=2.0)
        adj = perf_adj * sharpe_adj
        weights = adj.copy()
        for col in base.index:
            if col not in weights.columns:
                weights[col] = 1.0
        weights = weights[base.index]
        weights = weights.mul(base, axis=1)
        row_sum = weights.sum(axis=1)
        zero_rows = row_sum <= 0
        if zero_rows.any():
            weights.loc[zero_rows, :] = base.values
            row_sum = weights.sum(axis=1)
        weights = weights.div(row_sum.replace(0, np.nan), axis=0).fillna(base)
        return weights

    def _scale_row(self, row, cols, factor):
        if factor == 1.0 or not cols:
            return row
        valid = [c for c in cols if c in row.index]
        if not valid:
            return row
        row.loc[valid] *= factor
        return row

    def _regime_from_exposure(self, expo):
        if expo >= 0.95:
            return "bull"
        if expo >= 0.8:
            return "neutral"
        if expo >= 0.6:
            return "soft_bear"
        if expo >= 0.45:
            return "bear"
        return "ultra_bear"

    def _vol_target_series(self, exposures):
        if exposures is None or exposures.empty:
            return pd.Series(self.vol_target or 0.0, index=[])
        targets = []
        for expo in exposures:
            regime = self._regime_from_exposure(expo)
            targets.append(self.vol_target_map.get(regime, self.vol_target or 0.0))
        return pd.Series(targets, index=exposures.index)

    def _blend_role_target(self, row, expo):
        if row.sum() <= 0:
            return row
        regime_key = self._regime_from_exposure(expo)
        target = self.regime_role_targets.get(regime_key)
        if not target or self.role_target_blend <= 0:
            return row
        blend = self.role_target_blend
        total = row.sum()
        row = row / total
        for role, target_share in target.items():
            members = [c for c, r in self.strategy_roles.items() if r == role and c in row.index]
            if not members:
                continue
            current = row[members].sum()
            desired = (1 - blend) * current + blend * target_share
            if current <= 0:
                equal = desired / len(members)
                row.loc[members] = equal
            else:
                factor = desired / current if current > 0 else 0.0
                row.loc[members] *= factor
        total = row.sum()
        if total > 0:
            row = row / total
        return row

    def _apply_performance_filter(self, weights, ret_df):
        if weights.empty:
            return weights
        min_periods = max(20, self.performance_window // 2)
        roll_prod = (1.0 + ret_df).rolling(self.performance_window, min_periods=min_periods).apply(lambda x: np.prod(x) - 1.0, raw=True).shift(1)
        roll_sharpe = ret_df.rolling(self.performance_window, min_periods=min_periods).mean() / (ret_df.rolling(self.performance_window, min_periods=min_periods).std() + 1e-9)
        roll_sharpe = roll_sharpe.shift(1)
        rr = roll_prod.reindex_like(weights)
        sp = roll_sharpe.reindex_like(weights)
        scale = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)

        cond = rr < -0.04
        scale = scale.mask(cond, scale * 0.7)
        cond = rr < -0.10
        scale = scale.mask(cond, scale * 0.6)
        cond = sp < 0.0
        scale = scale.mask(cond, scale * 0.85)
        cond = sp < -0.25
        scale = scale.mask(cond, scale * 0.65)

        adjusted = weights * scale
        row_sum = adjusted.sum(axis=1).replace(0, np.nan)
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        adjusted = adjusted.div(row_sum, axis=0).fillna(base)
        return adjusted

    def _apply_fast_momentum_boost(self, weights, ret_df):
        if weights.empty:
            return weights
        fast = (1.0 + ret_df).rolling(self.fast_momentum_window, min_periods=max(10, self.fast_momentum_window // 2)).apply(lambda x: np.prod(x) - 1.0, raw=True)
        fast = fast.shift(1).reindex_like(weights).fillna(0.0)
        boosted = weights.copy()
        for date in boosted.index:
            row = boosted.loc[date]
            factors = 1.0 + fast.loc[date] * 1.8
            factors = factors.clip(lower=0.7, upper=1.35)
            row = row * factors
            boosted.loc[date] = row
        row_sum = boosted.sum(axis=1).replace(0, np.nan)
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        boosted = boosted.div(row_sum, axis=0).fillna(base)
        return boosted

    def _enforce_role_floor(self, row, floors):
        total = row.sum()
        if total <= 0:
            return row
        norm = row / total
        for role, min_share in floors.items():
            members = [c for c, r in self.strategy_roles.items() if r == role and c in norm.index]
            if not members:
                continue
            floor_each = min_share / len(members)
            norm.loc[members] = np.maximum(norm.loc[members], floor_each)
        norm = norm / norm.sum()
        return norm * total

    def _apply_recent_acceleration(self, weights, fast_signal):
        if weights.empty:
            return weights
        fast_signal = fast_signal.reindex(weights.index).fillna(0.0)
        accelerated = weights.copy()
        offensive = [k for k, v in self.strategy_roles.items() if v in ("offensive", "hybrid")]
        defensive = [k for k, v in self.strategy_roles.items() if v == "defensive"]
        short = [k for k, v in self.strategy_roles.items() if v == "short"]
        for date in accelerated.index:
            sig = fast_signal.loc[date]
            row = accelerated.loc[date]
            if sig > 0.02:
                row = self._scale_row(row, offensive, 1.25)
                row = self._scale_row(row, defensive, 0.9)
                row = self._scale_row(row, short, 0.7)
            elif sig < -0.025:
                row = self._scale_row(row, offensive, 0.75)
                row = self._scale_row(row, defensive, 1.15)
                row = self._scale_row(row, short, 1.25)
            accelerated.loc[date] = row
        row_sum = accelerated.sum(axis=1).replace(0, np.nan)
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        accelerated = accelerated.div(row_sum, axis=0).fillna(base)
        return accelerated

    def _meta_fast_signal(self, series):
        if series.empty:
            return pd.Series()
        window = self.fast_momentum_window
        min_periods = max(8, window // 2)
        fast = (1.0 + series).rolling(window, min_periods=min_periods).apply(lambda x: np.prod(x) - 1.0, raw=True)
        return fast.shift(1).fillna(0.0)

    def _apply_momentum_exposure_boost(self, exposures, fast_signal):
        if exposures.empty:
            return exposures
        fast_signal = fast_signal.reindex(exposures.index).fillna(0.0)
        adj = exposures.copy()
        for date in exposures.index:
            sig = fast_signal.loc[date]
            expo = adj.loc[date]
            if sig > 0.02:
                expo *= 1 + min(sig * 3.0, 0.12)
            elif sig < -0.025:
                expo *= 1 + max(sig * 3.0, -0.2)
            adj.loc[date] = expo
        return adj.clip(lower=self.exposure_floor, upper=1.2)

    def _performance_stress(self, exposures, blended_ret):
        if exposures is None or exposures.empty:
            return exposures, pd.Series(0, index=exposures.index if exposures is not None else [])
        blended = blended_ret.reindex(exposures.index).fillna(0.0)
        eq = (1.0 + blended).cumprod()
        running_max = eq.cummax().replace(0, np.nan)
        dd = eq / running_max - 1.0
        min_periods = max(20, self.protection_window // 2)
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
            if r < -0.02 or d < -0.03:
                f *= 0.9
                level = max(level, 1)
            if r < -0.045 or d < -0.05:
                f *= 0.8
                level = max(level, 1)
            if r < -0.08 or d < -0.08:
                f *= 0.65
                level = max(level, 2)
            if d < -0.2:
                f *= 0.5
                level = max(level, 2)
            if r > 0.04 and d > -0.04:
                f *= 1.08
            if r > 0.08 and d > -0.02:
                f *= 1.12
            cap = 1.05
            if d <= -0.03:
                cap = min(cap, 0.7)
            if d <= -0.05:
                cap = min(cap, 0.58)
            if level >= 2:
                cap = min(cap, 0.5)
            adj.loc[date] = min(expo * f, cap)
            stress_levels.loc[date] = level
        return adj.clip(lower=self.exposure_floor), stress_levels

    def _apply_regime_bias(self, weights, exposures, stress_levels=None):
        if weights.empty:
            return weights
        exposures = exposures.reindex(weights.index).ffill().fillna(self.regime_exposure.get("neutral", 0.7))
        biased = weights.copy()
        offensive = [k for k, v in self.strategy_roles.items() if v in ("offensive", "hybrid")]
        defensive = [k for k, v in self.strategy_roles.items() if v == "defensive"]
        short = [k for k, v in self.strategy_roles.items() if v == "short"]
        for date in biased.index:
            expo = exposures.loc[date]
            row = biased.loc[date]
            stress = stress_levels.loc[date] if stress_levels is not None and date in stress_levels.index else 0
            if expo >= 0.9:
                row = self._scale_row(row, offensive, 1.25)
                row = self._scale_row(row, short, 0.4)
                row = self._scale_row(row, defensive, 0.6)
            elif expo >= 0.75:
                row = self._scale_row(row, offensive, 1.1)
                row = self._scale_row(row, short, 0.65)
            elif expo <= 0.4:
                row = self._scale_row(row, offensive, 0.65)
                row = self._scale_row(row, short, 1.4)
                row = self._scale_row(row, defensive, 1.3)
            elif expo <= 0.55:
                row = self._scale_row(row, offensive, 0.85)
                row = self._scale_row(row, short, 1.15)
                row = self._scale_row(row, defensive, 1.15)
            if stress >= 1:
                row = self._scale_row(row, offensive, 0.7 if stress == 1 else 0.5)
                row = self._scale_row(row, defensive, 1.15 if stress == 1 else 1.25)
                row = self._scale_row(row, short, 1.25 if stress == 1 else 1.4)
            row = self._blend_role_target(row, expo)
            if stress in self.role_floor_stress:
                row = self._enforce_role_floor(row, self.role_floor_stress[stress])
            biased.loc[date] = row
        row_sum = biased.sum(axis=1).replace(0, np.nan)
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        biased = biased.div(row_sum, axis=0)
        biased = biased.fillna(base)
        return biased

    def _get_strategy(self, name):
        from strategies import get_strategy
        return get_strategy(name)

    def _run_child_strategies(self, enriched, market_index, weights_override=None, silent=True):
        child_results = {}
        for name in self.strategy_names:
            try:
                if not silent:
                    print(f"[multi_allocator] running child strategy: {name}")
                strat = self._get_strategy(name)
            except Exception as e:
                print(f"[WARN] multi-allocator: strategy {name} unavailable: {e}")
                continue

            try:
                ec, trades = strat.run_backtest(enriched, market_index=market_index, weights=weights_override, silent=silent)
                if ec is None or ec.empty:
                    if not silent:
                        print(f"[multi_allocator] {name} returned empty results")
                    continue
                weight_history = strat.get_target_weight_history()
                child_results[name] = {"equity": ec, "trades": trades, "weights": weight_history}
                if not silent:
                    print(f"[multi_allocator] completed {name}: {ec.shape}")
            except Exception as e:
                print(f"[WARN] multi-allocator: strategy {name} run failed: {e}")
        return child_results

    def _compute_daily_returns(self, equity_df):
        eq = equity_df["equity"].astype(float)
        return eq.pct_change().fillna(0.0)

    def _build_child_returns(self, child_results):
        ret_map = {}
        for name, data in child_results.items():
            ret_map[name] = self._compute_daily_returns(data["equity"])
        return ret_map

    def _rolling_sharpe_weights(self, ret_df):
        # ret_df: DataFrame (date x strategy) of daily returns
        if ret_df.empty:
            return pd.DataFrame()

        rolling_mean = ret_df.rolling(self.sharpe_window, min_periods=10).mean()
        rolling_std = ret_df.rolling(self.sharpe_window, min_periods=10).std()
        sharpe = rolling_mean / (rolling_std + 1e-9)
        sharpe = sharpe.fillna(0.0)

        weights = sharpe.clip(lower=0.0)
        weight_sum = weights.sum(axis=1).replace(0, np.nan)
        weights = weights.div(weight_sum, axis=0).fillna(0.0)
        return weights

    def _drawdown_penalty(self, ret_df):
        if ret_df.empty:
            return pd.DataFrame()
        equity = (1.0 + ret_df).cumprod()
        running_max = equity.cummax()
        dd = equity / (running_max + 1e-12) - 1.0  # negative values
        penalty = np.exp(dd * self.dd_penalty)
        penalty = penalty.clip(lower=0.1)
        return penalty

    def _risk_parity_weight(self, cov_mat):
        n = cov_mat.shape[0]
        if n == 0:
            return np.array([])
        w = np.ones(n) / n
        for _ in range(100):
            portfolio_var = w @ cov_mat @ w
            if portfolio_var <= 0:
                break
            contributions = w * (cov_mat @ w)
            target = portfolio_var / n
            grad = contributions - target
            w -= 0.01 * grad
            w = np.clip(w, 1e-4, None)
            w /= w.sum()
        return w

    def _risk_parity_weights(self, ret_df):
        if ret_df.empty:
            return pd.DataFrame()
        rolling_cov = ret_df.rolling(self.sharpe_window, min_periods=10).cov(pairwise=True)
        weights = []
        for date in ret_df.index:
            try:
                cov_slice = rolling_cov.xs(date, level=0)
            except KeyError:
                weights.append(pd.Series(np.nan, index=ret_df.columns, name=date))
                continue
            cov_slice = cov_slice.reindex(index=ret_df.columns, columns=ret_df.columns).fillna(0.0)
            w = self._risk_parity_weight(cov_slice.values)
            if w.size == 0:
                weights.append(pd.Series(np.nan, index=ret_df.columns, name=date))
            else:
                weights.append(pd.Series(w, index=ret_df.columns, name=date))
        rp_df = pd.DataFrame(weights)
        return rp_df

    def _get_price(self, df, date, field="close"):
        if df is None:
            return np.nan
        if date in df.index:
            col = field if field in df.columns else "close"
            val = df.loc[date, col]
            if np.isfinite(val):
                return float(val)
        idx = df.index[df.index <= date]
        if len(idx) == 0:
            return np.nan
        col = field if field in df.columns else "close"
        val = df.loc[idx.max(), col]
        return float(val) if np.isfinite(val) else np.nan

    def _convert_trades_to_weights(self, trades, equity_df, dates, enriched):
        if not trades:
            return pd.DataFrame(0.0, index=dates, columns=[])

        trade_map = defaultdict(list)
        for tr in trades:
            d = tr.get("date")
            if d is None:
                continue
            trade_map[pd.to_datetime(d)].append(tr)

        holdings = defaultdict(float)
        cash = float(equity_df.iloc[0]["equity"]) if not equity_df.empty else 1_000_000.0
        weight_rows = {}
        for date in dates:
            for tr in trade_map.get(date, []):
                ticker = tr.get("ticker")
                qty = tr.get("qty", 0)
                price = tr.get("price", 0)
                action = tr.get("action", "BUY")
                if ticker is None or qty <= 0 or price <= 0:
                    continue
                if action == "BUY":
                    holdings[ticker] += qty
                    cash -= qty * price
                else:
                    holdings[ticker] -= qty
                    cash += qty * price
                    if holdings[ticker] <= 0:
                        holdings.pop(ticker, None)

            values = {}
            total = cash
            for ticker, qty in holdings.items():
                price = self._get_price(enriched.get(ticker), date)
                if np.isnan(price):
                    continue
                val = qty * price
                if val <= 0:
                    continue
                values[ticker] = val
                total += val

            row = {}
            if total > 0:
                for ticker, val in values.items():
                    row[ticker] = val / total
                row["__CASH__"] = max(cash / total, 0.0)
            else:
                row["__CASH__"] = 1.0

            weight_rows[date] = row

        df = pd.DataFrame(weight_rows).T.fillna(0.0)
        df = df.reindex(dates).ffill().fillna(0.0)
        return df

    def _combine_strategy_targets(self, weight_frames, sharpe_weights, exposures):
        all_dates = sharpe_weights.index
        columns = sorted(set().union(*[df.columns for df in weight_frames.values()])) if weight_frames else []
        combined = pd.DataFrame(0.0, index=all_dates, columns=columns)

        for strat, frame in weight_frames.items():
            if strat not in sharpe_weights.columns:
                continue
            strat_w = sharpe_weights[strat]
            aligned = frame.reindex(all_dates).fillna(0.0)
            shared_cols = aligned.columns
            combined = combined.reindex(columns=sorted(set(combined.columns) | set(shared_cols)), fill_value=0.0)
            aligned = aligned.reindex(columns=combined.columns, fill_value=0.0)
            for date in all_dates:
                combined.loc[date] += strat_w.loc[date] * aligned.loc[date]

        result = combined.fillna(0.0)
        target = pd.DataFrame(0.0, index=all_dates, columns=result.columns)
        for date in all_dates:
            expo = exposures.loc[date]
            row = result.loc[date].copy()
            if "__CASH__" in row.index:
                cash_weight = row["__CASH__"]
                row = row.drop("__CASH__")
            else:
                cash_weight = 0.0
            total = row.sum()
            if total > 0:
                row = row * (expo / total)
            else:
                row = row * 0.0
            target.loc[date, row.index] = row
            target.loc[date, "__CASH__"] = max(1 - expo, 0.0)
        return target

    def _simulate_portfolio(self, target_weights, enriched, exposure_series=None):
        if target_weights.empty:
            return pd.DataFrame(), []

        holdings = defaultdict(float)
        cash = 1_000_000.0
        equity_curve = []
        trade_log = []

        for date in target_weights.index:
            row = target_weights.loc[date].fillna(0.0)
            target_cash_weight = row.get("__CASH__", 0.0)
            target_row = row.drop("__CASH__", errors="ignore")
            expo = exposure_series.loc[date] if exposure_series is not None and date in exposure_series.index else None

            # 1) 현재 에쿼티 및 포지션 가치
            equity = cash
            current_values = {}
            for ticker, qty in holdings.items():
                price = self._get_price(enriched.get(ticker), date)
                if np.isnan(price):
                    continue
                val = qty * price
                current_values[ticker] = val
                equity += val

            # 2) 타깃 가치
            desired_values = {}
            for ticker, weight in target_row.items():
                desired_values[ticker] = weight * equity

            # 3) delta 계산
            delta_values = {}
            tickers = set(current_values.keys()) | set(desired_values.keys())
            for ticker in tickers:
                cur_val = current_values.get(ticker, 0.0)
                des_val = desired_values.get(ticker, 0.0)
                delta_values[ticker] = des_val - cur_val

            # 4) 턴오버 계산 및 캡
            turnover = 0.0
            for val in delta_values.values():
                turnover += abs(val)
            turnover = turnover / (2 * equity) if equity > 0 else 0.0
            turnover_cap = self.max_turnover
            if expo is not None:
                if expo < 0.2:
                    turnover_cap = min(turnover_cap, 0.05)
                elif expo < 0.35:
                    turnover_cap = min(turnover_cap, 0.08)
                elif expo < 0.5:
                    turnover_cap = min(turnover_cap, 0.12)
                elif expo < 0.7:
                    turnover_cap = min(turnover_cap, 0.18)
            scale = 1.0
            if turnover_cap > 0 and turnover > turnover_cap:
                scale = turnover_cap / turnover

            # 5) 체결
            for ticker, delta_val in delta_values.items():
                delta_val *= scale
                if abs(delta_val) < 1e-6:
                    continue
                price = self._get_price(enriched.get(ticker), date, field="open")
                if np.isnan(price) or price <= 0:
                    continue
                shares_float = delta_val / price
                qty = int(abs(shares_float))
                if qty <= 0:
                    continue
                action = "BUY" if delta_val > 0 else "SELL"
                exec_price = price * (1 + self.slippage) if action == "BUY" else price * (1 - self.slippage)
                gross = qty * exec_price
                fee = gross * FEE_PER_SIDE
                if action == "BUY":
                    cash_out = gross + fee
                    if cash_out > cash:
                        continue
                    cash -= cash_out
                    holdings[ticker] += qty
                else:
                    proceeds = gross
                    tax = proceeds * TAX_RATE_SELL
                    cash_in = proceeds - fee - tax
                    cash += cash_in
                    holdings[ticker] -= qty
                    if holdings[ticker] <= 0:
                        holdings.pop(ticker, None)
                trade_log.append({
                    "date": date,
                    "ticker": ticker,
                    "action": action,
                    "price": exec_price,
                    "qty": qty,
                    "cash_after": cash,
                })

            # 6) 일별 에쿼티 기록
            equity = cash
            for ticker, qty in holdings.items():
                price = self._get_price(enriched.get(ticker), date)
                if np.isnan(price):
                    continue
                equity += qty * price
            equity_curve.append((date, equity))

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        return ec, trade_log

    # ---------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):
        if not enriched:
            return pd.DataFrame(), []

        if not silent:
            print("[multi_allocator] launching child strategies:", self.strategy_names)
        child_results = self._run_child_strategies(enriched, market_index, weights_override=weights, silent=True)
        if not silent:
            print(f"[multi_allocator] finished child strategies: {list(child_results.keys())}")
        if not child_results:
            return pd.DataFrame(), []

        # Build returns for weighting
        child_returns = self._build_child_returns(child_results)
        ret_df = pd.concat(child_returns.values(), axis=1).fillna(0.0)
        ret_df.columns = list(child_returns.keys())
        if not silent:
            print(f"[multi_allocator] ret_df shape: {ret_df.shape}, dates {ret_df.index.min()} ~ {ret_df.index.max()}")

        regime_df = self._prepare_regime(market_index)
        expos = self._dynamic_exposure(regime_df, ret_df.index)
        expos = expos.shift(1).fillna(self.regime_exposure.get("neutral", 0.7))
        expos = expos.clip(lower=self.exposure_floor, upper=1.0)

        shared_index = ret_df.index
        base = pd.Series(self.strategy_base_weight)
        base = base / base.sum()
        raw_weights = self._dynamic_strategy_weights(ret_df).reindex(shared_index).ffill().fillna(base)
        expos = expos.reindex(shared_index).fillna(self.regime_exposure["neutral"])
        base_blended = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
        fast_signal = self._meta_fast_signal(base_blended)
        expos, stress_levels = self._performance_stress(expos, base_blended)
        expos = self._apply_momentum_exposure_boost(expos, fast_signal)
        strategy_weights = self._apply_regime_bias(raw_weights, expos, stress_levels=stress_levels)
        strategy_weights = self._apply_performance_filter(strategy_weights, ret_df)
        strategy_weights = self._apply_fast_momentum_boost(strategy_weights, ret_df)
        strategy_weights = self._apply_recent_acceleration(strategy_weights, fast_signal)

        # Signal-level blending: use strategy weights & exposures to build meta return
        self.latest_target_weights = strategy_weights
        blended = (strategy_weights * ret_df.reindex(shared_index)).sum(axis=1)
        combined_ret = expos * blended
        ann_vol = combined_ret.std() * np.sqrt(252)
        target_vol_series = self._vol_target_series(expos)
        desired_vol = None
        if self.vol_target is not None:
            desired_vol = self.vol_target
        elif not target_vol_series.empty:
            desired_vol = target_vol_series.median()
        if desired_vol and desired_vol > 0 and ann_vol > 0:
            scaler = desired_vol / ann_vol
            combined_ret = combined_ret * scaler
        equity = (1.0 + combined_ret).cumprod() * 1_000_000.0
        ec = pd.DataFrame({"equity": equity})
        ec.index.name = "date"
        trade_log = []
        if not silent:
            stats = perf_stats(ec)
            print(f"✅ Multi strategy allocator stats: {stats}")
        return ec, trade_log

    def get_latest_target_weights(self):
        return self.latest_target_weights
