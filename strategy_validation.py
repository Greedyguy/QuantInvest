"""Causal, production-oriented helpers for strategy validation.

This module intentionally lives outside the strategy classes.  The legacy
backtest remains available as a comparison baseline while audited experiments
can share one explicit implementation of cash preservation, turnover limits,
and time-split metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


CASH = "__CASH__"


@dataclass(frozen=True)
class ValidationPeriod:
    """Named, inclusive evaluation interval."""

    name: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp


def _normalise_child_assets(frame: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    """Return long-only child assets without manufacturing invested weight.

    Child cash may be explicit or implicit.  Rows above 100% are scaled down
    because the audited Korean cash account does not assume leverage.
    """

    aligned = frame.reindex(dates).fillna(0.0).astype(float).clip(lower=0.0)
    assets = aligned.drop(columns=CASH, errors="ignore")
    asset_sum = assets.sum(axis=1)
    scale = pd.Series(1.0, index=dates)
    over = asset_sum > 1.0
    scale.loc[over] = 1.0 / asset_sum.loc[over]
    return assets.mul(scale, axis=0)


def combine_child_targets_preserving_cash(
    weight_frames: Mapping[str, pd.DataFrame],
    strategy_weights: pd.DataFrame,
    exposures: pd.Series,
    *,
    leverage_cap: float = 1.0,
) -> pd.DataFrame:
    """Combine child positions while treating outer exposure as a ceiling.

    The legacy allocator removed each child's cash and scaled the remaining
    securities *up* to the outer exposure.  Here the weighted child allocation
    is only scaled down when it exceeds the exposure ceiling.  Unallocated
    weight always remains cash.
    """

    if strategy_weights.empty:
        return pd.DataFrame(index=strategy_weights.index, columns=[CASH])
    if leverage_cap <= 0:
        raise ValueError("leverage_cap must be positive")

    dates = strategy_weights.index
    all_assets = sorted(
        {
            str(column)
            for frame in weight_frames.values()
            for column in frame.columns
            if str(column) != CASH
        }
    )
    combined = pd.DataFrame(0.0, index=dates, columns=all_assets)

    for strategy_name, frame in weight_frames.items():
        if strategy_name not in strategy_weights.columns or frame is None or frame.empty:
            continue
        child_assets = _normalise_child_assets(frame, dates)
        child_assets = child_assets.reindex(columns=all_assets, fill_value=0.0)
        meta_weight = strategy_weights[strategy_name].reindex(dates).fillna(0.0).clip(lower=0.0)
        combined = combined.add(child_assets.mul(meta_weight, axis=0), fill_value=0.0)

    invested = combined.sum(axis=1)
    ceiling = (
        exposures.reindex(dates)
        .ffill()
        .fillna(0.0)
        .clip(lower=0.0, upper=float(leverage_cap))
    )
    scale = pd.Series(1.0, index=dates)
    over = invested > ceiling
    scale.loc[over] = ceiling.loc[over] / invested.loc[over]
    combined = combined.mul(scale, axis=0).clip(lower=0.0)
    cash = (1.0 - combined.sum(axis=1)).clip(lower=0.0).rename(CASH)
    return pd.concat([combined, cash], axis=1).copy()


def cap_security_weights_to_cash(
    target_weights: pd.DataFrame,
    max_security_weight: float | None,
) -> pd.DataFrame:
    """Apply a per-security cap and leave every excess won in cash."""

    if target_weights.empty or max_security_weight is None:
        return target_weights.copy()
    if max_security_weight <= 0:
        raise ValueError("max_security_weight must be positive")
    capped = target_weights.copy().fillna(0.0)
    asset_columns = [column for column in capped.columns if column != CASH]
    capped[asset_columns] = capped[asset_columns].clip(
        lower=0.0, upper=float(max_security_weight)
    )
    capped[CASH] = (1.0 - capped[asset_columns].sum(axis=1)).clip(lower=0.0)
    return capped


def apply_turnover_cap(
    target_weights: pd.DataFrame,
    cap: float | Callable[[float], float | None] | None,
    *,
    initial_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Causally smooth targets using half-L1 portfolio turnover.

    The first row is measured from an all-cash account unless explicit initial
    weights are supplied.  This prevents a backtest starting with an unlimited
    first rebalance.
    """

    if target_weights.empty:
        return target_weights.copy()
    columns = list(target_weights.columns)
    if CASH not in columns:
        columns.append(CASH)
    desired_frame = target_weights.reindex(columns=columns, fill_value=0.0).fillna(0.0)

    if initial_weights is None:
        previous = pd.Series(0.0, index=columns)
        previous.loc[CASH] = 1.0
    else:
        previous = initial_weights.reindex(columns).fillna(0.0).clip(lower=0.0)
        total = float(previous.sum())
        previous = previous / total if total > 1.0 else previous
        previous.loc[CASH] += max(1.0 - float(previous.sum()), 0.0)

    rows: list[pd.Series] = []
    for _, desired in desired_frame.iterrows():
        desired = desired.clip(lower=0.0)
        total = float(desired.sum())
        if total <= 0:
            desired.loc[CASH] = 1.0
        elif total > 1.0:
            desired = desired / total
        else:
            desired.loc[CASH] += 1.0 - total

        exposure = float(desired.drop(CASH, errors="ignore").sum())
        row_cap = cap(exposure) if callable(cap) else cap
        if row_cap is not None:
            if row_cap < 0:
                raise ValueError("turnover cap must be non-negative")
            delta = desired - previous
            turnover = 0.5 * float(delta.abs().sum())
            if turnover > float(row_cap) and turnover > 0:
                desired = previous + delta * (float(row_cap) / turnover)
        desired = desired.clip(lower=0.0)
        desired = desired / float(desired.sum())
        rows.append(desired)
        previous = desired

    return pd.DataFrame(rows, index=target_weights.index, columns=columns)


def causal_volatility_scale(
    returns: pd.Series,
    target_volatility: float | pd.Series,
    *,
    window: int = 60,
    min_periods: int = 20,
    max_scale: float = 1.0,
) -> pd.Series:
    """Compute a volatility scale using only returns strictly before each day."""

    if window <= 1 or min_periods <= 1 or min_periods > window:
        raise ValueError("invalid volatility window/min_periods")
    if max_scale <= 0:
        raise ValueError("max_scale must be positive")

    values = returns.astype(float).fillna(0.0)
    realised = values.shift(1).rolling(window, min_periods=min_periods).std(ddof=1)
    realised = realised * np.sqrt(252.0)
    if isinstance(target_volatility, pd.Series):
        target = target_volatility.reindex(values.index).ffill()
    else:
        target = pd.Series(float(target_volatility), index=values.index)
    scale = target / realised.replace(0.0, np.nan)
    return scale.clip(lower=0.0, upper=float(max_scale)).fillna(1.0)


def performance_by_period(
    equity: pd.Series | pd.DataFrame,
    periods: Sequence[ValidationPeriod],
) -> pd.DataFrame:
    """Calculate non-overlapping return diagnostics without resetting history."""

    if isinstance(equity, pd.DataFrame):
        if "equity" not in equity.columns:
            raise ValueError("equity DataFrame must contain an 'equity' column")
        values = equity["equity"]
    else:
        values = equity
    values = values.astype(float).sort_index().dropna()
    returns = values.pct_change(fill_method=None).fillna(0.0)

    rows = []
    previous_end: pd.Timestamp | None = None
    for period in periods:
        start = pd.Timestamp(period.start)
        end = pd.Timestamp(period.end)
        if end < start:
            raise ValueError(f"period {period.name!r} ends before it starts")
        if previous_end is not None and start <= previous_end:
            raise ValueError("validation periods must be ordered and non-overlapping")
        previous_end = end

        selected = returns.loc[(returns.index >= start) & (returns.index <= end)]
        if selected.empty:
            rows.append({"period": period.name, "start": str(start.date()), "end": str(end.date()), "days": 0})
            continue
        curve = (1.0 + selected).cumprod()
        total_return = float(curve.iloc[-1] - 1.0)
        annualised = (1.0 + total_return) ** (252.0 / len(selected)) - 1.0
        volatility = float(selected.std(ddof=1) * np.sqrt(252.0)) if len(selected) > 1 else 0.0
        sharpe = (
            float(selected.mean() / selected.std(ddof=1) * np.sqrt(252.0))
            if len(selected) > 1 and selected.std(ddof=1) > 0
            else 0.0
        )
        drawdown = curve / curve.cummax() - 1.0
        rows.append(
            {
                "period": period.name,
                "start": str(selected.index.min().date()),
                "end": str(selected.index.max().date()),
                "days": int(len(selected)),
                "return_pct": total_return * 100.0,
                "cagr_pct": annualised * 100.0,
                "volatility_pct": volatility * 100.0,
                "sharpe": sharpe,
                "mdd_pct": float(drawdown.min()) * 100.0,
            }
        )
    return pd.DataFrame(rows).set_index("period")


DEFAULT_VALIDATION_PERIODS = (
    ValidationPeriod("train", "2020-01-01", "2023-12-31"),
    ValidationPeriod("validation", "2024-01-01", "2024-12-31"),
    ValidationPeriod("test", "2025-01-01", "2025-12-31"),
)
