#!/usr/bin/env python3
"""Development-only test of pre-registered volatility-budgeted leverage v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cash_distribution_data import (
    load_distribution_bundle,
    validate_distribution_coverage_manifest,
)
from market_benchmark import load_distribution_events
from scripts.backtest_k200_reentry import load_naver_prices
from scripts.backtest_kodex200_convex_risk_budget import (
    CORE_TICKER,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    LEVERAGE_TICKER,
    PRICE_WARMUP_START,
    _comparison,
    _simulate,
    _summary,
    load_development_execution_inputs,
    registered_gates,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


UNLEVERED = "unlevered"
MEDIUM_LEVERAGE = "medium_leverage"
HIGH_LEVERAGE = "high_leverage"
CASH = "cash"
WEIGHTS = {
    UNLEVERED: (0.95, 0.00, 0.05),
    MEDIUM_LEVERAGE: (0.75, 0.20, 0.05),
    HIGH_LEVERAGE: (0.60, 0.35, 0.05),
    CASH: (0.00, 0.00, 1.00),
}


def _tier(momentum_60: float, volatility_20: float) -> str:
    if not np.isfinite(momentum_60) or not np.isfinite(volatility_20):
        return UNLEVERED
    if momentum_60 <= 0.0 or volatility_20 > 0.22:
        return UNLEVERED
    if volatility_20 <= 0.15:
        return HIGH_LEVERAGE
    return MEDIUM_LEVERAGE


def apply_volatility_budget(
    signal_prices: pd.DataFrame, base_states: pd.DataFrame
) -> pd.DataFrame:
    """Set a causal monthly leverage tier while preserving emergency exits."""

    prices = signal_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    close = pd.to_numeric(prices["close"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame(
        {
            "momentum_60": close.pct_change(60, fill_method=None),
            "volatility_20": returns.rolling(20).std() * np.sqrt(252.0),
        }
    )
    states = base_states.copy().sort_index()
    states.index = pd.to_datetime(states.index)
    current_tier = CASH
    rows = []
    dates = list(states.index)
    for position, current_date in enumerate(dates):
        base_state = str(states.loc[current_date, "state"])
        reviewed = False
        feature_date = pd.NaT
        momentum = np.nan
        volatility = np.nan
        if base_state == K200LowTurnoverReentry.CASH:
            current_tier = CASH
        elif position == 0:
            current_tier = UNLEVERED
        elif position > 0:
            previous_date = dates[position - 1]
            first_session_of_month = current_date.month != previous_date.month
            reentered = (
                str(states.loc[previous_date, "state"])
                == K200LowTurnoverReentry.CASH
            )
            if first_session_of_month or reentered:
                reviewed = True
                feature_date = previous_date
                momentum = float(features.loc[previous_date, "momentum_60"])
                volatility = float(features.loc[previous_date, "volatility_20"])
                current_tier = _tier(momentum, volatility)
            elif current_tier == CASH:
                current_tier = UNLEVERED
        rows.append(
            {
                "date": current_date,
                "state": base_state,
                "tier": current_tier,
                "reviewed": reviewed,
                "feature_date": feature_date,
                "momentum_60": momentum,
                "volatility_20": volatility,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def targets_from_budget_history(
    budget: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Place effective next-open tiers on prior-close target rows."""

    effective_tier = budget["tier"].shift(-1).fillna(budget["tier"])
    effective_state = budget["state"].shift(-1).fillna(budget["state"])
    candidate_rows = [WEIGHTS[str(tier)] for tier in effective_tier]
    candidate = pd.DataFrame(
        candidate_rows,
        columns=[CORE_TICKER, LEVERAGE_TICKER, "__CASH__"],
        index=budget.index,
    )
    risk_on = effective_state.eq(K200LowTurnoverReentry.RISK_ON)
    same_timing = pd.DataFrame(
        {
            CORE_TICKER: np.where(risk_on, 0.95, 0.0),
            "__CASH__": np.where(risk_on, 0.05, 1.0),
        },
        index=budget.index,
    )
    continuous = pd.DataFrame(
        {CORE_TICKER: 1.0, "__CASH__": 0.0}, index=budget.index
    )
    return candidate, same_timing, continuous


def build_development_targets(
    signal_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = signal_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[PRICE_WARMUP_START:DEVELOPMENT_END]
    base_states = K200LowTurnoverReentry().compute_state_history(prices)
    budget = apply_volatility_budget(prices, base_states)
    candidate, same_timing, continuous = targets_from_budget_history(budget)
    selected = slice(DEVELOPMENT_START, DEVELOPMENT_END)
    return (
        candidate.loc[selected],
        same_timing.loc[selected],
        continuous.loc[selected],
        budget.loc[selected],
    )


def run_development_backtest(
    signal_prices: pd.DataFrame,
    execution_prices: dict[str, pd.DataFrame],
    tax_nav: pd.Series,
    kodex200_distributions: pd.DataFrame,
    *,
    initial_cash: float = 2_100_000.0,
) -> dict:
    candidate_targets, same_targets, continuous_targets, budget = (
        build_development_targets(signal_prices)
    )
    for ticker, prices in execution_prices.items():
        missing = candidate_targets.index.difference(prices.index)
        if not missing.empty:
            raise ValueError(
                f"official execution prices for {ticker} miss {missing[0].date()}"
            )
    distributions = {CORE_TICKER: kodex200_distributions}
    candidate, candidate_trades = _simulate(
        candidate_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
    )
    same, same_trades = _simulate(
        same_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
    )
    continuous, continuous_trades = _simulate(
        continuous_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
    )
    stressed, stressed_trades = _simulate(
        candidate_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
        cost_multiplier=2.0,
    )
    comparisons = {
        "continuous_kodex200": _comparison(
            candidate, continuous, annualised_excess=0.02
        ),
        "same_timing_kodex200": _comparison(
            candidate, same, annualised_excess=0.015
        ),
        "double_cost_continuous_kodex200": _comparison(
            stressed, continuous, annualised_excess=0.02
        ),
        "double_cost_same_timing_kodex200": _comparison(
            stressed, same, annualised_excess=0.015
        ),
    }
    gates = registered_gates(
        comparisons["continuous_kodex200"],
        comparisons["same_timing_kodex200"],
    )
    gates.update(
        registered_gates(
            comparisons["double_cost_continuous_kodex200"],
            comparisons["double_cost_same_timing_kodex200"],
            prefix="double_cost",
        )
    )
    passes = all(gates.values())
    tier_counts = budget["tier"].value_counts().to_dict()
    return {
        "candidate": _summary(candidate, initial_cash, candidate_trades),
        "continuous_kodex200": _summary(
            continuous, initial_cash, continuous_trades
        ),
        "same_timing_kodex200": _summary(same, initial_cash, same_trades),
        "double_cost_candidate": _summary(
            stressed, initial_cash, stressed_trades
        ),
        "comparisons": comparisons,
        "registered_gates": gates,
        "tier_sessions": {str(key): int(value) for key, value in tier_counts.items()},
        "monthly_overlay_reviews": int(budget["reviewed"].sum()),
        "development_passes": passes,
        "sealed_periods_opened": False,
        "next_step": (
            "open_2023_2024_walk_forward"
            if passes
            else "reject_without_opening_sealed_periods"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-price", type=Path, required=True)
    parser.add_argument("--actual-ohlc", type=Path, required=True)
    parser.add_argument("--tax-nav", type=Path, required=True)
    parser.add_argument(
        "--kodex200-distributions",
        type=Path,
        default=PROJECT_ROOT / "data" / "reference" / "kodex200_distributions.csv",
    )
    parser.add_argument(
        "--kodex200-distribution-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex200_distributions_manifest.json",
    )
    parser.add_argument(
        "--leverage-distributions",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex_leverage_distributions_2018_2022.csv",
    )
    parser.add_argument(
        "--leverage-distribution-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex_leverage_distributions_2018_2022_manifest.json",
    )
    parser.add_argument("--initial-cash", type=float, default=2_100_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    core_manifest = json.loads(
        args.kodex200_distribution_manifest.read_text(encoding="utf-8")
    )
    validate_distribution_coverage_manifest(
        core_manifest,
        required_tickers={CORE_TICKER},
        start_date=DEVELOPMENT_START,
        end_date=DEVELOPMENT_END,
    )
    leverage_bundle = load_distribution_bundle(
        args.leverage_distributions,
        args.leverage_distribution_manifest,
        required_tickers={LEVERAGE_TICKER},
        start_date=DEVELOPMENT_START,
        end_date=DEVELOPMENT_END,
    )
    if leverage_bundle:
        raise ValueError("pre-registered development input expected zero distributions")
    execution_prices, tax_nav = load_development_execution_inputs(
        args.actual_ohlc, args.tax_nav
    )
    signal_prices = load_naver_prices(args.signal_price).loc[
        PRICE_WARMUP_START:DEVELOPMENT_END
    ]
    result = run_development_backtest(
        signal_prices,
        execution_prices,
        tax_nav,
        load_distribution_events(args.kodex200_distributions),
        initial_cash=args.initial_cash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
