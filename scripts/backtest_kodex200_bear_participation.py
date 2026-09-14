#!/usr/bin/env python3
"""Development-only test of pre-registered KODEX bear-participation v3."""

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
    PRICE_WARMUP_START,
    _comparison,
    _simulate,
    _summary,
    load_development_execution_inputs,
    registered_gates,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


INVERSE_TICKER = "114800"
RISK_ON = "risk_on"
BEAR = "bear"
CASH = "cash"


def _bear_confirmed(close: float, moving_average_60: float, momentum_20: float) -> bool:
    return bool(
        np.isfinite(close)
        and np.isfinite(moving_average_60)
        and np.isfinite(momentum_20)
        and close < moving_average_60
        and momentum_20 < 0.0
    )


def apply_bear_participation(
    signal_prices: pd.DataFrame, base_states: pd.DataFrame
) -> pd.DataFrame:
    """Use prior-close evidence at monthly reviews and base-state transitions."""

    prices = signal_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    close = pd.to_numeric(prices["close"], errors="coerce")
    features = pd.DataFrame(
        {
            "close": close,
            "moving_average_60": close.rolling(60).mean(),
            "momentum_20": close.pct_change(20, fill_method=None),
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
        feature_close = np.nan
        moving_average = np.nan
        momentum = np.nan
        if base_state == K200LowTurnoverReentry.RISK_ON:
            current_tier = RISK_ON
        elif position == 0:
            current_tier = CASH
        else:
            previous_date = dates[position - 1]
            previous_state = str(states.loc[previous_date, "state"])
            first_session_of_month = current_date.month != previous_date.month
            transitioned_to_cash = (
                previous_state == K200LowTurnoverReentry.RISK_ON
            )
            if first_session_of_month or transitioned_to_cash:
                reviewed = True
                feature_date = previous_date
                row = features.loc[previous_date]
                feature_close = float(row["close"])
                moving_average = float(row["moving_average_60"])
                momentum = float(row["momentum_20"])
                current_tier = (
                    BEAR
                    if _bear_confirmed(feature_close, moving_average, momentum)
                    else CASH
                )
            elif current_tier == RISK_ON:
                current_tier = CASH
        rows.append(
            {
                "date": current_date,
                "state": base_state,
                "tier": current_tier,
                "reviewed": reviewed,
                "feature_date": feature_date,
                "close": feature_close,
                "moving_average_60": moving_average,
                "momentum_20": momentum,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def targets_from_bear_history(
    budget: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Place each effective next-open tier on the prior-close target row."""

    effective_tier = budget["tier"].shift(-1).fillna(budget["tier"])
    effective_state = budget["state"].shift(-1).fillna(budget["state"])
    candidate = pd.DataFrame(0.0, index=budget.index, columns=[
        CORE_TICKER,
        INVERSE_TICKER,
        "__CASH__",
    ])
    candidate.loc[effective_tier.eq(RISK_ON), [CORE_TICKER, "__CASH__"]] = [
        0.95,
        0.05,
    ]
    candidate.loc[effective_tier.eq(BEAR), [INVERSE_TICKER, "__CASH__"]] = [
        0.30,
        0.70,
    ]
    candidate.loc[effective_tier.eq(CASH), "__CASH__"] = 1.0
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
    budget = apply_bear_participation(prices, base_states)
    candidate, same_timing, continuous = targets_from_bear_history(budget)
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
        taxed_ticker=INVERSE_TICKER,
    )
    same, same_trades = _simulate(
        same_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
        taxed_ticker=INVERSE_TICKER,
    )
    continuous, continuous_trades = _simulate(
        continuous_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
        taxed_ticker=INVERSE_TICKER,
    )
    stressed, stressed_trades = _simulate(
        candidate_targets,
        execution_prices,
        distributions,
        tax_nav,
        initial_cash=initial_cash,
        cost_multiplier=2.0,
        taxed_ticker=INVERSE_TICKER,
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
        "tier_sessions": {
            str(key): int(value) for key, value in budget["tier"].value_counts().items()
        },
        "bear_reviews": int(
            (budget["reviewed"] & budget["tier"].eq(BEAR)).sum()
        ),
        "development_passes": passes,
        "sealed_periods_opened": False,
        "next_step": (
            "open_2023_2024_walk_forward"
            if passes
            else "reject_and_stop_2018_2022_candidate_search"
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
        "--inverse-distributions",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex_inverse_distributions_2018_2022.csv",
    )
    parser.add_argument(
        "--inverse-distribution-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex_inverse_distributions_2018_2022_manifest.json",
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
    inverse_bundle = load_distribution_bundle(
        args.inverse_distributions,
        args.inverse_distribution_manifest,
        required_tickers={INVERSE_TICKER},
        start_date=DEVELOPMENT_START,
        end_date=DEVELOPMENT_END,
    )
    if inverse_bundle:
        raise ValueError("pre-registered development input expected zero distributions")
    execution_prices, tax_nav = load_development_execution_inputs(
        args.actual_ohlc,
        args.tax_nav,
        secondary_ticker=INVERSE_TICKER,
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
