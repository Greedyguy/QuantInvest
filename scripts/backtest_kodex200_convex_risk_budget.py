#!/usr/bin/env python3
"""Development-only test of the pre-registered KODEX risk-budget candidate."""

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

from backtest_live_execution import HoldingPeriodTaxProfile, simulate
from cash_distribution_data import (
    load_distribution_bundle,
    validate_distribution_coverage_manifest,
)
from config import FEE_PER_SIDE, VENUE_FEE_PER_SIDE
from krx_execution_data import actual_ohlc_for_ticker, load_actual_close_panel
from market_benchmark import (
    MarketOutperformanceCriteria,
    evaluate_market_outperformance,
    load_distribution_events,
)
from scripts.backtest_k200_reentry import load_naver_prices
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


CORE_TICKER = "069500"
LEVERAGE_TICKER = "122630"
DEVELOPMENT_START = pd.Timestamp("2018-06-29")
DEVELOPMENT_END = pd.Timestamp("2022-12-29")
PRICE_WARMUP_START = pd.Timestamp("2017-06-01")
TOTAL_FEE_PER_SIDE = FEE_PER_SIDE + VENUE_FEE_PER_SIDE


def targets_from_state_history(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Translate next-open states into candidate and investable comparators."""

    if states.empty or "state" not in states:
        raise ValueError("risk-state history is empty")
    effective_next_open = states["state"].shift(-1).fillna(states["state"])
    risk_on = effective_next_open.eq(K200LowTurnoverReentry.RISK_ON)
    candidate = pd.DataFrame(
        {
            CORE_TICKER: np.where(risk_on, 0.70, 0.0),
            LEVERAGE_TICKER: np.where(risk_on, 0.25, 0.0),
            "__CASH__": np.where(risk_on, 0.05, 1.0),
        },
        index=states.index,
    )
    same_timing = pd.DataFrame(
        {
            CORE_TICKER: np.where(risk_on, 0.95, 0.0),
            "__CASH__": np.where(risk_on, 0.05, 1.0),
        },
        index=states.index,
    )
    continuous = pd.DataFrame(
        {CORE_TICKER: 1.0, "__CASH__": 0.0}, index=states.index
    )
    return candidate, same_timing, continuous


def build_development_targets(
    signal_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen causal state model and keep only development signals."""

    prices = signal_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[PRICE_WARMUP_START:DEVELOPMENT_END]
    strategy = K200LowTurnoverReentry()
    states = strategy.compute_state_history(prices)
    candidate, same_timing, continuous = targets_from_state_history(states)
    selected = slice(DEVELOPMENT_START, DEVELOPMENT_END)
    return (
        candidate.loc[selected],
        same_timing.loc[selected],
        continuous.loc[selected],
        states.loc[selected],
    )


def load_development_tax_nav(
    path: Path, *, ticker: str = LEVERAGE_TICKER
) -> pd.DataFrame:
    """Load only a normalized development panel; reject a broken seal."""

    frame = pd.read_csv(path, dtype={"ticker": "string"}, parse_dates=["date"])
    required = {"date", "ticker", "market_close", "tax_nav", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"tax-NAV panel is missing columns: {missing}")
    frame["ticker"] = frame["ticker"].str.zfill(6)
    expected_ticker = str(ticker).zfill(6)
    if set(frame["ticker"]) != {expected_ticker}:
        raise ValueError(f"tax-NAV panel must contain only {expected_ticker}")
    if frame["date"].max() > DEVELOPMENT_END:
        raise ValueError("tax-NAV input opens the sealed post-development period")
    for column in ("market_close", "tax_nav"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["market_close", "tax_nav"]].isna().any(axis=None):
        raise ValueError("tax-NAV panel contains missing values")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("tax-NAV panel contains duplicate dates")
    return frame.sort_values("date").reset_index(drop=True)


def load_development_execution_inputs(
    actual_ohlc_path: Path,
    tax_nav_path: Path,
    *,
    secondary_ticker: str = LEVERAGE_TICKER,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Require complete official OHLC and cross-source close agreement."""

    panel = load_actual_close_panel(actual_ohlc_path)
    if panel["date"].max() > DEVELOPMENT_END:
        raise ValueError("actual-price input opens the sealed post-development period")
    prices = {
        ticker: actual_ohlc_for_ticker(panel, ticker).loc[
            DEVELOPMENT_START:DEVELOPMENT_END
        ]
        for ticker in (CORE_TICKER, secondary_ticker)
    }
    if not prices[CORE_TICKER].index.equals(prices[secondary_ticker].index):
        raise ValueError("KODEX core and secondary execution calendars differ")
    tax_panel = load_development_tax_nav(
        tax_nav_path, ticker=secondary_ticker
    ).set_index("date")
    aligned = prices[secondary_ticker][["close"]].join(
        tax_panel[["market_close", "tax_nav"]], how="outer"
    )
    if aligned.isna().any(axis=None):
        raise ValueError("KODEX secondary official OHLC and tax NAV coverage differ")
    if not np.allclose(aligned["close"], aligned["market_close"], rtol=0, atol=0):
        raise ValueError("KRX and Samsung KODEX secondary market closes disagree")
    return prices, aligned["tax_nav"]


def _simulate(
    targets: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    distributions: dict[str, pd.DataFrame],
    tax_nav: pd.Series,
    *,
    initial_cash: float,
    cost_multiplier: float = 1.0,
    taxed_ticker: str = LEVERAGE_TICKER,
) -> tuple[pd.DataFrame, list[dict]]:
    return simulate(
        targets,
        prices,
        initial_cash=initial_cash,
        min_trade=50_000,
        price_band_pct=100.0,
        blocked_tickers=set(),
        sell_tax_rate_by_ticker={CORE_TICKER: 0.0, taxed_ticker: 0.0},
        rebalance_only_on_target_change=True,
        distribution_events_by_ticker=distributions,
        holding_period_tax_by_ticker={
            taxed_ticker: HoldingPeriodTaxProfile(tax_nav=tax_nav)
        },
        fee_per_side=TOTAL_FEE_PER_SIDE * cost_multiplier,
        slippage_entry=0.002 * cost_multiplier,
        slippage_exit=0.002 * cost_multiplier,
    )


def _summary(equity: pd.DataFrame, initial_cash: float, trades: list[dict]) -> dict:
    curve = equity.loc[DEVELOPMENT_START:DEVELOPMENT_END, "equity"]
    normalized = curve / float(curve.iloc[0])
    years = (curve.index[-1] - curve.index[0]).days / 365.2425
    orders = [trade for trade in trades if trade["action"] in {"BUY", "SELL"}]
    return {
        "return_pct": float(normalized.iloc[-1] - 1.0) * 100.0,
        "cagr_pct": float(normalized.iloc[-1] ** (1.0 / years) - 1.0) * 100.0,
        "mdd_pct": float((normalized / normalized.cummax() - 1.0).min()) * 100.0,
        "orders": len(orders),
        "distributions": sum(
            trade["action"] == "DISTRIBUTION" for trade in trades
        ),
        "fees_pct_initial": sum(float(trade.get("fee", 0.0)) for trade in trades)
        / initial_cash
        * 100.0,
        "distribution_tax_pct_initial": sum(
            float(trade.get("tax", 0.0))
            for trade in trades
            if trade["action"] == "DISTRIBUTION"
        )
        / initial_cash
        * 100.0,
        "holding_period_tax_pct_initial": sum(
            float(trade.get("holding_period_tax", 0.0)) for trade in orders
        )
        / initial_cash
        * 100.0,
    }


def _comparison(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    annualised_excess: float,
) -> dict:
    return evaluate_market_outperformance(
        strategy["equity"],
        benchmark["equity"],
        MarketOutperformanceCriteria(
            min_annualised_excess_return=annualised_excess,
            min_rolling_12m_beat_rate=0.60,
            max_mdd_disadvantage=0.05,
            max_positive_excess_year_share=0.60,
        ),
    )


def registered_gates(
    continuous: dict, same_timing: dict, *, prefix: str = ""
) -> dict[str, bool]:
    """Apply only the gates fixed in the candidate specification."""

    label = f"{prefix}_" if prefix else ""
    return {
        f"{label}cagr_excess_vs_continuous": continuous[
            "annualised_excess_return_pct_point"
        ]
        >= 2.0,
        f"{label}cagr_excess_vs_same_timing": same_timing[
            "annualised_excess_return_pct_point"
        ]
        >= 1.5,
        f"{label}rolling_12m_vs_continuous": continuous[
            "rolling_12m_beat_rate_pct"
        ]
        >= 60.0,
        f"{label}rolling_12m_vs_same_timing": same_timing[
            "rolling_12m_beat_rate_pct"
        ]
        >= 60.0,
        f"{label}mdd_disadvantage_vs_same_timing": same_timing[
            "mdd_disadvantage_pct_point"
        ]
        <= 5.0,
        f"{label}positive_year_concentration_vs_continuous": continuous[
            "positive_excess_year_concentration_pct"
        ]
        <= 60.0,
        f"{label}positive_year_concentration_vs_same_timing": same_timing[
            "positive_excess_year_concentration_pct"
        ]
        <= 60.0,
    }


def run_development_backtest(
    signal_prices: pd.DataFrame,
    execution_prices: dict[str, pd.DataFrame],
    tax_nav: pd.Series,
    kodex200_distributions: pd.DataFrame,
    *,
    initial_cash: float = 2_100_000.0,
) -> dict:
    candidate_targets, same_targets, continuous_targets, states = (
        build_development_targets(signal_prices)
    )
    expected_dates = candidate_targets.index
    for ticker, prices in execution_prices.items():
        missing = expected_dates.difference(prices.index)
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
        "risk_on_sessions_pct": float(
            states["state"].eq(K200LowTurnoverReentry.RISK_ON).mean() * 100.0
        ),
        "state_changes": int(states["state"].ne(states["state"].shift()).sum() - 1),
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
