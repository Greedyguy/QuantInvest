#!/usr/bin/env python3
"""Development-only backtest for the pre-registered DART factor candidate."""

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

from backtest_live_execution import simulate
from cash_distribution_data import (
    load_distribution_bundle,
    validate_distribution_coverage_manifest,
)
from config import FEE_PER_SIDE, VENUE_FEE_PER_SIDE
from fundamental_candidate import (
    build_dart_quality_value_targets,
    historical_kospi_sell_tax_rate,
)
from krx_execution_data import (
    assert_no_unmodelled_corporate_actions,
    load_actual_close_panel,
    restore_actual_price_panel,
)
from market_benchmark import (
    MarketOutperformanceCriteria,
    evaluate_market_outperformance,
    load_distribution_events,
    load_samsung_kodex_standard_xls,
    restore_actual_ohlc,
)
from point_in_time_constituents import load_point_in_time_constituents
from scripts.backtest_k200_reentry import load_naver_prices


DEVELOPMENT_START = pd.Timestamp("2018-06-29")
DEVELOPMENT_END = pd.Timestamp("2022-12-29")
PRICE_WARMUP_START = pd.Timestamp("2017-06-01")
CORE_TICKER = "069500"
TOTAL_FEE_PER_SIDE = FEE_PER_SIDE + VENUE_FEE_PER_SIDE


def same_timing_kodex200_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Replace every risk-on basket by 95% KODEX 200 and 5% cash."""

    invested = targets.drop(columns="__CASH__", errors="ignore").sum(axis=1).gt(0.0)
    return pd.DataFrame(
        {
            CORE_TICKER: np.where(invested, 0.95, 0.0),
            "__CASH__": np.where(invested, 0.05, 1.0),
        },
        index=targets.index,
    )


def continuous_kodex200_targets(index: pd.Index) -> pd.DataFrame:
    """Represent investable 100% KODEX 200 long-term ownership."""

    return pd.DataFrame(
        {CORE_TICKER: 1.0, "__CASH__": 0.0},
        index=pd.DatetimeIndex(index),
    )


def assert_development_coverage(decisions: pd.DataFrame) -> None:
    """Refuse to produce a selection result from a materially incomplete panel."""

    scheduled = decisions.loc[
        decisions["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & decisions["risk_on"]
        & decisions["reason"].eq("quarterly_snapshot")
    ]
    if scheduled.empty:
        raise RuntimeError("no risk-on development snapshots were evaluated")
    failed = scheduled.loc[
        scheduled["fundamental_members"].lt(18)
        | scheduled["index_weight_coverage_pct"].lt(55.0)
    ]
    if not failed.empty:
        diagnostics = failed[
            [
                "signal_date",
                "fundamental_members",
                "index_weight_coverage_pct",
            ]
        ].to_dict("records")
        raise RuntimeError(f"development fundamental coverage gate failed: {diagnostics}")


def _load_price_bases(
    constituents: pd.DataFrame,
    *,
    actual_closes_path: Path,
    adjusted_price_dir: Path,
    adjusted_price_template: str,
    core_adjusted_price_file: Path,
    core_standard_file: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    actual_closes = load_actual_close_panel(actual_closes_path)
    if actual_closes["date"].max() > DEVELOPMENT_END:
        raise ValueError("actual-close input opens the sealed post-development period")
    tickers = sorted(
        constituents.loc[
            constituents["as_of_date"].le(DEVELOPMENT_END), "ticker"
        ].unique()
    )
    adjusted: dict[str, pd.DataFrame] = {}
    missing = []
    for ticker in tickers:
        path = adjusted_price_dir / adjusted_price_template.format(ticker=ticker)
        if not path.exists():
            missing.append(str(path))
            continue
        adjusted[ticker] = load_naver_prices(path).loc[
            lambda frame: frame.index.to_series().between(
                PRICE_WARMUP_START, DEVELOPMENT_END
            )
        ]
    if missing:
        raise FileNotFoundError(f"missing adjusted signal price files: {missing}")
    prices = restore_actual_price_panel(adjusted, actual_closes)

    core_adjusted = load_naver_prices(core_adjusted_price_file).loc[
        lambda frame: frame.index.to_series().between(
            PRICE_WARMUP_START, DEVELOPMENT_END
        )
    ]
    core_official = load_samsung_kodex_standard_xls(core_standard_file)
    core_close = core_official.loc[
        core_official.index.to_series().between(PRICE_WARMUP_START, DEVELOPMENT_END),
        "market_close",
    ]
    adjusted[CORE_TICKER] = core_adjusted
    prices[CORE_TICKER] = restore_actual_ohlc(core_adjusted, core_close)
    return adjusted, prices


def _summary(equity: pd.DataFrame, initial_cash: float, trades: list[dict]) -> dict:
    curve = equity.loc[DEVELOPMENT_START:DEVELOPMENT_END, "equity"]
    curve = curve / float(curve.iloc[0])
    return {
        "return_pct": float(curve.iloc[-1] - 1.0) * 100.0,
        "mdd_pct": float((curve / curve.cummax() - 1.0).min()) * 100.0,
        "orders": sum(trade["action"] in {"BUY", "SELL"} for trade in trades),
        "distributions": sum(
            trade["action"] == "DISTRIBUTION" for trade in trades
        ),
        "explicit_cost_pct_initial": sum(
            float(trade.get("fee", 0.0)) + float(trade.get("tax", 0.0))
            for trade in trades
        )
        / initial_cash
        * 100.0,
    }


def _simulate(
    targets: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    distributions: dict[str, pd.DataFrame],
    *,
    initial_cash: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, list[dict]]:
    return simulate(
        targets,
        prices,
        initial_cash=initial_cash,
        min_trade=50_000,
        price_band_pct=100.0,
        blocked_tickers=set(),
        sell_tax_rate_resolver=historical_kospi_sell_tax_rate,
        rebalance_only_on_target_change=True,
        distribution_events_by_ticker=distributions,
        fee_per_side=TOTAL_FEE_PER_SIDE * cost_multiplier,
        slippage_entry=0.002 * cost_multiplier,
        slippage_exit=0.002 * cost_multiplier,
    )


def run_development_backtest(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    signal_prices: dict[str, pd.DataFrame],
    actual_prices: dict[str, pd.DataFrame],
    stock_distributions: dict[str, pd.DataFrame],
    kodex_distributions: pd.DataFrame,
    *,
    initial_cash: float = 2_100_000.0,
) -> dict:
    targets, decisions = build_dart_quality_value_targets(
        constituents,
        fundamentals,
        signal_prices,
        actual_prices=actual_prices,
    )
    assert_development_coverage(decisions)
    targets = targets.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    assert_no_unmodelled_corporate_actions(
        targets, signal_prices, actual_prices
    )
    same_timing_targets = same_timing_kodex200_targets(targets)
    continuous_targets = continuous_kodex200_targets(targets.index)
    selected = {
        ticker
        for value in decisions["selected"].dropna().astype(str)
        for ticker in value.split(",")
        if ticker
    }
    zero_event_tickers = sorted(selected - set(stock_distributions))
    candidate_distributions = {
        ticker: events
        for ticker, events in stock_distributions.items()
        if ticker in selected
    }
    candidate_distributions[CORE_TICKER] = kodex_distributions
    benchmark_distributions = {CORE_TICKER: kodex_distributions}

    candidate_equity, candidate_trades = _simulate(
        targets,
        actual_prices,
        initial_cash=initial_cash,
        distributions=candidate_distributions,
    )
    same_equity, same_trades = _simulate(
        same_timing_targets,
        actual_prices,
        initial_cash=initial_cash,
        distributions=benchmark_distributions,
    )
    continuous_equity, continuous_trades = _simulate(
        continuous_targets,
        actual_prices,
        initial_cash=initial_cash,
        distributions=benchmark_distributions,
    )
    stressed_equity, stressed_trades = _simulate(
        targets,
        actual_prices,
        initial_cash=initial_cash,
        distributions=candidate_distributions,
        cost_multiplier=2.0,
    )
    continuous_criteria = MarketOutperformanceCriteria(
        min_annualised_excess_return=0.01
    )
    same_criteria = MarketOutperformanceCriteria(
        min_annualised_excess_return=0.02
    )
    comparisons = {
        "continuous_kodex200": evaluate_market_outperformance(
            candidate_equity["equity"],
            continuous_equity["equity"],
            continuous_criteria,
        ),
        "same_timing_kodex200": evaluate_market_outperformance(
            candidate_equity["equity"], same_equity["equity"], same_criteria
        ),
        "double_cost_continuous_kodex200": evaluate_market_outperformance(
            stressed_equity["equity"],
            continuous_equity["equity"],
            continuous_criteria,
        ),
        "double_cost_same_timing_kodex200": evaluate_market_outperformance(
            stressed_equity["equity"], same_equity["equity"], same_criteria
        ),
    }
    passes = all(item["passes_all_gates"] for item in comparisons.values())
    return {
        "candidate": _summary(candidate_equity, initial_cash, candidate_trades),
        "continuous_kodex200": _summary(
            continuous_equity, initial_cash, continuous_trades
        ),
        "same_timing_kodex200": _summary(same_equity, initial_cash, same_trades),
        "double_cost_candidate": _summary(
            stressed_equity, initial_cash, stressed_trades
        ),
        "comparisons": comparisons,
        "selected_tickers": sorted(selected),
        "selected_zero_event_tickers": zero_event_tickers,
        "development_passes": passes,
        "next_step": (
            "open_2023_2024_walk_forward"
            if passes
            else "reject_without_opening_sealed_periods"
        ),
        "decisions": decisions.loc[
            decisions["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        ].replace({np.nan: None}).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fundamentals", type=Path, required=True)
    parser.add_argument("--actual-closes", type=Path, required=True)
    parser.add_argument("--stock-distributions", type=Path, required=True)
    parser.add_argument("--stock-distribution-manifest", type=Path, required=True)
    parser.add_argument(
        "--constituents",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex200_top30_pcf_quarterly_2018_2025.csv",
    )
    parser.add_argument("--adjusted-price-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--adjusted-price-template", default="pcf_naver_{ticker}.xml")
    parser.add_argument(
        "--core-adjusted-price-file",
        type=Path,
        default=Path("/private/tmp/naver_069500.xml"),
    )
    parser.add_argument(
        "--core-standard-file",
        type=Path,
        default=Path("/private/tmp/kodex200_standard.xls"),
    )
    parser.add_argument(
        "--kodex-distributions",
        type=Path,
        default=PROJECT_ROOT / "data" / "reference" / "kodex200_distributions.csv",
    )
    parser.add_argument(
        "--kodex-distribution-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex200_distributions_manifest.json",
    )
    parser.add_argument("--initial-cash", type=float, default=2_100_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    constituents = load_point_in_time_constituents(args.constituents)
    constituents = constituents.loc[constituents["as_of_date"].le(DEVELOPMENT_END)]
    fundamentals = pd.read_csv(args.fundamentals, dtype={"ticker": str})
    signal_prices, actual_prices = _load_price_bases(
        constituents,
        actual_closes_path=args.actual_closes,
        adjusted_price_dir=args.adjusted_price_dir,
        adjusted_price_template=args.adjusted_price_template,
        core_adjusted_price_file=args.core_adjusted_price_file,
        core_standard_file=args.core_standard_file,
    )
    targets, decisions = build_dart_quality_value_targets(
        constituents,
        fundamentals,
        signal_prices,
        actual_prices=actual_prices,
    )
    assert_development_coverage(decisions)
    selected = {
        ticker
        for value in decisions.loc[
            decisions["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END),
            "selected",
        ].dropna().astype(str)
        for ticker in value.split(",")
        if ticker
    }
    stock_distributions = load_distribution_bundle(
        args.stock_distributions,
        args.stock_distribution_manifest,
        required_tickers=selected,
        start_date=DEVELOPMENT_START,
        end_date=DEVELOPMENT_END,
    )
    kodex_manifest = json.loads(
        args.kodex_distribution_manifest.read_text(encoding="utf-8")
    )
    validate_distribution_coverage_manifest(
        kodex_manifest,
        required_tickers={CORE_TICKER},
        start_date=DEVELOPMENT_START,
        end_date=DEVELOPMENT_END,
    )
    result = run_development_backtest(
        constituents,
        fundamentals,
        signal_prices,
        actual_prices,
        stock_distributions,
        load_distribution_events(args.kodex_distributions),
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
