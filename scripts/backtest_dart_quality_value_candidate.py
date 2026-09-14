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
from fundamental_candidate import (
    build_dart_quality_value_targets,
    historical_kospi_sell_tax_rate,
)
from market_benchmark import evaluate_market_outperformance
from point_in_time_constituents import load_point_in_time_constituents
from scripts.backtest_k200_reentry import load_naver_prices


DEVELOPMENT_START = pd.Timestamp("2018-06-29")
DEVELOPMENT_END = pd.Timestamp("2022-12-29")
CORE_TICKER = "069500"


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


def _load_prices(
    constituents: pd.DataFrame,
    *,
    price_dir: Path,
    price_template: str,
    core_price_file: Path,
) -> dict[str, pd.DataFrame]:
    prices = {
        CORE_TICKER: load_naver_prices(core_price_file).loc[lambda frame: frame.index <= DEVELOPMENT_END]
    }
    tickers = sorted(
        constituents.loc[
            constituents["as_of_date"].le(DEVELOPMENT_END), "ticker"
        ].unique()
    )
    missing = []
    for ticker in tickers:
        path = price_dir / price_template.format(ticker=ticker)
        if not path.exists():
            missing.append(str(path))
            continue
        prices[ticker] = load_naver_prices(path).loc[
            lambda frame: frame.index <= DEVELOPMENT_END
        ]
    if missing:
        raise FileNotFoundError(f"missing price files: {missing}")
    return prices


def _summary(equity: pd.DataFrame, initial_cash: float, trades: list[dict]) -> dict:
    curve = equity.loc[DEVELOPMENT_START:DEVELOPMENT_END, "equity"]
    curve = curve / float(curve.iloc[0])
    return {
        "return_pct": float(curve.iloc[-1] - 1.0) * 100.0,
        "mdd_pct": float((curve / curve.cummax() - 1.0).min()) * 100.0,
        "orders": sum(trade["action"] in {"BUY", "SELL"} for trade in trades),
        "explicit_cost_pct_initial": sum(
            float(trade.get("fee", 0.0)) + float(trade.get("tax", 0.0))
            for trade in trades
        )
        / initial_cash
        * 100.0,
    }


def run_development_backtest(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    initial_cash: float = 2_100_000.0,
) -> dict:
    targets, decisions = build_dart_quality_value_targets(
        constituents, fundamentals, prices
    )
    assert_development_coverage(decisions)
    targets = targets.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    benchmark_targets = same_timing_kodex200_targets(targets)
    candidate_equity, candidate_trades = simulate(
        targets,
        prices,
        initial_cash=initial_cash,
        min_trade=50_000,
        price_band_pct=100.0,
        blocked_tickers=set(),
        sell_tax_rate_resolver=historical_kospi_sell_tax_rate,
        rebalance_only_on_target_change=True,
    )
    benchmark_equity, benchmark_trades = simulate(
        benchmark_targets,
        prices,
        initial_cash=initial_cash,
        min_trade=50_000,
        price_band_pct=100.0,
        blocked_tickers=set(),
        sell_tax_rate_by_ticker={CORE_TICKER: 0.0},
        rebalance_only_on_target_change=True,
    )
    stressed_equity, stressed_trades = simulate(
        targets,
        prices,
        initial_cash=initial_cash,
        min_trade=50_000,
        price_band_pct=100.0,
        blocked_tickers=set(),
        sell_tax_rate_resolver=historical_kospi_sell_tax_rate,
        rebalance_only_on_target_change=True,
        fee_per_side=0.000140527 * 2.0,
        slippage_entry=0.002 * 2.0,
        slippage_exit=0.002 * 2.0,
    )
    comparison = evaluate_market_outperformance(
        candidate_equity["equity"], benchmark_equity["equity"]
    )
    stress_comparison = evaluate_market_outperformance(
        stressed_equity["equity"], benchmark_equity["equity"]
    )
    passes = bool(comparison["passes_all_gates"] and stress_comparison["passes_all_gates"])
    return {
        "candidate": _summary(candidate_equity, initial_cash, candidate_trades),
        "same_timing_kodex200": _summary(
            benchmark_equity, initial_cash, benchmark_trades
        ),
        "double_cost_candidate": _summary(
            stressed_equity, initial_cash, stressed_trades
        ),
        "same_timing_comparison": comparison,
        "double_cost_same_timing_comparison": stress_comparison,
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
    parser.add_argument(
        "--constituents",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex200_top30_pcf_quarterly_2018_2025.csv",
    )
    parser.add_argument("--price-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--price-template", default="pcf_naver_{ticker}.xml")
    parser.add_argument(
        "--core-price-file", type=Path, default=Path("/private/tmp/naver_069500.xml")
    )
    parser.add_argument("--initial-cash", type=float, default=2_100_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    constituents = load_point_in_time_constituents(args.constituents)
    constituents = constituents.loc[constituents["as_of_date"].le(DEVELOPMENT_END)]
    fundamentals = pd.read_csv(args.fundamentals, dtype={"ticker": str})
    prices = _load_prices(
        constituents,
        price_dir=args.price_dir,
        price_template=args.price_template,
        core_price_file=args.core_price_file,
    )
    result = run_development_backtest(
        constituents, fundamentals, prices, initial_cash=args.initial_cash
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
