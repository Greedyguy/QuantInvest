#!/usr/bin/env python3
"""Audit the frozen KODEX sector-rotation paper-shadow candidate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_live_execution import simulate
from config import FEE_PER_SIDE, SLIPPAGE_ENTRY, SLIPPAGE_EXIT
from market_benchmark import (
    DOMESTIC_EQUITY_ETF,
    adjust_ohlc_for_distributions,
    evaluate_market_outperformance,
    load_distribution_events,
    load_samsung_distribution_json,
    load_samsung_kodex_standard_xls,
    restore_actual_ohlc,
)
from scripts.backtest_k200_reentry import (
    buy_and_hold,
    load_naver_prices,
    load_prices,
    merge_adjusted_prices,
    select_k200_file,
)
from strategies.kodex_sector_rotation import KodexSectorRotation


CORE_TICKER = "069500"
LEGACY_DISTRIBUTIONS = (
    PROJECT_ROOT / "data" / "reference" / "kodex_sector_distributions_2017_2019.csv"
)
SECTOR_PRODUCTS = {
    "091180": "자동차",
    "091170": "은행",
    "091160": "반도체",
    "102970": "증권",
    "102960": "기계장비",
    "117460": "에너지화학",
    "117700": "건설",
    "117680": "철강",
    "140700": "보험",
    "140710": "운송",
    "266410": "필수소비재",
    "266390": "경기소비재",
    "266360": "K-콘텐츠",
    "266370": "IT",
    "266420": "헬스케어",
}


def _official_execution_prices(
    ticker: str,
    naver_path: Path,
    standard_path: Path,
) -> pd.DataFrame:
    naver = load_naver_prices(naver_path)
    official = load_samsung_kodex_standard_xls(standard_path)
    common = naver.index.intersection(official.index)
    if common.empty:
        raise ValueError(f"no common actual-price dates for {ticker}")
    return restore_actual_ohlc(
        naver.loc[common], official.loc[common, "market_close"]
    )


def _load_official_inputs(data_dir: Path):
    legacy = pd.read_csv(LEGACY_DISTRIBUTIONS, dtype={"ticker": str})
    legacy["record_date"] = pd.to_datetime(legacy["record_date"])
    legacy["pay_date"] = pd.to_datetime(legacy["pay_date"])

    def with_legacy(ticker: str, recent: pd.DataFrame) -> pd.DataFrame:
        historical = legacy.loc[legacy["ticker"].eq(ticker)].drop(
            columns=["ticker", "source_sheet"]
        )
        combined = pd.concat([historical, recent], ignore_index=True, sort=False)
        return (
            combined.sort_values("record_date")
            .drop_duplicates("record_date", keep="last")
            .reset_index(drop=True)
        )

    execution = {
        CORE_TICKER: _official_execution_prices(
            CORE_TICKER,
            data_dir / f"naver_{CORE_TICKER}.xml",
            data_dir / "kodex200_standard.xls",
        )
    }
    total_return = {}
    distributions = {
        CORE_TICKER: with_legacy(
            CORE_TICKER,
            load_distribution_events(
                PROJECT_ROOT
                / "data"
                / "reference"
                / "kodex200_distributions.csv"
            ),
        )
    }
    for ticker in SECTOR_PRODUCTS:
        execution[ticker] = _official_execution_prices(
            ticker,
            data_dir / f"naver_{ticker}.xml",
            data_dir / f"kodex_actual_{ticker}.xls",
        )
        payload = json.loads((data_dir / f"kodex_tr_{ticker}.json").read_text())
        frame = pd.DataFrame(payload)
        frame["date"] = pd.to_datetime(frame["GIJUN_YMD"], format="%Y%m%d")
        total_return[ticker] = pd.Series(
            1.0 + pd.to_numeric(frame["SUIK_NAV"], errors="coerce").to_numpy() / 100.0,
            index=frame["date"],
        ).loc[lambda values: ~values.index.duplicated(keep="last")]
        distributions[ticker] = with_legacy(
            ticker,
            load_samsung_distribution_json(data_dir / f"kodex_div_{ticker}.json"),
        )
    return execution, pd.DataFrame(total_return).sort_index(), distributions


def _normalised_equity(equity: pd.DataFrame) -> pd.Series:
    values = equity["equity"].astype(float)
    return values / float(values.iloc[0])


def _account_summary(
    initial_cash: float,
    strategy_equity: pd.DataFrame,
    strategy_trades: list[dict],
    benchmark_equity: pd.DataFrame,
    benchmark_trades: list[dict],
) -> dict:
    result = evaluate_market_outperformance(
        strategy_equity["equity"], benchmark_equity["equity"]
    )
    result.update(
        {
            "initial_cash": initial_cash,
            "strategy_return_pct": (
                strategy_equity["equity"].iloc[-1]
                / strategy_equity["equity"].iloc[0]
                - 1.0
            )
            * 100.0,
            "benchmark_return_pct": (
                benchmark_equity["equity"].iloc[-1]
                / benchmark_equity["equity"].iloc[0]
                - 1.0
            )
            * 100.0,
            "strategy_orders": sum(
                row["action"] in {"BUY", "SELL"} for row in strategy_trades
            ),
            "benchmark_orders": sum(
                row["action"] in {"BUY", "SELL"} for row in benchmark_trades
            ),
            "strategy_distribution_payments": sum(
                row["action"] == "DISTRIBUTION" for row in strategy_trades
            ),
            "benchmark_distribution_payments": sum(
                row["action"] == "DISTRIBUTION" for row in benchmark_trades
            ),
        }
    )
    return result


def _run_candidate(
    targets: pd.DataFrame,
    execution: dict[str, pd.DataFrame],
    distributions: dict[str, pd.DataFrame],
    initial_cash: float,
    *,
    min_trade: int,
    cost_multiplier: float = 1.0,
):
    return simulate(
        targets,
        execution,
        initial_cash=initial_cash,
        min_trade=min_trade,
        price_band_pct=3.0,
        sell_tax_rate_by_ticker={ticker: 0.0 for ticker in execution},
        rebalance_only_on_target_change=True,
        distribution_events_by_ticker=distributions,
        fee_per_side=FEE_PER_SIDE * cost_multiplier,
        slippage_entry=SLIPPAGE_ENTRY * cost_multiplier,
        slippage_exit=SLIPPAGE_EXIT * cost_multiplier,
    )


def _run_market_benchmark(
    core_prices: pd.DataFrame,
    dates: pd.Index,
    core_distributions: pd.DataFrame,
    initial_cash: float,
    *,
    cost_multiplier: float = 1.0,
):
    return buy_and_hold(
        core_prices,
        dates,
        initial_cash,
        0.95,
        core_distributions,
        fee_per_side=FEE_PER_SIDE * cost_multiplier,
        slippage_entry=SLIPPAGE_ENTRY * cost_multiplier,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-data-dir", required=True)
    parser.add_argument("--core-signal-file")
    parser.add_argument(
        "--core-signal-source",
        choices=("cache", "distribution_adjusted_actual"),
        default="cache",
    )
    parser.add_argument("--start-date", default="2020-06-25")
    parser.add_argument("--end-date")
    parser.add_argument("--small-account", type=float, default=2_100_000.0)
    parser.add_argument("--reference-account", type=float, default=100_000_000.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()

    data_dir = Path(args.official_data_dir)
    execution, total_return, distributions = _load_official_inputs(data_dir)
    signal_path = None
    if args.core_signal_source == "distribution_adjusted_actual":
        core_signal = adjust_ohlc_for_distributions(
            execution[CORE_TICKER], distributions[CORE_TICKER]
        )
    else:
        signal_path = (
            Path(args.core_signal_file) if args.core_signal_file else select_k200_file()
        )
        core_signal = load_prices(signal_path)
        core_signal, _ = merge_adjusted_prices(
            core_signal, load_naver_prices(data_dir / f"naver_{CORE_TICKER}.xml")
        )

    common_end = min(frame.index.max() for frame in execution.values())
    end = min(pd.Timestamp(args.end_date), common_end) if args.end_date else common_end
    start = pd.Timestamp(args.start_date)
    if start >= end:
        raise ValueError("start date must be earlier than the common data end")
    execution = {
        ticker: frame.loc[frame.index <= end] for ticker, frame in execution.items()
    }
    core_signal = core_signal.loc[core_signal.index <= end]
    total_return = total_return.loc[total_return.index <= end]

    strategy = KodexSectorRotation(
        total_return_indices=total_return,
        signal_prices=core_signal,
        distribution_events_by_ticker=distributions,
    )
    all_targets = strategy.compute_security_targets(execution, silent=True)
    targets = all_targets.loc[(all_targets.index >= start) & (all_targets.index <= end)]
    if len(targets) < 253:
        raise RuntimeError("fewer than 253 common sessions; rolling validation is invalid")
    selections = strategy.latest_selection_history
    selections = selections.loc[
        (selections["execution_date"] >= targets.index.min())
        & (selections["execution_date"] <= targets.index.max())
    ]

    summaries = {}
    curve_columns = {}
    all_trade_rows = []
    for label, initial_cash in (
        ("small", args.small_account),
        ("reference", args.reference_account),
    ):
        strategy_equity, strategy_trades = _run_candidate(
            targets,
            execution,
            distributions,
            initial_cash,
            min_trade=strategy.min_trade,
        )
        benchmark_equity, benchmark_trades = _run_market_benchmark(
            execution[CORE_TICKER],
            targets.index,
            distributions[CORE_TICKER],
            initial_cash,
        )
        defensive_targets = pd.DataFrame(index=targets.index)
        risk_on = targets.drop(columns="__CASH__", errors="ignore").sum(axis=1).gt(0)
        defensive_targets[CORE_TICKER] = np.where(risk_on, 0.95, 0.0)
        defensive_targets["__CASH__"] = 1.0 - defensive_targets[CORE_TICKER]
        defensive_equity, defensive_trades = simulate(
            defensive_targets,
            {CORE_TICKER: execution[CORE_TICKER]},
            initial_cash=initial_cash,
            min_trade=strategy.min_trade,
            price_band_pct=strategy.price_band_pct,
            sell_tax_rate_by_ticker={CORE_TICKER: 0.0},
            rebalance_only_on_target_change=True,
            distribution_events_by_ticker={
                CORE_TICKER: distributions[CORE_TICKER]
            },
        )
        summaries[label] = {
            "candidate_vs_market": _account_summary(
                initial_cash,
                strategy_equity,
                strategy_trades,
                benchmark_equity,
                benchmark_trades,
            ),
            "candidate_vs_same_timing_kodex200": _account_summary(
                initial_cash,
                strategy_equity,
                strategy_trades,
                defensive_equity,
                defensive_trades,
            ),
        }
        if label == "small":
            stressed_equity, stressed_trades = _run_candidate(
                targets,
                execution,
                distributions,
                initial_cash,
                min_trade=strategy.min_trade,
                cost_multiplier=2.0,
            )
            stressed_benchmark, stressed_benchmark_trades = _run_market_benchmark(
                execution[CORE_TICKER],
                targets.index,
                distributions[CORE_TICKER],
                initial_cash,
                cost_multiplier=2.0,
            )
            summaries[label]["candidate_vs_market_double_cost"] = _account_summary(
                initial_cash,
                stressed_equity,
                stressed_trades,
                stressed_benchmark,
                stressed_benchmark_trades,
            )
            coarse_equity, coarse_trades = _run_candidate(
                targets,
                execution,
                distributions,
                initial_cash,
                min_trade=100_000,
            )
            summaries[label]["candidate_vs_market_100k_min_trade"] = _account_summary(
                initial_cash,
                coarse_equity,
                coarse_trades,
                benchmark_equity,
                benchmark_trades,
            )
        curve_columns[f"strategy_{label}"] = _normalised_equity(strategy_equity)
        curve_columns[f"benchmark_{label}"] = _normalised_equity(benchmark_equity)
        curve_columns[f"same_timing_kodex200_{label}"] = _normalised_equity(
            defensive_equity
        )
        for approach, rows in (
            ("strategy", strategy_trades),
            ("benchmark", benchmark_trades),
            ("same_timing_kodex200", defensive_trades),
        ):
            all_trade_rows.extend(
                {"account": label, "approach": approach, **row} for row in rows
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"kodex_sector_rotation_{stamp}"
    summary_path = output_dir / f"{stem}.json"
    equity_path = output_dir / f"{stem}_equity.csv"
    targets_path = output_dir / f"{stem}_targets.csv"
    selections_path = output_dir / f"{stem}_selections.csv"
    trades_path = output_dir / f"{stem}_trades.csv"
    historical_pass = bool(
        summaries["small"]["candidate_vs_market"]["passes_all_gates"]
    )
    payload = {
        "strategy": "kodex_sector_rotation",
        "status": "paper_shadow_only_no_capital",
        "production_approved": False,
        "historical_gate_result": historical_pass,
        "validation_status": (
            "retrospective seen data; this result cannot be relabelled as out-of-sample"
        ),
        "data_start": str(targets.index.min().date()),
        "data_end": str(targets.index.max().date()),
        "core_signal_source": args.core_signal_source,
        "core_signal_file": str(signal_path.resolve()) if signal_path else None,
        "sector_products": SECTOR_PRODUCTS,
        "parameters": {
            "core_weight": 0.40,
            "sector_weight": 0.55,
            "cash_weight": 0.05,
            "selected_sectors": 2,
            "score": "equal average of official 26-week and 52-week NAV total return",
            "rebalance": "monthly or on KODEX 200 risk-state change",
            "risk_state": "frozen k200_low_turnover_reentry defaults",
            "leverage_or_inverse": False,
        },
        "execution_assumptions": {
            "integer_shares": True,
            "buy_and_sell_slippage_pct": 0.20,
            "broker_fee_per_side_pct": 0.0140527,
            "domestic_equity_etf_sell_tax_pct": 0.0,
            "distribution_income_tax_pct": (
                DOMESTIC_EQUITY_ETF.distribution_income_tax_rate * 100.0
            ),
        },
        "accounts": summaries,
        "decision_rule": (
            "Never replace production from this replay. Freeze parameters and require "
            "at least 252 subsequent sessions before reconsideration."
        ),
        "artifacts": {
            "equity": str(equity_path),
            "targets": str(targets_path),
            "selections": str(selections_path),
            "trades": str(trades_path),
        },
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(curve_columns).to_csv(equity_path)
    targets.to_csv(targets_path)
    selections.to_csv(selections_path, index=False)
    pd.DataFrame(all_trade_rows).replace({np.nan: None}).to_csv(trades_path, index=False)
    print(json.dumps({**payload, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
