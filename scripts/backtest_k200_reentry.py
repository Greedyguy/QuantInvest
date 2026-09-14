#!/usr/bin/env python3
"""Backtest the low-turnover KODEX 200 re-entry strategy."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEE_PER_SIDE, SLIPPAGE_ENTRY
from market_benchmark import (
    DOMESTIC_EQUITY_ETF,
    evaluate_market_outperformance,
    load_distribution_events,
    load_samsung_kodex_standard_xls,
    load_samsung_kodex_total_return_json,
    period_return_asof,
    prepare_distribution_schedule,
    restore_actual_ohlc,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry
from strategy_validation import ValidationPeriod, performance_by_period


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIODS = (
    ValidationPeriod("historical_2020_2023_seen", "2020-01-01", "2023-12-31"),
    ValidationPeriod("historical_2024_seen", "2024-01-01", "2024-12-31"),
    ValidationPeriod("historical_2025_seen", "2025-01-01", "2025-12-31"),
    ValidationPeriod("historical_2026_seen", "2026-01-01", "2026-12-31"),
)
DEFAULT_DISTRIBUTIONS = PROJECT_ROOT / "data" / "reference" / "kodex200_distributions.csv"


def select_k200_file() -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / "enriched").glob("069500_*.parquet"))
    if not candidates:
        raise FileNotFoundError("no cached KODEX 200 data found")

    def coverage(path: Path) -> tuple[int, int]:
        parts = path.stem.rsplit("_", 2)
        start = pd.Timestamp(parts[-2])
        end = pd.Timestamp(parts[-1])
        return ((end - start).days, end.toordinal())

    return max(candidates, key=coverage)


def load_prices(path: Path, end_date: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if end_date:
        frame = frame.loc[frame.index <= pd.Timestamp(end_date)]
    required = {"open", "close"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"invalid KODEX 200 price file: {path}")
    return frame


def load_naver_prices(path: Path) -> pd.DataFrame:
    text = path.read_bytes().decode("euc-kr", errors="replace")
    rows = []
    for raw in re.findall(r'<item data="([^"]+)"', text):
        fields = raw.split("|")
        if len(fields) != 6:
            continue
        date, open_price, high, low, close, volume = fields
        rows.append(
            {
                "date": pd.to_datetime(date, format="%Y%m%d"),
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )
    if not rows:
        raise ValueError(f"Naver price response contains no rows: {path}")
    return pd.DataFrame(rows).set_index("date").sort_index()


def merge_adjusted_prices(
    historical: pd.DataFrame, recent: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Append recent prices after putting them on the historical price scale."""

    common = historical.index.intersection(recent.index)
    if common.empty:
        raise ValueError("historical and recent prices need an overlap")
    ratios = historical.loc[common, "close"] / recent.loc[common, "close"]
    adjustment = float(ratios.replace([np.inf, -np.inf], np.nan).dropna().tail(20).median())
    if not np.isfinite(adjustment) or adjustment <= 0:
        raise ValueError("could not determine recent-price adjustment")
    adjusted = recent.copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = adjusted[column] * adjustment
    combined = pd.concat([historical, adjusted], axis=0, sort=False)
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    return combined, adjustment


def buy_and_hold(
    prices: pd.DataFrame,
    dates: pd.Index,
    initial_cash: float,
    exposure: float,
    distribution_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    aligned = prices.reindex(dates).dropna(subset=["open", "close"])
    distribution_schedule = (
        prepare_distribution_schedule(distribution_events, aligned.index)
        if distribution_events is not None
        else pd.DataFrame()
    )
    entitlement_by_date: dict[pd.Timestamp, list[dict]] = {}
    for event in distribution_schedule.to_dict("records"):
        entitlement_by_date.setdefault(event["entitlement_date"], []).append(event)
    pending_distributions: list[dict] = []
    cash = float(initial_cash)
    quantity = 0
    rows = []
    trades = []
    for position, (date, row) in enumerate(aligned.iterrows()):
        if position == 1:
            execution_price = float(row["open"]) * (1.0 + SLIPPAGE_ENTRY)
            target_value = initial_cash * exposure
            quantity = min(
                int(target_value / execution_price),
                int(cash / (execution_price * (1.0 + FEE_PER_SIDE))),
            )
            gross = quantity * execution_price
            fee = gross * FEE_PER_SIDE
            cash -= gross + fee
            trades.append(
                {
                    "date": date,
                    "ticker": "069500",
                    "action": "BUY",
                    "price": execution_price,
                    "qty": quantity,
                    "fee": fee,
                    "tax": 0.0,
                    "reason": "buy_and_hold",
                }
            )
        for event in entitlement_by_date.get(date, []):
            pending_distributions.append({**event, "eligible_quantity": quantity})
        still_pending: list[dict] = []
        for event in pending_distributions:
            if event["credit_date"] > date:
                still_pending.append(event)
                continue
            eligible_quantity = int(event["eligible_quantity"])
            if eligible_quantity <= 0:
                continue
            gross_distribution = eligible_quantity * float(event["gross_unit"])
            distribution_tax = eligible_quantity * float(event["tax_unit"])
            cash += gross_distribution - distribution_tax
            trades.append(
                {
                    "signal_date": event["entitlement_date"],
                    "date": date,
                    "ticker": "069500",
                    "action": "DISTRIBUTION",
                    "price": float(event["gross_unit"]),
                    "qty": eligible_quantity,
                    "fee": 0.0,
                    "tax": distribution_tax,
                    "gross": gross_distribution,
                    "reason": "kodex_distribution",
                }
            )
        pending_distributions = still_pending
        rows.append(
            {
                "date": date,
                "equity": cash + quantity * float(row["close"]),
                "cash": cash,
                "quantity": quantity,
                "state": "risk_on" if quantity > 0 else "cash",
            }
        )
    return pd.DataFrame(rows).set_index("date"), trades


def summarise(
    label: str,
    equity: pd.DataFrame,
    trades: list[dict],
    initial_cash: float,
) -> tuple[dict, pd.DataFrame]:
    metrics = performance_by_period(equity, PERIODS)
    full_return = float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0)
    full_drawdown = equity["equity"] / equity["equity"].cummax() - 1.0
    invested = equity["quantity"].gt(0)
    explicit_cost = sum(float(row["fee"] + row["tax"]) for row in trades)
    orders = [row for row in trades if row["action"] in {"BUY", "SELL"}]
    distributions = [row for row in trades if row["action"] == "DISTRIBUTION"]
    gross_distributions = sum(float(row.get("gross", 0.0)) for row in distributions)
    distribution_tax = sum(float(row["tax"]) for row in distributions)
    summary = {
        "label": label,
        "initial_cash": float(initial_cash),
        "start": str(equity.index.min().date()),
        "end": str(equity.index.max().date()),
        "full_return_pct": full_return * 100.0,
        "full_mdd_pct": float(full_drawdown.min()) * 100.0,
        "orders": len(orders),
        "round_trips": sum(row["action"] == "SELL" for row in orders),
        "distribution_payments": len(distributions),
        "gross_distributions": gross_distributions,
        "distribution_tax": distribution_tax,
        "invested_days_pct": float(invested.mean()) * 100.0,
        "explicit_cost_pct_initial": explicit_cost / initial_cash * 100.0,
        "periods": metrics.reset_index().replace({np.nan: None}).to_dict("records"),
    }
    return summary, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest low-turnover KODEX 200 re-entry")
    parser.add_argument("--data-file")
    parser.add_argument("--naver-xml", help="Optional recent Naver chart XML to append")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--kodex-standard-xls",
        help="Official Samsung daily market-close/NAV workbook for actual-price execution",
    )
    parser.add_argument(
        "--kodex-total-return-json",
        help="Official Samsung total-return JSON used as a benchmark sanity check",
    )
    parser.add_argument(
        "--distribution-file",
        default=str(DEFAULT_DISTRIBUTIONS),
        help="KODEX 200 distribution CSV; applied only with actual-price execution",
    )
    parser.add_argument("--small-account", type=float, default=2_100_000.0)
    parser.add_argument("--reference-account", type=float, default=100_000_000.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()

    data_path = Path(args.data_file) if args.data_file else select_k200_file()
    prices = load_prices(data_path, args.end_date)
    recent_adjustment = None
    if args.naver_xml:
        prices, recent_adjustment = merge_adjusted_prices(
            prices,
            load_naver_prices(Path(args.naver_xml)),
        )
        if args.end_date:
            prices = prices.loc[prices.index <= pd.Timestamp(args.end_date)]
    execution_prices = prices
    distribution_events = None
    price_basis = "adjusted_proxy_without_cash_distributions"
    if args.kodex_standard_xls:
        official_daily = load_samsung_kodex_standard_xls(args.kodex_standard_xls)
        execution_prices = restore_actual_ohlc(prices, official_daily["market_close"])
        distribution_events = load_distribution_events(args.distribution_file)
        price_basis = "official_actual_close_scaled_ohlc_with_net_cash_distributions"

    official_total_return = None
    if args.kodex_total_return_json:
        official_total_return = load_samsung_kodex_total_return_json(
            args.kodex_total_return_json
        )
    comparisons = []
    metric_tables = []
    curves = {}
    state_history = None
    market_outperformance = {}

    for account_label, initial_cash in (
        ("small", args.small_account),
        ("reference", args.reference_account),
    ):
        strategy = K200LowTurnoverReentry(
            initial_cash=initial_cash,
            execution_prices=execution_prices if args.kodex_standard_xls else None,
            distribution_events=distribution_events,
            tax_profile=DOMESTIC_EQUITY_ETF,
        )
        strategy_equity, strategy_trades = strategy.run_backtest(
            {"069500": prices}, silent=True
        )
        if strategy_equity.empty:
            raise RuntimeError("re-entry strategy produced no equity curve")
        state_history = strategy.latest_state_history
        hold_equity, hold_trades = buy_and_hold(
            execution_prices,
            strategy_equity.index,
            initial_cash,
            strategy.risk_on_exposure,
            distribution_events,
        )
        market_outperformance[account_label] = evaluate_market_outperformance(
            strategy_equity["equity"], hold_equity["equity"]
        )

        for strategy_label, equity, trades in (
            ("low_turnover_reentry", strategy_equity, strategy_trades),
            ("kodex200_buy_hold_95pct", hold_equity, hold_trades),
        ):
            label = f"{strategy_label}_{account_label}"
            summary, metrics = summarise(label, equity, trades, initial_cash)
            comparisons.append(summary)
            table = metrics.reset_index()
            table.insert(0, "approach", label)
            metric_tables.append(table)
            curves[label] = equity["equity"] / initial_cash

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"k200_low_turnover_reentry_{stamp}"
    summary_path = output_dir / f"{stem}.json"
    metrics_path = output_dir / f"{stem}_periods.csv"
    equity_path = output_dir / f"{stem}_equity.csv"
    states_path = output_dir / f"{stem}_states.csv"
    payload = {
        "strategy": "k200_low_turnover_reentry",
        "data_file": str(data_path.resolve()),
        "data_start": str(prices.index.min().date()),
        "data_end": str(prices.index.max().date()),
        "recent_price_file": str(Path(args.naver_xml).resolve()) if args.naver_xml else None,
        "recent_price_adjustment": recent_adjustment,
        "price_basis": price_basis,
        "tax_profile": {
            "asset_class": "domestic_equity_etf",
            "sell_transaction_tax_rate": DOMESTIC_EQUITY_ETF.sell_transaction_tax_rate,
            "distribution_income_tax_rate": DOMESTIC_EQUITY_ETF.distribution_income_tax_rate,
        },
        "validation_status": (
            "historical replay only; all periods were visible during strategy design"
        ),
        "parameters": {
            "decision_frequency": "first trading day of each month",
            "risk_on_exposure": 0.95,
            "trend_window": 120,
            "momentum_window": 60,
            "entry": (
                "monthly: close > 1.01 * MA120, momentum60 > 0, "
                "close > MA60, and momentum20 > 0"
            ),
            "exit": "close < 0.99 * MA120 or momentum60 <= -0.03",
            "emergency_exit": (
                "daily: drawdown60 <= -0.12 and close < MA20, or "
                "close < 0.97 * MA60 and momentum20 <= -0.04"
            ),
        },
        "comparison": comparisons,
        "market_outperformance": market_outperformance,
        "covered_call_note": (
            "Covered-call ETF comparison is intentionally excluded until "
            "total-return prices including distributions are supplied."
        ),
        "artifacts": {
            "period_metrics": str(metrics_path),
            "equity_curves": str(equity_path),
            "state_history": str(states_path),
        },
    }
    if official_total_return is not None and state_history is not None:
        payload["official_pre_tax_sanity_check"] = {
            "start": str(state_history.index.min().date()),
            "end": str(state_history.index.max().date()),
            "market_price_return_pct": period_return_asof(
                official_total_return["market_price_index"],
                state_history.index.min(),
                state_history.index.max(),
            )
            * 100.0,
            "nav_total_return_pct": period_return_asof(
                official_total_return["nav_total_return_index"],
                state_history.index.min(),
                state_history.index.max(),
            )
            * 100.0,
            "note": "NAV total return assumes pre-tax distribution reinvestment.",
        }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(metric_tables, ignore_index=True).to_csv(metrics_path, index=False)
    pd.DataFrame(curves).to_csv(equity_path)
    if state_history is not None:
        state_history.to_csv(states_path)
    print(json.dumps({**payload, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
