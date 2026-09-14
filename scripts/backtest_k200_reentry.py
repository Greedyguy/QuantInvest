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
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry
from strategy_validation import ValidationPeriod, performance_by_period


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIODS = (
    ValidationPeriod("train", "2020-01-01", "2023-12-31"),
    ValidationPeriod("validation", "2024-01-01", "2024-12-31"),
    ValidationPeriod("test", "2025-01-01", "2025-12-31"),
    ValidationPeriod("live_oos", "2026-01-01", "2026-12-31"),
)


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
) -> tuple[pd.DataFrame, list[dict]]:
    aligned = prices.reindex(dates).dropna(subset=["open", "close"])
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
    summary = {
        "label": label,
        "initial_cash": float(initial_cash),
        "start": str(equity.index.min().date()),
        "end": str(equity.index.max().date()),
        "full_return_pct": full_return * 100.0,
        "full_mdd_pct": float(full_drawdown.min()) * 100.0,
        "orders": len(trades),
        "round_trips": sum(row["action"] == "SELL" for row in trades),
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
    comparisons = []
    metric_tables = []
    curves = {}
    state_history = None

    for account_label, initial_cash in (
        ("small", args.small_account),
        ("reference", args.reference_account),
    ):
        strategy = K200LowTurnoverReentry(initial_cash=initial_cash)
        strategy_equity, strategy_trades = strategy.run_backtest(
            {"069500": prices}, silent=True
        )
        if strategy_equity.empty:
            raise RuntimeError("re-entry strategy produced no equity curve")
        state_history = strategy.latest_state_history
        hold_equity, hold_trades = buy_and_hold(
            prices,
            strategy_equity.index,
            initial_cash,
            strategy.risk_on_exposure,
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
