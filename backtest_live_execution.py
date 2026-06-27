#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-style backtest for Multi Allocator PLUS.

This simulates the production flow:
- compute EOD security targets
- execute the previous EOD target on the next trading day's open
- apply min-trade filtering, price-band recheck, cash limits, fees/tax/slippage
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    FEE_PER_SIDE,
    SLIPPAGE_ENTRY,
    SLIPPAGE_EXIT,
    TAX_RATE_SELL,
)
from reports import load_data
from strategies import get_strategy
from utils import perf_stats


@dataclass
class SimOrder:
    signal_date: str
    exec_date: str
    ticker: str
    action: str
    planned_qty: int
    final_qty: int
    ref_price: float
    exec_price: float
    reason: str
    cash_after: float


def _price_on(enriched: Dict[str, pd.DataFrame], ticker: str, day, field: str) -> float:
    df = enriched.get(ticker)
    if df is None or df.empty or field not in df.columns:
        return np.nan
    if day not in df.index:
        return np.nan
    val = df.loc[day, field]
    return float(val) if np.isfinite(val) else np.nan


def _mark_to_market(cash: float, holdings: Dict[str, int], enriched: Dict[str, pd.DataFrame], day) -> float:
    equity = cash
    for ticker, qty in holdings.items():
        price = _price_on(enriched, ticker, day, "close")
        if np.isfinite(price) and price > 0:
            equity += qty * price
    return equity


def _build_orders(
    signal_date,
    exec_date,
    targets: pd.Series,
    cash: float,
    holdings: Dict[str, int],
    enriched: Dict[str, pd.DataFrame],
    min_trade: int,
    price_band_pct: float,
) -> List[SimOrder]:
    equity = cash
    for ticker, qty in holdings.items():
        px = _price_on(enriched, ticker, exec_date, "open")
        if np.isfinite(px) and px > 0:
            equity += qty * px

    orders: List[SimOrder] = []
    asset_targets = targets.drop("__CASH__", errors="ignore")
    asset_targets = asset_targets[asset_targets > 0]
    target_symbols = set(asset_targets.index)

    for ticker, weight in asset_targets.items():
        ref_price = _price_on(enriched, ticker, signal_date, "close")
        exec_open = _price_on(enriched, ticker, exec_date, "open")
        if not np.isfinite(ref_price) or ref_price <= 0 or not np.isfinite(exec_open) or exec_open <= 0:
            continue
        target_value = equity * float(weight)
        current_qty = int(holdings.get(ticker, 0))
        if target_value < min_trade:
            if current_qty > 0:
                orders.append(
                    SimOrder(
                        str(signal_date.date()),
                        str(exec_date.date()),
                        ticker,
                        "SELL",
                        current_qty,
                        current_qty,
                        ref_price,
                        exec_open,
                        "target_below_min_trade",
                        cash,
                    )
                )
            continue
        target_qty = int(target_value / ref_price)
        delta = target_qty - current_qty
        if delta == 0:
            continue
        diff_pct = abs(exec_open - ref_price) / ref_price * 100
        if diff_pct > price_band_pct:
            orders.append(
                SimOrder(
                    str(signal_date.date()),
                    str(exec_date.date()),
                    ticker,
                    "SKIP",
                    abs(delta),
                    0,
                    ref_price,
                    exec_open,
                    "price_band_exceeded",
                    cash,
                )
            )
            continue
        orders.append(
            SimOrder(
                str(signal_date.date()),
                str(exec_date.date()),
                ticker,
                "BUY" if delta > 0 else "SELL",
                abs(delta),
                abs(delta),
                ref_price,
                exec_open,
                "ok",
                cash,
            )
        )

    for ticker, qty in list(holdings.items()):
        if qty > 0 and ticker not in target_symbols:
            exec_open = _price_on(enriched, ticker, exec_date, "open")
            ref_price = _price_on(enriched, ticker, signal_date, "close")
            if np.isfinite(exec_open) and exec_open > 0:
                orders.append(
                    SimOrder(
                        str(signal_date.date()),
                        str(exec_date.date()),
                        ticker,
                        "SELL",
                        int(qty),
                        int(qty),
                        float(ref_price) if np.isfinite(ref_price) else exec_open,
                        exec_open,
                        "not_in_target",
                        cash,
                    )
                )

    orders.sort(key=lambda order: (0 if order.action == "SELL" else 1, -order.planned_qty * order.exec_price))
    return orders


def simulate(
    target_weights: pd.DataFrame,
    enriched: Dict[str, pd.DataFrame],
    initial_cash: float,
    min_trade: int,
    price_band_pct: float,
) -> tuple[pd.DataFrame, list[dict]]:
    cash = float(initial_cash)
    holdings: Dict[str, int] = {}
    equity_rows = []
    trade_rows: List[dict] = []

    dates = list(target_weights.index)
    for idx in range(len(dates) - 1):
        signal_date = dates[idx]
        exec_date = dates[idx + 1]
        targets = target_weights.loc[signal_date].fillna(0.0)
        orders = _build_orders(
            signal_date,
            exec_date,
            targets,
            cash,
            holdings,
            enriched,
            min_trade,
            price_band_pct,
        )

        for order in orders:
            if order.action == "SKIP" or order.final_qty <= 0:
                trade_rows.append(asdict(order))
                continue
            if order.action == "SELL":
                qty = min(int(order.final_qty), int(holdings.get(order.ticker, 0)))
                if qty <= 0:
                    continue
                exec_price = order.exec_price * (1 - SLIPPAGE_EXIT)
                gross = qty * exec_price
                fee = gross * FEE_PER_SIDE
                tax = gross * TAX_RATE_SELL
                cash += gross - fee - tax
                holdings[order.ticker] = int(holdings.get(order.ticker, 0)) - qty
                if holdings[order.ticker] <= 0:
                    holdings.pop(order.ticker, None)
                order.final_qty = qty
                order.exec_price = exec_price
                order.cash_after = cash
                trade_rows.append(asdict(order))
            elif order.action == "BUY":
                exec_price = order.exec_price * (1 + SLIPPAGE_ENTRY)
                cash_per_share = exec_price * (1 + FEE_PER_SIDE)
                qty = min(int(order.final_qty), int(cash / cash_per_share) if cash_per_share > 0 else 0)
                if qty <= 0:
                    order.action = "SKIP"
                    order.final_qty = 0
                    order.reason = "insufficient_cash"
                    order.cash_after = cash
                    trade_rows.append(asdict(order))
                    continue
                gross = qty * exec_price
                fee = gross * FEE_PER_SIDE
                cash -= gross + fee
                holdings[order.ticker] = int(holdings.get(order.ticker, 0)) + qty
                order.final_qty = qty
                order.exec_price = exec_price
                order.cash_after = cash
                trade_rows.append(asdict(order))

        equity = _mark_to_market(cash, holdings, enriched, exec_date)
        equity_rows.append({"date": exec_date, "equity": equity, "cash": cash, "positions": len(holdings)})

    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        equity_curve = equity_curve.set_index("date")
    return equity_curve, trade_rows


def main():
    parser = argparse.ArgumentParser(description="Run live-style execution backtest")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--strategy", default="multi_allocator_plus_safe_etf_kqm")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--min-trade", type=int, default=50_000)
    parser.add_argument("--price-band-pct", type=float, default=3.0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    enriched, idx_map = load_data(use_cache=not args.no_cache, start_date=args.start_date)
    strategy = get_strategy(args.strategy)
    target_weights = strategy.compute_security_targets(
        enriched,
        market_index=idx_map.get("KOSDAQ"),
        secondary_index=idx_map.get("KOSPI"),
        silent=True,
    )
    equity_curve, trades = simulate(
        target_weights,
        enriched,
        initial_cash=args.initial_cash,
        min_trade=args.min_trade,
        price_band_pct=args.price_band_pct,
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    equity_path = out_dir / f"live_execution_backtest_equity_{ts}.csv"
    trade_path = out_dir / f"live_execution_backtest_trades_{ts}.csv"
    summary_path = out_dir / f"live_execution_backtest_summary_{ts}.json"
    equity_curve.to_csv(equity_path)
    pd.DataFrame(trades).to_csv(trade_path, index=False)

    stats = perf_stats(equity_curve) if not equity_curve.empty else {}
    summary = {
        "strategy": args.strategy,
        "start_date": args.start_date,
        "initial_cash": args.initial_cash,
        "min_trade": args.min_trade,
        "price_band_pct": args.price_band_pct,
        "rows": int(len(equity_curve)),
        "trades": int(len(trades)),
        "stats": stats,
        "equity_path": str(equity_path),
        "trade_path": str(trade_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
