#!/usr/bin/env python3
"""Replay saved KR signals against recovery and KOFR shadow overlays."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEE_PER_SIDE, SLIPPAGE_ENTRY, SLIPPAGE_EXIT, TAX_RATE_SELL
from strategies.strategy_multi_allocator_plus_safe_etf_kqm import (
    MultiStrategyAllocatorPlusSafeETFKQM,
)


KOFR_TICKER = "423160"
LIQUID_RESERVE_WEIGHT = 0.30
MAX_PARKING_WEIGHT = 0.50
NAVER_CHART_URL = "https://fchart.stock.naver.com/sise.nhn"


def load_snapshots(report_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    snapshots = {}
    for path in report_dir.glob("signal_kr_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            signal_date = pd.Timestamp(payload["signal_date"])
        except Exception:
            continue
        if start <= signal_date <= end:
            snapshots[signal_date] = payload
    return dict(sorted(snapshots.items()))


def fetch_naver_price(ticker: str, count: int = 400) -> tuple[str, pd.DataFrame, str | None]:
    try:
        response = requests.get(
            NAVER_CHART_URL,
            params={
                "symbol": ticker,
                "timeframe": "day",
                "count": count,
                "requestType": 0,
            },
            timeout=20,
        )
        response.raise_for_status()
        text = response.content.decode("euc-kr", errors="replace")
        rows = []
        for raw in re.findall(r'<item data="([^"]+)"', text):
            fields = raw.split("|")
            if len(fields) != 6:
                continue
            day, open_px, high, low, close, volume = fields
            rows.append({
                "date": pd.to_datetime(day, format="%Y%m%d"),
                "open": float(open_px),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return ticker, frame, "empty_response"
        return ticker, frame.set_index("date").sort_index(), None
    except Exception as exc:
        return ticker, pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def fetch_prices(tickers: set[str]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    prices = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_naver_price, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker, frame, error = future.result()
            if error:
                errors[ticker] = error
            else:
                prices[ticker] = frame
    # The public endpoint can occasionally return an empty response under a
    # parallel burst. Retry only those symbols once, sequentially.
    for ticker in list(errors):
        retry_ticker, frame, error = fetch_naver_price(ticker)
        if error:
            errors[retry_ticker] = error
        else:
            prices[retry_ticker] = frame
            errors.pop(retry_ticker, None)
    return prices, errors


def latest_snapshot_before(snapshots: dict, day: pd.Timestamp):
    valid = [signal_date for signal_date in snapshots if signal_date < day]
    if not valid:
        return None, None
    signal_date = max(valid)
    return signal_date, snapshots[signal_date]


def price_on(prices: dict, ticker: str, day: pd.Timestamp, field: str) -> float:
    frame = prices.get(ticker)
    if frame is None or frame.empty or day not in frame.index:
        return math.nan
    value = frame.at[day, field]
    return float(value) if pd.notna(value) and float(value) > 0 else math.nan


def close_return(prices: dict, ticker: str, day: pd.Timestamp, previous: pd.Timestamp) -> float:
    current = price_on(prices, ticker, day, "close")
    prior = price_on(prices, ticker, previous, "close")
    if not np.isfinite(current) or not np.isfinite(prior) or prior <= 0:
        return math.nan
    return current / prior - 1.0


def build_recovery_targets(snapshots: dict, prices: dict, calendar: pd.DatetimeIndex) -> tuple[dict, pd.DataFrame]:
    risk_returns = pd.Series(0.0, index=calendar)
    stressed = pd.Series(np.nan, index=calendar)
    base = pd.Series(np.nan, index=calendar)
    levels = pd.Series(np.nan, index=calendar)

    for position, day in enumerate(calendar):
        signal_date, payload = latest_snapshot_before(snapshots, day + pd.Timedelta(days=1))
        if payload is None:
            continue
        targets = pd.Series(payload.get("targets", {}), dtype=float)
        assets = targets.drop("__CASH__", errors="ignore")
        assets = assets[assets > 0]
        exposure = float(assets.sum())
        stressed.loc[day] = exposure

        context = ((payload.get("decision_context") or {}).get("exposure") or {})
        level = context.get("stress_level")
        if level is None:
            level = 2 if exposure <= 0.25 else (1 if exposure <= 0.55 else 0)
        levels.loc[day] = int(level)
        base_exposure = context.get("base")
        if base_exposure is None:
            base_exposure = (
                max(exposure, 0.45)
                if int(level) >= 2
                else (max(exposure, 0.65) if int(level) == 1 else exposure)
            )
        base.loc[day] = float(base_exposure)

        if position == 0 or exposure <= 0:
            continue
        previous = calendar[position - 1]
        weighted = []
        used_weights = []
        for ticker, weight in assets.items():
            ret = close_return(prices, str(ticker), day, previous)
            if np.isfinite(ret):
                weighted.append(float(weight) * ret)
                used_weights.append(float(weight))
        if used_weights and sum(used_weights) > 0:
            risk_returns.loc[day] = sum(weighted) / sum(used_weights)

    stressed = stressed.ffill().fillna(0.0)
    base = base.ffill().fillna(stressed)
    levels = levels.ffill().fillna(0).astype(int)
    controller = MultiStrategyAllocatorPlusSafeETFKQM()
    candidate, context = controller._stress_recovery_candidate(
        base,
        stressed,
        levels,
        risk_returns,
    )

    recovered = {}
    for signal_date, payload in snapshots.items():
        targets = pd.Series(payload.get("targets", {}), dtype=float)
        assets = targets.drop("__CASH__", errors="ignore")
        exposure = float(assets[assets > 0].sum())
        candidate_exposure = float(candidate.asof(signal_date))
        if exposure > 0 and candidate_exposure > exposure:
            assets = assets * (candidate_exposure / exposure)
        cash = max(1.0 - float(assets[assets > 0].sum()), 0.0)
        recovered[signal_date] = pd.concat([
            assets[assets > 0],
            pd.Series({"__CASH__": cash}),
        ])

    diagnostics = context.copy()
    diagnostics["baseline_exposure"] = stressed
    diagnostics["candidate_exposure"] = candidate
    diagnostics["risk_basket_return"] = risk_returns
    diagnostics["stress_level"] = levels
    return recovered, diagnostics


def base_target_map(snapshots: dict) -> dict[pd.Timestamp, pd.Series]:
    return {
        signal_date: pd.Series(payload.get("targets", {}), dtype=float)
        for signal_date, payload in snapshots.items()
    }


def add_cash_parking(target_map: dict[pd.Timestamp, pd.Series]) -> dict[pd.Timestamp, pd.Series]:
    result = {}
    for signal_date, targets in target_map.items():
        parked = targets.copy()
        strategic_cash = max(float(parked.get("__CASH__", 0.0)), 0.0)
        parking_weight = min(
            max(strategic_cash - LIQUID_RESERVE_WEIGHT, 0.0),
            MAX_PARKING_WEIGHT,
        )
        parked.loc[KOFR_TICKER] = parking_weight
        parked.loc["__CASH__"] = max(strategic_cash - parking_weight, 0.0)
        result[signal_date] = parked
    return result


def simulate(
    target_map: dict[pd.Timestamp, pd.Series],
    snapshots: dict,
    prices: dict,
    calendar: pd.DatetimeIndex,
    initial_cash: float,
    minimum_trade: float,
    price_band_pct: float,
) -> tuple[pd.DataFrame, list[dict]]:
    cash = float(initial_cash)
    holdings: dict[str, int] = {}
    last_close: dict[str, float] = {}
    last_applied_signal = None
    equity_rows = []
    trades = []

    for day in calendar:
        signal_date, payload = latest_snapshot_before(snapshots, day)
        if signal_date is not None and signal_date != last_applied_signal:
            targets = target_map[signal_date]
            assets = targets.drop("__CASH__", errors="ignore")
            assets = assets[assets > 0]
            target_symbols = set(str(ticker) for ticker in assets.index)

            equity_open = cash
            for ticker, quantity in holdings.items():
                open_px = price_on(prices, ticker, day, "open")
                if not np.isfinite(open_px):
                    open_px = last_close.get(ticker, 0.0)
                equity_open += quantity * max(float(open_px), 0.0)

            orders = []
            refs = payload.get("ref_prices") or {}
            for ticker, weight in assets.items():
                ticker = str(ticker)
                # Naver back-adjusts historical prices after ETF splits/reverse splits,
                # while the saved live reference remains on its original price scale.
                # Keep sizing and the next-open price band on one adjusted scale.
                adjusted_close = price_on(prices, ticker, signal_date, "close")
                reference = (
                    adjusted_close
                    if np.isfinite(adjusted_close)
                    else float(refs.get(ticker) or math.nan)
                )
                open_px = price_on(prices, ticker, day, "open")
                if not np.isfinite(reference) or not np.isfinite(open_px):
                    continue
                target_value = equity_open * float(weight)
                current_qty = int(holdings.get(ticker, 0))
                if current_qty <= 0 and target_value < max(minimum_trade, reference):
                    continue
                if target_value < minimum_trade:
                    target_qty = 0
                else:
                    target_qty = int(target_value / reference)
                delta = target_qty - current_qty
                if delta:
                    orders.append({
                        "ticker": ticker,
                        "action": "BUY" if delta > 0 else "SELL",
                        "quantity": abs(delta),
                        "reference": reference,
                        "open": open_px,
                        "estimated": abs(delta) * reference,
                    })

            for ticker, quantity in list(holdings.items()):
                if quantity <= 0 or ticker in target_symbols:
                    continue
                open_px = price_on(prices, ticker, day, "open")
                if np.isfinite(open_px):
                    orders.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "quantity": quantity,
                        "reference": float(last_close.get(ticker, open_px)),
                        "open": open_px,
                        "estimated": quantity * open_px,
                    })

            orders.sort(key=lambda order: (
                0 if order["action"] == "SELL" else 1,
                1 if order["action"] == "BUY" and order["ticker"] == KOFR_TICKER else 0,
                -order["estimated"],
            ))
            for order in orders:
                ticker = order["ticker"]
                difference = abs(order["open"] - order["reference"]) / order["reference"] * 100
                if order["action"] == "BUY" and difference > price_band_pct:
                    continue
                if order["action"] == "SELL":
                    quantity = min(order["quantity"], holdings.get(ticker, 0))
                    if quantity <= 0:
                        continue
                    exec_price = order["open"] * (1.0 - SLIPPAGE_EXIT)
                    gross = quantity * exec_price
                    cash += gross * (1.0 - FEE_PER_SIDE - TAX_RATE_SELL)
                    holdings[ticker] -= quantity
                    if holdings[ticker] <= 0:
                        holdings.pop(ticker, None)
                else:
                    exec_price = order["open"] * (1.0 + SLIPPAGE_ENTRY)
                    cash_per_share = exec_price * (1.0 + FEE_PER_SIDE)
                    quantity = min(order["quantity"], int(cash / cash_per_share))
                    if quantity <= 0:
                        continue
                    cash -= quantity * cash_per_share
                    holdings[ticker] = holdings.get(ticker, 0) + quantity
                trades.append({
                    "date": str(day.date()),
                    "signal_date": str(signal_date.date()),
                    "ticker": ticker,
                    "action": order["action"],
                    "quantity": int(quantity),
                    "value": float(quantity * exec_price),
                })
            last_applied_signal = signal_date

        equity = cash
        risk_value = 0.0
        parking_value = 0.0
        for ticker, quantity in holdings.items():
            close_px = price_on(prices, ticker, day, "close")
            if np.isfinite(close_px):
                last_close[ticker] = close_px
            else:
                close_px = last_close.get(ticker, 0.0)
            value = quantity * max(float(close_px), 0.0)
            equity += value
            if ticker == KOFR_TICKER:
                parking_value += value
            else:
                risk_value += value
        equity_rows.append({
            "date": day,
            "equity": equity,
            "cash_weight": cash / equity if equity > 0 else math.nan,
            "risk_exposure": risk_value / equity if equity > 0 else math.nan,
            "parking_weight": parking_value / equity if equity > 0 else math.nan,
        })

    return pd.DataFrame(equity_rows).set_index("date"), trades


def simulate_continuous(
    target_map: dict[pd.Timestamp, pd.Series],
    snapshots: dict,
    prices: dict,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Frictionless signal-level replay without share or minimum-order effects."""
    close = pd.DataFrame(index=calendar)
    for ticker, frame in prices.items():
        close[ticker] = frame["close"].reindex(calendar).ffill()
    asset_returns = close.pct_change().fillna(0.0)
    equity = 1.0
    rows = []
    for day in calendar:
        signal_date, _ = latest_snapshot_before(snapshots, day)
        targets = target_map.get(signal_date, pd.Series(dtype=float))
        assets = targets.drop("__CASH__", errors="ignore")
        assets = assets[assets > 0]
        daily_return = 0.0
        risk_weight = 0.0
        parking_weight = 0.0
        for ticker, weight in assets.items():
            ticker = str(ticker)
            daily_return += float(weight) * float(asset_returns.at[day, ticker])
            if ticker == KOFR_TICKER:
                parking_weight += float(weight)
            else:
                risk_weight += float(weight)
        equity *= 1.0 + daily_return
        rows.append({
            "date": day,
            "equity": equity,
            "cash_weight": max(1.0 - float(assets.sum()), 0.0),
            "risk_exposure": risk_weight,
            "parking_weight": parking_weight,
        })
    return pd.DataFrame(rows).set_index("date")


def summarize(
    equity: pd.DataFrame,
    trades: list[dict],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict:
    period = equity.loc[period_start:period_end].copy()
    prior = equity.loc[equity.index < period_start]
    start_equity = float(prior.iloc[-1]["equity"]) if not prior.empty else float(period.iloc[0]["equity"])
    curve = pd.concat([
        pd.Series([start_equity], index=[period_start - pd.Timedelta(nanoseconds=1)]),
        period["equity"],
    ])
    returns = curve.pct_change().dropna()
    drawdown = curve / curve.cummax() - 1.0
    period_trades = [
        trade for trade in trades
        if period_start.date().isoformat() <= trade["date"] <= period_end.date().isoformat()
    ]
    return {
        "start_equity": start_equity,
        "final_equity": float(period.iloc[-1]["equity"]),
        "return_pct": (float(period.iloc[-1]["equity"]) / start_equity - 1.0) * 100.0,
        "mdd_pct": float(drawdown.min()) * 100.0,
        "annualized_vol_pct": float(returns.std(ddof=1) * np.sqrt(252)) * 100.0,
        "sharpe_zero_rf": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) > 0 else 0.0,
        "trade_count": len(period_trades),
        "buy_count": sum(trade["action"] == "BUY" for trade in period_trades),
        "sell_count": sum(trade["action"] == "SELL" for trade in period_trades),
        "turnover_pct": sum(trade["value"] for trade in period_trades) / start_equity * 100.0,
        "average_cash_weight_pct": float(period["cash_weight"].mean()) * 100.0,
        "average_risk_exposure_pct": float(period["risk_exposure"].mean()) * 100.0,
        "average_parking_weight_pct": float(period["parking_weight"].mean()) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, default=Path("reports/signals"))
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--warmup-start", default="2026-05-01")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--minimum-trade", type=float, default=50_000.0)
    parser.add_argument("--price-band-pct", type=float, default=3.0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    warmup_start = pd.Timestamp(args.warmup_start)
    snapshots = load_snapshots(args.signals, warmup_start, end)
    if not snapshots:
        raise RuntimeError("No KR signal snapshots found for the requested period")
    tickers = {KOFR_TICKER, "069500"}
    for payload in snapshots.values():
        tickers.update(
            ticker for ticker, weight in (payload.get("targets") or {}).items()
            if ticker != "__CASH__" and float(weight) > 0
        )
    prices, price_errors = fetch_prices(tickers)
    if "069500" not in prices:
        raise RuntimeError("KODEX 200 price data is required for the trading calendar")
    calendar = prices["069500"].loc[warmup_start:end].index

    baseline = base_target_map(snapshots)
    recovery, recovery_diagnostics = build_recovery_targets(
        snapshots,
        prices,
        calendar,
    )
    scenarios = {
        "existing": baseline,
        "recovery_only": recovery,
        "cash_parking_only": add_cash_parking(baseline),
        "combined_shadow": add_cash_parking(recovery),
    }
    results = {}
    continuous_results = {}
    for name, target_map in scenarios.items():
        equity, trades = simulate(
            target_map,
            snapshots,
            prices,
            calendar,
            args.initial_cash,
            args.minimum_trade,
            args.price_band_pct,
        )
        results[name] = summarize(equity, trades, start, end)
        continuous_equity = simulate_continuous(
            target_map,
            snapshots,
            prices,
            calendar,
        )
        continuous_results[name] = summarize(
            continuous_equity,
            [],
            start,
            end,
        )

    period_calendar = calendar[(calendar >= start) & (calendar <= end)]
    signal_dates = [date for date in snapshots if start <= date <= end]
    reference_checks = []
    for signal_date, payload in snapshots.items():
        if not (start <= signal_date <= end):
            continue
        for ticker, reference in (payload.get("ref_prices") or {}).items():
            naver_close = price_on(prices, str(ticker), signal_date, "close")
            reference = float(reference or 0.0)
            if reference <= 0 or not np.isfinite(naver_close):
                continue
            relative_error = abs(naver_close - reference) / reference
            reference_checks.append({
                "date": str(signal_date.date()),
                "ticker": str(ticker),
                "relative_error": relative_error,
            })
    reference_errors = [row["relative_error"] for row in reference_checks]
    report = {
        "period": {"start": args.start, "end": args.end},
        "assumptions": {
            "initial_cash": args.initial_cash,
            "minimum_trade": args.minimum_trade,
            "price_band_pct": args.price_band_pct,
            "cash_return": 0.0,
            "cash_parking_ticker": KOFR_TICKER,
            "liquid_reserve_weight": LIQUID_RESERVE_WEIGHT,
            "maximum_parking_weight": MAX_PARKING_WEIGHT,
            "execution": "previous EOD signal at next trading-day open",
            "reference_price_basis": "Naver adjusted close on signal date",
            "recovery_reconstruction": "saved target risk-basket return approximation",
        },
        "data_quality": {
            "signal_dates": len(signal_dates),
            "trading_dates": len(period_calendar),
            "missing_signal_dates": [
                str(date.date()) for date in period_calendar if date not in snapshots
            ],
            "requested_tickers": len(tickers),
            "downloaded_tickers": len(prices),
            "price_errors": price_errors,
            "price_source": "Naver Finance chart endpoint",
            "reference_price_checks": len(reference_errors),
            "reference_price_median_error_pct": (
                float(np.median(reference_errors)) * 100.0 if reference_errors else None
            ),
            "reference_price_p95_error_pct": (
                float(np.percentile(reference_errors, 95)) * 100.0 if reference_errors else None
            ),
            "reference_price_over_1pct": sum(error > 0.01 for error in reference_errors),
            "reference_price_over_5pct": sum(error > 0.05 for error in reference_errors),
            "largest_reference_price_errors": sorted(
                reference_checks,
                key=lambda row: row["relative_error"],
                reverse=True,
            )[:5],
        },
        "recovery_diagnostics": {
            "days_above_baseline": int((
                recovery_diagnostics.loc[start:end, "candidate_exposure"]
                > recovery_diagnostics.loc[start:end, "baseline_exposure"] + 1e-12
            ).sum()),
            "maximum_uplift_pct_points": float((
                recovery_diagnostics.loc[start:end, "candidate_exposure"]
                - recovery_diagnostics.loc[start:end, "baseline_exposure"]
            ).max()) * 100.0,
        },
        "results": results,
        "continuous_weight_results": continuous_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
