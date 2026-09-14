#!/usr/bin/env python3
"""Run a reproducible, production-oriented audit of the Korean allocator.

The audit deliberately compares executable security targets.  It does not use
the allocator's synthetic child-equity blend or its full-sample volatility
scaler, both of which can make historical results diverge from live orders.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest_live_execution import simulate
from config import FEE_PER_SIDE, TAX_RATE_SELL
from strategies import get_strategy
from strategy_validation import DEFAULT_VALIDATION_PERIODS, performance_by_period


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_enriched_cache_paths(
    enriched_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    """Select the best period-covering cache file for every KR ticker."""

    selected: dict[str, tuple[tuple[float, float, float], Path]] = {}
    for path in enriched_dir.glob("*.parquet"):
        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        ticker, raw_start, raw_end = parts
        try:
            file_start = pd.Timestamp(raw_start)
            file_end = pd.Timestamp(raw_end)
        except (TypeError, ValueError):
            continue
        overlap_start = max(start, file_start)
        overlap_end = min(end, file_end)
        overlap_days = max(float((overlap_end - overlap_start).days), -1.0)
        full_cover = float(file_start <= start and file_end >= end)
        end_distance = -abs(float((file_end - end).days))
        score = (full_cover, overlap_days, end_distance)
        current = selected.get(ticker)
        if current is None or score > current[0]:
            selected[ticker] = (score, path)
    return sorted(item[1] for item in selected.values())


def load_enriched_snapshot(
    cache_suffix: str,
    start_date: str,
    end_date: str | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load one immutable enriched-cache generation without network access."""

    enriched_dir = PROJECT_ROOT / "data" / "enriched"
    start = pd.Timestamp(start_date)
    index_end = pd.Timestamp(cache_suffix.rsplit("_", 1)[-1])
    end = min(pd.Timestamp(end_date), index_end) if end_date else index_end
    paths = select_enriched_cache_paths(enriched_dir, start, end)
    if not paths:
        raise FileNotFoundError("no compatible enriched cache files found")
    enriched: dict[str, pd.DataFrame] = {}
    for path in paths:
        ticker = path.name.split("_", 1)[0]
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.loc[frame.index >= start]
        frame = frame.loc[frame.index <= end]
        if not frame.empty and {"open", "close"}.issubset(frame.columns):
            enriched[ticker] = frame.sort_index()

    indexes = {}
    for market in ("KOSDAQ", "KOSPI"):
        path = PROJECT_ROOT / "data" / "index" / f"{market}_{cache_suffix}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing matching index cache: {path.name}")
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.loc[frame.index >= start]
        frame = frame.loc[frame.index <= end]
        indexes[market] = frame.sort_index()
    return enriched, indexes


def _summary_for(
    label: str,
    targets: pd.DataFrame,
    equity: pd.DataFrame,
    trades: list[dict],
    initial_cash: float,
    min_trade: int,
) -> tuple[dict, pd.DataFrame]:
    period_stats = performance_by_period(equity, DEFAULT_VALIDATION_PERIODS)
    executed = [row for row in trades if row.get("action") in {"BUY", "SELL"}]
    target_exposure = 1.0 - targets.get(
        "__CASH__", pd.Series(0.0, index=targets.index)
    )
    realised_exposure = (
        1.0 - equity["cash"] / equity["equity"]
        if not equity.empty
        else pd.Series(dtype=float)
    )
    period_records_frame = period_stats.reset_index().astype(object)
    period_records = period_records_frame.where(
        pd.notna(period_records_frame), None
    ).to_dict(orient="records")
    summary = {
        "label": label,
        "initial_cash": float(initial_cash),
        "min_trade": int(min_trade),
        "target_rows": int(len(targets)),
        "equity_rows": int(len(equity)),
        "executed_orders": int(len(executed)),
        "buy_orders": int(sum(row["action"] == "BUY" for row in executed)),
        "sell_orders": int(sum(row["action"] == "SELL" for row in executed)),
        "average_target_exposure_pct": float(target_exposure.mean() * 100.0),
        "average_realised_exposure_pct": float(realised_exposure.mean() * 100.0),
        "average_positions": float(equity["positions"].mean()),
        "explicit_fee_and_tax_pct_initial": float(
            sum(
                row["final_qty"]
                * row["exec_price"]
                * (FEE_PER_SIDE + (TAX_RATE_SELL if row["action"] == "SELL" else 0.0))
                for row in executed
            )
            / initial_cash
            * 100.0
        ),
        "periods": period_records,
    }
    return summary, period_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit allocator targets and execution")
    parser.add_argument("--strategy", default="multi_allocator_plus_safe_etf_kqm")
    parser.add_argument("--cache-suffix", default="20190101_20251113")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--initial-cash", type=float, default=2_100_000.0)
    parser.add_argument("--reference-cash", type=float, default=100_000_000.0)
    parser.add_argument("--min-trade", type=int, default=50_000)
    parser.add_argument("--price-band-pct", type=float, default=3.0)
    parser.add_argument("--legacy-targets", help="Previously saved legacy target parquet")
    parser.add_argument(
        "--cash-preserved-targets",
        help="Previously saved cash-preserved target parquet",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()

    enriched, indexes = load_enriched_snapshot(
        args.cache_suffix, args.start_date, args.end_date
    )
    if bool(args.legacy_targets) != bool(args.cash_preserved_targets):
        raise ValueError(
            "--legacy-targets and --cash-preserved-targets must be supplied together"
        )
    if args.legacy_targets:
        legacy_targets = pd.read_parquet(args.legacy_targets).sort_index()
        audited_targets = pd.read_parquet(args.cash_preserved_targets).sort_index()
    else:
        strategy = get_strategy(args.strategy)
        legacy_targets = strategy.compute_security_targets(
            enriched,
            market_index=indexes["KOSDAQ"],
            secondary_index=indexes["KOSPI"],
            silent=True,
            target_policy="legacy",
        )
        if legacy_targets.empty:
            raise RuntimeError("strategy returned no security targets")

        audited_targets = strategy._combine_strategy_targets_preserving_cash(
            strategy.latest_child_weight_frames,
            strategy.latest_target_weights,
            strategy.latest_final_exposure.reindex(
                strategy.latest_target_weights.index
            ),
            style_context=strategy.latest_style_context,
            security_style_map=strategy.latest_security_style_map,
        )

    target_start = pd.Timestamp(args.start_date)
    target_end = pd.Timestamp(args.end_date) if args.end_date else None
    legacy_targets = legacy_targets.loc[legacy_targets.index >= target_start]
    audited_targets = audited_targets.loc[audited_targets.index >= target_start]
    if target_end is not None:
        legacy_targets = legacy_targets.loc[legacy_targets.index <= target_end]
        audited_targets = audited_targets.loc[audited_targets.index <= target_end]

    comparisons = []
    period_tables = []
    curves = {}
    trade_tables = {}
    for label, targets, initial_cash, min_trade in (
        ("legacy_small_account", legacy_targets, args.initial_cash, args.min_trade),
        ("legacy_reference_account", legacy_targets, args.reference_cash, 0),
        ("cash_preserved_small_account", audited_targets, args.initial_cash, args.min_trade),
        ("cash_preserved_reference_account", audited_targets, args.reference_cash, 0),
    ):
        curve, trades = simulate(
            targets,
            enriched,
            initial_cash=initial_cash,
            min_trade=min_trade,
            price_band_pct=args.price_band_pct,
        )
        summary, periods = _summary_for(
            label, targets, curve, trades, initial_cash, min_trade
        )
        comparisons.append(summary)
        periods = periods.copy()
        periods.insert(0, "approach", label)
        period_tables.append(periods.reset_index())
        curves[label] = curve["equity"]
        trade_tables[label] = pd.DataFrame(trades)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"strategy_audit_{args.strategy}_{stamp}"
    summary_path = output_dir / f"{stem}.json"
    metrics_path = output_dir / f"{stem}_periods.csv"
    equity_path = output_dir / f"{stem}_equity.csv"
    legacy_target_path = output_dir / f"{stem}_legacy_targets.parquet"
    audited_target_path = output_dir / f"{stem}_cash_preserved_targets.parquet"

    payload = {
        "strategy": args.strategy,
        "cache_suffix": args.cache_suffix,
        "data_start": str(min(frame.index.min() for frame in indexes.values()).date()),
        "data_end": str(max(frame.index.max() for frame in indexes.values()).date()),
        "universe_size": len(enriched),
        "initial_cash": args.initial_cash,
        "reference_cash": args.reference_cash,
        "min_trade": args.min_trade,
        "price_band_pct": args.price_band_pct,
        "comparison": comparisons,
        "known_limitations": [
            "The cached universe is a later snapshot, not point-in-time constituents; survivorship bias remains.",
            "The 2025 test cache ends in November and does not cover the 2026 live period.",
            "Corporate actions are only as accurate as the stored adjusted cache.",
        ],
        "artifacts": {
            "period_metrics": str(metrics_path),
            "equity_curves": str(equity_path),
            "legacy_targets": str(legacy_target_path),
            "cash_preserved_targets": str(audited_target_path),
        },
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pd.concat(period_tables, ignore_index=True).to_csv(metrics_path, index=False)
    pd.DataFrame(curves).to_csv(equity_path)
    legacy_targets.to_parquet(legacy_target_path)
    audited_targets.to_parquet(audited_target_path)
    for label, trades in trade_tables.items():
        trades.to_csv(output_dir / f"{stem}_{label}_trades.csv", index=False)

    print(json.dumps({**payload, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
