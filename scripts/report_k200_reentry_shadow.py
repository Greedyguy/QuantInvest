#!/usr/bin/env python3
"""Write a sanitized, no-order prospective KODEX 200 re-entry signal record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry
from backtest_live_execution import simulate
from krx_execution_data import actual_ohlc_for_ticker, load_actual_close_panel
from market_benchmark import (
    adjust_ohlc_for_distributions,
    load_distribution_events,
    reconstruct_actual_ohlc_from_adjusted,
)


SPEC_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "k200_low_turnover_reentry_prospective_shadow_spec.json"
)
DEFAULT_DISTRIBUTIONS_PATH = (
    PROJECT_ROOT / "data" / "reference" / "kodex200_distributions.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the causal OHLC prefix independently of parquet metadata."""

    columns = [column for column in ("open", "high", "low", "close") if column in frame]
    canonical = frame.loc[:, columns].copy().sort_index()
    canonical.index = pd.to_datetime(canonical.index).strftime("%Y-%m-%d")
    payload = canonical.to_csv(float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _table_sha256(frame: pd.DataFrame) -> str:
    """Hash all normalized table values for injected official cash flows."""

    canonical = frame.copy()
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _business_day_gap(old: pd.Timestamp, new: pd.Timestamp) -> int:
    if new <= old:
        return 0
    return int(np.busday_count(np.datetime64(old.date()), np.datetime64(new.date())))


def _data_freshness(
    as_of: pd.Timestamp, timestamp: datetime, spec: dict
) -> dict:
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    timezone_name = str(spec["observation_policy"]["timezone"])
    requested = pd.Timestamp(timestamp.astimezone(ZoneInfo(timezone_name)).date())
    future_dated = as_of.normalize() > requested
    gap = _business_day_gap(as_of.normalize(), requested)
    tolerance = int(spec["observation_policy"]["max_source_business_day_gap"])
    return {
        "requested_local_date": requested.date().isoformat(),
        "source_date": as_of.date().isoformat(),
        "business_day_gap": gap,
        "tolerance_business_days": tolerance,
        "future_dated": bool(future_dated),
        "fresh": bool(not future_dated and gap <= tolerance),
    }


def _targets_from_states(
    states: pd.DataFrame,
    start: pd.Timestamp,
    *,
    risk_on_exposure: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Place each next-open state on the preceding close, matching execution."""

    effective_next_open = states["state"].shift(-1).fillna(states["state"])
    risk_on = effective_next_open.eq(K200LowTurnoverReentry.RISK_ON)
    candidate = pd.DataFrame(
        {
            "069500": np.where(risk_on, risk_on_exposure, 0.0),
            "__CASH__": np.where(risk_on, 1.0 - risk_on_exposure, 1.0),
        },
        index=states.index,
    ).loc[start:]
    benchmark = pd.DataFrame(
        {"069500": 1.0, "__CASH__": 0.0}, index=states.index
    ).loc[start:]
    return candidate, benchmark


def _ending_quantity(trades: list[dict]) -> int:
    quantity = 0
    for trade in trades:
        if trade.get("ticker") != "069500":
            continue
        if trade.get("action") == "BUY":
            quantity += int(trade.get("final_qty", 0))
        elif trade.get("action") == "SELL":
            quantity -= int(trade.get("final_qty", 0))
    if quantity < 0:
        raise ValueError("paper account produced a negative share quantity")
    return quantity


def _account_summary(
    equity: pd.DataFrame, trades: list[dict], initial_cash: float
) -> dict:
    curve = pd.to_numeric(equity["equity"], errors="raise")
    normalized = curve / float(initial_cash)
    orders = [trade for trade in trades if trade.get("action") in {"BUY", "SELL"}]
    distributions = [
        trade for trade in trades if trade.get("action") == "DISTRIBUTION"
    ]
    return {
        "equity_krw": float(curve.iloc[-1]),
        "cash_krw": float(equity["cash"].iloc[-1]),
        "distribution_receivable_krw": float(
            equity["distribution_receivable"].iloc[-1]
        ),
        "quantity": _ending_quantity(trades),
        "return_pct": float(normalized.iloc[-1] - 1.0) * 100.0,
        "mdd_pct": float((normalized / normalized.cummax() - 1.0).min()) * 100.0,
        "order_count": len(orders),
        "distribution_count": len(distributions),
        "fees_krw": float(sum(float(trade.get("fee", 0.0)) for trade in orders)),
        "taxes_krw": float(sum(float(trade.get("tax", 0.0)) for trade in trades)),
        "integer_share_execution": True,
    }


def _provisional_comparison(
    candidate: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    spec: dict,
) -> dict:
    aligned = pd.concat(
        [candidate["equity"].rename("candidate"), benchmark["equity"].rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    normalized = aligned / aligned.iloc[0]
    elapsed_days = int((normalized.index[-1] - normalized.index[0]).days)
    if elapsed_days > 0:
        years = elapsed_days / 365.2425
        candidate_cagr = float(normalized["candidate"].iloc[-1] ** (1.0 / years) - 1.0)
        benchmark_cagr = float(normalized["benchmark"].iloc[-1] ** (1.0 / years) - 1.0)
        annualised_excess = candidate_cagr - benchmark_cagr
    else:
        candidate_cagr = benchmark_cagr = annualised_excess = None

    candidate_mdd = float(
        (normalized["candidate"] / normalized["candidate"].cummax() - 1.0).min()
    )
    benchmark_mdd = float(
        (normalized["benchmark"] / normalized["benchmark"].cummax() - 1.0).min()
    )
    mdd_disadvantage = abs(candidate_mdd) - abs(benchmark_mdd)

    # Only fully completed calendar months count.  The current partial month is
    # never allowed to improve or worsen a registered checkpoint rate.
    current_period = normalized.index[-1].to_period("M")
    completed = normalized.loc[normalized.index.to_period("M") < current_period]
    month_ends = completed.groupby(completed.index.to_period("M")).tail(1)
    monthly_beats = month_ends["candidate"] > month_ends["benchmark"]
    monthly_beat_rate = float(monthly_beats.mean()) if len(monthly_beats) else None

    required = int(spec["evidence_policy"]["minimum_subsequent_sessions"])
    subsequent_sessions = max(len(aligned) - 1, 0)
    evidence_complete = subsequent_sessions >= required
    gates = None
    passes_all = None
    if evidence_complete:
        gates = {
            "annualised_excess_return": bool(
                annualised_excess is not None
                and annualised_excess
                >= float(
                    spec["evidence_policy"][
                        "final_annualised_excess_return_pct_point_min"
                    ]
                )
                / 100.0
            ),
            "mdd_disadvantage": bool(
                mdd_disadvantage
                <= float(
                    spec["evidence_policy"][
                        "final_mdd_disadvantage_pct_point_max"
                    ]
                )
                / 100.0
            ),
            "completed_month_checkpoint_beat_rate": bool(
                monthly_beat_rate is not None
                and monthly_beat_rate
                >= float(
                    spec["evidence_policy"][
                        "monthly_checkpoint_beat_rate_pct_min"
                    ]
                )
                / 100.0
            ),
        }
        passes_all = all(gates.values())
    return {
        "subsequent_sessions": subsequent_sessions,
        "required_subsequent_sessions": required,
        "evidence_complete": evidence_complete,
        "candidate_cagr_pct": (
            None if candidate_cagr is None else candidate_cagr * 100.0
        ),
        "benchmark_cagr_pct": (
            None if benchmark_cagr is None else benchmark_cagr * 100.0
        ),
        "annualised_excess_return_pct_point": (
            None if annualised_excess is None else annualised_excess * 100.0
        ),
        "candidate_mdd_pct": candidate_mdd * 100.0,
        "benchmark_mdd_pct": benchmark_mdd * 100.0,
        "mdd_disadvantage_pct_point": mdd_disadvantage * 100.0,
        "completed_month_checkpoints": int(len(month_ends)),
        "completed_month_checkpoint_beat_rate_pct": (
            None if monthly_beat_rate is None else monthly_beat_rate * 100.0
        ),
        "registered_gates": gates,
        "provisional_passes_all_gates": passes_all,
        "eligible_for_final_official_input_audit": bool(
            evidence_complete and passes_all
        ),
        "capital_authorized": False,
    }


def _paper_account_snapshot(
    execution_frame: pd.DataFrame,
    states: pd.DataFrame,
    distributions: pd.DataFrame,
    spec: dict,
) -> dict:
    start = pd.Timestamp(spec["first_eligible_signal_date"])
    candidate_targets, benchmark_targets = _targets_from_states(
        states,
        start,
        risk_on_exposure=float(spec["parameters"]["risk_on_exposure"]),
    )
    if candidate_targets.empty:
        return {
            "status": "not_started",
            "subsequent_sessions": 0,
            "required_subsequent_sessions": int(
                spec["evidence_policy"]["minimum_subsequent_sessions"]
            ),
            "capital_authorized": False,
        }

    execution = spec["execution_model"]
    initial_cash = float(spec["shadow_account_krw"])
    common = {
        "enriched": {"069500": execution_frame},
        "initial_cash": initial_cash,
        "min_trade": int(execution["minimum_trade_krw"]),
        "price_band_pct": float(execution["price_band_pct"]),
        "blocked_tickers": set(),
        "sell_tax_rate_by_ticker": {"069500": float(execution["sell_tax_rate"])},
        "rebalance_only_on_target_change": True,
        "distribution_events_by_ticker": {"069500": distributions},
        "fee_per_side": float(execution["fee_per_side"]),
        "slippage_entry": float(execution["slippage_entry"]),
        "slippage_exit": float(execution["slippage_exit"]),
    }
    candidate_equity, candidate_trades = simulate(
        candidate_targets, **common
    )
    benchmark_equity, benchmark_trades = simulate(
        benchmark_targets, **common
    )
    comparison = _provisional_comparison(
        candidate_equity, benchmark_equity, spec=spec
    )
    return {
        "status": (
            "awaiting_final_official_input_audit"
            if comparison["eligible_for_final_official_input_audit"]
            else "collecting"
        ),
        "provisional_until_final_official_input_audit": True,
        "initial_signal_date": candidate_targets.index[0].date().isoformat(),
        "latest_equity_date": candidate_equity.index[-1].date().isoformat(),
        "execution_model": execution,
        "candidate": _account_summary(
            candidate_equity, candidate_trades, initial_cash
        ),
        "benchmark": _account_summary(
            benchmark_equity, benchmark_trades, initial_cash
        ),
        "comparison": comparison,
        "final_audit_requirements": spec["final_audit_requirements"],
        "capital_authorized": False,
    }


def select_kodex200_price_file() -> Path:
    raw_cache = PROJECT_ROOT / "data" / "ohlcv" / "069500.parquet"
    if raw_cache.exists():
        return raw_cache
    candidates = sorted((PROJECT_ROOT / "data" / "enriched").glob("069500_*.parquet"))
    if not candidates:
        raise FileNotFoundError("no cached KODEX 200 history is available")

    def coverage(path: Path) -> tuple[pd.Timestamp, int]:
        try:
            end = pd.Timestamp(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            end = pd.Timestamp.min
        return end, path.stat().st_size

    return max(candidates, key=coverage)


def load_shadow_prices(path: Path, *, official_krx_input: bool = False) -> pd.DataFrame:
    if official_krx_input:
        return actual_ohlc_for_ticker(load_actual_close_panel(path), "069500")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame = frame.set_index("date")
        return frame
    raise ValueError(f"unsupported shadow price format: {path.suffix}")


def _normalize_shadow_frame(
    prices: pd.DataFrame,
    *,
    history_start: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    frame = prices.copy().sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if frame.empty or frame.index.min() > history_start:
        raise ValueError(f"{label} history does not reach the frozen history start")
    frame = frame.loc[history_start:]
    required_columns = {"open", "close"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} history is missing columns: {missing}")
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required_columns)].isna().any(axis=None):
        raise ValueError(f"{label} history contains invalid prices")
    if frame[list(required_columns)].le(0).any(axis=None):
        raise ValueError(f"{label} prices must be positive")
    return frame


def _relative_indicator_context(indicator: pd.Series) -> dict:
    close = float(indicator["close"])
    return {
        "close_to_trend_ma": close / float(indicator["trend_ma"]),
        "close_to_fast_ma": close / float(indicator["fast_ma"]),
        "close_to_medium_ma": close / float(indicator["medium_ma"]),
        "momentum": float(indicator["momentum"]),
        "fast_momentum": float(indicator["fast_momentum"]),
        "drawdown": float(indicator["drawdown"]),
    }


def build_shadow_payload(
    signal_prices: pd.DataFrame,
    *,
    source_path: Path,
    execution_prices: pd.DataFrame | None = None,
    execution_source_path: Path | None = None,
    spec_path: Path = SPEC_PATH,
    distribution_events: pd.DataFrame | None = None,
    distribution_path: Path = DEFAULT_DISTRIBUTIONS_PATH,
    generated_at: datetime | None = None,
    signal_price_authority: str | None = None,
    signal_price_source: str | None = None,
    execution_price_authority: str | None = None,
    execution_price_source: str | None = None,
) -> dict:
    """Freeze adjusted-price signals and actual-price paper accounting."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    strategy_path = PROJECT_ROOT / "strategies" / "k200_low_turnover_reentry.py"
    simulator_path = PROJECT_ROOT / "backtest_live_execution.py"
    cash_flow_path = PROJECT_ROOT / "market_benchmark.py"
    strategy_hash = _sha256(strategy_path)
    if strategy_hash != spec["strategy_source_sha256"]:
        raise RuntimeError("frozen KODEX 200 shadow strategy hash changed")
    if _sha256(simulator_path) != spec["execution_simulator_source_sha256"]:
        raise RuntimeError("frozen paper-account execution simulator hash changed")
    if _sha256(cash_flow_path) != spec["cash_flow_source_sha256"]:
        raise RuntimeError("frozen paper-account cash-flow engine hash changed")

    distributions = (
        distribution_events.copy()
        if distribution_events is not None
        else load_distribution_events(distribution_path)
    )
    distribution_hash = (
        _table_sha256(distributions)
        if distribution_events is not None
        else _sha256(distribution_path)
    )
    history_start = pd.Timestamp(spec["signal_history_start"])
    signal_frame = _normalize_shadow_frame(
        signal_prices,
        history_start=history_start,
        label="KODEX 200 adjusted signal",
    )
    if execution_prices is None:
        execution_frame = reconstruct_actual_ohlc_from_adjusted(
            signal_frame, distributions
        )
        execution_authority = (
            execution_price_authority
            or spec["observation_policy"]["daily_execution_price_authority"]
        )
        execution_source = (
            execution_price_source
            or spec["observation_policy"]["daily_execution_price_source"]
        )
        execution_file = source_path
    else:
        execution_frame = _normalize_shadow_frame(
            execution_prices,
            history_start=history_start,
            label="KODEX 200 actual execution",
        )
        execution_authority = execution_price_authority or "injected_actual_prices"
        execution_source = execution_price_source or "injected actual execution OHLC"
        execution_file = execution_source_path or source_path
    execution_frame = execution_frame.reindex(signal_frame.index)
    if execution_frame[["open", "close"]].isna().any(axis=None):
        raise ValueError("execution history does not cover every signal session")

    strategy = K200LowTurnoverReentry(**spec["parameters"])
    indicators = strategy._indicators(signal_frame)
    states = strategy.compute_state_history(signal_frame)
    if indicators.empty or states.empty:
        raise ValueError("KODEX 200 history is too short for the frozen strategy")
    as_of = states.index[-1]
    state_row = states.loc[as_of]
    indicator_row = indicators.loc[as_of]
    state = str(state_row["state"])
    exposure = float(state_row["target_exposure"])
    timestamp = generated_at or datetime.now(timezone.utc)
    freshness = _data_freshness(as_of, timestamp, spec)
    eligible = as_of >= pd.Timestamp(spec["first_eligible_signal_date"])
    if not freshness["fresh"]:
        status = "stale_input_rejected"
    elif eligible:
        status = "eligible_observation"
    else:
        status = "pre_start_diagnostic"

    signal_authority = (
        signal_price_authority
        or spec["observation_policy"]["daily_signal_price_authority"]
    )
    signal_source = (
        signal_price_source
        or spec["observation_policy"]["daily_signal_price_source"]
    )
    source_file_hash = _sha256(source_path) if source_path.exists() else None
    execution_file_hash = (
        _sha256(execution_file) if execution_file.exists() else None
    )
    return {
        "observation_version": 3,
        "mode": "prospective_paper_shadow",
        "status": status,
        "generated_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "signal_date": as_of.date().isoformat(),
        "strategy": spec["strategy"],
        "strategy_source_sha256": strategy_hash,
        "execution_simulator_source_sha256": _sha256(simulator_path),
        "cash_flow_source_sha256": _sha256(cash_flow_path),
        "spec_sha256": _sha256(spec_path),
        "execution_guard": "NO_ORDERS_SENT",
        "target_weights_effective_at_signal_open": {
            "069500": exposure,
            "__CASH__": 1.0 - exposure,
        },
        "state": state,
        "last_transition_signal_date": (
            None
            if pd.isna(state_row["signal_date"])
            else pd.Timestamp(state_row["signal_date"]).date().isoformat()
        ),
        "reason": str(state_row["reason"]),
        "observable_close_context": {
            key: float(indicator_row[key])
            for key in (
                "close",
                "trend_ma",
                "fast_ma",
                "medium_ma",
                "momentum",
                "fast_momentum",
                "drawdown",
            )
        },
        "scale_invariant_close_context": _relative_indicator_context(indicator_row),
        "source": {
            "signal": {
                "filename": source_path.name,
                "price_basis": "cash_distribution_adjusted",
                "price_authority": signal_authority,
                "price_source": signal_source,
                "data_start": signal_frame.index.min().date().isoformat(),
                "data_end": signal_frame.index.max().date().isoformat(),
                "causal_ohlc_prefix_sha256": _frame_sha256(signal_frame),
                "source_file_sha256": source_file_hash,
            },
            "execution": {
                "filename": execution_file.name,
                "price_basis": "actual_traded",
                "price_authority": execution_authority,
                "price_source": execution_source,
                "data_start": execution_frame.index.min().date().isoformat(),
                "data_end": execution_frame.index.max().date().isoformat(),
                "causal_ohlc_prefix_sha256": _frame_sha256(execution_frame),
                "source_file_sha256": execution_file_hash,
                "provisional": execution_authority
                != "official_krx_actual_traded",
            },
            "distribution_filename": distribution_path.name,
            "distribution_source_sha256": distribution_hash,
            "latest_known_distribution_record_date": (
                None
                if distributions.empty
                else pd.Timestamp(distributions["record_date"].max()).date().isoformat()
            ),
        },
        "data_freshness": freshness,
        "benchmark_target": {"069500": 1.0, "__CASH__": 0.0},
        "measurement_sessions_required": int(
            spec["evidence_policy"]["minimum_subsequent_sessions"]
        ),
        "paper_accounts": _paper_account_snapshot(
            execution_frame, states, distributions, spec
        ),
    }


def write_immutable_shadow_record(payload: dict, output_dir: Path) -> tuple[Path, bool]:
    """Create one immutable record per exchange session; reruns are idempotent."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"k200_reentry_shadow_{payload['signal_date']}.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))

        def stable(value: dict) -> dict:
            result = dict(value)
            result.pop("generated_at_utc", None)
            result.pop("data_freshness", None)
            return result

        if stable(existing) != stable(payload):
            raise RuntimeError(
                f"immutable shadow record conflicts with existing evidence: {output}"
            )
        return output, False
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "signals"
    )
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--distributions", type=Path, default=DEFAULT_DISTRIBUTIONS_PATH
    )
    parser.add_argument(
        "--allow-stale-diagnostic",
        action="store_true",
        help="write a rejected diagnostic instead of failing on stale input",
    )
    parser.add_argument(
        "--official-krx-input",
        action="store_true",
        help=(
            "treat --price-file as normalized KRX actual OHLC, derive adjusted "
            "signals from distributions, and use the original OHLC for execution"
        ),
    )
    args = parser.parse_args()
    price_path = args.price_file or select_kodex200_price_file()
    loaded_prices = load_shadow_prices(
        price_path, official_krx_input=args.official_krx_input
    )
    distributions = load_distribution_events(args.distributions)
    if args.official_krx_input:
        signal_prices = adjust_ohlc_for_distributions(loaded_prices, distributions)
        execution_prices = loaded_prices
    else:
        signal_prices = loaded_prices
        execution_prices = None
    payload = build_shadow_payload(
        signal_prices,
        source_path=price_path,
        execution_prices=execution_prices,
        execution_source_path=price_path if args.official_krx_input else None,
        spec_path=args.spec,
        distribution_path=args.distributions,
        signal_price_authority=(
            "official_krx_derived_distribution_adjusted"
            if args.official_krx_input
            else None
        ),
        signal_price_source=(
            "KRX Data Marketplace screen 13103 adjusted with official distributions"
            if args.official_krx_input
            else None
        ),
        execution_price_authority=(
            "official_krx_actual_traded" if args.official_krx_input else None
        ),
        execution_price_source=(
            "KRX Data Marketplace screen 13103"
            if args.official_krx_input
            else None
        ),
    )
    if payload["status"] == "stale_input_rejected" and not args.allow_stale_diagnostic:
        raise RuntimeError(
            "KODEX 200 shadow input is stale; no eligible observation was written"
        )
    output, created = write_immutable_shadow_record(payload, args.output_dir)
    action = "created" if created else "already verified"
    print(f"KODEX 200 prospective shadow record {action}: {output}")


if __name__ == "__main__":
    main()
