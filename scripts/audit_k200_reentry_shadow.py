#!/usr/bin/env python3
"""Audit frozen KODEX 200 shadow observations and produce a no-capital verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krx_execution_data import actual_ohlc_for_ticker, load_actual_close_panel
from market_benchmark import adjust_ohlc_for_distributions, load_distribution_events
from scripts.report_k200_reentry_shadow import (
    DEFAULT_DISTRIBUTIONS_PATH,
    SPEC_PATH,
    _frame_sha256,
    _paper_account_snapshot,
    _relative_indicator_context,
    _sha256,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


def load_shadow_records(directory: Path) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("k200_reentry_shadow_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        signal_date = str(payload.get("signal_date", ""))
        expected_name = f"k200_reentry_shadow_{signal_date}.json"
        if path.name != expected_name:
            raise ValueError(f"shadow filename and signal date disagree: {path.name}")
        if signal_date in seen:
            raise ValueError(f"duplicate shadow observation: {signal_date}")
        seen.add(signal_date)
        records.append(payload)
    return records


def _normalize_prices(prices: pd.DataFrame, spec: dict) -> pd.DataFrame:
    frame = prices.copy().sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    if frame.index.duplicated().any():
        raise ValueError("official KODEX 200 audit prices contain duplicate sessions")
    missing = sorted({"open", "high", "low", "close"} - set(frame.columns))
    if missing:
        raise ValueError(f"official KODEX 200 audit prices miss columns: {missing}")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "high", "low", "close"]].isna().any(axis=None):
        raise ValueError("official KODEX 200 audit prices contain invalid values")
    start = pd.Timestamp(spec["signal_history_start"])
    if start not in frame.index:
        raise ValueError("official audit prices do not include frozen history start")
    return frame.loc[start:]


def _manifest_checks(
    *,
    price_path: Path | None,
    price_manifest_path: Path | None,
    distribution_path: Path | None,
    distribution_manifest_path: Path | None,
    audit_end: pd.Timestamp,
) -> dict:
    price_errors: list[str] = []
    if price_path is None or price_manifest_path is None:
        price_errors.append("official price file and manifest are both required")
    else:
        manifest = json.loads(price_manifest_path.read_text(encoding="utf-8"))
        if not str(manifest.get("source", "")).startswith("KRX"):
            price_errors.append("official price manifest source is not KRX")
        if manifest.get("price_basis") != "actual_traded":
            price_errors.append("official price manifest is not actual_traded")
        if "069500" not in {str(value) for value in manifest.get("required_tickers", [])}:
            price_errors.append("official price manifest does not require 069500")
        fields = set(manifest.get("fields", []))
        if not {"open", "high", "low", "close"}.issubset(fields):
            price_errors.append("official price manifest does not cover full OHLC")
        normalized = manifest.get("normalized_file", {})
        if normalized.get("sha256") != _sha256(price_path):
            price_errors.append("official price file hash disagrees with manifest")

    distribution_errors: list[str] = []
    if distribution_path is None or distribution_manifest_path is None:
        distribution_errors.append(
            "official distribution file and manifest are both required"
        )
    else:
        manifest = json.loads(
            distribution_manifest_path.read_text(encoding="utf-8")
        )
        if not bool(manifest.get("complete_cash_distribution_history")):
            distribution_errors.append("cash-distribution history is not complete")
        if "069500" not in {str(value) for value in manifest.get("audited_tickers", [])}:
            distribution_errors.append("distribution manifest does not audit 069500")
        coverage_end = pd.to_datetime(
            manifest.get("history_coverage_end"), errors="coerce"
        )
        if pd.isna(coverage_end) or coverage_end < audit_end:
            distribution_errors.append(
                "distribution manifest does not cover the audit end date"
            )
        normalized_file = manifest.get("normalized_file")
        normalized_sha = (
            normalized_file.get("sha256")
            if isinstance(normalized_file, dict)
            else manifest.get("normalized_sha256")
        )
        if normalized_sha is not None and normalized_sha != _sha256(distribution_path):
            distribution_errors.append(
                "distribution file hash disagrees with manifest"
            )
    return {
        "price_input_complete": not price_errors,
        "distribution_input_complete": not distribution_errors,
        "price_errors": price_errors,
        "distribution_errors": distribution_errors,
        "complete": not price_errors and not distribution_errors,
    }


def audit_shadow_records(
    records: list[dict],
    official_prices: pd.DataFrame,
    distributions: pd.DataFrame,
    *,
    spec_path: Path = SPEC_PATH,
    audit_end: str | pd.Timestamp | None = None,
    price_path: Path | None = None,
    price_manifest_path: Path | None = None,
    distribution_path: Path | None = None,
    distribution_manifest_path: Path | None = None,
) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    actual_prices = _normalize_prices(official_prices, spec)
    signal_prices = adjust_ohlc_for_distributions(actual_prices, distributions)
    start = pd.Timestamp(spec["first_eligible_signal_date"])
    eligible_signal_dates = [
        pd.Timestamp(record["signal_date"])
        for record in records
        if pd.Timestamp(record["signal_date"]) >= start
    ]
    if len(eligible_signal_dates) != len(set(eligible_signal_dates)):
        raise ValueError("duplicate eligible shadow observations")
    eligible_records = {
        pd.Timestamp(record["signal_date"]): record
        for record in records
        if pd.Timestamp(record["signal_date"]) >= start
    }
    if audit_end is None:
        end = max(eligible_records) if eligible_records else actual_prices.index.max()
    else:
        end = pd.Timestamp(audit_end).normalize()
    if end > actual_prices.index.max():
        raise ValueError("official audit prices do not cover the requested audit end")
    expected_dates = actual_prices.loc[start:end].index
    expected_set = set(expected_dates)
    record_set = {date for date in eligible_records if date <= end}
    missing_dates = sorted(expected_set - record_set)
    unexpected_dates = sorted(record_set - expected_set)

    strategy = K200LowTurnoverReentry(**spec["parameters"])
    states = strategy.compute_state_history(signal_prices.loc[:end])
    strategy_hash = _sha256(
        PROJECT_ROOT / "strategies" / "k200_low_turnover_reentry.py"
    )
    simulator_hash = _sha256(PROJECT_ROOT / "backtest_live_execution.py")
    cash_flow_hash = _sha256(PROJECT_ROOT / "market_benchmark.py")
    spec_hash = _sha256(spec_path)
    errors: list[str] = []

    if strategy_hash != spec["strategy_source_sha256"]:
        errors.append("current strategy source no longer matches the frozen spec")
    if simulator_hash != spec["execution_simulator_source_sha256"]:
        errors.append("current execution simulator no longer matches the frozen spec")
    if cash_flow_hash != spec["cash_flow_source_sha256"]:
        errors.append("current cash-flow engine no longer matches the frozen spec")
    if missing_dates:
        errors.append(f"missing {len(missing_dates)} exchange-session observations")
    if unexpected_dates:
        errors.append(f"found {len(unexpected_dates)} non-exchange observations")

    def valid_hash(value: object) -> bool:
        return bool(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value)
        )

    for day in sorted(record_set & expected_set):
        record = eligible_records[day]
        actual_prefix = actual_prices.loc[:day]
        signal_prefix = adjust_ohlc_for_distributions(actual_prefix, distributions)
        row = strategy.compute_state_history(signal_prefix).loc[day]
        indicator = strategy._indicators(signal_prefix).loc[day]
        expected_transition = (
            None
            if pd.isna(row["signal_date"])
            else pd.Timestamp(row["signal_date"]).date().isoformat()
        )
        expected_exposure = float(row["target_exposure"])
        source = record.get("source", {})
        signal_source = source.get("signal", {})
        execution_source = source.get("execution", {})
        checks = {
            "observation_version": record.get("observation_version") == 3,
            "eligible_status": record.get("status") == "eligible_observation",
            "no_orders": record.get("execution_guard") == "NO_ORDERS_SENT",
            "strategy_hash": record.get("strategy_source_sha256") == strategy_hash,
            "simulator_hash": record.get("execution_simulator_source_sha256")
            == simulator_hash,
            "cash_flow_hash": record.get("cash_flow_source_sha256") == cash_flow_hash,
            "spec_hash": record.get("spec_sha256") == spec_hash,
            "signal_data_end": signal_source.get("data_end")
            == day.date().isoformat(),
            "execution_data_end": execution_source.get("data_end")
            == day.date().isoformat(),
            "signal_history_start": signal_source.get("data_start")
            == str(spec["signal_history_start"]),
            "execution_history_start": execution_source.get("data_start")
            == str(spec["signal_history_start"]),
            "signal_basis": signal_source.get("price_basis")
            == "cash_distribution_adjusted",
            "execution_basis": execution_source.get("price_basis")
            == "actual_traded",
            "signal_prefix_hash_present": valid_hash(
                signal_source.get("causal_ohlc_prefix_sha256")
            ),
            "execution_prefix_hash_present": valid_hash(
                execution_source.get("causal_ohlc_prefix_sha256")
            ),
            "state": record.get("state") == str(row["state"]),
            "transition": record.get("last_transition_signal_date")
            == expected_transition,
            "reason": record.get("reason") == str(row["reason"]),
            "candidate_capital_guard": record.get("paper_accounts", {}).get(
                "capital_authorized"
            )
            is False,
        }
        weights = record.get("target_weights_effective_at_signal_open", {})
        checks["target_weight"] = bool(
            np.isclose(weights.get("069500", np.nan), expected_exposure)
            and np.isclose(weights.get("__CASH__", np.nan), 1.0 - expected_exposure)
        )
        observed = record.get("scale_invariant_close_context", {})
        expected_context = _relative_indicator_context(indicator)
        checks["scale_invariant_indicator_values"] = all(
            np.isclose(
                observed.get(key, np.nan),
                value,
                rtol=2e-4,
                atol=1e-6,
            )
            for key, value in expected_context.items()
        )
        if execution_source.get("price_authority") == "official_krx_actual_traded":
            checks["official_execution_prefix_hash"] = (
                execution_source.get("causal_ohlc_prefix_sha256")
                == _frame_sha256(actual_prefix)
            )
        if signal_source.get("price_authority") == (
            "official_krx_derived_distribution_adjusted"
        ):
            checks["official_signal_prefix_hash"] = (
                signal_source.get("causal_ohlc_prefix_sha256")
                == _frame_sha256(signal_prefix)
            )
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            errors.append(f"{day.date().isoformat()} failed checks: {failed}")

    account_snapshot = _paper_account_snapshot(
        actual_prices.loc[:end], states.loc[:end], distributions, spec
    )
    manifest_checks = _manifest_checks(
        price_path=price_path,
        price_manifest_path=price_manifest_path,
        distribution_path=distribution_path,
        distribution_manifest_path=distribution_manifest_path,
        audit_end=end,
    )
    observation_complete = not errors
    comparison = account_snapshot.get("comparison", {})
    evidence_complete = bool(comparison.get("evidence_complete", False))
    gates_pass = comparison.get("provisional_passes_all_gates") is True
    if not observation_complete:
        decision = "invalid_observation_evidence"
    elif not evidence_complete:
        decision = "collecting_prospective_sessions"
    elif not manifest_checks["complete"]:
        decision = "awaiting_final_official_input_audit"
    elif gates_pass:
        decision = "eligible_for_separate_live_capital_decision"
    else:
        decision = "reject_after_prospective_validation"
    return {
        "strategy": spec["strategy"],
        "audit_end": end.date().isoformat(),
        "expected_exchange_sessions": int(len(expected_dates)),
        "recorded_exchange_sessions": int(len(record_set & expected_set)),
        "missing_signal_dates": [day.date().isoformat() for day in missing_dates],
        "unexpected_signal_dates": [
            day.date().isoformat() for day in unexpected_dates
        ],
        "observation_evidence_valid": observation_complete,
        "observation_errors": errors,
        "signal_recomputation": {
            "basis": "official_krx_actual_ohlc_adjusted_for_cash_distributions",
            "comparison_basis": "scale_invariant_indicators_and_exact_state",
        },
        "official_input_audit": manifest_checks,
        "paper_accounts": account_snapshot,
        "decision": decision,
        "capital_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-ohlc", type=Path, required=True)
    parser.add_argument("--official-ohlc-manifest", type=Path)
    parser.add_argument("--observations-dir", type=Path, required=True)
    parser.add_argument(
        "--distributions", type=Path, default=DEFAULT_DISTRIBUTIONS_PATH
    )
    parser.add_argument("--distribution-manifest", type=Path)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--audit-end")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prices = actual_ohlc_for_ticker(
        load_actual_close_panel(args.official_ohlc), "069500"
    )
    result = audit_shadow_records(
        load_shadow_records(args.observations_dir),
        prices,
        load_distribution_events(args.distributions),
        spec_path=args.spec,
        audit_end=args.audit_end,
        price_path=args.official_ohlc,
        price_manifest_path=args.official_ohlc_manifest,
        distribution_path=args.distributions,
        distribution_manifest_path=args.distribution_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
