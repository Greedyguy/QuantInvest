#!/usr/bin/env python3
"""Write a sanitized, no-order prospective KODEX 200 re-entry signal record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


SPEC_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "k200_low_turnover_reentry_prospective_shadow_spec.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_kodex200_price_file() -> Path:
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


def build_shadow_payload(
    prices: pd.DataFrame,
    *,
    source_path: Path,
    spec_path: Path = SPEC_PATH,
    generated_at: datetime | None = None,
) -> dict:
    """Freeze the observable state and indicators without planning any order."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    strategy_path = PROJECT_ROOT / "strategies" / "k200_low_turnover_reentry.py"
    strategy_hash = _sha256(strategy_path)
    if strategy_hash != spec["strategy_source_sha256"]:
        raise RuntimeError("frozen KODEX 200 shadow strategy hash changed")
    frame = prices.copy().sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    strategy = K200LowTurnoverReentry()
    indicators = strategy._indicators(frame)
    states = strategy.compute_state_history(frame)
    if indicators.empty or states.empty:
        raise ValueError("KODEX 200 history is too short for the frozen strategy")
    as_of = states.index[-1]
    state_row = states.loc[as_of]
    indicator_row = indicators.loc[as_of]
    state = str(state_row["state"])
    exposure = float(state_row["target_exposure"])
    timestamp = generated_at or datetime.now(timezone.utc)
    eligible = as_of >= pd.Timestamp(spec["first_eligible_signal_date"])
    return {
        "mode": "prospective_paper_shadow",
        "status": "eligible_observation" if eligible else "pre_start_diagnostic",
        "generated_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "signal_date": as_of.date().isoformat(),
        "strategy": spec["strategy"],
        "strategy_source_sha256": strategy_hash,
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
        "source": {
            "filename": source_path.name,
            "data_start": frame.index.min().date().isoformat(),
            "data_end": frame.index.max().date().isoformat(),
        },
        "benchmark_target": {"069500": 1.0, "__CASH__": 0.0},
        "measurement_sessions_required": int(
            spec["evidence_policy"]["minimum_subsequent_sessions"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "signals"
    )
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = parser.parse_args()
    price_path = args.price_file or select_kodex200_price_file()
    prices = pd.read_parquet(price_path)
    payload = build_shadow_payload(
        prices, source_path=price_path, spec_path=args.spec
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"k200_reentry_shadow_{payload['signal_date']}.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"KODEX 200 prospective shadow record: {output}")


if __name__ == "__main__":
    main()
