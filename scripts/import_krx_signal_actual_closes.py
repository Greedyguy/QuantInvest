#!/usr/bin/env python3
"""Import KRX all-stock snapshots used for actual-price candidate screening."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krx_execution_data import normalize_krx_actual_close, validate_actual_close_panel
from point_in_time_constituents import (
    constituents_asof,
    load_point_in_time_constituents,
)
from scripts.import_krx_fundamental_snapshot import _read_download


DEVELOPMENT_START = pd.Timestamp("2018-06-29")
DEVELOPMENT_END = pd.Timestamp("2022-12-29")
FILE_PATTERN = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\.(?:csv|xls|xlsx)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_signal_files(input_dir: Path) -> dict[pd.Timestamp, Path]:
    """Return one unambiguously dated KRX all-stock file per decision date."""

    files: dict[pd.Timestamp, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        date = pd.Timestamp(match.group("date")).normalize()
        if date in files:
            raise ValueError(f"duplicate KRX signal-date input: {date.date()}")
        files[date] = path
    if not files:
        raise ValueError(f"no dated KRX all-stock downloads found in {input_dir}")
    return dict(sorted(files.items()))


def load_required_signal_dates(path: Path) -> set[pd.Timestamp]:
    """Load the frozen development decision calendar."""

    frame = pd.read_csv(path)
    if "signal_date" not in frame:
        raise ValueError("decision-date file must contain signal_date")
    dates = pd.to_datetime(frame["signal_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("decision signal dates must be valid and unique")
    if not dates.between(DEVELOPMENT_START, DEVELOPMENT_END).all():
        raise ValueError("decision dates must remain inside the development period")
    return set(dates)


def build_signal_actual_panel(
    files: dict[pd.Timestamp, Path],
    constituents: pd.DataFrame,
    *,
    required_dates: set[pd.Timestamp],
) -> tuple[pd.DataFrame, list[dict]]:
    """Normalize exact decision-date closes and prove constituent coverage."""

    present = set(files)
    missing_dates = sorted(required_dates - present)
    unexpected_dates = sorted(present - required_dates)
    if missing_dates:
        raise ValueError(
            "missing KRX decision-date downloads: "
            f"{[date.date().isoformat() for date in missing_dates]}"
        )
    if unexpected_dates:
        raise ValueError(
            "input includes unregistered decision dates: "
            f"{[date.date().isoformat() for date in unexpected_dates]}"
        )

    rows: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for date in sorted(required_dates):
        path = files[date]
        raw = _read_download(path).copy()
        raw["date"] = date
        normalized = normalize_krx_actual_close(
            raw,
            source="KRX Data Marketplace screen 12001",
        )
        members = constituents_asof(constituents, date, max_age_days=100)
        if members.empty:
            raise ValueError(f"no point-in-time constituents for {date.date()}")
        required_tickers = set(members.index.astype(str))
        present_tickers = set(normalized["ticker"].astype(str))
        missing_tickers = sorted(required_tickers - present_tickers)
        if missing_tickers:
            raise ValueError(
                f"KRX decision-date coverage missing on {date.date()}: "
                f"{missing_tickers}"
            )
        selected = normalized.loc[normalized["ticker"].isin(required_tickers)]
        rows.append(selected)
        provenance.append(
            {
                "signal_date": date.date().isoformat(),
                "filename": path.name,
                "sha256": _sha256(path),
                "market_rows": int(len(normalized)),
                "constituent_rows": int(len(selected)),
            }
        )
    panel = validate_actual_close_panel(pd.concat(rows, ignore_index=True))
    return panel, provenance


def import_signal_actual_closes(
    input_dir: Path,
    decision_dates_path: Path,
    constituents_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[Path, Path]:
    """Write an immutable screening panel with content-addressed provenance."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("signal actual-close output is immutable and already exists")
    required_dates = load_required_signal_dates(decision_dates_path)
    constituents = load_point_in_time_constituents(constituents_path)
    panel, provenance = build_signal_actual_panel(
        discover_signal_files(input_dir),
        constituents,
        required_dates=required_dates,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        panel.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        panel.to_csv(output_path, index=False)
    else:
        raise ValueError("normalized output must be .csv or .parquet")
    manifest = {
        "source": "KRX Data Marketplace screen 12001",
        "price_basis": "actual_traded",
        "purpose": "candidate_screening_only",
        "development_only": True,
        "development_start": DEVELOPMENT_START.date().isoformat(),
        "development_end": DEVELOPMENT_END.date().isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_date_count": len(required_dates),
        "normalized_rows": int(len(panel)),
        "normalized_file": {
            "filename": output_path.name,
            "sha256": _sha256(output_path),
        },
        "raw_inputs": provenance,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--decision-dates",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "krx_small_account_decision_dates_2018_2022.csv",
    )
    parser.add_argument(
        "--constituents",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex200_top30_pcf_quarterly_2018_2025.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    output_path, manifest_path = import_signal_actual_closes(
        args.input_dir,
        args.decision_dates,
        args.constituents,
        args.output,
        manifest,
    )
    print(f"normalized signal actual closes: {output_path}")
    print(f"provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
