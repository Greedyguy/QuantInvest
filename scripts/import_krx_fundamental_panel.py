#!/usr/bin/env python3
"""Import paired KRX quarterly downloads into an immutable PIT factor panel."""

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

from market_benchmark import load_samsung_kodex_standard_xls
from point_in_time_constituents import load_point_in_time_constituents
from point_in_time_fundamentals import (
    normalize_krx_snapshot,
    validate_point_in_time_fundamentals,
)
from scripts.import_krx_fundamental_snapshot import _read_download


DEVELOPMENT_START = pd.Timestamp("2018-06-29")
DEVELOPMENT_END = pd.Timestamp("2022-12-29")
FILE_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<kind>fundamentals|trading)\.(?:csv|xls|xlsx)$",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_snapshot_pairs(input_dir: Path) -> dict[pd.Timestamp, dict[str, Path]]:
    """Match strict DATE_fundamentals and DATE_trading raw filenames."""

    pairs: dict[pd.Timestamp, dict[str, Path]] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        date = pd.Timestamp(match.group("date")).normalize()
        kind = match.group("kind").lower()
        if kind in pairs.setdefault(date, {}):
            raise ValueError(f"duplicate {kind} input for {date.date()}")
        pairs[date][kind] = path
    if not pairs:
        raise ValueError(f"no paired KRX snapshot downloads found in {input_dir}")
    incomplete = {
        date.date().isoformat(): sorted({"fundamentals", "trading"} - set(files))
        for date, files in pairs.items()
        if set(files) != {"fundamentals", "trading"}
    }
    if incomplete:
        raise ValueError(f"incomplete KRX snapshot pairs: {incomplete}")
    return dict(sorted(pairs.items()))


def required_development_snapshots(constituents: pd.DataFrame) -> set[pd.Timestamp]:
    """Return every pre-registered quarterly PCF date in development."""

    dates = pd.to_datetime(constituents["as_of_date"]).dt.normalize()
    return set(dates.loc[dates.between(DEVELOPMENT_START, DEVELOPMENT_END)])


def next_trading_session(snapshot: pd.Timestamp, trading_dates: pd.Index) -> pd.Timestamp:
    """Resolve after-close availability without a calendar-day shortcut."""

    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
    position = int(dates.searchsorted(snapshot, side="right"))
    if position >= len(dates):
        raise ValueError(f"no next KRX session after {snapshot.date()}")
    return pd.Timestamp(dates[position]).normalize()


def build_fundamental_panel(
    pairs: dict[pd.Timestamp, dict[str, Path]],
    *,
    required_snapshots: set[pd.Timestamp],
    trading_dates: pd.Index,
) -> tuple[pd.DataFrame, list[dict]]:
    """Normalize every required pair and preserve raw content hashes."""

    present = set(pairs)
    missing = sorted(required_snapshots - present)
    unexpected = sorted(present - required_snapshots)
    if missing:
        raise ValueError(
            f"missing required KRX snapshots: {[date.date().isoformat() for date in missing]}"
        )
    if unexpected:
        raise ValueError(
            "input includes non-development snapshots: "
            f"{[date.date().isoformat() for date in unexpected]}"
        )
    rows: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for snapshot in sorted(required_snapshots):
        if snapshot > DEVELOPMENT_END:
            raise ValueError("snapshot opens the sealed post-development period")
        files = pairs[snapshot]
        available = next_trading_session(snapshot, trading_dates)
        normalized = normalize_krx_snapshot(
            _read_download(files["fundamentals"]),
            _read_download(files["trading"]),
            snapshot_date=snapshot,
            available_date=available,
        )
        rows.append(normalized)
        provenance.append(
            {
                "snapshot_date": snapshot.date().isoformat(),
                "available_date": available.date().isoformat(),
                "rows": int(len(normalized)),
                "fundamentals": {
                    "filename": files["fundamentals"].name,
                    "sha256": _sha256(files["fundamentals"]),
                },
                "trading": {
                    "filename": files["trading"].name,
                    "sha256": _sha256(files["trading"]),
                },
            }
        )
    panel = validate_point_in_time_fundamentals(pd.concat(rows, ignore_index=True))
    return panel, provenance


def import_fundamental_panel(
    input_dir: Path,
    constituents_path: Path,
    core_standard_file: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[Path, Path]:
    """Write an immutable development panel and provenance manifest."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("fundamental panel output is immutable and already exists")
    constituents = load_point_in_time_constituents(constituents_path)
    required = required_development_snapshots(constituents)
    trading_dates = load_samsung_kodex_standard_xls(core_standard_file).index
    panel, provenance = build_fundamental_panel(
        discover_snapshot_pairs(input_dir),
        required_snapshots=required,
        trading_dates=trading_dates,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        panel.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        panel.to_csv(output_path, index=False)
    else:
        raise ValueError("normalized output must be .csv or .parquet")
    manifest = {
        "source": "KRX Data Marketplace",
        "development_only": True,
        "development_start": DEVELOPMENT_START.date().isoformat(),
        "development_end": DEVELOPMENT_END.date().isoformat(),
        "availability_rule": "first KRX trading session after each snapshot close",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_count": len(required),
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
        "--constituents",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reference"
        / "kodex200_top30_pcf_quarterly_2018_2025.csv",
    )
    parser.add_argument(
        "--core-standard-file",
        type=Path,
        default=Path("/private/tmp/kodex200_standard.xls"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    output_path, manifest_path = import_fundamental_panel(
        args.input_dir,
        args.constituents,
        args.core_standard_file,
        args.output,
        manifest,
    )
    print(f"normalized fundamental panel: {output_path}")
    print(f"provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
