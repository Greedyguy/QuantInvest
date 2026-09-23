#!/usr/bin/env python3
"""Normalize two authenticated KRX downloads into a point-in-time snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from point_in_time_fundamentals import normalize_krx_snapshot


DEVELOPMENT_START = pd.Timestamp("2018-01-01")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_download(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=encoding, dtype="string")
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError(f"CSV encoding could not be detected: {path}") from last_error
    raise ValueError(f"unsupported KRX download format: {suffix}")


def import_snapshot(
    fundamental_path: Path,
    trading_path: Path,
    *,
    snapshot_date,
    available_date,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write an immutable normalized Parquet plus provenance manifest."""
    snapshot = pd.Timestamp(snapshot_date).normalize()
    if snapshot < DEVELOPMENT_START:
        raise ValueError(
            "2015-2017 is sealed holdout data; this development importer refuses to open it"
        )
    available = pd.Timestamp(available_date).normalize()
    normalized = normalize_krx_snapshot(
        _read_download(fundamental_path),
        _read_download(trading_path),
        snapshot_date=snapshot,
        available_date=available,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = snapshot.date().isoformat()
    data_path = output_dir / f"{stem}.parquet"
    manifest_path = output_dir / f"{stem}.manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"snapshot is immutable and already exists: {stem}")
    normalized.to_parquet(data_path, index=False)
    manifest = {
        "snapshot_date": stem,
        "available_date": available.date().isoformat(),
        "source": "KRX Data Marketplace",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(normalized)),
        "raw_inputs": {
            "fundamentals": {
                "filename": fundamental_path.name,
                "sha256": _sha256(fundamental_path),
            },
            "trading": {
                "filename": trading_path.name,
                "sha256": _sha256(trading_path),
            },
        },
        "normalized": {
            "filename": data_path.name,
            "sha256": _sha256(data_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return data_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fundamentals", type=Path, required=True)
    parser.add_argument("--trading", type=Path, required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--available-date", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fundamentals" / "development",
    )
    args = parser.parse_args()
    data_path, manifest_path = import_snapshot(
        args.fundamentals,
        args.trading,
        snapshot_date=args.snapshot_date,
        available_date=args.available_date,
        output_dir=args.output_dir,
    )
    print(f"normalized snapshot: {data_path}")
    print(f"provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
