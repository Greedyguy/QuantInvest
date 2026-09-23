#!/usr/bin/env python3
"""Normalize one official KRX KODEX 200 history for prospective shadow audit."""

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

from krx_execution_data import normalize_krx_actual_close


TICKER = "069500"
DEFAULT_HISTORY_START = pd.Timestamp("2020-01-02")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_download(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path, dtype="string")
    if path.suffix.lower() == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=encoding, dtype="string")
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError(f"could not detect KRX CSV encoding: {path}") from last_error
    raise ValueError(f"unsupported KRX download format: {path.suffix}")


def build_official_kodex200_panel(
    raw: pd.DataFrame,
    *,
    expected_start: str | pd.Timestamp = DEFAULT_HISTORY_START,
    expected_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    panel = normalize_krx_actual_close(
        raw,
        ticker=TICKER,
        source="KRX Data Marketplace screen 13103",
    )
    start = pd.Timestamp(expected_start).normalize()
    if panel["date"].min() != start:
        raise ValueError(
            "KRX KODEX 200 history does not start on the frozen history date"
        )
    if expected_end is not None:
        end = pd.Timestamp(expected_end).normalize()
        if panel["date"].max() != end:
            raise ValueError(
                "KRX KODEX 200 history does not end on the requested audit date"
            )
    return panel


def import_official_kodex200_prices(
    raw_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_start: str | pd.Timestamp = DEFAULT_HISTORY_START,
    expected_end: str | pd.Timestamp | None = None,
) -> tuple[Path, Path]:
    """Create immutable normalized official prices and content provenance."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("official KRX shadow-audit output is immutable")
    panel = build_official_kodex200_panel(
        _read_download(raw_path),
        expected_start=expected_start,
        expected_end=expected_end,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        panel.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == ".parquet":
        panel.to_parquet(output_path, index=False)
    else:
        raise ValueError("normalized KRX output must be .csv or .parquet")
    manifest = {
        "source": "KRX Data Marketplace screen 13103",
        "source_url": (
            "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/"
            "index.cmd?menuId=MDC0201030103"
        ),
        "price_basis": "actual_traded",
        "purpose": "prospective_shadow_official_input_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_tickers": [TICKER],
        "fields": ["open", "high", "low", "close"],
        "history_coverage_start": panel["date"].min().date().isoformat(),
        "history_coverage_end": panel["date"].max().date().isoformat(),
        "normalized_rows": int(len(panel)),
        "normalized_file": {
            "filename": output_path.name,
            "sha256": _sha256(output_path),
        },
        "raw_input": {
            "filename": raw_path.name,
            "sha256": _sha256(raw_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-start", default="2020-01-02")
    parser.add_argument("--expected-end")
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    output, manifest = import_official_kodex200_prices(
        args.input,
        args.output,
        manifest,
        expected_start=args.expected_start,
        expected_end=args.expected_end,
    )
    print(f"normalized official KRX KODEX 200 prices: {output}")
    print(f"provenance manifest: {manifest}")


if __name__ == "__main__":
    main()
