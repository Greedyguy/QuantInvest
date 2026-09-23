#!/usr/bin/env python3
"""Extract a sealed-period-safe KODEX market-close and tax-NAV panel."""

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

from market_benchmark import load_samsung_kodex_standard_xls


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_development_tax_nav(
    official_daily: pd.DataFrame,
    *,
    ticker: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Keep only the predeclared development interval and required tax fields."""

    required = {"market_close", "tax_nav"}
    missing = sorted(required - set(official_daily.columns))
    if missing:
        raise ValueError(f"official KODEX workbook is missing columns: {missing}")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("development end must not precede start")
    frame = official_daily.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().loc[start:end, ["market_close", "tax_nav"]]
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise ValueError("official KODEX workbook does not cover development period")
    values = frame.apply(pd.to_numeric, errors="coerce")
    if values.isna().any(axis=None) or values.le(0.0).any(axis=None):
        raise ValueError("official KODEX market close and tax NAV must be positive")
    result = values.reset_index(names="date")
    result.insert(1, "ticker", str(ticker).zfill(6))
    result["source"] = "Samsung Asset Management official standard-price workbook"
    return result


def import_tax_nav(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path]:
    """Write an immutable development-only tax-NAV panel and provenance."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("tax-NAV output is immutable and already exists")
    panel = build_development_tax_nav(
        load_samsung_kodex_standard_xls(input_path),
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    manifest = {
        "source": "Samsung Asset Management official standard-price workbook",
        "ticker": str(ticker).zfill(6),
        "development_only": True,
        "requested_start": pd.Timestamp(start_date).date().isoformat(),
        "requested_end": pd.Timestamp(end_date).date().isoformat(),
        "first_trading_date": panel["date"].min().date().isoformat(),
        "last_trading_date": panel["date"].max().date().isoformat(),
        "rows": int(len(panel)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": {"filename": input_path.name, "sha256": _sha256(input_path)},
        "normalized_file": {
            "filename": output_path.name,
            "sha256": _sha256(output_path),
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
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    output, manifest = import_tax_nav(
        args.input,
        args.output,
        args.manifest or args.output.with_suffix(".manifest.json"),
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
    )
    print(f"normalized KODEX tax NAV: {output}")
    print(f"provenance manifest: {manifest}")


if __name__ == "__main__":
    main()
