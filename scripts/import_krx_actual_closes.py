#!/usr/bin/env python3
"""Import raw KRX daily-close downloads into one immutable actual-price panel."""

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
from point_in_time_constituents import load_point_in_time_constituents


DEVELOPMENT_END = pd.Timestamp("2022-12-29")
SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx"}
TICKER_PATTERN = re.compile(r"(?<![0-9A-Z])([0-9A-Z]{6})(?![0-9A-Z])")


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
        raise ValueError(f"CSV encoding could not be detected: {path}") from last_error
    raise ValueError(f"unsupported KRX download format: {path.suffix}")


def discover_raw_files(input_dir: Path) -> dict[str, list[Path]]:
    """Group supported files by an unambiguous six-character code in the name."""

    grouped: dict[str, list[Path]] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        matches = sorted(set(TICKER_PATTERN.findall(path.stem.upper())))
        if len(matches) != 1:
            raise ValueError(
                f"raw filename must contain exactly one KRX ticker: {path.name}"
            )
        grouped.setdefault(matches[0], []).append(path)
    if not grouped:
        raise ValueError(f"no KRX daily-close downloads found in {input_dir}")
    return grouped


def build_actual_close_panel(
    raw_files: dict[str, list[Path]],
    *,
    required_tickers: set[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Normalize raw chunks and prove that the required universe is present."""

    required = {str(ticker).zfill(6) for ticker in required_tickers}
    missing = sorted(required - set(raw_files))
    if missing:
        raise ValueError(f"missing required KRX actual-close downloads: {missing}")
    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for ticker in sorted(required):
        for path in raw_files[ticker]:
            normalized = normalize_krx_actual_close(
                _read_download(path), ticker=ticker
            )
            if normalized["date"].max() > DEVELOPMENT_END:
                raise ValueError(
                    f"{path.name} opens the sealed post-development period"
                )
            frames.append(normalized)
            provenance.append(
                {
                    "ticker": ticker,
                    "filename": path.name,
                    "sha256": _sha256(path),
                    "rows": int(len(normalized)),
                    "first_date": normalized["date"].min().date().isoformat(),
                    "last_date": normalized["date"].max().date().isoformat(),
                }
            )
    return validate_actual_close_panel(pd.concat(frames, ignore_index=True)), provenance


def import_actual_closes(
    input_dir: Path,
    constituents_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    required_tickers: set[str] | None = None,
) -> tuple[Path, Path]:
    """Write an immutable normalized panel and a content-addressed manifest."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("actual-close output is immutable and already exists")
    constituents = load_point_in_time_constituents(constituents_path)
    universe = set(
        constituents.loc[
            constituents["as_of_date"].le(DEVELOPMENT_END), "ticker"
        ].astype(str)
    )
    required = (
        {str(ticker).zfill(6) for ticker in required_tickers}
        if required_tickers is not None
        else universe
    )
    if not required:
        raise ValueError("required KRX actual-close ticker set cannot be empty")
    unexpected = sorted(required - universe)
    if unexpected:
        raise ValueError(f"requested tickers are outside the development universe: {unexpected}")
    panel, provenance = build_actual_close_panel(
        discover_raw_files(input_dir), required_tickers=required
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
        "price_basis": "actual_traded",
        "development_only": True,
        "sealed_after": DEVELOPMENT_END.date().isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_tickers": sorted(required),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--required-tickers",
        nargs="+",
        help="Optional selected development tickers; defaults to the full universe",
    )
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    output_path, manifest_path = import_actual_closes(
        args.input_dir,
        args.constituents,
        args.output,
        manifest,
        required_tickers=(set(args.required_tickers) if args.required_tickers else None),
    )
    print(f"normalized actual closes: {output_path}")
    print(f"provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
