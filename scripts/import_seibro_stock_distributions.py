#!/usr/bin/env python3
"""Import immutable KSD SEIBro common-stock dividend-history exports."""

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

from cash_distribution_data import validate_distribution_events


SOURCE = "KSD SEIBro official dividend history export"
SOURCE_URL = (
    "https://seibro.or.kr/websquare/control.jsp?"
    "w2xPath=/IPORTAL/user/company/BIP_CNTS01041V.xml&menuNo=278"
)
TICKER_PATTERN = re.compile(r"^(?P<ticker>[0-9A-Z]{6})\.(?:xls|html|jsp)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_exports(input_dir: Path) -> dict[str, Path]:
    """Return one clearly named SEIBro export per KRX ticker."""

    files: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = TICKER_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        ticker = match.group("ticker")
        if ticker in files:
            raise ValueError(f"duplicate SEIBro export for {ticker}")
        files[ticker] = path
    if not files:
        raise ValueError(f"no ticker-named SEIBro exports found in {input_dir}")
    return files


def _find_column(
    frame: pd.DataFrame,
    top: str,
    bottom: str | None = None,
) -> object:
    for column in frame.columns:
        parts = column if isinstance(column, tuple) else (column,)
        normalized = tuple(str(part).strip() for part in parts)
        if normalized[0] != top:
            continue
        if bottom is None or (len(normalized) > 1 and normalized[1] == bottom):
            return column
    label = top if bottom is None else f"{top}/{bottom}"
    raise ValueError(f"SEIBro export is missing column: {label}")


def _read_export(path: Path) -> pd.DataFrame:
    tables = pd.read_html(path, encoding="euc-kr")
    if len(tables) != 1:
        raise ValueError(f"SEIBro export must contain exactly one table: {path.name}")
    return tables[0]


def normalize_export(path: Path, ticker: str) -> pd.DataFrame:
    """Normalize one all-distribution-type common-stock history export."""

    frame = _read_export(path)
    record_column = _find_column(frame, "배정기준일")
    pay_column = _find_column(frame, "현금배당 지급일")
    ticker_column = _find_column(frame, "종목코드")
    stock_kind_column = _find_column(frame, "주식종류")
    distribution_kind_column = _find_column(frame, "배당구분")
    general_amount_column = _find_column(frame, "주당배당금", "일반")
    differential_amount_column = _find_column(frame, "주당배당금", "차등")

    raw_tickers = (
        frame[ticker_column]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
    )
    if set(raw_tickers.dropna()) != {ticker}:
        raise ValueError(f"{path.name} contains a different ticker")
    stock_kinds = frame[stock_kind_column].astype("string").str.strip()
    if not stock_kinds.eq("보통주").all():
        raise ValueError(f"{path.name} is not a common-stock-only export")

    distribution_kinds = frame[distribution_kind_column].astype("string").str.strip()
    has_cash = distribution_kinds.str.contains("현금", na=False)
    cash_rows = frame.loc[has_cash].copy()
    if cash_rows.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "record_date",
                "pay_date",
                "distribution_per_share",
                "taxable_per_share",
                "source",
            ]
        )

    general = pd.to_numeric(cash_rows[general_amount_column], errors="coerce")
    differential = pd.to_numeric(
        cash_rows[differential_amount_column], errors="coerce"
    )
    ambiguous = differential.gt(0.0) & ~differential.eq(general)
    if ambiguous.any():
        raise ValueError(f"{path.name} contains an unsupported differential dividend")
    result = pd.DataFrame(
        {
            "ticker": ticker,
            "record_date": pd.to_datetime(
                cash_rows[record_column].astype("string"),
                format="%Y%m%d",
                errors="coerce",
            ),
            "pay_date": pd.to_datetime(
                cash_rows[pay_column].astype("string"),
                format="%Y%m%d",
                errors="coerce",
            ),
            "distribution_per_share": general,
            "taxable_per_share": general,
            "source": SOURCE,
        }
    )
    return validate_distribution_events(result)


def build_distribution_panel(
    exports: dict[str, Path],
    *,
    required_tickers: set[str],
    history_start: pd.Timestamp,
    history_end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict]]:
    """Normalize all required exports and retain their content hashes."""

    required = {str(ticker).zfill(6) for ticker in required_tickers}
    if not required:
        raise ValueError("required SEIBro ticker set cannot be empty")
    missing = sorted(required - set(exports))
    unexpected = sorted(set(exports) - required)
    if missing:
        raise ValueError(f"missing required SEIBro exports: {missing}")
    if unexpected:
        raise ValueError(f"unexpected SEIBro exports: {unexpected}")

    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for ticker in sorted(required):
        path = exports[ticker]
        normalized = normalize_export(path, ticker)
        if not normalized.empty:
            outside = ~normalized["record_date"].between(history_start, history_end)
            if outside.any():
                raise ValueError(f"{path.name} contains dates outside the declared query")
            frames.append(normalized)
        provenance.append(
            {
                "ticker": ticker,
                "filename": path.name,
                "sha256": _sha256(path),
                "cash_event_rows": int(len(normalized)),
            }
        )
    if not frames:
        raise ValueError("SEIBro exports contain no cash-distribution events")
    panel = validate_distribution_events(pd.concat(frames, ignore_index=True))
    return panel, provenance


def import_distributions(
    input_dir: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    required_tickers: set[str],
    history_start: str | pd.Timestamp,
    history_end: str | pd.Timestamp,
) -> tuple[Path, Path]:
    """Write immutable normalized events and their completeness manifest."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("SEIBro distribution output is immutable and already exists")
    start = pd.Timestamp(history_start).normalize()
    end = pd.Timestamp(history_end).normalize()
    if start > end:
        raise ValueError("distribution history start cannot follow its end")
    required = {str(ticker).zfill(6) for ticker in required_tickers}
    panel, provenance = build_distribution_panel(
        discover_exports(input_dir),
        required_tickers=required,
        history_start=start,
        history_end=end,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.assign(
        record_date=panel["record_date"].dt.strftime("%Y-%m-%d"),
        pay_date=panel["pay_date"].dt.strftime("%Y-%m-%d"),
    ).to_csv(output_path, index=False)
    manifest = {
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "development_only": True,
        "stock_kind": "common_stock",
        "queried_distribution_types": "all",
        "history_coverage_start": start.date().isoformat(),
        "history_coverage_end": end.date().isoformat(),
        "complete_cash_distribution_history": True,
        "audited_tickers": sorted(required),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--required-tickers", nargs="+", required=True)
    parser.add_argument("--history-start", default="2018-01-01")
    parser.add_argument("--history-end", default="2022-12-31")
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    output, manifest_path = import_distributions(
        args.input_dir,
        args.output,
        manifest,
        required_tickers=set(args.required_tickers),
        history_start=args.history_start,
        history_end=args.history_end,
    )
    print(f"normalized stock distributions: {output}")
    print(f"coverage manifest: {manifest_path}")


if __name__ == "__main__":
    main()
