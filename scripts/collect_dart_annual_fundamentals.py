#!/usr/bin/env python3
"""Collect development-only annual fundamentals from public DART filings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dart_point_in_time import (
    DART_BASE_URL,
    DartPointInTimeError,
    normalize_annual_filing,
    parse_company_search,
    parse_report_sections,
    select_report_section,
)


DEVELOPMENT_END = pd.Timestamp("2022-12-31")
DEFAULT_CONSTITUENTS = (
    PROJECT_ROOT / "data" / "reference" / "kodex200_top30_pcf_quarterly_2018_2025.csv"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class PublicDartClient:
    def __init__(self, cache_dir: Path, pause_seconds: float = 0.20):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause_seconds = max(float(pause_seconds), 0.0)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; point-in-time-research/1.0; "
                    "+https://dart.fss.or.kr/)"
                )
            }
        )

    def _request(
        self,
        cache_name: str,
        url: str,
        *,
        data: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            payload = cache_path.read_bytes()
            return payload.decode("utf-8"), _sha256_bytes(payload)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.post(url, data=data, timeout=40) if data else self.session.get(url, timeout=40)
                response.raise_for_status()
                response.encoding = "utf-8"
                payload = response.content
                cache_path.write_bytes(payload)
                if self.pause_seconds:
                    time.sleep(self.pause_seconds)
                return payload.decode("utf-8"), _sha256_bytes(payload)
            except (requests.RequestException, UnicodeDecodeError) as error:
                last_error = error
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"DART request failed after retries: {url}") from last_error

    def search_annual_reports(
        self, ticker: str, start_date: str, end_date: str
    ) -> tuple[list, str]:
        url = f"{DART_BASE_URL}/dsab001/searchCorp.ax"
        data = {
            "currentPage": "1",
            "maxResults": "100",
            "maxLinks": "10",
            "sort": "date",
            "series": "desc",
            "pageGubun": "corp",
            "textCrpNm": ticker,
            "startDate": start_date.replace("-", ""),
            "endDate": end_date.replace("-", ""),
            # Deliberately omit finalReport=recent. Original receipts, not the
            # latest corrected view, define point-in-time availability.
            "publicType": "A001",
        }
        html, digest = self._request(
            f"{ticker}_search_{data['startDate']}_{data['endDate']}.html",
            url,
            data=data,
        )
        return parse_company_search(html, ticker), digest

    def get(self, cache_name: str, url: str) -> tuple[str, str]:
        return self._request(cache_name, url)


def collect(
    constituents_path: Path,
    *,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    pause_seconds: float,
) -> tuple[pd.DataFrame, list[dict], dict]:
    end = pd.Timestamp(end_date)
    if end > DEVELOPMENT_END:
        raise ValueError(
            "collector refuses to open 2023+ filings before the candidate passes development"
        )
    constituents = pd.read_csv(constituents_path, dtype={"ticker": str})
    constituents["as_of_date"] = pd.to_datetime(constituents["as_of_date"])
    development = constituents.loc[constituents["as_of_date"].le(DEVELOPMENT_END)]
    tickers = sorted(development["ticker"].astype(str).str.zfill(6).unique())
    client = PublicDartClient(cache_dir, pause_seconds=pause_seconds)
    rows: list[dict] = []
    errors: list[dict] = []
    raw_hashes: dict[str, str] = {}

    for position, ticker in enumerate(tickers, start=1):
        print(f"[{position:02d}/{len(tickers):02d}] {ticker}", flush=True)
        try:
            reports, digest = client.search_annual_reports(ticker, start_date, end_date)
            raw_hashes[f"{ticker}:search"] = digest
        except Exception as error:  # continue to quantify coverage
            errors.append({"ticker": ticker, "stage": "search", "error": str(error)})
            continue
        for report in reports:
            try:
                main_html, main_hash = client.get(
                    f"{ticker}_{report.receipt_no}_main.html", report.main_url
                )
                sections = parse_report_sections(main_html)
                statements = select_report_section(
                    sections, "consolidated_financial_statements"
                )
                shares = select_report_section(sections, "issued_shares")
                statement_html, statement_hash = client.get(
                    f"{ticker}_{report.receipt_no}_statements.html",
                    statements.viewer_url,
                )
                shares_html, shares_hash = client.get(
                    f"{ticker}_{report.receipt_no}_shares.html", shares.viewer_url
                )
                row = normalize_annual_filing(
                    report,
                    statement_html,
                    shares_html,
                    statement_url=statements.viewer_url,
                    shares_url=shares.viewer_url,
                )
                rows.append(row)
                raw_hashes[f"{ticker}:{report.receipt_no}:main"] = main_hash
                raw_hashes[f"{ticker}:{report.receipt_no}:statements"] = statement_hash
                raw_hashes[f"{ticker}:{report.receipt_no}:shares"] = shares_hash
            except Exception as error:  # continue to quantify parser limitations
                errors.append(
                    {
                        "ticker": ticker,
                        "receipt_no": report.receipt_no,
                        "period_end": report.period_end,
                        "stage": "normalize",
                        "error": str(error),
                    }
                )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["available_date", "ticker", "period_end"])
    metadata = {
        "source": "DART public original filings",
        "development_only": True,
        "search_start_date": start_date,
        "search_end_date": end_date,
        "universe_source": str(constituents_path.resolve()),
        "universe_tickers": len(tickers),
        "normalized_rows": len(frame),
        "normalized_tickers": int(frame["ticker"].nunique()) if not frame.empty else 0,
        "errors": len(errors),
        "availability_rule": "receipt_date plus one calendar day",
        "correction_rule": "original annual-report receipt only; finalReport=recent omitted",
        "raw_sha256": raw_hashes,
    }
    return frame, errors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constituents", type=Path, default=DEFAULT_CONSTITUENTS)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2022-12-31")
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/dart_pit_cache"))
    parser.add_argument("--pause-seconds", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    frame, errors, metadata = collect(
        args.constituents,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        pause_seconds=args.pause_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    errors_output = args.errors_output or args.output.with_suffix(".errors.json")
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")
    errors_output.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"normalized: {args.output} ({len(frame)} rows)")
    print(f"errors: {errors_output} ({len(errors)} rows)")
    print(f"manifest: {manifest_output}")


if __name__ == "__main__":
    main()
