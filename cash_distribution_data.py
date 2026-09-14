"""Auditable cash-distribution inputs for dividend-inclusive backtests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_EVENT_COLUMNS = (
    "ticker",
    "record_date",
    "pay_date",
    "distribution_per_share",
    "taxable_per_share",
    "source",
)
APPROVED_SOURCE_PREFIXES = (
    "KRX Data Marketplace",
    "KRX Open API",
    "KSD SEIBro",
    "DART",
    "Samsung Asset Management",
    "official_pdf",
    "official_api",
)


class CashDistributionDataError(ValueError):
    """Raised when dividend events or their completeness proof are unsafe."""


def validate_distribution_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize official per-share cash events without inventing missing rows."""

    missing = sorted(set(REQUIRED_EVENT_COLUMNS) - set(frame.columns))
    if missing:
        raise CashDistributionDataError(f"missing distribution columns: {missing}")
    result = frame.loc[:, REQUIRED_EVENT_COLUMNS].copy()
    for field in ("record_date", "pay_date"):
        result[field] = pd.to_datetime(result[field], errors="coerce").dt.normalize()
    if result[["record_date", "pay_date"]].isna().any().any():
        raise CashDistributionDataError("distribution dates must be valid")
    if result["pay_date"].lt(result["record_date"]).any():
        raise CashDistributionDataError("pay_date cannot precede record_date")
    result["ticker"] = result["ticker"].astype("string").str.strip().str.zfill(6)
    if not result["ticker"].str.fullmatch(r"[0-9A-Z]{6}").all():
        raise CashDistributionDataError("distribution ticker must be a KRX code")
    for field in ("distribution_per_share", "taxable_per_share"):
        result[field] = pd.to_numeric(result[field], errors="coerce")
    if result[["distribution_per_share", "taxable_per_share"]].isna().any().any():
        raise CashDistributionDataError("distribution amounts must be numeric")
    if result["distribution_per_share"].le(0.0).any():
        raise CashDistributionDataError("cash distributions must be positive")
    if result["taxable_per_share"].lt(0.0).any():
        raise CashDistributionDataError("taxable distribution cannot be negative")
    sources = result["source"].astype("string").str.strip()
    if not sources.str.startswith(APPROVED_SOURCE_PREFIXES).all():
        raise CashDistributionDataError("distribution source is not approved")
    if result.duplicated(["ticker", "record_date"]).any():
        raise CashDistributionDataError("duplicate ticker/record-date distribution")
    return result.sort_values(["record_date", "ticker"]).reset_index(drop=True)


def load_distribution_events(path: str | Path) -> pd.DataFrame:
    """Load a normalized multi-ticker event CSV."""

    return validate_distribution_events(
        pd.read_csv(path, dtype={"ticker": "string"})
    )


def validate_distribution_coverage_manifest(
    manifest: dict,
    *,
    required_tickers: set[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> None:
    """Require explicit proof that zero-event as well as paying names were checked."""

    required = {
        "history_coverage_start",
        "history_coverage_end",
        "complete_cash_distribution_history",
        "audited_tickers",
        "source",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise CashDistributionDataError(
            f"distribution coverage manifest is missing: {missing}"
        )
    if manifest["complete_cash_distribution_history"] is not True:
        raise CashDistributionDataError("distribution history must be explicitly complete")
    source = str(manifest["source"]).strip()
    if not source.startswith(APPROVED_SOURCE_PREFIXES):
        raise CashDistributionDataError("coverage source is not approved")
    coverage_start = pd.Timestamp(manifest["history_coverage_start"]).normalize()
    coverage_end = pd.Timestamp(manifest["history_coverage_end"]).normalize()
    if coverage_start > pd.Timestamp(start_date).normalize():
        raise CashDistributionDataError("distribution history starts after the test")
    if coverage_end < pd.Timestamp(end_date).normalize():
        raise CashDistributionDataError("distribution history ends before the test")
    audited = {str(ticker).zfill(6) for ticker in manifest["audited_tickers"]}
    missing_tickers = sorted({str(ticker).zfill(6) for ticker in required_tickers} - audited)
    if missing_tickers:
        raise CashDistributionDataError(
            f"distribution coverage missing tickers: {missing_tickers}"
        )


def load_distribution_bundle(
    events_path: str | Path,
    manifest_path: str | Path,
    *,
    required_tickers: set[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Load events only after their full-period/ticker coverage is proven."""

    events = load_distribution_events(events_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_distribution_coverage_manifest(
        manifest,
        required_tickers=required_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    audited = {str(ticker).zfill(6) for ticker in manifest["audited_tickers"]}
    unexpected = sorted(set(events["ticker"]) - audited)
    if unexpected:
        raise CashDistributionDataError(
            f"events contain unaudited tickers: {unexpected}"
        )
    return {
        str(ticker): group.drop(columns="ticker").reset_index(drop=True)
        for ticker, group in events.groupby("ticker", sort=True)
    }
