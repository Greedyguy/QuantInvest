"""Strict point-in-time contract for historical Korean equity fundamentals.

The module deliberately does not download data.  It validates normalized KRX
snapshots and exposes only records that were available by a requested signal
date, so a backtest cannot silently use future or subsequently loaded values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "snapshot_date",
    "available_date",
    "ticker",
    "market",
    "name",
    "close",
    "market_cap",
    "trading_value",
    "bps",
    "eps",
    "per",
    "pbr",
    "dps",
    "dividend_yield",
    "source",
)

NUMERIC_COLUMNS = (
    "close",
    "market_cap",
    "trading_value",
    "bps",
    "eps",
    "per",
    "pbr",
    "dps",
    "dividend_yield",
)

APPROVED_SOURCE_PREFIXES = ("KRX Data Marketplace", "KRX Open API")

KRX_COLUMN_ALIASES = {
    "ticker": ("ticker", "티커", "종목코드", "단축코드", "ISU_SRT_CD"),
    "market": ("market", "시장구분", "시장", "MKT_NM"),
    "name": ("name", "종목명", "한글 종목약명", "ISU_ABBRV"),
    "close": ("close", "종가", "TDD_CLSPRC"),
    "market_cap": ("market_cap", "시가총액", "MKTCAP"),
    "trading_value": ("trading_value", "거래대금", "ACC_TRDVAL"),
    "bps": ("bps", "BPS"),
    "eps": ("eps", "EPS"),
    "per": ("per", "PER"),
    "pbr": ("pbr", "PBR"),
    "dps": ("dps", "DPS"),
    "dividend_yield": ("dividend_yield", "배당수익률", "DIV", "DVD_YLD"),
}


class PointInTimeDataError(ValueError):
    """Raised when a snapshot could introduce leakage or ambiguity."""


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"ticker": "string"})
    raise PointInTimeDataError(f"unsupported snapshot format: {path.suffix}")


def _series_from_alias(frame: pd.DataFrame, field: str) -> pd.Series:
    for alias in KRX_COLUMN_ALIASES[field]:
        if alias in frame.columns:
            return frame[alias]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _numbers(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype("string").str.replace(",", "", regex=False).replace("-", pd.NA),
        errors="coerce",
    )


def normalize_krx_snapshot(
    fundamentals: pd.DataFrame,
    trading: pd.DataFrame,
    *,
    snapshot_date,
    available_date,
    source: str = "KRX Data Marketplace",
) -> pd.DataFrame:
    """Merge the two official KRX tables required for one monthly snapshot.

    ``available_date`` must be the first session on which the downloaded
    after-close snapshot could have been used by the strategy.
    """
    left = pd.DataFrame(
        {
            field: _series_from_alias(fundamentals, field)
            for field in ("ticker", "market", "name", "bps", "eps", "per", "pbr", "dps", "dividend_yield")
        }
    )
    right = pd.DataFrame(
        {
            field: _series_from_alias(trading, field)
            for field in ("ticker", "market", "name", "close", "market_cap", "trading_value")
        }
    )
    left["ticker"] = left["ticker"].astype("string").str.strip().str.zfill(6)
    right["ticker"] = right["ticker"].astype("string").str.strip().str.zfill(6)
    merged = left.merge(right, on="ticker", how="inner", suffixes=("_fundamental", "_trading"))
    merged["market"] = merged["market_trading"].fillna(merged["market_fundamental"])
    merged["name"] = merged["name_trading"].fillna(merged["name_fundamental"])
    merged = merged.drop(
        columns=["market_fundamental", "market_trading", "name_fundamental", "name_trading"]
    )
    merged["snapshot_date"] = pd.Timestamp(snapshot_date).normalize()
    merged["available_date"] = pd.Timestamp(available_date).normalize()
    merged["source"] = str(source)
    return validate_point_in_time_fundamentals(merged)


def validate_point_in_time_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a snapshot collection and fail closed on leakage hazards."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise PointInTimeDataError(f"missing required columns: {missing}")
    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    for field in ("snapshot_date", "available_date"):
        data[field] = pd.to_datetime(data[field], errors="coerce").dt.normalize()
    if data[["snapshot_date", "available_date"]].isna().any().any():
        raise PointInTimeDataError("snapshot_date and available_date must be valid dates")
    if data["available_date"].le(data["snapshot_date"]).any():
        raise PointInTimeDataError(
            "available_date must be after the after-close snapshot_date"
        )

    data["ticker"] = data["ticker"].astype("string").str.strip().str.zfill(6)
    if not data["ticker"].str.fullmatch(r"[0-9A-Z]{6}").all():
        raise PointInTimeDataError("ticker must be a six-character KRX short code")
    if data.duplicated(["snapshot_date", "available_date", "ticker"]).any():
        raise PointInTimeDataError("duplicate snapshot/availability/ticker rows")

    for field in NUMERIC_COLUMNS:
        data[field] = _numbers(data[field])
    if data[["close", "market_cap", "trading_value"]].isna().any().any():
        raise PointInTimeDataError("price, market cap, and trading value are required")
    if data["close"].le(0).any() or data["market_cap"].le(0).any():
        raise PointInTimeDataError("price and market cap must be positive")
    if data["trading_value"].lt(0).any():
        raise PointInTimeDataError("trading value cannot be negative")
    sources = data["source"].astype("string").str.strip()
    if not sources.str.startswith(APPROVED_SOURCE_PREFIXES).all():
        raise PointInTimeDataError("source must be an approved official KRX input")
    return data.sort_values(["available_date", "snapshot_date", "ticker"]).reset_index(drop=True)


def load_point_in_time_fundamentals(path: str | Path) -> pd.DataFrame:
    """Load and validate a normalized CSV or Parquet snapshot collection."""
    return validate_point_in_time_fundamentals(_read_table(path))


def fundamentals_asof(
    frame: pd.DataFrame,
    signal_date,
    *,
    max_age_days: int = 45,
) -> pd.DataFrame:
    """Return the latest per-ticker records legally available at signal time."""
    data = validate_point_in_time_fundamentals(frame)
    signal = pd.Timestamp(signal_date).normalize()
    eligible = data.loc[
        data["available_date"].le(signal) & data["snapshot_date"].le(signal)
    ].copy()
    if eligible.empty:
        return eligible.set_index("ticker")
    eligible = eligible.sort_values(["ticker", "snapshot_date", "available_date"])
    eligible = eligible.groupby("ticker", as_index=False).tail(1)
    age = (signal - eligible["snapshot_date"]).dt.days
    eligible = eligible.loc[age.le(int(max_age_days))].copy()
    return eligible.set_index("ticker").sort_index()
