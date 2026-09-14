"""Strict loaders for actual KRX price levels used by integer-share tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_benchmark import restore_actual_ohlc


REQUIRED_ACTUAL_CLOSE_COLUMNS = (
    "date",
    "ticker",
    "close",
    "source",
    "price_basis",
)
APPROVED_PRICE_SOURCE_PREFIXES = ("KRX Data Marketplace", "KRX Open API")
DATE_ALIASES = ("date", "일자", "거래일자", "기준일", "TRD_DD")
TICKER_ALIASES = ("ticker", "티커", "종목코드", "단축코드", "ISU_SRT_CD")
CLOSE_ALIASES = ("close", "종가", "TDD_CLSPRC")


class KrxExecutionDataError(ValueError):
    """Raised when price levels are not safe for integer-share execution."""


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for alias in aliases:
        if alias in frame:
            return frame[alias]
    raise KrxExecutionDataError(f"none of the required columns exist: {aliases}")


def validate_actual_close_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an official, unadjusted KRX daily-close collection."""

    missing = sorted(set(REQUIRED_ACTUAL_CLOSE_COLUMNS) - set(frame.columns))
    if missing:
        raise KrxExecutionDataError(f"missing actual-close columns: {missing}")
    result = frame.loc[:, REQUIRED_ACTUAL_CLOSE_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise KrxExecutionDataError("actual-close dates must be valid")
    result["ticker"] = result["ticker"].astype("string").str.strip().str.zfill(6)
    if not result["ticker"].str.fullmatch(r"[0-9A-Z]{6}").all():
        raise KrxExecutionDataError("ticker must be a six-character KRX short code")
    result["close"] = pd.to_numeric(
        result["close"].astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    )
    if result["close"].isna().any() or result["close"].le(0.0).any():
        raise KrxExecutionDataError("actual closes must be positive numbers")
    if not result["price_basis"].astype("string").eq("actual_traded").all():
        raise KrxExecutionDataError(
            "price_basis must be actual_traded; adjusted levels are forbidden"
        )
    sources = result["source"].astype("string").str.strip()
    if not sources.str.startswith(APPROVED_PRICE_SOURCE_PREFIXES).all():
        raise KrxExecutionDataError("price source must be an approved official KRX input")
    if result.duplicated(["date", "ticker"]).any():
        raise KrxExecutionDataError("duplicate date/ticker actual-close rows")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def normalize_krx_actual_close(
    raw: pd.DataFrame,
    *,
    ticker: str | None = None,
    source: str = "KRX Data Marketplace",
) -> pd.DataFrame:
    """Normalize one raw KRX daily-price download without changing its scale."""

    tickers = (
        pd.Series(str(ticker), index=raw.index)
        if ticker is not None
        else _column(raw, TICKER_ALIASES)
    )
    normalized = pd.DataFrame(
        {
            "date": _column(raw, DATE_ALIASES),
            "ticker": tickers,
            "close": _column(raw, CLOSE_ALIASES),
            "source": source,
            "price_basis": "actual_traded",
        }
    )
    return validate_actual_close_panel(normalized)


def load_actual_close_panel(path: str | Path) -> pd.DataFrame:
    """Load a normalized CSV or Parquet actual-close panel."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"ticker": "string"})
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise KrxExecutionDataError(f"unsupported actual-close format: {path.suffix}")
    return validate_actual_close_panel(frame)


def restore_actual_price_panel(
    adjusted_prices: dict[str, pd.DataFrame],
    actual_closes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Put adjusted OHLC histories onto official daily traded-price scales."""

    panel = validate_actual_close_panel(actual_closes)
    result: dict[str, pd.DataFrame] = {}
    for ticker, adjusted in adjusted_prices.items():
        official = panel.loc[panel["ticker"].eq(str(ticker))]
        if official.empty:
            raise KrxExecutionDataError(f"official actual closes missing for {ticker}")
        closes = official.set_index("date")["close"]
        try:
            result[str(ticker)] = restore_actual_ohlc(adjusted, closes)
        except ValueError as error:
            raise KrxExecutionDataError(
                f"could not restore actual prices for {ticker}: {error}"
            ) from error
    return result
