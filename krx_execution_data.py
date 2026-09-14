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
OPEN_ALIASES = ("open", "시가", "TDD_OPNPRC")
HIGH_ALIASES = ("high", "고가", "TDD_HGPRC")
LOW_ALIASES = ("low", "저가", "TDD_LWPRC")
OHLC_COLUMNS = ("open", "high", "low", "close")


class KrxExecutionDataError(ValueError):
    """Raised when price levels are not safe for integer-share execution."""


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for alias in aliases:
        if alias in frame:
            return frame[alias]
    raise KrxExecutionDataError(f"none of the required columns exist: {aliases}")


def _optional_column(
    frame: pd.DataFrame, aliases: tuple[str, ...]
) -> pd.Series | None:
    for alias in aliases:
        if alias in frame:
            return frame[alias]
    return None


def validate_actual_close_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an official, unadjusted KRX daily-close collection."""

    missing = sorted(set(REQUIRED_ACTUAL_CLOSE_COLUMNS) - set(frame.columns))
    if missing:
        raise KrxExecutionDataError(f"missing actual-close columns: {missing}")
    optional_ohlc = [column for column in OHLC_COLUMNS[:-1] if column in frame]
    if optional_ohlc and len(optional_ohlc) != len(OHLC_COLUMNS) - 1:
        raise KrxExecutionDataError(
            "actual OHLC must contain open, high, and low together"
        )
    ordered_columns = ["date", "ticker"]
    if optional_ohlc:
        ordered_columns.extend(OHLC_COLUMNS[:-1])
    ordered_columns.extend(["close", "source", "price_basis"])
    result = frame.loc[:, ordered_columns].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise KrxExecutionDataError("actual-close dates must be valid")
    result["ticker"] = result["ticker"].astype("string").str.strip().str.zfill(6)
    if not result["ticker"].str.fullmatch(r"[0-9A-Z]{6}").all():
        raise KrxExecutionDataError("ticker must be a six-character KRX short code")
    price_columns = OHLC_COLUMNS if optional_ohlc else ("close",)
    for column in price_columns:
        result[column] = pd.to_numeric(
            result[column].astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )
    if result.loc[:, price_columns].isna().any(axis=None):
        raise KrxExecutionDataError("actual prices must be valid numbers")
    if result.loc[:, price_columns].le(0.0).any(axis=None):
        raise KrxExecutionDataError("actual prices must be positive numbers")
    if optional_ohlc:
        if result["high"].lt(result[["open", "low", "close"]].max(axis=1)).any():
            raise KrxExecutionDataError("actual high must cover open, low, and close")
        if result["low"].gt(result[["open", "high", "close"]].min(axis=1)).any():
            raise KrxExecutionDataError("actual low must cover open, high, and close")
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
    open_price = _optional_column(raw, OPEN_ALIASES)
    high = _optional_column(raw, HIGH_ALIASES)
    low = _optional_column(raw, LOW_ALIASES)
    optional_prices = (open_price, high, low)
    if any(value is not None for value in optional_prices) and not all(
        value is not None for value in optional_prices
    ):
        raise KrxExecutionDataError(
            "raw actual OHLC must contain open, high, and low together"
        )
    columns = {
        "date": _column(raw, DATE_ALIASES),
        "ticker": tickers,
        "close": _column(raw, CLOSE_ALIASES),
        "source": source,
        "price_basis": "actual_traded",
    }
    if open_price is not None:
        columns.update({"open": open_price, "high": high, "low": low})
    normalized = pd.DataFrame(columns)
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


def actual_ohlc_for_ticker(
    panel: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """Extract authoritative traded OHLC for one ticker from a strict panel."""

    validated = validate_actual_close_panel(panel)
    missing = sorted(set(OHLC_COLUMNS) - set(validated.columns))
    if missing:
        raise KrxExecutionDataError(f"actual OHLC columns are missing: {missing}")
    selected = validated.loc[validated["ticker"].eq(str(ticker).zfill(6))]
    if selected.empty:
        raise KrxExecutionDataError(f"official actual OHLC missing for {ticker}")
    return selected.set_index("date").loc[:, OHLC_COLUMNS].sort_index()


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


def assert_no_unmodelled_corporate_actions(
    targets: pd.DataFrame,
    signal_prices: dict[str, pd.DataFrame],
    actual_prices: dict[str, pd.DataFrame],
    *,
    material_scale_change: float = 0.02,
) -> None:
    """Fail when a held name crosses a split or similar price-scale change.

    Actual/adjusted close ratios are stable between corporate actions.  The
    basic integer-share simulator does not mutate share counts or create
    spin-off entitlements, so it must not silently cross a material ratio
    change while the strategy is invested in that ticker.
    """

    target_index = pd.DatetimeIndex(pd.to_datetime(targets.index))
    failures: dict[str, list[str]] = {}
    for ticker in sorted(set(targets.columns) - {"__CASH__"}):
        if ticker not in signal_prices or ticker not in actual_prices:
            continue
        signal_close = pd.to_numeric(
            signal_prices[ticker]["close"], errors="coerce"
        ).rename("signal")
        actual_close = pd.to_numeric(
            actual_prices[ticker]["close"], errors="coerce"
        ).rename("actual")
        aligned = pd.concat([signal_close, actual_close], axis=1, join="inner").dropna()
        if aligned.empty:
            continue
        scale = aligned["actual"] / aligned["signal"]
        material = scale.pct_change().abs().gt(float(material_scale_change))
        event_dates = aligned.index[material]
        if event_dates.empty:
            continue
        intended_holdings = (
            pd.to_numeric(targets[ticker], errors="coerce")
            .reindex(target_index)
            .fillna(0.0)
            .shift(1, fill_value=0.0)
            .gt(0.0)
        )
        intended_holdings.index = target_index
        held_events = [
            pd.Timestamp(date)
            for date in event_dates
            if date in intended_holdings.index and bool(intended_holdings.loc[date])
        ]
        if held_events:
            failures[ticker] = [date.date().isoformat() for date in held_events]
    if failures:
        raise KrxExecutionDataError(
            "selected holdings cross unmodelled corporate-action price scales: "
            f"{failures}"
        )
