"""Point-in-time KODEX 200 constituent snapshots for leakage-safe research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"as_of_date", "ticker", "name", "weight_pct"}


class PointInTimeConstituentError(ValueError):
    """Raised when a constituent snapshot cannot be used without leakage."""


def validate_point_in_time_constituents(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize historical constituent observations."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise PointInTimeConstituentError(
            f"constituent data is missing columns: {sorted(missing)}"
        )
    result = frame.copy()
    result["as_of_date"] = pd.to_datetime(result["as_of_date"], errors="raise")
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["name"] = result["name"].astype(str).str.strip()
    result["weight_pct"] = pd.to_numeric(result["weight_pct"], errors="raise")
    if not result["ticker"].str.fullmatch(r"\d{6}").all():
        raise PointInTimeConstituentError("tickers must be six decimal digits")
    if result["name"].eq("").any():
        raise PointInTimeConstituentError("constituent names cannot be empty")
    if result["weight_pct"].le(0.0).any():
        raise PointInTimeConstituentError("constituent weights must be positive")
    if result.duplicated(["as_of_date", "ticker"]).any():
        raise PointInTimeConstituentError(
            "duplicate ticker in the same constituent snapshot"
        )
    return result.sort_values(
        ["as_of_date", "weight_pct", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def load_point_in_time_constituents(path: str | Path) -> pd.DataFrame:
    """Load a normalized constituent CSV without projecting members backward."""

    return validate_point_in_time_constituents(
        pd.read_csv(path, dtype={"ticker": str})
    )


def constituents_asof(
    frame: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    max_age_days: int = 100,
) -> pd.DataFrame:
    """Return only the latest snapshot observable by ``signal_date``.

    The snapshot is assumed to be read after its market close. Orders based on
    it must therefore execute no earlier than the next trading session.
    """

    data = validate_point_in_time_constituents(frame)
    signal = pd.Timestamp(signal_date)
    observable = data.loc[data["as_of_date"].le(signal)]
    if observable.empty:
        return observable.set_index("ticker")
    latest = observable["as_of_date"].max()
    if (signal - latest).days > int(max_age_days):
        return observable.iloc[0:0].set_index("ticker")
    return observable.loc[observable["as_of_date"].eq(latest)].set_index("ticker")
