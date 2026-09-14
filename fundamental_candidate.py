"""Pre-registered DART quality/value candidate scoring rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dart_point_in_time import annual_fundamentals_asof
from point_in_time_constituents import constituents_asof


QUALITY_FIELDS = (
    "roe",
    "operating_return_on_assets",
    "cash_return_on_assets",
    "cash_accrual_quality",
    "equity_to_assets",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    result = numerator / denominator.where(denominator.gt(0))
    return result.replace([np.inf, -np.inf], np.nan)


def _asof_close(prices: dict[str, pd.DataFrame], ticker: str, signal: pd.Timestamp) -> pd.Series:
    frame = prices.get(ticker)
    if frame is None or frame.empty or "close" not in frame.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce")
    close.index = pd.to_datetime(close.index)
    return close.sort_index().loc[lambda series: series.index <= signal].dropna()


def compute_dart_quality_value_scores(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    *,
    max_fundamental_age_days: int = 550,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build one leakage-safe quarterly cross-section using fixed v1 rules."""

    signal = pd.Timestamp(signal_date).normalize()
    members = constituents_asof(constituents, signal, max_age_days=100)
    filings = annual_fundamentals_asof(
        fundamentals, signal, max_age_days=max_fundamental_age_days
    )
    rows: list[dict[str, object]] = []
    for ticker, member in members.iterrows():
        if ticker not in filings.index:
            continue
        filing = filings.loc[ticker]
        close = _asof_close(prices, ticker, signal)
        if len(close) < 252:
            continue
        current_close = float(close.iloc[-1])
        average_assets = np.nanmean(
            [float(filing["assets"]), float(filing.get("previous_assets", np.nan))]
        )
        average_equity = np.nanmean(
            [float(filing["equity"]), float(filing.get("previous_equity", np.nan))]
        )
        market_cap = float(filing["ordinary_issued_shares"]) * current_close
        earnings_yield = float(filing["net_income"]) / market_cap if market_cap > 0 else np.nan
        book_to_market = float(filing["equity"]) / market_cap if market_cap > 0 else np.nan
        momentum_12_1 = float(close.iloc[-21] / close.iloc[-252] - 1.0)
        ma200 = float(close.tail(200).mean())
        rows.append(
            {
                "ticker": ticker,
                "name": member["name"],
                "index_weight_pct": float(member["weight_pct"]),
                "period_end": filing["period_end"],
                "available_date": filing["available_date"],
                "receipt_no": filing["receipt_no"],
                "close": current_close,
                "ma200": ma200,
                "momentum_12_1": momentum_12_1,
                "roe": float(filing["net_income"]) / average_equity,
                "operating_return_on_assets": float(filing["operating_income"]) / average_assets,
                "cash_return_on_assets": float(filing["cash_flow_from_operations"]) / average_assets,
                "cash_accrual_quality": (
                    float(filing["cash_flow_from_operations"])
                    - float(filing["net_income"])
                )
                / average_assets,
                "equity_to_assets": float(filing["equity"]) / float(filing["assets"]),
                "market_cap_proxy": market_cap,
                "earnings_yield": earnings_yield,
                "book_to_market": book_to_market,
            }
        )
    scored = pd.DataFrame(rows)
    coverage = {
        "constituent_members": float(len(members)),
        "fundamental_members": float(len(scored)),
        "index_weight_coverage_pct": float(scored["index_weight_pct"].sum())
        if not scored.empty
        else 0.0,
    }
    if scored.empty:
        return scored, coverage

    scored["quality_metric_count"] = scored.loc[:, QUALITY_FIELDS].notna().sum(axis=1)
    quality_ranks = scored.loc[:, QUALITY_FIELDS].rank(pct=True, method="average")
    scored["quality_score"] = quality_ranks.mean(axis=1, skipna=True)

    scored["value_plausible"] = (
        scored["earnings_yield"].between(-0.5, 0.5, inclusive="both")
        & scored["book_to_market"].gt(0.0)
        & scored["book_to_market"].le(3.0)
    )
    value_metrics = scored[["earnings_yield", "book_to_market"]].where(
        scored["value_plausible"], np.nan
    )
    scored["value_score"] = value_metrics.rank(pct=True, method="average").mean(
        axis=1, skipna=False
    )
    scored["momentum_score"] = scored["momentum_12_1"].rank(
        pct=True, method="average"
    )
    scored["eligible"] = (
        scored["quality_metric_count"].ge(3)
        & scored["value_plausible"]
        & scored["momentum_12_1"].gt(0.0)
        & scored["close"].gt(scored["ma200"])
    )
    scored["composite_score"] = (
        0.45 * scored["quality_score"]
        + 0.35 * scored["value_score"]
        + 0.20 * scored["momentum_score"]
    )
    scored = scored.sort_values(
        ["eligible", "composite_score", "ticker"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return scored, coverage


def select_dart_quality_value_satellite(
    scores: pd.DataFrame,
    coverage: dict[str, float],
    *,
    top_n: int = 4,
    minimum_members: int = 18,
    minimum_weight_coverage_pct: float = 55.0,
) -> list[str]:
    """Fail closed when the pre-registered data-coverage gate is not met."""

    if coverage.get("fundamental_members", 0.0) < minimum_members:
        return []
    if coverage.get("index_weight_coverage_pct", 0.0) < minimum_weight_coverage_pct:
        return []
    if scores.empty:
        return []
    eligible = scores.loc[scores["eligible"]]
    if len(eligible) < top_n:
        return []
    return eligible.head(top_n)["ticker"].tolist()
