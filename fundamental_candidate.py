"""Pre-registered DART quality/value candidate scoring rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dart_point_in_time import annual_fundamentals_asof
from point_in_time_constituents import constituents_asof
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


QUALITY_FIELDS = (
    "roe",
    "operating_return_on_assets",
    "cash_return_on_assets",
    "cash_accrual_quality",
    "equity_to_assets",
)

KOSPI_SELL_TAX_SCHEDULE = (
    (pd.Timestamp("1900-01-01"), 0.0030),
    (pd.Timestamp("2019-06-03"), 0.0025),
    (pd.Timestamp("2021-01-01"), 0.0023),
    (pd.Timestamp("2023-01-01"), 0.0020),
    (pd.Timestamp("2024-01-01"), 0.0018),
    (pd.Timestamp("2025-01-01"), 0.0015),
)


def historical_kospi_sell_tax_rate(ticker: str, execution_date) -> float:
    """Return total KOSPI sell tax, including the 0.15% rural surtax.

    KODEX 200 and other domestic equity ETFs are exempt from transaction tax.
    The individual-stock schedule follows the effective-date changes used in
    the pre-registered development simulation.
    """

    if str(ticker) == "069500":
        return 0.0
    date = pd.Timestamp(execution_date).normalize()
    rate = KOSPI_SELL_TAX_SCHEDULE[0][1]
    for effective_date, candidate in KOSPI_SELL_TAX_SCHEDULE:
        if date < effective_date:
            break
        rate = candidate
    return rate


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


def _actual_to_adjusted_scale(
    signal_prices: dict[str, pd.DataFrame],
    actual_prices: dict[str, pd.DataFrame],
    ticker: str,
    date: pd.Timestamp,
) -> float:
    """Return the latest same-session actual/adjusted corporate-action scale."""

    signal_close = _asof_close(signal_prices, ticker, date).rename("signal")
    actual_close = _asof_close(actual_prices, ticker, date).rename("actual")
    aligned = pd.concat([signal_close, actual_close], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan
    scale = float(aligned.iloc[-1]["actual"] / aligned.iloc[-1]["signal"])
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def compute_dart_quality_value_scores(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    *,
    actual_prices: dict[str, pd.DataFrame] | None = None,
    max_fundamental_age_days: int = 550,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build one leakage-safe quarterly cross-section using fixed v1 rules."""

    signal = pd.Timestamp(signal_date).normalize()
    members = constituents_asof(constituents, signal, max_age_days=100)
    filings = annual_fundamentals_asof(
        fundamentals, signal, max_age_days=max_fundamental_age_days
    )
    execution_prices = prices if actual_prices is None else actual_prices
    rows: list[dict[str, object]] = []
    for ticker, member in members.iterrows():
        if ticker not in filings.index:
            continue
        filing = filings.loc[ticker]
        signal_close = _asof_close(prices, ticker, signal)
        actual_close = _asof_close(execution_prices, ticker, signal)
        if len(signal_close) < 252 or actual_close.empty:
            continue
        current_signal_close = float(signal_close.iloc[-1])
        current_actual_close = float(actual_close.iloc[-1])
        average_assets = np.nanmean(
            [float(filing["assets"]), float(filing.get("previous_assets", np.nan))]
        )
        average_equity = np.nanmean(
            [float(filing["equity"]), float(filing.get("previous_equity", np.nan))]
        )
        period_scale = _actual_to_adjusted_scale(
            prices,
            execution_prices,
            ticker,
            pd.Timestamp(filing["period_end"]),
        )
        signal_scale = _actual_to_adjusted_scale(
            prices, execution_prices, ticker, signal
        )
        if not np.isfinite(period_scale) or not np.isfinite(signal_scale):
            continue
        split_adjusted_shares = (
            float(filing["ordinary_issued_shares"]) * period_scale / signal_scale
        )
        market_cap = split_adjusted_shares * current_actual_close
        earnings_yield = float(filing["net_income"]) / market_cap if market_cap > 0 else np.nan
        book_to_market = float(filing["equity"]) / market_cap if market_cap > 0 else np.nan
        momentum_12_1 = float(
            signal_close.iloc[-21] / signal_close.iloc[-252] - 1.0
        )
        ma200 = float(signal_close.tail(200).mean())
        rows.append(
            {
                "ticker": ticker,
                "name": member["name"],
                "index_weight_pct": float(member["weight_pct"]),
                "period_end": filing["period_end"],
                "available_date": filing["available_date"],
                "receipt_no": filing["receipt_no"],
                "close": current_actual_close,
                "signal_close": current_signal_close,
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
                "split_adjusted_issued_shares": split_adjusted_shares,
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
        & scored["signal_close"].gt(scored["ma200"])
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


def build_dart_quality_value_targets(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    actual_prices: dict[str, pd.DataFrame] | None = None,
    core_ticker: str = "069500",
    core_weight: float = 0.40,
    satellite_weight: float = 0.55,
    cash_weight: float = 0.05,
    top_n: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create daily-held targets for next-open execution and audit decisions."""

    if not np.isclose(core_weight + satellite_weight + cash_weight, 1.0):
        raise ValueError("core, satellite, and cash weights must sum to one")
    core = prices.get(core_ticker)
    if core is None or core.empty:
        return pd.DataFrame(), pd.DataFrame()
    state_model = K200LowTurnoverReentry(ticker=core_ticker)
    states = state_model.compute_state_history(core)
    if states.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = states.index
    snapshots = set(pd.to_datetime(constituents["as_of_date"]).dt.normalize())
    current = {"__CASH__": 1.0}
    current_risk_on = False
    target_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for position, signal_date in enumerate(dates):
        if position + 1 < len(dates):
            execution_date = dates[position + 1]
            risk_on = states.loc[execution_date, "state"] == state_model.RISK_ON
            scheduled = signal_date.normalize() in snapshots
            regime_changed = risk_on != current_risk_on
            if scheduled or regime_changed:
                selected: list[str] = []
                scores = pd.DataFrame()
                coverage = {
                    "constituent_members": 0.0,
                    "fundamental_members": 0.0,
                    "index_weight_coverage_pct": 0.0,
                }
                fallback = "risk_off_cash"
                if risk_on:
                    scores, coverage = compute_dart_quality_value_scores(
                        constituents,
                        fundamentals,
                        prices,
                        signal_date,
                        actual_prices=actual_prices,
                    )
                    selected = select_dart_quality_value_satellite(
                        scores, coverage, top_n=top_n
                    )
                    if selected:
                        current = {core_ticker: core_weight}
                        per_name = satellite_weight / len(selected)
                        current.update({ticker: per_name for ticker in selected})
                        current["__CASH__"] = cash_weight
                        fallback = "none"
                    else:
                        current = {
                            core_ticker: core_weight + satellite_weight,
                            "__CASH__": cash_weight,
                        }
                        fallback = "same_timing_kodex200"
                else:
                    current = {"__CASH__": 1.0}
                score_map = (
                    scores.set_index("ticker")["composite_score"].to_dict()
                    if not scores.empty
                    else {}
                )
                decisions.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "risk_on": risk_on,
                        "reason": (
                            "regime_change" if regime_changed else "quarterly_snapshot"
                        ),
                        "selected": ",".join(selected),
                        "selected_scores": ",".join(
                            f"{ticker}:{score_map[ticker]:.6f}" for ticker in selected
                        ),
                        "fallback": fallback,
                        **coverage,
                    }
                )
                current_risk_on = risk_on
        target_rows.append({"date": signal_date, **current})

    targets = pd.DataFrame(target_rows).set_index("date").fillna(0.0)
    targets = targets.reindex(columns=sorted(targets.columns)).fillna(0.0)
    return targets, pd.DataFrame(decisions)
