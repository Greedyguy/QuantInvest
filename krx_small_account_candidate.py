"""Pre-registered KRX value/profitability/momentum candidate for a small account."""

from __future__ import annotations

import numpy as np
import pandas as pd

from point_in_time_constituents import constituents_asof
from point_in_time_fundamentals import fundamentals_asof
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


CORE_TICKER = "069500"


def _asof_close(
    prices: dict[str, pd.DataFrame], ticker: str, signal_date: pd.Timestamp
) -> pd.Series:
    frame = prices.get(ticker)
    if frame is None or frame.empty or "close" not in frame:
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce")
    close.index = pd.to_datetime(close.index)
    return close.sort_index().loc[lambda values: values.index <= signal_date].dropna()


def compute_krx_small_account_scores(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    *,
    actual_prices: dict[str, pd.DataFrame] | None = None,
    maximum_share_price: float = 250_000.0,
    max_snapshot_age_days: int = 100,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """Build one leakage-safe cross-section from observable KRX snapshots.

    ``prices`` contains split-adjusted signal histories. ``actual_prices``
    contains traded price levels for the affordability check. Passing the same
    mapping for both remains useful for fixtures without corporate actions.
    """

    signal = pd.Timestamp(signal_date).normalize()
    members = constituents_asof(
        constituents, signal, max_age_days=max_snapshot_age_days
    )
    factors = fundamentals_asof(
        fundamentals, signal, max_age_days=max_snapshot_age_days
    )
    member_snapshot = (
        pd.Timestamp(members["as_of_date"].iloc[0]).normalize()
        if not members.empty
        else pd.NaT
    )
    factor_snapshot = (
        pd.Timestamp(factors["snapshot_date"].max()).normalize()
        if not factors.empty
        else pd.NaT
    )
    snapshots_aligned = bool(
        pd.notna(member_snapshot)
        and pd.notna(factor_snapshot)
        and member_snapshot == factor_snapshot
    )
    rows: list[dict[str, object]] = []
    execution_prices = prices if actual_prices is None else actual_prices
    if snapshots_aligned:
        for ticker, member in members.iterrows():
            if ticker not in factors.index:
                continue
            signal_close = _asof_close(prices, ticker, signal)
            actual_close = _asof_close(execution_prices, ticker, signal)
            if len(signal_close) < 252 or actual_close.empty:
                continue
            factor = factors.loc[ticker]
            current_signal_close = float(signal_close.iloc[-1])
            current_actual_close = float(actual_close.iloc[-1])
            eps = float(factor["eps"])
            bps = float(factor["bps"])
            per = float(factor["per"])
            pbr = float(factor["pbr"])
            earnings_yield = 1.0 / per if np.isfinite(per) and per > 0 else np.nan
            book_to_market = 1.0 / pbr if np.isfinite(pbr) and pbr > 0 else np.nan
            implied_roe = eps / bps if np.isfinite(eps) and bps > 0 else np.nan
            momentum_12_1 = float(
                signal_close.iloc[-21] / signal_close.iloc[-252] - 1.0
            )
            rows.append(
                {
                    "ticker": ticker,
                    "name": member["name"],
                    "index_weight_pct": float(member["weight_pct"]),
                    "snapshot_date": factor["snapshot_date"],
                    "available_date": factor["available_date"],
                    "close": current_actual_close,
                    "signal_close": current_signal_close,
                    "ma200": float(signal_close.tail(200).mean()),
                    "momentum_12_1": momentum_12_1,
                    "eps": eps,
                    "bps": bps,
                    "per": per,
                    "pbr": pbr,
                    "earnings_yield": earnings_yield,
                    "book_to_market": book_to_market,
                    "implied_roe": implied_roe,
                }
            )

    scored = pd.DataFrame(rows)
    coverage: dict[str, float | bool] = {
        "constituent_members": float(len(members)),
        "fundamental_members": float(len(scored)),
        "index_weight_coverage_pct": (
            float(scored["index_weight_pct"].sum()) if not scored.empty else 0.0
        ),
        "snapshots_aligned": snapshots_aligned,
    }
    if scored.empty:
        return scored, coverage

    scored["value_score"] = scored[["earnings_yield", "book_to_market"]].rank(
        pct=True, method="average"
    ).mean(axis=1, skipna=False)
    scored["profitability_score"] = scored["implied_roe"].rank(
        pct=True, method="average"
    )
    scored["momentum_score"] = scored["momentum_12_1"].rank(
        pct=True, method="average"
    )
    scored["eligible"] = (
        scored["eps"].gt(0.0)
        & scored["bps"].gt(0.0)
        & scored["per"].between(2.0, 50.0, inclusive="both")
        & scored["pbr"].between(0.1, 8.0, inclusive="both")
        & scored["implied_roe"].between(0.0, 1.0, inclusive="both")
        & scored["momentum_12_1"].gt(0.0)
        & scored["signal_close"].gt(scored["ma200"])
        & scored["close"].le(float(maximum_share_price))
    )
    scored["composite_score"] = (
        0.45 * scored["value_score"]
        + 0.25 * scored["profitability_score"]
        + 0.30 * scored["momentum_score"]
    )
    scored = scored.sort_values(
        ["eligible", "composite_score", "ticker"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return scored, coverage


def select_krx_small_account_satellite(
    scores: pd.DataFrame,
    coverage: dict[str, float | bool],
    *,
    top_n: int = 4,
    minimum_members: int = 23,
    minimum_weight_coverage_pct: float = 65.0,
) -> list[str]:
    """Choose the fixed-size sleeve only when the data panel is complete."""

    if not coverage.get("snapshots_aligned", False):
        return []
    if float(coverage.get("fundamental_members", 0.0)) < minimum_members:
        return []
    if (
        float(coverage.get("index_weight_coverage_pct", 0.0))
        < minimum_weight_coverage_pct
    ):
        return []
    eligible = scores.loc[scores["eligible"]] if not scores.empty else scores
    if len(eligible) < top_n:
        return []
    return eligible.head(top_n)["ticker"].tolist()


def build_krx_small_account_targets(
    constituents: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    actual_prices: dict[str, pd.DataFrame] | None = None,
    core_ticker: str = CORE_TICKER,
    core_weight: float = 0.40,
    satellite_weight: float = 0.55,
    cash_weight: float = 0.05,
    top_n: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create daily-held targets for next-session-open execution."""

    if not np.isclose(core_weight + satellite_weight + cash_weight, 1.0):
        raise ValueError("core, satellite, and cash weights must sum to one")
    core = prices.get(core_ticker)
    if core is None or core.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = pd.DatetimeIndex(pd.to_datetime(core.index)).sort_values().unique()
    core = core.reindex(dates)
    state_model = K200LowTurnoverReentry(ticker=core_ticker)
    states = state_model.compute_state_history(core)
    if states.empty:
        return pd.DataFrame(), pd.DataFrame()

    scheduled_dates = set(
        pd.to_datetime(fundamentals["available_date"]).dt.normalize().unique()
    )
    current = {"__CASH__": 1.0}
    current_risk_on = False
    target_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for position, signal_date in enumerate(states.index):
        if position + 1 < len(states):
            execution_date = states.index[position + 1]
            risk_on = states.loc[execution_date, "state"] == state_model.RISK_ON
            scheduled = signal_date.normalize() in scheduled_dates
            regime_changed = risk_on != current_risk_on
            if scheduled or regime_changed:
                selected: list[str] = []
                coverage: dict[str, float | bool] = {
                    "constituent_members": 0.0,
                    "fundamental_members": 0.0,
                    "index_weight_coverage_pct": 0.0,
                    "snapshots_aligned": False,
                }
                scores = pd.DataFrame()
                fallback = "risk_off_cash"
                if risk_on:
                    scores, coverage = compute_krx_small_account_scores(
                        constituents,
                        fundamentals,
                        prices,
                        signal_date,
                        actual_prices=actual_prices,
                    )
                    selected = select_krx_small_account_satellite(
                        scores, coverage, top_n=top_n
                    )
                    if selected:
                        current = {core_ticker: core_weight}
                        current.update(
                            {
                                ticker: satellite_weight / len(selected)
                                for ticker in selected
                            }
                        )
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
                            "regime_change" if regime_changed else "krx_snapshot"
                        ),
                        "selected": ",".join(selected),
                        "selected_scores": ",".join(
                            f"{ticker}:{score_map[ticker]:.6f}"
                            for ticker in selected
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
