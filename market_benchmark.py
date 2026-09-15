"""Investable market benchmarks and ETF cash-flow helpers.

The project stores adjusted OHLCV for signal research.  That is useful for
indicators, but a small-account simulation needs actual traded prices, integer
shares, taxes, and cash distributions.  This module keeps those two price
bases explicit and evaluates a strategy against an investable benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetTaxProfile:
    """Tax rates used by the account simulator, not by signal generation."""

    sell_transaction_tax_rate: float
    distribution_income_tax_rate: float


DOMESTIC_EQUITY_ETF = AssetTaxProfile(
    sell_transaction_tax_rate=0.0,
    distribution_income_tax_rate=0.154,
)


@dataclass(frozen=True)
class MarketOutperformanceCriteria:
    """Pre-committed gates for promoting a candidate beyond research."""

    min_annualised_excess_return: float = 0.02
    min_rolling_12m_beat_rate: float = 0.60
    max_mdd_disadvantage: float = 0.05
    max_positive_excess_year_share: float = 0.60
    rolling_sessions: int = 252


def load_samsung_kodex_standard_xls(path: str | Path) -> pd.DataFrame:
    """Load official daily market close, NAV, and tax NAV from Samsung XLS."""

    try:
        import xlrd
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise ImportError("xlrd is required to read the official KODEX .xls file") from exc

    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    rows: list[dict] = []
    for row_number in range(sheet.nrows):
        values = sheet.row_values(row_number)
        try:
            date = pd.to_datetime(str(int(values[0])), format="%Y%m%d")
            market_close = float(values[1])
            nav = float(values[5])
            tax_nav = float(values[8])
        except (IndexError, TypeError, ValueError):
            continue
        rows.append(
            {
                "date": date,
                "market_close": market_close,
                "nav": nav,
                "tax_nav": tax_nav,
            }
        )
    if not rows:
        raise ValueError(f"official KODEX workbook contains no price rows: {path}")
    return (
        pd.DataFrame(rows)
        .set_index("date")
        .sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
    )


def load_samsung_kodex_total_return_json(path: str | Path) -> pd.DataFrame:
    """Load Samsung's since-inception price, NAV-total-return, and index series."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("official total-return JSON must contain a list")
    frame = pd.DataFrame(payload)
    required = {"GIJUN_YMD", "SUIK_PRICE", "SUIK_NAV", "SUIK_JISU"}
    if not required.issubset(frame.columns):
        raise ValueError("official total-return JSON is missing required fields")
    frame["date"] = pd.to_datetime(frame["GIJUN_YMD"], format="%Y%m%d")
    frame = frame.set_index("date").sort_index()
    result = pd.DataFrame(index=frame.index)
    result["market_price_index"] = 1.0 + pd.to_numeric(
        frame["SUIK_PRICE"], errors="coerce"
    ) / 100.0
    result["nav_total_return_index"] = 1.0 + pd.to_numeric(
        frame["SUIK_NAV"], errors="coerce"
    ) / 100.0
    result["underlying_index"] = 1.0 + pd.to_numeric(
        frame["SUIK_JISU"], errors="coerce"
    ) / 100.0
    return result.dropna().loc[lambda out: ~out.index.duplicated(keep="last")]


def load_distribution_events(path: str | Path) -> pd.DataFrame:
    """Load per-share distributions with official record and payment dates."""

    frame = pd.read_csv(path, dtype={"record_date": str, "pay_date": str})
    required = {"record_date", "pay_date", "distribution_per_share"}
    if not required.issubset(frame.columns):
        raise ValueError("distribution file is missing required fields")
    frame = frame.copy()
    frame["record_date"] = pd.to_datetime(frame["record_date"], format="%Y-%m-%d")
    frame["pay_date"] = pd.to_datetime(frame["pay_date"], format="%Y-%m-%d")
    frame["distribution_per_share"] = pd.to_numeric(
        frame["distribution_per_share"], errors="raise"
    )
    if "taxable_per_share" not in frame:
        frame["taxable_per_share"] = frame["distribution_per_share"]
    else:
        frame["taxable_per_share"] = pd.to_numeric(
            frame["taxable_per_share"], errors="coerce"
        ).fillna(frame["distribution_per_share"])
    return frame.sort_values("record_date").reset_index(drop=True)


def load_samsung_distribution_json(path: str | Path) -> pd.DataFrame:
    """Load Samsung's KODEX distribution API response."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("dividList", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame(
            columns=[
                "record_date",
                "pay_date",
                "distribution_per_share",
                "taxable_per_share",
            ]
        )
    frame = pd.DataFrame(rows).rename(
        columns={
            "basicD": "record_date",
            "payD": "pay_date",
            "dividA": "distribution_per_share",
            "taxDividA": "taxable_per_share",
        }
    )
    frame["record_date"] = pd.to_datetime(frame["record_date"], format="%Y%m%d")
    frame["pay_date"] = pd.to_datetime(frame["pay_date"], format="%Y%m%d")
    frame["distribution_per_share"] = pd.to_numeric(
        frame["distribution_per_share"], errors="raise"
    )
    frame["taxable_per_share"] = pd.to_numeric(
        frame["taxable_per_share"], errors="coerce"
    ).fillna(frame["distribution_per_share"])
    return frame.sort_values("record_date").reset_index(drop=True)


def restore_actual_ohlc(
    adjusted_prices: pd.DataFrame,
    official_market_close: pd.Series,
) -> pd.DataFrame:
    """Put adjusted OHLC on each day's actual traded-price scale.

    Corporate-action adjustment factors are constant within a trading day, so
    the official-close ratio can be applied to open/high/low while the close is
    replaced with the authoritative value.
    """

    adjusted = adjusted_prices.copy().sort_index()
    adjusted.index = pd.to_datetime(adjusted.index)
    actual_close = official_market_close.copy().sort_index().astype(float)
    actual_close.index = pd.to_datetime(actual_close.index)
    aligned_close = actual_close.reindex(adjusted.index)
    missing = aligned_close.isna()
    if missing.any():
        first = str(adjusted.index[missing][0].date())
        raise ValueError(f"official market close is missing for {first}")
    adjusted_close = pd.to_numeric(adjusted["close"], errors="coerce")
    factor = aligned_close / adjusted_close
    if (~np.isfinite(factor) | factor.le(0)).any():
        raise ValueError("could not derive a valid actual-price adjustment")
    result = adjusted.copy()
    for column in ("open", "high", "low", "close"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce") * factor
    result["close"] = aligned_close
    return result


def adjust_ohlc_for_distributions(
    actual_prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    settlement_lag: int = 2,
) -> pd.DataFrame:
    """Back-adjust actual OHLC for cash distributions without future signals.

    Each distribution factor is applied to the entitlement session and every
    earlier row.  A later distribution therefore changes only the absolute
    scale of an earlier prefix, not its returns or moving-average decisions.
    """

    prices = actual_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    schedule = prepare_distribution_schedule(
        events,
        prices.index,
        settlement_lag=settlement_lag,
    )
    cumulative_factor = pd.Series(1.0, index=prices.index)
    for event in schedule.to_dict("records"):
        entitlement_date = pd.Timestamp(event["entitlement_date"])
        reference_close = float(prices.loc[entitlement_date, "close"])
        distribution = float(event["gross_unit"])
        factor = (reference_close - distribution) / reference_close
        if not np.isfinite(factor) or factor <= 0 or factor > 1:
            raise ValueError(
                f"invalid distribution adjustment on {entitlement_date.date()}"
            )
        cumulative_factor.loc[cumulative_factor.index <= entitlement_date] *= factor

    adjusted = prices.copy()
    for column in ("open", "high", "low", "close"):
        if column in adjusted:
            adjusted[column] = (
                pd.to_numeric(adjusted[column], errors="coerce")
                * cumulative_factor
            )
    return adjusted


def reconstruct_actual_ohlc_from_adjusted(
    adjusted_prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    settlement_lag: int = 2,
) -> pd.DataFrame:
    """Approximately restore traded OHLC from a cash-adjusted price history.

    The project cache rounds adjusted ETF prices to whole won, so this inverse
    is suitable only for daily provisional paper accounting.  A final audit
    must replace it with official actual-traded KRX OHLC.
    """

    adjusted = adjusted_prices.copy().sort_index()
    adjusted.index = pd.to_datetime(adjusted.index)
    adjusted = adjusted.loc[~adjusted.index.duplicated(keep="last")]
    schedule = prepare_distribution_schedule(
        events,
        adjusted.index,
        settlement_lag=settlement_lag,
    )
    event_factors: list[tuple[pd.Timestamp, float]] = []
    later_factor = 1.0
    for event in reversed(schedule.to_dict("records")):
        entitlement_date = pd.Timestamp(event["entitlement_date"])
        adjusted_close = float(adjusted.loc[entitlement_date, "close"])
        distribution = float(event["gross_unit"])
        actual_close = adjusted_close / later_factor + distribution
        factor = (actual_close - distribution) / actual_close
        if not np.isfinite(factor) or factor <= 0 or factor > 1:
            raise ValueError(
                f"invalid reverse distribution adjustment on "
                f"{entitlement_date.date()}"
            )
        event_factors.append((entitlement_date, factor))
        later_factor *= factor

    cumulative_factor = pd.Series(1.0, index=adjusted.index)
    for entitlement_date, factor in event_factors:
        cumulative_factor.loc[cumulative_factor.index <= entitlement_date] *= factor

    actual = adjusted.copy()
    for column in ("open", "high", "low", "close"):
        if column in actual:
            actual[column] = (
                pd.to_numeric(actual[column], errors="coerce")
                / cumulative_factor
            )
    return actual


def prepare_distribution_schedule(
    events: pd.DataFrame,
    trading_dates: pd.Index,
    *,
    settlement_lag: int = 2,
    tax_profile: AssetTaxProfile = DOMESTIC_EQUITY_ETF,
) -> pd.DataFrame:
    """Resolve record/payment dates to tradable entitlement/credit sessions."""

    if settlement_lag < 0:
        raise ValueError("settlement_lag must be non-negative")
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
    empty_columns = [
        "record_date",
        "pay_date",
        "entitlement_date",
        "credit_date",
        "gross_unit",
        "tax_unit",
        "net_unit",
    ]
    if dates.empty:
        return pd.DataFrame(columns=empty_columns)
    rows: list[dict] = []
    for event in events.to_dict("records"):
        record_date = pd.Timestamp(event["record_date"])
        pay_date = pd.Timestamp(event["pay_date"])
        # A calendar record date can fall just after the last exchange session
        # (for example, Saturday 31 December after Thursday 29 December).  Keep
        # that entitlement, but never drag genuinely future events backwards
        # into a truncated test calendar.
        if record_date > dates[-1] + pd.Timedelta(days=7):
            continue
        record_position = int(dates.searchsorted(record_date, side="right") - 1)
        entitlement_position = record_position - settlement_lag
        credit_position = int(dates.searchsorted(pay_date, side="left"))
        if entitlement_position < 0:
            continue
        credit_date = (
            dates[credit_position]
            if credit_position < len(dates)
            else pay_date.normalize()
        )
        gross_unit = float(event["distribution_per_share"])
        taxable_unit = float(event.get("taxable_per_share", gross_unit))
        tax_unit = max(taxable_unit, 0.0) * tax_profile.distribution_income_tax_rate
        rows.append(
            {
                **event,
                "record_date": record_date,
                "pay_date": pay_date,
                "entitlement_date": dates[entitlement_position],
                "credit_date": credit_date,
                "gross_unit": gross_unit,
                "tax_unit": tax_unit,
                "net_unit": gross_unit - tax_unit,
            }
        )
    if not rows:
        return pd.DataFrame(columns=empty_columns)
    return pd.DataFrame(rows).sort_values("entitlement_date").reset_index(drop=True)


def period_return_asof(
    index_values: pd.Series,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> float:
    """Return a point-to-point index return without looking past either date."""

    values = index_values.astype(float).sort_index().dropna()
    start_values = values.loc[values.index <= pd.Timestamp(start)]
    end_values = values.loc[values.index <= pd.Timestamp(end)]
    if start_values.empty or end_values.empty:
        raise ValueError("index does not cover requested period")
    return float(end_values.iloc[-1] / start_values.iloc[-1] - 1.0)


def evaluate_market_outperformance(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    criteria: MarketOutperformanceCriteria = MarketOutperformanceCriteria(),
) -> dict:
    """Evaluate fixed market-beating gates on two investable equity curves."""

    aligned = pd.concat(
        [strategy_equity.rename("strategy"), benchmark_equity.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    aligned = aligned.sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]
    if len(aligned) < 2:
        raise ValueError("equity curves do not have enough overlapping observations")
    aligned = aligned / aligned.iloc[0]
    elapsed_years = (aligned.index[-1] - aligned.index[0]).days / 365.2425
    if elapsed_years <= 0:
        raise ValueError("equity curves need distinct dates")

    strategy_cagr = float(aligned["strategy"].iloc[-1] ** (1.0 / elapsed_years) - 1.0)
    benchmark_cagr = float(aligned["benchmark"].iloc[-1] ** (1.0 / elapsed_years) - 1.0)
    annualised_excess = strategy_cagr - benchmark_cagr

    rolling = aligned.pct_change(criteria.rolling_sessions, fill_method=None).dropna()
    rolling_beat_rate = (
        float((rolling["strategy"] > rolling["benchmark"]).mean())
        if not rolling.empty
        else 0.0
    )
    strategy_mdd = float((aligned["strategy"] / aligned["strategy"].cummax() - 1.0).min())
    benchmark_mdd = float((aligned["benchmark"] / aligned["benchmark"].cummax() - 1.0).min())
    mdd_disadvantage = abs(strategy_mdd) - abs(benchmark_mdd)

    daily = aligned.pct_change(fill_method=None).fillna(0.0)
    annual = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    annual_excess = annual["strategy"] - annual["benchmark"]
    positive_excess = annual_excess.clip(lower=0.0)
    positive_total = float(positive_excess.sum())
    concentration = (
        float(positive_excess.max() / positive_total) if positive_total > 0 else 1.0
    )

    gates = {
        "annualised_excess_return": annualised_excess
        >= criteria.min_annualised_excess_return,
        "rolling_12m_beat_rate": rolling_beat_rate
        >= criteria.min_rolling_12m_beat_rate,
        "mdd_disadvantage": mdd_disadvantage <= criteria.max_mdd_disadvantage,
        "positive_excess_year_concentration": concentration
        <= criteria.max_positive_excess_year_share,
    }
    return {
        "strategy_cagr_pct": strategy_cagr * 100.0,
        "benchmark_cagr_pct": benchmark_cagr * 100.0,
        "annualised_excess_return_pct_point": annualised_excess * 100.0,
        "rolling_12m_beat_rate_pct": rolling_beat_rate * 100.0,
        "strategy_mdd_pct": strategy_mdd * 100.0,
        "benchmark_mdd_pct": benchmark_mdd * 100.0,
        "mdd_disadvantage_pct_point": mdd_disadvantage * 100.0,
        "positive_excess_year_concentration_pct": concentration * 100.0,
        "annual_excess_return_pct_point": {
            str(year): float(value) * 100.0 for year, value in annual_excess.items()
        },
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }
