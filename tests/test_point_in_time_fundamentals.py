import pandas as pd
import pytest

from point_in_time_fundamentals import (
    PointInTimeDataError,
    fundamentals_asof,
    normalize_krx_snapshot,
    validate_point_in_time_fundamentals,
)


def _snapshot(snapshot_date="2020-01-31", available_date="2020-02-03", eps=1000):
    return pd.DataFrame(
        {
            "snapshot_date": [snapshot_date],
            "available_date": [available_date],
            "ticker": ["005930"],
            "market": ["KOSPI"],
            "name": ["삼성전자"],
            "close": [50000],
            "market_cap": [300_000_000_000_000],
            "trading_value": [1_000_000_000_000],
            "bps": [30000],
            "eps": [eps],
            "per": [50.0],
            "pbr": [1.67],
            "dps": [1000],
            "dividend_yield": [2.0],
            "source": ["KRX Data Marketplace"],
        }
    )


def test_asof_never_exposes_a_future_available_snapshot():
    old = _snapshot()
    corrected = _snapshot(available_date="2020-03-02", eps=1200)
    data = pd.concat([old, corrected], ignore_index=True)

    february = fundamentals_asof(data, "2020-02-28")
    march = fundamentals_asof(data, "2020-03-02")

    assert february.loc["005930", "eps"] == 1000
    assert march.loc["005930", "eps"] == 1200


def test_asof_drops_stale_monthly_records():
    assert fundamentals_asof(_snapshot(), "2020-04-01", max_age_days=45).empty


def test_validator_rejects_impossible_availability_date():
    data = _snapshot(available_date="2020-01-30")
    with pytest.raises(PointInTimeDataError, match="must be after"):
        validate_point_in_time_fundamentals(data)


def test_validator_rejects_nonofficial_source():
    data = _snapshot()
    data["source"] = "unknown portal"
    with pytest.raises(PointInTimeDataError, match="official KRX"):
        validate_point_in_time_fundamentals(data)


def test_normalize_krx_snapshot_merges_official_tables():
    fundamentals = pd.DataFrame(
        {
            "종목코드": ["5930"],
            "종목명": ["삼성전자"],
            "BPS": ["30,000"],
            "EPS": ["1,000"],
            "PER": ["50.0"],
            "PBR": ["1.67"],
            "DPS": ["1,000"],
            "배당수익률": ["2.0"],
        }
    )
    trading = pd.DataFrame(
        {
            "종목코드": ["005930"],
            "시장구분": ["KOSPI"],
            "종목명": ["삼성전자"],
            "종가": ["50,000"],
            "시가총액": ["300,000,000,000,000"],
            "거래대금": ["1,000,000,000,000"],
        }
    )

    result = normalize_krx_snapshot(
        fundamentals,
        trading,
        snapshot_date="2020-01-31",
        available_date="2020-02-03",
    )

    assert result.loc[0, "ticker"] == "005930"
    assert result.loc[0, "market_cap"] == 300_000_000_000_000
    assert result.loc[0, "eps"] == 1000
