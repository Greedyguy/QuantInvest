import pandas as pd
import pytest

from krx_execution_data import (
    KrxExecutionDataError,
    assert_no_unmodelled_corporate_actions,
    normalize_krx_actual_close,
    restore_actual_price_panel,
    validate_actual_close_panel,
)


def test_corporate_action_guard_rejects_scale_change_while_held():
    dates = pd.bdate_range("2020-01-02", periods=5)
    signal = {
        "005930": pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
    }
    actual = {
        "005930": pd.DataFrame({"close": [500, 505, 102, 103, 104]}, index=dates)
    }
    targets = pd.DataFrame(
        {"005930": [1.0] * 5, "__CASH__": [0.0] * 5}, index=dates
    )

    with pytest.raises(KrxExecutionDataError, match="corporate-action"):
        assert_no_unmodelled_corporate_actions(targets, signal, actual)


def test_corporate_action_guard_allows_change_before_entry():
    dates = pd.bdate_range("2020-01-02", periods=5)
    signal = {
        "005930": pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
    }
    actual = {
        "005930": pd.DataFrame({"close": [500, 505, 102, 103, 104]}, index=dates)
    }
    targets = pd.DataFrame(
        {
            "005930": [0.0, 0.0, 0.0, 1.0, 1.0],
            "__CASH__": [1.0, 1.0, 1.0, 0.0, 0.0],
        },
        index=dates,
    )

    assert_no_unmodelled_corporate_actions(targets, signal, actual)


def test_normalizer_preserves_official_actual_close_level():
    raw = pd.DataFrame(
        {
            "일자": ["2018-04-27", "2018-04-30"],
            "종목코드": ["5930", "5930"],
            "종가": ["2,650,000", "2,650,000"],
        }
    )

    result = normalize_krx_actual_close(raw)

    assert result["ticker"].tolist() == ["005930", "005930"]
    assert result["close"].tolist() == [2_650_000, 2_650_000]
    assert result["price_basis"].eq("actual_traded").all()


def test_validator_rejects_adjusted_price_basis():
    frame = pd.DataFrame(
        {
            "date": ["2018-04-30"],
            "ticker": ["005930"],
            "close": [53_000],
            "source": ["KRX Data Marketplace"],
            "price_basis": ["split_adjusted"],
        }
    )

    with pytest.raises(KrxExecutionDataError, match="adjusted levels are forbidden"):
        validate_actual_close_panel(frame)


def test_restore_panel_scales_adjusted_ohlc_to_actual_daily_close():
    dates = pd.to_datetime(["2018-04-27", "2018-04-30"])
    adjusted = {
        "005930": pd.DataFrame(
            {
                "open": [49.0, 51.0],
                "high": [51.0, 54.0],
                "low": [48.0, 50.0],
                "close": [50.0, 53.0],
                "volume": [100, 110],
            },
            index=dates,
        )
    }
    actual = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["005930", "005930"],
            "close": [2_500.0, 2_650.0],
            "source": ["KRX Data Marketplace"] * 2,
            "price_basis": ["actual_traded"] * 2,
        }
    )

    restored = restore_actual_price_panel(adjusted, actual)["005930"]

    assert restored.loc[dates[0], "open"] == 2_450.0
    assert restored.loc[dates[1], "close"] == 2_650.0


def test_restore_panel_fails_when_any_execution_date_lacks_official_close():
    dates = pd.to_datetime(["2018-04-27", "2018-04-30"])
    adjusted = {
        "005930": pd.DataFrame(
            {"open": [49.0, 51.0], "close": [50.0, 53.0]}, index=dates
        )
    }
    actual = pd.DataFrame(
        {
            "date": [dates[0]],
            "ticker": ["005930"],
            "close": [2_500.0],
            "source": ["KRX Data Marketplace"],
            "price_basis": ["actual_traded"],
        }
    )

    with pytest.raises(KrxExecutionDataError, match="missing for 2018-04-30"):
        restore_actual_price_panel(adjusted, actual)
