import pandas as pd
import pytest

from scripts.import_kodex_tax_nav import build_development_tax_nav


def test_build_tax_nav_truncates_unopened_dates_and_preserves_values():
    dates = pd.to_datetime(["2022-12-28", "2022-12-29", "2023-01-02"])
    official = pd.DataFrame(
        {
            "market_close": [13_285.0, 12_805.0, 12_900.0],
            "nav": [13_365.0, 12_852.0, 12_950.0],
            "tax_nav": [10_820.87, 10_887.85, 10_890.0],
        },
        index=dates,
    )

    result = build_development_tax_nav(
        official,
        ticker="122630",
        start_date="2018-06-29",
        end_date="2022-12-29",
    )

    assert result["date"].max() == pd.Timestamp("2022-12-29")
    assert result["ticker"].eq("122630").all()
    assert result.iloc[-1]["market_close"] == 12_805.0
    assert result.iloc[-1]["tax_nav"] == 10_887.85


def test_build_tax_nav_rejects_invalid_values():
    official = pd.DataFrame(
        {"market_close": [100.0], "tax_nav": [float("nan")]},
        index=pd.to_datetime(["2022-12-29"]),
    )

    with pytest.raises(ValueError, match="must be positive"):
        build_development_tax_nav(
            official,
            ticker="122630",
            start_date="2022-01-01",
            end_date="2022-12-29",
        )
