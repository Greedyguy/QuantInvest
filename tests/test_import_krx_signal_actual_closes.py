from pathlib import Path

import pandas as pd
import pytest

from scripts.import_krx_signal_actual_closes import (
    build_signal_actual_panel,
    discover_signal_files,
)


def _raw(path: Path, tickers: list[str]) -> None:
    pd.DataFrame(
        {
            "종목코드": tickers,
            "종가": ["50,000"] * len(tickers),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _constituents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2020-03-31", "2020-03-31"],
            "ticker": ["005930", "000660"],
            "name": ["Samsung", "Hynix"],
            "weight_pct": [20.0, 10.0],
        }
    )


def test_discovery_accepts_only_unambiguous_iso_dated_files(tmp_path):
    first = tmp_path / "2020-04-01.csv"
    second = tmp_path / "2020-07-01.csv"
    _raw(first, ["005930"])
    _raw(second, ["005930"])
    _raw(tmp_path / "notes.csv", ["005930"])

    result = discover_signal_files(tmp_path)

    assert result == {
        pd.Timestamp("2020-04-01"): first,
        pd.Timestamp("2020-07-01"): second,
    }


def test_builder_requires_every_registered_decision_date(tmp_path):
    path = tmp_path / "2020-04-01.csv"
    _raw(path, ["005930", "000660"])

    with pytest.raises(ValueError, match="2020-07-01"):
        build_signal_actual_panel(
            discover_signal_files(tmp_path),
            _constituents(),
            required_dates={
                pd.Timestamp("2020-04-01"),
                pd.Timestamp("2020-07-01"),
            },
        )


def test_builder_rejects_missing_point_in_time_constituent(tmp_path):
    path = tmp_path / "2020-04-01.csv"
    _raw(path, ["005930"])

    with pytest.raises(ValueError, match="000660"):
        build_signal_actual_panel(
            discover_signal_files(tmp_path),
            _constituents(),
            required_dates={pd.Timestamp("2020-04-01")},
        )


def test_builder_normalizes_cross_section_and_hashes_source(tmp_path):
    path = tmp_path / "2020-04-01.csv"
    _raw(path, ["005930", "000660", "035420"])

    panel, provenance = build_signal_actual_panel(
        discover_signal_files(tmp_path),
        _constituents(),
        required_dates={pd.Timestamp("2020-04-01")},
    )

    assert set(panel["ticker"]) == {"005930", "000660"}
    assert panel["date"].eq(pd.Timestamp("2020-04-01")).all()
    assert panel["price_basis"].eq("actual_traded").all()
    assert provenance[0]["market_rows"] == 3
    assert provenance[0]["constituent_rows"] == 2
    assert len(provenance[0]["sha256"]) == 64
