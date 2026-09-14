from pathlib import Path

import pandas as pd
import pytest

from scripts.import_krx_actual_closes import (
    build_actual_close_panel,
    discover_raw_files,
)


def _raw(path: Path, dates: list[str]) -> None:
    pd.DataFrame(
        {
            "일자": dates,
            "종가": ["50,000"] * len(dates),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_discovery_groups_date_chunks_by_filename_ticker(tmp_path):
    first = tmp_path / "KRX_005930_2017_2019.csv"
    second = tmp_path / "KRX_005930_2020_2022.csv"
    other = tmp_path / "KRX_000660_2017_2022.csv"
    for path in (first, second, other):
        _raw(path, ["2020-01-02"])

    result = discover_raw_files(tmp_path)

    assert result["005930"] == [first, second]
    assert result["000660"] == [other]


def test_discovery_rejects_ambiguous_filename(tmp_path):
    _raw(tmp_path / "005930_000660.csv", ["2020-01-02"])

    with pytest.raises(ValueError, match="exactly one KRX ticker"):
        discover_raw_files(tmp_path)


def test_builder_rejects_missing_universe_ticker(tmp_path):
    samsung = tmp_path / "KRX_005930.csv"
    _raw(samsung, ["2020-01-02"])

    with pytest.raises(ValueError, match="000660"):
        build_actual_close_panel(
            discover_raw_files(tmp_path),
            required_tickers={"005930", "000660"},
        )


def test_builder_rejects_post_development_data(tmp_path):
    samsung = tmp_path / "KRX_005930.csv"
    _raw(samsung, ["2022-12-29", "2023-01-02"])

    with pytest.raises(ValueError, match="sealed post-development"):
        build_actual_close_panel(
            discover_raw_files(tmp_path), required_tickers={"005930"}
        )


def test_builder_combines_nonoverlapping_chunks_and_records_hashes(tmp_path):
    first = tmp_path / "KRX_005930_2017_2019.csv"
    second = tmp_path / "KRX_005930_2020_2022.csv"
    _raw(first, ["2019-12-30"])
    _raw(second, ["2020-01-02"])

    panel, provenance = build_actual_close_panel(
        discover_raw_files(tmp_path), required_tickers={"005930"}
    )

    assert panel["date"].tolist() == list(
        pd.to_datetime(["2019-12-30", "2020-01-02"])
    )
    assert len(provenance) == 2
    assert all(len(item["sha256"]) == 64 for item in provenance)
