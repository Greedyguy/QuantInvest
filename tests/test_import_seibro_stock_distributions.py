from pathlib import Path

import pandas as pd
import pytest

from scripts.import_seibro_stock_distributions import (
    build_distribution_panel,
    discover_exports,
    normalize_export,
)


def _export(
    path: Path,
    ticker: str,
    *,
    amount: float = 361.0,
    differential: float | None = None,
) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("배정기준일", "배정기준일"),
            ("현금배당 지급일", "현금배당 지급일"),
            ("종목코드", "종목코드"),
            ("주식종류", "주식종류"),
            ("배당구분", "배당구분"),
            ("주당배당금", "일반"),
            ("주당배당금", "차등"),
        ]
    )
    frame = pd.DataFrame(
        [[20221231, 20230414, ticker, "보통주", "현금배당", amount, differential]],
        columns=columns,
    )
    path.write_text(frame.to_html(index=False), encoding="euc-kr")


def test_discovery_accepts_one_six_digit_named_export(tmp_path):
    expected = tmp_path / "005930.xls"
    _export(expected, "005930")
    _export(tmp_path / "notes.xls", "005930")

    assert discover_exports(tmp_path) == {"005930": expected}


def test_normalizer_maps_official_dates_amount_and_taxable_base(tmp_path):
    path = tmp_path / "005930.xls"
    _export(path, "005930")

    result = normalize_export(path, "005930")

    assert result.iloc[0]["ticker"] == "005930"
    assert result.iloc[0]["record_date"] == pd.Timestamp("2022-12-31")
    assert result.iloc[0]["pay_date"] == pd.Timestamp("2023-04-14")
    assert result.iloc[0]["distribution_per_share"] == 361.0
    assert result.iloc[0]["taxable_per_share"] == 361.0
    assert result.iloc[0]["source"].startswith("KSD SEIBro")


def test_normalizer_rejects_ambiguous_differential_dividend(tmp_path):
    path = tmp_path / "005930.xls"
    _export(path, "005930", amount=361.0, differential=400.0)

    with pytest.raises(ValueError, match="differential"):
        normalize_export(path, "005930")


def test_builder_requires_exact_audited_ticker_set(tmp_path):
    path = tmp_path / "005930.xls"
    _export(path, "005930")

    with pytest.raises(ValueError, match="000660"):
        build_distribution_panel(
            discover_exports(tmp_path),
            required_tickers={"005930", "000660"},
            history_start=pd.Timestamp("2018-01-01"),
            history_end=pd.Timestamp("2022-12-31"),
        )
