from pathlib import Path

import pandas as pd
import pytest

from scripts.import_krx_fundamental_panel import (
    build_fundamental_panel,
    discover_snapshot_pairs,
    next_trading_session,
)


def _fundamental(path: Path) -> None:
    pd.DataFrame(
        {
            "종목코드": ["005930"],
            "종목명": ["삼성전자"],
            "BPS": ["30,000"],
            "EPS": ["1,000"],
            "PER": ["50.0"],
            "PBR": ["1.67"],
            "DPS": ["1,000"],
            "배당수익률": ["2.0"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _trading(path: Path) -> None:
    pd.DataFrame(
        {
            "종목코드": ["005930"],
            "시장구분": ["KOSPI"],
            "종목명": ["삼성전자"],
            "종가": ["50,000"],
            "시가총액": ["300,000,000,000,000"],
            "거래대금": ["1,000,000,000,000"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _pair(directory: Path, date: str) -> None:
    _fundamental(directory / f"{date}_fundamentals.csv")
    _trading(directory / f"{date}_trading.csv")


def test_discovery_requires_both_tables_for_each_date(tmp_path):
    _fundamental(tmp_path / "2020-03-31_fundamentals.csv")

    with pytest.raises(ValueError, match="incomplete KRX snapshot pairs"):
        discover_snapshot_pairs(tmp_path)


def test_next_session_uses_trading_calendar_not_calendar_day():
    sessions = pd.to_datetime(["2020-04-29", "2020-05-06"])

    assert next_trading_session(pd.Timestamp("2020-04-29"), sessions) == pd.Timestamp(
        "2020-05-06"
    )


def test_builder_rejects_any_missing_registered_snapshot(tmp_path):
    _pair(tmp_path, "2020-03-31")

    with pytest.raises(ValueError, match="2020-06-30"):
        build_fundamental_panel(
            discover_snapshot_pairs(tmp_path),
            required_snapshots={
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
            },
            trading_dates=pd.to_datetime(["2020-04-01", "2020-07-01"]),
        )


def test_builder_normalizes_complete_pairs_and_hashes_sources(tmp_path):
    _pair(tmp_path, "2020-03-31")
    _pair(tmp_path, "2020-06-30")

    panel, provenance = build_fundamental_panel(
        discover_snapshot_pairs(tmp_path),
        required_snapshots={pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")},
        trading_dates=pd.to_datetime(
            ["2020-04-01", "2020-07-01", "2020-07-02"]
        ),
    )

    assert panel["snapshot_date"].nunique() == 2
    assert panel["available_date"].tolist() == list(
        pd.to_datetime(["2020-04-01", "2020-07-01"])
    )
    assert len(provenance) == 2
    assert len(provenance[0]["fundamentals"]["sha256"]) == 64


def test_builder_rejects_unregistered_extra_date(tmp_path):
    _pair(tmp_path, "2020-03-31")
    _pair(tmp_path, "2020-06-30")

    with pytest.raises(ValueError, match="non-development snapshots"):
        build_fundamental_panel(
            discover_snapshot_pairs(tmp_path),
            required_snapshots={pd.Timestamp("2020-03-31")},
            trading_dates=pd.to_datetime(["2020-04-01", "2020-07-01"]),
        )
