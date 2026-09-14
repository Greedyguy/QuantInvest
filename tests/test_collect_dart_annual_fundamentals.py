from pathlib import Path

import pandas as pd
import pytest

from scripts.collect_dart_annual_fundamentals import collect


def test_collector_refuses_to_open_post_development_filings(tmp_path: Path):
    constituents = tmp_path / "constituents.csv"
    pd.DataFrame(
        {
            "as_of_date": ["2022-12-29"],
            "ticker": ["005930"],
            "name": ["삼성전자"],
            "weight_pct": [25.0],
        }
    ).to_csv(constituents, index=False)

    with pytest.raises(ValueError, match="refuses to open 2023"):
        collect(
            constituents,
            start_date="2018-01-01",
            end_date="2023-01-01",
            cache_dir=tmp_path / "cache",
            pause_seconds=0,
        )
