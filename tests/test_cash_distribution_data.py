import json
from pathlib import Path

import pandas as pd
import pytest

from cash_distribution_data import (
    CashDistributionDataError,
    load_distribution_bundle,
    validate_distribution_coverage_manifest,
    validate_distribution_events,
)


def test_kodex_reference_manifest_proves_development_coverage():
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "reference"
        / "kodex200_distributions_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    validate_distribution_coverage_manifest(
        manifest,
        required_tickers={"069500"},
        start_date="2018-06-29",
        end_date="2022-12-29",
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["005930"],
            "record_date": ["2020-12-31"],
            "pay_date": ["2021-04-16"],
            "distribution_per_share": [1932],
            "taxable_per_share": [1932],
            "source": ["DART original filing and issuer payment notice"],
        }
    )


def test_event_validator_rejects_unapproved_source():
    events = _events()
    events["source"] = "finance portal"

    with pytest.raises(CashDistributionDataError, match="source is not approved"):
        validate_distribution_events(events)


def test_bundle_requires_zero_event_tickers_in_coverage_manifest(tmp_path):
    events_path = tmp_path / "events.csv"
    manifest_path = tmp_path / "manifest.json"
    _events().to_csv(events_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "history_coverage_start": "2020-01-01",
                "history_coverage_end": "2022-12-31",
                "complete_cash_distribution_history": True,
                "audited_tickers": ["005930"],
                "source": "DART original filing and issuer payment notice",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CashDistributionDataError, match="missing tickers"):
        load_distribution_bundle(
            events_path,
            manifest_path,
            required_tickers={"005930", "000660"},
            start_date="2020-01-01",
            end_date="2022-12-31",
        )


def test_bundle_accepts_audited_zero_event_ticker(tmp_path):
    events_path = tmp_path / "events.csv"
    manifest_path = tmp_path / "manifest.json"
    _events().to_csv(events_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "history_coverage_start": "2020-01-01",
                "history_coverage_end": "2022-12-31",
                "complete_cash_distribution_history": True,
                "audited_tickers": ["005930", "000660"],
                "source": "DART original filing and issuer payment notice",
            }
        ),
        encoding="utf-8",
    )

    result = load_distribution_bundle(
        events_path,
        manifest_path,
        required_tickers={"005930", "000660"},
        start_date="2020-01-01",
        end_date="2022-12-31",
    )

    assert set(result) == {"005930"}
