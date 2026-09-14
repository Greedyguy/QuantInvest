from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_k200_reentry_shadow import audit_shadow_records
from scripts.report_k200_reentry_shadow import build_shadow_payload


def _prices(end: str) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", end=end)
    close = 100.0 * np.cumprod(np.repeat(1.0005, len(dates)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )


def _no_distributions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "record_date",
            "pay_date",
            "distribution_per_share",
            "taxable_per_share",
        ]
    )


def _records(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> list[dict]:
    records = []
    for day in dates:
        records.append(
            build_shadow_payload(
                prices.loc[:day],
                source_path=Path(f"069500_{day:%Y%m%d}.parquet"),
                distribution_events=_no_distributions(),
                generated_at=datetime(
                    day.year, day.month, day.day, 7, tzinfo=timezone.utc
                ),
            )
        )
    return records


def test_audit_accepts_complete_causal_daily_observations_while_collecting():
    prices = _prices("2026-09-21")
    eligible = prices.loc["2026-09-16":].index
    result = audit_shadow_records(
        _records(prices, eligible),
        prices,
        _no_distributions(),
        audit_end=eligible[-1],
    )

    assert result["observation_evidence_valid"] is True
    assert result["missing_signal_dates"] == []
    assert result["decision"] == "collecting_prospective_sessions"
    assert result["capital_authorized"] is False


def test_audit_fails_closed_when_an_exchange_session_is_missing():
    prices = _prices("2026-09-21")
    eligible = prices.loc["2026-09-16":].index
    result = audit_shadow_records(
        _records(prices, eligible[:-1]),
        prices,
        _no_distributions(),
        audit_end=eligible[-1],
    )

    assert result["observation_evidence_valid"] is False
    assert result["missing_signal_dates"] == [eligible[-1].date().isoformat()]
    assert result["decision"] == "invalid_observation_evidence"


def test_audit_detects_a_tampered_frozen_signal():
    prices = _prices("2026-09-18")
    eligible = prices.loc["2026-09-16":].index
    records = _records(prices, eligible)
    records[-1]["state"] = "tampered"

    result = audit_shadow_records(
        records,
        prices,
        _no_distributions(),
        audit_end=eligible[-1],
    )

    assert result["observation_evidence_valid"] is False
    assert any("state" in error for error in result["observation_errors"])
    assert result["capital_authorized"] is False


def test_audit_treats_current_distribution_manifest_as_incomplete_for_future_end():
    prices = _prices("2025-12-10")
    reference = Path(__file__).resolve().parents[1] / "data" / "reference"
    result = audit_shadow_records(
        [],
        prices,
        _no_distributions(),
        audit_end="2025-12-10",
        distribution_path=reference / "kodex200_distributions.csv",
        distribution_manifest_path=(
            reference / "kodex200_distributions_manifest.json"
        ),
    )

    audit = result["official_input_audit"]
    assert audit["distribution_input_complete"] is False
    assert any("audit end date" in error for error in audit["distribution_errors"])
