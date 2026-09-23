import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_k200_reentry_shadow import audit_shadow_records
from scripts.report_k200_reentry_shadow import SPEC_PATH, build_shadow_payload


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


def test_audit_exactly_recomputes_injected_official_signal_and_execution_hashes():
    prices = _prices("2026-09-16")
    day = prices.index[-1]
    record = build_shadow_payload(
        prices,
        source_path=Path("official.parquet"),
        execution_prices=prices,
        execution_source_path=Path("official.parquet"),
        distribution_events=_no_distributions(),
        signal_price_authority="official_krx_derived_distribution_adjusted",
        execution_price_authority="official_krx_actual_traded",
        generated_at=datetime(day.year, day.month, day.day, 7, tzinfo=timezone.utc),
    )

    result = audit_shadow_records(
        [record], prices, _no_distributions(), audit_end=day
    )

    assert result["observation_evidence_valid"] is True
    assert result["observation_errors"] == []


def test_full_shortened_lifecycle_only_makes_strategy_eligible_for_live_decision(
    tmp_path,
):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["signal_history_start"] = "2026-01-02"
    spec["first_eligible_signal_date"] = "2026-01-28"
    spec["parameters"].update(
        {
            "trend_window": 5,
            "momentum_window": 3,
            "fast_trend_window": 2,
            "medium_trend_window": 4,
        }
    )
    spec["evidence_policy"]["minimum_subsequent_sessions"] = 5
    spec_path = tmp_path / "shortened_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    dates = pd.bdate_range("2026-01-02", "2026-02-04")
    close = np.linspace(100.0, 80.0, len(dates))
    prices = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )
    eligible = prices.loc[spec["first_eligible_signal_date"] :].index
    events = _no_distributions()
    official_path = tmp_path / "official.parquet"
    official_path.write_bytes(b"immutable official fixture")
    official_hash = hashlib.sha256(official_path.read_bytes()).hexdigest()
    official_manifest = tmp_path / "official.manifest.json"
    official_manifest.write_text(
        json.dumps(
            {
                "source": "KRX test fixture",
                "price_basis": "actual_traded",
                "required_tickers": ["069500"],
                "fields": ["open", "high", "low", "close"],
                "normalized_file": {"sha256": official_hash},
            }
        ),
        encoding="utf-8",
    )
    distribution_path = tmp_path / "distributions.csv"
    events.to_csv(distribution_path, index=False)
    distribution_hash = hashlib.sha256(distribution_path.read_bytes()).hexdigest()
    distribution_manifest = tmp_path / "distributions.manifest.json"
    distribution_manifest.write_text(
        json.dumps(
            {
                "complete_cash_distribution_history": True,
                "audited_tickers": ["069500"],
                "history_coverage_end": eligible[-1].date().isoformat(),
                "normalized_file": {"sha256": distribution_hash},
            }
        ),
        encoding="utf-8",
    )
    records = [
        build_shadow_payload(
            prices.loc[:day],
            source_path=official_path,
            execution_prices=prices.loc[:day],
            execution_source_path=official_path,
            spec_path=spec_path,
            distribution_events=events,
            distribution_path=distribution_path,
            signal_price_authority="official_krx_derived_distribution_adjusted",
            execution_price_authority="official_krx_actual_traded",
            generated_at=datetime(
                day.year, day.month, day.day, 7, tzinfo=timezone.utc
            ),
        )
        for day in eligible
    ]

    result = audit_shadow_records(
        records,
        prices,
        events,
        spec_path=spec_path,
        audit_end=eligible[-1],
        price_path=official_path,
        price_manifest_path=official_manifest,
        distribution_path=distribution_path,
        distribution_manifest_path=distribution_manifest,
    )

    assert result["observation_evidence_valid"] is True
    assert result["official_input_audit"]["complete"] is True
    assert result["paper_accounts"]["comparison"]["evidence_complete"] is True
    assert result["paper_accounts"]["comparison"][
        "provisional_passes_all_gates"
    ] is True
    assert result["decision"] == "eligible_for_separate_live_capital_decision"
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
