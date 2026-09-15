import hashlib
import json
from pathlib import Path

import pandas as pd

from krx_execution_data import actual_ohlc_for_ticker, load_actual_close_panel
from market_benchmark import (
    load_distribution_events,
    reconstruct_actual_ohlc_from_adjusted,
)


REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference"
RAW = REFERENCE / "kodex200_krx_preflight_raw_20200102_20260914.csv"
NORMALIZED = (
    REFERENCE / "kodex200_krx_preflight_actual_20200102_20260914.parquet"
)
MANIFEST = REFERENCE / (
    "kodex200_krx_preflight_actual_20200102_20260914_manifest.json"
)
PREFLIGHT = REFERENCE / "k200_reentry_shadow_official_preflight_20260914.json"
PREFLIGHT_AUDIT = (
    REFERENCE / "k200_reentry_shadow_official_preflight_audit_20260914.json"
)
DISTRIBUTIONS = REFERENCE / "kodex200_distributions.csv"
DISTRIBUTION_MANIFEST = (
    REFERENCE / "kodex200_distributions_through_20260915_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_preflight_files_match_content_addressed_krx_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prices = actual_ohlc_for_ticker(load_actual_close_panel(NORMALIZED), "069500")

    assert manifest["source"] == "KRX Data Marketplace screen 13103"
    assert manifest["price_basis"] == "actual_traded"
    assert manifest["normalized_rows"] == len(prices) == 1645
    assert prices.index.min() == pd.Timestamp("2020-01-02")
    assert prices.index.max() == pd.Timestamp("2026-09-14")
    assert manifest["normalized_file"]["sha256"] == _sha256(NORMALIZED)
    assert manifest["raw_input"]["sha256"] == _sha256(RAW)


def test_provisional_actual_price_reconstruction_matches_krx_within_ten_won():
    adjusted = pd.read_parquet(REFERENCE.parent / "ohlcv" / "069500.parquet")
    adjusted.index = pd.to_datetime(adjusted.index)
    official = actual_ohlc_for_ticker(load_actual_close_panel(NORMALIZED), "069500")
    reconstructed = reconstruct_actual_ohlc_from_adjusted(
        adjusted, load_distribution_events(DISTRIBUTIONS)
    )
    overlap = reconstructed.index.intersection(official.index)
    close_error = (
        reconstructed.loc[overlap, "close"] - official.loc[overlap, "close"]
    ).abs()

    assert len(overlap) == 1458
    assert close_error.mean() < 2.5
    assert close_error.max() < 10.0


def test_official_preflight_is_before_measurement_and_authorizes_no_capital():
    payload = json.loads(PREFLIGHT.read_text(encoding="utf-8"))

    assert payload["observation_version"] == 3
    assert payload["status"] == "pre_start_diagnostic"
    assert payload["signal_date"] == "2026-09-14"
    assert payload["source"]["signal"]["price_authority"] == (
        "official_krx_derived_distribution_adjusted"
    )
    assert payload["source"]["signal"]["price_basis"] == (
        "cash_distribution_adjusted"
    )
    assert payload["source"]["execution"]["price_authority"] == (
        "official_krx_actual_traded"
    )
    assert payload["source"]["execution"]["price_basis"] == "actual_traded"
    assert payload["source"]["execution"]["provisional"] is False
    assert payload["source"]["execution"]["source_file_sha256"] == _sha256(
        NORMALIZED
    )
    assert payload["state"] == "cash"
    assert payload["target_weights_effective_at_signal_open"] == {
        "069500": 0.0,
        "__CASH__": 1.0,
    }
    assert payload["execution_guard"] == "NO_ORDERS_SENT"
    assert payload["paper_accounts"]["status"] == "not_started"
    assert payload["paper_accounts"]["capital_authorized"] is False


def test_preflight_distribution_manifest_is_complete_only_through_query_date():
    manifest = json.loads(DISTRIBUTION_MANIFEST.read_text(encoding="utf-8"))
    events = pd.read_csv(DISTRIBUTIONS)

    assert manifest["history_coverage_end"] == "2026-09-15"
    assert manifest["complete_cash_distribution_history"] is True
    assert manifest["event_count"] == len(events) == 33
    assert manifest["latest_event_record_date"] == events["record_date"].max()
    assert manifest["normalized_file"]["sha256"] == _sha256(DISTRIBUTIONS)


def test_preflight_audit_passes_inputs_but_keeps_prospective_evidence_empty():
    audit = json.loads(PREFLIGHT_AUDIT.read_text(encoding="utf-8"))

    assert audit["official_input_audit"]["complete"] is True
    assert audit["observation_evidence_valid"] is True
    assert audit["expected_exchange_sessions"] == 0
    assert audit["paper_accounts"]["subsequent_sessions"] == 0
    assert audit["decision"] == "collecting_prospective_sessions"
    assert audit["capital_authorized"] is False
