import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from point_in_time_constituents import (
    PointInTimeConstituentError,
    constituents_asof,
    load_point_in_time_constituents,
    validate_point_in_time_constituents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "kodex200_top30_pcf_quarterly_2018_2025.csv"
)
FACTOR_REFERENCES = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "kodex_value_lowvol_top30_pcf_quarterly_2018_2025.csv",
    PROJECT_ROOT
    / "data"
    / "reference"
    / "kodex_quality_top30_pcf_quarterly_2018_2025.csv",
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2020-03-31", "2020-06-30"],
            "ticker": ["005930", "000660"],
            "name": ["삼성전자", "SK하이닉스"],
            "weight_pct": [30.0, 6.0],
        }
    )


def test_asof_does_not_expose_a_future_constituent_snapshot():
    data = _sample()

    march = constituents_asof(data, "2020-06-29")
    june = constituents_asof(data, "2020-06-30")

    assert list(march.index) == ["005930"]
    assert list(june.index) == ["000660"]


def test_asof_rejects_stale_quarterly_membership():
    assert constituents_asof(_sample(), "2020-10-15", max_age_days=100).empty


def test_validator_rejects_duplicate_membership():
    duplicate = pd.concat([_sample().iloc[[0]], _sample().iloc[[0]]])
    with pytest.raises(PointInTimeConstituentError, match="duplicate"):
        validate_point_in_time_constituents(duplicate)


def test_reference_has_complete_unique_quarterly_top_rows():
    data = load_point_in_time_constituents(REFERENCE)
    counts = data.groupby("as_of_date").size()

    assert len(data) == 928
    assert counts.size == 32
    assert counts.eq(29).all()
    assert data["ticker"].nunique() == 53
    assert data["as_of_date"].min() == pd.Timestamp("2018-03-30")
    assert data["as_of_date"].max() == pd.Timestamp("2025-12-30")


def test_reference_hash_matches_immutable_manifest():
    manifest_path = REFERENCE.with_name("kodex200_top30_pcf_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == (
        manifest["normalized_file"]["sha256"]
    )


def test_factor_proxy_references_have_matching_quarterly_snapshots():
    loaded = [load_point_in_time_constituents(path) for path in FACTOR_REFERENCES]

    for data in loaded:
        assert len(data) == 928
        assert data.groupby("as_of_date").size().eq(29).all()
    assert loaded[0]["as_of_date"].drop_duplicates().tolist() == (
        loaded[1]["as_of_date"].drop_duplicates().tolist()
    )


def test_factor_proxy_hashes_match_manifest():
    manifest_path = REFERENCE.with_name("kodex_factor_pcf_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for path in FACTOR_REFERENCES:
        expected = manifest["normalized_files"][path.name]["sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
