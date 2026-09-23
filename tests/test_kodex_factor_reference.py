import hashlib
import json
from pathlib import Path

import pandas as pd

from market_benchmark import load_distribution_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "kodex_factor_distributions_2018_2026.csv"
)
MANIFEST = REFERENCE.with_name("kodex_factor_distribution_manifest.json")


def test_factor_distribution_reference_is_unique_and_officially_sourced():
    raw = pd.read_csv(REFERENCE, dtype={"ticker": str})
    parsed = load_distribution_events(REFERENCE)

    assert set(raw["ticker"]) == {
        "211900",
        "223190",
        "244620",
        "244660",
        "252650",
    }
    assert not raw.duplicated(["ticker", "record_date"]).any()
    assert raw["source_url"].str.startswith("https://m.samsungfund.com/").all()
    assert parsed["distribution_per_share"].gt(0).all()
    assert parsed["taxable_per_share"].le(parsed["distribution_per_share"]).all()


def test_factor_distribution_hash_matches_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == manifest["sha256"]
