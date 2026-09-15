import hashlib
import json

import pandas as pd
import pytest

from krx_execution_data import load_actual_close_panel
from scripts.import_krx_kodex200_shadow_prices import (
    build_official_kodex200_panel,
    import_official_kodex200_prices,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "일자": ["2020/01/03", "2020/01/02"],
            "종가": [29_460, 29_465],
            "시가": [29_725, 29_805],
            "고가": [29_890, 29_880],
            "저가": [29_305, 29_410],
        }
    )


def test_build_official_panel_preserves_actual_ohlc_and_fixed_start():
    result = build_official_kodex200_panel(
        _raw(), expected_end="2020-01-03"
    )

    assert result["ticker"].eq("069500").all()
    assert result["price_basis"].eq("actual_traded").all()
    assert result.iloc[0]["open"] == 29_805
    assert result.iloc[-1]["close"] == 29_460


def test_build_official_panel_fails_when_frozen_start_is_missing():
    with pytest.raises(ValueError, match="frozen history date"):
        build_official_kodex200_panel(_raw().iloc[:1])


def test_import_writes_hash_verified_immutable_manifest(tmp_path):
    raw_path = tmp_path / "krx.csv"
    output = tmp_path / "official.parquet"
    manifest_path = tmp_path / "official.manifest.json"
    _raw().to_csv(raw_path, index=False, encoding="cp949")

    import_official_kodex200_prices(
        raw_path,
        output,
        manifest_path,
        expected_end="2020-01-03",
    )

    loaded = load_actual_close_panel(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert manifest["history_coverage_start"] == "2020-01-02"
    assert manifest["history_coverage_end"] == "2020-01-03"
    assert manifest["normalized_file"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError, match="immutable"):
        import_official_kodex200_prices(raw_path, output, manifest_path)
