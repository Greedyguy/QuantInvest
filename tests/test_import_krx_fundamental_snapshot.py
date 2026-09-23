from pathlib import Path

import pandas as pd
import pytest

from scripts.import_krx_fundamental_snapshot import import_snapshot


def _write_inputs(root: Path) -> tuple[Path, Path]:
    fundamental_path = root / "fundamentals.csv"
    trading_path = root / "trading.csv"
    pd.DataFrame(
        {
            "종목코드": ["005930"],
            "종목명": ["삼성전자"],
            "BPS": ["30,000"],
            "EPS": ["1,000"],
            "PER": ["50"],
            "PBR": ["1.67"],
            "DPS": ["1,000"],
            "배당수익률": ["2.0"],
        }
    ).to_csv(fundamental_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "종목코드": ["005930"],
            "시장구분": ["KOSPI"],
            "종목명": ["삼성전자"],
            "종가": ["50,000"],
            "시가총액": ["300,000,000,000,000"],
            "거래대금": ["1,000,000,000,000"],
        }
    ).to_csv(trading_path, index=False, encoding="utf-8-sig")
    return fundamental_path, trading_path


def test_importer_writes_data_and_hash_manifest(tmp_path):
    fundamentals, trading = _write_inputs(tmp_path)
    data_path, manifest_path = import_snapshot(
        fundamentals,
        trading,
        snapshot_date="2020-01-31",
        available_date="2020-02-03",
        output_dir=tmp_path / "normalized",
    )

    assert data_path.exists()
    assert manifest_path.exists()
    loaded = pd.read_parquet(data_path)
    assert loaded.loc[0, "ticker"] == "005930"
    assert '"sha256"' in manifest_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        import_snapshot(
            fundamentals,
            trading,
            snapshot_date="2020-01-31",
            available_date="2020-02-03",
            output_dir=tmp_path / "normalized",
        )


def test_importer_refuses_to_open_sealed_holdout(tmp_path):
    fundamentals, trading = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="sealed holdout"):
        import_snapshot(
            fundamentals,
            trading,
            snapshot_date="2017-12-28",
            available_date="2018-01-02",
            output_dir=tmp_path / "normalized",
        )
