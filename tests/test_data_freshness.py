import pandas as pd
import pytest

from data_loader import get_index_close, validate_market_data_freshness


def test_market_data_freshness_records_asof_and_rejects_stale_data():
    frame = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-08-21", "2026-08-24"]),
    )
    metadata = validate_market_data_freshness(
        frame, "2026-08-26", "KOSDAQ index", tolerance_bdays=3
    )
    assert metadata["last_date"] == "2026-08-24"
    assert metadata["business_day_gap"] <= 3

    with pytest.raises(RuntimeError, match="오래되었습니다"):
        validate_market_data_freshness(
            frame, "2026-09-04", "KOSDAQ index", tolerance_bdays=3
        )


def test_index_download_includes_requested_end_date(monkeypatch):
    import yfinance as yf

    captured = {}

    def fake_download(_symbol, start, end, auto_adjust, progress):
        captured["start"] = pd.Timestamp(start)
        captured["end"] = pd.Timestamp(end)
        return pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-08-26"]),
        )

    monkeypatch.setattr(yf, "download", fake_download)
    result = get_index_close("KOSDAQ", "2026-08-01", "2026-08-26")

    assert captured["end"] == pd.Timestamp("2026-08-27")
    assert result.index.max() == pd.Timestamp("2026-08-26")
