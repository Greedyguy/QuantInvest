from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_k200_reentry_shadow import build_shadow_payload


def test_shadow_report_is_sanitized_and_never_plans_orders():
    dates = pd.bdate_range("2026-01-02", periods=180)
    close = 100.0 * np.cumprod(np.repeat(1.001, len(dates)))
    prices = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )

    payload = build_shadow_payload(
        prices,
        source_path=Path("069500_test.parquet"),
        generated_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )

    assert payload["execution_guard"] == "NO_ORDERS_SENT"
    assert set(payload["target_weights_effective_at_signal_open"]) == {
        "069500",
        "__CASH__",
    }
    assert "account" not in payload
    assert "orders" not in payload
    assert payload["source"]["filename"] == "069500_test.parquet"
