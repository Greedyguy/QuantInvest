from types import SimpleNamespace

import json
import pandas as pd

from multi_allocator_plus_trader import MultiAllocatorPlusTrader


def test_small_account_shadow_compares_policies_without_balance_leak(tmp_path):
    trader = MultiAllocatorPlusTrader.__new__(MultiAllocatorPlusTrader)
    trader.run_id = "shadow-test"
    trader.market = "kr"
    trader.dry_run = True
    trader.min_trade_value = 50_000
    trader.cash_policy = "preserve"
    trader.loaded_signal_snapshot_payload = {
        "decision_context": {"signal_date": "2026-08-25", "exposure": {}},
        "meta": {"data_as_of": {"primary_index": "2026-08-25"}},
    }
    trader.strategy = SimpleNamespace()
    trader.enriched = {}
    trader.market_index = None
    trader.secondary_index = None
    trader.kis = SimpleNamespace(account="12345678-01")
    trader._shadow_report_path = lambda _date: tmp_path / "shadow.json"

    targets = pd.Series({"069500": 0.62, "091160": 0.03, "__CASH__": 0.35})
    account = {
        "account_no": "12345678-01",
        "total_value": 1_000_000,
        "available_cash": 1_000_000,
        "stock_value": 0,
    }
    path = trader.run_small_account_shadow(
        pd.Timestamp("2026-08-25"),
        targets,
        account,
        {},
        {"069500": 400_000, "091160": 10_000},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution_guard"] == "NO_ORDERS_SENT"
    assert [row["policy"] for row in payload["comparisons"]] == [
        "floor_50k", "nearest_etf_50k", "nearest_etf_20k"
    ]
    floor_qty = payload["comparisons"][0]["plans"][0]["target_qty"]
    nearest_qty = payload["comparisons"][1]["plans"][0]["target_qty"]
    assert floor_qty == 1
    assert nearest_qty == 2

    text = path.read_text(encoding="utf-8")
    assert "12345678-01" not in text
    for sensitive_key in [
        '"account_no"', '"total_value"', '"available_cash"', '"stock_value"'
    ]:
        assert sensitive_key not in text


def test_shadow_failure_report_is_sanitized(tmp_path):
    trader = MultiAllocatorPlusTrader.__new__(MultiAllocatorPlusTrader)
    trader.run_id = "shadow-failure-test"
    trader.market = "kr"
    trader.loaded_signal_snapshot_payload = {"meta": {"source": "test"}}
    trader.kis = SimpleNamespace(account="12345678-01")
    trader._shadow_report_path = lambda _date: tmp_path / "shadow-failure.json"

    path = trader.save_shadow_failure(
        pd.Timestamp("2026-08-31"),
        RuntimeError("계좌 12345678-01 잔고 조회 실패"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["execution_guard"] == "NO_ORDERS_SENT"
    assert "12345678-01" not in path.read_text(encoding="utf-8")
    assert "[REDACTED_ACCOUNT]" in payload["reason"]
