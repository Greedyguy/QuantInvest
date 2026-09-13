from types import SimpleNamespace

import json
import pandas as pd
import pytest

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
        {"069500": 400_000, "091160": 10_000, "423160": 110_000},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution_guard"] == "NO_ORDERS_SENT"
    assert [row["policy"] for row in payload["comparisons"]] == [
        "floor_50k",
        "nearest_etf_50k",
        "nearest_etf_20k",
        "kofr_parking_50k_shadow",
    ]
    floor_qty = payload["comparisons"][0]["plans"][0]["target_qty"]
    nearest_qty = payload["comparisons"][1]["plans"][0]["target_qty"]
    assert floor_qty == 1
    assert nearest_qty == 2
    parking = payload["comparisons"][-1]
    assert parking["cash_parking"]["status"] == "ready"
    assert parking["cash_parking"]["execution_guard"] == "SHADOW_ONLY"
    assert parking["cash_parking"]["target_quantity"] == 0

    text = path.read_text(encoding="utf-8")
    assert "12345678-01" not in text
    for sensitive_key in [
        '"account_no"', '"total_value"', '"available_cash"', '"stock_value"'
    ]:
        assert sensitive_key not in text


def test_cash_parking_shadow_keeps_liquid_reserve_and_excludes_kofr_from_risk_exposure():
    trader = MultiAllocatorPlusTrader.__new__(MultiAllocatorPlusTrader)
    trader.run_id = "cash-parking-test"
    trader.market = "kr"
    trader.dry_run = True
    trader.min_trade_value = 50_000
    trader.cash_policy = "preserve"
    trader.loaded_signal_snapshot_payload = {
        "decision_context": {"signal_date": "2026-09-10", "exposure": {}},
        "meta": {},
    }
    trader.strategy = SimpleNamespace()
    trader.enriched = {}
    trader.market_index = None
    trader.secondary_index = None
    trader.kis = SimpleNamespace(account="12345678-01")

    targets = pd.Series({"069500": 0.18, "__CASH__": 0.82})
    account = {
        "total_value": 1_000_000,
        "available_cash": 1_000_000,
        "stock_value": 0,
    }
    parked, metadata, prices = trader._cash_parking_shadow_targets(
        targets,
        account,
        {},
        {"069500": 70_000, "423160": 110_000},
    )

    assert metadata["status"] == "ready"
    assert metadata["target_quantity"] == 4
    assert parked["423160"] == pytest.approx(0.44)
    assert parked["__CASH__"] == pytest.approx(0.38)

    plans, decisions = trader.build_order_plan(
        parked,
        account,
        {},
        price_cache_override=prices,
        min_trade_value_override=50_000,
        return_decisions=True,
    )
    exposure = trader._exposure_diagnostics(
        parked,
        account,
        {},
        plans,
        price_cache_override=prices,
        planning_decisions=decisions,
        cash_equivalent_tickers={"423160"},
    )

    assert any(plan.symbol == "423160" and plan.quantity == 4 for plan in plans)
    buy_symbols = [plan.symbol for plan in plans if plan.action == "BUY"]
    assert buy_symbols.index("423160") > buy_symbols.index("069500")
    assert exposure["target_exposure"] == pytest.approx(0.18)
    assert exposure["executable_exposure"] == pytest.approx(0.14)


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
