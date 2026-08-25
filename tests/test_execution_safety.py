from types import SimpleNamespace

import pandas as pd

from multi_allocator_plus_trader import MultiAllocatorPlusTrader, OrderPlan


def _bare_trader():
    trader = MultiAllocatorPlusTrader.__new__(MultiAllocatorPlusTrader)
    trader.execution_recheck = True
    trader.recheck_price_band_pct = 3.0
    trader.run_id = "test-run"
    trader.market = "kr"
    trader.dry_run = False
    trader.kis = SimpleNamespace(account="12345678-01")
    return trader


def test_sell_is_not_blocked_by_price_band_but_buy_is():
    trader = _bare_trader()
    trader._safe_get_current_price = lambda _symbol: 80.0
    trader._safe_get_orderable_qty = lambda _symbol, _price: 100
    account = {"available_cash": 1_000_000}

    sell = OrderPlan("069500", "SELL", 2, 100.0, 200.0, 0.0, 2, 0)
    reviewed, logs = trader.apply_execution_recheck([sell], account)
    assert len(reviewed) == 1
    assert reviewed[0].action == "SELL"
    assert logs[0]["reason"] == "sell_price_band_bypassed"

    buy = OrderPlan("069500", "BUY", 2, 100.0, 200.0, 0.2, 0, 2)
    reviewed, logs = trader.apply_execution_recheck([buy], account)
    assert reviewed == []
    assert logs[0]["reason"] == "price_band_exceeded"


def test_execution_summary_omits_account_number_and_exact_balances(tmp_path):
    trader = _bare_trader()
    trader.signal_mode = "eod_fixed"
    trader.virtual_account = False
    trader.enriched = {}
    trader.market_index = None
    trader.secondary_index = None
    trader.loaded_signal_snapshot_payload = {
        "style_attribution": {"signal_date": "2026-08-25"},
        "decision_context": {"signal_date": "2026-08-25", "exposure": {}},
        "meta": {"data_as_of": {"primary_index": "2026-08-25"}},
    }
    trader.strategy = SimpleNamespace()
    trader._execution_summary_path = lambda _date: tmp_path / "summary.json"
    account = {
        "account_no": "12345678-01",
        "total_value": 1_000_000,
        "available_cash": 400_000,
        "stock_value": 600_000,
    }
    holdings = {
        "069500": {
            "symbol": "069500",
            "name": "KODEX 200",
            "quantity": 10,
            "avg_price": 55_000,
            "current_price": 60_000,
            "market_value": 600_000,
            "unrealized_pnl": 50_000,
            "unrealized_pnl_rate": 9.09,
        }
    }
    plan = OrderPlan("069500", "SELL", 2, 60_000, 120_000, 0.48, 10, 8)
    path = trader.save_execution_summary(
        signal_date=pd.Timestamp("2026-08-25"),
        targets=pd.Series({"069500": 0.48, "__CASH__": 0.52}),
        account=account,
        holdings=holdings,
        raw_plans=[plan],
        plans=[plan],
        recheck_logs=[{"cash_before": 400_000, "decision": "send"}],
        execution_result={"status": "failed", "reason": "account 12345678-01"},
        completed_for_signal=False,
        planning_decisions=[],
    )

    text = path.read_text(encoding="utf-8")
    assert "12345678-01" not in text
    for sensitive_key in [
        '"account_no"', '"total_value"', '"available_cash"',
        '"stock_value"', '"market_value"', '"avg_price"',
        '"unrealized_pnl"', '"cash_before"', '"est_value"',
    ]:
        assert sensitive_key not in text
    assert '"account_allocation"' in text
    assert '"current_actual_exposure"' in text
