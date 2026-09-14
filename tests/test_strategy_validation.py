import numpy as np
import pandas as pd
import pytest

from backtest_live_execution import _build_orders, simulate
from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus
from strategy_validation import (
    ValidationPeriod,
    apply_turnover_cap,
    cap_security_weights_to_cash,
    causal_volatility_scale,
    combine_child_targets_preserving_cash,
    performance_by_period,
)


def test_child_cash_is_preserved_instead_of_scaled_to_outer_exposure():
    dates = pd.bdate_range("2025-01-02", periods=2)
    child = pd.DataFrame({"AAA": [0.20, 0.20], "__CASH__": [0.80, 0.80]}, index=dates)
    meta = pd.DataFrame({"child": 1.0}, index=dates)
    exposure = pd.Series(0.90, index=dates)

    result = combine_child_targets_preserving_cash(
        {"child": child}, meta, exposure
    )

    assert result.loc[dates[0], "AAA"] == pytest.approx(0.20)
    assert result.loc[dates[0], "__CASH__"] == pytest.approx(0.80)


def test_child_return_input_deduplicates_and_sorts_dates():
    strategy = MultiStrategyAllocatorPlus()
    dates = pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-03"])
    equity = pd.DataFrame({"equity": [100.0, 90.0, 110.0]}, index=dates)

    result = strategy._compute_daily_returns(equity)

    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.loc[pd.Timestamp("2025-01-03")] == pytest.approx(110.0 / 90.0 - 1.0)


def test_outer_exposure_only_scales_risk_down():
    dates = pd.bdate_range("2025-01-02", periods=1)
    child = pd.DataFrame({"AAA": [0.60], "BBB": [0.40]}, index=dates)
    meta = pd.DataFrame({"child": [1.0]}, index=dates)

    result = combine_child_targets_preserving_cash(
        {"child": child}, meta, pd.Series(0.30, index=dates)
    )

    assert result.loc[dates[0], "AAA"] == pytest.approx(0.18)
    assert result.loc[dates[0], "BBB"] == pytest.approx(0.12)
    assert result.loc[dates[0], "__CASH__"] == pytest.approx(0.70)


def test_security_cap_and_turnover_excess_both_remain_cash():
    dates = pd.bdate_range("2025-01-02", periods=3)
    desired = pd.DataFrame(
        {"AAA": [0.80, 0.80, 0.00], "__CASH__": [0.20, 0.20, 1.00]},
        index=dates,
    )
    capped = cap_security_weights_to_cash(desired, 0.25)
    smoothed = apply_turnover_cap(capped, 0.10)

    assert capped.iloc[0]["AAA"] == pytest.approx(0.25)
    assert capped.iloc[0]["__CASH__"] == pytest.approx(0.75)
    assert smoothed.iloc[0]["AAA"] == pytest.approx(0.10)
    assert smoothed.iloc[1]["AAA"] == pytest.approx(0.20)
    assert smoothed.sum(axis=1).eq(1.0).all()


def test_turnover_cap_is_prefix_invariant():
    dates = pd.bdate_range("2025-01-02", periods=8)
    desired = pd.DataFrame(
        {
            "AAA": [0.7, 0.7, 0.1, 0.5, 0.0, 0.6, 0.2, 0.4],
            "__CASH__": [0.3, 0.3, 0.9, 0.5, 1.0, 0.4, 0.8, 0.6],
        },
        index=dates,
    )

    prefix = apply_turnover_cap(desired.iloc[:5], 0.12)
    full = apply_turnover_cap(desired, 0.12)

    pd.testing.assert_frame_equal(prefix, full.iloc[:5])


def test_causal_volatility_scale_is_prefix_invariant_and_ignores_current_shock():
    dates = pd.bdate_range("2025-01-02", periods=100)
    returns = pd.Series(np.r_[np.full(80, 0.002), np.full(20, -0.01)], index=dates)

    prefix = causal_volatility_scale(returns.iloc[:85], 0.10, window=20, min_periods=10)
    full = causal_volatility_scale(returns, 0.10, window=20, min_periods=10)

    pd.testing.assert_series_equal(prefix, full.iloc[:85])
    shocked = returns.copy()
    shocked.iloc[60] = -0.20
    scale = causal_volatility_scale(shocked, 0.10, window=20, min_periods=10)
    assert scale.iloc[60] == pytest.approx(scale.iloc[59])
    assert scale.iloc[61] < scale.iloc[60]


def test_period_metrics_keep_first_return_after_boundary():
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2025-01-02"])
    equity = pd.Series([100.0, 110.0, 99.0, 108.9], index=dates)
    periods = [
        ValidationPeriod("validation", "2024-01-01", "2024-12-31"),
        ValidationPeriod("test", "2025-01-01", "2025-12-31"),
    ]

    result = performance_by_period(equity, periods)

    assert result.loc["validation", "return_pct"] == pytest.approx(-1.0)
    assert result.loc["test", "return_pct"] == pytest.approx(10.0)


def test_execution_bypasses_price_band_for_sells_but_not_buys():
    dates = pd.bdate_range("2025-01-02", periods=2)
    prices = pd.DataFrame(
        {"close": [100.0, 80.0], "open": [100.0, 80.0]}, index=dates
    )
    enriched = {"AAA": prices}

    sell_orders = _build_orders(
        dates[0], dates[1], pd.Series({"AAA": 0.5}), 0.0, {"AAA": 10},
        enriched, 0, 3.0,
    )
    buy_orders = _build_orders(
        dates[0], dates[1], pd.Series({"AAA": 0.5}), 1_000.0, {},
        enriched, 0, 3.0,
    )

    assert sell_orders[0].action == "SELL"
    assert buy_orders[0].action == "SKIP"
    assert buy_orders[0].reason == "price_band_exceeded"


def test_execution_uses_integer_shares_and_charges_every_sell():
    dates = pd.bdate_range("2025-01-02", periods=3)
    prices = pd.DataFrame(
        {"close": [50_000.0] * 3, "open": [50_000.0] * 3}, index=dates
    )
    targets = pd.DataFrame(
        {"AAA": [0.50, 0.00, 0.00], "__CASH__": [0.50, 1.00, 1.00]},
        index=dates,
    )

    equity, trades = simulate(
        targets,
        {"AAA": prices},
        initial_cash=210_000.0,
        min_trade=50_000,
        price_band_pct=3.0,
        blocked_tickers=set(),
    )

    executed = [trade for trade in trades if trade["action"] != "SKIP"]
    assert [trade["final_qty"] for trade in executed] == [2, 2]
    assert [trade["action"] for trade in executed] == ["BUY", "SELL"]
    assert equity.index[0] == dates[0]
    assert equity.iloc[-1]["equity"] < 210_000.0


def test_execution_can_exempt_domestic_equity_etf_sell_tax():
    dates = pd.bdate_range("2025-01-02", periods=3)
    prices = pd.DataFrame(
        {"close": [50_000.0] * 3, "open": [50_000.0] * 3}, index=dates
    )
    targets = pd.DataFrame(
        {"069500": [0.50, 0.00, 0.00], "__CASH__": [0.50, 1.00, 1.00]},
        index=dates,
    )

    taxed, _ = simulate(
        targets,
        {"069500": prices},
        initial_cash=210_000.0,
        min_trade=50_000,
        price_band_pct=3.0,
        blocked_tickers=set(),
    )
    exempt, _ = simulate(
        targets,
        {"069500": prices},
        initial_cash=210_000.0,
        min_trade=50_000,
        price_band_pct=3.0,
        blocked_tickers=set(),
        sell_tax_rate_by_ticker={"069500": 0.0},
    )

    assert exempt.iloc[-1]["equity"] > taxed.iloc[-1]["equity"]


def test_execution_can_resolve_historical_sell_tax_by_date():
    dates = pd.bdate_range("2020-01-02", periods=3)
    prices = pd.DataFrame(
        {"close": [50_000.0] * 3, "open": [50_000.0] * 3}, index=dates
    )
    targets = pd.DataFrame(
        {"AAA": [0.50, 0.00, 0.00], "__CASH__": [0.50, 1.00, 1.00]},
        index=dates,
    )
    calls = []

    _, trades = simulate(
        targets,
        {"AAA": prices},
        initial_cash=210_000.0,
        min_trade=50_000,
        price_band_pct=3.0,
        blocked_tickers=set(),
        sell_tax_rate_resolver=lambda ticker, date: calls.append((ticker, date))
        or 0.0025,
    )

    sell = next(trade for trade in trades if trade["action"] == "SELL")
    assert calls == [("AAA", dates[2])]
    assert sell["tax"] == pytest.approx(
        sell["final_qty"] * sell["exec_price"] * 0.0025
    )


def test_execution_can_avoid_daily_weight_maintenance_churn():
    dates = pd.bdate_range("2025-01-02", periods=5)
    prices = pd.DataFrame(
        {"close": [50_000.0, 55_000.0, 70_000.0, 90_000.0, 100_000.0],
         "open": [50_000.0, 55_000.0, 70_000.0, 90_000.0, 100_000.0]},
        index=dates,
    )
    targets = pd.DataFrame(
        {"AAA": [0.50] * 5, "__CASH__": [0.50] * 5}, index=dates
    )

    _, trades = simulate(
        targets,
        {"AAA": prices},
        initial_cash=210_000.0,
        min_trade=0,
        price_band_pct=100.0,
        blocked_tickers=set(),
        rebalance_only_on_target_change=True,
    )

    executed = [trade for trade in trades if trade["action"] != "SKIP"]
    assert [trade["action"] for trade in executed] == ["BUY"]


def test_execution_credits_distribution_to_entitled_integer_shares():
    dates = pd.bdate_range("2025-01-27", "2025-02-05")
    prices = pd.DataFrame(
        {"close": [50_000.0] * len(dates), "open": [50_000.0] * len(dates)},
        index=dates,
    )
    targets = pd.DataFrame(
        {"069500": [0.50] * len(dates), "__CASH__": [0.50] * len(dates)},
        index=dates,
    )
    distributions = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2025-01-31")],
            "pay_date": [pd.Timestamp("2025-02-04")],
            "distribution_per_share": [100.0],
            "taxable_per_share": [80.0],
        }
    )

    equity, trades = simulate(
        targets,
        {"069500": prices},
        initial_cash=210_000.0,
        min_trade=0,
        price_band_pct=3.0,
        blocked_tickers=set(),
        sell_tax_rate_by_ticker={"069500": 0.0},
        rebalance_only_on_target_change=True,
        distribution_events_by_ticker={"069500": distributions},
    )

    payments = [trade for trade in trades if trade["action"] == "DISTRIBUTION"]
    assert len(payments) == 1
    assert payments[0]["final_qty"] == 2
    assert payments[0]["tax"] == pytest.approx(2 * 80.0 * 0.154)
    prior_cash = equity.loc[pd.Timestamp("2025-02-03"), "cash"]
    paid_cash = equity.loc[pd.Timestamp("2025-02-04"), "cash"]
    assert paid_cash - prior_cash == pytest.approx(2 * (100.0 - 80.0 * 0.154))


def test_execution_values_post_horizon_distribution_receivable():
    dates = pd.bdate_range("2022-12-20", "2022-12-29")
    prices = pd.DataFrame(
        {"close": [50_000.0] * len(dates), "open": [50_000.0] * len(dates)},
        index=dates,
    )
    targets = pd.DataFrame(
        {"005930": [0.50] * len(dates), "__CASH__": [0.50] * len(dates)},
        index=dates,
    )
    distributions = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2022-12-31")],
            "pay_date": [pd.Timestamp("2023-04-14")],
            "distribution_per_share": [100.0],
            "taxable_per_share": [100.0],
        }
    )

    equity, trades = simulate(
        targets,
        {"005930": prices},
        initial_cash=210_000.0,
        min_trade=0,
        price_band_pct=3.0,
        blocked_tickers=set(),
        sell_tax_rate_by_ticker={"005930": 0.0},
        rebalance_only_on_target_change=True,
        distribution_events_by_ticker={"005930": distributions},
    )

    expected = 2 * 100.0 * (1.0 - 0.154)
    assert equity.loc[pd.Timestamp("2022-12-27"), "distribution_receivable"] == pytest.approx(expected)
    assert equity.loc[pd.Timestamp("2022-12-29"), "distribution_receivable"] == pytest.approx(expected)
    assert not [trade for trade in trades if trade["action"] == "DISTRIBUTION"]
