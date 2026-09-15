import json
from pathlib import Path

import pandas as pd
import pytest

from market_benchmark import (
    MarketOutperformanceCriteria,
    adjust_ohlc_for_distributions,
    evaluate_market_outperformance,
    load_distribution_events,
    load_samsung_distribution_json,
    prepare_distribution_schedule,
    restore_actual_ohlc,
    reconstruct_actual_ohlc_from_adjusted,
)
from scripts.audit_strategy_validation import select_enriched_cache_paths
from scripts.backtest_k200_reentry import load_naver_prices, merge_adjusted_prices
from strategies import get_strategy
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


def test_kodex200_distribution_reference_covers_full_development_period():
    reference = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "reference"
        / "kodex200_distributions.csv"
    )
    events = load_distribution_events(reference)
    amounts = events.set_index("record_date")["distribution_per_share"]

    assert amounts.loc[pd.Timestamp("2018-04-30")] == 460
    assert amounts.loc[pd.Timestamp("2019-10-31")] == 70
    assert events["record_date"].min() <= pd.Timestamp("2018-04-30")


def _monthly_synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", "2025-04-10")
    closes = []
    for date in dates:
        if date.month == 1:
            closes.append(100.0)
        elif date.month == 2:
            closes.append(100.0 + len(closes) * 0.8)
        elif date.month == 3:
            closes.append(130.0 if date.day < 27 else 75.0)
        else:
            closes.append(74.0)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        },
        index=dates,
    )


def _short_window_strategy() -> K200LowTurnoverReentry:
    return K200LowTurnoverReentry(
        trend_window=5,
        momentum_window=3,
        fast_trend_window=2,
        medium_trend_window=4,
        entry_buffer=0.0,
        exit_buffer=0.01,
        exit_momentum=-0.03,
    )


def test_state_history_is_prefix_invariant():
    prices = _monthly_synthetic_prices()
    strategy = _short_window_strategy()

    prefix = strategy.compute_state_history(prices.iloc[:50])
    full = strategy.compute_state_history(prices)

    pd.testing.assert_frame_equal(prefix, full.loc[prefix.index])


def test_entries_only_happen_at_first_trading_day_of_month():
    prices = _monthly_synthetic_prices()
    strategy = _short_window_strategy()
    states = strategy.compute_state_history(prices)
    entries = states.loc[
        states["state"].eq(strategy.RISK_ON)
        & states["state"].ne(states["state"].shift())
    ]

    assert not entries.empty
    for date in entries.index:
        prior_date = states.index[states.index.get_loc(date) - 1]
        assert date.month != prior_date.month


def test_backtest_trades_only_on_state_changes_and_does_not_tax_etf_sell():
    prices = _monthly_synthetic_prices()
    strategy = _short_window_strategy()

    equity, trades = strategy.run_backtest({"069500": prices}, silent=True)

    assert not equity.empty
    assert [trade["action"] for trade in trades] == ["BUY", "SELL"]
    assert trades[-1]["price"] < trades[0]["price"]
    assert trades[-1]["tax"] == 0.0
    assert len(trades) == int(equity["state"].ne(equity["state"].shift()).sum() - 1)


def test_strategy_is_registered():
    assert isinstance(get_strategy("k200_low_turnover_reentry"), K200LowTurnoverReentry)


def test_distribution_schedule_uses_settlement_lag_and_net_taxable_amount():
    dates = pd.bdate_range("2025-01-27", "2025-02-05")
    events = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2025-01-31")],
            "pay_date": [pd.Timestamp("2025-02-04")],
            "distribution_per_share": [100.0],
            "taxable_per_share": [80.0],
        }
    )

    schedule = prepare_distribution_schedule(events, dates)

    assert schedule.iloc[0]["entitlement_date"] == pd.Timestamp("2025-01-29")
    assert schedule.iloc[0]["credit_date"] == pd.Timestamp("2025-02-04")
    assert schedule.iloc[0]["tax_unit"] == pytest.approx(12.32)
    assert schedule.iloc[0]["net_unit"] == pytest.approx(87.68)


def test_distribution_schedule_keeps_nearby_post_horizon_payment_as_receivable():
    dates = pd.bdate_range("2022-12-20", "2022-12-29")
    events = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2022-12-31")],
            "pay_date": [pd.Timestamp("2023-04-14")],
            "distribution_per_share": [100.0],
            "taxable_per_share": [100.0],
        }
    )

    schedule = prepare_distribution_schedule(events, dates)

    assert len(schedule) == 1
    assert schedule.iloc[0]["entitlement_date"] == pd.Timestamp("2022-12-27")
    assert schedule.iloc[0]["credit_date"] == pd.Timestamp("2023-04-14")


def test_distribution_schedule_does_not_pull_distant_future_event_backwards():
    dates = pd.bdate_range("2022-12-20", "2022-12-29")
    events = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2023-01-31")],
            "pay_date": [pd.Timestamp("2023-02-02")],
            "distribution_per_share": [100.0],
            "taxable_per_share": [100.0],
        }
    )

    assert prepare_distribution_schedule(events, dates).empty


def test_samsung_distribution_json_loader_maps_official_fields(tmp_path):
    response = tmp_path / "distribution.json"
    response.write_text(
        json.dumps(
            {
                "dividList": [
                    {
                        "basicD": "20250430",
                        "payD": "20250507",
                        "dividA": "100",
                        "taxDividA": "80",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = load_samsung_distribution_json(response)

    assert result.iloc[0]["record_date"] == pd.Timestamp("2025-04-30")
    assert result.iloc[0]["pay_date"] == pd.Timestamp("2025-05-07")
    assert result.iloc[0]["distribution_per_share"] == 100
    assert result.iloc[0]["taxable_per_share"] == 80


def test_strategy_credits_net_distribution_only_when_eligible():
    prices = _monthly_synthetic_prices()
    events = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2025-03-20")],
            "pay_date": [pd.Timestamp("2025-03-24")],
            "distribution_per_share": [100.0],
            "taxable_per_share": [100.0],
        }
    )
    strategy = K200LowTurnoverReentry(
        trend_window=5,
        momentum_window=3,
        fast_trend_window=2,
        medium_trend_window=4,
        entry_buffer=0.0,
        exit_buffer=0.01,
        exit_momentum=-0.03,
        execution_prices=prices,
        distribution_events=events,
    )

    _, trades = strategy.run_backtest({"069500": prices}, silent=True)

    payments = [trade for trade in trades if trade["action"] == "DISTRIBUTION"]
    assert len(payments) == 1
    assert payments[0]["date"] == pd.Timestamp("2025-03-24")
    assert payments[0]["tax"] == pytest.approx(
        payments[0]["qty"] * 100.0 * 0.154
    )


def test_restore_actual_ohlc_keeps_signals_separate_from_execution_scale():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    adjusted = pd.DataFrame(
        {"open": [90.0, 99.0], "high": [101.0, 111.0], "low": [89.0, 98.0], "close": [100.0, 110.0]},
        index=dates,
    )
    official_close = pd.Series([120.0, 132.0], index=dates)

    actual = restore_actual_ohlc(adjusted, official_close)

    assert actual.loc[dates[0], "open"] == pytest.approx(108.0)
    assert actual.loc[dates[1], "close"] == pytest.approx(132.0)


def test_distribution_adjustment_removes_ex_distribution_price_gap():
    dates = pd.bdate_range("2025-01-27", "2025-02-05")
    close = pd.Series(100.0, index=dates)
    close.loc[dates >= pd.Timestamp("2025-01-30")] = 90.0
    prices = pd.DataFrame({"open": close, "close": close})
    events = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2025-01-31")],
            "pay_date": [pd.Timestamp("2025-02-04")],
            "distribution_per_share": [10.0],
            "taxable_per_share": [10.0],
        }
    )

    adjusted = adjust_ohlc_for_distributions(prices, events)

    assert adjusted.loc[pd.Timestamp("2025-01-29"), "close"] == pytest.approx(90.0)
    assert adjusted.loc[pd.Timestamp("2025-01-30"), "close"] == pytest.approx(90.0)


def test_distribution_adjustment_can_be_reversed_for_provisional_execution():
    dates = pd.bdate_range("2025-01-27", "2025-02-05")
    close = pd.Series(100.0, index=dates)
    close.loc[dates >= pd.Timestamp("2025-01-30")] = 90.0
    actual = pd.DataFrame({"open": close, "close": close})
    events = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2025-01-31")],
            "pay_date": [pd.Timestamp("2025-02-04")],
            "distribution_per_share": [10.0],
            "taxable_per_share": [10.0],
        }
    )

    adjusted = adjust_ohlc_for_distributions(actual, events)
    reconstructed = reconstruct_actual_ohlc_from_adjusted(adjusted, events)

    pd.testing.assert_frame_equal(reconstructed, actual)


def test_distribution_adjustment_does_not_change_frozen_prefix_states():
    prices = _monthly_synthetic_prices()
    prefix_end = prices.index[50]
    future_record = prices.index[-3]
    events = pd.DataFrame(
        {
            "record_date": [future_record],
            "pay_date": [prices.index[-1]],
            "distribution_per_share": [5.0],
            "taxable_per_share": [5.0],
        }
    )
    full_adjusted = adjust_ohlc_for_distributions(prices, events)
    strategy = _short_window_strategy()

    raw_prefix_states = strategy.compute_state_history(
        prices.loc[prices.index <= prefix_end]
    )["state"]
    adjusted_prefix_states = strategy.compute_state_history(
        full_adjusted.loc[full_adjusted.index <= prefix_end]
    )["state"]

    pd.testing.assert_series_equal(raw_prefix_states, adjusted_prefix_states)


def test_market_outperformance_requires_repeatable_excess_return():
    dates = pd.bdate_range("2020-01-02", periods=800)
    benchmark = pd.Series((1.0002 ** pd.RangeIndex(len(dates))).to_numpy(), index=dates)
    strategy = pd.Series((1.00035 ** pd.RangeIndex(len(dates))).to_numpy(), index=dates)
    result = evaluate_market_outperformance(
        strategy,
        benchmark,
        MarketOutperformanceCriteria(max_positive_excess_year_share=1.0),
    )

    assert result["annualised_excess_return_pct_point"] > 2.0
    assert result["rolling_12m_beat_rate_pct"] == 100.0
    assert result["passes_all_gates"]


def test_cache_selection_includes_etf_with_different_suffix(tmp_path):
    exact_stock = tmp_path / "005930_20190101_20251113.parquet"
    short_etf = tmp_path / "069500_20240603_20251127.parquet"
    long_etf = tmp_path / "069500_20200102_20251119.parquet"
    for path in (exact_stock, short_etf, long_etf):
        path.touch()

    selected = select_enriched_cache_paths(
        tmp_path, pd.Timestamp("2020-01-01"), pd.Timestamp("2025-11-13")
    )

    assert exact_stock in selected
    assert long_etf in selected
    assert short_etf not in selected


def test_naver_prices_are_scaled_to_historical_overlap(tmp_path):
    response = tmp_path / "naver.xml"
    response.write_bytes(
        b'<item data="20250102|50|51|49|50|1000" />\n'
        b'<item data="20250103|55|56|54|55|1100" />'
    )
    recent = load_naver_prices(response)
    historical = pd.DataFrame(
        {"open": [100.0], "close": [100.0]},
        index=pd.to_datetime(["2025-01-02"]),
    )

    merged, adjustment = merge_adjusted_prices(historical, recent)

    assert adjustment == 2.0
    assert merged.loc[pd.Timestamp("2025-01-03"), "close"] == 110.0
