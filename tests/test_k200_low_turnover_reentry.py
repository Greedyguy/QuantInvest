import pandas as pd

from scripts.audit_strategy_validation import select_enriched_cache_paths
from scripts.backtest_k200_reentry import load_naver_prices, merge_adjusted_prices
from strategies import get_strategy
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


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


def test_backtest_trades_only_on_state_changes_and_taxes_losing_sell():
    prices = _monthly_synthetic_prices()
    strategy = _short_window_strategy()

    equity, trades = strategy.run_backtest({"069500": prices}, silent=True)

    assert not equity.empty
    assert [trade["action"] for trade in trades] == ["BUY", "SELL"]
    assert trades[-1]["price"] < trades[0]["price"]
    assert trades[-1]["tax"] > 0.0
    assert len(trades) == int(equity["state"].ne(equity["state"].shift()).sum() - 1)


def test_strategy_is_registered():
    assert isinstance(get_strategy("k200_low_turnover_reentry"), K200LowTurnoverReentry)


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
