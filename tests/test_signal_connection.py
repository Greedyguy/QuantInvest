"""Signal replay must preserve holdings, real exits, and legacy backtests."""
import numpy as np
import pandas as pd
import pytest

from strategies import get_strategy
from strategies.strategy_multi_allocator import MultiStrategyAllocator


def prices(length=126):
    index = pd.bdate_range('2025-01-02', periods=length)
    return pd.DataFrame(dict(open=10000., high=10000., low=10000., close=10000.,
                             volume=1000000., value=1e10), index=index)


def setup_strategy(name):
    strategy = get_strategy(name)
    if name.startswith('kqm_small'):
        strategy._compute_factors = lambda df, date: dict(
            mom3=.1, mom1=.05, quality=.1, inv_vol=20.,
            rs_raw=1., bo=1., vcp=1.)
    if name == 'kqm_small_cap_v22_short':
        strategy.max_hold_days = 1000
    if name == 'etf_defensive_safe':
        strategy._select_target = lambda universe, date, state: '069500'
    if name == 'kqm':
        strategy._compute_factors = lambda df, date: dict(val_ma20=1e12, inv_vol=20., close=10000.)
        strategy._calculate_factor_score = lambda frame: pd.Series(1., index=frame.index)
        # One stock at exactly 100% has no room for fees in the native code;
        # use two candidate stocks so this tests signal delivery, not sizing.
        strategy.holdings_count = 1
    if name == 'k200_mean_rev':
        strategy._compute_indicators = lambda df: df.assign(ma20=10000., std20=10., z20=-2., ret5=-.05)
        strategy.max_hold_days = 1000
    return strategy


@pytest.mark.parametrize('name', ['kqm_small_cap_v22', 'kqm_small_cap_v22_short',
                                'etf_defensive_safe', 'kqm', 'k200_mean_rev'])
def test_endpoint_delivers_holdings_and_repeat_resets_history(name):
    data = {'069500': prices(), '0001A0': prices()}
    strategy = setup_strategy(name)
    equity, trades = strategy.run_signal_backtest(data, market='KR')
    weights = strategy.get_target_weight_history()
    assert weights.index[-1] == data['069500'].index[-1]
    assert weights.iloc[-1].drop('__CASH__').sum() > .1
    assert np.allclose(weights.sum(axis=1), 1.)
    assert not any(t.get('reason') in ('final', 'final_liq', 'FORCE_END') for t in trades)
    strategy.run_signal_backtest({t:f.iloc[:-1] for t,f in data.items()}, market='KR')
    assert strategy.get_target_weight_history().index.max() == data['069500'].index[-2]
    assert not strategy._signal_generation


@pytest.mark.parametrize('name', ['kqm_small_cap_v22', 'kqm_small_cap_v22_short',
                                'etf_defensive_safe', 'k200_mean_rev'])
def test_legacy_still_liquidates_and_signal_prefix_is_causal(name):
    frame = prices(129)
    full = {'069500': frame}
    old = setup_strategy(name)
    eq, _ = old.run_backtest(full, silent=True)
    if name != 'k200_mean_rev':
        assert old.get_target_weight_history().iloc[-1]['__CASH__'] == pytest.approx(1.)
    repaired = setup_strategy(name)
    repaired.run_signal_backtest(full, market='KR')
    prefix = setup_strategy(name)
    prefix.run_signal_backtest({'069500': frame.iloc[:-1]}, market='KR')
    expected = repaired.get_target_weight_history().loc[frame.index[-2]]
    pd.testing.assert_series_equal(expected, prefix.get_target_weight_history().iloc[-1])


def test_short_real_stop_still_runs_on_cutoff():
    frame = prices(124)
    frame.loc[frame.index[-1], ['open', 'high', 'low', 'close']] = 8000.
    strategy = setup_strategy('kqm_small_cap_v22_short')
    _, trades = strategy.run_signal_backtest({'069500': frame}, market='KR')
    assert any(t.get('reason') == 'stop_loss' and t['date'] == frame.index[-1] for t in trades)
    assert strategy.get_target_weight_history().iloc[-1]['__CASH__'] == pytest.approx(1.)


def test_mean_reversion_real_exit_still_runs():
    frame = prices(126)
    strategy = setup_strategy('k200_mean_rev')
    strategy._compute_indicators = lambda df: df.assign(ma20=10000., std20=10.,
        z20=np.r_[np.full(len(df)-2, -2.), 1., 1.],
        ret5=np.r_[np.full(len(df)-2, -.05), 0., 0.])
    _, trades = strategy.run_signal_backtest({'069500': frame}, market='KR')
    assert any(t.get('reason') == 'Z_REVERT' for t in trades)
    assert strategy.get_target_weight_history().iloc[-1]['__CASH__'] == pytest.approx(1.)


def test_allocator_opts_in_to_direct_weights_not_trade_conversion(monkeypatch):
    strategy = MultiStrategyAllocator(strategy_configs=[{'name':'kqm', 'weight':1.}],
                                     child_signal_mode=True, signal_market='KR')
    monkeypatch.setattr(strategy, '_get_strategy', setup_strategy)
    result = strategy._run_child_strategies({'069500':prices(), '0001A0':prices()}, None)
    assert result['kqm']['weights'].iloc[-1].drop('__CASH__').sum() > .1
    assert MultiStrategyAllocator().child_signal_mode is False


def test_explicit_kr_overrides_alphanumeric_heuristic():
    strategy = get_strategy('k200_trend_sleeve')
    strategy.run_signal_backtest({'0001A0': prices(), '069500': prices()}, market='KR')
    assert strategy.ticker == '069500'
    assert '_is_us_market' not in strategy.__dict__


def test_signal_context_restored_on_exception(monkeypatch):
    strategy = get_strategy('k200_mean_rev')
    def fail(*args, **kwargs):
        assert strategy._signal_generation
        raise RuntimeError('test')
    monkeypatch.setattr(strategy, 'run_backtest', fail)
    with pytest.raises(RuntimeError):
        strategy.run_signal_backtest({}, market='KR')
    assert strategy._signal_generation is False
    assert '_is_us_market' not in strategy.__dict__


def test_invalid_market_rejected():
    with pytest.raises(ValueError):
        get_strategy('kqm').run_signal_backtest({}, market='invalid')
