from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from strategies import get_strategy
from scripts.backtest_kodex_sector_rotation import _validate_sealed_holdout
from strategies.kodex_sector_rotation import KodexSectorRotation


def _synthetic_data(periods: int = 430):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    core_close = 30_000.0 * np.exp(np.arange(periods) * 0.0006)
    execution = {
        "069500": pd.DataFrame(
            {"open": core_close, "close": core_close}, index=dates
        )
    }
    weekly_dates = dates[::5]
    total_return = {"069500": pd.Series(1.0, index=weekly_dates)}
    for rank in range(1, 5):
        ticker = f"09{rank:04d}"
        close = 10_000.0 * np.exp(np.arange(periods) * (0.0002 + rank * 0.0001))
        execution[ticker] = pd.DataFrame(
            {"open": close, "close": close}, index=dates
        )
        total_return[ticker] = pd.Series(
            np.exp(np.arange(len(weekly_dates)) * (0.001 + rank * 0.001)),
            index=weekly_dates,
        )
    return execution, pd.DataFrame(total_return), execution["069500"]


def test_sector_rotation_targets_are_prefix_invariant():
    execution, total_return, signal_core = _synthetic_data()
    strategy = KodexSectorRotation(
        total_return_indices=total_return,
        signal_prices=signal_core,
        short_momentum_weeks=8,
        long_momentum_weeks=16,
    )
    cutoff = execution["069500"].index[370]
    prefix_execution = {
        ticker: frame.loc[frame.index <= cutoff] for ticker, frame in execution.items()
    }
    prefix_strategy = KodexSectorRotation(
        total_return_indices=total_return.loc[total_return.index <= cutoff],
        signal_prices=signal_core.loc[signal_core.index <= cutoff],
        short_momentum_weeks=8,
        long_momentum_weeks=16,
    )

    prefix = prefix_strategy.compute_security_targets(prefix_execution)
    full = strategy.compute_security_targets(execution)

    pd.testing.assert_frame_equal(
        prefix,
        full.loc[prefix.index].reindex(columns=prefix.columns),
    )


def test_sector_rotation_uses_core_two_sectors_and_cash_buffer():
    execution, total_return, signal_core = _synthetic_data()
    strategy = KodexSectorRotation(
        total_return_indices=total_return,
        signal_prices=signal_core,
        short_momentum_weeks=8,
        long_momentum_weeks=16,
    )

    targets = strategy.compute_security_targets(execution)
    invested = targets.loc[targets["069500"].gt(0)].iloc[-1]
    sector_weights = invested.drop(["069500", "__CASH__"])

    assert sector_weights.gt(0).sum() == 2
    assert invested["069500"] == pytest.approx(0.40)
    assert invested["__CASH__"] == pytest.approx(0.05)
    assert invested.sum() == pytest.approx(1.0)


def test_sector_rotation_is_registered():
    assert isinstance(get_strategy("kodex_sector_rotation"), KodexSectorRotation)


def test_sealed_holdout_requires_preregistered_period_and_signal_source():
    valid = Namespace(
        start_date="2018-04-02",
        end_date="2019-12-30",
        core_signal_source="distribution_adjusted_actual",
    )
    assert _validate_sealed_holdout(valid)["one_shot_evaluation"] is True

    invalid = Namespace(
        start_date="2018-05-01",
        end_date="2019-12-30",
        core_signal_source="distribution_adjusted_actual",
    )
    with pytest.raises(ValueError, match="sealed holdout dates"):
        _validate_sealed_holdout(invalid)
