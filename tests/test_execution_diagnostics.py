from datetime import datetime
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from execution_diagnostics import execution_diagnostics, summary_markdown
import multi_allocator_plus_trader as module
from multi_allocator_plus_trader import MultiAllocatorPlusTrader


def bare():
    t = MultiAllocatorPlusTrader.__new__(MultiAllocatorPlusTrader)
    t.market = 'kr'; t.dry_run = False; t.cash_policy = 'preserve'
    t.signal_repair_mode = 'on'; t.min_trade_value = 10_000
    t.strategy = SimpleNamespace(max_security_weight=.3)
    t.signal_mode = 'eod_fixed'; t.skip_if_executed = True
    t.loaded_signal_snapshot_payload = None
    return t


def test_lower_minimum_unlocks_only_affordable_whole_shares_within_original_weights():
    t = bare()
    targets = pd.Series({'069500': .0660149813838391, '446540': .030440565892484134,
        '036540': .030359592474405804, '092870': .023563817650774314,
        '459580': .006109279548924274})
    targets['__CASH__'] = 1 - targets.sum()
    original = targets.copy()
    refs = {'069500': 110745, '446540': 11650, '036540': 11590,
            '092870': 54300, '459580': 1073860}
    account = {'total_value': 700000, 'available_cash': 590000, 'stock_value': 0}
    plans, decisions = t.build_order_plan(targets, account, {}, price_cache_override=refs,
        min_trade_value_override=50000, return_decisions=True)
    assert plans == []
    assert execution_diagnostics(targets, decisions, 0, 0, {}, 50000)['status'] == 'no_orders_sizing_limits'
    plans, decisions = t.build_order_plan(targets, account, {}, price_cache_override=refs,
        return_decisions=True)
    assert {(p.symbol, p.quantity) for p in plans} == {('446540', 1), ('036540', 1)}
    for p in plans:
        assert p.est_value <= account['total_value'] * targets[p.symbol]
        assert p.quantity > 0 and p.action == 'BUY'
    pd.testing.assert_series_equal(targets, original)
    expensive = next(d for d in decisions if d['ticker'] == '069500')
    assert expensive['below_one_share'] and not expensive['below_minimum_trade']


@pytest.mark.parametrize('decisions,raw,final,result,expected', [
    ([], 0, 0, {'status': 'skipped_already_executed'}, 'skipped_already_executed'),
    ([{'reason': 'new_position_below_minimum_or_one_share'}], 0, 0, {}, 'no_orders_sizing_limits'),
    ([{'reason': 'quantity_already_at_target'}], 0, 0, {}, 'no_orders_at_target'),
    ([], 1, 0, {}, 'no_orders_recheck'),
    ([], 1, 1, {'executed_orders': 1}, 'orders_submitted'),
    ([], 1, 1, {'executed_orders': 1, 'failed_orders': 1}, 'order_submission_failed'),
    ([], 0, 0, {'status': 'failed'}, 'order_submission_failed'),
])
def test_execution_outcomes_are_distinct(decisions, raw, final, result, expected):
    d = execution_diagnostics({'069500': .18, '__CASH__': .82}, decisions, raw, final, result, 10000)
    assert d['status'] == expected
    assert d['selected_security_count'] == 1
    assert d['target_cash_weight'] == .82


def test_duplicate_guard_only_blocks_completed_live_run_same_day_and_signal(tmp_path, monkeypatch):
    class Clock:
        @classmethod
        def now(cls): return datetime(2026, 10, 7, 0, 15)
    monkeypatch.setattr(module, 'datetime', Clock)
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    directory = tmp_path / 'reports' / 'execution'; directory.mkdir(parents=True)
    def write(day, signal, run, completed=True, dry=False):
        (directory / f'summary_{day}_kr_{signal}_{run}.json').write_text(json.dumps(
            dict(run_id=run, completed_for_signal=completed, dry_run=dry)))
    t = bare()
    write('2026-10-06', '2026-10-02', 'yesterday')
    assert t._completed_execution_for_signal(pd.Timestamp('2026-10-02')) is None
    write('2026-10-07', '2026-10-02', 'other-signal')
    write('2026-10-07', '2026-10-06', 'no-orders-warning', completed=False)
    write('2026-10-07', '2026-10-06', 'dry', dry=True)
    signal = pd.Timestamp('2026-10-06')
    assert t._completed_execution_for_signal(signal) is None
    write('2026-10-07', '2026-10-06', 'submitted')
    assert t._completed_execution_for_signal(signal)['run_id'] == 'submitted'


def test_summary_explains_previous_execution_without_loading_account():
    execution = dict(status='skipped_already_executed', prior_execution=dict(
        run_id='prior', timestamp='2026-10-07T00:14:48', submitted_order_count=1,
        github_run_id='37549830399'))
    d = execution_diagnostics({'069500': .18, '__CASH__': .82}, [], 0, 0, execution, 10000)
    text = summary_markdown(dict(diagnostics=d, execution=execution,
        trade_date='2026-10-07', signal_date='2026-10-06'))
    assert '중복 주문 차단' in text and '37549830399' in text and '이전 주문 전송: 1건' in text


def test_sizing_summary_is_not_reported_as_duplicate():
    d = execution_diagnostics({'069500': .18, '__CASH__': .82},
        [{'reason': 'new_position_below_minimum_or_one_share'}], 0, 0, {}, 10000)
    text = summary_markdown(dict(diagnostics=d, execution={}, trade_date='2026-10-07', signal_date='2026-10-06'))
    assert '10,000원' in text and '최소금액·1주 조건 제외: 1개' in text
    assert '중복 실행 차단이 아닙니다' in text
