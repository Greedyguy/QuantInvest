import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import multi_allocator_plus_trader as module
from multi_allocator_plus_trader import MultiAllocatorPlusTrader, OrderPlan
from signal_safety import SIGNAL_PATH_VERSION, constrain_targets, validate_targets, validate_snapshot
from strategies.strategy_multi_allocator import MultiStrategyAllocator


def payload():
    return dict(signal_date='2026-10-01',strategy='multi_allocator_plus_safe_etf_kqm',
        targets={'069500':.3,'__CASH__':.7},ref_prices={'069500':60000.},
        meta=dict(signal_path_version=SIGNAL_PATH_VERSION,market='kr',allocation_policy='legacy',
            data_as_of=dict(primary_index='2026-10-01',secondary_index='2026-10-01',
                            target_securities={'069500':'2026-10-01'})))


def bare():
    t=MultiAllocatorPlusTrader.__new__(MultiAllocatorPlusTrader)
    t.market='kr'; t.signal_repair_mode='on'; t.cash_policy='preserve'
    t.strategy=SimpleNamespace(max_security_weight=.3)
    t.dry_run=True; t.min_trade_value=50000; t.run_id='offline-test'
    t.execution_recheck=True; t.recheck_price_band_pct=3.
    t.kis=SimpleNamespace(account='test',send_order=lambda **kw: pytest.fail('Unexpected broker call'))
    return t


def test_underinvested_budget_and_relative_weights_preserved():
    row=pd.Series({'A':.10,'B':.20,'__CASH__':.70})
    pd.testing.assert_series_equal(constrain_targets(row).sort_index(),row.sort_index())


def test_only_excess_shrinks_and_security_cap_leaves_cash():
    row=constrain_targets({'A':.27,'B':.27,'C':.27,'D':.27,'__CASH__':0})
    assert row.drop('__CASH__').tolist()==pytest.approx([.25]*4)
    assert row['__CASH__']==pytest.approx(0)
    row=constrain_targets({'A':.8,'__CASH__':.2})
    assert row.to_dict()==pytest.approx({'A':.3,'__CASH__':.7})


@pytest.mark.parametrize('value',[np.nan,np.inf,-.1])
def test_corrupt_generated_weight_fails(value):
    with pytest.raises(ValueError):
        constrain_targets({'A':value,'__CASH__':1.})


@pytest.mark.parametrize('row',[{'A':.4,'__CASH__':.6},{'A':.3,'__CASH__':.8},
                               {'A':.2},{'A':np.nan,'__CASH__':.8}])
def test_order_boundary_rejects_not_silently_repairs_invalid_target(row):
    with pytest.raises(ValueError): validate_targets(row)


def test_current_repaired_snapshot_valid():
    assert validate_snapshot(payload(),today='2026-10-02')['069500']==.3


@pytest.mark.parametrize('today',['2026-10-01','2026-09-30','2026-10-08'])
def test_same_future_or_stale_snapshot_rejected(today):
    with pytest.raises(ValueError): validate_snapshot(payload(),today=today)


@pytest.mark.parametrize('field,value',[('signal_path_version','old'),('market','us'),
                                       ('allocation_policy','top5')])
def test_contract_changes_cannot_reach_order_planning(field,value):
    p=payload(); p['meta'][field]=value
    with pytest.raises(ValueError): validate_snapshot(p,today='2026-10-02')


def test_reference_and_asof_must_be_current_and_valid():
    p=payload(); p['ref_prices']['069500']=np.nan
    with pytest.raises(ValueError): validate_snapshot(p,today='2026-10-02')
    p=payload(); p['meta']['data_as_of']['target_securities']['069500']='2026-09-30'
    with pytest.raises(ValueError): validate_snapshot(p,today='2026-10-02')


def test_unversioned_snapshot_rejected_before_account_fetch(tmp_path):
    t=bare(); p=payload(); p['meta'].pop('signal_path_version')
    path=tmp_path/'old.json'; path.write_text(json.dumps(p))
    t._signal_snapshot_path=lambda:path
    with pytest.raises(ValueError,match='version'): t.load_signal_snapshot()


def test_rollback_cannot_execute_repaired_snapshot(tmp_path):
    t=bare(); t.signal_repair_mode='off'
    path=tmp_path/'repaired.json'; path.write_text(json.dumps(payload()))
    t._signal_snapshot_path=lambda:path
    with pytest.raises(ValueError,match='disabled'): t.load_signal_snapshot()


def test_end_to_end_snapshot_to_plan_without_broker(tmp_path):
    t=bare()
    date=(pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).normalize()-pd.offsets.BDay(1))
    t.enriched={name:pd.DataFrame({'close':[50000.]},index=[date]) for name in ['069500','305720']}
    t.market_index=t.enriched['069500']; t.secondary_index=t.market_index
    t.loaded_signal_snapshot_payload=None
    t.start_date='2026-01-01'; t.stress_recovery_mode='shadow'
    t._latest_prices=lambda tickers:{ticker:50000. for ticker in tickers}
    t._git_revision=lambda:'offline-test'
    t._build_style_attribution=lambda *args:{}
    t._build_decision_context=lambda *args:{}
    t.strategy.get_name=lambda:'multi_allocator_plus_safe_etf_kqm'
    path=tmp_path/'signal.json'; t._signal_snapshot_path=lambda *args:path
    # Unchanged target allocation: no money is reassigned from the blocked ETF.
    targets=pd.Series({'069500':.3,'305720':.2,'__CASH__':.5})
    t.save_signal_snapshot(date,targets)
    loaded_date,loaded,refs=t.load_signal_snapshot()
    pd.testing.assert_series_equal(loaded.sort_index(),targets.sort_index())
    assert loaded_date==date
    plans=t.build_order_plan(loaded,{'total_value':1e6,'available_cash':500000},
        {'305720':dict(quantity=2,current_price=50000)},price_cache_override=refs)
    assert len(plans)==1 and plans[0].symbol=='069500' and plans[0].quantity==6


@pytest.mark.parametrize('quantity,weight,expected',[(0,.2,None),(10,.2,'SELL'),(10,.1,'SELL')])
def test_305720_no_new_buy_normal_reduction_allowed(quantity,weight,expected):
    t=bare()
    holdings={'305720':dict(quantity=quantity,current_price=50000)} if quantity else {}
    plans,decisions=t.build_order_plan(pd.Series({'305720':weight,'__CASH__':1-weight}),
        {'total_value':1_000_000,'available_cash':500000},holdings,
        price_cache_override={'305720':50000},return_decisions=True)
    assert [p.action for p in plans]==([expected] if expected else [])
    if expected is None: assert decisions[0]['reason']=='buy_blocked_instrument_identity'


def test_305720_existing_position_not_forced_out_just_for_buy_block():
    t=bare()
    plans=t.build_order_plan(pd.Series({'305720':.2,'__CASH__':.8}),
        {'total_value':1_000_000,'available_cash':500000},
        {'305720':dict(quantity=4,current_price=50000)},price_cache_override={'305720':50000})
    assert plans==[]


def test_305720_normal_full_exit_still_allowed():
    t=bare()
    plans=t.build_order_plan(pd.Series({'__CASH__':1.}),
        {'total_value':1_000_000,'available_cash':500000},
        {'305720':dict(quantity=4,current_price=50000)},price_cache_override={})
    assert plans[0].action=='SELL' and plans[0].quantity==4


def test_blocked_buy_filtered_even_without_recheck():
    t=bare(); t.execution_recheck=False
    plan=OrderPlan('305720','BUY',1,50000,50000,.1,0,1)
    reviewed,logs=t.apply_execution_recheck([plan],{'available_cash':1e6})
    assert reviewed==[] and logs[0]['reason']=='buy_blocked_instrument_identity'


def test_final_send_boundary_cannot_bypass_buy_block():
    t=bare(); t.dry_run=False
    t.reporter=SimpleNamespace(save_report=lambda *a:'offline')
    t.append_execution_logs=lambda *a:None
    plan=OrderPlan('305720','BUY',1,50000,50000,.1,0,1)
    result=t.execute([plan],{'total_value':1e6},{},pd.Timestamp('2026-10-02'))
    assert result['executed_orders']==0
    assert result['order_logs'][0]['reason']=='buy_blocked_instrument_identity'


def test_us_not_changed_by_kr_repair():
    t=bare(); t.market='us'
    assert not t._signal_repair_enabled()
    assert not t._buy_blocked('305720')


@pytest.mark.parametrize('quote',[None,0.,-1.,np.nan,np.inf])
def test_live_buy_fails_closed_without_valid_current_quote(quote):
    t=bare(); t.dry_run=False
    t._safe_get_current_price=lambda symbol:quote
    t._safe_get_orderable_qty=lambda *a:pytest.fail('Sizing unavailable quote')
    plan=OrderPlan('069500','BUY',1,50000,50000,.1,0,1)
    reviewed,logs=t.apply_execution_recheck([plan],{'available_cash':1e6})
    assert reviewed==[] and logs[0]['reason']=='live_buy_quote_unavailable'


def test_missing_quote_does_not_block_normal_sell():
    t=bare(); t.dry_run=False
    t._safe_get_current_price=lambda symbol:None
    plan=OrderPlan('305720','SELL',1,50000,50000,0,1,0)
    reviewed,logs=t.apply_execution_recheck([plan],{'available_cash':1e6})
    assert reviewed==[plan]


def test_valid_current_quote_still_allows_normal_buy():
    t=bare(); t.dry_run=False
    t._safe_get_current_price=lambda symbol:50000.
    t._safe_get_orderable_qty=lambda *args:10
    plan=OrderPlan('069500','BUY',1,50000,50000,.1,0,1)
    reviewed,logs=t.apply_execution_recheck([plan],{'available_cash':1e6})
    assert reviewed==[plan] and logs[0]['decision']=='send'


def test_repaired_real_execution_cannot_bypass_snapshot_contract(monkeypatch):
    monkeypatch.setattr(module,'KoreaInvestmentConnector',lambda **k:pytest.fail('Broker initialized'))
    with pytest.raises(ValueError,match='eod_fixed'):
        MultiAllocatorPlusTrader(dry_run=False,signal_mode='live')


@pytest.mark.parametrize('failure',['exception','empty','no_weights','unavailable'])
def test_child_failure_cannot_silently_reallocate_portfolio(failure):
    s=MultiStrategyAllocator(strategy_configs=[{'name':'fake','weight':1}],child_signal_mode=True,signal_market='KR')
    def run(*args,**kwargs):
        if failure=='exception': raise RuntimeError('fixture')
        return (pd.DataFrame() if failure=='empty' else pd.DataFrame({'equity':[100.]},index=[pd.Timestamp('2026-10-01')])),[]
    def get(name):
        if failure=='unavailable': raise RuntimeError('fixture')
        return SimpleNamespace(run_signal_backtest=run,get_target_weight_history=lambda:pd.DataFrame())
    s._get_strategy=get
    with pytest.raises(RuntimeError): s._run_child_strategies({},None)


def test_unsafe_live_options_rejected_before_connector_creation(monkeypatch):
    monkeypatch.setattr(module,'KoreaInvestmentConnector',lambda **k:pytest.fail('Broker initialized'))
    with pytest.raises(ValueError,match='rechecks'):
        MultiAllocatorPlusTrader(dry_run=False,execution_recheck=False)
    with pytest.raises(ValueError,match='cash'):
        MultiAllocatorPlusTrader(cash_policy='legacy_renorm')


def test_three_kr_workflows_opt_in_and_us_not_changed():
    from pathlib import Path
    for path in ['daily-eod-signal.yml','daily-open-exec-a.yml','daily-open-exec-b-shadow.yml']:
        assert '--signal-repair-mode on' in (Path('.github/workflows')/path).read_text()
    for path in ['daily-eod-signal-us.yml','daily-open-exec-us.yml']:
        assert '--signal-repair-mode' not in (Path('.github/workflows')/path).read_text()


def test_prepare_only_never_initializes_broker(monkeypatch):
    monkeypatch.setattr(module,'KoreaInvestmentConnector',lambda **kw:pytest.fail('Broker initialized'))
    t=MultiAllocatorPlusTrader(prepare_signal_only=True,market_store_path='private/store',market_data_commit='a'*40)
    assert t.kis is None


def test_private_store_path_does_not_fall_back_to_legacy_download(monkeypatch):
    import production_market_inputs
    t=bare(); t.market_store_path='missing/store'; t.market_data_commit='a'*40; t.start_date='2026-01-01'
    monkeypatch.setattr(module,'load_data',lambda **kw:pytest.fail('Legacy download fallback'))
    def fail(*a,**kw): raise ValueError('Missing private data')
    monkeypatch.setattr(production_market_inputs,'load_production_inputs',fail)
    with pytest.raises(ValueError,match='Missing private'): t._load_market_data_kr()
