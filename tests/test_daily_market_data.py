from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from market_data_store import MarketStore, DataQualityError
from kis_market_collection import KISPriceClient
from daily_market_data import run_daily, previous_open_day, normalize_calendar, load_daily_selection_inputs

NOW=datetime(2026,9,29,10,tzinfo=timezone.utc)  # 19:00 KST, prior open day 09-28 in fixture.


class Client:
    def __init__(self, holidays=(), missing_ticker=None):
        self.holidays=set(holidays)
        self.calendar_calls=0
        self.price_calls=0
        self.missing_ticker=missing_ticker

    def fetch_calendar(self,base):
        self.calendar_calls+=1
        return {'rt_cd':'0','output':[{'bass_dt':d.strftime('%Y%m%d'),
            'opnd_yn':'Y' if d.weekday()<5 and str(d.date()) not in self.holidays else 'N'}
            for d in pd.date_range(base,periods=31)]}

    def fetch(self,ticker,start,end,basis):
        self.price_calls+=1
        rows=[]
        for d in pd.date_range(start,end):
            if d.weekday()>=5 or str(d.date()) in self.holidays or ticker==self.missing_ticker:
                continue
            rows.append(dict(stck_bsop_date=d.strftime('%Y%m%d'),stck_oprc='100',stck_hgpr='110',
                stck_lwpr='90',stck_clpr='105',acml_vol='1000',acml_tr_pbmn='105000'))
        return {'rt_cd':'0','output1':{'stck_shrn_iscd':ticker},'output2':rows}


def master_loader(store,observed_date,extra=None):
    frames=[]
    for market,ticker,asset in [('kospi','069500','etf'),('kosdaq','005930','stock')]:
        frame=pd.DataFrame([dict(ticker=ticker,isin='KR7'+ticker+'003',name=ticker,market=market,
            asset_type=asset,listed_date='2000-01-01',collection_eligible=True,
            classification='stock' if asset=='stock' else 'non_leveraged_name_unverified')])
        if extra and market=='kosdaq':
            frame=pd.concat([frame,pd.DataFrame([extra])],ignore_index=True)
        store.put_table('masters',f'{observed_date}/{market}',frame,b'mock master',{})
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


@pytest.fixture
def store(tmp_path,monkeypatch):
    import daily_market_data
    monkeypatch.setattr(daily_market_data,'_utc_now',lambda:NOW)
    return MarketStore(tmp_path/'store')


def test_previous_day_calendar_reused_once(store):
    client=Client()
    assert previous_open_day(store,client,'2026-09-29')[0]=='2026-09-28'
    assert previous_open_day(MarketStore(store.root),client,'2026-09-29')[0]=='2026-09-28'
    assert client.calendar_calls==1


def test_weekend_and_explicit_holiday_not_inferred(store):
    client=Client(holidays=['2026-09-25'])
    assert previous_open_day(store,client,'2026-09-28')[0]=='2026-09-24'
    assert client.calendar_calls==1


@pytest.mark.parametrize('payload',[
    {'rt_cd':'1','output':[]}, {'rt_cd':'0','output':[]},
    {'rt_cd':'0','output':[{'bass_dt':'20260928'}]},
    {'rt_cd':'0','output':[{'bass_dt':'20260928','opnd_yn':'Y'}]*2}])
def test_bad_calendar(payload):
    with pytest.raises(DataQualityError): normalize_calendar(payload)


def test_calendar_missing_dates_blocks_no_weekday_fallback(store):
    client=Client()
    client.fetch_calendar=lambda _: {'rt_cd':'0','output':[{'bass_dt':'20260928','opnd_yn':'Y'}]}
    with pytest.raises(DataQualityError): previous_open_day(store,client,'2026-09-29')


def test_daily_complete_no_orders_and_no_top_n(store):
    client=Client()
    state=run_daily(store,client,now=NOW,master_loader=master_loader,delay=0)
    assert state['status']=='daily_price_inputs_ready'
    assert state['price_date']=='2026-09-28' and state['master_date']=='2026-09-29'
    assert state['target_tickers']==2 and state['requested']==4
    assert not state['orders_enabled'] and not state['strategy_history_verified']
    frame=load_daily_selection_inputs(MarketStore(store.root),now=NOW)
    assert set(frame.ticker)=={'005930','069500'}
    assert frame.stock_candidate_universe.sum()==1
    # Stable target/range resumes without another calendar or price request.
    resumed=run_daily(MarketStore(store.root),client,now=NOW,master_loader=master_loader,delay=0)
    assert resumed['requested']==0 and resumed['reused']==4
    assert client.calendar_calls==1 and client.price_calls==4


def test_budget_limit_cannot_publish_selection(store):
    state=run_daily(store,Client(),now=NOW,master_loader=master_loader,max_requests=1,delay=0)
    assert state['status']=='checkpoint_budget_exhausted'
    assert not store.manifest.get('daily_inputs')
    with pytest.raises(DataQualityError): load_daily_selection_inputs(store,now=NOW)


def test_missing_prices_block_instead_of_silently_shrinking_universe(store):
    state=run_daily(store,Client(missing_ticker='005930'),now=NOW,master_loader=master_loader,delay=0)
    assert state['status']=='blocked_missing_prices'
    assert state['missing_raw']==['005930']
    assert not store.manifest.get('daily_inputs')


def test_stale_and_not_yet_available_inputs_rejected(store):
    state=run_daily(store,Client(),now=NOW,master_loader=master_loader,delay=0)
    with pytest.raises(DataQualityError): load_daily_selection_inputs(store,now=NOW+timedelta(days=1))
    # Deterministically move availability past the requested time without editing prices.
    state['completed_at']=(NOW+timedelta(seconds=1)).isoformat()
    (store.root/'daily_status/latest.json').write_text(json.dumps(state))
    with pytest.raises(DataQualityError): load_daily_selection_inputs(store,now=NOW)


def test_new_listing_after_price_date_excluded(store):
    extra=dict(ticker='123456',isin='KR7123456003',name='new',market='kosdaq',asset_type='stock',
               listed_date='2026-09-29',collection_eligible=True,classification='stock')
    def loader(s,observed_date): return master_loader(s,observed_date,extra)
    state=run_daily(store,Client(),now=NOW,master_loader=loader,delay=0)
    assert state['target_tickers']==2


def test_error_retires_old_ready_pointer(store):
    run_daily(store,Client(),now=NOW,master_loader=master_loader,delay=0)
    def fail(*args,**kwargs): raise RuntimeError('simulated private provider error')
    with pytest.raises(RuntimeError):
        run_daily(store,Client(),now=NOW,master_loader=fail,delay=0)
    state=json.loads((store.root/'daily_status/latest.json').read_text())
    assert state['status']=='blocked_collection_or_quality_error'
    assert 'simulated private provider' not in json.dumps(state)
    with pytest.raises(DataQualityError): load_daily_selection_inputs(store,now=NOW)


def test_snapshot_tamper_rejected(store):
    state=run_daily(store,Client(),now=NOW,master_loader=master_loader,delay=0)
    record=store.manifest['daily_inputs'][state['input_key']]
    store.checked_path(record['table_path']).write_bytes(b'corrupt')
    with pytest.raises(DataQualityError): load_daily_selection_inputs(store,now=NOW)


def test_calendar_endpoint_is_read_only_and_sanitized():
    calls=[]
    class Session:
        def post(self,url,**kwargs):return SimpleNamespace(status_code=200,json=lambda:{'access_token':'fake','expires_in':3600})
        def get(self,url,**kwargs):
            calls.append((url,kwargs))
            return SimpleNamespace(status_code=200,json=lambda:Client().fetch_calendar('2026-09-15'))
    client=KISPriceClient('key','secret',session=Session())
    assert client.fetch_calendar('2026-09-15')['rt_cd']=='0'
    assert calls[0][0].endswith('/quotations/chk-holiday')
    assert calls[0][1]['headers']['tr_id']=='CTCA0903R'
    assert calls[0][1]['params']=={'BASS_DT':'20260915','CTX_AREA_FK':'','CTX_AREA_NK':''}
    assert calls[0][1]['allow_redirects'] is False


def test_daily_workflow_does_not_trade_or_race_backfill():
    from pathlib import Path
    import yaml
    text=(Path(__file__).resolve().parents[1]/'.github/workflows/market-data-daily.yml').read_text()
    workflow=yaml.safe_load(text)
    assert set(workflow['on'])=={'workflow_call'}
    assert workflow['permissions']=={'contents':'read'}
    assert workflow['concurrency']=={'group':'market-data-private-writer','cancel-in-progress':False}
    assert 'KIS_ACCOUNT' not in text and 'multi_allocator_plus_trader' not in text
    assert 'upload-artifact' not in text and '--force' not in text and 'git reset' not in text
    assert 'steps.audit.outcome' in text and '!cancelled()' in text
    assert 'daily_price_inputs_ready' in text


def test_unknown_calendar_blocks_before_prices(store):
    client=Client()
    client.fetch_calendar=lambda _: {'rt_cd':'1','output':[]}
    with pytest.raises(DataQualityError):
        run_daily(store,client,now=NOW,master_loader=master_loader,delay=0)
    assert client.price_calls==0
    assert json.loads((store.root/'daily_status/latest.json').read_text())['status']=='blocked_collection_or_quality_error'
