import copy
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from market_data_store import DataQualityError, canonical, digest
from security_lifecycle import normalize_security_info
from kis_market_collection import KISPriceClient
from daily_market_data import run_daily, load_daily_selection_inputs
from test_daily_market_data import Client, NOW, master_loader, store


def info(date='20260928'):
    return dict(rt_cd='0',output=dict(pdno='005930',std_pdno='KR7005930003',
        kosdaq_mket_lstg_abol_dt=date,lstg_abol_dt=date,tr_stop_yn='Y'))


class LifecycleClient(Client):
    def __init__(self,date='20260928'):
        super().__init__(missing_ticker='005930')
        self.date=date
        self.status_calls=0
    def fetch_security_info(self,ticker):
        self.status_calls+=1
        return info(self.date)


def test_delisted_remains_in_full_input_without_prices(store):
    client=LifecycleClient()
    state=run_daily(store,client,now=NOW,master_loader=master_loader,delay=0)
    assert state['status']=='daily_price_inputs_ready'
    assert state['target_tickers']==2
    assert state['confirmed_delisted_tickers']==['005930']
    assert state['missing_price_retry_requests']==2
    f=load_daily_selection_inputs(store,now=NOW).set_index('ticker')
    row=f.loc['005930']
    assert not row.tradable and not row.stock_candidate_universe
    assert row.price_status=='confirmed_delisted' and pd.isna(row.raw_close)
    assert row.lifecycle_key=='2026-09-29/005930'
    assert client.status_calls==1


@pytest.mark.parametrize('date',['','00000000','20260929'])
def test_halt_or_future_delisting_cannot_excuse_missing_prior_prices(store,date):
    state=run_daily(store,LifecycleClient(date),now=NOW,master_loader=master_loader,delay=0)
    assert state['status']=='blocked_missing_prices'
    assert not store.manifest.get('daily_inputs')


@pytest.mark.parametrize('key,value',[('pdno','000660'),('std_pdno','KR7000660001'),
                                    ('lstg_abol_dt','20260927'),('lstg_abol_dt','bad')])
def test_identity_or_date_mismatch_rejected(key,value):
    p=info();p['output'][key]=value
    with pytest.raises((DataQualityError,ValueError)):
        normalize_security_info(p,'005930','KR7005930003','kosdaq')


def test_empty_security_response_never_confirms_delisting():
    with pytest.raises(DataQualityError):
        normalize_security_info({'rt_cd':'7','output':{}},'005930','KR7005930003','kosdaq')


def test_observed_internal_product_id_requires_exact_prefix_and_isin():
    p=info();p['output']['pdno']='00000A005930'
    assert normalize_security_info(p,'005930','KR7005930003','kosdaq')['delisted_date']=='2026-09-28'
    p['output']['pdno']='anything005930'
    with pytest.raises(DataQualityError):normalize_security_info(p,'005930','KR7005930003','kosdaq')


def test_lifecycle_original_corruption_blocks_consumer(store):
    run_daily(store,LifecycleClient(),now=NOW,master_loader=master_loader,delay=0)
    r=store.manifest['security_status']['2026-09-29/005930']
    store.checked_path(r['raw_path']).write_bytes(b'bad')
    with pytest.raises(DataQualityError):load_daily_selection_inputs(store,now=NOW)


def test_lifecycle_table_rewrite_with_updated_hash_still_rejected(store):
    run_daily(store,LifecycleClient(),now=NOW,master_loader=master_loader,delay=0)
    r=store.manifest['security_status']['2026-09-29/005930']
    path=store.checked_path(r['table_path']);f=pd.read_parquet(path)
    f['delisted_date']='2020-01-01';f.to_parquet(path,index=False)
    r['table_sha256']=digest(path.read_bytes())
    with pytest.raises(DataQualityError,match='original'):load_daily_selection_inputs(store,now=NOW)


def test_missing_prices_recovered_with_targeted_day_refresh(store):
    class Transient(Client):
        def fetch(self,ticker,start,end,basis):
            self.missing_ticker='005930' if start!=end else None
            return super().fetch(ticker,start,end,basis)
    client=Transient()
    state=run_daily(store,client,now=NOW,master_loader=master_loader,delay=0)
    assert state['status']=='daily_price_inputs_ready'
    assert state['confirmed_delisted_tickers']==[]
    assert client.price_calls==6


def test_one_sided_missing_not_excused_by_delisting(store):
    class Partial(LifecycleClient):
        def fetch(self,ticker,start,end,basis):
            self.missing_ticker='005930' if basis=='raw' else None
            return super().fetch(ticker,start,end,basis)
    client=Partial()
    state=run_daily(store,client,now=NOW,master_loader=master_loader,delay=0)
    assert state['status']=='blocked_missing_prices' and client.status_calls==0


def test_security_endpoint_read_only_and_identity_params():
    calls=[]
    class Session:
        def post(self,url,**kw):return SimpleNamespace(status_code=200,json=lambda:{'access_token':'fake','expires_in':3600})
        def get(self,url,**kw):
            calls.append((url,kw));return SimpleNamespace(status_code=200,json=info)
    c=KISPriceClient('key','secret',session=Session())
    assert c.fetch_security_info('005930')['rt_cd']=='0'
    url,kw=calls[0]
    assert url.endswith('/quotations/search-stock-info')
    assert kw['params']=={'PRDT_TYPE_CD':'300','PDNO':'005930'}
    assert kw['headers']['tr_id']=='CTPF1002R' and not kw['allow_redirects']
