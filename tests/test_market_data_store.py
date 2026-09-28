import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import market_data_store as md
import kis_market_collection as kis
from scripts.market_data_pipeline import collect


def api_row(ticker='005930',name='삼성전자'):
    return dict(BAS_DD='20200803',ISU_CD=ticker,ISU_NM=name,
        TDD_OPNPRC='10,000',TDD_HGPRC='11,000',TDD_LWPRC='9,000',TDD_CLSPRC='10,500',
        ACC_TRDVOL='1,000',ACC_TRDVAL='10,400,000',MKTCAP='100000000')


def kis_payload(date='20200102'):
    return {'rt_cd':'0','output1':{'stck_shrn_iscd':'005930'},'output2':[
        dict(stck_bsop_date=date,stck_oprc='100',stck_hgpr='110',stck_lwpr='90',
             stck_clpr='105',acml_vol='1000',acml_tr_pbmn='104000')]}


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(md,'MIN_ROWS',dict.fromkeys(md.GROUPS,1))
    return md.MarketStore(tmp_path/'store')


@pytest.mark.parametrize('value',['2020-01-02','20200102'])
def test_day(value):
    assert md.day(value)=='2020-01-02'


@pytest.mark.parametrize('value',['../../etc/passwd','2020-99-99','NaT'])
def test_invalid_day(value):
    with pytest.raises(ValueError):md.day(value)


@pytest.mark.parametrize('name,expected',[
    ('KODEX 200','non_leveraged_name_unverified'),
    ('KODEX 200배당커버드콜','non_leveraged_name_unverified'),
    ('KODEX 레버리지','excluded_leverage_or_multiplier_name'),
    ('선물인버스2X','excluded_leverage_or_multiplier_name'),
    ('해외1.5X','excluded_leverage_or_multiplier_name'),
    ('KODEX 인버스','inverse_name_unverified')])
def test_etf_classification(name,expected):
    assert md.etf_name_classification(name)==expected


def test_raw_value_not_proxy():
    frame,_,_=md.normalize_api({'OutBlock_1':[api_row()]},'kospi','20200803')
    assert frame.value.iloc[0]==10400000
    assert frame.value.iloc[0]!=frame.close.iloc[0]*frame.volume.iloc[0]


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(BAS_DD='20200804'),lambda r:r.update(ISU_CD='../../bad'),
    lambda r:r.update(TDD_HGPRC='1'),lambda r:r.update(ACC_TRDVAL='-'),
    lambda r:r.update(ACC_TRDVOL='NaN'),lambda r:r.update(TDD_OPNPRC='0')])
def test_bad_rows_rejected(mutation):
    row=api_row();mutation(row)
    with pytest.raises(ValueError):md.normalize_api({'OutBlock_1':[row]},'kospi','20200803')


def test_error_is_not_holiday():
    with pytest.raises(md.DataQualityError):md.normalize_api({'error':'bad key'},'kospi','20200803')


def test_immutable_and_idempotent(store):
    payload={'OutBlock_1':[api_row()]}
    assert store.ingest('kospi','20200803',payload)
    original=copy.deepcopy(store.manifest)
    assert not store.ingest('kospi','20200803',payload)
    assert original==store.manifest
    payload['OutBlock_1'][0]['TDD_CLSPRC']='10,600'
    assert store.ingest('kospi','20200803',payload)
    old=original['snapshots']['kospi/2020-08-03']
    store.verify_record(old)
    assert store.manifest['snapshots']['kospi/2020-08-03']['previous_version']==old['raw_sha256']


def test_corruption_stops_reuse(store):
    store.ingest('kospi','20200803',{'OutBlock_1':[api_row()]})
    record=store.manifest['snapshots']['kospi/2020-08-03']
    store.checked_path(record['table_path']).write_bytes(b'corrupt')
    with pytest.raises(md.DataQualityError):store.has('kospi','20200803')


def test_path_escape(store):
    with pytest.raises(md.DataQualityError):store.checked_path('../../outside')


def test_empty_not_complete(store):
    store.ingest('kospi','20200803',{'OutBlock_1':[]})
    assert store.has('kospi','20200803')
    with pytest.raises(md.DataQualityError):store.market_day('20200803')


def test_etfs_quarantined_and_explicit_research_opt_in(store):
    for group,ticker,name in [('kospi','005930','삼성전자'),('kosdaq','035720','예시'),('etf','069500','KODEX 200')]:
        store.ingest(group,'20200803',{'OutBlock_1':[api_row(ticker,name)]})
    assert len(store.select_universe('20200803')['candidates'])==2
    selected=store.select_universe('20200803',include_unverified_etfs=True)
    assert len(selected['candidates'])==3 and not selected['orders_enabled']
    with pytest.raises(md.DataQualityError):store.select_universe('20200804')


def test_leverage_rows_not_admitted(store):
    store.ingest('etf','20200803',{'OutBlock_1':[api_row('122630','KODEX 레버리지')]})
    record=store.manifest['snapshots']['etf/2020-08-03']
    assert record['rows']==0 and len(record['excluded'])==1


def test_request_budget_and_resume(store):
    class Client:
        def fetch(self,g,d):return {'OutBlock_1':[api_row()]}
    first=collect(store,Client(),'20200803','20200803',max_requests=1,delay=0)
    assert first['status']=='checkpoint_budget_exhausted'
    second=collect(store,Client(),'20200803','20200803',max_requests=3,delay=0)
    assert second['requested']==2 and second['reused']==1


def test_raw_adjusted_kis_flags_and_no_orders():
    class Session:
        def __init__(self):self.calls=[]
        def post(self,url,**kwargs):
            self.calls.append(('post',url,kwargs))
            return SimpleNamespace(status_code=200,json=lambda:{'access_token':'fake-token','expires_in':3600})
        def get(self,url,**kwargs):
            self.calls.append(('get',url,kwargs))
            return SimpleNamespace(status_code=200,json=kis_payload)
    session=Session();client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    client.fetch('005930','20200101','20200330','raw')
    client.fetch('005930','20200101','20200330','adjusted')
    assert len([c for c in session.calls if c[0]=='post'])==1
    assert session.calls[1][2]['params']['FID_ORG_ADJ_PRC']=='1'
    assert session.calls[2][2]['params']['FID_ORG_ADJ_PRC']=='0'
    assert all('order' not in c[1] for c in session.calls)
    assert all(c[2]['allow_redirects'] is False for c in session.calls)
    with pytest.raises(ValueError):client.fetch('005930','20190101','20200101','raw')


def test_kis_secret_not_in_error():
    class Session:
        def post(self,*a,**kw):return SimpleNamespace(status_code=403,json=lambda:{'error_description':'fake-secret'})
    with pytest.raises(RuntimeError) as error:kis.KISPriceClient('fake-key','fake-secret',session=Session())._auth()
    assert 'fake-secret' not in str(error.value)


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(rt_cd='1'),
    lambda p:p['output1'].update(stck_shrn_iscd='000660'),
    lambda p:p['output2'][0].update(stck_bsop_date='20250102'),
    lambda p:p['output2'].append(p['output2'][0].copy())])
def test_bad_kis_payload(mutation):
    p=kis_payload();mutation(p)
    with pytest.raises(md.DataQualityError):kis.normalize_prices(p,'005930','20200101','20200131','raw')


def test_kis_resume_and_listing_date(store):
    master=pd.DataFrame([dict(ticker='005930',isin='KR7005930003',listed_date='1975-06-11',collection_eligible=True),
                         dict(ticker='000000',isin='KR7000000000',listed_date='2021-01-01',collection_eligible=True)])
    class Client:
        def fetch(self,*args):return kis_payload()
    result=kis.collect_prices(store,Client(),master,'20260928','20200101','20200131',max_requests=1,delay=0)
    assert result['requested']==1 and not result['requested_universe_queried']
    result=kis.collect_prices(store,Client(),master,'20260928','20200101','20200131',max_requests=5,delay=0)
    assert result['requested']==1 and result['reused']==1 and result['requested_universe_queried']
    assert len(store.manifest['kis_segments'])==2
    reloaded=md.MarketStore(store.root)
    assert len(reloaded.manifest['kis_segments'])==2
    panel=kis.load_kis_panel(store,{'005930'},'20200101','20200131','raw')
    assert len(panel)==1 and panel.price_basis.eq('raw').all()


def test_normalizer_revision_keeps_previous_files(store):
    frame=pd.DataFrame({'value':[1]})
    store.put_table('seeds','same/source',frame,b'same raw',{})
    old=copy.deepcopy(store.manifest['seeds']['same/source'])
    store.put_table('seeds','same/source',pd.DataFrame({'value':[2]}),b'same raw',{})
    store.verify_record(old)
    assert old['table_path']!=store.manifest['seeds']['same/source']['table_path']


def test_shard_interrupted_root_commit_recovers(store):
    frame=pd.DataFrame({'value':[1]})
    store.put_table('kis_segments','005930/raw/a',frame,b'a',{})
    committed_root=store.manifest_path.read_bytes()
    store.put_table('kis_segments','005930/raw/b',frame,b'b',{})
    # Simulate process termination before the atomic root manifest switch.
    store.manifest_path.write_bytes(committed_root)
    reloaded=md.MarketStore(store.root)
    assert set(reloaded.manifest['kis_segments'])=={'005930/raw/a'}


def test_master_cannot_be_relabelled_historical(store):
    with pytest.raises(md.DataQualityError):kis.select_kis(store,'20260928','20200102')


def test_seed_never_becomes_market_coverage(store):
    frame=pd.DataFrame([dict(date='2020-08-03',ticker='005930',open=1,high=1,low=1,close=1)])
    store.put_table('seeds','test/005930',frame,b'original',dict(ticker='005930',price_basis='adjusted',start='2020-08-03',end='2020-08-03'))
    assert len(store.load_seed_panel({'005930'},'20200803','20200803',price_basis='adjusted'))==1
    with pytest.raises(md.DataQualityError):store.market_day('20200803')
    with pytest.raises(md.DataQualityError):store.load_seed_panel({'005930'},'20200803','20200803',price_basis='raw')


def test_workflow_safety():
    import yaml
    root=Path(__file__).resolve().parents[1]
    text=(root/'.github/workflows/market-data.yml').read_text()
    workflow=yaml.safe_load(text)
    assert workflow['permissions']=={'contents':'read'}
    assert workflow['concurrency']['cancel-in-progress'] is False
    assert "r.json().get('private') is not True" in text
    assert 'KIS_ACCOUNT' not in text and 'multi_allocator_plus_trader' not in text
    assert 'git reset' not in text and 'git clean' not in text and '--force' not in text
    assert 'steps.audit.outcome' in text
