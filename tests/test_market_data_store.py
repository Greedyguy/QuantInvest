import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

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


class PriceSession:
    def __init__(self, replies):
        self.replies=iter(replies)
        self.calls=[]

    def post(self,*args,**kwargs):
        return SimpleNamespace(status_code=200,json=lambda:{'access_token':'fake-token','expires_in':3600})

    def get(self,url,**kwargs):
        self.calls.append((url,kwargs))
        value=next(self.replies)
        if isinstance(value,Exception):
            raise value
        return value


def price_response(status=200,body=None,headers=None):
    def content():
        if isinstance(body,Exception):raise body
        return kis_payload() if body is None else body
    return SimpleNamespace(status_code=status,json=content,headers=headers or {})


@pytest.mark.parametrize('reply,kind',[
    (requests.ReadTimeout('fake-secret'),'timeout'),
    (requests.ConnectTimeout('fake-secret'),'timeout'),
    (requests.ConnectionError('fake-secret'),'connection_error'),
    (requests.exceptions.ChunkedEncodingError('fake-secret'),'incomplete_response'),
    (price_response(body=ValueError('fake-secret')),'invalid_json'),
    (price_response(body=requests.exceptions.JSONDecodeError('fake-secret','bad',0)),'invalid_json'),
    (price_response(503),'http_error'),
    (price_response(429),'http_error'),
    (price_response(body={'rt_cd':'1','msg_cd':'EGW00201','msg1':'fake-secret'}),'provider_error'),
])
def test_kis_transient_retry_same_segment(reply,kind,monkeypatch,capsys):
    waits=[];monkeypatch.setattr(kis.time,'sleep',waits.append)
    session=PriceSession([reply,price_response()])
    client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    assert client.fetch('005930','20200101','20200131','raw')['rt_cd']=='0'
    assert len(session.calls)==2 and session.calls[0]==session.calls[1]
    assert len(waits)==1 and waits[0]>=2
    log=capsys.readouterr().out
    assert json.loads(log)['kind']==kind
    assert all(secret not in log for secret in ('fake-key','fake-secret','fake-token'))


def test_kis_retry_exhaustion_is_bounded_and_sanitized(monkeypatch,capsys):
    waits=[];monkeypatch.setattr(kis.time,'sleep',waits.append)
    session=PriceSession([requests.ReadTimeout('fake-secret') for _ in range(5)])
    client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    with pytest.raises(kis.KISPriceRequestError) as failure:
        client.fetch('005930','20200101','20200131','raw')
    assert len(session.calls)==5 and waits==[2,4,8,16]
    assert failure.value.diagnostic==dict(kind='timeout',attempts=5,http_status=None,provider_code=None)
    assert 'fake-secret' not in str(failure.value)+capsys.readouterr().out
    assert failure.value.__suppress_context__


@pytest.mark.parametrize('reply',[
    price_response(401,body=ValueError('fake-secret')),
    price_response(403),price_response(404),price_response(302),
    price_response(body={'rt_cd':'1','msg_cd':'EGW00123','msg1':'fake-secret'}),
    price_response(body={'rt_cd':'1','msg_cd':'fake-secret'}),
    price_response(body=[]),
    requests.exceptions.SSLError('fake-secret'),
    requests.RequestException('fake-secret'),
])
def test_kis_permanent_failures_are_not_retried(reply,monkeypatch,capsys):
    waits=[];monkeypatch.setattr(kis.time,'sleep',waits.append)
    session=PriceSession([reply]);client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    with pytest.raises(kis.KISPriceRequestError) as failure:
        client.fetch('005930','20200101','20200131','raw')
    assert len(session.calls)==1 and waits==[]
    assert 'fake-secret' not in str(failure.value)+capsys.readouterr().out


def test_kis_retry_after_and_success_on_final_attempt(monkeypatch):
    waits=[];monkeypatch.setattr(kis.time,'sleep',waits.append)
    replies=[price_response(503,headers={'Retry-After':'20'})]*4+[price_response()]
    session=PriceSession(replies);client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    assert client.fetch('005930','20200101','20200131','raw')['rt_cd']=='0'
    assert waits==[20]*4 and len(session.calls)==5


def test_failed_collection_records_progress_and_resumes(store,monkeypatch):
    monkeypatch.setattr(kis.time,'sleep',lambda _:None)
    master=pd.DataFrame([dict(ticker='005930',isin='KR7005930003',listed_date='1975-06-11',collection_eligible=True)])
    replies=[price_response()]+[requests.ReadTimeout('fake-secret')]*5
    client=kis.KISPriceClient('fake-key','fake-secret',session=PriceSession(replies))
    with pytest.raises(kis.CollectionInterrupted) as failure:
        kis.collect_prices(store,client,master,'20260928','20200101','20200131',max_requests=2,delay=0)
    p=failure.value.progress
    assert p['requested']==1 and p['reused']==0
    assert p['next_key']=='005930/adjusted/2020-01-01_2020-01-31'
    assert p['error']['kind']=='timeout' and not p['requested_universe_queried']
    reloaded=md.MarketStore(store.root)
    assert len(reloaded.manifest['kis_segments'])==1
    session=PriceSession([price_response()])
    client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    result=kis.collect_prices(reloaded,client,master,'20260928','20200101','20200131',max_requests=2,delay=0)
    assert result['requested']==1 and result['reused']==1 and result['requested_universe_queried']
    assert session.calls[0][1]['params']['FID_ORG_ADJ_PRC']=='0'


def test_failed_job_replaces_stale_summary_and_stays_failed(store,monkeypatch,tmp_path):
    import scripts.run_market_data_job as job
    monkeypatch.setenv('MARKET_STORE_PATH',str(store.root))
    monkeypatch.setenv('COLLECTION_MODE','backfill')
    monkeypatch.setenv('MASTER_DATE','2026-09-28')
    monkeypatch.setenv('BACKFILL_START','2020-01-01')
    monkeypatch.setenv('BACKFILL_END','2020-01-31')
    summary=tmp_path/'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY',str(summary))
    master=pd.DataFrame([dict(ticker='005930',isin='KR7005930003',listed_date='1975-06-11',collection_eligible=True)])
    monkeypatch.setattr(job,'read_master',lambda *args:master)
    monkeypatch.setattr(kis.time,'sleep',lambda _:None)
    session=PriceSession([price_response()]+[requests.ReadTimeout('fake-secret')]*5)
    client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    monkeypatch.setattr(job,'KISPriceClient',lambda *args:client)
    store.root.mkdir(parents=True,exist_ok=True)
    (store.root/'last_kis_collection.json').write_text('{"status":"requested_universe_queried"}')
    with pytest.raises(kis.CollectionInterrupted):job.main()
    result=json.loads((store.root/'last_kis_collection.json').read_text())
    assert result['status']=='blocked_collection_error' and result['requested']==1
    assert result['next_key']=='005930/adjusted/2020-01-01_2020-01-31'
    assert result['start']=='2020-01-01' and result['updated_at']
    assert result['error']['attempts']==5 and result['orders_enabled'] is False
    assert '오류로 중단' in summary.read_text()
    assert 'fake-secret' not in json.dumps(result)+summary.read_text()


def test_collection_quality_failure_is_not_retried_or_skipped(store,monkeypatch):
    waits=[];monkeypatch.setattr(kis.time,'sleep',waits.append)
    master=pd.DataFrame([dict(ticker='005930',isin='KR7005930003',listed_date='1975-06-11',collection_eligible=True)])
    payload=kis_payload();payload['output1']['stck_shrn_iscd']='000660'
    session=PriceSession([price_response(body=payload)])
    client=kis.KISPriceClient('fake-key','fake-secret',session=session)
    with pytest.raises(kis.CollectionInterrupted) as failure:
        kis.collect_prices(store,client,master,'20260928','20200101','20200131',delay=0)
    assert failure.value.progress['error']['kind']=='quality_error'
    assert failure.value.progress['requested']==0 and not store.manifest['kis_segments']
    assert len(session.calls)==1 and waits==[]


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
    assert 'push' not in workflow['on']
    for trigger in ('workflow_call','workflow_dispatch'):
        assert workflow['on'][trigger]['inputs']['start']['default']=='2025-01-01'
        assert workflow['on'][trigger]['inputs']['end']['default']=='2026-08-31'
        assert workflow['on'][trigger]['inputs']['master_date']['default']=='2026-09-28'


def test_job_explicit_today_master_is_reused(monkeypatch,tmp_path):
    import scripts.run_market_data_job as job
    observed=job.datetime.now(job.ZoneInfo('Asia/Seoul')).date().isoformat()
    monkeypatch.setenv('MARKET_STORE_PATH',str(tmp_path/'store'))
    monkeypatch.setenv('COLLECTION_MODE','master')
    monkeypatch.setenv('MASTER_DATE',observed)
    calls=[]
    def read_existing(store,date):
        calls.append(date)
        return pd.DataFrame({'ticker':['005930']})
    def no_download(*args,**kwargs):
        raise AssertionError('Explicit master date must reuse the stored snapshot, even today')
    monkeypatch.setattr(job,'read_master',read_existing)
    monkeypatch.setattr(job,'collect_masters',no_download)
    job.main()
    assert calls==[observed]
def test_zero_activity_inconsistent_quote_retained_but_never_usable_candle():
    import kis_market_collection as kis
    p={'rt_cd':'0','output1':{'stck_shrn_iscd':'0191M0'},'output2':[
        dict(stck_bsop_date='20261002',stck_oprc='100875',stck_hgpr='100875',
             stck_lwpr='100875',stck_clpr='100905',acml_vol='0',acml_tr_pbmn='0')]}
    f=kis.normalize_prices(p,'0191M0','20261002','20261002','raw')
    assert f.iloc[0]['close']==100905 and not kis.usable_ohlc(f).any()
    for field in ('acml_vol','acml_tr_pbmn'):
        p['output2'][0][field]='1'
        with pytest.raises(md.DataQualityError,match='range mismatch'):
            kis.normalize_prices(p,'0191M0','20261002','20261002','raw')
        p['output2'][0][field]='0'
