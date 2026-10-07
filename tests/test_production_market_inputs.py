from datetime import datetime,timezone
import copy
import pandas as pd
import pytest
import daily_market_data as daily
import kis_market_collection as kis
from market_data_store import MarketStore,DataQualityError
from kis_index_store import collect_indices,load_indices,normalize_index
from production_market_inputs import load_production_inputs

NOW=datetime(2026,10,2,0,tzinfo=timezone.utc)
SHA='a'*40


class Client:
    def fetch_calendar(self,base):
        return {'rt_cd':'0','output':[dict(bass_dt=d.strftime('%Y%m%d'),opnd_yn='Y' if d.weekday()<5 else 'N')
            for d in pd.date_range(base,periods=31)]}

    def fetch(self,ticker,start,end,basis):
        scale=2 if basis=='adjusted' else 1
        return {'rt_cd':'0','output1':{'stck_shrn_iscd':ticker},'output2':[
            dict(stck_bsop_date=d.strftime('%Y%m%d'),stck_oprc=100*scale,stck_hgpr=110*scale,
                stck_lwpr=90*scale,stck_clpr=(105+int(d.weekday()==2))*scale,acml_vol=1000,acml_tr_pbmn=105000)
            for d in pd.bdate_range(start,end)]}

    def fetch_index(self,code,start,end):
        return {'rt_cd':'0','output1':{'bstp_cls_code':code},'output2':[
            dict(stck_bsop_date=d.strftime('%Y%m%d'),bstp_nmix_oprc=100,
                bstp_nmix_hgpr=110,bstp_nmix_lwpr=90,bstp_nmix_prpr=105)
            for d in pd.bdate_range(start,end)]}


def masters(store,observed_date):
    all_frames=[]
    for market,ticker in [('kospi','069500'),('kosdaq','005930')]:
        frame=pd.DataFrame([dict(ticker=ticker,isin='KR7'+ticker+'003',name='test',market=market,
            listed_date='2000-01-01',collection_eligible=True,asset_type='stock' if market=='kosdaq' else 'etf',
            classification='stock' if market=='kosdaq' else 'non_leveraged_name_unverified')])
        store.put_table('masters',observed_date+'/'+market,frame,b'master',{})
        all_frames.append(frame)
    return pd.concat(all_frames,ignore_index=True)


@pytest.fixture
def ready(tmp_path,monkeypatch):
    monkeypatch.setattr(daily,'_utc_now',lambda:NOW)
    store=MarketStore(tmp_path/'store'); client=Client()
    daily.run_daily(store,client,now=NOW,start='2026-01-01',master_loader=masters,delay=0)
    collect_indices(store,client,'2026-01-01','2026-10-01')
    return store


def test_complete_offline_bridge_keeps_real_index_and_raw_execution_price(ready,monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session,'request',lambda *a,**k:pytest.fail('Network in offline bridge'))
    enriched,indices,refs,meta=load_production_inputs(ready.root,start='2026-01-01',data_commit=SHA,now=NOW)
    assert set(enriched)=={'069500','005930'}
    assert enriched['005930'].iloc[-1]['close']==210
    assert refs['005930']==105
    assert indices['KOSPI'].iloc[-1]['close']==105
    assert meta['full_current_input_count']==2 and meta['data_commit']==SHA
    assert meta['index_source']=='kis_actual_indices'


def test_native_seven_child_signal_preparation_without_network_or_broker(ready,tmp_path,monkeypatch):
    import requests
    import multi_allocator_plus_trader as trader_module
    import production_market_inputs as bridge
    from signal_safety import validate_snapshot
    import json
    monkeypatch.setattr(requests.Session,'request',lambda *a,**k:pytest.fail('Network in signal preparation'))
    monkeypatch.setattr(trader_module,'KoreaInvestmentConnector',lambda **k:pytest.fail('Broker initialized'))
    original=bridge.load_production_inputs
    monkeypatch.setattr(bridge,'load_production_inputs',lambda *a,**k:original(*a,**k,now=NOW))
    t=trader_module.MultiAllocatorPlusTrader(start_date='2026-01-01',prepare_signal_only=True,
        market_store_path=str(ready.root),market_data_commit=SHA,dry_run=True)
    path=tmp_path/'signal.json'
    t._signal_snapshot_path=lambda *a:path
    t.run()
    payload=json.loads(path.read_text())
    validate_snapshot(payload,today='2026-10-02')
    assert len(t.strategy.latest_child_results)==7
    assert payload['meta']['market_inputs']['data_commit']==SHA
    assert t.kis is None


@pytest.mark.parametrize('mutation',['index','history','hash','stale','commit'])
def test_bridge_never_falls_back_after_bad_inputs(ready,mutation):
    commit=SHA; now=NOW
    if mutation=='index': ready.manifest['kis_indices']={}; ready.save()
    if mutation=='history':
        key=next(k for k in ready.manifest['kis_segments'] if k.startswith('005930/raw/'))
        del ready.manifest['kis_segments'][key]
        ready._save_kis_shard(key); ready.save()
    if mutation=='hash':
        r=next(iter(ready.manifest['kis_segments'].values()))
        ready.checked_path(r['table_path']).write_bytes(b'corrupt fixture')
    if mutation=='stale': now=datetime(2026,10,3,0,tzinfo=timezone.utc)
    if mutation=='commit': commit='main'
    with pytest.raises((DataQualityError,ValueError)):
        load_production_inputs(ready.root,start='2026-01-01',data_commit=commit,now=now)


def test_index_identity_and_range_required():
    p=Client().fetch_index('0001','2026-01-01','2026-01-05')
    with pytest.raises(DataQualityError): normalize_index(p,'1001','2026-01-01','2026-01-05')
    p['output2'][0]['bstp_nmix_prpr']=999
    with pytest.raises(DataQualityError): normalize_index(p,'0001','2026-01-01','2026-01-05')


def test_index_provider_fifty_row_cap_cannot_truncate_monthly_requests(tmp_path):
    class Capped(Client):
        def fetch_index(self,code,start,end):
            assert (pd.Timestamp(end)-pd.Timestamp(start)).days<=29
            p=super().fetch_index(code,start,end)
            p['output2']=p['output2'][-50:]
            return p
    store=MarketStore(tmp_path/'indices')
    collect_indices(store,Capped(),'2026-01-01','2026-10-01')
    loaded=load_indices(store,'2026-01-01','2026-10-01')
    assert loaded['KOSPI'].index.equals(pd.bdate_range('2026-01-01','2026-10-01'))


def test_eod_workflow_requires_collection_and_offline_no_order_verification():
    from pathlib import Path
    import yaml
    text=Path('.github/workflows/daily-eod-signal.yml').read_text()
    workflow=yaml.safe_load(text)
    assert workflow['jobs']['prep-signal']['needs']=='collect'
    assert workflow['jobs']['collect']['uses']=='./.github/workflows/market-data-daily.yml'
    prep=workflow['jobs']['prep-signal']
    assert 'KIS_APP_KEY' not in str(prep) and 'KIS_ACCOUNT' not in str(prep) and 'KRX_PW' not in str(prep)
    assert 'scripts/verify_kis_production_signal.py' in text
    assert "github.ref == 'refs/heads/main'" in text
    assert 'inputs.publish_signal' in text
    assert 'daily-open-exec-a' not in text


def test_verifier_uses_same_minimum_order_as_live(tmp_path,monkeypatch):
    import sys,re,json
    from pathlib import Path
    from scripts import verify_kis_production_signal as verifier
    settings={}
    def stop_before_data(**kwargs):
        settings.update(kwargs)
        raise RuntimeError('fixture: stop before data')
    monkeypatch.setattr(verifier,'MultiAllocatorPlusTrader',stop_before_data)
    report=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['verify','--store',str(tmp_path),
        '--data-commit',SHA,'--report',str(report)])
    with pytest.raises(RuntimeError,match='fixture'):verifier.main()
    workflow=Path('.github/workflows/daily-open-exec-a.yml').read_text()
    live_min=int(re.search(r'--min-trade\s+(\d+)',workflow).group(1))
    assert settings['min_trade_value']==live_min==10000
    assert settings['dry_run'] and settings['prepare_signal_only'] and settings['require_private_inputs']
    assert json.loads(report.read_text())['min_trade_value']==live_min


@pytest.mark.parametrize('permanent',[False,True])
def test_quality_retry_keeps_rejected_payload_private_not_in_prices(tmp_path,monkeypatch,permanent):
    store=MarketStore(tmp_path/'store'); master=masters(store,'2026-10-02').iloc[:1]
    monkeypatch.setattr(kis.time,'sleep',lambda seconds:None)
    client=Client(); real_fetch=client.fetch; calls=[]
    def fetch(*args):
        calls.append(args)
        payload=real_fetch(*args)
        if permanent or len(calls)==1:
            payload['output2'][0]['stck_hgpr']=1
        return payload
    client.fetch=fetch
    if permanent:
        with pytest.raises(kis.CollectionInterrupted) as exc:
            kis.collect_prices(store,client,master,'2026-10-02','2026-09-01','2026-10-01',delay=0)
        assert len(calls)==3 and not store.manifest['kis_segments']
        assert exc.value.progress['error']['reason']=='KIS OHLC range mismatch'
    else:
        result=kis.collect_prices(store,client,master,'2026-10-02','2026-09-01','2026-10-01',delay=0)
        assert result['requested']==2 and len(calls)==3
    rejected=store.manifest['quarantines']
    assert len(rejected)==1 and not next(iter(rejected.values()))['eligible_for_prices']
    for r in rejected.values(): store.verify_record(r)
