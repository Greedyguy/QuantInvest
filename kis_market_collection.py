"""KIS read-only price collection. No account number and no order API surface."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
import os
import re
import time
import zipfile

import pandas as pd
import requests

from market_data_store import (DataQualityError, canonical, day, digest,
                               etf_name_classification, number)

MASTER_URLS = {m:f'https://new.real.download.dws.co.kr/common/master/{m}_code.mst.zip'
               for m in ('kospi','kosdaq')}
PRICE_PATH = '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
BASE = 'https://openapi.koreainvestment.com:9443'
PRICE_FIELDS = {'open':'stck_oprc','high':'stck_hgpr','low':'stck_lwpr',
                'close':'stck_clpr','volume':'acml_vol','value':'acml_tr_pbmn'}
STOCK_GROUPS = {'ST','RT','IF','MF','DR','FS'}


class KISPriceRequestError(RuntimeError):
    """Only bounded, sanitized diagnostics; never broker bodies or exceptions."""
    def __init__(self, kind, attempts, *, http_status=None, provider_code=None):
        self.diagnostic = dict(kind=kind, attempts=attempts,
                               http_status=http_status, provider_code=provider_code)
        super().__init__('KIS price request failed: '+json.dumps(self.diagnostic)+
                         '; validated checkpoint retained; do not skip this segment')


class CollectionInterrupted(RuntimeError):
    def __init__(self, progress):
        self.progress = progress
        super().__init__('KIS collection interrupted; validated segments retained; '
                         'resume at '+progress['next_key']+'; '+json.dumps(progress['error']))


def parse_master(blob, market):
    # Official KIS examples use decoded characters and include LF in the tail.
    tail = {'kospi':228, 'kosdaq':222}[market]
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        info=archive.getinfo(f'{market}_code.mst')
        if info.file_size>20_000_000:
            raise DataQualityError('Unexpected master size')
        text=archive.read(info).decode('cp949').replace('\r\n','\n')
    rows=[]
    for line in text.splitlines(keepends=True):
        if not line.endswith('\n') or len(line)<=tail+21:
            raise DataQualityError('Unexpected KIS master fixed-width layout')
        prefix, suffix=line[:-tail],line[-tail:]
        ticker,isin,name=prefix[:9].strip(),prefix[9:21].strip(),prefix[21:].strip()
        group=suffix[:2]
        if group not in STOCK_GROUPS|{'EF'}:
            continue  # ETNs, warrants, subscription rights and non-listed funds.
        if not re.fullmatch('[0-9A-Z]{6}',ticker) or not re.fullmatch('[A-Z]{2}[0-9A-Z]{10}',isin) or not name:
            raise DataQualityError('Invalid master security identity')
        classification=etf_name_classification(name) if group=='EF' else 'stock'
        offset=105 if market=='kospi' else 100
        listed=suffix[offset:offset+8]
        if not re.fullmatch(r'\d{8}',listed):
            raise DataQualityError('Unrecognized listing date in master')
        rows.append(dict(ticker=ticker,isin=isin,name=name,market=market,security_group=group,
            listed_date=day(listed),
            asset_type='etf' if group=='EF' else 'stock',classification=classification,
            collection_eligible=not classification.startswith('excluded_'),
            classification_verified=group!='EF'))
    frame=pd.DataFrame(rows)
    if len(frame)<500 or frame.ticker.duplicated().any():
        raise DataQualityError('Incomplete/duplicate master list')
    return frame


def collect_masters(store, *, observed_date, session=None):
    session=session or requests.Session()
    frames=[]
    for market,url in MASTER_URLS.items():
        try:
            response=session.get(url,timeout=(10,45),allow_redirects=False)
        except requests.RequestException:
            raise RuntimeError('KIS public master download failed') from None
        if response.status_code!=200:
            raise RuntimeError(f'KIS master HTTP {response.status_code}')
        frame=parse_master(response.content,market)
        store.put_table('masters',f'{day(observed_date)}/{market}',frame,response.content,
            dict(source='kis_public_master',observed_date=day(observed_date),market=market,
                 coverage='current_snapshot_not_historical_universe',source_url=url))
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    if result.ticker.duplicated().any():
        raise DataQualityError('Cross-market master identity conflict')
    return result


def read_master(store, observed_date):
    frames=[]
    for market in MASTER_URLS:
        record=store.manifest.get('masters',{}).get(f'{day(observed_date)}/{market}')
        if not record:
            raise DataQualityError('Both KIS masters must be collected first')
        store.verify_record(record)
        frames.append(pd.read_parquet(store.checked_path(record['table_path'])))
    return pd.concat(frames,ignore_index=True)


class KISPriceClient:
    def __init__(self, app_key, app_secret, *, session=None, demo=False):
        if not app_key or not app_secret:
            raise ValueError('KIS_APP_KEY and KIS_APP_SECRET are required')
        self._key,self._secret=app_key,app_secret
        self._token=None
        self._expiry=0
        self.session=session or requests.Session()
        self.base='https://openapivts.koreainvestment.com:29443' if demo else BASE

    def _auth(self):
        if self._token and time.monotonic()<self._expiry:
            return
        try:
            response=self.session.post(self.base+'/oauth2/tokenP',json={
                'grant_type':'client_credentials','appkey':self._key,'appsecret':self._secret},
                timeout=(10,45),allow_redirects=False)
            body=response.json()
        except (requests.RequestException,ValueError):
            raise RuntimeError('KIS authentication failed; credentials/response not logged') from None
        if response.status_code!=200 or not body.get('access_token'):
            code=str(body.get('error_code',''))
            code=code if re.fullmatch('[A-Z0-9_]{1,30}',code) else 'unavailable'
            raise RuntimeError(f'KIS authentication rejected (HTTP {response.status_code}, code {code}); credentials/response not logged')
        self._token=body['access_token']
        self._expiry=time.monotonic()+max(0,int(body.get('expires_in',3600))-120)

    def fetch(self,ticker,start,end,basis):
        if not re.fullmatch('[0-9A-Z]{6}',ticker) or basis not in ('raw','adjusted'):
            raise ValueError('Invalid ticker/basis')
        start,end=day(start),day(end)
        if not 0 <= (pd.Timestamp(end)-pd.Timestamp(start)).days <= 89:
            raise ValueError('KIS request windows are capped at 90 calendar days (<100 daily rows)')
        self._auth()
        params={'FID_COND_MRKT_DIV_CODE':'J','FID_INPUT_ISCD':ticker,
                'FID_INPUT_DATE_1':start.replace('-',''),'FID_INPUT_DATE_2':end.replace('-',''),
                'FID_PERIOD_DIV_CODE':'D','FID_ORG_ADJ_PRC':'1' if basis=='raw' else '0'}
        for attempt in range(1,6):
            status,code=None,None
            retryable=False
            retry_after=0
            try:
                response=self.session.get(self.base+PRICE_PATH,params=params,headers={
                    'authorization':'Bearer '+self._token,'appkey':self._key,'appsecret':self._secret,
                    'tr_id':'FHKST03010100','custtype':'P'},timeout=(10,45),allow_redirects=False)
                status=response.status_code
                if status!=200:
                    kind='http_error'
                    retryable=status in (408,429,500,502,503,504)
                    # Respect bounded numeric Retry-After without logging headers.
                    header=str(getattr(response,'headers',{}).get('Retry-After',''))
                    if re.fullmatch(r'\d{1,3}',header):
                        retry_after=min(int(header),300)
                else:
                    body=response.json()
                    if not isinstance(body,dict):
                        kind='invalid_json_shape'
                    elif body.get('rt_cd')=='0':
                        return body
                    else:
                        kind='provider_error'
                        value=str(body.get('msg_cd',''))
                        code=value if re.fullmatch(r'[A-Z]{3,4}\d{4,5}',value) else None
                        retryable=code=='EGW00201'
            except requests.exceptions.SSLError:
                kind='tls_error'  # Do not retry certificate/configuration failures.
            except requests.Timeout:
                kind='timeout';retryable=True
            except requests.ConnectionError:
                kind='connection_error';retryable=True
            except requests.exceptions.ChunkedEncodingError:
                kind='incomplete_response';retryable=True
            except requests.exceptions.JSONDecodeError:
                kind='invalid_json';retryable=True
            except requests.RequestException:
                kind='request_error'
            except ValueError:
                kind='invalid_json';retryable=True
            diagnostic=KISPriceRequestError(kind,attempt,http_status=status,provider_code=code)
            if not retryable or attempt==5:
                raise diagnostic from None
            delay=max(2**attempt,retry_after,60 if status==429 or code=='EGW00201' else 0)
            print(json.dumps(dict(event='kis_price_retry',ticker=ticker,basis=basis,
                start=start,end=end,next_attempt=attempt+1,wait_seconds=delay,
                **diagnostic.diagnostic)),flush=True)
            time.sleep(delay)

    def fetch_security_info(self, ticker):
        """Read-only identity/lifecycle evidence; never infer dates from prices."""
        if not re.fullmatch('[0-9A-Z]{6}', ticker):
            raise ValueError('Invalid security ticker')
        self._auth()
        try:
            response = self.session.get(self.base+'/uapi/domestic-stock/v1/quotations/search-stock-info',
                params={'PRDT_TYPE_CD':'300', 'PDNO':ticker},
                headers={'authorization':'Bearer '+self._token,'appkey':self._key,'appsecret':self._secret,
                         'tr_id':'CTPF1002R','custtype':'P'},timeout=(10,45),allow_redirects=False)
            body = response.json()
        except (requests.RequestException, ValueError):
            raise RuntimeError('KIS security information request failed; no inferred status') from None
        if response.status_code != 200 or not isinstance(body,dict) or body.get('rt_cd') != '0':
            raise RuntimeError('KIS security information rejected; no inferred status')
        return body

    def fetch_calendar(self, base_date):
        """KIS asks that CTCA0903R be called sparingly, preferably once per day.

        The daily orchestrator persists and reuses this response; it never polls
        individual dates or follows pages just to obtain a full holiday database.
        """
        self._auth()
        try:
            response=self.session.get(self.base+'/uapi/domestic-stock/v1/quotations/chk-holiday',
                params={'BASS_DT':day(base_date).replace('-',''), 'CTX_AREA_FK':'', 'CTX_AREA_NK':''},
                headers={'authorization':'Bearer '+self._token,'appkey':self._key,'appsecret':self._secret,
                         'tr_id':'CTCA0903R','custtype':'P'}, timeout=(10,45),allow_redirects=False)
            body=response.json()
        except (requests.RequestException,ValueError):
            raise RuntimeError('KIS calendar request failed; do not infer holidays') from None
        if response.status_code!=200 or body.get('rt_cd')!='0':
            raise RuntimeError('KIS calendar request rejected; do not infer holidays')
        return body

    def fetch_index(self, code, start, end):
        if code not in ('0001','1001') or not 0 <= (pd.Timestamp(day(end))-pd.Timestamp(day(start))).days <= 89:
            raise ValueError('Invalid index/window')
        self._auth()
        try:
            response=self.session.get(self.base+'/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice',
                params={'FID_COND_MRKT_DIV_CODE':'U','FID_INPUT_ISCD':code,
                    'FID_INPUT_DATE_1':day(start).replace('-',''),'FID_INPUT_DATE_2':day(end).replace('-',''),
                    'FID_PERIOD_DIV_CODE':'D'},headers={'authorization':'Bearer '+self._token,
                    'appkey':self._key,'appsecret':self._secret,'tr_id':'FHKUP03500100','custtype':'P'},
                timeout=(10,45),allow_redirects=False)
            body=response.json()
        except (requests.RequestException,ValueError):
            raise RuntimeError('KIS index request failed; no ETF proxy fallback') from None
        if response.status_code!=200 or body.get('rt_cd')!='0':
            raise RuntimeError('KIS index response rejected; no ETF proxy fallback')
        return body


def normalize_prices(payload,ticker,start,end,basis):
    rows=payload.get('output2')
    if payload.get('rt_cd')!='0' or not isinstance(rows,list):
        raise DataQualityError('Invalid KIS response')
    claimed=payload.get('output1',{}).get('stck_shrn_iscd')
    if claimed and claimed!=ticker:
        raise DataQualityError('KIS returned another ticker')
    result=[]
    for row in rows:
        if not row or not row.get('stck_bsop_date'):
            continue  # KIS may return blank padding rows.
        date=day(row['stck_bsop_date'])
        if not day(start)<=date<=day(end):
            raise DataQualityError('KIS returned a date outside requested window')
        values={k:number(row.get(v)) for k,v in PRICE_FIELDS.items()}
        if any(v is None for v in values.values()):
            raise DataQualityError('Missing KIS price/value; no synthetic fill')
        if all(values[k]>0 for k in ('open','high','low','close')):
            if not values['low']<=min(values['open'],values['close'])<=max(values['open'],values['close'])<=values['high']:
                raise DataQualityError('KIS OHLC range mismatch')
        elif values['volume']>0 or values['value']>0:
            raise DataQualityError('KIS trading activity with zero OHLC')
        result.append(dict(date=date,ticker=ticker,price_basis=basis,**values))
    frame=pd.DataFrame(result,columns=['date','ticker','price_basis',*PRICE_FIELDS])
    if frame.date.duplicated().any():
        raise DataQualityError('KIS duplicate date')
    return frame.sort_values('date').reset_index(drop=True)


def windows(start,end):
    start,end=pd.Timestamp(day(start)),pd.Timestamp(day(end))
    if start>end:
        raise ValueError('Inverted range')
    # Stable 90-calendar-day slices for repeated invocations of the same range.
    cursor=start
    while cursor<=end:
        finish=min(cursor+pd.Timedelta(days=89),end)
        yield cursor.strftime('%Y-%m-%d'),finish.strftime('%Y-%m-%d')
        cursor=finish+pd.Timedelta(days=1)


def collect_prices(store,client,master,master_date,start,end,*,max_requests=300,refresh=False,delay=.6):
    if not 1<=max_requests<=20000:
        raise ValueError('Invalid request budget')
    requested,reused=0,0
    eligible=master.loc[master.collection_eligible].sort_values('ticker')
    for first,last in windows(start,end):
        for row in eligible.itertuples():
            if row.listed_date>last:
                continue
            request_first=max(first,row.listed_date)
            for basis in ('raw','adjusted'):
                key=f'{row.ticker}/{basis}/{request_first}_{last}'
                previous=store.manifest.get('kis_segments',{}).get(key)
                if previous and not refresh:
                    store.verify_record(previous)
                    if previous.get('isin')!=row.isin:
                        raise DataQualityError('Ticker reuse/ISIN change requires manual identity reconciliation')
                    reused+=1
                    continue
                if requested>=max_requests:
                    return dict(status='checkpoint_budget_exhausted',requested=requested,reused=reused,
                        next_key=key,requested_universe_queried=False,full_universe_certified=False)
                try:
                    for attempt in range(1,4):
                        payload=client.fetch(row.ticker,request_first,last,basis)
                        try:
                            frame=normalize_prices(payload,row.ticker,request_first,last,basis)
                            break
                        except DataQualityError as quality:
                            # Rejected observations never enter the reusable price layer.
                            reason=str(quality)
                            store.put_table('quarantines',key+'/'+digest(canonical(payload)),
                                pd.DataFrame([dict(reason=reason)]),canonical(payload),dict(
                                    source='kis_rejected_price_response',ticker=row.ticker,
                                    start=request_first,end=last,price_basis=basis,reason=reason,
                                    eligible_for_prices=False))
                            if attempt==3 or reason=='KIS returned another ticker':
                                raise
                            print(json.dumps(dict(event='kis_quality_retry',ticker=row.ticker,
                                basis=basis,next_attempt=attempt+1,reason=reason)),flush=True)
                            time.sleep(2**attempt)
                except (RuntimeError,ValueError) as error:
                    diagnostic=(error.diagnostic if isinstance(error,KISPriceRequestError)
                                else dict(kind='quality_error' if isinstance(error,DataQualityError)
                                          else 'collection_error',
                                          reason=str(error) if isinstance(error,DataQualityError) else None))
                    progress=dict(status='blocked_collection_error',requested=requested,reused=reused,
                        next_key=key,requested_universe_queried=False,full_universe_certified=False,
                        error=diagnostic)
                    raise CollectionInterrupted(progress) from None
                store.put_table('kis_segments',key,frame,canonical(payload),dict(
                    source='kis_period_price',ticker=row.ticker,isin=row.isin,price_basis=basis,
                    start=request_first,end=last,master_date=day(master_date),
                    coverage='current_universe_history_not_pit',
                    response_status='observed_rows' if len(frame) else 'empty_unconfirmed'))
                requested+=1
                if requested%20==0:
                    print(json.dumps({'price_requests':requested,'reused':reused,'last_ticker':row.ticker}),flush=True)
                if delay:
                    time.sleep(delay)
    return dict(status='requested_universe_queried',requested=requested,reused=reused,
        requested_universe_queried=True,full_universe_certified=False,
        note='Current master excludes historical delistings; empty periods are unconfirmed, not zero returns')


def load_kis_panel(store,tickers,start,end,basis='raw'):
    if basis not in ('raw','adjusted'):
        raise ValueError('Explicit raw/adjusted basis required')
    frames=[]
    seen_identities={}
    for record in sorted(store.manifest.get('kis_segments',{}).values(),key=lambda r:r['collected_at']):
        if record['ticker'] not in tickers or record['price_basis']!=basis:
            continue
        if record['end']<day(start) or record['start']>day(end):
            continue
        if record['ticker'] in seen_identities and seen_identities[record['ticker']]!=record['isin']:
            raise DataQualityError('Mixed ISIN histories for ticker')
        seen_identities[record['ticker']]=record['isin']
        store.verify_record(record)
        frame=pd.read_parquet(store.checked_path(record['table_path']))
        frame['source_version']=record['raw_sha256']
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise DataQualityError('No stored KIS prices for requested range')
    panel=pd.concat(frames,ignore_index=True)
    panel=panel.loc[panel.date.between(day(start),day(end))]
    # Explicit latest-observation precedence for overlapping corrections. Older
    # inputs remain reproducible by pinning the private data repository commit.
    return panel.drop_duplicates(['ticker','date'],keep='last').sort_values(['ticker','date']).reset_index(drop=True)


def select_kis(store,master_date,as_of,*,allow_historical_master=False,include_etfs=False,min_value=1e9,limit=100):
    master_date,as_of=day(master_date),day(as_of)
    if not allow_historical_master and master_date!=as_of:
        raise DataQualityError('Selection requires same-day master; no future/stale universe substitution')
    master=read_master(store,master_date)
    eligible=master.loc[master.collection_eligible].copy()
    if not include_etfs:
        eligible=eligible.loc[eligible.asset_type.eq('stock')]
    eligible=eligible.loc[~eligible.classification.str.startswith('inverse_')]
    panel=load_kis_panel(store,set(eligible.ticker),as_of,as_of,'raw')
    missing=sorted(set(eligible.ticker)-set(panel.ticker))
    if missing:
        raise DataQualityError(f'Selection blocked: {len(missing)} eligible names lack exact-date prices')
    result=eligible.merge(panel,on='ticker',validate='one_to_one')
    result=result.loc[result.close.gt(0)&result.open.gt(0)&result.volume.gt(0)&result.value.ge(min_value)]
    result=result.sort_values(['value','ticker'],ascending=[False,True]).head(limit)
    return dict(as_of=as_of,master_date=master_date,orders_enabled=False,
        purpose='liquidity_ranked_research_universe_not_buy_signal',
        historical_universe_verified=not allow_historical_master,
        etf_classification_verified=not include_etfs,
        available_for='after_close_only',candidates=json.loads(result.to_json(orient='records')))
