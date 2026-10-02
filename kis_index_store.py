"""Actual KOSPI/KOSDAQ index history, not ETF proxies. Read-only KIS APIs."""
import gzip
import json
import pandas as pd
from market_data_store import DataQualityError,canonical,day,number
from kis_market_collection import windows

INDEX_CODES={'KOSPI':'0001','KOSDAQ':'1001'}
INDEX_FIELDS={'open':'bstp_nmix_oprc','high':'bstp_nmix_hgpr',
              'low':'bstp_nmix_lwpr','close':'bstp_nmix_prpr'}


def normalize_index(payload,code,start,end):
    claimed=payload.get('output1',{}).get('bstp_cls_code')
    if claimed and str(claimed).zfill(4)!=code:
        raise DataQualityError('Index identity mismatch')
    if payload.get('rt_cd')!='0' or not isinstance(payload.get('output2'),list):
        raise DataQualityError('Invalid index response')
    rows=[]
    for r in payload['output2']:
        if not r or not r.get('stck_bsop_date'): continue
        date=day(r['stck_bsop_date'])
        if not day(start)<=date<=day(end):
            raise DataQualityError('Index date outside requested window')
        values={k:number(r.get(v)) for k,v in INDEX_FIELDS.items()}
        if any(v is None or v<=0 for v in values.values()):
            raise DataQualityError('Missing/nonpositive index OHLC')
        if not values['low']<=min(values['open'],values['close'])<=max(values['open'],values['close'])<=values['high']:
            raise DataQualityError('Invalid index OHLC range')
        rows.append(dict(date=date,**values))
    f=pd.DataFrame(rows,columns=['date',*INDEX_FIELDS])
    if f.empty or f.date.duplicated().any():
        raise DataQualityError('Empty/duplicate index history')
    return f.sort_values('date').reset_index(drop=True)


def collect_indices(store,client,start,end):
    for market,code in INDEX_CODES.items():
        for first,last in windows(start,end):
            key=f'{market}/{first}_{last}'
            record=store.manifest.get('kis_indices',{}).get(key)
            if record:
                store.verify_record(record)
                continue
            payload=client.fetch_index(code,first,last)
            frame=normalize_index(payload,code,first,last)
            store.put_table('kis_indices',key,frame,canonical(payload),dict(source='kis_index_period',
                market=market,code=code,start=first,end=last))


def load_indices(store,start,end):
    result={}
    for market,code in INDEX_CODES.items():
        frames=[]
        records=sorted(store.manifest.get('kis_indices',{}).values(),key=lambda r:r['collected_at'])
        for record in records:
            if record['market']!=market or record['start']>end or record['end']<start: continue
            if record['source']!='kis_index_period' or record['code']!=code:
                raise DataQualityError('Index metadata mismatch')
            store.verify_record(record)
            expected=normalize_index(json.loads(gzip.decompress(store.checked_path(record['raw_path']).read_bytes())),
                                     code,record['start'],record['end'])
            f=pd.read_parquet(store.checked_path(record['table_path']))
            pd.testing.assert_frame_equal(f,expected,check_dtype=False)
            frames.append(f)
        if not frames: raise DataQualityError(f'Missing {market} index history')
        f=pd.concat(frames).drop_duplicates('date',keep='last').sort_values('date')
        f=f.loc[f.date.between(start,end)].set_index('date')
        f.index=pd.to_datetime(f.index)
        if f.empty or f.index.max()!=pd.Timestamp(end):
            raise DataQualityError(f'Stale {market} index')
        result[market]=f
    if not result['KOSPI'].index.equals(result['KOSDAQ'].index):
        raise DataQualityError('Index session mismatch')
    return result
