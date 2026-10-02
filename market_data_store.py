"""Versioned, offline-first market snapshots; deliberately independent of orders.

Official market-wide dates and partial legacy imports are separate namespaces.
Raw prices are never silently replaced with adjusted prices or forward-filled.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

import pandas as pd
import requests

SCHEMA = 1
GROUPS = ('kospi', 'kosdaq', 'etf')
ENDPOINTS = {'kospi': 'sto/stk_bydd_trd', 'kosdaq': 'sto/ksq_bydd_trd', 'etf': 'etp/etf_bydd_trd'}
API_BASE = 'https://data-dbg.krx.co.kr/svc/apis/'
FIELDS = {'open':'TDD_OPNPRC', 'high':'TDD_HGPRC', 'low':'TDD_LWPRC',
          'close':'TDD_CLSPRC', 'volume':'ACC_TRDVOL', 'value':'ACC_TRDVAL'}
MIN_ROWS = {'kospi': 500, 'kosdaq': 500, 'etf': 100}


class DataQualityError(ValueError):
    pass


def day(value):
    value = str(value)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}|\d{8}', value):
        raise ValueError('Use YYYY-MM-DD or YYYYMMDD')
    return pd.Timestamp(value).strftime('%Y-%m-%d')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temp = Path(handle.name)
    temp.replace(path)


def number(value):
    if value in (None, '', '-'):
        return None
    try:
        result = float(str(value).replace(',', ''))
    except ValueError:
        raise DataQualityError('Non-numeric API price/value') from None
    if not math.isfinite(result) or result < 0:
        raise DataQualityError('Negative or non-finite price/value')
    return result


def etf_name_classification(name):
    """A conservative exclusion screen, NOT authoritative leverage metadata."""
    name = name.upper().replace(' ', '')
    if re.search(r'레버리지|LEVERAGE|(?<!\d)(?:[2-9](?:\.\d+)?|1\.\d+)X|[2-9]배(?!당)', name):
        return 'excluded_leverage_or_multiplier_name'
    if '인버스' in name or 'INVERSE' in name:
        return 'inverse_name_unverified'
    return 'non_leveraged_name_unverified'


def normalize_api(payload, group, date):
    if group not in GROUPS:
        raise DataQualityError('Unknown market group')
    rows = payload.get('OutBlock_1') if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise DataQualityError('Missing OutBlock_1: empty/auth/error responses are not market data')
    result, excluded, identities = [], [], set()
    for row in rows:
        if not isinstance(row, dict) or day(row.get('BAS_DD', '')) != day(date):
            raise DataQualityError('API returned a different date')
        ticker, name = str(row.get('ISU_CD', '')), str(row.get('ISU_NM', '')).strip()
        if not re.fullmatch('[0-9A-Z]{6}', ticker) or not name:
            raise DataQualityError('Unrecognized security identity; do not infer ticker from ISIN')
        if ticker in identities:
            raise DataQualityError('Duplicate security on a market date')
        identities.add(ticker)
        values = {key:number(row.get(field)) for key,field in FIELDS.items()}
        if any(v is None for v in values.values()):
            raise DataQualityError('Missing OHLC/volume/value; never substitute close*volume')
        valid_prices = all(values[k] > 0 for k in ('open','high','low','close'))
        if valid_prices and not (values['low'] <= min(values['open'], values['close'])
                <= max(values['open'], values['close']) <= values['high']):
            raise DataQualityError('OHLC range inconsistency')
        if not valid_prices and (values['volume'] > 0 or values['value'] > 0):
            raise DataQualityError('Trading activity with missing execution prices')
        classification = etf_name_classification(name) if group == 'etf' else 'stock'
        if classification.startswith('excluded_'):
            excluded.append({'ticker':ticker, 'name':name, 'reason':classification})
            continue
        result.append(dict(date=day(date), ticker=ticker, name=name, group=group,
            asset_type='etf' if group == 'etf' else 'stock', price_basis='raw',
            classification=classification, classification_verified=group != 'etf',
            tradable=valid_prices and values['volume'] > 0 and values['value'] > 0,
            market_cap=number(row.get('MKTCAP')), **values))
    return pd.DataFrame(result), excluded, len(rows)


class KRXClient:
    """No browser password, cookie, broker credential or alternate-source fallback."""
    def __init__(self, key, session=None):
        if not key or not key.strip():
            raise ValueError('KRX_OPENAPI_KEY is required (approved services also required)')
        self._key = key
        self.session = session or requests.Session()

    def fetch(self, group, date):
        if group not in ENDPOINTS:
            raise ValueError('Unknown API service')
        try:
            response = self.session.get(API_BASE+ENDPOINTS[group],
                params={'basDd':day(date).replace('-', '')}, headers={'AUTH_KEY':self._key},
                timeout=(10, 45), allow_redirects=False)
        except requests.RequestException:
            raise RuntimeError('KRX connection failed; checkpoint retained, no fallback') from None
        if response.status_code != 200:
            raise RuntimeError(f'KRX HTTP {response.status_code}; check service approval/quota. No response body logged.')
        try:
            payload = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            raise DataQualityError('KRX non-JSON response; not an empty market day') from None
        # Reject service errors before persisting anything as a successful day.
        normalize_api(payload, group, date)
        return payload


class MarketStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root/'manifest.json'
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get('schema_version') != SCHEMA:
                raise DataQualityError('Unsupported store schema')
        else:
            self.manifest = dict(schema_version=SCHEMA, snapshots={}, seeds={},
                corporate_actions_complete=False, distributions_complete=False,
                full_universe_certified=False, orders_enabled=False)
        self._indexes=self.manifest.pop('kis_index_shards',{})
        self.manifest.setdefault('kis_segments',{})
        for record in self._indexes.values():
            data=self._index_bytes(record)
            self.manifest['kis_segments'].update(json.loads(data))

    def _index_bytes(self,record):
        # A crash between replacing a shard and replacing the small root
        # manifest recovers the last committed shard, never an uncommitted one.
        for relative in (record['path'],record['path']+'.previous'):
            path=self.checked_path(relative)
            if path.exists():
                data=path.read_bytes()
                if digest(data)==record['sha256']:
                    return data
        raise DataQualityError('Corrupt KIS manifest shard')

    def save(self):
        serial=dict(self.manifest)
        serial.pop('kis_segments',None)
        serial['kis_index_shards']=self._indexes
        atomic_bytes(self.manifest_path, canonical(serial))

    def _save_kis_shard(self,key):
        shard=key[:2]
        records={k:r for k,r in self.manifest['kis_segments'].items() if k.startswith(shard)}
        data=canonical(records)
        path=f'indexes/kis/{shard}.json'
        if shard in self._indexes:
            atomic_bytes(self.checked_path(path+'.previous'),self._index_bytes(self._indexes[shard]))
        atomic_bytes(self.checked_path(path),data)
        self._indexes[shard]={'path':path,'sha256':digest(data)}

    def put_table(self, namespace, key, frame, original, metadata):
        """Content-addressed import/KIS layer; cannot impersonate full-market KRX."""
        if namespace not in ('seeds', 'kis_segments', 'masters', 'calendars', 'daily_inputs'):
            raise ValueError('Unsupported namespace')
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', key) or '..' in key:
            raise ValueError('Invalid record key')
        import io
        raw = gzip.compress(original, mtime=0)
        version = digest(raw)
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        table_hash=digest(buf.getvalue())
        records = self.manifest.setdefault(namespace, {})
        previous = records.get(key)
        if previous and previous['raw_sha256'] == version and previous['table_sha256']==table_hash:
            self.verify_record(previous)
            return False
        prefix = f'{namespace}/{key}/{version}_{table_hash}'
        raw_path, table_path = f'{prefix}.gz', f'{prefix}.parquet'
        atomic_bytes(self.checked_path(raw_path), raw)
        atomic_bytes(self.checked_path(table_path), buf.getvalue())
        records[key] = dict(**metadata, rows=len(frame), raw_path=raw_path, raw_sha256=version,
            table_path=table_path, table_sha256=digest(buf.getvalue()),
            collected_at=datetime.now(timezone.utc).isoformat(),
            previous_version=previous['raw_sha256'] if previous else None)
        if namespace=='kis_segments':
            self._save_kis_shard(key)
        self.save()
        return True

    def checked_path(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root):
            raise DataQualityError('Manifest path escapes store')
        return path

    def verify_record(self, record):
        for path_key, hash_key in (('raw_path','raw_sha256'), ('table_path','table_sha256')):
            path = self.checked_path(record[path_key])
            if not path.is_file() or digest(path.read_bytes()) != record[hash_key]:
                raise DataQualityError('Missing/corrupt snapshot; not eligible for reuse')

    def has(self, group, date):
        record = self.manifest['snapshots'].get(f'{group}/{day(date)}')
        if record is None:
            return False
        self.verify_record(record)
        return True

    def ingest(self, group, date, payload):
        date = day(date)
        frame, excluded, source_count = normalize_api(payload, group, date)
        if source_count and source_count < MIN_ROWS[group]:
            raise DataQualityError('Suspiciously small market response: refused as a full snapshot')
        raw = gzip.compress(canonical(payload), mtime=0)
        version = digest(raw)
        import io
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        table_hash=digest(buf.getvalue())
        key = f'{group}/{date}'
        previous = self.manifest['snapshots'].get(key)
        if previous and previous['raw_sha256'] == version and previous['table_sha256']==table_hash:
            self.verify_record(previous)
            return False
        # Raw responses retain all source rows for audit; leveraged ETFs are NOT
        # admitted to the normalized eligible collection. No source mutation.
        prefix = f'snapshots/{group}/{date}/{version}_{table_hash}'
        raw_path, table_path = f'{prefix}.json.gz', f'{prefix}.parquet'
        atomic_bytes(self.checked_path(raw_path), raw)
        atomic_bytes(self.checked_path(table_path), buf.getvalue())
        record = dict(group=group, date=date, source='krx_openapi', price_basis='raw',
            coverage='market_snapshot' if source_count else 'empty_unconfirmed',
            source_rows=source_count, rows=len(frame), excluded=excluded,
            raw_path=raw_path, raw_sha256=version, table_path=table_path,
            table_sha256=digest(buf.getvalue()), collected_at=datetime.now(timezone.utc).isoformat(),
            previous_version=previous['raw_sha256'] if previous else None)
        self.manifest['snapshots'][key] = record
        self.save()
        return True

    def market_day(self, date):
        frames, versions = [], {}
        for group in GROUPS:
            key = f'{group}/{day(date)}'
            record = self.manifest['snapshots'].get(key)
            if not record or record['coverage'] != 'market_snapshot':
                raise DataQualityError(f'Incomplete official market date: {key}')
            self.verify_record(record)
            frames.append(pd.read_parquet(self.checked_path(record['table_path'])))
            versions[group] = record['raw_sha256']
        frame = pd.concat(frames, ignore_index=True)
        if frame.ticker.duplicated().any():
            raise DataQualityError('Cross-market duplicate ticker: identity review required')
        return frame, versions

    def load_panel(self, tickers, dates):
        """Explicit calendar required. Missing securities remain missing, not filled."""
        wanted = set(tickers)
        frames = [self.market_day(date)[0] for date in dates]
        panel = pd.concat(frames, ignore_index=True)
        return panel.loc[panel.ticker.isin(wanted)].sort_values(['ticker','date']).reset_index(drop=True)

    def load_seed_panel(self, tickers, start, end, *, price_basis):
        """Explicit partial research reader. Never promoted to market_day()."""
        if price_basis not in ('raw','adjusted'):
            raise ValueError('Explicit price basis required')
        frames=[]
        for record in self.manifest['seeds'].values():
            if record['ticker'] not in tickers or record['price_basis']!=price_basis:
                continue
            if record['end']<day(start) or record['start']>day(end):
                continue
            self.verify_record(record)
            frame=pd.read_parquet(self.checked_path(record['table_path']))
            frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
            frame=frame.loc[frame.date.between(day(start),day(end))]
            frame['source_version']=record['raw_sha256']
            frames.append(frame)
        if not frames:
            raise DataQualityError('No seed rows in requested range/basis')
        panel=pd.concat(frames,ignore_index=True)
        # Source overlap may be identical, but conflicting observations must not
        # be silently resolved by file order.
        for _,g in panel.loc[panel.duplicated(['ticker','date'],keep=False)].groupby(['ticker','date']):
            if g[['open','high','low','close']].drop_duplicates().shape[0]>1:
                raise DataQualityError('Conflicting seed observations')
        return panel.drop_duplicates(['ticker','date']).sort_values(['ticker','date']).reset_index(drop=True)

    def select_universe(self, date, *, min_value=0, max_price=None, budget=None,
                        include_unverified_etfs=False, include_inverse=False, limit=None):
        """Data-only, as-of-day eligibility; NOT a trading strategy or order signal.

        By default ETFs with only name-based classification are quarantined.
        Explicit exploratory inclusion still emits orders_enabled=False.
        """
        frame, versions = self.market_day(date)
        eligible = frame.tradable & frame.value.ge(min_value)
        if not include_unverified_etfs:
            eligible &= frame.classification_verified
        if not include_inverse:
            eligible &= ~frame.classification.str.startswith('inverse_')
        if max_price is not None:
            eligible &= frame.close.le(max_price)
        if budget is not None:
            eligible &= (frame.close*1.003).le(budget)  # Explicit screening buffer, not fill modelling.
        candidates = frame.loc[eligible].sort_values(['value','ticker'], ascending=[False,True])
        if limit is not None:
            candidates = candidates.head(limit)
        return dict(as_of=day(date), available_for='after_close_only', orders_enabled=False,
            purpose='liquidity_ranked_research_universe_not_buy_signal', source_versions=versions,
            etf_classification='name_screen_only_unverified',
            candidates=json.loads(candidates.to_json(orient='records')))
