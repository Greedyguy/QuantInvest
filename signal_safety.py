"""Versioned KR signal contract and unlevered order-boundary validation."""
import numpy as np
import pandas as pd
import re

SIGNAL_PATH_VERSION = 'kr-signal-repair-v1'
KR_BUY_BLOCKED = frozenset({'305720'})  # Industry ETF, not the claimed US Treasury hedge.


def validate_targets(targets, max_security_weight=.30):
    row = pd.Series(targets, dtype=float)
    if row.empty or row.index.has_duplicates or '__CASH__' not in row:
        raise ValueError('Missing or duplicate target fields')
    if not np.isfinite(row.to_numpy()).all() or (row < 0).any():
        raise ValueError('Targets must be finite and nonnegative')
    assets = row.drop('__CASH__')
    if assets.sum() > 1 + 1e-9 or abs(row.sum()-1) > 1e-8:
        raise ValueError('Unlevered target budget must sum to one including cash')
    if (assets > max_security_weight + 1e-9).any():
        raise ValueError('Per-security target limit exceeded')
    return row


def constrain_targets(targets, max_security_weight=.30):
    """Producer only: preserve underinvested budgets; shrink excess to limits."""
    row = pd.Series(targets, dtype=float)
    if row.empty or row.index.has_duplicates or not np.isfinite(row.to_numpy()).all() or (row < 0).any():
        raise ValueError('Invalid generated targets')
    assets = row.drop('__CASH__', errors='ignore').clip(upper=max_security_weight)
    total = float(assets.sum())
    if total > 1:
        assets = assets / total
    assets = assets.loc[assets > 0].sort_values(ascending=False)
    result = pd.concat([assets, pd.Series({'__CASH__':max(0.,1-float(assets.sum()))})])
    return validate_targets(result,max_security_weight)


def validate_snapshot(payload, *, today=None, max_security_weight=.30, require_private_inputs=False):
    meta = payload.get('meta') or {}
    if meta.get('signal_path_version') != SIGNAL_PATH_VERSION:
        raise ValueError('Old or missing signal path version; regenerate EOD snapshot')
    if meta.get('market') != 'kr' or meta.get('allocation_policy') != 'legacy':
        raise ValueError('Snapshot market/allocation policy mismatch')
    if payload.get('strategy') != 'multi_allocator_plus_safe_etf_kqm':
        raise ValueError('Snapshot strategy mismatch')
    signal_date = pd.Timestamp(payload.get('signal_date'))
    current = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None)
    if pd.isna(signal_date) or signal_date.tzinfo is not None or signal_date != signal_date.normalize():
        raise ValueError('Invalid EOD signal date')
    current = current.normalize()
    # Conservative weekday limit. Long exchange holidays can pause trading;
    # never silently execute an older snapshot just to keep a batch running.
    if signal_date >= current or np.busday_count(signal_date.date(),current.date()) > 3:
        raise ValueError('EOD signal must precede today and be at most three weekdays old')
    targets = validate_targets(payload.get('targets',{}),max_security_weight)
    provenance=meta.get('market_inputs') or {}
    if require_private_inputs:
        if (provenance.get('source')!='private_kis_store'
                or provenance.get('decision_date')!=current.date().isoformat()
                or provenance.get('price_date')!=signal_date.date().isoformat()
                or not re.fullmatch('[a-f0-9]{40}',provenance.get('data_commit',''))
                or any(not re.fullmatch('[a-f0-9]{64}',provenance.get(k,''))
                       for k in ('manifest_sha256','input_table_sha256','universe_membership_sha256'))):
            raise ValueError('Fresh verified private market inputs required; no legacy snapshot fallback')
    blocked=provenance.get('order_blocked_tickers',[])
    if (not isinstance(blocked,list) or any(not isinstance(t,str) or not re.fullmatch('[0-9A-Z]{6}',t) for t in blocked)
            or any(targets.get(t,0)>0 for t in blocked)):
        raise ValueError('Invalid delisted security targets/metadata')
    refs = payload.get('ref_prices') or {}
    asof = meta.get('data_as_of') or {}
    required_dates = [asof.get('primary_index'),asof.get('secondary_index')]
    for ticker,weight in targets.drop('__CASH__').items():
        if weight <= 0:
            continue
        price = refs.get(ticker)
        if price is None or not np.isfinite(float(price)) or float(price) <= 0:
            raise ValueError(f'Missing valid reference price: {ticker}')
        required_dates.append((asof.get('target_securities') or {}).get(ticker))
    for value in required_dates:
        if value is None or pd.isna(pd.Timestamp(value)) or pd.Timestamp(value).normalize() != signal_date:
            raise ValueError('Snapshot inputs must all match the signal session')
    return targets
