#!/usr/bin/env python3
"""Preserve already downloaded research inputs in explicitly partial namespaces."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from market_data_store import MarketStore,digest


def seed(store, research):
    count,rows=0,0
    plans=[('research_relative_strength_20260928','adjusted','normalized_path','normalized_sha256','raw_path','raw_sha256'),
           ('research_relative_strength_krx_20260928','raw','path','sha256','raw_path','raw_sha256'),
           ('research_relative_strength_krx_extension_20260928','raw','path','sha256','raw_path','raw_sha256')]
    for folder,basis,path_key,hash_key,raw_key,raw_hash in plans:
        manifest_path=research/'data'/folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        for record in manifest['files']:
            path=research/record[path_key]
            original=research/record[raw_key]
            if digest(path.read_bytes())!=record[hash_key] or digest(original.read_bytes())!=record[raw_hash]:
                raise ValueError('Existing research source hash mismatch')
            frame=pd.read_parquet(path).copy()
            frame.index=pd.to_datetime(frame.index)
            frame=frame.loc['2019-01-01':'2026-08-31']
            if frame.empty:
                continue
            frame.index.name='date'
            frame=frame.reset_index()
            frame['ticker']=record['ticker']
            frame['price_basis']=basis
            key=f'{folder}/{path.stem}'
            changed=store.put_table('seeds',key,frame,original.read_bytes(),dict(
                source='existing_research_download',source_manifest_sha256=digest(manifest_path.read_bytes()),
                source_table_sha256=record[hash_key],ticker=record['ticker'],price_basis=basis,
                coverage='partial_fixed_universe_not_whole_market',
                start=str(frame.date.min().date()),end=str(frame.date.max().date())))
            count+=int(changed)
            rows+=len(frame)
    return dict(new_imports=count,referenced_rows=rows,full_universe_certified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store',required=True,type=Path)
    parser.add_argument('--research-root',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(seed(MarketStore(args.store),args.research_root)))
