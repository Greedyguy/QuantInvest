#!/usr/bin/env python3
"""Bounded Actions entry point, read-only KIS APIs; no trading imports."""
from datetime import datetime,timedelta
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from market_data_store import MarketStore,atomic_bytes,canonical,day
from kis_market_collection import (CollectionInterrupted,KISPriceClient,collect_masters,
                                  collect_prices,read_master,select_kis)


def main():
    store=MarketStore(os.environ['MARKET_STORE_PATH'])
    mode=os.environ.get('COLLECTION_MODE','incremental')
    today=datetime.now(ZoneInfo('Asia/Seoul')).date()
    observed=today.isoformat()
    if mode=='select':
        as_of=day(os.environ['SELECTION_DATE'])
        result=select_kis(store,as_of,as_of)
        atomic_bytes(store.root.parent/'signals'/f'{as_of}.json',canonical(result))
        return
    if mode not in ('master','backfill','incremental'):
        raise ValueError('Invalid collection mode')
    pinned_master=os.environ.get('MASTER_DATE')
    master_date=pinned_master or observed
    if not pinned_master:
        master=collect_masters(store,observed_date=observed)
    else:
        master=read_master(store,master_date)
    if mode=='master':
        print(f'Master rows: {len(master)}; historical universe not certified')
        return
    if mode=='backfill':
        start=day(os.environ.get('BACKFILL_START') or '2025-01-01')
        end=day(os.environ.get('BACKFILL_END') or '2026-08-31')
    else:
        end=(today-timedelta(days=1)).isoformat()
        start=(today-timedelta(days=8)).isoformat()
    if end>=observed:
        raise ValueError('Only completed prior calendar days may be collected')
    client=KISPriceClient(os.environ.get('KIS_APP_KEY'),os.environ.get('KIS_APP_SECRET'))
    failure=None
    try:
        result=collect_prices(store,client,master,master_date,start,end,
            max_requests=int(os.environ.get('MAX_REQUESTS') or '300'),refresh=mode=='incremental')
    except CollectionInterrupted as error:
        result=error.progress
        failure=error
    result.update(start=start,end=end,master_date=master_date,mode=mode,orders_enabled=False)
    result['updated_at']=datetime.now(ZoneInfo('UTC')).isoformat()
    atomic_bytes(store.root/'last_kis_collection.json',canonical(result))
    print({k:v for k,v in result.items() if k!='candidates'})
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        status=('오류로 중단 — 검증·비공개 저장 단계의 성공 여부를 확인하세요' if failure else
                '요청한 현재 종목 목록의 조회 완료' if result['requested_universe_queried']
                else '중간 저장 — 같은 설정으로 Run workflow를 다시 실행하세요')
        with open(summary,'a',encoding='utf-8') as handle:
            handle.write(f'## 과거 데이터 수집\n\n- 기간: {start} ~ {end}\n'
                         f'- 상태: {status}\n- 이번 신규 조회: {result["requested"]}회\n'
                         f'- 기존 자료 재사용: {result["reused"]}개\n'
                         '- 아래 비공개 저장 단계까지 성공해야 이번 결과가 보관됩니다.\n'
                         '- 상장폐지 종목 전체 복원 및 빈 응답 검증 완료를 뜻하지 않습니다.\n')
    if failure:
        raise failure


if __name__=='__main__':
    main()
