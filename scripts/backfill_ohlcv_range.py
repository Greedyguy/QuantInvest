#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지정한 날짜 구간의 OHLCV 데이터를 강제로 캐시에 채워 넣는 보조 스크립트.

사용 예시:
    python scripts/backfill_ohlcv_range.py --start 2025-11-01 --end 2025-11-27 --workers 12
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import data_loader  # pylint: disable=wrong-import-position


def _merge_and_save(ticker: str, new_df: pd.DataFrame, data_dir: str) -> Tuple[bool, str]:
    if new_df is None or new_df.empty:
        return False, "신규 데이터 없음"
    fp = os.path.join(data_dir, f"{ticker}.parquet")
    frames = [new_df]
    if os.path.exists(fp):
        try:
            old_df = pd.read_parquet(fp)
            if isinstance(old_df, pd.DataFrame) and not old_df.empty:
                frames.append(old_df)
        except Exception as exc:  # pragma: no cover
            return False, f"기존 캐시 로드 실패: {exc}"
    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    try:
        merged.to_parquet(fp, index=True)
    except Exception as exc:
        return False, f"저장 실패: {exc}"
    return True, f"{len(new_df)}건 갱신"


def _fetch_one(
    ticker: str,
    start: str,
    end: str,
    include_market_cap: bool,
    delay: float,
) -> Tuple[str, bool, str]:
    try:
        if delay > 0:
            time.sleep(delay)
        df = data_loader._download_ohlcv_block(  # pylint: disable=protected-access
            ticker,
            start,
            end,
            include_market_cap=include_market_cap,
            min_rows=1,
        )
        ok, message = _merge_and_save(ticker, df, data_loader.DATA_DIR)
        return ticker, ok, message
    except Exception as exc:  # pragma: no cover - 네트워크/IO 오류
        return ticker, False, f"예외 발생: {exc}"


def main():
    parser = argparse.ArgumentParser(description="지정 구간 OHLCV 강제 백필")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=8, help="병렬 다운로드 워커 수")
    parser.add_argument("--no-etf", dest="include_etf", action="store_false", help="ETF 제외")
    parser.add_argument("--no-index-etf", dest="include_index_etf", action="store_false", help="주요 인덱스 ETF 제외")
    parser.add_argument("--market-cap", action="store_true", help="시가총액도 함께 저장")
    parser.add_argument("--tickers", type=str, default="", help="쉼표로 구분된 티커 리스트 (전체 대신 사용)")
    parser.add_argument("--tickers-file", type=str, default="", help="티커 리스트가 담긴 텍스트 파일 경로")
    parser.add_argument("--offset", type=int, default=0, help="유니버스 오프셋 (기본 0)")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 티커 수 (0이면 전체)")
    parser.add_argument("--sleep", type=float, default=0.0, help="각 다운로드 사이 지연(초)")
    parser.set_defaults(include_etf=True, include_index_etf=True)
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("종료일은 시작일보다 늦어야 합니다.")

    custom_tickers: Optional[List[str]] = None
    if args.tickers:
        custom_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        path = Path(args.tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"티커 파일을 찾을 수 없습니다: {path}")
        with open(path, "r", encoding="utf-8") as f:
            custom_tickers = [line.strip() for line in f if line.strip()]

    if custom_tickers:
        target_universe = custom_tickers
    else:
        universe = data_loader.get_universe(include_etf=args.include_etf, include_index_etf=args.include_index_etf)
        if not universe:
            raise RuntimeError("유효한 유니버스를 가져오지 못했습니다.")
        target_universe = universe

    if args.offset or args.limit:
        start_idx = max(args.offset, 0)
        end_idx = len(target_universe) if args.limit <= 0 else start_idx + args.limit
        target_universe = target_universe[start_idx:end_idx]

    if not target_universe:
        raise RuntimeError("처리할 티커가 없습니다.")

    print(
        f"총 {len(target_universe)}개 종목에 대해 "
        f"{args.start}~{args.end} 구간을 백필합니다. "
        f"(workers={args.workers}, sleep={args.sleep}s)"
    )

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _fetch_one,
                ticker,
                args.start,
                args.end,
                args.market_cap,
                args.sleep,
            ): ticker
            for ticker in target_universe
        }
        for idx, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            ok, message = False, ""
            try:
                ticker, ok, message = future.result()
            except Exception as exc:  # pragma: no cover
                ok = False
                message = f"예외 발생: {exc}"
            if ok:
                success += 1
            else:
                failed += 1
            if idx % 50 == 0 or idx == len(futures):
                print(f"[{idx}/{len(futures)}] {ticker}: {'OK' if ok else 'FAIL'} ({message})")

    print(f"완료: 성공 {success}개 / 실패 {failed}개")


if __name__ == "__main__":
    main()
