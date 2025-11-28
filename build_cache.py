"""
build_cache.py
--------------------------------
전체 코스피+코스닥 주식의 일별 OHLCV 데이터를
/data/ohlcv/*.parquet 형태로 병렬 다운로드 & 캐시 구축

처음 실행 시 2~3분, 이후 10초 내 완료.
"""

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pykrx import stock
from datetime import date
from data_loader import get_ohlcv_one, get_universe, DATA_DIR
import sys

# 다운로드 기간: 최근 5년
START = "2020-01-01"
END = date.today().strftime("%Y-%m-%d")
MAX_WORKERS = 16  # 시스템에 맞게 조정 (맥북 M1은 8~12 정도)

os.makedirs(DATA_DIR, exist_ok=True)
MARKETS = ("KOSPI","KOSDAQ")
if len(sys.argv) > 1:
    m = sys.argv[1].upper()
    if m in ("KOSPI","KOSDAQ"):
        MARKETS = (m,)

def build_all_cache():
    tickers = get_universe(MARKETS)
    existing = {f.split(".")[0] for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}
    to_download = [t for t in tickers if t not in existing]

    print(f"✅ 전체 유니버스: {len(tickers)}개 종목")
    print(f"📦 이미 캐시된 종목: {len(existing)}개")
    print(f"⬇️ 새로 다운로드할 종목: {len(to_download)}개")
    if not to_download:
        print("모든 데이터가 최신 상태입니다!")
        return

    start_t = time.time()
    success, fail = [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_ohlcv_one, t, START, END): t for t in to_download}
        for i, f in enumerate(as_completed(futures), 1):
            t = futures[f]
            df = f.result()
            if not df.empty:
                success.append(t)
            else:
                fail.append(t)

            if i % 100 == 0 or i == len(futures):
                pct = 100 * i / len(futures)
                print(f"  진행률: {i}/{len(futures)} ({pct:.1f}%)")

    elapsed = time.time() - start_t
    print(f"🏁 완료: 성공 {len(success)}개, 실패 {len(fail)}개, 소요 {elapsed/60:.1f}분")

    summary = pd.DataFrame({
        "success": [len(success)],
        "fail": [len(fail)],
        "elapsed_min": [elapsed / 60]
    })
    pd.DataFrame({"success_tickers": success}).to_csv("cache_success.csv", index=False)
    pd.DataFrame({"fail_tickers": fail}).to_csv("cache_fail.csv", index=False)
    summary.to_csv("cache_summary.csv", index=False)

if __name__ == "__main__":
    build_all_cache()