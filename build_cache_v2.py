"""
build_cache_v2.py
---------------------------
안정형 캐시 빌더
 - 시장별 (KOSPI/KOSDAQ)
 - tqdm 진행률
 - 재시도(3회)
 - 100개 단위 순차 다운로드
"""

import os, sys, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import date
from pykrx import stock
from data_loader import get_ohlcv_one, get_universe, DATA_DIR

# ============ 기본 설정 ============
MARKETS = ("KOSPI", "KOSDAQ")
if len(sys.argv) > 1:
    m = sys.argv[1].upper()
    if m in ("KOSPI", "KOSDAQ"):
        MARKETS = (m,)
START = "2020-01-01"
END = date.today().strftime("%Y-%m-%d")
MAX_WORKERS = 4     # 병렬 수 제한 (KRX 서버 안전치)
CHUNK_SIZE = 100    # 한 번에 처리할 종목 수
RETRY_LIMIT = 3
# ==================================

os.makedirs(DATA_DIR, exist_ok=True)

def safe_download(ticker):
    for attempt in range(RETRY_LIMIT):
        try:
            df = get_ohlcv_one(ticker, START, END)
            if not df.empty:
                return True
        except Exception as e:
            time.sleep(1)
        time.sleep(0.5)
    return False

def build_cache():
    tickers = get_universe(MARKETS)
    existing = {f.split(".")[0] for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}
    to_download = [t for t in tickers if t not in existing]

    print(f"✅ 시장: {MARKETS}, 전체 {len(tickers)}개 중 신규 {len(to_download)}개 다운로드 예정")
    if not to_download:
        print("모든 데이터가 최신 상태입니다.")
        return

    success, fail = [], []
    start_t = time.time()

    for i in range(0, len(to_download), CHUNK_SIZE):
        chunk = to_download[i:i+CHUNK_SIZE]
        print(f"\n📦 {i+1}~{i+len(chunk)} / {len(to_download)} 처리 중...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(safe_download, t): t for t in chunk}
            for f in tqdm(as_completed(futures), total=len(chunk)):
                t = futures[f]
                ok = f.result()
                if ok:
                    success.append(t)
                else:
                    fail.append(t)

        # 청크별 휴식 (KRX 서버 부하 방지)
        print(f"💤 서버 cool-down (10초)...")
        time.sleep(10)

    elapsed = time.time() - start_t
    print(f"\n🏁 완료: 성공 {len(success)}개, 실패 {len(fail)}개, 소요 {elapsed/60:.1f}분")

    pd.DataFrame({"success": success}).to_csv(f"cache_success_{MARKETS[0]}.csv", index=False)
    pd.DataFrame({"fail": fail}).to_csv(f"cache_fail_{MARKETS[0]}.csv", index=False)

if __name__ == "__main__":
    build_cache()