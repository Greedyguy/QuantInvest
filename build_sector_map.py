import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pykrx import stock
import pickle, os, time, re
from tqdm import tqdm
from bs4 import BeautifulSoup

# ---------------------------------------------
# ⚙️ 설정
# ---------------------------------------------
MARKETS = ["KOSPI", "KOSDAQ"]
MAX_WORKERS = 10
SAVE_DIR = "./data/meta"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------
# 🔍 네이버 금융에서 업종명 크롤링
# ---------------------------------------------
def get_sector_from_naver(ticker):
    """티커(6자리) 기준으로 Naver Finance에서 업종명 추출"""
    url = f"https://finance.naver.com/item/main.nhn?code={ticker}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            return None
        html = BeautifulSoup(res.text, "html.parser")

        # HTML 내에서 '업종명' 추출
        # 예: <a href="/sise/sise_group_detail.naver?type=upjong&no=xxx">반도체</a>
        link = html.select_one("div.wrap_company a[href*='sise_group_detail']")
        if link:
            sector = re.sub(r"\s+", "", link.text.strip())
            return sector
    except Exception:
        return None
    return None

# ---------------------------------------------
# 📦 전체 종목 수집
# ---------------------------------------------
def get_all_tickers():
    tickers = []
    for m in MARKETS:
        tks = stock.get_market_ticker_list(market=m)
        for t in tks:
            tickers.append((t, m))
    return tickers

# ---------------------------------------------
# 🧠 메인 로직 (병렬 수집)
# ---------------------------------------------
def build_sector_map():
    tickers = get_all_tickers()
    print(f"✅ 전체 티커 수: {len(tickers)}개")

    sector_map = {}
    fails = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(get_sector_from_naver, t): (t, m) for t, m in tickers}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Sector Fetch"):
            t, m = futs[f]
            try:
                sec = f.result()
                if sec:
                    sector_map[t] = sec
                else:
                    fails.append(t)
            except Exception:
                fails.append(t)

    print(f"✅ 성공: {len(sector_map)} | 실패: {len(fails)}")

    # 결과 저장
    with open(os.path.join(SAVE_DIR, "sector_map.pkl"), "wb") as f:
        pickle.dump(sector_map, f)

    pd.DataFrame([
        {"ticker": t, "sector": s}
        for t, s in sector_map.items()
    ]).to_csv(os.path.join(SAVE_DIR, "sector_map.csv"), index=False)

    print(f"💾 저장 완료 → {SAVE_DIR}/sector_map.pkl, sector_map.csv")
    return sector_map

# ---------------------------------------------
# 🚀 실행
# ---------------------------------------------
if __name__ == "__main__":
    sector_map = build_sector_map()