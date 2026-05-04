import pandas as pd, numpy as np, re
pd.set_option('future.no_silent_downcasting', True)
from datetime import datetime, date, timedelta
from pykrx import stock
import os
import io
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config import DATA_STALE_TOLERANCE_BDAYS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ohlcv")
US_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ohlcv_us")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(US_DATA_DIR, exist_ok=True)

_MCAP_MARKET_MAP_CACHE = None
_MCAP_MARKET_MAP_DATE = None

def _bizdays(start, end):
    return pd.bdate_range(start, end, freq="C")


def _business_day_gap(old: pd.Timestamp, new: pd.Timestamp) -> int:
    if old is None or pd.isna(old) or new is None or pd.isna(new):
        return DATA_STALE_TOLERANCE_BDAYS + 1
    if new <= old:
        return 0
    try:
        old_day = np.datetime64(old.date())
        new_day = np.datetime64(new.date())
        if new_day <= old_day:
            return 0
        return int(np.busday_count(old_day, new_day))
    except Exception:
        return max(int((new - old).days), 0)


def _is_cache_stale(cache_end: pd.Timestamp, req_end: pd.Timestamp) -> bool:
    if cache_end is None or pd.isna(cache_end):
        return True
    if req_end is None or pd.isna(req_end):
        return True
    if cache_end >= req_end:
        return False
    return _business_day_gap(cache_end, req_end) > DATA_STALE_TOLERANCE_BDAYS


def _download_ohlcv_block(
    ticker: str,
    start: str,
    end: str,
    include_market_cap: bool = True,
    min_rows: int = 120,
) -> pd.DataFrame:
    try:
        raw = stock.get_market_ohlcv_by_date(start, end, ticker)
    except Exception as e:
        print(f"[ERR] {ticker}: pykrx-call -> {type(e).__name__}: {e}")
        return pd.DataFrame()

    try:
        df = _normalize_df(raw, min_rows=min_rows)
    except Exception as e:
        print(f"[ERR] {ticker}: normalize -> {type(e).__name__}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if include_market_cap:
        try:
            df_cap = get_market_cap(ticker, start, end)
            if not df_cap.empty:
                df = df.join(df_cap, how="left")
        except Exception as e:
            print(f"[WARN] {ticker}: market_cap -> {type(e).__name__}: {e}")
    return df

def get_ohlcv(ticker, start, end):
    df = stock.get_market_ohlcv_by_date(start, end, ticker)

    # 컬럼 표준화: 기본 기대 컬럼
    rename_map = {"시가":"open","고가":"high","저가":"low","종가":"close",
                  "거래량":"volume","거래대금":"value"}
    # 혹시 MultiIndex/영문 혼용 방지
    df.columns = [str(c) for c in df.columns]
    df = df.rename(columns=rename_map)

    # 필수 컬럼 채우기
    for c_old, c_new in rename_map.items():
        if c_new not in df.columns and c_old in df.columns:
            df[c_new] = df[c_old]

    # 거래대금 없으면 계산식으로 대체
    if "value" not in df.columns:
        if ("close" in df.columns) and ("volume" in df.columns):
            df["value"] = df["close"].astype("float64") * df["volume"].astype("float64")
        else:
            # 정말로 만들 수 없으면 빈 프레임 반환
            return pd.DataFrame()

    # 숫자형 강제
    for c in ["open","high","low","close","volume","value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 결측/비정상 제거
    df = df.dropna(subset=["open","high","low","close","volume","value"])
    # 최소 길이 보장 (나중 compute에서 가정)
    if len(df) < 120:
        return pd.DataFrame()

    return df[["open","high","low","close","volume","value"]]

REQ_COLS = ["open","high","low","close","volume","value"]  # 필수 컬럼

def _is_bad_column(col):
    """컬럼 객체가 None/스칼라/길이0/전부 NaN 등인 경우 True"""
    if col is None:
        return True
    # 스칼라(숫자/문자)면 DF 컬럼이 아님
    if np.isscalar(col):
        return True
    try:
        # 길이 없는 이상 객체
        if len(col) == 0:
            return True
    except Exception:
        return True
    # 전부 NaN
    try:
        if pd.isna(col).all():
            return True
    except Exception:
        pass
    return False

def _normalize_df(raw, min_rows=120):
    """pykrx 반환을 표준 DF로 정규화. 실패시 빈 DF"""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    # 컬럼명 표준화
    raw.columns = [str(c) for c in raw.columns]
    rename_map = {"시가":"open","고가":"high","저가":"low","종가":"close",
                  "거래량":"volume","거래대금":"value"}
    df = raw.rename(columns=rename_map).copy()

    # 거래대금 미제공시 계산
    if "value" not in df.columns and {"close","volume"}.issubset(df.columns):
        df["value"] = df["close"] * df["volume"]

    # 필수 컬럼이 하나라도 없거나 "이상 컬럼"이면 버림
    for c in REQ_COLS:
        if c not in df.columns:
            return pd.DataFrame()
        if _is_bad_column(df[c]):
            return pd.DataFrame()

    # 숫자형 강제 변환 (여기서도 예외시 버림)
    for c in REQ_COLS:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            return pd.DataFrame()

    # 결측 제거 + 최소 길이
    df = df.dropna(subset=REQ_COLS)
    if len(df) < min_rows:
        return pd.DataFrame()

    return df[REQ_COLS]

def get_market_cap(ticker, start, end):
    """
    시가총액 데이터 수집
    
    Args:
        ticker: 종목 코드
        start: 시작일 (YYYY-MM-DD)
        end: 종료일 (YYYY-MM-DD)
    
    Returns:
        DataFrame with market_cap column
    """
    try:
        # pykrx에서 시가총액 조회
        df_cap = stock.get_market_cap_by_date(start, end, ticker)
        
        if df_cap is None or df_cap.empty:
            return pd.DataFrame()
        
        # 컬럼명 정규화
        df_cap.columns = [str(c) for c in df_cap.columns]
        rename_map = {
            "시가총액": "market_cap",
            "거래량": "volume_cap",  # 중복 방지
            "거래대금": "value_cap",
            "상장주식수": "shares",
        }
        df_cap = df_cap.rename(columns=rename_map)
        
        # 시가총액만 추출
        if "market_cap" in df_cap.columns:
            df_result = pd.DataFrame(index=df_cap.index)
            df_result["market_cap"] = pd.to_numeric(df_cap["market_cap"], errors="coerce")
            return df_result
        
        return pd.DataFrame()
    
    except Exception as e:
        # pykrx API 오류 시 조용히 무시 (시가총액 없이 진행)
        return pd.DataFrame()


def get_ohlcv_one(ticker, start, end, include_market_cap=True):
    """
    절대 None을 반환하지 않는 안전 버전
    
    Args:
        ticker: 종목 코드
        start: 시작일
        end: 종료일
        include_market_cap: 시가총액 데이터 포함 여부 (기본: True)
    """
    fp = os.path.join(DATA_DIR, f"{ticker}.parquet")

    # 1) 캐시 우선
    if os.path.exists(fp):
        try:
            df = pd.read_parquet(fp)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 캐시가 이상하면 여기서도 검사
                df = _normalize_df(df, min_rows=1)
                if not df.empty:
                    full_df = df
                    cached_end = full_df.index.max()
                    req_end = pd.to_datetime(end)
                    if _is_cache_stale(cached_end, req_end):
                        fetch_start_ts = cached_end + pd.Timedelta(days=1)
                        fetch_start = fetch_start_ts.strftime("%Y-%m-%d")
                        print(f"[INFO] {ticker}: cached OHLCV stale (last={cached_end.date()}, req={req_end.date()}) → incremental download from {fetch_start}")
                        incremental = _download_ohlcv_block(
                            ticker,
                            fetch_start,
                            end,
                            include_market_cap,
                            min_rows=1,
                        )
                        if incremental.empty:
                            # pykrx에서 신데이터를 못 받은 경우 전체 재다운로드 시도
                            full_df = pd.DataFrame()
                        else:
                            full_df = pd.concat([full_df, incremental])
                            full_df = full_df[~full_df.index.duplicated(keep='last')]
                            full_df = full_df.sort_index()
                            try:
                                full_df.to_parquet(fp, index=True)
                            except Exception as e:
                                print(f"[WARN] {ticker}: parquet-save(update) -> {type(e).__name__}: {e}")
                    else:
                        if include_market_cap and "market_cap" not in full_df.columns:
                            df_cap = get_market_cap(ticker, start, end)
                            if not df_cap.empty:
                                full_df = full_df.join(df_cap, how="left")
                                try:
                                    full_df.to_parquet(fp, index=True)
                                except Exception as e:
                                    print(f"[WARN] {ticker}: parquet-save(cap) -> {type(e).__name__}: {e}")
                    if full_df is not None and not full_df.empty:
                        req_start = pd.to_datetime(start)
                        trimmed = full_df[(full_df.index >= req_start) & (full_df.index <= req_end)]
                        if not trimmed.empty:
                            return trimmed
        except Exception:
            pass  # 손상시 재다운

    # 2) 다운로드
    df = _download_ohlcv_block(ticker, start, end, include_market_cap)

    # 5) 캐시 저장 (실패해도 무시)
    if not df.empty:
        try:
            df.to_parquet(fp, index=True)
        except Exception as e:
            print(f"[WARN] {ticker}: parquet-save -> {type(e).__name__}: {e}")

    return df if not df.empty else pd.DataFrame()

from pykrx import stock

def get_index_close(market, start, end):
    """
    KOSPI/KOSDAQ 지수 종가를 yfinance로 가져오는 버전
    market: "KOSPI" or "KOSDAQ"
    start, end: "YYYY-MM-DD" 또는 datetime-like
    """
    import yfinance as yf
    import pandas as pd

    m = str(market).upper().strip()
    if m == "KOSPI":
        symbol = "^KS11"
    elif m == "KOSDAQ":
        symbol = "^KQ11"
    elif m in ("US", "SP500", "S&P500"):
        symbol = "^GSPC"
    else:
        print(f"[WARN] get_index_close: 지원하지 않는 market={market}")
        return pd.DataFrame()

    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    try:
        # auto_adjust=False 로 명시 + progress bar 끄기
        df_raw = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[ERROR] {market}({symbol}) yfinance 다운로드 실패: {type(e).__name__}: {e}")
        return pd.DataFrame()

    if df_raw is None or df_raw.empty:
        print(f"[WARN] {market}({symbol}) yfinance: 빈 데이터")
        return pd.DataFrame()

    # 🔹 MultiIndex(필드, 티커) / 단일컬럼 모두 대응
    if isinstance(df_raw.columns, pd.MultiIndex):
        # level 0 에서 'Close' 찾기
        lvl0 = [str(c).lower() for c in df_raw.columns.get_level_values(0)]
        if "close" in lvl0:
            # level 0 에서 'Close' slice
            close_df = df_raw.xs("Close", axis=1, level=0)
        else:
            # 혹시 모르니 마지막 레벨0 사용
            first_field = df_raw.columns.get_level_values(0)[0]
            close_df = df_raw.xs(first_field, axis=1, level=0)

        # 티커가 1개일테니 첫 번째 컬럼만 사용
        close_series = close_df.iloc[:, 0]
    else:
        # 일반 DataFrame: Close 컬럼 사용
        if "Close" in df_raw.columns:
            close_series = df_raw["Close"]
        elif "close" in [c.lower() for c in df_raw.columns]:
            # 혹시 소문자 등
            c = [c for c in df_raw.columns if c.lower() == "close"][0]
            close_series = df_raw[c]
        else:
            print(f"[WARN] {market}({symbol}) Close 컬럼 없음: {df_raw.columns}")
            return pd.DataFrame()

    close = pd.to_numeric(close_series, errors="coerce")

    df = pd.DataFrame({"close": close})
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df

def infer_market(ticker):
    # pykrx가 티커로 시장 구분 제공X → 시총표에서 유추
    # 종목마다 반복 호출하지 않고 당일 1회만 받아 캐시 재사용
    global _MCAP_MARKET_MAP_CACHE, _MCAP_MARKET_MAP_DATE
    try:
        today_str = date.today().strftime("%Y%m%d")
        if _MCAP_MARKET_MAP_CACHE is None or _MCAP_MARKET_MAP_DATE != today_str:
            # pykrx 내부에서 stderr/stdout으로 JSON 파싱 오류를 출력하는 경우가 있어 억제
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    mcap = stock.get_market_cap_by_ticker(today_str)
            except Exception:
                mcap = pd.DataFrame()
            if mcap is None or mcap.empty or "시장구분" not in mcap.columns:
                _MCAP_MARKET_MAP_CACHE = {}
            else:
                _MCAP_MARKET_MAP_CACHE = mcap["시장구분"].astype(str).to_dict()
            _MCAP_MARKET_MAP_DATE = today_str

        market = _MCAP_MARKET_MAP_CACHE.get(ticker)
        if market == "KOSDAQ":
            return "KOSDAQ"
        if market == "KOSPI":
            return "KOSPI"
        return None
    except Exception:
        return None


def infer_market_us(ticker):
    return "US"


_DEFAULT_US_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V",
    "UNH","XOM","LLY","AVGO","COST","WMT","PG","HD","MA","JNJ",
    "NFLX","ORCL","ABBV","BAC","KO","MRK","AMD","PEP","CVX","ADBE",
    "CRM","TMO","LIN","MCD","QCOM","ACN","CSCO","GE","ABT","INTU",
    "DHR","WFC","AMAT","TXN","PFE","PM","IBM","CAT","NOW","GS"
]
_DEFAULT_US_ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "TLT", "SH", "PSQ", "SQQQ"]


def get_universe_us(limit=200):
    """
    US Universe 조회. 네트워크/소스 실패 시 대형주 기본 리스트 fallback.
    """
    try:
        import yfinance as yf
        # 대표 ETF 보유종목 페이지에서 상위 티커 추출 시도 (환경 제한 시 fallback)
        # yfinance는 보유종목 추출 API가 제한적이라 안정적으로는 fallback 사용
        _ = yf.Ticker("SPY").history(period="5d")
    except Exception:
        pass
    merged = _DEFAULT_US_UNIVERSE + [t for t in _DEFAULT_US_ETF_UNIVERSE if t not in _DEFAULT_US_UNIVERSE]
    return merged[:max(1, int(limit))]

def get_universe(markets=("KOSPI","KOSDAQ"), include_etf=True, include_index_etf=True):
    """
    Universe 종목 리스트 조회 (캐시 Fallback 버전)
    
    Args:
        markets: ("KOSPI", "KOSDAQ") 등
        include_etf: ETF 포함 여부
        include_index_etf: 주요 인덱스 ETF 강제 포함 (KODEX 200 등)
    
    Returns:
        종목 코드 리스트
    """
    import os
    import glob
    from datetime import datetime, timedelta
    
    tickers = []
    
    # 📌 최근 거래일 찾기 (주말/공휴일 대응)
    today = datetime.now()
    last_error = None
    attempted_dates = []
    
    for days_ago in range(14):  # 최대 14일 전까지 시도
        try_date = (today - timedelta(days=days_ago)).strftime("%Y%m%d")
        attempted_dates.append(try_date)
        
        try:
            temp_tickers = []
            for m in markets:
                # 명시적 날짜 전달
                ticker_list = stock.get_market_ticker_list(date=try_date, market=m)
                if ticker_list is None:
                    raise ValueError(f"pykrx가 None을 반환했습니다 (market={m}, date={try_date})")
                temp_tickers += ticker_list
            
            # 성공하면 저장하고 중단
            if len(temp_tickers) > 0:
                tickers = temp_tickers
                print(f"[INFO] Universe (pykrx): {len(tickers)}개 종목 (날짜: {try_date[:4]}-{try_date[4:6]}-{try_date[6:]})")
                break
            else:
                # 빈 리스트 반환 시에도 에러로 기록
                if last_error is None:
                    last_error = ValueError(f"모든 시도 날짜에서 빈 리스트 반환 (시도한 날짜: {', '.join(attempted_dates[:3])}...)")
                
        except Exception as e:
            last_error = e
            continue  # 다음 날짜 시도
    
    # pykrx 실패 시 캐시에서 종목 리스트 추출
    if len(tickers) == 0:
        error_msg = str(last_error) if last_error is not None else "예외 없이 빈 리스트 반환 (14일 모두 시도)"
        print(f"[WARN] pykrx Universe 조회 실패: {error_msg}")
        print(f"       → 캐시된 enriched 파일에서 종목 리스트 추출 시도...")
        
        enriched_dir = "data/enriched"
        if os.path.exists(enriched_dir):
            cache_files = glob.glob(f"{enriched_dir}/*.parquet")
            if len(cache_files) > 0:
                cached_tickers = sorted(set([
                    os.path.basename(f).split("_")[0] 
                    for f in cache_files
                ]))
                if len(cached_tickers) > 0:
                    tickers = cached_tickers
                    print(f"[INFO] Universe (캐시): {len(tickers)}개 종목 추출 완료")
    
    if len(tickers) == 0:
        print("[ERROR] Universe가 완전히 비어있습니다 (pykrx 및 캐시 모두 실패)")
        return []
    
    # 📌 주요 인덱스 ETF 강제 추가
    index_etfs = [
        "069500",  # KODEX 200
        "102110",  # TIGER 200
        "114800",  # KODEX 인버스
        "122630",  # KODEX 레버리지
        "229200",  # KODEX 코스닥150
        "091160",  # KODEX 반도체
        "091180",  # TIGER 은행
        "152100",  # ARIRANG 200
    ]
    
    if include_index_etf:
        for etf_code in index_etfs:
            if etf_code not in tickers:
                tickers.append(etf_code)
        print(f"[INFO] 주요 인덱스 ETF {len(index_etfs)}개 추가됨")
    
    # 종목명 조회 및 필터링
    names = {}
    for t in tickers:
        try:
            name_result = stock.get_market_ticker_name(t)
            # DataFrame이나 Series인 경우 문자열로 변환
            if isinstance(name_result, (pd.DataFrame, pd.Series)):
                names[t] = str(name_result.iloc[0] if hasattr(name_result, 'iloc') else name_result)
            else:
                names[t] = str(name_result) if name_result is not None else ""
        except:
            names[t] = ""
    
    def ok(t):
        n = names.get(t, "")
        # 문자열이 아닌 경우 문자열로 변환
        if not isinstance(n, str):
            n = str(n)
        
        # 주요 인덱스 ETF는 항상 허용
        if include_index_etf and t in index_etfs:
            return True
        
        # ETF 포함 여부에 따라 필터링
        if include_etf:
            # ETF는 허용하되, 리츠/스팩은 제외
            bad_kw = ["우", "리츠", "스팩", "SPAC"]
        else:
            # ETF도 제외
            bad_kw = ["우", "리츠", "스팩", "SPAC", "ETF", "ETN"]
        
        return not any(k in n.upper() or k in n for k in bad_kw)
    
    result = [t for t in tickers if ok(t)]

    # 통계 출력
    etf_count = sum(1 for t in result if t in index_etfs or "ETF" in names.get(t, "").upper() or "KODEX" in names.get(t, "") or "TIGER" in names.get(t, ""))
    print(f"[INFO] Universe: {len(result)} 종목 ({len(tickers)} → 필터링, ETF: {etf_count}개)")
    
    return result


def get_ohlcv_one_us(ticker, start, end, min_rows=120):
    """
    US 종목 OHLCV 로더(yfinance). 캐시 우선 + 증분 업데이트.
    """
    import yfinance as yf

    safe_ticker = ticker.replace("/", "_")
    fp = os.path.join(US_DATA_DIR, f"{safe_ticker}.parquet")

    if os.path.exists(fp):
        try:
            cached = pd.read_parquet(fp)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                req_start = pd.to_datetime(start)
                req_end = pd.to_datetime(end)
                cached = cached.sort_index()
                cache_end = cached.index.max()
                if not _is_cache_stale(cache_end, req_end):
                    trimmed = cached[(cached.index >= req_start) & (cached.index <= req_end)]
                    if len(trimmed) >= min_rows:
                        return trimmed
        except Exception:
            pass

    try:
        raw = yf.download(
            ticker,
            start=pd.to_datetime(start),
            end=pd.to_datetime(end) + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[ERR] {ticker}: yfinance-call -> {type(e).__name__}: {e}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    col_map = {"adj close": "adj_close"}
    raw = raw.rename(columns=col_map)

    need = {"open", "high", "low", "close", "volume"}
    if not need.issubset(set(raw.columns)):
        return pd.DataFrame()

    def _to_series(frame: pd.DataFrame, col: str) -> pd.Series:
        obj = frame[col]
        if isinstance(obj, pd.DataFrame):
            # 중복 컬럼이 생기면 첫 번째 컬럼 사용
            obj = obj.iloc[:, 0]
        return pd.to_numeric(obj, errors="coerce")

    df = pd.DataFrame(index=raw.index)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = _to_series(raw, c)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df["value"] = df["close"] * df["volume"]
    if len(df) < min_rows:
        return pd.DataFrame()
    df = df.sort_index()
    try:
        df.to_parquet(fp, index=True)
    except Exception as e:
        print(f"[WARN] {ticker}: parquet-save(us) -> {type(e).__name__}: {e}")
    return df


def load_panel(universe, start, end, max_workers=6, include_market_cap=True):
    """
    전체 종목 패널 데이터 로드
    
    Args:
        universe: 종목 코드 리스트
        start: 시작일
        end: 종료일
        max_workers: 병렬 처리 워커 수
        include_market_cap: 시가총액 포함 여부 (기본: True)
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    datas = {}
    print(f"[INFO] Loading {len(universe)} tickers (parallel={max_workers}, market_cap={include_market_cap})")
    t0 = time.time()

    def _safe_one(t):
        df = get_ohlcv_one(t, start, end, include_market_cap=include_market_cap)
        # DataFrame이 아니거나 비어있으면 제외
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        # 필수 컬럼 최종 검증
        if any(col not in df.columns for col in REQ_COLS):
            return None, None
        return t, df

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_safe_one, t): t for t in universe}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                k, df = f.result()
                if k is not None:
                    df["ticker"] = k
                    datas[k] = df
            except Exception as e:
                print(f"[WARN] {t}: load -> {type(e).__name__}: {e}")

            if i % 50 == 0 or i == len(futs):
                print(f"  Progress: {i}/{len(futs)} ({i/len(futs)*100:.1f}%)")

    # 시가총액 수집 통계
    if include_market_cap and len(datas) > 0:
        cap_count = sum(1 for df in datas.values() if "market_cap" in df.columns)
        print(f"[INFO] Market cap collected: {cap_count}/{len(datas)} tickers ({cap_count/len(datas)*100:.1f}%)")
    elif include_market_cap:
        print(f"[WARN] Market cap requested but no data loaded")

    print(f"[INFO] Loaded {len(datas)} tickers in {time.time()-t0:.1f}s")
    return datas


def load_panel_us(universe, start, end, max_workers=6):
    datas = {}
    print(f"[INFO] Loading US {len(universe)} tickers (parallel={max_workers})")
    t0 = time.time()

    def _safe_one(t):
        df = get_ohlcv_one_us(t, start, end, min_rows=120)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        if any(col not in df.columns for col in REQ_COLS):
            return None, None
        return t, df

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_safe_one, t): t for t in universe}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                k, df = f.result()
                if k is not None:
                    df["ticker"] = k
                    datas[k] = df
            except Exception as e:
                print(f"[WARN] {t}: load(us) -> {type(e).__name__}: {e}")
            if i % 50 == 0 or i == len(futs):
                print(f"  Progress: {i}/{len(futs)} ({i/len(futs)*100:.1f}%)")

    print(f"[INFO] Loaded US {len(datas)} tickers in {time.time()-t0:.1f}s")
    return datas
