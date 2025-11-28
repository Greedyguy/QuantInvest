import numpy as np, pandas as pd

def compute_indicators(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # 보조 가드: 필수 컬럼 재확인
    need = ["open","high","low","close","volume","value"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()

    df["ret"] = df["close"].pct_change()
    df["returns"] = df["ret"]  # returns는 ret의 별칭
    df["val_ma20"] = df["value"].rolling(20, min_periods=20).mean()

    # ✅ 여기서 None 방지: Series 보장 + replace 호출 전에 타입 보정
    vma = df["val_ma20"]
    if vma is None or np.isscalar(vma):
        return pd.DataFrame()
    vma = vma.replace(0, np.nan)

    df["v_mult"] = df["value"] / vma
    
    # 이동평균선 계산
    df["ma5"] = df["close"].rolling(5, min_periods=5).mean()
    df["ma10"] = df["close"].rolling(10, min_periods=10).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=20).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=60).mean()
    
    # RSI 계산 (14일 기준)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)  # 0으로 나누기 방지
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi14"] = df["rsi"]  # rsi14는 rsi의 별칭

    df["hh_55"] = df["high"].rolling(55, min_periods=55).max()
    df["bo"] = (df["close"] >= df["hh_55"]).astype(int)

    r = df["close"].pct_change()
    df["std20"] = r.rolling(20, min_periods=20).std()
    df["std60"] = r.rolling(60, min_periods=60).std()

    tr = np.maximum(df["high"]-df["low"],
                    np.maximum((df["high"]-df["close"].shift(1)).abs(),
                               (df["low"]-df["close"].shift(1)).abs()))
    df["atr60"] = tr.rolling(60, min_periods=60).mean()
    df["vcp"] = ((df["std20"]/(df["std60"]+1e-12) <= 0.6) &
                 (tr/(df["atr60"]+1e-12) >= 1.5)).astype(int)

    df["gap"] = (df["open"] > df["high"].shift(1))
    df["go"]  = (df["close"] > df["open"])
    df["gg"] = (df["gap"] & df["go"]).astype(int)

    df["lc"] = (df["ret"] >= 0.25).astype(int)

    return df

def add_rel_strength(df, idx_close, look=20):
    """
    상대강도 계산
    
    Args:
        df: 종목 데이터프레임
        idx_close: 지수 데이터프레임 (close 컬럼 포함)
        look: 기간 (기본 20일)
    """
    # idx_close가 비어있거나 close 컬럼이 없으면 rs_raw를 0으로 설정
    if idx_close is None or idx_close.empty or "close" not in idx_close.columns:
        df["rs_raw"] = 0.0
        return df
    
    try:
        rs = (df["close"].pct_change(look)) - (idx_close["close"].pct_change(look).reindex(df.index).values)
        df["rs_raw"] = rs
    except Exception as e:
        # 오류 발생 시 rs_raw를 0으로 설정
        df["rs_raw"] = 0.0
    
    return df

def scoring(day_table, W):
    # RS는 당일 단면 퍼센타일 랭크
    rs_rank = day_table["rs_raw"].rank(pct=True)
    score = (W["LC"]*day_table["lc"] 
             + W["VS"]*(day_table["v_mult"]>=5).astype(int)
             + W["BO"]*day_table["bo"]
             + W["RS"]*rs_rank.fillna(0)
             + W["VCP"]*day_table["vcp"]
             + W["GG"]*day_table["gg"])
    return score