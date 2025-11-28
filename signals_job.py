# signals_job.py (노트북 맨 위 또는 별도 파일)
import pandas as pd
from signals import compute_indicators, add_rel_strength

def build_signals_for_one(ticker, df, idx_map, infer_market_fn):
    mkt = infer_market_fn(ticker) or "KOSPI"
    idx_close = idx_map[mkt]
    df = compute_indicators(df)
    if df.empty: 
        return ticker, pd.DataFrame()
    df = add_rel_strength(df, idx_close)
    return ticker, df