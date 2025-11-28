import pandas as pd, numpy as np

def select_candidates(day_table, max_holdings, max_weight):
    # 유동성 필터
    pool = day_table[(day_table["val_ma20"] >= 5e8) & (day_table["price_ok"])]
    picks = (pool.sort_values("score", ascending=False)
                 .head(max_holdings)
                 .copy())
    # 비중 동일배분 후 상한 적용
    w = min(1.0/len(picks) if len(picks)>0 else 0, max_weight)
    picks["weight"] = w
    return picks[["ticker","weight"]]