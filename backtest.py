import pandas as pd, numpy as np
from datetime import datetime, date
from pykrx import stock
from config import *
from data_loader import get_universe, load_panel, get_index_close, infer_market
from signals import compute_indicators, add_rel_strength, scoring
from positioner import select_candidates

rng = np.random.default_rng(RANDOM_SEED)

def _apply_tax(rate, gross_cash):
    return gross_cash * rate

def _tax_rate_for_day(yyyymmdd):
    d = int(yyyymmdd)
    rate = TAX_RATE_SELL
    for eff, r in sorted(TAX_SCHEDULE):
        if d >= int(eff):
            rate = r
    return rate

def run():
    start = START
    end = (date.today().strftime("%Y-%m-%d") if END is None else END)

    # universe = get_universe(MARKETS)
    universe = get_universe(MARKETS)[:10]  # ✅ 테스트용 (10종목만)
    panel = load_panel(universe, start, end)

    # 지수 수집
    idx_map = {"KOSPI": get_index_close("KOSPI", start, end),
               "KOSDAQ": get_index_close("KOSDAQ", start, end)}

    # 인디케이터/RS
    enriched = {}
    for t,df in panel.items():
        mkt = infer_market(t) or "KOSPI"
        df = compute_indicators(df)
        df = add_rel_strength(df, idx_map[mkt])
        enriched[t] = df

    # 백테스트 루프 (EOD 시그널 → 다음날 시가 체결)
    dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash, equity = 1_000_000_000.0, 0.0  # 10억 가정(원)
    positions = {}  # ticker -> dict(entry_px, qty, entry_date)
    equity_curve = []

    for i in range(60, len(dates)-1):
        d0, d1 = dates[i], dates[i+1]
        # 당일 단면 테이블
        rows = []
        for t,df in enriched.items():
            if d0 not in df.index or d1 not in df.index: 
                continue
            r = df.loc[d0]
            # 체결불가 보호: 상한가 당일 진입 금지
            price_ok = True
            if FORBID_LIMIT_UP_ENTRY and r["lc"]==1:
                price_ok = False
            rows.append({
                "ticker": t,
                "close": r["close"], "open_next": df.loc[d1,"open"],
                "val_ma20": r["val_ma20"], "v_mult": r["v_mult"],
                "bo": r["bo"], "vcp": r["vcp"], "gg": r["gg"],
                "lc": r["lc"], "rs_raw": r.get("rs_raw", np.nan),
                "price_ok": price_ok
            })
        if not rows: 
            equity_curve.append((d0, cash+_mtm(positions, enriched, d0)))
            continue
        day = pd.DataFrame(rows).set_index("ticker")

        # 점수
        day["score"] = scoring(day, W)

        # 기존 포트 평가
        port_value = cash + _mtm(positions, enriched, d0)

        # 청산 로직: 손절/익절/시간청산
        to_close = []
        for t,pos in positions.items():
            df = enriched[t]
            if d0 not in df.index: 
                continue
            px = df.loc[d0,"close"]
            ret = px/pos["entry_px"] - 1
            held = (d0 - pos["entry_date"]).days
            if (ret <= STOP_LOSS) or (ret >= TAKE_PROFIT) or (held >= MAX_HOLD_DAYS):
                to_close.append(t)
        for t in to_close:
            df = enriched[t]; px = df.loc[d1,"open"] * (1 - SLIPPAGE_EXIT)
            qty = positions[t]["qty"]
            gross = px * qty
            # 비용: 수수료+유관기관(매도편)+세금(시가 기준)
            fee = gross*(FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
            tax = gross*_tax_rate_for_day(d1.strftime("%Y%m%d"))
            cash += (gross - fee - tax)
            del positions[t]

        # 신규 진입: 상위 N
        picks = day.sort_values("score", ascending=False)
        picks = picks[picks["val_ma20"] >= MIN_AVG_TRD_AMT_20]
        picks = picks[picks["price_ok"]].head(MAX_HOLDINGS)

        # 잔여 슬롯 계산
        slots = max(0, MAX_HOLDINGS - len(positions))
        picks = picks.head(slots)

        # 균등비중(상한 30%), 가용현금 내에서 배분
        if len(picks) > 0 and slots>0:
            target_w = min(1/ max(1,len(positions)+len(picks)), MAX_WEIGHT_PER_NAME)
            alloc_cash = (cash) * target_w
            for t,r in picks.iterrows():
                px = r["open_next"] * (1 + SLIPPAGE_ENTRY)
                if px <= 0: 
                    continue
                qty = int(alloc_cash / px)
                if qty <= 0: 
                    continue
                notional = qty*px
                fee = notional*(FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                cash_post = cash - notional - fee
                if cash_post < 0:
                    continue
                cash = cash_post
                positions[t] = {"entry_px": px, "qty": qty, "entry_date": d1}

        # 기록
        equity_curve.append((d0, cash+_mtm(positions, enriched, d0)))

    ec = pd.DataFrame(equity_curve, columns=["date","equity"]).set_index("date")
    return ec, positions

def _mtm(positions, enriched, date_):
    val = 0.0
    for t,pos in positions.items():
        df = enriched.get(t)
        if df is None or date_ not in df.index: 
            continue
        val += df.loc[date_,"close"] * pos["qty"]
    return val

if __name__ == "__main__":
    ec, pos = run()
    print(ec.tail())