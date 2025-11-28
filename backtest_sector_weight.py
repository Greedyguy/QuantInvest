import pickle
import pandas as pd
import numpy as np
from signals import scoring
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# 0️⃣ 섹터 매핑 로드
# ------------------------------------------------------------
with open("./data/meta/sector_map.pkl", "rb") as f:
    sector_map = pickle.load(f)

# 없는 종목은 "기타"로 분류
def get_sector(ticker):
    return sector_map.get(ticker, "기타")

# ------------------------------------------------------------
# 1️⃣ 섹터 모멘텀 계산 함수
# ------------------------------------------------------------
def compute_sector_momentum(enriched, window=20):
    """
    각 섹터의 평균 수익률(최근 20일)을 모멘텀 점수로 계산
    반환: {sector: 평균수익률}
    """
    sec_rets = {}
    sec_counts = {}

    for t, df in enriched.items():
        if df.empty or len(df) < window:
            continue
        sec = get_sector(t)
        ret = df["close"].pct_change(window).iloc[-1]
        sec_rets[sec] = sec_rets.get(sec, 0) + ret
        sec_counts[sec] = sec_counts.get(sec, 0) + 1

    sector_momentum = {}
    for sec, total_ret in sec_rets.items():
        sector_momentum[sec] = total_ret / sec_counts[sec]
    return sector_momentum

# ------------------------------------------------------------
# 2️⃣ 섹터 기반 가중치 함수
# ------------------------------------------------------------
def sector_weight(sector_momentum, sector):
    """섹터 모멘텀 기반 비중 조정"""
    mom = sector_momentum.get(sector, 0)
    if mom > 0.05:
        return 1.2  # 최근 강세 섹터
    elif mom < -0.03:
        return 0.8  # 약세 섹터
    else:
        return 1.0  # 중립

# ------------------------------------------------------------
# 3️⃣ 백테스트 루프 (섹터 가중 포함)
# ------------------------------------------------------------
def backtest_with_sector_weight(enriched, W, START_CASH=1_000_000_000):
    cash = START_CASH
    positions = {}
    equity_curve = []
    trade_log = []

    dates = sorted(set().union(*[df.index for df in enriched.values()]))

    for i in tqdm(range(60, len(dates)-1), desc="SectorBacktest"):
        d0, d1 = dates[i], dates[i+1]

        # 매일 섹터 모멘텀 갱신
        if i % 20 == 0:
            sector_mom = compute_sector_momentum(enriched, window=20)

        rows = []
        for t, df in enriched.items():
            if d0 not in df.index or d1 not in df.index:
                continue
            r = df.loc[d0]
            rows.append({
                "ticker": t,
                "close": r["close"],
                "open_next": df.loc[d1, "open"],
                "val_ma20": r["val_ma20"],
                "v_mult": r["v_mult"],
                "bo": r["bo"],
                "vcp": r["vcp"],
                "gg": r["gg"],
                "rs_raw": r.get("rs_raw", np.nan),
                "sector": get_sector(t)
            })

        if not rows:
            equity_curve.append((d0, cash))
            continue

        day = pd.DataFrame(rows).set_index("ticker")
        day["score"] = scoring(day, W)

        # 📉 청산
        to_close = []
        for t, pos in positions.items():
            df = enriched[t]
            if d0 not in df.index:
                continue
            px = df.loc[d0, "close"]
            ret = px / pos["entry_px"] - 1
            held = (d0 - pos["entry_date"]).days
            if (ret <= STOP_LOSS) or (ret >= TAKE_PROFIT) or (held >= MAX_HOLD_DAYS):
                to_close.append(t)

        for t in to_close:
            df = enriched[t]
            px = df.loc[d1, "open"] * (1 - SLIPPAGE_EXIT)
            qty = positions[t]["qty"]
            gross = px * qty
            fee = gross * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
            tax = gross * TAX_RATE_SELL
            cash += (gross - fee - tax)
            trade_log.append({
                "date": d1, "ticker": t,
                "exit_px": px,
                "ret": px / positions[t]["entry_px"] - 1
            })
            del positions[t]

        # 🚀 신규 진입
        picks = (
            day[day["val_ma20"] >= MIN_AVG_TRD_AMT_20]
            .sort_values("score", ascending=False)
            .head(MAX_HOLDINGS * 2)  # 후보 2배 확보
        )

        # ✅ 동일 섹터 최대 2개만 보유
        sector_counts = {}
        new_positions = []

        for t, r in picks.iterrows():
            sec = r["sector"]
            sector_counts[sec] = sector_counts.get(sec, 0)
            if sector_counts[sec] >= 2:
                continue
            sector_counts[sec] += 1
            new_positions.append((t, r))

        slots = max(0, MAX_HOLDINGS - len(positions))
        picks = new_positions[:slots]

        if len(picks) > 0 and slots > 0:
            for t, r in picks:
                sec = r["sector"]
                sw = sector_weight(sector_mom, sec)
                target_w = min(MAX_WEIGHT_PER_NAME * sw, 0.5)
                alloc_cash = cash * target_w
                px = r["open_next"] * (1 + SLIPPAGE_ENTRY)
                qty = int(alloc_cash / px)
                if qty <= 0:
                    continue
                notional = qty * px
                fee = notional * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                cash -= (notional + fee)
                positions[t] = {"entry_px": px, "qty": qty, "entry_date": d1}

        # 💰 Equity 기록
        equity = cash
        for t, pos in positions.items():
            if d0 in enriched[t].index:
                equity += enriched[t].loc[d0, "close"] * pos["qty"]
        equity_curve.append((d0, equity))

    ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return ec, trade_log