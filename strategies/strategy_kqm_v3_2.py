"""
K-Quality Momentum v3.2 전략

Optuna 최적화 결과를 반영한 버전:
- 최적 팩터 가중치 적용 (MOM6: 30.54%, MOM3: 6.04%, QUALITY: 9.55%, VOL: 30.73%, VAL: 23.14%)
- 최적 리밸런싱 주기: 17일
- Train Sharpe: 2.97, Valid Sharpe: 0.74
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMStrategyV3_2(BaseStrategy):
    """
    K-Quality Momentum v3.2 전략 (Optuna 최적화 버전)
    
    개선사항:
    1. Optuna로 최적화된 팩터 가중치 적용
    2. 최적 리밸런싱 주기 17일 적용
    3. 기존 v3의 모든 기능 유지
    """
    
    def __init__(self):
        """
        Optuna 최적화 결과 적용:
        - Best Score: 1.9879 (Sharpe + 0.5*MDD)
        - Train: Sharpe 2.97, CAGR 45.28%, MDD -5.64%
        - Valid: Sharpe 0.74, CAGR 17.15%, MDD -36.23%
        """
        # Optuna 최적화 결과
        self.rebal_days = 17  # 최적 리밸런싱 주기
        self.n_stocks = 30
        self.sector_limit = 5
        
        # 최적 팩터 가중치 (정규화된 값)
        self.w_mom6 = 0.3054
        self.w_mom3 = 0.0604
        self.w_quality = 0.0955
        self.w_vol = 0.3073
        self.w_val = 0.2314
        
        # 거래 비용
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = 0.001
    
    def get_name(self) -> str:
        return "kqm_v3_2"
    
    def get_description(self) -> str:
        return "K-Quality Momentum v3.2 (Optuna Optimized: 17d rebal, MOM6:30.5% VOL:30.7% VAL:23.1%)"
    
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
        """팩터 계산"""
        if df is None or len(df) < 120:
            return None
        
        subset = df[df.index <= current_date]
        if len(subset) < 120:
            return None
        
        close = subset["close"].values
        
        # 1) Momentum 6개월
        if len(close) < 120:
            return None
        mom_6m = (close[-1] / close[-120]) - 1.0
        
        # 2) Momentum 3개월
        if len(close) < 60:
            return None
        mom_3m = (close[-1] / close[-60]) - 1.0
        
        # 3) Quality Proxy (가격 안정성)
        if len(close) < 60:
            return None
        ret_60 = pd.Series(close[-60:]).pct_change().dropna()
        quality_proxy = ret_60.mean() / (ret_60.std() + 1e-9) if len(ret_60) > 0 else 0.0
        
        # 4) Inverse Volatility (Smoothed)
        vol_20 = pd.Series(close[-20:]).pct_change().ewm(halflife=10).std().iloc[-1]
        inv_vol_smooth = 1.0 / (vol_20 + 1e-9)
        
        # 5) Value Proxy (PER/PBR 역수 대용: 최근 수익률 평균)
        ret_120 = pd.Series(close[-120:]).pct_change().dropna()
        value_proxy = ret_120.mean() if len(ret_120) > 0 else 0.0
        
        return {
            "mom6m": mom_6m,
            "mom3m": mom_3m,
            "quality": quality_proxy,
            "inv_vol_smooth": inv_vol_smooth,
            "val_proxy": value_proxy,
        }
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """백테스트 실행"""
        if not silent:
            print("\n" + "="*60)
            print("📈 K-Quality Momentum v3.2 백테스트 시작...")
            print("="*60)
            print(f"⚙️  리밸런싱 주기: {self.rebal_days}일 (Optuna 최적화)")
            print(f"⚙️  보유 종목: {self.n_stocks}개")
            print(f"⚙️  섹터 제한: {self.sector_limit}개/섹터")
            print(f"⚙️  팩터 가중치:")
            print(f"     MOM6: {self.w_mom6:.1%}, MOM3: {self.w_mom3:.1%}")
            print(f"     QUALITY: {self.w_quality:.1%}, VOL: {self.w_vol:.1%}, VAL: {self.w_val:.1%}")
        
        # 날짜 리스트
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        
        if len(dates) < 120:
            return pd.DataFrame(), []
        
        # 리밸런싱 날짜 (120일 이후부터 시작, 이후 17일마다)
        rebalance_dates = dates[120::self.rebal_days]
        
        # 초기 설정
        init_cash = 1_000_000_000
        cash = init_cash
        positions = {}
        equity_curve = []
        trade_log = []
        
        # 리밸런싱 루프
        for rebal_idx in tqdm(range(len(rebalance_dates)), desc="KQM v3.2", disable=silent):
            d0 = rebalance_dates[rebal_idx]
            
            # 다음 리밸런싱 날짜 (마지막이면 종료일)
            if rebal_idx < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[rebal_idx + 1]
            else:
                next_rebal_date = dates[-1]
            
            # === 1) 팩터 스냅샷 ===
            rows = []
            for ticker, df in enriched.items():
                if df is None or len(df) == 0:
                    continue
                
                factors = self._compute_factors(df, d0)
                if factors is None:
                    continue
                
                # 음수 모멘텀 제외
                if factors["mom6m"] <= 0 or factors["mom3m"] <= 0:
                    continue
                
                # 섹터 정보
                sector = df.attrs.get("sector", "Unknown")
                
                rows.append({
                    "ticker": ticker,
                    "mom6m": factors["mom6m"],
                    "mom3m": factors["mom3m"],
                    "quality": factors["quality"],
                    "inv_vol_smooth": factors["inv_vol_smooth"],
                    "val_proxy": factors["val_proxy"],
                    "sector": sector,
                })
            
            if len(rows) == 0:
                # 데이터 없으면 현금 유지, equity 기록
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            day = pd.DataFrame(rows)
            
            # === 2) 팩터 랭킹 (0~1 정규화) ===
            for col in ["mom6m", "mom3m", "quality", "inv_vol_smooth", "val_proxy"]:
                day[f"{col}_rank"] = day[col].rank(pct=True)
            
            # === 3) 복합 팩터 스코어 (Optuna 최적화 가중치) ===
            day["score"] = (
                self.w_mom6 * day["mom6m_rank"] +
                self.w_mom3 * day["mom3m_rank"] +
                self.w_quality * day["quality_rank"] +
                self.w_vol * day["inv_vol_smooth_rank"] +
                self.w_val * day["val_proxy_rank"]
            )
            
            # === 4) 상위 종목 선택 (섹터 제한) ===
            day_sorted = day.sort_values("score", ascending=False)
            selected = []
            sector_count = {}
            
            for _, row in day_sorted.iterrows():
                ticker = row["ticker"]
                sector = row["sector"]
                
                if len(selected) >= self.n_stocks:
                    break
                
                # 섹터 제한 확인
                if sector_count.get(sector, 0) >= self.sector_limit:
                    continue
                
                selected.append(ticker)
                sector_count[sector] = sector_count.get(sector, 0) + 1
            
            if len(selected) == 0:
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            # === 5) 팩터 기반 가중치 (Equal Risk Contribution) ===
            top_n_stocks = day[day["ticker"].isin(selected)].copy()
            top_n_stocks["w"] = top_n_stocks["inv_vol_smooth"] / top_n_stocks["inv_vol_smooth"].sum()
            
            # === 6) 포지션 청산 (보유 중인데 선택 안 된 종목) ===
            for ticker in list(positions.keys()):
                if ticker not in selected:
                    pos = positions.pop(ticker)
                    
                    # 청산 가격
                    df_t = enriched.get(ticker)
                    if df_t is None or d0 not in df_t.index:
                        continue
                    
                    exit_px = df_t.loc[d0, "close"] * (1 - self.slippage)
                    qty = pos["qty"]
                    entry_px = pos["entry_px"]
                    
                    # 수익 계산
                    proceeds = exit_px * qty
                    cost = entry_px * qty
                    pnl = proceeds - cost
                    
                    # 거래 비용
                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0
                    net_proceeds = proceeds - fee_out - tax
                    
                    cash += net_proceeds
                    
                    # 거래 로그
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_px,
                        "qty": qty,
                        "pnl": pnl,
                        "cash_after": cash,
                    })
            
            # === 7) 신규 진입 & 비중 조정 ===
            portfolio_value = cash + sum(
                enriched[t].loc[d0, "close"] * pos["qty"]
                for t, pos in positions.items()
                if enriched.get(t) is not None and d0 in enriched[t].index
            )
            
            for _, row in top_n_stocks.iterrows():
                ticker = row["ticker"]
                target_w = row["w"]
                target_value = portfolio_value * target_w
                
                df_t = enriched.get(ticker)
                if df_t is None or d0 not in df_t.index:
                    continue
                
                entry_px = df_t.loc[d0, "close"] * (1 + self.slippage)
                target_qty = int(target_value / entry_px)
                
                if target_qty <= 0:
                    continue
                
                # 현재 보유 수량
                current_qty = positions.get(ticker, {}).get("qty", 0)
                delta_qty = target_qty - current_qty
                
                if delta_qty == 0:
                    continue
                
                # 매수
                if delta_qty > 0:
                    cost = entry_px * delta_qty
                    fee_in = cost * self.fee
                    total_cost = cost + fee_in
                    
                    if total_cost > cash:
                        continue
                    
                    cash -= total_cost
                    
                    if ticker in positions:
                        # 기존 포지션 평균 단가 업데이트
                        old_qty = positions[ticker]["qty"]
                        old_px = positions[ticker]["entry_px"]
                        new_qty = old_qty + delta_qty
                        new_avg_px = (old_px * old_qty + entry_px * delta_qty) / new_qty
                        positions[ticker] = {"qty": new_qty, "entry_px": new_avg_px}
                    else:
                        positions[ticker] = {"qty": delta_qty, "entry_px": entry_px}
                    
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "action": "BUY",
                        "price": entry_px,
                        "qty": delta_qty,
                        "pnl": 0,
                        "cash_after": cash,
                    })
                
                # 매도 (비중 줄이기)
                elif delta_qty < 0:
                    sell_qty = -delta_qty
                    pos = positions[ticker]
                    
                    exit_px = entry_px * (1 - self.slippage)
                    proceeds = exit_px * sell_qty
                    cost = pos["entry_px"] * sell_qty
                    pnl = proceeds - cost
                    
                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0
                    net_proceeds = proceeds - fee_out - tax
                    
                    cash += net_proceeds
                    
                    # 포지션 업데이트
                    pos["qty"] -= sell_qty
                    if pos["qty"] <= 0:
                        positions.pop(ticker)
                    
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_px,
                        "qty": sell_qty,
                        "pnl": pnl,
                        "cash_after": cash,
                    })
            
            # === 8) 리밸런싱 ~ 다음 리밸런싱까지 equity 기록 ===
            period_dates = [d for d in dates if d0 <= d < next_rebal_date]
            for eval_date in period_dates:
                equity = self._calculate_equity(cash, positions, enriched, eval_date)
                equity_curve.append((eval_date, equity))
        
        # === 9) 마지막 날 최종 청산 & equity 기록 ===
        final_date = dates[-1]
        for ticker in list(positions.keys()):
            pos = positions.pop(ticker)
            df_t = enriched.get(ticker)
            
            if df_t is None or final_date not in df_t.index:
                continue
            
            exit_px = df_t.loc[final_date, "close"] * (1 - self.slippage)
            qty = pos["qty"]
            proceeds = exit_px * qty
            fee_out = proceeds * self.fee
            pnl = proceeds - (pos["entry_px"] * qty)
            tax = proceeds * self.tax if pnl > 0 else 0
            
            cash += proceeds - fee_out - tax
            
            trade_log.append({
                "date": final_date,
                "ticker": ticker,
                "action": "SELL",
                "price": exit_px,
                "qty": qty,
                "pnl": pnl,
                "cash_after": cash,
            })
        
        equity = cash
        equity_curve.append((final_date, equity))
        
        # === 10) 결과 반환 ===
        ec_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        
        if not silent:
            print(f"✅ KQM v3.2 백테스트 완료: {len(ec_df)}개 데이터 포인트")
        
        return ec_df, trade_log

