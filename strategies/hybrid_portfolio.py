#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Portfolio Strategy v2.0

- Korean Aggressive (70%) + Production Portfolio (30%)
- 소액 계좌(기본 100만원) + 한국 시장에 맞춘 유동성/변동성 필터 내장
- 기존 v1의 현금/포지션 계산 버그를 완전히 수정
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class HybridPortfolioStrategy(BaseStrategy):
    """
    Hybrid Portfolio: Korean Aggressive (70%) + Production Portfolio (30%)

    v2.0 주요 변경 사항
    -------------------
    1) 유니버스 필터 내장
       - 최근 20일 평균 거래대금 >= 20억
       - 가격 2,000 ~ 80,000원
       - 최근 20일 중 15일 이상 거래

    2) 시그널 개선
       - Korean Aggressive: RSI / MA10 / 최근 3일 구조 / 거래량 증가 필터
       - Production: MA20 위 추세 + 60일 고점 근처 + 중간 RSI 구간

    3) 포트폴리오 레이어
       - Korean 포지션 최대 4개, Production 최대 3개 (기본)
       - 전략별 별도 현금 계정(korean_cash / portfolio_cash)으로 관리
       - 단일 종목 최대 비중 max_single_stock_ratio (기본 20%)
    """

    def __init__(
        self,
        korean_aggressive_ratio: float = 0.70,
        production_portfolio_ratio: float = 0.30,
        korean_max_positions: int = 4,
        portfolio_max_positions: int = 3,
        korean_position_size: float = 0.25,     # 전략 내 캐시의 25%씩
        portfolio_position_size: float = 0.33,  # 전략 내 캐시의 33%씩
        max_single_stock_ratio: float = 0.20,   # 전체 포트폴리오 대비 단일 종목 최대 20%
        min_tvalue_20d: float = 2_000_000_000,  # 20일 평균 거래대금 최소 20억
        min_price: int = 2_000,
        max_price: int = 80_000,
        min_active_days_20d: int = 15,
        max_vol_20d: float = 0.12,  # 20일 일간 변동성 최대 12%
        slippage: float = 0.001
    ):
        self.korean_aggressive_ratio = korean_aggressive_ratio
        self.production_portfolio_ratio = production_portfolio_ratio
        self.korean_max_positions = korean_max_positions
        self.portfolio_max_positions = portfolio_max_positions
        self.korean_position_size = korean_position_size
        self.portfolio_position_size = portfolio_position_size
        self.max_single_stock_ratio = max_single_stock_ratio

        # 유니버스/유동성 파라미터
        self.min_tvalue_20d = min_tvalue_20d
        self.min_price = min_price
        self.max_price = max_price
        self.min_active_days_20d = min_active_days_20d
        self.max_vol_20d = max_vol_20d

        # 비용
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = slippage

    # ------------------------------------------------------------------
    # 기본 정보
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "hybrid_portfolio_v2"

    def get_description(self) -> str:
        return (
            f"Hybrid Portfolio v2.0 (Korean {self.korean_aggressive_ratio:.0%} + "
            f"Portfolio {self.production_portfolio_ratio:.0%})"
        )

    # ------------------------------------------------------------------
    # 유틸: 유동성 + 변동성 필터
    # ------------------------------------------------------------------
    def _passes_universe_filter(self, df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
        """
        현실적인 유동성/가격/활성도 필터.
        - 최근 20일 평균 거래대금 >= min_tvalue_20d
        - 최근 20일 중 거래 활성일 >= min_active_days_20d
        - 현재 가격 [min_price, max_price]
        - 20일 일간 수익률 표준편차 <= max_vol_20d (너무 미친 놈 제거)
        """
        if df is None or current_date not in df.index:
            return False

        hist = df.loc[:current_date].tail(20)
        if len(hist) < 20:
            return False

        close = hist["close"]
        volume = hist["volume"]

        price = close.iloc[-1]
        if not (self.min_price <= price <= self.max_price):
            return False

        # 거래대금
        tvalue = (close * volume).mean()
        if tvalue < self.min_tvalue_20d:
            return False

        # 활성일
        active_days = (volume > 0).sum()
        if active_days < self.min_active_days_20d:
            return False

        # 20일 변동성
        rets = close.pct_change().dropna()
        if len(rets) >= 5:
            vol_20d = rets.std()
            if vol_20d > self.max_vol_20d:
                return False

        return True

    # ------------------------------------------------------------------
    # 유틸: 포트폴리오 가치 계산
    # ------------------------------------------------------------------
    def _portfolio_value(
        self,
        current_date: pd.Timestamp,
        enriched: dict,
        korean_cash: float,
        portfolio_cash: float,
        korean_positions: dict,
        portfolio_positions: dict
    ) -> float:
        total = korean_cash + portfolio_cash

        for ticker, pos in korean_positions.items():
            df = enriched.get(ticker)
            if df is not None and current_date in df.index:
                price = df.loc[current_date, "close"]
            else:
                price = pos["entry_px"]
            total += price * pos["qty"]

        for ticker, pos in portfolio_positions.items():
            df = enriched.get(ticker)
            if df is not None and current_date in df.index:
                price = df.loc[current_date, "close"]
            else:
                price = pos["entry_px"]
            total += price * pos["qty"]

        return total

    # ------------------------------------------------------------------
    # 유틸: 단일 종목 최대 비중 체크
    # ------------------------------------------------------------------
    def _can_open_position(
        self,
        ticker: str,
        price: float,
        qty: int,
        current_date: pd.Timestamp,
        enriched: dict,
        korean_cash: float,
        portfolio_cash: float,
        korean_positions: dict,
        portfolio_positions: dict
    ) -> bool:
        """
        신규 포지션 추가 시 단일 종목 비중이 max_single_stock_ratio를 넘지 않는지 체크
        """
        if qty <= 0:
            return False

        new_position_value = price * qty
        total_equity = self._portfolio_value(
            current_date,
            enriched,
            korean_cash,
            portfolio_cash,
            korean_positions,
            portfolio_positions
        )

        # total_equity가 0이면(초기 직후 등) 우선 허용
        if total_equity <= 0:
            return True

        new_ratio = new_position_value / total_equity
        return new_ratio <= self.max_single_stock_ratio + 1e-6

    # ------------------------------------------------------------------
    # 시그널: Korean Aggressive 후보 스캔
    # ------------------------------------------------------------------
    def _scan_korean_candidates(
        self,
        current_date: pd.Timestamp,
        enriched: dict,
        korean_positions: dict,
        portfolio_positions: dict
    ) -> List[dict]:
        """
        Korean Aggressive 개선 버전 시그널:
        - 유니버스 필터 통과
        - RSI 45~78
        - close > MA10
        - 최근 3일 연속 양봉 금지
        - 최근 3일 거래량 증가 또는 유지
        """
        candidates = []

        for ticker, df in enriched.items():
            if df is None or current_date not in df.index:
                continue

            # 이미 보유 중인 종목 제외
            if ticker in korean_positions or ticker in portfolio_positions:
                continue

            # 유니버스 필터
            if not self._passes_universe_filter(df, current_date):
                continue

            hist = df.loc[:current_date]
            if len(hist) < 60:
                continue

            row = hist.iloc[-1]
            close = row["close"]
            volume = row["volume"]

            # RSI / MA10
            rsi = row.get("rsi", np.nan)
            ma10 = row.get("ma10", np.nan)
            if np.isnan(rsi) or np.isnan(ma10):
                continue

            if not (45 <= rsi <= 78):
                continue

            if close <= ma10:
                continue

            # 최근 3일 수익률 / 거래량
            recent = hist.tail(4)  # 오늘 포함 4캔들
            if len(recent) < 4:
                continue

            # 연속 양봉 방지 (이전 3일 모두 양봉 금지)
            rets = recent["close"].pct_change().dropna()
            if len(rets) >= 3 and all(rets[-3:] > 0):
                continue

            # 거래량 증가 (3일 평균 >= 직전 10일 평균 * 1.2)
            vol3 = recent["volume"].tail(3).mean()
            vol10 = hist["volume"].tail(10).mean()
            if vol10 <= 0:
                continue
            if vol3 < vol10 * 1.2:
                continue

            # 5일 변동폭 너무 큰 종목 제거 (over-extended 방지)
            last5 = hist.tail(5)["close"]
            rng = last5.max() / last5.min() - 1
            if rng > 0.20:  # 20% 이상 넓게 흔들린 놈 제외
                continue

            score = (rsi - 45) / 33 + vol3 / (vol10 + 1e-9)
            candidates.append(
                {
                    "ticker": ticker,
                    "price": close,
                    "volume": volume,
                    "rsi": rsi,
                    "score": score,
                }
            )

        # 점수 순 정렬
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 시그널: Production Portfolio 후보 스캔
    # ------------------------------------------------------------------
    def _scan_production_candidates(
        self,
        current_date: pd.Timestamp,
        enriched: dict,
        korean_positions: dict,
        portfolio_positions: dict
    ) -> List[dict]:
        """
        Production Portfolio 개선 버전 시그널:
        - 유니버스 필터 통과
        - MA20 위에서 머무는 우상향 종목
        - 최근 60일 고점 근처 (0.97 * 60일 고가 이상)
        - RSI 45~68 (중간 구간)
        """
        candidates = []

        for ticker, df in enriched.items():
            if df is None or current_date not in df.index:
                continue

            if ticker in korean_positions or ticker in portfolio_positions:
                continue

            if not self._passes_universe_filter(df, current_date):
                continue

            hist = df.loc[:current_date]
            if len(hist) < 80:
                continue

            row = hist.iloc[-1]
            close = row["close"]
            volume = row["volume"]

            rsi = row.get("rsi", np.nan)
            ma20 = row.get("ma20", np.nan)
            if np.isnan(rsi) or np.isnan(ma20):
                continue

            # RSI 중간 구간
            if not (45 <= rsi <= 68):
                continue

            # MA20 위
            if close < ma20:
                continue

            # 60일 고점 근처
            last60 = hist.tail(60)
            high60 = last60["close"].max()
            if high60 <= 0:
                continue
            if close < high60 * 0.97:
                continue

            # 거래량 감소 추세는 제외 (최근 10일 평균 >= 이전 20일 평균 * 0.9)
            vol10 = last60["volume"].tail(10).mean()
            vol20_prev = last60["volume"].head(40).mean()
            if vol20_prev > 0 and vol10 < vol20_prev * 0.9:
                continue

            score = (close / high60) + vol10 / (vol20_prev + 1e-9)
            candidates.append(
                {
                    "ticker": ticker,
                    "price": close,
                    "volume": volume,
                    "rsi": rsi,
                    "score": score,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 메인 백테스트 루프
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        enriched: dict,
        market_index=None,
        weights: dict = None,
        silent: bool = False
    ) -> tuple:
        """
        enriched: {ticker: df(OHLCV + indicators)}
        return: (equity_df, trade_log)
        """

        if not silent:
            print("\n" + "=" * 60)
            print(f"🔥 {self.get_description()} 백테스트 시작 (v2.0)")
            print("=" * 60)

        # 날짜 세트 구성
        all_dates = set()
        for df in enriched.values():
            if df is not None and len(df) > 0:
                all_dates.update(df.index)

        dates = sorted(all_dates)
        if len(dates) < 80:
            if not silent:
                print("⚠️ 데이터가 너무 짧음 (80일 미만)")
            return pd.DataFrame(), []

        # 초기 자본
        initial_capital = 1_000_000.0  # 100만원
        korean_cash = initial_capital * self.korean_aggressive_ratio
        portfolio_cash = initial_capital * self.production_portfolio_ratio

        # 포지션 관리
        korean_positions: Dict[str, dict] = {}
        portfolio_positions: Dict[str, dict] = {}

        equity_curve: List[dict] = []
        trade_log: List[dict] = []

        # 워밍업 기간 (각종 지표 계산 여유)
        start_idx = 80
        total_days = len(dates) - start_idx

        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            
            # 진행률 표시 (10% 간격)
            if not silent and (i - start_idx) % max(1, total_days // 10) == 0:
                progress = ((i - start_idx) / total_days) * 100
                print(f"  📊 진행률: {progress:.1f}% ({i - start_idx}/{total_days}일 처리 완료, 현재: {current_date.strftime('%Y-%m-%d')})")

            # ----------------------------------------------------------
            # 1) 기존 포지션 청산 조건 체크 (손절/익절/기간 만료)
            # ----------------------------------------------------------
            for book_name, positions in [("korean", korean_positions), ("portfolio", portfolio_positions)]:
                to_close = []

                for ticker, pos in positions.items():
                    df = enriched.get(ticker)
                    if df is None or current_date not in df.index:
                        continue

                    row = df.loc[current_date]
                    price = row["close"]
                    entry = pos["entry_px"]
                    qty = pos["qty"]
                    entry_date = pos["entry_date"]

                    if qty <= 0:
                        to_close.append(ticker)
                        continue

                    pnl_pct = (price / entry) - 1.0
                    days_held = (current_date - entry_date).days

                    reason = None

                    # 공통 청산 룰 (v2 튜닝)
                    if pnl_pct <= -0.10:
                        reason = "STOP_LOSS_-10%"
                    elif pnl_pct >= 0.20:
                        reason = "TAKE_PROFIT_+20%"
                    elif days_held >= 10 and pnl_pct >= 0.05:
                        reason = "TIME_PROFIT_10D_+5%"
                    elif days_held >= 20:
                        reason = "TIME_EXIT_20D"

                    if reason is not None:
                        # 슬리피지 반영
                        exit_price = price * (1 - self.slippage)
                        gross = exit_price * qty
                        fee = gross * self.fee
                        tax = gross * self.tax if pnl_pct > 0 else 0.0
                        net = gross - fee - tax

                        cost = entry * qty * (1 + self.fee)
                        pnl = net - cost

                        if book_name == "korean":
                            korean_cash += net
                        else:
                            portfolio_cash += net

                        trade_log.append(
                            {
                                "date": current_date,
                                "ticker": ticker,
                                "strategy": pos["strategy"],
                                "action": "SELL",
                                "price": exit_price,
                                "qty": qty,
                                "amount": net,
                                "pnl": pnl,
                                "pnl_pct": pnl_pct,
                                "days_held": days_held,
                                "reason": reason,
                            }
                        )
                        to_close.append(ticker)

                        if not silent:
                            print(
                                f"  💰 [{book_name}] SELL {ticker} @ {exit_price:,.0f} x {qty} "
                                f"= {pnl:+,.0f}원 ({reason})"
                            )

                for t in to_close:
                    positions.pop(t, None)

            # ----------------------------------------------------------
            # 2) 매수 신호: Korean Aggressive
            # ----------------------------------------------------------
            if len(korean_positions) < self.korean_max_positions and korean_cash > 0:
                k_candidates = self._scan_korean_candidates(
                    current_date, enriched, korean_positions, portfolio_positions
                )
                # 하루 최대 1개만 매수
                for c in k_candidates[:3]:
                    price = c["price"]
                    # 포지션 크기
                    target_notional = korean_cash * self.korean_position_size
                    qty = int(target_notional / (price * (1 + self.fee + self.slippage)))
                    if qty <= 0:
                        continue

                    # 포트폴리오 단일 종목 비중 체크
                    if not self._can_open_position(
                        c["ticker"],
                        price,
                        qty,
                        current_date,
                        enriched,
                        korean_cash,
                        portfolio_cash,
                        korean_positions,
                        portfolio_positions,
                    ):
                        continue

                    notional = price * qty
                    fee = notional * self.fee
                    total_cost = notional * (1 + self.slippage) + fee

                    if total_cost > korean_cash:
                        continue

                    korean_cash -= total_cost
                    korean_positions[c["ticker"]] = {
                        "qty": qty,
                        "entry_px": price * (1 + self.slippage),
                        "entry_date": current_date,
                        "strategy": "korean_aggressive",
                    }

                    trade_log.append(
                        {
                            "date": current_date,
                            "ticker": c["ticker"],
                            "strategy": "korean_aggressive",
                            "action": "BUY",
                            "price": price * (1 + self.slippage),
                            "qty": qty,
                            "amount": total_cost,
                            "pnl": 0.0,
                        }
                    )

                    if not silent:
                        print(
                            f"  💚 [korean] BUY {c['ticker']} @ {price*(1+self.slippage):,.0f} x {qty} "
                            f"(cash left {korean_cash:,.0f})"
                        )
                    break  # 하루 한 종목만

            # ----------------------------------------------------------
            # 3) 매수 신호: Production Portfolio
            # ----------------------------------------------------------
            if len(portfolio_positions) < self.portfolio_max_positions and portfolio_cash > 0:
                p_candidates = self._scan_production_candidates(
                    current_date, enriched, korean_positions, portfolio_positions
                )

                for c in p_candidates[:3]:
                    price = c["price"]
                    target_notional = portfolio_cash * self.portfolio_position_size
                    qty = int(target_notional / (price * (1 + self.fee + self.slippage)))
                    if qty <= 0:
                        continue

                    if not self._can_open_position(
                        c["ticker"],
                        price,
                        qty,
                        current_date,
                        enriched,
                        korean_cash,
                        portfolio_cash,
                        korean_positions,
                        portfolio_positions,
                    ):
                        continue

                    notional = price * qty
                    fee = notional * self.fee
                    total_cost = notional * (1 + self.slippage) + fee

                    if total_cost > portfolio_cash:
                        continue

                    portfolio_cash -= total_cost
                    portfolio_positions[c["ticker"]] = {
                        "qty": qty,
                        "entry_px": price * (1 + self.slippage),
                        "entry_date": current_date,
                        "strategy": "production_portfolio",
                    }

                    trade_log.append(
                        {
                            "date": current_date,
                            "ticker": c["ticker"],
                            "strategy": "production_portfolio",
                            "action": "BUY",
                            "price": price * (1 + self.slippage),
                            "qty": qty,
                            "amount": total_cost,
                            "pnl": 0.0,
                        }
                    )

                    if not silent:
                        print(
                            f"  📊 [portfolio] BUY {c['ticker']} @ {price*(1+self.slippage):,.0f} x {qty} "
                            f"(cash left {portfolio_cash:,.0f})"
                        )
                    break

            # ----------------------------------------------------------
            # 4) Equity Curve 기록 (End-of-Day)
            # ----------------------------------------------------------
            total_equity = self._portfolio_value(
                current_date,
                enriched,
                korean_cash,
                portfolio_cash,
                korean_positions,
                portfolio_positions,
            )
            equity_curve.append({"date": current_date, "equity": total_equity})

        # --------------------------------------------------------------
        # 5) 마지막 날 포지션 강제 청산 (선택적, equity는 동일하지만 로그용)
        # --------------------------------------------------------------
        final_date = dates[-1]
        for book_name, positions in [("korean", korean_positions), ("portfolio", portfolio_positions)]:
            to_close = list(positions.keys())
            for ticker in to_close:
                pos = positions[ticker]
                df = enriched.get(ticker)
                if df is None or final_date not in df.index:
                    continue

                price = df.loc[final_date, "close"]
                qty = pos["qty"]
                entry = pos["entry_px"]
                days_held = (final_date - pos["entry_date"]).days
                pnl_pct = (price / entry) - 1.0

                exit_price = price * (1 - self.slippage)
                gross = exit_price * qty
                fee = gross * self.fee
                tax = gross * self.tax if pnl_pct > 0 else 0.0
                net = gross - fee - tax
                cost = entry * qty * (1 + self.fee)
                pnl = net - cost

                if book_name == "korean":
                    korean_cash += net
                else:
                    portfolio_cash += net

                trade_log.append(
                    {
                        "date": final_date,
                        "ticker": ticker,
                        "strategy": pos["strategy"],
                        "action": "SELL",
                        "price": exit_price,
                        "qty": qty,
                        "amount": net,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "days_held": days_held,
                        "reason": "FINAL_LIQ",
                    }
                )
                positions.pop(ticker, None)

        # 최종 equity 재계산
        final_equity = korean_cash + portfolio_cash
        equity_curve.append({"date": final_date, "equity": final_equity})

        equity_df = pd.DataFrame(equity_curve).drop_duplicates("date", keep="last")
        equity_df.set_index("date", inplace=True)
        equity_df.sort_index(inplace=True)

        if not silent:
            print("\n✅ Hybrid v2.0 백테스트 완료")
            print(f"   최종 자산: {final_equity:,.0f}원 (초기 1,000,000원 기준)")
            print(f"   총 거래 수: {len(trade_log)}회")

        return equity_df, trade_log