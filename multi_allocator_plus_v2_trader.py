#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실거래용 Multi Allocator Plus v2 트레이더
- 최신 데이터로 multi_allocator_plus_v2 목표 비중을 계산
- 한국투자증권(KIS) API를 통해 주문 계획을 생성/전송
- cleaned_stock_proj의 hybrid_portfolio_trader 구현 패턴을 참조
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from reports import load_data
from strategies import get_strategy
from universe_filter import filter_universe
from automation.telegram_notifier import TelegramNotifier, format_alert
from automation.daily_reporter import DailyReporter

# .env 로컬 테스트 지원
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# cleaned_stock_proj 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent
CLEANED_ROOT = PROJECT_ROOT.parent / "cleaned_stock_proj"
if CLEANED_ROOT.exists():
    sys.path.append(str(CLEANED_ROOT))

try:
    from kiwoom_api.core.korea_investment_connector import KoreaInvestmentConnector
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "KoreaInvestmentConnector를 불러올 수 없습니다. "
        "cleaned_stock_proj 경로가 존재하는지 확인하세요."
    ) from exc


logger = logging.getLogger("multi_allocator_plus_v2_trader")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger.setLevel(logging.INFO)


@dataclass
class OrderPlan:
    symbol: str
    action: str  # BUY / SELL
    quantity: int
    est_price: float
    est_value: float
    target_weight: float
    current_qty: int
    target_qty: int


class MultiAllocatorPlusV2Trader:
    def __init__(self, start_date: str = None, use_cache: bool = True,
                 dry_run: bool = True, virtual_account: bool = True,
                 min_trade_value: int = 200_000):
        self.start_date = start_date
        self.use_cache = use_cache
        self.dry_run = dry_run
        self.virtual_account = virtual_account
        self.min_trade_value = min_trade_value

        self.kis = KoreaInvestmentConnector(virtual_account=virtual_account)
        self.telegram = TelegramNotifier()
        self.reporter = DailyReporter(PROJECT_ROOT / "reports" / "daily")
        self.strategy = get_strategy("multi_allocator_plus_v2")
        if self.strategy is None:
            raise RuntimeError("multi_allocator_plus_v2 전략을 찾을 수 없습니다.")

        self.enriched = {}
        self.market_index = None

    def load_market_data(self):
        enriched, idx_map = load_data(
            use_cache=self.use_cache,
            start_date=self.start_date
        )
        # multi allocator는 전체 enriched를 필요로 함
        self.enriched = enriched
        self.market_index = idx_map.get("KOSDAQ")
        universe = filter_universe(enriched)
        logger.info("✅ 데이터 로드 완료 - 유니버스 %d개", len(universe))

    def compute_target_weights(self) -> Tuple[pd.Timestamp, pd.Series]:
        targets = self.strategy.compute_security_targets(
            self.enriched, market_index=self.market_index, silent=True
        )
        if targets is None or targets.empty:
            raise RuntimeError("타깃 비중 계산 실패")
        latest_date = targets.index.max()
        latest_row = targets.loc[latest_date].drop("__CASH__", errors="ignore")
        latest_row = latest_row[latest_row > 0].sort_values(ascending=False)
        total = latest_row.sum()
        if total > 0:
            latest_row = latest_row / total
        logger.info("🎯 타깃 비중 산출 완료 (%s)", latest_date.date())
        for ticker, weight in latest_row.items():
            logger.info("  %s -> %.2f%%", ticker, weight * 100)
        return latest_date, latest_row

    def fetch_account_snapshot(self) -> Tuple[Dict, Dict]:
        balance_raw = self.kis.get_account_balance()
        account = self.kis.parse_account_balance_data(balance_raw)
        holdings_list = self.kis.get_account_stocks()
        holdings = {}
        for item in holdings_list:
            symbol = self._normalize_symbol(item.get("symbol", ""))
            holdings[symbol] = item
        logger.info("💰 계좌 총자산: %s원 / 매수가능: %s원",
                    f"{account.get('total_value', 0):,.0f}",
                    f"{account.get('available_cash', 0):,.0f}")
        return account, holdings

    def build_order_plan(self, targets: pd.Series, account: Dict,
                         holdings: Dict) -> List[OrderPlan]:
        total_equity = account.get("total_value") or (
            account.get("available_cash", 0) + account.get("stock_value", 0)
        )
        plans: List[OrderPlan] = []
        price_cache = self._latest_prices(targets.index)

        # 타깃 종목 우선 처리
        for ticker, weight in targets.items():
            price = price_cache.get(ticker)
            if price is None or price <= 0:
                continue
            target_value = total_equity * weight
            if target_value < self.min_trade_value:
                continue
            target_qty = int(target_value / price)
            current_qty = holdings.get(self._normalize_symbol(ticker), {}).get("quantity", 0)
            delta = target_qty - current_qty
            if abs(delta) == 0:
                continue
            action = "BUY" if delta > 0 else "SELL"
            plans.append(OrderPlan(
                symbol=ticker,
                action=action,
                quantity=abs(delta),
                est_price=price,
                est_value=abs(delta) * price,
                target_weight=weight,
                current_qty=current_qty,
                target_qty=target_qty
            ))

        # 타깃에서 제외된 기존 포지션 정리
        target_set = set(self._normalize_symbol(t) for t in targets.index)
        for symbol, pos in holdings.items():
            if symbol not in target_set and pos.get("quantity", 0) > 0:
                price = pos.get("current_price") or price_cache.get(symbol)
                if price is None:
                    price = 0
                plans.append(OrderPlan(
                    symbol=symbol,
                    action="SELL",
                    quantity=pos["quantity"],
                    est_price=price,
                    est_value=pos["quantity"] * (price or 0),
                    target_weight=0.0,
                    current_qty=pos["quantity"],
                    target_qty=0
                ))

        plans.sort(key=lambda x: (-1 if x.action == "SELL" else 1, -x.est_value))
        return plans

    def execute(self, plans: List[OrderPlan], account: Dict, holdings: Dict, as_of: datetime):
        if not plans:
            logger.info("🚫 실행할 주문이 없습니다.")
            return
        account_no = self.kis.account
        executed_orders = 0
        for plan in plans:
            logger.info("➡️ %s %s x %s (목표 %.2f%%)",
                        plan.action, plan.symbol, plan.quantity, plan.target_weight * 100)
            if self.dry_run:
                continue
            order_type = 1 if plan.action == "BUY" else 2
            try:
                result = self.kis.send_order(
                    request_name=f"multi_alloc_{plan.action.lower()}",
                    screen_no="9000",
                    account_no=account_no,
                    order_type=order_type,
                    stock_code=self._normalize_symbol(plan.symbol),
                    quantity=int(plan.quantity),
                    price=0,
                    quote_type="03"
                )
                if result == 0:
                    executed_orders += 1
            except Exception as exc:
                logger.error("주문 실패: %s (%s)", plan.symbol, exc)
        equity = account.get("total_value") or (
            account.get("available_cash", 0) + account.get("stock_value", 0)
        )
        snapshot = {"account": account, "holdings": list(holdings.values())}
        report_path = self.reporter.save_report(as_of, equity, [plan.__dict__ for plan in plans], snapshot)
        if self.dry_run:
            logger.info("dry-run 모드이므로 텔레그램 알림을 생략합니다.")
            return
        if executed_orders == 0:
            logger.info("실제 체결된 주문이 없어 텔레그램 알림을 생략합니다.")
            return
        self._notify(
            latest_equity=equity,
            plans=plans,
            report_path=report_path,
            report_date=as_of,
            trade_time=datetime.now(),
        )

    def _notify(
        self,
        latest_equity: float,
        plans: List[OrderPlan],
        report_path: Path,
        report_date: datetime,
        trade_time: datetime | None = None,
    ):
        if not self.telegram.can_send():
            return
        trade_date = trade_time.date() if trade_time else report_date.date()
        lines = [
            f"날짜: {trade_date}",
            f"총자산: {latest_equity:,.0f}원",
            f"주문 수: {len(plans)}",
        ]
        for plan in plans[:5]:
            lines.append(f"- {plan.action} {plan.symbol} {plan.quantity}주")
        if len(plans) > 5:
            lines.append(f"...외 {len(plans) - 5}건")
        lines.append(f"리포트: {report_path.name}")
        self.telegram.send_message(format_alert("Multi Allocator PLUS v2", lines))

    def run(self):
        self.load_market_data()
        last_date, targets = self.compute_target_weights()
        account, holdings = self.fetch_account_snapshot()
        plans = self.build_order_plan(targets, account, holdings)
        if plans:
            logger.info("📋 주문 계획 (%s):", last_date.date())
            for plan in plans:
                logger.info(
                    "  %s %s주 @ %.0f원 (보유 %s주 → 목표 %s주)",
                    plan.action, plan.quantity, plan.est_price,
                    plan.current_qty, plan.target_qty
                )
        else:
            logger.info("📋 주문 계획 없음")
        self.execute(plans, account, holdings, last_date)

    def _latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        prices = {}
        for ticker in tickers:
            df = self.enriched.get(ticker)
            if df is None or "close" not in df.columns:
                continue
            last_close = df["close"].dropna()
            if not last_close.empty:
                prices[ticker] = float(last_close.iloc[-1])
        return prices

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if symbol is None:
            return ""
        symbol = symbol.replace("A", "")
        return symbol.strip()


def main():
    parser = argparse.ArgumentParser(description="Multi Allocator Plus v2 실거래 트레이더")
    parser.add_argument("--start-date", type=str, default=None,
                        help="데이터 로딩 시작일 (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true",
                        help="데이터 캐시 사용 안 함")
    parser.add_argument("--real", action="store_true",
                        help="실거래 모드 (기본값: 모의투자)")
    parser.add_argument("--dry-run", action="store_true",
                        help="주문 미전송, 계획만 출력")
    parser.add_argument("--min-trade", type=int, default=200_000,
                        help="최소 매매 금액 기준")

    args = parser.parse_args()
    trader = MultiAllocatorPlusV2Trader(
        start_date=args.start_date,
        use_cache=not args.no_cache,
        dry_run=args.dry_run,
        virtual_account=not args.real,
        min_trade_value=args.min_trade
    )
    trader.run()


if __name__ == "__main__":
    main()
