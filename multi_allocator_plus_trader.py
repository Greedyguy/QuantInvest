#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi Allocator PLUS 실거래 트레이더
cleaned_stock_proj의 hybrid_portfolio_trader 패턴을 참고하여
multi_allocator_plus 전략 목표 비중을 계산하고 한국투자증권 API로 주문 계획 생성
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from reports import load_data
from strategies import get_strategy
from universe_filter import filter_universe
from automation.telegram_notifier import TelegramNotifier, format_alert
from automation.daily_reporter import DailyReporter
from config import BLOCKED_TICKERS

# .env 로컬 테스트 지원
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
CLEANED_ROOT = PROJECT_ROOT.parent / "cleaned_stock_proj"
if CLEANED_ROOT.exists():
    sys.path.append(str(CLEANED_ROOT))

try:
    from kiwoom_api.core.korea_investment_connector import (
        KoreaInvestmentConnector,
        ORDER_ERR_ACCOUNT_NOT_ELIGIBLE,
        ORDER_ERR_MARKET_OPERATION_DATE_MISMATCH,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("cleaned_stock_proj 경로에서 KoreaInvestmentConnector를 찾을 수 없습니다.") from exc

logger = logging.getLogger("multi_allocator_plus_trader")
logging.basicConfig(
    level=logging.WARNING,  # 루트 로거는 WARNING 이상만
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger.setLevel(logging.INFO)


@dataclass
class OrderPlan:
    symbol: str
    action: str
    quantity: int
    est_price: float
    est_value: float
    target_weight: float
    current_qty: int
    target_qty: int


class MultiAllocatorPlusTrader:
    def __init__(
        self,
        start_date: str = None,
        use_cache: bool = True,
        dry_run: bool = True,
        virtual_account: bool = True,
        min_trade_value: int = 200_000,
        cache_only: bool = False,
    ):
        self.start_date = start_date
        self.use_cache = use_cache
        self.dry_run = dry_run
        self.virtual_account = virtual_account
        self.min_trade_value = min_trade_value
        self.cache_only = cache_only

        self.kis = KoreaInvestmentConnector(virtual_account=virtual_account)
        self.telegram = TelegramNotifier()
        self.reporter = DailyReporter(PROJECT_ROOT / "reports" / "daily")
        # 파생 ETF를 사용하지 않는 버전 사용
        # (multi_allocator_plus_no_etf: etf_defensive 자식 전략 제거)
        self.strategy = get_strategy("multi_allocator_plus_no_etf")
        if self.strategy is None:
            raise RuntimeError("multi_allocator_plus 전략을 찾을 수 없습니다.")

        self.enriched = {}
        self.market_index = None

    def load_market_data(self):
        enriched, idx_map = load_data(
            use_cache=self.use_cache,
            start_date=self.start_date,
        )
        self.enriched = enriched
        self.market_index = idx_map.get("KOSDAQ")
        universe = filter_universe(enriched)
        logger.info("✅ 데이터 로드 완료 - 유니버스 %d개", len(universe))

    def compute_target_weights(self) -> Tuple[pd.Timestamp, pd.Series]:
        targets = self.strategy.compute_security_targets(
            self.enriched,
            market_index=self.market_index,
            silent=True,
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
        # dry-run 모드에서는 KIS API를 부르지 않고 가상 계좌 스냅샷 사용
        if self.dry_run:
            account = {
                "account_no": "SIM_ACCOUNT",
                "total_cash": 1_000_000,
                "available_cash": 1_000_000,
                "total_value": 1_000_000,
                "stock_value": 0,
            }
            holdings: Dict[str, Dict] = {}
            logger.info("💡 dry-run 모드: KIS 잔고 대신 가상 자본 1,000,000원 사용")
            return account, holdings

        balance_raw = self.kis.get_account_balance()
        account = self.kis.parse_account_balance_data(balance_raw)
        holdings_list = self.kis.get_account_stocks()
        holdings: Dict[str, Dict] = {}
        for item in holdings_list:
            symbol = self._normalize_symbol(item.get("symbol", ""))
            holdings[symbol] = item
        logger.info(
            "💰 계좌 총자산: %s원 / 매수가능: %s원",
            f"{account.get('total_value', 0):,.0f}",
            f"{account.get('available_cash', 0):,.0f}",
        )
        return account, holdings

    def build_order_plan(
        self,
        targets: pd.Series,
        account: Dict,
        holdings: Dict,
    ) -> List[OrderPlan]:
        blocked = set(BLOCKED_TICKERS or set())
        if blocked:
            targets = targets.drop(
                [ticker for ticker in targets.index if self._normalize_symbol(ticker) in blocked],
                errors="ignore",
            )

        total_equity = account.get("total_value") or (
            account.get("available_cash", 0) + account.get("stock_value", 0)
        )
        # dry-run 모드에서는 최소 100만원 기준으로 수량 계산
        if self.dry_run and (not total_equity or total_equity <= 0):
            total_equity = 1_000_000
            logger.info("💡 dry-run 모드: 가상 초기 자본 1,000,000원으로 주문 수량 계산")

        effective_min_trade = self.min_trade_value
        if self.dry_run:
            effective_min_trade = min(self.min_trade_value, 50_000)
        plans: List[OrderPlan] = []
        price_cache = self._latest_prices(targets.index)

        for ticker, weight in targets.items():
            price = price_cache.get(ticker)
            if price is None or price <= 0:
                continue
            target_value = total_equity * weight
            if target_value < effective_min_trade:
                continue
            target_qty = int(target_value / price)
            current_qty = holdings.get(self._normalize_symbol(ticker), {}).get("quantity", 0)
            delta = target_qty - current_qty
            if delta == 0:
                continue
            action = "BUY" if delta > 0 else "SELL"
            plans.append(
                OrderPlan(
                    symbol=ticker,
                    action=action,
                    quantity=abs(delta),
                    est_price=price,
                    est_value=abs(delta) * price,
                    target_weight=weight,
                    current_qty=current_qty,
                    target_qty=target_qty,
                )
            )

        target_set = set(self._normalize_symbol(t) for t in targets.index)
        for symbol, pos in holdings.items():
            if symbol not in target_set and pos.get("quantity", 0) > 0:
                price = pos.get("current_price") or price_cache.get(symbol, 0)
                plans.append(
                    OrderPlan(
                        symbol=symbol,
                        action="SELL",
                        quantity=pos["quantity"],
                        est_price=price,
                        est_value=pos["quantity"] * price if price else 0,
                        target_weight=0.0,
                        current_qty=pos["quantity"],
                        target_qty=0,
                    )
                )

        plans.sort(key=lambda x: (-1 if x.action == "SELL" else 1, -x.est_value))
        return plans

    def execute(self, plans: List[OrderPlan], account: Dict, holdings: Dict, as_of: datetime):
        if not plans:
            logger.info("🚫 실행할 주문이 없습니다.")
            return
        account_no = self.kis.account
        executed_orders = 0
        failed_orders = 0
        not_eligible_orders = 0
        abort_reason: str | None = None
        for plan in plans:
            logger.info(
                "➡️ %s %s x %s (목표 %.2f%%)",
                plan.action,
                plan.symbol,
                plan.quantity,
                plan.target_weight * 100,
            )
            if self.dry_run:
                continue
            order_type = 1 if plan.action == "BUY" else 2
            try:
                result = self.kis.send_order(
                    request_name=f"multi_alloc_plus_{plan.action.lower()}",
                    screen_no="9001",
                    account_no=account_no,
                    order_type=order_type,
                    stock_code=self._normalize_symbol(plan.symbol),
                    quantity=int(plan.quantity),
                    price=0,
                    quote_type="03",
                )
                if result == 0:
                    executed_orders += 1
                else:
                    failed_orders += 1
                    if result == ORDER_ERR_MARKET_OPERATION_DATE_MISMATCH:
                        abort_reason = "휴장/영업일 불일치(KIS: 장운영일자≠주문일)"
                        break
                    if result == ORDER_ERR_ACCOUNT_NOT_ELIGIBLE:
                        not_eligible_orders += 1
            except Exception as exc:
                logger.error("주문 실패: %s (%s)", plan.symbol, exc)
                failed_orders += 1
        equity = account.get("total_value") or (
            account.get("available_cash", 0) + account.get("stock_value", 0)
        )
        if self.dry_run and (not equity or equity <= 0):
            equity = 1_000_000
        snapshot = {"account": account, "holdings": list(holdings.values())}
        report_path = self.reporter.save_report(as_of, equity, [plan.__dict__ for plan in plans], snapshot)
        if self.dry_run:
            logger.info("dry-run 모드이므로 텔레그램 알림을 생략합니다.")
            return
        if executed_orders == 0:
            if failed_orders == 0:
                logger.info("실제 체결된 주문이 없어 텔레그램 알림을 생략합니다.")
                return
            if self.telegram.can_send():
                lines = [
                    f"날짜: {datetime.now().date()}",
                    f"총자산: {equity:,.0f}원",
                    f"계획 주문 수: {len(plans)}",
                    "실제 체결: 0건",
                    f"실패: {failed_orders}건",
                ]
                if abort_reason:
                    lines.append(f"중단 사유: {abort_reason}")
                if not_eligible_orders:
                    lines.append(f"자격요건 미충족: {not_eligible_orders}건")
                lines.append(f"리포트: {report_path.name}")
                self.telegram.send_message(format_alert("Multi Allocator PLUS (주문 실패)", lines))
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
        self.telegram.send_message(format_alert("Multi Allocator PLUS", lines))

    def run(self):
        self.load_market_data()
        if self.cache_only:
            logger.info("🗂️ 캐시 리프레시 전용 실행이 완료되었습니다. 트레이딩 루틴은 건너뜁니다.")
            return
        last_date, targets = self.compute_target_weights()
        account, holdings = self.fetch_account_snapshot()
        plans = self.build_order_plan(targets, account, holdings)
        if plans:
            logger.info("📋 주문 계획 (%s):", last_date.date())
            for plan in plans:
                logger.info(
                    "  %s %s주 @ %.0f원 (보유 %s주 → 목표 %s주)",
                    plan.action,
                    plan.quantity,
                    plan.est_price,
                    plan.current_qty,
                    plan.target_qty,
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
            close_series = df["close"].dropna()
            if not close_series.empty:
                prices[ticker] = float(close_series.iloc[-1])
        return prices

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if symbol is None:
            return ""
        return symbol.replace("A", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Multi Allocator PLUS 실거래 트레이더")
    parser.add_argument("--start-date", type=str, default=None, help="데이터 로드 시작일 (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="데이터 캐시 사용 안 함")
    parser.add_argument("--real", action="store_true", help="실거래 모드 (기본: 모의투자)")
    parser.add_argument("--dry-run", action="store_true", help="주문 미전송, 계획만 출력")
    parser.add_argument("--min-trade", type=int, default=200_000, help="최소 매매 금액 기준")
    parser.add_argument("--cache-only", action="store_true", help="캐시 업데이트만 수행하고 주문 단계 생략")
    args = parser.parse_args()

    trader = MultiAllocatorPlusTrader(
        start_date=args.start_date,
        use_cache=not args.no_cache,
        dry_run=args.dry_run,
        virtual_account=not args.real,
        min_trade_value=args.min_trade,
        cache_only=args.cache_only,
    )
    trader.run()


if __name__ == "__main__":
    main()
