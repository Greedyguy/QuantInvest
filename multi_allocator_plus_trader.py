#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi Allocator PLUS 실거래 트레이더
cleaned_stock_proj의 hybrid_portfolio_trader 패턴을 참고하여
multi_allocator_plus 전략 목표 비중을 계산하고 한국투자증권 API로 주문 계획 생성
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import pandas as pd

from reports import load_data
from strategies import get_strategy
from universe_filter import filter_universe
from automation.telegram_notifier import TelegramNotifier, format_alert
from automation.daily_reporter import DailyReporter
from config import BLOCKED_TICKERS
from data_loader import get_universe_us, load_panel_us, get_index_close
from signals import compute_indicators, add_rel_strength

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
        cash_policy: str = "preserve",
        signal_mode: str = "live",
        signal_snapshot: str | None = None,
        prepare_signal_only: bool = False,
        execution_recheck: bool = True,
        recheck_price_band_pct: float = 3.0,
        market: str = "kr",
        us_universe_limit: int = 50,
    ):
        self.start_date = start_date
        self.use_cache = use_cache
        self.dry_run = dry_run
        self.virtual_account = virtual_account
        self.min_trade_value = min_trade_value
        self.cache_only = cache_only
        self.cash_policy = cash_policy
        self.signal_mode = signal_mode
        self.signal_snapshot = signal_snapshot
        self.prepare_signal_only = prepare_signal_only
        self.execution_recheck = execution_recheck
        self.recheck_price_band_pct = recheck_price_band_pct
        self.market = market.lower().strip()
        self.us_universe_limit = us_universe_limit
        self.run_id = uuid4().hex[:12]
        self.execution_log_path = self._execution_log_path()

        self.kis = KoreaInvestmentConnector(virtual_account=virtual_account)
        self.telegram = TelegramNotifier()
        self.reporter = DailyReporter(PROJECT_ROOT / "reports" / "daily")
        self.strategy = get_strategy("multi_allocator_plus_no_etf")
        if self.strategy is None:
            raise RuntimeError("multi_allocator_plus 전략을 찾을 수 없습니다.")

        self.enriched = {}
        self.market_index = None

    def _signal_snapshot_path(self, signal_date: datetime | pd.Timestamp | None = None) -> Path:
        out_dir = PROJECT_ROOT / "reports" / "signals"
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.signal_snapshot:
            return Path(self.signal_snapshot)
        mkt = self.market.lower()
        if signal_date is not None:
            dt = signal_date.date() if hasattr(signal_date, "date") else signal_date
            return out_dir / f"signal_{mkt}_{dt.isoformat()}.json"
        # 로드 시점(eod_fixed): 오늘 파일 → 가장 최근 파일 순으로 fallback
        today_path = out_dir / f"signal_{mkt}_{datetime.now().date().isoformat()}.json"
        if today_path.exists():
            return today_path
        existing = sorted(out_dir.glob(f"signal_{mkt}_*.json"))
        if existing:
            return existing[-1]
        return today_path

    def _execution_log_path(self) -> Path:
        out_dir = PROJECT_ROOT / "reports" / "execution"
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "A_live" if self.signal_mode == "live" else "B_eod_fixed"
        mkt = getattr(self, "market", "kr")
        return out_dir / f"execution_{datetime.now().date().isoformat()}_{mkt}_{mode}.jsonl"

    def load_market_data(self):
        if self.market == "us":
            self._load_market_data_us()
        else:
            self._load_market_data_kr()

    def _load_market_data_kr(self):
        enriched, idx_map = load_data(
            use_cache=self.use_cache,
            start_date=self.start_date,
        )
        self.enriched = enriched
        self.market_index = idx_map.get("KOSDAQ")
        universe = filter_universe(enriched)
        logger.info("✅ [KR] 데이터 로드 완료 - 유니버스 %d개", len(universe))

    def _load_market_data_us(self):
        start = self.start_date or "2020-01-01"
        from datetime import date as _date
        end = _date.today().strftime("%Y-%m-%d")
        universe = get_universe_us(limit=self.us_universe_limit)
        logger.info("🌐 [US] 유니버스 %d개 로드 중 ...", len(universe))
        panel = load_panel_us(universe, start, end, max_workers=6)
        idx = get_index_close("US", start, end)
        enriched = {}
        for ticker, df in panel.items():
            df = compute_indicators(df)
            if df is None or df.empty:
                continue
            df = add_rel_strength(df, idx)
            enriched[ticker] = df
        self.enriched = enriched
        self.market_index = idx
        logger.info("✅ [US] 데이터 로드 완료 - %d개 종목", len(enriched))

    def compute_target_weights(self) -> Tuple[pd.Timestamp, pd.Series]:
        targets = self.strategy.compute_security_targets(
            self.enriched,
            market_index=self.market_index,
            silent=True,
        )
        if targets is None or targets.empty:
            raise RuntimeError("타깃 비중 계산 실패")
        latest_date = targets.index.max()
        latest_row = targets.loc[latest_date].fillna(0.0)
        latest_row = latest_row[latest_row >= 0]
        # __CASH__ 비중은 리스크-오프 신호를 살리기 위해 보존하지만,
        # 자산 측 weight==0 종목은 drop 해야 보유 중일 때 매도 주문이 정상 생성된다.
        # (build_order_plan의 고아 정리 루프가 target_set 비교를 하므로 weight=0이 남아 있으면 매도가 누락됨)
        cash_weight = float(latest_row.get("__CASH__", 0.0))
        asset_row = latest_row.drop("__CASH__", errors="ignore")
        asset_row = asset_row[asset_row > 0].sort_values(ascending=False)
        latest_row = pd.concat([asset_row, pd.Series({"__CASH__": cash_weight})])
        logger.info("🎯 타깃 비중 산출 완료 (%s)", latest_date.date())
        logger.info("  __CASH__ -> %.2f%%", cash_weight * 100)
        for ticker, weight in asset_row.items():
            logger.info("  %s -> %.2f%%", ticker, weight * 100)
        return latest_date, latest_row

    def save_signal_snapshot(self, signal_date: pd.Timestamp, targets: pd.Series):
        ref_prices = self._latest_prices([ticker for ticker in targets.index if ticker != "__CASH__"])
        payload = {
            "signal_date": str(signal_date.date()),
            "strategy": "multi_allocator_plus_no_etf",
            "signal_mode": "eod_fixed",
            "targets": {k: float(v) for k, v in targets.fillna(0.0).items()},
            "ref_prices": {k: float(v) for k, v in ref_prices.items()},
            "meta": {
                "start_date": self.start_date,
                "generated_at": datetime.now().isoformat(),
            },
        }
        snapshot_path = self._signal_snapshot_path(signal_date)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("🧾 신호 스냅샷 저장: %s", snapshot_path)
        return snapshot_path

    def load_signal_snapshot(self) -> Tuple[pd.Timestamp, pd.Series, Dict[str, float]]:
        snapshot_path = self._signal_snapshot_path()
        if not snapshot_path.exists():
            raise FileNotFoundError(f"신호 스냅샷 파일이 없습니다: {snapshot_path}")
        with open(snapshot_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        signal_date = pd.to_datetime(payload.get("signal_date"))
        targets = pd.Series(payload.get("targets", {}), dtype=float).fillna(0.0)
        # 자산 측 weight==0 종목은 compute_target_weights와 동일하게 drop (cash는 보존).
        # 과거 스냅샷 호환을 위해 load 시점에서도 가드.
        cash_weight = float(targets.get("__CASH__", 0.0))
        asset_targets = targets.drop("__CASH__", errors="ignore")
        asset_targets = asset_targets[asset_targets > 0]
        targets = pd.concat([asset_targets, pd.Series({"__CASH__": cash_weight})])
        ref_prices = {k: float(v) for k, v in (payload.get("ref_prices") or {}).items()}
        logger.info("🧾 신호 스냅샷 로드: %s (date=%s)", snapshot_path, signal_date.date())
        return signal_date, targets, ref_prices

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
        price_cache_override: Dict[str, float] | None = None,
    ) -> List[OrderPlan]:
        targets = targets.copy()
        raw_cash_weight = float(targets.get("__CASH__", 0.0))
        targets = targets.drop("__CASH__", errors="ignore")

        # 운영 안정성을 위해 기존 동작(100% 재정규화)과 개선 동작(현금 비중 유지)을 선택 가능하게 둔다.
        if self.cash_policy == "legacy_renorm":
            positive = targets[targets > 0]
            total = positive.sum()
            targets = (positive / total) if total > 0 else positive
            cash_weight = 0.0
        else:
            cash_weight = max(raw_cash_weight, 0.0)

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
        price_cache = price_cache_override or self._latest_prices(targets.index)
        logger.info("💼 현금 정책: %s / 목표 현금 비중: %.2f%%", self.cash_policy, cash_weight * 100)

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

    def _safe_get_current_price(self, symbol: str) -> float | None:
        code = self._normalize_symbol(symbol)
        if self.dry_run:
            return None
        method_names = [
            "get_current_price",
            "get_stock_price",
            "get_price",
            "get_quote",
        ]
        for name in method_names:
            method = getattr(self.kis, name, None)
            if method is None:
                continue
            try:
                raw = method(code)
                if isinstance(raw, (int, float)):
                    return float(raw)
                if isinstance(raw, dict):
                    for key in ["current_price", "price", "stck_prpr"]:
                        val = raw.get(key)
                        if val is not None:
                            return float(val)
            except Exception:
                continue
        return None

    def _safe_get_orderable_qty(self, symbol: str, price: float) -> int | None:
        code = self._normalize_symbol(symbol)
        if self.dry_run:
            return None
        method_names = [
            "get_orderable_quantity",
            "get_available_buy_qty",
            "get_buyable_quantity",
        ]
        for name in method_names:
            method = getattr(self.kis, name, None)
            if method is None:
                continue
            try:
                raw = method(code, price)
                if isinstance(raw, int):
                    return raw
                if isinstance(raw, dict):
                    for key in ["orderable_qty", "quantity", "qty"]:
                        val = raw.get(key)
                        if val is not None:
                            return int(val)
            except Exception:
                continue
        return None

    def apply_execution_recheck(self, plans: List[OrderPlan], account: Dict) -> Tuple[List[OrderPlan], List[Dict]]:
        if not self.execution_recheck:
            return plans, []
        reviewed: List[OrderPlan] = []
        logs: List[Dict] = []
        remaining_cash = float(account.get("available_cash", 0) or 0)
        for plan in plans:
            recheck_price = self._safe_get_current_price(plan.symbol) or plan.est_price
            if recheck_price <= 0:
                logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "decision": "skip",
                    "reason": "invalid_recheck_price",
                })
                continue
            price_diff_pct = abs(recheck_price - plan.est_price) / plan.est_price * 100 if plan.est_price > 0 else 0.0
            if price_diff_pct > self.recheck_price_band_pct:
                logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "decision": "skip",
                    "reason": "price_band_exceeded",
                    "signal_price": plan.est_price,
                    "recheck_price": recheck_price,
                    "price_diff_pct": price_diff_pct,
                })
                continue

            adjusted_qty = int(plan.quantity)
            orderable_qty = self._safe_get_orderable_qty(plan.symbol, recheck_price)
            if orderable_qty is not None:
                adjusted_qty = min(adjusted_qty, max(orderable_qty, 0))
            if plan.action == "BUY":
                max_by_cash = int(remaining_cash / recheck_price) if recheck_price > 0 else 0
                adjusted_qty = min(adjusted_qty, max(max_by_cash, 0))
                remaining_cash -= adjusted_qty * recheck_price

            if adjusted_qty <= 0:
                logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "decision": "skip",
                    "reason": "insufficient_cash_or_orderable",
                    "signal_price": plan.est_price,
                    "recheck_price": recheck_price,
                    "orderable_qty": orderable_qty,
                })
                continue

            reviewed.append(
                OrderPlan(
                    symbol=plan.symbol,
                    action=plan.action,
                    quantity=adjusted_qty,
                    est_price=recheck_price,
                    est_value=adjusted_qty * recheck_price,
                    target_weight=plan.target_weight,
                    current_qty=plan.current_qty,
                    target_qty=plan.target_qty,
                )
            )
            logs.append({
                "run_id": self.run_id,
                "ticker": plan.symbol,
                "decision": "send",
                "reason": "ok",
                "signal_price": plan.est_price,
                "recheck_price": recheck_price,
                "planned_qty": plan.quantity,
                "final_qty": adjusted_qty,
                "orderable_qty": orderable_qty,
            })
        return reviewed, logs

    def append_execution_logs(self, logs: List[Dict]):
        if not logs:
            return
        with open(self.execution_log_path, "a", encoding="utf-8") as f:
            for row in logs:
                row["timestamp"] = datetime.now().isoformat()
                row["signal_mode"] = self.signal_mode
                row["dry_run"] = self.dry_run
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def execute(self, plans: List[OrderPlan], account: Dict, holdings: Dict, as_of: datetime):
        if not plans:
            logger.info("🚫 실행할 주문이 없습니다.")
            return
        account_no = self.kis.account
        executed_orders = 0
        failed_orders = 0
        not_eligible_orders = 0
        abort_reason: str | None = None
        order_logs: List[Dict] = []
        for plan in plans:
            logger.info(
                "➡️ %s %s x %s (목표 %.2f%%)",
                plan.action,
                plan.symbol,
                plan.quantity,
                plan.target_weight * 100,
            )
            if self.dry_run:
                order_logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "action": plan.action,
                    "decision": "dry_run",
                    "reason": "no_order_sent",
                    "final_qty": int(plan.quantity),
                    "price": float(plan.est_price),
                })
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
                    order_logs.append({
                        "run_id": self.run_id,
                        "ticker": plan.symbol,
                        "action": plan.action,
                        "decision": "sent",
                        "reason": "ok",
                        "final_qty": int(plan.quantity),
                        "price": float(plan.est_price),
                        "order_result_code": int(result),
                    })
                else:
                    failed_orders += 1
                    order_logs.append({
                        "run_id": self.run_id,
                        "ticker": plan.symbol,
                        "action": plan.action,
                        "decision": "failed",
                        "reason": "send_order_nonzero",
                        "final_qty": int(plan.quantity),
                        "price": float(plan.est_price),
                        "order_result_code": int(result),
                    })
                    if result == ORDER_ERR_MARKET_OPERATION_DATE_MISMATCH:
                        abort_reason = "휴장/영업일 불일치(KIS: 장운영일자≠주문일)"
                        break
                    if result == ORDER_ERR_ACCOUNT_NOT_ELIGIBLE:
                        not_eligible_orders += 1
            except Exception as exc:
                logger.error("주문 실패: %s (%s)", plan.symbol, exc)
                failed_orders += 1
                order_logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "action": plan.action,
                    "decision": "failed",
                    "reason": str(exc),
                    "final_qty": int(plan.quantity),
                    "price": float(plan.est_price),
                })
        equity = account.get("total_value") or (
            account.get("available_cash", 0) + account.get("stock_value", 0)
        )
        if self.dry_run and (not equity or equity <= 0):
            equity = 1_000_000
        snapshot = {"account": account, "holdings": list(holdings.values())}
        report_path = self.reporter.save_report(as_of, equity, [plan.__dict__ for plan in plans], snapshot)
        self.append_execution_logs(order_logs)
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
        price_cache_override = None
        if self.signal_mode == "live":
            self.load_market_data()
            last_date, targets = self.compute_target_weights()
            snapshot_path = self.save_signal_snapshot(last_date, targets)
            if self.prepare_signal_only:
                logger.info("🧾 신호 준비 전용 실행 완료: %s", snapshot_path)
                return
        else:
            last_date, targets, ref_prices = self.load_signal_snapshot()
            price_cache_override = ref_prices
            if self.prepare_signal_only:
                logger.warning("eod_fixed 모드에서는 --prepare-signal-only를 무시합니다.")
        if self.cache_only:
            logger.info("🗂️ 캐시 리프레시 전용 실행이 완료되었습니다. 트레이딩 루틴은 건너뜁니다.")
            return
        account, holdings = self.fetch_account_snapshot()
        plans = self.build_order_plan(targets, account, holdings, price_cache_override=price_cache_override)
        plans, recheck_logs = self.apply_execution_recheck(plans, account)
        self.append_execution_logs(recheck_logs)
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

    def _normalize_symbol(self, symbol: str) -> str:
        if symbol is None:
            return ""
        s = symbol.strip()
        if getattr(self, "market", "kr") == "us":
            return s
        # KR: pykrx 접두사 'A' 제거 (예: 'A005930' → '005930')
        if s.startswith("A") and s[1:].isdigit():
            return s[1:]
        return s


def main():
    parser = argparse.ArgumentParser(description="Multi Allocator PLUS 실거래 트레이더")
    parser.add_argument("--start-date", type=str, default=None, help="데이터 로드 시작일 (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="데이터 캐시 사용 안 함")
    parser.add_argument("--real", action="store_true", help="실거래 모드 (기본: 모의투자)")
    parser.add_argument("--dry-run", action="store_true", help="주문 미전송, 계획만 출력")
    parser.add_argument("--min-trade", type=int, default=200_000, help="최소 매매 금액 기준")
    parser.add_argument("--cache-only", action="store_true", help="캐시 업데이트만 수행하고 주문 단계 생략")
    parser.add_argument(
        "--cash-policy",
        type=str,
        default="preserve",
        choices=["preserve", "legacy_renorm"],
        help="현금 비중 처리 방식 (preserve: 전략 현금 비중 유지, legacy_renorm: 기존 100%% 재정규화)",
    )
    parser.add_argument(
        "--signal-mode",
        type=str,
        default="live",
        choices=["live", "eod_fixed"],
        help="신호 생성 방식 (live: 실행 시 계산, eod_fixed: 저장된 전일 신호 사용)",
    )
    parser.add_argument("--signal-snapshot", type=str, default=None, help="신호 스냅샷 파일 경로(JSON)")
    parser.add_argument("--prepare-signal-only", action="store_true", help="신호 스냅샷만 생성하고 주문 단계 생략")
    parser.add_argument(
        "--execution-recheck",
        dest="execution_recheck",
        action="store_true",
        help="주문 직전 가격/현금/주문가능수량 재검증 활성화",
    )
    parser.add_argument(
        "--no-execution-recheck",
        dest="execution_recheck",
        action="store_false",
        help="주문 직전 재검증 비활성화",
    )
    parser.set_defaults(execution_recheck=True)
    parser.add_argument(
        "--recheck-price-band-pct",
        type=float,
        default=3.0,
        help="신호 기준가 대비 허용 가격 괴리(%%)",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="kr",
        choices=["kr", "us"],
        help="마켓 선택 (kr: 한국, us: 미국)",
    )
    parser.add_argument(
        "--us-universe-limit",
        type=int,
        default=50,
        help="US 유니버스 종목 수",
    )
    args = parser.parse_args()

    trader = MultiAllocatorPlusTrader(
        start_date=args.start_date,
        use_cache=not args.no_cache,
        dry_run=args.dry_run,
        virtual_account=not args.real,
        min_trade_value=args.min_trade,
        cache_only=args.cache_only,
        cash_policy=args.cash_policy,
        signal_mode=args.signal_mode,
        signal_snapshot=args.signal_snapshot,
        prepare_signal_only=args.prepare_signal_only,
        execution_recheck=args.execution_recheck,
        recheck_price_band_pct=args.recheck_price_band_pct,
        market=args.market,
        us_universe_limit=args.us_universe_limit,
    )
    trader.run()


if __name__ == "__main__":
    main()
