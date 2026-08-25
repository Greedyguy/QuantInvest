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
import subprocess
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
from config import (
    BLOCKED_TICKERS,
    FEE_PER_SIDE,
    FEE_PER_SIDE_US,
    SLIPPAGE_ENTRY,
    SLIPPAGE_ENTRY_US,
    TAX_RATE_SELL,
    US_TAX_RATE_SELL,
)
from data_loader import (
    get_universe_us,
    load_panel_us,
    get_index_close,
    validate_market_data_freshness,
)
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


INDEX_ETF_TICKERS = {
    "069500", "102110", "152100", "229200", "091160", "091180",
    "305720", "233740", "114800", "122630",
}

SENSITIVE_REPORT_KEYS = {
    "account", "account_no", "available_cash", "total_cash", "total_value",
    "stock_value", "equity", "market_value", "avg_price", "unrealized_pnl",
    "est_value", "cash_before", "cash_after",
}


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
        skip_if_executed: bool = True,
        small_account_shadow: bool = False,
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
        self.skip_if_executed = skip_if_executed
        self.small_account_shadow = small_account_shadow
        self.run_id = uuid4().hex[:12]
        self.execution_log_path = self._execution_log_path()

        self.kis = KoreaInvestmentConnector(virtual_account=virtual_account)
        self.telegram = TelegramNotifier()
        self.reporter = DailyReporter(PROJECT_ROOT / "reports" / "daily")
        self.strategy = get_strategy("multi_allocator_plus_safe_etf_kqm")
        if self.strategy is None:
            raise RuntimeError("multi_allocator_plus 전략을 찾을 수 없습니다.")

        self.enriched = {}
        self.market_index = None
        self.secondary_index = None
        self.loaded_signal_snapshot_payload = None

    def _shadow_report_path(self, signal_date: datetime | pd.Timestamp) -> Path:
        out_dir = PROJECT_ROOT / "reports" / "shadow"
        out_dir.mkdir(parents=True, exist_ok=True)
        sig_day = signal_date.date() if hasattr(signal_date, "date") else signal_date
        return out_dir / (
            f"small_account_{datetime.now().date().isoformat()}_"
            f"{self.market}_{sig_day.isoformat()}_{self.run_id}.json"
        )

    @staticmethod
    def _git_revision() -> str | None:
        github_sha = os.getenv("GITHUB_SHA")
        if github_sha:
            return github_sha
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

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

    def _execution_summary_path(self, signal_date: datetime | pd.Timestamp) -> Path:
        out_dir = PROJECT_ROOT / "reports" / "execution"
        out_dir.mkdir(parents=True, exist_ok=True)
        sig_day = signal_date.date() if hasattr(signal_date, "date") else signal_date
        trade_day = datetime.now().date()
        return out_dir / (
            f"summary_{trade_day.isoformat()}_{self.market}_"
            f"{sig_day.isoformat()}_{self.run_id}.json"
        )

    def _completed_execution_for_signal(self, signal_date: datetime | pd.Timestamp) -> Dict | None:
        if self.dry_run or not self.skip_if_executed or self.signal_mode != "eod_fixed":
            return None
        sig_day = signal_date.date() if hasattr(signal_date, "date") else signal_date
        trade_day = datetime.now().date().isoformat()
        pattern = f"summary_{trade_day}_{self.market}_{sig_day.isoformat()}_*.json"
        for path in sorted((PROJECT_ROOT / "reports" / "execution").glob(pattern)):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if payload.get("completed_for_signal") and not payload.get("dry_run"):
                return payload
        return None

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
        self.secondary_index = idx_map.get("KOSPI")
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
            secondary_index=self.secondary_index,
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
        validate_market_data_freshness(self.market_index, latest_date, "primary index")
        if self.secondary_index is not None and not self.secondary_index.empty:
            validate_market_data_freshness(self.secondary_index, latest_date, "secondary index")
        for ticker in asset_row.index:
            validate_market_data_freshness(
                self.enriched.get(ticker), latest_date, f"target security {ticker}"
            )
        logger.info("🎯 타깃 비중 산출 완료 (%s)", latest_date.date())
        logger.info("  __CASH__ -> %.2f%%", cash_weight * 100)
        for ticker, weight in asset_row.items():
            logger.info("  %s -> %.2f%%", ticker, weight * 100)
        return latest_date, latest_row

    @staticmethod
    def _json_scalar(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float, str)):
            if isinstance(value, float) and not pd.notna(value):
                return None
            return value
        if hasattr(value, "item"):
            try:
                return MultiAllocatorPlusTrader._json_scalar(value.item())
            except Exception:
                return str(value)
        return str(value)

    def _row_asof(self, frame, as_of):
        if frame is None or getattr(frame, "empty", True):
            return None
        ts = pd.to_datetime(as_of)
        valid = frame.index[frame.index <= ts]
        if len(valid) == 0:
            return None
        return frame.loc[valid.max()]

    def _series_value_asof(self, series, as_of):
        if series is None or getattr(series, "empty", True):
            return None
        ts = pd.to_datetime(as_of)
        valid = series.index[series.index <= ts]
        if len(valid) == 0:
            return None
        return self._json_scalar(series.loc[valid.max()])

    def _data_as_of(self, targets: pd.Series | None = None) -> Dict:
        fallback = ((self.loaded_signal_snapshot_payload or {}).get("meta") or {}).get("data_as_of")
        if not self.enriched and fallback:
            return fallback

        def last_date(frame):
            if frame is None or getattr(frame, "empty", True):
                return None
            return pd.to_datetime(frame.index.max()).date().isoformat()

        selected = []
        if targets is not None:
            selected = [str(t) for t in targets.index if str(t) != "__CASH__"]
        target_dates = {
            ticker: last_date(self.enriched.get(ticker))
            for ticker in selected
            if self.enriched.get(ticker) is not None
        }
        universe_dates = [
            pd.to_datetime(df.index.max())
            for df in self.enriched.values()
            if df is not None and not df.empty
        ]
        return {
            "primary_index": last_date(self.market_index),
            "secondary_index": last_date(self.secondary_index),
            "universe_latest_min": min(universe_dates).date().isoformat() if universe_dates else None,
            "universe_latest_max": max(universe_dates).date().isoformat() if universe_dates else None,
            "target_securities": target_dates,
        }

    def _build_decision_context(self, signal_date: pd.Timestamp) -> Dict:
        fallback = (self.loaded_signal_snapshot_payload or {}).get("decision_context")
        regime = getattr(self.strategy, "latest_regime_context", None)
        if (regime is None or getattr(regime, "empty", True)) and fallback:
            return fallback

        regime_row = self._row_asof(regime, signal_date)
        regime_payload = {}
        if regime_row is not None:
            for key in ["regime", "close", "ma60", "mom5", "mom20", "mom1m", "mom3m", "vol20"]:
                if key in regime_row.index:
                    regime_payload[key] = self._json_scalar(regime_row.get(key))
        return {
            "signal_date": str(signal_date.date()),
            "regime": regime_payload,
            "exposure": {
                "base": self._series_value_asof(
                    getattr(self.strategy, "latest_base_exposure", None), signal_date
                ),
                "after_stress": self._series_value_asof(
                    getattr(self.strategy, "latest_stress_exposure", None), signal_date
                ),
                "final": self._series_value_asof(
                    getattr(self.strategy, "latest_final_exposure", None), signal_date
                ),
                "stress_level": self._series_value_asof(
                    getattr(self.strategy, "latest_stress_levels", None), signal_date
                ),
                "fast_signal": self._series_value_asof(
                    getattr(self.strategy, "latest_fast_signal", None), signal_date
                ),
            },
        }

    def _build_style_attribution(self, signal_date: pd.Timestamp, targets: pd.Series) -> Dict:
        fallback = (self.loaded_signal_snapshot_payload or {}).get("style_attribution")
        style_context = getattr(self.strategy, "latest_style_context", None)
        strategy_weights = getattr(self.strategy, "latest_target_weights", None)
        security_style_map = getattr(self.strategy, "latest_security_style_map", {}) or {}

        if (
            fallback
            and (style_context is None or getattr(style_context, "empty", True))
            and (strategy_weights is None or getattr(strategy_weights, "empty", True))
        ):
            return fallback

        style_row = self._row_asof(style_context, signal_date)
        strategy_row = self._row_asof(strategy_weights, signal_date)

        style_payload = {}
        if style_row is not None:
            for key, value in style_row.to_dict().items():
                style_payload[key] = self._json_scalar(value)

        strategy_payload = {}
        if strategy_row is not None:
            strategy_payload = {
                str(k): float(v)
                for k, v in strategy_row.fillna(0.0).sort_values(ascending=False).items()
            }

        target_rows = []
        style_sums: Dict[str, float] = {}
        for ticker, weight in targets.drop("__CASH__", errors="ignore").fillna(0.0).items():
            if weight <= 0:
                continue
            info = dict(security_style_map.get(ticker, {}))
            if not info:
                df = self.enriched.get(ticker)
                price = None
                market_cap = None
                avg_value20 = None
                if df is not None and not df.empty:
                    if "close" in df.columns:
                        close = pd.to_numeric(df["close"], errors="coerce").dropna()
                        price = float(close.iloc[-1]) if not close.empty else None
                    if "market_cap" in df.columns:
                        mcap = pd.to_numeric(df["market_cap"], errors="coerce").dropna()
                        market_cap = float(mcap.iloc[-1]) if not mcap.empty else None
                    if "value" in df.columns:
                        value = pd.to_numeric(df["value"], errors="coerce").tail(20).dropna()
                        avg_value20 = float(value.mean()) if not value.empty else None
                info = {
                    "style": "unknown",
                    "price": price,
                    "market_cap": market_cap,
                    "avg_value20": avg_value20,
                }
            style = str(info.get("style") or "unknown")
            style_sums[style] = style_sums.get(style, 0.0) + float(weight)
            target_rows.append({
                "ticker": str(ticker),
                "weight": float(weight),
                "style": style,
                "price": self._json_scalar(info.get("price")),
                "market_cap": self._json_scalar(info.get("market_cap")),
                "avg_value20": self._json_scalar(info.get("avg_value20")),
            })

        target_rows.sort(key=lambda row: row["weight"], reverse=True)
        return {
            "signal_date": str(signal_date.date() if hasattr(signal_date, "date") else signal_date),
            "style_context": style_payload,
            "strategy_weights": strategy_payload,
            "target_style_sums": {k: float(v) for k, v in sorted(style_sums.items())},
            "cash_weight": float(targets.get("__CASH__", 0.0)),
            "targets": target_rows,
        }

    def save_signal_snapshot(self, signal_date: pd.Timestamp, targets: pd.Series):
        ref_prices = self._latest_prices([ticker for ticker in targets.index if ticker != "__CASH__"])
        strategy_name = (
            self.strategy.get_name()
            if hasattr(self.strategy, "get_name")
            else self.strategy.__class__.__name__
        )
        payload = {
            "signal_date": str(signal_date.date()),
            "strategy": strategy_name,
            "signal_mode": "eod_fixed",
            "targets": {k: float(v) for k, v in targets.fillna(0.0).items()},
            "ref_prices": {k: float(v) for k, v in ref_prices.items()},
            "style_attribution": self._build_style_attribution(signal_date, targets),
            "decision_context": self._build_decision_context(signal_date),
            "meta": {
                "start_date": self.start_date,
                "generated_at": datetime.now().isoformat(),
                "git_revision": self._git_revision(),
                "data_as_of": self._data_as_of(targets),
                "max_security_weight": getattr(self.strategy, "max_security_weight", None),
                "target_turnover_cap": getattr(self.strategy, "target_turnover_cap", None),
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
        self.loaded_signal_snapshot_payload = payload
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

    def fetch_account_snapshot(self, force_live_read: bool = False) -> Tuple[Dict, Dict]:
        # dry-run 모드에서는 KIS API를 부르지 않고 가상 계좌 스냅샷 사용
        if self.dry_run and not force_live_read:
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

        if force_live_read:
            logger.info("🔎 shadow 비교용 실제 계좌 스냅샷 조회 (주문 전송 없음)")

        balance_raw = self.kis.get_account_balance()
        if not balance_raw:
            raise RuntimeError("계좌 잔고 조회 실패: 빈 응답을 받아 실거래를 중단합니다.")

        account = self.kis.parse_account_balance_data(balance_raw)
        if not account or not account.get("total_value"):
            raise RuntimeError("계좌 잔고 파싱 실패: 총자산을 확인할 수 없어 실거래를 중단합니다.")

        holdings = self._parse_holdings_from_balance(balance_raw)
        stock_value = float(account.get("stock_value", 0) or 0)
        if stock_value > 1_000 and not holdings:
            raise RuntimeError(
                "보유 종목 조회 실패: 주식 평가금액은 있으나 보유 목록이 비어 있어 실거래를 중단합니다."
            )
        allocation = self._account_allocation(account)
        logger.info(
            "💰 계좌 스냅샷 로드 완료 (현금 %.1f%% / 주식 %.1f%%)",
            float(allocation.get("cash_weight") or 0) * 100,
            float(allocation.get("stock_weight") or 0) * 100,
        )
        if holdings:
            holding_summary = ", ".join(
                f"{symbol}:{int(pos.get('quantity', 0))}주"
                for symbol, pos in sorted(holdings.items())
            )
            logger.info("📦 보유 종목: %s", holding_summary)
        else:
            logger.info("📦 보유 종목: 없음")
        return account, holdings

    def _parse_holdings_from_balance(self, balance_raw: Dict) -> Dict[str, Dict]:
        holdings: Dict[str, Dict] = {}
        for stock_data in balance_raw.get("output1") or []:
            quantity = self._safe_int(stock_data.get("hldg_qty", "0"))
            if quantity <= 0:
                continue

            symbol = self._normalize_symbol(stock_data.get("pdno", ""))
            if not symbol:
                continue

            current_price = self._safe_float(stock_data.get("prpr", "0"))
            avg_price = self._safe_float(stock_data.get("pchs_avg_pric", "0"))
            market_value = self._safe_float(stock_data.get("evlu_amt", "0"))
            if market_value <= 0 and current_price > 0:
                market_value = current_price * quantity
            unrealized_pnl = self._safe_float(stock_data.get("evlu_pfls_amt", "0"))
            invested = avg_price * quantity
            unrealized_pnl_rate = (
                unrealized_pnl / invested * 100
                if invested > 0
                else self._safe_float(stock_data.get("evlu_pfls_rt", "0"))
            )

            holdings[symbol] = {
                "symbol": symbol,
                "name": stock_data.get("prdt_name", ""),
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_rate": unrealized_pnl_rate,
                "purchase_date": stock_data.get("ord_dt", ""),
            }

        logger.info("📦 보유 종목 파싱 완료: %d개", len(holdings))
        return holdings

    @staticmethod
    def _safe_int(value) -> int:
        try:
            if value is None or value == "":
                return 0
            cleaned = "".join(c for c in str(value) if c.isdigit() or c in "-.")
            if not cleaned or cleaned in {".", "-", "-."}:
                return 0
            return int(float(cleaned))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value) -> float:
        try:
            if value is None or value == "":
                return 0.0
            cleaned = "".join(c for c in str(value) if c.isdigit() or c in "-.")
            if not cleaned or cleaned in {".", "-", "-."}:
                return 0.0
            return float(cleaned)
        except (TypeError, ValueError):
            return 0.0

    def build_order_plan(
        self,
        targets: pd.Series,
        account: Dict,
        holdings: Dict,
        price_cache_override: Dict[str, float] | None = None,
        min_trade_value_override: int | None = None,
        quantity_rounding: str = "floor",
        return_decisions: bool = False,
    ):
        targets = targets.copy()
        raw_cash_weight = float(targets.get("__CASH__", 0.0))
        targets = targets.drop("__CASH__", errors="ignore")
        decisions: List[Dict] = []

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
            blocked_targets = [
                ticker for ticker in targets.index
                if self._normalize_symbol(ticker) in blocked
            ]
            decisions.extend({
                "ticker": str(ticker),
                "action": "SKIP",
                "reason": "blocked_ticker",
                "target_weight": float(targets.get(ticker, 0.0)),
            } for ticker in blocked_targets)
            targets = targets.drop(
                blocked_targets,
                errors="ignore",
            )

        total_equity = account.get("total_value") or (
            account.get("available_cash", 0) + account.get("stock_value", 0)
        )
        # dry-run 모드에서는 최소 100만원 기준으로 수량 계산
        if self.dry_run and (not total_equity or total_equity <= 0):
            total_equity = 1_000_000
            logger.info("💡 dry-run 모드: 가상 초기 자본 1,000,000원으로 주문 수량 계산")

        effective_min_trade = (
            int(min_trade_value_override)
            if min_trade_value_override is not None
            else self.min_trade_value
        )
        if self.dry_run and min_trade_value_override is None:
            effective_min_trade = min(self.min_trade_value, 50_000)
        plans: List[OrderPlan] = []
        price_cache = price_cache_override or self._latest_prices(targets.index)
        logger.info("💼 현금 정책: %s / 목표 현금 비중: %.2f%%", self.cash_policy, cash_weight * 100)

        for ticker, weight in targets.items():
            price = price_cache.get(ticker)
            if price is None or price <= 0:
                logger.warning("계획 제외: %s 기준가 없음", ticker)
                decisions.append({
                    "ticker": str(ticker),
                    "action": "SKIP",
                    "reason": "invalid_reference_price",
                    "target_weight": float(weight),
                })
                continue
            target_value = total_equity * weight
            current_qty = holdings.get(self._normalize_symbol(ticker), {}).get("quantity", 0)
            required_buy_value = max(float(effective_min_trade), float(price))
            if current_qty <= 0 and target_value < required_buy_value:
                logger.info(
                    "계획 제외: %s 목표 %.2f%%가 신규매수 기준 미달 "
                    "(최소매매 %.0f원, 1주 %.0f원)",
                    ticker,
                    weight * 100,
                    effective_min_trade,
                    price,
                )
                decisions.append({
                    "ticker": str(ticker),
                    "action": "SKIP",
                    "reason": "new_position_below_minimum_or_one_share",
                    "target_weight": float(weight),
                    "current_qty": int(current_qty),
                    "target_qty": 0,
                    "reference_price": float(price),
                })
                continue
            if target_value < effective_min_trade:
                logger.info(
                    "계획 제외: %s 목표 %.2f%%가 최소매매 %.0f원 기준 미달",
                    ticker,
                    weight * 100,
                    effective_min_trade,
                )
                if current_qty > 0:
                    logger.info(
                        "계획 생성: SELL %s %s주 (목표금액이 최소매매 미만이라 잔량 정리)",
                        ticker,
                        current_qty,
                    )
                    plans.append(
                        OrderPlan(
                            symbol=ticker,
                            action="SELL",
                            quantity=current_qty,
                            est_price=price,
                            est_value=current_qty * price,
                            target_weight=0.0,
                            current_qty=current_qty,
                            target_qty=0,
                        )
                    )
                    decisions.append({
                        "ticker": str(ticker),
                        "action": "SELL",
                        "reason": "target_below_minimum_exit",
                        "target_weight": float(weight),
                        "current_qty": int(current_qty),
                        "target_qty": 0,
                        "quantity": int(current_qty),
                        "reference_price": float(price),
                    })
                else:
                    decisions.append({
                        "ticker": str(ticker),
                        "action": "SKIP",
                        "reason": "target_below_minimum",
                        "target_weight": float(weight),
                        "current_qty": 0,
                        "target_qty": 0,
                        "reference_price": float(price),
                    })
                continue
            raw_target_qty = target_value / price
            if (
                quantity_rounding == "nearest_etf"
                and self._normalize_symbol(ticker) in INDEX_ETF_TICKERS
            ):
                target_qty = int(raw_target_qty + 0.5)
            else:
                target_qty = int(raw_target_qty)
            delta = target_qty - current_qty
            if delta == 0:
                logger.info(
                    "계획 제외: %s 보유 %s주 = 목표 %s주 (목표 %.2f%%, 기준가 %.0f원)",
                    ticker,
                    current_qty,
                    target_qty,
                    weight * 100,
                    price,
                )
                decisions.append({
                    "ticker": str(ticker),
                    "action": "HOLD",
                    "reason": "quantity_already_at_target",
                    "target_weight": float(weight),
                    "current_qty": int(current_qty),
                    "target_qty": int(target_qty),
                    "reference_price": float(price),
                })
                continue
            action = "BUY" if delta > 0 else "SELL"
            logger.info(
                "계획 생성: %s %s %s주 (보유 %s주 -> 목표 %s주, 목표 %.2f%%)",
                action,
                ticker,
                abs(delta),
                current_qty,
                target_qty,
                weight * 100,
            )
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
            decisions.append({
                "ticker": str(ticker),
                "action": action,
                "reason": "planned",
                "target_weight": float(weight),
                "current_qty": int(current_qty),
                "target_qty": int(target_qty),
                "quantity": int(abs(delta)),
                "reference_price": float(price),
                "rounding": quantity_rounding,
            })

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
                decisions.append({
                    "ticker": str(symbol),
                    "action": "SELL",
                    "reason": "not_in_target_exit",
                    "target_weight": 0.0,
                    "current_qty": int(pos["quantity"]),
                    "target_qty": 0,
                    "quantity": int(pos["quantity"]),
                    "reference_price": float(price or 0),
                })

        plans.sort(key=lambda x: (-1 if x.action == "SELL" else 1, -x.est_value))
        if return_decisions:
            return plans, decisions
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
                    containers = [raw]
                    output = raw.get("output")
                    if isinstance(output, dict):
                        containers.append(output)
                    for data in containers:
                        for key in ["current_price", "price", "stck_prpr", "prpr"]:
                            price = self._safe_float(data.get(key))
                            if price > 0:
                                return price
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
                    containers = [raw]
                    output = raw.get("output")
                    if isinstance(output, dict):
                        containers.append(output)
                    for data in containers:
                        for key in ["orderable_qty", "quantity", "qty", "ord_psbl_qty"]:
                            val = data.get(key)
                            if val is not None:
                                return self._safe_int(val)
            except Exception:
                continue
        return None

    def _execution_cost_rates(self) -> Tuple[float, float, float]:
        if self.market == "us":
            return FEE_PER_SIDE_US, US_TAX_RATE_SELL, SLIPPAGE_ENTRY_US
        return FEE_PER_SIDE, TAX_RATE_SELL, SLIPPAGE_ENTRY

    def apply_execution_recheck(self, plans: List[OrderPlan], account: Dict) -> Tuple[List[OrderPlan], List[Dict]]:
        if not self.execution_recheck:
            return plans, []
        reviewed: List[OrderPlan] = []
        logs: List[Dict] = []
        remaining_cash = float(account.get("available_cash", 0) or 0)
        fee_rate, sell_tax_rate, entry_slippage = self._execution_cost_rates()
        for plan in plans:
            cash_before = remaining_cash
            recheck_price = self._safe_get_current_price(plan.symbol) or plan.est_price
            if recheck_price <= 0:
                logger.warning("재검증 제외: %s %s - 현재가 확인 실패", plan.action, plan.symbol)
                logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "action": plan.action,
                    "decision": "skip",
                    "reason": "invalid_recheck_price",
                    "cash_before": cash_before,
                })
                continue
            price_diff_pct = abs(recheck_price - plan.est_price) / plan.est_price * 100 if plan.est_price > 0 else 0.0
            price_band_bypassed = (
                plan.action == "SELL" and price_diff_pct > self.recheck_price_band_pct
            )
            if price_diff_pct > self.recheck_price_band_pct and plan.action == "BUY":
                logger.warning(
                    "재검증 제외: %s %s - 가격 괴리 %.2f%% > %.2f%% (신호 %.0f원, 현재 %.0f원)",
                    plan.action,
                    plan.symbol,
                    price_diff_pct,
                    self.recheck_price_band_pct,
                    plan.est_price,
                    recheck_price,
                )
                logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "action": plan.action,
                    "decision": "skip",
                    "reason": "price_band_exceeded",
                    "signal_price": plan.est_price,
                    "recheck_price": recheck_price,
                    "price_diff_pct": price_diff_pct,
                    "cash_before": cash_before,
                })
                continue
            if price_band_bypassed:
                logger.warning(
                    "매도 계속 진행: %s - 가격 괴리 %.2f%% > %.2f%%; "
                    "리스크 축소 주문은 가격 밴드로 차단하지 않음",
                    plan.symbol,
                    price_diff_pct,
                    self.recheck_price_band_pct,
                )

            adjusted_qty = int(plan.quantity)
            orderable_qty = None
            if plan.action == "BUY":
                orderable_qty = self._safe_get_orderable_qty(plan.symbol, recheck_price)
                if orderable_qty is not None:
                    adjusted_qty = min(adjusted_qty, max(orderable_qty, 0))
                cash_per_share = recheck_price * (1 + fee_rate + entry_slippage)
                max_by_cash = int(remaining_cash / cash_per_share) if cash_per_share > 0 else 0
                adjusted_qty = min(adjusted_qty, max(max_by_cash, 0))
                remaining_cash -= adjusted_qty * cash_per_share
            else:
                cash_per_share = recheck_price * max(1 - fee_rate - sell_tax_rate, 0)
                remaining_cash += adjusted_qty * cash_per_share

            if adjusted_qty <= 0:
                logger.warning(
                    "재검증 제외: %s %s - 현금/주문가능 부족 (계획 %s주)",
                    plan.action,
                    plan.symbol,
                    plan.quantity,
                )
                logs.append({
                    "run_id": self.run_id,
                    "ticker": plan.symbol,
                    "action": plan.action,
                    "decision": "skip",
                    "reason": "insufficient_cash_or_orderable",
                    "signal_price": plan.est_price,
                    "recheck_price": recheck_price,
                    "orderable_qty": orderable_qty,
                    "cash_before": cash_before,
                    "cash_after": remaining_cash,
                    "cash_per_share": cash_per_share,
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
            if adjusted_qty != int(plan.quantity):
                logger.info(
                    "재검증 수량 조정: %s %s %s주 -> %s주",
                    plan.action,
                    plan.symbol,
                    plan.quantity,
                    adjusted_qty,
                )
            logs.append({
                "run_id": self.run_id,
                "ticker": plan.symbol,
                "action": plan.action,
                "decision": "send",
                "reason": "sell_price_band_bypassed" if price_band_bypassed else "ok",
                "signal_price": plan.est_price,
                "recheck_price": recheck_price,
                "planned_qty": plan.quantity,
                "final_qty": adjusted_qty,
                "orderable_qty": orderable_qty,
                "cash_before": cash_before,
                "cash_after": remaining_cash,
                "cash_per_share": cash_per_share,
            })
        return reviewed, logs

    def _sanitize_report_value(self, value):
        if isinstance(value, dict):
            return {
                str(key): self._sanitize_report_value(item)
                for key, item in value.items()
                if str(key).lower() not in SENSITIVE_REPORT_KEYS
            }
        if isinstance(value, (list, tuple)):
            return [self._sanitize_report_value(item) for item in value]
        if isinstance(value, str):
            account_no = str(getattr(getattr(self, "kis", None), "account", "") or "")
            if account_no:
                value = value.replace(account_no, "[REDACTED_ACCOUNT]")
            return value
        return self._json_scalar(value)

    def _sanitized_plan(self, plan: OrderPlan) -> Dict:
        return {
            "symbol": str(plan.symbol),
            "action": str(plan.action),
            "quantity": int(plan.quantity),
            "est_price": float(plan.est_price),
            "target_weight": float(plan.target_weight),
            "current_qty": int(plan.current_qty),
            "target_qty": int(plan.target_qty),
        }

    def _sanitized_holdings(self, holdings: Dict | None) -> List[Dict]:
        rows = []
        for symbol, position in sorted((holdings or {}).items()):
            rows.append({
                "symbol": str(position.get("symbol") or symbol),
                "name": str(position.get("name") or ""),
                "quantity": int(position.get("quantity", 0) or 0),
                "unrealized_pnl_rate": self._json_scalar(
                    position.get("unrealized_pnl_rate")
                ),
            })
        return rows

    def _account_allocation(self, account: Dict | None) -> Dict:
        account = account or {}
        total = float(account.get("total_value", 0) or 0)
        if total <= 0:
            total = float(account.get("available_cash", 0) or 0) + float(
                account.get("stock_value", 0) or 0
            )
        has_stock_value = account.get("stock_value") is not None
        stock = float(account.get("stock_value", 0) or 0)
        if account.get("total_cash") is not None:
            cash = float(account.get("total_cash", 0) or 0)
        elif total > 0 and stock > 0:
            cash = max(total - stock, 0.0)
        else:
            cash = float(account.get("available_cash", 0) or 0)
        return {
            "cash_weight": cash / total if total > 0 else None,
            "stock_weight": stock / total if total > 0 and has_stock_value else None,
        }

    def _exposure_diagnostics(
        self,
        targets: pd.Series,
        account: Dict,
        holdings: Dict,
        plans: List[OrderPlan],
        price_cache_override: Dict[str, float] | None = None,
        planning_decisions: List[Dict] | None = None,
    ) -> Dict:
        total = float(account.get("total_value", 0) or 0)
        if total <= 0:
            total = float(account.get("available_cash", 0) or 0) + float(
                account.get("stock_value", 0) or 0
            )
        if total <= 0:
            return {
                "status": "unavailable",
                "reason": "total_equity_unavailable",
            }

        target_assets = targets.drop("__CASH__", errors="ignore").fillna(0.0)
        target_assets = target_assets[target_assets > 0]
        target_map = {
            self._normalize_symbol(str(ticker)): float(weight)
            for ticker, weight in target_assets.items()
        }
        prices = {
            self._normalize_symbol(str(ticker)): float(price)
            for ticker, price in (price_cache_override or {}).items()
            if price is not None and float(price) > 0
        }
        for plan in plans:
            if plan.est_price > 0:
                prices[self._normalize_symbol(plan.symbol)] = float(plan.est_price)

        current_qty = {}
        current_values = {}
        for symbol, position in (holdings or {}).items():
            code = self._normalize_symbol(symbol)
            qty = int(position.get("quantity", 0) or 0)
            price = float(position.get("current_price", 0) or prices.get(code, 0) or 0)
            market_value = float(position.get("market_value", 0) or 0)
            if market_value <= 0 and price > 0:
                market_value = qty * price
            current_qty[code] = qty
            current_values[code] = market_value
            if price > 0:
                prices[code] = price

        projected_qty = dict(current_qty)
        for plan in plans:
            code = self._normalize_symbol(plan.symbol)
            before = projected_qty.get(code, 0)
            projected_qty[code] = max(
                before + plan.quantity if plan.action == "BUY" else before - plan.quantity,
                0,
            )

        all_symbols = sorted(set(target_map) | set(current_qty) | set(projected_qty))
        position_rows = []
        current_position_weight_sum = 0.0
        for symbol in all_symbols:
            price = prices.get(symbol, 0.0)
            current_value = current_values.get(symbol, current_qty.get(symbol, 0) * price)
            executable_value = projected_qty.get(symbol, 0) * price
            current_weight = current_value / total
            executable_weight = executable_value / total
            current_position_weight_sum += current_weight
            position_rows.append({
                "ticker": symbol,
                "target_weight": target_map.get(symbol, 0.0),
                "current_actual_weight": current_weight,
                "executable_weight": executable_weight,
                "current_qty": int(current_qty.get(symbol, 0)),
                "executable_qty": int(projected_qty.get(symbol, 0)),
            })

        account_stock_weight = self._account_allocation(account).get("stock_weight")
        current_actual = (
            float(account_stock_weight)
            if account_stock_weight is not None
            else current_position_weight_sum
        )
        executable = current_actual + sum(
            (1 if plan.action == "BUY" else -1) * plan.quantity * plan.est_price / total
            for plan in plans
        )
        target_exposure = float(target_assets.sum())
        target_cash = float(targets.get("__CASH__", max(1.0 - target_exposure, 0.0)))
        executable_cash = 1.0 - executable
        tracking_l1 = 0.5 * (
            sum(
                abs(row["target_weight"] - row["executable_weight"])
                for row in position_rows
            )
            + abs(target_cash - executable_cash)
        )
        skipped = [
            row for row in (planning_decisions or [])
            if row.get("action") == "SKIP"
        ]
        gap = target_exposure - executable
        warning = abs(gap) >= 0.05 or tracking_l1 >= 0.10 or bool(skipped)
        result = {
            "status": "warning" if warning else "ok",
            "target_exposure": target_exposure,
            "current_actual_exposure": current_actual,
            "executable_exposure": executable,
            "target_minus_executable": gap,
            "tracking_error_l1": tracking_l1,
            "planned_turnover_weight": sum(plan.est_value for plan in plans) / total,
            "order_count": len(plans),
            "skipped_target_count": len(skipped),
            "positions": position_rows,
        }
        if warning:
            logger.warning(
                "⚠️ 노출 추적 경고: 목표 %.1f%% / 현재 %.1f%% / 집행가능 %.1f%% / "
                "gap %.1f%%p / tracking L1 %.1f%%",
                target_exposure * 100,
                current_actual * 100,
                executable * 100,
                gap * 100,
                tracking_l1 * 100,
            )
        else:
            logger.info(
                "📐 노출 추적: 목표 %.1f%% / 현재 %.1f%% / 집행가능 %.1f%%",
                target_exposure * 100,
                current_actual * 100,
                executable * 100,
            )
        return result

    def run_small_account_shadow(
        self,
        signal_date: pd.Timestamp,
        targets: pd.Series,
        account: Dict,
        holdings: Dict,
        price_cache_override: Dict[str, float] | None,
    ) -> Path:
        policies = [
            ("floor_50k", 50_000, "floor"),
            ("nearest_etf_50k", 50_000, "nearest_etf"),
            ("nearest_etf_20k", 20_000, "nearest_etf"),
        ]
        comparisons = []
        for name, min_trade, rounding in policies:
            plans, decisions = self.build_order_plan(
                targets,
                account,
                holdings,
                price_cache_override=price_cache_override,
                min_trade_value_override=min_trade,
                quantity_rounding=rounding,
                return_decisions=True,
            )
            comparisons.append({
                "policy": name,
                "minimum_trade_value": min_trade,
                "quantity_rounding": rounding,
                "exposure": self._exposure_diagnostics(
                    targets,
                    account,
                    holdings,
                    plans,
                    price_cache_override=price_cache_override,
                    planning_decisions=decisions,
                ),
                "plans": [self._sanitized_plan(plan) for plan in plans],
                "decisions": self._sanitize_report_value(decisions),
            })

        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "trade_date": datetime.now().date().isoformat(),
            "signal_date": str(signal_date.date()),
            "market": self.market,
            "mode": "small_account_shadow",
            "execution_guard": "NO_ORDERS_SENT",
            "targets": {k: float(v) for k, v in targets.fillna(0.0).items()},
            "account_allocation": self._account_allocation(account),
            "holdings": self._sanitized_holdings(holdings),
            "decision_context": self._build_decision_context(signal_date),
            "source_meta": (self.loaded_signal_snapshot_payload or {}).get("meta", {}),
            "comparisons": comparisons,
        }
        report_path = self._shadow_report_path(signal_date)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self._sanitize_report_value(payload), f, ensure_ascii=False, indent=2)
        logger.info("🧪 소액계좌 shadow 비교 저장: %s", report_path)
        return report_path

    def append_execution_logs(self, logs: List[Dict]):
        if not logs:
            return
        with open(self.execution_log_path, "a", encoding="utf-8") as f:
            for row in logs:
                sanitized = self._sanitize_report_value(row)
                sanitized["timestamp"] = datetime.now().isoformat()
                sanitized["signal_mode"] = self.signal_mode
                sanitized["dry_run"] = self.dry_run
                f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")

    def save_execution_summary(
        self,
        signal_date: datetime | pd.Timestamp,
        targets: pd.Series,
        account: Dict | None,
        holdings: Dict | None,
        raw_plans: List[OrderPlan],
        plans: List[OrderPlan],
        recheck_logs: List[Dict],
        execution_result: Dict,
        completed_for_signal: bool,
        planning_decisions: List[Dict] | None = None,
    ) -> Path:
        summary_path = self._execution_summary_path(signal_date)
        exposure = (
            self._exposure_diagnostics(
                targets,
                account,
                holdings or {},
                plans,
                planning_decisions=planning_decisions,
            )
            if account
            else {"status": "unavailable", "reason": "account_snapshot_not_loaded"}
        )
        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "trade_date": datetime.now().date().isoformat(),
            "signal_date": str(signal_date.date() if hasattr(signal_date, "date") else signal_date),
            "market": self.market,
            "signal_mode": self.signal_mode,
            "dry_run": self.dry_run,
            "virtual_account": self.virtual_account,
            "completed_for_signal": bool(completed_for_signal),
            "targets": {k: float(v) for k, v in targets.fillna(0.0).items()},
            "style_attribution": self._build_style_attribution(pd.to_datetime(signal_date), targets),
            "decision_context": self._build_decision_context(pd.to_datetime(signal_date)),
            "meta": {
                "git_revision": self._git_revision(),
                "data_as_of": self._data_as_of(targets),
            },
            "account_allocation": self._account_allocation(account),
            "holdings": self._sanitized_holdings(holdings),
            "raw_plans": [self._sanitized_plan(plan) for plan in raw_plans],
            "final_plans": [self._sanitized_plan(plan) for plan in plans],
            "planning_decisions": self._sanitize_report_value(planning_decisions or []),
            "exposure": exposure,
            "recheck_logs": self._sanitize_report_value(recheck_logs),
            "execution": self._sanitize_report_value(execution_result),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._sanitize_report_value(payload), f, ensure_ascii=False, indent=2)
        logger.info("🧾 실행 요약 저장: %s", summary_path)
        return summary_path

    def execute(self, plans: List[OrderPlan], account: Dict, holdings: Dict, as_of: datetime) -> Dict:
        result_summary = {
            "executed_orders": 0,
            "failed_orders": 0,
            "not_eligible_orders": 0,
            "abort_reason": None,
            "order_logs": [],
            "report_path": None,
        }
        if not plans:
            logger.info("🚫 실행할 주문이 없습니다.")
            return result_summary
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
        snapshot = {"holdings": self._sanitized_holdings(holdings)}
        report_path = self.reporter.save_report(
            as_of,
            None,
            [self._sanitized_plan(plan) for plan in plans],
            snapshot,
        )
        self.append_execution_logs(order_logs)
        result_summary.update(
            {
                "executed_orders": executed_orders,
                "failed_orders": failed_orders,
                "not_eligible_orders": not_eligible_orders,
                "abort_reason": abort_reason,
                "order_logs": order_logs,
                "report_path": str(report_path),
            }
        )
        if self.dry_run:
            logger.info("dry-run 모드이므로 텔레그램 알림을 생략합니다.")
            return result_summary
        if executed_orders == 0:
            if failed_orders == 0:
                logger.info("실제 체결된 주문이 없어 텔레그램 알림을 생략합니다.")
                return result_summary
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
            return result_summary
        self._notify(
            latest_equity=equity,
            plans=plans,
            report_path=report_path,
            report_date=as_of,
            trade_time=datetime.now(),
        )
        return result_summary

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

    def _notify_recheck_filtered(
        self,
        raw_plans: List[OrderPlan],
        recheck_logs: List[Dict],
        as_of: datetime,
    ):
        if self.dry_run or not self.telegram.can_send():
            return
        skipped = [row for row in recheck_logs if row.get("decision") == "skip"]
        lines = [
            f"신호일: {as_of.date()}",
            f"원 주문 계획: {len(raw_plans)}건",
            f"재검증 스킵: {len(skipped)}건",
        ]
        for row in skipped[:5]:
            lines.append(
                f"- {row.get('action')} {row.get('ticker')}: {row.get('reason')}"
            )
        if len(skipped) > 5:
            lines.append(f"...외 {len(skipped) - 5}건")
        self.telegram.send_message(format_alert("Multi Allocator PLUS (재검증 스킵)", lines))

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
        if self.small_account_shadow:
            if self.signal_mode != "eod_fixed":
                raise RuntimeError("소액계좌 shadow 비교는 저장된 eod_fixed 신호만 사용합니다.")
            account, holdings = self.fetch_account_snapshot(force_live_read=True)
            self.run_small_account_shadow(
                last_date,
                targets,
                account,
                holdings,
                price_cache_override,
            )
            return
        prior_execution = self._completed_execution_for_signal(last_date)
        if prior_execution is not None:
            logger.info(
                "이미 완료된 실행이 있어 스킵합니다: signal_date=%s prior_run_id=%s",
                last_date.date(),
                prior_execution.get("run_id"),
            )
            self.save_execution_summary(
                signal_date=last_date,
                targets=targets,
                account=None,
                holdings=None,
                raw_plans=[],
                plans=[],
                recheck_logs=[],
                execution_result={
                    "status": "skipped_already_executed",
                    "prior_run_id": prior_execution.get("run_id"),
                },
                completed_for_signal=False,
            )
            return
        if self.cache_only:
            logger.info("🗂️ 캐시 리프레시 전용 실행이 완료되었습니다. 트레이딩 루틴은 건너뜁니다.")
            return
        account = None
        holdings = None
        raw_plans: List[OrderPlan] = []
        plans: List[OrderPlan] = []
        planning_decisions: List[Dict] = []
        recheck_logs: List[Dict] = []
        try:
            account, holdings = self.fetch_account_snapshot()
            raw_plans, planning_decisions = self.build_order_plan(
                targets,
                account,
                holdings,
                price_cache_override=price_cache_override,
                return_decisions=True,
            )
            self.append_execution_logs([
                {"log_type": "planning", **row}
                for row in planning_decisions
            ])
            plans, recheck_logs = self.apply_execution_recheck(raw_plans, account)
            self.append_execution_logs(recheck_logs)
            exposure = self._exposure_diagnostics(
                targets,
                account,
                holdings,
                plans,
                price_cache_override=price_cache_override,
                planning_decisions=planning_decisions,
            )
            self.append_execution_logs([{"log_type": "exposure", **exposure}])
            if raw_plans and not plans:
                logger.warning("재검증 후 모든 주문이 스킵되었습니다.")
                self._notify_recheck_filtered(raw_plans, recheck_logs, last_date)
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
            execution_result = self.execute(plans, account, holdings, last_date)
            completed_for_signal = (
                not self.dry_run
                and execution_result.get("failed_orders", 0) == 0
                and execution_result.get("abort_reason") is None
                and (
                    execution_result.get("executed_orders", 0) > 0
                    or (
                        len(raw_plans) == 0
                        and exposure.get("status") == "ok"
                    )
                )
            )
            self.save_execution_summary(
                signal_date=last_date,
                targets=targets,
                account=account,
                holdings=holdings,
                raw_plans=raw_plans,
                plans=plans,
                recheck_logs=recheck_logs,
                execution_result=execution_result,
                completed_for_signal=completed_for_signal,
                planning_decisions=planning_decisions,
            )
        except Exception as exc:
            logger.exception("실행 중 예외 발생: %s", exc)
            self.save_execution_summary(
                signal_date=last_date,
                targets=targets,
                account=account,
                holdings=holdings,
                raw_plans=raw_plans,
                plans=plans,
                recheck_logs=recheck_logs,
                execution_result={
                    "status": "failed_exception",
                    "reason": str(exc),
                },
                completed_for_signal=False,
                planning_decisions=planning_decisions,
            )
            raise

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
    parser.add_argument(
        "--small-account-shadow",
        action="store_true",
        help="실계좌 잔고를 읽어 소액계좌 집행 정책 3종을 비교하되 주문은 전송하지 않음",
    )
    parser.add_argument(
        "--skip-if-executed",
        dest="skip_if_executed",
        action="store_true",
        help="같은 거래일/신호일의 완료된 실행 요약이 있으면 주문을 스킵",
    )
    parser.add_argument(
        "--no-skip-if-executed",
        dest="skip_if_executed",
        action="store_false",
        help="완료된 실행 요약이 있어도 강제 실행",
    )
    parser.set_defaults(skip_if_executed=True)
    args = parser.parse_args()
    if args.small_account_shadow and not args.real:
        parser.error("--small-account-shadow에는 실계좌 조회를 위한 --real이 필요합니다.")
    if args.small_account_shadow and args.signal_mode != "eod_fixed":
        parser.error("--small-account-shadow는 --signal-mode eod_fixed와 함께 사용해야 합니다.")

    trader = MultiAllocatorPlusTrader(
        start_date=args.start_date,
        use_cache=not args.no_cache,
        dry_run=args.dry_run or args.small_account_shadow,
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
        skip_if_executed=args.skip_if_executed,
        small_account_shadow=args.small_account_shadow,
    )
    trader.run()


if __name__ == "__main__":
    main()
