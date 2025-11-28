#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily trading report generator
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


@dataclass
class TradeSnapshot:
    symbol: str
    action: str
    quantity: int
    price: float
    amount: float
    reason: str = ""


class DailyReporter:
    """일별 거래 보고서를 CSV/JSON으로 저장"""

    def __init__(self, report_dir: Path):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self,
        run_date: datetime,
        equity: float,
        orders: List[Dict[str, Any]],
        portfolio_snapshot: Dict[str, Any],
    ) -> Path:
        report_date = run_date.strftime("%Y%m%d")
        csv_path = self.report_dir / f"daily_report_{report_date}.csv"
        json_path = self.report_dir / f"daily_report_{report_date}.json"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "action", "qty", "est_price", "est_value", "target_weight", "current_qty", "target_qty"])
            for plan in orders:
                writer.writerow([
                    plan.get("symbol"),
                    plan.get("action"),
                    plan.get("quantity"),
                    plan.get("est_price"),
                    plan.get("est_value"),
                    plan.get("target_weight"),
                    plan.get("current_qty"),
                    plan.get("target_qty"),
                ])

        payload = {
            "date": report_date,
            "equity": equity,
            "orders": orders,
            "portfolio": portfolio_snapshot,
        }
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2, ensure_ascii=False)

        logger.info("📄 일일 리포트 저장: %s", csv_path)
        return csv_path
