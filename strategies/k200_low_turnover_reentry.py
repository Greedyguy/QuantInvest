"""Low-turnover KODEX 200 trend exit and re-entry strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEE_PER_SIDE, SLIPPAGE_ENTRY, SLIPPAGE_EXIT, TAX_RATE_SELL
from strategies.base_strategy import BaseStrategy


class K200LowTurnoverReentry(BaseStrategy):
    """Hold KODEX 200 in confirmed trends and otherwise preserve cash.

    Decisions are made once per month.  On the first observed trading day of a
    new month, only the prior trading day's close and indicators are used, and
    a state change is executed at that day's open.  Holdings are untouched
    between state changes, avoiding target-weight maintenance churn.
    """

    CASH = "cash"
    RISK_ON = "risk_on"

    def __init__(
        self,
        ticker: str = "069500",
        risk_on_exposure: float = 0.95,
        trend_window: int = 120,
        momentum_window: int = 60,
        fast_trend_window: int = 20,
        medium_trend_window: int = 60,
        entry_buffer: float = 0.01,
        exit_buffer: float = 0.01,
        exit_momentum: float = -0.03,
        emergency_drawdown: float = -0.12,
        emergency_momentum: float = -0.04,
        initial_cash: float = 2_100_000.0,
    ):
        super().__init__()
        self.ticker = ticker
        self.risk_on_exposure = float(risk_on_exposure)
        self.trend_window = int(trend_window)
        self.momentum_window = int(momentum_window)
        self.fast_trend_window = int(fast_trend_window)
        self.medium_trend_window = int(medium_trend_window)
        self.entry_buffer = float(entry_buffer)
        self.exit_buffer = float(exit_buffer)
        self.exit_momentum = float(exit_momentum)
        self.emergency_drawdown = float(emergency_drawdown)
        self.emergency_momentum = float(emergency_momentum)
        self.initial_cash = float(initial_cash)
        self.latest_state_history = pd.DataFrame()

    def get_name(self) -> str:
        return "k200_low_turnover_reentry"

    def get_description(self) -> str:
        return (
            "KODEX 200 monthly low-turnover trend exit/re-entry "
            f"({self.risk_on_exposure:.0%} risk-on)"
        )

    def _indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy().sort_index()
        out.index = pd.to_datetime(out.index)
        out = out.loc[~out.index.duplicated(keep="last")]
        close = pd.to_numeric(out["close"], errors="coerce")
        out["trend_ma"] = close.rolling(self.trend_window).mean()
        out["fast_ma"] = close.rolling(self.fast_trend_window).mean()
        out["medium_ma"] = close.rolling(self.medium_trend_window).mean()
        out["momentum"] = close.pct_change(self.momentum_window)
        out["fast_momentum"] = close.pct_change(self.fast_trend_window)
        out["drawdown"] = close / close.rolling(self.medium_trend_window).max() - 1.0
        return out.dropna(
            subset=[
                "open",
                "close",
                "trend_ma",
                "fast_ma",
                "medium_ma",
                "momentum",
                "fast_momentum",
                "drawdown",
            ]
        )

    def _next_state(
        self,
        row: pd.Series,
        current_state: str,
        monthly_decision: bool,
    ) -> tuple[str, str]:
        close = float(row["close"])
        trend = float(row["trend_ma"])
        momentum = float(row["momentum"])
        fast_ma = float(row["fast_ma"])
        medium_ma = float(row["medium_ma"])
        fast_momentum = float(row["fast_momentum"])
        drawdown = float(row["drawdown"])
        emergency_exit = (
            drawdown <= self.emergency_drawdown and close < fast_ma
        ) or (
            close < medium_ma * 0.97
            and fast_momentum <= self.emergency_momentum
        )
        if current_state == self.RISK_ON and emergency_exit:
            return self.CASH, "emergency_trend_exit"
        if not monthly_decision:
            return current_state, "no_monthly_decision"
        if current_state == self.CASH:
            if (
                close > trend * (1.0 + self.entry_buffer)
                and momentum > 0.0
                and close > medium_ma
                and fast_momentum > 0.0
            ):
                return self.RISK_ON, "monthly_trend_reentry"
            return current_state, "cash_hold"
        if (
            close < trend * (1.0 - self.exit_buffer)
            or momentum <= self.exit_momentum
        ):
            return self.CASH, "monthly_trend_exit"
        return current_state, "risk_hold"

    def compute_state_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the state effective at each open using only prior data."""

        data = self._indicators(frame)
        if data.empty:
            return pd.DataFrame(
                columns=["state", "target_exposure", "signal_date", "reason"]
            )
        dates = list(data.index)
        state = self.CASH
        rows = []
        for position, current_date in enumerate(dates):
            signal_date = pd.NaT
            reason = "no_monthly_decision"
            if position > 0:
                prior_date = dates[position - 1]
                monthly_decision = current_date.month != prior_date.month
                next_state, reason = self._next_state(
                    data.loc[prior_date], state, monthly_decision
                )
                if next_state != state:
                    signal_date = prior_date
                state = next_state
            rows.append(
                {
                    "date": current_date,
                    "state": state,
                    "target_exposure": (
                        self.risk_on_exposure if state == self.RISK_ON else 0.0
                    ),
                    "signal_date": signal_date,
                    "reason": reason,
                }
            )
        return pd.DataFrame(rows).set_index("date")

    def run_backtest(
        self,
        enriched: dict,
        market_index=None,
        weights=None,
        silent: bool = False,
    ) -> tuple[pd.DataFrame, list[dict]]:
        self._reset_weight_history()
        frame = enriched.get(self.ticker)
        if frame is None or frame.empty:
            return pd.DataFrame(), []
        data = self._indicators(frame)
        states = self.compute_state_history(frame)
        self.latest_state_history = states.copy()
        if states.empty:
            return pd.DataFrame(), []

        cash = self.initial_cash
        quantity = 0
        effective_target = 0.0
        trades: list[dict] = []
        equity_rows = []

        first_date = states.index[0]
        equity_rows.append((first_date, cash, cash, quantity, self.CASH))
        self._record_weights(first_date, cash, {}, {self.ticker: data})

        for current_date in states.index[1:]:
            row = states.loc[current_date]
            target = float(row["target_exposure"])
            open_price = float(data.loc[current_date, "open"])
            close_price = float(data.loc[current_date, "close"])
            if not np.isfinite(open_price) or open_price <= 0:
                open_price = close_price

            if target != effective_target:
                equity_at_open = cash + quantity * open_price
                target_quantity = int(equity_at_open * target / open_price)
                delta = target_quantity - quantity
                if delta < 0:
                    sell_quantity = min(quantity, -delta)
                    execution_price = open_price * (1.0 - SLIPPAGE_EXIT)
                    gross = sell_quantity * execution_price
                    fee = gross * FEE_PER_SIDE
                    tax = gross * TAX_RATE_SELL
                    cash += gross - fee - tax
                    quantity -= sell_quantity
                    trades.append(
                        {
                            "signal_date": row["signal_date"],
                            "date": current_date,
                            "ticker": self.ticker,
                            "action": "SELL",
                            "price": execution_price,
                            "qty": sell_quantity,
                            "fee": fee,
                            "tax": tax,
                            "reason": row["reason"],
                        }
                    )
                elif delta > 0:
                    execution_price = open_price * (1.0 + SLIPPAGE_ENTRY)
                    cash_per_share = execution_price * (1.0 + FEE_PER_SIDE)
                    buy_quantity = min(delta, int(cash / cash_per_share))
                    if buy_quantity > 0:
                        gross = buy_quantity * execution_price
                        fee = gross * FEE_PER_SIDE
                        cash -= gross + fee
                        quantity += buy_quantity
                        trades.append(
                            {
                                "signal_date": row["signal_date"],
                                "date": current_date,
                                "ticker": self.ticker,
                                "action": "BUY",
                                "price": execution_price,
                                "qty": buy_quantity,
                                "fee": fee,
                                "tax": 0.0,
                                "reason": row["reason"],
                            }
                        )
                effective_target = target

            equity = cash + quantity * close_price
            state = str(row["state"])
            equity_rows.append((current_date, equity, cash, quantity, state))
            positions = (
                {self.ticker: {"qty": quantity, "entry_px": close_price}}
                if quantity > 0
                else {}
            )
            self._record_weights(current_date, cash, positions, {self.ticker: data})

        equity_curve = pd.DataFrame(
            equity_rows,
            columns=["date", "equity", "cash", "quantity", "state"],
        ).set_index("date")
        return equity_curve, trades
