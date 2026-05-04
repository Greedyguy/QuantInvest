"""
Hybrid Portfolio Strategy v3.0
Korean Aggressive (70%) + Production Portfolio (30%)

- signal.py 에서 생성한 지표(rsi, ma5, ma20, returns, volume 등)를 활용
- 백테스트에서도 실제로 매매가 발생하도록 조건 완화 및 버그 수정
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL, VENUE_FEE_PER_SIDE, FEE_PER_SIDE_US, US_TAX_RATE_SELL


class HybridPortfolioStrategyV24(BaseStrategy):
    """Hybrid Portfolio: Korean Aggressive (70%) + Production Portfolio (30%)"""
    
    def __init__(
        self,
        korean_aggressive_ratio: float = 0.45,
        production_portfolio_ratio: float = 0.55,
        korean_max_positions: int = 6,
        portfolio_max_positions: int = 6,
        korean_position_size: float = 0.15,   # Korean Aggressive 포지션 크기 (캐시 대비)
        portfolio_position_size: float = 0.35, # Production 포지션 크기 (캐시 대비)
        max_single_stock_ratio: float = 0.20   # 전체 포트 대비 단일 종목 최대 비중
    ):
        """
        Args:
            korean_aggressive_ratio: Korean Aggressive 전략 자본 비율 (기본 70%)
            production_portfolio_ratio: Production Portfolio 전략 자본 비율 (기본 30%)
            korean_max_positions: Korean Aggressive 최대 포지션 수
            portfolio_max_positions: Production Portfolio 최대 포지션 수
            korean_position_size: Korean Aggressive 포지션 크기 비율
            portfolio_position_size: Production Portfolio 포지션 크기 비율
            max_single_stock_ratio: 전체 포트폴리오에서 단일 종목 최대 비중
        """
        self.korean_aggressive_ratio = korean_aggressive_ratio
        self.production_portfolio_ratio = production_portfolio_ratio
        self.korean_max_positions = korean_max_positions
        self.portfolio_max_positions = portfolio_max_positions
        self.korean_position_size = korean_position_size
        self.portfolio_position_size = portfolio_position_size
        self.max_single_stock_ratio = max_single_stock_ratio
        
        # 거래 비용
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = 0.001
        self.risk_on_buffer_days = 3
        self._market_profile_set = False
        self._is_us = False

        # KR 기본 필터
        self.korean_price_min = 1000
        self.korean_price_max = 100000
        self.korean_min_volume = 10000
        self.korean_rsi_min = 35
        self.korean_rsi_max = 80
        self.korean_ma5_ratio = 0.995
        self.korean_min_gap_pct = 0.1

        self.portfolio_price_min = 3000
        self.portfolio_price_max = 500000
        self.portfolio_min_volume = 10000
        self.portfolio_rsi_min = 30
        self.portfolio_rsi_max = 80
        self.portfolio_ma20_ratio = 0.99
        self.portfolio_extra_price_floor = 5000

    def _detect_us_market(self, tickers):
        sample = list(tickers)[:30]
        if not sample:
            return False
        return any((not str(t).isdigit()) for t in sample)

    def _set_market_profile(self, enriched):
        if self._market_profile_set:
            return
        self._is_us = self._detect_us_market(enriched.keys())
        if self._is_us:
            self.korean_price_min = 5
            self.korean_price_max = 1500
            self.korean_min_volume = 200000
            self.korean_min_gap_pct = 0.05

            self.portfolio_price_min = 5
            self.portfolio_price_max = 1500
            self.portfolio_min_volume = 200000
            self.portfolio_extra_price_floor = 5
            self.max_single_stock_ratio = max(self.max_single_stock_ratio, 0.40)
            self.fee = FEE_PER_SIDE_US
            self.tax = US_TAX_RATE_SELL
        self._market_profile_set = True

    def _compute_market_regime(self, market_index):
        if market_index is None or "close" not in market_index.columns:
            return {}
        idx = market_index.copy().sort_index()
        idx["ma20"] = idx["close"].rolling(20).mean()
        idx["ma60"] = idx["close"].rolling(60).mean()
        idx["risk_on"] = (idx["close"] > idx["ma60"]) & (idx["ma20"] > idx["ma60"])
        flags = idx["risk_on"].fillna(False)
        # apply buffer so that risk_on requires consecutive days
        buffered = flags.rolling(self.risk_on_buffer_days).sum() == self.risk_on_buffer_days
        return buffered.to_dict()
    
    def get_name(self) -> str:
        return "hybrid_portfolio_v2_4"
    
    def get_description(self) -> str:
        return f"Hybrid Portfolio v2.4 (Korean {self.korean_aggressive_ratio:.0%} + Portfolio {self.production_portfolio_ratio:.0%})"
    
    # ------------------------------------------------------------------
    # 메인 백테스트 루프
    # ------------------------------------------------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """백테스트 실행"""
        self._set_market_profile(enriched)
        
        self._reset_weight_history()
        if not silent:
            print(f"\n{'='*60}")
            print(f"🔥 {self.get_description()} 백테스트 시작 (v3.0)")
            print(f"{'='*60}")
        
        # 초기 자본 설정
        initial_capital = 1_000_000  # 100만원
        korean_capital = initial_capital * self.korean_aggressive_ratio
        portfolio_capital = initial_capital * self.production_portfolio_ratio
        cash_state = {
            'korean': korean_capital,
            'portfolio': portfolio_capital,
        }
        
        # 전략별 포지션 관리
        korean_positions = {}    # {ticker: {qty, entry_px, entry_date, strategy}}
        portfolio_positions = {} # {ticker: {qty, entry_px, entry_date, strategy}}
        
        # 거래 기록 & 에쿼티 커브
        trade_log = []
        equity_curve = []
        
        # 전체 날짜 리스트
        all_dates = set()
        for df in enriched.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)
        
        if len(dates) < 60:
            if not silent:
                print("⚠️ 데이터 부족: 최소 60일 필요")
            return pd.DataFrame(), []
        
        regime_flags = self._compute_market_regime(market_index)

        # 백테스트 시작 (60일 warming up)
        start_idx = 60
        
        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            
            is_risk_on = regime_flags.get(current_date, True)

            # 1) 매도 신호 먼저 처리
            self._check_sell_signals(
                current_date, 
                korean_positions, 
                portfolio_positions,
                enriched, 
                dates, 
                i, 
                trade_log,
                silent,
                cash_state
            )
            
            # 2) 현재 자본(캐시) 계산
            korean_cash = cash_state['korean']
            portfolio_cash = cash_state['portfolio']
            
            # 3) 매수 신호 체크
            # Korean Aggressive
            current_korean_fraction = self.korean_position_size if is_risk_on else self.korean_position_size * 0.4
            if len(korean_positions) < self.korean_max_positions and korean_cash > 0 and is_risk_on:
                self._check_korean_aggressive_buy(
                    current_date,
                    korean_positions,
                    portfolio_positions,
                    enriched,
                    cash_state,
                    initial_capital,
                    trade_log,
                    silent,
                    position_fraction=current_korean_fraction
                )
            
            # Production Portfolio
            current_portfolio_fraction = self.portfolio_position_size if is_risk_on else self.portfolio_position_size * 0.6
            if len(portfolio_positions) < self.portfolio_max_positions and portfolio_cash > 0:
                self._check_production_portfolio_buy(
                    current_date,
                    korean_positions,
                    portfolio_positions,
                    enriched,
                    cash_state,
                    initial_capital,
                    trade_log,
                    silent,
                    position_fraction=current_portfolio_fraction
                )
            
            # 4) 포트폴리오 가치 계산
            total_value = self._calculate_portfolio_value(
                current_date,
                korean_positions,
                portfolio_positions,
                enriched,
                cash_state['korean'],
                cash_state['portfolio']
            )
            
            equity_curve.append({
                'date': current_date,
                'equity': total_value
            })
            combined_positions = {**korean_positions, **portfolio_positions}
            total_cash = cash_state['korean'] + cash_state['portfolio']
            self._record_weights(current_date, total_cash, combined_positions, enriched)
        
        # 5) 최종 청산
        final_date = dates[-1]
        self._liquidate_all_positions(
            final_date,
            korean_positions,
            portfolio_positions,
            enriched,
            trade_log,
            silent,
            cash_state
        )
        final_equity = cash_state['korean'] + cash_state['portfolio']
        if equity_curve and equity_curve[-1]['date'] == final_date:
            equity_curve[-1]['equity'] = final_equity
        else:
            equity_curve.append({'date': final_date, 'equity': final_equity})
        
        # 결과 정리
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        if not silent:
            print(f"\n✅ 백테스트 완료: {len(trade_log)}개 거래 (BUY+SELL)")
            print(f"   Korean Aggressive 거래: {sum(1 for t in trade_log if t.get('strategy') == 'korean_aggressive')}개")
            print(f"   Production Portfolio 거래: {sum(1 for t in trade_log if t.get('strategy') == 'production_portfolio')}개")
        
        return equity_df, trade_log
    
    # ------------------------------------------------------------------
    # 공통 매도 로직
    # ------------------------------------------------------------------
    def _check_sell_signals(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        dates: list,
        current_idx: int,
        trade_log: list,
        silent: bool,
        cash_state: dict
    ):
        """매도 신호 체크 (양쪽 전략 공통)"""
        
        all_positions = {
            **{k: {**v, 'source': 'korean_aggressive'} for k, v in korean_positions.items()},
            **{k: {**v, 'source': 'production_portfolio'} for k, v in portfolio_positions.items()}
        }
        
        to_sell = []
        
        for ticker, pos in all_positions.items():
            if ticker not in enriched:
                continue
            
            df = enriched[ticker]
            if current_date not in df.index:
                continue
            
            row = df.loc[current_date]
            current_price = row['close']
            entry_price = pos['entry_px']
            entry_date = pos['entry_date']
            
            pnl_pct = (current_price - entry_price) / entry_price
            days_held = (current_date - entry_date).days
            
            # 기본 매도 조건 (너무 빡세지 않게 유지)
            reason = None
            
            # 1) 손절 -10%
            if pnl_pct <= -0.10:
                reason = "손절"
            # 2) 익절 +20%
            elif pnl_pct >= 0.20:
                reason = "익절"
            # 3) 기간 만료 + 수익
            elif days_held >= 10 and pnl_pct >= 0.05:
                reason = "기간만료_수익실현"
            # 4) 최대 보유일
            elif days_held >= 20:
                reason = "기간만료"
            
            if reason:
                to_sell.append((ticker, pos, current_price, reason))
        
        # 매도 실행
        for ticker, pos, current_price, reason in to_sell:
            strategy = pos['source']
            qty = pos['qty']
            entry_price = pos['entry_px']
            
            sell_amount = qty * current_price
            commission = sell_amount * self.fee
            tax = sell_amount * self.tax
            net_amount = sell_amount - commission - tax
            
            buy_cost = qty * entry_price * (1 + self.fee)
            pnl = net_amount - buy_cost
            
            trade_log.append({
                'date': current_date,
                'ticker': ticker,
                'strategy': strategy,
                'action': 'SELL',
                'price': current_price,
                'qty': qty,
                'amount': net_amount,
                'pnl': pnl,
                'reason': reason
            })
            
            if strategy == 'korean_aggressive':
                cash_state['korean'] += net_amount
                del korean_positions[ticker]
            else:
                cash_state['portfolio'] += net_amount
                del portfolio_positions[ticker]
            
            if not silent:
                print(f"  💰 매도: {ticker} ({strategy}) @ {current_price:,.0f}원 x {qty}주 = {pnl:+,.0f}원 ({reason})")
    
    # ------------------------------------------------------------------
    # Korean Aggressive 매수 로직 (완화 버전)
    # ------------------------------------------------------------------
    def _check_korean_aggressive_buy(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        cash_state: dict,
        initial_capital: float,
        trade_log: list,
        silent: bool,
        position_fraction: Optional[float] = None,
    ):
        """Korean Aggressive 매수 신호 체크 (백테스트 친화 조건)"""
        
        candidates = []
        
        for ticker, df in enriched.items():
            if current_date not in df.index:
                continue
            
            # 이미 보유 중이면 제외
            if ticker in korean_positions or ticker in portfolio_positions:
                continue
            
            row = df.loc[current_date]
            close_price = row['close']
            volume = row['volume']
            returns = row.get('returns', 0.0)
            rsi = row.get('rsi', 50.0)
            ma5 = row.get('ma5', close_price)
            
            # 1) 가격 필터 (원래 범위 유지)
            if not (self.korean_price_min <= close_price <= self.korean_price_max):
                continue
            
            # 2) 거래량 필터 (완화: 50,000 → 10,000)
            if volume < self.korean_min_volume:
                continue
            
            # 3) RSI 필터 (완화: 40~85 → 35~80)
            if not (self.korean_rsi_min < rsi < self.korean_rsi_max):
                continue
            
            # 4) MA5 상방 (완화: close > ma5 → close > ma5 * 0.995)
            if close_price <= ma5 * self.korean_ma5_ratio:
                continue
            
            # 5) 전일 대비 변동성 (완화: 0.5% → 0.1%)
            gap_pct = abs(returns) * 100
            if gap_pct <= self.korean_min_gap_pct:
                continue
            
            # 후보 추가
            candidates.append({
                'ticker': ticker,
                'price': close_price,
                'volume': volume,
                'rsi': rsi,
                'score': volume * (1 + gap_pct / 100.0)  # 거래량 * (1+변동)
            })
        
        if not candidates:
            return
        
        # 점수 기준 정렬
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        available_cash = cash_state.get('korean', 0.0)
        if available_cash <= 0:
            return
        frac = position_fraction if position_fraction is not None else self.korean_position_size

        for candidate in candidates[:5]:
            ticker = candidate['ticker']
            price = candidate['price']
            
            if not self._should_allow_duplicate(
                ticker,
                korean_positions,
                portfolio_positions,
                price,
                available_cash,
                initial_capital,
                position_fraction=frac
            ):
                continue
            
            position_size = available_cash * frac
            qty = int(position_size / price)
            if qty <= 0:
                continue
            
            buy_cost = qty * price * (1 + self.fee)
            if buy_cost > available_cash:
                continue
            
            cash_state['korean'] -= buy_cost
            available_cash = cash_state['korean']
            korean_positions[ticker] = {
                'qty': qty,
                'entry_px': price,
                'entry_date': current_date,
                'strategy': 'korean_aggressive'
            }
            
            trade_log.append({
                'date': current_date,
                'ticker': ticker,
                'strategy': 'korean_aggressive',
                'action': 'BUY',
                'price': price,
                'qty': qty,
                'amount': buy_cost,
                'pnl': 0
            })
            
            if not silent:
                print(f"  💚 매수: {ticker} (Korean) @ {price:,.0f}원 x {qty}주 = {buy_cost:,.0f}원")
            
            break  # 하루에 한 종목만 신규 진입
    
    # ------------------------------------------------------------------
    # Production Portfolio 매수 로직 (완화 버전)
    # ------------------------------------------------------------------
    def _check_production_portfolio_buy(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        cash_state: dict,
        initial_capital: float,
        trade_log: list,
        silent: bool,
        position_fraction: Optional[float] = None,
    ):
        """Production Portfolio 매수 신호 체크 (완화 조건)"""
        
        candidates = []
        
        for ticker, df in enriched.items():
            if current_date not in df.index:
                continue
            
            if ticker in korean_positions or ticker in portfolio_positions:
                continue
            
            row = df.loc[current_date]
            close_price = row['close']
            volume = row['volume']
            rsi = row.get('rsi', 50.0)
            ma20 = row.get('ma20', close_price)
            
            # 1) 가격 필터 (그대로 유지)
            if not (self.portfolio_price_min <= close_price <= self.portfolio_price_max):
                continue
            
            # 2) 거래량 필터 (완화: 30,000 → 10,000)
            if volume < self.portfolio_min_volume:
                continue
            
            # 3) RSI 필터 (완화: 35~75 → 30~80)
            if not (self.portfolio_rsi_min < rsi < self.portfolio_rsi_max):
                continue
            
            # 4) MA20 위 (완화: close > ma20*0.98 → close > ma20*0.99)
            if close_price <= ma20 * self.portfolio_ma20_ratio:
                continue
            
            # 5) 가격 하한 (완화: 8,000 → 5,000)
            if close_price <= self.portfolio_extra_price_floor:
                continue
            
            candidates.append({
                'ticker': ticker,
                'price': close_price,
                'volume': volume,
                'rsi': rsi,
                'score': volume
            })
        
        if not candidates:
            return
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        available_cash = cash_state.get('portfolio', 0.0)
        if available_cash <= 0:
            return
        frac = position_fraction if position_fraction is not None else self.portfolio_position_size

        for candidate in candidates[:5]:
            ticker = candidate['ticker']
            price = candidate['price']

            if not self._should_allow_duplicate(
                ticker,
                korean_positions,
                portfolio_positions,
                price,
                available_cash,
                initial_capital,
                position_fraction=frac
            ):
                continue

            position_size = available_cash * frac
            qty = int(position_size / price)
            if qty <= 0:
                continue

            buy_cost = qty * price * (1 + self.fee)
            if buy_cost > available_cash:
                continue

            cash_state['portfolio'] -= buy_cost
            available_cash = cash_state['portfolio']
            portfolio_positions[ticker] = {
                'qty': qty,
                'entry_px': price,
                'entry_date': current_date,
                'strategy': 'production_portfolio'
            }

            trade_log.append({
                'date': current_date,
                'ticker': ticker,
                'strategy': 'production_portfolio',
                'action': 'BUY',
                'price': price,
                'qty': qty,
                'amount': buy_cost,
                'pnl': 0
            })

            if not silent:
                print(f"  📊 매수: {ticker} (Portfolio) @ {price:,.0f}원 x {qty}주 = {buy_cost:,.0f}원")

            break
    
    # ------------------------------------------------------------------
    # 단일종목 비중 체크 (버그 수정)
    # ------------------------------------------------------------------
    def _should_allow_duplicate(
        self,
        ticker: str,
        korean_positions: dict,
        portfolio_positions: dict,
        price: float,
        cash: float,
        initial_capital: float,
        position_fraction: float
    ) -> bool:
        """
        종목 중복/과도 비중 체크
        - 이미 보유 중이면 False
        - 새 포지션을 position_fraction * cash 로 가정하여
          initial_capital 대비 비중이 max_single_stock_ratio 이내인지 체크
        """
        if ticker in korean_positions or ticker in portfolio_positions:
            return False
        
        # 새 포지션 가정
        planned_value = price * int(cash * position_fraction / price)
        if planned_value <= 0:
            return False
        
        total_value = max(initial_capital, 1.0)
        proposed_ratio = planned_value / total_value
        
        if proposed_ratio > self.max_single_stock_ratio:
            return False
        
        return True
    
    # ------------------------------------------------------------------
    # 포트폴리오 평가
    # ------------------------------------------------------------------
    def _calculate_portfolio_value(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        korean_cash: float,
        portfolio_cash: float
    ) -> float:
        """전체 포트폴리오 가치 계산"""
        
        korean_position_value = 0.0
        for ticker, pos in korean_positions.items():
            if ticker in enriched and current_date in enriched[ticker].index:
                current_price = enriched[ticker].loc[current_date, 'close']
                korean_position_value += pos['qty'] * current_price
            else:
                korean_position_value += pos['qty'] * pos['entry_px']
        
        portfolio_position_value = 0.0
        for ticker, pos in portfolio_positions.items():
            if ticker in enriched and current_date in enriched[ticker].index:
                current_price = enriched[ticker].loc[current_date, 'close']
                portfolio_position_value += pos['qty'] * current_price
            else:
                portfolio_position_value += pos['qty'] * pos['entry_px']
        
        total_value = korean_position_value + portfolio_position_value + max(korean_cash, 0.0) + max(portfolio_cash, 0.0)
        return total_value
    
    # ------------------------------------------------------------------
    # 최종 청산
    # ------------------------------------------------------------------
    def _liquidate_all_positions(
        self,
        final_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        trade_log: list,
        silent: bool,
        cash_state: dict
    ):
        """모든 포지션 최종 청산"""
        
        all_positions = list(korean_positions.items()) + list(portfolio_positions.items())
        
        for ticker, pos in all_positions:
            if ticker not in enriched:
                continue
            
            df = enriched[ticker]
            if final_date not in df.index:
                continue
            
            final_price = df.loc[final_date, 'close']
            qty = pos['qty']
            entry_price = pos['entry_px']
            
            sell_amount = qty * final_price
            commission = sell_amount * self.fee
            tax = sell_amount * self.tax
            net_amount = sell_amount - commission - tax
            
            buy_cost = qty * entry_price * (1 + self.fee)
            pnl = net_amount - buy_cost
            
            strategy = pos.get('strategy', 'unknown')
            
            trade_log.append({
                'date': final_date,
                'ticker': ticker,
                'strategy': strategy,
                'action': 'SELL',
                'price': final_price,
                'qty': qty,
                'amount': net_amount,
                'pnl': pnl,
                'reason': '최종청산'
            })

            if strategy == 'korean_aggressive':
                cash_state['korean'] += net_amount
                korean_positions.pop(ticker, None)
            else:
                cash_state['portfolio'] += net_amount
                portfolio_positions.pop(ticker, None)
            
            if not silent:
                print(f"  🔚 최종청산: {ticker} ({strategy}) @ {final_price:,.0f}원 x {qty}주 = {pnl:+,.0f}원")
