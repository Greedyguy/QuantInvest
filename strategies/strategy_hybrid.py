"""
Hybrid Portfolio Strategy
Korean Aggressive (70%) + Production Portfolio (30%)

hybrid_portfolio_trader.py의 실시간 매매 로직을 백테스트용으로 변환
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL, VENUE_FEE_PER_SIDE


class HybridPortfolioStrategy(BaseStrategy):
    """Hybrid Portfolio: Korean Aggressive (70%) + Production Portfolio (30%)"""
    
    def __init__(
        self,
        korean_aggressive_ratio: float = 0.70,
        production_portfolio_ratio: float = 0.30,
        korean_max_positions: int = 7,
        portfolio_max_positions: int = 6,
        korean_position_size: float = 0.20,  # 20% 포지션 크기
        portfolio_position_size: float = 0.50,  # 50% 포지션 크기
        max_single_stock_ratio: float = 0.20  # 단일 종목 최대 비중
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
    
    def get_name(self) -> str:
        return "hybrid_portfolio"
    
    def get_description(self) -> str:
        return f"Hybrid Portfolio (Korean {self.korean_aggressive_ratio:.0%} + Portfolio {self.production_portfolio_ratio:.0%})"
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """백테스트 실행"""
        
        if not silent:
            print(f"\n{'='*60}")
            print(f"🔥 {self.get_description()} 백테스트 시작")
            print(f"{'='*60}")
        
        # 초기 자본 설정
        initial_capital = 1_000_000  # 100만원
        korean_capital = initial_capital * self.korean_aggressive_ratio
        portfolio_capital = initial_capital * self.production_portfolio_ratio
        
        # 전략별 포지션 관리
        korean_positions = {}  # {ticker: {qty, entry_px, entry_date, strategy}}
        portfolio_positions = {}
        
        # 거래 기록
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
        
        # 백테스트 시작 (60일 warming up)
        start_idx = 60
        
        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            
            # 🔥 1. 매도 신호 체크 (먼저 처리)
            self._check_sell_signals(
                current_date, 
                korean_positions, 
                portfolio_positions,
                enriched, 
                dates, 
                i, 
                trade_log,
                silent
            )
            
            # 🔥 2. 현재 자본 계산
            korean_cash = korean_capital
            portfolio_cash = portfolio_capital
            
            # 포지션 평가액 차감
            for pos in korean_positions.values():
                korean_cash -= pos['qty'] * pos['entry_px']
            for pos in portfolio_positions.values():
                portfolio_cash -= pos['qty'] * pos['entry_px']
            
            # 🔥 3. 매수 신호 체크
            # Korean Aggressive 전략
            if len(korean_positions) < self.korean_max_positions and korean_cash > 0:
                self._check_korean_aggressive_buy(
                    current_date,
                    korean_positions,
                    portfolio_positions,
                    enriched,
                    korean_cash,
                    initial_capital,
                    trade_log,
                    silent
                )
            
            # Production Portfolio 전략
            if len(portfolio_positions) < self.portfolio_max_positions and portfolio_cash > 0:
                self._check_production_portfolio_buy(
                    current_date,
                    korean_positions,
                    portfolio_positions,
                    enriched,
                    portfolio_cash,
                    initial_capital,
                    trade_log,
                    silent
                )
            
            # 🔥 4. 포트폴리오 가치 계산
            total_value = self._calculate_portfolio_value(
                current_date,
                korean_positions,
                portfolio_positions,
                enriched,
                korean_capital,
                portfolio_capital
            )
            
            equity_curve.append({
                'date': current_date,
                'equity': total_value
            })
        
        # 🔥 5. 최종 청산
        final_date = dates[-1]
        self._liquidate_all_positions(
            final_date,
            korean_positions,
            portfolio_positions,
            enriched,
            trade_log,
            silent
        )
        
        # 결과 정리
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        if not silent:
            print(f"\n✅ 백테스트 완료: {len(trade_log)}개 거래")
            print(f"   Korean Aggressive 거래: {sum(1 for t in trade_log if t.get('strategy') == 'korean_aggressive')}개")
            print(f"   Production Portfolio 거래: {sum(1 for t in trade_log if t.get('strategy') == 'production_portfolio')}개")
        
        return equity_df, trade_log
    
    def _check_sell_signals(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        dates: list,
        current_idx: int,
        trade_log: list,
        silent: bool
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
            
            # 손익률 계산
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 보유 기간
            days_held = (current_date - entry_date).days
            
            # 매도 조건
            reason = None
            
            # 1. 손절 (-10%)
            if pnl_pct <= -0.10:
                reason = "손절"
            
            # 2. 익절 (+20%)
            elif pnl_pct >= 0.20:
                reason = "익절"
            
            # 3. 기간 만료
            elif days_held >= 10 and pnl_pct >= 0.05:
                reason = "기간만료_수익실현"
            elif days_held >= 20:
                reason = "기간만료"
            
            if reason:
                to_sell.append((ticker, pos, current_price, reason))
        
        # 매도 실행
        for ticker, pos, current_price, reason in to_sell:
            strategy = pos['source']
            qty = pos['qty']
            entry_price = pos['entry_px']
            
            # 매도금 계산
            sell_amount = qty * current_price
            commission = sell_amount * self.fee
            tax = sell_amount * self.tax
            net_amount = sell_amount - commission - tax
            
            # 매수 비용
            buy_cost = qty * entry_price * (1 + self.fee)
            
            # 손익
            pnl = net_amount - buy_cost
            
            # 거래 기록
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
            
            # 포지션 제거
            if strategy == 'korean_aggressive':
                del korean_positions[ticker]
            else:
                del portfolio_positions[ticker]
            
            if not silent:
                print(f"  💰 매도: {ticker} ({strategy}) @ {current_price:,.0f}원 x {qty}주 = {pnl:+,.0f}원 ({reason})")
    
    def _check_korean_aggressive_buy(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        korean_cash: float,
        initial_capital: float,
        trade_log: list,
        silent: bool
    ):
        """Korean Aggressive 매수 신호 체크"""
        
        # 후보 종목 스캔
        candidates = []
        
        for ticker, df in enriched.items():
            if current_date not in df.index:
                continue
            
            # 이미 보유 중이면 제외
            if ticker in korean_positions or ticker in portfolio_positions:
                continue
            
            row = df.loc[current_date]
            
            # 기본 필터
            close_price = row['close']
            volume = row['volume']
            
            if close_price < 1000 or close_price > 100000:
                continue
            if volume < 50000:
                continue
            
            # 기술적 지표
            rsi = row.get('rsi', 50)
            ma5 = row.get('ma5', close_price)
            
            # Korean Aggressive 조건
            # 1. RSI 40-85
            if rsi <= 40 or rsi >= 85:
                continue
            
            # 2. 가격 > MA5
            if close_price <= ma5:
                continue
            
            # 3. GAP 조건 (전일 대비 변화율)
            gap_pct = abs(row.get('returns', 0) * 100)
            if gap_pct <= 0.5:
                continue
            
            # 후보 추가
            candidates.append({
                'ticker': ticker,
                'price': close_price,
                'volume': volume,
                'rsi': rsi,
                'score': volume * (1 + gap_pct)  # 거래량 * 갭 점수
            })
        
        if not candidates:
            return
        
        # 거래량 순 정렬
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 최상위 종목 매수 시도
        for candidate in candidates[:3]:  # 상위 3개 시도
            ticker = candidate['ticker']
            price = candidate['price']
            
            # 종목 중복 체크
            if not self._should_allow_duplicate(
                ticker,
                korean_positions,
                portfolio_positions,
                price,
                korean_cash,
                initial_capital
            ):
                continue
            
            # 포지션 크기 계산
            position_size = korean_cash * self.korean_position_size
            qty = int(position_size / price)
            
            if qty <= 0:
                continue
            
            # 매수 실행
            buy_cost = qty * price * (1 + self.fee)
            
            if buy_cost <= korean_cash:
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
                
                break  # 한 번에 하나씩만 매수
    
    def _check_production_portfolio_buy(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        portfolio_cash: float,
        initial_capital: float,
        trade_log: list,
        silent: bool
    ):
        """Production Portfolio 매수 신호 체크"""
        
        # 후보 종목 스캔
        candidates = []
        
        for ticker, df in enriched.items():
            if current_date not in df.index:
                continue
            
            # 이미 보유 중이면 제외
            if ticker in korean_positions or ticker in portfolio_positions:
                continue
            
            row = df.loc[current_date]
            
            # 기본 필터
            close_price = row['close']
            volume = row['volume']
            
            if close_price < 3000 or close_price > 500000:
                continue
            if volume < 30000:
                continue
            
            # 기술적 지표
            rsi = row.get('rsi', 50)
            ma20 = row.get('ma20', close_price)
            
            # Production Portfolio 조건
            # 1. RSI 35-75
            if rsi <= 35 or rsi >= 75:
                continue
            
            # 2. 가격 > MA20 * 0.98
            if close_price <= ma20 * 0.98:
                continue
            
            # 3. 가격 필터 (8,000원 이상)
            if close_price <= 8000:
                continue
            
            # 후보 추가
            candidates.append({
                'ticker': ticker,
                'price': close_price,
                'volume': volume,
                'rsi': rsi,
                'score': volume  # 거래량 기준
            })
        
        if not candidates:
            return
        
        # 거래량 순 정렬
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 최상위 종목 매수 시도
        for candidate in candidates[:3]:  # 상위 3개 시도
            ticker = candidate['ticker']
            price = candidate['price']
            
            # 종목 중복 체크
            if not self._should_allow_duplicate(
                ticker,
                korean_positions,
                portfolio_positions,
                price,
                portfolio_cash,
                initial_capital
            ):
                continue
            
            # 포지션 크기 계산
            position_size = portfolio_cash * self.portfolio_position_size
            qty = int(position_size / price)
            
            if qty <= 0:
                continue
            
            # 매수 실행
            buy_cost = qty * price * (1 + self.fee)
            
            if buy_cost <= portfolio_cash:
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
                
                break  # 한 번에 하나씩만 매수
    
    def _should_allow_duplicate(
        self,
        ticker: str,
        korean_positions: dict,
        portfolio_positions: dict,
        price: float,
        cash: float,
        initial_capital: float
    ) -> bool:
        """종목 중복 체크 (전체 포트폴리오 관점)"""
        
        # 현재 보유 중인지 확인
        if ticker in korean_positions or ticker in portfolio_positions:
            return False
        
        # 전체 포트폴리오 가치 계산
        total_value = initial_capital  # 근사값
        
        # 새로운 포지션 추가시 비중 계산
        proposed_position_value = price * int(cash * 0.5 / price)
        proposed_ratio = proposed_position_value / total_value
        
        # 단일 종목 최대 비중 체크
        if proposed_ratio > self.max_single_stock_ratio:
            return False
        
        return True
    
    def _calculate_portfolio_value(
        self,
        current_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        korean_capital: float,
        portfolio_capital: float
    ) -> float:
        """전체 포트폴리오 가치 계산"""
        
        total_value = 0.0
        
        # Korean Aggressive 포지션 평가
        korean_position_value = 0.0
        for ticker, pos in korean_positions.items():
            if ticker in enriched and current_date in enriched[ticker].index:
                current_price = enriched[ticker].loc[current_date, 'close']
                korean_position_value += pos['qty'] * current_price
            else:
                korean_position_value += pos['qty'] * pos['entry_px']
        
        # Production Portfolio 포지션 평가
        portfolio_position_value = 0.0
        for ticker, pos in portfolio_positions.items():
            if ticker in enriched and current_date in enriched[ticker].index:
                current_price = enriched[ticker].loc[current_date, 'close']
                portfolio_position_value += pos['qty'] * current_price
            else:
                portfolio_position_value += pos['qty'] * pos['entry_px']
        
        # 현금 계산
        korean_cash = korean_capital
        for pos in korean_positions.values():
            korean_cash -= pos['qty'] * pos['entry_px']
        
        portfolio_cash = portfolio_capital
        for pos in portfolio_positions.values():
            portfolio_cash -= pos['qty'] * pos['entry_px']
        
        total_value = korean_position_value + portfolio_position_value + korean_cash + portfolio_cash
        
        return total_value
    
    def _liquidate_all_positions(
        self,
        final_date: pd.Timestamp,
        korean_positions: dict,
        portfolio_positions: dict,
        enriched: dict,
        trade_log: list,
        silent: bool
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
            
            # 매도금 계산
            sell_amount = qty * final_price
            commission = sell_amount * self.fee
            tax = sell_amount * self.tax
            net_amount = sell_amount - commission - tax
            
            # 매수 비용
            buy_cost = qty * entry_price * (1 + self.fee)
            
            # 손익
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
            
            if not silent:
                print(f"  🔚 최종청산: {ticker} ({strategy}) @ {final_price:,.0f}원 x {qty}주 = {pnl:+,.0f}원")

