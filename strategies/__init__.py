#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategies 모듈 - 백테스팅 전략 레지스트리
"""

from typing import Dict, List, Type
from strategies.base_strategy import BaseStrategy
from strategies.strategy_baseline import BaselineStrategy
from strategies.strategy_sector_weighted import SectorWeightedStrategy
from strategies.strategy_improved import ImprovedStrategy
from strategies.strategy_reversal import ReversalStrategy
from strategies.strategy_kqm import KQMStrategy
from strategies.strategy_kqm_v2 import KQMStrategyV2
from strategies.strategy_kqm_v3 import KQMStrategyV3
from strategies.strategy_kqm_v3_1 import KQMStrategyV3_1
from strategies.strategy_kqm_v3_2 import KQMStrategyV3_2
from strategies.strategy_kqm_small_cap import KQMSmallCapStrategy
from strategies.strategy_ksms import KSMSStrategy
from strategies.k200_mean_reversion import K200MeanReversion
from strategies.strategy_kqm_small_cap_v2 import KQMSmallCapStrategyV21
from strategies.kmr_midcap_reversion import KMRMidcapReversion
from strategies.strategy_kqm_small_cap_v3 import KQMSmallCapStrategyV3
from strategies.strategy_kqm_small_cap_v3_2 import KQMSmallCapStrategyV32
from strategies.br_pullback import BreakoutPullbackStrategy
from strategies.event_swing import EventSwingStrategy
from strategies.mr_fast import MRFastStrategy
from strategies.strategy_hybrid import HybridPortfolioStrategy
from strategies.strategy_korean_aggressive import KoreanAggressiveStrategy
from strategies.strategy_production_portfolio import ProductionPortfolioStrategy
from strategies.strategy_kqm_small_cap_v2_2 import KQMSmallCapStrategyV22
from strategies.strategy_kqm_small_cap_v2_2_regime import KQMSmallCapStrategyV22Regime
from strategies.strategy_kqm_small_cap_v2_2_short import KQMSmallCapStrategyV22Short
from strategies.strategy_kqm_small_cap_v2_2_blend import KQMSmallCapStrategyV22Blend
from strategies.strategy_kqm_small_cap_v2_2_hybrid import KQMSmallCapStrategyV22Hybrid
from strategies.strategy_multi_allocator import MultiStrategyAllocator
from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus
from strategies.strategy_multi_allocator_plus_v2 import MultiStrategyAllocatorPlusV2
from strategies.hybrid_portfolio_v2_4 import HybridPortfolioStrategyV24
from strategies.strategy_etf_defensive import ETFRiskOverlayStrategy
from strategies.strategy_multi_allocator_plus_no_etf import MultiStrategyAllocatorPlusNoETF

# 전략 레지스트리
_strategy_registry: Dict[str, BaseStrategy] = {}


def _register_strategies():
    """전략 자동 등록"""
    global _strategy_registry
    
    # 기본 전략 클래스 등록 (인스턴스가 아닌 클래스)
    strategy_classes = [
        BaselineStrategy,
        SectorWeightedStrategy,
        ImprovedStrategy,
        ReversalStrategy, 
        KQMStrategy,
        KQMStrategyV2,
        KQMStrategyV3,
        KQMStrategyV3_1,
        KQMStrategyV3_2,
        KQMSmallCapStrategy,
        KSMSStrategy,
        K200MeanReversion,
        KQMSmallCapStrategyV21,
        KQMSmallCapStrategyV22,
        KQMSmallCapStrategyV22Regime,
        KQMSmallCapStrategyV22Short,
        KQMSmallCapStrategyV22Blend,
        KQMSmallCapStrategyV22Hybrid,
        MultiStrategyAllocator,
        MultiStrategyAllocatorPlusV2,
        KMRMidcapReversion,
        KQMSmallCapStrategyV3,
        KQMSmallCapStrategyV32,
        BreakoutPullbackStrategy,
        EventSwingStrategy,
        MRFastStrategy,
        HybridPortfolioStrategy,
        KoreanAggressiveStrategy,
        ProductionPortfolioStrategy,
        HybridPortfolioStrategyV24,
        ETFRiskOverlayStrategy,
        MultiStrategyAllocatorPlus,
        MultiStrategyAllocatorPlusNoETF
    ]
    
    for strategy_class in strategy_classes:
        # 임시 인스턴스로 이름 확인
        temp_instance = strategy_class()
        name = temp_instance.get_name()
        
        # 클래스 자체를 저장
        _strategy_registry[name] = strategy_class


def get_strategy(name: str) -> BaseStrategy:
    """
    전략 이름으로 전략 인스턴스 반환
    
    Args:
        name: 전략 이름
        
    Returns:
        BaseStrategy 인스턴스
    """
    if not _strategy_registry:
        _register_strategies()
    
    strategy_class = _strategy_registry.get(name)
    if strategy_class is None:
        available = ", ".join(_strategy_registry.keys())
        raise ValueError(f"전략 '{name}'을 찾을 수 없습니다. 사용 가능: {available}")
    
    # 새 인스턴스 생성
    return strategy_class()


def list_strategies() -> List[tuple]:
    """
    등록된 모든 전략 목록 반환
    
    Returns:
        [(name, description), ...] 형태의 리스트
    """
    if not _strategy_registry:
        _register_strategies()
    
    strategies = []
    for name, strategy_class in _strategy_registry.items():
        # 임시 인스턴스로 설명 가져오기
        temp_instance = strategy_class()
        desc = temp_instance.get_description()
        strategies.append((name, desc))
    
    return strategies


# 모듈 로드 시 자동 등록
_register_strategies()


__all__ = [
    'BaseStrategy',
    'get_strategy',
    'list_strategies',
]
