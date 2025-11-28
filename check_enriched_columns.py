#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enriched 데이터셋의 컬럼 확인 스크립트
"""

import pandas as pd
from cache_manager import load_enriched
from data_loader import get_universe, load_panel
from signals import compute_indicators, add_rel_strength
from data_loader import get_index_close, infer_market

# 필요한 컬럼 목록
required_columns = ['volume', 'close', 'rsi', 'ma10', 'ma20', 'returns']

print("=" * 80)
print("🔍 Enriched 데이터셋 컬럼 확인")
print("=" * 80)

# 샘플 종목 몇 개 로드
print("\n1️⃣ 캐시에서 샘플 데이터 로드 중...")
universe = get_universe(["KOSPI", "KOSDAQ"], include_etf=False, include_index_etf=False)
sample_tickers = list(universe)[:5]  # 처음 5개만

all_columns = set()
missing_columns = {col: [] for col in required_columns}
found_columns = {col: [] for col in required_columns}

for ticker in sample_tickers:
    # 캐시에서 로드 시도
    cached = load_enriched(ticker, "2020-01-02", "2025-11-19")
    
    if cached is not None and not cached.empty:
        columns = set(cached.columns)
        all_columns.update(columns)
        
        print(f"\n📊 {ticker}:")
        print(f"   전체 컬럼 수: {len(columns)}")
        print(f"   컬럼 목록: {sorted(columns)}")
        
        # 필요한 컬럼 확인
        for col in required_columns:
            if col in columns:
                found_columns[col].append(ticker)
                print(f"   ✅ {col}: 존재")
            else:
                missing_columns[col].append(ticker)
                print(f"   ❌ {col}: 없음")
        
        # 샘플 데이터 확인
        if len(cached) > 0:
            print(f"\n   샘플 데이터 (마지막 3행):")
            print(cached[['close', 'volume']].tail(3) if 'close' in cached.columns and 'volume' in cached.columns else "   (close/volume 없음)")
    else:
        print(f"\n⚠️ {ticker}: 캐시에 데이터 없음")

print("\n" + "=" * 80)
print("📋 전체 컬럼 요약")
print("=" * 80)

print(f"\n전체 발견된 컬럼 ({len(all_columns)}개):")
for col in sorted(all_columns):
    print(f"  - {col}")

print("\n" + "=" * 80)
print("✅ 필수 컬럼 존재 여부")
print("=" * 80)

for col in required_columns:
    if found_columns[col]:
        print(f"✅ {col}: {len(found_columns[col])}개 종목에서 발견")
    else:
        print(f"❌ {col}: 없음")

# 실제 데이터 생성 과정 확인
print("\n" + "=" * 80)
print("🔧 compute_indicators로 생성되는 컬럼 확인")
print("=" * 80)

# 샘플 OHLCV 데이터 로드
panel = load_panel(sample_tickers[:1], "2024-01-01", "2024-12-31", max_workers=1)
if panel:
    ticker, df = list(panel.items())[0]
    print(f"\n원본 OHLCV 데이터 ({ticker}):")
    print(f"  컬럼: {list(df.columns)}")
    
    # compute_indicators 실행
    df_with_indicators = compute_indicators(df)
    if not df_with_indicators.empty:
        print(f"\ncompute_indicators 후:")
        print(f"  컬럼: {list(df_with_indicators.columns)}")
        
        # RSI, MA 계산 확인
        print(f"\n  'ret' 컬럼 존재: {'ret' in df_with_indicators.columns}")
        print(f"  'rsi' 컬럼 존재: {'rsi' in df_with_indicators.columns}")
        print(f"  'ma10' 컬럼 존재: {'ma10' in df_with_indicators.columns}")
        print(f"  'ma20' 컬럼 존재: {'ma20' in df_with_indicators.columns}")
        
        # RSI, MA가 없으면 계산해보기
        if 'rsi' not in df_with_indicators.columns:
            print("\n  ⚠️ RSI가 없습니다. 계산 방법 확인 필요")
        if 'ma10' not in df_with_indicators.columns:
            print("  ⚠️ MA10이 없습니다. 계산 방법 확인 필요")
        if 'ma20' not in df_with_indicators.columns:
            print("  ⚠️ MA20이 없습니다. 계산 방법 확인 필요")
        
        # returns vs ret 확인
        if 'returns' not in df_with_indicators.columns and 'ret' in df_with_indicators.columns:
            print("\n  ℹ️ 'returns'는 없지만 'ret' 컬럼이 있습니다 (pct_change 결과)")

print("\n" + "=" * 80)
print("💡 결론")
print("=" * 80)

print("\n필수 컬럼 상태:")
for col in required_columns:
    status = "✅ 존재" if found_columns[col] else "❌ 없음"
    print(f"  {col}: {status}")

if not all(found_columns[col] for col in required_columns):
    print("\n⚠️ 일부 컬럼이 없습니다. 전략 코드에서 이 컬럼들을 계산하거나")
    print("   signals.py의 compute_indicators에 추가해야 할 수 있습니다.")

