#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국 시장 데이터 품질 종합 점검
- Universe 커버리지
- 데이터 완전성
- 시계열 연속성
- 상장폐지/관리종목 처리
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pykrx.stock as stock

from cache_manager import load_enriched
try:
    from config import START, END
except:
    START = "2020-01-01"
    END = None

print("=" * 80)
print("📊 한국 시장 데이터 품질 종합 점검")
print("=" * 80)

# =============================================================================
# 1. Universe 범위 확인
# =============================================================================
print("\n1️⃣ Universe 범위 확인")
print("-" * 80)

try:
    # 현재 상장 종목 수
    kospi_tickers = stock.get_market_ticker_list(market="KOSPI")
    kosdaq_tickers = stock.get_market_ticker_list(market="KOSDAQ")
    konex_tickers = stock.get_market_ticker_list(market="KONEX")
    
    print(f"✅ KOSPI 종목: {len(kospi_tickers):,}개")
    print(f"✅ KOSDAQ 종목: {len(kosdaq_tickers):,}개")
    print(f"✅ KONEX 종목: {len(konex_tickers):,}개")
    print(f"   총계: {len(kospi_tickers) + len(kosdaq_tickers) + len(konex_tickers):,}개")
    
    all_tickers = set(kospi_tickers + kosdaq_tickers + konex_tickers)
    
except Exception as e:
    print(f"❌ Universe 조회 실패: {e}")
    all_tickers = set()

# =============================================================================
# 2. 캐시된 데이터 확인
# =============================================================================
print("\n2️⃣ 캐시된 데이터 확인")
print("-" * 80)

import glob
import os

# 올바른 경로 사용
enriched_dir = "data/enriched"
ohlcv_dir = "data/ohlcv"

enriched_files = glob.glob(f"{enriched_dir}/*_*.parquet")

print(f"캐시된 enriched 파일: {len(enriched_files):,}개")

# 파일별 날짜 범위 추출
cached_tickers = set()
date_ranges = {}
file_sizes = []

for f in enriched_files:
    basename = os.path.basename(f)
    parts = basename.replace(".parquet", "").split("_")
    if len(parts) >= 3:
        ticker = parts[0]
        cached_tickers.add(ticker)
        
        # 파일 크기
        size_mb = os.path.getsize(f) / (1024 * 1024)
        file_sizes.append(size_mb)
        
        # 날짜 범위 (파일명에서 추출)
        if len(parts) == 3:
            start_date = parts[1]
            end_date = parts[2]
            date_ranges[ticker] = (start_date, end_date)

print(f"캐시된 종목: {len(cached_tickers):,}개")
print(f"평균 파일 크기: {np.mean(file_sizes):.2f} MB")
print(f"총 캐시 크기: {np.sum(file_sizes):.2f} MB")

# Universe와 캐시 비교
if all_tickers:
    cached_ratio = len(cached_tickers) / len(all_tickers) * 100
    print(f"\n캐시 커버리지: {cached_ratio:.1f}% ({len(cached_tickers)}/{len(all_tickers)})")
    
    missing = all_tickers - cached_tickers
    if len(missing) > 0:
        print(f"⚠️  캐시 누락 종목: {len(missing):,}개")
        print(f"   예시: {list(missing)[:10]}")

# =============================================================================
# 3. 데이터 완전성 점검 (샘플링)
# =============================================================================
print("\n3️⃣ 데이터 완전성 점검 (샘플 100개)")
print("-" * 80)

sample_tickers = list(cached_tickers)[:100] if len(cached_tickers) > 100 else list(cached_tickers)

completeness_report = {
    "total": 0,
    "has_ohlcv": 0,
    "has_volume": 0,
    "has_indicators": 0,
    "missing_data": 0,
    "date_gaps": 0,
    "zero_volume_days": [],
}

expected_columns = ["open", "high", "low", "close", "volume"]
indicator_columns = ["rsi14", "ma20", "ma60"]

for ticker in sample_tickers:
    completeness_report["total"] += 1
    
    df = load_enriched(ticker, START, END)
    
    if df is None or len(df) == 0:
        completeness_report["missing_data"] += 1
        continue
    
    # OHLCV 존재 여부
    if all(col in df.columns for col in expected_columns):
        completeness_report["has_ohlcv"] += 1
    
    # Volume 데이터 품질
    if "volume" in df.columns:
        vol_series = df["volume"]
        if vol_series.notna().sum() > 0:
            completeness_report["has_volume"] += 1
            
            # 거래량 0인 날 비율
            zero_vol_ratio = (vol_series == 0).sum() / len(vol_series)
            if zero_vol_ratio > 0.1:  # 10% 이상
                completeness_report["zero_volume_days"].append((ticker, zero_vol_ratio))
    
    # 지표 존재 여부
    if any(col in df.columns for col in indicator_columns):
        completeness_report["has_indicators"] += 1
    
    # 날짜 연속성 (거래일 기준)
    if len(df) > 1:
        date_diffs = df.index.to_series().diff().dt.days
        # 거래일 기준 5일 이상 갭은 비정상
        long_gaps = (date_diffs > 5).sum()
        if long_gaps > 0:
            completeness_report["date_gaps"] += 1

print(f"총 샘플: {completeness_report['total']:,}개")
print(f"OHLCV 완전: {completeness_report['has_ohlcv']:,}개 ({completeness_report['has_ohlcv']/completeness_report['total']*100:.1f}%)")
print(f"Volume 데이터: {completeness_report['has_volume']:,}개 ({completeness_report['has_volume']/completeness_report['total']*100:.1f}%)")
print(f"지표 계산됨: {completeness_report['has_indicators']:,}개 ({completeness_report['has_indicators']/completeness_report['total']*100:.1f}%)")
print(f"데이터 누락: {completeness_report['missing_data']:,}개")
print(f"날짜 갭 발견: {completeness_report['date_gaps']:,}개")

if len(completeness_report["zero_volume_days"]) > 0:
    print(f"\n⚠️  거래량 0 비율 높은 종목 (>10%): {len(completeness_report['zero_volume_days'])}개")
    for ticker, ratio in completeness_report["zero_volume_days"][:5]:
        print(f"   {ticker}: {ratio*100:.1f}%")

# =============================================================================
# 4. 시계열 범위 점검
# =============================================================================
print("\n4️⃣ 시계열 범위 점검")
print("-" * 80)

time_ranges = []
for ticker in sample_tickers[:50]:
    df = load_enriched(ticker, START, END)
    if df is not None and len(df) > 0:
        time_ranges.append({
            "ticker": ticker,
            "start": df.index[0],
            "end": df.index[-1],
            "days": len(df),
        })

if len(time_ranges) > 0:
    df_ranges = pd.DataFrame(time_ranges)
    
    print(f"최소 시작일: {df_ranges['start'].min().date()}")
    print(f"최대 종료일: {df_ranges['end'].max().date()}")
    print(f"평균 데이터 일수: {df_ranges['days'].mean():.0f}일")
    print(f"최소 데이터 일수: {df_ranges['days'].min():.0f}일")
    print(f"최대 데이터 일수: {df_ranges['days'].max():.0f}일")
    
    # 최근 데이터 신선도
    today = pd.Timestamp.now()
    latest_end = df_ranges['end'].max()
    days_old = (today - latest_end).days
    
    print(f"\n데이터 신선도:")
    print(f"  최근 데이터: {latest_end.date()}")
    print(f"  오늘 기준: {days_old}일 전")
    
    if days_old > 3:
        print(f"  ⚠️  데이터가 {days_old}일 오래됨. 업데이트 필요!")

# =============================================================================
# 5. 시가총액 데이터 확인
# =============================================================================
print("\n5️⃣ 시가총액 데이터 확인")
print("-" * 80)

market_cap_count = 0
for ticker in sample_tickers[:50]:
    df = load_enriched(ticker, START, END)
    if df is not None and "market_cap" in df.columns:
        if df["market_cap"].notna().sum() > 0:
            market_cap_count += 1

print(f"시가총액 데이터 보유: {market_cap_count}/50 ({market_cap_count/50*100:.1f}%)")

if market_cap_count == 0:
    print("⚠️  시가총액 데이터 없음 - Universe 필터링 제한됨!")

# =============================================================================
# 6. 상장폐지/관리종목 처리 확인
# =============================================================================
print("\n6️⃣ 상장폐지/관리종목 처리")
print("-" * 80)

# 캐시에는 있지만 현재 상장 종목 목록에 없는 종목
if all_tickers:
    delisted = cached_tickers - all_tickers
    print(f"상장폐지/제외 종목 (캐시에만 존재): {len(delisted):,}개")
    if len(delisted) > 0:
        print(f"  예시: {list(delisted)[:10]}")
        print(f"  ✅ 과거 백테스트에 반영됨 (Survivorship Bias 최소화)")

# =============================================================================
# 7. 종합 점수
# =============================================================================
print("\n" + "=" * 80)
print("📊 종합 평가")
print("=" * 80)

scores = []

# Universe 커버리지
if all_tickers and len(cached_tickers) > 0:
    coverage_score = min(100, cached_ratio)
    scores.append(("Universe 커버리지", coverage_score))
else:
    scores.append(("Universe 커버리지", 0))

# 데이터 완전성
if completeness_report["total"] > 0:
    completeness_score = (completeness_report["has_ohlcv"] / completeness_report["total"]) * 100
    scores.append(("데이터 완전성", completeness_score))

# 지표 계산
if completeness_report["total"] > 0:
    indicator_score = (completeness_report["has_indicators"] / completeness_report["total"]) * 100
    scores.append(("지표 계산", indicator_score))

# 시계열 연속성
if completeness_report["total"] > 0:
    continuity_score = ((completeness_report["total"] - completeness_report["date_gaps"]) / completeness_report["total"]) * 100
    scores.append(("시계열 연속성", continuity_score))

# 데이터 신선도
if 'days_old' in locals():
    if days_old <= 1:
        freshness_score = 100
    elif days_old <= 3:
        freshness_score = 80
    elif days_old <= 7:
        freshness_score = 60
    else:
        freshness_score = 40
    scores.append(("데이터 신선도", freshness_score))

print("\n개별 점수:")
for name, score in scores:
    status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
    print(f"  {status} {name}: {score:.1f}/100")

if len(scores) > 0:
    overall_score = np.mean([s[1] for s in scores])
    print(f"\n종합 점수: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("✅ 데이터 품질: 우수 - 백테스트 신뢰 가능")
    elif overall_score >= 60:
        print("⚠️  데이터 품질: 양호 - 주의하여 사용")
    else:
        print("❌ 데이터 품질: 불량 - 데이터 업데이트 필요")

# =============================================================================
# 8. 권장 사항
# =============================================================================
print("\n" + "=" * 80)
print("💡 권장 사항")
print("=" * 80)

recommendations = []

if all_tickers and cached_ratio < 90:
    recommendations.append("• Universe 커버리지 향상을 위해 전체 데이터 재수집 권장")

if completeness_report["missing_data"] > 10:
    recommendations.append("• 데이터 누락 종목 재수집 필요")

if completeness_report["date_gaps"] > 5:
    recommendations.append("• 시계열 갭이 있는 종목들 데이터 보완 필요")

if 'days_old' in locals() and days_old > 3:
    recommendations.append(f"• 데이터가 {days_old}일 오래됨 - 즉시 업데이트 필요")

if market_cap_count < 40:
    recommendations.append("• 시가총액 데이터 수집 로직 추가 권장 (Universe 필터링 개선)")

if len(completeness_report["zero_volume_days"]) > 10:
    recommendations.append("• 거래량 0 종목 많음 - 상장폐지 또는 데이터 오류 확인")

if len(recommendations) > 0:
    for rec in recommendations:
        print(rec)
else:
    print("✅ 현재 데이터 품질 우수 - 추가 조치 불필요")

print("\n" + "=" * 80)
print("점검 완료!")
print("=" * 80)

