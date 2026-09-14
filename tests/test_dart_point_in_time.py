import pandas as pd
import pytest

from dart_point_in_time import (
    DartPointInTimeError,
    annual_fundamentals_asof,
    parse_company_search,
    parse_consolidated_financial_statements,
    parse_ordinary_issued_shares,
    parse_report_sections,
    select_report_section,
)


def test_company_search_preserves_original_receipt_and_availability_date():
    html = """
    <table><tbody id="tbody"><tr>
      <td>1</td><td><a onclick="openCorpInfoNew('00126380', 'x', 'y');">삼성전자</a></td>
      <td><a href="/dsaf001/main.do?rcpNo=20190401004781">사업보고서 (2018.12)</a></td>
      <td>삼성전자</td><td>2019.04.01</td><td>연</td>
    </tr><tr>
      <td>2</td><td><a onclick="openCorpInfoNew('00126380', 'x', 'y');">삼성전자</a></td>
      <td><a href="/dsaf001/main.do?rcpNo=20190501000001">[기재정정] 사업보고서 (2018.12)</a></td>
      <td>삼성전자</td><td>2019.05.01</td><td></td>
    </tr></tbody></table>
    """

    reports = parse_company_search(html, "5930")

    assert len(reports) == 1
    assert reports[0].ticker == "005930"
    assert reports[0].corp_code == "00126380"
    assert reports[0].receipt_no == "20190401004781"
    assert reports[0].period_end == "2018-12-31"
    assert reports[0].available_date == "2019-04-02"


def test_report_tree_selects_statements_and_shares_without_notes():
    html = """
    var node2 = {};
    node2['text'] = "2. 연결재무제표";
    node2['rcpNo'] = "20190401004781";
    node2['dcmNo'] = "6616741";
    node2['eleId'] = "13";
    node2['offset'] = "625579";
    node2['length'] = "120141";
    node2['dtd'] = "dart3.xsd";
    var node2 = {};
    node2['text'] = "3. 연결재무제표 주석";
    node2['rcpNo'] = "20190401004781";
    node2['dcmNo'] = "6616741";
    node2['eleId'] = "14";
    node2['offset'] = "745724";
    node2['length'] = "586931";
    node2['dtd'] = "dart3.xsd";
    var node2 = {};
    node2['text'] = "4. 주식의 총수 등";
    node2['rcpNo'] = "20190401004781";
    node2['dcmNo'] = "6616741";
    node2['eleId'] = "7";
    node2['offset'] = "247247";
    node2['length'] = "63793";
    node2['dtd'] = "dart3.xsd";
    """
    sections = parse_report_sections(html)

    statements = select_report_section(sections, "consolidated_financial_statements")
    shares = select_report_section(sections, "issued_shares")

    assert statements.element_id == "13"
    assert "eleId=13" in statements.viewer_url
    assert shares.element_id == "7"


def test_statement_parser_normalizes_units_and_parenthesized_losses():
    html = """
    <p>(단위 : 백만원)</p>
    <table><tr><th></th><th>제 50 기</th><th>제 49 기</th></tr>
      <tr><td>자산총계</td><td>339,357,244</td><td>301,752,090</td></tr>
      <tr><td>부채총계</td><td>91,604,067</td><td>87,260,662</td></tr>
      <tr><td>자본총계</td><td>247,753,177</td><td>214,491,428</td></tr></table>
    <table><tr><th></th><th>제 50 기</th><th>제 49 기</th></tr>
      <tr><td>수익(매출액)</td><td>243,771,415</td><td>239,575,376</td></tr>
      <tr><td>영업이익(손실)</td><td>58,886,669</td><td>53,645,038</td></tr>
      <tr><td>당기순이익(손실)</td><td>(44,344,857)</td><td>42,186,747</td></tr></table>
    <table><tr><th></th><th>제 50 기</th><th>제 49 기</th></tr>
      <tr><td>영업활동 현금흐름</td><td>67,031,863</td><td>62,162,041</td></tr></table>
    """

    result = parse_consolidated_financial_statements(html)

    assert result["assets"] == 339_357_244_000_000
    assert result["previous_assets"] == 301_752_090_000_000
    assert result["net_income"] == -44_344_857_000_000
    assert result["cash_flow_from_operations"] == 67_031_863_000_000


def test_statement_parser_accepts_numbered_and_consolidated_labels():
    html = """
    <p>(단위 : 원)</p>
    <table><tr><th></th><th>당기</th><th>전기</th></tr>
      <tr><td>자 산 총 계</td><td>1000</td><td>900</td></tr>
      <tr><td>부 채 총 계</td><td>400</td><td>350</td></tr>
      <tr><td>자 본 총 계</td><td>600</td><td>550</td></tr></table>
    <table><tr><th></th><th>당기</th><th>전기</th></tr>
      <tr><td>Ⅰ. 영업수익</td><td>800</td><td>700</td></tr>
      <tr><td>Ⅱ. 영업이익</td><td>100</td><td>90</td></tr>
      <tr><td>VIII. 연결당기순이익 (준비금 반영 전)</td><td>70</td><td>60</td></tr>
      <tr><td>영업활동으로 인한 순현금흐름</td><td>85</td><td>75</td></tr></table>
    """

    result = parse_consolidated_financial_statements(html)

    assert result["revenue"] == 800
    assert result["operating_income"] == 100
    assert result["net_income"] == 70
    assert result["cash_flow_from_operations"] == 85


def test_share_parser_selects_ordinary_issued_total_not_all_classes():
    html = """
    <table><thead><tr><th rowspan="2">구분</th><th colspan="3">주식의 종류</th></tr>
      <tr><th>보통주</th><th>우선주</th><th>합계</th></tr></thead>
      <tbody><tr><td>Ⅳ. 발행주식의 총수 (Ⅱ-Ⅲ)</td>
      <td>5,969,782,550</td><td>822,886,700</td><td>6,792,669,250</td></tr></tbody>
    </table>
    """

    assert parse_ordinary_issued_shares(html) == 5_969_782_550


def test_share_parser_accepts_legacy_voting_share_header():
    html = """
    <table><thead><tr><th rowspan="2">구분</th><th colspan="3">주식의 종류</th></tr>
      <tr><th>의결권 있는 주식</th><th>의결권 없는 주식</th><th>합계</th></tr></thead>
      <tbody><tr><td>Ⅳ. 발행주식의 총수 (Ⅱ-Ⅲ)</td>
      <td>87,186,835</td><td>-</td><td>87,186,835</td></tr></tbody>
    </table>
    """

    assert parse_ordinary_issued_shares(html) == 87_186_835


def test_asof_hides_future_filing_and_drops_stale_annual_data():
    data = pd.DataFrame(
        {
            "ticker": ["005930", "005930"],
            "period_end": ["2018-12-31", "2019-12-31"],
            "receipt_date": ["2019-04-01", "2020-03-30"],
            "available_date": ["2019-04-02", "2020-03-31"],
            "receipt_no": ["old", "future"],
        }
    )

    march = annual_fundamentals_asof(data, "2020-03-30")
    april = annual_fundamentals_asof(data, "2020-04-01")

    assert march.loc["005930", "receipt_no"] == "old"
    assert april.loc["005930", "receipt_no"] == "future"
    assert annual_fundamentals_asof(data.iloc[:1], "2021-01-01", max_age_days=550).empty


def test_asof_rejects_same_day_availability():
    data = pd.DataFrame(
        {
            "ticker": ["005930"],
            "period_end": ["2019-12-31"],
            "receipt_date": ["2020-03-30"],
            "available_date": ["2020-03-30"],
            "receipt_no": ["leak"],
        }
    )
    with pytest.raises(DartPointInTimeError, match="must be after"):
        annual_fundamentals_asof(data, "2020-04-01")
