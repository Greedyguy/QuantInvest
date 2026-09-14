"""Point-in-time annual fundamentals from the public DART filing viewer.

The functions in this module are deliberately split between downloading and
parsing.  Backtests consume only normalized rows whose DART receipt date was
observable by the signal date; network access is never performed implicitly by
the research code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import StringIO
import re
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from lxml import html as lxml_html


DART_BASE_URL = "https://dart.fss.or.kr"


class DartPointInTimeError(ValueError):
    """Raised when a filing cannot be normalized without ambiguity."""


@dataclass(frozen=True)
class DartAnnualReport:
    ticker: str
    corp_code: str
    name: str
    receipt_no: str
    receipt_date: str
    period_end: str

    @property
    def available_date(self) -> str:
        # DART receipt timestamps are not carried by the public result table.
        # Waiting one calendar day therefore fails closed for same-day signals.
        return str((pd.Timestamp(self.receipt_date) + pd.Timedelta(days=1)).date())

    @property
    def main_url(self) -> str:
        return f"{DART_BASE_URL}/dsaf001/main.do?rcpNo={self.receipt_no}"


@dataclass(frozen=True)
class DartDocumentSection:
    text: str
    receipt_no: str
    document_no: str
    element_id: str
    offset: str
    length: str
    dtd: str

    @property
    def viewer_url(self) -> str:
        query = urlencode(
            {
                "rcpNo": self.receipt_no,
                "dcmNo": self.document_no,
                "eleId": self.element_id,
                "offset": self.offset,
                "length": self.length,
                "dtd": self.dtd,
            }
        )
        return f"{DART_BASE_URL}/report/viewer.do?{query}"


_NODE_START = re.compile(r"var node\d+ = \{\};")
_NODE_FIELD = re.compile(
    r"node\d+\['(?P<field>[^']+)'\]\s*=\s*\"(?P<value>.*?)\";"
)


def _compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).replace("\u3000", "")


def parse_company_search(html_text: str, ticker: str) -> list[DartAnnualReport]:
    """Parse original annual-report receipts from a DART company search."""

    ticker = str(ticker).zfill(6)
    root = lxml_html.fromstring(html_text)
    reports: list[DartAnnualReport] = []
    for row in root.xpath("//tbody[@id='tbody']/tr"):
        links = row.xpath(".//a[contains(@href, '/dsaf001/main.do?rcpNo=')]")
        if not links:
            continue
        link = links[0]
        title = " ".join(link.text_content().split())
        if not title.startswith("사업보고서") or "정정" in title:
            continue
        receipt_match = re.search(r"rcpNo=(\d{14})", link.get("href", ""))
        period_match = re.search(r"\((\d{4})\.(\d{2})\)", title)
        corp_anchor = row.xpath(".//a[contains(@onclick, 'openCorpInfoNew')]")
        cells = row.xpath("./td")
        if not receipt_match or not period_match or not corp_anchor or len(cells) < 5:
            continue
        corp_match = re.search(r"openCorpInfoNew\('(\d{8})'", corp_anchor[0].get("onclick", ""))
        if not corp_match:
            continue
        receipt_date = "".join(cells[4].text_content().split()).replace(".", "-")
        period_end = pd.Timestamp(
            year=int(period_match.group(1)), month=int(period_match.group(2)), day=1
        ) + pd.offsets.MonthEnd(0)
        reports.append(
            DartAnnualReport(
                ticker=ticker,
                corp_code=corp_match.group(1),
                name=" ".join(corp_anchor[0].text_content().split()),
                receipt_no=receipt_match.group(1),
                receipt_date=str(pd.Timestamp(receipt_date).date()),
                period_end=str(period_end.date()),
            )
        )
    unique = {report.receipt_no: report for report in reports}
    return sorted(unique.values(), key=lambda report: report.receipt_date)


def parse_report_sections(html_text: str) -> list[DartDocumentSection]:
    """Parse the document-tree records embedded in a DART report page."""

    starts = list(_NODE_START.finditer(html_text))
    sections: list[DartDocumentSection] = []
    for position, start in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(html_text)
        fields = {
            match.group("field"): match.group("value")
            for match in _NODE_FIELD.finditer(html_text[start.start() : end])
        }
        required = {"text", "rcpNo", "dcmNo", "eleId", "offset", "length", "dtd"}
        if not required.issubset(fields):
            continue
        sections.append(
            DartDocumentSection(
                text=fields["text"],
                receipt_no=fields["rcpNo"],
                document_no=fields["dcmNo"],
                element_id=fields["eleId"],
                offset=fields["offset"],
                length=fields["length"],
                dtd=fields["dtd"],
            )
        )
    return sections


def select_report_section(
    sections: list[DartDocumentSection], section_type: str
) -> DartDocumentSection:
    """Select the consolidated statements or issued-share-count section."""

    if section_type == "consolidated_financial_statements":
        candidates = [
            section
            for section in sections
            if "연결재무제표" in _compact(section.text)
            and "주석" not in _compact(section.text)
        ]
    elif section_type == "issued_shares":
        candidates = [
            section for section in sections if "주식의총수" in _compact(section.text)
        ]
    else:
        raise DartPointInTimeError(f"unknown section type: {section_type}")
    if not candidates:
        raise DartPointInTimeError(f"DART section not found: {section_type}")
    return min(candidates, key=lambda section: int(section.element_id))


def _parse_number(value: object) -> float:
    text = _compact(value).replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    number = float(text)
    return -number if negative else number


def _column_text(column: object) -> str:
    if isinstance(column, tuple):
        return " ".join(_compact(part) for part in column if _compact(part) != "nan")
    return _compact(column)


def _first_row_value(
    tables: list[pd.DataFrame], aliases: tuple[str, ...]
) -> tuple[float, float]:
    compact_aliases = {_compact(alias) for alias in aliases}

    def matches(value: object) -> bool:
        label = _compact(value)
        label = re.sub(r"^[0-9IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ.()]+", "", label)
        return any(
            label == alias or label.startswith(f"{alias}(")
            for alias in compact_aliases
        )

    for table in tables:
        for _, row in table.iterrows():
            values = row.tolist()
            label_position = next(
                (
                    index
                    for index, value in enumerate(values)
                    if matches(value)
                ),
                None,
            )
            if label_position is None:
                continue
            numbers = [
                _parse_number(value) for value in values[label_position + 1 :]
            ]
            numbers = [number for number in numbers if np.isfinite(number)]
            if numbers:
                current = numbers[0]
                previous = numbers[1] if len(numbers) > 1 else np.nan
                return current, previous
    return np.nan, np.nan


_STATEMENT_ALIASES = {
    "assets": ("자산총계", "자산 총계"),
    "liabilities": ("부채총계", "부채 총계"),
    "equity": ("자본총계", "자본 총계"),
    "revenue": ("수익(매출액)", "매출액", "영업수익"),
    "operating_income": ("영업이익(손실)", "영업이익", "영업손실"),
    "net_income": (
        "당기순이익(손실)",
        "연결당기순이익",
        "당기순이익",
        "당기순손익",
        "당기순손실",
    ),
    "cash_flow_from_operations": (
        "영업활동 현금흐름",
        "영업활동현금흐름",
        "영업활동으로 인한 현금흐름",
        "영업활동으로 인한 순현금흐름",
        "영업활동으로인한순현금흐름",
    ),
}


def parse_consolidated_financial_statements(html_text: str) -> dict[str, float]:
    """Extract current/prior annual values and normalize them to Korean won."""

    unit_match = re.search(r"단위\s*[:：]\s*([^\)<]+)", html_text)
    unit = _compact(unit_match.group(1)) if unit_match else ""
    scale_by_unit = {"원": 1.0, "천원": 1_000.0, "백만원": 1_000_000.0, "억원": 100_000_000.0}
    if unit not in scale_by_unit:
        raise DartPointInTimeError(f"unsupported or missing statement unit: {unit!r}")
    tables = pd.read_html(StringIO(html_text))
    result: dict[str, float] = {"statement_scale_won": scale_by_unit[unit]}
    for field, aliases in _STATEMENT_ALIASES.items():
        current, previous = _first_row_value(tables, aliases)
        result[field] = current * scale_by_unit[unit]
        result[f"previous_{field}"] = previous * scale_by_unit[unit]
    required = ("assets", "liabilities", "equity", "net_income")
    if any(not np.isfinite(result[field]) for field in required):
        raise DartPointInTimeError("required consolidated statement values are missing")
    return result


def parse_ordinary_issued_shares(html_text: str) -> float:
    """Extract issued ordinary/common shares used by KRX market capitalization."""

    tables = pd.read_html(StringIO(html_text))
    for table in tables:
        columns = [_column_text(column) for column in table.columns]
        common_positions = [
            index
            for index, name in enumerate(columns)
            if "보통주" in name and "우선" not in name
        ]
        if not common_positions:
            continue
        for _, row in table.iterrows():
            values = row.tolist()
            if not any("발행주식의총수" in _compact(value) for value in values):
                continue
            for position in common_positions:
                number = _parse_number(values[position])
                if np.isfinite(number) and number > 0:
                    return number
    raise DartPointInTimeError("ordinary issued-share count not found")


def normalize_annual_filing(
    report: DartAnnualReport,
    statement_html: str,
    shares_html: str,
    *,
    statement_url: str,
    shares_url: str,
) -> dict[str, object]:
    """Combine one original filing into an auditable point-in-time row."""

    values = parse_consolidated_financial_statements(statement_html)
    row: dict[str, object] = {
        **asdict(report),
        "available_date": report.available_date,
        "ordinary_issued_shares": parse_ordinary_issued_shares(shares_html),
        **values,
        "source": "DART public original filing",
        "report_url": report.main_url,
        "statement_url": statement_url,
        "shares_url": shares_url,
    }
    return row


def annual_fundamentals_asof(
    frame: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    max_age_days: int = 550,
) -> pd.DataFrame:
    """Return the latest original annual filing available for each ticker."""

    required = {"ticker", "period_end", "receipt_date", "available_date", "receipt_no"}
    missing = required.difference(frame.columns)
    if missing:
        raise DartPointInTimeError(f"annual fundamentals missing columns: {sorted(missing)}")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.zfill(6)
    for field in ("period_end", "receipt_date", "available_date"):
        data[field] = pd.to_datetime(data[field], errors="raise").dt.normalize()
    if data["available_date"].le(data["receipt_date"]).any():
        raise DartPointInTimeError("available_date must be after the DART receipt date")
    signal = pd.Timestamp(signal_date).normalize()
    eligible = data.loc[data["available_date"].le(signal)].copy()
    if eligible.empty:
        return eligible.set_index("ticker")
    eligible = eligible.sort_values(["ticker", "period_end", "receipt_date"])
    eligible = eligible.groupby("ticker", as_index=False).tail(1)
    age = (signal - eligible["period_end"]).dt.days
    eligible = eligible.loc[age.le(int(max_age_days))]
    return eligible.set_index("ticker").sort_index()
