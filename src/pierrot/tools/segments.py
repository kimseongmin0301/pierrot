"""사업보고서 '매출 및 수주상황' 절의 부문별·제품별 매출 표를 뽑습니다.

왜 필요한가
-----------
3단계 실측에서 확인 불가 55건 중 가장 큰 덩어리가 **부문별·제품별 매출**이었습니다.

    "2024년 전체 HBM 매출은 45조 원 이상을 기록했다"
    "NAVER는 2024년 1분기 광고 부문에서 5조 7,839억 원의 매출을 기록했다"
    "카카오의 광고 사업 부문 매출은 1,509억 원이다"
    "2024년 1분기 POSCO홀딩스 철강 부문 영업이익은 750억원이다"

재무제표 주요계정(14개)에도, 전체 재무제표(121개)에도 부문별 수치는 없습니다.
부문 정보는 **사업보고서 본문의 표**에 있습니다. 2단계에서 본문을 수집할 때
표는 `looks_like_table` 로 걸러 버렸습니다. 문장 단위 NLI 에 표 조각을 넣으면
거짓 모순이 쏟아졌기 때문입니다(실측 64.2%). 그 판단 자체는 맞았지만,
**표를 문장으로 다루려 한 것이 문제였습니다.** 표는 표로 읽어야 합니다.

표의 모양 (실측)
----------------
회사마다 다르지만 공통 구조가 있습니다. 앞쪽에 이름 칸이 오고 뒤쪽에 숫자 칸이
옵니다. 숫자 칸은 당기/전기/전전기 순서이고, 비중(%)이 끼어 있기도 합니다.

    삼성전자      부 문 | 매출유형 | 품 목 | 제56기 | 제55기 | 제54기
                 DX 부문 | 제·상품... | TV, 모니터... | 1,748,877 | 1,699,923 | 1,824,897

    삼성바이오    사업부문 | 매출유형 | 품목 | 2024년도 | 2023년도 | 2022년도
                           (각 연도마다 금액 | 비중)
                 CDMO | 제품 등 | 항체의약품 | 28,983 | 63.7% | 23,956 | 64.8% | ...

    카카오        구분 | 제30 (당) 기 | 제29 (전) 기 | 제28 (전전) 기
                 플랫폼 부문 | 3,900,700 | 49.6% | 3,554,442 | 47.0% | ...

기수 표기("제56기", "제30 (당) 기")는 회사마다 달라 그대로 쓸 수 없습니다.
대신 **열 순서**를 씁니다. 첫 금액 열이 보고서의 사업연도이고, 그다음이 1년 전,
그다음이 2년 전입니다. 이 규칙은 모든 표에서 같습니다.

단위도 표마다 다릅니다(억원 / 백만원 / 천원). 단위는 표 바로 앞에 별도의 작은
표로 들어 있어서("(단위 : 억원)"), 절을 앞에서부터 훑으며 마지막으로 본 단위를
기억했다가 씁니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_TABLE = re.compile(r"<TABLE\b.*?</TABLE>", re.I | re.S)
_ROW = re.compile(r"<TR\b.*?</TR>", re.I | re.S)
_CELL = re.compile(r"<T[HD]\b.*?</T[HD]>", re.I | re.S)

# "(단위 : 억원)", "(단위: 백만원, %)" 등
_UNIT = re.compile(r"\(\s*단위\s*[:：]\s*([^)]+)\)")

UNIT_SCALE = {
    "원": 1,
    "천원": 1_000,
    "백만원": 1_000_000,
    "억원": 100_000_000,
    "십억원": 1_000_000_000,
    "조원": 1_000_000_000_000,
}

# 숫자 칸. "1,748,877", "△285,155", "(285,155)", "63.7%", "-"
_NUMBER = re.compile(r"^[△▲\-(]?\s*[\d,]+(?:\.\d+)?\s*\)?%?$")

# 합계·조정 행. 부문 하나로 세면 안 됩니다.
# "총 합 계"(POSCO), "단순합계", "내부거래제거" 까지 포함합니다.
# 처음에는 "합계|소계|총계|계" 만 봤다가 POSCO 의 "총 합 계 726,881억" 을
# 일반 부문으로 셌습니다.
_TOTAL = re.compile(r"^(합계|소계|총계|총합계|계|단순합계|내부거래제거|연결조정)$")

# 머리글 낱말. 이름 칸에 이것만 들어 있으면 자료 행이 아니라 제목 행입니다.
# 병합 셀 때문에 제목 행이 자료 행처럼 보이는 경우가 있습니다.
# 실측: 삼성전자 수출 비중 표에서 "비 중 | 16" 이 매출 16억원으로 잡혔습니다.
# 주의: 실제 부문 이름을 여기 넣으면 자료가 조용히 사라집니다.
#   "영업수익" 은 NAVER 의 합계 행 이름이고 "서비스" 는 삼성바이오의 부문입니다.
#   처음에 둘 다 넣었다가 NAVER 의 합계를 잃어 상한 판정이 동작하지 않았습니다.
#   열 제목으로만 쓰이는 낱말로 좁힙니다.
_HEADER_WORDS = {
    "비중", "구분", "구 분", "품목", "품 목", "매출유형", "부문", "사업부문",
    "주요영업", "금액",
}

# 항목 앞에 붙는 글머리표. NAVER 는 "- 서치플랫폼" 처럼 적습니다.
_LEADING_MARK = re.compile(r"^[\s\-ㆍ·•※*]+")
_HEADER_WORDS_FLAT = {word.replace(" ", "") for word in _HEADER_WORDS}


def plain(html: str) -> str:
    """태그를 벗기고 공백을 정리합니다."""
    return _WHITESPACE.sub(" ", _TAG.sub(" ", html)).strip()


def parse_number(text: str) -> float | None:
    """표 한 칸을 숫자로. 못 읽으면 None.

    △ 와 괄호는 음수 표기입니다. DART 표에서 둘 다 쓰입니다.
    """
    value = (text or "").strip()
    if not value or value in {"-", "—", "–"}:
        return None

    negative = value.startswith(("△", "▲")) or (value.startswith("(") and value.endswith(")"))
    digits = re.sub(r"[^\d.]", "", value)
    if not digits or digits == ".":
        return None
    try:
        number = float(digits)
    except ValueError:
        return None
    return -number if negative else number


def is_ratio(text: str) -> bool:
    return "%" in (text or "")


def table_rows(table_html: str) -> list[list[str]]:
    """표 하나를 행 목록으로."""
    rows = []
    for row_html in _ROW.findall(table_html):
        cells = [plain(cell) for cell in _CELL.findall(row_html)]
        if any(cells):
            rows.append(cells)
    return rows


@dataclass(frozen=True)
class SegmentFact:
    """부문·제품·지역별 매출 한 칸."""

    code: str
    bsns_year: int
    """보고서의 사업연도. 표의 당기입니다."""

    rcept_no: str
    name: str
    """부문/품목 이름. 표의 이름 칸들을 그대로 이어 붙였습니다."""

    labels: tuple[str, ...]
    """이름 칸 하나하나. 부분 일치로 찾을 때 씁니다."""

    year: int
    """이 수치가 가리키는 연도. 당기면 bsns_year, 전기면 -1, 전전기면 -2."""

    amount: float
    """원 단위로 환산한 금액."""

    ratio: float | None = None
    """비중(%). 표에 있으면 채웁니다."""

    unit_text: str = ""
    """표에 적혀 있던 단위 표기. 환산이 맞는지 사람이 확인할 때 씁니다."""

    is_total: bool = False
    """합계 행인지. 부문 하나로 세면 안 되지만, 상한 비교에는 씁니다."""

    def describe(self) -> str:
        share = f", 비중 {self.ratio:.1f}%" if self.ratio is not None else ""
        return f"{self.name} {self.year}년 매출 {self.amount / 1e8:,.0f}억원{share}"


def _split_labels_and_values(cells: list[str]) -> tuple[list[str], list[str]]:
    """앞쪽 이름 칸과 뒤쪽 숫자 칸을 가릅니다.

    열 제목을 믿지 않는 이유: 회사마다 제목이 다르고(부 문 / 사업부문 / 구분),
    병합 셀(rowspan) 때문에 행마다 칸 수가 다릅니다. 대신 **뒤에서부터
    숫자로 읽히는 칸을 세는** 방식을 씁니다. 이건 모든 표에서 같습니다.
    """
    index = len(cells)
    while index > 0 and _NUMBER.match(cells[index - 1].replace(" ", "")):
        index -= 1
    return cells[:index], cells[index:]


# 기간을 가리키는 머리글 표기. 회사마다 다릅니다.
#   "제56기"  "제81기('24.1.1~12.31)"  "2024년도 (제14기)"
#   "제30 (당) 기"  "당기"  "전기"  "전전기"
_PERIOD_MARK = re.compile(r"제\s*\d+\s*\(?\s*[당전]?\s*\)?\s*기|\d{4}\s*년|^\s*(당|전|전전)\s*기\s*$")


def count_period_columns(rows: list[list[str]]) -> int:
    """머리글에서 기간 열이 몇 개인지 셉니다.

    **이 함수가 없으면 표를 통째로 오독합니다.** 처음에는 "뒤쪽 숫자 칸은
    모두 기간"이라고 가정했는데, 기아 보고서의 지역별 매출표가 이렇게 생겼습니다.

        구 분 | 국내 | 북중미 | 유럽 | 기타 | 연결조정 | 연결후금액
        총매출액 | 63,256,745 | 39,408,225 | ... | 107,448,752

    숫자 칸이 7 개인데 기간이 아니라 **지역**입니다. 그대로 읽으면
    국내 매출 63조를 "2024년 매출", 북중미 39조를 "2023년 매출"로 기록합니다.
    실제로 합계 검수에서 93% 차이로 걸렸습니다.

    그래서 머리글에 기간 표기가 있는 표만 받습니다. 없으면 그 표는 버립니다.
    버리는 쪽이 잘못 읽는 쪽보다 낫습니다.
    """
    best = 0
    for cells in rows:
        _, values = _split_labels_and_values(cells)
        if values:
            continue
        hits = sum(1 for cell in cells if _PERIOD_MARK.search(cell))
        best = max(best, hits)
    return best


def count_ratio_columns(rows: list[list[str]]) -> int:
    """머리글에서 비중 열이 몇 개인지 셉니다.

    기간 수를 3 으로 고정했다가 NAVER 표를 통째로 놓쳤습니다. 그 표는 기간이
    당기·전기 두 개뿐이라 숫자 칸이 4 개였고, 3 도 6 도 아니라서 버려졌습니다.
    기간 수는 회사마다 다르므로 **표에서 읽어야 합니다.**

        NAVER        금액 | 비중 | 금액 | 비중              -> 비중 2개, 기간 2
        삼성바이오    금액 | 비중 | 금액 | 비중 | 금액 | 비중  -> 비중 3개, 기간 3
        삼성전자      (비중 열 없음)                        -> 기간 = 숫자 칸 수
    """
    best = 0
    for cells in rows:
        _, values = _split_labels_and_values(cells)
        if values:
            continue  # 숫자가 있으면 머리글이 아닙니다
        hits = sum(1 for cell in cells if "비중" in cell)
        best = max(best, hits)
    return best


def _value_columns(
    values: list[str], *, periods: int, ratio_columns: int = 0
) -> list[tuple[float | None, float | None]]:
    """숫자 칸들을 (금액, 비중) 짝으로 묶습니다.

    칸 수가 머리글에서 읽은 기간 수와 맞아야 합니다. 맞지 않으면 해석하지
    않습니다. 억지로 맞추면 엉뚱한 숫자를 금액으로 읽습니다.
    """
    if ratio_columns and len(values) == ratio_columns * 2:
        return [
            (parse_number(values[index * 2]), parse_number(values[index * 2 + 1]))
            for index in range(ratio_columns)
        ]
    if len(values) == periods:
        return [(parse_number(value), None) for value in values]
    return []


def extract_segment_facts(
    section_html: str, *, code: str, bsns_year: int, rcept_no: str
) -> list[SegmentFact]:
    """'매출 및 수주상황' 절에서 부문별 수치를 뽑습니다."""
    facts: list[SegmentFact] = []
    unit_text = ""
    scale = 0
    cursor = 0

    for table_html in _TABLE.finditer(section_html):
        # 표 **앞쪽 본문**까지 훑어 단위를 찾습니다.
        #
        # 처음에는 표 안에서만 "(단위 : 억원)" 을 찾았습니다. 보통은 단위만 적힌
        # 작은 표가 앞에 붙어 있어서 그걸로 됐는데, 현대차 2023·2024 보고서는
        # 단위를 표 바깥 문단에 적어 두었습니다. 그래서 단위를 못 읽었고
        # 그 회사의 부문 수치를 **한 건도 못 뽑았습니다**(수집 로그에서 0개).
        between = plain(section_html[cursor: table_html.start()])
        for match in _UNIT.finditer(between):
            unit_text = match.group(1).strip()
            scale = _resolve_scale(unit_text) or scale
        cursor = table_html.end()

        rows = table_rows(table_html.group(0))
        if not rows:
            continue

        # 단위만 적힌 작은 표. 다음 표들에 적용됩니다.
        flat = " ".join(" ".join(row) for row in rows)
        unit_match = _UNIT.search(flat)
        if unit_match:
            unit_text = unit_match.group(1).strip()
            scale = _resolve_scale(unit_text) or scale
            if len(rows) == 1:
                continue

        if not scale:
            continue

        periods = count_period_columns(rows)
        if not periods:
            # 기간 머리글이 없는 표는 읽지 않습니다. 지역별·항목별 표를
            # 기간별로 오독하는 것을 막습니다.
            continue
        ratio_columns = count_ratio_columns(rows)

        for cells in rows:
            labels, values = _split_labels_and_values(cells)
            if not labels or not values:
                continue
            pairs = _value_columns(
                values, periods=periods, ratio_columns=ratio_columns
            )
            if not pairs:
                continue

            name = " ".join(label for label in labels if label).strip()
            name = _LEADING_MARK.sub("", name).strip()
            if not name or name.replace(" ", "") in _HEADER_WORDS_FLAT:
                continue
            # 합계 행은 이름으로도 잡고, 비중 100% 로도 잡습니다.
            # NAVER 는 합계 행 이름이 "영업수익" 이라 이름만으로는 못 가립니다.
            flat_name = name.replace(" ", "")
            total = bool(_TOTAL.search(flat_name)) or any(
                ratio is not None and abs(ratio - 100.0) < 0.05 for _, ratio in pairs
            )

            # 비중만 있는 표(금액 없음)는 건너뜁니다.
            for offset, (amount, ratio) in enumerate(pairs):
                if amount is None:
                    continue
                facts.append(
                    SegmentFact(
                        code=code,
                        bsns_year=bsns_year,
                        rcept_no=rcept_no,
                        name=name,
                        labels=tuple(label for label in labels if label),
                        year=bsns_year - offset,
                        amount=amount * scale,
                        ratio=ratio,
                        unit_text=unit_text,
                        is_total=total,
                    )
                )
    return facts


def _resolve_scale(unit_text: str) -> int:
    """단위 표기를 배수로. '백만원, %' 처럼 뒤에 다른 것이 붙어 있어도 읽습니다."""
    cleaned = unit_text.replace(" ", "")
    for name in sorted(UNIT_SCALE, key=len, reverse=True):
        if name in cleaned:
            return UNIT_SCALE[name]
    return 0


# ---------------------------------------------------------------------------
# 스냅샷
# ---------------------------------------------------------------------------

COLUMNS = [
    "code", "bsns_year", "rcept_no", "name", "labels",
    "year", "amount", "ratio", "unit_text", "is_total",
]


class SegmentSnapshot:
    """수집해 둔 부문별 매출을 조회합니다.

    검증 중에는 외부에 접속하지 않습니다. 수집과 검증을 분리해 두어야
    몇 달 뒤에도 같은 숫자가 나옵니다. 주가·재무 스냅샷과 같은 방식입니다.
    """

    def __init__(self, frame) -> None:
        missing = [column for column in COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"필요한 열이 없습니다: {missing}")
        self.frame = frame

    @classmethod
    def load(cls, path=None) -> "SegmentSnapshot":
        import pandas as pd

        from ..config import get_settings

        path = path or (get_settings().snapshot_dir / "dart_segments.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"부문별 매출 스냅샷이 없습니다: {path}\n"
                "먼저 scripts/fetch_segment_snapshot.py 를 실행해 주십시오."
            )
        return cls(pd.read_parquet(path))

    @classmethod
    def from_facts(cls, facts: list[SegmentFact]) -> "SegmentSnapshot":
        import pandas as pd

        records = [
            {
                "code": fact.code,
                "bsns_year": fact.bsns_year,
                "rcept_no": fact.rcept_no,
                "name": fact.name,
                "labels": " | ".join(fact.labels),
                "year": fact.year,
                "amount": fact.amount,
                "ratio": fact.ratio if fact.ratio is not None else float("nan"),
                "unit_text": fact.unit_text,
                "is_total": fact.is_total,
            }
            for fact in facts
        ]
        return cls(pd.DataFrame(records, columns=COLUMNS))

    def names(self, code: str, year: int) -> list[str]:
        """그 회사·그 해에 공시된 부문 이름들. 못 찾았을 때 사유에 적습니다."""
        rows = self.frame[
            (self.frame["code"] == code)
            & (self.frame["year"] == year)
            & (~self.frame["is_total"])
        ]
        return sorted({str(name) for name in rows["name"]})

    def lookup(self, code: str, year: int, keyword: str) -> list[SegmentFact]:
        """부문 이름에 keyword 가 들어간 수치들.

        완전 일치를 요구하지 않습니다. 공시의 부문 이름("DS 부문")과 답변의
        표현("반도체 부문")이 같을 이유가 없기 때문입니다. 대신 찾지 못하면
        있는 이름을 전부 돌려주어 사유에 적게 합니다.
        """
        key = (keyword or "").replace(" ", "")
        if not key:
            return []
        rows = self.frame[(self.frame["code"] == code) & (self.frame["year"] == year)]
        # 같은 연도 수치가 여러 보고서에 들어 있습니다(2024년 수치는 2024년·2025년
        # 보고서 양쪽에 있습니다). 해당 연도 보고서를 우선합니다. 그쪽이 당기라
        # 표기가 더 자세합니다.
        same_year = rows[rows["bsns_year"] == year]
        if not same_year.empty:
            rows = same_year
        hits = [
            row
            for _, row in rows.iterrows()
            if key in str(row["name"]).replace(" ", "")
        ]
        return [self._to_fact(row) for row in hits]

    def total(self, code: str, year: int) -> SegmentFact | None:
        """그 해 매출 합계 행. 부문 수치가 상한을 넘는지 볼 때 씁니다."""
        rows = self.frame[
            (self.frame["code"] == code)
            & (self.frame["year"] == year)
            & (self.frame["is_total"])
        ]
        if rows.empty:
            return None
        best = rows.loc[rows["amount"].idxmax()]
        return self._to_fact(best)

    @staticmethod
    def _to_fact(row) -> SegmentFact:
        ratio = row["ratio"]
        return SegmentFact(
            code=str(row["code"]),
            bsns_year=int(row["bsns_year"]),
            rcept_no=str(row["rcept_no"]),
            name=str(row["name"]),
            labels=tuple(str(row["labels"]).split(" | ")) if row["labels"] else (),
            year=int(row["year"]),
            amount=float(row["amount"]),
            ratio=None if ratio != ratio else float(ratio),
            unit_text=str(row["unit_text"]),
            is_total=bool(row["is_total"]),
        )
