"""한국어 날짜 표현을 날짜 범위로 바꿉니다.

주가 주장에는 "2024년 4월 9일"처럼 정확한 날짜도 있지만 "4월 중순", "1월 초",
"2024년 상반기"처럼 폭이 있는 표현이 더 흔합니다. 조회를 하려면 이것을
구체적인 구간으로 바꿔야 합니다.

## 왜 파이썬이 계산하는가

LLM에게 "4월 중순이 며칠인지 계산해라"라고 시키면 매번 다른 답이 나오고,
윤년이나 월말 계산에서 틀립니다. 1단계에서 원문 위치를 LLM에게 맡기지 않은 것과
같은 이유입니다. LLM은 "어떤 표현이 날짜인지" 찾는 일만 하고, 그 표현을 구간으로
바꾸는 일은 규칙으로 처리합니다. 그래야 결과가 재현됩니다.

## 범위로 다루는 이유

"4월 중순에 36만원이었다"를 4월 15일 하루와만 비교하면, 사람이 보기에 맞는 주장을
틀렸다고 판정하게 됩니다(오탐). 오탐이 많은 검증기는 실무에서 쓰이지 않습니다.
그래서 중순을 4월 11~20일 구간으로 보고, 그 구간 안에서 성립하면 인정합니다.

대신 관대한 기준이라는 점을 숨기지 않습니다. DatePrecision 으로 어떤 주장이
정확한 날짜였고 어떤 주장이 폭 있는 표현이었는지 구분해 두고, 평가 지표를
"전체"와 "날짜가 명확한 주장만"으로 나누어 공개합니다.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum


class DatePrecision(str, Enum):
    """날짜 표현이 얼마나 구체적이었는지."""

    DAY = "DAY"
    """하루로 특정됩니다. 예: 2024년 4월 9일"""

    RANGE = "RANGE"
    """폭이 있는 표현입니다. 예: 4월 중순, 2024년 1분기"""

    UNKNOWN = "UNKNOWN"
    """날짜를 알 수 없습니다. 검증 대상이 될 수 없습니다."""


@dataclass(frozen=True)
class DateRange:
    """해석된 날짜 구간. 양 끝을 포함합니다."""

    start: date
    end: date
    precision: DatePrecision
    raw: str = ""

    anchor: str = ""
    """구간 안에서 **어느 한 날**을 가리키는 표현이면 그 쪽을 적습니다.

    "FIRST" 는 구간의 첫 거래일, "LAST" 는 마지막 거래일입니다.

    왜 필요한가: "연초 대비 10% 상승" 에서 '연초' 를 1월 전체로 읽으면
    기준값이 통째로 벌어집니다. 실측에서 KB금융 2023년 1월 종가가
    47,600~60,000원이라 등락률 범위가 -9.3%~+14.3% 가 됐고, 틀린 주장
    (+10%, 실제 +14.3%)이 그 안에 들어가 **근거 있음으로 통과**했습니다.
    한 주로 좁혀도 폭이 19.1% 라 모자랍니다. '연초' 는 그 해 **첫 거래일**
    하나를 뜻하므로, 구간이 아니라 점으로 다뤄야 합니다.
    """

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def is_single_day(self) -> bool:
        return self.start == self.end

    def describe(self) -> str:
        if self.is_single_day:
            return f"{self.start}"
        return f"{self.start} ~ {self.end} ({self.days}일)"


def _month_end(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _clamp_day(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, _month_end(year, month)))


# "초순"은 초, "하순"은 말과 같게 봅니다.
_TENDAY = {
    "초": (1, 10),
    "초순": (1, 10),
    "중순": (11, 20),
    "중": (11, 20),
    "하순": (21, 31),
    "말": (21, 31),
}

_QUARTER = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
_HALF = {"상반기": (1, 6), "하반기": (7, 12)}

_ISO = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_Y_M_D = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_M_D = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_Y_M_TENDAY = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(초순|초|중순|하순|말)")
_M_TENDAY = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(초순|초|중순|하순|말)")
_Y_M = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
_M_ONLY = re.compile(r"(?<!\d)(\d{1,2})\s*월")
_Y_QUARTER = re.compile(r"(\d{4})\s*년\s*(\d)\s*분기")
_QUARTER_ONLY = re.compile(r"(?<!\d)(\d)\s*분기")
_Y_HALF = re.compile(r"(\d{4})\s*년\s*(상반기|하반기)")
_HALF_ONLY = re.compile(r"(상반기|하반기)")
_YEAR = re.compile(r"(\d{4})\s*년")

# 연초·연말. 그 해의 첫/마지막 거래일 **한 날**을 가리킵니다.
# 연도가 앞에 붙은 것부터 맞춰야 "2023년 연초" 가 제대로 잡힙니다.
_Y_YEAR_EDGE = re.compile(r"(\d{4})\s*년\s*(연초|연말)")
_YEAR_EDGE = re.compile(r"(연초|연말)")


def resolve_date_expression(
    expression: str,
    *,
    reference: date | None = None,
) -> DateRange | None:
    """한국어 날짜 표현 하나를 구간으로 바꿉니다.

    Args:
        expression: "2024년 4월 중순", "2024-04-09" 같은 표현.
        reference: 연도가 생략된 표현("4월 9일")을 해석할 기준일.
            없으면 연도를 알 수 없으므로 None 을 돌려줍니다.

    Returns:
        해석된 구간. 날짜로 볼 수 없으면 None.
    """
    if not expression:
        return None

    text = expression.strip()
    raw = text
    reference_year = reference.year if reference else None

    # 0) 연초·연말. 다른 패턴보다 **먼저** 봅니다. "2023년 연초" 를 뒤쪽
    #    _YEAR 패턴이 먼저 집으면 그 해 전체(365일)가 되어 버립니다.
    match = _Y_YEAR_EDGE.search(text) or (
        _YEAR_EDGE.search(text) if reference_year else None
    )
    if match:
        groups = match.groups()
        if len(groups) == 2:
            year, label = int(groups[0]), groups[1]
        else:
            year, label = reference_year, groups[0]
        edge = _year_edge_range(year, label, raw)
        if edge is not None:
            return edge

    # 1) 정확한 하루
    match = _ISO.search(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        try:
            point = date(year, month, day)
        except ValueError:
            return None
        return DateRange(point, point, DatePrecision.DAY, raw)

    match = _Y_M_D.search(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        if not 1 <= month <= 12:
            return None
        point = _clamp_day(year, month, day)
        return DateRange(point, point, DatePrecision.DAY, raw)

    # 2) 연도가 생략된 하루
    match = _M_D.search(text)
    if match and reference_year:
        month, day = (int(g) for g in match.groups())
        if not 1 <= month <= 12:
            return None
        point = _clamp_day(reference_year, month, day)
        return DateRange(point, point, DatePrecision.DAY, raw)

    # 3) 순(旬) 표현
    match = _Y_M_TENDAY.search(text)
    if match:
        year, month, label = int(match.group(1)), int(match.group(2)), match.group(3)
        return _tenday_range(year, month, label, raw)

    match = _M_TENDAY.search(text)
    if match and reference_year:
        month, label = int(match.group(1)), match.group(2)
        return _tenday_range(reference_year, month, label, raw)

    # 4) 분기
    match = _Y_QUARTER.search(text)
    if match:
        return _quarter_range(int(match.group(1)), int(match.group(2)), raw)

    match = _QUARTER_ONLY.search(text)
    if match and reference_year:
        return _quarter_range(reference_year, int(match.group(1)), raw)

    # 5) 반기
    match = _Y_HALF.search(text)
    if match:
        return _half_range(int(match.group(1)), match.group(2), raw)

    match = _HALF_ONLY.search(text)
    if match and reference_year:
        return _half_range(reference_year, match.group(1), raw)

    # 6) 월 전체
    match = _Y_M.search(text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            return None
        return DateRange(
            date(year, month, 1),
            date(year, month, _month_end(year, month)),
            DatePrecision.RANGE,
            raw,
        )

    match = _M_ONLY.search(text)
    if match and reference_year:
        month = int(match.group(1))
        if not 1 <= month <= 12:
            return None
        return DateRange(
            date(reference_year, month, 1),
            date(reference_year, month, _month_end(reference_year, month)),
            DatePrecision.RANGE,
            raw,
        )

    # 7) 연도 전체
    match = _YEAR.search(text)
    if match:
        year = int(match.group(1))
        return DateRange(date(year, 1, 1), date(year, 12, 31), DatePrecision.RANGE, raw)

    return None


def _year_edge_range(year: int, label: str, raw: str) -> DateRange | None:
    """'연초'/'연말' 을 그 해 첫·마지막 거래일을 담는 좁은 구간으로.

    달력만으로는 어느 날이 거래일인지 알 수 없습니다. 그래서 며칠짜리
    구간으로 두되 `anchor` 에 어느 쪽 끝을 쓸지 적어 둡니다. 실제로 어느
    날을 쓸지는 주가 스냅샷을 가진 쪽(`price_verify`)이 정합니다.
    """
    if not year:
        return None
    if label == "연초":
        return DateRange(date(year, 1, 1), date(year, 1, 10),
                         DatePrecision.RANGE, raw, anchor="FIRST")
    if label == "연말":
        return DateRange(date(year, 12, 20), date(year, 12, 31),
                         DatePrecision.RANGE, raw, anchor="LAST")
    return None


def _tenday_range(year: int, month: int, label: str, raw: str) -> DateRange | None:
    if not 1 <= month <= 12:
        return None
    first, last = _TENDAY[label]
    return DateRange(
        _clamp_day(year, month, first),
        _clamp_day(year, month, last),
        DatePrecision.RANGE,
        raw,
    )


def _quarter_range(year: int, quarter: int, raw: str) -> DateRange | None:
    if quarter not in _QUARTER:
        return None
    first, last = _QUARTER[quarter]
    return DateRange(
        date(year, first, 1),
        date(year, last, _month_end(year, last)),
        DatePrecision.RANGE,
        raw,
    )


def _half_range(year: int, label: str, raw: str) -> DateRange | None:
    first, last = _HALF[label]
    return DateRange(
        date(year, first, 1),
        date(year, last, _month_end(year, last)),
        DatePrecision.RANGE,
        raw,
    )


def combine(start: DateRange, end: DateRange) -> DateRange:
    """구간 주장의 시작과 끝을 하나로 합칩니다.

    "3월 초부터 4월 중순까지"처럼 두 표현이 따로 해석된 경우에 씁니다.
    둘 중 하나라도 폭이 있으면 결과도 RANGE 입니다.
    """
    precision = (
        DatePrecision.DAY
        if start.precision is DatePrecision.DAY and end.precision is DatePrecision.DAY
        else DatePrecision.RANGE
    )
    return DateRange(
        min(start.start, end.start),
        max(start.end, end.end),
        precision,
        f"{start.raw} ~ {end.raw}".strip(" ~"),
    )


def narrow_to_year_edge(window: DateRange | None, source: str, label: str) -> DateRange | None:
    """문장에 '연초'/'연말' 이 남아 있으면 구간을 그 한 날로 좁힙니다.

    왜 필요한가
    -----------
    날짜 해석기는 '연초' 를 이미 한 날로 봅니다(`_year_edge_range`). 그런데
    그 말이 해석기까지 **도달하지 못하는 길**이 있습니다. 6단계 다듬기가
    맥락을 채우면서 '연초 대비' 를 "2023년 1월부터 ... 연초 대비" 로 펼쳐
    놓으면, 파서는 더 또렷해 보이는 "2023년 1월" 을 시작 표현으로 고릅니다.
    그 순간 한 날이 한 달이 되고, 기준값이 47,600~60,000원으로 퍼집니다.

    실측(gpt-4o-mini 답변 40건):

        "KB금융의 주가는 2023년 1월부터 2023년 10월 25일까지 연초 대비 약 10% 상승했다"
        -> 1월 전체를 기준으로 삼아 등락률 -9.33% ~ +14.29%
        -> 주장 +10% 가 그 안에 들어가 **근거 있음** 으로 통과
        실제는 47,600 -> 54,400 으로 +14.29% 라 틀린 주장입니다.

    안전장치
    --------
    좁힌 구간이 원래 구간 **안에 들어갈 때만** 바꿉니다. 파서가 "2023년 3월"
    처럼 더 좁은 시점을 집어냈다면 그쪽이 더 정확한 정보이므로 건드리지
    않습니다. 문장 어딘가의 '연말' 때문에 엉뚱한 구간이 끌려가는 것을 막습니다.

    Args:
        window: 파서가 고른 표현으로 해석한 구간.
        source: 주장 원문. 여기에 '연초'/'연말' 이 있는지 봅니다.
        label: "연초" 또는 "연말".
    """
    if window is None or window.anchor or label not in source:
        return window
    edge = _year_edge_range(window.start.year, label, window.raw)
    if edge is None:
        return window
    if not (window.start <= edge.start and edge.end <= window.end):
        return window
    return DateRange(edge.start, edge.end, edge.precision, window.raw, anchor=edge.anchor)
