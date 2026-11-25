"""휴장일이라 못 본 것을 자료가 없어서 못 본 것처럼 보고하던 문제를 고정합니다.

한국 증시는 1월 1일과 12월 31일에 쉽니다. 그런데 사람도 LLM 도
"2023년 12월 31일 기준 주가" 라고 씁니다. 그 말은 실무에서 **그 해 마지막
거래일 종가**를 가리킵니다.

실측(gpt-4o-mini 답변 40건): 확인 불가 112건 중 **14건**이 이것뿐이었습니다.
"거래일이 없습니다" 라고만 적혀 있어서 공시 자료가 모자란 줄 알았습니다.
사유를 원인별로 갈라 보지 않았으면 못 찾았을 결함입니다.

거슬러 올라간 사실은 **사유에 남깁니다.** 어느 날을 봤는지 읽는 사람이
알아야 합니다. 말없이 다른 날을 보면 11단계에서 고친 '뭉개진 사유' 와
같은 문제가 됩니다.
"""

from __future__ import annotations

from datetime import date

import pytest

from pierrot.tools.dates import DatePrecision, DateRange, resolve_date_expression
from pierrot.tools.price import PriceRow, PriceSnapshot
from pierrot.tools.price_verify import HOLIDAY_LOOKBACK_DAYS, _with_holiday_fallback


class FakeSnapshot:
    """`rows_between` 만 쓰므로 그것만 흉내 냅니다."""

    def __init__(self, days: list[str]) -> None:
        self.rows = [
            PriceRow(
                code="005930", date=date.fromisoformat(day),
                open=100.0, high=100.0, low=100.0, close=100.0,
                volume=0, change=0.0,
                open_raw=100.0, high_raw=100.0, low_raw=100.0, close_raw=100.0,
            )
            for day in days
        ]

    def rows_between(self, code: str, start: date, end: date) -> list[PriceRow]:
        return [row for row in self.rows if start <= row.date <= end]


def day(value: str) -> DateRange:
    point = date.fromisoformat(value)
    return DateRange(point, point, DatePrecision.DAY, value)


def test_휴장일이면_직전_거래일을_본다():
    snapshot = FakeSnapshot(["2023-12-27", "2023-12-28"])
    rows, note = _with_holiday_fallback(snapshot, "005930", day("2023-12-31"))
    assert len(rows) == 1
    assert rows[0].date == date(2023, 12, 28)
    assert "휴장" in note and "2023-12-28" in note


def test_거래일이_있으면_거슬러_가지_않는다():
    """멀쩡한 날까지 과거로 끌고 가면 엉뚱한 날을 보게 됩니다."""
    snapshot = FakeSnapshot(["2023-12-27", "2023-12-28"])
    rows, note = _with_holiday_fallback(snapshot, "005930", day("2023-12-28"))
    assert [row.date for row in rows] == [date(2023, 12, 28)]
    assert note == "", "거슬러 가지 않았으면 사유에 덧붙일 것이 없습니다"


def test_너무_멀면_포기한다():
    """연휴를 덮을 만큼만 봅니다. 그 이상 벌어지면 다른 기간입니다."""
    snapshot = FakeSnapshot(["2023-12-01"])
    rows, note = _with_holiday_fallback(snapshot, "005930", day("2023-12-31"))
    assert rows == []
    assert note == ""


def test_되돌아보는_날수가_연휴를_덮는다():
    assert HOLIDAY_LOOKBACK_DAYS >= 5, "설 연휴·연말 휴장을 덮어야 합니다"


def test_자료가_아예_없으면_빈_채로_둔다():
    rows, note = _with_holiday_fallback(FakeSnapshot([]), "005930", day("2023-12-31"))
    assert rows == []
    assert note == ""


@pytest.mark.parametrize("expression", ["2023년 12월 31일", "2024년 1월 1일"])
def test_휴장일_표현이_날짜로는_멀쩡히_해석된다(expression):
    """해석 자체는 문제가 없었습니다. 그 날 장이 안 선 것뿐입니다."""
    found = resolve_date_expression(expression)
    assert found is not None
    assert found.precision is DatePrecision.DAY
