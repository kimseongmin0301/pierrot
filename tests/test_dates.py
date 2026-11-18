"""한국어 날짜 표현 해석 테스트.

여기서 틀리면 엉뚱한 날짜의 주가와 비교하게 되므로 판정 전체가 무의미해집니다.
실제 1단계 주장에 등장한 표현을 그대로 가져다 썼습니다.
"""

from __future__ import annotations

from datetime import date

import pytest

from pierrot.tools.dates import (
    DatePrecision,
    DateRange,
    combine,
    resolve_date_expression,
)

REF = date(2024, 6, 1)


# ---------------------------------------------------------------------------
# 정확한 하루
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2024년 4월 9일", date(2024, 4, 9)),
        ("2024년 3월 29일", date(2024, 3, 29)),
        ("2024-04-09", date(2024, 4, 9)),
        ("2024/04/09", date(2024, 4, 9)),
        ("현재(2024년 7월 26일)", date(2024, 7, 26)),
    ],
)
def test_exact_day(expression, expected):
    result = resolve_date_expression(expression)
    assert result is not None
    assert result.start == result.end == expected
    assert result.precision is DatePrecision.DAY
    assert result.is_single_day


def test_month_day_without_year_uses_reference():
    """연도가 생략된 표현은 기준일의 연도를 씁니다."""
    result = resolve_date_expression("4월 9일", reference=REF)
    assert result is not None and result.start == date(2024, 4, 9)


def test_month_day_without_reference_fails():
    """기준일이 없으면 연도를 지어내지 않고 포기합니다."""
    assert resolve_date_expression("4월 9일") is None


def test_invalid_day_is_clamped_to_month_end():
    """2월 31일 같은 표기는 그 달의 마지막 날로 맞춥니다."""
    result = resolve_date_expression("2024년 2월 31일")
    assert result is not None and result.start == date(2024, 2, 29)


# ---------------------------------------------------------------------------
# 순(旬) 표현 - 이 프로젝트에서 가장 자주 나옵니다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,start,end",
    [
        ("2024년 1월 초", date(2024, 1, 1), date(2024, 1, 10)),
        ("2024년 4월 초순", date(2024, 4, 1), date(2024, 4, 10)),
        ("2024년 4월 중순", date(2024, 4, 11), date(2024, 4, 20)),
        ("2024년 5월 말", date(2024, 5, 21), date(2024, 5, 31)),
        ("2024년 2월 말", date(2024, 2, 21), date(2024, 2, 29)),
        ("2024년 4월 하순", date(2024, 4, 21), date(2024, 4, 30)),
    ],
)
def test_tenday_expressions(expression, start, end):
    result = resolve_date_expression(expression)
    assert result is not None
    assert (result.start, result.end) == (start, end)
    assert result.precision is DatePrecision.RANGE


def test_tenday_without_year_uses_reference():
    result = resolve_date_expression("4월 중순", reference=REF)
    assert result is not None
    assert (result.start, result.end) == (date(2024, 4, 11), date(2024, 4, 20))


def test_february_end_is_leap_year_aware():
    """윤년 처리를 규칙으로 하는 이유가 여기 있습니다."""
    leap = resolve_date_expression("2024년 2월 말")
    common = resolve_date_expression("2025년 2월 말")
    assert leap is not None and leap.end == date(2024, 2, 29)
    assert common is not None and common.end == date(2025, 2, 28)


# ---------------------------------------------------------------------------
# 분기 / 반기 / 월 / 연도
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,start,end",
    [
        ("2024년 1분기", date(2024, 1, 1), date(2024, 3, 31)),
        ("2024년 3분기", date(2024, 7, 1), date(2024, 9, 30)),
        ("2024년 4분기", date(2024, 10, 1), date(2024, 12, 31)),
        ("2024년 상반기", date(2024, 1, 1), date(2024, 6, 30)),
        ("2024년 하반기", date(2024, 7, 1), date(2024, 12, 31)),
        ("2024년 4월", date(2024, 4, 1), date(2024, 4, 30)),
        ("2026년", date(2026, 1, 1), date(2026, 12, 31)),
    ],
)
def test_period_expressions(expression, start, end):
    result = resolve_date_expression(expression)
    assert result is not None
    assert (result.start, result.end) == (start, end)
    assert result.precision is DatePrecision.RANGE


def test_day_wins_over_month():
    """'2024년 4월 9일'을 4월 전체로 해석하면 안 됩니다."""
    result = resolve_date_expression("2024년 4월 9일")
    assert result is not None and result.is_single_day


def test_tenday_wins_over_month():
    """'2024년 4월 중순'을 4월 전체로 넓히면 판정이 지나치게 관대해집니다."""
    result = resolve_date_expression("2024년 4월 중순")
    assert result is not None and result.days == 10


# ---------------------------------------------------------------------------
# 해석 실패
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["", "   ", "최근", "현재까지", "조만간", "2024년 13월"])
def test_unparseable_returns_none(expression):
    """해석할 수 없으면 추측하지 않고 포기합니다.

    억지로 날짜를 만들어 내면 엉뚱한 날 주가와 비교해 틀린 판정을 냅니다.
    검증할 수 없는 주장은 UNVERIFIABLE 로 남겨야 합니다.
    """
    assert resolve_date_expression(expression) is None


# ---------------------------------------------------------------------------
# 구간 결합
# ---------------------------------------------------------------------------


def test_combine_two_ranges():
    """'3월 초부터 4월 중순까지' 같은 구간 주장."""
    start = resolve_date_expression("2024년 3월 초")
    end = resolve_date_expression("2024년 4월 중순")
    merged = combine(start, end)
    assert merged.start == date(2024, 3, 1)
    assert merged.end == date(2024, 4, 20)
    assert merged.precision is DatePrecision.RANGE


def test_combine_two_days_stays_day_precision():
    """양 끝이 모두 정확한 날짜면 정밀도를 유지합니다."""
    start = resolve_date_expression("2024년 3월 4일")
    end = resolve_date_expression("2024년 4월 9일")
    merged = combine(start, end)
    assert merged.precision is DatePrecision.DAY


def test_date_range_describe():
    single = DateRange(date(2024, 4, 9), date(2024, 4, 9), DatePrecision.DAY)
    span = DateRange(date(2024, 4, 11), date(2024, 4, 20), DatePrecision.RANGE)
    assert single.describe() == "2024-04-09"
    assert "10일" in span.describe()
