"""한국어 금액 표기 파싱 테스트.

여기서 자릿수를 틀리면 1만 배 차이가 나므로, 판정 전체가 무의미해집니다.
실제 샘플 답변에 등장한 표기를 그대로 가져다 썼습니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.amounts import (
    format_korean_amount,
    parse_korean_amount,
    parse_percent,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("9조 1,800억원", 9_180_000_000_000),
        ("32조 7,259억원", 32_725_900_000_000),
        ("1조 6,785억원", 1_678_500_000_000),
        ("3,000억원", 300_000_000_000),
        ("1,500억원", 150_000_000_000),
        ("1조원", 1_000_000_000_000),
        ("15조 원", 15_000_000_000_000),
        ("71조 9,857억 원", 71_985_700_000_000),
        ("6조 1,879억", 6_187_900_000_000),
    ],
)
def test_parse_korean_amount(text, expected):
    assert parse_korean_amount(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("약 9.2조원", 9_200_000_000_000),
        ("3.5조", 3_500_000_000_000),
    ],
)
def test_parse_decimal_units(text, expected):
    """'약 9.2조원'처럼 소수로 말하는 경우도 처리합니다."""
    assert parse_korean_amount(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("71,800원", 71_800),
        ("236,000", 236_000),
    ],
)
def test_parse_plain_number(text, expected):
    """단위가 없으면 원 단위로 봅니다."""
    assert parse_korean_amount(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "영업이익", "알 수 없음"])
def test_parse_without_number_returns_none(text):
    """숫자가 없으면 추측하지 않고 포기합니다."""
    assert parse_korean_amount(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-1조 5,000억원", -1_500_000_000_000),
        ("영업손실 1조원", -1_000_000_000_000),
        ("적자 3,000억원", -300_000_000_000),
    ],
)
def test_negative_amounts(text, expected):
    """적자와 손실은 음수입니다. 부호를 놓치면 판정이 뒤집힙니다."""
    assert parse_korean_amount(text) == expected


def test_unit_and_plain_mixed_prefers_units():
    """'9조 1,800억원' 에서 1,800 을 원 단위로 잘못 읽으면 안 됩니다."""
    assert parse_korean_amount("9조 1,800억원") == 9_180_000_000_000


@pytest.mark.parametrize(
    "amount,expected",
    [
        (9_180_000_000_000, "9조 1,800억원"),
        (300_000_000_000, "3,000억원"),
        (1_000_000_000_000, "1조원"),
        (71_800, "7만 1,800원"),
    ],
)
def test_format_korean_amount(amount, expected):
    assert format_korean_amount(amount) == expected


def test_roundtrip():
    """표기 -> 숫자 -> 표기 가 일치해야 합니다."""
    for text in ["9조 1,800억원", "3,000억원", "1조원"]:
        assert format_korean_amount(parse_korean_amount(text)) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("277% 증가", 277.0),
        ("약 15.7%", 15.7),
        ("-3.5%", -3.5),
        ("전년 대비 1,200% 성장", 1200.0),
        ("증가했다", None),
    ],
)
def test_parse_percent(text, expected):
    assert parse_percent(text) == expected
