"""종목 식별 테스트.

주장 텍스트에서 종목을 못 찾으면 조회 조건을 만들 수 없고, 그 주장은
검증되지 못한 채 UNVERIFIABLE 로 떨어집니다. 그래서 여기가 재현율의 출발점입니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.universe import (
    BY_CODE,
    UNIVERSE,
    find_all_stocks,
    find_stock,
    get_stock,
)


def test_universe_codes_are_unique():
    """종목코드가 겹치면 매핑이 망가집니다."""
    codes = [stock.code for stock in UNIVERSE]
    assert len(codes) == len(set(codes))


def test_universe_codes_are_six_digits():
    for stock in UNIVERSE:
        assert len(stock.code) == 6 and stock.code.isdigit(), stock


def test_get_stock_by_code():
    assert get_stock("005930").name == "삼성전자"
    assert get_stock("999999") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다", "005930"),
        ("SK하이닉스의 연간 매출은 71조원이다", "000660"),
        ("현대자동차는 배당 성향을 확대해 왔다", "005380"),
        ("네이버의 광고 매출이 증가했다", "035420"),
        ("포스코홀딩스 철강 부문 영업이익이 감소했다", "005490"),
        ("KB금융지주의 순이익은 1조원이다", "105560"),
        ("종목코드 005930 주가는 7만원이다", "005930"),
    ],
)
def test_find_stock(text, expected):
    """이름, 별칭, 종목코드 어느 쪽으로도 찾을 수 있어야 합니다."""
    stock = find_stock(text)
    assert stock is not None, text
    assert stock.code == expected


def test_longer_name_wins():
    """짧은 이름이 긴 이름을 가로채면 안 됩니다.

    "삼성바이오로직스"를 "삼성"이나 "삼성전자"로 잘못 잡으면 엉뚱한 종목을
    조회하게 됩니다.
    """
    stock = find_stock("삼성바이오로직스의 수주가 늘었다")
    assert stock is not None and stock.code == "207940"


def test_unknown_company_returns_none():
    """유니버스 밖의 기업은 None 입니다. 억지로 끼워 맞추면 안 됩니다."""
    assert find_stock("애플의 주가가 상승했다") is None
    assert find_stock("") is None


def test_find_all_stocks():
    """한 문장에 여러 종목이 나오면 모두 찾습니다."""
    found = find_all_stocks("현대차와 기아의 합산 영업이익이 증가했다")
    assert {stock.code for stock in found} == {"005380", "000270"}


def test_find_all_stocks_has_no_duplicates():
    found = find_all_stocks("삼성전자와 삼성전자우는 같은 회사다")
    assert [stock.code for stock in found] == ["005930"]


def test_by_code_matches_universe():
    assert set(BY_CODE) == {stock.code for stock in UNIVERSE}
