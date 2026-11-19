"""라우팅 규칙 테스트.

합성 예시만으로는 의미가 없어서, **실제로 잘못 라우팅됐던 주장**을 그대로 넣습니다.
이 문장들은 2단계 실측에서 "재무제표 주요계정에 없습니다" 로 끝났던 것들입니다.
"""

from __future__ import annotations

import pytest

from pierrot.verifier.routing import classify, is_forecast
from pierrot.verifier.state import Route


# ---------------------------------------------------------------------------
# 실측에서 잘못 갔던 주장들
# ---------------------------------------------------------------------------


def test_주가_수익률은_재무가_아니라_주가로_간다():
    """실측 사례. NUMERIC 으로 분류돼 재무제표를 뒤지다 실패했습니다."""
    text = "2024년 5월 말 POSCO홀딩스의 전년 동기 대비 주가 수익률은 약 18% 수준이다"
    route, reason = classify(text, "NUMERIC")
    assert route is Route.PRICE
    assert "뒤집음" in reason


def test_금액이_있는_NARRATIVE_는_재무로_돌린다():
    text = "삼성바이오로직스는 2024년 1분기 및 2분기에만 총 15조 원 이상의 CDMO 수주를 확보했다"
    route, _ = classify(text, "NARRATIVE")
    assert route is Route.NUMERIC


def test_낱말_증거가_없으면_추출기_유형을_따른다():
    text = "이 회사는 업계에서 좋은 평가를 받고 있다"
    assert classify(text, "NARRATIVE")[0] is Route.NARRATIVE
    assert classify(text, "PRICE")[0] is Route.PRICE


# ---------------------------------------------------------------------------
# 전망 판별
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "2024년 현대차의 영업이익은 약 1.3조 원 수준으로 예상된다",
        "KB금융은 2024년 배당금 총액을 1,500억원 이상으로 지급할 계획이다",
        "KB금융의 2024년 배당수익률은 약 6~7%를 넘어설 것이다",
        "2024년 현대차의 주당별도배당금은 1,500원 이상이 될 것으로 기대된다",
    ],
)
def test_전망_주장은_도구를_타지_않는다(text):
    """실측에서 1단계 휴리스틱이 놓쳤던 네 가지 형태입니다."""
    assert is_forecast(text)
    route, reason = classify(text, "NUMERIC")
    assert route is Route.FORECAST
    assert "전망" in reason or "권고" in reason


@pytest.mark.parametrize(
    "text",
    [
        "삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다",
        "현대차의 2023년 매출액은 162조 6,636억원을 기록했다",
        "삼성전자 주가는 2024년 4월 9일 종가 기준 83,600원이다",
    ],
)
def test_확정된_사실은_전망으로_보지_않는다(text):
    """전망 규칙이 과하면 이미 잡아내던 모순을 놓칩니다. 그 반대 방향 고정입니다."""
    assert not is_forecast(text)
    assert classify(text, "")[0] is not Route.FORECAST


# ---------------------------------------------------------------------------
# 회귀 고정: 전망 규칙이 기존 판정을 삼키지 않아야 합니다
# ---------------------------------------------------------------------------

# 2단계 실측에서 CONTRADICTED 로 잡아냈던 주장들입니다.
# 전망 규칙을 넓히다가 이것들이 FORECAST 로 새면 모순 적발을 그만큼 잃습니다.
CAUGHT_CLAIMS = [
    ("삼성전자 주가는 2024년 4월 9일 종가 기준 71,800원이다", Route.PRICE),
    ("셀트리온 주가는 2024년 4월 9일 종가 기준 218,000원이다", Route.PRICE),
    ("LG에너지솔루션의 2024년 1분기 영업이익은 전년 대비 28% 증가했다", Route.NUMERIC),
    ("NAVER의 2023년 연간 매출액은 6조 7,700억원이다", Route.NUMERIC),
]


@pytest.mark.parametrize("text,expected", CAUGHT_CLAIMS)
def test_이미_잡던_모순은_여전히_도구로_간다(text, expected):
    route, _ = classify(text, "")
    assert route is expected
