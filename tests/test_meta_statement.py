"""답변이 스스로에 대해 말한 문장을 회사 주장으로 다루던 문제를 고정합니다.

gpt-4o-mini 답변 40건의 의미 비교 판정 8건을 전수로 되짚다 나왔습니다.
8건 중 2건이 **회사에 대한 주장이 아니었는데** 모순 판정을 받았습니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.meta_statement import is_meta_statement
from pierrot.verifier.routing import classify
from pierrot.verifier.state import Route


@pytest.mark.parametrize(
    "text",
    [
        "SK하이닉스는 이미 종료된 기간에 대한 실적이나 수치를 제공할 수 있다",
        "SK하이닉스는 미래에 대한 예측은 할 수 없다",
        "2024년 주가 상승률에 대한 정보를 제공할 수 없다",
        "죄송하지만 해당 기간의 자료는 확인되지 않는다",
    ],
)
def test_답변_자신에_대한_문장은_판정하지_않는다(text):
    """공시에 답이 있을 리 없는데 모순 0.997~0.999 가 났습니다."""
    assert is_meta_statement(text)
    route, _ = classify(text, "NARRATIVE")
    assert route is Route.FORECAST


@pytest.mark.parametrize(
    "text,why",
    [
        ("SK하이닉스는 2024년 실적을 아직 공시하지 않았다", "공시 여부는 확인할 수 있는 사실입니다"),
        ("죄송하지만 SK텔레콤의 2024년 영업이익은 발표된 수치가 아니다",
         "사과 투여도 '발표' 여부를 말하면 사실 주장입니다"),
        ("SK하이닉스의 2024년 영업이익은 23조 원이다", "평범한 수치 주장입니다"),
        ("삼성전자는 2024년 배당을 확대할 수 있다", "회사의 행위입니다"),
        ("현대차의 글로벌 전략이 긍정적 영향을 미쳤다", "인과 주장입니다"),
        ("기아는 2024년 전기차 판매를 늘렸다", "평범한 방향 주장입니다"),
    ],
)
def test_회사에_대한_주장은_막지_않는다(text, why):
    assert not is_meta_statement(text), why


def test_빈_문자열을_견딘다():
    assert not is_meta_statement("")
    assert not is_meta_statement(None)
