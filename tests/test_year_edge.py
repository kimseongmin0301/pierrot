"""'연초 대비' 는 1월 전체가 아니라 그 해 첫 거래일입니다.

실측 오탐(gpt-4o-mini 가 쓴 답변을 검증하다 나옴):

    "KB금융의 주가는 2023년 10월 25일 기준 56,000원이었으며,
     연초 대비 약 10% 상승했습니다."

'연초' 를 2023년 1월 전체로 읽었습니다. 그 달 종가가 47,600~60,000원이라
기준값이 통째로 퍼졌고, 나올 수 있는 등락률이 -9.33% ~ +14.29% 가 됐습니다.
주장한 +10% 가 그 안에 들어가 **근거 있음(CALCULATED)** 으로 통과했습니다.
실제는 47,600 -> 54,400 으로 **+14.29%** 라 틀린 주장입니다.

구간 폭 상한(25%p)이 이미 있었지만 이 건은 23.6%p 라 간신히 빠져나갔습니다.
상한을 조이는 것은 답이 아닙니다. 다른 멀쩡한 주장까지 막게 됩니다.
'연초' 는 애초에 구간이 아니라 **한 날**을 가리키는 말입니다.

이 결함은 gemma3:4b 답변으로는 영영 못 찾았습니다. 그 답변에서 뽑은
주장 94개 중 참인 것이 0개라, '맞는 것을 틀렸다고 하는가' 를 잴 분모가
아예 없었기 때문입니다.
"""

from __future__ import annotations

from datetime import date

import pytest

from pierrot.tools.dates import (
    DatePrecision,
    narrow_to_year_edge,
    resolve_date_expression,
)
from pierrot.tools.price import PriceRow
from pierrot.tools.price_verify import _anchored


def row(day: str, close: float) -> PriceRow:
    return PriceRow(
        code="105560",
        date=date.fromisoformat(day),
        open=close, high=close, low=close, close=close,
        volume=0, change=0.0,
        open_raw=close, high_raw=close, low_raw=close, close_raw=close,
    )


# ---------------------------------------------------------------------------
# 날짜 해석
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,anchor,why",
    [
        ("2023년 연초", "FIRST", "연도가 붙은 표현"),
        ("2024년 연말", "LAST", "연말은 마지막 거래일"),
    ],
)
def test_연초_연말은_한_날을_가리킨다(expression, anchor, why):
    found = resolve_date_expression(expression)
    assert found is not None, why
    assert found.anchor == anchor
    assert found.days <= 12, "구간이 넓으면 기준값이 퍼집니다"


def test_연도가_없으면_기준일에서_가져온다():
    found = resolve_date_expression("연초", reference=date(2023, 10, 25))
    assert found is not None
    assert found.start.year == 2023
    assert found.anchor == "FIRST"


def test_기준일이_없으면_연도를_모르므로_해석하지_않는다():
    assert resolve_date_expression("연초") is None


def test_연초가_그_해_전체로_읽히지_않는다():
    """_YEAR 패턴이 먼저 집으면 365일 구간이 되어 버립니다."""
    found = resolve_date_expression("2023년 연초")
    assert found is not None
    assert found.days < 365


@pytest.mark.parametrize("expression", ["2023년 1월", "2023년", "2024년 3분기"])
def test_보통_기간_표현에는_anchor_가_붙지_않는다(expression):
    """가드가 과하면 멀쩡한 구간 주장까지 한 점으로 좁혀 버립니다."""
    found = resolve_date_expression(expression)
    assert found is not None
    assert found.anchor == ""
    assert found.precision is DatePrecision.RANGE


# ---------------------------------------------------------------------------
# 구간을 한 거래일로 좁히기
# ---------------------------------------------------------------------------


def test_첫_거래일만_남긴다():
    rows = [row("2023-01-02", 47600), row("2023-01-03", 50000),
            row("2023-01-06", 60000)]
    picked = _anchored(rows, "FIRST")
    assert len(picked) == 1
    assert picked[0].date == date(2023, 1, 2)
    assert picked[0].close == 47600


def test_마지막_거래일만_남긴다():
    rows = [row("2024-12-26", 90000), row("2024-12-30", 95000)]
    picked = _anchored(rows, "LAST")
    assert len(picked) == 1
    assert picked[0].date == date(2024, 12, 30)


def test_날짜_순서가_뒤섞여_있어도_맞게_고른다():
    rows = [row("2023-01-06", 60000), row("2023-01-02", 47600),
            row("2023-01-03", 50000)]
    assert _anchored(rows, "FIRST")[0].close == 47600
    assert _anchored(rows, "LAST")[0].close == 60000


def test_anchor_가_없으면_구간을_그대로_둔다():
    """'2023년 1월 중' 같은 주장은 구간 전체를 봐야 맞습니다."""
    rows = [row("2023-01-02", 47600), row("2023-01-06", 60000)]
    assert _anchored(rows, "") == rows


def test_거래일이_없으면_빈_채로_둔다():
    assert _anchored([], "FIRST") == []


# ---------------------------------------------------------------------------
# '연초' 가 한 달로 펼쳐져 들어오는 길
# ---------------------------------------------------------------------------
#
# 위의 anchor 를 넣고도 같은 주장이 또 통과했습니다. 날짜 해석기는 멀쩡했고,
# 거기까지 '연초' 가 **도달하지 못한** 것이 원인이었습니다. 6단계 다듬기가
# 맥락을 채우며 "연초 대비" 를 "2023년 1월부터 ... 연초 대비" 로 펼쳐 놓자,
# 파서는 더 또렷해 보이는 "2023년 1월" 을 시작 표현으로 골랐습니다.
#
# 고친 데를 테스트로 덮었다고 결함이 사라지는 것이 아닙니다. 같은 입력을
# 끝까지 다시 흘려 보내야 보입니다.


def test_한_달로_펼쳐져도_원문의_연초를_따른다():
    window = resolve_date_expression("2023년 1월")
    narrowed = narrow_to_year_edge(
        window, "2023년 1월부터 2023년 10월 25일까지 연초 대비 약 10% 상승했다", "연초"
    )
    assert narrowed.anchor == "FIRST"
    assert narrowed.days <= 12


def test_파서가_더_좁게_집었으면_건드리지_않는다():
    """"2023년 3월" 은 '연초' 보다 정확한 정보입니다. 덮어쓰면 손해입니다."""
    window = resolve_date_expression("2023년 3월")
    narrowed = narrow_to_year_edge(window, "연초 대비 상승했다", "연초")
    assert narrowed.anchor == ""
    assert narrowed.start == date(2023, 3, 1)


def test_원문에_연초가_없으면_그대로_둔다():
    window = resolve_date_expression("2023년 1월")
    assert narrow_to_year_edge(window, "2023년 1월 대비 상승했다", "연초") is window


def test_이미_anchor_가_있으면_두_번_좁히지_않는다():
    window = resolve_date_expression("2023년 연초")
    assert narrow_to_year_edge(window, "연초 대비", "연초") is window


def test_해석되지_않은_구간은_그대로_None():
    assert narrow_to_year_edge(None, "연초 대비", "연초") is None


def test_연말도_같게_동작한다():
    window = resolve_date_expression("2024년 12월")
    narrowed = narrow_to_year_edge(window, "2024년 12월 연말 기준", "연말")
    assert narrowed.anchor == "LAST"
