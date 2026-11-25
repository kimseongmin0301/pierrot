"""'흑자 전환' 은 증가가 아니라 부호 변화입니다.

실측 오탐(가짜 데이터셋 20번):

    "카카오의 2025년 연결 영업이익은 전년 대비 흑자로 전환됐습니다"

    카카오 연결 영업이익  2024년 +4,602억  ->  2025년 +7,320억

늘어난 것은 맞지만 **2024년에 이미 흑자**였습니다. 전환이 아니라 증가입니다.
그런데도 '근거 있음' 이 났습니다. 원인은 `direction_rescue.UP_WORDS` 에
'흑자 전환' 이 증가·상승과 나란히 들어 있었던 것입니다.

최상급 주장('역대 최대')을 방향 주장으로 읽어 통과시켰던 것과 같은 계열입니다.
**참인 일부(증가)로 더 강한 주장(전환)을 통과시킨 것**입니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.direction_rescue import find_direction, rescue
from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    ParsedNumericClaim,
    build_query,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim
from pierrot.tools.turnaround import LOSS, PROFIT, TO_LOSS, TO_PROFIT, detect

CODE = "035720"  # 카카오


def record(year: int, account: str, amount: int, fs_div: str = "CFS") -> dict:
    return {
        "code": CODE,
        "corp_code": "00258801",
        "bsns_year": year,
        "reprt_code": "11011",
        "period_label": "연간",
        "quarter": 0,
        "period_months": 12,
        "fs_div": fs_div,
        "fs_nm": "연결재무제표" if fs_div == "CFS" else "재무제표",
        "account": account,
        "amount": amount,
        "period_start": "",
        "period_end": "",
        "rcept_no": "X",
    }


@pytest.fixture
def snapshot():
    """실제 카카오 모양.

    영업이익은 **두 해 모두 흑자**입니다. 늘기만 했을 뿐 전환이 아닙니다.
    당기순이익은 2024년 적자에서 2025년 흑자로 **실제로 전환**했습니다.
    """
    return FinancialSnapshot.from_records(
        [
            record(2024, "영업이익", 460_200_000_000),
            record(2025, "영업이익", 732_000_000_000),
            record(2024, "당기순이익", -161_800_000_000),
            record(2025, "당기순이익", 517_900_000_000),
        ]
    )


def claim(text: str, year: int, account: str = "영업이익"):
    """파서가 방향 주장(증가)으로 읽었다고 치고 build_query 에 넣습니다.

    실제로 그렇게 읽었습니다. 근거에 "주장은 증가/개선" 이 찍혔습니다.
    """
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="카카오",
        account=account,
        bsns_year=year,
        quarter=0,
        assertion=FinancialAssertion.DIRECTION,
        amount_text="",
        growth_percent=0.0,
        direction=Direction.UP,
        baseline=Baseline.UNKNOWN,
        fs_hint="연결",
        note="",
    )
    return build_query(parsed, claim_text=text)


# ---------------------------------------------------------------------------
# 가려내기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("카카오의 2025년 영업이익은 흑자로 전환됐다", (TO_PROFIT, "영업이익")),
        ("카카오의 2025년 영업이익이 흑자 전환했다", (TO_PROFIT, "영업이익")),
        ("카카오의 2025년 순이익은 흑자로 돌아섰다", (TO_PROFIT, "당기순이익")),
        ("카카오의 2025년 영업이익은 적자로 전환됐다", (TO_LOSS, "영업이익")),
        ("카카오의 2023년 영업이익은 적자를 기록하였다", (LOSS, "영업이익")),
        ("카카오의 2023년 영업이익은 흑자였다", (PROFIT, "영업이익")),
    ],
)
def test_부호_주장을_가려낸다(text, expected):
    assert detect(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "카카오의 2025년 영업이익은 2,000억 원 흑자로 전환됐다",   # 금액 주장이 맡습니다
        "카카오의 2025년 광고 사업 영업이익이 흑자로 전환됐다",     # 부문 주장
        "비용 절감 때문에 영업이익이 흑자로 전환됐다",              # 인과 주장
        "매출은 증가했고 영업이익은 흑자로 전환됐다",               # 계정이 둘
        "카카오의 2025년 매출이 증가했다",                         # 부호 주장이 아님
    ],
)
def test_손대지_않는_경우(text):
    assert detect(text) == ("", "")


def test_방향_되살리기가_흑자전환을_집어삼키지_않는다():
    """전에는 UP_WORDS 에 '흑자 전환' 이 들어 있어 증가로 읽혔습니다."""
    assert find_direction("영업이익이 흑자로 전환됐다") == ""
    assert rescue("카카오의 2025년 영업이익이 흑자로 전환됐다") == ("", "")


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------


def test_이미_흑자였으면_전환이_아니다(snapshot):
    """실측 오탐을 고정합니다. 전에는 '근거 있음' 이 났습니다."""
    query = claim("카카오의 2025년 연결 영업이익은 전년 대비 흑자로 전환됐다", 2025)
    assert query.turnaround == TO_PROFIT
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "이미 흑자" in result.reason


def test_진짜_흑자_전환이면_근거_있음이다(snapshot):
    """가드가 과해서 맞는 주장까지 막으면 안 됩니다."""
    query = claim(
        "카카오의 2025년 연결 순이익은 전년 대비 흑자로 전환됐다", 2025, "당기순이익"
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert "적자" in result.reason and "흑자" in result.reason


def test_적자_전환_주장도_부호로_본다(snapshot):
    """2024년 -1,618억 -> 2025년 +5,179억. 적자 전환이 아닙니다."""
    query = claim(
        "카카오의 2025년 연결 순이익은 적자로 전환됐다", 2025, "당기순이익"
    )
    assert query.turnaround == TO_LOSS
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED


def test_상태_주장은_직전_수치_없이도_판정한다(snapshot):
    """'적자를 기록했다' 는 증감이 아니라 지금 부호에 대한 주장입니다."""
    query = claim("카카오의 2024년 연결 순이익은 적자를 기록했다", 2024, "당기순이익")
    assert query.turnaround == LOSS
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


def test_흑자인데_적자라고_하면_모순이다(snapshot):
    query = claim("카카오의 2025년 연결 영업이익은 적자를 기록했다", 2025)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "적자가 아닙니다" in result.reason


def test_비교할_직전_수치가_없으면_확인_불가다(snapshot):
    """전환은 직전 수치가 있어야 판정할 수 있습니다."""
    query = claim("카카오의 2024년 연결 영업이익은 흑자로 전환됐다", 2024)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "부호가 바뀌었는지" in result.reason


# ---------------------------------------------------------------------------
# 인과 표시는 '원인' 말고도 많습니다
# ---------------------------------------------------------------------------
#
# gpt-4o-mini 답변 40건의 참 판정을 전수로 되짚다 나왔습니다.
#
#   "한국전력은 2023년에 원자재 가격 상승과 전력 판매 단가 하락이 겹치면서
#    영업손실이 발생했다"
#
# 적자는 사실입니다(연결 -4조 5,416억원). 그런데 **왜 적자였는지는 아무것도
# 확인하지 않고** 근거 있음이 났습니다. 인과 가드는 이미 있었지만
# '겹치면서' 가 목록에 없었습니다.


@pytest.mark.parametrize(
    "text",
    [
        "한국전력은 2023년에 원자재 가격 상승과 전력 판매 단가 하락이 겹치면서 영업손실이 발생했다",
        "LG화학은 2023년 업황 부진의 여파로 영업이익이 적자였다",
        "포스코홀딩스는 철강 가격 하락의 영향으로 영업손실을 기록했다",
        "카카오는 일회성 비용 탓에 영업이익이 적자였다",
    ],
)
def test_원인이_붙은_부호_주장은_판정하지_않는다(text):
    """방향(적자)이 맞아도 **원인까지 맞은 것은 아닙니다.**"""
    assert detect(text) == ("", "")


def test_원인이_없으면_그대로_판정한다():
    """가드가 과하면 멀쩡한 부호 주장을 버립니다."""
    assert detect("한국전력은 2023년에 영업손실이 발생했다") == ("LOSS", "영업이익")
