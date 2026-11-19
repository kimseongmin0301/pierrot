"""크기 없이 방향만 말한 주장을 판정합니다.

실제 글에는 수치 없이 방향만 적는 문장이 많습니다.

    "2023년에는 매출이 증가하였으며, 영업이익과 순이익 또한 상승하였습니다"
    "2024년에는 영업이익 증가폭이 크게 나타났습니다"

크기를 안 밝혔어도 **방향은 대조할 수 있습니다.** 실제로 삼성전자 2023년은
매출·영업이익·순이익이 모두 줄어든 해였으므로 위 문장은 세 항목 모두 거짓입니다.

이 파일은 두 가지를 고정합니다.

1. 방향 주장이 판정 경로를 타는지 (전에는 부문 경로로 새서 버려졌습니다)
2. 금액이 적혀 있으면 **언제나 금액 주장이 먼저인지**
   (방향 판정을 넓히다가 "2.2조원 수준이다" 까지 방향으로 읽어
    4단계 커버리지가 100% 에서 99% 로 떨어졌습니다)
"""

from __future__ import annotations

import pytest

from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    NumericQuery,
    ParsedNumericClaim,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim
from pierrot.tools.universe import find_stock


@pytest.fixture(scope="module")
def snapshot():
    return FinancialSnapshot.load()


def direction_query(
    account: str, year: int, direction: Direction, *, company: str = "삼성전자"
) -> NumericQuery:
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company=company,
        account=account,
        bsns_year=year,
        quarter=0,
        assertion=FinancialAssertion.DIRECTION,
        amount_text="",
        growth_percent=0.0,
        direction=direction,
        baseline=Baseline.UNKNOWN,
        fs_hint="",
        note="",
    )
    return NumericQuery(
        parsed=parsed, stock=find_stock(company), amount=None, unresolved=""
    )


# ---------------------------------------------------------------------------
# 2022년을 받아야 2023년 증감을 판정할 수 있습니다
# ---------------------------------------------------------------------------


def test_스냅샷에_2022년이_있다(snapshot):
    """증감은 전년 수치가 있어야 계산됩니다.

    2023년부터 받았을 때는 "2023년에 매출이 증가했다" 가 전부 확인 불가였습니다.
    보고 싶은 기간보다 **한 해 앞서** 받아야 합니다.
    """
    facts = snapshot.lookup("005930", 2022, "영업이익", quarter=0, fs_div="CFS")
    assert facts, "2022년이 없으면 2023년 증감을 판정할 수 없습니다"


@pytest.mark.parametrize("account", ["매출액", "영업이익", "당기순이익"])
def test_삼성전자_2023년은_모두_감소했다(snapshot, account):
    """실제 사용자 글에 들어 있던 문장입니다.

    "2023년에는 매출이 증가하였으며, 영업이익과 순이익 또한 상승하였습니다"
    -> 세 항목 모두 반대입니다.
    """
    result = verify_numeric_claim(
        direction_query(account, 2023, Direction.UP), snapshot
    )
    assert result.verdict is Verdict.CONTRADICTED
    assert "방향이 맞지 않습니다" in result.reason


def test_방향이_맞으면_인정한다(snapshot):
    """가드가 과해서 맞는 주장까지 막으면 안 됩니다."""
    result = verify_numeric_claim(
        direction_query("영업이익", 2024, Direction.UP), snapshot
    )
    assert result.verdict is Verdict.SUPPORTED


# ---------------------------------------------------------------------------
# 금액이 있으면 금액 주장이 먼저입니다
# ---------------------------------------------------------------------------

# 파서가 고르는 주장 종류를 고정합니다. LLM 호출 없이 프롬프트의 규칙만
# 글로 남겨 두면 다음에 프롬프트를 고칠 때 근거가 사라집니다.
PRECEDENCE_CASES = [
    ("NAVER의 2025년 영업이익은 2.2조원 수준이다", "AMOUNT"),
    ("삼성전자의 2024년 매출은 약 300조원이다", "AMOUNT"),
    ("삼성전자는 2024년에 영업이익 증가폭이 크게 나타났다", "DIRECTION"),
    ("삼성전자는 2026년 1분기에 매출이 우수하게 증가하였다", "DIRECTION"),
]


@pytest.mark.parametrize("text,expected", PRECEDENCE_CASES)
def test_주장_종류_판별_기준이_문서로_남아_있다(text, expected):
    """프롬프트에 적힌 경계를 테스트로 박아 둡니다.

    "수준이다" 가 붙어도 금액이 적혀 있으면 AMOUNT 입니다.
    이 경계를 잃으면 4단계 커버리지가 떨어집니다(실제로 100% -> 99% 로 떨어졌습니다).
    LLM 호출 없이 프롬프트 본문에 규칙이 남아 있는지만 확인합니다.
    """
    from pierrot.tools.numeric_claim import SYSTEM_PROMPT

    assert "(1) 금액이 적혀 있는가?" in SYSTEM_PROMPT
    assert "DIRECTION 은 **금액도 증감률도 적혀 있지 않을 때만** 씁니다" in SYSTEM_PROMPT
    # 부문 주장은 금액이 있어도 OTHER 입니다. 이 경계를 잃으면
    # 부문 매출을 전사 매출과 비교해 거짓 '근거 있음' 이 납니다.
    assert "수치가 있든 없든** OTHER" in SYSTEM_PROMPT
