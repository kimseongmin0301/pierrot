"""연결과 별도가 엇갈렸을 때 그 사실을 사유에 남깁니다.

기준을 밝히지 않은 주장은 **둘 중 하나라도 맞으면 인정**합니다.
그 편이 오탐을 막기 때문입니다. 주장이 어느 기준을 가리키는지 알 수 없는데
한쪽으로만 판정하면 맞는 주장을 틀렸다고 하게 됩니다.

하지만 **한쪽만 맞았다는 사실 자체가 정보**입니다.

    현대차 2026년 2분기 당기순이익
      연결 3조 2,503억 -> 2조 8,879억 (감소)
      별도 1조 3,513억 -> 3조   344억 (증가)   <- 주장과 일치

"순이익이 증가했다" 는 별도 기준으로만 맞습니다. 보통 '현대차 순이익' 이라고
하면 연결을 가리키므로, 사유에 적어 두지 않으면 **연결 기준으로는 반대였다는
사실이 가려집니다.**

판정 자체는 바꾸지 않습니다. 사유에 덧붙이기만 합니다. 그래야 4단계 지표가
흔들리지 않으면서 읽는 사람이 판단할 수 있습니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.financials import FinancialFact, FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    NumericQuery,
    ParsedNumericClaim,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim
from pierrot.tools.universe import find_stock

CODE = "005380"  # 현대차


def fact(year: int, quarter: int, account: str, amount: int, fs_div: str) -> dict:
    return {
        "code": CODE,
        "corp_code": "00164742",
        "bsns_year": year,
        "reprt_code": "11011" if quarter == 0 else "11012",
        "period_label": "연간" if quarter == 0 else f"{quarter}분기",
        "quarter": quarter,
        "period_months": 12 if quarter == 0 else 3,
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
    """연결은 줄고 별도는 늘어난 해. 실제 현대차 2026년 2분기 모양입니다."""
    return FinancialSnapshot.from_records(
        [
            fact(2025, 2, "당기순이익", 3_250_300_000_000, "CFS"),
            fact(2025, 2, "당기순이익", 1_351_300_000_000, "OFS"),
            fact(2026, 2, "당기순이익", 2_887_900_000_000, "CFS"),
            fact(2026, 2, "당기순이익", 3_034_400_000_000, "OFS"),
        ]
    )


def direction_query(direction: Direction, *, fs_hint: str = "") -> NumericQuery:
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="현대차",
        account="당기순이익",
        bsns_year=2026,
        quarter=2,
        assertion=FinancialAssertion.DIRECTION,
        amount_text="",
        growth_percent=0.0,
        direction=direction,
        baseline=Baseline.YOY,
        fs_hint=fs_hint,
        note="",
    )
    return NumericQuery(
        parsed=parsed, stock=find_stock("현대차"), amount=None, unresolved=""
    )


def test_한쪽만_맞으면_사유에_적는다(snapshot):
    """판정은 그대로 SUPPORTED 입니다. 사실만 덧붙입니다."""
    result = verify_numeric_claim(direction_query(Direction.UP), snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.mixed_basis == "연결"
    assert "연결 기준으로는 감소했습니다" in result.reason


def test_반대_방향도_같게_적는다(snapshot):
    result = verify_numeric_claim(direction_query(Direction.DOWN), snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.mixed_basis == "별도"
    assert "별도 기준으로는 증가했습니다" in result.reason


def test_기준을_밝힌_주장은_덧붙이지_않는다(snapshot):
    """주장이 '연결 기준' 이라고 적었으면 엇갈릴 일이 없습니다."""
    result = verify_numeric_claim(
        direction_query(Direction.DOWN, fs_hint="연결"), snapshot
    )
    assert result.verdict is Verdict.SUPPORTED
    assert result.mixed_basis == ""
    assert "다만" not in result.reason


def test_둘_다_맞으면_덧붙이지_않는다():
    """엇갈리지 않았는데 적으면 없는 경고를 만드는 것입니다."""
    both_up = FinancialSnapshot.from_records(
        [
            fact(2025, 0, "매출액", 100_000_000_000_000, "CFS"),
            fact(2025, 0, "매출액", 50_000_000_000_000, "OFS"),
            fact(2026, 0, "매출액", 120_000_000_000_000, "CFS"),
            fact(2026, 0, "매출액", 60_000_000_000_000, "OFS"),
        ]
    )
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="현대차",
        account="매출액",
        bsns_year=2026,
        quarter=0,
        assertion=FinancialAssertion.DIRECTION,
        amount_text="",
        growth_percent=0.0,
        direction=Direction.UP,
        baseline=Baseline.YOY,
        fs_hint="",
        note="",
    )
    query = NumericQuery(
        parsed=parsed, stock=find_stock("현대차"), amount=None, unresolved=""
    )
    result = verify_numeric_claim(query, both_up)
    assert result.verdict is Verdict.SUPPORTED
    assert result.mixed_basis == ""
    assert "다만" not in result.reason


def test_둘_다_틀리면_모순이고_덧붙이지_않는다(snapshot):
    """맞은 기준이 없으면 '엇갈림' 이 아니라 그냥 틀린 것입니다."""
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="현대차",
        account="당기순이익",
        bsns_year=2026,
        quarter=2,
        assertion=FinancialAssertion.AMOUNT,
        amount_text="100조원",
        growth_percent=0.0,
        direction=Direction.NONE,
        baseline=Baseline.UNKNOWN,
        fs_hint="",
        note="",
    )
    query = NumericQuery(
        parsed=parsed,
        stock=find_stock("현대차"),
        amount=100_000_000_000_000,
        unresolved="",
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.mixed_basis == ""
