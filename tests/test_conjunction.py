"""한 문장에 사실이 여럿인 주장.

분해 단계가 "한 주장에는 사실 하나만" 을 늘 지키지는 못합니다.

    "2024년에는 매출이 다시 증가하였으나 영업이익과 순이익은
     각각 줄어드는 추세를 보였다"

이걸 하나의 방향 주장으로 다루면 **앞부분만 보고 통과시킵니다.**
실제로 그랬습니다. 매출은 늘었으므로 '근거 있음' 이 났는데,
SK하이닉스 2024년 영업이익은 -7조 7,303억에서 +23조 4,673억으로 **늘었습니다.**
주장은 거짓인데 참으로 판정됐습니다. 지금까지 중 가장 나쁜 오탐입니다.

분해를 고치는 것과 별개로 **판정기가 문장 일부만 보고 통과시키지 않도록**
막아야 합니다. 분해가 완벽해질 수는 없기 때문입니다.

묶는 규칙은 '그리고' 입니다. 한 문장이 여러 사실을 이어 붙였으면
**하나라도 틀리면 그 문장은 틀린 것**입니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.direction_rescue import find_all_accounts, find_claims, split_segments
from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    ParsedNumericClaim,
    build_query,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim

CODE = "000660"  # SK하이닉스


def record(year: int, account: str, amount: int) -> dict:
    return {
        "code": CODE,
        "corp_code": "00164779",
        "bsns_year": year,
        "reprt_code": "11011",
        "period_label": "연간",
        "quarter": 0,
        "period_months": 12,
        "fs_div": "CFS",
        "fs_nm": "연결재무제표",
        "account": account,
        "amount": amount,
        "period_start": "",
        "period_end": "",
        "rcept_no": "X",
    }


@pytest.fixture
def snapshot():
    """실제 SK하이닉스 모양. 2023년은 적자, 2024년은 크게 흑자 전환입니다.

    증감은 전년 수치가 있어야 계산되므로 2022년도 넣습니다.
    """
    return FinancialSnapshot.from_records(
        [
            record(2022, "매출액", 44_621_500_000_000),
            record(2022, "영업이익", 6_809_400_000_000),
            record(2022, "당기순이익", 2_241_600_000_000),
            record(2023, "매출액", 32_765_700_000_000),
            record(2023, "영업이익", -7_730_300_000_000),
            record(2023, "당기순이익", -9_137_500_000_000),
            record(2024, "매출액", 66_192_900_000_000),
            record(2024, "영업이익", 23_467_300_000_000),
            record(2024, "당기순이익", 19_796_900_000_000),
        ]
    )


def claim(text: str, year: int):
    """파서가 방향 주장으로 읽었다고 치고 build_query 에 넣습니다."""
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="SK하이닉스",
        account="매출액",
        bsns_year=year,
        quarter=0,
        assertion=FinancialAssertion.DIRECTION,
        amount_text="",
        growth_percent=0.0,
        direction=Direction.UP,
        baseline=Baseline.UNKNOWN,
        fs_hint="",
        note="",
    )
    return build_query(parsed, claim_text=text)


# ---------------------------------------------------------------------------
# 문장 가르기
# ---------------------------------------------------------------------------


def test_대조_이음말로_절을_가른다():
    parts = split_segments("매출이 증가하였으나 영업이익은 감소하였다")
    assert len(parts) == 2
    assert "매출" in parts[0] and "영업이익" in parts[1]


def test_한_절의_계정을_모두_찾는다():
    """'영업이익과 순이익은 줄었다' 에서 둘 다 찾아야 합니다."""
    assert set(find_all_accounts("영업이익과 순이익은 줄어드는 추세")) == {
        "영업이익",
        "당기순이익",
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "2023년에 매출이 증가하였으나 영업이익은 다소 감소하였다",
            {("매출액", "UP"), ("영업이익", "DOWN")},
        ),
        (
            "2024년에 매출이 다시 증가하였으나 영업이익과 순이익은 각각 줄어드는 추세를 보였다",
            {("매출액", "UP"), ("영업이익", "DOWN"), ("당기순이익", "DOWN")},
        ),
        (
            "2026년 1분기에 매출과 영업이익이 모두 증가하였다",
            {("매출액", "UP"), ("영업이익", "UP")},
        ),
    ],
)
def test_문장에서_사실을_모두_뽑는다(text, expected):
    assert set(find_claims(text)) == expected


def test_사실이_하나면_그대로_하나다():
    assert find_claims("매출이 지속적으로 상승하였다") == [("매출액", "UP")]


def test_방향을_못_읽는_절이_있으면_통째로_포기한다():
    """일부만 확인하고 넘어가는 것이 가장 위험합니다."""
    assert find_claims("매출이 증가하였고 영업이익은 그대로였다") == []


# ---------------------------------------------------------------------------
# 판정 — 하나라도 틀리면 문장이 틀린 것입니다
# ---------------------------------------------------------------------------


def test_뒷부분이_틀리면_모순이다(snapshot):
    """실측 오탐을 고정합니다.

    전에는 매출만 보고 '근거 있음' 이 났습니다. 영업이익은 -7.7조에서
    +23.5조로 늘었는데도요.
    """
    query = claim(
        "SK하이닉스는 2024년에 매출이 다시 증가하였으나 영업이익과 순이익은 각각 줄어드는 추세를 보였다",
        2024,
    )
    assert len(query.conjunction) == 3
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "영업이익" in result.reason
    assert "틀림" in result.reason


def test_앞부분이_틀려도_모순이다(snapshot):
    query = claim(
        "SK하이닉스는 2023년에 매출이 증가하였으나 영업이익은 다소 감소하였다", 2023
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "매출액" in result.reason


def test_전부_맞으면_근거_있음이다(snapshot):
    """가드가 과해서 맞는 문장까지 막으면 안 됩니다."""
    query = claim(
        "SK하이닉스는 2024년에 매출과 영업이익이 모두 증가하였다", 2024
    )
    assert len(query.conjunction) == 2
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert "전부 맞습니다" in result.reason


def test_확인_못_한_부분이_있으면_확정하지_않는다(snapshot):
    """일부를 못 보면 문장 전체를 확정할 수 없습니다."""
    query = claim(
        "SK하이닉스는 2024년에 매출이 증가하였으나 영업활동현금흐름은 감소하였다",
        2024,
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "확정할 수 없습니다" in result.reason
