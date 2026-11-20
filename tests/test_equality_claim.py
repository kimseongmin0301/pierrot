"""'전년과 같은 수준' — 변화가 없다는 것도 주장입니다.

놓쳤던 실측(가짜 데이터셋):

    07번 "하나금융지주 2025년 순이익은 전년과 같은 수준입니다"
         실제 3조 7,700억 -> 4조 366억  (+7.1%)

    13번 "SKT 2026년 2분기 순이익은 전년 동기와 같았습니다"
         실제 800억 -> 4조 7,000억 수준  (+488%)

둘 다 "증감 방향이 분명하지 않습니다" 로 흘려보냈습니다. 파서가
direction=NONE 을 돌려주기 때문입니다. 오탐은 아니지만 **잡을 수 있는
거짓을 놓친 것**입니다. 13번은 누가 봐도 같은 수준이 아닙니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.equality_claim import detect
from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    ParsedNumericClaim,
    build_query,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim

CODE = "086790"  # 하나금융지주


def record(year: int, account: str, amount: int, fs_div: str = "CFS") -> dict:
    return {
        "code": CODE,
        "corp_code": "00547583",
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
    """세 가지 구간을 모두 담습니다.

        당기순이익  3조 7,700억 -> 4조   366억  (+7.1%)  애매한 구간
        매출액      10조        -> 10조 1,000억 (+1.0%)  같은 수준
        영업이익     2조        ->  4조          (+100%)  분명히 다름
    """
    return FinancialSnapshot.from_records(
        [
            record(2024, "당기순이익", 3_770_000_000_000),
            record(2025, "당기순이익", 4_036_600_000_000),
            record(2024, "매출액", 10_000_000_000_000),
            record(2025, "매출액", 10_100_000_000_000),
            record(2024, "영업이익", 2_000_000_000_000),
            record(2025, "영업이익", 4_000_000_000_000),
        ]
    )


def claim(text: str, year: int, account: str):
    """파서가 방향 없음(NONE)으로 읽었다고 치고 넣습니다. 실제로 그랬습니다."""
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="하나금융지주",
        account=account,
        bsns_year=year,
        quarter=0,
        assertion=FinancialAssertion.DIRECTION,
        amount_text="",
        growth_percent=0.0,
        direction=Direction.NONE,
        baseline=Baseline.UNKNOWN,
        fs_hint="연결",
        note="",
    )
    return build_query(parsed, claim_text=text)


# ---------------------------------------------------------------------------
# 가려내기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,account",
    [
        ("하나금융지주의 2025년 순이익은 전년과 같은 수준이다", "당기순이익"),
        ("하나금융지주의 2026년 2분기 순이익은 전년 동기와 같았다", "당기순이익"),
        ("하나금융지주의 2025년 매출은 전년과 비슷한 수준이다", "매출액"),
        ("하나금융지주의 2025년 영업이익은 전년과 차이가 없었다", "영업이익"),
        ("하나금융지주의 2025년 매출은 보합세를 보였다", "매출액"),
    ],
)
def test_동등_주장을_가려낸다(text, account):
    assert detect(text) == account


@pytest.mark.parametrize(
    "text,why",
    [
        ("하나금융지주의 2025년 순이익은 전년과 같은 4조 원이다", "금액이 있으면 금액 주장"),
        ("하나금융지주의 광고 부문 매출은 전년과 같은 수준이다", "부문 주장"),
        ("비용 절감 덕분에 순이익이 전년과 같은 수준이다", "인과 주장"),
        ("매출은 증가했고 순이익은 전년과 같은 수준이다", "계정이 둘"),
        ("하나금융지주의 2025년 순이익은 성장세를 유지했다", "'유지'는 계속 늘었다는 뜻"),
        ("하나금융지주의 2025년 순이익은 늘어난 뒤 비슷한 수준이다", "증가 낱말이 섞임"),
    ],
)
def test_손대지_않는_경우(text, why):
    assert detect(text) == "", why


# ---------------------------------------------------------------------------
# 판정 — 구간 셋
# ---------------------------------------------------------------------------


def test_분명히_다르면_모순이다(snapshot):
    """13번 실측을 고정합니다. +100% 는 누가 봐도 같은 수준이 아닙니다."""
    query = claim(
        "하나금융지주의 2025년 연결 영업이익은 전년과 같은 수준이다", 2025, "영업이익"
    )
    assert query.equality == "영업이익"
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "같은 수준이라 할 수 없습니다" in result.reason


def test_정말_같은_수준이면_근거_있음이다(snapshot):
    """가드가 과해서 맞는 주장까지 막으면 안 됩니다."""
    query = claim(
        "하나금융지주의 2025년 연결 매출은 전년과 같은 수준이다", 2025, "매출액"
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert "안쪽입니다" in result.reason


def test_애매한_구간은_실제_변동률을_적고_확인_불가로_둔다(snapshot):
    """07번 실측. +7.1% 는 억지로 한쪽으로 몰 수 없습니다.

    다만 전처럼 "증감 방향이 분명하지 않습니다" 로 뭉개지 않고
    실제 변동률을 적어 둡니다. 읽는 사람이 직접 판단할 수 있습니다.
    """
    query = claim(
        "하나금융지주의 2025년 연결 순이익은 전년과 같은 수준이다", 2025, "당기순이익"
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "+7.1%" in result.reason
    assert "가르기 어렵습니다" in result.reason


def test_비교_대상이_없으면_확인_불가다(snapshot):
    query = claim(
        "하나금융지주의 2024년 연결 매출은 전년과 같은 수준이다", 2024, "매출액"
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "변화가 없었는지" in result.reason
