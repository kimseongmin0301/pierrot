"""'역대 최대' 같은 최상급 주장.

증명과 반증의 난이도가 다릅니다.

    증명 — 전체 이력이 필요합니다. 수집 범위가 2022년부터이므로
           그 이전을 모르는 한 "역대" 를 확정할 수 없습니다.
    반증 — **한 건이면 충분합니다.** 범위 안에 더 큰 해가 있으면
           그 주장은 확실히 거짓입니다.

그래서 반증되면 모순, 아니면 확인 불가로 두되 **무엇이 없어서 못 보는지**를
사유에 적습니다. 확인 불가의 사유는 이 프로젝트에서 결과 그 자체입니다.

가장 조심할 것은 **참인 부분으로 더 강한 주장을 통과시키는 것**입니다.
"2023년 역대 최대 영업이익" 을 "영업이익이 늘었다" 로 읽으면 늘어난 것은
맞으므로 '근거 있음' 이 납니다. 역대 최대인지는 아무것도 확인하지 않았는데도요.
인과 주장을 방향 주장으로 읽어 거짓 판정이 났던 것과 같은 종류입니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    ParsedNumericClaim,
    build_query,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim
from pierrot.tools.superlative import detect

CODE = "005380"


def record(year: int, quarter: int, account: str, amount: int) -> dict:
    return {
        "code": CODE,
        "corp_code": "00164742",
        "bsns_year": year,
        "reprt_code": "11011" if quarter == 0 else "11013",
        "period_label": "연간" if quarter == 0 else f"{quarter}분기",
        "quarter": quarter,
        "period_months": 12 if quarter == 0 else 3,
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
    """실제 현대차 모양입니다. 매출은 계속 늘고 영업이익은 2023년이 꼭지입니다.

    두 모양이 다 필요합니다. 매출로는 "범위 안에서 최대" 경우를,
    영업이익으로는 "더 큰 해가 있어 반증되는" 경우를 봅니다.
    """
    return FinancialSnapshot.from_records(
        [
            record(2022, 0, "매출액", 142_527_500_000_000),
            record(2023, 0, "매출액", 162_663_500_000_000),
            record(2024, 0, "매출액", 175_231_100_000_000),
            record(2025, 0, "매출액", 186_254_400_000_000),
            record(2026, 1, "매출액", 45_938_800_000_000),
            record(2022, 0, "영업이익", 9_819_700_000_000),
            record(2023, 0, "영업이익", 15_126_900_000_000),   # 꼭지
            record(2024, 0, "영업이익", 14_239_500_000_000),
            record(2025, 0, "영업이익", 11_467_800_000_000),
        ]
    )


def claim(text: str, year: int, *, company: str = "현대차", assertion=None):
    """파서가 돌려줄 법한 결과를 만들어 build_query 에 넣습니다."""
    parsed = ParsedNumericClaim(
        is_financial_claim=assertion is not None,
        company=company,
        account="매출액" if assertion else "",
        bsns_year=year,
        quarter=0,
        assertion=assertion or FinancialAssertion.OTHER,
        amount_text="",
        growth_percent=0.0,
        direction=Direction.UP if assertion else Direction.NONE,
        baseline=Baseline.UNKNOWN,
        fs_hint="",
        note="",
    )
    return build_query(parsed, claim_text=text)


# ---------------------------------------------------------------------------
# 가려내기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,account",
    [
        ("현대차는 2026년 역대 최대 매출을 이어가고 있다", "매출액"),
        ("삼성전자는 2025년 사상 최대 영업이익을 기록했다", "영업이익"),
        ("SK하이닉스의 2024년 매출이 최대치를 경신했다", "매출액"),
        ("현대차는 2025년 역대급 순이익을 올렸다", "당기순이익"),
    ],
)
def test_최상급_주장을_가려낸다(text, account):
    assert detect(text) == account


@pytest.mark.parametrize(
    "text,why",
    [
        ("현대차의 2026년 매출은 190조원이다", "금액이 있으면 금액 주장입니다"),
        ("SK하이닉스는 HBM 부문에서 역대 최대 매출을 올렸다", "부문 주장입니다"),
        ("현대차는 2025년에 매출이 증가하였다", "최상급 표현이 없습니다"),
        ("역대 최대 실적의 원인은 환율이다", "인과 주장입니다"),
        ("현대차는 역대 최대 규모의 투자를 집행했다", "계정 낱말이 없습니다"),
    ],
)
def test_최상급이_아니면_손대지_않는다(text, why):
    assert detect(text) == "", why


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------


def test_범위_안에_더_큰_해가_있으면_모순이다(snapshot):
    """반증은 한 건이면 충분합니다.

    현대차 영업이익은 2023년이 꼭지(15.1조)이고 2024년은 14.2조입니다.
    "2024년 역대 최대" 는 확실히 거짓입니다.
    """
    result = verify_numeric_claim(
        claim("현대차는 2024년 역대 최대 영업이익을 기록했다", 2024), snapshot
    )
    assert result.verdict is Verdict.CONTRADICTED
    assert "2023년" in result.reason
    assert "'역대 최대'가 아닙니다" in result.reason


def test_나중_연도로_과거_주장을_반박하지_않는다(snapshot):
    """2023년 주장을 2024·2025년 수치로 깎으면 안 됩니다.

    "2023년에 역대 최대였다" 는 **그때 기준** 이야기입니다.
    뒤에 더 큰 해가 와도 당시 주장이 거짓이 되지는 않습니다.
    """
    result = verify_numeric_claim(
        claim("현대차는 2023년 역대 최대 매출을 기록했다", 2023), snapshot
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "2024" not in result.reason  # 나중 해는 비교 대상이 아닙니다


def test_범위_안에서_최대면_확인_불가로_두되_이유를_적는다(snapshot):
    """증명하려면 전체 이력이 필요합니다. 없는 것을 있다고 하면 안 됩니다."""
    result = verify_numeric_claim(
        claim("현대차는 2025년 역대 최대 매출을 기록했다", 2025), snapshot
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "수집 범위(2022년~)" in result.reason
    assert "확정할 수 없습니다" in result.reason
    # 무엇과 비교했는지 남겨 둡니다.
    assert "2024년" in result.reason


def test_연간이_아직_없으면_그렇게_말한다(snapshot):
    """2026년은 1분기까지만 있습니다."""
    result = verify_numeric_claim(
        claim("현대차는 2026년 역대 최대 매출을 이어가고 있다", 2026), snapshot
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "아직 공시되지 않아" in result.reason
    assert "1분기까지만" in result.reason


def test_방향_주장으로_새지_않는다(snapshot):
    """가장 조심할 부분입니다.

    파서가 "역대 최대 매출을 기록했다" 를 DIRECTION 으로 읽어도,
    최상급 검사가 **먼저** 돌아 방향 판정으로 새지 않아야 합니다.
    방향만 보면 "늘었다" 는 맞으므로 거짓 '근거 있음' 이 납니다.
    """
    query = claim(
        "현대차는 2024년 역대 최대 영업이익을 기록했다",
        2024,
        assertion=FinancialAssertion.DIRECTION,
    )
    assert query.superlative == "영업이익"
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
