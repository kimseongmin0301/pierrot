"""'2조 원에서 3조 원으로 늘었다' — 숫자가 둘인 문장.

실측 오탐(가짜 데이터셋 08번):

    "LG엔솔의 2025년 연결 영업이익은 2조 원에서 3조 원으로 늘었습니다"

    실제 연결 영업이익  2024년 5,753억  ->  2025년 1조 3,461억

늘어난 것은 맞지만 **2조도 3조도 아닙니다.** 파서가 이 문장을 방향 주장으로
읽어 적힌 숫자를 통째로 무시했습니다.

같은 문형인데 처리가 갈렸던 것이 문제의 핵심입니다.

    10번 "2조 원에서 3조 원으로 늘었습니다"  -> 금액 주장으로 읽어 적발
    16번 "20조 원에서 18조 원으로 감소"      -> 증감률로 읽어 적발
    08번 "2조 원에서 3조 원으로 늘었다"      -> 방향 주장으로 읽어 **통과**

프롬프트로 부탁해서 될 일이 아닙니다. 파이썬이 먼저 집습니다.
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
from pierrot.tools.range_claim import detect

CODE = "373220"  # LG에너지솔루션


def record(year: int, account: str, amount: int, fs_div: str = "CFS") -> dict:
    return {
        "code": CODE,
        "corp_code": "01515323",
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
    """실제 LG에너지솔루션 연결 영업이익 모양."""
    return FinancialSnapshot.from_records(
        [
            record(2024, "영업이익", 575_300_000_000),
            record(2025, "영업이익", 1_346_100_000_000),
            record(2024, "매출액", 25_620_000_000_000),
            record(2025, "매출액", 28_000_000_000_000),
        ]
    )


def claim(text: str, year: int, account: str = "영업이익"):
    """파서가 방향 주장(증가)으로 읽었다고 치고 넣습니다. 실제로 그랬습니다."""
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="LG에너지솔루션",
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
    "text,start,end",
    [
        ("영업이익은 2조 원에서 3조 원으로 늘었다", 2_000_000_000_000, 3_000_000_000_000),
        ("매출은 20조 원에서 18조 원으로 감소했다", 20_000_000_000_000, 18_000_000_000_000),
        ("순이익은 2.2조 원에서 2.0조 원으로 줄었다", 2_200_000_000_000, 2_000_000_000_000),
        ("영업이익은 전년 2조 원에서 3조 원으로 늘었다", 2_000_000_000_000, 3_000_000_000_000),
    ],
)
def test_양_끝을_읽어_낸다(text, start, end):
    assert detect(text) == (start, end)


@pytest.mark.parametrize(
    "text",
    [
        "매출이 2023년에서 2025년으로 가면서 늘었다",   # 금액이 아니라 연도
        "HBM 매출이 2조 원에서 3조 원으로 늘었다",      # 부문 주장
        "원가 하락 때문에 영업이익이 2조 원에서 3조 원으로 늘었다",  # 인과 주장
        "영업이익이 크게 늘었다",                       # 숫자가 없음
        "영업이익은 3조 원이다",                        # 숫자가 하나
    ],
)
def test_손대지_않는_경우(text):
    assert detect(text) is None


# ---------------------------------------------------------------------------
# 판정 — 양 끝을 둘 다 봅니다
# ---------------------------------------------------------------------------


def test_방향만_맞고_숫자가_틀리면_모순이다(snapshot):
    """실측 오탐을 고정합니다. 전에는 '근거 있음' 이 났습니다."""
    query = claim(
        "LG엔솔의 2025년 연결 영업이익은 2조 원에서 3조 원으로 늘었다", 2025
    )
    assert query.value_range == (2_000_000_000_000, 3_000_000_000_000)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "끝값" in result.reason
    assert "시작값" in result.reason


def test_끝값만_틀려도_모순이다(snapshot):
    query = claim(
        "LG엔솔의 2025년 연결 영업이익은 5,753억 원에서 3조 원으로 늘었다", 2025
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "끝값" in result.reason


def test_시작값만_틀려도_모순이다(snapshot):
    """끝값만 보는 구현이면 이 건이 통과합니다. 그래서 따로 고정합니다."""
    query = claim(
        "LG엔솔의 2025년 연결 영업이익은 2조 원에서 1조 3,461억 원으로 늘었다", 2025
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "시작값" in result.reason


def test_양_끝이_모두_맞으면_근거_있음이다(snapshot):
    """가드가 과해서 맞는 주장까지 막으면 안 됩니다."""
    query = claim(
        "LG엔솔의 2025년 연결 영업이익은 5,753억 원에서 1조 3,461억 원으로 늘었다",
        2025,
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert "둘 다 맞습니다" in result.reason


def test_비교_기준_수치가_없으면_확정하지_않는다(snapshot):
    """끝값만 맞았다고 '근거 있음' 이라고 하면 확인하지 않은 것을 확인했다고
    말하는 셈입니다."""
    query = claim(
        "LG엔솔의 2024년 연결 영업이익은 2조 원에서 5,753억 원으로 줄었다", 2024
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "시작값을 확인하지 못했습니다" in result.reason
