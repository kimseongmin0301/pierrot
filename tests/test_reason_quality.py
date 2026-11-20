"""사유가 아무것도 설명하지 못하면 결과가 아닙니다.

이 프로젝트에서 **확인 불가의 사유는 결과 그 자체**입니다. 무엇이 없어서
못 봤는지 적혀 있어야 다음에 무엇을 모을지 정할 수 있습니다.

실측(가짜 데이터셋)에서 사유가 뭉개진 자리가 넷 있었습니다.

    19번  "삼성물산은 2027년 연간 실적에서 매출 목표를 달성했다"
          -> "부문별 실적 주장이 아닙니다"   (부문 주장인 적이 없습니다)

    03번  "2020년부터 2025년까지 6개 연도 중 가장 높은 수치이다"
          -> "구체적인 수치가 없으므로 OTHER로 판별."
             최상급 주장인데 '역대' 라는 낱말이 없어 가려내지 못했습니다.

    10번  "전년 대비 증가액은 1,000억 원입니다"
          -> "주장 +1000.0%"  (증감액을 증감률로 읽었습니다)

    06번  "2027년 연결 매출도 2026년보다 늘어날 전망입니다"
          -> 아무 데도 나오지 않습니다. 문장이 통째로 사라졌습니다.
"""

from __future__ import annotations

import pytest

from pierrot.reporter.report import ClaimReport, find_uncovered
from pierrot.tools.delta_claim import detect as detect_delta
from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.numeric_claim import (
    Baseline,
    Direction,
    FinancialAssertion,
    ParsedNumericClaim,
    build_query,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim
from pierrot.tools.superlative import detect as detect_superlative
from pierrot.verifier.nodes import _says_nothing

CODE = "015760"  # 한국전력


def record(year: int, account: str, amount: int) -> dict:
    return {
        "code": CODE,
        "corp_code": "00159193",
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
    """실제 한국전력 연결 영업이익. 2024년 8조 3,600억 -> 2025년 13조 4,900억."""
    return FinancialSnapshot.from_records(
        [
            record(2024, "영업이익", 8_360_000_000_000),
            record(2025, "영업이익", 13_490_000_000_000),
        ]
    )


def claim(text: str, year: int, account: str = "영업이익"):
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="한국전력",
        account=account,
        bsns_year=year,
        quarter=0,
        assertion=FinancialAssertion.GROWTH,
        amount_text="",
        growth_percent=1000.0,   # 파서가 실제로 이렇게 읽었습니다
        direction=Direction.UP,
        baseline=Baseline.YOY,
        fs_hint="연결",
        note="",
    )
    return build_query(parsed, claim_text=text)


# ---------------------------------------------------------------------------
# 19번 — 아무것도 설명하지 못하는 사유는 최종 사유가 되면 안 됩니다
# ---------------------------------------------------------------------------


def test_정보가_없는_사유를_가려낸다():
    assert _says_nothing("부문별 실적 주장이 아닙니다")
    assert _says_nothing("판정 방법이 정해지지 않은 주장입니다")
    assert not _says_nothing(
        "SK하이닉스 2025년 공시에 'HBM' 부문이 없습니다. 공시된 부문: DRAM, NAND"
    )
    assert not _says_nothing("재무제표 주요계정에 없습니다")


# ---------------------------------------------------------------------------
# 03번 — '역대' 가 없는 최상급
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,account",
    [
        ("KB금융의 2025년 순이익은 6개 연도 중 가장 높은 수치이다", "당기순이익"),
        ("삼성전자의 2025년 매출은 최고 수준이다", "매출액"),
        ("현대차의 2025년 매출은 역대 최대이다", "매출액"),
    ],
)
def test_역대가_없어도_최상급으로_본다(text, account):
    assert detect_superlative(text) == account


def test_변화폭에_붙은_최상급은_수준_비교가_아니다():
    """"매출이 가장 높은 성장을 보였다" 는 매출 크기 주장이 아닙니다.

    수준 비교로 다루면 전년 매출이 더 컸다는 이유로 엉뚱한 모순이 납니다.
    """
    assert detect_superlative("현대차의 매출이 가장 높은 성장을 보였다") == ""
    assert detect_superlative("영업이익이 최대 감소폭을 기록했다") == ""


# ---------------------------------------------------------------------------
# 10번 — 증감액은 증감률이 아닙니다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,account,delta",
    [
        ("한전의 2025년 영업이익 증가액은 1,000억 원이다", "영업이익", 100_000_000_000),
        ("한전의 2025년 영업이익 증가폭은 5조 1,300억 원이다", "영업이익", 5_130_000_000_000),
        ("한전의 2025년 영업이익 감소액은 1조 원이다", "영업이익", -1_000_000_000_000),
    ],
)
def test_증감액을_읽어_낸다(text, account, delta):
    assert detect_delta(text) == (account, delta)


@pytest.mark.parametrize(
    "text",
    [
        "한전의 2025년 영업이익 증가율은 61.3%이다",   # 률은 증감률 주장
        "한전의 2025년 영업이익이 증가했다",           # 금액이 없음
        "원가 하락 때문에 영업이익 증가액이 1조 원이다",   # 인과 주장
    ],
)
def test_증감액이_아닌_것에는_손대지_않는다(text):
    assert detect_delta(text) is None


def test_알려진_한계_조사형_인과는_가려내지_못한다():
    """"원가 하락으로 ~" 처럼 조사 '으로' 만 쓴 인과는 못 가려냅니다.

    `CAUSAL_WORDS` 에 '으로' 를 넣으면 "전년 대비 5% 증가로 나타났다" 같은
    **인과가 아닌** 문장까지 막혀 커버리지가 떨어집니다. 지금은 넣지 않고
    한계로 둡니다. 이런 문장은 증감액만 확인되고 원인은 확인되지 않습니다.
    """
    assert detect_delta("원가 하락으로 영업이익 증가액이 1조 원이다") is not None


def test_틀린_증감액은_모순이고_사유에_금액이_찍힌다(snapshot):
    """전에는 사유에 '주장 +1000.0%' 가 찍혔습니다. 문장과 단위가 다릅니다."""
    query = claim("한전의 2025년 연결 영업이익 증가액은 1,000억 원이다", 2025)
    assert query.delta == 100_000_000_000
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "%" not in result.reason
    assert "1,000억원" in result.reason
    assert "5조 1,300억원" in result.reason   # 실제 증감액


def test_맞는_증감액은_근거_있음이다(snapshot):
    query = claim("한전의 2025년 연결 영업이익 증가액은 5조 1,300억 원이다", 2025)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


# ---------------------------------------------------------------------------
# 06번 — 버린 문장을 적어 둡니다
# ---------------------------------------------------------------------------


def report_claim(start: int, end: int) -> ClaimReport:
    return ClaimReport(
        claim_id="c1",
        text="x",
        claim_type="NUMERIC",
        route="NUMERIC",
        verdict="SUPPORTED",
        confidence="CALCULATED",
        reason="",
        evidence=[],
        start=start,
        end=end,
        is_forecast=False,
        route_changed=False,
    )


def test_전망_문장이_사라지지_않고_사유와_함께_남는다():
    answer = (
        "현대차의 2025년 연결 매출은 전년보다 증가했습니다. "
        "2027년 연결 매출도 2026년보다 늘어날 전망입니다."
    )
    first = "현대차의 2025년 연결 매출은 전년보다 증가했습니다."
    found = find_uncovered(answer, [report_claim(0, len(first))])
    assert len(found) == 1
    assert "2027년" in found[0].text
    assert "전망" in found[0].why


def test_모든_문장이_다뤄졌으면_비어_있다():
    answer = "현대차의 2025년 매출은 늘었습니다."
    found = find_uncovered(answer, [report_claim(0, len(answer))])
    assert found == []


def test_위치를_모르는_주장이_있으면_단정하지_않는다():
    """위치 없는 주장이 그 문장을 다뤘을 수도 있습니다.

    다루지 않았다고 단정하면 거짓말이 됩니다.
    """
    answer = "첫 문장입니다. 둘째 문장입니다."
    nowhere = ClaimReport(
        claim_id="c1", text="x", claim_type="NUMERIC", route="NUMERIC",
        verdict="SUPPORTED", confidence="CALCULATED", reason="", evidence=[],
        start=None, end=None, is_forecast=False, route_changed=False,
    )
    assert find_uncovered(answer, [nowhere]) == []


def test_금액_없는_부문_주장의_사유를_구분한다():
    """전에는 "금액을 읽을 수 없습니다: (표기 없음)" 하나로 끝났습니다.

    금액이 아예 없는 것과 표기를 못 읽은 것은 전혀 다른 문제입니다.
    앞은 도구의 한계이고 뒤는 고칠 수 있는 버그입니다.
    """
    from pierrot.tools.segment_claim import ParsedSegmentClaim
    from pierrot.tools.segment_claim import build_query as build_segment_query

    def parsed(amount_text: str) -> ParsedSegmentClaim:
        return ParsedSegmentClaim(
            is_segment_claim=True,
            company="삼성SDI",
            segment="전지",
            metric="매출",
            bsns_year=2025,
            quarter=0,
            amount_text=amount_text,
            note="",
        )

    empty = build_segment_query(parsed(""))
    assert "금액이 적혀 있지 않습니다" in empty.unresolved
    assert "전지" in empty.unresolved

    unreadable = build_segment_query(parsed("약 스무 조"))
    assert "읽지 못했습니다" in unreadable.unresolved
    assert "약 스무 조" in unreadable.unresolved
