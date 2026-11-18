"""재무 주장 판정 테스트.

LLM 호출은 하지 않습니다. 파싱 결과를 손으로 만들어 넣습니다.
수치는 DART 에서 실제로 받은 삼성전자 2023~2024년 값입니다.
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


def record(**kwargs) -> dict:
    base = {
        "code": "005930",
        "corp_code": "00126380",
        "bsns_year": 2024,
        "reprt_code": "11014",
        "period_label": "3분기",
        "quarter": 3,
        "period_months": 3,
        "fs_div": "CFS",
        "fs_nm": "연결재무제표",
        "account": "영업이익",
        "amount": 0,
        "period_start": "2024.01.01",
        "period_end": "2024.09.30",
        "currency": "KRW",
        "rcept_no": "20241114",
    }
    base.update(kwargs)
    return base


# 삼성전자 실제 수치 (연결 / 별도)
RECORDS = [
    # 2024년 3분기
    record(amount=9_183_371_000_000),
    record(fs_div="OFS", fs_nm="재무제표", amount=3_900_000_000_000),
    # 2023년 3분기 (전년 동기) — 2024년 3분기가 전년 대비 크게 늘었습니다
    record(bsns_year=2023, amount=2_433_000_000_000),
    record(bsns_year=2023, fs_div="OFS", fs_nm="재무제표", amount=-2_000_000_000_000),
    # 2024년 연간
    record(reprt_code="11011", period_label="연간", quarter=0, period_months=12,
           amount=32_725_961_000_000, period_end="2024.12.31"),
    record(reprt_code="11011", period_label="연간", quarter=0, period_months=12,
           fs_div="OFS", fs_nm="재무제표", amount=12_361_034_000_000,
           period_end="2024.12.31"),
    # 2024년 1~2분기 (4분기 계산용)
    record(reprt_code="11013", period_label="1분기", quarter=1, amount=6_606_009_000_000),
    record(reprt_code="11012", period_label="2분기", quarter=2, amount=10_443_878_000_000),
    # 매출액
    record(account="매출액", amount=79_100_000_000_000),
]


@pytest.fixture
def snapshot() -> FinancialSnapshot:
    return FinancialSnapshot.from_records(RECORDS)


def make_query(**kwargs):
    base = dict(
        is_financial_claim=True,
        company="삼성전자",
        account="영업이익",
        bsns_year=2024,
        quarter=3,
        fs_hint="",
        assertion=FinancialAssertion.AMOUNT,
        amount_text="",
        growth_percent=0.0,
        baseline=Baseline.UNKNOWN,
        direction=Direction.NONE,
        note="",
    )
    base.update(kwargs)
    return build_query(ParsedNumericClaim(**base))


# ---------------------------------------------------------------------------
# 조회 조건 만들기
# ---------------------------------------------------------------------------


def test_query_parses_korean_amount():
    """금액 변환은 파이썬이 합니다. 자릿수를 틀리면 1만 배 차이가 납니다."""
    query = make_query(amount_text="9조 1,800억원")
    assert query.is_resolved
    assert query.amount == 9_180_000_000_000


def test_query_fails_for_unknown_company():
    query = make_query(company="애플", amount_text="9조원")
    assert not query.is_resolved
    assert "유니버스에 없는 종목" in query.unresolved


def test_query_fails_for_unknown_account():
    """재무제표에 없는 계정은 조회할 수 없습니다."""
    query = make_query(account="수주액", amount_text="15조원")
    assert not query.is_resolved
    assert "다룰 수 없는 계정" in query.unresolved


def test_query_fails_without_year():
    query = make_query(bsns_year=0, amount_text="9조원")
    assert not query.is_resolved
    assert "사업연도" in query.unresolved


def test_query_fails_for_unreadable_amount():
    query = make_query(amount_text="상당한 규모")
    assert not query.is_resolved
    assert "금액을 읽을 수 없습니다" in query.unresolved


def test_other_assertion_is_not_resolved():
    query = make_query(assertion=FinancialAssertion.OTHER, is_financial_claim=False)
    assert not query.is_resolved


@pytest.mark.parametrize(
    "hint,expected",
    [("연결", "CFS"), ("별도", "OFS"), ("개별", "OFS"), ("", None)],
)
def test_fs_hint_mapping(hint, expected):
    assert make_query(fs_hint=hint, amount_text="9조원").fs_div == expected


# ---------------------------------------------------------------------------
# 금액 판정
# ---------------------------------------------------------------------------


def test_amount_matches_consolidated(snapshot):
    """연결 기준 9조 1,833억원. '9조 1,800억원' 주장은 인정합니다."""
    query = make_query(amount_text="9조 1,800억원")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.matched_fs == "CFS"
    assert result.fs_explicit is False


def test_amount_matches_separate_statement(snapshot):
    """별도 기준(3조 9,000억원)과 맞아도 인정하되 어느 쪽인지 남깁니다.

    LLM 답변은 연결/별도를 밝히지 않으므로 하나만 보면 오탐이 납니다.
    """
    query = make_query(amount_text="3조 9,000억원")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.matched_fs == "OFS"


def test_amount_contradicted(snapshot):
    """실제 9조원대인데 6조 1,879억원이라고 하면 모순입니다."""
    query = make_query(amount_text="6조 1,879억원")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.matched_fs == ""


def test_explicit_fs_hint_narrows_comparison(snapshot):
    """연결이라고 밝혔으면 별도와 맞아도 인정하지 않습니다."""
    query = make_query(amount_text="3조 9,000억원", fs_hint="연결")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.fs_explicit is True


def test_annual_and_quarterly_are_not_mixed(snapshot):
    """연간 32조원을 3분기 주장으로 인정하면 안 됩니다."""
    quarterly = make_query(quarter=3, amount_text="32조 7,259억원")
    annual = make_query(quarter=0, amount_text="32조 7,259억원")
    assert verify_numeric_claim(quarterly, snapshot).verdict is Verdict.CONTRADICTED
    assert verify_numeric_claim(annual, snapshot).verdict is Verdict.SUPPORTED


def test_missing_data_is_unverifiable(snapshot):
    query = make_query(bsns_year=2019, amount_text="9조원")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "스냅샷에 해당 수치가 없습니다" in result.reason


def test_reason_is_human_readable(snapshot):
    """판정만 있고 근거가 없으면 사람이 신뢰할 수 없습니다."""
    query = make_query(amount_text="6조원")
    result = verify_numeric_claim(query, snapshot)
    assert "삼성전자" in result.reason
    assert "9조 1,833억원" in result.reason
    assert result.evidence


# ---------------------------------------------------------------------------
# 4분기 계산
# ---------------------------------------------------------------------------


def test_fourth_quarter_is_computed(snapshot):
    """DART 에 4분기 보고서가 없으므로 연간에서 1~3분기를 뺍니다.

    32조 7,259억 - (6조 6,060억 + 10조 4,438억 + 9조 1,833억) = 6조 4,926억
    """
    query = make_query(quarter=4, amount_text="6조 4,900억원")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


def test_fourth_quarter_wrong_value_is_contradicted(snapshot):
    query = make_query(quarter=4, amount_text="20조원")
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED


# ---------------------------------------------------------------------------
# 증감률
# ---------------------------------------------------------------------------


def test_growth_rate_supported(snapshot):
    """2023년 3분기 2조 4,330억 -> 2024년 3분기 9조 1,833억 = +277.4%"""
    query = make_query(
        assertion=FinancialAssertion.GROWTH,
        growth_percent=277,
        direction=Direction.UP,
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.unit == "%"


def test_growth_direction_is_applied(snapshot):
    """실제로는 늘었는데 277% 감소라고 하면 모순입니다."""
    query = make_query(
        assertion=FinancialAssertion.GROWTH,
        growth_percent=277,
        direction=Direction.DOWN,
    )
    assert verify_numeric_claim(query, snapshot).verdict is Verdict.CONTRADICTED


def test_growth_without_previous_year_is_unverifiable(snapshot):
    query = make_query(
        bsns_year=2023,
        assertion=FinancialAssertion.GROWTH,
        growth_percent=10,
        direction=Direction.UP,
    )
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "비교 대상" in result.reason and "2022년 3분기" in result.reason


# ---------------------------------------------------------------------------
# 방향만 말한 주장
# ---------------------------------------------------------------------------


def test_direction_claim_supported(snapshot):
    query = make_query(assertion=FinancialAssertion.DIRECTION, direction=Direction.UP)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


def test_direction_claim_contradicted(snapshot):
    """연결과 별도 모두 늘었으므로 '감소했다'는 모순입니다."""
    query = make_query(assertion=FinancialAssertion.DIRECTION, direction=Direction.DOWN)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED


# ---------------------------------------------------------------------------
# 어림수 주장 (실측으로 찾은 오탐)
#
# 정답을 아는 주장 160건으로 재 보니, 일률 5% 상한이 한 자릿수 금액에서
# 자릿수 규칙을 무력화해 오탐률 58.8% 가 나왔습니다. 상한을 10% 로 올려
# 0% 로 내렸습니다. 다시 느슨해지거나 조여지지 않도록 고정합니다.
# ---------------------------------------------------------------------------


def make_annual_snapshot(amount: int) -> FinancialSnapshot:
    return FinancialSnapshot.from_records([
        record(reprt_code="11011", period_label="연간", quarter=0,
               period_months=12, amount=amount, period_end="2023.12.31",
               bsns_year=2023),
    ])


@pytest.mark.parametrize(
    "amount_text,actual,expected,note",
    [
        # 사람은 6조 5,669억원을 "약 7조원"이라고 말합니다. 인정해야 합니다.
        ("7조원", 6_566_900_000_000, Verdict.SUPPORTED, "차이 6.6%"),
        ("2조원", 2_163_200_000_000, Verdict.SUPPORTED, "차이 7.5%"),
        # 5,753억원을 "1조원"이라고 하면 74% 과장입니다. 잡아야 합니다.
        ("1조원", 575_300_000_000, Verdict.CONTRADICTED, "차이 73.8%"),
        # 정확히 말한 주장은 엄격하게 봅니다.
        ("6조 5,669억원", 6_566_900_000_000, Verdict.SUPPORTED, "정확"),
        ("5조원", 6_566_900_000_000, Verdict.CONTRADICTED, "차이 23.9%"),
    ],
)
def test_rounded_amount_claims(amount_text, actual, expected, note):
    snapshot = make_annual_snapshot(actual)
    query = make_query(bsns_year=2023, quarter=0, amount_text=amount_text)
    result = verify_numeric_claim(query, snapshot)
    assert result.verdict is expected, f"{amount_text} vs {actual} ({note}): {result.reason}"
