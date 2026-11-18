"""주가 주장 판정 테스트.

LLM 호출은 하지 않습니다. 파싱 결과를 손으로 만들어 넣습니다.
판정 로직만 따로 고정해 두어야, 나중에 파서를 고쳐도 판정 기준이 흔들리지 않습니다.
"""

from __future__ import annotations

from datetime import date

import pytest

from pierrot.tools.dates import DatePrecision
from pierrot.tools.price import PriceSnapshot
from pierrot.tools.price_claim import (
    Direction,
    ParsedPriceClaim,
    PriceAssertion,
    build_query,
)
from pierrot.tools.price_verify import Verdict, verify_price_claim

# 삼성전자 2024년 4월 가상 시세.
# 4월 11~20일(중순) 구간의 종가는 80,000 ~ 84,000 입니다.
RECORDS = [
    {"code": "005930", "date": "2024-04-01", "open": 70000, "high": 71000, "low": 69000,
     "close": 70000, "volume": 100, "change": 0.0},
    {"code": "005930", "date": "2024-04-09", "open": 71000, "high": 72500, "low": 70500,
     "close": 71800, "volume": 100, "change": 0.025},
    {"code": "005930", "date": "2024-04-11", "open": 72000, "high": 81000, "low": 72000,
     "close": 80000, "volume": 100, "change": 0.114},
    {"code": "005930", "date": "2024-04-15", "open": 80000, "high": 83000, "low": 79000,
     "close": 82000, "volume": 100, "change": 0.025},
    {"code": "005930", "date": "2024-04-19", "open": 82000, "high": 86000, "low": 81000,
     "close": 84000, "volume": 100, "change": 0.024},
    {"code": "005930", "date": "2024-04-30", "open": 84000, "high": 85000, "low": 83000,
     "close": 84500, "volume": 100, "change": 0.006},
]


@pytest.fixture
def snapshot() -> PriceSnapshot:
    return PriceSnapshot.from_records(RECORDS)


def make_query(**kwargs):
    base = dict(
        is_price_claim=True,
        company="삼성전자",
        assertion=PriceAssertion.CLOSE,
        point_date_expr="",
        start_date_expr="",
        end_date_expr="",
        direction=Direction.NONE,
        value_low=0.0,
        value_high=0.0,
        note="",
    )
    base.update(kwargs)
    return build_query(ParsedPriceClaim(**base))


# ---------------------------------------------------------------------------
# 조회 조건 만들기
# ---------------------------------------------------------------------------


def test_query_resolves_stock_and_date():
    query = make_query(point_date_expr="2024년 4월 9일", value_low=71800, value_high=71800)
    assert query.is_resolved
    assert query.stock is not None and query.stock.code == "005930"
    assert query.window is not None and query.window.start == date(2024, 4, 9)
    assert query.precision is DatePrecision.DAY


def test_query_fails_for_unknown_company():
    """유니버스 밖 종목은 조회 조건을 만들 수 없습니다."""
    query = make_query(company="애플", point_date_expr="2024년 4월 9일")
    assert not query.is_resolved
    assert "유니버스에 없는 종목" in query.unresolved


def test_query_fails_without_date():
    """기간이 없으면 조회할 수 없습니다. 날짜를 지어내지 않습니다."""
    query = make_query(point_date_expr="")
    assert not query.is_resolved
    assert "시점을 특정할 수 없습니다" in query.unresolved


def test_query_fails_for_other_assertion():
    query = make_query(assertion=PriceAssertion.OTHER)
    assert not query.is_resolved


def test_query_tenday_expression_becomes_range():
    query = make_query(point_date_expr="2024년 4월 중순")
    assert query.is_resolved
    assert query.window is not None
    assert (query.window.start, query.window.end) == (date(2024, 4, 11), date(2024, 4, 20))
    assert query.precision is DatePrecision.RANGE


# ---------------------------------------------------------------------------
# 판정 - 정확한 날짜
# ---------------------------------------------------------------------------


def test_exact_date_correct_price_is_supported(snapshot):
    query = make_query(point_date_expr="2024년 4월 9일", value_low=71800, value_high=71800)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.is_date_exact


def test_exact_date_wrong_price_is_contradicted(snapshot):
    """실제 71,800원인데 50,000원이라고 하면 허용 오차로도 못 덮습니다."""
    query = make_query(point_date_expr="2024년 4월 9일", value_low=50000, value_high=50000)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED
    assert "겹치지 않습니다" in result.reason


def test_tolerance_accepts_rounded_claim(snapshot):
    """'7만원 수준'처럼 어림수로 말한 주장을 틀렸다고 하면 안 됩니다."""
    query = make_query(point_date_expr="2024년 4월 9일", value_low=70000, value_high=70000)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


def test_tolerance_is_recorded(snapshot):
    """어떤 기준으로 판정했는지 결과에 남아야 합니다."""
    query = make_query(point_date_expr="2024년 4월 9일", value_low=71800, value_high=71800)
    result = verify_price_claim(query, snapshot)
    assert result.tolerance == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# 판정 - 범위 날짜 (A안의 핵심)
# ---------------------------------------------------------------------------


def test_range_date_matches_any_day_in_window(snapshot):
    """4월 중순 종가는 80,000~84,000. 82,000 주장은 구간 안이므로 인정합니다."""
    query = make_query(point_date_expr="2024년 4월 중순", value_low=82000, value_high=82000)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.precision is DatePrecision.RANGE
    assert result.observed_low == 80000 and result.observed_high == 84000


def test_range_date_still_catches_wrong_claim(snapshot):
    """관대한 기준이어도 명백히 틀린 값은 잡아야 합니다.

    이것이 A안을 택한 근거입니다. 범위로 봐도 환각은 걸립니다.
    """
    query = make_query(point_date_expr="2024년 4월 중순", value_low=300000, value_high=300000)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED


def test_range_date_is_marked_as_range(snapshot):
    """지표를 '날짜가 명확한 주장만'으로 나눠 낼 수 있어야 합니다."""
    query = make_query(point_date_expr="2024년 4월 중순", value_low=82000, value_high=82000)
    result = verify_price_claim(query, snapshot)
    assert result.is_date_exact is False


def test_claimed_range_value(snapshot):
    """'8만~8만 5천원' 같은 범위 주장도 처리합니다."""
    query = make_query(point_date_expr="2024년 4월 중순", value_low=80000, value_high=85000)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


# ---------------------------------------------------------------------------
# 판정 - 고점 / 저점
# ---------------------------------------------------------------------------


def test_high_assertion_uses_intraday_high(snapshot):
    """'한때 86,000원까지 상승'은 종가가 아니라 고가로 확인해야 합니다."""
    query = make_query(
        assertion=PriceAssertion.HIGH,
        point_date_expr="2024년 4월 19일",
        direction=Direction.UP,
        value_low=86000,
        value_high=86000,
    )
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


def test_low_assertion_uses_intraday_low(snapshot):
    query = make_query(
        assertion=PriceAssertion.LOW,
        point_date_expr="2024년 4월 11일",
        direction=Direction.DOWN,
        value_low=72000,
        value_high=72000,
    )
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED


# ---------------------------------------------------------------------------
# 판정 - 등락률
# ---------------------------------------------------------------------------


def test_change_rate_supported(snapshot):
    """4월 1일 70,000 -> 4월 30일 84,500 이면 약 +20.7% 입니다."""
    query = make_query(
        assertion=PriceAssertion.CHANGE_RATE,
        start_date_expr="2024년 4월 1일",
        end_date_expr="2024년 4월 30일",
        direction=Direction.UP,
        value_low=20,
        value_high=20,
    )
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.unit == "%"


def test_change_rate_direction_is_applied(snapshot):
    """실제로는 올랐는데 '20% 하락'이라고 하면 모순입니다."""
    query = make_query(
        assertion=PriceAssertion.CHANGE_RATE,
        start_date_expr="2024년 4월 1일",
        end_date_expr="2024년 4월 30일",
        direction=Direction.DOWN,
        value_low=20,
        value_high=20,
    )
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.CONTRADICTED


def test_change_rate_without_period_is_unverifiable(snapshot):
    """기간이 없는 등락률 주장은 검증할 수 없습니다."""
    query = make_query(
        assertion=PriceAssertion.CHANGE_RATE,
        direction=Direction.UP,
        value_low=135,
        value_high=135,
    )
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "기간을 특정할 수 없습니다" in result.reason


# ---------------------------------------------------------------------------
# 판정 불가
# ---------------------------------------------------------------------------


def test_out_of_snapshot_range_is_unverifiable(snapshot):
    """스냅샷에 없는 기간은 모른다고 해야 합니다. 추측하면 안 됩니다."""
    query = make_query(point_date_expr="2019년 4월 9일", value_low=71800, value_high=71800)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "스냅샷 범위" in result.reason


def test_window_without_trading_days_is_unverifiable(snapshot):
    """구간 안에 거래일이 하나도 없으면 판정할 수 없습니다."""
    query = make_query(point_date_expr="2024년 4월 2일", value_low=71800, value_high=71800)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_reason_is_human_readable(snapshot):
    """판정만 있고 근거가 없으면 사람이 신뢰할 수 없습니다."""
    query = make_query(point_date_expr="2024년 4월 9일", value_low=50000, value_high=50000)
    result = verify_price_claim(query, snapshot)
    assert "삼성전자" in result.reason
    assert "71,800원" in result.reason
    assert result.evidence and "2024-04-09" in result.evidence[0]


# ---------------------------------------------------------------------------
# 원주가가 있을 때 (수정주가 종목도 값 주장을 판정할 수 있습니다)
# ---------------------------------------------------------------------------

# 셀트리온처럼 보정된 종목을 흉내 냅니다.
# close 는 보정된 값(호가 단위에 맞지 않음), close_raw 는 당시 실제 호가입니다.
ADJUSTED_RECORDS = [
    {"code": "068270", "date": "2024-04-09", "open": 166604, "high": 167217,
     "low": 158975, "close": 158975, "volume": 1, "change": -0.038,
     "open_raw": 189000, "high_raw": 190000, "low_raw": 180000, "close_raw": 180000},
    {"code": "068270", "date": "2024-04-10", "open": 158975, "high": 162111,
     "low": 157333, "close": 161234, "volume": 1, "change": 0.014,
     "open_raw": 180000, "high_raw": 183500, "low_raw": 178000, "close_raw": 182500},
]


@pytest.fixture
def adjusted_snapshot() -> PriceSnapshot:
    return PriceSnapshot.from_records(ADJUSTED_RECORDS)


def make_celltrion_query(**kwargs):
    base = dict(
        is_price_claim=True,
        company="셀트리온",
        assertion=PriceAssertion.CLOSE,
        point_date_expr="2024년 4월 9일",
        start_date_expr="",
        end_date_expr="",
        direction=Direction.NONE,
        value_low=0.0,
        value_high=0.0,
        note="",
    )
    base.update(kwargs)
    return build_query(ParsedPriceClaim(**base))


def test_snapshot_detects_raw_columns(adjusted_snapshot):
    assert adjusted_snapshot.has_raw("068270") is True


def test_level_claim_uses_raw_price(adjusted_snapshot):
    """당시 실제 호가 180,000원 주장은 인정되어야 합니다.

    보정된 종가(158,975원)와 비교하면 13% 차이로 모순 판정이 나옵니다.
    원주가가 있으면 이 오탐이 사라집니다.
    """
    query = make_celltrion_query(value_low=180000, value_high=180000)
    result = verify_price_claim(query, adjusted_snapshot)
    assert result.verdict is Verdict.SUPPORTED
    assert result.price_basis == "원주가"
    assert result.observed_low == 180000


def test_adjusted_price_claim_is_now_contradicted(adjusted_snapshot):
    """반대로 보정된 값(158,975원)을 주장하면 모순입니다. 당시 호가가 아니기 때문입니다."""
    query = make_celltrion_query(value_low=158975, value_high=158975)
    result = verify_price_claim(query, adjusted_snapshot)
    assert result.verdict is Verdict.CONTRADICTED


def test_without_raw_adjusted_stock_is_skipped():
    """원주가가 없으면 보정된 종목의 값 주장은 판정하지 않습니다."""
    records = [
        {k: v for k, v in row.items() if not k.endswith("_raw")}
        for row in ADJUSTED_RECORDS
    ]
    snapshot = PriceSnapshot.from_records(records)
    query = make_celltrion_query(value_low=180000, value_high=180000)
    result = verify_price_claim(query, snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "보정된 종목" in result.reason


def test_change_rate_still_uses_adjusted(adjusted_snapshot):
    """등락률은 원주가가 있어도 수정주가로 계산합니다.

    분할 전후를 비교 가능하게 만드는 것이 수정주가의 목적이기 때문입니다.
    """
    query = make_celltrion_query(
        assertion=PriceAssertion.CHANGE_RATE,
        point_date_expr="",
        start_date_expr="2024년 4월 9일",
        end_date_expr="2024년 4월 10일",
        direction=Direction.UP,
        value_low=1.4,
        value_high=1.4,
    )
    result = verify_price_claim(query, adjusted_snapshot)
    assert result.price_basis == "수정주가"
    assert result.verdict is Verdict.SUPPORTED
