"""DART 응답 파싱과 재무 스냅샷 조회 테스트.

API 호출은 하지 않습니다. 실제 응답에서 확인한 값을 그대로 가져다 썼습니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.dart import (
    ACCOUNT_ALIASES,
    REPORT_CODES,
    normalize_account,
    parse_amount,
)
from pierrot.tools.financials import FinancialSnapshot, format_won


# ---------------------------------------------------------------------------
# 금액 파싱
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("32,725,961,000,000", 32_725_961_000_000),
        ("-11,526,297,000,000", -11_526_297_000_000),
        ("1000", 1000),
        ("", None),
        ("-", None),
        (None, None),
        ("알수없음", None),
    ],
)
def test_parse_amount(raw, expected):
    """DART 금액에는 쉼표가 들어 있고 빈 값도 옵니다."""
    assert parse_amount(raw) == expected


# ---------------------------------------------------------------------------
# 계정명 정규화
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("매출액", "매출액"),
        ("수익(매출액)", "매출액"),
        ("영업수익", "매출액"),
        ("영업이익", "영업이익"),
        ("당기순이익(손실)", "당기순이익"),
        ("자산총계", "자산총계"),
    ],
)
def test_normalize_account(raw, expected):
    """회사마다 표기가 달라 표준 이름으로 맞춥니다."""
    assert normalize_account(raw) == expected


def test_unknown_account_returns_none():
    """모르는 계정은 조용히 넘깁니다. 억지로 매핑하면 엉뚱한 값을 비교하게 됩니다."""
    assert normalize_account("기타포괄손익누계액") is None
    assert normalize_account("") is None


def test_account_aliases_map_to_known_targets():
    """별칭의 대상이 모두 표준 이름 집합 안에 있어야 합니다."""
    targets = set(ACCOUNT_ALIASES.values())
    for target in targets:
        assert target in targets


# ---------------------------------------------------------------------------
# 보고서 코드
# ---------------------------------------------------------------------------


def test_report_codes_cover_four_periods():
    assert set(REPORT_CODES) == {"11011", "11012", "11013", "11014"}


def test_quarterly_reports_are_three_months():
    """분기보고서 금액은 당분기 3개월치입니다.

    DART 는 thstrm_dt 를 누적 기간으로 적지만 금액은 당분기입니다.
    실측으로 확인했습니다(삼성전자 2024: 1분기 6.61조 + 2분기 10.44조 +
    3분기 9.18조 = 26.23조, 연간 32.73조 -> 4분기 6.49조로 맞아떨어짐).
    """
    for code, meta in REPORT_CODES.items():
        if code == "11011":
            assert meta["months"] == 12 and meta["quarter"] == 0
        else:
            assert meta["months"] == 3 and meta["quarter"] in (1, 2, 3)


# ---------------------------------------------------------------------------
# 금액 표기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount,expected",
    [
        (9_180_000_000_000, "9조 1,800억원"),
        (32_725_961_000_000, "32조 7,259억원"),
        (300_000_000_000, "3,000억원"),
        (1_000_000_000_000, "1조원"),
        (-11_526_297_000_000, "-11조 5,262억원"),
        (5_000_000, "500만원"),
    ],
)
def test_format_won(amount, expected):
    """LLM 이 '9조 1,800억원'처럼 말하므로 판정 근거도 같은 단위로 보여 줍니다."""
    assert format_won(amount) == expected


# ---------------------------------------------------------------------------
# 재무 스냅샷 조회
# ---------------------------------------------------------------------------


def make_record(**kwargs) -> dict:
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
        "amount": 9_183_371_000_000,
        "period_start": "2024.01.01",
        "period_end": "2024.09.30",
        "currency": "KRW",
        "rcept_no": "20241114000000",
    }
    base.update(kwargs)
    return base


# 삼성전자 2024년 실제 수치입니다.
RECORDS = [
    make_record(),  # 3분기 연결 영업이익 9.18조
    make_record(fs_div="OFS", fs_nm="재무제표", amount=3_900_000_000_000),
    make_record(reprt_code="11011", period_label="연간", quarter=0, period_months=12,
                amount=32_725_961_000_000, period_end="2024.12.31"),
    make_record(reprt_code="11011", period_label="연간", quarter=0, period_months=12,
                fs_div="OFS", fs_nm="재무제표", amount=12_361_034_000_000,
                period_end="2024.12.31"),
    make_record(account="매출액", amount=79_100_000_000_000),
]


@pytest.fixture
def snapshot() -> FinancialSnapshot:
    return FinancialSnapshot.from_records(RECORDS)


def test_snapshot_basic_info(snapshot):
    assert len(snapshot) == len(RECORDS)
    assert snapshot.codes() == ["005930"]
    assert set(snapshot.accounts()) == {"영업이익", "매출액"}
    assert snapshot.year_range() == (2024, 2024)


def test_lookup_returns_both_fs_divisions(snapshot):
    """연결과 별도를 모두 돌려줍니다.

    LLM 답변이 어느 쪽인지 밝히지 않으므로, 하나만 골라 두면 맞는 주장을
    틀렸다고 판정하게 됩니다.
    """
    facts = snapshot.lookup("005930", 2024, "영업이익", quarter=3)
    assert {f.fs_div for f in facts} == {"CFS", "OFS"}


def test_lookup_filters_by_quarter(snapshot):
    """분기와 연간은 다른 값입니다. 섞이면 안 됩니다."""
    quarterly = snapshot.lookup("005930", 2024, "영업이익", quarter=3, fs_div="CFS")
    annual = snapshot.lookup("005930", 2024, "영업이익", quarter=0, fs_div="CFS")
    assert quarterly[0].amount == 9_183_371_000_000
    assert annual[0].amount == 32_725_961_000_000


def test_lookup_unknown_returns_empty(snapshot):
    assert snapshot.lookup("005930", 2019, "영업이익") == []
    assert snapshot.lookup("000660", 2024, "영업이익") == []


def test_has_data(snapshot):
    assert snapshot.has_data("005930", 2024) is True
    assert snapshot.has_data("005930", 2019) is False


def test_fact_describe(snapshot):
    """판정 근거로 쓰려면 사람이 읽을 수 있어야 합니다."""
    fact = snapshot.lookup("005930", 2024, "영업이익", quarter=3, fs_div="CFS")[0]
    text = fact.describe()
    assert "삼성전자" in text
    assert "9조 1,833억원" in text
    assert "연결" in text


def test_missing_column_is_rejected():
    import pandas as pd

    with pytest.raises(ValueError):
        FinancialSnapshot(pd.DataFrame([{"code": "005930"}]))
