"""테스트셋 생성 테스트.

정답 라벨이 틀리면 평가 지표 전체가 무의미해집니다.
생성 규칙이 라벨과 어긋나지 않는지 고정합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.testset import (  # noqa: E402
    TestSetBuilder,
    perturb_amount,
    summarize,
)
from pierrot.tools.financials import FinancialSnapshot  # noqa: E402
from pierrot.tools.price import PriceSnapshot  # noqa: E402


def financial_record(**kwargs) -> dict:
    base = {
        "code": "005930", "corp_code": "00126380", "bsns_year": 2024,
        "reprt_code": "11011", "period_label": "연간", "quarter": 0,
        "period_months": 12, "fs_div": "CFS", "fs_nm": "연결재무제표",
        "account": "영업이익", "amount": 32_725_961_000_000,
        "period_start": "2024.01.01", "period_end": "2024.12.31",
        "currency": "KRW", "rcept_no": "20250311",
    }
    base.update(kwargs)
    return base


PRICE_RECORDS = [
    {"code": "005930", "date": f"2024-01-{day:02d}", "open": 70000 + day,
     "high": 71000 + day, "low": 69000 + day, "close": 70000 + day * 10,
     "volume": 1, "change": 0.01}
    for day in range(2, 32)
]


# ---------------------------------------------------------------------------
# 금액 변형
# ---------------------------------------------------------------------------


def test_perturb_keeps_sign_and_magnitude():
    amount = 32_725_961_000_000
    bigger = perturb_amount(amount, 0.3)
    smaller = perturb_amount(amount, -0.3)
    assert bigger > amount and smaller < amount
    assert 0.25 < (bigger - amount) / amount < 0.35
    assert bigger % 10**10 == 0, "조 단위 금액은 자릿수를 유지해야 합니다"


def test_perturb_small_amount():
    assert perturb_amount(1000, 0.5) == 1500


# ---------------------------------------------------------------------------
# 재무 주장
# ---------------------------------------------------------------------------


@pytest.fixture
def financials() -> FinancialSnapshot:
    return FinancialSnapshot.from_records([
        financial_record(),
        financial_record(bsns_year=2023, amount=6_566_900_000_000),
    ])


def test_numeric_claims_have_all_three_labels(financials):
    claims = TestSetBuilder().numeric_claims(
        financials, years=(2024,), accounts=("영업이익",)
    )
    labels = {claim.label for claim in claims}
    assert labels == {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}


def test_true_claim_contains_the_actual_amount(financials):
    claims = TestSetBuilder().numeric_claims(
        financials, years=(2024,), accounts=("영업이익",)
    )
    true_claim = next(c for c in claims if c.label == "SUPPORTED")
    assert "32조 7,259억원" in true_claim.text
    assert true_claim.error_kind == "none"


def test_unverifiable_claim_uses_year_outside_snapshot(financials):
    claims = TestSetBuilder().numeric_claims(
        financials, years=(2024,), accounts=("영업이익",)
    )
    claim = next(c for c in claims if c.label == "UNVERIFIABLE")
    assert "2019년" in claim.text
    assert claim.error_kind == "out_of_range"


def test_every_claim_records_its_source(financials):
    """판정을 검토하려면 어떤 사실에서 나온 주장인지 알아야 합니다."""
    claims = TestSetBuilder().numeric_claims(
        financials, years=(2024,), accounts=("영업이익",)
    )
    assert all(claim.source_fact for claim in claims)
    assert all(claim.code == "005930" for claim in claims)


# ---------------------------------------------------------------------------
# 증감 방향
# ---------------------------------------------------------------------------


def test_direction_claims_are_paired(financials):
    """방향을 뒤집은 주장은 반드시 거짓입니다."""
    claims = TestSetBuilder().direction_claims(financials, years=(2024,))
    assert len(claims) == 2
    truthy = next(c for c in claims if c.label == "SUPPORTED")
    falsy = next(c for c in claims if c.label == "CONTRADICTED")
    assert "증가" in truthy.text
    assert "감소" in falsy.text
    assert falsy.error_kind == "direction_flip"


def test_direction_needs_previous_year(financials):
    """전년 수치가 없으면 방향 주장을 만들지 않습니다."""
    claims = TestSetBuilder().direction_claims(financials, years=(2023,))
    assert claims == []


# ---------------------------------------------------------------------------
# 주가 주장
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> PriceSnapshot:
    return PriceSnapshot.from_records(PRICE_RECORDS)


def test_price_claims_have_all_three_labels(prices):
    claims = TestSetBuilder().price_claims(
        prices, samples_per_stock=2, start="2024-01-01", end="2024-12-31"
    )
    assert {c.label for c in claims} == {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}
    assert all(c.claim_type == "PRICE" for c in claims)


def test_price_true_claim_matches_snapshot(prices):
    claims = TestSetBuilder().price_claims(
        prices, samples_per_stock=1, start="2024-01-01", end="2024-12-31"
    )
    true_claim = next(c for c in claims if c.label == "SUPPORTED")
    # 주장에 적힌 금액이 근거에 적힌 종가와 같아야 합니다.
    amount = true_claim.text.split("종가 기준 ")[1].split("원")[0]
    assert amount in true_claim.source_fact


# ---------------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_testset(financials, prices):
    """지표를 비교하려면 테스트셋이 고정되어야 합니다."""
    first = TestSetBuilder(seed=42).price_claims(prices, samples_per_stock=3)
    second = TestSetBuilder(seed=42).price_claims(prices, samples_per_stock=3)
    assert [c.text for c in first] == [c.text for c in second]


def test_different_seed_produces_different_testset(prices):
    first = TestSetBuilder(seed=1).price_claims(prices, samples_per_stock=3)
    second = TestSetBuilder(seed=2).price_claims(prices, samples_per_stock=3)
    assert [c.text for c in first] != [c.text for c in second]


def test_claim_ids_are_unique(financials, prices):
    builder = TestSetBuilder()
    claims = (
        builder.numeric_claims(financials, years=(2024,), accounts=("영업이익",))
        + builder.direction_claims(financials, years=(2024,))
        + builder.price_claims(prices, samples_per_stock=2)
    )
    ids = [claim.claim_id for claim in claims]
    assert len(ids) == len(set(ids))


def test_summarize(financials):
    claims = TestSetBuilder().numeric_claims(
        financials, years=(2024,), accounts=("영업이익",)
    )
    stats = summarize(claims)
    assert stats["total"] == len(claims)
    assert sum(stats["by_label"].values()) == len(claims)
