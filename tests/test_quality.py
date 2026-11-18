"""분해 품질 휴리스틱 테스트.

이 휴리스틱은 정답 판정기가 아니라 "사람이 어디를 먼저 볼지" 정해 주는 장치입니다.
그래서 테스트도 실제로 걸러 내야 할 문장과 걸리면 안 되는 문장을 고정하는 데 집중합니다.
"""

from __future__ import annotations

import pytest

from pierrot.extractor import (
    Claim,
    ClaimType,
    MatchMode,
    has_amount,
    looks_like_forecast,
    quality_report,
    suspect_forecast,
    suspect_type,
)


# 실제 v2/v3 결과에서 잘못 추출된 문장들입니다.
FORECASTS = [
    "NAVER의 AI 기술 경쟁력이 주가에 중요한 영향을 미칠 것으로 예상된다",
    "셀트리온은 코로나19 백신 사업을 축소할 것이다",
    "삼성바이오로직스는 리스크 요인을 지속적으로 주시해야 한다",
    "카카오는 광고 사업 효율화에 집중해야 한다",
    "2024년 현대차의 주당별도배당금은 1,500원 이상이 될 것으로 기대된다",
    "배터리 시장 경쟁 심화가 중요한 변수가 될 것이다",
]

# 과거·현재 사실이므로 걸리면 안 되는 문장들입니다.
FACTS = [
    "삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다",
    "삼성전자 주가는 2024년 10월 9일 5.2% 상승했다",
    "반도체 사업 부문의 실적이 크게 개선되었다",
    "현대차는 배당 성향을 꾸준히 확대해 왔다",
    "셀트리온은 2024년 1분기에 매출 1조 6,785억원을 기록했다",
]


@pytest.mark.parametrize("text", FORECASTS)
def test_forecast_is_detected(text):
    """전망과 권고 표현은 걸러져야 합니다."""
    assert looks_like_forecast(text) is True


@pytest.mark.parametrize("text", FACTS)
def test_fact_is_not_flagged(text):
    """이미 일어난 사실은 걸리면 안 됩니다. 오탐이 많으면 쓸모가 없습니다."""
    assert looks_like_forecast(text) is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("15조 원 이상의 CDMO 수주를 확보했다", True),
        ("3억 유로 규모의 파트너십 계약을 체결했다", True),
        ("배당금 총액을 1,500억원 이상으로 지급했다", True),
        ("주가는 71,800원에 마감했다", True),
        ("반도체 부문의 실적이 개선되었다", False),
        ("영업이익률은 8.5%를 기록했다", False),
    ],
)
def test_amount_detection(text, expected):
    """금액 표현을 찾아냅니다. 비율(%)은 금액이 아닙니다."""
    assert has_amount(text) is expected


def make_claim(claim_id, text, claim_type, mode=MatchMode.EXACT) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=claim_type,
        source_text=text,
        start=0,
        end=len(text),
        span_found=mode is not MatchMode.NOT_FOUND,
        match_mode=mode,
        match_ratio=1.0,
    )


def test_suspect_forecast_collects_ids():
    claims = [
        make_claim("c01", FACTS[0], ClaimType.NUMERIC),
        make_claim("c02", FORECASTS[1], ClaimType.NARRATIVE),
    ]
    assert [c.claim_id for c in suspect_forecast(claims)] == ["c02"]


def test_amount_in_narrative_is_type_suspect():
    """금액이 있는데 NARRATIVE 면 NLI 로 잘못 라우팅됩니다.

    실제로 v2 프롬프트에서 "15조원 수주 확보"가 NARRATIVE 로 분류되어
    공시 조회 대상에서 빠질 뻔했습니다.
    """
    claims = [
        make_claim("c01", "15조 원 이상의 CDMO 수주를 확보했다", ClaimType.NARRATIVE),
        make_claim("c02", "15조 원 이상의 CDMO 수주를 확보했다", ClaimType.NUMERIC),
        make_claim("c03", "반도체 부문 실적이 개선되었다", ClaimType.NARRATIVE),
    ]
    assert [c.claim_id for c in suspect_type(claims)] == ["c01"]


def test_quality_report_counts():
    claims = [
        make_claim("c01", FACTS[0], ClaimType.NUMERIC),
        make_claim("c02", FORECASTS[0], ClaimType.NARRATIVE),
        make_claim("c03", "3억 유로 규모의 계약을 체결했다", ClaimType.NARRATIVE),
        make_claim("c04", FACTS[2], ClaimType.NARRATIVE, mode=MatchMode.FUZZY),
    ]
    report = quality_report(claims)
    assert report["total"] == 4
    assert report["forecast_count"] == 1
    assert report["type_count"] == 1
    assert report["span_review_count"] == 1
