"""부문별 매출 표 파싱과 판정 테스트.

표 HTML 은 실제 DART 사업보고서에서 그대로 가져온 모양입니다.
합성 예시로 테스트하면 실제 표의 변형(병합 셀, 단위가 바깥에 있음,
기간이 아닌 열)을 못 잡습니다. 실제로 그 세 가지 모두에서 틀렸습니다.
"""

from __future__ import annotations

import pytest

from pierrot.tools.segment_verify import (
    OVER_TOTAL_MARGIN,
    Verdict,
    normalize_keyword,
    verify_segment_claim,
)
from pierrot.tools.segments import (
    SegmentSnapshot,
    count_period_columns,
    count_ratio_columns,
    extract_segment_facts,
    parse_number,
)
from pierrot.tools.universe import find_stock

# ---------------------------------------------------------------------------
# 실제 보고서에서 가져온 표 모양
# ---------------------------------------------------------------------------

# 삼성전자 2024: 기간 3개, 비중 없음, 단위는 앞선 1행짜리 표에
SAMSUNG = """
<TABLE><TR><TD>(단위 : 억원)</TD></TR></TABLE>
<TABLE>
<TR><TH>부 문</TH><TH>매출유형</TH><TH>품 목</TH><TH>제56기</TH><TH>제55기</TH><TH>제54기</TH></TR>
<TR><TD>DX 부문</TD><TD>제ㆍ상품</TD><TD>TV, 모니터</TD><TD>1,748,877</TD><TD>1,699,923</TD><TD>1,824,897</TD></TR>
<TR><TD>DS 부문</TD><TD>제ㆍ상품</TD><TD>DRAM 등</TD><TD>1,110,660</TD><TD>665,945</TD><TD>984,553</TD></TR>
<TR><TD>기 타</TD><TD>부문간 내부거래 제거 등</TD><TD>△ 285,155</TD><TD>△230,152</TD><TD>△263,099</TD></TR>
<TR><TD>합 계</TD><TD>3,008,709</TD><TD>2,589,355</TD><TD>3,022,314</TD></TR>
</TABLE>
"""

# NAVER 2024: 기간 2개(당기/전기), 비중 열 있음
NAVER = """
<TABLE><TR><TD>(누적)</TD><TD>(단위: 백만원, %)</TD></TR></TABLE>
<TABLE>
<TR><TH>구분</TH><TH>연결</TH></TR>
<TR><TH>당기</TH><TH>전기</TH></TR>
<TR><TH>금액</TH><TH>비중</TH><TH>금액</TH><TH>비중</TH></TR>
<TR><TD>영업수익</TD><TD>10,737,719</TD><TD>100.0</TD><TD>9,670,644</TD><TD>100.0</TD></TR>
<TR><TD>- 서치플랫폼</TD><TD>3,946,166</TD><TD>36.8</TD><TD>3,589,061</TD><TD>37.1</TD></TR>
<TR><TD>- 클라우드</TD><TD>563,748</TD><TD>5.3</TD><TD>447,184</TD><TD>4.6</TD></TR>
</TABLE>
"""

# 기아 2024: 열이 기간이 아니라 지역. 읽으면 안 되는 표입니다.
KIA_REGION = """
<TABLE><TR><TD>(단위:백만원)</TD></TR></TABLE>
<TABLE>
<TR><TH>구 분</TH><TH>국내</TH><TH>북중미</TH><TH>유럽</TH><TH>기타</TH><TH>연결조정</TH><TH>연결후금액</TH></TR>
<TR><TD>총매출액</TD><TD>63,256,745</TD><TD>39,408,225</TD><TD>26,684,947</TD><TD>49,256,915</TD><TD>8,463,201</TD><TD>107,448,752</TD></TR>
</TABLE>
"""

# 현대차 2024: 단위가 표 바깥 문단에 있음
HYUNDAI = """
<P>(단위 : 백만원)</P>
<TABLE>
<TR><TH>매출유형</TH><TH>품 목</TH><TH>2024년(제57기)</TH><TH>2023년(제56기)</TH><TH>2022년(제55기)</TH></TR>
<TR><TD>제품</TD><TD>승용</TD><TD>9,667,086</TD><TD>12,000,528</TD><TD>11,406,316</TD></TR>
</TABLE>
"""


def facts_of(html: str, *, code: str = "005930", year: int = 2024):
    return extract_segment_facts(html, code=code, bsns_year=year, rcept_no="X")


# ---------------------------------------------------------------------------
# 숫자 읽기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,748,877", 1748877.0),
        ("△ 285,155", -285155.0),      # DART 는 음수를 △ 로 적습니다
        ("(230,152)", -230152.0),      # 괄호도 음수입니다
        ("63.7%", 63.7),
        ("-", None),
        ("", None),
    ],
)
def test_표_한_칸을_숫자로_읽는다(text, expected):
    assert parse_number(text) == expected


# ---------------------------------------------------------------------------
# 기간 열 인식 — 이게 틀리면 표를 통째로 오독합니다
# ---------------------------------------------------------------------------


def test_기간_머리글이_없는_표는_읽지_않는다():
    """기아 지역별 표. 숫자 칸 7개가 기간이 아니라 지역입니다.

    이 가드가 없으면 국내 매출 63조를 '2024년 매출', 북중미 39조를
    '2023년 매출' 로 기록합니다. 실제로 그렇게 기록됐다가 합계 검수에서
    93% 차이로 걸렸습니다.
    """
    assert facts_of(KIA_REGION, code="000270") == []


def test_기간_수는_머리글에서_읽는다():
    samsung_rows = [["부 문", "매출유형", "품 목", "제56기", "제55기", "제54기"]]
    assert count_period_columns(samsung_rows) == 3

    naver_rows = [["당기", "전기"]]
    assert count_period_columns(naver_rows) == 2

    region_rows = [["구 분", "국내", "북중미", "유럽", "연결조정", "연결후금액"]]
    assert count_period_columns(region_rows) == 0


def test_비중_열_수도_머리글에서_읽는다():
    assert count_ratio_columns([["금액", "비중", "금액", "비중"]]) == 2
    assert count_ratio_columns([["부 문", "매출유형", "품 목"]]) == 0


# ---------------------------------------------------------------------------
# 표 읽기
# ---------------------------------------------------------------------------


def test_기간_3개_표를_연도별로_편다():
    facts = facts_of(SAMSUNG)
    dx = [f for f in facts if f.name.startswith("DX")]
    assert {f.year for f in dx} == {2024, 2023, 2022}
    current = next(f for f in dx if f.year == 2024)
    assert current.amount == pytest.approx(1_748_877 * 1e8)


def test_음수_표기를_읽는다():
    facts = facts_of(SAMSUNG)
    other = next(f for f in facts if f.name.startswith("기 타") and f.year == 2024)
    assert other.amount == pytest.approx(-285_155 * 1e8)


def test_합계_행을_표시한다():
    facts = facts_of(SAMSUNG)
    total = next(f for f in facts if f.name == "합 계" and f.year == 2024)
    assert total.is_total


def test_기간_2개에_비중이_있는_표도_읽는다():
    """NAVER. 기간 수를 3으로 고정했다가 이 표를 통째로 놓쳤습니다."""
    facts = facts_of(NAVER, code="035420")
    search = next(f for f in facts if f.name == "서치플랫폼" and f.year == 2024)
    assert search.amount == pytest.approx(3_946_166 * 1e6)
    assert search.ratio == pytest.approx(36.8)


def test_비중_100퍼센트_행은_합계로_본다():
    """NAVER 는 합계 행 이름이 '영업수익' 이라 이름만으로는 못 가립니다."""
    facts = facts_of(NAVER, code="035420")
    revenue = next(f for f in facts if f.name == "영업수익" and f.year == 2024)
    assert revenue.is_total


def test_글머리표를_뗀다():
    facts = facts_of(NAVER, code="035420")
    assert any(f.name == "클라우드" for f in facts)


def test_단위가_표_바깥에_있어도_읽는다():
    """현대차 2023·2024. 이걸 못 읽어 그 회사 수치를 한 건도 못 뽑았습니다."""
    facts = facts_of(HYUNDAI, code="005380")
    passenger = next(f for f in facts if "승용" in f.name and f.year == 2024)
    assert passenger.amount == pytest.approx(9_667_086 * 1e6)


# ---------------------------------------------------------------------------
# 부문 이름 다듬기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "account,expected",
    [
        ("HBM 매출", "HBM"),
        ("광고 사업 부문 매출", "광고"),
        ("철강 부문 영업이익", "철강"),
        ("백신 사업 매출", "백신"),
        ("매출액", ""),
    ],
)
def test_계정_표기에서_부문_이름을_뽑는다(account, expected):
    assert normalize_keyword(account) == expected


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot():
    return SegmentSnapshot.from_facts(
        facts_of(SAMSUNG) + facts_of(NAVER, code="035420")
    )


def test_이름이_맞는_부문과_금액이_같으면_인정한다(snapshot):
    result = verify_segment_claim(
        stock=find_stock("NAVER"),
        year=2024,
        account="서치플랫폼 매출",
        claimed=3_946_166 * 1e6,
        snapshot=snapshot,
    )
    assert result.verdict is Verdict.SUPPORTED
    assert result.matched_name == "서치플랫폼"


def test_이름이_맞는데_금액이_다르면_모순이다(snapshot):
    result = verify_segment_claim(
        stock=find_stock("NAVER"),
        year=2024,
        account="서치플랫폼 매출",
        claimed=1_000_000 * 1e6,
        snapshot=snapshot,
    )
    assert result.verdict is Verdict.CONTRADICTED


def test_없는_부문이라도_전사_매출을_넘으면_모순이다(snapshot):
    """'NAVER 광고 부문 매출 5조 7,839억' 같은 주장입니다.

    NAVER 에 '광고' 부문은 없지만, 전사 매출(10조 7,377억)의 1.5배를 넘으면
    어느 부문이든 성립할 수 없습니다.
    """
    result = verify_segment_claim(
        stock=find_stock("NAVER"),
        year=2024,
        account="광고 부문 매출",
        claimed=20_000_000 * 1e6,
        snapshot=snapshot,
    )
    assert result.verdict is Verdict.CONTRADICTED
    assert result.used_upper_bound


def test_없는_부문이고_상한도_안_넘으면_확인_불가다(snapshot):
    """여기서 모순이라고 하면 오탐입니다. 공시 표에 없다고 사실이 아닌 건 아닙니다."""
    result = verify_segment_claim(
        stock=find_stock("NAVER"),
        year=2024,
        account="광고 부문 매출",
        claimed=1_509 * 1e8,
        snapshot=snapshot,
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "서치플랫폼" in result.reason  # 있는 부문을 사유에 적습니다


def test_상한_여유를_지킨다(snapshot):
    """전사 매출을 조금 넘는 정도로는 모순이라고 하지 않습니다.

    공시 표의 합계가 내부거래 제거 전 단순합계인 경우가 있어
    (한국전력은 재무제표 매출의 152%) 여유가 없으면 오탐이 납니다.
    """
    total = 10_737_719 * 1e6
    result = verify_segment_claim(
        stock=find_stock("NAVER"),
        year=2024,
        account="광고 부문 매출",
        claimed=total * (OVER_TOTAL_MARGIN - 0.1),
        snapshot=snapshot,
    )
    assert result.verdict is Verdict.UNVERIFIABLE
