"""가상 샘플 100건이 찾아낸 구조적 결함 셋을 고정합니다.

이 셋은 **수치와 무관한** 결함입니다. 가상 문단의 숫자가 실제 공시와
달라도(중앙값 4.5배) 검증기의 행동은 같아야 하기 때문에, 실제 데이터로도
그대로 시험됩니다.

    1. 문장 안의 절이 흔적 없이 사라짐   (7건)
    2. 모호한 기간을 연도로 바꿔치기     (5건)
    3. 평가 표현을 의미 비교로 판정      (5건)
"""

from __future__ import annotations

import pytest

from pierrot.extractor.decontextualize import invented_period
from pierrot.reporter.report import ClaimReport, find_uncovered
from pierrot.tools.value_judgment import is_value_judgment
from pierrot.verifier.routing import classify
from pierrot.verifier.state import Route


def claim(start: int, end: int) -> ClaimReport:
    return ClaimReport(
        claim_id="c1", text="x", claim_type="NUMERIC", route="NUMERIC",
        verdict="CONTRADICTED", confidence="CALCULATED", reason="", evidence=[],
        start=start, end=end, is_forecast=False, route_changed=False,
    )


# ---------------------------------------------------------------------------
# 1. 문장 안의 절이 사라지던 문제
# ---------------------------------------------------------------------------


def test_문장_뒤쪽_절이_사라지지_않는다():
    """실측 결함을 고정합니다.

    앞 절에 주장이 붙으면 문장이 '다뤄진 것' 으로 보여, 뒤 절의 미래 단정이
    주장 목록에도 uncovered 에도 남지 않았습니다. 100건 중 7건이 그랬습니다.
    """
    answer = (
        "하이닉스의 2025년 연결 영업이익은 2,200억 원이었습니다. "
        "비용 절감 덕분에 전년보다 동일했으며, 2027년에는 역대 최대 이익을 달성할 것입니다."
    )
    found = find_uncovered(answer, [claim(0, 35), claim(36, 56)])
    assert len(found) == 1
    assert "2027년" in found[0].text
    assert "역대 최대" in found[0].text
    assert "전망" in found[0].why


def test_맨_앞_쉼표와_이음말은_다듬는다():
    answer = (
        "하이닉스의 2025년 연결 영업이익은 2,200억 원이었습니다. "
        "비용 절감 덕분에 전년보다 동일했으며, 2027년에는 역대 최대 이익을 달성할 것입니다."
    )
    text = find_uncovered(answer, [claim(0, 35), claim(36, 56)])[0].text
    assert not text.startswith(",")
    assert text.startswith("2027년")


def test_문장_단위_동작은_그대로다():
    """14단계에서 넣은 문장 단위 탐지가 깨지면 안 됩니다."""
    answer = (
        "현대차의 2025년 연결 매출은 전년보다 증가했습니다. "
        "2027년 연결 매출도 2026년보다 늘어날 전망입니다."
    )
    first = "현대차의 2025년 연결 매출은 전년보다 증가했습니다."
    found = find_uncovered(answer, [claim(0, len(first))])
    assert len(found) == 1
    assert "2027년" in found[0].text


def test_전부_덮였으면_비어_있다():
    """가드가 과해서 멀쩡한 문장까지 '안 다뤘다' 고 하면 안 됩니다."""
    answer = "현대차의 2025년 매출은 늘었습니다."
    assert find_uncovered(answer, [claim(0, len(answer))]) == []


def test_부스러기는_보고하지_않는다():
    """한두 글자 토막까지 내놓으면 읽는 사람이 피곤합니다."""
    answer = "매출은 늘었고 이익도 늘었습니다."
    found = find_uncovered(answer, [claim(0, 6), claim(8, len(answer))])
    assert all(len(u.text) >= 8 for u in found)


# ---------------------------------------------------------------------------
# 2. 모호한 기간을 연도로 바꿔치기하던 문제
# ---------------------------------------------------------------------------


def test_원문에_없는_연도를_지어내면_되돌린다():
    """실측 결함을 고정합니다.

    "Kakao의 최근 연결 매출" 이 "Kakao의 2024년 연결 매출" 로 바뀌었습니다.
    '최근' 을 지우고 연도로 바꿔치기해 **흔적조차 남지 않았습니다.**
    그 연도로 조회하면 엉뚱한 해와 대조하고 근거는 CALCULATED 로 찍힙니다.
    """
    answer = "Kakao의 최근 연결 매출은 크게 증가했습니다. 연말에는 영업이익도 예전보다 나아졌습니다."
    assert invented_period(
        "Kakao의 최근 연결 매출은 크게 증가했다",
        "Kakao의 2024년 연결 매출이 크게 증가했다",
        answer,
    ) == "2024년"


@pytest.mark.parametrize(
    "original,rewritten,answer,why",
    [
        ("매출은 증가했다", "삼성전자의 2024년 매출은 증가했다",
         "삼성전자의 2024년 실적입니다. 매출은 증가했습니다.",
         "답변 맥락에 연도가 있으면 끌어온 것이므로 막지 않습니다"),
        ("2024년 매출은 증가했다", "삼성전자의 2024년 매출은 증가했다",
         "삼성전자 2024년 매출은 증가했습니다.",
         "원래 주장에 이미 연도가 있습니다"),
        ("최근 주가가 올랐다", "SK하이닉스의 주가가 올랐다",
         "SK하이닉스의 최근 주가가 올랐습니다.",
         "연도를 넣지 않았으면 문제가 없습니다"),
    ],
)
def test_맥락에서_끌어온_연도는_막지_않는다(original, rewritten, answer, why):
    assert invented_period(original, rewritten, answer) == "", why


# ---------------------------------------------------------------------------
# 3. 평가 표현을 의미 비교로 판정하던 문제
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "삼바는 2025년에 좋은 실적을 냈다",
        "포스코홀딩스는 2025년에 좋은 실적을 냈다",
        "회사는 양호한 성과를 거두었다",
        "전반적으로 부진한 흐름을 보였다",
    ],
)
def test_평가_표현은_의미_비교로_보내지_않는다(text):
    """NLI 로 넘기면 함의 0.998 / 모순 0.999 처럼 양쪽이 동시에 높게 나옵니다.

    어느 쪽으로 갈라져도 근거가 되지 못하는데 0.99 라는 숫자만 남습니다.
    """
    assert is_value_judgment(text)
    route, _ = classify(text, "NARRATIVE")
    assert route is Route.FORECAST


@pytest.mark.parametrize(
    "text,why",
    [
        ("반도체 부문 실적이 개선되었다", "부문이라는 잴 대상이 있습니다"),
        ("영업이익이 좋은 흐름을 보였다", "계정이 있으면 잴 수 있습니다"),
        ("실적이 전년 대비 20% 늘었다", "수치가 있으면 잴 수 있습니다"),
        ("중국 시장의 부진이 큰 영향을 미치고 있다", "인과 서술은 1단계가 뽑으라고 한 것입니다"),
        ("순이익이 좋은 성장세를 보였다", "계정이 있습니다"),
    ],
)
def test_잴_거리가_있으면_막지_않는다(text, why):
    """잘못 막으면 확인할 수 있는 주장을 버리게 됩니다. 막는 쪽보다 나쁩니다."""
    assert not is_value_judgment(text), why
