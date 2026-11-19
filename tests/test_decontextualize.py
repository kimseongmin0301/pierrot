"""탈맥락화 2단계 테스트.

API 를 호출하지 않습니다. 파이썬이 맡은 부분(무엇이 빠졌는지 판정, 개수
안전장치)만 봅니다. 문장을 잘 다시 쓰는지는 LLM 의 몫이고 측정으로 봅니다
(`scripts/check_decontextualization.py`).
"""

from __future__ import annotations

from datetime import date

import pytest

from pierrot.extractor.decontextualize import (
    RewrittenClaim,
    RewrittenClaimList,
    decontextualize,
    invented_company,
    missing_parts,
)
from pierrot.extractor.models import Claim, ClaimType
from pierrot.tools.dates import DatePrecision, DateRange
from pierrot.tools.price import PriceSnapshot
from pierrot.tools.price_claim import (
    Direction,
    ParsedPriceClaim,
    PriceAssertion,
    PriceQuery,
)
from pierrot.tools.price_verify import (
    MAX_LEVEL_SPREAD_RATIO,
    MAX_RATE_SPREAD_PP,
    Verdict,
    verify_price_claim,
)
from pierrot.tools.universe import find_stock

NAMES = ("삼성전자", "SK하이닉스", "카카오", "POSCO홀딩스")


def claim(text: str, claim_id: str = "c1") -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=ClaimType.NARRATIVE,
        source_text=text,
        span_found=True,
        start=0,
        end=len(text),
    )


# ---------------------------------------------------------------------------
# 무엇이 빠졌는지 판정 — 파이썬이 합니다
# ---------------------------------------------------------------------------


def test_회사가_없으면_짚어_준다():
    assert "회사 없음" in missing_parts(claim("HBM4 개발이 성공했다"), NAMES)


def test_연도가_없으면_짚어_준다():
    assert "연도 없음" in missing_parts(claim("삼성전자의 영업이익이 늘었다"), NAMES)


def test_둘_다_빠질_수_있다():
    parts = missing_parts(claim("HBM4 개발이 성공했다"), NAMES)
    assert parts == ["회사 없음", "연도 없음"]


def test_기준일_없는_상대_표현을_짚어_준다():
    parts = missing_parts(
        claim("삼성전자 주가는 2024년 1월 8일부터 현재까지 올랐다"), NAMES
    )
    assert "기준일 없는 상대 표현" in parts


def test_등락률_주장에_한쪽만_있으면_짚어_준다():
    parts = missing_parts(
        claim("POSCO홀딩스 주가는 2024년 4월 중순에 20% 이상 급등했다"), NAMES
    )
    assert "구간 한쪽만" in parts


def test_다_갖춘_주장은_손대지_않는다():
    assert missing_parts(
        claim("삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다"), NAMES
    ) == []


# ---------------------------------------------------------------------------
# 안전장치 — 주장을 잃지 않아야 합니다
# ---------------------------------------------------------------------------


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.result


def test_손댈_것이_없으면_호출하지_않는다():
    claims = [claim("삼성전자의 2024년 3분기 영업이익은 9조원이다")]
    model = FakeModel(None)
    result, problem = decontextualize("답변", claims, model=model, names=NAMES)
    assert model.calls == 0
    assert problem == ""
    assert result == claims


def test_개수가_줄어들면_통째로_버린다():
    """LLM 이 주장을 지워 버리는 것을 **구조로** 막습니다.

    프롬프트로 "지우지 마십시오"라고 부탁하는 것만으로는 부족합니다.
    누락은 되돌릴 수 없고, 어느 주장이 사라졌는지 맞춰 볼 방법도 없습니다.
    """
    claims = [claim("HBM4 개발이 성공했다", "c1"), claim("D램 매출이 줄었다", "c2")]
    model = FakeModel(
        RewrittenClaimList(
            claims=[RewrittenClaim(index=1, text="SK하이닉스는 2024년에 HBM4 개발에 성공했다", filled_from="답변")]
        )
    )
    result, problem = decontextualize("답변", claims, model=model, names=NAMES)
    assert "개수가 다릅니다" in problem
    assert result == claims  # 원래 주장 그대로


def test_번호가_어긋나면_통째로_버린다():
    claims = [claim("HBM4 개발이 성공했다", "c1"), claim("D램 매출이 줄었다", "c2")]
    model = FakeModel(
        RewrittenClaimList(
            claims=[
                RewrittenClaim(index=1, text="A", filled_from=""),
                RewrittenClaim(index=5, text="B", filled_from=""),
            ]
        )
    )
    result, problem = decontextualize("답변", claims, model=model, names=NAMES)
    assert "번호가" in problem
    assert result == claims


ANSWER_WITH_COMPANY = (
    "SK하이닉스의 2024년 영업이익은 23조 4,673억 원입니다. "
    "HBM4 개발 성공을 통해 고부가 제품 비중을 확대했습니다."
)


def test_손댈_주장만_보내고_나머지는_그대로_둔다():
    """다 갖춘 주장은 LLM 에 보내지 않습니다. 보내면 멀쩡한 것을 망칠 수 있습니다."""
    good = claim("삼성전자의 2024년 3분기 영업이익은 9조원이다", "c1")
    bad = claim("HBM4 개발이 성공했다", "c2")
    model = FakeModel(
        RewrittenClaimList(
            claims=[
                RewrittenClaim(
                    index=1, text="SK하이닉스는 2024년에 HBM4 개발에 성공했다", filled_from="답변 앞부분"
                )
            ]
        )
    )
    result, note = decontextualize(
        ANSWER_WITH_COMPANY, [good, bad], model=model, names=NAMES
    )
    assert note == ""
    assert result[0].text == good.text          # 손대지 않음
    assert result[1].text.startswith("SK하이닉스")  # 다시 쓰임
    assert result[1].claim_id == "c2"           # 짝이 어긋나지 않음


# ---------------------------------------------------------------------------
# 없는 회사를 지어내는 것을 막습니다
# ---------------------------------------------------------------------------


def test_원문에_없는_회사를_넣으면_되돌린다():
    """실측에서 나온 오탐을 고정합니다.

    회사 이름이 한 번도 안 나오는 글을 넣었는데 다듬기 단계가
    "SK하이닉스" 를 지어냈고, 그 회사의 수치로 조회해 모순 판정까지
    났습니다. 프롬프트에 "지어내지 마십시오" 라고 적어 두었지만
    지켜지지 않았습니다. 그래서 코드로 검사합니다.
    """
    answer = "펀더멘탈 분석 결과, 2023년부터 2025년까지 실적 성장세가 이어지고 있습니다."
    target = claim("2023년부터 2025년까지 실적 성장세가 이어지고 있다", "c1")
    model = FakeModel(
        RewrittenClaimList(
            claims=[
                RewrittenClaim(
                    index=1,
                    text="SK하이닉스의 2023년부터 2025년까지 실적 성장세가 이어지고 있다",
                    filled_from="지어냄",
                )
            ]
        )
    )
    result, note = decontextualize(answer, [target], model=model, names=NAMES)
    assert result[0].text == target.text        # 원래대로 되돌림
    assert "SK하이닉스" in note


def test_원문에_있는_회사는_그대로_넣는다():
    """가드가 과하면 정상 동작까지 막습니다. 반대 방향 고정입니다."""
    target = claim("HBM4 개발이 성공했다", "c1")
    model = FakeModel(
        RewrittenClaimList(
            claims=[
                RewrittenClaim(
                    index=1, text="SK하이닉스는 2024년에 HBM4 개발에 성공했다", filled_from="답변"
                )
            ]
        )
    )
    result, note = decontextualize(
        ANSWER_WITH_COMPANY, [target], model=model, names=NAMES
    )
    assert result[0].text.startswith("SK하이닉스")
    assert note == ""


def test_한_문장이_되돌려져도_나머지는_살린다():
    """한 문장 때문에 멀쩡하게 다듬어진 나머지를 잃으면 안 됩니다."""
    a = claim("HBM4 개발이 성공했다", "c1")
    b = claim("영업이익이 늘었다", "c2")
    model = FakeModel(
        RewrittenClaimList(
            claims=[
                RewrittenClaim(index=1, text="SK하이닉스는 2024년에 HBM4 개발에 성공했다", filled_from="답변"),
                RewrittenClaim(index=2, text="카카오의 2024년 영업이익이 늘었다", filled_from="지어냄"),
            ]
        )
    )
    result, note = decontextualize(ANSWER_WITH_COMPANY, [a, b], model=model, names=NAMES)
    assert result[0].text.startswith("SK하이닉스")  # 살아남음
    assert result[1].text == b.text                 # 되돌려짐
    assert "카카오" in note


@pytest.mark.parametrize(
    "original,rewritten,answer,expected",
    [
        ("실적이 늘었다", "SK하이닉스의 실적이 늘었다", "분석 결과입니다", "SK하이닉스"),
        ("실적이 늘었다", "SK하이닉스의 실적이 늘었다", "SK하이닉스 분석", ""),
        ("SK하이닉스 실적", "SK하이닉스의 실적이 늘었다", "분석 결과입니다", ""),
        ("실적이 늘었다", "2024년 실적이 늘었다", "분석 결과입니다", ""),
    ],
)
def test_지어낸_회사만_골라낸다(original, rewritten, answer, expected):
    assert invented_company(original, rewritten, answer, NAMES) == expected


def test_호출이_실패해도_주장을_잃지_않는다():
    class Broken:
        def invoke(self, messages):
            raise RuntimeError("API 실패")

    claims = [claim("HBM4 개발이 성공했다")]
    result, problem = decontextualize("답변", claims, model=Broken(), names=NAMES)
    assert "실패" in problem
    assert result == claims


# ---------------------------------------------------------------------------
# 다듬기가 만들어 낸 오탐을 막는 가드
# ---------------------------------------------------------------------------


def _rate_query(start: DateRange, end: DateRange, rate: float) -> PriceQuery:
    parsed = ParsedPriceClaim(
        is_price_claim=True,
        company="삼성전자",
        assertion=PriceAssertion.CHANGE_RATE,
        point_date_expr="",
        start_date_expr=start.raw,
        end_date_expr=end.raw,
        direction=Direction.DOWN,
        value_low=rate,
        value_high=rate,
        note="",
    )
    window = DateRange(start.start, end.end, DatePrecision.RANGE, "")
    return PriceQuery(
        parsed=parsed,
        stock=find_stock("삼성전자"),
        window=window,
        start_window=start,
        end_window=end,
        unresolved="",
    )


@pytest.fixture
def wide_snapshot():
    """한 해 동안 크게 움직인 종목. 연 단위 구간이면 등락률 폭이 벌어집니다."""
    records = []
    for index, (day, close) in enumerate(
        [
            (date(2023, 1, 2), 600_000.0),
            (date(2023, 12, 28), 380_000.0),
            (date(2024, 1, 2), 430_000.0),
            (date(2024, 12, 27), 320_000.0),
        ]
    ):
        records.append(
            {
                "code": "005930",
                "date": day,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
                "change": 0.0,
            }
        )
    return PriceSnapshot.from_records(records)


def test_기간이_너무_넓으면_판정하지_않는다(wide_snapshot):
    """실측 오탐을 고정합니다.

    "LG에너지솔루션의 주가는 2024년 전년 동기간 동안 약 13% 하락했다" 가
    2023년 전체 -> 2024년 전체로 해석되어 실제 등락률 -47.6% ~ +15.6% 가
    나왔고, 주장 -13% 가 그 안에 들어가 SUPPORTED 로 통과했습니다.
    그 정도로 넓은 구간은 어떤 주장이든 통과시킵니다.
    """
    start = DateRange(date(2023, 1, 1), date(2023, 12, 31), DatePrecision.RANGE, "2023년")
    end = DateRange(date(2024, 1, 1), date(2024, 12, 31), DatePrecision.RANGE, "2024년")
    result = verify_price_claim(_rate_query(start, end, 13.0), wide_snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "너무 넓어" in result.reason
    assert result.observed_high - result.observed_low > MAX_RATE_SPREAD_PP


def test_좁은_구간은_그대로_판정한다(wide_snapshot):
    """가드가 과하면 정상 주장까지 막습니다. 반대 방향 고정입니다."""
    start = DateRange(date(2024, 1, 1), date(2024, 1, 5), DatePrecision.RANGE, "2024년 1월 초")
    end = DateRange(date(2024, 12, 20), date(2024, 12, 31), DatePrecision.RANGE, "2024년 12월 말")
    result = verify_price_claim(_rate_query(start, end, 25.6), wide_snapshot)
    assert result.verdict is Verdict.SUPPORTED


# ---------------------------------------------------------------------------
# 값 주장도 구간이 넓으면 판정하지 않습니다
# ---------------------------------------------------------------------------


def _level_query(window: DateRange, value: float) -> PriceQuery:
    parsed = ParsedPriceClaim(
        is_price_claim=True,
        company="삼성전자",
        assertion=PriceAssertion.LOW,
        point_date_expr=window.raw,
        start_date_expr="",
        end_date_expr="",
        direction=Direction.NONE,
        value_low=value,
        value_high=value,
        note="",
    )
    return PriceQuery(
        parsed=parsed,
        stock=find_stock("삼성전자"),
        window=window,
        start_window=window,
        end_window=window,
        unresolved="",
    )


@pytest.fixture
def quarter_snapshot():
    """한 분기 동안 크게 움직인 종목. 구간이 길면 폭이 벌어집니다."""
    # 값은 모두 호가 단위(5만~10만원 구간은 100원)에 맞춥니다.
    # 어긋나면 "수정주가로 보정된 종목" 가드가 먼저 걸려 이 테스트가
    # 보려는 것을 가립니다.
    records = []
    for day, low, high, close in [
        (date(2024, 7, 1), 86_000.0, 88_800.0, 87_400.0),
        (date(2024, 8, 5), 70_000.0, 74_000.0, 72_000.0),
        (date(2024, 9, 30), 61_500.0, 63_000.0, 62_200.0),
    ]:
        records.append(
            {
                "code": "005930",
                "date": day,
                "open": high,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
                "change": 0.0,
            }
        )
    return PriceSnapshot.from_records(records)


def test_값_주장도_기간이_넓으면_판정하지_않는다(quarter_snapshot):
    """API 연기 테스트에서 나온 오탐을 고정합니다.

    "삼성전자 주가는 2024년 3분기 실적 발표 직후 71,800원까지 하락했다" 가
    92일 구간으로 해석되어 저가 61,500 ~ 88,800원(44% 폭)이 나왔고,
    주장 71,800원이 그 안에 들어가 SUPPORTED 로 통과했습니다.
    그 정도로 넓은 구간은 그 사이 어떤 값이든 통과시킵니다.
    """
    window = DateRange(
        date(2024, 7, 1), date(2024, 9, 30), DatePrecision.RANGE, "2024년 3분기"
    )
    result = verify_price_claim(_level_query(window, 71_800.0), quarter_snapshot)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "너무 넓어" in result.reason
    assert result.observed_high / result.observed_low - 1 > MAX_LEVEL_SPREAD_RATIO


def test_좁은_구간의_값_주장은_그대로_판정한다(quarter_snapshot):
    """가드가 과하면 정상 주장까지 막습니다. 반대 방향 고정입니다."""
    window = DateRange(
        date(2024, 9, 29), date(2024, 9, 30), DatePrecision.RANGE, "2024년 9월 말"
    )
    assert verify_price_claim(
        _level_query(window, 62_000.0), quarter_snapshot
    ).verdict is Verdict.SUPPORTED
    assert verify_price_claim(
        _level_query(window, 90_000.0), quarter_snapshot
    ).verdict is Verdict.CONTRADICTED
