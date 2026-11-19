"""수치를 말하지 않은 주장이 모순으로 판정되지 않도록 고정합니다.

배경: 파서는 주장에 숫자가 없으면 값 필드를 0 으로 채웁니다. "없음"을 나타내는
필드가 따로 없기 때문입니다. 판정기가 그 0 을 그대로 비교하면 주장 구간이
[0, 0] 이 되어 실제 값과 절대 겹치지 않고, **숫자를 말하지도 않은 주장이
모순으로 판정됩니다.**

실제로 3단계 에이전트가 라우팅을 바꾸면서 드러났습니다.

    "셀트리온 주가는 2024년 1월 8일 장마감 기준으로 전장 최저점을 찍은 후
     회복세를 보였다"
    -> LOW 주장 / 주장값 0원 / 실제 저가 204,500원 -> 모순 (오탐)

이 주장은 그 전까지 NARRATIVE 로 분류돼 주가 도구에 들어온 적이 없었습니다.
에이전트가 PRICE 로 돌리면서 가려져 있던 버그가 나왔습니다.
"""

from __future__ import annotations

from datetime import date

from pierrot.tools.dates import DatePrecision, DateRange
from pierrot.tools.numeric_claim import (
    Direction as NumDirection,
    FinancialAssertion,
    NumericQuery,
    ParsedNumericClaim,
)
from pierrot.tools.numeric_verify import Verdict as NumVerdict, verify_numeric_claim
from pierrot.tools.price import PriceSnapshot
from pierrot.tools.price_claim import (
    Direction as PriceDirection,
    ParsedPriceClaim,
    PriceAssertion,
    PriceQuery,
)
from pierrot.tools.price_verify import Verdict as PriceVerdict, verify_price_claim
from pierrot.tools.universe import find_stock


def _price_snapshot() -> PriceSnapshot:
    return PriceSnapshot.from_records(
        [
            {
                "code": "068270",
                "date": date(2024, 1, 8),
                "open": 213000.0,
                "high": 216000.0,
                "low": 204500.0,
                "close": 206000.0,
                "volume": 100000.0,
                "change": -3.96,
            }
        ]
    )


def _price_query(assertion: PriceAssertion, low: float, high: float) -> PriceQuery:
    parsed = ParsedPriceClaim(
        is_price_claim=True,
        company="셀트리온",
        assertion=assertion,
        point_date_expr="2024년 1월 8일",
        start_date_expr="",
        end_date_expr="",
        direction=PriceDirection.NONE,
        value_low=low,
        value_high=high,
        note="",
    )
    window = DateRange(date(2024, 1, 8), date(2024, 1, 8), DatePrecision.DAY, "2024년 1월 8일")
    return PriceQuery(
        parsed=parsed,
        stock=find_stock("셀트리온"),
        window=window,
        start_window=window,
        end_window=window,
        unresolved="",
    )


def test_가격_수치가_없으면_모순이_아니라_확인_불가다():
    """실측에서 오탐이 났던 바로 그 조건입니다."""
    result = verify_price_claim(
        _price_query(PriceAssertion.LOW, 0.0, 0.0), _price_snapshot()
    )
    assert result.verdict is PriceVerdict.UNVERIFIABLE
    assert "수치가 없습니다" in result.reason


def test_등락률_수치가_없어도_확인_불가다():
    result = verify_price_claim(
        _price_query(PriceAssertion.CHANGE_RATE, 0.0, 0.0), _price_snapshot()
    )
    assert result.verdict is PriceVerdict.UNVERIFIABLE


def test_수치가_있는_주장은_그대로_판정된다():
    """가드가 과해서 정상 주장까지 막으면 안 됩니다. 반대 방향 고정입니다."""
    snapshot = _price_snapshot()

    matched = verify_price_claim(_price_query(PriceAssertion.LOW, 204500.0, 204500.0), snapshot)
    assert matched.verdict is PriceVerdict.SUPPORTED

    wrong = verify_price_claim(_price_query(PriceAssertion.LOW, 150000.0, 150000.0), snapshot)
    assert wrong.verdict is PriceVerdict.CONTRADICTED


# ---------------------------------------------------------------------------
# 재무 증감률 쪽 같은 가드
# ---------------------------------------------------------------------------


def _numeric_query(growth_percent: float) -> NumericQuery:
    parsed = ParsedNumericClaim(
        is_financial_claim=True,
        company="삼성전자",
        account="매출액",
        bsns_year=2024,
        quarter=1,
        assertion=FinancialAssertion.GROWTH,
        amount_text="",
        growth_percent=growth_percent,
        direction=NumDirection.UP,
        baseline="YOY",
        fs_hint="",
        note="",
    )
    return NumericQuery(parsed=parsed, stock=find_stock("삼성전자"), amount=None, unresolved="")


def test_증감률_수치가_없으면_확인_불가다():
    """'증가했다'처럼 크기를 말하지 않은 주장이 0% 주장으로 둔갑하면 안 됩니다."""
    from pierrot.tools.financials import FinancialSnapshot

    snapshot = FinancialSnapshot.load()
    result = verify_numeric_claim(_numeric_query(0.0), snapshot)
    assert result.verdict is NumVerdict.UNVERIFIABLE
    assert "수치가 없습니다" in result.reason
