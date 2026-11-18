"""주가 주장 판정.

조회 조건(PriceQuery)과 스냅샷을 받아 SUPPORTED / CONTRADICTED / UNVERIFIABLE 중
하나로 판정합니다. 숫자는 전부 파이썬이 계산합니다.

## 범위로 판정하는 방식 (A안)

"2024년 4월 중순에 36만원이었다"를 4월 15일 하루와만 비교하면, 사람이 보기에 맞는
주장을 틀렸다고 판정하게 됩니다. 오탐이 많은 검증기는 실무에서 쓰이지 않습니다.
그래서 날짜 표현이 가리키는 구간 전체에서 관측된 값의 범위를 구하고, 주장한 값이
그 범위와 겹치면 인정합니다.

관대한 기준이라는 점은 숨기지 않습니다. 각 판정에 날짜 정밀도(DAY / RANGE)를
남겨 두고, 평가 지표를 "전체"와 "날짜가 명확한 주장만"으로 나누어 공개합니다.

## 허용 오차 — 주장이 스스로 드러내는 정밀도를 씁니다

"30만원 수준"처럼 어림수로 말한 주장을 299,500원과 비교해 틀렸다고 하면 곤란합니다.
그렇다고 일률적인 비율(예: 3%)을 쓰면 반대 방향으로 틀립니다.

실제로 겪은 사례입니다. 3% 일률 적용 시
"2024년 4월 중순 POSCO홀딩스 주가는 36만원이었다"가 근거 있음으로 통과했습니다.
그 기간 실제 종가는 371,000~400,000원이고 장중 저가 최저도 371,000원이라
**36만원은 한 번도 나온 적 없는 가격**인데, 3% 하한(359,870원)을 130원 차이로
넘겨 통과한 것입니다. 환각을 놓친 것(미탐)입니다.

그래서 허용 오차를 주장에 적힌 **자릿수**에서 읽어 냅니다.

| 주장 표기 | 읽히는 단위 | 허용 오차 |
|---|---|---|
| 36만원 (360000) | 만원 | ±5,000원 |
| 71,800원 | 백원 | ±359원 (0.5% 하한 적용) |
| 7만원 (70000) | 만원 | ±3,500원 (5% 상한 적용) |

정밀하게 말할수록 엄격하게, 어림수로 말할수록 너그럽게 봅니다.
사람이 수치를 읽는 방식과 같고, 무엇보다 **왜 그렇게 판정했는지 설명할 수 있습니다.**

하한(0.5%)과 상한(5%)을 두는 이유: 하한이 없으면 "71,800원"에 ±50원만 허용해
반올림 차이로도 모순이 나고, 상한이 없으면 "30만원"에 ±50,000원을 허용해
너무 관대해집니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .dates import DatePrecision, DateRange
from .price import PriceRow, PriceSnapshot
from .price_claim import Direction, PriceAssertion, PriceQuery
from .tolerance import (
    MAX_RELATIVE_TOLERANCE,
    MIN_RELATIVE_TOLERANCE,
    intervals_overlap,
    rounding_unit,
    tolerance_for,
)

MIN_RATE_TOLERANCE_PP = 0.5
MAX_RATE_TOLERANCE_PP = 3.0
"""등락률 허용 오차의 하한과 상한(퍼센트포인트)."""

# 이전 이름. 외부에서 참조하던 곳이 있어 남겨 둡니다.
DEFAULT_PRICE_TOLERANCE = MAX_RELATIVE_TOLERANCE
DEFAULT_RATE_TOLERANCE_PP = MAX_RATE_TOLERANCE_PP


def price_tolerance_for(value: float) -> float:
    """가격 주장 하나의 허용 오차(원)."""
    return tolerance_for(value)


def rate_tolerance_for(value: float) -> float:
    """등락률 주장 하나의 허용 오차(퍼센트포인트)."""
    return tolerance_for(
        value,
        min_absolute=MIN_RATE_TOLERANCE_PP,
        max_absolute=MAX_RATE_TOLERANCE_PP,
    )


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class PriceVerification:
    """판정 결과 하나."""

    verdict: Verdict
    reason: str
    """판정 근거를 사람이 읽을 수 있게. 숫자만으로는 설명이 되지 않습니다."""

    claimed_low: float | None = None
    claimed_high: float | None = None
    observed_low: float | None = None
    observed_high: float | None = None
    unit: str = ""
    """'원' 또는 '%'."""

    window: DateRange | None = None
    precision: DatePrecision = DatePrecision.UNKNOWN
    trading_days: int = 0
    tolerance: float = 0.0
    price_basis: str = ""
    """값 주장을 어떤 시세로 판정했는지. '원주가' 또는 '수정주가'."""

    evidence: list[str] = field(default_factory=list)

    @property
    def is_date_exact(self) -> bool:
        return self.precision is DatePrecision.DAY


_interval_overlaps = intervals_overlap


def _format(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:+.2f}%"
    return f"{value:,.0f}원"


def _claimed_rate_interval(query: PriceQuery) -> tuple[float, float]:
    """주장한 등락률을 부호 있는 구간으로 바꿉니다."""
    low, high = query.parsed.value_low, query.parsed.value_high
    low, high = min(low, high), max(low, high)
    if query.parsed.direction is Direction.DOWN:
        return (-high, -low)
    return (low, high)


def verify_price_claim(
    query: PriceQuery,
    snapshot: PriceSnapshot,
    *,
    skip_adjusted_levels: bool = True,
) -> PriceVerification:
    """조회 조건 하나를 판정합니다.

    Args:
        skip_adjusted_levels: 수정주가로 보정된 종목의 "얼마였다" 주장을
            판정하지 않고 UNVERIFIABLE 로 둡니다. 보정된 시세는 당시 호가와 달라
            맞는 주장을 틀렸다고 판정할 수 있기 때문입니다(오탐).
            등락률 주장은 수정주가가 오히려 정확하므로 그대로 판정합니다.
    """
    if not query.is_resolved:
        return PriceVerification(Verdict.UNVERIFIABLE, query.unresolved)

    assert query.stock is not None and query.window is not None
    code = query.stock.code

    is_level = query.parsed.assertion in (
        PriceAssertion.CLOSE,
        PriceAssertion.HIGH,
        PriceAssertion.LOW,
    )
    # 원주가가 있으면 보정 여부와 무관하게 값 주장을 판정할 수 있습니다.
    if (
        skip_adjusted_levels
        and is_level
        and not snapshot.has_raw(code)
        and snapshot.is_adjusted(code)
    ):
        return PriceVerification(
            Verdict.UNVERIFIABLE,
            f"{query.stock.name}은 분할·합병으로 과거 시세가 보정된 종목입니다"
            f"(호가 단위 정상 비율 {snapshot.tick_validity(code):.1%}). "
            f"보정된 시세는 당시 실제 호가와 달라 '얼마였다'는 주장을 판정할 수 없습니다. "
            f"등락률 주장은 수정주가로 판정 가능합니다.",
            window=query.window,
            precision=query.window.precision,
        )

    snapshot_span = snapshot.date_range(code)
    if snapshot_span is None:
        return PriceVerification(
            Verdict.UNVERIFIABLE, f"스냅샷에 {query.stock.name} 데이터가 없습니다"
        )

    if query.window.end < snapshot_span[0] or query.window.start > snapshot_span[1]:
        return PriceVerification(
            Verdict.UNVERIFIABLE,
            f"스냅샷 범위({snapshot_span[0]} ~ {snapshot_span[1]}) 밖의 기간입니다: "
            f"{query.window.describe()}",
            window=query.window,
            precision=query.window.precision,
        )

    if query.parsed.assertion is PriceAssertion.CHANGE_RATE:
        return _verify_change_rate(query, snapshot)
    return _verify_level(query, snapshot)


def _verify_level(query: PriceQuery, snapshot: PriceSnapshot) -> PriceVerification:
    """CLOSE / HIGH / LOW 처럼 '얼마였다'는 주장."""
    assert query.stock is not None and query.window is not None
    code = query.stock.code
    window = query.window
    rows = snapshot.rows_between(code, window.start, window.end)

    if not rows:
        return PriceVerification(
            Verdict.UNVERIFIABLE,
            f"해당 기간에 거래일이 없습니다: {window.describe()}",
            window=window,
            precision=window.precision,
        )

    # 값 주장은 원주가(당시 실제 호가)로 보는 것이 맞습니다.
    # LLM 이 말하는 "그날 얼마였다"는 보정된 시세가 아니라 당시 호가를 가리킵니다.
    use_raw = snapshot.has_raw(code)
    basis = "원주가" if use_raw else "수정주가"

    assertion = query.parsed.assertion
    if assertion is PriceAssertion.CLOSE:
        observed_low = min(row.level("close", prefer_raw=use_raw) for row in rows)
        observed_high = max(row.level("close", prefer_raw=use_raw) for row in rows)
        what = "종가"
    else:
        # 고점·저점 주장은 "그 가격에 닿았는가"를 봅니다.
        observed_low = min(row.level("low", prefer_raw=use_raw) for row in rows)
        observed_high = max(row.level("high", prefer_raw=use_raw) for row in rows)
        what = "고가" if assertion is PriceAssertion.HIGH else "저가"

    claimed_low = min(query.parsed.value_low, query.parsed.value_high)
    claimed_high = max(query.parsed.value_low, query.parsed.value_high)

    # 관측값은 실제 데이터이므로 넓히지 않습니다.
    # 넓혀야 할 쪽은 "어림수로 말한" 주장입니다.
    tolerance = max(
        price_tolerance_for(claimed_low), price_tolerance_for(claimed_high)
    )
    supported = _interval_overlaps(
        claimed_low - tolerance, claimed_high + tolerance, observed_low, observed_high
    )

    claimed_text = (
        _format(claimed_low, "원")
        if claimed_low == claimed_high
        else f"{_format(claimed_low, '원')} ~ {_format(claimed_high, '원')}"
    )
    observed_text = (
        _format(observed_low, "원")
        if observed_low == observed_high
        else f"{_format(observed_low, '원')} ~ {_format(observed_high, '원')}"
    )

    reason = (
        f"{query.stock.name} {window.describe()} 실제 {what}({basis}) {observed_text}, "
        f"주장 {claimed_text}"
    )
    reason += f" (허용 오차 ±{tolerance:,.0f}원)"
    if not supported:
        reason += " -> 겹치지 않습니다"

    return PriceVerification(
        verdict=Verdict.SUPPORTED if supported else Verdict.CONTRADICTED,
        reason=reason,
        claimed_low=claimed_low,
        claimed_high=claimed_high,
        observed_low=observed_low,
        observed_high=observed_high,
        unit="원",
        window=window,
        precision=window.precision,
        trading_days=len(rows),
        tolerance=tolerance,
        price_basis=basis,
        evidence=_evidence_lines(rows, prefer_raw=use_raw),
    )


def _verify_change_rate(query: PriceQuery, snapshot: PriceSnapshot) -> PriceVerification:
    """두 시점 사이 등락률 주장."""
    assert query.stock is not None and query.window is not None
    assert query.start_window is not None and query.end_window is not None
    code = query.stock.code

    start_rows = snapshot.rows_between(
        code, query.start_window.start, query.start_window.end
    )
    end_rows = snapshot.rows_between(code, query.end_window.start, query.end_window.end)

    if not start_rows or not end_rows:
        missing = "시작" if not start_rows else "끝"
        return PriceVerification(
            Verdict.UNVERIFIABLE,
            f"{missing} 구간에 거래일이 없습니다",
            window=query.window,
            precision=query.window.precision,
        )

    start_low = min(row.close for row in start_rows)
    start_high = max(row.close for row in start_rows)
    end_low = min(row.close for row in end_rows)
    end_high = max(row.close for row in end_rows)

    # 구간 안에서 나올 수 있는 등락률의 최소·최대를 구합니다.
    observed_low = (end_low / start_high - 1) * 100
    observed_high = (end_high / start_low - 1) * 100

    claimed_low, claimed_high = _claimed_rate_interval(query)
    tolerance_pp = max(
        rate_tolerance_for(claimed_low), rate_tolerance_for(claimed_high)
    )
    supported = _interval_overlaps(
        claimed_low - tolerance_pp, claimed_high + tolerance_pp,
        observed_low, observed_high,
    )

    claimed_text = (
        _format(claimed_low, "%")
        if claimed_low == claimed_high
        else f"{_format(claimed_low, '%')} ~ {_format(claimed_high, '%')}"
    )
    observed_text = (
        _format(observed_low, "%")
        if abs(observed_low - observed_high) < 1e-9
        else f"{_format(observed_low, '%')} ~ {_format(observed_high, '%')}"
    )

    reason = (
        f"{query.stock.name} {query.start_window.describe()} 종가 "
        f"{_format(start_low, '원')}~{_format(start_high, '원')} -> "
        f"{query.end_window.describe()} 종가 "
        f"{_format(end_low, '원')}~{_format(end_high, '원')}, "
        f"실제 등락률 {observed_text}, 주장 {claimed_text}"
    )
    reason += f" (허용 오차 ±{tolerance_pp:.2f}%p)"
    if not supported:
        reason += " -> 겹치지 않습니다"

    return PriceVerification(
        verdict=Verdict.SUPPORTED if supported else Verdict.CONTRADICTED,
        reason=reason,
        claimed_low=claimed_low,
        claimed_high=claimed_high,
        observed_low=observed_low,
        observed_high=observed_high,
        unit="%",
        window=query.window,
        precision=query.window.precision,
        trading_days=len(start_rows) + len(end_rows),
        tolerance=tolerance_pp,
        price_basis="수정주가",
        evidence=_evidence_lines(start_rows[:2] + end_rows[-2:], prefer_raw=False),
    )


def _evidence_lines(rows: list[PriceRow], *, prefer_raw: bool = True) -> list[str]:
    """판정 근거로 보여 줄 실제 시세 몇 줄."""
    return [
        f"{row.date} 시가 {row.level('open', prefer_raw=prefer_raw):,.0f} "
        f"고가 {row.level('high', prefer_raw=prefer_raw):,.0f} "
        f"저가 {row.level('low', prefer_raw=prefer_raw):,.0f} "
        f"종가 {row.level('close', prefer_raw=prefer_raw):,.0f} "
        f"({row.change * 100:+.2f}%)"
        for row in rows[:6]
    ]
