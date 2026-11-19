"""재무 주장 판정.

조회 조건(NumericQuery)과 재무 스냅샷을 받아
SUPPORTED / CONTRADICTED / UNVERIFIABLE 중 하나로 판정합니다.

## 연결과 별도를 모두 보는 이유

같은 회사, 같은 기간이라도 수치가 크게 다릅니다.

    삼성전자 2024 영업이익: 연결 32.73조 / 별도 12.36조

LLM 답변은 보통 어느 쪽인지 밝히지 않습니다. 하나만 골라 비교하면 맞는 주장을
틀렸다고 판정하게 됩니다(오탐). 그래서 **둘 중 하나라도 맞으면 인정**하되,
**어느 쪽과 맞았는지 기록**합니다. 날짜 범위를 다룬 방식과 같은 원리입니다.

관대한 기준이므로 숨기지 않습니다. matched_fs 를 남겨 두면
"별도 기준으로만 맞은 주장이 몇 건인지"를 따로 집계할 수 있습니다.

## 4분기는 계산합니다

DART 에는 4분기 보고서가 따로 없습니다. 연간에서 1~3분기를 빼야 합니다.
이것을 안 하면 "4분기 영업이익" 주장이 전부 확인 불가가 됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .amounts import format_korean_amount
from .financials import FinancialFact, FinancialSnapshot
from .numeric_claim import Baseline, Direction, FinancialAssertion, NumericQuery
from .tolerance import intervals_overlap, tolerance_for

MIN_GROWTH_TOLERANCE_PP = 1.0
MAX_GROWTH_TOLERANCE_PP = 10.0
"""증감률 허용 오차의 하한과 상한(퍼센트포인트).

증감률은 기준 시점 값이 작으면 수백 퍼센트가 나오므로 주가 등락률보다 넓게 둡니다.
예: 삼성전자 2024년 3분기 영업이익은 전년 동기 대비 277% 증가했습니다.
"""


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class NumericVerification:
    """판정 결과 하나."""

    verdict: Verdict
    reason: str

    claimed: float | None = None
    observed: float | None = None
    unit: str = ""
    """'원' 또는 '%'."""

    matched_fs: str = ""
    """어느 기준과 맞았는지. 'CFS'(연결) 또는 'OFS'(별도). 못 맞았으면 빈 문자열."""

    fs_explicit: bool = False
    """주장이 연결/별도를 직접 밝혔는지. False 면 둘 다 보고 판정한 것입니다."""

    tolerance: float = 0.0
    evidence: list[str] = field(default_factory=list)


def _fact_line(fact: FinancialFact) -> str:
    return (
        f"{fact.describe()}  (기간 {fact.period_start}~{fact.period_end}, "
        f"접수번호 {fact.rcept_no})"
    )


def _resolve_facts(
    query: NumericQuery, snapshot: FinancialSnapshot
) -> tuple[list[FinancialFact], str]:
    """주장이 가리키는 수치들을 모읍니다. 4분기는 계산합니다."""
    assert query.stock is not None
    code = query.stock.code
    year = query.parsed.bsns_year
    account = query.parsed.account
    quarter = query.parsed.quarter
    fs_div = query.fs_div

    if quarter != 4:
        return snapshot.lookup(code, year, account, quarter=quarter, fs_div=fs_div), ""

    # 4분기 = 연간 - (1분기 + 2분기 + 3분기)
    computed: list[FinancialFact] = []
    for division in ([fs_div] if fs_div else ["CFS", "OFS"]):
        annual = snapshot.lookup(code, year, account, quarter=0, fs_div=division)
        if not annual:
            continue
        quarters = [
            snapshot.lookup(code, year, account, quarter=q, fs_div=division)
            for q in (1, 2, 3)
        ]
        if not all(quarters):
            continue
        total = sum(group[0].amount for group in quarters)
        base = annual[0]
        computed.append(
            FinancialFact(
                code=base.code,
                bsns_year=base.bsns_year,
                quarter=4,
                period_label="4분기(연간에서 1~3분기를 뺀 값)",
                period_months=3,
                fs_div=base.fs_div,
                account=base.account,
                amount=base.amount - total,
                period_start=base.period_start,
                period_end=base.period_end,
                rcept_no=base.rcept_no,
            )
        )
    if not computed:
        return [], "4분기를 계산하려면 연간과 1~3분기가 모두 있어야 합니다"
    return computed, ""


def verify_numeric_claim(
    query: NumericQuery, snapshot: FinancialSnapshot
) -> NumericVerification:
    """조회 조건 하나를 판정합니다."""
    if not query.is_resolved:
        return NumericVerification(Verdict.UNVERIFIABLE, query.unresolved)

    assert query.stock is not None
    facts, problem = _resolve_facts(query, snapshot)
    if problem:
        return NumericVerification(Verdict.UNVERIFIABLE, problem)

    period = "연간" if query.parsed.quarter == 0 else f"{query.parsed.quarter}분기"
    if not facts:
        return NumericVerification(
            Verdict.UNVERIFIABLE,
            f"스냅샷에 해당 수치가 없습니다: {query.stock.name} "
            f"{query.parsed.bsns_year}년 {period} {query.parsed.account}"
            f"{' (' + query.parsed.fs_hint + ' 기준)' if query.fs_div else ''}",
        )

    assertion = query.parsed.assertion
    if assertion is FinancialAssertion.AMOUNT:
        return _verify_amount(query, facts)
    if assertion is FinancialAssertion.GROWTH:
        return _verify_growth(query, facts, snapshot)
    if assertion is FinancialAssertion.DIRECTION:
        return _verify_direction(query, facts, snapshot)
    return NumericVerification(Verdict.UNVERIFIABLE, "판정 방법이 정해지지 않은 주장입니다")


def _verify_amount(query: NumericQuery, facts: list[FinancialFact]) -> NumericVerification:
    """금액 주장. 연결과 별도 중 하나라도 맞으면 인정합니다."""
    assert query.amount is not None
    claimed = float(query.amount)
    tolerance = tolerance_for(claimed)

    matched: FinancialFact | None = None
    for fact in facts:
        if intervals_overlap(
            claimed - tolerance, claimed + tolerance, fact.amount, fact.amount
        ):
            matched = fact
            break

    observed_text = " / ".join(
        f"{fact.fs_label} {format_korean_amount(fact.amount)}" for fact in facts
    )
    reason = (
        f"{facts[0].stock.name if facts[0].stock else facts[0].code} "
        f"{facts[0].bsns_year}년 {facts[0].period_label} {facts[0].account} "
        f"실제 {observed_text}, 주장 {format_korean_amount(claimed)} "
        f"(허용 오차 ±{format_korean_amount(tolerance)})"
    )
    if matched is None:
        reason += " -> 어느 기준과도 맞지 않습니다"
    elif not query.fs_div:
        reason += f" -> {matched.fs_label} 기준과 일치"

    return NumericVerification(
        verdict=Verdict.SUPPORTED if matched else Verdict.CONTRADICTED,
        reason=reason,
        claimed=claimed,
        observed=float(matched.amount) if matched else float(facts[0].amount),
        unit="원",
        matched_fs=matched.fs_div if matched else "",
        fs_explicit=bool(query.fs_div),
        tolerance=tolerance,
        evidence=[_fact_line(fact) for fact in facts],
    )


def _baseline_target(query: NumericQuery) -> tuple[int, int] | None:
    """비교 대상의 (사업연도, 분기). 정할 수 없으면 None.

    전년 동기와 전 분기는 전혀 다른 숫자입니다. 삼성전자 2024년 3분기 영업이익은
    전년 동기 대비 +277%, 전 분기 대비 -12% 입니다. 이것을 구분하지 않으면
    엉뚱한 근거를 대게 됩니다.
    """
    year, quarter = query.parsed.bsns_year, query.parsed.quarter
    baseline = query.parsed.baseline

    if baseline is Baseline.QOQ:
        if quarter <= 0:
            return None  # 연간 수치에 "전 분기"는 성립하지 않습니다
        if quarter == 1:
            return None  # 전 분기는 작년 4분기인데 DART 에 4분기 보고서가 없습니다
        return (year, quarter - 1)

    # YOY 와 UNKNOWN 은 전년 동기로 봅니다. 재무 보도에서 가장 흔한 기준입니다.
    return (year - 1, quarter)


def _baseline_label(query: NumericQuery) -> str:
    if query.parsed.baseline is Baseline.QOQ:
        return "전 분기 대비"
    if query.parsed.baseline is Baseline.YOY:
        return "전년 동기 대비"
    return "전년 동기 대비(기준 미표기)"


def _period_name(quarter: int) -> str:
    return "연간" if quarter == 0 else f"{quarter}분기"


def _growth_pairs(
    query: NumericQuery, facts: list[FinancialFact], snapshot: FinancialSnapshot
) -> list[tuple[FinancialFact, FinancialFact]]:
    """비교 기준에 맞는 과거 수치와 짝지어 줍니다."""
    assert query.stock is not None
    target = _baseline_target(query)
    if target is None:
        return []
    base_year, base_quarter = target

    pairs: list[tuple[FinancialFact, FinancialFact]] = []
    for fact in facts:
        previous = snapshot.lookup(
            query.stock.code,
            base_year,
            query.parsed.account,
            quarter=base_quarter,
            fs_div=fact.fs_div,
        )
        if previous:
            pairs.append((fact, previous[0]))
    return pairs


def _verify_growth(
    query: NumericQuery, facts: list[FinancialFact], snapshot: FinancialSnapshot
) -> NumericVerification:
    """증감률 주장. 전년 동기 수치와 비교해 직접 계산합니다."""
    pairs = _growth_pairs(query, facts, snapshot)
    if not pairs:
        target = _baseline_target(query)
        if target is None:
            return NumericVerification(
                Verdict.UNVERIFIABLE,
                f"{_baseline_label(query)} 기준으로는 비교 대상을 정할 수 없습니다 "
                "(DART 에 4분기 보고서가 없어 1분기의 전 분기는 조회 불가)",
            )
        return NumericVerification(
            Verdict.UNVERIFIABLE,
            f"비교 대상({target[0]}년 {_period_name(target[1])}) 수치가 "
            "스냅샷에 없어 증감률을 계산할 수 없습니다",
        )

    magnitude = abs(query.parsed.growth_percent)

    # 비교할 수치가 없는 주장을 거릅니다.
    #
    # 파서는 증감률 수치를 못 찾으면 0 을 돌려줍니다. "없음"을 나타내는 필드가
    # 없기 때문입니다. 그대로 비교하면 "매출이 증가했다"처럼 **크기를 말하지 않은
    # 주장이 0% 주장으로 둔갑해** 실제 -29.9% 와 어긋나 모순 판정을 받습니다.
    # 방향만 말한 주장은 DIRECTION 으로 가야 하고, GROWTH 로 왔는데 수치가 0 이면
    # 파싱이 어긋난 것입니다.
    #
    # 같은 모양의 버그를 주가 쪽에서 실제로 겪었습니다(price_verify.py 참고).
    # 거기서는 "주장 0원" 과 실제 204,500원을 비교해 오탐이 났습니다.
    if magnitude == 0:
        return NumericVerification(
            Verdict.UNVERIFIABLE,
            "주장에 비교할 증감률 수치가 없습니다. 크기를 말하지 않은 주장은 "
            "증감률 비교로 참·거짓을 가릴 수 없습니다.",
        )

    claimed = -magnitude if query.parsed.direction is Direction.DOWN else magnitude
    tolerance = tolerance_for(
        claimed,
        min_absolute=MIN_GROWTH_TOLERANCE_PP,
        max_absolute=MAX_GROWTH_TOLERANCE_PP,
    )

    observed_parts: list[str] = []
    matched: tuple[FinancialFact, float] | None = None
    for current, previous in pairs:
        if previous.amount == 0:
            continue
        # 기준값이 음수면 증감률의 부호가 뒤집혀 해석이 어렵습니다.
        rate = (current.amount - previous.amount) / abs(previous.amount) * 100
        observed_parts.append(f"{current.fs_label} {rate:+.1f}%")
        if matched is None and intervals_overlap(
            claimed - tolerance, claimed + tolerance, rate, rate
        ):
            matched = (current, rate)

    if not observed_parts:
        return NumericVerification(
            Verdict.UNVERIFIABLE, "비교 대상 수치가 0이라 증감률을 계산할 수 없습니다"
        )

    first = pairs[0][0]
    reason = (
        f"{first.stock.name if first.stock else first.code} "
        f"{first.bsns_year}년 {first.period_label} {first.account} "
        f"{_baseline_label(query)} 실제 {' / '.join(observed_parts)}, "
        f"주장 {claimed:+.1f}% (허용 오차 ±{tolerance:.1f}%p)"
    )
    if matched is None:
        reason += " -> 어느 기준과도 맞지 않습니다"
    elif not query.fs_div:
        reason += f" -> {matched[0].fs_label} 기준과 일치"

    return NumericVerification(
        verdict=Verdict.SUPPORTED if matched else Verdict.CONTRADICTED,
        reason=reason,
        claimed=claimed,
        observed=matched[1] if matched else None,
        unit="%",
        matched_fs=matched[0].fs_div if matched else "",
        fs_explicit=bool(query.fs_div),
        tolerance=tolerance,
        evidence=[_fact_line(current) + " / 전년 " + _fact_line(previous)
                  for current, previous in pairs],
    )


def _verify_direction(
    query: NumericQuery, facts: list[FinancialFact], snapshot: FinancialSnapshot
) -> NumericVerification:
    """'개선되었다', '감소했다'처럼 방향만 말한 주장."""
    pairs = _growth_pairs(query, facts, snapshot)
    if not pairs:
        return NumericVerification(
            Verdict.UNVERIFIABLE,
            f"{_baseline_label(query)} 비교 대상 수치가 없어 방향을 판단할 수 없습니다",
        )

    wanted = query.parsed.direction
    if wanted is Direction.NONE:
        return NumericVerification(Verdict.UNVERIFIABLE, "증감 방향이 분명하지 않습니다")

    observed_parts: list[str] = []
    matched: FinancialFact | None = None
    for current, previous in pairs:
        delta = current.amount - previous.amount
        actual = Direction.UP if delta > 0 else (Direction.DOWN if delta < 0 else Direction.NONE)
        observed_parts.append(
            f"{current.fs_label} {format_korean_amount(previous.amount)} -> "
            f"{format_korean_amount(current.amount)}"
        )
        if matched is None and actual is wanted:
            matched = current

    first = pairs[0][0]
    label = "증가/개선" if wanted is Direction.UP else "감소/악화"
    reason = (
        f"{first.stock.name if first.stock else first.code} "
        f"{first.bsns_year}년 {first.period_label} {first.account} "
        f"{_baseline_label(query)} {' / '.join(observed_parts)}, 주장은 {label}"
    )
    if matched is None:
        reason += " -> 방향이 맞지 않습니다"
    elif not query.fs_div:
        reason += f" -> {matched.fs_label} 기준과 일치"

    return NumericVerification(
        verdict=Verdict.SUPPORTED if matched else Verdict.CONTRADICTED,
        reason=reason,
        matched_fs=matched.fs_div if matched else "",
        fs_explicit=bool(query.fs_div),
        evidence=[_fact_line(current) for current, _ in pairs],
    )
