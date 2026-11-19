"""부문별·제품별 매출 주장을 판정합니다.

재무제표 주요계정으로는 판정할 수 없던 주장들이 대상입니다.

    "2024년 전체 HBM 매출은 45조 원 이상을 기록했다"
    "카카오의 광고 사업 부문 매출은 1,509억 원이다"
    "NAVER는 2024년 1분기 광고 부문에서 5조 7,839억 원의 매출을 기록했다"

판정 방법이 두 갈래입니다.

**1. 이름이 맞는 부문이 있으면 직접 비교합니다.**
   공시 표의 부문 이름과 답변의 표현이 같을 이유가 없으므로 부분 일치로 찾습니다.
   "반도체" 로 찾으면 SK하이닉스의 "반도체 부문" 이 걸립니다.

**2. 이름이 없으면 합계로 상한을 겁니다.**
   공시에 없는 부문이라고 해서 판정을 포기하면 안 됩니다. 전사 매출을 넘는
   부문 매출은 있을 수 없기 때문입니다.

       "NAVER 2024년 1분기 광고 부문 매출 5조 7,839억원"
       -> NAVER 에 '광고' 부문은 없지만, 2024년 전사 매출이 10조 7,377억원입니다.
          한 분기 한 부문이 연간 전사 매출의 54% 라는 것은 성립하지 않습니다.

   상한을 넘지 않으면 UNVERIFIABLE 로 두고, **공시에 있는 부문 이름을 사유에
   적습니다.** "없다"로 끝내면 다음에 무엇을 고쳐야 할지 알 수 없습니다.

오탐을 만들지 않는 것이 상한 판정의 전제입니다. 그래서 상한에는 여유를 둡니다.
자세한 것은 `OVER_TOTAL_MARGIN` 설명을 보십시오.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .segments import SegmentFact, SegmentSnapshot
from .tolerance import tolerance_for
from .universe import Stock

# 부문 이름에서 떼어 낼 꼬리말. "HBM 매출" 과 "HBM" 은 같은 것을 가리킵니다.
_SUFFIX = re.compile(
    r"\s*(사업\s*)?(부문|사업부|사업)?\s*(매출액|매출|수익|영업이익|영업수익)\s*$"
)
_TRAILING_UNIT = re.compile(r"\s*(부문|사업부문|사업부|사업)\s*$")

# 전사 매출 대비 얼마를 넘으면 모순으로 볼지.
#
# 1.0 (전사 매출과 같음) 이 이론적 상한이지만 그대로 쓰면 오탐이 납니다.
#
#   - 공시 표의 "매출" 이 내부거래 제거 전 단순합계인 경우가 있습니다.
#     한국전력은 표의 합계가 재무제표 매출액의 152% 였습니다.
#   - 주장이 가리키는 연도와 스냅샷의 연도가 한 해 어긋날 수 있습니다.
#
# 이 프로젝트는 미탐보다 오탐을 비싸게 봅니다. 그래서 1.5 배를 넘을 때만
# 모순으로 봅니다. 확실히 불가능한 것만 잡습니다.
OVER_TOTAL_MARGIN = 1.5


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class SegmentVerification:
    """판정 결과 하나."""

    verdict: Verdict
    reason: str

    keyword: str = ""
    """공시 표에서 찾을 때 쓴 이름."""

    claimed: float | None = None
    observed: float | None = None
    tolerance: float = 0.0

    matched_name: str = ""
    """실제로 맞은 공시 부문 이름."""

    total: float | None = None
    """그 해 전사 매출(표 합계). 상한 판정에 썼으면 채웁니다."""

    available: tuple[str, ...] = ()
    """공시에 있던 부문 이름들. 못 찾았을 때 사유에 적습니다."""

    evidence: list[str] = field(default_factory=list)

    @property
    def used_upper_bound(self) -> bool:
        return self.matched_name == "" and self.total is not None


def normalize_keyword(account: str) -> str:
    """주장에서 뽑힌 계정 표기를 부문 이름으로 다듬습니다.

    "HBM 매출" -> "HBM",  "광고 사업 부문 매출" -> "광고",  "철강 부문 영업이익" -> "철강"
    """
    text = (account or "").strip()
    previous = None
    while text and text != previous:
        previous = text
        text = _SUFFIX.sub("", text).strip()
        text = _TRAILING_UNIT.sub("", text).strip()
    return text


def verify_segment_claim(
    *,
    stock: Stock,
    year: int,
    account: str,
    claimed: float | None,
    snapshot: SegmentSnapshot,
    period_total: float | None = None,
    period_label: str = "",
) -> SegmentVerification:
    """부문 매출 주장 하나를 판정합니다.

    Args:
        account: 주장에서 뽑힌 계정 표기. "HBM 매출" 처럼 들어옵니다.
        claimed: 주장한 금액(원). 없으면 판정하지 않습니다.
        period_total: 주장이 가리키는 기간의 **전사 매출**. 분기 주장이면
            그 분기 매출을 넘겨 주십시오. 부문별 표는 연간이라, 분기 주장을
            연간 합계와 견주면 상한이 네 배로 느슨해집니다.

            실측 예: "NAVER 2024년 1분기 광고 부문 매출 5조 7,839억원"
            연간 상한(10조 7,377억)으로는 안 걸리지만, 1분기 전사 매출과
            견주면 명백히 불가능합니다.
        period_label: 사유에 적을 기간 표기. "2024년 1분기" 처럼.
    """
    keyword = normalize_keyword(account)
    if not keyword:
        return SegmentVerification(
            Verdict.UNVERIFIABLE, f"부문 이름을 읽을 수 없습니다: {account or '(표기 없음)'}"
        )

    available = tuple(snapshot.names(stock.code, year))
    if not available:
        return SegmentVerification(
            Verdict.UNVERIFIABLE,
            f"{stock.name} {year}년 사업보고서에서 부문별 매출 표를 찾지 못했습니다. "
            f"금융지주처럼 '매출 및 수주상황' 절이 없는 보고서가 있습니다.",
            keyword=keyword,
        )

    total_fact = snapshot.total(stock.code, year)
    # 상한은 더 촘촘한 쪽을 씁니다. 분기 매출을 받았으면 그쪽이 더 정확합니다.
    bound = period_total if period_total is not None else (
        total_fact.amount if total_fact else None
    )
    bound_label = period_label or f"{year}년 전사"
    matches = [fact for fact in snapshot.lookup(stock.code, year, keyword) if not fact.is_total]

    if claimed is None:
        return SegmentVerification(
            Verdict.UNVERIFIABLE,
            f"주장에 비교할 금액이 없습니다 ({keyword})",
            keyword=keyword,
            available=available,
        )

    if matches:
        return _compare(keyword, claimed, matches, available, total_fact)

    # 이름이 없을 때: 전사 매출 상한으로만 판단합니다.
    if bound is not None and claimed > bound * OVER_TOTAL_MARGIN:
        return SegmentVerification(
            Verdict.CONTRADICTED,
            f"{stock.name} {bound_label} 매출이 {_eok(bound)}인데 "
            f"'{keyword}' 하나가 {_eok(claimed)}이라는 것은 성립하지 않습니다. "
            f"공시된 부문: {', '.join(available[:6])}",
            keyword=keyword,
            claimed=claimed,
            total=bound,
            available=available,
            evidence=[total_fact.describe()] if total_fact else [],
        )

    return SegmentVerification(
        Verdict.UNVERIFIABLE,
        f"{stock.name} {year}년 공시에 '{keyword}' 부문이 없습니다. "
        f"공시된 부문: {', '.join(available[:8])}",
        keyword=keyword,
        claimed=claimed,
        total=bound,
        available=available,
    )


def _compare(
    keyword: str,
    claimed: float,
    matches: list[SegmentFact],
    available: tuple[str, ...],
    total_fact: SegmentFact | None,
) -> SegmentVerification:
    """이름이 맞는 부문들과 금액을 견줍니다.

    여러 개가 맞을 수 있습니다("냉연" 이 국내·해외 두 줄). 하나라도 맞으면
    인정합니다. 어느 줄을 가리킨 주장인지 공시만 보고는 알 수 없기 때문입니다.
    """
    tolerance = tolerance_for(claimed)
    hit = next(
        (fact for fact in matches if abs(fact.amount - claimed) <= tolerance), None
    )
    observed_text = " / ".join(
        f"{fact.name} {_eok(fact.amount)}" for fact in matches[:4]
    )
    reason = (
        f"공시 부문별 매출: {observed_text}, 주장 {_eok(claimed)} "
        f"(허용 오차 ±{_eok(tolerance)})"
    )
    if hit is not None:
        return SegmentVerification(
            Verdict.SUPPORTED,
            reason + f" -> '{hit.name}' 과 일치",
            keyword=keyword,
            claimed=claimed,
            observed=hit.amount,
            tolerance=tolerance,
            matched_name=hit.name,
            total=total_fact.amount if total_fact else None,
            available=available,
            evidence=[fact.describe() for fact in matches[:4]],
        )
    return SegmentVerification(
        Verdict.CONTRADICTED,
        reason + " -> 어느 것과도 맞지 않습니다",
        keyword=keyword,
        claimed=claimed,
        observed=matches[0].amount,
        tolerance=tolerance,
        total=total_fact.amount if total_fact else None,
        available=available,
        evidence=[fact.describe() for fact in matches[:4]],
    )


def _eok(amount: float) -> str:
    """원 단위 금액을 사람이 읽는 표기로."""
    if abs(amount) >= 1e12:
        return f"{amount / 1e12:,.2f}조원"
    return f"{amount / 1e8:,.0f}억원"
