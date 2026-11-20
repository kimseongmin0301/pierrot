"""부문별 매출 주장을 조회 조건으로 바꿉니다.

왜 재무 파서를 쓰지 않는가
--------------------------
재무 파서(`numeric_claim`)는 부문 주장을 만나면 `assertion=OTHER` 로 분류하고
**계정·금액 필드를 비워 둡니다.** 사유 문구에만 "광고 부문 매출은 재무제표
주요계정에 없습니다" 처럼 이름이 남습니다. 구조화된 값이 없으니 부문 표를
조회할 수가 없습니다.

재무 파서 프롬프트를 고쳐서 OTHER 일 때도 필드를 채우게 할 수도 있었습니다.
하지만 그 프롬프트는 4단계에서 측정한 지표(도구 재현율 100%, 오탐률 0%)에
묶여 있습니다. 프롬프트를 바꾸면 그 수치를 다시 재야 하고, 캐시도 전부
무효가 됩니다. **측정이 끝난 부품은 건드리지 않는 쪽**을 택했습니다.

대신 부문 전용 파서를 따로 둡니다. 재무 경로가 "주요계정에 없습니다" 로
끝났을 때만 호출하므로 비용도 그만큼만 듭니다(91건 중 20건 안팎).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..llm import get_openai_chat
from .amounts import parse_korean_amount
from .universe import Stock, find_stock

PARSER_VERSION = "segment-v2"


class ParsedSegmentClaim(BaseModel):
    """LLM 이 돌려주는 파싱 결과."""

    is_segment_claim: bool = Field(
        description=(
            "회사 전체가 아니라 **특정 사업부문·제품·서비스**의 실적에 대한 "
            "주장이면 true. 예: HBM 매출, 광고 부문 매출, 철강 부문 영업이익. "
            "회사 전체 매출이나 영업이익이면 false."
        )
    )
    company: str = Field(
        description="주장에 나온 회사 표기를 원문 그대로. 없으면 빈 문자열."
    )
    segment: str = Field(
        description=(
            "부문·제품·서비스 이름만. 꾸밈말과 '매출', '부문', '사업' 같은 "
            "꼬리말은 빼십시오. "
            "'HBM 매출' -> 'HBM', '광고 사업 부문 매출' -> '광고', "
            "'철강 부문 영업이익' -> '철강', '코로나19 변이 대응 백신 사업 매출' -> '백신'."
        )
    )
    metric: str = Field(
        description=(
            "무엇에 대한 주장인지. '매출' / '영업이익' / '기타' 중 하나만 "
            "적으십시오. 매출액·영업수익·판매액은 '매출' 입니다."
        )
    )
    bsns_year: int = Field(
        description=(
            "주장이 가리키는 사업연도. 네 자리 숫자. "
            "문장에 없으면 0. 추측하지 마십시오."
        )
    )
    quarter: int = Field(
        description=(
            "주장이 가리키는 분기. 1~4. 연간이거나 분기를 말하지 않았으면 0. "
            "'2024년 1분기' 는 1, '2024년' 은 0 입니다."
        )
    )
    amount_text: str = Field(
        description=(
            "주장한 금액 표기를 원문 그대로. '1,509억 원', '45조 원', '5조 7,839억 원'. "
            "금액이 없으면 빈 문자열."
        )
    )
    note: str = Field(description="애매했던 부분이 있으면 한 문장으로. 없으면 빈 문자열.")


# 프롬프트 끝에 파서 버전을 붙입니다.
#
# 응답 캐시(SQLite)는 프롬프트 문자열로 키를 만듭니다. 출력 스키마만 바꾸고
# 프롬프트를 그대로 두면 **옛 스키마로 저장된 응답이 그대로 돌아옵니다.**
# 실제로 `quarter` 필드를 추가했을 때 전부 "Field required" 로 터졌습니다.
# 버전을 프롬프트에 넣어 두면 스키마를 바꿀 때 캐시가 자동으로 무효가 됩니다.
SYSTEM_PROMPT = """당신은 사업부문별 실적 주장을 조회 조건으로 바꾸는 도구입니다.

주장 문장 하나를 받아 아래를 뽑아냅니다.

- 어느 회사인가
- 어느 부문·제품·서비스인가
- 어느 사업연도인가
- 얼마라고 말했는가

## 규칙

1. **문장에 없는 것은 만들지 마십시오.** 연도가 없으면 0, 금액이 없으면
   빈 문자열입니다. 그럴듯하게 채우면 틀린 조회가 됩니다.
2. 부문 이름은 **이름만** 남기십시오. "광고 사업 부문 매출"에서 쓸 것은
   "광고"입니다. 공시 표의 이름과 맞춰 찾을 것이기 때문입니다.
3. 회사 전체 실적 주장이면 `is_segment_claim` 을 false 로 두십시오.
   "삼성전자의 2024년 매출은 300조원이다" 는 부문 주장이 아닙니다.
4. 금액 표기는 **원문 그대로** 옮기십시오. 단위 환산은 하지 마십시오.
   숫자 변환은 파이썬이 합니다.
5. `metric` 은 주장이 무엇을 말하는지입니다. 매출인지 영업이익인지
   구분해야 합니다. 둘은 다른 숫자입니다.
"""
SYSTEM_PROMPT += f"\n(parser: {PARSER_VERSION})\n"

USER_PROMPT = """주장: {claim}"""


# 공시의 '매출 및 수주상황' 표에는 **매출**만 있습니다. 영업이익·순이익은
# 부문별로 공시되지 않습니다(주석의 영업부문 정보에 일부 있지만 형식이
# 제각각입니다). 매출이 아닌 주장을 이 표로 판정하면 오탐이 납니다.
#
# 실제로 났습니다.
#   "2024년 1분기 POSCO홀딩스 철강 부문 영업이익은 750억원이다"
#   -> 철강부문 **매출** 5.27조와 비교해 모순 판정 (오탐)
SUPPORTED_METRICS = {"매출", "매출액", "영업수익", "판매액"}


@dataclass(frozen=True)
class SegmentQuery:
    """조회할 수 있는 형태로 확정된 부문 주장."""

    parsed: ParsedSegmentClaim
    stock: Stock | None
    amount: int | None
    unresolved: str

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved

    def describe(self) -> str:
        if not self.is_resolved:
            return f"조회 불가: {self.unresolved}"
        name = self.stock.name if self.stock else "?"
        return (
            f"{name} {self.parsed.bsns_year}년 '{self.parsed.segment}' 부문 "
            f"{self.parsed.metric}"
            + (f" {self.parsed.quarter}분기" if self.parsed.quarter else " 연간")
        )


def build_query(parsed: ParsedSegmentClaim) -> SegmentQuery:
    """파싱 결과를 조회 조건으로 확정합니다. 판단은 전부 파이썬이 합니다."""
    if not parsed.is_segment_claim:
        return SegmentQuery(parsed, None, None, "부문별 실적 주장이 아닙니다")

    stock = find_stock(parsed.company)
    if stock is None:
        return SegmentQuery(
            parsed, None, None,
            f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
        )
    if not parsed.segment.strip():
        return SegmentQuery(parsed, stock, None, "부문 이름을 찾을 수 없습니다")

    metric = (parsed.metric or "").strip()
    if metric not in SUPPORTED_METRICS:
        return SegmentQuery(
            parsed, stock, None,
            f"공시의 부문별 표에는 매출만 있어 '{metric or '해당 지표'}' 주장은 "
            f"판정할 수 없습니다. 부문별 영업이익은 공시되지 않습니다.",
        )
    if not parsed.bsns_year or parsed.bsns_year < 1900:
        return SegmentQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")
    if parsed.quarter < 0 or parsed.quarter > 4:
        return SegmentQuery(parsed, stock, None, "기간을 특정할 수 없습니다")

    amount = parse_korean_amount(parsed.amount_text)
    if amount is None:
        # 사유를 두 가지로 나눕니다. 전에는 둘 다 "금액을 읽을 수 없습니다:
        # (표기 없음)" 으로 끝나 **무엇이 문제인지 알 수 없었습니다.**
        #
        #   금액이 아예 없다   -> 지금 도구로는 다룰 수 없는 종류입니다
        #   금액이 있는데 못 읽음 -> 표기를 못 읽은 것이므로 고칠 수 있습니다
        if not (parsed.amount_text or "").strip():
            return SegmentQuery(
                parsed, stock, None,
                f"'{parsed.segment}' 부문 주장에 금액이 적혀 있지 않습니다. "
                f"부문별 표와는 금액으로만 대조할 수 있어 증감 방향만 말한 "
                f"부문 주장은 아직 판정하지 못합니다.",
            )
        return SegmentQuery(
            parsed, stock, None,
            f"금액 표기를 읽지 못했습니다: {parsed.amount_text}",
        )
    return SegmentQuery(parsed, stock, amount, "")


def build_segment_parser(
    *, settings: Settings | None = None, temperature: float = 0.0
) -> Any:
    settings = settings or get_settings()
    chat = get_openai_chat(
        "verifier",
        temperature=temperature,
        max_tokens=1024,
        source="segment_parser",
        settings=settings,
    )
    return chat.with_structured_output(
        ParsedSegmentClaim, method="json_schema", strict=True
    )


def parse_segment_claim(
    claim_text: str,
    *,
    parser: Any | None = None,
    settings: Settings | None = None,
) -> SegmentQuery:
    """주장 문장 하나를 부문 조회 조건으로 바꿉니다."""
    settings = settings or get_settings()
    if parser is None:
        parser = build_segment_parser(settings=settings)

    parsed = parser.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(claim=claim_text)),
        ]
    )
    if isinstance(parsed, dict):
        parsed = ParsedSegmentClaim.model_validate(parsed)
    return build_query(parsed)
