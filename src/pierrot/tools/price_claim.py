"""PRICE 주장을 조회 조건으로 바꿉니다.

"NAVER의 주가는 2024년 6월 18일 기준 2만 7천원 수준까지 회복되었다" 같은 문장을
(종목, 기간, 주장 유형, 주장 값)으로 분해해야 스냅샷을 조회할 수 있습니다.

## 역할 분담

LLM이 하는 일과 파이썬이 하는 일을 엄격히 나눕니다.

- LLM: 문장에서 "무엇을 말하는가"를 뽑습니다. 회사 표기, 날짜 표현(원문 그대로),
  주장 유형, 숫자. 자연어 이해가 필요한 부분만 맡깁니다.
- 파이썬: 회사 표기를 종목코드로, 날짜 표현을 구간으로, 숫자를 조회 조건으로
  바꿉니다. 계산과 매핑은 전부 규칙으로 처리합니다.

이렇게 나누는 이유는 1단계와 같습니다. LLM에게 날짜 산술이나 코드 매핑을 시키면
매번 다른 답이 나와 평가를 재현할 수 없습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..llm import get_openai_chat
from .dates import (
    DatePrecision,
    DateRange,
    combine,
    narrow_to_year_edge,
    resolve_date_expression,
)
from .universe import Stock, find_stock

PARSER_VERSION = "v1"


class PriceAssertion(str, Enum):
    """주가 주장의 종류. 무엇을 조회해 비교할지가 여기서 갈립니다."""

    CLOSE = "CLOSE"
    """특정 시점의 주가(종가) 값 주장. 예: 4월 9일 종가 71,800원이다"""

    HIGH = "HIGH"
    """기간 중 도달한 고점 주장. 예: 한때 70,800원까지 상승했다"""

    LOW = "LOW"
    """기간 중 도달한 저점 주장. 예: 5만원까지 떨어졌다"""

    CHANGE_RATE = "CHANGE_RATE"
    """두 시점 사이의 등락률 주장. 예: 3월 초부터 4월 중순까지 15% 하락했다"""

    OTHER = "OTHER"
    """시가총액, 거래량 등 주가 조회로는 확인할 수 없는 주장."""


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class ParsedPriceClaim(BaseModel):
    """LLM이 돌려주는 파싱 결과."""

    is_price_claim: bool = Field(
        description="주가(종가, 고가, 저가, 등락률)에 대한 주장이면 true."
    )
    company: str = Field(
        description="주장에 나온 회사 표기를 원문 그대로. 없으면 빈 문자열."
    )
    assertion: PriceAssertion = Field(
        description=(
            "CLOSE(특정 시점의 주가 값), HIGH(기간 중 고점), LOW(기간 중 저점), "
            "CHANGE_RATE(두 시점 사이 등락률), OTHER(시가총액·거래량 등) 중 하나."
        )
    )
    point_date_expr: str = Field(
        description=(
            "CLOSE / HIGH / LOW 일 때 그 시점을 가리키는 날짜 표현을 원문 그대로. "
            "예: '2024년 4월 9일', '2024년 4월 중순'. "
            "연도가 문장에 있으면 반드시 포함하십시오. 해당 없으면 빈 문자열."
        )
    )
    start_date_expr: str = Field(
        description=(
            "CHANGE_RATE 일 때 시작 시점의 날짜 표현을 원문 그대로. "
            "해당 없으면 빈 문자열."
        )
    )
    end_date_expr: str = Field(
        description=(
            "CHANGE_RATE 일 때 끝 시점의 날짜 표현을 원문 그대로. "
            "해당 없으면 빈 문자열."
        )
    )
    direction: Direction = Field(
        description=(
            "등락 방향. 상승이면 UP, 하락이면 DOWN, 방향이 없는 값 주장이면 NONE."
        )
    )
    value_low: float = Field(
        description=(
            "주장한 숫자의 절댓값. 범위로 말했으면 작은 쪽. "
            "가격이면 원 단위 정수(예: 71800), 등락률이면 퍼센트 수치(예: 15). "
            "'8만 6천 700원'은 86700, '약 135%'는 135 입니다."
        )
    )
    value_high: float = Field(
        description="범위로 말했으면 큰 쪽. 단일 값이면 value_low 와 같은 값."
    )
    note: str = Field(
        description="판단이 애매했던 부분이 있으면 한 문장으로. 없으면 빈 문자열."
    )


SYSTEM_PROMPT = """당신은 주가에 대한 주장을 조회 조건으로 바꾸는 도구입니다.

주장 문장 하나를 받아 아래를 뽑아냅니다.

- 어느 회사인가
- 어떤 종류의 주장인가 (주가 값 / 고점 / 저점 / 등락률 / 그 외)
- 어느 시점 또는 어느 기간인가
- 주장한 숫자는 무엇인가

## 중요한 규칙

1. 날짜는 **원문 표현 그대로** 옮깁니다. 직접 계산하지 마십시오.
   - "2024년 4월 중순" -> point_date_expr="2024년 4월 중순"
   - 4월 중순이 며칠인지 계산하는 것은 다음 단계의 일입니다.
   - 문장에 연도가 있으면 반드시 함께 옮기십시오. "4월 9일"만 적으면 어느 해인지
     알 수 없어 검증할 수 없게 됩니다.

2. 숫자는 **절댓값**으로 적고 방향은 direction 에 따로 적습니다.
   - "3.5% 하락했다" -> value_low=3.5, value_high=3.5, direction=DOWN
   - "15~20% 하락했다" -> value_low=15, value_high=20, direction=DOWN

3. 한국어 숫자 표기를 숫자로 바꿉니다. 이것은 계산이 아니라 표기 변환입니다.
   - "8만 6천 700원" -> 86700
   - "2만 7천원" -> 27000
   - "30만원" -> 300000
   - "71,800원" -> 71800
   - "₩236,000원" -> 236000

4. 주장이 주가 값과 등락률을 **둘 다** 담고 있으면, 더 구체적인 쪽을 고릅니다.
   - "80만원 수준에서 19만 5천원까지 상승했다" -> CHANGE_RATE 가 아니라,
     끝 시점의 주가 값 주장(CLOSE)으로 봅니다. 시작값과 끝값이 모두 있으므로
     start_date_expr 과 end_date_expr 도 채우되, assertion 은 CLOSE 로 두고
     value 는 끝 시점 값(195000)을 적습니다.

5. 시가총액, 거래량, 배당수익률처럼 주가 조회로 확인할 수 없는 것은 OTHER 입니다.

## 예시

입력: "삼성전자 주가는 2024년 4월 9일 종가 기준으로 71,800원이다"
출력: is_price_claim=true, company="삼성전자", assertion=CLOSE,
     point_date_expr="2024년 4월 9일", direction=NONE,
     value_low=71800, value_high=71800

입력: "삼성전자 주가는 2024년 3월 29일 한때 70,800원까지 상승했다"
출력: is_price_claim=true, company="삼성전자", assertion=HIGH,
     point_date_expr="2024년 3월 29일", direction=UP,
     value_low=70800, value_high=70800

입력: "NAVER의 주가는 2024년 3월 초부터 4월 중순까지 약 15~20% 하락했다"
출력: is_price_claim=true, company="NAVER", assertion=CHANGE_RATE,
     start_date_expr="2024년 3월 초", end_date_expr="2024년 4월 중순",
     direction=DOWN, value_low=15, value_high=20

입력: "삼성바이오로직스는 약 135%의 높은 상승률을 기록했다"
출력: is_price_claim=true, company="삼성바이오로직스", assertion=CHANGE_RATE,
     start_date_expr="", end_date_expr="", direction=UP,
     value_low=135, value_high=135,
     note="기간이 명시되지 않았습니다"
"""

USER_PROMPT = """다음 주장을 조회 조건으로 바꾸십시오.

주장: {claim}
"""


# ---------------------------------------------------------------------------
# 파싱 결과에 파이썬이 해석한 값을 붙인 것
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceQuery:
    """스냅샷을 조회할 수 있는 형태로 확정된 주장."""

    parsed: ParsedPriceClaim
    stock: Stock | None
    window: DateRange | None
    """조회할 구간. CHANGE_RATE 면 전체 구간이고, 세부는 start_window/end_window."""

    start_window: DateRange | None
    end_window: DateRange | None
    unresolved: str
    """조회 조건을 만들지 못한 이유. 만들었으면 빈 문자열."""

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved

    @property
    def precision(self) -> DatePrecision:
        return self.window.precision if self.window else DatePrecision.UNKNOWN

    def describe(self) -> str:
        if not self.is_resolved:
            return f"조회 불가: {self.unresolved}"
        name = self.stock.name if self.stock else "?"
        return (
            f"{name}({self.stock.code if self.stock else '?'}) "
            f"{self.parsed.assertion.value} "
            f"{self.window.describe() if self.window else '?'}"
        )


def build_query(
    parsed: ParsedPriceClaim,
    *,
    reference: date | None = None,
    source: str = "",
) -> PriceQuery:
    """파싱 결과를 조회 조건으로 확정합니다.

    여기서 종목코드 매핑과 날짜 해석이 일어납니다. 실패하면 그 이유를 남깁니다.
    이유를 남기는 것이 중요합니다. "검증 못 했다"가 아니라 "왜 못 했다"가 있어야
    분해 품질을 개선할 지점을 찾을 수 있습니다.

    `source` 는 주장 원문입니다. 파서가 고른 날짜 표현만으로는 알 수 없는
    것을 되살리는 데 씁니다. '연초 대비' 가 그렇습니다.
    """
    if not parsed.is_price_claim or parsed.assertion is PriceAssertion.OTHER:
        return PriceQuery(parsed, None, None, None, None, "주가 조회로 확인할 수 있는 주장이 아닙니다")

    stock = find_stock(parsed.company) or find_stock(parsed.note)
    if stock is None:
        return PriceQuery(parsed, None, None, None, None,
                          f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}")

    if parsed.assertion is PriceAssertion.CHANGE_RATE:
        start = resolve_date_expression(parsed.start_date_expr, reference=reference)
        end = resolve_date_expression(parsed.end_date_expr, reference=reference)
        # '연초 대비' 가 한 달로 펼쳐져 들어오는 길이 있습니다. 원문을 보고 되돌립니다.
        start = narrow_to_year_edge(start, source, "연초")
        end = narrow_to_year_edge(end, source, "연말")
        if start is None or end is None:
            missing = []
            if start is None:
                missing.append(f"시작({parsed.start_date_expr or '표기 없음'})")
            if end is None:
                missing.append(f"끝({parsed.end_date_expr or '표기 없음'})")
            return PriceQuery(parsed, stock, None, start, end,
                              "기간을 특정할 수 없습니다: " + ", ".join(missing))
        return PriceQuery(parsed, stock, combine(start, end), start, end, "")

    point = resolve_date_expression(parsed.point_date_expr, reference=reference)
    if point is None:
        return PriceQuery(parsed, stock, None, None, None,
                          f"시점을 특정할 수 없습니다: {parsed.point_date_expr or '표기 없음'}")
    return PriceQuery(parsed, stock, point, None, None, "")


# ---------------------------------------------------------------------------
# LLM 호출
# ---------------------------------------------------------------------------


def build_price_parser(
    *, settings: Settings | None = None, temperature: float = 0.0
) -> Any:
    settings = settings or get_settings()
    chat = get_openai_chat(
        "verifier",
        temperature=temperature,
        max_tokens=1024,
        source="price_parser",
        settings=settings,
    )
    return chat.with_structured_output(
        ParsedPriceClaim, method="json_schema", strict=True
    )


def parse_price_claim(
    claim_text: str,
    *,
    parser: Any | None = None,
    settings: Settings | None = None,
    reference: date | None = None,
) -> PriceQuery:
    """주장 문장 하나를 조회 조건으로 바꿉니다."""
    settings = settings or get_settings()
    if parser is None:
        parser = build_price_parser(settings=settings)

    parsed = parser.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(claim=claim_text)),
        ]
    )
    if isinstance(parsed, dict):
        parsed = ParsedPriceClaim.model_validate(parsed)
    return build_query(parsed, reference=reference, source=claim_text)
