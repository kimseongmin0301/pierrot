"""NUMERIC 주장을 재무 조회 조건으로 바꿉니다.

"삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다" 를
(종목, 사업연도, 분기, 계정, 금액)으로 분해해야 스냅샷을 조회할 수 있습니다.

역할 분담은 주가와 같습니다. LLM 은 "무엇을 말하는가"만 뽑고, 종목코드 매핑과
금액 변환은 파이썬이 규칙으로 처리합니다. "9조 1,800억원"을 LLM 에게 숫자로
바꾸라고 하면 자릿수를 틀리고, 그 실수는 1만 배 차이를 만듭니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..llm import get_openai_chat
from .amounts import parse_korean_amount
from .dart import ACCOUNT_ALIASES
from .universe import Stock, find_stock

PARSER_VERSION = "v1"

KNOWN_ACCOUNTS: tuple[str, ...] = tuple(sorted(set(ACCOUNT_ALIASES.values())))


class FinancialAssertion(str, Enum):
    """재무 주장의 종류."""

    AMOUNT = "AMOUNT"
    """금액 주장. 예: 3분기 영업이익은 9조 1,800억원이다"""

    GROWTH = "GROWTH"
    """증감률 주장. 예: 전년 동기 대비 277% 증가했다"""

    DIRECTION = "DIRECTION"
    """방향만 말한 주장. 예: 전년 대비 개선되었다"""

    OTHER = "OTHER"
    """재무제표 조회로는 확인할 수 없는 주장. 예: 수주액, 시장 점유율, 배당수익률"""


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class Baseline(str, Enum):
    """증감률을 무엇과 비교한 것인지.

    이것을 구분하지 않으면 전혀 다른 숫자와 대조하게 됩니다.
    실제로 "전 분기 대비 15.7% 증가"라는 주장을 전년 동기(+277%)와 비교해
    엉뚱한 근거를 댄 적이 있습니다.
    """

    YOY = "YOY"
    """전년 동기 대비. 같은 분기의 작년 값과 비교합니다."""

    QOQ = "QOQ"
    """전 분기 대비. 바로 앞 분기와 비교합니다."""

    UNKNOWN = "UNKNOWN"
    """무엇과 비교했는지 알 수 없습니다. 판정할 수 없습니다."""


class ParsedNumericClaim(BaseModel):
    """LLM 이 돌려주는 파싱 결과."""

    is_financial_claim: bool = Field(
        description="재무제표(매출액, 영업이익, 순이익 등)에 대한 주장이면 true."
    )
    company: str = Field(description="주장에 나온 회사 표기를 원문 그대로. 없으면 빈 문자열.")
    account: str = Field(
        description=(
            "재무 계정 이름. 다음 중 하나로 적으십시오: "
            + ", ".join(KNOWN_ACCOUNTS)
            + ". 해당하는 것이 없으면 빈 문자열."
        )
    )
    bsns_year: int = Field(description="사업연도 4자리. 알 수 없으면 0.")
    quarter: int = Field(
        description=(
            "0 이면 연간(한 해 전체), 1~4 면 해당 분기(3개월). "
            "'3분기 영업이익'은 3, '2024년 영업이익'은 0. 알 수 없으면 -1."
        )
    )
    fs_hint: str = Field(
        description="연결 기준이면 '연결', 별도 기준이면 '별도'. 밝히지 않았으면 빈 문자열."
    )
    assertion: FinancialAssertion = Field(
        description=(
            "AMOUNT(금액을 말함), GROWTH(증감률을 말함), "
            "DIRECTION(방향만 말함), OTHER(재무제표로 확인 불가) 중 하나."
        )
    )
    amount_text: str = Field(
        description=(
            "AMOUNT 일 때 금액 표기를 원문 그대로. 예: '9조 1,800억원', '1조 6,785억원'. "
            "숫자로 바꾸지 마십시오. 해당 없으면 빈 문자열."
        )
    )
    growth_percent: float = Field(
        description="GROWTH 일 때 증감률의 절댓값. 예: '277% 증가' -> 277. 해당 없으면 0."
    )
    baseline: Baseline = Field(
        description=(
            "증감률이나 방향을 무엇과 비교했는지. "
            "'전년 동기 대비', '전년 대비' -> YOY. "
            "'전 분기 대비', '직전 분기 대비' -> QOQ. "
            "밝히지 않았으면 UNKNOWN. 해당 없으면 UNKNOWN."
        )
    )
    direction: Direction = Field(
        description="증가/개선이면 UP, 감소/악화면 DOWN, 방향이 없으면 NONE."
    )
    note: str = Field(description="판단이 애매했던 부분. 없으면 빈 문자열.")


SYSTEM_PROMPT = f"""당신은 기업 재무에 대한 주장을 조회 조건으로 바꾸는 도구입니다.

주장 문장 하나를 받아 아래를 뽑아냅니다.

- 어느 회사인가
- 어느 사업연도, 어느 기간인가
- 어떤 계정인가
- 금액을 말했는가, 증감률을 말했는가, 방향만 말했는가

## 중요한 규칙

1. 금액은 **원문 표기 그대로** 옮깁니다. 숫자로 바꾸지 마십시오.
   - "9조 1,800억원" -> amount_text="9조 1,800억원"
   - 몇 원인지 계산하는 것은 다음 단계의 일입니다.

2. 증감률은 **무엇과 비교했는지**를 반드시 구분하십시오.
   - "전년 동기 대비", "전년 대비" -> baseline=YOY
   - "전 분기 대비", "직전 분기 대비" -> baseline=QOQ
   - 이것을 틀리면 전혀 다른 숫자와 대조하게 됩니다.
     삼성전자 2024년 3분기 영업이익은 전년 동기 대비 +277%, 전 분기 대비 -12% 입니다.

3. 기간을 정확히 구분하십시오. 이것을 틀리면 전혀 다른 값과 비교하게 됩니다.
   - "2024년 3분기 영업이익" -> bsns_year=2024, quarter=3
   - "2024년 영업이익" / "2024년 연간" -> bsns_year=2024, quarter=0
   - "2024년 상반기" -> quarter=2 (2분기)
   - 연도가 없으면 bsns_year=0 입니다. 추측하지 마십시오.

4. 계정은 다음 중 하나로만 적습니다: {", ".join(KNOWN_ACCOUNTS)}
   - "매출", "영업수익", "수익" -> 매출액
   - "순이익", "당기순손익" -> 당기순이익
   - "지배주주 순이익", "지배기업 소유주지분 순이익" -> 지배주주순이익
     당기순이익과는 **다른 값**입니다(비지배지분이 빠짐). 묶지 마십시오.
   - "배당금 총액", "배당금 지급액", "현금배당금" -> 배당금총액
   - "주당 배당금", "주당 현금배당금", "DPS" -> 주당배당금
   - 목록에 없으면 account 를 빈 문자열로 두고 assertion 을 OTHER 로 하십시오.

5. 재무제표 주요계정에 없는 것은 OTHER 입니다.
   - 수주액, 계약 규모, 시장 점유율, 주가, 임직원 수
   - **비율 지표**: 영업이익률, 배당성향, 배당수익률, 배당금 지급 비율
     배당 **금액**은 조회할 수 있지만 **비율**은 아직 다루지 않습니다.
   - **사업 부문별 수치**: "광고 부문 매출", "반도체 부문 영업이익",
     "HBM 매출", "백신 사업 매출" 등
     주요계정에는 회사 전체 수치만 있습니다. 부문 수치를 전사 수치와 비교하면
     전혀 다른 숫자를 대조하게 되므로 반드시 OTHER 로 두십시오.
   - 이런 값은 재무제표 주요계정으로 확인할 수 없습니다.

6. 연결/별도를 밝혔으면 fs_hint 에 적습니다. 대부분은 밝히지 않으므로 빈 문자열입니다.

## 예시

입력: "삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다"
출력: is_financial_claim=true, company="삼성전자", account="영업이익",
     bsns_year=2024, quarter=3, assertion=AMOUNT,
     amount_text="9조 1,800억원", direction=NONE

입력: "삼성전자의 2024년 3분기 영업이익은 전년 동기 대비 277% 증가했다"
출력: is_financial_claim=true, company="삼성전자", account="영업이익",
     bsns_year=2024, quarter=3, assertion=GROWTH,
     growth_percent=277, direction=UP, baseline=YOY

입력: "삼성전자의 2024년 3분기 영업이익은 전 분기 대비 15.7% 증가했다"
출력: is_financial_claim=true, company="삼성전자", account="영업이익",
     bsns_year=2024, quarter=3, assertion=GROWTH,
     growth_percent=15.7, direction=UP, baseline=QOQ

입력: "SK하이닉스의 2024년 연간 매출은 71조 9,857억원이다"
출력: is_financial_claim=true, company="SK하이닉스", account="매출액",
     bsns_year=2024, quarter=0, assertion=AMOUNT,
     amount_text="71조 9,857억 원", direction=NONE

입력: "삼성바이오로직스는 2024년 1분기 및 2분기에만 총 15조 원 이상의 CDMO 수주를 확보했다"
출력: is_financial_claim=false, assertion=OTHER, account="",
     note="수주액은 재무제표 주요계정에 없습니다"
"""

USER_PROMPT = """다음 주장을 재무 조회 조건으로 바꾸십시오.

주장: {claim}
"""


@dataclass(frozen=True)
class NumericQuery:
    """스냅샷을 조회할 수 있는 형태로 확정된 재무 주장."""

    parsed: ParsedNumericClaim
    stock: Stock | None
    amount: int | None
    """원 단위로 변환된 주장 금액. AMOUNT 가 아니면 None."""

    unresolved: str
    """조회 조건을 만들지 못한 이유. 만들었으면 빈 문자열."""

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved

    @property
    def fs_div(self) -> str | None:
        """연결/별도를 밝혔으면 DART 코드로. 안 밝혔으면 None(둘 다 봄)."""
        hint = (self.parsed.fs_hint or "").strip()
        if "연결" in hint:
            return "CFS"
        if "별도" in hint or "개별" in hint:
            return "OFS"
        return None

    def describe(self) -> str:
        if not self.is_resolved:
            return f"조회 불가: {self.unresolved}"
        name = self.stock.name if self.stock else "?"
        period = "연간" if self.parsed.quarter == 0 else f"{self.parsed.quarter}분기"
        scope = {"CFS": " 연결", "OFS": " 별도"}.get(self.fs_div or "", "")
        return (
            f"{name}({self.stock.code if self.stock else '?'}) "
            f"{self.parsed.bsns_year}년 {period} {self.parsed.account}"
            f"{scope} / {self.parsed.assertion.value}"
        )


def build_query(parsed: ParsedNumericClaim) -> NumericQuery:
    """파싱 결과를 조회 조건으로 확정합니다."""
    if not parsed.is_financial_claim or parsed.assertion is FinancialAssertion.OTHER:
        reason = parsed.note or "재무제표 조회로 확인할 수 있는 주장이 아닙니다"
        return NumericQuery(parsed, None, None, reason)

    stock = find_stock(parsed.company)
    if stock is None:
        return NumericQuery(
            parsed, None, None,
            f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
        )

    if parsed.account not in KNOWN_ACCOUNTS:
        return NumericQuery(
            parsed, stock, None,
            f"다룰 수 없는 계정입니다: {parsed.account or '(표기 없음)'}",
        )

    if not parsed.bsns_year or parsed.bsns_year < 1900:
        return NumericQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")

    if parsed.quarter < 0 or parsed.quarter > 4:
        return NumericQuery(parsed, stock, None, "기간(연간/분기)을 특정할 수 없습니다")

    amount = None
    if parsed.assertion is FinancialAssertion.AMOUNT:
        amount = parse_korean_amount(parsed.amount_text)
        if amount is None:
            return NumericQuery(
                parsed, stock, None,
                f"금액을 읽을 수 없습니다: {parsed.amount_text or '(표기 없음)'}",
            )

    return NumericQuery(parsed, stock, amount, "")


def build_numeric_parser(
    *, settings: Settings | None = None, temperature: float = 0.0
) -> Any:
    settings = settings or get_settings()
    chat = get_openai_chat(
        "verifier",
        temperature=temperature,
        max_tokens=1024,
        source="numeric_parser",
        settings=settings,
    )
    return chat.with_structured_output(
        ParsedNumericClaim, method="json_schema", strict=True
    )


def parse_numeric_claim(
    claim_text: str,
    *,
    parser: Any | None = None,
    settings: Settings | None = None,
) -> NumericQuery:
    """주장 문장 하나를 재무 조회 조건으로 바꿉니다."""
    settings = settings or get_settings()
    if parser is None:
        parser = build_numeric_parser(settings=settings)

    parsed = parser.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(claim=claim_text)),
        ]
    )
    if isinstance(parsed, dict):
        parsed = ParsedNumericClaim.model_validate(parsed)
    return build_query(parsed)
