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
from .direction_rescue import find_claims, rescue
from .equality_claim import detect as detect_equality
from .range_claim import detect as detect_range
from .superlative import detect as detect_superlative
from .turnaround import detect as detect_turnaround
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
    """크기 없이 방향만 말한 주장.

    "전년 대비 개선되었다", "영업이익 증가폭이 크게 나타났다",
    "매출이 우수하게 증가하였다" 가 모두 여기에 해당합니다.
    꾸밈말(크게, 우수하게, 좋은)이 붙어도 재는 것은 **방향 하나**입니다.
    """

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
   - "주당순이익", "EPS" -> 주당순이익
     당기순이익과 **다른 값**입니다(주식 수로 나눈 것). 묶지 마십시오.
   - 목록에 없으면 account 를 빈 문자열로 두고 assertion 을 OTHER 로 하십시오.

5. 재무제표 주요계정에 없는 것은 OTHER 입니다.
   - 수주액, 계약 규모, 시장 점유율, 주가, 임직원 수
   - **비율 지표**: 영업이익률, 배당성향, 배당수익률, 배당금 지급 비율
     배당 **금액**은 조회할 수 있지만 **비율**은 아직 다루지 않습니다.

   부문·제품 주장은 **수치가 있든 없든** OTHER 입니다.
   - "백신 사업 매출이 3,000억원을 넘어섰다" -> OTHER (금액이 있어도 부문입니다)
   - "광고 부문 매출이 늘었다"              -> OTHER

   다만 **수치가 없다는 이유만으로 OTHER 로 두지 마십시오.**
   회사 **전체**의 계정과 기간을 알 수 있으면 DIRECTION 으로 판정할 수 있습니다.

   아래는 전부 DIRECTION 입니다. "구체적인 수치가 없다" 는 OTHER 의 사유가
   되지 못합니다. 방향은 전년 수치와 대조해 가릴 수 있기 때문입니다.
     "매출이 대폭 증가하였다"      "매출이 지속적으로 상승하였다"
     "영업이익이 흑자로 전환되었다"  "순이익이 적자를 기록하였다"
     "EPS 가 크게 증가하였다"      "실적이 개선되었다"(계정을 알 수 있을 때)
   꾸밈말(대폭·크게·지속적으로·우수하게)은 **방향을 바꾸지 않습니다.** 무시하십시오.
   - "영업이익 증가폭이 크게 나타났다" -> account=영업이익, DIRECTION, UP
   - "매출이 우수하게 증가하였다"       -> account=매출액, DIRECTION, UP
   - "순이익이 좋은 성장세를 보였다"     -> account=당기순이익, DIRECTION, UP
   - "수익성이 둔화되었다"              -> 어느 계정인지 알 수 없으므로 OTHER

   **종류를 고르는 순서입니다.** 회사 전체 수치일 때만 적용합니다.
   (부문·제품 수치는 이 순서와 무관하게 OTHER 입니다. 위 항목을 보십시오.)

   (1) 금액이 적혀 있는가?         -> AMOUNT
   (2) 증감률이 적혀 있는가?        -> GROWTH
   (3) 숫자는 없지만 방향은 있는가?  -> DIRECTION

   - "영업이익은 2.2조원 수준이다"   -> AMOUNT (수준·가량·안팎이 붙어도 금액)
   - "매출이 3,000억원을 넘어섰다"   -> AMOUNT (넘어섰다·돌파했다도 금액)
   - "영업이익이 크게 늘었다"        -> DIRECTION (금액도 증감률도 없음)

   DIRECTION 은 **금액도 증감률도 적혀 있지 않을 때만** 씁니다.
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

    conjunction: tuple[tuple[str, str], ...] = ()
    """한 문장에 사실이 여럿일 때 (계정, 방향) 쌍 전부.

    비어 있지 않으면 **전부 확인해야** 판정할 수 있습니다. 하나만 보고
    통과시키면 거짓 주장이 참으로 판정됩니다. 실제로 그랬습니다.

        "매출이 증가하였으나 영업이익과 순이익은 줄어드는 추세"
        -> 매출만 보고 '근거 있음'. 영업이익은 -7.7조에서 +23.5조로 늘었는데도.
    """

    equality: str = ""
    """'전년과 같은 수준' 처럼 변화가 없다고 말한 주장이면 그 계정 이름.

    전에는 전부 "증감 방향이 분명하지 않습니다" 로 흘러갔습니다. 같다는
    것도 주장입니다. SKT 2026년 2분기 순이익은 +488% 변했는데도
    "전년 동기와 같았습니다" 가 확인 불가로 빠져나갔습니다.
    """

    value_range: tuple[int, int] | None = None
    """'2조 원에서 3조 원으로' 처럼 양 끝을 말한 주장의 (시작값, 끝값).

    숫자가 둘인 문장이 방향 주장으로 떨어지면 **적힌 숫자를 통째로 무시**
    하게 됩니다. 실제로 그랬습니다. LG에너지솔루션 연결 영업이익은
    5,753억에서 1조 3,461억으로 늘었는데 "2조에서 3조로 늘었다" 가
    '근거 있음' 으로 통과했습니다.
    """

    turnaround: str = ""
    """부호 주장이면 그 종류(TO_PROFIT / TO_LOSS / PROFIT / LOSS).

    '흑자 전환' 은 증감이 아니라 **부호가 바뀌었는지**에 대한 주장입니다.
    증가로 읽으면 이미 흑자였던 해에도 통과합니다(실측 오탐). `turnaround`
    를 보십시오.
    """

    superlative: str = ""
    """최상급 주장('역대 최대 매출')이면 그 계정 이름.

    LLM 출력 스키마에는 넣지 않았습니다. 스키마를 바꾸면 캐시가 전부
    무효가 되고 4단계 지표를 다시 재야 합니다. 이건 파이썬이 문장을 보고
    판정할 수 있는 일이라 여기서 채웁니다(`superlative.detect`).
    """

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


def build_query(
    parsed: ParsedNumericClaim, *, claim_text: str = ""
) -> NumericQuery:
    """파싱 결과를 조회 조건으로 확정합니다.

    Args:
        claim_text: 원래 주장 문장. 파서가 방향 주장을 OTHER 로 버렸을 때
            파이썬이 다시 보기 위해 받습니다(`direction_rescue`).
    """
    # 최상급 주장을 **파서 결과보다 먼저** 봅니다.
    #
    # "2023년 역대 최대 영업이익을 기록했다" 를 파서가 DIRECTION 으로 읽으면
    # "영업이익이 늘었다" 만 확인하고 SUPPORTED 가 납니다. 늘어난 것은 맞지만
    # **역대 최대인지는 아무것도 확인하지 않았습니다.** 인과 주장을 방향으로
    # 읽어 거짓 '근거 있음' 이 났던 것과 같은 종류입니다.
    top = detect_superlative(claim_text)
    if top:
        stock = find_stock(parsed.company)
        if stock is None:
            return NumericQuery(
                parsed, None, None,
                f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
            )
        if not parsed.bsns_year or parsed.bsns_year < 1900:
            return NumericQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")
        return NumericQuery(
            parsed.model_copy(update={"account": top, "quarter": 0}),
            stock, None, "", superlative=top,
        )

    # 동등 주장도 **파서 결과보다 먼저** 봅니다.
    #
    # 파서는 '같은 수준' 에 direction=NONE 을 돌려줍니다. 그러면 판정기가
    # "증감 방향이 분명하지 않습니다" 로 끝냅니다. 방향이 없는 것이 아니라
    # **변화가 없다고 주장한 것**입니다. 확인할 수 있습니다.
    same = detect_equality(claim_text)
    if same:
        stock = find_stock(parsed.company)
        if stock is None:
            return NumericQuery(
                parsed, None, None,
                f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
            )
        if not parsed.bsns_year or parsed.bsns_year < 1900:
            return NumericQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")
        return NumericQuery(
            parsed.model_copy(
                update={"is_financial_claim": True, "account": same}
            ),
            stock, None, "", equality=same,
        )

    # 숫자가 둘인 문장은 **파서 결과보다 먼저** 집습니다.
    #
    # "2조 원에서 3조 원으로 늘었다" 를 파서가 DIRECTION 으로 읽으면
    # 방향만 보고 통과시킵니다. 적힌 숫자 둘을 아무것도 확인하지 않습니다.
    # 같은 문형을 실행마다 다르게 읽으므로 파이썬이 먼저 막습니다.
    span = detect_range(claim_text)
    if span:
        stock = find_stock(parsed.company)
        if stock is None:
            return NumericQuery(
                parsed, None, None,
                f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
            )
        if not parsed.bsns_year or parsed.bsns_year < 1900:
            return NumericQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")
        if parsed.account not in KNOWN_ACCOUNTS:
            return NumericQuery(
                parsed, stock, None,
                f"다룰 수 없는 계정입니다: {parsed.account or '(표기 없음)'}",
            )
        return NumericQuery(
            parsed.model_copy(update={"is_financial_claim": True}),
            stock, None, "", value_range=span,
        )

    # 부호 주장도 **파서 결과보다 먼저** 봅니다.
    #
    # "영업이익이 흑자로 전환됐다" 를 파서가 DIRECTION(UP) 으로 읽으면
    # "늘었다" 만 확인하고 SUPPORTED 가 납니다. 이미 흑자였던 해에도
    # 통과합니다. 최상급 주장과 같은 계열의 오탐입니다.
    kind, account = detect_turnaround(claim_text)
    if kind:
        stock = find_stock(parsed.company)
        if stock is None:
            return NumericQuery(
                parsed, None, None,
                f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
            )
        if not parsed.bsns_year or parsed.bsns_year < 1900:
            return NumericQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")
        return NumericQuery(
            parsed.model_copy(
                update={"is_financial_claim": True, "account": account}
            ),
            stock, None, "", turnaround=kind,
        )

    # 한 문장에 사실이 여럿이면 **전부** 확인해야 합니다.
    #
    # 분해 단계가 "한 주장에는 사실 하나만" 을 늘 지키지는 못합니다.
    # 쪼개지지 않은 문장을 하나의 방향 주장으로 다루면 앞부분만 보고
    # 통과시킵니다. 분해를 고치는 것과 별개로 판정기가 막아야 합니다.
    pairs = find_claims(claim_text)
    if len(pairs) > 1:
        stock = find_stock(parsed.company)
        if stock is None:
            return NumericQuery(
                parsed, None, None,
                f"유니버스에 없는 종목입니다: {parsed.company or '(표기 없음)'}",
            )
        if not parsed.bsns_year or parsed.bsns_year < 1900:
            return NumericQuery(parsed, stock, None, "사업연도를 특정할 수 없습니다")
        return NumericQuery(
            parsed.model_copy(update={"is_financial_claim": True}),
            stock, None, "", conjunction=tuple(pairs),
        )

    if not parsed.is_financial_claim or parsed.assertion is FinancialAssertion.OTHER:
        # 수치 없는 방향 주장을 파이썬이 건져 냅니다.
        #
        # 파서가 "구체적인 수치가 없다" 며 OTHER 로 버리는 일이 반복됩니다.
        # 프롬프트로 여러 번 막아 봤지만 일관되지 않습니다. 같은 형태인데
        # 하나는 통과하고 하나는 버려집니다. 그래서 검사로 바꿨습니다.
        account, direction = rescue(claim_text)
        if account and direction:
            parsed = parsed.model_copy(
                update={
                    "is_financial_claim": True,
                    "account": account,
                    "assertion": FinancialAssertion.DIRECTION,
                    "direction": Direction(direction),
                    "amount_text": "",
                    "growth_percent": 0.0,
                    "note": (parsed.note or "") + " [파이썬이 방향 주장으로 되살림]",
                }
            )
        else:
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
    return build_query(parsed, claim_text=claim_text)
