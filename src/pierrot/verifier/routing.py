"""주장을 어느 도구로 보낼지 정합니다.

왜 추출기 유형을 그대로 쓰지 않는가: 실측에서 잘못 붙은 유형이 나왔습니다.
"주가 수익률" 주장이 NUMERIC 으로 분류되어 재무제표에서 찾다가
"주요계정에 없습니다" 로 끝났습니다. 주가 스냅샷으로 가면 계산되는 주장입니다.
금액이 들어 있는데 NARRATIVE 로 분류된 경우도 마찬가지입니다(수주액, 계약 규모).

왜 LLM 을 쓰지 않는가: 세 갈래 분류에 호출당 비용을 치를 이유가 없습니다.
이 프로젝트는 **LLM 이 식별하고 파이썬이 판단하는** 방식을 지켜 왔습니다.
규칙으로 먼저 처리하고, 규칙이 틀리는 사례가 실측으로 나오면 그때 바꿉니다.
규칙은 테스트로 고정할 수 있지만 LLM 분류는 그렇지 않습니다.

규칙이 뒤집은 경우에는 사유를 반드시 남깁니다. 남기지 않으면 나중에
"왜 이 주장이 주가로 갔지?" 를 코드를 읽어서 역추적해야 합니다.
"""

from __future__ import annotations

import re

from ..extractor.quality import has_amount, looks_like_forecast
from .state import Route

# 1단계 품질 휴리스틱(`quality.looks_like_forecast`)이 못 잡는 전망 표현입니다.
# 실측에서 다음이 빠져나갔습니다.
#   "약 1.3조 원 수준으로 예상된다"  "2024년 예상 영업이익률"
#   "1,500억원 이상으로 지급할 계획이다"  "6~7%를 넘어설 것이다"
# quality.py 를 고치지 않는 이유: 그 패턴은 1단계에서 측정한 수치
# (전망 과다추출 7건)에 묶여 있습니다. 바꾸면 그 수치가 재현되지 않습니다.
# 라우팅에만 필요한 보강이므로 여기 둡니다.
FORECAST_EXTRA_PATTERN = re.compile(
    r"예상(된다|됩니다|되는|치|되며)"
    r"|예상\s*[가-힣]{2,}(은|는|이|가)"
    r"|계획(이다|입니다|이며)"
    r"|[가-힣]\s*것(이다|입니다)"
    r"|전망(이|을|에)?\s*(이다|입니다|된다|됩니다)?"
)


def is_forecast(text: str) -> bool:
    """전망·권고 주장인지. 1단계 휴리스틱에 보강 규칙을 더해 봅니다."""
    return looks_like_forecast(text) or bool(FORECAST_EXTRA_PATTERN.search(text or ""))

# 주가 스냅샷으로 조회·계산할 수 있는 표현.
# "수익률" 은 단독으로는 모호합니다(배당수익률은 재무 쪽). 그래서 넣지 않고,
# "주가" 같은 확실한 낱말과 함께 나올 때만 PRICE 로 갑니다.
PRICE_PATTERN = re.compile(
    r"주가|주식\s*가격|종가|시초가|고가|저가|상한가|하한가"
    r"|시가총액|거래량|거래대금|52주|주당\s*가격|주식\s*시세"
)

# DART 재무제표·공시로 대조할 수 있는 표현.
NUMERIC_PATTERN = re.compile(
    r"매출|영업이익|순이익|당기순|영업손실|적자|흑자"
    r"|자산|부채|자본|이익잉여금|현금흐름"
    r"|배당|수주|영업이익률|이익률|마진"
)


def classify(claim_text: str, declared_type: str) -> tuple[Route, str]:
    """(경로, 사유)를 돌려줍니다.

    Args:
        claim_text: 주장 문장.
        declared_type: 추출기가 붙인 유형. NUMERIC / PRICE / NARRATIVE.

    판정 순서가 곧 우선순위입니다.

    1. 전망·권고면 도구를 타지 않습니다. 지금 참·거짓을 가릴 수 없습니다.
    2. 낱말 증거가 한쪽에만 있으면 그쪽으로 보냅니다. 추출기 유형보다 우선합니다.
    3. 양쪽 다 있거나 양쪽 다 없으면 추출기 유형을 따릅니다.
    4. NARRATIVE 인데 금액이 들어 있으면 NUMERIC 으로 돌립니다.
    """
    text = claim_text or ""
    declared = (declared_type or "").upper()

    if is_forecast(text):
        return Route.FORECAST, "전망·권고 표현이 있어 지금 참·거짓을 가릴 수 없습니다"

    price_hits = PRICE_PATTERN.findall(text)
    numeric_hits = NUMERIC_PATTERN.findall(text)

    # 2. 한쪽에만 증거가 있는 경우
    if price_hits and not numeric_hits:
        return _with_override(
            Route.PRICE, declared, f"주가 낱말({', '.join(sorted(set(price_hits)))})이 있습니다"
        )
    if numeric_hits and not price_hits:
        return _with_override(
            Route.NUMERIC, declared, f"재무 낱말({', '.join(sorted(set(numeric_hits)))})이 있습니다"
        )

    # 양쪽 다 걸린 경우. "주가" 는 강한 증거로 보고 PRICE 를 택합니다.
    # "주가 수익률", "주가가 올라 시가총액이 매출을 넘었다" 같은 문장 때문입니다.
    if price_hits and numeric_hits:
        if re.search(r"주가|종가|시가총액", text):
            return _with_override(
                Route.PRICE, declared, "재무 낱말도 있지만 주가 표현이 더 구체적입니다"
            )
        return _with_override(Route.NUMERIC, declared, "재무 낱말과 주가 낱말이 섞여 있습니다")

    # 3~4. 낱말 증거가 없을 때는 추출기 유형을 따릅니다.
    if declared == "PRICE":
        return Route.PRICE, "낱말 증거는 없고 추출기 유형을 따릅니다"
    if declared == "NUMERIC":
        return Route.NUMERIC, "낱말 증거는 없고 추출기 유형을 따릅니다"

    if has_amount(text):
        return Route.NUMERIC, "NARRATIVE 로 분류됐지만 금액 표현이 있어 재무로 돌립니다"

    return Route.NARRATIVE, "수치 주장으로 볼 근거가 없어 의미 비교로 보냅니다"


def _with_override(route: Route, declared: str, reason: str) -> tuple[Route, str]:
    """추출기 유형을 뒤집었다면 사유에 그 사실을 적습니다."""
    if declared and declared != route.value:
        return route, f"{reason} (추출기 유형 {declared} 을 뒤집음)"
    return route, reason
