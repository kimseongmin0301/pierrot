"""분해 품질 자동 점검.

프롬프트를 고칠 때마다 결과를 사람이 눈으로 훑으면 반드시 놓칩니다.
실제로 v2 프롬프트에서 전망 문장이 주장으로 섞여 들어온 것을 뒤늦게 발견했습니다.
그래서 기계가 먼저 훑고 의심스러운 것을 올려 주도록 합니다.

여기 있는 것은 **휴리스틱**입니다. 정답 판정기가 아닙니다.
걸렸다고 반드시 잘못된 주장은 아니고, 안 걸렸다고 올바른 주장도 아닙니다.
사람이 어디를 먼저 볼지 정해 주는 용도입니다.
"""

from __future__ import annotations

import re

from .models import Claim, ClaimType

# 미래 시제와 권고 표현. 이런 문장은 지금 참·거짓을 가릴 수 없으므로
# 주장으로 뽑히면 검증 단계에서 판정 불가 항목이 되어 지표를 망칩니다.
FORECAST_PATTERN = re.compile(
    r"것으로\s*(보|예상|전망|분석|관측)"
    r"|할\s*것(입니다|이다)"
    r"|될\s*것(입니다|이다)"
    r"|전망(이다|입니다|된다|됩니다)"
    r"|기대(된다|됩니다|감)"
    r"|주시해야"
    r"|관찰해야"
    r"|주목할\s*필요"
    r"|해야\s*(한다|합니다|할\s*것)"
    r"|변수가\s*될"
)

# 금액·수량 표현. 이런 값이 들어 있는데 NARRATIVE 로 분류되면
# NLI 로 라우팅되어 숫자를 검증하지 못합니다.
AMOUNT_PATTERN = re.compile(
    r"\d[\d,\.]*\s*(조|억|만|천)\s*(원|달러|유로|엔)"
    r"|\d[\d,\.]*\s*(원|달러|유로|엔)"
)


def looks_like_forecast(text: str) -> bool:
    """전망·권고 문장으로 보이는지."""
    return bool(FORECAST_PATTERN.search(text or ""))


def has_amount(text: str) -> bool:
    """금액 표현이 들어 있는지."""
    return bool(AMOUNT_PATTERN.search(text or ""))


def suspect_forecast(claims: list[Claim]) -> list[Claim]:
    """전망·권고로 의심되는 주장들. 추출되지 말았어야 할 후보입니다."""
    return [claim for claim in claims if looks_like_forecast(claim.text)]


def suspect_type(claims: list[Claim]) -> list[Claim]:
    """유형이 잘못 붙었을 가능성이 있는 주장들.

    지금은 한 가지만 봅니다. 금액이 들어 있는데 NARRATIVE 로 분류된 경우입니다.
    수주액이나 계약 규모가 여기에 걸립니다. 공시의 금액과 대조해야 하므로
    NUMERIC 이어야 합니다.
    """
    return [
        claim
        for claim in claims
        if claim.claim_type is ClaimType.NARRATIVE and has_amount(claim.text)
    ]


def quality_report(claims: list[Claim]) -> dict[str, object]:
    """분해 품질 지표를 한 번에 계산합니다."""
    forecasts = suspect_forecast(claims)
    types = suspect_type(claims)
    review = [claim for claim in claims if claim.needs_review]
    return {
        "total": len(claims),
        "forecast_suspects": [c.claim_id for c in forecasts],
        "type_suspects": [c.claim_id for c in types],
        "span_review": [c.claim_id for c in review],
        "forecast_count": len(forecasts),
        "type_count": len(types),
        "span_review_count": len(review),
    }
