"""주장 분해. 답변 텍스트를 검증 가능한 최소 단위로 나눕니다.

사용 예:

    from pierrot.extractor import extract_claims

    result = extract_claims("삼성전자의 3분기 영업이익은 9조원입니다.")
    for claim in result.claims:
        print(claim.claim_type, claim.text)
"""

from .extractor import (
    SpanMatch,
    assemble_result,
    build_extractor,
    extract_claims,
    find_span,
    resolve_span,
)
from .quality import (
    has_amount,
    looks_like_forecast,
    quality_report,
    suspect_forecast,
    suspect_type,
)
from .models import (
    Claim,
    ClaimType,
    ExtractedClaim,
    ExtractedClaimList,
    ExtractionResult,
    MatchMode,
)

__all__ = [
    "Claim",
    "ClaimType",
    "ExtractedClaim",
    "ExtractedClaimList",
    "ExtractionResult",
    "MatchMode",
    "SpanMatch",
    "assemble_result",
    "build_extractor",
    "extract_claims",
    "find_span",
    "has_amount",
    "looks_like_forecast",
    "quality_report",
    "resolve_span",
    "suspect_forecast",
    "suspect_type",
]
