"""주장(claim) 데이터 구조.

두 종류의 모델이 있습니다.

- ``ExtractedClaim`` / ``ExtractedClaimList``: LLM에게 강제할 출력 스키마입니다.
  OpenAI Structured Outputs 에 그대로 넘겨 JSON 형식을 보장받습니다.
- ``Claim`` / ``ExtractionResult``: 파이썬 쪽에서 쓰는 최종 결과입니다.
  LLM 출력에 원문 위치(span)를 계산해 붙인 것입니다.

왜 나누는가: LLM에게 글자 단위 위치(start/end)를 직접 내놓으라고 하면 거의 틀립니다.
사람이 글자를 세지 않듯 LLM도 세지 못합니다. 그래서 LLM에게는 "원문에서 그대로 복사한
문장"만 받고, 위치 계산은 파이썬이 문자열 검색으로 처리합니다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """주장의 유형. 검증 단계에서 어떤 도구를 쓸지가 여기서 갈립니다."""

    NUMERIC = "NUMERIC"
    """재무·실적 수치. 영업이익, 매출, 순이익, 수주액 등. DART 공시로 검증합니다."""

    PRICE = "PRICE"
    """시장 데이터. 주가, 등락률, 시가총액, 거래량 등. 주가 스냅샷으로 검증합니다."""

    NARRATIVE = "NARRATIVE"
    """위 둘에 속하지 않는 서술·인과 진술. 근거 문서와 의미 비교(NLI)로 검증합니다."""


# ---------------------------------------------------------------------------
# LLM에게 강제할 출력 스키마
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    """LLM이 돌려주는 주장 하나."""

    text: str = Field(
        description=(
            "검증 가능한 주장 한 문장. 이 문장만 따로 떼어 놓아도 무엇에 대한 "
            "주장인지 알 수 있도록 주어와 대상을 명시해야 합니다. "
            "원문의 수치와 날짜는 틀렸더라도 그대로 옮깁니다."
        )
    )
    claim_type: ClaimType = Field(
        description=(
            "NUMERIC(재무·실적 수치), PRICE(주가·등락률·시가총액 등 시장 데이터), "
            "NARRATIVE(그 밖의 서술·인과 진술) 중 하나."
        )
    )
    source_text: str = Field(
        description=(
            "이 주장의 근거가 된 원문 구간을 글자 그대로 복사한 것. "
            "요약하거나 다듬지 말고 원문에 있는 연속된 문자열을 그대로 적습니다."
        )
    )


class ExtractedClaimList(BaseModel):
    """LLM이 돌려주는 주장 목록."""

    claims: list[ExtractedClaim] = Field(
        description="답변에서 찾아낸 검증 가능한 주장 전체. 없으면 빈 목록."
    )


# ---------------------------------------------------------------------------
# 파이썬 쪽 최종 결과
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """원문 위치까지 확정된 주장 하나."""

    claim_id: str
    text: str
    claim_type: ClaimType
    source_text: str

    start: int | None = None
    """원문에서 source_text 가 시작하는 글자 위치. 찾지 못하면 None."""

    end: int | None = None
    """원문에서 source_text 가 끝나는 글자 위치(미포함). 찾지 못하면 None."""

    span_found: bool = False
    """원문에서 위치를 찾았는지 여부.

    False 라면 LLM이 원문에 없는 문장을 source_text 로 적었다는 뜻입니다.
    분해 단계에서 내용을 지어낸 것이므로 검토 대상입니다.
    """

    @property
    def source_span(self) -> tuple[int, int] | None:
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)


class ExtractionResult(BaseModel):
    """답변 하나에 대한 분해 결과."""

    answer: str
    claims: list[Claim]
    model: str

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def unmatched_claims(self) -> list[Claim]:
        """원문에서 위치를 찾지 못한 주장들. 분해 품질 점검에 씁니다."""
        return [claim for claim in self.claims if not claim.span_found]

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {t.value: 0 for t in ClaimType}
        for claim in self.claims:
            counts[claim.claim_type.value] += 1
        return counts
