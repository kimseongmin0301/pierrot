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


class MatchMode(str, Enum):
    """원문 위치를 어떤 방식으로 찾았는지.

    분해 품질을 모니터링하기 위한 값입니다. EXACT 에서 멀어질수록 LLM이 원문을
    그대로 옮기지 않았다는 뜻이고, NOT_FOUND 는 아예 없는 문장을 적었다는 뜻입니다.
    """

    EXACT = "EXACT"
    """원문에 글자 그대로 있었습니다."""

    WHITESPACE = "WHITESPACE"
    """공백과 줄바꿈 차이만 있었습니다."""

    NORMALIZED = "NORMALIZED"
    """마크다운 기호나 앞뒤 문장부호 차이까지 무시하니 일치했습니다."""

    FUZZY = "FUZZY"
    """위 방법으로는 못 찾고 유사도 기준을 넘겨 인정했습니다. 검토가 필요합니다."""

    NOT_FOUND = "NOT_FOUND"
    """원문에서 찾지 못했습니다. LLM이 지어냈을 가능성이 높습니다."""


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

    match_mode: MatchMode = MatchMode.NOT_FOUND
    """위치를 어떤 방식으로 찾았는지. 분해 품질 모니터링에 씁니다."""

    match_ratio: float = 0.0
    """FUZZY 로 찾았을 때의 유사도(0~1). 그 외에는 1.0 또는 0.0."""

    @property
    def source_span(self) -> tuple[int, int] | None:
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)

    @property
    def needs_review(self) -> bool:
        """사람이 들여다봐야 하는 주장인지.

        원문을 못 찾았거나 유사도로 겨우 찾은 경우입니다.
        """
        return self.match_mode in (MatchMode.NOT_FOUND, MatchMode.FUZZY)


class ExtractionResult(BaseModel):
    """답변 하나에 대한 분해 결과."""

    answer: str
    claims: list[Claim]
    model: str

    prompt_version: str = ""
    """어떤 버전의 프롬프트로 분해했는지.

    프롬프트를 고치면 결과가 달라지므로, 지표를 비교하려면 어떤 프롬프트로 나온
    숫자인지가 남아 있어야 합니다.
    """

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def unmatched_claims(self) -> list[Claim]:
        """원문에서 위치를 찾지 못한 주장들. 분해 품질 점검에 씁니다."""
        return [claim for claim in self.claims if not claim.span_found]

    @property
    def review_needed_claims(self) -> list[Claim]:
        """사람이 확인해야 하는 주장들 (찾지 못했거나 유사도로 겨우 찾은 것)."""
        return [claim for claim in self.claims if claim.needs_review]

    def count_by_match_mode(self) -> dict[str, int]:
        counts: dict[str, int] = {mode.value: 0 for mode in MatchMode}
        for claim in self.claims:
            counts[claim.match_mode.value] += 1
        return counts

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {t.value: 0 for t in ClaimType}
        for claim in self.claims:
            counts[claim.claim_type.value] += 1
        return counts
