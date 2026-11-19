"""API 요청·응답 스키마.

응답에 **운영 정보를 반드시 담습니다.** 판정만 돌려주면 이 시스템이
얼마나 걸리고 얼마를 쓰는지 밖에서 알 수 없습니다. 월 예산이 $5 인
프로젝트에서 호출 하나가 얼마를 쓰는지 모르면 운영이 안 됩니다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    """답변 하나를 통째로 검증합니다."""

    answer: str = Field(
        min_length=1,
        max_length=20_000,
        description="검증할 LLM 답변 텍스트.",
    )
    allow_fallback: bool = Field(
        default=False,
        description=(
            "도구가 판정하지 못했을 때 의미 비교(NLI)로 한 번 더 시도합니다. "
            "기본은 꺼짐입니다. 실측에서 켜면 커버리지가 10%p 오르지만 "
            "늘어난 9건 중 8건은 근거가 무관했습니다."
        ),
    )
    refine: bool = Field(
        default=True,
        description=(
            "주장에 회사·시점을 채우는 2단계를 돌립니다. "
            "끄면 LLM 호출이 줄지만 판정하지 못하는 주장이 늘어납니다."
        ),
    )
    annotate: bool = Field(
        default=True, description="원본 답변에 모순 표시를 붙여 함께 돌려줍니다."
    )


class VerifyClaimRequest(BaseModel):
    """주장 문장 하나만 검증합니다. 분해 단계를 건너뜁니다."""

    claim: str = Field(min_length=1, max_length=2_000)
    declared_type: str = Field(
        default="",
        description=(
            "미리 아는 유형이 있으면 NUMERIC / PRICE / NARRATIVE 중 하나. "
            "비워 두면 라우팅이 문장을 보고 정합니다."
        ),
    )
    allow_fallback: bool = Field(default=False)


class UsageInfo(BaseModel):
    """이 요청이 쓴 자원. 운영을 위해 응답마다 붙입니다."""

    elapsed_seconds: float = Field(description="요청 처리에 걸린 시간.")
    cost_usd: float = Field(description="이 요청으로 늘어난 API 비용. 캐시 적중이면 0.")
    month_spent_usd: float = Field(description="이번 달 누적 비용.")
    month_budget_usd: float = Field(description="월 예산 상한.")
    llm_calls: int = Field(
        default=0, description="이 요청에서 실제로 나간 LLM 호출 수(캐시 적중 제외)."
    )


class VerifyResponse(BaseModel):
    answer: str
    annotated_answer: str = ""
    claims: list[dict[str, Any]]
    summary: dict[str, Any]
    usage: UsageInfo
    extractor: dict[str, Any] = Field(
        default_factory=dict, description="분해 단계 정보(모델, 프롬프트 버전 등)."
    )


class VerifyClaimResponse(BaseModel):
    result: dict[str, Any]
    usage: UsageInfo


class HealthResponse(BaseModel):
    """준비 상태. 스냅샷이 없으면 검증이 되지 않습니다."""

    status: str = Field(description="ok / degraded")
    snapshots: dict[str, bool] = Field(description="스냅샷 파일이 있는지.")
    openai_key: bool = Field(description="OpenAI 키가 설정되어 있는지.")
    budget_remaining_usd: float
    notes: list[str] = Field(default_factory=list, description="문제가 있으면 사유.")


class UsageResponse(BaseModel):
    month: str
    spent_usd: float
    budget_usd: float
    remaining_usd: float
    by_source: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="어느 단계가 얼마를 썼는지. 추출/파서/판정을 나눠 봅니다.",
    )
