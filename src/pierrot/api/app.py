"""FastAPI 앱.

    uvicorn pierrot.api.app:app --reload
    .venv/Scripts/python.exe scripts/run_api.py

엔드포인트

    GET  /health        준비 상태. 스냅샷과 키가 있는지.
    POST /verify        답변 하나를 분해하고 주장마다 판정
    POST /verify/claim  주장 문장 하나만 판정 (분해 건너뜀)
    GET  /usage         이번 달 비용을 단계별로

응답마다 `usage` 를 붙입니다. 이 시스템은 월 예산이 $5 이므로
**호출 하나가 얼마를 썼는지 응답에서 바로 보여야** 운영할 수 있습니다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..config import get_settings
from ..cost import BudgetExceededError
from .schemas import (
    HealthResponse,
    UsageResponse,
    VerifyClaimRequest,
    VerifyClaimResponse,
    VerifyRequest,
    VerifyResponse,
)
from .service import VerificationService

logger = logging.getLogger("pierrot.api")

DESCRIPTION = """LLM 답변을 검증 가능한 주장 단위로 분해하고,
공시·주가 데이터로 하나씩 대조해 주장별 판정을 붙입니다.

판정은 세 가지입니다.

- `SUPPORTED` 근거와 일치합니다.
- `CONTRADICTED` 근거와 어긋납니다. 환각으로 의심되는 항목입니다.
- `UNVERIFIABLE` 판정할 수 없습니다. **사유를 반드시 함께 돌려줍니다.**

판정의 성격(`confidence`)도 함께 봅니다.

- `CALCULATED` 공시·주가 수치와 직접 대조했습니다. 근거를 숫자로 제시합니다.
- `INFERRED` 의미 비교(NLI)로 추정했습니다. 근거 문장이 주장과 무관할 수
  있으므로 그대로 믿으면 안 됩니다.
- `NONE` 판정하지 않았습니다.
"""


def create_app(service: VerificationService | None = None) -> FastAPI:
    """앱을 만듭니다.

    Args:
        service: 테스트에서 가짜를 끼워 넣기 위한 자리입니다.
            생략하면 시작할 때 진짜를 만듭니다(스냅샷·모델이 올라갑니다).
    """
    state: dict[str, VerificationService | None] = {"service": service}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 무거운 것은 **시작할 때 한 번만** 올립니다.
        # 요청마다 올리면 주장 하나에 몇 초씩 더 걸립니다.
        if state["service"] is None:
            state["service"] = VerificationService(get_settings())
        logger.info("준비 완료: %s", state["service"].health()["status"])
        yield

    app = FastAPI(
        title="PIERROT",
        description=DESCRIPTION,
        version="0.7.0",
        lifespan=lifespan,
    )

    def current() -> VerificationService:
        instance = state["service"]
        if instance is None:  # pragma: no cover - lifespan 이 먼저 돕니다
            raise HTTPException(status_code=503, detail="아직 준비되지 않았습니다")
        return instance

    @app.get("/health", response_model=HealthResponse, summary="준비 상태")
    def health() -> HealthResponse:
        return HealthResponse(**current().health())

    @app.get("/usage", response_model=UsageResponse, summary="이번 달 비용")
    def usage() -> UsageResponse:
        service = current()
        summary = service.cost.summary()
        return UsageResponse(
            month=summary["month"],
            spent_usd=summary["spent_usd"],
            budget_usd=summary["budget_usd"],
            remaining_usd=summary["remaining_usd"],
            by_source=service.usage_by_source(),
        )

    @app.post("/verify", response_model=VerifyResponse, summary="답변 검증")
    def verify(request: VerifyRequest) -> VerifyResponse:
        service = current()
        try:
            report, used, extractor_info = service.verify_answer(
                request.answer,
                allow_fallback=request.allow_fallback,
                refine=request.refine,
            )
        except BudgetExceededError as exc:
            # 예산 초과는 서버 잘못이 아니라 "지금은 받을 수 없음" 입니다.
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("검증 실패")
            raise HTTPException(status_code=500, detail=f"검증 실패: {exc}") from exc

        payload = report.to_dict()
        return VerifyResponse(
            answer=payload["answer"],
            annotated_answer=report.annotated_answer() if request.annotate else "",
            claims=payload["claims"],
            summary=payload["summary"],
            usage=used.to_dict(),
            extractor=extractor_info,
        )

    @app.post(
        "/verify/claim", response_model=VerifyClaimResponse, summary="주장 하나 검증"
    )
    def verify_one(request: VerifyClaimRequest) -> VerifyClaimResponse:
        service = current()
        try:
            result, used = service.verify_single_claim(
                request.claim,
                declared_type=request.declared_type,
                allow_fallback=request.allow_fallback,
            )
        except BudgetExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("판정 실패")
            raise HTTPException(status_code=500, detail=f"판정 실패: {exc}") from exc

        return VerifyClaimResponse(result=result.to_dict(), usage=used.to_dict())

    return app


app = create_app()
