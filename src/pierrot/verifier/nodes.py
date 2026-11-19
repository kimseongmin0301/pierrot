"""그래프의 노드들.

노드 하나가 하는 일은 작습니다. 상태를 받아서 자기가 바꾼 키만 돌려줍니다.
LangGraph 가 그걸 상태에 합쳐 줍니다.

설계에서 중요한 점 두 가지입니다.

1. **도구가 세 종류라 결과 모양도 세 가지**입니다(`PriceVerification`,
   `NumericVerification`, `NarrativeVerification`). 필드가 서로 다릅니다.
   이걸 노드 안에서 하나의 모양으로 바꿔 둡니다. 그래야 아래 단계가
   도구별 분기 없이 결과를 다룹니다.
2. **판정 못 한 사유를 반드시 남깁니다.** 확인 불가는 결과가 없는 게 아니라
   "왜 못 했는지"가 결과입니다. 4단계에서 커버리지를 쪼갤 때 이 사유로 갈랐습니다.
"""

from __future__ import annotations

from typing import Any, Callable

from .routing import classify
from .state import Route, VerifierState
from .tools import VerifierTools

NodeFn = Callable[[VerifierState], dict[str, Any]]


def _attempt(
    route: Route, verdict: str, reason: str, evidence: list[str], detail: dict[str, Any]
) -> dict[str, Any]:
    return {
        "route": route.value,
        "verdict": verdict,
        "reason": reason,
        "evidence": list(evidence),
        "detail": dict(detail),
    }


# ---------------------------------------------------------------------------
# 1. 유형 분류
# ---------------------------------------------------------------------------


def make_classify_node() -> NodeFn:
    """주장을 어느 도구로 보낼지 정합니다. LLM 호출이 없습니다."""

    def node(state: VerifierState) -> dict[str, Any]:
        route, reason = classify(state["claim_text"], state.get("declared_type", ""))
        return {
            "route": route.value,
            "route_reason": reason,
            "is_forecast": route is Route.FORECAST,
            "attempts": [],
        }

    return node


# ---------------------------------------------------------------------------
# 2. 주가 경로
# ---------------------------------------------------------------------------


def make_price_node(tools: VerifierTools) -> NodeFn:
    def node(state: VerifierState) -> dict[str, Any]:
        from ..tools.price_claim import parse_price_claim
        from ..tools.price_verify import verify_price_claim

        try:
            query = parse_price_claim(
                state["claim_text"], parser=tools.price_parser(), settings=tools.settings
            )
            result = verify_price_claim(query, tools.prices())
        except Exception as exc:  # noqa: BLE001
            # 파서나 조회가 터져도 파이프라인 전체를 멈추지 않습니다.
            # 주장 하나가 실패했다는 사실 자체가 기록되어야 합니다.
            return _failed(Route.PRICE, f"주가 판정 중 오류: {exc}")

        detail = {
            "claimed_low": result.claimed_low,
            "claimed_high": result.claimed_high,
            "observed_low": result.observed_low,
            "observed_high": result.observed_high,
            "unit": result.unit,
            "tolerance": result.tolerance,
            "price_basis": result.price_basis,
            "date_precision": result.precision.value,
            "trading_days": result.trading_days,
            "window_start": str(result.window.start) if result.window else "",
            "window_end": str(result.window.end) if result.window else "",
            "query": query.describe(),
        }
        return _done(Route.PRICE, result.verdict.value, result.reason, result.evidence, detail)

    return node


# ---------------------------------------------------------------------------
# 3. 재무 경로
# ---------------------------------------------------------------------------


def make_numeric_node(tools: VerifierTools) -> NodeFn:
    def node(state: VerifierState) -> dict[str, Any]:
        from ..tools.numeric_claim import parse_numeric_claim
        from ..tools.numeric_verify import verify_numeric_claim

        try:
            query = parse_numeric_claim(
                state["claim_text"], parser=tools.numeric_parser(), settings=tools.settings
            )
            result = verify_numeric_claim(query, tools.financials())
        except Exception as exc:  # noqa: BLE001
            return _failed(Route.NUMERIC, f"재무 판정 중 오류: {exc}")

        detail = {
            "claimed": result.claimed,
            "observed": result.observed,
            "unit": result.unit,
            "tolerance": result.tolerance,
            "matched_fs": result.matched_fs,
            "fs_explicit": result.fs_explicit,
            "query": query.describe(),
        }
        return _done(Route.NUMERIC, result.verdict.value, result.reason, result.evidence, detail)

    return node


# ---------------------------------------------------------------------------
# 4. 의미 비교 경로 (검색 + NLI)
# ---------------------------------------------------------------------------


def make_narrative_node(tools: VerifierTools) -> NodeFn:
    def node(state: VerifierState) -> dict[str, Any]:
        from ..tools.narrative_verify import verify_narrative_claim

        try:
            result = verify_narrative_claim(
                state["claim_text"],
                model=tools.nli(),
                financials=tools.financials(),
                prices=tools.prices(),
                documents=tools.documents(),
            )
        except Exception as exc:  # noqa: BLE001
            return _failed(Route.NARRATIVE, f"의미 비교 중 오류: {exc}")

        detail = {
            "best_entailment": result.best_entailment,
            "best_contradiction": result.best_contradiction,
            "best_evidence": result.best_evidence,
            "evidence_count": result.evidence_count,
            "retrieved_count": result.retrieved_count,
            "stocks": list(result.stocks),
            "years": list(result.years),
            "threshold": result.threshold,
        }
        return _done(
            Route.NARRATIVE, result.verdict.value, result.reason, result.evidence, detail
        )

    return node


# ---------------------------------------------------------------------------
# 5. 전망 주장
# ---------------------------------------------------------------------------


def make_forecast_node() -> NodeFn:
    """도구를 타지 않습니다. 거르지 않고 사유를 붙여 흘려보냅니다.

    걸러 버리면 분모에서 사라져 지표가 좋아 보입니다. 그건 성능이 아니라
    측정 대상을 줄인 것입니다. 그래서 남깁니다.
    """

    reason = (
        "전망·예상·권고 주장이라 지금 시점에 참·거짓을 가릴 수 없습니다. "
        "검증 대상이 아니므로 확인 불가로 둡니다."
    )

    def node(state: VerifierState) -> dict[str, Any]:
        return _done(Route.FORECAST, "UNVERIFIABLE", reason, [], {})

    return node


# ---------------------------------------------------------------------------
# 6. 마무리
# ---------------------------------------------------------------------------


def make_finalize_node() -> NodeFn:
    """여러 번 시도했다면 어느 결과를 최종으로 삼을지 정합니다.

    규칙: **판정이 난 시도 중 마지막 것**을 씁니다. 전부 확인 불가면
    **첫 시도**의 사유를 씁니다. 첫 시도가 더 구체적이기 때문입니다
    ("재무제표 주요계정에 없습니다" vs "기준을 넘는 관계가 없습니다").
    """

    def node(state: VerifierState) -> dict[str, Any]:
        attempts = state.get("attempts") or []
        if not attempts:
            return {}

        decided = [a for a in attempts if a["verdict"] != "UNVERIFIABLE"]
        chosen = decided[-1] if decided else attempts[0]
        return {
            "verdict": chosen["verdict"],
            "reason": chosen["reason"],
            "route": chosen["route"],
            "evidence": chosen.get("evidence", []),
            "detail": chosen.get("detail", {}),
        }

    return node


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _done(
    route: Route, verdict: str, reason: str, evidence: list[str], detail: dict[str, Any]
) -> dict[str, Any]:
    """시도 기록만 덧붙입니다.

    최상단 verdict/reason/evidence 는 여기서 건드리지 않습니다. 대체 경로가
    돌면 나중 노드가 앞 노드의 값을 덮어쓰기 때문입니다. 무엇을 최종으로
    삼을지는 `finalize` 한 군데에서만 정합니다.
    """
    return {"attempts": [_attempt(route, verdict, reason, evidence, detail)]}


def _failed(route: Route, reason: str) -> dict[str, Any]:
    return _done(route, "UNVERIFIABLE", reason, [], {})
