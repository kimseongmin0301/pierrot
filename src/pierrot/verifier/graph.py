"""검증 에이전트 그래프.

    주장 ─→ classify ─┬─ PRICE ─────→ price ────┐
                      ├─ NUMERIC ───→ numeric ──┼─→ (확인 불가면) ─→ narrative ─┐
                      ├─ NARRATIVE ─→ narrative ┘                               │
                      └─ FORECAST ──→ forecast ────────────────────────────────┴─→ finalize

왜 스크립트 세 개를 그래프 하나로 묶는가:

- 지금은 `verify_price_claims.py` 와 `verify_numeric_claims.py` 가 따로 돌고,
  어떤 주장을 어느 쪽에 넣을지는 사람이 정합니다. 도구를 더 붙일수록
  스크립트가 늘어납니다.
- 라우팅을 코드로 고정하면 "왜 이 주장이 이 도구로 갔는지"가 기록에 남고
  테스트로 묶을 수 있습니다.
- 대체 경로를 켜고 끄며 비교할 수 있습니다. 끄면 1차 경로 성능,
  켜면 대체까지 포함한 성능이 나옵니다.

대체 경로를 기본으로 끈 이유는 실측 때문입니다. NLI 는 어려운 주장에서
오탐률이 36.5% 였습니다. 지금 도구 경로는 오탐률 0% 입니다. 대체 경로를
기본으로 켜면 **없던 오탐을 들여오게 됩니다.** 켜고 끈 결과를 둘 다 재서
숫자로 결정할 문제이지, 처음부터 켜 둘 문제가 아닙니다.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    make_classify_node,
    make_finalize_node,
    make_forecast_node,
    make_narrative_node,
    make_numeric_node,
    make_price_node,
)
from .state import Attempt, Route, VerificationResult, Verdict, VerifierState
from .tools import VerifierTools

NODE_CLASSIFY = "classify"
NODE_PRICE = "price"
NODE_NUMERIC = "numeric"
NODE_NARRATIVE = "narrative"
NODE_FORECAST = "forecast"
NODE_FINALIZE = "finalize"

_ROUTE_TO_NODE = {
    Route.PRICE.value: NODE_PRICE,
    Route.NUMERIC.value: NODE_NUMERIC,
    Route.NARRATIVE.value: NODE_NARRATIVE,
    Route.FORECAST.value: NODE_FORECAST,
}


def _route_after_classify(state: VerifierState) -> str:
    return _ROUTE_TO_NODE.get(state.get("route", ""), NODE_NARRATIVE)


def _route_after_tool(state: VerifierState) -> str:
    """1차 경로가 확인 불가면 의미 비교로 한 번 더 시도할지 정합니다."""
    if not state.get("allow_fallback"):
        return NODE_FINALIZE

    attempts = state.get("attempts") or []
    if not attempts:
        return NODE_FINALIZE
    if attempts[-1]["verdict"] != "UNVERIFIABLE":
        return NODE_FINALIZE
    # 이미 의미 비교를 거쳤으면 또 보내지 않습니다. 무한 루프 방지입니다.
    if any(attempt["route"] == Route.NARRATIVE.value for attempt in attempts):
        return NODE_FINALIZE
    return NODE_NARRATIVE


def build_graph(tools: VerifierTools | None = None):
    """컴파일된 그래프를 돌려줍니다."""
    tools = tools or VerifierTools()

    builder = StateGraph(VerifierState)
    builder.add_node(NODE_CLASSIFY, make_classify_node())
    builder.add_node(NODE_PRICE, make_price_node(tools))
    builder.add_node(NODE_NUMERIC, make_numeric_node(tools))
    builder.add_node(NODE_NARRATIVE, make_narrative_node(tools))
    builder.add_node(NODE_FORECAST, make_forecast_node())
    builder.add_node(NODE_FINALIZE, make_finalize_node())

    builder.add_edge(START, NODE_CLASSIFY)
    builder.add_conditional_edges(
        NODE_CLASSIFY,
        _route_after_classify,
        {
            NODE_PRICE: NODE_PRICE,
            NODE_NUMERIC: NODE_NUMERIC,
            NODE_NARRATIVE: NODE_NARRATIVE,
            NODE_FORECAST: NODE_FORECAST,
        },
    )

    # 도구 경로는 실패하면 의미 비교로 한 번 더 갈 수 있습니다.
    for node in (NODE_PRICE, NODE_NUMERIC):
        builder.add_conditional_edges(
            node,
            _route_after_tool,
            {NODE_NARRATIVE: NODE_NARRATIVE, NODE_FINALIZE: NODE_FINALIZE},
        )

    builder.add_edge(NODE_NARRATIVE, NODE_FINALIZE)
    builder.add_edge(NODE_FORECAST, NODE_FINALIZE)
    builder.add_edge(NODE_FINALIZE, END)

    return builder.compile()


def verify_claim(
    claim_text: str,
    *,
    claim_id: str = "",
    declared_type: str = "",
    graph=None,
    tools: VerifierTools | None = None,
    allow_fallback: bool = False,
) -> VerificationResult:
    """주장 하나를 판정합니다.

    Args:
        allow_fallback: 1차 경로가 확인 불가일 때 의미 비교(NLI)로 한 번 더
            시도합니다. 기본은 끔입니다. 이유는 모듈 설명을 보십시오.
        graph: 미리 컴파일한 그래프. 여러 건을 돌릴 때는 반드시 넘기십시오.
            넘기지 않으면 주장마다 스냅샷을 다시 읽습니다.
    """
    graph = graph or build_graph(tools)
    state: VerifierState = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "declared_type": declared_type,
        "allow_fallback": allow_fallback,
        "attempts": [],
    }
    final: dict[str, Any] = graph.invoke(state)
    return to_result(final)


def to_result(state: dict[str, Any]) -> VerificationResult:
    """그래프가 끝낸 상태를 최종 결과 구조로 옮깁니다."""
    attempts = tuple(
        Attempt(
            route=Route(attempt["route"]),
            verdict=Verdict(attempt["verdict"]),
            reason=attempt["reason"],
            evidence=list(attempt.get("evidence", [])),
            detail=dict(attempt.get("detail", {})),
        )
        for attempt in state.get("attempts", [])
    )
    return VerificationResult(
        claim_id=state.get("claim_id", ""),
        claim_text=state.get("claim_text", ""),
        verdict=Verdict(state.get("verdict", Verdict.UNVERIFIABLE.value)),
        reason=state.get("reason", ""),
        route=Route(state.get("route", Route.NARRATIVE.value)),
        route_reason=state.get("route_reason", ""),
        declared_type=state.get("declared_type", ""),
        is_forecast=bool(state.get("is_forecast", False)),
        attempts=attempts,
        evidence=list(state.get("evidence", [])),
        detail=dict(state.get("detail", {})),
    )
