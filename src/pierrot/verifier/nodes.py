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
        update = _done(
            Route.NUMERIC, result.verdict.value, result.reason, result.evidence, detail
        )

        # 계정을 못 찾아 끝났다면 부문별 매출 표로 넘길 거리를 남깁니다.
        # 파서를 다시 부르지 않으려고 1차 결과를 그대로 전달합니다(호출은 돈입니다).
        if result.verdict.value == "UNVERIFIABLE":
            update["segment_hint"] = _segment_hint(result.reason)
        return update

    return node


# 재무 경로가 "부문 수치라 못 본다"고 끝났을 때 나오는 사유의 표지입니다.
# 이 말이 들어 있을 때만 부문 경로로 넘깁니다. 넓히면 "기간을 특정할 수
# 없습니다" 같은 전혀 다른 실패까지 부문 표로 보내 LLM 호출만 낭비합니다.
_SEGMENT_MARKS = ("주요계정에 없습니다", "부문별", "사업 부문", "주요계정으로 확인할 수 없습니다")

# 위 표지에 걸리지만 부문 주장이 **아닌** 것들입니다. 실측에서 부문 경로를 탄
# 21건 중 12건이 여기 해당했고, 전부 "부문별 실적 주장이 아닙니다"로 끝났습니다.
# 결과는 맞지만 그만큼 LLM 호출을 헛되이 썼습니다.
_NOT_SEGMENT_MARKS = (
    "배당", "영업이익률", "이익률", "지배주주", "예상치", "수익률",
    "매출 증가의 원인", "주가",
)


def _segment_hint(reason: str) -> dict[str, Any]:
    """재무 경로의 실패 사유를 보고 부문 경로로 넘길지 정합니다.

    종목·연도·금액을 여기서 뽑지 않는 이유: 재무 파서는 부문 주장을 OTHER 로
    분류하면서 **계정·금액 필드를 비워 둡니다.** 사유 문구에만 이름이 남습니다.
    그래서 부문 경로가 전용 파서로 다시 읽습니다(`segment_claim`).
    """
    text = reason or ""
    if any(mark in text for mark in _NOT_SEGMENT_MARKS):
        return {}
    if any(mark in text for mark in _SEGMENT_MARKS):
        return {"reason": text}
    return {}


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
# 4-2. 부문별 매출 경로
# ---------------------------------------------------------------------------


def make_segment_node(tools: VerifierTools) -> NodeFn:
    """사업보고서의 부문별·제품별 매출 표와 대조합니다.

    재무제표 주요계정에는 부문 수치가 없습니다. 3단계 실측에서 확인 불가
    55건 중 가장 큰 덩어리가 이것이었습니다(HBM, 광고 부문, 철강 부문 등).
    """

    def node(state: VerifierState) -> dict[str, Any]:
        from ..tools.segment_claim import parse_segment_claim
        from ..tools.segment_verify import verify_segment_claim

        snapshot = tools.segments()
        if snapshot is None:
            return _done(
                Route.SEGMENT,
                "UNVERIFIABLE",
                "부문별 매출 스냅샷이 없습니다. "
                "scripts/fetch_segment_snapshot.py 를 먼저 실행해 주십시오.",
                [],
                {},
            )

        try:
            query = parse_segment_claim(
                state["claim_text"],
                parser=tools.segment_parser(),
                settings=tools.settings,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed(Route.SEGMENT, f"부문 주장 해석 중 오류: {exc}")

        if not query.is_resolved:
            return _done(Route.SEGMENT, "UNVERIFIABLE", query.unresolved, [], {})

        assert query.stock is not None
        parsed = query.parsed

        # 분기 주장이면 그 분기의 전사 매출을 상한으로 씁니다.
        # 부문별 표는 연간이라, 분기 주장을 연간 합계와 견주면 상한이
        # 네 배로 느슨해져 명백히 불가능한 주장도 빠져나갑니다.
        period_total = None
        period_label = ""
        if parsed.quarter:
            facts = tools.financials().lookup(
                query.stock.code, parsed.bsns_year, "매출액",
                quarter=parsed.quarter, fs_div="CFS",
            )
            if facts:
                period_total = float(facts[0].amount)
                period_label = f"{parsed.bsns_year}년 {parsed.quarter}분기"

        try:
            result = verify_segment_claim(
                stock=query.stock,
                year=parsed.bsns_year,
                account=parsed.segment,
                claimed=float(query.amount) if query.amount is not None else None,
                snapshot=snapshot,
                period_total=period_total,
                period_label=period_label,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed(Route.SEGMENT, f"부문 판정 중 오류: {exc}")

        detail = {
            "query": query.describe(),
            "keyword": result.keyword,
            "claimed": result.claimed,
            "observed": result.observed,
            "tolerance": result.tolerance,
            "matched_name": result.matched_name,
            "total": result.total,
            "available": list(result.available),
            "used_upper_bound": result.used_upper_bound,
        }
        return _done(
            Route.SEGMENT, result.verdict.value, result.reason, result.evidence, detail
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
        if decided:
            chosen = decided[-1]
        else:
            # 전부 확인 불가일 때 어느 사유를 남길지.
            #
            # 도구 경로(주가·재무·부문)의 사유가 의미 비교 사유보다 구체적입니다.
            #   도구:      "NAVER 2024년 공시에 '광고' 부문이 없습니다.
            #              공시된 부문: 서치플랫폼, 커머스, ..."
            #   의미 비교: "기준(0.70)을 넘는 관계가 없습니다"
            # 그래서 도구 경로 중 **마지막** 것을 씁니다. 뒤로 갈수록
            # 더 좁혀 들어간 판단이기 때문입니다(재무 -> 부문).
            tool_attempts = [a for a in attempts if a["route"] != "NARRATIVE"]
            chosen = tool_attempts[-1] if tool_attempts else attempts[0]
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
