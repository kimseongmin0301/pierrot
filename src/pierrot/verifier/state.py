"""검증 에이전트가 들고 다니는 상태와 최종 결과 구조.

왜 따로 두는가: 도구 세 개(`price_verify`, `numeric_verify`, `narrative_verify`)가
각자 `Verdict` 와 결과 dataclass 를 따로 갖고 있습니다. 필드도 서로 다릅니다.
에이전트는 이것들을 **하나의 모양**으로 합쳐서 내보내야 합니다. 그래야
평가 파이프라인과 API 가 도구별 분기 없이 결과를 다룰 수 있습니다.

판정이 왜 그렇게 나왔는지를 추적할 수 있어야 하므로, 어떤 경로를 거쳤고
무엇을 시도했는지(`attempts`)를 함께 남깁니다. 판정만 남기면 나중에
"이건 왜 확인 불가지?" 를 코드를 읽어서 역추적해야 합니다.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, TypedDict


class Verdict(str, Enum):
    """에이전트의 최종 판정. 도구 세 개의 Verdict 를 이 하나로 모읍니다."""

    SUPPORTED = "SUPPORTED"
    """근거와 일치합니다."""

    CONTRADICTED = "CONTRADICTED"
    """근거와 어긋납니다. 환각으로 의심되는 항목입니다."""

    UNVERIFIABLE = "UNVERIFIABLE"
    """판정할 수 없습니다. 사유를 반드시 남깁니다."""


class Route(str, Enum):
    """주장을 어느 도구로 보낼지."""

    PRICE = "PRICE"
    """주가 스냅샷으로 조회·계산합니다."""

    NUMERIC = "NUMERIC"
    """DART 재무 스냅샷과 대조합니다."""

    NARRATIVE = "NARRATIVE"
    """근거 문장을 검색해 NLI 로 의미를 비교합니다."""

    FORECAST = "FORECAST"
    """전망·권고 주장. 지금 참·거짓을 가릴 수 없으므로 도구를 타지 않습니다.

    거르지 않고 흘려보냅니다. 걸러 버리면 "검증기가 무엇을 못 하는지" 가
    지표에서 사라지기 때문입니다. 대신 `is_forecast` 로 표시해 두어
    전망을 포함한 커버리지와 제외한 커버리지를 따로 낼 수 있게 합니다.
    """


class Confidence(str, Enum):
    """판정의 성격. 같은 CONTRADICTED 라도 근거의 질이 다릅니다.

    실측으로 확인한 차이입니다. 91건을 돌린 결과, 도구 경로가 잡은 34건은
    실제 수치와의 차이를 숫자로 제시합니다. 반면 NLI 경로가 잡은 2건은
    근거 문장이 주장과 무관했습니다.

        주장: 카카오 게임 사업부의 수익성 악화가 지속되었다
        근거: 2019년말 카카오비즈보드 출시 이후 ... 광고매출의 수익성이 높아지고 있습니다
        -> 모순 0.999  (광고 매출 이야기인데 게임 사업부 주장을 반박했다고 봄)

    둘을 같은 칸에 넣고 세면 "모순 36건"이 되지만, 그중 2건은 믿을 수 없습니다.
    그래서 결과에 성격을 붙여 두고 지표를 따로 낼 수 있게 합니다.
    """

    CALCULATED = "CALCULATED"
    """공시·주가 수치를 직접 조회해 계산으로 대조했습니다. 근거를 숫자로 제시합니다."""

    INFERRED = "INFERRED"
    """문장 의미 비교(NLI)로 추정했습니다. 근거 문장이 주장과 무관할 수 있습니다."""

    NONE = "NONE"
    """판정하지 않았습니다."""


_ROUTE_CONFIDENCE = {
    "PRICE": Confidence.CALCULATED,
    "NUMERIC": Confidence.CALCULATED,
    "NARRATIVE": Confidence.INFERRED,
    "FORECAST": Confidence.NONE,
}


@dataclass(frozen=True)
class Attempt:
    """경로 하나를 시도한 기록.

    근거와 상세까지 시도 단위로 들고 있습니다. 1차 경로가 확인 불가라
    대체 경로로 넘어간 경우, 최종 채택한 쪽의 근거를 써야 하는데
    상태 최상단에만 두면 나중 노드가 앞 노드의 근거를 덮어써 버립니다.
    """

    route: Route
    verdict: Verdict
    reason: str
    """그 도구가 내놓은 사유. 대체 경로로 넘어간 이유가 여기 남습니다."""

    evidence: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """에이전트의 최종 산출물."""

    claim_id: str
    claim_text: str
    verdict: Verdict
    reason: str

    route: Route
    """최종 판정을 낸 경로."""

    route_reason: str = ""
    """왜 그 경로를 골랐는지. 추출기 유형을 뒤집었다면 그 사유가 들어갑니다."""

    declared_type: str = ""
    """추출기가 붙였던 유형. 라우팅이 뒤집었는지 비교하려면 원래 값이 필요합니다."""

    is_forecast: bool = False
    """전망·권고 주장인지. 지표를 두 갈래로 낼 때 분모를 가르는 기준입니다."""

    attempts: tuple[Attempt, ...] = ()
    """거쳐 온 경로들. 마지막 항목이 최종 판정입니다."""

    evidence: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    """도구별 상세 필드(허용 오차, 관측값, 기준 등)를 그대로 담습니다."""

    @property
    def confidence(self) -> Confidence:
        """판정의 성격. 판정이 나지 않았으면 NONE 입니다."""
        if self.verdict is Verdict.UNVERIFIABLE:
            return Confidence.NONE
        return _ROUTE_CONFIDENCE.get(self.route.value, Confidence.INFERRED)

    @property
    def used_fallback(self) -> bool:
        """첫 경로가 실패해서 대체 경로로 판정했는지."""
        return len(self.attempts) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "route": self.route.value,
            "route_reason": self.route_reason,
            "declared_type": self.declared_type,
            "is_forecast": self.is_forecast,
            "confidence": self.confidence.value,
            "used_fallback": self.used_fallback,
            "attempts": [
                {
                    "route": attempt.route.value,
                    "verdict": attempt.verdict.value,
                    "reason": attempt.reason,
                }
                for attempt in self.attempts
            ],
            "evidence": list(self.evidence),
            "detail": dict(self.detail),
        }


class VerifierState(TypedDict, total=False):
    """LangGraph 노드 사이를 오가는 상태.

    TypedDict 를 쓰는 이유: LangGraph 가 노드의 반환값(부분 dict)을 받아
    상태에 합쳐 줍니다. 노드는 자기가 바꾼 키만 돌려주면 됩니다.
    """

    claim_id: str
    claim_text: str
    declared_type: str

    route: str
    route_reason: str
    is_forecast: bool

    verdict: str
    reason: str
    evidence: list[str]
    detail: dict[str, Any]

    attempts: Annotated[list[dict[str, Any]], operator.add]
    """시도 기록.

    `operator.add` 리듀서를 붙인 이유: LangGraph 는 노드가 돌려준 값으로
    상태 키를 **덮어씁니다**. 그대로 두면 대체 경로 노드가 1차 경로의
    기록을 지워 버려서 "무엇을 시도하다 실패했는지"가 사라집니다.
    리듀서를 붙이면 이어 붙습니다.
    """

    allow_fallback: bool
    """1차 경로가 확인 불가일 때 NARRATIVE 로 한 번 더 시도할지."""
