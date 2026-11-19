"""검증 에이전트 그래프 테스트.

API 를 호출하지 않습니다. 파서와 판정 함수를 가짜로 바꿔 끼워서
**그래프의 흐름**만 봅니다. 흐름이 맞는지와 도구가 맞는지는 다른 문제이고,
도구 쪽은 이미 각자 테스트가 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pierrot.verifier import VerifierTools, build_graph, verify_claim
from pierrot.verifier.graph import _route_after_classify, _route_after_tool
from pierrot.verifier.state import Route, Verdict


# ---------------------------------------------------------------------------
# 가짜 도구
# ---------------------------------------------------------------------------


@dataclass
class FakeQuery:
    text: str = ""

    def describe(self) -> str:
        return f"가짜 조회: {self.text}"


@dataclass
class FakeVerdict:
    value: str


@dataclass
class FakeResult:
    """세 도구의 결과 dataclass 가 공통으로 갖는 필드만 흉내 냅니다."""

    verdict: FakeVerdict
    reason: str
    evidence: list[str] = field(default_factory=list)

    claimed: float | None = None
    observed: float | None = None
    claimed_low: float | None = None
    claimed_high: float | None = None
    observed_low: float | None = None
    observed_high: float | None = None
    unit: str = ""
    tolerance: float = 0.0
    matched_fs: str = ""
    fs_explicit: bool = False
    price_basis: str = ""
    trading_days: int = 0
    window = None
    precision = FakeVerdict("DAY")

    best_entailment: float = 0.0
    best_contradiction: float = 0.0
    best_evidence: str = ""
    evidence_count: int = 0
    retrieved_count: int = 0
    stocks: tuple = ()
    years: tuple = ()
    threshold: float = 0.0


def _patch(monkeypatch, *, price=None, numeric=None, narrative=None):
    """세 도구의 파서·판정 함수를 가짜로 바꿔 끼웁니다.

    노드가 함수 안에서 import 하므로 원본 모듈을 패치하면 걸립니다.
    """
    import pierrot.tools.narrative_verify as nv
    import pierrot.tools.numeric_claim as nc
    import pierrot.tools.numeric_verify as nver
    import pierrot.tools.price_claim as pc
    import pierrot.tools.price_verify as pver

    monkeypatch.setattr(pc, "parse_price_claim", lambda text, **kw: FakeQuery(text))
    monkeypatch.setattr(nc, "parse_numeric_claim", lambda text, **kw: FakeQuery(text))

    if price is not None:
        monkeypatch.setattr(pver, "verify_price_claim", lambda q, s, **kw: price)
    if numeric is not None:
        monkeypatch.setattr(nver, "verify_numeric_claim", lambda q, s: numeric)
    if narrative is not None:
        monkeypatch.setattr(nv, "verify_narrative_claim", lambda text, **kw: narrative)


@pytest.fixture
def tools(monkeypatch):
    """스냅샷을 읽지 않는 도구 묶음."""
    container = VerifierTools()
    monkeypatch.setattr(type(container), "prices", lambda self: object())
    monkeypatch.setattr(type(container), "financials", lambda self: object())
    monkeypatch.setattr(type(container), "documents", lambda self: None)
    monkeypatch.setattr(type(container), "nli", lambda self: object())
    monkeypatch.setattr(type(container), "price_parser", lambda self: object())
    monkeypatch.setattr(type(container), "numeric_parser", lambda self: object())
    return container


# ---------------------------------------------------------------------------
# 분기 함수 (순수 함수라 그래프 없이 봅니다)
# ---------------------------------------------------------------------------


def test_분류_결과대로_노드를_고른다():
    assert _route_after_classify({"route": "PRICE"}) == "price"
    assert _route_after_classify({"route": "NUMERIC"}) == "numeric"
    assert _route_after_classify({"route": "FORECAST"}) == "forecast"
    assert _route_after_classify({"route": ""}) == "narrative"


def test_대체_경로는_기본으로_꺼져_있다():
    state = {"attempts": [{"route": "NUMERIC", "verdict": "UNVERIFIABLE", "reason": ""}]}
    assert _route_after_tool(state) == "finalize"


def test_확인_불가면_대체_경로로_넘어간다():
    state = {
        "allow_fallback": True,
        "attempts": [{"route": "NUMERIC", "verdict": "UNVERIFIABLE", "reason": ""}],
    }
    assert _route_after_tool(state) == "narrative"


def test_판정이_났으면_대체_경로로_가지_않는다():
    state = {
        "allow_fallback": True,
        "attempts": [{"route": "NUMERIC", "verdict": "CONTRADICTED", "reason": ""}],
    }
    assert _route_after_tool(state) == "finalize"


def test_의미_비교를_이미_거쳤으면_또_가지_않는다():
    """무한 루프 방지. 이 분기가 없으면 narrative 를 계속 다시 탑니다."""
    state = {
        "allow_fallback": True,
        "attempts": [
            {"route": "NUMERIC", "verdict": "UNVERIFIABLE", "reason": ""},
            {"route": "NARRATIVE", "verdict": "UNVERIFIABLE", "reason": ""},
        ],
    }
    assert _route_after_tool(state) == "finalize"


# ---------------------------------------------------------------------------
# 지연 로딩 회귀 고정
# ---------------------------------------------------------------------------


def test_그래프를_컴파일해도_모델과_스냅샷이_올라오지_않는다():
    """LangGraph 가 노드 클로저의 속성을 컴파일 시점에 getattr 합니다.

    접근자를 `@property` 로 두면 그래프를 만드는 것만으로 NLI 모델과
    parquet 스냅샷이 전부 올라옵니다. 실제로 그렇게 만들었다가 발견했습니다.
    메서드로 바꿔서 막았고, 이 테스트가 그 상태를 고정합니다.
    """
    container = VerifierTools()
    build_graph(container)
    assert not any(container.loaded().values()), (
        f"컴파일만 했는데 올라온 것이 있습니다: {container.loaded()}"
    )


# ---------------------------------------------------------------------------
# 전체 흐름
# ---------------------------------------------------------------------------


def test_주가_주장은_주가_경로로_판정된다(monkeypatch, tools):
    _patch(monkeypatch, price=FakeResult(FakeVerdict("CONTRADICTED"), "실제 종가와 다릅니다"))
    graph = build_graph(tools)

    result = verify_claim(
        "삼성전자 주가는 2024년 4월 9일 종가 기준 71,800원이다", graph=graph
    )
    assert result.route is Route.PRICE
    assert result.verdict is Verdict.CONTRADICTED
    assert result.reason == "실제 종가와 다릅니다"
    assert not result.used_fallback


def test_전망_주장은_도구를_타지_않고_사유가_남는다(monkeypatch, tools):
    _patch(monkeypatch, numeric=FakeResult(FakeVerdict("SUPPORTED"), "맞습니다"))
    graph = build_graph(tools)

    result = verify_claim(
        "2024년 현대차의 영업이익은 약 1.3조 원 수준으로 예상된다", graph=graph
    )
    assert result.route is Route.FORECAST
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.is_forecast
    assert "전망" in result.reason
    # 도구를 타지 않았으므로 시도 기록은 forecast 하나뿐입니다.
    assert [a.route for a in result.attempts] == [Route.FORECAST]


def test_대체_경로를_켜면_확인_불가를_의미_비교가_받는다(monkeypatch, tools):
    _patch(
        monkeypatch,
        numeric=FakeResult(FakeVerdict("UNVERIFIABLE"), "재무제표 주요계정에 없습니다"),
        narrative=FakeResult(FakeVerdict("CONTRADICTED"), "근거 문장과 모순입니다"),
    )
    graph = build_graph(tools)

    result = verify_claim(
        "카카오의 광고 사업 부문 매출은 1,509억 원이다",
        graph=graph,
        allow_fallback=True,
    )
    assert result.used_fallback
    assert [a.route for a in result.attempts] == [Route.NUMERIC, Route.NARRATIVE]
    assert result.verdict is Verdict.CONTRADICTED
    assert result.route is Route.NARRATIVE


def test_둘_다_확인_불가면_첫_시도의_사유를_남긴다(monkeypatch, tools):
    """1차 사유가 더 구체적입니다. 대체 경로 사유로 덮으면 진단이 어려워집니다."""
    _patch(
        monkeypatch,
        numeric=FakeResult(FakeVerdict("UNVERIFIABLE"), "재무제표 주요계정에 없습니다"),
        narrative=FakeResult(FakeVerdict("UNVERIFIABLE"), "기준을 넘는 관계가 없습니다"),
    )
    graph = build_graph(tools)

    result = verify_claim(
        "카카오의 광고 사업 부문 매출은 1,509억 원이다",
        graph=graph,
        allow_fallback=True,
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.reason == "재무제표 주요계정에 없습니다"
    assert len(result.attempts) == 2


def test_도구가_터져도_파이프라인은_계속된다(monkeypatch, tools):
    import pierrot.tools.price_claim as pc

    monkeypatch.setattr(
        pc, "parse_price_claim", lambda text, **kw: (_ for _ in ()).throw(RuntimeError("파서 실패"))
    )
    graph = build_graph(tools)

    result = verify_claim("삼성전자 주가는 2024년 4월 9일 종가 기준 71,800원이다", graph=graph)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "파서 실패" in result.reason


def test_결과를_dict_로_직렬화할_수_있다(monkeypatch, tools):
    _patch(monkeypatch, price=FakeResult(FakeVerdict("SUPPORTED"), "일치합니다"))
    graph = build_graph(tools)

    result = verify_claim(
        "삼성전자 주가는 2024년 4월 9일 종가 기준 83,600원이다",
        claim_id="c1",
        declared_type="PRICE",
        graph=graph,
    )
    payload = result.to_dict()
    assert payload["claim_id"] == "c1"
    assert payload["verdict"] == "SUPPORTED"
    assert payload["route"] == "PRICE"
    assert payload["attempts"][0]["route"] == "PRICE"
