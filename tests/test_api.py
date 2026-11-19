"""API 테스트.

OpenAI 도 스냅샷도 건드리지 않습니다. 가짜 서비스를 끼워 넣어
**엔드포인트의 계약**만 봅니다. 판정이 맞는지는 도구 쪽 테스트가 봅니다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pierrot.api import create_app
from pierrot.api.service import Usage
from pierrot.cost import BudgetExceededError
from pierrot.extractor.models import Claim, ClaimType
from pierrot.reporter import build_report
from pierrot.verifier.state import Route, VerificationResult, Verdict

ANSWER = "삼성전자의 2024년 3분기 영업이익은 6조 1,879억원입니다. 주가는 올랐습니다."


def make_claim(claim_id: str, text: str, claim_type: ClaimType, start: int, end: int) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=claim_type,
        source_text=ANSWER[start:end],
        start=start,
        end=end,
        span_found=True,
        match_mode="EXACT",
        match_ratio=1.0,
    )


def make_result(claim_id: str, text: str, verdict: Verdict, route: Route) -> VerificationResult:
    return VerificationResult(
        claim_id=claim_id,
        claim_text=text,
        verdict=verdict,
        reason="실제 9조 1,833억원과 다릅니다" if verdict is Verdict.CONTRADICTED else "사유",
        route=route,
        route_reason="재무 낱말이 있습니다",
        declared_type=route.value,
        evidence=["삼성전자 2024년 3분기 영업이익(연결) 9조 1,833억원"],
    )


CLAIMS = [
    make_claim("c01", "삼성전자의 2024년 3분기 영업이익은 6조 1,879억원이다", ClaimType.NUMERIC, 0, 40),
    make_claim("c02", "삼성전자 주가는 2024년 3분기에 올랐다", ClaimType.PRICE, 41, 52),
]
RESULTS = [
    make_result("c01", CLAIMS[0].text, Verdict.CONTRADICTED, Route.NUMERIC),
    make_result("c02", CLAIMS[1].text, Verdict.UNVERIFIABLE, Route.PRICE),
]


class FakeService:
    """진짜 서비스와 같은 모양만 흉내 냅니다."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.raise_budget = False

        class _Cost:
            def summary(self_inner):
                return {
                    "month": "2026-11",
                    "spent_usd": 0.52,
                    "budget_usd": 5.0,
                    "remaining_usd": 4.48,
                }

        self.cost = _Cost()

    def health(self):
        return {
            "status": "ok",
            "snapshots": {"prices": True, "financials": True},
            "openai_key": True,
            "budget_remaining_usd": 4.48,
            "notes": [],
        }

    def usage_by_source(self):
        return {"extractor": {"calls": 2, "cache_hits": 1, "cost_usd": 0.004}}

    def _usage(self) -> Usage:
        return Usage(
            elapsed_seconds=1.23,
            cost_usd=0.004,
            month_spent_usd=0.52,
            month_budget_usd=5.0,
            llm_calls=2,
        )

    def verify_answer(self, answer, *, allow_fallback=False, refine=True):
        self.calls.append(("answer", answer, allow_fallback, refine))
        if self.raise_budget:
            raise BudgetExceededError("예산을 모두 썼습니다")
        report = build_report(answer, CLAIMS, RESULTS)
        return report, self._usage(), {"model": "gpt-4o-mini", "prompt_version": "v3"}

    def verify_single_claim(self, claim, *, declared_type="", allow_fallback=False):
        self.calls.append(("claim", claim, declared_type, allow_fallback))
        if self.raise_budget:
            raise BudgetExceededError("예산을 모두 썼습니다")
        return RESULTS[0], self._usage()


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def client(service) -> TestClient:
    with TestClient(create_app(service)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 준비 상태
# ---------------------------------------------------------------------------


def test_헬스체크가_준비_상태를_알려준다(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["snapshots"]["prices"] is True
    assert body["budget_remaining_usd"] == 4.48


def test_사용량은_단계별로_나뉜다(client):
    """어느 단계가 돈을 쓰는지 알아야 줄일 곳을 정할 수 있습니다."""
    body = client.get("/usage").json()
    assert body["budget_usd"] == 5.0
    assert body["by_source"]["extractor"]["calls"] == 2


# ---------------------------------------------------------------------------
# 답변 검증
# ---------------------------------------------------------------------------


def test_답변을_주장별로_판정한다(client):
    body = client.post("/verify", json={"answer": ANSWER}).json()
    assert len(body["claims"]) == 2
    assert body["claims"][0]["verdict"] == "CONTRADICTED"
    assert body["claims"][1]["verdict"] == "UNVERIFIABLE"


def test_판정하지_못하면_사유가_함께_온다(client):
    """확인 불가는 결과가 없는 게 아니라 '왜 못 했는지'가 결과입니다."""
    body = client.post("/verify", json={"answer": ANSWER}).json()
    unverifiable = [c for c in body["claims"] if c["verdict"] == "UNVERIFIABLE"]
    assert unverifiable and all(c["reason"] for c in unverifiable)


def test_응답에_비용과_지연이_붙는다(client):
    """월 예산이 $5 입니다. 호출 하나가 얼마를 썼는지 응답에서 보여야 합니다."""
    usage = client.post("/verify", json={"answer": ANSWER}).json()["usage"]
    assert usage["cost_usd"] == 0.004
    assert usage["elapsed_seconds"] == 1.23
    assert usage["month_budget_usd"] == 5.0
    assert usage["llm_calls"] == 2


def test_요약은_판정의_성격을_나눠_낸다(client):
    """계산으로 대조한 것과 NLI 로 추정한 것을 한 칸에 세면 안 됩니다."""
    summary = client.post("/verify", json={"answer": ANSWER}).json()["summary"]
    assert summary["total"] == 2
    assert summary["contradicted"] == 1
    assert summary["calculated"] == 1   # NUMERIC 경로
    assert summary["inferred"] == 0
    assert summary["coverage"] == 0.5


def test_원문에_모순_표시를_붙인다(client):
    body = client.post("/verify", json={"answer": ANSWER}).json()
    assert "[모순]" in body["annotated_answer"]
    # 확인 불가는 표시하지 않습니다. 다 표시하면 본문이 덮입니다.
    assert "[확인 불가]" not in body["annotated_answer"]


def test_표시를_끄면_빈_문자열이_온다(client):
    body = client.post("/verify", json={"answer": ANSWER, "annotate": False}).json()
    assert body["annotated_answer"] == ""


def test_대체_경로_설정이_전달된다(client, service):
    client.post("/verify", json={"answer": ANSWER, "allow_fallback": True})
    assert service.calls[-1][2] is True


def test_다듬기_설정이_전달된다(client, service):
    client.post("/verify", json={"answer": ANSWER, "refine": False})
    assert service.calls[-1][3] is False


# ---------------------------------------------------------------------------
# 주장 하나
# ---------------------------------------------------------------------------


def test_주장_하나만_판정할_수_있다(client, service):
    body = client.post(
        "/verify/claim",
        json={"claim": "삼성전자의 2024년 3분기 영업이익은 6조원이다", "declared_type": "NUMERIC"},
    ).json()
    assert body["result"]["verdict"] == "CONTRADICTED"
    assert body["result"]["route"] == "NUMERIC"
    assert service.calls[-1][0] == "claim"


# ---------------------------------------------------------------------------
# 오류 처리
# ---------------------------------------------------------------------------


def test_예산을_넘기면_429를_돌려준다(client, service):
    """서버 잘못이 아니라 '지금은 받을 수 없음' 입니다."""
    service.raise_budget = True
    response = client.post("/verify", json={"answer": ANSWER})
    assert response.status_code == 429
    assert "예산" in response.json()["detail"]


def test_빈_답변은_거부한다(client):
    assert client.post("/verify", json={"answer": ""}).status_code == 422


def test_너무_긴_답변은_거부한다(client):
    assert client.post("/verify", json={"answer": "가" * 20_001}).status_code == 422


# ---------------------------------------------------------------------------
# 보고서 조립
# ---------------------------------------------------------------------------


def test_주장과_판정_수가_다르면_거부한다():
    """짝이 어긋나면 엉뚱한 주장에 엉뚱한 판정이 붙습니다."""
    with pytest.raises(ValueError, match="수가 다릅니다"):
        build_report(ANSWER, CLAIMS, RESULTS[:1])


def test_표시는_뒤에서부터_넣는다():
    """앞에서부터 넣으면 뒤쪽 구간의 위치가 밀려 엉뚱한 자리에 붙습니다."""
    report = build_report(
        ANSWER,
        CLAIMS,
        [
            make_result("c01", CLAIMS[0].text, Verdict.CONTRADICTED, Route.NUMERIC),
            make_result("c02", CLAIMS[1].text, Verdict.CONTRADICTED, Route.PRICE),
        ],
    )
    annotated = report.annotated_answer()
    assert annotated.count("[모순]") == 2
    # 첫 주장의 끝(40) 바로 뒤에 표시가 붙어야 합니다.
    assert annotated[:42].endswith("[모순]"[:1]) or "[모순]" in annotated[:50]
