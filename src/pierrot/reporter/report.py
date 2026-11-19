"""검증 결과를 사람이 읽을 수 있는 보고서로 조립합니다.

주장별 판정만 늘어놓으면 "그래서 이 답변을 믿어도 되는가"에 답하지 못합니다.
원본 답변의 **어느 자리**가 문제인지 함께 보여야 합니다.
1단계에서 주장마다 원문 구간(span)을 기록해 둔 것이 여기서 쓰입니다.

세 가지를 담습니다.

1. 주장별 판정 — 무엇을, 어떤 경로로, 왜 그렇게 판정했는지
2. 원본 답변에 표시 — 모순으로 판정된 구간을 원문 위에 표시
3. 요약 — 판정 분포와 커버리지. **판정의 성격(계산/추정)을 나눠서** 냅니다

왜 성격을 나누는가: 5단계 실측에서 도구 경로가 잡은 모순은 실제 수치와의
차이를 숫자로 제시했지만, NLI 경로가 잡은 2건은 근거 문장이 주장과
무관했습니다. 둘을 한 칸에 세면 "모순 41건"이 되지만 그중 2건은 믿을 수
없습니다. 보고서를 읽는 사람이 그 차이를 알아야 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..extractor.models import Claim
from ..verifier.state import Confidence, Route, VerificationResult, Verdict

# 원문에 표시할 때 쓰는 기호. 터미널에서도 읽히도록 기호만 씁니다.
MARKS: dict[Verdict, str] = {
    Verdict.SUPPORTED: "[근거 있음]",
    Verdict.CONTRADICTED: "[모순]",
    Verdict.UNVERIFIABLE: "[확인 불가]",
}


@dataclass(frozen=True)
class ClaimReport:
    """주장 하나에 대한 보고."""

    claim_id: str
    text: str
    claim_type: str
    """추출기가 붙인 유형."""

    route: str
    """에이전트가 실제로 고른 경로. 유형과 다르면 뒤집힌 것입니다."""

    verdict: str
    confidence: str
    reason: str
    evidence: list[str] = field(default_factory=list)

    start: int | None = None
    end: int | None = None
    """원문 구간. 없으면 원문에서 찾지 못한 것입니다."""

    is_forecast: bool = False
    route_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "route": self.route,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "start": self.start,
            "end": self.end,
            "is_forecast": self.is_forecast,
            "route_changed": self.route_changed,
        }


@dataclass(frozen=True)
class Summary:
    """판정 분포와 커버리지."""

    total: int
    supported: int
    contradicted: int
    unverifiable: int

    forecast: int
    """전망·권고 주장. 거르지 않고 흘려보낸 것들입니다."""

    calculated: int
    """공시·주가 수치와 직접 대조해 판정한 건수."""

    inferred: int
    """의미 비교(NLI)로 추정한 건수. 그대로 믿으면 안 됩니다."""

    @property
    def judged(self) -> int:
        return self.supported + self.contradicted

    @property
    def coverage(self) -> float:
        """판정이 난 비율. 전체 기준입니다."""
        return self.judged / self.total if self.total else 0.0

    @property
    def coverage_excluding_forecast(self) -> float:
        """전망 주장을 분모에서 뺀 커버리지.

        전망은 애초에 검증 대상이 아닙니다. 두 숫자를 함께 내야
        "검증 못 한 것"과 "검증 대상이 아닌 것"이 섞이지 않습니다.
        """
        denominator = self.total - self.forecast
        return self.judged / denominator if denominator else 0.0

    @property
    def calculated_coverage(self) -> float:
        """계산으로 대조해 판정한 비율. 가장 믿을 수 있는 숫자입니다."""
        return self.calculated / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "supported": self.supported,
            "contradicted": self.contradicted,
            "unverifiable": self.unverifiable,
            "forecast": self.forecast,
            "calculated": self.calculated,
            "inferred": self.inferred,
            "judged": self.judged,
            "coverage": round(self.coverage, 4),
            "coverage_excluding_forecast": round(self.coverage_excluding_forecast, 4),
            "calculated_coverage": round(self.calculated_coverage, 4),
        }


@dataclass(frozen=True)
class Report:
    """답변 하나에 대한 검증 보고서."""

    answer: str
    claims: list[ClaimReport]
    summary: Summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "summary": self.summary.to_dict(),
        }

    def annotated_answer(self, *, only_flagged: bool = True) -> str:
        """원본 답변에 판정 표시를 붙입니다.

        Args:
            only_flagged: 모순으로 판정된 것만 표시합니다. 전부 표시하면
                표시가 본문을 덮어 읽기 어려워집니다.
        """
        marked = [
            claim
            for claim in self.claims
            if claim.start is not None
            and claim.end is not None
            and (not only_flagged or claim.verdict == Verdict.CONTRADICTED.value)
        ]
        if not marked:
            return self.answer

        # 뒤에서부터 끼워 넣습니다. 앞에서부터 하면 뒤쪽 구간의
        # 위치가 밀려 엉뚱한 자리에 표시됩니다.
        text = self.answer
        for claim in sorted(marked, key=lambda c: c.end or 0, reverse=True):
            mark = MARKS.get(Verdict(claim.verdict), "")
            text = text[: claim.end] + f" {mark}" + text[claim.end :]
        return text


def build_report(
    answer: str,
    claims: list[Claim],
    results: list[VerificationResult],
) -> Report:
    """주장 목록과 판정 결과를 합쳐 보고서를 만듭니다.

    주장과 결과는 **같은 순서**여야 합니다. 짝이 어긋나면 엉뚱한 주장에
    엉뚱한 판정이 붙으므로, 길이가 다르면 거부합니다.
    """
    if len(claims) != len(results):
        raise ValueError(
            f"주장과 판정 수가 다릅니다: 주장 {len(claims)}개, 판정 {len(results)}개"
        )

    reports: list[ClaimReport] = []
    for claim, result in zip(claims, results):
        claim_type = (
            claim.claim_type.value
            if hasattr(claim.claim_type, "value")
            else str(claim.claim_type)
        )
        reports.append(
            ClaimReport(
                claim_id=claim.claim_id,
                text=result.claim_text or claim.text,
                claim_type=claim_type,
                route=result.route.value,
                verdict=result.verdict.value,
                confidence=result.confidence.value,
                reason=result.reason,
                evidence=list(result.evidence),
                start=claim.start if claim.span_found else None,
                end=claim.end if claim.span_found else None,
                is_forecast=result.is_forecast,
                route_changed=claim_type != result.route.value
                and result.route is not Route.FORECAST,
            )
        )

    return Report(answer=answer, claims=reports, summary=summarize(results))


def summarize(results: list[VerificationResult]) -> Summary:
    """판정 분포를 셉니다."""

    def count(verdict: Verdict) -> int:
        return sum(1 for result in results if result.verdict is verdict)

    return Summary(
        total=len(results),
        supported=count(Verdict.SUPPORTED),
        contradicted=count(Verdict.CONTRADICTED),
        unverifiable=count(Verdict.UNVERIFIABLE),
        forecast=sum(1 for result in results if result.is_forecast),
        calculated=sum(
            1 for result in results if result.confidence is Confidence.CALCULATED
        ),
        inferred=sum(
            1 for result in results if result.confidence is Confidence.INFERRED
        ),
    )
