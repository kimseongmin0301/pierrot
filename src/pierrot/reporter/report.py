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

import re
from dataclasses import dataclass, field
from typing import Any

from ..extractor.models import Claim
from ..extractor.quality import looks_like_forecast
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
class UncoveredSentence:
    """주장이 하나도 나오지 않은 문장.

    분해 단계는 전망·권고 문장을 **일부러 버립니다.** 지금 참·거짓을 가릴
    수 없기 때문입니다. 그 자체는 옳지만, 버린 사실을 아무 데도 적지 않으면
    읽는 사람은 그 문장이 **검사를 통과한 것인지 아예 보지 않은 것인지**
    구분할 수 없습니다.

    실측(가짜 데이터셋 06번):

        "현대차의 2025년 연결 매출은 전년보다 증가했습니다.
         2027년 연결 매출도 2026년보다 늘어날 전망입니다."

    두 문장인데 주장은 하나만 나왔습니다. 두 번째 문장은 흔적도 없이
    사라졌습니다. 검증기가 할 말은 "보지 않았다" 가 아니라
    **"전망이라 지금은 가릴 수 없다"** 입니다.
    """

    text: str
    start: int
    end: int
    why: str
    """왜 주장이 안 나왔는지. '전망·권고로 보입니다' 또는 '분해되지 않았습니다'."""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end, "why": self.why}


@dataclass(frozen=True)
class Report:
    """답변 하나에 대한 검증 보고서."""

    answer: str
    claims: list[ClaimReport]
    summary: Summary
    uncovered: list[UncoveredSentence] = field(default_factory=list)
    """주장이 하나도 나오지 않은 문장들. 비어 있으면 모든 문장이 다뤄졌습니다."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "summary": self.summary.to_dict(),
            "uncovered": [item.to_dict() for item in self.uncovered],
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

    return Report(
        answer=answer,
        claims=reports,
        summary=summarize(results),
        uncovered=find_uncovered(answer, reports),
    )


# 문장 끝. 한국어 종결 부호 뒤에 공백이나 끝이 오는 자리에서 자릅니다.
_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")


def find_uncovered(answer: str, claims: list[ClaimReport]) -> list[UncoveredSentence]:
    """주장이 하나도 걸치지 않은 문장을 찾습니다.

    주장마다 원문 위치(start, end)가 붙어 있으므로, 어느 주장과도 겹치지
    않는 문장은 **분해 단계가 버린 문장**입니다. 대부분 전망·권고입니다.

    위치를 못 찾은 주장(span_found=False)이 하나라도 있으면 빈 목록을
    돌려줍니다. 위치를 모르는 주장이 그 문장을 다뤘을 수도 있어, 다루지
    않았다고 단정하면 거짓말이 되기 때문입니다.
    """
    text = answer or ""
    if not text.strip():
        return []
    if any(claim.start is None or claim.end is None for claim in claims):
        return []

    spans = [(claim.start, claim.end) for claim in claims]

    found: list[UncoveredSentence] = []
    offset = 0
    for piece in _SENTENCE_END.split(text):
        start = text.find(piece, offset)
        if start < 0:
            start = offset
        end = start + len(piece)
        offset = end
        if not piece.strip():
            continue
        for chunk_start, chunk_end in _uncovered_chunks(text, start, end, spans):
            chunk = text[chunk_start:chunk_end]
            if not _worth_reporting(chunk):
                continue
            why = (
                "전망·권고로 보입니다. 지금 참·거짓을 가릴 수 없습니다"
                if looks_like_forecast(chunk)
                else "주장으로 분해되지 않았습니다"
            )
            found.append(
                UncoveredSentence(
                    text=_TRIM.sub("", chunk), start=chunk_start, end=chunk_end, why=why
                )
            )
    return found


# 문장 안에서 절을 가르는 자리. 쉼표는 넣지 않습니다 — "32,300억" 이 부서집니다.
_CLAUSE_SPLIT = re.compile(
    r"(?:으며|하며|이며|했으며|였으며|으나|이나|지만|하나|했으나|하였으나|하고|했고)\s*"
    r"|(?:반면|하지만|다만|또한)\s"
)

# 절 하나로 볼 최소 길이. 조각이 너무 짧으면 잡음입니다.
_MIN_CHUNK = 8

# 조각 양 끝의 공백과 문장 부호. 이것만 남은 조각은 보여 줄 가치가 없습니다.
_TRIM = re.compile(r"^[\s.,·…]+|[\s.,·…]+$")


def _worth_reporting(chunk: str) -> bool:
    """사람에게 보여 줄 만한 조각인지. 토막 부스러기는 거릅니다."""
    return len(_TRIM.sub("", chunk)) >= _MIN_CHUNK


def _uncovered_chunks(
    text: str, start: int, end: int, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """문장 [start, end) 안에서 **어느 주장도 덮지 않은 구간**을 돌려줍니다.

    전에는 문장 단위로만 봤습니다. 그래서 문장 앞쪽 절에 주장이 하나라도
    붙으면 **뒤쪽 절이 통째로 사라져도 알 수 없었습니다.**

    실측(가상 샘플 100건, 009 등 7건):

        "비용 절감 덕분에 전년보다 동일했으며, 2027년에는 역대 최대 이익을 달성할 것입니다."
         └─ 주장으로 뽑힌 구간 ─┘  └──── 흔적 없이 사라진 절 ────┘

    앞 절에 주장이 붙어 문장이 '다뤄진 것' 으로 보였고, 뒤 절의 미래 단정은
    주장 목록에도 uncovered 에도 남지 않았습니다. 읽는 사람은 그 주장이
    검사를 통과한 줄 압니다.
    """
    inside = sorted(
        (max(s, start), min(e, end))
        for s, e in spans
        if s < end and start < e
    )
    if not inside:
        return [(start, end)]

    # 주장이 덮은 구간을 뺀 나머지가 '빈 곳' 입니다.
    gaps: list[tuple[int, int]] = []
    cursor = start
    for span_start, span_end in inside:
        if span_start > cursor:
            gaps.append((cursor, span_start))
        cursor = max(cursor, span_end)
    if cursor < end:
        gaps.append((cursor, end))

    # 빈 곳을 절 경계로 다듬습니다. 이음말이 앞에 남으면 읽기 나쁩니다.
    trimmed: list[tuple[int, int]] = []
    for gap_start, gap_end in gaps:
        chunk = text[gap_start:gap_end]
        offset = 0
        for match in _CLAUSE_SPLIT.finditer(chunk):
            if match.start() <= 2:          # 조각 맨 앞의 이음말만 떼어 냅니다
                offset = match.end()
        trimmed.append((gap_start + offset, gap_end))
    return trimmed


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
