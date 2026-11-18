"""NARRATIVE 주장 판정 (NLI 베이스라인).

주장 하나를 받아 근거 문장들과 하나씩 대조하고, 가장 강한 관계로 판정합니다.
이것이 이 프로젝트의 **베이스라인 ①** 입니다. 도구 기반 검증과 비교하기 위한
기준선이지 최종 판정기가 아닙니다.

## 가장 강한 관계가 이깁니다

근거 문장이 수십 개면 그중 일부는 반드시 거짓 모순을 냅니다.
실측으로 확인한 사례입니다.

    전제: 삼성전자의 2024년 3분기 영업이익은 연결 기준 9조 1,833억원이다.
    가설: 삼성전자의 2024년 3분기 영업이익은 9.18조원이다.
    -> CONTRADICTION 0.808

같은 값인데 표기가 달라 모순으로 판정합니다.

처음에는 이 때문에 "함의를 먼저 보고, 함의가 없을 때만 모순을 본다"로 만들었는데
실제로 돌려 보니 반대 방향의 오류가 났습니다.

    주장: SK하이닉스의 2024년 연간 매출은 전년 대비 31.6% 증가했다 (실제 +102%)
    -> 최대 함의 0.712, 최대 모순 0.968 인데 함의 우선 규칙 때문에 SUPPORTED

그래서 **모든 근거 x 두 관계 중 가장 확률이 높은 하나**로 판정합니다.
표기 차이로 생긴 모순(0.808)은 같은 값을 그대로 적은 근거의 함의(0.99)에 지고,
위 사례처럼 모순이 확실할 때는 모순이 이깁니다.

판정 근거가 된 문장과 두 최고 점수를 모두 남겨 두므로, 왜 그렇게 판정했는지
사람이 확인할 수 있습니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .evidence import Evidence, build_evidence
from .financials import FinancialSnapshot
from .nli import NliLabel, NliModel, NliResult
from .price import PriceSnapshot
from .retrieval import search_by_source
from .universe import Stock, find_all_stocks

DEFAULT_ENTAILMENT_THRESHOLD = 0.7
DEFAULT_CONTRADICTION_THRESHOLD = 0.7
"""판정에 필요한 최소 확률.

낮추면 판정이 늘지만 오탐도 늘어납니다. 이 값이 지표를 좌우하므로 결과에 기록합니다.
"""

DEFAULT_TOP_K = 30
"""NLI 에 넘길 근거 문장 수.

0 이면 검색 없이 전부 넘깁니다.

처음에는 "검색이 약하니 전부 넘기는 것이 공정하다"고 보고 0 으로 돌렸는데
결과가 망가졌습니다. 공시 본문까지 넣으니 주장 하나에 근거가 500개가 되고,
그중 최댓값을 취하니 관련 없는 문장에서 거짓 모순이 쏟아졌습니다.

    주장: 2023년 현대차의 배당금 지급액은 4조 6천억 원이었다
    판정 근거로 뽑힌 문장: "... 이자율스왑 160,000 - 808 파생상품부채 ..."
    -> 모순 0.995

실제 NLI 팩트체킹은 검색으로 추린 뒤 판정합니다. 전부에 돌려 최댓값을 쓰는 것은
표준적인 방식이 아니고, 베이스라인을 공정하게 만들기는커녕 쓰레기를 만듭니다.
그래서 검색을 기본으로 켜고, 끈 결과도 함께 공개합니다.
"""

_YEAR_PATTERN = re.compile(r"(20\d{2})\s*년")


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class NarrativeVerification:
    """판정 결과 하나."""

    verdict: Verdict
    reason: str

    best_entailment: float = 0.0
    best_contradiction: float = 0.0
    best_evidence: str = ""
    """판정 근거가 된 문장."""

    evidence_count: int = 0
    retrieved_count: int = 0
    """검색으로 추려 NLI 에 실제로 넘긴 문장 수.

    판정하지 못했을 때 "검색이 못 찾았는지" 와 "NLI 가 못 맞혔는지" 를
    구분하려면 이 값이 필요합니다.
    """

    stocks: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    threshold: float = 0.0
    evidence: list[str] = field(default_factory=list)


def extract_years(text: str, *, fallback: tuple[int, ...]) -> tuple[int, ...]:
    """주장에서 연도를 찾습니다. 없으면 기본 범위를 씁니다."""
    found = tuple(sorted({int(match) for match in _YEAR_PATTERN.findall(text or "")}))
    return found or fallback


def collect_evidence(
    claim_text: str,
    *,
    financials: FinancialSnapshot | None,
    prices: PriceSnapshot | None,
    documents=None,
    default_years: tuple[int, ...] = (2023, 2024, 2025, 2026),
) -> tuple[list[Evidence], list[Stock], tuple[int, ...]]:
    """주장에 등장하는 회사와 연도로 근거 문장을 모읍니다."""
    stocks = find_all_stocks(claim_text)
    years = extract_years(claim_text, fallback=default_years)

    pool: list[Evidence] = []
    for stock in stocks:
        pool.extend(
            build_evidence(
                stock.code, years,
                financials=financials, prices=prices, documents=documents,
            )
        )
    return pool, stocks, years


def verify_narrative_claim(
    claim_text: str,
    *,
    model: NliModel,
    financials: FinancialSnapshot | None = None,
    prices: PriceSnapshot | None = None,
    documents=None,
    top_k: int = DEFAULT_TOP_K,
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    evidence_pool: list[Evidence] | None = None,
) -> NarrativeVerification:
    """주장 하나를 NLI 로 판정합니다.

    Args:
        evidence_pool: 미리 만들어 둔 근거. 주면 수집 단계를 건너뜁니다.
    """
    if evidence_pool is None:
        pool, stocks, years = collect_evidence(
            claim_text, financials=financials, prices=prices, documents=documents
        )
    else:
        pool = evidence_pool
        stocks = find_all_stocks(claim_text)
        years = extract_years(claim_text, fallback=())

    stock_names = tuple(stock.name for stock in stocks)

    if not stocks:
        return NarrativeVerification(
            Verdict.UNVERIFIABLE,
            "주장에서 유니버스 안의 종목을 찾지 못해 근거를 모을 수 없습니다",
            stocks=stock_names,
            years=years,
        )
    if not pool:
        return NarrativeVerification(
            Verdict.UNVERIFIABLE,
            f"근거 문장을 만들 수 없습니다 (종목 {', '.join(stock_names)}, "
            f"연도 {', '.join(str(year) for year in years) or '미상'})",
            stocks=stock_names,
            years=years,
        )

    # 검색으로 관련 있는 문장만 추립니다. top_k=0 이면 전부 넘깁니다.
    retrieved = len(pool)
    if top_k:
        grouped: dict[str, list[str]] = {}
        for item in pool:
            grouped.setdefault(item.source, []).append(item.text)
        premises = search_by_source(claim_text, grouped, per_source=top_k)
        if not premises:
            premises = [item.text for item in pool[:top_k]]
        retrieved = len(premises)
    else:
        premises = [item.text for item in pool]

    results: list[NliResult] = model.predict_batch(
        [(premise, claim_text) for premise in premises]
    )

    best_entail = max(results, key=lambda r: r.score(NliLabel.ENTAILMENT))
    best_contra = max(results, key=lambda r: r.score(NliLabel.CONTRADICTION))
    entail_score = best_entail.score(NliLabel.ENTAILMENT)
    contra_score = best_contra.score(NliLabel.CONTRADICTION)

    common = dict(
        best_entailment=entail_score,
        best_contradiction=contra_score,
        evidence_count=len(pool),
        retrieved_count=retrieved,
        stocks=stock_names,
        years=years,
    )

    # 가장 강한 관계가 이깁니다. 함의와 모순 중 확률이 높은 쪽을 택합니다.
    entail_wins = entail_score >= contra_score

    if entail_wins and entail_score >= entailment_threshold:
        return NarrativeVerification(
            Verdict.SUPPORTED,
            f"근거 문장과 함의 관계입니다 (함의 {entail_score:.3f} > "
            f"모순 {contra_score:.3f}). 근거: {best_entail.premise}",
            best_evidence=best_entail.premise,
            threshold=entailment_threshold,
            evidence=[best_entail.premise],
            **common,
        )

    if not entail_wins and contra_score >= contradiction_threshold:
        return NarrativeVerification(
            Verdict.CONTRADICTED,
            f"근거 문장과 모순입니다 (모순 {contra_score:.3f} > "
            f"함의 {entail_score:.3f}). 근거: {best_contra.premise}",
            best_evidence=best_contra.premise,
            threshold=contradiction_threshold,
            evidence=[best_contra.premise],
            **common,
        )

    return NarrativeVerification(
        Verdict.UNVERIFIABLE,
        f"근거 {len(pool)}개 중 {retrieved}개를 검토했으나 "
        f"기준({entailment_threshold:.2f})을 넘는 관계가 "
        f"없습니다 (최대 함의 {entail_score:.3f}, 최대 모순 {contra_score:.3f})",
        best_evidence=best_entail.premise,
        threshold=entailment_threshold,
        evidence=[best_entail.premise, best_contra.premise],
        **common,
    )
