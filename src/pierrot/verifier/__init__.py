"""검증 에이전트. 주장 유형에 맞는 도구를 선택해 판정합니다 (LangGraph).

쓰는 법:

    from pierrot.verifier import build_graph, verify_claim

    graph = build_graph()            # 스냅샷·모델을 한 번만 올립니다
    result = verify_claim("삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다",
                          graph=graph)
    print(result.verdict, result.reason)

여러 건을 돌릴 때는 `graph` 를 반드시 재사용하십시오. 넘기지 않으면
주장마다 parquet 스냅샷을 다시 읽습니다.
"""

from .graph import build_graph, to_result, verify_claim
from .routing import classify, is_forecast
from .state import (
    Attempt,
    Confidence,
    Route,
    VerificationResult,
    Verdict,
    VerifierState,
)
from .tools import VerifierTools

__all__ = [
    "Attempt",
    "Confidence",
    "Route",
    "VerificationResult",
    "Verdict",
    "VerifierState",
    "VerifierTools",
    "build_graph",
    "classify",
    "is_forecast",
    "to_result",
    "verify_claim",
]
