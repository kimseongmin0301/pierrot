"""최종 출력 조립. 원본 답변에 주장별 신뢰 표시를 붙입니다.

    from pierrot.reporter import build_report

    report = build_report(answer, claims, results)
    print(report.annotated_answer())
    print(report.summary.to_dict())
"""

from .report import (
    MARKS,
    ClaimReport,
    Report,
    Summary,
    build_report,
    summarize,
)

__all__ = [
    "MARKS",
    "ClaimReport",
    "Report",
    "Summary",
    "build_report",
    "summarize",
]
