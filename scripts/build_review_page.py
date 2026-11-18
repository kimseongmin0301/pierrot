"""분해 결과를 사람이 검토할 수 있는 HTML 페이지로 만듭니다.

터미널 출력으로는 "답변의 어느 부분이 주장으로 뽑히지 않았는가"를 보기 어렵습니다.
주장이 커버한 구간을 원문 위에 직접 표시해야 누락이 눈에 들어옵니다.

실행:
    .venv/Scripts/python.exe scripts/build_review_page.py
    .venv/Scripts/python.exe scripts/build_review_page.py --out build/review.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CLAIMS_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_claims.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "build" / "review.html"

TYPE_TOOL = {
    "NUMERIC": "DART 공시로 검증",
    "PRICE": "주가 스냅샷으로 검증",
    "NARRATIVE": "근거 문서 대조(NLI)로 검증",
}

MATCH_LABEL = {
    "EXACT": "원문 그대로",
    "WHITESPACE": "공백 차이",
    "NORMALIZED": "표기 차이",
    "FUZZY": "유사도 매칭",
    "NOT_FOUND": "원문에 없음",
}


def build_segments(answer: str, claims: list[dict]) -> list[dict]:
    """원문을 "같은 주장 집합이 덮고 있는 구간" 단위로 쪼갭니다.

    주장 두 개가 같은 문장에서 나오는 일이 흔하므로 구간이 겹칩니다.
    글자마다 어떤 주장이 덮고 있는지 기록한 뒤, 덮개가 바뀌는 지점에서 끊습니다.
    """
    length = len(answer)
    owners: list[list[str]] = [[] for _ in range(length)]
    type_of: dict[str, str] = {}

    for claim in claims:
        type_of[claim["claim_id"]] = claim["claim_type"]
        start, end = claim.get("start"), claim.get("end")
        if start is None or end is None:
            continue
        for index in range(max(0, start), min(end, length)):
            owners[index].append(claim["claim_id"])

    segments: list[dict] = []
    current: tuple[str, ...] | None = None
    start_index = 0

    def flush(end_index: int) -> None:
        if current is None or end_index <= start_index:
            return
        ids = list(current)
        types = {type_of[i] for i in ids}
        segments.append(
            {
                "text": answer[start_index:end_index],
                "claims": ids,
                "type": types.pop() if len(types) == 1 else ("multi" if types else ""),
            }
        )

    for index in range(length):
        key = tuple(owners[index])
        if current is None:
            current, start_index = key, index
        elif key != current:
            flush(index)
            current, start_index = key, index
    flush(length)
    return segments


def coverage_ratio(answer: str, claims: list[dict]) -> float:
    """공백을 뺀 글자 기준으로 주장이 덮은 비율을 구합니다."""
    length = len(answer)
    covered = [False] * length
    for claim in claims:
        start, end = claim.get("start"), claim.get("end")
        if start is None or end is None:
            continue
        for index in range(max(0, start), min(end, length)):
            covered[index] = True

    total = sum(1 for ch in answer if not ch.isspace())
    if not total:
        return 0.0
    hit = sum(1 for index, ch in enumerate(answer) if not ch.isspace() and covered[index])
    return hit / total


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"분해 결과가 없습니다: {path}\n"
            "먼저 scripts/review_claims.py 를 실행해 주십시오."
        )
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            claims = row["claims"]
            cases.append(
                {
                    "id": row["id"],
                    "company": row["company"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "extractorModel": row.get("extractor_model", ""),
                    "segments": build_segments(row["answer"], claims),
                    "coverage": round(coverage_ratio(row["answer"], claims) * 100),
                    "claims": [
                        {
                            "id": c["claim_id"],
                            "text": c["text"],
                            "type": c["claim_type"],
                            "sourceText": c["source_text"],
                            "start": c.get("start"),
                            "end": c.get("end"),
                            "matchMode": c.get("match_mode", "NOT_FOUND"),
                            "matchRatio": c.get("match_ratio", 0.0),
                            "needsReview": c.get("match_mode") in ("FUZZY", "NOT_FOUND"),
                        }
                        for c in claims
                    ],
                }
            )
    return cases


def summarize(cases: list[dict]) -> dict:
    claims = [c for case in cases for c in case["claims"]]
    by_type = {key: 0 for key in TYPE_TOOL}
    by_mode = {key: 0 for key in MATCH_LABEL}
    for claim in claims:
        by_type[claim["type"]] = by_type.get(claim["type"], 0) + 1
        by_mode[claim["matchMode"]] = by_mode.get(claim["matchMode"], 0) + 1
    coverages = [case["coverage"] for case in cases]
    return {
        "answers": len(cases),
        "claims": len(claims),
        "perAnswer": round(len(claims) / len(cases), 1) if cases else 0,
        "byType": by_type,
        "byMode": by_mode,
        "needsReview": sum(1 for c in claims if c["needsReview"]),
        "avgCoverage": round(sum(coverages) / len(coverages)) if coverages else 0,
        "minCoverage": min(coverages) if coverages else 0,
        "model": cases[0]["extractorModel"] if cases else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="분해 결과 검토 페이지 생성")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases = load_cases(CLAIMS_PATH)
    summary = summarize(cases)
    payload = json.dumps({"cases": cases, "summary": summary}, ensure_ascii=False)

    template = (PROJECT_ROOT / "scripts" / "review_page_template.html").read_text(
        encoding="utf-8"
    )
    page = template.replace("/*__PAYLOAD__*/null", payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8", newline="\n")

    print(f"생성 완료: {args.out}")
    print(f"  답변 {summary['answers']}건 / 주장 {summary['claims']}개 "
          f"(답변당 평균 {summary['perAnswer']}개)")
    print(f"  평균 커버리지 {summary['avgCoverage']}% (최저 {summary['minCoverage']}%)")
    print(f"  검토 필요 {summary['needsReview']}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
