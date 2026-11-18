"""주장 분해 결과를 사람이 검토할 수 있는 형태로 출력.

1단계의 완료 기준은 "샘플 답변 10개에서 주장 분리 결과를 사람이 검토해 합격"입니다.
이 스크립트는 그 검토를 위해 원문과 분해 결과를 나란히 보여 줍니다.

검토할 때 볼 것:

1. 누락 - 답변에 있는 검증 가능한 진술이 주장 목록에서 빠지지 않았는가.
   누락된 주장은 뒤 단계에서 영원히 검증되지 않습니다. 가장 치명적인 오류입니다.
2. 자립성 - 주장 하나만 떼어 놓고 읽어도 무엇에 대한 이야기인지 알 수 있는가.
   "그것은 5% 올랐다" 같은 문장은 검증 단계에서 조회할 대상을 특정할 수 없습니다.
3. 값 보존 - 원문의 수치와 날짜가 그대로 옮겨졌는가. LLM이 틀린 값을 고쳐 버리면
   환각이 사라져서 검증할 것이 없어집니다.
4. 유형 분류 - NUMERIC / PRICE / NARRATIVE 가 검증 도구와 맞게 붙었는가.
5. 지어내기 - [검토 필요] 표시가 붙은 주장의 근거 문장이 원문과 같은 뜻인가.

실행:
    .venv/Scripts/python.exe scripts/review_claims.py
    .venv/Scripts/python.exe scripts/review_claims.py --limit 3
    .venv/Scripts/python.exe scripts/review_claims.py --id q01 q05
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

from pierrot.config import get_settings  # noqa: E402
from pierrot.extractor import build_extractor, extract_claims  # noqa: E402
from pierrot.extractor.models import ClaimType, ExtractionResult, MatchMode  # noqa: E402
from pierrot.llm import get_cost_logger  # noqa: E402

SAMPLES_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_answers.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_claims.jsonl"

TYPE_LABEL = {
    ClaimType.NUMERIC: "NUMERIC   (공시로 검증)",
    ClaimType.PRICE: "PRICE     (주가로 검증)",
    ClaimType.NARRATIVE: "NARRATIVE (문서 대조로 검증)",
}

MATCH_LABEL = {
    MatchMode.EXACT: "원문 그대로",
    MatchMode.WHITESPACE: "공백 차이",
    MatchMode.NORMALIZED: "표기 차이",
    MatchMode.FUZZY: "유사도 매칭",
    MatchMode.NOT_FOUND: "원문에 없음",
}


def load_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"샘플 파일이 없습니다: {path}\n"
            "먼저 scripts/generate_samples.py 를 실행해 주십시오."
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wrap(text: str, indent: str = "       ") -> str:
    """긴 문장을 줄바꿈합니다. 두 번째 줄부터는 라벨 대신 공백으로 맞춥니다."""
    return textwrap.fill(
        text.replace("\n", " "),
        width=100,
        initial_indent=indent,
        subsequent_indent=" " * len(indent),
    )


def print_review(sample: dict, result: ExtractionResult, elapsed: float) -> None:
    print()
    print("=" * 104)
    print(f"{sample['id']}  {sample['company']}   "
          f"(주장 {result.claim_count}개, {elapsed:.1f}초)")
    print("=" * 104)
    print("[질문]")
    print(wrap(sample["question"]))
    print()
    print("[원문 답변]")
    print(wrap(result.answer))
    print()
    print("[분해된 주장]")

    if not result.claims:
        print("       (추출된 주장이 없습니다)")
        return

    for claim in result.claims:
        position = f"{claim.start}~{claim.end}" if claim.span_found else "미확인"
        mode = MATCH_LABEL[claim.match_mode]
        if claim.match_mode is MatchMode.FUZZY:
            mode = f"{mode} 유사도 {claim.match_ratio:.3f}"
        marker = "   <-- 검토 필요" if claim.needs_review else ""
        print(f"  {claim.claim_id}  {TYPE_LABEL[claim.claim_type]}  "
              f"원문 {position}  [{mode}]{marker}")
        print(wrap(claim.text, indent="        주장: "))
        print(wrap(claim.source_text, indent="        원문: "))
        print()


def print_summary(results: list[tuple[dict, ExtractionResult]]) -> None:
    print()
    print("=" * 104)
    print("요약")
    print("=" * 104)

    total_claims = sum(r.claim_count for _, r in results)
    totals = {t.value: 0 for t in ClaimType}
    modes = {m.value: 0 for m in MatchMode}
    review_needed: list[str] = []

    print(f"{'ID':6} {'회사':16} {'길이':>6} {'주장':>4} "
          f"{'NUM':>4} {'PRICE':>6} {'NARR':>5} | "
          f"{'그대로':>6} {'표기차이':>8} {'유사도':>6} {'없음':>5}")
    print("-" * 104)
    for sample, result in results:
        counts = result.count_by_type()
        for key, value in counts.items():
            totals[key] += value
        mode_counts = result.count_by_match_mode()
        for key, value in mode_counts.items():
            modes[key] += value
        review_needed.extend(
            f"{sample['id']}/{c.claim_id}" for c in result.review_needed_claims
        )
        exact = mode_counts["EXACT"] + mode_counts["WHITESPACE"]
        print(f"{sample['id']:6} {sample['company']:16} {len(result.answer):5}자 "
              f"{result.claim_count:4} {counts['NUMERIC']:4} {counts['PRICE']:6} "
              f"{counts['NARRATIVE']:5} | {exact:6} {mode_counts['NORMALIZED']:8} "
              f"{mode_counts['FUZZY']:6} {mode_counts['NOT_FOUND']:5}")
    print("-" * 104)
    exact_total = modes["EXACT"] + modes["WHITESPACE"]
    print(f"{'합계':21} {'':6} {total_claims:4} "
          f"{totals['NUMERIC']:4} {totals['PRICE']:6} {totals['NARRATIVE']:5} | "
          f"{exact_total:6} {modes['NORMALIZED']:8} {modes['FUZZY']:6} {modes['NOT_FOUND']:5}")

    if results:
        print()
        print(f"답변당 평균 주장 수: {total_claims / len(results):.1f}개")
    if review_needed:
        print(f"검토가 필요한 주장 {len(review_needed)}개: {', '.join(review_needed)}")
        print("  - [유사도 매칭]: LLM이 원문을 고쳐 적었습니다. 내용이 바뀌지 않았는지 확인하십시오.")
        print("  - [원문에 없음]: LLM이 원문에 없는 문장을 지어냈습니다.")
    else:
        print("모든 주장이 원문에서 위치를 확인했습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="주장 분해 결과 검토용 출력")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개만 처리합니다")
    parser.add_argument("--id", nargs="*", default=None, help="특정 샘플 ID만 처리합니다")
    parser.add_argument("--no-save", action="store_true", help="결과 파일을 쓰지 않습니다")
    args = parser.parse_args()

    settings = get_settings()
    samples = load_samples(SAMPLES_PATH)

    if args.id:
        wanted = set(args.id)
        samples = [s for s in samples if s["id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("처리할 샘플이 없습니다.")
        return 1

    logger = get_cost_logger(settings)
    spent_before = logger.month_total_usd()

    print(f"분해 모델: {settings.openai_extractor_model}")
    print(f"샘플 수  : {len(samples)}")

    extractor = build_extractor(settings=settings)
    results: list[tuple[dict, ExtractionResult]] = []

    for sample in samples:
        started = time.perf_counter()
        result = extract_claims(
            sample["answer"], extractor=extractor, settings=settings
        )
        elapsed = time.perf_counter() - started
        results.append((sample, result))
        print_review(sample, result, elapsed)

    print_summary(results)

    spent = logger.month_total_usd() - spent_before
    print()
    print(f"이번 실행 비용: ${spent:.6f} "
          f"(이번 달 누적 ${logger.month_total_usd():.6f} / 상한 ${logger.monthly_budget_usd:.2f})")

    if not args.no_save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            for sample, result in results:
                handle.write(
                    json.dumps(
                        {
                            "id": sample["id"],
                            "company": sample["company"],
                            "question": sample["question"],
                            "answer": result.answer,
                            "extractor_model": result.model,
                            "claims": [claim.model_dump() for claim in result.claims],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"결과 저장: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
