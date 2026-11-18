"""NLI 베이스라인과 도구 기반 검증을 같은 주장으로 비교합니다.

이 프로젝트의 결론은 "도구 기반 검증이 NLI 보다 낫다"입니다. 그 주장이 성립하려면
**같은 주장, 같은 사실**을 두 방법에 똑같이 주고 비교해야 합니다.

그래서 NLI 에게 주는 근거 문장도 DART 와 거래소에서 받은 **검증된 사실**로 만듭니다.
근거를 부실하게 주고 이기면 그 결론은 조작입니다.

실행 (NLI 는 로컬 CPU, 도구 판정은 저장된 결과 재사용 - API 비용 없음):
    .venv/Scripts/python.exe scripts/compare_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pierrot.tools.dart_documents import DocumentStore  # noqa: E402
from pierrot.tools.financials import FinancialSnapshot  # noqa: E402
from pierrot.tools.narrative_verify import verify_narrative_claim  # noqa: E402
from pierrot.tools.nli import NliModel  # noqa: E402
from pierrot.tools.price import PriceSnapshot  # noqa: E402

DATASETS = PROJECT_ROOT / "evaluation" / "datasets"
CLAIMS_PATH = DATASETS / "sample_claims.jsonl"
PRICE_PATH = DATASETS / "price_verifications.jsonl"
NUMERIC_PATH = DATASETS / "numeric_verifications.jsonl"
OUTPUT_PATH = DATASETS / "baseline_comparison.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_claims() -> list[dict]:
    claims = []
    for row in load_jsonl(CLAIMS_PATH):
        for claim in row["claims"]:
            claims.append({"sample_id": row["id"], **claim})
    return claims


def tool_verdicts() -> dict[tuple[str, str], str]:
    """저장된 도구 판정 결과를 (샘플, 주장) 키로 모읍니다."""
    verdicts: dict[tuple[str, str], str] = {}
    for path in (PRICE_PATH, NUMERIC_PATH):
        for row in load_jsonl(path):
            verdicts[(row["sample_id"], row["claim_id"])] = row["verdict"]
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description="NLI 베이스라인 vs 도구 기반 검증")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--top-k", type=int, default=30,
                        help="NLI 에 넘길 근거 수. 0 이면 검색 없이 전부")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-documents", action="store_true",
                        help="공시 본문 없이 수치 근거만 사용합니다 (이전 설정 재현용)")
    args = parser.parse_args()

    claims = load_claims()
    if args.limit:
        claims = claims[: args.limit]
    tools = tool_verdicts()

    financials = FinancialSnapshot.load()
    prices = PriceSnapshot.load()
    documents = None if args.no_documents else DocumentStore.load()

    print("=" * 100)
    print("NLI 베이스라인 vs 도구 기반 검증")
    print("=" * 100)
    print(f"주장      : {len(claims)}개")
    print(f"재무 근거 : {financials.describe_coverage()}")
    print(f"공시 본문 : {documents.describe() if documents else '사용 안 함'}")
    print(f"검색      : {'top-' + str(args.top_k) if args.top_k else '사용 안 함(전부 비교)'}")
    print(f"도구 판정 : {len(tools)}건 (저장된 결과 재사용)")
    print("NLI 모델 적재 중...")

    model = NliModel().load()
    print(f"  {model.model_name} / 라벨 {model.label_order} ({model.label_source})")
    print()

    results = []
    for index, claim in enumerate(claims, start=1):
        verification = verify_narrative_claim(
            claim["text"], model=model, financials=financials,
            prices=prices, documents=documents, top_k=args.top_k,
        )
        key = (claim["sample_id"], claim["claim_id"])
        results.append(
            {
                "sample_id": claim["sample_id"],
                "claim_id": claim["claim_id"],
                "claim_type": claim["claim_type"],
                "claim_text": claim["text"],
                "nli_verdict": verification.verdict.value,
                "nli_entailment": round(verification.best_entailment, 4),
                "nli_contradiction": round(verification.best_contradiction, 4),
                "nli_evidence_count": verification.evidence_count,
                "nli_retrieved_count": verification.retrieved_count,
                "nli_best_evidence": verification.best_evidence,
                "tool_verdict": tools.get(key, ""),
            }
        )
        if index % 20 == 0:
            print(f"  {index}/{len(claims)} 처리")

    print_summary(results)

    if not args.no_save:
        out = args.out or OUTPUT_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n결과 저장: {OUTPUT_PATH}")
    return 0


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("요약")
    print("=" * 100)

    print()
    print("주장 유형별 NLI 판정")
    print(f"  {'유형':12} {'건수':>5} {'근거있음':>8} {'모순':>6} {'확인불가':>8}")
    for claim_type in ("NUMERIC", "PRICE", "NARRATIVE"):
        subset = [r for r in results if r["claim_type"] == claim_type]
        if not subset:
            continue
        counts = Counter(r["nli_verdict"] for r in subset)
        print(f"  {claim_type:12} {len(subset):5} {counts.get('SUPPORTED', 0):8} "
              f"{counts.get('CONTRADICTED', 0):6} {counts.get('UNVERIFIABLE', 0):8}")

    # 도구가 판정한 주장만 놓고 비교합니다.
    paired = [r for r in results if r["tool_verdict"]]
    if not paired:
        return

    print()
    print(f"도구와 NLI 를 모두 돌린 주장 {len(paired)}개 비교")
    print()
    print(f"  {'':14} {'도구':>8} {'NLI':>8}")
    for verdict in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
        tool_count = sum(1 for r in paired if r["tool_verdict"] == verdict)
        nli_count = sum(1 for r in paired if r["nli_verdict"] == verdict)
        print(f"  {verdict:14} {tool_count:8} {nli_count:8}")

    agree = sum(1 for r in paired if r["tool_verdict"] == r["nli_verdict"])
    print()
    print(f"  판정 일치 {agree}/{len(paired)} ({agree / len(paired) * 100:.1f}%)")

    print()
    print("  판정이 갈린 조합 (도구 -> NLI)")
    for (tool, nli), count in Counter(
        (r["tool_verdict"], r["nli_verdict"]) for r in paired
        if r["tool_verdict"] != r["nli_verdict"]
    ).most_common():
        print(f"    {tool:14} -> {nli:14} {count:3}건")

    # 도구는 환각을 잡았는데 NLI 는 놓친 경우가 이 프로젝트의 핵심 논거입니다.
    missed = [
        r for r in paired
        if r["tool_verdict"] == "CONTRADICTED" and r["nli_verdict"] != "CONTRADICTED"
    ]
    print()
    print(f"  도구는 모순으로 잡았으나 NLI 가 놓친 주장: {len(missed)}건")
    for row in missed[:5]:
        print(f"    [{row['nli_verdict']:13}] {row['claim_text'][:62]}")
        print(f"       최대 함의 {row['nli_entailment']:.3f} / 최대 모순 {row['nli_contradiction']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
