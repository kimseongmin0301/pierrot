"""NLI 베이스라인의 천장을 잽니다.

평가에서 NLI 재현율이 15.4% 로 낮았습니다. 원인이 둘 중 무엇인지 갈라야
개선 방향이 나옵니다.

    (가) 검색이 근거를 못 찾아서  → 검색을 고치면 올라갑니다
    (나) NLI 모델 자체의 한계      → 검색을 고쳐도 안 올라갑니다

그래서 **정답 근거 문장 하나만** 전제로 주고 판정시킵니다. 방해 문장이 0개이고
찾아야 할 문장을 손에 쥐여 준 상태이므로, 이것이 NLI 가 낼 수 있는 최대 성능입니다.
여기서도 낮으면 원인은 (나) 이고, 검색·임계값을 손봐야 소용이 없습니다.

임계값도 함께 훑습니다. 0.7 이 나빠서 생긴 문제인지 확인하기 위해서입니다.

실행 (API 비용 없음, 로컬 모델):
    .venv/Scripts/python.exe scripts/probe_nli_ceiling.py
    .venv/Scripts/python.exe scripts/probe_nli_ceiling.py --testset easy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate, format_metrics  # noqa: E402
from pierrot.config import get_settings  # noqa: E402
from pierrot.tools.nli import NliLabel, NliModel  # noqa: E402
from scripts.premise_render import render  # noqa: E402

DATASETS = {
    "hard": PROJECT_ROOT / "evaluation" / "datasets" / "testset_hard.jsonl",
    "easy": PROJECT_ROOT / "evaluation" / "datasets" / "testset.jsonl",
}
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "nli_ceiling.jsonl"

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def decide(entail: float, contra: float, threshold: float) -> str:
    """narrative_verify 와 같은 규칙입니다. 강한 관계가 이깁니다."""
    if entail >= contra:
        return "SUPPORTED" if entail >= threshold else "UNVERIFIABLE"
    return "CONTRADICTED" if contra >= threshold else "UNVERIFIABLE"


def main() -> int:
    parser = argparse.ArgumentParser(description="NLI 천장 측정")
    parser.add_argument("--testset", choices=sorted(DATASETS), default="hard")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--premise", choices=("raw", "sentence"), default="raw",
                        help="raw=토막 라벨 그대로, sentence=문장으로 바꿔서")
    args = parser.parse_args()

    path = DATASETS[args.testset]
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    rows = [row for row in rows if row.get("source_fact")]

    print("=" * 88)
    mode = "토막 라벨 그대로" if args.premise == "raw" else "문장으로 변환"
    print(f"NLI 천장 측정 — 정답 근거 하나만 전제로 제공 / 전제 형식: {mode} "
          f"({args.testset}, {len(rows)}건)")
    print("=" * 88)

    premises = []
    rules: dict[str, int] = {}
    for row in rows:
        if args.premise == "sentence":
            premise, rule = render(row["source_fact"])
        else:
            premise, rule = row["source_fact"], "raw"
        premises.append(premise)
        rules[rule] = rules.get(rule, 0) + 1
    print(f"  전제 변환 규칙: {rules}")
    if args.premise == "sentence":
        print(f"  예시: {premises[0]}")

    model = NliModel(settings=get_settings()).load()
    results = model.predict_batch(list(zip(premises, [r["text"] for r in rows])))

    scored = []
    for row, premise, result in zip(rows, premises, results):
        scored.append(
            {
                "claim_id": row["claim_id"],
                "claim_type": row["claim_type"],
                "label": row["label"],
                "error_kind": row.get("error_kind", ""),
                "text": row["text"],
                "source_fact": row["source_fact"],
                "premise": premise,
                "entailment": result.score(NliLabel.ENTAILMENT),
                "contradiction": result.score(NliLabel.CONTRADICTION),
                "neutral": result.score(NliLabel.NEUTRAL),
            }
        )

    # 임계값별 지표
    print()
    print("임계값별 지표 (정답 근거를 쥐여 준 상태)")
    print("-" * 88)
    print(f"{'임계값':>6} {'재현율':>8} {'오탐률':>8} {'정밀도':>8} {'커버리지':>9}")
    for threshold in THRESHOLDS:
        pairs = [
            (item["label"], decide(item["entailment"], item["contradiction"], threshold))
            for item in scored
        ]
        metrics = evaluate(pairs)
        print(
            f"{threshold:>6.1f} {metrics.recall * 100:>7.1f}% "
            f"{metrics.false_positive_rate * 100:>7.1f}% "
            f"{metrics.precision * 100:>7.1f}% {metrics.coverage * 100:>8.1f}%"
        )

    # 기본 임계값 0.7 로 유형별
    print()
    print("유형별 (임계값 0.7)")
    print("-" * 88)
    for claim_type in ("NUMERIC", "PRICE"):
        subset = [item for item in scored if item["claim_type"] == claim_type]
        if not subset:
            continue
        pairs = [
            (item["label"], decide(item["entailment"], item["contradiction"], 0.7))
            for item in subset
        ]
        print(format_metrics(f"{claim_type} ({len(subset)}건)", evaluate(pairs)))

    # 놓친 거짓 주장 — NLI 가 잡아야 했는데 못 잡은 것
    print()
    print("놓친 거짓 주장 (정답 근거를 줬는데도 못 잡음)")
    print("-" * 88)
    misses = [
        item
        for item in scored
        if item["label"] == "CONTRADICTED"
        and decide(item["entailment"], item["contradiction"], 0.7) != "CONTRADICTED"
    ]
    for item in misses[:8]:
        print(f"  [{item['claim_id']}] {item['text']}")
        print(f"      근거: {item['source_fact']}")
        print(
            f"      함의 {item['entailment']:.3f} / 모순 {item['contradiction']:.3f} "
            f"/ 중립 {item['neutral']:.3f}  → {decide(item['entailment'], item['contradiction'], 0.7)}"
        )
    print(f"  ... 놓침 합계 {len(misses)}건")

    # 거짓 경보 — 맞는 주장을 틀렸다고 한 것
    print()
    print("거짓 경보 (맞는 주장인데 모순이라고 판정)")
    print("-" * 88)
    alarms = [
        item
        for item in scored
        if item["label"] == "SUPPORTED"
        and decide(item["entailment"], item["contradiction"], 0.7) == "CONTRADICTED"
    ]
    for item in alarms[:8]:
        print(f"  [{item['claim_id']}] ({item['error_kind']}) {item['text']}")
        print(f"      근거: {item['source_fact']}")
        print(f"      모순 {item['contradiction']:.3f}")
    print(f"  ... 거짓 경보 합계 {len(alarms)}건")

    # 오류 유형별 적중률 — 어떤 표현에서 무너지는지
    print()
    print("주장 변형 종류별 성적 (임계값 0.7)")
    print("-" * 88)
    kinds: dict[str, list] = {}
    for item in scored:
        kinds.setdefault(item["error_kind"] or "(없음)", []).append(item)
    print(f"{'변형':<22} {'건수':>5} {'정답':>6} {'적중률':>8}")
    for kind, items in sorted(kinds.items(), key=lambda pair: -len(pair[1])):
        hit = sum(
            1
            for item in items
            if decide(item["entailment"], item["contradiction"], 0.7) == item["label"]
        )
        print(f"{kind:<22} {len(items):>5} {hit:>6} {hit / len(items) * 100:>7.1f}%")

    out = OUTPUT_PATH.with_name(f"nli_ceiling_{args.testset}_{args.premise}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for item in scored:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print()
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
