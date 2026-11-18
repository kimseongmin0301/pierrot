"""정답 라벨이 붙은 테스트셋을 만들어 저장합니다.

스냅샷에서 직접 생성하므로 라벨이 확실합니다. seed 를 고정해 다시 돌려도
같은 테스트셋이 나옵니다. 평가 지표를 비교하려면 테스트셋이 고정되어야 합니다.

실행 (API 비용 없음):
    .venv/Scripts/python.exe evaluation/build_testset.py
    .venv/Scripts/python.exe evaluation/build_testset.py --target 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.testset import TestSetBuilder, summarize  # noqa: E402
from pierrot.tools.financials import FinancialSnapshot  # noqa: E402
from pierrot.tools.price import PriceSnapshot  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "testset.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="라벨된 테스트셋 생성")
    parser.add_argument("--target", type=int, default=300, help="목표 주장 수")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--hard", action="store_true",
                        help="어림수·다른 단위·모호한 날짜로 만든 어려운 셋")
    args = parser.parse_args()

    financials = FinancialSnapshot.load()
    prices = PriceSnapshot.load()

    builder = TestSetBuilder(seed=args.seed)
    if args.hard:
        numeric = builder.hard_numeric_claims(financials)
        direction = []
        price = builder.hard_price_claims(prices)
    else:
        numeric = builder.numeric_claims(financials)
        direction = builder.direction_claims(financials)
        price = builder.price_claims(prices)

    out = args.out or (
        OUTPUT_PATH.with_name("testset_hard.jsonl") if args.hard else OUTPUT_PATH
    )

    print("=" * 88)
    print("테스트셋 생성")
    print("=" * 88)
    print(f"  재무 금액 주장 {len(numeric):4}개")
    print(f"  증감 방향 주장 {len(direction):4}개")
    print(f"  주가 주장     {len(price):4}개")
    print(f"  합계         {len(numeric) + len(direction) + len(price):4}개")

    pool = numeric + direction + price

    # 목표 수에 맞춰 줄이되, 라벨과 유형 비율이 유지되도록 고르게 뽑습니다.
    if args.target and len(pool) > args.target:
        builder.random.shuffle(pool)
        buckets: dict[tuple[str, str], list] = {}
        for claim in pool:
            buckets.setdefault((claim.claim_type, claim.label), []).append(claim)

        selected = []
        index = 0
        while len(selected) < args.target:
            added = False
            for items in buckets.values():
                if index < len(items):
                    selected.append(items[index])
                    added = True
                    if len(selected) >= args.target:
                        break
            if not added:
                break
            index += 1
        pool = selected

    pool.sort(key=lambda claim: claim.claim_id)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for claim in pool:
            handle.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")

    stats = summarize(pool)
    print()
    print(f"저장 완료: {out} ({len(pool)}개)")
    print(f"  라벨별: {stats['by_label']}")
    print(f"  유형별: {stats['by_type']}")
    print(f"  오류별: {stats['by_error']}")
    print()
    print("샘플")
    for claim in pool[:3]:
        print(f"  [{claim.label:13}] {claim.text}")
        print(f"      근거: {claim.source_fact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
