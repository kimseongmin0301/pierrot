"""NLI 모델이 어디서 무너지는지 실측합니다.

계획 초안에는 "NLI 는 숫자를 값으로 보지 못해 9조원과 10조원을 함의로 판정한다"고
적었습니다. **실측해 보니 틀렸습니다.** 이 모델은 9조원과 10조원을 정확히 구분합니다.

그래서 진짜 약점이 어디인지 조건을 바꿔 가며 확인합니다. 베이스라인의 한계를
추측이 아니라 측정으로 적어야, "도구 기반 검증이 낫다"는 결론이 근거를 갖습니다.

실행 (API 비용 없음, 로컬 CPU):
    .venv/Scripts/python.exe scripts/probe_nli_limits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pierrot.tools.nli import NliLabel, NliModel  # noqa: E402

PREMISE_AMOUNT = "삼성전자의 2024년 3분기 영업이익은 연결 기준 9조 1,833억원이다."
PREMISE_PRICE = "삼성전자의 주가는 2024년 3분기에 81,800원에서 61,500원으로 24.8% 하락했다."
PREMISE_SEPARATE = "삼성전자의 2024년 연간 영업이익은 별도 기준 12조 3,610억원이다."

CASES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "숫자 구분 능력",
        PREMISE_AMOUNT,
        [
            ("삼성전자의 2024년 3분기 영업이익은 9조 1,833억원이다.", "ENTAILMENT"),
            ("삼성전자의 2024년 3분기 영업이익은 9조원이다.", "ENTAILMENT"),
            ("삼성전자의 2024년 3분기 영업이익은 10조원이다.", "CONTRADICTION"),
            ("삼성전자의 2024년 3분기 영업이익은 5조원이다.", "CONTRADICTION"),
            ("삼성전자의 2024년 3분기 영업이익은 3,000억원이다.", "CONTRADICTION"),
        ],
    ),
    (
        "표기 형식이 다를 때",
        PREMISE_AMOUNT,
        [
            ("삼성전자의 2024년 3분기 영업이익은 9.18조원이다.", "ENTAILMENT"),
            ("삼성전자의 2024년 3분기 영업이익은 91,833억원이다.", "ENTAILMENT"),
            ("삼성전자의 2024년 3분기 영업이익은 9,183,300백만원이다.", "ENTAILMENT"),
        ],
    ),
    (
        "파생 계산이 필요할 때",
        PREMISE_PRICE,
        [
            ("삼성전자 주가는 2024년 3분기에 약 25% 하락했다.", "ENTAILMENT"),
            ("삼성전자 주가는 2024년 3분기에 5% 하락했다.", "CONTRADICTION"),
            ("삼성전자 주가는 2024년 3분기에 20,300원 하락했다.", "ENTAILMENT"),
        ],
    ),
    (
        "근거에 없는 사실",
        PREMISE_AMOUNT,
        [
            ("삼성전자의 2024년 1분기 영업이익은 6조 6,060억원이다.", "NEUTRAL"),
            ("삼성전자의 2024년 1분기 영업이익은 20조원이다.", "NEUTRAL"),
        ],
    ),
    (
        "숫자는 같지만 기준이 다를 때",
        PREMISE_SEPARATE,
        [
            ("삼성전자의 2024년 연간 영업이익은 12조 3,610억원이다.", "NEUTRAL"),
            ("삼성전자의 2024년 3분기 영업이익은 12조 3,610억원이다.", "NEUTRAL"),
        ],
    ),
]


def main() -> int:
    model = NliModel().load()
    print("=" * 100)
    print("NLI 한계 실측")
    print("=" * 100)
    print(f"모델      : {model.model_name}")
    print(f"라벨 순서 : {model.label_order} ({model.label_source})")
    print()

    totals = {"hit": 0, "miss": 0}
    for title, premise, cases in CASES:
        print("-" * 100)
        print(f"[{title}]")
        print(f"전제: {premise}")
        pairs = [(premise, hypothesis) for hypothesis, _ in cases]
        results = model.predict_batch(pairs)

        for (hypothesis, expected), result in zip(cases, results):
            hit = result.label.value == expected
            totals["hit" if hit else "miss"] += 1
            print(
                f"  [{'OK  ' if hit else 'MISS'}] 기대 {expected:13} "
                f"실제 {result.label.value:13} "
                f"(함의 {result.score(NliLabel.ENTAILMENT):.3f} / "
                f"모순 {result.score(NliLabel.CONTRADICTION):.3f})"
            )
            print(f"         {hypothesis}")
        print()

    total = totals["hit"] + totals["miss"]
    print("=" * 100)
    print(f"일치 {totals['hit']}/{total}, 불일치 {totals['miss']}/{total}")
    print()
    print("관찰된 약점")
    print("  1. 같은 값이라도 표기가 다르면 모순으로 판정합니다.")
    print("     '9조 1,833억원' 과 '9.18조원' 을 다른 값으로 봅니다. 오탐의 주원인입니다.")
    print("  2. 뺄셈 같은 간단한 계산도 하지 못합니다.")
    print("     81,800 - 61,500 = 20,300 인데 '20,300원 하락' 을 모순으로 판정합니다.")
    print("  3. 근거 문장이 사실을 미리 말해 두지 않으면 판정하지 못합니다(NEUTRAL).")
    print("     도구는 임의의 조건으로 조회하고 계산할 수 있습니다.")
    print("  4. 숫자가 같으면 연결/별도 같은 기준 차이를 무시하고 함의로 봅니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
