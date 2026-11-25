"""실제 LLM 답변에 대한 PIERROT 판정을 스냅샷으로 되짚어 봅니다.

왜 필요한가
-----------
실제 LLM 답변에는 **정답 라벨이 없습니다.** 테스트셋처럼 자동 채점이 안 됩니다.
그래서 PIERROT 가 내놓은 판정이 맞는지는 사람이 확인해야 하는데, 눈으로
세면 빠뜨립니다.

이 스크립트는 사람이 봐야 할 것을 **골라 줍니다.** 특히 `근거 있음` 은
한 건도 빠짐없이 봐야 합니다. 거짓을 참이라고 한 것이 가장 위험한 오류이고,
실제 글에서의 오탐률은 오직 여기서만 나오기 때문입니다.

왜 gemma3:4b 로는 이게 불가능했나
---------------------------------
gemma3:4b 답변에서 뽑은 주장 94개 중 `근거 있음` 이 **0개**였습니다.
참이라고 판정한 것이 하나도 없으면 "참이라 한 것이 정말 참인가" 를
**잴 분모가 없습니다.** 커버리지만 재고 끝납니다.

gpt-4o-mini 로 바꾸니 참·거짓이 섞여 나왔고, 첫 10건에서 바로 오탐 1건이
드러났습니다("연초" 를 1월 전체로 읽어 +10% 주장을 통과시킨 건).

사용:
    python scripts/audit_real_answers.py result.json
    python scripts/audit_real_answers.py result.json --all   # 모순까지 전부
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def wrap(text: str, prefix: str, width: int = 100) -> str:
    return textwrap.fill(
        text, width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="실제 답변 판정 감사")
    parser.add_argument("result", type=Path, help="verify_batch.py --out 으로 만든 JSON")
    parser.add_argument("--all", action="store_true", help="모순 판정도 전부 보여 줍니다")
    args = parser.parse_args()

    rows = json.loads(args.result.read_text(encoding="utf-8"))
    claims = [(r["label"], c) for r in rows for c in r["claims"]]

    tally = collections.Counter(c["verdict"] for _, c in claims)
    conf = collections.Counter(c["confidence"] for _, c in claims)
    judged = tally["SUPPORTED"] + tally["CONTRADICTED"]

    print("=" * 100)
    print(f"실제 LLM 답변 {len(rows)}건 -> 주장 {len(claims)}개")
    print("=" * 100)
    print(f"  틀렸다고 판정   {tally['CONTRADICTED']:4}")
    print(f"  맞다고 확인     {tally['SUPPORTED']:4}   <- 전수 확인 대상")
    print(f"  가릴 수 없었음  {tally['UNVERIFIABLE']:4}")
    print(f"\n  판정까지 간 비율 {judged}/{len(claims)} = "
          f"{judged / max(len(claims), 1) * 100:.1f}%")
    print(f"  판정의 성격      " + "  ".join(f"{k} {v}" for k, v in conf.most_common()))

    print()
    print("=" * 100)
    print("'맞다고 확인' 전수 — 이게 틀렸으면 거짓을 참이라 한 것입니다")
    print("=" * 100)
    supported = [(label, c) for label, c in claims if c["verdict"] == "SUPPORTED"]
    if not supported:
        print("\n  없습니다. 참이라 판정한 것이 하나도 없으면")
        print("  **실제 글에서의 오탐률을 잴 수 없습니다**(분모가 0).")
        print("  생성 모델을 참·거짓이 섞여 나오는 것으로 바꿔야 합니다.")
    for label, c in supported:
        print(f"\n  [{label}] ({c['confidence']})")
        print(wrap(c["text"], "    주장: "))
        print(wrap(c["reason"], "    근거: "))
        for line in c.get("evidence", [])[:2]:
            print(wrap(line, "          "))

    if args.all:
        print()
        print("=" * 100)
        print("'틀렸다고 판정' 전수")
        print("=" * 100)
        for label, c in claims:
            if c["verdict"] != "CONTRADICTED":
                continue
            print(f"\n  [{label}] ({c['confidence']})")
            print(wrap(c["text"], "    주장: "))
            print(wrap(c["reason"], "    근거: "))

    # 추정(INFERRED) 판정은 근거가 무관할 수 있어 따로 모아 둡니다.
    inferred = [(label, c) for label, c in claims
                if c["confidence"] == "INFERRED" and c["verdict"] != "UNVERIFIABLE"]
    if inferred:
        print()
        print("=" * 100)
        print(f"의미 비교로 추정한 판정 {len(inferred)}건 — 근거가 무관할 수 있습니다")
        print("=" * 100)
        for label, c in inferred:
            print(f"\n  [{label}] {c['verdict']}")
            print(wrap(c["text"], "    주장: "))
            print(wrap(c["reason"], "    근거: "))

    cost = sum(r.get("cost_usd", 0) for r in rows)
    print(f"\n  이번 실행 비용 ${cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
