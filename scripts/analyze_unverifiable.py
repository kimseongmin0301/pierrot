"""판정하지 못한 주장을 원인별로 가릅니다.

왜 필요한가: "확인 불가 54건"은 그 자체로는 아무것도 알려주지 않습니다.
데이터가 없어서인지, 주장에 연도가 없어서인지, 애초에 검증 대상이 아닌지에
따라 **고쳐야 할 곳이 완전히 다릅니다.**

5단계에서 부문별 매출 도구를 붙였는데 판정이 1건만 늘었습니다. 도구가
부실해서인지, 그 도구로 풀 주장이 애초에 적어서인지 숫자로 갈라야 합니다.

실행 (API 비용 없음):
    .venv/Scripts/python.exe scripts/analyze_unverifiable.py
    .venv/Scripts/python.exe scripts/analyze_unverifiable.py --result 결과.json

`--result` 는 `scripts/verify_batch.py --out` 이 만든 JSON 도 받습니다.
실제 LLM 답변을 돌린 결과에 같은 분류를 쓰기 위한 것입니다. 분류 기준을
두 벌로 두면 수치가 어긋나므로 한 벌만 둡니다.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

RESULT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "agent_verifications.jsonl"

# 사유 문구 -> 원인 분류.
#
# 분류 기준은 "무엇을 고쳐야 이 주장이 판정되는가" 입니다.
# 도구를 더 붙여야 하는 것과, 1단계 주장 분해를 고쳐야 하는 것과,
# 애초에 검증 대상이 아닌 것은 서로 다른 일입니다.
RULES: list[tuple[str, tuple[str, ...], str]] = [
    (
        "검증 대상 아님 (전망·권고)",
        ("전망·예상·권고 주장이라",),
        "설계대로입니다. 거르지 않고 흘려보낸 결과입니다.",
    ),
    (
        "주장에 시점이 없음",
        ("기간을 특정할 수 없습니다", "시점을 특정할 수 없습니다",
         "사업연도를 특정할 수 없습니다", "기간(연간/분기)을 특정할 수 없습니다"),
        "1단계 주장 분해 문제입니다. 원문에는 있던 시점이 주장에 안 담겼습니다.",
    ),
    (
        "주장에 종목이 없음",
        ("유니버스 안의 종목을 찾지 못해", "유니버스에 없는 종목입니다"),
        "1단계 주장 분해 문제입니다. 탈맥락화가 덜 됐습니다.",
    ),
    (
        "기간이 너무 넓어 판정 불가",
        ("기간이 너무 넓어",),
        "구간이 넓으면 어떤 주장이든 통과하므로 판정하지 않습니다. 오탐 방지 가드입니다.",
    ),
    (
        "주장에 수치가 없음",
        ("비교할 가격 수치가 없습니다", "비교할 증감률 수치가 없습니다",
         "금액을 읽을 수 없습니다", "비교할 금액이 없습니다",
         "수치가 없고 방향만", "수치가 없으므로", "구체적인 수치가 없",
         "비교 대상 수치가 없어"),
        "값을 말하지 않은 주장입니다. 수치 대조로는 가릴 수 없습니다.",
    ),
    (
        "부문 수치인데 매출이 아님",
        ("부문별 표에는 매출만 있어",),
        "공시가 부문별 영업이익·수주를 내놓지 않습니다. 데이터가 없습니다.",
    ),
    (
        "공시에 그 부문이 없음",
        ("부문이 없습니다",),
        "회사가 그 단위로 공시하지 않습니다(HBM, 광고 등). 데이터가 없습니다.",
    ),
    (
        "재무제표에 없는 계정",
        ("주요계정에 없습니다", "다룰 수 없는 계정입니다",
         "주요계정으로 확인할 수 없습니다", "주요계정과 다르므로"),
        "배당·지배주주순이익 등. DART 전체 계정/배당 API 를 붙이면 일부 풀립니다.",
    ),
    (
        "스냅샷에 수치가 없음",
        ("스냅샷에 해당 수치가 없습니다", "수치가 스냅샷에 없어",
         "스냅샷 범위", "스냅샷에"),
        "수집 범위 문제입니다. 분기 보고서를 더 받으면 됩니다.",
    ),
    (
        "도구가 다룰 수 없는 주장",
        ("주가 조회로 확인할 수 있는 주장이 아닙니다", "부문별 실적 주장이 아닙니다",
         "판정 방법이 정해지지 않은 주장입니다", "부문 이름을 찾을 수 없습니다",
         "재무제표 조회로 확인할 수 있는 주장이 아닙니다",
         "재무 조회 조건으로 변환할 수 없습니다"),
        "변동성·인과 등 수치 대조 대상이 아닙니다.",
    ),
    (
        "부문 주장인데 금액이 없음",
        ("부문 주장에 금액이 적혀 있지 않습니다",),
        "부문별 표와는 금액으로만 대조합니다. 자료에 연도별 수치가 있으니 고칠 수 있습니다.",
    ),
    (
        "휴장·스냅샷 밖이라 거래일이 없음",
        ("구간에 거래일이 없습니다", "해당 기간에 거래일이 없습니다"),
        "직전 거래일까지 거슬러 봐도 없는 경우입니다. 수집 범위를 넓히면 일부 풀립니다.",
    ),
    (
        "근거를 찾았으나 판정 못 함 (NLI)",
        ("기준(", "넘는 관계가 없습니다"),
        "검색·NLI 한계입니다. 임베딩 검색이 다음 후보입니다.",
    ),
]


def classify(reason: str) -> str:
    text = reason or ""
    for name, marks, _ in RULES:
        if any(mark in text for mark in marks):
            return name
    return "기타"


def load_rows(path: Path) -> list[dict]:
    """판정 한 건을 담은 사전들의 목록으로 바꿉니다.

    두 가지 형식을 받습니다.
      - JSONL: 한 줄에 판정 하나 (`scripts/run_verifier.py`)
      - JSON : 답변마다 `claims` 목록 (`scripts/verify_batch.py --out`)
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    rows = []
    for answer in json.loads(text):
        for claim in answer["claims"]:
            rows.append({
                "verdict": claim["verdict"],
                "reason": claim["reason"],
                "claim_text": claim["text"],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="확인 불가 원인 분류")
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()

    if not args.result.exists():
        print(f"결과 파일이 없습니다: {args.result}")
        print("먼저 scripts/run_verifier.py 를 실행해 주십시오.")
        return 1

    rows = load_rows(args.result)
    unverifiable = [row for row in rows if row["verdict"] == "UNVERIFIABLE"]

    print("=" * 100)
    print(f"확인 불가 원인 분류 — 전체 {len(rows)}건 중 {len(unverifiable)}건")
    print("=" * 100)

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in unverifiable:
        buckets[classify(row["reason"])].append(row)

    notes = {name: note for name, _, note in RULES}
    order = [name for name, _, _ in RULES] + ["기타"]

    for name in order:
        items = buckets.get(name)
        if not items:
            continue
        print()
        print(f"[{len(items):2}건] {name}")
        print(f"        {notes.get(name, '')}")
        for row in items[:4]:
            print(f"          - {row['claim_text'][:76]}")
        if len(items) > 4:
            print(f"          ... 외 {len(items) - 4}건")

    # 고쳐야 할 곳별로 묶습니다.
    groups = {
        "1단계 주장 분해를 고쳐야 함": ["주장에 시점이 없음", "주장에 종목이 없음"],
        "데이터를 더 붙이면 풀림": ["재무제표에 없는 계정", "스냅샷에 수치가 없음",
                           "부문 주장인데 금액이 없음", "휴장·스냅샷 밖이라 거래일이 없음"],
        "공시에 데이터가 없음 (한계)": ["부문 수치인데 매출이 아님", "공시에 그 부문이 없음"],
        "검증 대상이 아님": ["검증 대상 아님 (전망·권고)", "도구가 다룰 수 없는 주장",
                       "주장에 수치가 없음", "기간이 너무 넓어 판정 불가"],
        "검색·NLI 한계": ["근거를 찾았으나 판정 못 함 (NLI)"],
    }
    print()
    print("=" * 100)
    print("무엇을 고쳐야 하는가")
    print("=" * 100)
    total = len(rows)
    for label, names in groups.items():
        count = sum(len(buckets.get(name, [])) for name in names)
        if not count:
            continue
        print(f"  {count:3}건 ({count / total * 100:4.1f}%)  {label}")
    leftover = sum(len(buckets.get(name, [])) for name in ["기타"])
    if leftover:
        print(f"  {leftover:3}건              미분류")
    return 0


if __name__ == "__main__":
    sys.exit(main())
