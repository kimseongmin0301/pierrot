"""판정하지 못한 주장을 원인별로 가릅니다.

왜 필요한가: "확인 불가 54건"은 그 자체로는 아무것도 알려주지 않습니다.
데이터가 없어서인지, 주장에 연도가 없어서인지, 애초에 검증 대상이 아닌지에
따라 **고쳐야 할 곳이 완전히 다릅니다.**

5단계에서 부문별 매출 도구를 붙였는데 판정이 1건만 늘었습니다. 도구가
부실해서인지, 그 도구로 풀 주장이 애초에 적어서인지 숫자로 갈라야 합니다.

실행 (API 비용 없음):
    .venv/Scripts/python.exe scripts/analyze_unverifiable.py
"""

from __future__ import annotations

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
        "주장에 수치가 없음",
        ("비교할 가격 수치가 없습니다", "비교할 증감률 수치가 없습니다",
         "금액을 읽을 수 없습니다", "비교할 금액이 없습니다"),
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
         "판정 방법이 정해지지 않은 주장입니다", "부문 이름을 찾을 수 없습니다"),
        "변동성·인과 등 수치 대조 대상이 아닙니다.",
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


def main() -> int:
    if not RESULT_PATH.exists():
        print(f"결과 파일이 없습니다: {RESULT_PATH}")
        print("먼저 scripts/run_verifier.py 를 실행해 주십시오.")
        return 1

    rows = [json.loads(line) for line in RESULT_PATH.open(encoding="utf-8") if line.strip()]
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
        "데이터를 더 붙이면 풀림": ["재무제표에 없는 계정", "스냅샷에 수치가 없음"],
        "공시에 데이터가 없음 (한계)": ["부문 수치인데 매출이 아님", "공시에 그 부문이 없음"],
        "검증 대상이 아님": ["검증 대상 아님 (전망·권고)", "도구가 다룰 수 없는 주장",
                       "주장에 수치가 없음"],
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
