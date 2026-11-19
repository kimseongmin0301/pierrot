"""주장이 혼자 읽어도 조회 가능한지 점검합니다.

5단계에서 확인 불가 54건을 원인별로 갈랐더니 **12건이 주장에 시점이나
주체가 없어서** 판정되지 못했습니다. 원문에는 있던 정보입니다.

프롬프트를 고치기 전에 기계가 셀 수 있는 기준이 필요합니다. 고치고 나서
좋아졌는지 같은 기준으로 다시 재야 하기 때문입니다.

6단계 전후 비교:
    --claims evaluation/datasets/sample_claims_before_refine.jsonl   (다듬기 전)
    --claims evaluation/datasets/sample_claims_after_refine.jsonl    (다듬기 후)

실행 (API 비용 없음):
    .venv/Scripts/python.exe scripts/check_decontextualization.py
    .venv/Scripts/python.exe scripts/check_decontextualization.py --claims <경로>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from pierrot.extractor.models import Claim  # noqa: E402
from pierrot.extractor.quality import (  # noqa: E402
    has_relative_time,
    has_year,
    has_span_endpoints,
    looks_like_span_claim,
    missing_subject,
    missing_time,
    suspect_forecast,
)
from pierrot.tools.universe import UNIVERSE  # noqa: E402

DEFAULT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_claims.jsonl"


def universe_names() -> tuple[str, ...]:
    names: list[str] = []
    for stock in UNIVERSE:
        names.append(stock.name)
        names.extend(getattr(stock, "aliases", ()) or ())
    return tuple(names)


def load_claims(path: Path) -> list[Claim]:
    claims: list[Claim] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        for item in sample["claims"]:
            claims.append(
                Claim(
                    claim_id=f"{sample['id']}/{item['claim_id']}",
                    text=item["text"],
                    claim_type=item["claim_type"],
                    source_text=item["source_text"],
                    start=item.get("start"),
                    end=item.get("end"),
                    span_found=item.get("span_found", False),
                    match_mode=item.get("match_mode", "NOT_FOUND"),
                    match_ratio=item.get("match_ratio", 0.0),
                )
            )
    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description="탈맥락화 점검")
    parser.add_argument("--claims", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--show", type=int, default=6, help="사례를 몇 개 보일지")
    args = parser.parse_args()

    claims = load_claims(args.claims)
    names = universe_names()

    no_time = missing_time(claims)
    no_subject = missing_subject(claims, names)
    forecasts = suspect_forecast(claims)

    # 시점 누락을 다시 세부로 가릅니다. 고칠 방법이 서로 다릅니다.
    no_year = [c for c in claims if not has_year(c.text)]
    relative = [c for c in claims if has_year(c.text) and has_relative_time(c.text)]
    open_span = [
        c
        for c in claims
        if has_year(c.text)
        and not has_relative_time(c.text)
        and looks_like_span_claim(c.text)
        and not has_span_endpoints(c.text)
    ]

    total = len(claims)
    print("=" * 96)
    print(f"탈맥락화 점검 — {args.claims.name}  주장 {total}개")
    print("=" * 96)
    print()
    print(f"  시점 없음        {len(no_time):3}개 ({len(no_time) / total * 100:5.1f}%)")
    print(f"      연도 자체가 없음        {len(no_year):3}개")
    print(f"      '현재·최근' 등 상대 표현 {len(relative):3}개")
    print(f"      기간 주장인데 한쪽만    {len(open_span):3}개")
    print(f"  주체 없음        {len(no_subject):3}개 ({len(no_subject) / total * 100:5.1f}%)")
    print(f"  전망 의심        {len(forecasts):3}개 ({len(forecasts) / total * 100:5.1f}%)")

    both = [c for c in claims if c in no_time and c in no_subject]
    print(f"  둘 다 없음       {len(both):3}개")

    for label, items in (
        ("연도 자체가 없음", no_year),
        ("'현재·최근' 등 상대 표현", relative),
        ("기간 주장인데 한쪽만", open_span),
        ("주체 없음", no_subject),
    ):
        if not items:
            continue
        print()
        print(f"  [{label}] 사례")
        for claim in items[: args.show]:
            print(f"    - {claim.text[:84]}")
        if len(items) > args.show:
            print(f"    ... 외 {len(items) - args.show}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
