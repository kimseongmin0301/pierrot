"""PRICE 주장을 실제로 검증합니다.

1단계에서 뽑은 주장 중 PRICE 유형만 골라 파싱 -> 조회 -> 판정까지 돌립니다.
2단계의 end-to-end 동작 확인이자, 도구 기반 검증이 실제로 환각을 잡는지 보는 자리입니다.

실행:
    .venv/Scripts/python.exe scripts/verify_price_claims.py
    .venv/Scripts/python.exe scripts/verify_price_claims.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

from pierrot.config import get_settings  # noqa: E402
from pierrot.llm import get_cost_logger  # noqa: E402
from pierrot.tools.dates import DatePrecision  # noqa: E402
from pierrot.tools.price import PriceSnapshot  # noqa: E402
from pierrot.tools.price_claim import PARSER_VERSION, build_price_parser, parse_price_claim  # noqa: E402
from pierrot.tools.price_verify import (  # noqa: E402
    DEFAULT_PRICE_TOLERANCE,
    DEFAULT_RATE_TOLERANCE_PP,
    Verdict,
    verify_price_claim,
)

CLAIMS_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_claims.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "price_verifications.jsonl"

MARK = {
    Verdict.SUPPORTED: "[ 근거 있음  ]",
    Verdict.CONTRADICTED: "[ 근거와 모순 ]",
    Verdict.UNVERIFIABLE: "[ 확인 불가   ]",
}


def wrap(text: str, indent: str) -> str:
    return textwrap.fill(
        text, width=104, initial_indent=indent, subsequent_indent=" " * len(indent)
    )


def load_price_claims() -> list[dict]:
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"분해 결과가 없습니다: {CLAIMS_PATH}\n"
            "먼저 scripts/review_claims.py 를 실행해 주십시오."
        )
    claims = []
    with CLAIMS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for claim in row["claims"]:
                if claim["claim_type"] == "PRICE":
                    claims.append({"sample_id": row["id"], "company": row["company"], **claim})
    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description="PRICE 주장 검증")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개만 처리")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--include-adjusted", action="store_true",
                        help="수정주가 보정 종목의 값 주장도 판정합니다 (오탐 위험)")
    args = parser.parse_args()

    settings = get_settings()
    snapshot = PriceSnapshot.load()
    claims = load_price_claims()
    if args.limit:
        claims = claims[: args.limit]

    span = snapshot.date_range()
    logger = get_cost_logger(settings)
    spent_before = logger.month_total_usd()

    print("=" * 104)
    print("PRICE 주장 검증")
    print("=" * 104)
    print(f"스냅샷     : {len(snapshot):,}행 / {len(snapshot.codes())}종목 / {span[0]} ~ {span[1]}")
    print(f"파서       : {settings.openai_verifier_model} (파서 {PARSER_VERSION})")
    print(f"허용 오차  : 가격 {DEFAULT_PRICE_TOLERANCE:.0%} / 등락률 {DEFAULT_RATE_TOLERANCE_PP:.0f}%p")
    print(f"대상 주장  : {len(claims)}개")
    adjusted = snapshot.adjusted_codes()
    if adjusted:
        names = ", ".join(
            f"{next((st.name for st in snapshot.stocks() if st.code == c), c)}({snapshot.tick_validity(c):.0%})"
            for c in adjusted
        )
        print(f"수정주가   : {names}")
        print(f"             값 주장 판정 "
              f"{'포함(오탐 위험)' if args.include_adjusted else '제외'}")

    price_parser = build_price_parser(settings=settings)
    results = []

    for index, claim in enumerate(claims, start=1):
        query = parse_price_claim(claim["text"], parser=price_parser, settings=settings)
        verification = verify_price_claim(
            query, snapshot, skip_adjusted_levels=not args.include_adjusted
        )

        print()
        print("-" * 104)
        print(f"{index:2}. {claim['sample_id']}/{claim['claim_id']}  {MARK[verification.verdict]}")
        print(wrap(claim["text"], "    주장: "))
        print(wrap(query.describe(), "    조회: "))
        print(wrap(verification.reason, "    근거: "))
        for line in verification.evidence[:3]:
            print(f"          {line}")

        results.append(
            {
                "sample_id": claim["sample_id"],
                "claim_id": claim["claim_id"],
                "claim_text": claim["text"],
                "parser_version": PARSER_VERSION,
                "assertion": query.parsed.assertion.value,
                "stock_code": query.stock.code if query.stock else "",
                "window_start": str(query.window.start) if query.window else "",
                "window_end": str(query.window.end) if query.window else "",
                "date_precision": verification.precision.value,
                "verdict": verification.verdict.value,
                "reason": verification.reason,
                "claimed_low": verification.claimed_low,
                "claimed_high": verification.claimed_high,
                "observed_low": verification.observed_low,
                "observed_high": verification.observed_high,
                "unit": verification.unit,
                "tolerance": verification.tolerance,
                "price_adjusted": (
                    snapshot.is_adjusted(query.stock.code) if query.stock else False
                ),
            }
        )

    print_summary(results)

    spent = logger.month_total_usd() - spent_before
    print()
    print(f"이번 실행 비용: ${spent:.6f} "
          f"(이번 달 누적 ${logger.month_total_usd():.6f} / 상한 ${logger.monthly_budget_usd:.2f})")

    if not args.no_save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"결과 저장     : {OUTPUT_PATH}")

    return 0


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 104)
    print("요약")
    print("=" * 104)

    verdicts = Counter(row["verdict"] for row in results)
    total = len(results)
    for verdict in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
        count = verdicts.get(verdict, 0)
        share = count / total * 100 if total else 0
        print(f"  {verdict:14} {count:3}개 ({share:4.1f}%)")

    # A안을 택했으므로 날짜 정밀도별로 나누어 공개합니다.
    # 범위 날짜는 관대한 기준이라 지표가 좋게 나올 수 있기 때문입니다.
    print()
    print("  날짜 정밀도별")
    for precision in (DatePrecision.DAY, DatePrecision.RANGE, DatePrecision.UNKNOWN):
        subset = [r for r in results if r["date_precision"] == precision.value]
        if not subset:
            continue
        sub = Counter(r["verdict"] for r in subset)
        label = {"DAY": "정확한 날짜", "RANGE": "폭 있는 표현", "UNKNOWN": "날짜 불명"}[precision.value]
        print(f"    {label:12} {len(subset):3}개  "
              f"근거있음 {sub.get('SUPPORTED', 0)} / "
              f"모순 {sub.get('CONTRADICTED', 0)} / "
              f"확인불가 {sub.get('UNVERIFIABLE', 0)}")

    unverifiable = [r for r in results if r["verdict"] == "UNVERIFIABLE"]
    if unverifiable:
        print()
        print("  확인 불가 사유")
        for row in unverifiable:
            print(f"    {row['sample_id']}/{row['claim_id']}: {row['reason'][:84]}")


if __name__ == "__main__":
    sys.exit(main())
