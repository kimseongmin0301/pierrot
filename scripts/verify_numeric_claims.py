"""NUMERIC 주장을 실제로 검증합니다.

1단계에서 뽑은 주장 중 NUMERIC 유형만 골라 파싱 -> 조회 -> 판정까지 돌립니다.

실행:
    .venv/Scripts/python.exe scripts/verify_numeric_claims.py
    .venv/Scripts/python.exe scripts/verify_numeric_claims.py --limit 5
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
from pierrot.tools.financials import FinancialSnapshot  # noqa: E402
from pierrot.tools.numeric_claim import (  # noqa: E402
    PARSER_VERSION,
    build_numeric_parser,
    parse_numeric_claim,
)
from pierrot.tools.numeric_verify import Verdict, verify_numeric_claim  # noqa: E402

CLAIMS_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_claims.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "numeric_verifications.jsonl"

MARK = {
    Verdict.SUPPORTED: "[ 근거 있음  ]",
    Verdict.CONTRADICTED: "[ 근거와 모순 ]",
    Verdict.UNVERIFIABLE: "[ 확인 불가   ]",
}

FS_LABEL = {"CFS": "연결", "OFS": "별도", "": "-"}


def wrap(text: str, indent: str) -> str:
    return textwrap.fill(
        text, width=104, initial_indent=indent, subsequent_indent=" " * len(indent)
    )


def load_numeric_claims() -> list[dict]:
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
                if claim["claim_type"] == "NUMERIC":
                    claims.append({"sample_id": row["id"], **claim})
    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description="NUMERIC 주장 검증")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    snapshot = FinancialSnapshot.load()
    claims = load_numeric_claims()
    if args.limit:
        claims = claims[: args.limit]

    logger = get_cost_logger(settings)
    spent_before = logger.month_total_usd()

    print("=" * 104)
    print("NUMERIC 주장 검증")
    print("=" * 104)
    print(f"재무 스냅샷: {snapshot.describe_coverage()}")
    print(f"파서       : {settings.openai_verifier_model} (파서 {PARSER_VERSION})")
    print("판정 기준  : 연결/별도 중 하나라도 맞으면 인정하되 어느 쪽인지 기록")
    print(f"대상 주장  : {len(claims)}개")

    numeric_parser = build_numeric_parser(settings=settings)
    results = []

    for index, claim in enumerate(claims, start=1):
        query = parse_numeric_claim(claim["text"], parser=numeric_parser, settings=settings)
        verification = verify_numeric_claim(query, snapshot)

        print()
        print("-" * 104)
        print(f"{index:2}. {claim['sample_id']}/{claim['claim_id']}  {MARK[verification.verdict]}")
        print(wrap(claim["text"], "    주장: "))
        print(wrap(query.describe(), "    조회: "))
        print(wrap(verification.reason, "    근거: "))
        for line in verification.evidence[:2]:
            print(wrap(line, "          "))

        results.append(
            {
                "sample_id": claim["sample_id"],
                "claim_id": claim["claim_id"],
                "claim_text": claim["text"],
                "parser_version": PARSER_VERSION,
                "assertion": query.parsed.assertion.value,
                "stock_code": query.stock.code if query.stock else "",
                "account": query.parsed.account,
                "bsns_year": query.parsed.bsns_year,
                "quarter": query.parsed.quarter,
                "fs_hint": query.parsed.fs_hint,
                "verdict": verification.verdict.value,
                "reason": verification.reason,
                "claimed": verification.claimed,
                "observed": verification.observed,
                "unit": verification.unit,
                "matched_fs": verification.matched_fs,
                "fs_explicit": verification.fs_explicit,
                "tolerance": verification.tolerance,
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

    total = len(results)
    verdicts = Counter(row["verdict"] for row in results)
    for verdict in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
        count = verdicts.get(verdict, 0)
        print(f"  {verdict:14} {count:3}개 ({count / total * 100 if total else 0:4.1f}%)")

    print()
    print("  주장 종류별")
    for assertion, count in Counter(r["assertion"] for r in results).most_common():
        sub = Counter(r["verdict"] for r in results if r["assertion"] == assertion)
        print(f"    {assertion:12} {count:3}개  "
              f"근거있음 {sub.get('SUPPORTED', 0)} / 모순 {sub.get('CONTRADICTED', 0)} / "
              f"확인불가 {sub.get('UNVERIFIABLE', 0)}")

    # 연결/별도를 밝히지 않은 주장을 둘 다 보고 판정했으므로,
    # 어느 쪽으로 맞았는지 공개해야 관대한 기준이 숨겨지지 않습니다.
    supported = [r for r in results if r["verdict"] == "SUPPORTED"]
    if supported:
        print()
        print("  근거 있음 판정이 맞은 기준")
        for fs, count in Counter(r["matched_fs"] for r in supported).most_common():
            explicit = sum(1 for r in supported if r["matched_fs"] == fs and r["fs_explicit"])
            print(f"    {FS_LABEL.get(fs, fs):6} {count:3}개 "
                  f"(주장이 기준을 직접 밝힌 것 {explicit}개)")

    unverifiable = [r for r in results if r["verdict"] == "UNVERIFIABLE"]
    if unverifiable:
        print()
        print(f"  확인 불가 {len(unverifiable)}건의 사유")
        for reason, count in Counter(
            r["reason"].split(":")[0][:60] for r in unverifiable
        ).most_common():
            print(f"    {count:2}건  {reason}")


if __name__ == "__main__":
    sys.exit(main())
