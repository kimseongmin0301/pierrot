"""검증 에이전트를 실제 주장에 돌립니다.

지금까지는 주장 유형별로 스크립트를 따로 돌렸습니다(`verify_price_claims.py`,
`verify_numeric_claims.py`). 어느 도구로 보낼지는 사람이 정했습니다.
이 스크립트는 **에이전트가 스스로 경로를 고르게** 하고 그 결과를 봅니다.

실행:
    .venv/Scripts/python.exe scripts/run_verifier.py
    .venv/Scripts/python.exe scripts/run_verifier.py --fallback    # 대체 경로 켜기
    .venv/Scripts/python.exe scripts/run_verifier.py --limit 5
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from pierrot.config import get_settings  # noqa: E402
from pierrot.cost import CostLogger  # noqa: E402
from pierrot.verifier import VerifierTools, build_graph, verify_claim  # noqa: E402
from pierrot.verifier.state import Confidence, Route, Verdict  # noqa: E402

CLAIMS_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_claims.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "agent_verifications.jsonl"

MARK = {
    Verdict.SUPPORTED: "[ 근거 있음   ]",
    Verdict.CONTRADICTED: "[ 근거와 모순 ]",
    Verdict.UNVERIFIABLE: "[ 확인 불가   ]",
}


def wrap(text: str, prefix: str) -> str:
    return textwrap.fill(
        text, width=104, initial_indent=prefix, subsequent_indent=" " * len(prefix)
    )


def load_claims(limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in CLAIMS_PATH.open(encoding="utf-8") if line.strip()]
    claims = []
    for row in rows:
        for claim in row["claims"]:
            claims.append({**claim, "sample_id": row["id"]})
    return claims[:limit] if limit else claims


def main() -> int:
    parser = argparse.ArgumentParser(description="검증 에이전트 실행")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="1차 경로가 확인 불가일 때 의미 비교(NLI)로 한 번 더 시도합니다",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="주장별 출력을 줄입니다")
    args = parser.parse_args()

    settings = get_settings()
    claims = load_claims(args.limit)

    print("=" * 104)
    print("검증 에이전트 (LangGraph)")
    print("=" * 104)
    print(f"  대상 주장  : {len(claims)}개")
    print(f"  대체 경로  : {'켬 (확인 불가 -> 의미 비교)' if args.fallback else '끔'}")

    tools = VerifierTools(settings=settings)
    graph = build_graph(tools)
    print(f"  그래프 컴파일 직후 로딩 상태: {tools.loaded()}")

    started = time.time()
    results = []
    for index, claim in enumerate(claims, start=1):
        result = verify_claim(
            claim["text"],
            claim_id=f"{claim['sample_id']}/{claim['claim_id']}",
            declared_type=claim["claim_type"],
            graph=graph,
            allow_fallback=args.fallback,
        )
        results.append(result)

        if args.quiet:
            continue
        print()
        print("-" * 104)
        flipped = result.declared_type and result.declared_type != result.route.value
        print(
            f"{index:2}. {result.claim_id}  {MARK[result.verdict]}  "
            f"경로 {result.route.value}" + ("  <- 추출기 유형을 뒤집음" if flipped else "")
        )
        print(wrap(result.claim_text, "    주장: "))
        print(wrap(result.route_reason, "    라우팅: "))
        print(wrap(result.reason, "    근거: "))
        if result.used_fallback:
            chain = " -> ".join(
                f"{a.route.value}({a.verdict.value})" for a in result.attempts
            )
            print(f"    시도: {chain}")
        for line in result.evidence[:2]:
            print(wrap(line, "          "))

    elapsed = time.time() - started

    # ---------------- 요약 ----------------
    verdicts = collections.Counter(r.verdict for r in results)
    routes = collections.Counter(r.route for r in results)
    declared = collections.Counter(r.declared_type for r in results)
    flips = [r for r in results if r.declared_type and r.declared_type != r.route.value]
    fallbacks = [r for r in results if r.used_fallback]
    forecasts = [r for r in results if r.is_forecast]

    judged = verdicts[Verdict.SUPPORTED] + verdicts[Verdict.CONTRADICTED]
    total = len(results)
    non_forecast = total - len(forecasts)

    print()
    print("=" * 104)
    print("요약")
    print("=" * 104)
    print(f"  소요 시간 : {elapsed:.1f}초  ({elapsed / max(total, 1):.2f}초/주장)")
    print()
    print("  판정")
    for verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNVERIFIABLE):
        print(f"    {verdict.value:14} {verdicts[verdict]:3}건")
    print()
    print("  경로 (추출기 유형 -> 에이전트 경로)")
    for route in (Route.PRICE, Route.NUMERIC, Route.NARRATIVE, Route.FORECAST):
        print(
            f"    {route.value:10} 에이전트 {routes[route]:3}건"
            f"   (추출기 {declared.get(route.value, 0):3}건)"
        )
    print(f"    유형을 뒤집은 주장: {len(flips)}건")
    for result in flips[:8]:
        print(f"      {result.declared_type} -> {result.route.value}  {result.claim_text[:62]}")

    calculated = [r for r in results if r.confidence is Confidence.CALCULATED]
    inferred = [r for r in results if r.confidence is Confidence.INFERRED]

    print()
    print("  커버리지 (판정이 난 비율)")
    print(f"    전체 기준          {judged:3}/{total:3} = {judged / max(total, 1) * 100:5.1f}%")
    print(
        f"    전망 제외 기준      {judged:3}/{non_forecast:3} = "
        f"{judged / max(non_forecast, 1) * 100:5.1f}%   (전망 {len(forecasts)}건 제외)"
    )
    print(
        f"    계산 기반만        {len(calculated):3}/{total:3} = "
        f"{len(calculated) / max(total, 1) * 100:5.1f}%"
    )
    print()
    print("  판정의 성격")
    print(
        f"    계산으로 대조      {len(calculated):3}건   공시·주가 수치와 직접 비교. "
        f"근거를 숫자로 제시합니다."
    )
    print(
        f"    의미 비교로 추정    {len(inferred):3}건   NLI 판정. "
        f"근거 문장이 주장과 무관할 수 있어 그대로 믿으면 안 됩니다."
    )

    if args.fallback:
        rescued = [r for r in fallbacks if r.verdict is not Verdict.UNVERIFIABLE]
        print()
        print(f"  대체 경로: {len(fallbacks)}건 시도, {len(rescued)}건이 판정으로 바뀜")
        for result in rescued[:10]:
            print(f"    {result.verdict.value:13} {result.claim_text[:70]}")

    print()
    print("  확인 불가 사유 (상위)")
    reasons = collections.Counter(
        r.reason[:46] for r in results if r.verdict is Verdict.UNVERIFIABLE
    )
    for reason, count in reasons.most_common(10):
        print(f"    {count:3}건  {reason}")

    out = args.out or (
        OUTPUT_PATH.with_name("agent_verifications_fallback.jsonl")
        if args.fallback
        else OUTPUT_PATH
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    print()
    print(f"  저장: {out}")

    logger = CostLogger(
        log_path=settings.cost_log_path, monthly_budget_usd=settings.monthly_budget_usd
    )
    print(
        f"  누적 비용: ${logger.month_total_usd():.4f} / ${settings.monthly_budget_usd:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
