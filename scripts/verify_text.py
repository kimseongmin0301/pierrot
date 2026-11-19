"""내가 쓴 글을 검증기에 넣어 봅니다.

API 를 띄우고 curl 을 치는 대신, 파일이나 표준입력으로 바로 넣습니다.

    .venv/Scripts/python.exe scripts/verify_text.py my_analysis.txt
    .venv/Scripts/python.exe scripts/verify_text.py --text "삼성전자의 2024년 매출은 300조원이다"
    type my_analysis.txt | .venv/Scripts/python.exe scripts/verify_text.py

돈을 쓰기 전에 먼저 봅니다
--------------------------
이 검증기는 **수집해 둔 자료 안에서만** 대조할 수 있습니다.
유니버스 20종목, 2023년 이후, 공시·주가 수치입니다.

그 밖의 글을 넣으면 LLM 호출만 하고 전부 "확인 불가" 가 나옵니다.
그래서 **먼저 글을 훑어 종목이 있는지 확인하고**, 없으면 물어봅니다.
확인 불가만 잔뜩 받으려고 돈을 쓸 이유가 없습니다.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from pierrot.config import get_settings  # noqa: E402
from pierrot.tools.universe import UNIVERSE, find_all_stocks  # noqa: E402
from pierrot.verifier.state import Verdict  # noqa: E402

MARK = {
    Verdict.SUPPORTED.value: "[ 근거 있음   ]",
    Verdict.CONTRADICTED.value: "[ 모순        ]",
    Verdict.UNVERIFIABLE.value: "[ 확인 불가   ]",
}

# 자료 범위. 글을 넣기 전에 보여 줍니다.
PRICE_RANGE = "2023-01-02 ~ 2026-11-18"
FINANCIAL_YEARS = "2023 ~ 2026"


def wrap(text: str, prefix: str, width: int = 100) -> str:
    return textwrap.fill(
        text, width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix)
    )


def read_input(args) -> str:
    if args.text:
        return args.text
    if args.path:
        return Path(args.path).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def show_scope() -> None:
    names = ", ".join(stock.name for stock in UNIVERSE)
    print("=" * 100)
    print("대조할 수 있는 범위")
    print("=" * 100)
    print(wrap(names, "  종목   : "))
    print(f"  주가   : {PRICE_RANGE}")
    print(f"  재무   : {FINANCIAL_YEARS} (연간·분기)")
    print("  이 밖의 종목·기간·항목은 '확인 불가' 로 나옵니다.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="내 글을 검증기에 넣어 보기")
    parser.add_argument("path", nargs="?", help="검증할 텍스트 파일")
    parser.add_argument("--text", help="파일 대신 문장을 직접 넘깁니다")
    parser.add_argument("--out", type=Path, help="결과를 JSON 으로 저장")
    parser.add_argument(
        "--fallback", action="store_true",
        help="도구가 판정 못 할 때 의미 비교(NLI)로 한 번 더. 오탐이 늘 수 있습니다",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="회사·시점 채우기 단계를 건너뜁니다(LLM 호출이 줄지만 판정률이 떨어집니다)",
    )
    parser.add_argument("--yes", action="store_true", help="확인 질문 없이 진행")
    args = parser.parse_args()

    text = read_input(args).strip()
    if not text:
        parser.print_help()
        print()
        print("검증할 글이 없습니다. 파일 경로를 주거나 --text 를 쓰십시오.")
        return 1

    show_scope()

    # 돈을 쓰기 전에 훑어봅니다.
    stocks = find_all_stocks(text)
    print("=" * 100)
    print(f"넣을 글: {len(text):,}자")
    print("=" * 100)
    if stocks:
        print(f"  찾은 종목: {', '.join(stock.name for stock in stocks)}")
    else:
        print("  찾은 종목: 없음")
        print()
        print("  이 글에서 유니버스 20종목을 찾지 못했습니다.")
        print("  이대로 넣으면 LLM 호출 비용만 쓰고 대부분 '확인 불가' 가 나옵니다.")
        if not args.yes:
            answer = input("  그래도 진행할까요? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("  중단했습니다.")
                return 0
    print()

    settings = get_settings()
    print("준비 중입니다. 스냅샷과 모델을 올립니다(몇 초 걸립니다)...")
    from pierrot.api.service import VerificationService

    service = VerificationService(settings)

    started = time.time()
    report, usage, extractor_info = service.verify_answer(
        text, allow_fallback=args.fallback, refine=not args.no_refine
    )

    # ---------------- 주장별 ----------------
    print()
    print("=" * 100)
    print(f"주장별 판정  ({len(report.claims)}개)")
    print("=" * 100)
    for index, claim in enumerate(report.claims, 1):
        flip = "  <- 유형 뒤집음" if claim.route_changed else ""
        print()
        print(f"{index:2}. {MARK[claim.verdict]} ({claim.confidence}) {claim.route}{flip}")
        print(wrap(claim.text, "    주장: "))
        print(wrap(claim.reason, "    근거: "))
        for line in claim.evidence[:2]:
            print(wrap(line, "          "))

    # ---------------- 표시된 원문 ----------------
    print()
    print("=" * 100)
    print("원문에 표시")
    print("=" * 100)
    for line in report.annotated_answer().splitlines():
        print(wrap(line, "  ") if line.strip() else "")

    # ---------------- 요약 ----------------
    summary = report.summary
    print()
    print("=" * 100)
    print("요약")
    print("=" * 100)
    print(f"  주장           {summary.total:3}개")
    print(f"    모순         {summary.contradicted:3}개")
    print(f"    근거 있음    {summary.supported:3}개")
    print(f"    확인 불가    {summary.unverifiable:3}개  (전망 {summary.forecast}개 포함)")
    print()
    print(f"  판정의 성격")
    print(f"    계산으로 대조 {summary.calculated:3}개   수치를 직접 비교했습니다")
    print(f"    의미 비교 추정 {summary.inferred:3}개   근거가 무관할 수 있습니다")
    print()
    print(f"  판정 가능 비율  전체 {summary.coverage * 100:5.1f}%  "
          f"전망 제외 {summary.coverage_excluding_forecast * 100:5.1f}%")

    if summary.unverifiable:
        print()
        print("  확인 불가 사유")
        reasons: dict[str, int] = {}
        for claim in report.claims:
            if claim.verdict == Verdict.UNVERIFIABLE.value:
                key = claim.reason[:60]
                reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
            print(f"    {count:2}건  {reason}")

    print()
    print(f"  소요 {time.time() - started:.1f}초   "
          f"이번 비용 ${usage.cost_usd:.5f}   "
          f"이번 달 누적 ${usage.month_spent_usd:.4f} / ${usage.month_budget_usd:.2f}")
    print(f"  분해 {extractor_info['model']} / 프롬프트 {extractor_info['prompt_version']}"
          + ("  (다듬기 적용)" if extractor_info.get("refined") else ""))

    if args.out:
        payload = report.to_dict()
        payload["usage"] = usage.to_dict()
        payload["extractor"] = extractor_info
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
