"""여러 글을 한 번에 검증합니다.

한 줄에 한 건씩 적힌 파일을 받습니다. 앞에 번호가 붙어 있으면 떼어 냅니다.

    .venv/Scripts/python.exe scripts/verify_batch.py cases.txt
    .venv/Scripts/python.exe scripts/verify_batch.py cases.txt --out result.json

`verify_text.py` 는 한 건씩 보기 좋게 보여 주지만, 여러 건을 볼 때는
**종합 표**가 필요합니다. 어느 건이 걸렸고 어느 건이 통과했는지 한눈에
보여야 무엇을 고칠지 정할 수 있습니다.

무거운 것(스냅샷, NLI 모델, 그래프)은 **한 번만** 올려 재사용합니다.
건마다 올리면 20건에 몇 분이 더 걸립니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from pierrot.config import get_settings  # noqa: E402
from pierrot.verifier.state import Verdict  # noqa: E402

MARK = {
    Verdict.SUPPORTED.value: "근거 있음",
    Verdict.CONTRADICTED.value: "모순",
    Verdict.UNVERIFIABLE.value: "확인 불가",
}

# 줄 앞의 번호. "01. ", "1) ", "- " 를 떼어 냅니다.
_NUMBER = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*])\s*")


def load_cases(path: Path) -> list[tuple[str, str]]:
    """(라벨, 본문) 목록. 라벨은 원래 붙어 있던 번호입니다."""
    cases: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _NUMBER.match(stripped)
        label = match.group(0).strip() if match else str(len(cases) + 1)
        cases.append((label.rstrip(".) "), _NUMBER.sub("", stripped)))
    return cases


def wrap(text: str, prefix: str, width: int = 104) -> str:
    return textwrap.fill(
        text, width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="여러 글을 한 번에 검증")
    parser.add_argument("path", type=Path, help="한 줄에 한 건씩 적힌 파일")
    parser.add_argument("--out", type=Path, help="결과를 JSON 으로 저장")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="건별 상세를 줄입니다")
    args = parser.parse_args()

    cases = load_cases(args.path)
    if not cases:
        print(f"읽을 내용이 없습니다: {args.path}")
        return 1

    settings = get_settings()
    print("=" * 110)
    print(f"일괄 검증 — {len(cases)}건  ({args.path.name})")
    print("=" * 110)
    print("준비 중입니다. 스냅샷과 모델을 올립니다...")

    from pierrot.api.service import VerificationService

    service = VerificationService(settings)

    started = time.time()
    rows: list[dict] = []
    for label, text in cases:
        report, usage, _ = service.verify_answer(text, allow_fallback=args.fallback)
        summary = report.summary
        rows.append(
            {
                "label": label,
                "text": text,
                "claims": [claim.to_dict() for claim in report.claims],
                "summary": summary.to_dict(),
                "cost_usd": usage.cost_usd,
            }
        )

        verdicts = [claim.verdict for claim in report.claims]
        head = (
            f"[{label}] 주장 {summary.total}  "
            f"모순 {summary.contradicted}  근거있음 {summary.supported}  "
            f"확인불가 {summary.unverifiable}"
        )
        print()
        print("-" * 110)
        print(head)
        print(wrap(text, "  글: "))
        if args.quiet:
            continue
        for index, claim in enumerate(report.claims, 1):
            print(f"    {index}. [{MARK[claim.verdict]:9}] ({claim.confidence}) {claim.route}")
            print(wrap(claim.text, "       주장: "))
            print(wrap(claim.reason, "       근거: "))

    elapsed = time.time() - started

    # ---------------- 종합 ----------------
    print()
    print("=" * 110)
    print("종합")
    print("=" * 110)
    print(f"  {'번호':<5}{'주장':>5}{'모순':>5}{'근거':>5}{'불가':>5}  글")
    total = {"claims": 0, "contradicted": 0, "supported": 0, "unverifiable": 0}
    for row in rows:
        s = row["summary"]
        total["claims"] += s["total"]
        total["contradicted"] += s["contradicted"]
        total["supported"] += s["supported"]
        total["unverifiable"] += s["unverifiable"]
        flag = "<-" if s["contradicted"] else "  "
        print(
            f"  {row['label']:<5}{s['total']:>5}{s['contradicted']:>5}"
            f"{s['supported']:>5}{s['unverifiable']:>5} {flag} {row['text'][:58]}"
        )
    print(
        f"  {'합계':<5}{total['claims']:>5}{total['contradicted']:>5}"
        f"{total['supported']:>5}{total['unverifiable']:>5}"
    )

    judged = total["contradicted"] + total["supported"]
    print()
    print(f"  판정 가능 비율 {judged}/{total['claims']} = "
          f"{judged / max(total['claims'], 1) * 100:.1f}%")
    print(f"  소요 {elapsed:.1f}초   비용 ${sum(r['cost_usd'] for r in rows):.5f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
