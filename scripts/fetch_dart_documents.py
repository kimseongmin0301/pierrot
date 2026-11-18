"""DART 공시 본문에서 서술 문장을 수집합니다.

사업보고서(11011)의 "사업의 내용"과 "이사의 경영진단" 절만 잘라 문장으로 저장합니다.
NARRATIVE 주장을 판정하려면 수치가 아니라 서술이 필요하기 때문입니다.

본문 XML 은 한 건에 약 6MB 이고 대부분이 재무제표 표와 주석입니다.
그래서 받는 즉시 필요한 절만 추출하고 문장만 저장합니다.

실행:
    .venv/Scripts/python.exe scripts/fetch_dart_documents.py
    .venv/Scripts/python.exe scripts/fetch_dart_documents.py --from-year 2023 --to-year 2025
    .venv/Scripts/python.exe scripts/fetch_dart_documents.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from pierrot.config import get_settings  # noqa: E402
from pierrot.tools.dart_documents import (  # noqa: E402
    document_sentences,
    fetch_document_text,
)
from pierrot.tools.financials import SNAPSHOT_FILENAME as FIN_FILENAME  # noqa: E402
from pierrot.tools.universe import UNIVERSE, get_stock  # noqa: E402

OUTPUT_FILENAME = "dart_sentences.parquet"
ANNUAL_REPORT = "11011"
COLUMNS = ["code", "bsns_year", "rcept_no", "section", "sentence"]


def annual_receipts(snapshot_dir: Path, years: range) -> list[tuple[str, int, str]]:
    """재무 스냅샷에서 사업보고서 접수번호를 꺼냅니다.

    이미 받아 둔 스냅샷에 접수번호가 들어 있으므로 따로 조회하지 않습니다.
    """
    frame = pd.read_parquet(snapshot_dir / FIN_FILENAME)
    frame = frame[
        (frame["reprt_code"] == ANNUAL_REPORT)
        & (frame["bsns_year"].isin(list(years)))
        & (frame["rcept_no"].astype(str).str.len() > 0)
    ]
    frame = frame.drop_duplicates(subset=["code", "bsns_year"])
    return [
        (str(row["code"]), int(row["bsns_year"]), str(row["rcept_no"]))
        for _, row in frame.iterrows()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="DART 공시 본문 문장 수집")
    parser.add_argument("--from-year", type=int, default=2023)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    out = args.out or (settings.snapshot_dir / OUTPUT_FILENAME)

    if out.exists() and not args.force:
        frame = pd.read_parquet(out)
        print(f"이미 존재합니다: {out}")
        print(f"  {len(frame):,}문장 / {frame['code'].nunique()}종목")
        print("다시 받으려면 --force 를 붙여 주십시오.")
        return 0

    years = range(args.from_year, args.to_year + 1)
    targets = annual_receipts(settings.snapshot_dir, years)
    if not targets:
        print("대상 접수번호가 없습니다. 먼저 fetch_dart_snapshot.py 를 실행해 주십시오.")
        return 1

    print(f"대상    : 사업보고서 {len(targets)}건 ({args.from_year}~{args.to_year}년)")
    print(f"저장 경로: {out}")
    print()

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    for index, (code, year, rcept_no) in enumerate(targets, start=1):
        stock = get_stock(code)
        name = stock.name if stock else code
        started = time.perf_counter()
        try:
            raw = fetch_document_text(rcept_no, settings=settings)
        except Exception as exc:
            failed.append((f"{code}/{year}", f"{type(exc).__name__}: {exc}"))
            print(f"[{index:2}/{len(targets)}] {name:16} {year} 실패: {str(exc)[:50]}")
            continue

        sentences = document_sentences(code, year, rcept_no, raw)
        rows.extend(
            {
                "code": item.code,
                "bsns_year": item.bsns_year,
                "rcept_no": item.rcept_no,
                "section": item.section,
                "sentence": item.sentence,
            }
            for item in sentences
        )
        elapsed = time.perf_counter() - started
        print(f"[{index:2}/{len(targets)}] {name:16} {year} "
              f"본문 {len(raw) / 1_000_000:4.1f}MB -> 문장 {len(sentences):4}개  {elapsed:4.1f}초")

    if not rows:
        print("\n수집된 문장이 없습니다.")
        return 1

    frame = pd.DataFrame(rows)[COLUMNS]
    frame = frame.sort_values(["code", "bsns_year", "section"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    print()
    print(f"저장 완료: {out} ({out.stat().st_size / 1024:,.0f} KB)")
    print(f"  {len(frame):,}문장 / {frame['code'].nunique()}종목 / "
          f"{frame['bsns_year'].min()}~{frame['bsns_year'].max()}년")
    print(f"  절별: {frame['section'].value_counts().to_dict()}")
    print(f"  종목당 평균 {len(frame) / frame['code'].nunique():.0f}문장")

    missing = {stock.code for stock in UNIVERSE} - set(frame["code"])
    if missing:
        print(f"  문장이 없는 종목: {sorted(missing)}")
    if failed:
        print()
        print(f"실패 {len(failed)}건:")
        for target, reason in failed[:10]:
            print(f"  {target}: {reason[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
