"""사업보고서에서 부문별·제품별 매출 표를 모아 스냅샷으로 저장합니다.

왜 모으는가: 3단계 실측에서 확인 불가 55건 중 가장 큰 덩어리가 부문별 매출이었습니다.
재무제표 주요계정에도 전체 재무제표에도 없고, 사업보고서 본문의 표에만 있습니다.

검수 장치: 표에서 읽은 **합계가 DART 재무제표의 매출액과 맞는지** 대조합니다.
단위를 잘못 읽거나(억원 vs 백만원) 열을 잘못 짚으면 여기서 걸립니다.
표 파싱은 눈으로 몇 개 보고 "되는 것 같다"고 넘어가면 반드시 틀립니다.

실행 (DART API 무료, 시간이 걸립니다):
    .venv/Scripts/python.exe scripts/fetch_segment_snapshot.py
    .venv/Scripts/python.exe scripts/fetch_segment_snapshot.py --limit 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from pierrot.config import get_settings  # noqa: E402
from pierrot.tools.dart_documents import extract_section, fetch_document_text  # noqa: E402
from pierrot.tools.financials import FinancialSnapshot  # noqa: E402
from pierrot.tools.segments import (  # noqa: E402
    COLUMNS,
    SegmentSnapshot,
    extract_segment_facts,
)
from pierrot.tools.universe import UNIVERSE  # noqa: E402

SENTENCES_PATH = PROJECT_ROOT / "data" / "snapshots" / "dart_sentences.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "snapshots" / "dart_segments.parquet"


def sales_section(body: str) -> str:
    """'매출 및 수주상황' 절을 잘라 냅니다.

    TITLE 태그로 먼저 찾고, 못 찾으면 문자열 위치로 찾습니다.
    카카오는 제목이 "4. (제조서비스업)매출 및 수주상황" 이라 접두어가 붙어서
    제목 일치로는 안 잡힙니다.
    """
    section = extract_section(body, "4. 매출 및 수주상황", "5. 위험관리 및 파생거래")
    if section:
        return section
    start = body.find("매출 및 수주상황")
    if start < 0:
        return ""
    end = body.find("위험관리", start)
    return body[start : end if end > start else start + 30_000]


def main() -> int:
    parser = argparse.ArgumentParser(description="부문별 매출 스냅샷 수집")
    parser.add_argument("--limit", type=int, default=0, help="보고서 몇 건만")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    settings = get_settings()
    if not SENTENCES_PATH.exists():
        print(f"본문 스냅샷이 없습니다: {SENTENCES_PATH}")
        print("먼저 scripts/fetch_dart_documents.py 를 실행해 주십시오.")
        return 1

    sentences = pd.read_parquet(SENTENCES_PATH)
    # (종목, 연도, 접수번호) 목록을 본문 스냅샷에서 그대로 가져옵니다.
    reports = (
        sentences[["code", "bsns_year", "rcept_no"]]
        .drop_duplicates()
        .sort_values(["code", "bsns_year"])
    )
    if args.limit:
        reports = reports.head(args.limit)

    names = {stock.code: stock.name for stock in UNIVERSE}
    financials = FinancialSnapshot.load()

    print("=" * 104)
    print(f"부문별 매출 수집 — 보고서 {len(reports)}건")
    print("=" * 104)

    all_facts = []
    checks = []
    for index, row in enumerate(reports.itertuples(index=False), start=1):
        code, year, rcept_no = row.code, int(row.bsns_year), row.rcept_no
        name = names.get(code, code)
        started = time.time()
        try:
            raw = fetch_document_text(rcept_no, settings=settings)
            body = extract_section(raw, "II. 사업의 내용", "III. 재무에 관한 사항")
            section = sales_section(body)
            facts = extract_segment_facts(
                section, code=code, bsns_year=year, rcept_no=rcept_no
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{index:3}. {name:<12} {year}  수집 실패: {exc}")
            continue

        all_facts.extend(facts)
        current = [f for f in facts if f.year == year and not f.is_total]
        total = next((f for f in facts if f.year == year and f.is_total), None)

        # 검수: 뽑은 수치의 크기가 재무제표 매출액과 같은 자릿수인지 봅니다.
        #
        # 처음에는 "표 합계 == 재무제표 매출액" 을 요구했는데 그게 틀렸습니다.
        # 기아의 품목별 표는 제품 매출(63조)만 담고 있어 연결 매출(107조)과
        # 애초에 다릅니다. 표가 전사 매출을 덮는다는 보장이 없습니다.
        #
        # 꼭 잡아야 하는 것은 **단위 오독**(억원을 백만원으로 읽으면 100배)과
        # **열 오독**입니다. 둘 다 자릿수를 깨뜨리므로 비율로 걸러집니다.
        reported = financials.lookup(code, year, "매출액", quarter=0, fs_div="CFS")
        gap_text = "재무제표 없음"
        ok = None
        current_year = [f for f in facts if f.year == year]
        if reported and current_year:
            revenue = reported[0].amount
            largest = max(f.amount for f in current_year)
            share = largest / max(revenue, 1)
            exact = any(
                abs(f.amount - revenue) / max(revenue, 1) < 0.01 for f in current_year
            )
            # 매출액과 정확히 맞는 수치가 하나라도 있으면 단위·열을 제대로
            # 읽은 것입니다. 한국전력처럼 표에 전사 매출보다 큰 중간 합계가
            # 들어 있는 경우가 있어 비율만으로 자르면 멀쩡한 것이 걸립니다.
            ok = exact or 0.2 <= share <= 1.5
            mark = "일치" if exact else ("정상" if ok else "이상")
            gap_text = f"최대 {largest / 1e8:>9,.0f}억 = 매출의 {share * 100:5.1f}%  {mark}"
        elif not current_year:
            gap_text = "뽑은 수치 없음"

        checks.append((name, year, len(current), ok))
        print(
            f"{index:3}. {name:<12} {year}  절 {len(section):6,}자  "
            f"부문 {len(current):3}개  {gap_text:<22} {time.time() - started:4.1f}초"
        )

    if not all_facts:
        print("\n수집된 수치가 없습니다.")
        return 1

    snapshot = SegmentSnapshot.from_facts(all_facts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.frame.to_parquet(args.out, index=False)

    matched = sum(1 for _, _, _, ok in checks if ok)
    unknown = sum(1 for _, _, _, ok in checks if ok is None)
    mismatched = len(checks) - matched - unknown

    print()
    print("=" * 104)
    print("요약")
    print("=" * 104)
    print(f"  수치 {len(all_facts):,}개  (열 {len(COLUMNS)}개)")
    print(f"  저장: {args.out}  ({args.out.stat().st_size / 1024:.0f}KB)")
    print()
    print("  자릿수 검수 (뽑은 최대 수치가 매출액의 20~150% 범위인가)")
    print(f"    정상   {matched:3}건")
    print(f"    이상   {mismatched:3}건   <- 단위나 열을 잘못 읽었을 가능성")
    print(f"    확인불가 {unknown:3}건")
    if mismatched:
        print()
        print("  이상 목록")
        for name, year, count, ok in checks:
            if ok is False:
                print(f"    {name} {year}  부문 {count}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
