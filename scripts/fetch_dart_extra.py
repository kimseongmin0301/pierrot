"""배당·지배주주순이익 등 주요계정 바깥의 수치를 모읍니다.

주요계정 스냅샷(`dart_financials.parquet`)과 **같은 모양**의 행을 만들어
별도 파일에 저장합니다. 조회할 때 둘을 합칩니다.

왜 같은 파일에 덧붙이지 않는가: 주요계정 수집(`fetch_dart_snapshot.py`)을
다시 돌리면 덮어써집니다. 따로 두면 각각 독립적으로 다시 모을 수 있습니다.

실행 (DART API 무료):
    .venv/Scripts/python.exe scripts/fetch_dart_extra.py
    .venv/Scripts/python.exe scripts/fetch_dart_extra.py --limit 3
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
from pierrot.tools.dart_extra import fetch_dividends, fetch_extra_accounts  # noqa: E402
from pierrot.tools.financials import COLUMNS  # noqa: E402
from pierrot.tools.universe import UNIVERSE  # noqa: E402

CORP_PATH = PROJECT_ROOT / "data" / "snapshots" / "dart_corp_codes.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "snapshots" / "dart_financials_extra.parquet"

# 재무 스냅샷과 같은 이유로 2022 부터 받습니다(증감 계산의 기준 해).
YEARS = (2022, 2023, 2024, 2025)
# 배당은 사업보고서(연간)에만 있습니다. 분기보고서에는 없습니다.
REPORT_CODE = "11011"


def main() -> int:
    parser = argparse.ArgumentParser(description="배당·추가 계정 수집")
    parser.add_argument("--limit", type=int, default=0, help="종목 몇 개만")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    settings = get_settings()
    if not CORP_PATH.exists():
        print(f"기업코드 스냅샷이 없습니다: {CORP_PATH}")
        return 1

    corps = pd.read_parquet(CORP_PATH)
    corp_code = {str(row["code"]): str(row["corp_code"]) for _, row in corps.iterrows()}

    stocks = list(UNIVERSE)[: args.limit] if args.limit else list(UNIVERSE)

    print("=" * 100)
    print(f"추가 계정 수집 — 종목 {len(stocks)}개 x {len(YEARS)}년")
    print("=" * 100)

    rows: list[dict] = []
    for index, stock in enumerate(stocks, start=1):
        code = corp_code.get(stock.code)
        if not code:
            print(f"{index:3}. {stock.name:<12} 기업코드 없음")
            continue
        started = time.time()
        counts = {"배당": 0, "추가": 0}
        for year in YEARS:
            try:
                dividends = fetch_dividends(
                    code, year, REPORT_CODE, code=stock.code, settings=settings
                )
                extra = fetch_extra_accounts(
                    code, year, REPORT_CODE, code=stock.code, settings=settings
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{index:3}. {stock.name:<12} {year} 실패: {exc}")
                continue
            rows.extend(dividends)
            rows.extend(extra)
            counts["배당"] += len(dividends)
            counts["추가"] += len(extra)
        print(
            f"{index:3}. {stock.name:<12} 배당 {counts['배당']:3}행  "
            f"추가계정 {counts['추가']:3}행  {time.time() - started:4.1f}초"
        )

    if not rows:
        print("\n수집된 행이 없습니다.")
        return 1

    frame = pd.DataFrame(rows, columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    print()
    print("=" * 100)
    print("요약")
    print("=" * 100)
    print(f"  {len(frame):,}행 저장: {args.out} ({args.out.stat().st_size / 1024:.0f}KB)")
    print()
    print("  계정별")
    for account, count in frame["account"].value_counts().items():
        print(f"    {account:16} {count:4}행")
    print()
    print("  표본 (현대차)")
    sample = frame[frame["code"] == "005380"].head(8)
    for _, row in sample.iterrows():
        print(
            f"    {row['bsns_year']} {row['fs_div']:4} {row['account']:16} "
            f"{row['amount'] / 1e8:>12,.1f}억원"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
