"""DART 재무 스냅샷 수집.

유니버스 20종목의 주요계정을 연도·분기별로 받아 parquet 한 파일로 저장합니다.
이 스크립트만 DART 에 접속합니다. 검증은 저장된 파일만 읽습니다.

호출 수: 20종목 x 연도 수 x 보고서 4종. 2023~2026 이면 320회입니다.
DART 는 키당 일일 호출 한도가 있으므로 한 번 받아 커밋해 두고 재사용합니다.

실행:
    .venv/Scripts/python.exe scripts/fetch_dart_snapshot.py
    .venv/Scripts/python.exe scripts/fetch_dart_snapshot.py --from-year 2023 --to-year 2026
    .venv/Scripts/python.exe scripts/fetch_dart_snapshot.py --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from pierrot.config import get_settings  # noqa: E402
from pierrot.tools.dart import (  # noqa: E402
    REPORT_CODES,
    DartError,
    DartKeyMissingError,
    fetch_corp_codes,
    fetch_financials,
)
from pierrot.tools.financials import (  # noqa: E402
    COLUMNS,
    SNAPSHOT_FILENAME,
    FinancialSnapshot,
)
from pierrot.tools.universe import UNIVERSE  # noqa: E402

DEFAULT_FROM_YEAR = 2023
CORP_MAP_FILENAME = "dart_corp_codes.parquet"


def resolve_corp_codes(settings, snapshot_dir: Path, force: bool) -> dict[str, str]:
    """종목코드 -> DART 고유번호 매핑.

    전체 목록이 3.5MB ZIP 으로 오므로 한 번 받아 유니버스만 추려 저장합니다.
    """
    path = snapshot_dir / CORP_MAP_FILENAME
    if path.exists() and not force:
        frame = pd.read_parquet(path)
        print(f"고유번호 매핑: 기존 파일 사용 ({len(frame)}종목, {path.name})")
        return dict(zip(frame["code"], frame["corp_code"]))

    print("고유번호 매핑: DART 에서 전체 목록을 받는 중...")
    wanted = {stock.code for stock in UNIVERSE}
    entries = [c for c in fetch_corp_codes(settings=settings) if c.stock_code in wanted]

    mapping = {entry.stock_code: entry.corp_code for entry in entries}
    missing = wanted - set(mapping)
    if missing:
        print(f"  경고: 고유번호를 못 찾은 종목 {sorted(missing)}")

    frame = pd.DataFrame(
        [{"code": e.stock_code, "corp_code": e.corp_code, "corp_name": e.corp_name}
         for e in entries]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"  {len(mapping)}종목 매핑 저장: {path.name}")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="DART 재무 스냅샷 수집")
    parser.add_argument("--from-year", type=int, default=DEFAULT_FROM_YEAR)
    parser.add_argument("--to-year", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true", help="기존 파일을 덮어씁니다")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    out = args.out or (settings.snapshot_dir / SNAPSHOT_FILENAME)

    if out.exists() and not args.force:
        snapshot = FinancialSnapshot.load(out)
        print(f"이미 존재합니다: {out}")
        print(f"  {snapshot.describe_coverage()}")
        print("다시 받으려면 --force 를 붙여 주십시오.")
        return 0

    years = list(range(args.from_year, args.to_year + 1))
    total_calls = len(UNIVERSE) * len(years) * len(REPORT_CODES)
    print(f"수집 범위 : {years[0]} ~ {years[-1]}년 / {len(UNIVERSE)}종목 / 보고서 {len(REPORT_CODES)}종")
    print(f"예상 호출 : {total_calls}회")
    print(f"저장 경로 : {out}")
    print()

    try:
        corp_map = resolve_corp_codes(settings, settings.snapshot_dir, args.force)
    except DartKeyMissingError as exc:
        print(exc)
        return 1

    print()
    rows: list[dict] = []
    empty: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, stock in enumerate(UNIVERSE, start=1):
        corp_code = corp_map.get(stock.code)
        if not corp_code:
            failed.append((stock.code, "고유번호 없음"))
            print(f"[{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} 고유번호 없음")
            continue

        collected = 0
        for year in years:
            for reprt_code in REPORT_CODES:
                try:
                    fetched = fetch_financials(
                        corp_code, year, reprt_code, settings=settings
                    )
                except DartError as exc:
                    failed.append((f"{stock.code}/{year}/{reprt_code}", str(exc)))
                    continue
                if not fetched:
                    empty.append(f"{stock.code}/{year}/{REPORT_CODES[reprt_code]['label']}")
                    continue
                for row in fetched:
                    row["code"] = stock.code
                    rows.append(row)
                collected += len(fetched)

        print(f"[{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} {collected:5}행")

    if not rows:
        print("\n수집된 데이터가 없습니다.")
        return 1

    frame = pd.DataFrame(rows)[COLUMNS]
    frame = frame.sort_values(["code", "bsns_year", "quarter", "fs_div", "account"])
    frame = frame.reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    print()
    print(f"저장 완료: {out} ({out.stat().st_size / 1024:,.0f} KB)")

    snapshot = FinancialSnapshot.load(out)
    print(f"  {snapshot.describe_coverage()}")
    print(f"  계정: {', '.join(snapshot.accounts())}")

    if empty:
        print()
        print(f"데이터 없음 {len(empty)}건 (아직 공시되지 않은 기간은 정상입니다)")
        print(f"  예: {', '.join(empty[:6])}")
    if failed:
        print()
        print(f"실패 {len(failed)}건:")
        for target, reason in failed[:10]:
            print(f"  {target}: {reason[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
