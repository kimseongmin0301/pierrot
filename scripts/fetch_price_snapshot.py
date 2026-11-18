"""주가 스냅샷 수집.

FinanceDataReader 로 OHLCV 를 받아 parquet 한 파일로 저장합니다.
이 스크립트만 외부에 접속합니다. 검증 도구와 평가 파이프라인은 저장된 파일만 읽습니다.

왜 분리하는가: FinanceDataReader 는 거래소·포털을 긁는 비공식 라이브러리라
대상 사이트가 바뀌면 멈추고, 수정주가 반영 시점에 따라 같은 날짜의 값이 달라질 수
있습니다. 수집 시점을 한 번으로 고정하고 결과를 커밋해 두면, 라이브러리가 깨져도
평가는 계속 돌아가고 몇 달 뒤에도 같은 숫자가 나옵니다.

실행:
    .venv/Scripts/python.exe scripts/fetch_price_snapshot.py
    .venv/Scripts/python.exe scripts/fetch_price_snapshot.py --start 2023-01-01 --end 2026-11-18
    .venv/Scripts/python.exe scripts/fetch_price_snapshot.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from pierrot.config import get_settings  # noqa: E402
from pierrot.tools.krx_raw import (  # noqa: E402
    KrxCredentialsMissingError,
    fetch_raw_ohlcv,
    merge_raw,
)
from pierrot.tools.price import COLUMNS, SNAPSHOT_FILENAME, PriceSnapshot  # noqa: E402
from pierrot.tools.universe import UNIVERSE  # noqa: E402

DEFAULT_START = "2023-01-01"

# FinanceDataReader 가 돌려주는 컬럼 -> 스냅샷 컬럼
RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Change": "change",
}


def fetch_one(code: str, start: str, end: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader(code, start, end)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    frame = frame.rename(columns=RENAME).reset_index()
    frame["date"] = pd.to_datetime(frame["Date"]).dt.date
    frame["code"] = code

    for column in ("open", "high", "low", "close", "volume", "change"):
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[COLUMNS]


def attach_raw_prices(merged, start: str, end: str, settings):
    """수정주가 스냅샷에 원주가 컬럼을 붙입니다.

    실패해도 수집 전체를 중단하지 않습니다. 원주가가 없으면 보정된 종목의
    값 주장만 판정 보류로 남고, 나머지는 그대로 동작합니다.
    """
    print()
    print("원주가 수집 (KRX 정보데이터시스템)")
    if not settings.has_krx_credentials:
        print("  건너뜀: KRX_ID / KRX_PW 가 .env 에 없습니다.")
        print("  https://data.krx.co.kr 무료 가입 후 .env 에 넣으면 원주가도 받습니다.")
        return merged

    raw_frames = []
    failed = []
    for index, stock in enumerate(UNIVERSE, start=1):
        try:
            result = fetch_raw_ohlcv(stock.code, start, end, settings=settings)
        except KrxCredentialsMissingError as exc:
            print(f"  중단: {exc}")
            return merged

        if result.ok:
            raw_frames.append(result.frame)
            print(f"  [{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} {len(result.frame):5}행")
        else:
            failed.append((stock.code, result.error))
            print(f"  [{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} 실패: {result.error[:60]}")

    if not raw_frames:
        print("  원주가를 하나도 받지 못했습니다. 수정주가만 저장합니다.")
        return merged

    merged = merge_raw(merged, raw_frames)
    filled = merged["close_raw"].notna().sum()
    print(f"  원주가 채워진 행: {filled:,} / {len(merged):,} ({filled / len(merged):.1%})")
    if failed:
        print(f"  실패 {len(failed)}종목: {', '.join(code for code, _ in failed)}")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="주가 스냅샷 수집")
    parser.add_argument("--start", default=DEFAULT_START, help="시작일 (기본 2023-01-01)")
    parser.add_argument("--end", default=date.today().isoformat(), help="종료일 (기본 오늘)")
    parser.add_argument("--force", action="store_true", help="기존 파일을 덮어씁니다")
    parser.add_argument("--out", type=Path, default=None, help="저장 경로")
    parser.add_argument("--skip-raw", action="store_true",
                        help="원주가 수집을 건너뜁니다 (KRX 계정 불필요)")
    args = parser.parse_args()

    settings = get_settings()
    out = args.out or (settings.snapshot_dir / SNAPSHOT_FILENAME)

    if out.exists() and not args.force:
        snapshot = PriceSnapshot.load(out)
        span = snapshot.date_range()
        print(f"이미 존재합니다: {out}")
        print(f"  {len(snapshot):,}행, {len(snapshot.codes())}종목, {span[0]} ~ {span[1]}")
        print("다시 받으려면 --force 를 붙여 주십시오.")
        return 0

    print(f"수집 범위: {args.start} ~ {args.end}")
    print(f"종목 수  : {len(UNIVERSE)}")
    print(f"저장 경로: {out}")
    print()

    frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []

    for index, stock in enumerate(UNIVERSE, start=1):
        started = time.perf_counter()
        try:
            frame = fetch_one(stock.code, args.start, args.end)
        except Exception as exc:
            failed.append((stock.code, f"{type(exc).__name__}: {exc}"))
            print(f"[{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} 실패: {exc}")
            continue

        elapsed = time.perf_counter() - started
        if frame.empty:
            failed.append((stock.code, "데이터 없음"))
            print(f"[{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} 데이터 없음")
            continue

        frames.append(frame)
        print(f"[{index:2}/{len(UNIVERSE)}] {stock.code} {stock.name:16} "
              f"{len(frame):5}행  {frame['date'].min()} ~ {frame['date'].max()}  "
              f"{elapsed:4.1f}초")

    if not frames:
        print("\n수집된 데이터가 없습니다.")
        return 1

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["close"])
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)

    # 원주가(당시 실제 호가)를 KRX 에서 함께 받습니다.
    # 값 주장("4월 9일 71,800원")은 원주가로, 등락률 주장은 수정주가로 판정합니다.
    if not args.skip_raw:
        merged = attach_raw_prices(merged, args.start, args.end, settings)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    size_kb = out.stat().st_size / 1024
    print()
    print(f"저장 완료: {out} ({size_kb:,.0f} KB)")
    print(f"  {len(merged):,}행 / {merged['code'].nunique()}종목 / "
          f"{merged['date'].min()} ~ {merged['date'].max()}")

    if failed:
        print()
        print(f"실패한 종목 {len(failed)}개:")
        for code, reason in failed:
            print(f"  {code}: {reason}")

    # 저장한 파일을 다시 읽어 실제로 쓸 수 있는지 확인합니다.
    snapshot = PriceSnapshot.load(out)
    print()
    print("검증 (저장된 파일을 다시 읽음):")
    if "close_raw" in merged.columns:
        check = PriceSnapshot.load(out)
        adjusted_codes = check.adjusted_codes()
        for code in adjusted_codes:
            name = next((s.name for s in check.stocks() if s.code == code), code)
            status = "원주가 있음" if check.has_raw(code) else "원주가 없음"
            print(f"  보정 종목 {name}: {status}")
    for stock in snapshot.stocks()[:3]:
        print(f"  {snapshot.describe_stock(stock.code)}")
    print(f"  ... 총 {len(snapshot.codes())}종목")
    return 0


if __name__ == "__main__":
    sys.exit(main())
