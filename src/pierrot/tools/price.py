"""주가 검증 도구.

PRICE 유형 주장("주가가 5% 상승했다", "7만원에 마감했다")을 확인합니다.

## 설계 원칙 두 가지

1. **LLM에게 묻지 않습니다.** 등락률은 파이썬이 종가로 직접 계산합니다.
   이것이 도구 기반 검증이 NLI나 LLM-as-judge보다 나은 이유의 전부입니다.
   계산 결과는 재현되고, 설명할 수 있고, 틀리면 어디서 틀렸는지 보입니다.

2. **검증 중에는 외부를 조회하지 않습니다.** 미리 받아 둔 로컬 스냅샷만 읽습니다.
   FinanceDataReader 는 거래소·포털을 긁는 비공식 라이브러리라 언제든 멈출 수
   있고, 수정주가 반영 시점에 따라 값이 달라집니다. 외부 의존성을 재현성 경계
   밖으로 밀어내기 위해 수집과 검증을 분리합니다.
   수집은 scripts/fetch_price_snapshot.py 가 담당합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from ..config import get_settings
from .universe import UNIVERSE, Stock, get_stock

SNAPSHOT_FILENAME = "kospi_ohlcv.parquet"

COLUMNS = ["code", "date", "open", "high", "low", "close", "volume", "change"]

RAW_COLUMNS = ["open_raw", "high_raw", "low_raw", "close_raw"]
"""원주가(수정되지 않은 당시 실제 호가) 컬럼. 선택 사항입니다.

있으면 "얼마였다"는 값 주장을 원주가로 판정하고, 없으면 수정주가로 보정된 종목의
값 주장은 판정하지 않습니다. 등락률 주장은 어느 경우에도 수정주가를 씁니다.
"""


class SnapshotMissingError(FileNotFoundError):
    """스냅샷 파일이 없습니다."""


def default_snapshot_path() -> Path:
    return get_settings().snapshot_dir / SNAPSHOT_FILENAME


def tick_size(price: float) -> int:
    """KRX 호가 단위. 2023년 1월 개편 기준입니다.

    실제로 거래된 가격은 반드시 호가 단위의 배수입니다. 그렇지 않은 값이 섞여
    있으면 수정주가로 보정된 것입니다.
    """
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def is_tick_valid(price: float) -> bool:
    return price > 0 and float(price) % tick_size(price) == 0


def to_date(value: str | date | datetime | pd.Timestamp) -> date:
    """여러 형태의 날짜 표기를 date 로 맞춥니다."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(str(value)).date()


@dataclass(frozen=True)
class PriceRow:
    """어느 종목의 하루치 시세."""

    code: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    change: float
    """전일 종가 대비 등락률. 0.05 이면 5% 상승입니다."""

    open_raw: float | None = None
    high_raw: float | None = None
    low_raw: float | None = None
    close_raw: float | None = None
    """원주가. 수집하지 않았으면 None 입니다."""

    @property
    def has_raw(self) -> bool:
        return self.close_raw is not None

    def level(self, field: str, *, prefer_raw: bool = True) -> float:
        """값 주장 판정에 쓸 가격을 돌려줍니다.

        원주가가 있으면 그쪽을 씁니다. LLM 이 말하는 "그날 얼마였다"는 당시 호가를
        가리키므로, 보정된 시세와 비교하면 맞는 주장을 틀렸다고 판정하게 됩니다.
        """
        if prefer_raw:
            value = getattr(self, f"{field}_raw", None)
            if value is not None:
                return value
        return float(getattr(self, field))


@dataclass(frozen=True)
class ChangeResult:
    """두 시점 사이의 등락률 계산 결과.

    검증 결과를 사람에게 설명하려면 "몇 퍼센트"만으로는 부족합니다.
    어느 날 얼마에서 어느 날 얼마가 되었는지가 함께 있어야 근거가 됩니다.
    """

    code: str
    start: PriceRow
    end: PriceRow

    @property
    def rate(self) -> float:
        """시작 종가 대비 끝 종가의 변화율. 0.05 이면 5% 상승입니다."""
        return self.end.close / self.start.close - 1.0

    @property
    def percent(self) -> float:
        return self.rate * 100.0

    def describe(self) -> str:
        return (
            f"{self.code} {self.start.date} 종가 {self.start.close:,.0f}원 -> "
            f"{self.end.date} 종가 {self.end.close:,.0f}원 "
            f"({self.percent:+.2f}%)"
        )


class PriceSnapshot:
    """로컬 OHLCV 스냅샷 조회기."""

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [column for column in COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"스냅샷에 없는 컬럼: {missing}")
        present_raw = [c for c in RAW_COLUMNS if c in frame.columns]
        self._raw_columns = present_raw if len(present_raw) == len(RAW_COLUMNS) else []
        frame = frame[COLUMNS + self._raw_columns].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        self._frame = frame.sort_values(["code", "date"]).reset_index(drop=True)
        self._by_code = {
            code: group.reset_index(drop=True)
            for code, group in self._frame.groupby("code", sort=False)
        }

    # -- 생성 -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PriceSnapshot":
        path = Path(path) if path else default_snapshot_path()
        if not path.exists():
            raise SnapshotMissingError(
                f"주가 스냅샷이 없습니다: {path}\n"
                "먼저 scripts/fetch_price_snapshot.py 를 실행해 주십시오."
            )
        return cls(pd.read_parquet(path))

    @classmethod
    def from_records(cls, records: list[dict]) -> "PriceSnapshot":
        """테스트에서 손으로 만든 데이터로 스냅샷을 만듭니다."""
        has_raw = any(column in record for record in records for column in RAW_COLUMNS)
        columns = COLUMNS + (RAW_COLUMNS if has_raw else [])
        return cls(pd.DataFrame(records, columns=columns))

    # -- 기본 정보 ---------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def codes(self) -> list[str]:
        return list(self._by_code)

    def stocks(self) -> list[Stock]:
        return [s for s in UNIVERSE if s.code in self._by_code]

    def date_range(self, code: str | None = None) -> tuple[date, date] | None:
        frame = self._by_code.get(code) if code else self._frame
        if frame is None or frame.empty:
            return None
        return (frame["date"].min(), frame["date"].max())

    def __len__(self) -> int:
        return len(self._frame)

    # -- 수정주가 탐지 -----------------------------------------------------

    def has_raw(self, code: str | None = None) -> bool:
        """원주가 컬럼이 채워져 있는지."""
        if not self._raw_columns:
            return False
        frame = self._rows_for(code) if code else self._frame
        return bool(frame["close_raw"].notna().all())

    def tick_validity(self, code: str) -> float:
        """종가가 KRX 호가 단위에 맞는 비율. 1.0 이면 원본 호가입니다."""
        frame = self._rows_for(code)
        if frame.empty:
            return 0.0
        return float(frame["close"].map(is_tick_valid).mean())

    def is_adjusted(self, code: str, *, threshold: float = 0.95) -> bool:
        """이 종목의 시세가 수정주가로 보정되었는지.

        액면분할, 합병, 인적분할 같은 일이 있으면 과거 시세가 보정되어
        **그때 실제로 거래된 가격과 달라집니다.**

        이것이 왜 문제인가: LLM이 "2024년 7월 26일 주가는 85만원이었다"고 말할 때
        그 값은 당시 호가를 가리킵니다. 보정된 시세와 비교하면 맞는 주장을
        틀렸다고 판정하게 됩니다. 오탐은 이 프로젝트가 가장 피해야 할 오류입니다.

        반대로 등락률 주장에는 수정주가가 **맞습니다.** 분할 전후를 비교 가능하게
        만드는 것이 수정주가의 목적이기 때문입니다.

        목록을 코드에 적지 않고 매번 계산하는 이유: 스냅샷을 다시 받으면 보정 대상이
        달라질 수 있는데, 하드코딩하면 조용히 틀린 채로 남습니다.
        """
        return self.tick_validity(code) < threshold

    def adjusted_codes(self, *, threshold: float = 0.95) -> list[str]:
        """수정주가로 보정된 종목코드 목록."""
        return [code for code in self._by_code if self.is_adjusted(code, threshold=threshold)]

    # -- 조회 -------------------------------------------------------------

    def _rows_for(self, code: str) -> pd.DataFrame:
        frame = self._by_code.get(code)
        if frame is None:
            raise KeyError(
                f"스냅샷에 없는 종목입니다: {code}. "
                f"수집 범위는 tools/universe.py 의 UNIVERSE 입니다."
            )
        return frame

    @staticmethod
    def _to_row(record: pd.Series) -> PriceRow:
        def raw(name: str) -> float | None:
            if name not in record.index:
                return None
            value = record[name]
            return None if pd.isna(value) else float(value)

        return PriceRow(
            code=str(record["code"]),
            date=to_date(record["date"]),
            open=float(record["open"]),
            high=float(record["high"]),
            low=float(record["low"]),
            close=float(record["close"]),
            volume=float(record["volume"]),
            change=float(record["change"]),
            open_raw=raw("open_raw"),
            high_raw=raw("high_raw"),
            low_raw=raw("low_raw"),
            close_raw=raw("close_raw"),
        )

    def row_on(self, code: str, day: str | date) -> PriceRow | None:
        """그날의 시세. 휴장일이면 None 입니다.

        None 을 돌려주는 것이 중요합니다. 임의로 가까운 날을 끌어다 쓰면
        "10월 9일에 5% 올랐다"는 주장을 다른 날 데이터로 판정하게 됩니다.
        가까운 거래일이 필요하면 nearest_row 를 명시적으로 쓰십시오.
        """
        day = to_date(day)
        frame = self._rows_for(code)
        hit = frame[frame["date"] == day]
        return None if hit.empty else self._to_row(hit.iloc[0])

    def nearest_row(
        self,
        code: str,
        day: str | date,
        direction: Literal["backward", "forward"] = "backward",
    ) -> PriceRow | None:
        """그날 또는 그 이전(이후)의 가장 가까운 거래일 시세."""
        day = to_date(day)
        frame = self._rows_for(code)
        if direction == "backward":
            hit = frame[frame["date"] <= day]
            return None if hit.empty else self._to_row(hit.iloc[-1])
        hit = frame[frame["date"] >= day]
        return None if hit.empty else self._to_row(hit.iloc[0])

    def rows_between(self, code: str, start: str | date, end: str | date) -> list[PriceRow]:
        start, end = to_date(start), to_date(end)
        frame = self._rows_for(code)
        hit = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        return [self._to_row(record) for _, record in hit.iterrows()]

    # -- 계산 -------------------------------------------------------------

    def daily_change(self, code: str, day: str | date) -> float | None:
        """그날의 전일 대비 등락률. 휴장일이면 None 입니다."""
        row = self.row_on(code, day)
        return None if row is None else row.change

    def change_between(
        self,
        code: str,
        start: str | date,
        end: str | date,
        *,
        snap: bool = True,
    ) -> ChangeResult | None:
        """두 날짜 종가로 등락률을 계산합니다.

        Args:
            snap: True 면 휴장일일 때 가장 가까운 거래일로 당깁니다.
                시작일은 이전 거래일로, 종료일은 이전 거래일로 맞춥니다.
                False 면 해당일에 데이터가 없을 때 None 을 돌려줍니다.
        """
        if snap:
            start_row = self.nearest_row(code, start, "backward")
            end_row = self.nearest_row(code, end, "backward")
        else:
            start_row = self.row_on(code, start)
            end_row = self.row_on(code, end)

        if start_row is None or end_row is None:
            return None
        if start_row.close == 0:
            return None
        return ChangeResult(code=code, start=start_row, end=end_row)

    def period_high(self, code: str, start: str | date, end: str | date) -> PriceRow | None:
        rows = self.rows_between(code, start, end)
        return max(rows, key=lambda r: r.high) if rows else None

    def period_low(self, code: str, start: str | date, end: str | date) -> PriceRow | None:
        rows = self.rows_between(code, start, end)
        return min(rows, key=lambda r: r.low) if rows else None

    def describe_stock(self, code: str) -> str:
        stock = get_stock(code)
        span = self.date_range(code)
        name = stock.name if stock else code
        if span is None:
            return f"{name}({code}): 데이터 없음"
        return f"{name}({code}): {span[0]} ~ {span[1]}, {len(self._rows_for(code))}행"
