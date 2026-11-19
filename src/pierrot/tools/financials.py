"""재무 스냅샷 조회.

DART 에서 받아 저장해 둔 재무 수치를 읽습니다. 주가 스냅샷과 같은 원칙으로,
검증 중에는 외부에 접속하지 않습니다.

## 연결과 별도를 모두 들고 있는 이유

같은 회사, 같은 기간이라도 연결(CFS)과 별도(OFS)의 수치가 크게 다릅니다.

    삼성전자 2024 영업이익: 연결 32.73조 / 별도 12.36조

LLM 답변은 보통 어느 쪽인지 밝히지 않습니다. 그래서 둘 다 들고 있다가
판정 단계에서 "어느 쪽과 맞았는지"를 기록합니다. 하나만 골라 두면 맞는 주장을
틀렸다고 판정하게 됩니다(오탐).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import get_settings
from .amounts import format_korean_amount
from .universe import Stock, get_stock

SNAPSHOT_FILENAME = "dart_financials.parquet"

COLUMNS = [
    "code",
    "corp_code",
    "bsns_year",
    "reprt_code",
    "period_label",
    "quarter",
    "period_months",
    "fs_div",
    "fs_nm",
    "account",
    "amount",
    "period_start",
    "period_end",
    "currency",
    "rcept_no",
]

FS_LABEL = {"CFS": "연결", "OFS": "별도"}


class FinancialsMissingError(FileNotFoundError):
    """재무 스냅샷 파일이 없습니다."""


def default_snapshot_path() -> Path:
    return get_settings().snapshot_dir / SNAPSHOT_FILENAME


@dataclass(frozen=True)
class FinancialFact:
    """어느 회사, 어느 기간, 어느 계정의 수치 하나."""

    code: str
    bsns_year: int
    quarter: int
    """0 이면 연간, 1~3 이면 해당 분기(3개월)."""

    period_label: str
    period_months: int
    fs_div: str
    """CFS(연결) 또는 OFS(별도)."""

    account: str
    amount: int
    period_start: str
    period_end: str
    rcept_no: str

    @property
    def fs_label(self) -> str:
        return FS_LABEL.get(self.fs_div, self.fs_div)

    @property
    def stock(self) -> Stock | None:
        return get_stock(self.code)

    def describe(self) -> str:
        name = self.stock.name if self.stock else self.code
        return (
            f"{name} {self.bsns_year}년 {self.period_label} "
            f"{self.account}({self.fs_label}) {format_won(self.amount)}"
        )


# 금액 표기는 amounts 모듈 하나만 씁니다. 두 벌을 두면 판정 근거와 보고서의
# 표기가 어긋나 비교가 어려워집니다.
format_won = format_korean_amount


class FinancialSnapshot:
    """재무 스냅샷 조회기."""

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [column for column in COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"재무 스냅샷에 없는 컬럼: {missing}")
        self._frame = frame[COLUMNS].copy().reset_index(drop=True)

    # -- 생성 -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "FinancialSnapshot":
        path = Path(path) if path else default_snapshot_path()
        if not path.exists():
            raise FinancialsMissingError(
                f"재무 스냅샷이 없습니다: {path}\n"
                "먼저 scripts/fetch_dart_snapshot.py 를 실행해 주십시오."
            )
        frame = pd.read_parquet(path)

        # 주요계정 밖에서 모은 수치(배당금, 지배주주순이익 등)를 합칩니다.
        # 파일을 나눠 둔 이유: 주요계정 수집을 다시 돌려도 이쪽이 날아가지
        # 않게 하기 위해서입니다. 없으면 그냥 넘어갑니다.
        extra_path = path.with_name("dart_financials_extra.parquet")
        if extra_path.exists():
            extra = pd.read_parquet(extra_path)
            frame = pd.concat([frame, extra], ignore_index=True)
        return cls(frame)

    @classmethod
    def from_records(cls, records: list[dict]) -> "FinancialSnapshot":
        return cls(pd.DataFrame(records, columns=COLUMNS))

    # -- 기본 정보 ---------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def __len__(self) -> int:
        return len(self._frame)

    def codes(self) -> list[str]:
        return sorted(self._frame["code"].unique())

    def accounts(self) -> list[str]:
        return sorted(self._frame["account"].unique())

    def year_range(self) -> tuple[int, int] | None:
        if self._frame.empty:
            return None
        return (int(self._frame["bsns_year"].min()), int(self._frame["bsns_year"].max()))

    # -- 조회 -------------------------------------------------------------

    def _to_fact(self, record: pd.Series) -> FinancialFact:
        return FinancialFact(
            code=str(record["code"]),
            bsns_year=int(record["bsns_year"]),
            quarter=int(record["quarter"]),
            period_label=str(record["period_label"]),
            period_months=int(record["period_months"]),
            fs_div=str(record["fs_div"]),
            account=str(record["account"]),
            amount=int(record["amount"]),
            period_start=str(record["period_start"]),
            period_end=str(record["period_end"]),
            rcept_no=str(record["rcept_no"]),
        )

    def lookup(
        self,
        code: str,
        year: int,
        account: str,
        *,
        quarter: int | None = None,
        fs_div: str | None = None,
    ) -> list[FinancialFact]:
        """조건에 맞는 수치를 모두 돌려줍니다.

        fs_div 를 지정하지 않으면 연결과 별도가 모두 나옵니다.
        어느 쪽인지 밝히지 않은 주장을 판정하려면 둘 다 필요하기 때문입니다.

        Args:
            quarter: 0 이면 연간, 1~3 이면 해당 분기. None 이면 전부.
        """
        frame = self._frame
        selected = frame[
            (frame["code"] == code)
            & (frame["bsns_year"] == year)
            & (frame["account"] == account)
        ]
        if quarter is not None:
            selected = selected[selected["quarter"] == quarter]
        if fs_div is not None:
            selected = selected[selected["fs_div"] == fs_div]
        return [self._to_fact(record) for _, record in selected.iterrows()]

    def has_data(self, code: str, year: int) -> bool:
        frame = self._frame
        return bool(
            ((frame["code"] == code) & (frame["bsns_year"] == year)).any()
        )

    def describe_coverage(self) -> str:
        span = self.year_range()
        if span is None:
            return "데이터 없음"
        return (
            f"{len(self):,}행 / {len(self.codes())}종목 / "
            f"{span[0]} ~ {span[1]}년 / 계정 {len(self.accounts())}종"
        )
