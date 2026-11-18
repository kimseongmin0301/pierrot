"""KRX 정보데이터시스템에서 원주가(수정되지 않은 당시 실제 호가)를 받습니다.

## 왜 필요한가

FinanceDataReader 가 주는 시세는 수정주가입니다. 분할·합병·인적분할이 있으면
과거 가격이 소급 조정되어, **그때 실제로 거래된 가격과 달라집니다.**

LLM 이 "2024년 7월 26일 주가는 85만원이었다"고 말할 때 그 값은 당시 호가를
가리킵니다. 보정된 시세와 비교하면 맞는 주장을 틀렸다고 판정하게 됩니다.
오탐은 이 프로젝트가 가장 피해야 할 오류입니다.

반대로 등락률 주장에는 수정주가가 맞습니다. 분할 전후를 비교 가능하게 만드는 것이
수정주가의 목적이기 때문입니다. 그래서 둘 다 저장하고 주장 유형에 따라 골라 씁니다.

## 로그인이 필요합니다

KRX 가 비로그인 API 접근을 막고 있어, data.krx.co.kr 무료 회원 계정이 필요합니다.
`.env` 에 KRX_ID / KRX_PW 를 넣으십시오. `.env` 는 .gitignore 에 등록되어 있습니다.

**로그인은 수집할 때만 필요합니다.** 검증은 저장된 스냅샷만 읽으므로,
KRX 가 막히거나 pykrx 가 깨져도 평가는 계속 돌아갑니다. 재현성 원칙은 그대로입니다.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd

from ..config import Settings, get_settings

RAW_RENAME = {"시가": "open_raw", "고가": "high_raw", "저가": "low_raw", "종가": "close_raw"}


class KrxCredentialsMissingError(RuntimeError):
    """KRX 자격증명이 설정되지 않았습니다."""


@dataclass(frozen=True)
class RawFetchResult:
    code: str
    frame: pd.DataFrame
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.frame.empty


def _load_pykrx(settings: Settings):
    """pykrx 를 불러옵니다.

    pykrx 는 **임포트 시점에** 환경 변수를 읽어 로그인을 시도합니다.
    그래서 임포트 전에 환경 변수를 채워야 합니다.
    """
    if not settings.has_krx_credentials:
        raise KrxCredentialsMissingError(
            "KRX_ID / KRX_PW 가 설정되지 않았습니다.\n"
            "https://data.krx.co.kr 에서 무료 가입한 뒤 .env 에 넣어 주십시오.\n"
            "원주가 없이 진행하려면 --skip-raw 를 쓰십시오."
        )
    os.environ["KRX_ID"] = settings.krx_id
    os.environ["KRX_PW"] = settings.krx_pw

    from pykrx import stock  # noqa: PLC0415

    return stock


def fetch_raw_ohlcv(
    code: str,
    start: str,
    end: str,
    *,
    settings: Settings | None = None,
    pause: float = 0.3,
) -> RawFetchResult:
    """종목 하나의 원주가 시세를 받습니다.

    Args:
        start, end: "2023-01-01" 또는 "20230101" 형식.
        pause: 요청 사이 대기 시간. KRX 에 부담을 주지 않기 위함입니다.
    """
    settings = settings or get_settings()
    stock = _load_pykrx(settings)

    from_date = start.replace("-", "")
    to_date = end.replace("-", "")

    try:
        frame = stock.get_market_ohlcv(from_date, to_date, code, adjusted=False)
    except Exception as exc:
        return RawFetchResult(code, pd.DataFrame(), f"{type(exc).__name__}: {exc}")
    finally:
        if pause:
            time.sleep(pause)

    if frame is None or frame.empty:
        return RawFetchResult(code, pd.DataFrame(), "빈 결과")

    missing = [column for column in RAW_RENAME if column not in frame.columns]
    if missing:
        return RawFetchResult(
            code, pd.DataFrame(), f"예상한 컬럼이 없습니다: {missing} (받은 컬럼: {list(frame.columns)})"
        )

    frame = frame.rename(columns=RAW_RENAME).reset_index()
    date_column = frame.columns[0]
    frame["date"] = pd.to_datetime(frame[date_column]).dt.date
    frame["code"] = code
    return RawFetchResult(code, frame[["code", "date", *RAW_RENAME.values()]])


def merge_raw(adjusted: pd.DataFrame, raw_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """수정주가 스냅샷에 원주가 컬럼을 붙입니다.

    (code, date) 로 맞춥니다. 원주가가 없는 행은 비어 있는 채로 둡니다.
    억지로 채우면 어느 값이 실제 호가인지 알 수 없게 됩니다.
    """
    if not raw_frames:
        return adjusted
    raw = pd.concat(raw_frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["code", "date"], keep="last")
    return adjusted.merge(raw, on=["code", "date"], how="left")
