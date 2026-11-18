"""주가 조회·계산 테스트.

대부분은 손으로 만든 데이터로 돌립니다. 실제 스냅샷에 의존하면 데이터를 다시
받을 때마다 테스트가 흔들립니다. 마지막 두 개만 실제 스냅샷이 읽히는지 확인합니다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pierrot.tools.price import (
    PriceSnapshot,
    SnapshotMissingError,
    default_snapshot_path,
    to_date,
)

# 2026-01-05(월) ~ 2026-01-09(금) 중 1월 7일은 휴장일이라고 가정합니다.
RECORDS = [
    {"code": "005930", "date": "2026-01-05", "open": 100, "high": 105, "low": 99,
     "close": 100, "volume": 1000, "change": 0.00},
    {"code": "005930", "date": "2026-01-06", "open": 100, "high": 112, "low": 100,
     "close": 110, "volume": 1200, "change": 0.10},
    # 2026-01-07 휴장
    {"code": "005930", "date": "2026-01-08", "open": 110, "high": 115, "low": 104,
     "close": 105, "volume": 900, "change": -0.0454545},
    {"code": "005930", "date": "2026-01-09", "open": 105, "high": 126, "low": 105,
     "close": 125, "volume": 1500, "change": 0.1904762},
    {"code": "000660", "date": "2026-01-05", "open": 200, "high": 200, "low": 190,
     "close": 200, "volume": 500, "change": 0.00},
    {"code": "000660", "date": "2026-01-06", "open": 200, "high": 200, "low": 178,
     "close": 180, "volume": 600, "change": -0.10},
]


@pytest.fixture
def snapshot() -> PriceSnapshot:
    return PriceSnapshot.from_records(RECORDS)


# ---------------------------------------------------------------------------
# 기본
# ---------------------------------------------------------------------------


def test_to_date_accepts_many_forms():
    assert to_date("2026-01-05") == date(2026, 1, 5)
    assert to_date(date(2026, 1, 5)) == date(2026, 1, 5)


def test_missing_column_is_rejected():
    """컬럼이 빠진 데이터를 조용히 받아들이면 나중에 엉뚱한 곳에서 터집니다."""
    import pandas as pd

    with pytest.raises(ValueError):
        PriceSnapshot(pd.DataFrame([{"code": "005930", "date": "2026-01-05"}]))


def test_len_and_codes(snapshot):
    assert len(snapshot) == len(RECORDS)
    assert set(snapshot.codes()) == {"005930", "000660"}


def test_date_range(snapshot):
    assert snapshot.date_range("005930") == (date(2026, 1, 5), date(2026, 1, 9))


def test_unknown_code_raises(snapshot):
    """유니버스 밖 종목을 조회하면 조용히 빈 값이 아니라 예외여야 합니다."""
    with pytest.raises(KeyError):
        snapshot.row_on("999999", "2026-01-05")


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------


def test_row_on_trading_day(snapshot):
    row = snapshot.row_on("005930", "2026-01-06")
    assert row is not None
    assert row.close == 110
    assert row.change == pytest.approx(0.10)


def test_row_on_holiday_returns_none(snapshot):
    """휴장일에 임의로 가까운 날을 끌어다 쓰면 안 됩니다.

    "1월 7일에 5% 올랐다"는 주장을 1월 8일 데이터로 판정하면 틀린 판정입니다.
    """
    assert snapshot.row_on("005930", "2026-01-07") is None


def test_nearest_row_backward(snapshot):
    row = snapshot.nearest_row("005930", "2026-01-07", "backward")
    assert row is not None and row.date == date(2026, 1, 6)


def test_nearest_row_forward(snapshot):
    row = snapshot.nearest_row("005930", "2026-01-07", "forward")
    assert row is not None and row.date == date(2026, 1, 8)


def test_nearest_row_out_of_range(snapshot):
    assert snapshot.nearest_row("005930", "2020-01-01", "backward") is None


def test_rows_between(snapshot):
    rows = snapshot.rows_between("005930", "2026-01-06", "2026-01-08")
    assert [r.date for r in rows] == [date(2026, 1, 6), date(2026, 1, 8)]


# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------


def test_daily_change(snapshot):
    assert snapshot.daily_change("005930", "2026-01-06") == pytest.approx(0.10)
    assert snapshot.daily_change("005930", "2026-01-07") is None


def test_change_between(snapshot):
    """100원에서 125원이면 +25% 입니다. LLM에게 묻지 않고 직접 계산합니다."""
    result = snapshot.change_between("005930", "2026-01-05", "2026-01-09")
    assert result is not None
    assert result.rate == pytest.approx(0.25)
    assert result.percent == pytest.approx(25.0)


def test_change_between_negative(snapshot):
    result = snapshot.change_between("000660", "2026-01-05", "2026-01-06")
    assert result is not None
    assert result.percent == pytest.approx(-10.0)


def test_change_between_snaps_holiday(snapshot):
    """종료일이 휴장일이면 직전 거래일 종가를 씁니다."""
    result = snapshot.change_between("005930", "2026-01-05", "2026-01-07")
    assert result is not None
    assert result.end.date == date(2026, 1, 6)
    assert result.percent == pytest.approx(10.0)


def test_change_between_without_snap_returns_none(snapshot):
    """snap=False 면 해당일 데이터가 없을 때 계산하지 않습니다."""
    assert snapshot.change_between("005930", "2026-01-05", "2026-01-07", snap=False) is None


def test_change_result_describe(snapshot):
    """판정 근거로 쓰려면 숫자만이 아니라 설명이 함께 나와야 합니다."""
    result = snapshot.change_between("005930", "2026-01-05", "2026-01-09")
    text = result.describe()
    assert "2026-01-05" in text and "2026-01-09" in text and "+25.00%" in text


def test_period_high_and_low(snapshot):
    high = snapshot.period_high("005930", "2026-01-05", "2026-01-09")
    low = snapshot.period_low("005930", "2026-01-05", "2026-01-09")
    assert high is not None and high.high == 126
    assert low is not None and low.low == 99


def test_period_high_empty_range(snapshot):
    assert snapshot.period_high("005930", "2020-01-01", "2020-12-31") is None


# ---------------------------------------------------------------------------
# 실제 스냅샷
# ---------------------------------------------------------------------------


def test_missing_snapshot_raises():
    with pytest.raises(SnapshotMissingError):
        PriceSnapshot.load(Path("없는경로") / "없는파일.parquet")


@pytest.mark.skipif(
    not default_snapshot_path().exists(), reason="주가 스냅샷이 아직 수집되지 않았습니다"
)
def test_real_snapshot_loads():
    """커밋된 스냅샷이 실제로 읽히고 유니버스 전체를 담고 있는지 확인합니다."""
    from pierrot.tools.universe import UNIVERSE

    snapshot = PriceSnapshot.load()
    assert set(snapshot.codes()) == {stock.code for stock in UNIVERSE}

    span = snapshot.date_range()
    assert span is not None
    assert span[0] <= date(2023, 1, 3)
    assert span[1] >= date(2026, 11, 1)
