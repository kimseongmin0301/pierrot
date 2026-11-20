"""요청이 쓴 비용을 실제로 재는지.

실측 버그:

    20건을 일괄 검증했는데 "비용 $0.00000" 이 찍혔습니다.
    그런데 `logs/cost.jsonl` 누적은 $1.2040 -> $1.2293 으로
    **약 $0.025 늘었습니다.** 돈은 나갔는데 0 으로 보고한 것입니다.

원인은 **비용 로거가 둘**이었던 것입니다.

    llm.py          : 모듈 수준 `_cost_logger`. LLM 호출이 여기에 기록됩니다.
    api/service.py  : `CostLogger(...)` 를 **새로** 만들었습니다.

비용은 월 누적액의 **차이**로 잽니다. 그런데 service 쪽 인스턴스에는
아무것도 기록되지 않으니 차이가 늘 0 입니다. FastAPI 응답의 cost_usd 도
전부 0 이었습니다. 월 예산 $5 를 지키려고 만든 장치가 아무것도 못 본 것입니다.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pierrot.api.service import VerificationService
from pierrot.config import get_settings
from pierrot.llm import get_cost_logger


@pytest.fixture
def settings(tmp_path):
    """비용 기록을 임시 파일로 돌립니다. 실제 운영 로그를 더럽히지 않습니다."""
    real = get_settings()
    temporary = replace(real, cost_log_path=tmp_path / "cost.jsonl")
    get_cost_logger(temporary, reload=True)
    yield temporary
    # 다른 테스트가 쓰는 공용 로거를 원래대로 돌려 놓습니다.
    get_cost_logger(real, reload=True)


def test_서비스는_프로세스_공용_로거를_쓴다(settings):
    """새로 만들면 LLM 호출을 한 건도 보지 못합니다."""
    service = VerificationService(settings, graph=object())
    assert service.cost is get_cost_logger(settings)


def test_공용_로거에_기록하면_서비스가_본다(settings):
    """차이로 비용을 재므로, 같은 인스턴스여야 변화가 보입니다."""
    service = VerificationService(settings, graph=object())
    before = service.cost.month_total_usd()

    get_cost_logger(settings).record(
        source="test",
        model="gpt-4o-mini",
        input_tokens=100_000,
        output_tokens=100_000,
        cache_hit=False,
    )

    after = service.cost.month_total_usd()
    assert after > before


def test_캐시에_맞으면_비용이_늘지_않는다(settings):
    """캐시 적중은 돈이 나가지 않습니다. 0 이 맞습니다."""
    service = VerificationService(settings, graph=object())
    before = service.cost.month_total_usd()

    get_cost_logger(settings).record(
        source="test",
        model="gpt-4o-mini",
        input_tokens=100_000,
        output_tokens=100_000,
        cache_hit=True,
    )

    assert service.cost.month_total_usd() == before
