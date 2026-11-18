"""비용 계산과 기록에 대한 단위 테스트.

API 호출은 하지 않습니다. 과금 없이 돌릴 수 있어야 하는 테스트입니다.

함수 이름을 영문으로 쓰는 이유: Windows 콘솔에서 한글 테스트 이름이 깨져
실패한 테스트를 식별할 수 없기 때문입니다. 설명은 docstring에 한글로 남깁니다.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from pierrot.cost import (
    BudgetExceededError,
    CostCallbackHandler,
    CostLogger,
    UnknownModelPriceError,
    estimate_cost_usd,
    normalize_model_name,
    summarize_log,
)


# ---------------------------------------------------------------------------
# 모델 이름 정규화
# ---------------------------------------------------------------------------


def test_exact_model_name_is_kept():
    """단가표에 그대로 있는 이름은 변환하지 않습니다."""
    assert normalize_model_name("gpt-4o-mini") == "gpt-4o-mini"


def test_dated_model_name_is_resolved():
    """OpenAI가 돌려주는 날짜 붙은 이름도 단가표 키로 찾습니다."""
    assert normalize_model_name("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"


def test_longest_prefix_wins():
    """gpt-5 도 접두사로 일치하지만 더 긴 gpt-5-mini 가 선택되어야 합니다."""
    assert normalize_model_name("gpt-5-mini-2025-08-07") == "gpt-5-mini"


def test_ollama_tag_is_resolved():
    """gemma3:4b 처럼 태그가 붙은 로컬 모델 이름도 처리합니다."""
    assert normalize_model_name("gemma3:4b") == "gemma3"


def test_unknown_model_raises():
    """단가를 모르는 모델은 조용히 0원 처리하지 않고 예외를 냅니다."""
    with pytest.raises(UnknownModelPriceError):
        normalize_model_name("llama4:70b")


# ---------------------------------------------------------------------------
# 비용 계산
# ---------------------------------------------------------------------------


def test_input_price_per_million_tokens():
    """gpt-4o-mini 입력 단가는 100만 토큰당 $0.15 입니다."""
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)


def test_output_price_per_million_tokens():
    """gpt-4o-mini 출력 단가는 100만 토큰당 $0.60 입니다."""
    assert estimate_cost_usd("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)


def test_cached_input_uses_discounted_rate():
    """prompt_tokens 에 포함된 캐시 토큰은 할인 단가로 계산해야 합니다.

    입력 100만 중 50만이 캐시 적중이면
    정상 50만 x $0.15 + 캐시 50만 x $0.075 가 됩니다.
    """
    cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 0, cached_input_tokens=500_000)
    assert cost == pytest.approx(0.075 + 0.0375)


def test_cached_tokens_larger_than_input_is_safe():
    """캐시 토큰이 입력보다 크게 들어와도 음수 비용이 나오면 안 됩니다."""
    cost = estimate_cost_usd("gpt-4o-mini", 100, 0, cached_input_tokens=999_999)
    assert cost >= 0


def test_local_model_is_free():
    """Ollama 로컬 모델은 비용이 발생하지 않습니다."""
    assert estimate_cost_usd("gemma3:4b", 10_000, 10_000) == 0.0


# ---------------------------------------------------------------------------
# 기록과 예산
# ---------------------------------------------------------------------------


def make_logger(tmp_path, budget=5.0) -> CostLogger:
    return CostLogger(log_path=tmp_path / "cost.jsonl", monthly_budget_usd=budget)


def test_record_is_written_to_file(tmp_path):
    """호출 1건이 JSONL 한 줄로 남습니다."""
    logger = make_logger(tmp_path)
    logger.record(model="gpt-4o-mini", source="test", input_tokens=1000, output_tokens=500)

    lines = (tmp_path / "cost.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["model"] == "gpt-4o-mini"
    assert row["input_tokens"] == 1000
    assert row["cost_usd"] == pytest.approx(1000 * 0.15 / 1e6 + 500 * 0.60 / 1e6)


def test_monthly_total_accumulates(tmp_path):
    """같은 달의 호출 비용이 합산됩니다."""
    logger = make_logger(tmp_path)
    for _ in range(3):
        logger.record(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    assert logger.month_total_usd() == pytest.approx(0.45)
    assert logger.remaining_usd() == pytest.approx(5.0 - 0.45)


def test_existing_log_is_reloaded(tmp_path):
    """프로세스를 새로 띄워도 기존 로그를 읽어 누적액을 복원합니다."""
    first = make_logger(tmp_path)
    first.record(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)

    second = make_logger(tmp_path)
    assert second.month_total_usd() == pytest.approx(0.15)


def test_budget_exceeded_raises(tmp_path):
    """예산을 넘기면 예외가 발생합니다."""
    logger = make_logger(tmp_path, budget=0.10)
    with pytest.raises(BudgetExceededError):
        logger.record(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)


def test_preflight_check_blocks_further_calls(tmp_path):
    """이미 예산을 넘긴 상태라면 호출 전 단계에서 막습니다."""
    logger = make_logger(tmp_path, budget=0.10)
    with pytest.raises(BudgetExceededError):
        logger.record(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)

    reopened = make_logger(tmp_path, budget=0.10)
    with pytest.raises(BudgetExceededError):
        reopened.assert_within_budget()


def test_cache_hit_costs_nothing(tmp_path):
    """캐시 적중으로 기록하면 비용이 0입니다."""
    logger = make_logger(tmp_path)
    record = logger.record(
        model="gpt-4o-mini", cache_hit=True, tokens_saved_by_cache=1_000_000
    )
    assert record.cost_usd == 0.0
    assert logger.month_total_usd() == 0.0


# ---------------------------------------------------------------------------
# LangChain 콜백
# ---------------------------------------------------------------------------


def make_result(llm_output: dict | None, usage: dict | None) -> LLMResult:
    """LangChain이 콜백에 넘기는 LLMResult 를 흉내 냅니다."""
    if usage is None:
        message = AIMessage(content="PONG")
    else:
        message = AIMessage(content="PONG", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


FRESH_LLM_OUTPUT = {
    "token_usage": {
        "completion_tokens": 2,
        "prompt_tokens": 13,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0},
    },
    "model_name": "gpt-4o-mini",
}

USAGE_METADATA = {"input_tokens": 13, "output_tokens": 2, "total_tokens": 15}


def test_fresh_call_records_tokens_and_cost(tmp_path):
    """실제 API 호출은 토큰 수와 비용이 모두 기록됩니다."""
    logger = make_logger(tmp_path)
    handler = CostCallbackHandler(logger, model="gpt-4o-mini", source="unit")
    handler.on_llm_end(make_result(FRESH_LLM_OUTPUT, USAGE_METADATA))

    row = json.loads((tmp_path / "cost.jsonl").read_text(encoding="utf-8").strip())
    assert row["cache_hit"] is False
    assert row["input_tokens"] == 13
    assert row["output_tokens"] == 2
    assert row["cost_usd"] > 0


def test_cache_hit_is_not_billed(tmp_path):
    """이 테스트가 이 모듈의 핵심입니다.

    캐시 적중 시 llm_output["token_usage"] 는 비어 있지만 usage_metadata 에는
    과거 토큰 수가 그대로 남아 있습니다. usage_metadata 를 기준으로 계산하면
    쓰지도 않은 돈이 매번 누적됩니다.
    """
    logger = make_logger(tmp_path)
    handler = CostCallbackHandler(logger, model="gpt-4o-mini", source="unit")

    cached_output = {"token_usage": {}, "model_name": "gpt-4o-mini"}
    handler.on_llm_end(make_result(cached_output, USAGE_METADATA))

    row = json.loads((tmp_path / "cost.jsonl").read_text(encoding="utf-8").strip())
    assert row["cache_hit"] is True
    assert row["cost_usd"] == 0.0
    assert row["input_tokens"] == 0
    assert row["tokens_saved_by_cache"] == 15
    assert logger.month_total_usd() == 0.0


def test_local_provider_records_tokens_without_cost(tmp_path):
    """로컬 모델은 토큰 수만 남기고 비용은 0으로 기록합니다."""
    logger = make_logger(tmp_path)
    handler = CostCallbackHandler(
        logger, model="gemma3:4b", source="generator", provider="local"
    )
    usage = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
    handler.on_llm_end(make_result(None, usage))

    row = json.loads((tmp_path / "cost.jsonl").read_text(encoding="utf-8").strip())
    assert row["provider"] == "local"
    assert row["input_tokens"] == 50
    assert row["output_tokens"] == 20
    assert row["cost_usd"] == 0.0


def test_summarize_log(tmp_path):
    """월별 집계가 호출 수와 캐시 적중 수를 올바르게 셉니다."""
    logger = make_logger(tmp_path)
    logger.record(model="gpt-4o-mini", input_tokens=1000, output_tokens=100)
    logger.record(model="gpt-4o-mini", cache_hit=True, tokens_saved_by_cache=1100)

    totals = summarize_log(tmp_path / "cost.jsonl")
    month = logger.current_month()
    assert totals[month]["calls"] == 2
    assert totals[month]["cache_hits"] == 1
    assert totals[month]["cost_usd"] > 0
