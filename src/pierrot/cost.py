"""토큰 사용량과 비용 기록.

월 예산이 $5이므로 "얼마 썼는지 모르는 상태"를 만들지 않는 것이 이 모듈의 목적입니다.
모든 LLM 호출의 토큰 수를 받아 단가를 곱하고, JSONL 파일에 한 줄씩 덧붙입니다.

## 캐시 적중을 반드시 구분해야 하는 이유

LangChain의 응답 캐시가 적중하면 실제 API 호출은 일어나지 않지만, 캐시에 저장된
메시지에는 원래 호출의 토큰 수(usage_metadata)가 그대로 남아 있습니다.
usage_metadata 를 기준으로 비용을 계산하면 캐시가 적중할 때마다 쓰지도 않은 돈을
중복으로 계상하게 됩니다.

실측으로 확인한 차이는 다음과 같습니다.

| 상황          | llm_output["token_usage"] | message.usage_metadata |
|---------------|---------------------------|------------------------|
| 실제 API 호출 | 채워짐                    | 채워짐                 |
| 캐시 적중     | 빈 dict                   | 채워짐(과거 값)        |

따라서 OpenAI 호출의 과금 판단은 llm_output["token_usage"] 하나만 봅니다.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ---------------------------------------------------------------------------
# 단가표
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """100만 토큰당 미국 달러 단가."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float | None = None

    @property
    def is_free(self) -> bool:
        return self.input_usd_per_mtok == 0.0 and self.output_usd_per_mtok == 0.0


# 출처: https://developers.openai.com/api/docs/pricing (2026-11-18 확인)
# 단가가 바뀌면 이 표만 고치면 됩니다.
PRICING: dict[str, ModelPrice] = {
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.075),
    "gpt-4o": ModelPrice(2.50, 10.00, 1.25),
    "gpt-4.1": ModelPrice(2.00, 8.00, 0.50),
    "gpt-5-nano": ModelPrice(0.05, 0.40, 0.005),
    "gpt-5-mini": ModelPrice(0.25, 2.00, 0.025),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075),
    "gpt-5.2": ModelPrice(1.75, 14.00, 0.175),
    "gpt-5.1": ModelPrice(1.25, 10.00, 0.125),
    "gpt-5": ModelPrice(1.25, 10.00, 0.125),
    # 로컬(Ollama) 모델은 비용이 발생하지 않습니다.
    "gemma3": ModelPrice(0.0, 0.0),
    "qwen3": ModelPrice(0.0, 0.0),
    "embeddinggemma": ModelPrice(0.0, 0.0),
}


class UnknownModelPriceError(LookupError):
    """단가표에 없는 모델입니다."""


class BudgetExceededError(RuntimeError):
    """이번 달 누적 비용이 예산 상한을 넘었습니다."""


def normalize_model_name(model: str) -> str:
    """API가 돌려주는 상세 모델명을 단가표의 키로 맞춥니다.

    OpenAI는 상황에 따라 gpt-4o-mini 또는 gpt-4o-mini-2024-07-18 처럼 날짜가 붙은
    이름을 돌려줍니다. Ollama는 gemma3:4b 처럼 태그가 붙습니다.
    가장 길게 일치하는 키를 고릅니다.
    """
    if not model:
        raise UnknownModelPriceError("모델 이름이 비어 있습니다.")
    if model in PRICING:
        return model
    candidates = [key for key in PRICING if model.startswith(key)]
    if not candidates:
        raise UnknownModelPriceError(
            f"단가표에 없는 모델입니다: {model!r}. "
            "src/pierrot/cost.py 의 PRICING 에 추가해 주십시오."
        )
    return max(candidates, key=len)


def get_price(model: str) -> ModelPrice:
    return PRICING[normalize_model_name(model)]


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """호출 1회의 비용을 계산합니다.

    주의: OpenAI가 보고하는 prompt_tokens 에는 캐시로 읽은 토큰이 포함되어 있습니다.
    그래서 캐시 분을 빼고 나머지에 정상 단가를, 캐시 분에는 할인 단가를 적용합니다.
    """
    price = get_price(model)
    cached = max(0, min(cached_input_tokens, input_tokens))
    fresh_input = input_tokens - cached

    cost = fresh_input * price.input_usd_per_mtok / 1_000_000
    cost += output_tokens * price.output_usd_per_mtok / 1_000_000
    if cached:
        rate = (
            price.cached_input_usd_per_mtok
            if price.cached_input_usd_per_mtok is not None
            else price.input_usd_per_mtok
        )
        cost += cached * rate / 1_000_000
    return cost


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRecord:
    timestamp: str
    month: str
    source: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    cache_hit: bool
    tokens_saved_by_cache: int

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CostLogger:
    """호출별 비용을 JSONL로 남기고, 월 누적액이 예산을 넘는지 감시합니다.

    월 기준은 UTC입니다. OpenAI 청구가 UTC 기준이므로 거기에 맞췄습니다.
    """

    def __init__(self, log_path: Path, monthly_budget_usd: float) -> None:
        self.log_path = Path(log_path)
        self.monthly_budget_usd = float(monthly_budget_usd)
        self._lock = threading.Lock()
        self._month_totals: dict[str, float] = {}
        self._load_existing()

    # -- 내부 --------------------------------------------------------------

    def _load_existing(self) -> None:
        if not self.log_path.exists():
            return
        with self.log_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                month = row.get("month")
                cost = row.get("cost_usd") or 0.0
                if month:
                    self._month_totals[month] = self._month_totals.get(month, 0.0) + cost

    # -- 조회 --------------------------------------------------------------

    def current_month(self) -> str:
        return _utc_now().strftime("%Y-%m")

    def month_total_usd(self, month: str | None = None) -> float:
        return self._month_totals.get(month or self.current_month(), 0.0)

    def remaining_usd(self) -> float:
        return self.monthly_budget_usd - self.month_total_usd()

    def assert_within_budget(self) -> None:
        """호출하기 전에 예산이 남았는지 확인합니다.

        콜백은 호출이 끝난 뒤에야 동작하므로, 이미 예산을 넘긴 상태에서 새 호출이
        나가는 것을 막으려면 호출 전에 이 함수를 써야 합니다.
        """
        spent = self.month_total_usd()
        if spent >= self.monthly_budget_usd:
            raise BudgetExceededError(
                f"이번 달({self.current_month()}) 누적 비용 ${spent:.4f} 가 "
                f"예산 상한 ${self.monthly_budget_usd:.2f} 에 도달했습니다. "
                ".env 의 MONTHLY_BUDGET_USD 를 조정하거나 다음 달까지 기다려 주십시오."
            )

    def summary(self) -> dict[str, Any]:
        month = self.current_month()
        return {
            "month": month,
            "spent_usd": round(self.month_total_usd(month), 6),
            "budget_usd": self.monthly_budget_usd,
            "remaining_usd": round(self.remaining_usd(), 6),
            "log_path": str(self.log_path),
        }

    # -- 기록 --------------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        provider: str = "openai",
        source: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_hit: bool = False,
        tokens_saved_by_cache: int = 0,
    ) -> CostRecord:
        now = _utc_now()
        cost = (
            0.0
            if cache_hit
            else estimate_cost_usd(model, input_tokens, output_tokens, cached_input_tokens)
        )

        record = CostRecord(
            timestamp=now.isoformat(),
            month=now.strftime("%Y-%m"),
            source=source,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=round(cost, 8),
            cache_hit=cache_hit,
            tokens_saved_by_cache=tokens_saved_by_cache,
        )

        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json_line() + "\n")
            self._month_totals[record.month] = (
                self._month_totals.get(record.month, 0.0) + record.cost_usd
            )

        spent = self._month_totals[record.month]
        if spent > self.monthly_budget_usd:
            raise BudgetExceededError(
                f"이번 달({record.month}) 누적 비용 ${spent:.4f} 가 "
                f"예산 상한 ${self.monthly_budget_usd:.2f} 를 넘었습니다. "
                "마지막 호출은 이미 과금되었으므로 로그에는 남아 있습니다."
            )
        return record


# ---------------------------------------------------------------------------
# LangChain 콜백
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class CostCallbackHandler(BaseCallbackHandler):
    """LangChain 호출이 끝날 때마다 토큰 수를 읽어 CostLogger에 넘깁니다.

    provider 에 따라 토큰을 읽는 위치가 다릅니다.

    - openai: llm_output["token_usage"] 만 봅니다. 이 값이 비어 있으면 캐시
      적중이므로 비용 0으로 기록합니다.
    - local: 비용이 0이므로 토큰 수만 참고용으로 usage_metadata 에서 읽습니다.
    """

    def __init__(
        self,
        logger: CostLogger,
        *,
        model: str,
        source: str = "",
        provider: str = "openai",
    ) -> None:
        self.logger = logger
        self.model = model
        self.source = source or model
        self.provider = provider

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            self._handle(response)
        except BudgetExceededError:
            raise
        except Exception as exc:  # 비용 기록 실패가 본 작업을 멈추면 안 됩니다.
            print(f"[cost] 사용량 기록 실패: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _handle(self, response: LLMResult) -> None:
        llm_output = response.llm_output or {}
        model = llm_output.get("model_name") or self.model

        if self.provider == "local":
            usage = self._usage_metadata(response)
            self.logger.record(
                model=model,
                provider="local",
                source=self.source,
                input_tokens=_as_int(usage.get("input_tokens")),
                output_tokens=_as_int(usage.get("output_tokens")),
            )
            return

        token_usage = llm_output.get("token_usage") or {}
        if not token_usage:
            # 캐시 적중. 실제 호출이 없었으므로 비용은 0입니다.
            saved = self._usage_metadata(response)
            self.logger.record(
                model=model,
                provider=self.provider,
                source=self.source,
                cache_hit=True,
                tokens_saved_by_cache=_as_int(saved.get("total_tokens")),
            )
            return

        details = token_usage.get("prompt_tokens_details") or {}
        self.logger.record(
            model=model,
            provider=self.provider,
            source=self.source,
            input_tokens=_as_int(token_usage.get("prompt_tokens")),
            output_tokens=_as_int(token_usage.get("completion_tokens")),
            cached_input_tokens=_as_int(details.get("cached_tokens")),
        )

    @staticmethod
    def _usage_metadata(response: LLMResult) -> dict[str, Any]:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    return dict(usage)
        return {}


def summarize_log(log_path: Path) -> dict[str, Any]:
    """비용 로그 전체를 읽어 월별로 집계합니다. 보고용 함수입니다."""
    log_path = Path(log_path)
    totals: dict[str, dict[str, Any]] = {}
    if not log_path.exists():
        return totals

    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            month = row.get("month", "?")
            bucket = totals.setdefault(
                month,
                {
                    "calls": 0,
                    "cache_hits": 0,
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "by_model": {},
                },
            )
            bucket["calls"] += 1
            bucket["cache_hits"] += 1 if row.get("cache_hit") else 0
            bucket["cost_usd"] += row.get("cost_usd") or 0.0
            bucket["input_tokens"] += row.get("input_tokens") or 0
            bucket["output_tokens"] += row.get("output_tokens") or 0
            model = row.get("model", "?")
            bucket["by_model"][model] = round(
                bucket["by_model"].get(model, 0.0) + (row.get("cost_usd") or 0.0), 8
            )
    for bucket in totals.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 8)
    return totals
