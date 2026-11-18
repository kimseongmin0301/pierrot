"""LangChain 기반 LLM 클라이언트 생성.

이 모듈을 거치지 않고 ChatOpenAI 를 직접 만들면 두 가지가 빠집니다.

1. 응답 캐시 — 같은 프롬프트를 다시 보낼 때 돈을 또 내게 됩니다.
2. 비용 기록 — 얼마를 썼는지 아무 데도 남지 않습니다.

월 예산이 $5이므로 둘 다 빠지면 안 됩니다. 그래서 LLM 객체는 항상 여기서 만듭니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# langchain-community 는 sunset 예정이라는 경고를 띄웁니다. SQLiteCache 는 아직
# 정상 동작하므로 그대로 쓰되, 동작이 깨지면 BaseCache 를 상속해 직접 구현합니다.
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings
from .cost import CostCallbackHandler, CostLogger

OpenAIRole = Literal["extractor", "verifier", "judge"]

_cache_path: Path | None = None
_cost_logger: CostLogger | None = None


def setup_cache(settings: Settings | None = None, *, force: bool = False) -> Path:
    """프롬프트 단위 응답 캐시를 켭니다.

    같은 (모델, 프롬프트) 조합이 다시 들어오면 API를 호출하지 않고 저장된 응답을
    돌려줍니다. 평가를 여러 번 돌릴 때 비용이 누적되지 않게 하는 핵심 장치입니다.

    캐시 파일은 .gitignore 에 등록되어 있어 저장소에는 올라가지 않습니다.
    """
    global _cache_path
    settings = settings or get_settings()
    if _cache_path is not None and not force:
        return _cache_path

    settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(settings.cache_path)))
    _cache_path = settings.cache_path
    return _cache_path


def get_cost_logger(settings: Settings | None = None, *, reload: bool = False) -> CostLogger:
    """비용 로거를 돌려줍니다. 프로세스 안에서 하나만 씁니다."""
    global _cost_logger
    settings = settings or get_settings()
    if _cost_logger is None or reload:
        _cost_logger = CostLogger(
            log_path=settings.cost_log_path,
            monthly_budget_usd=settings.monthly_budget_usd,
        )
    return _cost_logger


def get_openai_chat(
    role: OpenAIRole,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    source: str | None = None,
    settings: Settings | None = None,
    enable_cache: bool = True,
) -> ChatOpenAI:
    """OpenAI 채팅 모델을 만듭니다.

    temperature 기본값이 0인 이유: 평가 결과를 재현할 수 있어야 하기 때문입니다.
    같은 입력에 매번 다른 출력이 나오면 지표를 비교할 수 없습니다.

    호출 전에 예산을 확인합니다. 이미 상한에 도달했다면 여기서 BudgetExceededError 가
    발생해 더 이상 과금되지 않습니다.
    """
    settings = settings or get_settings()
    settings.require_openai_key()
    if enable_cache:
        setup_cache(settings)

    logger = get_cost_logger(settings)
    logger.assert_within_budget()

    model = settings.model_for(role)
    handler = CostCallbackHandler(
        logger, model=model, source=source or role, provider="openai"
    )
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[handler],
    )


def get_ollama_chat(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    num_predict: int | None = None,
    source: str = "generator",
    settings: Settings | None = None,
    enable_cache: bool = True,
) -> ChatOllama:
    """로컬 Ollama 채팅 모델을 만듭니다.

    Ollama 는 ai-server 의 127.0.0.1:11434 에만 바인딩되어 있으므로, 먼저 터널을
    열어야 합니다.

        ssh -N -L 11434:127.0.0.1:11434 ai-server

    비용은 들지 않지만 토큰 수는 동일한 로그에 남겨 둡니다. 로컬 모델이 토큰을
    얼마나 쓰는지도 운영 지표이기 때문입니다.
    """
    settings = settings or get_settings()
    if enable_cache:
        setup_cache(settings)

    model = model or settings.ollama_generator_model
    handler = CostCallbackHandler(
        get_cost_logger(settings), model=model, source=source, provider="local"
    )
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        callbacks=[handler],
    )
