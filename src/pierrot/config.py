"""설정 로딩.

모델 ID, 경로, 예산 상한처럼 코드 여기저기에 흩어지면 곤란한 값을 전부 여기에 모읍니다.
값의 출처는 프로젝트 루트의 `.env` 파일 하나뿐입니다.

왜 이렇게 하는가:
- 모델 ID가 코드 안에 박혀 있으면 모델을 바꿀 때 찾아다녀야 합니다.
- 평가 결과를 재현하려면 "그때 어떤 설정으로 돌렸는지"가 한곳에 남아 있어야 합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv

# src/pierrot/config.py -> src/pierrot -> src -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

_env_loaded = False


def _ensure_env_loaded() -> None:
    """`.env`를 한 번만 읽어 환경 변수로 올립니다."""
    global _env_loaded
    if not _env_loaded:
        # override=False: 이미 셸에 설정된 환경 변수가 있으면 그쪽을 우선합니다.
        load_dotenv(ENV_PATH, override=False)
        _env_loaded = True


def _get(key: str, default: str = "") -> str:
    _ensure_env_loaded()
    value = os.environ.get(key, default)
    return value.strip() if value else default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f".env 의 {key} 값을 숫자로 읽을 수 없습니다: {raw!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    """이 프로젝트가 사용하는 모든 설정값."""

    # --- OpenAI (유료) ---
    openai_api_key: str
    openai_extractor_model: str
    openai_verifier_model: str
    openai_judge_model: str

    # --- Ollama (ai-server, 무료) ---
    ollama_base_url: str
    ollama_generator_model: str
    ollama_generator_alt_model: str
    ollama_embedding_model: str

    # --- 검증 도구 ---
    dart_api_key: str
    nli_model: str
    nli_device: str

    # --- 비용 통제 ---
    monthly_budget_usd: float

    # --- 경로 ---
    project_root: Path
    cache_path: Path
    cost_log_path: Path
    snapshot_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        _ensure_env_loaded()
        return cls(
            openai_api_key=_get("OPENAI_API_KEY"),
            openai_extractor_model=_get("OPENAI_EXTRACTOR_MODEL", "gpt-4o-mini"),
            openai_verifier_model=_get("OPENAI_VERIFIER_MODEL", "gpt-4o-mini"),
            openai_judge_model=_get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
            ollama_base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_generator_model=_get("OLLAMA_GENERATOR_MODEL", "gemma3:4b"),
            ollama_generator_alt_model=_get("OLLAMA_GENERATOR_ALT_MODEL", "qwen3:4b"),
            ollama_embedding_model=_get("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
            dart_api_key=_get("DART_API_KEY"),
            nli_model=_get("NLI_MODEL", "Huffon/klue-roberta-base-nli"),
            nli_device=_get("NLI_DEVICE", "cpu"),
            monthly_budget_usd=_get_float("MONTHLY_BUDGET_USD", 5.0),
            project_root=PROJECT_ROOT,
            cache_path=PROJECT_ROOT / ".cache" / "llm_cache.sqlite",
            cost_log_path=PROJECT_ROOT / "logs" / "cost.jsonl",
            snapshot_dir=PROJECT_ROOT / "data" / "snapshots",
        )

    def model_for(self, role: str) -> str:
        """역할 이름으로 OpenAI 모델 ID를 돌려줍니다."""
        table = {
            "extractor": self.openai_extractor_model,
            "verifier": self.openai_verifier_model,
            "judge": self.openai_judge_model,
        }
        if role not in table:
            raise KeyError(
                f"알 수 없는 역할입니다: {role!r} (가능한 값: {sorted(table)})"
            )
        return table[role]

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 가 비어 있습니다. "
                f"{ENV_PATH} 에 값을 채워 주십시오."
            )
        return self.openai_api_key

    def masked(self) -> dict:
        """로그나 화면에 출력해도 안전한 형태로 바꿉니다 (키는 가립니다)."""
        data = asdict(self)
        for key in ("openai_api_key", "dart_api_key"):
            value = data.get(key) or ""
            data[key] = f"설정됨(길이 {len(value)})" if value else "(비어 있음)"
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """설정 객체를 돌려줍니다. 기본적으로 한 번만 읽고 재사용합니다."""
    global _settings
    if _settings is None or reload:
        _settings = Settings.load()
    return _settings
