"""한국어 NLI 모델 래퍼.

두 문장(전제, 가설)을 받아 관계를 세 가지로 분류합니다.

    entailment     전제가 가설을 뒷받침한다   -> SUPPORTED
    contradiction  전제와 가설이 충돌한다     -> CONTRADICTED
    neutral        전제만으로는 알 수 없다    -> UNVERIFIABLE

이 프로젝트에서 NLI 는 **베이스라인**입니다. 도구 기반 검증과 비교하기 위한
기준선이지, 최종 판정기가 아닙니다.

## 라벨 순서를 직접 확인하는 이유

모델마다 라벨 순서가 다릅니다. KLUE-NLI 는 0=entailment, 1=neutral,
2=contradiction 이지만, 다른 데이터로 학습한 모델은 순서가 뒤바뀝니다.
순서를 잘못 잡으면 entailment 를 contradiction 으로 읽어 **판정이 통째로
뒤집히는데, 겉으로는 멀쩡해 보입니다.**

그래서 config 의 id2label 을 먼저 읽고, 이름으로 알 수 없을 때만
KLUE 순서를 가정합니다. 어느 쪽을 썼는지 label_source 에 남겨 둡니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..config import Settings, get_settings

KLUE_FALLBACK_ORDER = ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
"""config 에서 라벨 이름을 읽지 못했을 때 가정하는 순서 (KLUE-NLI 기준)."""


class NliLabel(str, Enum):
    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class NliResult:
    """판정 결과 하나."""

    label: NliLabel
    scores: dict[str, float]
    premise: str
    hypothesis: str

    @property
    def confidence(self) -> float:
        return self.scores.get(self.label.value, 0.0)

    def score(self, label: NliLabel) -> float:
        return self.scores.get(label.value, 0.0)

    def describe(self) -> str:
        parts = " / ".join(
            f"{name.lower()} {value:.3f}" for name, value in sorted(self.scores.items())
        )
        return f"{self.label.value} ({parts})"


def _normalize_label(raw: str) -> str | None:
    text = (raw or "").strip().lower()
    if "entail" in text:
        return NliLabel.ENTAILMENT.value
    if "contradic" in text:
        return NliLabel.CONTRADICTION.value
    if "neutral" in text:
        return NliLabel.NEUTRAL.value
    return None


def resolve_label_order(id2label: dict[int, str]) -> tuple[list[str], str]:
    """모델 설정에서 라벨 순서를 정합니다.

    Returns:
        (인덱스 순서대로의 표준 라벨 목록, 어떻게 정했는지)
    """
    ordered = [id2label[key] for key in sorted(id2label)]
    mapped = [_normalize_label(name) for name in ordered]

    if all(item is not None for item in mapped) and len(set(mapped)) == 3:
        return [item for item in mapped if item], "config"

    if len(ordered) == 3:
        return list(KLUE_FALLBACK_ORDER), "KLUE 기본 순서 가정"

    raise ValueError(
        f"라벨 순서를 정할 수 없습니다: {id2label}. "
        "이 모델은 3분류 NLI 모델이 아닐 수 있습니다."
    )


class NliModel:
    """허깅페이스 NLI 모델.

    API 비용이 들지 않습니다. 모델 파일을 한 번 내려받으면 그 뒤로는
    로컬 CPU 에서 돕니다.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        settings: Settings | None = None,
        max_length: int = 512,
    ) -> None:
        settings = settings or get_settings()
        self.model_name = model_name or settings.nli_model
        self.device = device or settings.nli_device
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._drop_token_type_ids = False
        self.label_order: list[str] = []
        self.label_source: str = ""

    # -- 적재 -------------------------------------------------------------

    def load(self) -> "NliModel":
        """모델을 메모리에 올립니다. 처음 호출 시 다운로드가 일어납니다."""
        if self._model is not None:
            return self

        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        self._torch = torch

        id2label = dict(self._model.config.id2label)
        self.label_order, self.label_source = resolve_label_order(id2label)

        # 문장 구분 임베딩을 쓰지 않는 모델인지 확인합니다.
        self._drop_token_type_ids = getattr(self._model.config, "type_vocab_size", 2) <= 1
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- 추론 -------------------------------------------------------------

    def predict_batch(
        self, pairs: list[tuple[str, str]], *, batch_size: int = 16
    ) -> list[NliResult]:
        """(전제, 가설) 쌍 여러 개를 한 번에 판정합니다."""
        if not pairs:
            return []
        self.load()
        torch = self._torch

        results: list[NliResult] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            encoded = self._tokenizer(
                [premise for premise, _ in chunk],
                [hypothesis for _, hypothesis in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # RoBERTa 계열은 문장 구분 임베딩을 쓰지 않습니다(type_vocab_size=1).
            # 그런데 토크나이저는 두 번째 문장에 token_type_ids=1 을 붙여 주므로
            # 그대로 넘기면 임베딩 인덱스를 벗어나 IndexError 가 납니다.
            if self._drop_token_type_ids:
                encoded.pop("token_type_ids", None)
            encoded = encoded.to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

            for (premise, hypothesis), row in zip(chunk, probabilities):
                scores = {
                    self.label_order[index]: float(value)
                    for index, value in enumerate(row)
                }
                best = max(scores, key=scores.get)
                results.append(
                    NliResult(
                        label=NliLabel(best),
                        scores=scores,
                        premise=premise,
                        hypothesis=hypothesis,
                    )
                )
        return results

    def predict(self, premise: str, hypothesis: str) -> NliResult:
        return self.predict_batch([(premise, hypothesis)])[0]


_shared: NliModel | None = None


def get_nli_model(settings: Settings | None = None) -> NliModel:
    """프로세스 안에서 모델을 하나만 띄웁니다. 적재가 느리기 때문입니다."""
    global _shared
    if _shared is None:
        _shared = NliModel(settings=settings)
    return _shared
