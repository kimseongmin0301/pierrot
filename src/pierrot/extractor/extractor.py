"""주장 분해 실행.

LLM에게 구조화된 JSON 출력을 강제해 주장 목록을 받고, 각 주장이 원문의 어느 위치에서
나왔는지 파이썬이 직접 찾아 붙입니다.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings, get_settings
from ..llm import get_openai_chat
from .models import Claim, ExtractedClaimList, ExtractionResult
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 원문 위치 찾기
# ---------------------------------------------------------------------------


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """공백을 하나로 줄인 문자열과, 각 글자가 원문 어디에 있었는지의 대응표를 만듭니다.

    LLM이 ``source_text`` 를 옮기면서 줄바꿈을 공백으로 바꾸거나 공백 개수를 바꾸는 일이
    흔합니다. 그 정도 차이는 "원문에 없다"고 볼 이유가 없으므로 공백을 무시하고 한 번 더
    찾아봅니다. 대응표가 있어야 찾은 위치를 원문 좌표로 되돌릴 수 있습니다.
    """
    chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = False

    for position, char in enumerate(text):
        if char.isspace():
            if previous_was_space or not chars:
                continue
            chars.append(" ")
            index_map.append(position)
            previous_was_space = True
        else:
            chars.append(char)
            index_map.append(position)
            previous_was_space = False

    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()

    return "".join(chars), index_map


def resolve_span(answer: str, source_text: str) -> tuple[int, int] | None:
    """``source_text`` 가 원문의 어디에 있는지 (시작, 끝) 위치를 찾습니다.

    찾지 못하면 None 을 돌려줍니다. 이는 LLM이 원문에 없는 문장을 지어냈다는 신호입니다.
    """
    if not answer or not source_text:
        return None

    exact = answer.find(source_text)
    if exact >= 0:
        return (exact, exact + len(source_text))

    normalized_answer, index_map = _normalize_with_map(answer)
    normalized_source, _ = _normalize_with_map(source_text)
    if not normalized_source:
        return None

    found = normalized_answer.find(normalized_source)
    if found < 0:
        return None

    start = index_map[found]
    end = index_map[found + len(normalized_source) - 1] + 1
    return (start, end)


# ---------------------------------------------------------------------------
# 분해
# ---------------------------------------------------------------------------


def build_extractor(
    *,
    settings: Settings | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 4096,
) -> Any:
    """구조화 출력이 강제된 LLM 체인을 만듭니다.

    ``method="json_schema"`` + ``strict=True`` 는 OpenAI Structured Outputs 입니다.
    모델이 스키마를 벗어난 JSON을 만들지 못하게 API 쪽에서 막아 주므로,
    파싱 실패를 처리하는 코드를 따로 쓰지 않아도 됩니다.
    """
    settings = settings or get_settings()
    chat = get_openai_chat(
        "extractor",
        temperature=temperature,
        max_tokens=max_tokens,
        source="extractor",
        settings=settings,
    )
    return chat.with_structured_output(
        ExtractedClaimList, method="json_schema", strict=True
    )


def extract_claims(
    answer: str,
    *,
    extractor: Any | None = None,
    settings: Settings | None = None,
    model_name: str | None = None,
) -> ExtractionResult:
    """답변 텍스트 하나를 주장 목록으로 분해합니다.

    Args:
        answer: 분해할 답변 텍스트.
        extractor: 구조화 출력 체인. 생략하면 설정대로 새로 만듭니다.
            테스트에서는 API를 호출하지 않는 가짜 객체를 넣습니다.
        settings: 설정 객체. 생략하면 기본 설정을 씁니다.
        model_name: 결과에 기록할 모델 이름. 생략하면 설정에서 가져옵니다.
    """
    settings = settings or get_settings()
    if model_name is None:
        model_name = settings.openai_extractor_model
    if extractor is None:
        extractor = build_extractor(settings=settings)

    answer = answer or ""
    if not answer.strip():
        return ExtractionResult(answer=answer, claims=[], model=model_name)

    raw = extractor.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(answer=answer)),
        ]
    )
    return assemble_result(answer, raw, model_name=model_name)


def assemble_result(
    answer: str,
    raw: ExtractedClaimList | dict,
    *,
    model_name: str,
    id_prefix: str = "c",
) -> ExtractionResult:
    """LLM 출력에 원문 위치를 붙여 최종 결과로 만듭니다."""
    if isinstance(raw, dict):
        raw = ExtractedClaimList.model_validate(raw)

    claims: list[Claim] = []
    for index, extracted in enumerate(raw.claims, start=1):
        span = resolve_span(answer, extracted.source_text)
        claims.append(
            Claim(
                claim_id=f"{id_prefix}{index:02d}",
                text=extracted.text.strip(),
                claim_type=extracted.claim_type,
                source_text=extracted.source_text,
                start=span[0] if span else None,
                end=span[1] if span else None,
                span_found=span is not None,
            )
        )

    return ExtractionResult(answer=answer, claims=claims, model=model_name)
