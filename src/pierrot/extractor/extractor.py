"""주장 분해 실행.

LLM에게 구조화된 JSON 출력을 강제해 주장 목록을 받고, 각 주장이 원문의 어느 위치에서
나왔는지 파이썬이 직접 찾아 붙입니다.
"""

from __future__ import annotations

import difflib
from typing import Any, NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings, get_settings
from ..llm import get_openai_chat
from .models import Claim, ExtractedClaimList, ExtractionResult, MatchMode
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# gemma3 같은 모델이 답변에 넣는 마크다운 강조 기호입니다.
# LLM이 source_text 를 옮길 때 이 기호를 빼고 적는 일이 흔합니다.
MARKDOWN_CHARS = "*_`~"

# 인용 구간의 양 끝에서 달라지기 쉬운 문장부호입니다.
# LLM이 복문의 앞 절만 떼어내면서 쉼표를 마침표로 바꾸는 일이 흔합니다.
EDGE_PUNCTUATION = " \t\n.,;:!?\"'()[]{}"

FUZZY_THRESHOLD = 0.85
"""유사도가 이 값 이상이면 같은 구간으로 인정합니다.

너무 낮추면 LLM이 지어낸 문장도 아무 데나 끼워 맞춰져 탐지 기능이 죽습니다.
"""


# ---------------------------------------------------------------------------
# 원문 위치 찾기
# ---------------------------------------------------------------------------


class SpanMatch(NamedTuple):
    """위치 탐색 결과."""

    start: int
    end: int
    mode: MatchMode
    ratio: float


def _normalize_with_map(
    text: str, *, drop_markdown: bool = False
) -> tuple[str, list[int]]:
    """정규화한 문자열과, 각 글자가 원문 어디에 있었는지의 대응표를 만듭니다.

    대응표가 있어야 정규화된 좌표에서 찾은 위치를 원문 좌표로 되돌릴 수 있습니다.

    Args:
        text: 원본 문자열.
        drop_markdown: True 면 마크다운 강조 기호를 제거합니다.
    """
    chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = False

    for position, char in enumerate(text):
        if drop_markdown and char in MARKDOWN_CHARS:
            continue
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


def _find_in_normalized(
    answer: str, source_text: str, *, drop_markdown: bool, mode: MatchMode
) -> SpanMatch | None:
    """정규화한 뒤 부분 문자열로 찾고, 원문 좌표로 되돌립니다."""
    normalized_answer, index_map = _normalize_with_map(answer, drop_markdown=drop_markdown)
    normalized_source, _ = _normalize_with_map(source_text, drop_markdown=drop_markdown)
    if not normalized_source:
        return None

    found = normalized_answer.find(normalized_source)
    if found < 0:
        return None

    start = index_map[found]
    end = index_map[found + len(normalized_source) - 1] + 1
    return SpanMatch(start, end, mode, 1.0)


def _find_fuzzy(answer: str, source_text: str) -> SpanMatch | None:
    """유사도로 가장 비슷한 구간을 찾습니다.

    여기까지 왔다는 것은 LLM이 원문을 제법 고쳐 적었다는 뜻이므로, 찾더라도
    FUZZY 로 표시해 검토 대상으로 남깁니다.
    """
    normalized_answer, index_map = _normalize_with_map(answer, drop_markdown=True)
    normalized_source, _ = _normalize_with_map(source_text, drop_markdown=True)
    if not normalized_source or not normalized_answer:
        return None

    matcher = difflib.SequenceMatcher(
        None, normalized_answer, normalized_source, autojunk=False
    )
    block = matcher.find_longest_match(
        0, len(normalized_answer), 0, len(normalized_source)
    )
    if block.size == 0:
        return None

    # 겹치는 구간을 기준으로 source 길이만큼 창을 잡습니다.
    window_start = max(0, block.a - block.b)
    window_end = min(len(normalized_answer), window_start + len(normalized_source))
    window = normalized_answer[window_start:window_end]

    ratio = difflib.SequenceMatcher(None, window, normalized_source, autojunk=False).ratio()
    if ratio < FUZZY_THRESHOLD:
        return None

    start = index_map[window_start]
    end = index_map[window_end - 1] + 1
    return SpanMatch(start, end, MatchMode.FUZZY, ratio)


def find_span(answer: str, source_text: str) -> SpanMatch | None:
    """``source_text`` 가 원문의 어디에 있는지 단계적으로 찾습니다.

    1. 글자 그대로 찾기                      -> EXACT
    2. 공백·줄바꿈 차이 무시                  -> WHITESPACE
    3. 마크다운 기호와 양 끝 문장부호까지 무시   -> NORMALIZED
    4. 유사도 기준(0.85)으로 인정             -> FUZZY
    5. 그래도 못 찾음                        -> None

    단계를 나누고 결과를 기록하는 이유: 모두 "찾았다"로 뭉뚱그리면 LLM이 원문을
    얼마나 충실히 옮겼는지 알 수 없습니다. 뒤로 갈수록 품질이 나쁘다는 신호입니다.
    """
    if not answer or not source_text:
        return None

    exact = answer.find(source_text)
    if exact >= 0:
        return SpanMatch(exact, exact + len(source_text), MatchMode.EXACT, 1.0)

    match = _find_in_normalized(
        answer, source_text, drop_markdown=False, mode=MatchMode.WHITESPACE
    )
    if match:
        return match

    match = _find_in_normalized(
        answer, source_text, drop_markdown=True, mode=MatchMode.NORMALIZED
    )
    if match:
        return match

    # 양 끝 문장부호를 벗겨내고 다시 시도합니다.
    trimmed = source_text.strip(EDGE_PUNCTUATION)
    if trimmed and trimmed != source_text:
        match = _find_in_normalized(
            answer, trimmed, drop_markdown=True, mode=MatchMode.NORMALIZED
        )
        if match:
            return match

    return _find_fuzzy(answer, source_text)


def resolve_span(answer: str, source_text: str) -> tuple[int, int] | None:
    """``find_span`` 의 결과에서 위치만 꺼냅니다."""
    match = find_span(answer, source_text)
    return None if match is None else (match.start, match.end)


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
    decontextualizer: Any | None = None,
    refine: bool = True,
) -> ExtractionResult:
    """답변 텍스트 하나를 주장 목록으로 분해합니다.

    두 단계로 나뉩니다.

    1. 분해 — 답변에서 검증 가능한 진술을 빠짐없이 뽑습니다(`prompts.py`).
    2. 다듬기 — 각 주장에 회사와 시점을 채웁니다(`decontextualize.py`).

    왜 나누는가: 한 프롬프트에 "빠짐없이 뽑아라"와 "엄밀하게 다듬어라"를 같이
    넣었더니 추출이 보수적으로 바뀌어 서술 주장을 놓쳤습니다
    (커버리지 83.2% -> 71.7%). 자세한 측정은 `prompts.py` 의 버전 이력에 있습니다.

    Args:
        answer: 분해할 답변 텍스트.
        extractor: 구조화 출력 체인. 생략하면 설정대로 새로 만듭니다.
            테스트에서는 API를 호출하지 않는 가짜 객체를 넣습니다.
        settings: 설정 객체. 생략하면 기본 설정을 씁니다.
        model_name: 결과에 기록할 모델 이름. 생략하면 설정에서 가져옵니다.
        decontextualizer: 다듬기 체인. 생략하면 설정대로 새로 만듭니다.
        refine: 다듬기 단계를 돌릴지. 끄면 분해 결과를 그대로 돌려줍니다.
    """
    settings = settings or get_settings()
    if model_name is None:
        model_name = settings.openai_extractor_model
    if extractor is None:
        extractor = build_extractor(settings=settings)

    answer = answer or ""
    if not answer.strip():
        return ExtractionResult(
            answer=answer, claims=[], model=model_name, prompt_version=PROMPT_VERSION
        )

    raw = extractor.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(answer=answer)),
        ]
    )
    result = assemble_result(answer, raw, model_name=model_name)
    if not refine or not result.claims:
        return result

    from .decontextualize import decontextualize

    refined, note = decontextualize(
        answer, result.claims, model=decontextualizer, settings=settings
    )
    # 돌려받은 목록을 **항상** 씁니다.
    #
    # 문제가 생긴 문장은 `decontextualize` 안에서 이미 원래대로 되돌려 놓습니다
    # (개수가 어긋나면 전체를, 없는 회사를 지어냈으면 그 문장만).
    # 여기서 note 가 있다고 전체를 버리면, 한 문장 때문에 멀쩡하게 다듬어진
    # 나머지까지 잃습니다. note 는 무슨 일이 있었는지 알리는 용도입니다.
    return result.model_copy(
        update={"claims": refined, "refined": True, "refine_note": note}
    )


def assemble_result(
    answer: str,
    raw: ExtractedClaimList | dict,
    *,
    model_name: str,
    id_prefix: str = "c",
    prompt_version: str = PROMPT_VERSION,
) -> ExtractionResult:
    """LLM 출력에 원문 위치를 붙여 최종 결과로 만듭니다."""
    if isinstance(raw, dict):
        raw = ExtractedClaimList.model_validate(raw)

    claims: list[Claim] = []
    for index, extracted in enumerate(raw.claims, start=1):
        match = find_span(answer, extracted.source_text)
        claims.append(
            Claim(
                claim_id=f"{id_prefix}{index:02d}",
                text=extracted.text.strip(),
                claim_type=extracted.claim_type,
                source_text=extracted.source_text,
                start=match.start if match else None,
                end=match.end if match else None,
                span_found=match is not None,
                match_mode=match.mode if match else MatchMode.NOT_FOUND,
                match_ratio=match.ratio if match else 0.0,
            )
        )

    return ExtractionResult(
        answer=answer, claims=claims, model=model_name, prompt_version=prompt_version
    )
