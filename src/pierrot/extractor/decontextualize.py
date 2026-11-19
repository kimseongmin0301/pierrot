"""주장에 회사와 시점을 채워 넣는 2단계.

왜 추출 프롬프트에서 같이 하지 않는가
-------------------------------------
해 봤고, 실패했습니다. 추출 프롬프트에 "회사와 연도를 반드시 넣어라"를 더했더니
탈맥락화는 좋아졌지만 **추출이 보수적으로 바뀌어 서술 주장을 놓쳤습니다.**

    버전   주장   NARRATIVE   커버리지   시점없음   주체없음
    v3      91       38       83.2%    33.0%     7.7%
    v4      79       30       74.5%    19.0%     0.0%
    v5      74       24       71.7%    17.6%     2.7%

프롬프트 하나에 "빠짐없이 뽑아라"와 "엄밀하게 다듬어라"를 같이 넣으면 둘 중
하나가 눌립니다. v2->v3 에서 한 번 겪은 일이 반복됐습니다.
누락은 되돌릴 수 없지만 문구는 언제든 다듬을 수 있으므로, 뽑는 쪽에 여유를
주고 다듬는 쪽을 따로 뗐습니다.

왜 "고칠 것만" 보내는가
-----------------------
처음에는 주장 전부를 넘기고 "필요하면 고치라"고 했습니다. **모델이 12개를
하나도 고치지 않고 그대로 돌려줬습니다.** 예시에 적어 둔 바로 그 문장
("HBM4 개발이 성공했다")조차 그대로였습니다. 대부분 손댈 필요가 없는
목록에서는 전체를 복사하는 쪽으로 기울어집니다.

그래서 **파이썬이 먼저 판정합니다.** 회사가 없는지, 연도가 없는지, 등락
주장인데 한쪽만 있는지는 규칙으로 가릴 수 있습니다(`quality.py`).
빠진 것이 있는 주장만, 무엇이 빠졌는지와 함께 보냅니다.
이 프로젝트가 지켜 온 방식 그대로입니다. **파이썬이 판단하고 LLM 은 쓰기만
합니다.**

안전장치
--------
이 단계는 주장을 **다시 쓰기만** 합니다. 개수가 달라지면 결과를 통째로 버리고
원래 주장을 씁니다. "지우지 마십시오"라고 부탁하는 대신 구조로 막습니다.

**없는 회사를 지어내는 것도 코드로 막습니다.** 프롬프트에 "답변에 없는 정보를
만들어 넣지 마십시오"라고 적어 두었지만 지켜지지 않았습니다. 실측 사례입니다.

    원문: "펀더멘탈 분석 결과, 최근 2023년부터 2025년까지의 실적 성장세가
           이어지고 있으며 ..."  (회사 이름이 한 번도 안 나옴)
    결과: "**SK하이닉스**의 2023년부터 2025년까지의 실적 성장세가 이어지고 있다"
          -> 지어낸 회사로 조회해 모순 판정까지 났습니다. 오탐입니다.

그래서 다시 쓴 문장에 들어간 회사 이름이 **원문에 실제로 있는지** 대조하고,
없으면 그 문장만 원래대로 되돌립니다. 부탁이 아니라 검사입니다.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from .models import Claim
from .quality import (
    has_relative_time,
    has_span_endpoints,
    has_year,
    looks_like_span_claim,
)
from ..llm import get_openai_chat

DECONTEXT_VERSION = "decontext-v4"


class RewrittenClaim(BaseModel):
    index: int = Field(description="입력에 적힌 번호를 그대로.")
    text: str = Field(description="빠진 것을 채워 다시 쓴 주장 문장.")
    filled_from: str = Field(
        description=(
            "답변의 어느 부분에서 가져왔는지 짧게. "
            "답변에서 찾지 못해 그대로 두었다면 빈 문자열."
        )
    )


class RewrittenClaimList(BaseModel):
    claims: list[RewrittenClaim] = Field(
        description="입력과 **같은 개수**를, 같은 번호로 돌려주십시오."
    )


SYSTEM_PROMPT = f"""당신은 검증용 주장 문장에 빠진 맥락을 채워 넣는 도구입니다.

답변 원문과, 그 답변에서 뽑아낸 주장 중 **무언가 빠진 것들**을 받습니다.
주장마다 무엇이 빠졌는지도 함께 적혀 있습니다. 그것을 채워 돌려주십시오.

왜 채워야 하는가: 검증 단계는 주장 하나만 따로 받습니다. 답변을 다시 보지
않습니다. 회사나 시점이 없으면 무엇을 조회할지 정할 수 없어 그 주장은
검증되지 못하고 버려집니다.

## 절대 규칙

- 주장을 **지우거나 합치지 마십시오.** 입력이 N개면 출력도 N개, 번호도 그대로입니다.
- **답변에 없는 정보를 만들어 넣지 마십시오.** 답변을 뒤져도 찾을 수 없으면
  그대로 두고 filled_from 을 빈 문자열로 두십시오.
  없는 연도를 지어내면 엉뚱한 값과 대조하게 되어 검증이 틀립니다.
- 수치, 단위, 금액은 **한 글자도 바꾸지 마십시오.** 값이 틀려 보여도 그대로
  두십시오. 틀렸는지 확인하는 것이 다음 단계의 일입니다.
- 뜻을 바꾸지 마십시오. 넣는 것은 **답변에 이미 있는 맥락**뿐입니다.

## 빠진 것이 여러 개면 **전부** 채우십시오

대괄호 안에 둘 이상이 적혀 있으면 하나만 고치고 끝내지 마십시오.
    "[회사 없음, 연도 없음] HBM 3구역 생산이 본격화되었다"
    -> "SK하이닉스는 2024년에 HBM 3구역 생산을 본격화했다"   (둘 다 채움)
    -> "HBM 3구역 생산이 SK하이닉스에서 본격화되었다"        (연도를 빠뜨림. 나쁨)

## 빠진 것별로 할 일

[회사 없음] 답변이 다루는 회사 이름을 넣습니다.
    회사 이름은 **문장 앞에 주어나 소유격으로** 넣으십시오.
    뒤에 "~에서"를 붙이면 말이 어색해지고 뜻도 달라집니다.
    "HBM4 개발이 성공했다"
    -> "SK하이닉스는 HBM4 개발에 성공했다"           (좋음)
    -> "HBM4 개발이 SK하이닉스에서 성공했다"          (어색함. 쓰지 마십시오)
    "D램 시장 경쟁이 심화되고 있다"
    -> "SK하이닉스가 속한 D램 시장의 경쟁이 심화되고 있다"   (좋음)
    -> "D램 시장 경쟁이 SK하이닉스에서 심화되고 있다"        (뜻이 달라짐. 나쁨)

[연도 없음] 답변에서 그 사실이 속한 연도를 찾아 넣습니다.
    "광고 사업 부문 매출은 1,509억 원이다"   (답변이 2024년 1분기 실적을 다룸)
    -> "카카오의 2024년 1분기 광고 사업 부문 매출은 1,509억 원이다"

[구간 한쪽만] 등락·증감 주장에 시작과 끝을 둘 다 적습니다.
    답변: "2024년 1월 초 30만원 수준에서 4월 중순에는 36만원까지 상승하며
           20% 이상 급등했습니다"
    "POSCO홀딩스 주가는 2024년 4월 중순에 20% 이상 급등했다"
    -> "POSCO홀딩스 주가는 2024년 1월 초부터 4월 중순까지 20% 이상 급등했다"

[기준일 없는 상대 표현] "현재", "최근" 을 답변의 기준일로 바꿉니다.
    답변: "... 현재(2024년 7월 26일) ..."
    "셀트리온 주가는 현재까지 17.5% 상승했다"
    -> "셀트리온 주가는 2024년 1월 8일부터 2024년 7월 26일까지 17.5% 상승했다"

## 주의

- 문장을 길게 늘이지 마십시오. 빠진 것만 채웁니다.
- 답변에 연도가 여러 개 있으면, **그 주장이 가리키는** 연도를 고르십시오.
  헷갈리면 그대로 두는 편이 낫습니다.

(version: {DECONTEXT_VERSION})
"""

USER_PROMPT_TEMPLATE = """[답변 시작]
{answer}
[답변 끝]

아래 주장 {count}개에 빠진 맥락을 채워 **{count}개 그대로** 돌려주십시오.

{claims}
"""


def missing_parts(claim: Claim, names: tuple[str, ...]) -> list[str]:
    """주장에 무엇이 빠졌는지. 비어 있으면 손댈 것이 없습니다.

    파이썬이 판정하는 이유: 규칙으로 가릴 수 있는 일이고, 규칙은 테스트로
    고정할 수 있습니다. 그리고 **무엇이 빠졌는지 짚어 주어야** 모델이
    실제로 고칩니다. 전부 넘기고 "알아서 고치라"고 하면 그대로 복사합니다.
    """
    text = claim.text
    parts: list[str] = []
    if names and not any(name and name in text for name in names):
        parts.append("회사 없음")
    if not has_year(text):
        parts.append("연도 없음")
    elif has_relative_time(text):
        parts.append("기준일 없는 상대 표현")
    if looks_like_span_claim(text) and not has_span_endpoints(text):
        parts.append("구간 한쪽만")
    return parts


def invented_company(original: str, rewritten: str, answer: str, names: tuple[str, ...]) -> str:
    """다시 쓴 문장이 **원문에 없는 회사**를 새로 넣었으면 그 이름을 돌려줍니다.

    프롬프트로 "지어내지 마십시오"라고 해도 지켜지지 않습니다.
    지어낸 회사로 조회하면 전혀 다른 회사의 수치와 대조하게 되어
    **판정 자체가 거짓**이 됩니다. 누락보다 나쁩니다.
    """
    for name in names:
        if not name or name in original:
            continue
        if name in rewritten and name not in answer:
            return name
    return ""


def build_decontextualizer(
    *, settings: Settings | None = None, temperature: float = 0.0
) -> Any:
    settings = settings or get_settings()
    chat = get_openai_chat(
        "extractor",
        temperature=temperature,
        max_tokens=4096,
        source="decontextualize",
        settings=settings,
    )
    return chat.with_structured_output(
        RewrittenClaimList, method="json_schema", strict=True
    )


def _universe_names() -> tuple[str, ...]:
    from ..tools.universe import UNIVERSE

    names: list[str] = []
    for stock in UNIVERSE:
        names.append(stock.name)
        names.extend(getattr(stock, "aliases", ()) or ())
    return tuple(names)


def decontextualize(
    answer: str,
    claims: list[Claim],
    *,
    model: Any | None = None,
    settings: Settings | None = None,
    names: tuple[str, ...] | None = None,
) -> tuple[list[Claim], str]:
    """빠진 맥락이 있는 주장만 골라 다듬습니다.

    Returns:
        (다듬은 주장 목록, 무슨 일이 있었는지 적은 쪽지).

        **돌려주는 목록은 언제나 바로 쓸 수 있습니다.** 문제가 생긴 것은
        안에서 이미 원래대로 되돌려 놓습니다.

          - 개수나 번호가 어긋남 -> 전체를 원래대로
          - 원문에 없는 회사를 지어냄 -> 그 문장만 원래대로

        쪽지는 무엇이 되돌려졌는지 알리는 용도이지, 결과를 버리라는 신호가
        아닙니다. 한 문장 때문에 멀쩡한 나머지를 잃으면 안 됩니다.
    """
    if not claims:
        return claims, ""

    settings = settings or get_settings()
    names = _universe_names() if names is None else names

    targets = [
        (index, claim, parts)
        for index, claim in enumerate(claims)
        if (parts := missing_parts(claim, names))
    ]
    if not targets:
        return claims, ""

    model = model or build_decontextualizer(settings=settings)
    listing = "\n".join(
        f"{number}. [{', '.join(parts)}] {claim.text}"
        for number, (_, claim, parts) in enumerate(targets, 1)
    )
    try:
        result = model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=USER_PROMPT_TEMPLATE.format(
                        answer=answer, count=len(targets), claims=listing
                    )
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return claims, f"다듬기 호출 실패: {exc}"

    if isinstance(result, dict):
        result = RewrittenClaimList.model_validate(result)

    # 개수가 다르면 통째로 버립니다. 어느 주장이 사라졌는지 맞춰 볼 방법이
    # 없고, 잘못 맞추면 주장과 근거가 어긋납니다.
    if len(result.claims) != len(targets):
        return claims, f"개수가 다릅니다: 입력 {len(targets)} -> 출력 {len(result.claims)}"

    by_number = {item.index: item for item in result.claims}
    if sorted(by_number) != list(range(1, len(targets) + 1)):
        return claims, "번호가 1..N 과 맞지 않습니다"

    rewritten = list(claims)
    invented: list[str] = []
    for number, (position, claim, _) in enumerate(targets, 1):
        text = (by_number[number].text or "").strip()
        if not text or text == claim.text:
            continue

        # 원문에 없는 회사를 새로 넣었으면 그 문장은 되돌립니다.
        # 지어낸 회사로 조회하면 전혀 다른 회사의 수치와 대조하게 됩니다.
        made_up = invented_company(claim.text, text, answer, names)
        if made_up:
            invented.append(f"{made_up}({claim.claim_id})")
            continue

        rewritten[position] = claim.model_copy(update={"text": text})

    note = ""
    if invented:
        note = "원문에 없는 회사를 넣으려 해 되돌렸습니다: " + ", ".join(invented)
    return rewritten, note
