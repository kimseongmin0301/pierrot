"""'2조 원에서 3조 원으로 늘었다' 처럼 시작값과 끝값을 함께 말한 주장.

왜 따로 다루는가
----------------
같은 문형인데 처리가 갈렸습니다. 실측(가짜 데이터셋):

    10번 "2조 원에서 3조 원으로 늘었습니다"   -> 금액 주장으로 읽어 적발
    16번 "20조 원에서 18조 원으로 감소"       -> 증감률로 읽어 적발
    08번 "2조 원에서 3조 원으로 늘었다"       -> **방향 주장**으로 읽어 통과

08번이 문제입니다. LG에너지솔루션 연결 영업이익은 5,753억에서 1조 3,461억으로
늘었습니다. 늘어난 것은 맞지만 **2조도 3조도 아닙니다.** 주장에 적힌 숫자를
통째로 무시하고 '방향만 맞으면 통과' 시킨 것입니다.

파서(gpt-4o-mini)가 같은 문형을 실행마다 다르게 읽는 것이 원인입니다.
이 프로젝트에서 네 번째로 겪는 같은 일입니다. 해법도 같습니다.
**부탁을 검사로 바꿉니다.** 숫자가 둘인 문장은 파이썬이 먼저 집어서
방향 주장으로 흘러가지 못하게 막습니다.

무엇을 확인하는가
-----------------
양 끝을 **둘 다** 봅니다. 끝값만 보면 시작값에 적힌 거짓이 통과합니다.

    끝값   = 주장이 가리키는 기간의 실제 수치
    시작값 = 비교 기준 기간(보통 전년 동기)의 실제 수치

비교 기준 기간 수치가 없으면 끝값만 봅니다. 그래도 방향만 보는 것보다
훨씬 강합니다.
"""

from __future__ import annotations

import re

from .amounts import parse_korean_amount
from .direction_rescue import CAUSAL_WORDS, SEGMENT_WORDS

# "A에서 B로" 꼴. A 와 B 는 금액 표기입니다.
#
# '에서'와 '로/으로' 사이에 다른 말이 끼어들 수 있어 느슨하게 둡니다.
#     "2조 원에서 3조 원으로"
#     "2조원에서 3조원으로"
#     "전년 2조 원에서 3조 원으로"
# 금액은 여러 토막으로 적습니다. "1조 3,461억 원" 은 토막이 둘입니다.
# 한 토막만 받으면 "1조" 까지만 읽고 뒤를 놓칩니다.
_PART = r"\d[\d,]*(?:\.\d+)?\s*(?:조|억|만|천)?"
_AMOUNT = rf"[-+]?{_PART}(?:\s*{_PART})*\s*원?"
RANGE_PATTERN = re.compile(
    rf"({_AMOUNT})\s*에서\s*({_AMOUNT})\s*(?:으로|로)"
)


def detect(text: str) -> tuple[int, int] | None:
    """(시작값, 끝값)을 원 단위로 돌려줍니다. 범위 주장이 아니면 None.

    손대지 않는 조건은 다른 가드와 같습니다.
      - 부문 주장은 전사 수치와 견주면 안 됩니다
      - 인과 주장은 원인까지 확인한 것이 아닙니다
    """
    target = (text or "").strip()
    if not target:
        return None
    if SEGMENT_WORDS.search(target):
        return None
    if CAUSAL_WORDS.search(target):
        return None

    match = RANGE_PATTERN.search(target)
    if match is None:
        return None

    start = parse_korean_amount(match.group(1))
    end = parse_korean_amount(match.group(2))
    if start is None or end is None:
        return None
    # 둘 중 하나라도 '원' 단위 금액으로 읽히지 않으면(연도 같은 맨 숫자)
    # 손대지 않습니다. "2023년에서 2025년으로" 같은 문장을 금액으로 읽으면
    # 안 됩니다.
    if not _looks_like_money(match.group(1)) or not _looks_like_money(match.group(2)):
        return None
    return start, end


_MONEY_MARK = re.compile(r"조|억|만|천|원")


def _looks_like_money(chunk: str) -> bool:
    """'3조 원' 은 금액이고 '2023년' 은 금액이 아닙니다."""
    return bool(_MONEY_MARK.search(chunk or ""))
