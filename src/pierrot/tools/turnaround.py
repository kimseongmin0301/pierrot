"""'흑자 전환' 같은 부호 주장을 가려냅니다.

왜 따로 다루는가
----------------
지금까지 "흑자 전환"은 **단순한 증가**로 취급됐습니다.
`direction_rescue.UP_WORDS` 에 증가·상승과 나란히 들어 있었기 때문입니다.
그래서 숫자가 늘기만 하면 통과했습니다.

실측 오탐(가짜 데이터셋 20번):

    "카카오의 2025년 연결 영업이익은 전년 대비 흑자로 전환됐습니다"

    카카오 연결 영업이익  2024년 +4,602억  ->  2025년 +7,320억

늘어난 것은 맞지만 **2024년에 이미 흑자였습니다.** 전환이 아니라 증가입니다.
주장은 거짓인데 '근거 있음' 이 났습니다. 최상급 주장('역대 최대')을 방향
주장으로 읽어 통과시켰던 것과 **완전히 같은 계열**입니다. 참인 일부(증가)로
더 강한 주장(전환)을 통과시킨 것입니다.

무엇이 달라지는가
-----------------
전환은 증감이 아니라 **부호가 바뀌었는지**로 판정해야 합니다.

    흑자 전환  = 직전이 음수이고 현재가 양수
    적자 전환  = 직전이 양수이고 현재가 음수

상태 주장도 같이 다룹니다. "순이익이 적자를 기록하였다" 는 방향 주장이
아니라 **현재 부호**에 대한 주장입니다. 전에는 `적자` 가 DOWN_WORDS 에
들어 있어서 '감소했다' 로 읽혔습니다. 적자인데 적자 폭이 줄어든 해에는
거짓 '모순' 이 날 수 있는 자리였습니다.

    흑자 상태  = 현재가 양수
    적자 상태  = 현재가 음수

손대지 않는 조건
----------------
`superlative` 와 같습니다. 금액·부문·인과가 섞이면 비켜섭니다.
금액이 적힌 전환 주장("1,000억 흑자로 전환")은 금액 주장이 맡습니다.
이 경우 전환 여부 자체는 확인되지 않지만, 지금과 달라지는 것이 없어
새로 생기는 위험은 없습니다.
"""

from __future__ import annotations

import re

from .direction_rescue import (
    CAUSAL_WORDS,
    HAS_NUMBER,
    SEGMENT_WORDS,
    find_all_accounts,
)

TO_PROFIT = "TO_PROFIT"
"""적자에서 흑자로. 직전 < 0 이고 현재 > 0 이어야 참입니다."""

TO_LOSS = "TO_LOSS"
"""흑자에서 적자로. 직전 > 0 이고 현재 < 0 이어야 참입니다."""

PROFIT = "PROFIT"
"""흑자 상태. 현재 > 0 이어야 참입니다."""

LOSS = "LOSS"
"""적자 상태. 현재 < 0 이어야 참입니다."""

# 전환을 먼저 봅니다. "적자에서 흑자로 전환" 에는 '적자' 도 들어 있어서
# 상태 주장을 먼저 보면 엉뚱하게 적자 주장으로 읽힙니다.
_TO_PROFIT = re.compile(
    r"흑자\s*(전환|달성|기조로)|흑자로\s*(전환|돌아|바뀌|돌아섰|전환되)"
    r"|적자를?\s*(벗어|탈출|면)|흑자\s*로\s*전환"
)

_TO_LOSS = re.compile(
    r"적자\s*(전환|기록으로)|적자로\s*(전환|돌아|바뀌|돌아섰|전환되)"
    r"|흑자에서\s*적자|적자\s*로\s*전환"
)

# 상태 주장. 전환 표현이 아닌 '흑자' / '적자' 입니다.
_PROFIT = re.compile(r"흑자")
_LOSS = re.compile(r"적자|순손실|영업손실")


def detect(text: str) -> tuple[str, str]:
    """(종류, 계정)을 돌려줍니다. 부호 주장이 아니면 ("", "").

    종류는 TO_PROFIT / TO_LOSS / PROFIT / LOSS 중 하나입니다.
    """
    target = (text or "").strip()
    if not target:
        return "", ""
    if HAS_NUMBER.search(target):
        return "", ""          # 금액이 적혀 있으면 금액 주장이 맡습니다
    if SEGMENT_WORDS.search(target):
        return "", ""          # 부문 주장은 전사 수치와 견주면 안 됩니다
    if CAUSAL_WORDS.search(target):
        return "", ""          # 방향이 맞아도 원인까지 맞은 것은 아닙니다

    if _TO_PROFIT.search(target):
        kind = TO_PROFIT
    elif _TO_LOSS.search(target):
        kind = TO_LOSS
    elif _PROFIT.search(target) and _LOSS.search(target):
        # 전환 표현 없이 흑자와 적자가 같이 나오면 해석할 수 없습니다.
        return "", ""
    elif _PROFIT.search(target):
        kind = PROFIT
    elif _LOSS.search(target):
        kind = LOSS
    else:
        return "", ""

    # 계정이 둘 이상이면 비켜섭니다.
    #
    #     "매출은 증가했고 영업이익은 흑자 전환했다"
    #
    # 전환만 확인하고 통과시키면 **매출 부분은 아무것도 보지 않은 것**입니다.
    # 문장 일부만 보고 판정하지 않는다는 규칙은 여기에도 그대로 적용됩니다.
    accounts = find_all_accounts(target)
    if len(accounts) != 1:
        return "", ""
    return kind, accounts[0]


def needs_previous(kind: str) -> bool:
    """직전 수치가 있어야 판정할 수 있는 종류인지."""
    return kind in (TO_PROFIT, TO_LOSS)
