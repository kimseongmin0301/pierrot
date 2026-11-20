"""'전년과 같은 수준입니다' 처럼 변화가 없다고 말한 주장.

왜 필요한가
-----------
지금은 이런 문장이 전부 "증감 방향이 분명하지 않습니다" 로 흘러갑니다.
파서가 direction=NONE 을 돌려주기 때문입니다. 하지만 **같다는 것도 주장**
입니다. 확인할 수 있고, 틀렸으면 잡아야 합니다.

놓친 실측(가짜 데이터셋):

    07번 "하나금융지주 2025년 순이익은 전년과 같은 수준입니다"
         실제 3조 7,700억 -> 4조   366억  (+7.2%)

    13번 "SKT 2026년 2분기 순이익은 전년 동기와 같았습니다"
         실제   800억 -> 4조 7,000억 수준  (+488%)

13번은 누가 봐도 '같은 수준' 이 아닙니다. 그런데 확인 불가로 흘려보냈습니다.
오탐은 아니지만 **잡을 수 있는 거짓을 놓친 것**입니다.

어디까지를 '같다' 고 볼 것인가
------------------------------
'같은 수준' 에는 정확한 경계가 없습니다. 그래서 구간을 셋으로 나눕니다.

    |변동률| <= 3%   -> 같은 수준이 맞습니다          (근거 있음)
    |변동률| >  10%  -> 같은 수준이라 할 수 없습니다  (모순)
    그 사이          -> 애매합니다                    (확인 불가)

가운데를 억지로 한쪽으로 몰지 않습니다. 이 프로젝트에서 확인 불가의
**사유가 결과 그 자체**입니다. 다만 사유에 실제 변동률을 적어 두어,
읽는 사람이 직접 판단할 수 있게 합니다. "증감 방향이 분명하지 않습니다"
보다는 "실제로는 +7.2% 변했습니다" 가 훨씬 쓸모 있습니다.
"""

from __future__ import annotations

import re

from .direction_rescue import (
    CAUSAL_WORDS,
    DOWN_WORDS,
    HAS_NUMBER,
    SEGMENT_WORDS,
    UP_WORDS,
    find_all_accounts,
)

SAME_TOLERANCE_PCT = 3.0
"""이 안쪽이면 '같은 수준' 이 맞다고 봅니다."""

CLEARLY_DIFFERENT_PCT = 10.0
"""이 바깥이면 '같은 수준' 이라 할 수 없습니다."""

# 변화가 없다는 뜻.
#
# '유지' 는 일부러 뺐습니다. "성장세를 유지했다" 는 계속 늘었다는 뜻이지
# 그대로였다는 뜻이 아닙니다. 낱말 하나 때문에 정반대로 읽을 수 있습니다.
EQUAL_WORDS = re.compile(
    r"같은\s*수준|비슷한\s*수준|유사한\s*수준|동일한\s*수준"
    r"|전년\s*수준|예년\s*수준"
    r"|비슷했|유사했|같았|동일했|다르지\s*않"
    r"|변화가\s*없|변동이\s*없|차이가\s*없"
    r"|제자리|보합|횡보"
)


def detect(text: str) -> str:
    """동등 주장이면 대상 계정 이름을, 아니면 빈 문자열을 돌려줍니다.

    손대지 않는 조건은 다른 가드와 같습니다. 여기에 하나가 더 있습니다.
    증가·감소 낱말이 같이 나오면 해석할 수 없으므로 비켜섭니다.
    """
    target = (text or "").strip()
    if not target:
        return ""
    if not EQUAL_WORDS.search(target):
        return ""
    if HAS_NUMBER.search(target):
        return ""          # 금액이 적혀 있으면 금액 주장이 맡습니다
    if SEGMENT_WORDS.search(target):
        return ""          # 부문 주장은 전사 수치와 견주면 안 됩니다
    if CAUSAL_WORDS.search(target):
        return ""          # 원인까지 확인한 것이 아닙니다
    if UP_WORDS.search(target) or DOWN_WORDS.search(target):
        return ""          # "늘어난 뒤 비슷한 수준" 류는 해석할 수 없습니다

    # 계정이 둘 이상이면 문장 일부만 보고 판정하게 됩니다.
    accounts = find_all_accounts(target)
    if len(accounts) != 1:
        return ""
    return accounts[0]
