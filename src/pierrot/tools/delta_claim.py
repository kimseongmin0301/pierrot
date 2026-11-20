"""'증가액은 1,000억 원이다' 처럼 **증감 금액**을 말한 주장.

왜 따로 다루는가
----------------
파서가 이것을 증감**률**로 읽습니다. 실측(가짜 데이터셋 10번):

    "한전의 2025년 연결 영업이익은 전년 대비 증가액이 1,000억 원입니다"
    -> growth_percent = 1000.0  (즉 +1000%)

결론은 우연히 맞았습니다(실제 +61.3% 이므로 모순). 하지만 사유에
"주장 +1000.0%" 라고 찍힙니다. **읽는 사람이 보기에 시스템이 문장을
이해하지 못한 것**으로 보입니다. 이 프로젝트에서 사유는 결과 그 자체입니다.

증감액은 증감률과 전혀 다른 단위입니다. 1,000억 원 증가와 1,000% 증가는
100배가 아니라 비교 자체가 불가능한 값입니다.

무엇을 하는가
-------------
'증가액 / 감소액 / 증가폭' 과 금액 표기가 함께 나오면 파이썬이 먼저
집어서 **금액 차이**로 판정하게 합니다.

    실제 증감액 = 해당 기간 수치 - 비교 기준 기간 수치
"""

from __future__ import annotations

import re

from .amounts import parse_korean_amount
from .direction_rescue import CAUSAL_WORDS, SEGMENT_WORDS, find_all_accounts

# 증감 '액' 을 가리키는 말. '증가율' 과 헷갈리지 않도록 률/율은 넣지 않습니다.
DELTA_WORDS = re.compile(
    r"증가액|감소액|증감액|증가폭|감소폭|증감폭|늘어난\s*금액|줄어든\s*금액"
    r"|증가분|감소분"
)

# 금액 표기. range_claim 과 같은 모양입니다.
_PART = r"\d[\d,]*(?:\.\d+)?\s*(?:조|억|만|천)?"
_AMOUNT = re.compile(rf"[-+]?{_PART}(?:\s*{_PART})*\s*원")

# 금액 **바로 뒤**에 증감 동사가 붙은 모양. 사이에 '만큼/가량' 정도만 허용합니다.
#
#   "1,000억 원 증가했다"       -> 걸립니다
#   "3조 원으로 증가했다"        -> 걸리지 않습니다('으로' 가 끼어 있음. 금액 주장입니다)
#   "2조 원에서 3조 원으로 늘었다" -> 걸리지 않습니다(범위 주장이 맡습니다)
_ADJACENT = re.compile(
    rf"([-+]?{_PART}(?:\s*{_PART})*\s*원)\s*(?:만큼|가량)?\s*(증가|감소|늘어|늘었|줄어|줄었)"
)


def detect(text: str) -> tuple[str, int] | None:
    """(계정, 증감액)을 돌려줍니다. 증감액 주장이 아니면 None.

    증감액의 부호는 낱말로 정합니다. '감소액' 이면 음수입니다.
    """
    target = (text or "").strip()
    if not target:
        return None
    if SEGMENT_WORDS.search(target):
        return None
    if CAUSAL_WORDS.search(target):
        return None

    # 두 가지 모양이 있습니다.
    #
    #   (1) "증가액은 1,000억 원이다"       — 낱말이 앞에
    #   (2) "전년 대비 1,000억 원 증가했다" — 금액 바로 뒤에 증감 동사
    #
    # (2) 는 분해 단계가 (1) 을 고쳐 쓰면서 생깁니다. 실측 10번이 그랬습니다.
    # 원문은 "증가액은 1,000억 원입니다" 인데 주장은 "1,000억 원 증가했다" 로
    # 바뀌어 낱말이 사라졌습니다.
    match = _ADJACENT.search(target)
    sign_down = False
    if match is not None:
        chunk = match.group(1)
        sign_down = bool(re.search(r"감소|줄", match.group(2)))
    elif DELTA_WORDS.search(target):
        found = _AMOUNT.search(target)
        if found is None:
            return None
        chunk = found.group(0)
        sign_down = bool(re.search(r"감소액|감소폭|감소분|줄어든\s*금액", target))
    else:
        return None

    amount = parse_korean_amount(chunk)
    if amount is None:
        return None

    accounts = find_all_accounts(target)
    if len(accounts) != 1:
        return None

    if sign_down:
        amount = -abs(amount)
    return accounts[0], amount
