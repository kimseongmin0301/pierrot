"""'역대 최대' 같은 최상급 주장을 가려냅니다.

왜 따로 다루는가
----------------
최상급 주장은 **방향 주장도 금액 주장도 아닙니다.**

    "현대차는 2026년 역대 최대 매출을 이어가고 있다"

지금 시스템은 이걸 "재무제표 조회로 확인할 수 있는 주장이 아닙니다" 로
끝냈습니다. 판정은 맞지만 **사유가 틀렸습니다.** 매출 주장이 맞고,
재무제표로 볼 수 있는 항목입니다. 못 보는 이유는 따로 있습니다.

    1. "역대" 는 수집 범위(2022년~) 밖의 과거까지 봐야 합니다
    2. 2026년 연간 매출은 아직 공시되지 않았습니다(3분기까지만 있습니다)

사유가 "확인할 수 있는 주장이 아니다" 로 뭉개지면 **무엇을 더 모아야
판정할 수 있는지** 알 수 없습니다. 이 프로젝트에서 확인 불가의 사유는
결과 그 자체입니다.

한쪽은 확실히 판정할 수 있습니다
--------------------------------
"역대 최대" 를 증명하려면 전체 이력이 필요하지만, **반증하는 데는
한 건이면 충분합니다.** 수집 범위 안에 더 큰 해가 있으면 그 주장은
확실히 거짓입니다. 부문별 매출에서 쓴 상한 판정과 같은 꼴입니다.
"""

from __future__ import annotations

import re

from .direction_rescue import CAUSAL_WORDS, HAS_NUMBER, SEGMENT_WORDS, find_account

# 최상급 표현. "역대" 가 없어도 "사상 최대", "최대치" 로 쓰기도 합니다.
SUPERLATIVE_WORDS = re.compile(
    r"역대\s*(최대|최고|최다)|사상\s*(최대|최고)|최대\s*치|최고\s*치"
    r"|신기록|기록적\s*인?\s*(매출|실적)|역대급"
)


def detect(text: str) -> str:
    """최상급 주장이면 대상 계정 이름을, 아니면 빈 문자열을 돌려줍니다.

    손대지 않는 조건은 방향 주장 되살리기와 같습니다.
      - 금액이 적혀 있으면 금액 주장으로 보는 편이 낫습니다
      - 부문 주장은 전사 수치와 견주면 안 됩니다
      - 인과 주장은 원인까지 확인한 것이 아닙니다
    """
    target = (text or "").strip()
    if not target:
        return ""
    if not SUPERLATIVE_WORDS.search(target):
        return ""
    if HAS_NUMBER.search(target):
        return ""
    if SEGMENT_WORDS.search(target):
        return ""
    if CAUSAL_WORDS.search(target):
        return ""
    return find_account(target)
