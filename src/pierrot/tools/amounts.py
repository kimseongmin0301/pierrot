"""한국어 금액 표기를 숫자로 바꿉니다.

"9조 1,800억원" -> 9,180,000,000,000

## 왜 파이썬이 하는가

LLM 에게 "9조 1,800억원이 몇 원이냐"고 물으면 자릿수를 틀립니다.
조는 10의 12승, 억은 10의 8승이라 0을 네 개 더 붙이거나 덜 붙이는 실수가 잦고,
그 실수는 1만 배 차이를 만듭니다. 주가의 날짜 계산을 LLM 에게 맡기지 않은 것과
같은 이유입니다.

LLM 은 "이 문장에서 금액이 어디인가"만 찾고, 변환은 규칙으로 처리합니다.
"""

from __future__ import annotations

import re

UNITS: list[tuple[str, int]] = [
    ("조", 10**12),
    ("억", 10**8),
    ("만", 10**4),
]

SCALES: dict[str, int] = dict(UNITS)

SUB_SCALES: dict[str, int] = {"천": 1000, "백": 100, "십": 10}
"""단위 앞에 붙는 배수. "9천억" 은 9 x 1000 x 1억 입니다.

실제 샘플에 "7조 9천억 원" 이 나왔는데 이것을 못 읽어 7조로 파싱한 적이 있습니다.
9천억(9000억)이 통째로 사라져 판정이 틀렸습니다.
"""

# 숫자 + (천/백/십) + 단위. "9조", "1,800억", "9천억", "3.5조" 를 모두 받습니다.
_CHUNK = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(천|백|십)?\s*(조|억|만)?")

_NEGATIVE_WORDS = ("적자", "손실", "마이너스")


def parse_korean_amount(text: str) -> int | None:
    """한국어 금액 표기를 원 단위 정수로 바꿉니다.

    받는 형태:
        "9조 1,800억원"  -> 9_180_000_000_000
        "32조 7,259억원" -> 32_725_900_000_000
        "3,000억원"      -> 300_000_000_000
        "1조원"          -> 1_000_000_000_000
        "약 9.2조원"     -> 9_200_000_000_000
        "71,800원"       -> 71_800

    단위가 하나도 없으면 원 단위 숫자로 봅니다.
    숫자를 찾지 못하면 None 입니다. 추측하지 않습니다.
    """
    if not text:
        return None

    cleaned = text.strip()
    negative = cleaned.startswith("-") or any(word in cleaned for word in _NEGATIVE_WORDS)

    total = 0
    found = False
    plain: float | None = None

    for match in _CHUNK.finditer(cleaned):
        raw_number, sub_unit, unit = match.group(1), match.group(2), match.group(3)
        try:
            number = float(raw_number.replace(",", ""))
        except ValueError:
            continue

        if sub_unit:
            number *= SUB_SCALES[sub_unit]

        if unit:
            total += number * SCALES[unit]
            found = True
        elif sub_unit:
            # "9천원" 처럼 큰 단위 없이 천/백/십만 붙은 경우.
            total += number
            found = True
        elif found:
            # 단위가 이미 나온 뒤의 숫자는 나머지(원 단위)입니다.
            # "8만 6천 700원" 의 700 이 여기 해당합니다.
            total += number
        else:
            # 단위가 하나도 없으면 그 숫자를 원 단위로 봅니다.
            plain = number

    if not found:
        if plain is None:
            return None
        value = int(round(plain))
        return -value if negative else value

    value = int(round(total))
    return -value if negative else value


def format_korean_amount(amount: float) -> str:
    """숫자를 한국어 금액 표기로. 판정 근거를 보여 줄 때 씁니다.

    LLM 답변이 "9조 1,800억원"처럼 말하므로 근거도 같은 단위로 보여 줘야
    사람이 바로 비교할 수 있습니다. 큰 금액은 두 단위까지만 쓰고 나머지는
    버립니다. 재무 수치를 읽을 때 사람이 하는 방식과 같습니다.
    """
    sign = "-" if amount < 0 else ""
    value = abs(int(amount))

    cho, remainder = divmod(value, 10**12)
    eok = remainder // 10**8
    if cho and eok:
        return f"{sign}{cho:,}조 {eok:,}억원"
    if cho:
        return f"{sign}{cho:,}조원"
    if eok:
        return f"{sign}{eok:,}억원"

    man, rest = divmod(value, 10**4)
    if man and rest:
        return f"{sign}{man:,}만 {rest:,}원"
    if man:
        return f"{sign}{man:,}만원"
    return f"{sign}{value:,}원"


def parse_percent(text: str) -> float | None:
    """'277%', '약 15.7%' 같은 표기에서 숫자를 꺼냅니다."""
    if not text:
        return None
    match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*%", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None
