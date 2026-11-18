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

# 숫자 + 단위 묶음. "9조", "1,800억", "3.5조" 를 모두 받습니다.
_CHUNK = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만)?")

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
        raw_number, unit = match.group(1), match.group(2)
        try:
            number = float(raw_number.replace(",", ""))
        except ValueError:
            continue

        if unit:
            total += number * SCALES[unit]
            found = True
        else:
            # 단위 없는 숫자는 마지막 것만 기억했다가, 단위가 하나도 없을 때 씁니다.
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
