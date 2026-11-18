"""테스트셋의 `source_fact` 토막 라벨을 자연스러운 문장으로 바꿉니다.

왜 필요한가: NLI 모델(KLUE NLI)은 **문장 쌍**으로 학습했습니다. 전제가
"삼성전자 2025-04-25 종가 55,700원" 같은 토막이면 문장으로 보지 않아
주장과 값이 완전히 같아도 중립을 냅니다. 실제로 측정해 보니 PRICE 주장의
중립 확률 평균이 0.912 였습니다.

이 상태로 "NLI 는 주가를 못 본다"고 결론 내리면 **모델의 한계가 아니라
제 입력 형식 때문에 생긴 결과를 모델 탓으로 돌리는 것**이 됩니다.
그래서 전제를 문장으로 바꾼 조건을 따로 만들어 비교합니다.

문장 형식은 `pierrot.tools.evidence.financial_evidence` 가 실제 파이프라인에서
만드는 문장과 같은 모양으로 맞췄습니다.
"""

from __future__ import annotations

import re

# 삼성전자 2025-04-25 종가 55,700원
_DAILY = re.compile(r"^(?P<name>\S+) (?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2}) 종가 (?P<close>[\d,]+)원$")

# 삼성전자 2025년 11월 저가 94,500 ~ 고가 112,400원 (2025-11-25 종가 99,300원)
_RANGE = re.compile(
    r"^(?P<name>\S+) (?P<y>\d{4})년 (?P<m>\d{1,2})월 저가 (?P<low>[\d,]+) ~ 고가 (?P<high>[\d,]+)원"
    r" \((?P<date>\d{4}-\d{2}-\d{2}) 종가 (?P<close>[\d,]+)원\)$"
)

# 삼성전자 2024년 연간 영업이익 연결 32조 7,259억원 / ... 1분기 ... -3조 4,023억원
_AMOUNT = re.compile(
    r"^(?P<name>\S+) (?P<y>\d{4})년 (?P<period>연간|\d분기) (?P<account>\S+) "
    r"(?P<fs>연결|별도) (?P<amount>-?[\d조,억원 ]+)$"
)

# 기아 2024년 영업이익 11조 6,078억원 -> 12조 6,671억원 (+9.1%)
_GROWTH = re.compile(
    r"^(?P<name>\S+) (?P<y>\d{4})년 (?P<account>\S+) (?P<before>-?[\d조,억원 ]+)"
    r" -> (?P<after>-?[\d조,억원 ]+) \((?P<sign>[+-])(?P<rate>[\d.]+)%\)$"
)


def render(source_fact: str) -> tuple[str, str]:
    """(문장, 적용한 규칙 이름)을 돌려줍니다. 못 바꾸면 원문 그대로 돌려줍니다."""
    text = (source_fact or "").strip()

    match = _DAILY.match(text)
    if match:
        g = match.groupdict()
        return (
            f"{g['name']}의 주가는 {g['y']}년 {int(g['m'])}월 {int(g['d'])}일 "
            f"종가 {g['close']}원이다.",
            "daily",
        )

    match = _RANGE.match(text)
    if match:
        g = match.groupdict()
        return (
            f"{g['name']}의 주가는 {g['y']}년 {int(g['m'])}월 한 달 동안 "
            f"최저 {g['low']}원에서 최고 {g['high']}원 사이에서 움직였고, "
            f"{g['date']} 종가는 {g['close']}원이었다.",
            "range",
        )

    match = _AMOUNT.match(text)
    if match:
        g = match.groupdict()
        period = "연간" if g["period"] == "연간" else g["period"]
        return (
            f"{g['name']}의 {g['y']}년 {period} {g['account']}은 "
            f"{g['fs']} 기준 {g['amount'].strip()}이다.",
            "amount",
        )

    match = _GROWTH.match(text)
    if match:
        g = match.groupdict()
        word = "증가" if g["sign"] == "+" else "감소"
        return (
            f"{g['name']}의 {g['y']}년 {g['account']}은 전년 {g['before'].strip()}에서 "
            f"{g['after'].strip()}으로 {g['rate']}% {word}했다.",
            "growth",
        )

    return text, "none"
