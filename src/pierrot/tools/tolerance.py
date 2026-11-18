"""허용 오차 계산.

주장에 적힌 **자릿수**에서 허용 오차를 읽어 냅니다.
정밀하게 말할수록 엄격하게, 어림수로 말할수록 너그럽게 봅니다.

## 왜 일률적인 비율을 쓰지 않는가

주가 검증에서 실제로 겪은 일입니다. 가격 주장에 일률 3%를 적용했더니
"2024년 4월 중순 POSCO홀딩스 주가는 36만원이었다"가 근거 있음으로 통과했습니다.
그 기간 실제 종가는 371,000~400,000원이고 장중 저가 최저도 371,000원이라
36만원은 한 번도 나온 적 없는 가격인데, 3% 하한(359,870원)을 130원 차이로
넘겨 통과한 것입니다. 환각을 놓친 것(미탐)입니다.

자릿수 기반으로 바꾸면 "36만원"은 만원 단위로 말한 것이므로 ±5,000원이 되고,
상한 365,000원이 실제 최저 371,000원에 못 미쳐 제대로 잡힙니다.

주가와 재무 수치가 같은 규칙을 쓰도록 이 모듈 하나에 둡니다.
복사해 두면 한쪽만 고쳐지는 일이 생깁니다.
"""

from __future__ import annotations

MIN_RELATIVE_TOLERANCE = 0.005
"""허용 오차의 하한(값 대비 비율). 반올림 차이로 모순 판정이 나는 것을 막습니다."""

MAX_RELATIVE_TOLERANCE = 0.10
"""허용 오차의 상한(값 대비 비율). 어림수 표기가 지나치게 관대해지는 것을 막습니다.

처음에는 5% 였는데, 정답을 아는 주장으로 재 보니 **한 자릿수 금액에서
자릿수 규칙을 무력화**하고 있었습니다.

    주장 "7조원", 실제 6조 5,669억원 (차이 6.6%)
    조 단위로 말했으니 허용 오차는 반올림 단위의 절반인 ±5,000억원이 맞는데,
    5% 상한이 3,500억원으로 깎아 버려 모순으로 판정했습니다.

사람은 6조 5,669억원을 "약 7조원"이라고 말합니다. 이것을 틀렸다고 하면
검증기를 쓸 수 없습니다. 상한을 10% 로 올려 측정한 결과:

    7조원   vs 6조 5,669억 (6.6%)  -> 모순에서 근거 있음으로 (고쳐짐)
    2조원   vs 2조 1,632억 (7.5%)  -> 모순에서 근거 있음으로 (고쳐짐)
    1조원   vs 5,753억    (73.8%)  -> 모순 유지 (과장이므로 맞음)
    36만원  vs 371,000원   (3.0%)  -> 모순 유지 (기존 회귀 사례 보존)

상한은 한 자릿수 값에서만 작동합니다. 자릿수가 많으면 반올림 단위의 절반이
이미 작아서 상한에 닿지 않습니다.
"""


def rounding_unit(value: float) -> float:
    """숫자 표기에서 읽히는 반올림 단위.

    360000 -> 10000 (만원 단위로 말했다)
    71800  -> 100   (백원 단위로 말했다)
    3.5    -> 0.1   (소수 첫째 자리까지 말했다)
    """
    if not value:
        return 1.0
    magnitude = abs(float(value))

    text = f"{magnitude:.10f}".rstrip("0")
    if "." in text and not text.endswith("."):
        decimals = len(text.split(".", 1)[1])
        return 10.0 ** (-decimals)

    integer = int(round(magnitude))
    if integer == 0:
        return 1.0

    unit = 1
    while integer % (unit * 10) == 0 and unit < 10**12:
        unit *= 10
    return float(unit)


def tolerance_for(
    value: float,
    *,
    min_relative: float = MIN_RELATIVE_TOLERANCE,
    max_relative: float = MAX_RELATIVE_TOLERANCE,
    min_absolute: float | None = None,
    max_absolute: float | None = None,
) -> float:
    """값 하나의 허용 오차(절대값).

    반올림 단위의 절반을 쓰되, 비율 하한과 상한으로 묶습니다.
    등락률처럼 절대값으로 묶고 싶으면 min_absolute / max_absolute 를 씁니다.
    """
    magnitude = abs(float(value))
    result = rounding_unit(magnitude) / 2

    if min_absolute is not None or max_absolute is not None:
        if min_absolute is not None:
            result = max(result, min_absolute)
        if max_absolute is not None:
            result = min(result, max_absolute)
        return result

    floor = magnitude * min_relative
    ceiling = magnitude * max_relative
    return min(max(result, floor), ceiling)


def intervals_overlap(
    low_a: float, high_a: float, low_b: float, high_b: float
) -> bool:
    return not (high_a < low_b or high_b < low_a)
