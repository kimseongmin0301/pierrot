"""분해 품질 자동 점검.

프롬프트를 고칠 때마다 결과를 사람이 눈으로 훑으면 반드시 놓칩니다.
실제로 v2 프롬프트에서 전망 문장이 주장으로 섞여 들어온 것을 뒤늦게 발견했습니다.
그래서 기계가 먼저 훑고 의심스러운 것을 올려 주도록 합니다.

여기 있는 것은 **휴리스틱**입니다. 정답 판정기가 아닙니다.
걸렸다고 반드시 잘못된 주장은 아니고, 안 걸렸다고 올바른 주장도 아닙니다.
사람이 어디를 먼저 볼지 정해 주는 용도입니다.
"""

from __future__ import annotations

import re

from .models import Claim, ClaimType

# 미래 시제와 권고 표현. 이런 문장은 지금 참·거짓을 가릴 수 없으므로
# 주장으로 뽑히면 검증 단계에서 판정 불가 항목이 되어 지표를 망칩니다.
FORECAST_PATTERN = re.compile(
    r"것으로\s*(보|예상|전망|분석|관측)"
    r"|할\s*것(입니다|이다)"
    r"|될\s*것(입니다|이다)"
    r"|전망(이다|입니다|된다|됩니다)"
    r"|기대(된다|됩니다|감)"
    r"|주시해야"
    r"|관찰해야"
    r"|주목할\s*필요"
    r"|해야\s*(한다|합니다|할\s*것)"
    r"|변수가\s*될"
)

# 금액·수량 표현. 이런 값이 들어 있는데 NARRATIVE 로 분류되면
# NLI 로 라우팅되어 숫자를 검증하지 못합니다.
AMOUNT_PATTERN = re.compile(
    r"\d[\d,\.]*\s*(조|억|만|천)\s*(원|달러|유로|엔)"
    r"|\d[\d,\.]*\s*(원|달러|유로|엔)"
)


def looks_like_forecast(text: str) -> bool:
    """전망·권고 문장으로 보이는지."""
    return bool(FORECAST_PATTERN.search(text or ""))


def has_amount(text: str) -> bool:
    """금액 표현이 들어 있는지."""
    return bool(AMOUNT_PATTERN.search(text or ""))


def suspect_forecast(claims: list[Claim]) -> list[Claim]:
    """전망·권고로 의심되는 주장들. 추출되지 말았어야 할 후보입니다."""
    return [claim for claim in claims if looks_like_forecast(claim.text)]


def suspect_type(claims: list[Claim]) -> list[Claim]:
    """유형이 잘못 붙었을 가능성이 있는 주장들.

    지금은 한 가지만 봅니다. 금액이 들어 있는데 NARRATIVE 로 분류된 경우입니다.
    수주액이나 계약 규모가 여기에 걸립니다. 공시의 금액과 대조해야 하므로
    NUMERIC 이어야 합니다.
    """
    return [
        claim
        for claim in claims
        if claim.claim_type is ClaimType.NARRATIVE and has_amount(claim.text)
    ]


def quality_report(
    claims: list[Claim], *, names: tuple[str, ...] = ()
) -> dict[str, object]:
    """분해 품질 지표를 한 번에 계산합니다.

    Args:
        names: 주체 점검에 쓸 회사 표기들. 비우면 주체 점검을 건너뜁니다.
    """
    forecasts = suspect_forecast(claims)
    types = suspect_type(claims)
    review = [claim for claim in claims if claim.needs_review]
    no_time = missing_time(claims)
    no_subject = missing_subject(claims, names) if names else []
    return {
        "total": len(claims),
        "forecast_suspects": [c.claim_id for c in forecasts],
        "type_suspects": [c.claim_id for c in types],
        "span_review": [c.claim_id for c in review],
        "missing_time": [c.claim_id for c in no_time],
        "missing_subject": [c.claim_id for c in no_subject],
        "forecast_count": len(forecasts),
        "type_count": len(types),
        "span_review_count": len(review),
        "missing_time_count": len(no_time),
        "missing_subject_count": len(no_subject),
    }

# ---------------------------------------------------------------------------
# 탈맥락화 점검 (v4 프롬프트 작업에서 추가)
# ---------------------------------------------------------------------------
#
# 5단계에서 확인 불가 54건을 원인별로 갈랐더니 12건이 **주장에 시점이나
# 주체가 없어서** 판정되지 못한 것이었습니다. 원문에는 있던 정보입니다.
#
#     원문: "POSCO홀딩스 주가는 2024년 1월 초 30만원 수준에서 4월 중순에는
#            36만원까지 상승하며 20% 이상 급등했습니다"
#     주장: "2024년 4월 중순 POSCO홀딩스 주가는 20% 이상 급등했다"
#            -> 시작 시점(1월 초)이 빠져 등락률을 계산할 구간이 없습니다
#
# 프롬프트를 고치기 전에 **기계가 셀 수 있는 기준**이 필요합니다.
# 사람이 눈으로 세면 고친 뒤에 좋아졌는지 알 수 없습니다.

# 연도 표기. "2024년", "24년", "2024-04-09", "'24"
_YEAR = re.compile(r"(19|20)\d{2}\s*년|(19|20)\d{2}-\d{1,2}|'\d{2}년")

# 기간 주장임을 알리는 표현. 이런 주장은 시작과 끝이 **둘 다** 있어야 합니다.
_SPAN_WORDS = re.compile(
    r"상승률|하락률|등락률|수익률|증가했|감소했|상승했|하락했|급등|급락"
    r"|성장했|개선되|악화되|늘었|줄었|확대|축소"
)

# 시작과 끝을 함께 가리키는 표현.
_SPAN_MARK = re.compile(
    r"부터|까지|에서\s*.{0,20}(으로|로)|~|사이|동안"
    r"|전년\s*(동기|대비)|전\s*분기\s*대비|직전\s*분기"
)

# 비율 표기. 기간 주장인지 가릴 때 함께 봅니다.
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%?p")

# 지금을 가리키는 상대 표현. 답변 밖에서는 뜻이 정해지지 않습니다.
#
# "이후", "그 후" 는 뺐습니다. "2024년 3분기 실적 발표 이후" 처럼 앞에 시점이
# 붙어 쓰이는 경우가 대부분이라, 넣어 두면 멀쩡한 주장을 무더기로 잡습니다.
_RELATIVE_TIME = re.compile(r"현재|최근|지금|오늘|요즘|당분간")

# 상대 표현 앞에서 "이 날짜는 시작점일 뿐" 임을 알리는 조사.
# "2024년 1월 8일부터 현재까지" 에서 1월 8일은 끝점이 아닙니다.
_START_MARK = re.compile(r"(부터|에서|이래|이후)\s*$")


def has_year(text: str) -> bool:
    """주장에 연도 표기가 있는지."""
    return bool(_YEAR.search(text or ""))


def has_relative_time(text: str) -> bool:
    """'현재', '최근' 처럼 기준일이 있어야 뜻이 정해지는 표현이 있는지.

    **바로 뒤에** 날짜가 붙어 있으면 기준일이 정해진 것이므로 세지 않습니다.
    "현재(2024년 5월 16일) 주가는 ..." 는 문제가 없습니다.

    앞쪽 날짜는 조사를 함께 봅니다.
      - "2024년 5월 말 현재 33만원"      -> 앞의 날짜가 곧 기준일입니다. 셈하지 않습니다.
      - "2024년 1월 8일부터 현재까지"     -> 1월 8일은 **시작점**이고 '현재'는
                                           여전히 정해지지 않았습니다. 셈합니다.
    처음에는 앞뒤를 가리지 않고 날짜만 찾았다가 뒤쪽 경우를 놓쳤고,
    그다음에는 앞쪽을 아예 안 봤다가 앞쪽 경우를 모두 잘못 셌습니다.
    """
    text = text or ""
    for match in _RELATIVE_TIME.finditer(text):
        after = text[match.end(): match.end() + 14]
        if _YEAR.search(after):
            continue  # "현재(2024년 7월 26일)" — 기준일이 붙어 있습니다

        before = text[max(0, match.start() - 14): match.start()]
        if _YEAR.search(before) and not _START_MARK.search(before):
            continue  # "2024년 5월 말 현재" — 앞의 날짜가 곧 기준일입니다

        return True
    return False


def looks_like_span_claim(text: str) -> bool:
    """두 시점을 비교하는 **비율** 주장인지.

    비율이 적혀 있을 때만 봅니다. "판매 비중이 확대되었다" 처럼 수치가 없는
    서술은 의미 비교(NLI)로 가지, 구간을 계산하지 않습니다. 그런 것까지 세면
    숫자가 부풀려져 프롬프트를 고친 효과를 가릴 수 없습니다.
    """
    text = text or ""
    if not _PERCENT.search(text):
        return False
    return bool(_SPAN_WORDS.search(text))


def has_span_endpoints(text: str) -> bool:
    """기간 주장에 시작과 끝이 함께 적혀 있는지."""
    return bool(_SPAN_MARK.search(text or ""))


def missing_time(claims: list[Claim]) -> list[Claim]:
    """시점을 특정할 수 없는 주장들.

    세 가지를 봅니다.
      - 연도가 없다
      - '현재', '최근' 처럼 기준일이 필요한 표현이 있다
      - 기간 주장인데 시작·끝 표현이 없다
    """
    result = []
    for claim in claims:
        text = claim.text
        if not has_year(text):
            result.append(claim)
        elif has_relative_time(text):
            result.append(claim)
        elif looks_like_span_claim(text) and not has_span_endpoints(text):
            result.append(claim)
    return result


def missing_subject(claims: list[Claim], names: tuple[str, ...]) -> list[Claim]:
    """주체(회사)가 적히지 않은 주장들.

    Args:
        names: 찾을 회사 표기들. 유니버스의 이름과 별칭을 넘깁니다.
    """
    return [
        claim
        for claim in claims
        if not any(name and name in claim.text for name in names)
    ]
