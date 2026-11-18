"""검증 대상 종목 목록.

목록을 코드에 고정하는 이유: "코스피 시가총액 상위 20종목"처럼 동적으로 정하면
돌리는 시점마다 구성이 바뀌어 평가 결과를 재현할 수 없습니다. 유니버스는 실험
조건의 일부이므로 고정하고, 바꿀 때는 커밋으로 흔적을 남깁니다.

구성: 1단계 샘플 답변에 등장한 10개 기업 + 코스피 대형주 10개.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Stock:
    """종목 하나."""

    code: str
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


# 별칭은 LLM이 답변에 쓸 법한 표기를 모은 것입니다.
# 주장 텍스트에서 종목을 찾아내려면 "현대자동차"도 현대차로 이어져야 합니다.
UNIVERSE: tuple[Stock, ...] = (
    Stock("005930", "삼성전자", ("삼성전자우",)),
    Stock("000660", "SK하이닉스", ("에스케이하이닉스", "하이닉스")),
    Stock("373220", "LG에너지솔루션", ("엘지에너지솔루션", "LG엔솔", "LG 에너지솔루션")),
    Stock("207940", "삼성바이오로직스", ("삼바", "삼성바이오")),
    Stock("005380", "현대차", ("현대자동차", "현대자동차그룹")),
    Stock("000270", "기아", ("기아자동차", "기아차")),
    Stock("068270", "셀트리온", ()),
    Stock("005490", "POSCO홀딩스", ("포스코홀딩스", "포스코", "POSCO")),
    Stock("035420", "NAVER", ("네이버", "naver")),
    Stock("035720", "카카오", ("Kakao", "kakao")),
    Stock("105560", "KB금융", ("KB금융지주", "국민은행", "케이비금융")),
    Stock("055550", "신한지주", ("신한금융", "신한금융지주")),
    Stock("086790", "하나금융지주", ("하나금융", "하나은행")),
    Stock("316140", "우리금융지주", ("우리금융", "우리은행")),
    Stock("012330", "현대모비스", ("모비스",)),
    Stock("051910", "LG화학", ("엘지화학",)),
    Stock("006400", "삼성SDI", ("삼성에스디아이",)),
    Stock("028260", "삼성물산", ()),
    Stock("015760", "한국전력", ("한전", "한국전력공사")),
    Stock("017670", "SK텔레콤", ("에스케이텔레콤", "SKT")),
)

BY_CODE: dict[str, Stock] = {stock.code: stock for stock in UNIVERSE}

# 긴 이름을 먼저 보도록 정렬합니다.
# "삼성전자"와 "삼성바이오로직스"가 모두 후보일 때 "삼성"만 보고 고르면 안 됩니다.
_NAME_INDEX: tuple[tuple[str, Stock], ...] = tuple(
    sorted(
        ((name, stock) for stock in UNIVERSE for name in stock.all_names),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)

_CODE_PATTERN = re.compile(r"\b(\d{6})\b")


def get_stock(code: str) -> Stock | None:
    """종목코드로 찾습니다."""
    return BY_CODE.get(code)


def find_stock(text: str) -> Stock | None:
    """주장 텍스트에서 종목을 찾습니다.

    종목코드가 직접 적혀 있으면 그것을 먼저 쓰고, 없으면 이름과 별칭을 찾습니다.
    찾지 못하면 None 입니다. 그 주장은 조회 조건을 만들 수 없으므로
    검증 단계에서 UNVERIFIABLE 로 가야 합니다.
    """
    if not text:
        return None

    match = _CODE_PATTERN.search(text)
    if match and match.group(1) in BY_CODE:
        return BY_CODE[match.group(1)]

    for name, stock in _NAME_INDEX:
        if name in text:
            return stock
    return None


def find_all_stocks(text: str) -> list[Stock]:
    """텍스트에 등장하는 종목을 모두 찾습니다. 중복은 제거합니다."""
    found: list[Stock] = []
    for name, stock in _NAME_INDEX:
        if name in text and stock not in found:
            found.append(stock)
    return found
