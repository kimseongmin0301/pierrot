"""근거 문장 생성.

NLI 는 두 문장을 비교하는 모델이라 "비교할 근거 문장"이 있어야 합니다.
그런데 `generator` 는 환각을 유도하려고 RAG 없이 답변을 만들었으므로
참고 문서가 존재하지 않습니다.

그래서 이미 수집한 스냅샷(재무 수치, 주가)에서 사실 문장을 기계적으로 만들어
근거로 씁니다. 이렇게 하면 두 가지가 좋습니다.

1. 근거가 **검증된 사실**입니다. DART 와 거래소에서 받은 값이므로 틀릴 수 없습니다.
2. 같은 사실을 도구(DART/주가 조회)와 NLI 가 각각 판정하므로,
   **공정한 비교**가 됩니다. NLI 쪽에만 불리한 근거를 준 것이 아닙니다.

이것이 중요한 이유: 이 프로젝트의 결론은 "도구 기반 검증이 NLI 보다 낫다"입니다.
NLI 에게 근거를 부실하게 주면 그 결론은 조작이 됩니다. 같은 사실을 문장으로
제대로 전달하고도 NLI 가 숫자를 못 본다면, 그것이 진짜 한계입니다.

## 한계

여기서 만드는 문장은 수치에서 나온 것이라 "반도체 부문 실적이 개선되면서
영업이익 성장을 견인했다" 같은 인과 서술은 담지 못합니다.
그런 주장을 다루려면 공시 본문이 필요하고, 그것은 다음 단계에서 붙입니다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .amounts import format_korean_amount
from .financials import FinancialSnapshot
from .price import PriceSnapshot
from .universe import Stock, get_stock


@dataclass(frozen=True)
class Evidence:
    """근거 문장 하나."""

    text: str
    source: str
    """어디서 나온 문장인지. 'DART' 또는 '주가'."""

    code: str
    year: int
    kind: str
    """문장의 종류. 'amount', 'growth', 'price_level', 'price_change' 등."""

    def __str__(self) -> str:
        return self.text


def _period_name(quarter: int) -> str:
    return "연간" if quarter == 0 else f"{quarter}분기"


def financial_evidence(
    snapshot: FinancialSnapshot,
    code: str,
    year: int,
    *,
    accounts: tuple[str, ...] = ("매출액", "영업이익", "당기순이익"),
    include_growth: bool = True,
) -> list[Evidence]:
    """재무 스냅샷에서 그 회사·그 해의 사실 문장을 만듭니다."""
    stock = get_stock(code)
    name = stock.name if stock else code
    result: list[Evidence] = []

    for quarter in (0, 1, 2, 3):
        for account in accounts:
            facts = snapshot.lookup(code, year, account, quarter=quarter)
            for fact in facts:
                period = _period_name(quarter)
                result.append(
                    Evidence(
                        text=(
                            f"{name}의 {year}년 {period} {account}은 "
                            f"{fact.fs_label} 기준 {format_korean_amount(fact.amount)}이다."
                        ),
                        source="DART",
                        code=code,
                        year=year,
                        kind="amount",
                    )
                )

                if not include_growth:
                    continue
                previous = snapshot.lookup(
                    code, year - 1, account, quarter=quarter, fs_div=fact.fs_div
                )
                if not previous or previous[0].amount == 0:
                    continue
                rate = (fact.amount - previous[0].amount) / abs(previous[0].amount) * 100
                word = "증가" if rate >= 0 else "감소"
                result.append(
                    Evidence(
                        text=(
                            f"{name}의 {year}년 {period} {account}은 "
                            f"{fact.fs_label} 기준 전년 동기 "
                            f"{format_korean_amount(previous[0].amount)}에서 "
                            f"{format_korean_amount(fact.amount)}으로 "
                            f"{abs(rate):.1f}% {word}했다."
                        ),
                        source="DART",
                        code=code,
                        year=year,
                        kind="growth",
                    )
                )
    return result


def price_evidence(
    snapshot: PriceSnapshot, code: str, year: int
) -> list[Evidence]:
    """주가 스냅샷에서 그 회사·그 해의 사실 문장을 만듭니다.

    연간 흐름과 분기별 등락을 담습니다. 일별 시세를 전부 문장으로 만들면
    근거가 수백 개가 되어 NLI 가 찾지 못하므로, 사람이 요약하듯 줄입니다.
    """
    stock = get_stock(code)
    name = stock.name if stock else code
    start, end = f"{year}-01-01", f"{year}-12-31"

    rows = snapshot.rows_between(code, start, end)
    if not rows:
        return []

    use_raw = snapshot.has_raw(code)
    result: list[Evidence] = []

    first, last = rows[0], rows[-1]
    first_close = first.level("close", prefer_raw=use_raw)
    last_close = last.level("close", prefer_raw=use_raw)
    rate = (last_close / first_close - 1) * 100 if first_close else 0.0
    word = "상승" if rate >= 0 else "하락"

    result.append(
        Evidence(
            text=(
                f"{name}의 주가는 {year}년 첫 거래일({first.date}) 종가 "
                f"{first_close:,.0f}원에서 마지막 거래일({last.date}) 종가 "
                f"{last_close:,.0f}원으로 {abs(rate):.1f}% {word}했다."
            ),
            source="주가",
            code=code,
            year=year,
            kind="price_change",
        )
    )

    high = max(rows, key=lambda r: r.level("high", prefer_raw=use_raw))
    low = min(rows, key=lambda r: r.level("low", prefer_raw=use_raw))
    result.append(
        Evidence(
            text=(
                f"{name}의 {year}년 주가 최고가는 {high.date}의 "
                f"{high.level('high', prefer_raw=use_raw):,.0f}원이고, "
                f"최저가는 {low.date}의 "
                f"{low.level('low', prefer_raw=use_raw):,.0f}원이다."
            ),
            source="주가",
            code=code,
            year=year,
            kind="price_level",
        )
    )

    # 분기별 등락
    for quarter, (begin, finish) in enumerate(
        [("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")],
        start=1,
    ):
        part = snapshot.rows_between(code, f"{year}-{begin}", f"{year}-{finish}")
        if len(part) < 2:
            continue
        open_close = part[0].level("close", prefer_raw=use_raw)
        close_close = part[-1].level("close", prefer_raw=use_raw)
        if not open_close:
            continue
        quarter_rate = (close_close / open_close - 1) * 100
        quarter_word = "상승" if quarter_rate >= 0 else "하락"
        result.append(
            Evidence(
                text=(
                    f"{name}의 주가는 {year}년 {quarter}분기에 "
                    f"{open_close:,.0f}원에서 {close_close:,.0f}원으로 "
                    f"{abs(quarter_rate):.1f}% {quarter_word}했다."
                ),
                source="주가",
                code=code,
                year=year,
                kind="price_change",
            )
        )
    return result


def build_evidence(
    code: str,
    years: tuple[int, ...],
    *,
    financials: FinancialSnapshot | None = None,
    prices: PriceSnapshot | None = None,
) -> list[Evidence]:
    """한 회사에 대한 근거 문장을 모읍니다."""
    result: list[Evidence] = []
    for year in years:
        if financials is not None:
            result.extend(financial_evidence(financials, code, year))
        if prices is not None:
            result.extend(price_evidence(prices, code, year))
    return result


def candidate_stocks(text: str) -> list[Stock]:
    """주장 텍스트에서 근거를 모을 종목을 찾습니다."""
    from .universe import find_all_stocks

    return find_all_stocks(text)
