"""정답 라벨이 붙은 테스트셋 생성.

## 왜 스냅샷에서 만드는가

평가를 하려면 "이 주장이 참인지 거짓인지"를 확실히 알아야 합니다.
사람이 손으로 라벨을 붙이면 300개를 만들기도 어렵고, 붙인 라벨이 맞는지도
다시 검증해야 합니다.

대신 **이미 검증된 스냅샷에서 주장을 만들어 냅니다.**
DART 와 거래소에서 받은 값이므로 근거가 확실하고, 값을 일부러 틀리게 바꾸면
그 주장은 반드시 거짓입니다. 라벨이 생성 규칙에서 나오므로 의심할 여지가 없습니다.

## 주의: 이 방식의 한계

여기서 만든 주장은 문장 틀이 정해져 있어 실제 LLM 환각과 분포가 다릅니다.
그래서 1단계에서 gemma3:4b 가 실제로 만들어 낸 주장 91개를 **보조 세트**로
함께 둡니다. 다만 그쪽은 정답 라벨이 없으므로 지표 계산에는 쓰지 않고,
"실제 환각에서도 같은 경향이 나오는가"를 보는 용도입니다.

## 라벨

    SUPPORTED     근거와 일치. 스냅샷 값을 그대로 또는 어림수로 옮긴 주장
    CONTRADICTED  근거와 모순. 값, 방향, 기간을 일부러 틀리게 바꾼 주장
    UNVERIFIABLE  근거로 확인할 수 없음. 스냅샷에 없는 기간이나 계정
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

ERROR_KINDS = (
    "none",
    "amount_small",
    "amount_large",
    "direction_flip",
    "period_shift",
    "out_of_range",
    "unknown_account",
)


@dataclass(frozen=True)
class TestClaim:
    """정답이 붙은 주장 하나."""

    claim_id: str
    text: str
    label: str
    """SUPPORTED / CONTRADICTED / UNVERIFIABLE"""

    claim_type: str
    """NUMERIC / PRICE"""

    error_kind: str
    """어떤 오류를 주입했는지. 'none' 이면 참인 주장입니다."""

    code: str
    source_fact: str
    """이 주장이 어떤 사실에서 나왔는지. 판정을 검토할 때 씁니다."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round_to_unit(value: float, unit: int) -> int:
    return int(round(value / unit) * unit)


def perturb_amount(amount: int, ratio: float) -> int:
    """금액을 비율만큼 바꿉니다. 자릿수가 유지되도록 반올림합니다."""
    changed = amount * (1 + ratio)
    if abs(changed) >= 10**12:
        return _round_to_unit(changed, 10**10)
    if abs(changed) >= 10**8:
        return _round_to_unit(changed, 10**7)
    return int(round(changed))


class TestSetBuilder:
    """스냅샷에서 라벨된 주장을 만듭니다.

    난수를 쓰지만 seed 를 고정합니다. 평가를 다시 돌렸을 때 같은 테스트셋이
    나와야 지표를 비교할 수 있기 때문입니다.
    """

    def __init__(self, seed: int = 20261118) -> None:
        self.random = random.Random(seed)
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:04d}"

    # -- NUMERIC ----------------------------------------------------------

    def numeric_claims(
        self,
        financials,
        *,
        years: tuple[int, ...] = (2023, 2024, 2025),
        accounts: tuple[str, ...] = ("매출액", "영업이익", "당기순이익"),
        per_fact: int = 1,
    ) -> list[TestClaim]:
        """재무 수치에서 주장을 만듭니다."""
        from pierrot.tools.amounts import format_korean_amount
        from pierrot.tools.universe import UNIVERSE

        claims: list[TestClaim] = []
        for stock in UNIVERSE:
            for year in years:
                for account in accounts:
                    for quarter in (0, 1, 3):
                        facts = financials.lookup(
                            stock.code, year, account, quarter=quarter, fs_div="CFS"
                        )
                        if not facts or facts[0].amount == 0:
                            continue
                        amount = facts[0].amount
                        period = "연간" if quarter == 0 else f"{quarter}분기"
                        source = (
                            f"{stock.name} {year}년 {period} {account} "
                            f"연결 {format_korean_amount(amount)}"
                        )

                        for _ in range(per_fact):
                            claims.extend(
                                self._numeric_variants(
                                    stock, year, period, account, amount, source
                                )
                            )
        return claims

    def _numeric_variants(
        self, stock, year: int, period: str, account: str, amount: int, source: str
    ) -> list[TestClaim]:
        from pierrot.tools.amounts import format_korean_amount

        def claim(text: str, label: str, error: str) -> TestClaim:
            return TestClaim(
                claim_id=self._next_id("n"),
                text=text,
                label=label,
                claim_type="NUMERIC",
                error_kind=error,
                code=stock.code,
                source_fact=source,
            )

        head = f"{stock.name}의 {year}년 {period} {account}은"
        exact = format_korean_amount(amount)

        result = [claim(f"{head} {exact}이다.", "SUPPORTED", "none")]

        # 거짓: 30% 틀리게 (사람이 보기에도 명백한 오류)
        big = format_korean_amount(perturb_amount(amount, self.random.choice([0.3, -0.3])))
        result.append(claim(f"{head} {big}이다.", "CONTRADICTED", "amount_large"))

        # 거짓: 허용 오차 바로 바깥 (경계 시험)
        small = format_korean_amount(perturb_amount(amount, self.random.choice([0.15, -0.15])))
        if small != exact:
            result.append(claim(f"{head} {small}이다.", "CONTRADICTED", "amount_small"))

        # 확인 불가: 스냅샷에 없는 연도
        result.append(
            claim(
                f"{stock.name}의 2019년 {period} {account}은 {exact}이다.",
                "UNVERIFIABLE",
                "out_of_range",
            )
        )
        return result

    # -- PRICE ------------------------------------------------------------

    def price_claims(
        self,
        prices,
        *,
        samples_per_stock: int = 3,
        start: str = "2024-01-01",
        end: str = "2025-12-31",
    ) -> list[TestClaim]:
        """주가에서 주장을 만듭니다."""
        from pierrot.tools.universe import UNIVERSE

        claims: list[TestClaim] = []
        available = set(prices.codes())
        for stock in UNIVERSE:
            # 스냅샷에 없는 종목은 건너뜁니다. 수집 범위가 바뀌어도 생성기가
            # 멈추지 않아야 합니다.
            if stock.code not in available:
                continue
            rows = prices.rows_between(stock.code, start, end)
            if len(rows) < 30:
                continue
            use_raw = prices.has_raw(stock.code)
            picks = self.random.sample(rows, min(samples_per_stock, len(rows)))

            for row in picks:
                close = row.level("close", prefer_raw=use_raw)
                source = f"{stock.name} {row.date} 종가 {close:,.0f}원"
                claims.extend(self._price_variants(stock, row, close, source))
        return claims

    def _price_variants(self, stock, row, close: float, source: str) -> list[TestClaim]:
        def claim(text: str, label: str, error: str) -> TestClaim:
            return TestClaim(
                claim_id=self._next_id("p"),
                text=text,
                label=label,
                claim_type="PRICE",
                error_kind=error,
                code=stock.code,
                source_fact=source,
            )

        date_text = f"{row.date.year}년 {row.date.month}월 {row.date.day}일"
        head = f"{stock.name} 주가는 {date_text}"

        result = [
            claim(f"{head} 종가 기준 {close:,.0f}원이다.", "SUPPORTED", "none")
        ]

        wrong = _round_to_unit(close * self.random.choice([1.25, 0.75]), 100)
        result.append(
            claim(f"{head} 종가 기준 {wrong:,.0f}원이다.", "CONTRADICTED", "amount_large")
        )

        near = _round_to_unit(close * self.random.choice([1.12, 0.88]), 100)
        if near != int(close):
            result.append(
                claim(f"{head} 종가 기준 {near:,.0f}원이다.", "CONTRADICTED", "amount_small")
            )

        result.append(
            claim(
                f"{stock.name} 주가는 2019년 5월 2일 종가 기준 {close:,.0f}원이다.",
                "UNVERIFIABLE",
                "out_of_range",
            )
        )
        return result

    # -- 방향 주장 --------------------------------------------------------

    def direction_claims(
        self, financials, *, years: tuple[int, ...] = (2024, 2025)
    ) -> list[TestClaim]:
        """증감 방향 주장을 만듭니다. 방향을 뒤집으면 반드시 거짓입니다."""
        from pierrot.tools.amounts import format_korean_amount
        from pierrot.tools.universe import UNIVERSE

        claims: list[TestClaim] = []
        for stock in UNIVERSE:
            for year in years:
                for account in ("매출액", "영업이익"):
                    current = financials.lookup(
                        stock.code, year, account, quarter=0, fs_div="CFS"
                    )
                    previous = financials.lookup(
                        stock.code, year - 1, account, quarter=0, fs_div="CFS"
                    )
                    if not current or not previous or previous[0].amount == 0:
                        continue

                    delta = current[0].amount - previous[0].amount
                    if delta == 0:
                        continue
                    rose = delta > 0
                    rate = abs(delta) / abs(previous[0].amount) * 100
                    source = (
                        f"{stock.name} {year}년 {account} "
                        f"{format_korean_amount(previous[0].amount)} -> "
                        f"{format_korean_amount(current[0].amount)} "
                        f"({'+' if rose else '-'}{rate:.1f}%)"
                    )

                    head = f"{stock.name}의 {year}년 연간 {account}은 전년 대비"
                    claims.append(
                        TestClaim(
                            claim_id=self._next_id("d"),
                            text=f"{head} {rate:.1f}% {'증가' if rose else '감소'}했다.",
                            label="SUPPORTED",
                            claim_type="NUMERIC",
                            error_kind="none",
                            code=stock.code,
                            source_fact=source,
                        )
                    )
                    claims.append(
                        TestClaim(
                            claim_id=self._next_id("d"),
                            text=f"{head} {rate:.1f}% {'감소' if rose else '증가'}했다.",
                            label="CONTRADICTED",
                            claim_type="NUMERIC",
                            error_kind="direction_flip",
                            code=stock.code,
                            source_fact=source,
                        )
                    )
        return claims


    # -- 어려운 변형 ------------------------------------------------------
    #
    # 위 생성기는 스냅샷 값을 포맷터 표기 그대로 옮깁니다. 그래서 도구가
    # 자기 데이터를 자기가 조회하는 꼴이 되어 쉽습니다.
    #
    # 실제 LLM 답변은 어림수로 말하고, 단위 표기가 다르고, 날짜가 모호합니다.
    # 그 조건에서도 동작하는지 따로 재야 "실무에서 쓸 수 있다" 고 말할 수 있습니다.

    def hard_numeric_claims(
        self,
        financials,
        *,
        years: tuple[int, ...] = (2023, 2024, 2025),
        accounts: tuple[str, ...] = ("매출액", "영업이익"),
    ) -> list[TestClaim]:
        """어림수와 다른 단위 표기로 말한 주장."""
        from pierrot.tools.amounts import format_korean_amount
        from pierrot.tools.universe import UNIVERSE

        claims: list[TestClaim] = []
        for stock in UNIVERSE:
            for year in years:
                for account in accounts:
                    facts = financials.lookup(
                        stock.code, year, account, quarter=0, fs_div="CFS"
                    )
                    if not facts or facts[0].amount <= 0:
                        continue
                    amount = facts[0].amount
                    source = (
                        f"{stock.name} {year}년 연간 {account} "
                        f"연결 {format_korean_amount(amount)}"
                    )
                    head = f"{stock.name}의 {year}년 {account}은"

                    def add(text: str, label: str, error: str) -> None:
                        claims.append(TestClaim(
                            claim_id=self._next_id("hn"), text=text, label=label,
                            claim_type="NUMERIC", error_kind=error,
                            code=stock.code, source_fact=source,
                        ))

                    # 라벨은 연결과 별도를 **둘 다** 보고 정합니다.
                    #
                    # 처음에는 연결만 보고 라벨을 붙였는데, 도구는 둘 다 보므로
                    # 별도 기준으로 맞는 주장을 "놓쳤다"고 잘못 세었습니다.
                    # 예: 셀트리온 2024년 매출 "약 4조원"
                    #     연결 3조 5,573억(12.4% 차이)  별도 3조 7,092억(7.8% 차이)
                    # 기준을 밝히지 않은 주장이므로 별도와 맞으면 참입니다.
                    separate_facts = financials.lookup(
                        stock.code, year, account, quarter=0, fs_div="OFS"
                    )
                    bases = [amount]
                    if separate_facts and separate_facts[0].amount > 0:
                        bases.append(separate_facts[0].amount)

                    def closest_gap(claimed: float) -> float:
                        return min(abs(claimed - base) / base for base in bases)

                    # 조 단위로 어림잡아 말합니다. 라벨은 실제 차이로 정합니다.
                    cho = amount / 1e12
                    if cho >= 1:
                        rounded = round(cho)
                        gap = closest_gap(rounded * 1e12)
                        add(
                            f"{head} 약 {rounded}조원이다.",
                            "SUPPORTED" if gap <= 0.10 else "CONTRADICTED",
                            "rounded" if gap <= 0.10 else "rounded_overstated",
                        )

                        # 소수 한 자리로 말합니다. 표기가 달라도 같은 값입니다.
                        decimal = round(cho, 1)
                        gap = closest_gap(decimal * 1e12)
                        add(
                            f"{head} {decimal}조원 수준이다.",
                            "SUPPORTED" if gap <= 0.10 else "CONTRADICTED",
                            "decimal_unit" if gap <= 0.10 else "decimal_overstated",
                        )

                    # 억 단위로만 말합니다. 조를 쓰지 않는 표기입니다.
                    eok = round(amount / 1e8)
                    add(f"{head} {eok:,}억원이다.", "SUPPORTED", "eok_unit")

                    # 기준을 밝히지 않고 별도 수치를 말합니다.
                    if separate_facts and separate_facts[0].amount > 0:
                        add(
                            f"{head} {format_korean_amount(separate_facts[0].amount)}이다.",
                            "SUPPORTED",
                            "separate_basis",
                        )
        return claims

    def hard_price_claims(
        self,
        prices,
        *,
        samples_per_stock: int = 2,
        start: str = "2024-01-01",
        end: str = "2025-12-31",
    ) -> list[TestClaim]:
        """모호한 날짜와 어림수 가격으로 말한 주장."""
        from pierrot.tools.universe import UNIVERSE

        tenday = {1: "초", 2: "중순", 3: "말"}
        claims: list[TestClaim] = []
        available = set(prices.codes())

        for stock in UNIVERSE:
            if stock.code not in available:
                continue
            rows = prices.rows_between(stock.code, start, end)
            if len(rows) < 30:
                continue
            use_raw = prices.has_raw(stock.code)

            for row in self.random.sample(rows, min(samples_per_stock, len(rows))):
                close = row.level("close", prefer_raw=use_raw)
                part = tenday[min((row.date.day - 1) // 10 + 1, 3)]
                window = prices.rows_between(
                    stock.code,
                    f"{row.date.year}-{row.date.month:02d}-01",
                    f"{row.date.year}-{row.date.month:02d}-28",
                )
                lows = [r.level("low", prefer_raw=use_raw) for r in window]
                highs = [r.level("high", prefer_raw=use_raw) for r in window]
                source = (
                    f"{stock.name} {row.date.year}년 {row.date.month}월 "
                    f"저가 {min(lows):,.0f} ~ 고가 {max(highs):,.0f}원 "
                    f"({row.date} 종가 {close:,.0f}원)"
                )

                def add(text: str, label: str, error: str) -> None:
                    claims.append(TestClaim(
                        claim_id=self._next_id("hp"), text=text, label=label,
                        claim_type="PRICE", error_kind=error,
                        code=stock.code, source_fact=source,
                    ))

                # 모호한 날짜 + 만원 단위 어림수. 실제 LLM 이 말하는 방식입니다.
                unit = 10000 if close >= 50000 else 1000
                rounded = round(close / unit) * unit
                head = (
                    f"{stock.name} 주가는 {row.date.year}년 "
                    f"{row.date.month}월 {part}"
                )
                add(f"{head} {rounded:,.0f}원 수준이었다.", "SUPPORTED", "vague_date")

                # 같은 모호한 날짜에 그 달 범위를 크게 벗어난 값.
                far = round(max(highs) * 1.4 / unit) * unit
                add(f"{head} {far:,.0f}원 수준이었다.", "CONTRADICTED", "vague_date_wrong")
        return claims


def summarize(claims: list[TestClaim]) -> dict[str, Any]:
    from collections import Counter

    return {
        "total": len(claims),
        "by_label": dict(Counter(c.label for c in claims)),
        "by_type": dict(Counter(c.claim_type for c in claims)),
        "by_error": dict(Counter(c.error_kind for c in claims)),
    }
