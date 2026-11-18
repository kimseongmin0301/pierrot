"""정답을 아는 주장으로 오탐률을 측정합니다.

## 왜 필요한가

지금까지 판정 결과에 SUPPORTED 가 한 건도 없었습니다. 전부 모순이거나 확인 불가입니다.
이것만으로는 **"검증기가 잘 작동한다"와 "전부 거부한다"를 구분할 수 없습니다.**
둘 다 같은 숫자가 나오기 때문입니다.

그래서 스냅샷에서 **참이라고 확신할 수 있는 주장**을 만들어 넣어 봅니다.
근거가 스냅샷 자신이므로 정답은 SUPPORTED 입니다. 여기서 모순이 나오면
그것은 오탐이고, 실무에서 검증기를 못 쓰게 만드는 오류입니다.

함께 거짓 주장도 만들어 넣습니다. 참만 넣으면 "전부 인정"하는 검증기와
구분되지 않습니다.

LLM 파서를 거치지 않고 조회 조건을 직접 만듭니다. 판정 로직만 따로 보기 위해서입니다.
API 비용이 들지 않고 즉시 돌아갑니다.

실행:
    .venv/Scripts/python.exe scripts/probe_false_positives.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pierrot.tools.amounts import format_korean_amount  # noqa: E402
from pierrot.tools.dates import DatePrecision  # noqa: E402
from pierrot.tools.financials import FinancialSnapshot  # noqa: E402
from pierrot.tools.numeric_claim import (  # noqa: E402
    Baseline,
    Direction,
    FinancialAssertion,
    ParsedNumericClaim,
    build_query,
)
from pierrot.tools.numeric_verify import Verdict as NumericVerdict  # noqa: E402
from pierrot.tools.numeric_verify import verify_numeric_claim  # noqa: E402
from pierrot.tools.price import PriceSnapshot  # noqa: E402
from pierrot.tools.price_claim import (  # noqa: E402
    Direction as PriceDirection,
)
from pierrot.tools.price_claim import (
    ParsedPriceClaim,
    PriceAssertion,
)
from pierrot.tools.price_claim import build_query as build_price_query  # noqa: E402
from pierrot.tools.price_verify import Verdict as PriceVerdict  # noqa: E402
from pierrot.tools.price_verify import verify_price_claim  # noqa: E402
from pierrot.tools.universe import UNIVERSE, get_stock  # noqa: E402


@dataclass
class Case:
    kind: str
    label: str
    """정답. 'SUPPORTED' 또는 'CONTRADICTED'."""

    description: str
    verdict: str
    reason: str


def numeric_cases(snapshot: FinancialSnapshot, limit: int) -> list[Case]:
    """재무 스냅샷의 값을 그대로 주장으로 만들어 넣습니다."""
    cases: list[Case] = []
    for stock in UNIVERSE:
        for year in (2023, 2024):
            for account in ("매출액", "영업이익"):
                for quarter in (0, 3):
                    facts = snapshot.lookup(
                        stock.code, year, account, quarter=quarter, fs_div="CFS"
                    )
                    if not facts:
                        continue
                    amount = facts[0].amount
                    if amount <= 0:
                        continue

                    base = dict(
                        is_financial_claim=True,
                        company=stock.name,
                        account=account,
                        bsns_year=year,
                        quarter=quarter,
                        fs_hint="",
                        assertion=FinancialAssertion.AMOUNT,
                        growth_percent=0.0,
                        baseline=Baseline.UNKNOWN,
                        direction=Direction.NONE,
                        note="",
                    )

                    # 참: 스냅샷 값을 그대로 적습니다.
                    truthy = format_korean_amount(amount)
                    query = build_query(ParsedNumericClaim(**base, amount_text=truthy))
                    result = verify_numeric_claim(query, snapshot)
                    cases.append(
                        Case(
                            "NUMERIC", "SUPPORTED",
                            f"{stock.name} {year}년 "
                            f"{'연간' if quarter == 0 else str(quarter) + '분기'} "
                            f"{account} = {truthy}",
                            result.verdict.value, result.reason,
                        )
                    )

                    # 거짓: 값을 2배로 바꿉니다.
                    falsy = format_korean_amount(amount * 2)
                    query = build_query(ParsedNumericClaim(**base, amount_text=falsy))
                    result = verify_numeric_claim(query, snapshot)
                    cases.append(
                        Case(
                            "NUMERIC", "CONTRADICTED",
                            f"{stock.name} {year}년 {account} = {falsy} (실제의 2배)",
                            result.verdict.value, result.reason,
                        )
                    )
                    if len(cases) >= limit * 2:
                        return cases
    return cases


def price_cases(snapshot: PriceSnapshot, limit: int) -> list[Case]:
    """주가 스냅샷의 종가를 그대로 주장으로 만들어 넣습니다."""
    cases: list[Case] = []
    for stock in UNIVERSE:
        rows = snapshot.rows_between(stock.code, "2024-03-01", "2024-03-20")
        for row in rows[:2]:
            use_raw = snapshot.has_raw(stock.code)
            close = row.level("close", prefer_raw=use_raw)
            base = dict(
                is_price_claim=True,
                company=stock.name,
                assertion=PriceAssertion.CLOSE,
                point_date_expr=f"{row.date.year}년 {row.date.month}월 {row.date.day}일",
                start_date_expr="",
                end_date_expr="",
                direction=PriceDirection.NONE,
                note="",
            )

            query = build_price_query(
                ParsedPriceClaim(**base, value_low=close, value_high=close)
            )
            result = verify_price_claim(query, snapshot)
            cases.append(
                Case(
                    "PRICE", "SUPPORTED",
                    f"{stock.name} {row.date} 종가 = {close:,.0f}원",
                    result.verdict.value, result.reason,
                )
            )

            wrong = close * 1.5
            query = build_price_query(
                ParsedPriceClaim(**base, value_low=wrong, value_high=wrong)
            )
            result = verify_price_claim(query, snapshot)
            cases.append(
                Case(
                    "PRICE", "CONTRADICTED",
                    f"{stock.name} {row.date} 종가 = {wrong:,.0f}원 (실제의 1.5배)",
                    result.verdict.value, result.reason,
                )
            )
            if len(cases) >= limit * 2:
                return cases
    return cases


def report(cases: list[Case]) -> dict[str, int]:
    """정답과 판정을 대조해 혼동행렬을 만듭니다."""
    matrix = Counter((case.label, case.verdict) for case in cases)

    true_cases = [c for c in cases if c.label == "SUPPORTED"]
    false_cases = [c for c in cases if c.label == "CONTRADICTED"]

    hit = sum(1 for c in true_cases if c.verdict == "SUPPORTED")
    false_alarm = sum(1 for c in true_cases if c.verdict == "CONTRADICTED")
    missed_true = sum(1 for c in true_cases if c.verdict == "UNVERIFIABLE")

    caught = sum(1 for c in false_cases if c.verdict == "CONTRADICTED")
    missed = sum(1 for c in false_cases if c.verdict == "SUPPORTED")
    skipped = sum(1 for c in false_cases if c.verdict == "UNVERIFIABLE")

    print(f"  참인 주장 {len(true_cases)}건")
    print(f"    근거 있음 (정답)      {hit:4}  {hit / max(len(true_cases), 1):6.1%}")
    print(f"    모순 (오탐)           {false_alarm:4}  "
          f"{false_alarm / max(len(true_cases), 1):6.1%}  <- 실무에서 가장 치명적")
    print(f"    확인 불가             {missed_true:4}")
    print()
    print(f"  거짓인 주장 {len(false_cases)}건")
    print(f"    모순 (정답)           {caught:4}  {caught / max(len(false_cases), 1):6.1%}")
    print(f"    근거 있음 (미탐)      {missed:4}  "
          f"{missed / max(len(false_cases), 1):6.1%}  <- 환각을 놓침")
    print(f"    확인 불가             {skipped:4}")
    return dict(matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description="정답을 아는 주장으로 오탐률 측정")
    parser.add_argument("--limit", type=int, default=40, help="유형별 참 주장 수")
    parser.add_argument("--easy-only", action="store_true", help="쉬운 시험만 돌립니다")
    args = parser.parse_args()

    financials = FinancialSnapshot.load()
    prices = PriceSnapshot.load()

    print("=" * 96)
    print("오탐률 측정 - 스냅샷에서 만든 '정답을 아는 주장'")
    print("=" * 96)
    print("참인 주장은 스냅샷 값을 그대로 옮긴 것이므로 정답이 SUPPORTED 입니다.")
    print("여기서 모순이 나오면 오탐이고, 검증기를 실무에서 못 쓰게 만듭니다.")
    print()

    groups = [
        ("NUMERIC (DART 재무) - 쉬움", numeric_cases(financials, args.limit)),
        ("PRICE (주가) - 쉬움", price_cases(prices, args.limit)),
    ]
    if not args.easy_only:
        groups += [
            ("NUMERIC - 어려움 (어림수 참 / 7% 틀린 거짓)",
             hard_numeric_cases(financials, args.limit)),
            ("PRICE - 어려움 (어림수 참 / 5% 틀린 거짓)",
             hard_price_cases(prices, args.limit)),
        ]

    for title, cases in groups:
        print("-" * 96)
        print(title)
        print("-" * 96)
        report(cases)

        wrong = [c for c in cases if c.label != c.verdict]
        if wrong:
            print()
            print(f"  틀린 판정 {len(wrong)}건 중 앞 4건")
            for case in wrong[:4]:
                print(f"    [정답 {case.label} -> 판정 {case.verdict}] {case.description}")
                print(f"       {case.reason[:92]}")
        print()
    return 0




# ---------------------------------------------------------------------------
# 어려운 조건
#
# 위 시험은 쉽습니다. 참 주장을 포맷터가 만든 표기 그대로 넣었고, 거짓은 2배로
# 크게 틀렸습니다. 실제 LLM 답변은 어림수로 말하고 틀려도 5~10% 수준입니다.
# 경계 근처에서 어떻게 판정하는지가 실무 사용 가능성을 가릅니다.
# ---------------------------------------------------------------------------


def hard_numeric_cases(snapshot: FinancialSnapshot, limit: int) -> list[Case]:
    """어림수로 말한 참 주장과, 조금만 틀린 거짓 주장."""
    from pierrot.tools.amounts import parse_korean_amount

    cases: list[Case] = []
    for stock in UNIVERSE:
        for year in (2023, 2024):
            facts = snapshot.lookup(stock.code, year, "영업이익", quarter=0, fs_div="CFS")
            if not facts or facts[0].amount <= 0:
                continue
            amount = facts[0].amount

            base = dict(
                is_financial_claim=True, company=stock.name, account="영업이익",
                bsns_year=year, quarter=0, fs_hint="",
                assertion=FinancialAssertion.AMOUNT, growth_percent=0.0,
                baseline=Baseline.UNKNOWN, direction=Direction.NONE, note="",
            )

            # 조 단위로 반올림해 어림수로 말합니다. 사람이 쓰는 방식입니다.
            #
            # 정답 라벨은 **실제 차이로 정합니다.** 반올림했다고 전부 참이 아닙니다.
            # 5,753억원을 "1조원"이라고 하면 74% 과장이라 거짓입니다.
            # 처음에는 반올림 결과를 무조건 참으로 라벨링해 오탐률이 부풀려졌습니다.
            cho = round(amount / 1e12)
            if cho >= 1:
                rounded = f"{cho}조원"
                parsed = parse_korean_amount(rounded)
                gap = abs(parsed - amount) / amount
                label = "SUPPORTED" if gap <= 0.10 else "CONTRADICTED"
                query = build_query(ParsedNumericClaim(**base, amount_text=rounded))
                result = verify_numeric_claim(query, snapshot)
                cases.append(Case(
                    "NUMERIC-어림수", label,
                    f"{stock.name} {year}년 영업이익 = {rounded} "
                    f"(실제 {format_korean_amount(amount)}, 차이 {gap:.1%})",
                    result.verdict.value, result.reason,
                ))

            # 거짓: 7%만 틀립니다. 경계 근처입니다.
            wrong = int(amount * 1.07)
            text = format_korean_amount(wrong)
            query = build_query(ParsedNumericClaim(**base, amount_text=text))
            result = verify_numeric_claim(query, snapshot)
            cases.append(Case(
                "NUMERIC-근소", "CONTRADICTED",
                f"{stock.name} {year}년 영업이익 = {text} (실제보다 7% 많음)",
                result.verdict.value, result.reason,
            ))
            if len(cases) >= limit * 2:
                return cases
    return cases


def hard_price_cases(snapshot: PriceSnapshot, limit: int) -> list[Case]:
    """만원 단위로 어림잡은 참 주장과, 5%만 틀린 거짓 주장."""
    cases: list[Case] = []
    for stock in UNIVERSE:
        rows = snapshot.rows_between(stock.code, "2024-06-01", "2024-06-10")
        for row in rows[:1]:
            use_raw = snapshot.has_raw(stock.code)
            close = row.level("close", prefer_raw=use_raw)
            base = dict(
                is_price_claim=True, company=stock.name,
                assertion=PriceAssertion.CLOSE,
                point_date_expr=f"{row.date.year}년 {row.date.month}월 {row.date.day}일",
                start_date_expr="", end_date_expr="",
                direction=PriceDirection.NONE, note="",
            )

            # 참: 만원 단위로 어림잡습니다.
            unit = 10000 if close >= 50000 else 1000
            rounded = round(close / unit) * unit
            gap = abs(rounded - close) / close
            query = build_price_query(
                ParsedPriceClaim(**base, value_low=rounded, value_high=rounded)
            )
            result = verify_price_claim(query, snapshot)
            cases.append(Case(
                "PRICE-어림수", "SUPPORTED",
                f"{stock.name} {row.date} 종가 = {rounded:,.0f}원 "
                f"(실제 {close:,.0f}원, 차이 {gap:.1%})",
                result.verdict.value, result.reason,
            ))

            # 거짓: 5%만 틀립니다.
            wrong = round(close * 1.05)
            query = build_price_query(
                ParsedPriceClaim(**base, value_low=wrong, value_high=wrong)
            )
            result = verify_price_claim(query, snapshot)
            cases.append(Case(
                "PRICE-근소", "CONTRADICTED",
                f"{stock.name} {row.date} 종가 = {wrong:,.0f}원 (실제보다 5% 높음)",
                result.verdict.value, result.reason,
            ))
            if len(cases) >= limit * 2:
                return cases
    return cases

if __name__ == "__main__":
    sys.exit(main())
