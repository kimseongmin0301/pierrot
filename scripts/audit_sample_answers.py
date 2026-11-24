"""가상 검증샘플 100건과 정답표를 두 방향으로 감사합니다.

왜 필요한가
-----------
`docs/가상_종목별_100개_검증샘플.csv` 와 `docs/가상_종목별_100개_정답표.csv` 는
**사람이 만든 가상 데이터**입니다. 정답표의 '의도된_판정' 이 정말 정답인지는
아무도 확인하지 않았습니다. 라벨을 의심할 수 있으면 그 테스트셋으로 잰
숫자도 의심해야 합니다.

두 가지를 따로 봅니다. 섞으면 결론이 틀립니다.

1. **내부 정합성** — 문단에 적힌 수치·방향이 그 종목의 '가상 기준값' 과
   맞는가. 정답표가 자기 기준으로도 틀렸다면 그 라벨은 못 씁니다.
   이 감사는 파이썬 계산이라 LLM 없이 끝납니다.

2. **실제 공시와의 거리** — 가상 기준값이 실제 DART 수치와 얼마나 다른가.
   이것이 중요한 이유는, 이 가상 문단을 PIERROT 에 넣으면 **실제 공시와**
   대조하기 때문입니다. 가상 기준값이 실제와 다르면 '수치상 참' 으로
   라벨된 문단도 PIERROT 는 올바르게 '모순' 이라고 판정합니다.
   그 불일치는 PIERROT 의 오류가 아니라 **서로 다른 세계를 재고 있다**는 뜻입니다.

사용:
    python scripts/audit_sample_answers.py
    python scripts/audit_sample_answers.py --show 일부러\\ 틀림
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SAMPLES = PROJECT_ROOT / "docs" / "가상_종목별_100개_검증샘플.csv"
ANSWERS = PROJECT_ROOT / "docs" / "가상_종목별_100개_정답표.csv"

# 정답표 열 이름 -> (연도, 계정 키)
BASELINE_COLUMNS = {
    "가상_기준값_억원_2024_매출": (2024, "매출"),
    "가상_기준값_억원_2025_매출": (2025, "매출"),
    "가상_기준값_억원_2024_영업이익": (2024, "영업이익"),
    "가상_기준값_억원_2025_영업이익": (2025, "영업이익"),
    "가상_기준값_억원_2024_순이익": (2024, "순이익"),
    "가상_기준값_억원_2025_순이익": (2025, "순이익"),
    "가상_기준값_억원_2024_별도영업이익": (2024, "별도영업이익"),
    "가상_기준값_억원_2025_별도영업이익": (2025, "별도영업이익"),
    "가상_기준값_억원_2024_부문매출": (2024, "부문매출"),
    "가상_기준값_억원_2025_부문매출": (2025, "부문매출"),
    "가상_기준값_억원_2025_1분기매출": (2025, "1분기매출"),
    "가상_기준값_억원_2026_1분기매출": (2026, "1분기매출"),
}

# 문단에서 "17,500억 원" 꼴을 뽑습니다. 억 단위로만 적혀 있습니다.
AMOUNT = re.compile(r"([\d,]+)\s*억\s*원")

# 계정 낱말. 긴 것부터 맞춰야 '별도 영업이익' 이 '영업이익' 으로 잘리지 않습니다.
ACCOUNT_WORDS = (
    ("별도 영업이익", "별도영업이익"),
    ("별도영업이익", "별도영업이익"),
    ("부문 매출", "부문매출"),
    ("부문매출", "부문매출"),
    ("영업이익", "영업이익"),
    ("순이익", "순이익"),
    ("매출", "매출"),
)

UP = re.compile(r"증가|증가습니다|늘었|상승|개선|나아졌|성장")
DOWN = re.compile(r"감소|줄었|하락|악화|축소")
SAME = re.compile(r"동일|유지|같았|변동\s*없")


def read_csv(path: Path) -> list[dict]:
    text = path.read_bytes().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def baseline_of(answer: dict) -> dict[tuple[int, str], float]:
    """정답표 한 행에서 (연도, 계정) -> 억원 값을 뽑습니다."""
    out: dict[tuple[int, str], float] = {}
    for column, key in BASELINE_COLUMNS.items():
        raw = (answer.get(column) or "").replace(",", "").strip()
        if raw:
            out[key] = float(raw)
    return out


def amounts_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in AMOUNT.findall(text)]


def direction_of(text: str) -> str:
    """절 하나의 방향. 섞여 있으면 빈 문자열."""
    hits = {"UP": bool(UP.search(text)), "DOWN": bool(DOWN.search(text)),
            "SAME": bool(SAME.search(text))}
    marked = [k for k, v in hits.items() if v]
    return marked[0] if len(marked) == 1 else ""


def account_of(text: str) -> str:
    for word, key in ACCOUNT_WORDS:
        if word in text:
            return key
    return ""


def clauses(paragraph: str) -> list[tuple[str, str]]:
    """(계정, 그 계정에 딸린 글토막) 목록.

    PIERROT 의 `split_segments` 는 쓰지 않습니다. 그쪽은 쉼표로도 쪼개기
    때문에 "32,300억" 이 "32" 와 "300억" 으로 부서집니다(PIERROT 본체에서는
    숫자가 있으면 그 경로를 타지 않아 문제가 없습니다).

    대신 **계정 낱말이 나온 자리를 기준으로** 다음 계정 낱말 직전까지를
    그 계정의 토막으로 봅니다. 문장 부호에 기대지 않아 더 안전합니다.

        "연결 매출은 증가했습니다. 별도 영업이익은 전년 대비 증가했습니다."
        -> [("매출", "매출은 증가했습니다. "), ("별도영업이익", "별도 영업이익은 ...")]
    """
    text = paragraph or ""
    hits: list[tuple[int, int, str]] = []
    position = 0
    while position < len(text):
        best = None
        for word, key in ACCOUNT_WORDS:          # 긴 낱말이 앞에 옵니다
            if text.startswith(word, position):
                best = (position, position + len(word), key)
                break
        if best:
            hits.append(best)
            position = best[1]
        else:
            position += 1

    out = []
    for index, (start, _, key) in enumerate(hits):
        end = hits[index + 1][0] if index + 1 < len(hits) else len(text)
        out.append((key, text[start:end]))
    return out


def trend(base: dict, account: str) -> str:
    prev, curr = base.get((2024, account)), base.get((2025, account))
    if prev is None or curr is None:
        return ""
    return "UP" if curr > prev else ("DOWN" if curr < prev else "SAME")


def audit_special(paragraph: str, base: dict) -> list[str]:
    """계정 토막으로는 못 보는 두 패턴.

    1) "연결과 별도의 증감 추이는 서로 같았다/엇갈렸다"
    2) "2026년 1분기 연결 매출은 전년 동기 대비 증가/감소했다"
    """
    problems: list[str] = []

    if "연결과 별도" in paragraph:
        cfs, ofs = trend(base, "영업이익"), trend(base, "별도영업이익")
        if cfs and ofs:
            same_actual = cfs == ofs
            if re.search(r"서로\s*같", paragraph):
                claimed = True
            elif re.search(r"엇갈|서로\s*다르|반대", paragraph):
                claimed = False
            else:
                claimed = None
            if claimed is not None and claimed != same_actual:
                problems.append(
                    f"연결·별도 추이: 문단 {'같음' if claimed else '엇갈림'} vs "
                    f"기준값 연결 {cfs} / 별도 {ofs}"
                )

    if "1분기" in paragraph and re.search(r"전년\s*동기", paragraph):
        prev, curr = base.get((2025, "1분기매출")), base.get((2026, "1분기매출"))
        if prev is not None and curr is not None:
            actual = "UP" if curr > prev else ("DOWN" if curr < prev else "SAME")
            tail = paragraph[paragraph.find("1분기"):]
            wanted = direction_of(tail)
            if wanted and wanted != actual:
                problems.append(
                    f"1분기 매출 전년 동기 대비: 문단 {wanted} vs 기준값 {actual}"
                    f" ({prev:,.0f} -> {curr:,.0f}억)"
                )
    return problems


def audit_sample(paragraph: str, base: dict) -> list[str]:
    """문단의 수치·방향이 그 종목의 가상 기준값과 맞는지. 어긋난 것만 돌려줍니다."""
    problems: list[str] = audit_special(paragraph, base)
    for account, chunk in clauses(paragraph):
        values = amounts_in(chunk)
        allowed = {v for (_, acct), v in base.items() if acct == account}
        for value in values:
            if allowed and value not in allowed:
                problems.append(
                    f"{account} 금액 {value:,.0f}억은 기준값"
                    f"({', '.join(f'{v:,.0f}' for v in sorted(allowed))})에 없음"
                )

        wanted = direction_of(chunk)
        if not wanted or account == "1분기매출":
            continue
        prev, curr = base.get((2024, account)), base.get((2025, account))
        if prev is None or curr is None:
            continue
        actual = "UP" if curr > prev else ("DOWN" if curr < prev else "SAME")
        if actual != wanted:
            problems.append(
                f"{account} 방향: 문단 {wanted} vs 기준값 {actual}"
                f" ({prev:,.0f} -> {curr:,.0f}억)"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="가상 샘플·정답표 감사")
    parser.add_argument("--show", help="이 판정 범주의 상세를 전부 출력")
    args = parser.parse_args()

    samples = {r["sample_id"]: r for r in read_csv(SAMPLES)}
    answers = {r["sample_id"]: r for r in read_csv(ANSWERS)}

    print("=" * 100)
    print("감사 1 — 정답표가 자기 '가상 기준값' 에 비춰 맞는가")
    print("=" * 100)

    # 범주별로 기대하는 바가 다릅니다.
    #   수치상 참      -> 불일치가 하나도 없어야 합니다
    #   일부러 틀림     -> 불일치가 적어도 하나 있어야 합니다
    #   그 밖(보류 계열) -> 수치 부분만 확인하고 판단은 보류합니다
    rows = []
    for sid in sorted(samples):
        sample, answer = samples[sid], answers[sid]
        base = baseline_of(answer)
        problems = audit_sample(sample["가상_문단"], base)
        rows.append((sid, sample, answer, problems))

    tally = collections.Counter()
    bad: list[tuple] = []
    for sid, sample, answer, problems in rows:
        verdict = answer["의도된_판정"]
        if verdict == "수치상 참":
            ok = not problems
        elif verdict == "일부러 틀림":
            ok = bool(problems)
        else:
            ok = None  # 보류 계열은 수치만으로 가릴 수 없습니다
        tally[(verdict, ok)] += 1
        if ok is False:
            bad.append((sid, sample, answer, problems))

    print(f"\n{'판정':18}{'건수':>6}{'라벨 성립':>10}{'라벨 불성립':>12}{'판단 보류':>10}")
    print("-" * 60)
    for verdict in ["수치상 참", "일부러 틀림", "보류", "일부 확인·나머지 보류", "보류/조건부"]:
        total = sum(v for (vd, _), v in tally.items() if vd == verdict)
        if not total:
            continue
        print(f"{verdict:18}{total:>6}{tally[(verdict, True)]:>10}"
              f"{tally[(verdict, False)]:>12}{tally[(verdict, None)]:>10}")

    if bad:
        print(f"\n라벨이 자기 기준값과 어긋나는 건 {len(bad)}개:")
        for sid, sample, answer, problems in bad:
            print(f"\n  [{sid}] {sample['종목명']} · 의도된_판정={answer['의도된_판정']}")
            print(f"      문단: {sample['가상_문단'][:88]}")
            print(f"      근거: {answer['판정_근거']}")
            for p in problems:
                print(f"      -> {p}")
    else:
        print("\n라벨이 자기 기준값과 어긋나는 건: 없음")

    # ---------------- 감사 2 ----------------
    print()
    print("=" * 100)
    print("감사 2 — 가상 기준값이 실제 DART 공시와 얼마나 다른가")
    print("=" * 100)
    print("PIERROT 는 실제 공시와 대조합니다. 가상 기준값이 실제와 다르면")
    print("'수치상 참' 라벨이 붙은 문단도 실제로는 거짓이고,")
    print("PIERROT 가 '모순' 이라고 하는 것이 맞습니다.")
    print()

    import statistics

    from pierrot.tools.financials import FinancialSnapshot
    from pierrot.tools.universe import find_stock

    real_account = {"매출": "매출액", "영업이익": "영업이익",
                    "순이익": "당기순이익", "별도영업이익": "영업이익"}
    snapshot = FinancialSnapshot.load()

    pairs: list[tuple] = []
    seen: set[str] = set()
    for answer in answers.values():
        if answer["종목명"] in seen:
            continue
        seen.add(answer["종목명"])
        stock = find_stock(answer["종목명"])
        if stock is None:
            print(f"  유니버스에 없는 종목: {answer['종목명']}")
            continue
        base = baseline_of(answer)
        for (year, account), fake in base.items():
            name = real_account.get(account)
            if name is None or year not in (2024, 2025) or not fake:
                continue
            fs_div = "OFS" if account == "별도영업이익" else "CFS"
            found = snapshot.lookup(stock.code, year, name, quarter=0, fs_div=fs_div)
            if found:
                pairs.append((answer["종목명"], year, account,
                              fake, found[0].amount / 1e8))

    if pairs:
        ratios = [real / fake for _, _, _, fake, real in pairs]
        close = sum(1 for r in ratios if 0.5 <= r <= 2.0)
        print(f"  비교 가능한 (종목, 연도, 계정) 조합: {len(pairs)}개")
        print(f"  실제/가상 배율  중앙값 {statistics.median(ratios):.1f}배  "
              f"최소 {min(ratios):.1f}배  최대 {max(ratios):.1f}배")
        print(f"  실제의 0.5~2배 안에 드는 것: {close}/{len(pairs)} "
              f"({close / len(pairs) * 100:.0f}%)")
        print()
        print("  가장 크게 벌어진 5개:")
        for name, year, account, fake, real in sorted(
                pairs, key=lambda r: -(r[4] / r[3]))[:5]:
            print(f"    {name:12} {year} {account:8} "
                  f"가상 {fake:9,.0f}억  실제 {real:12,.0f}억  {real / fake:7.1f}배")
        print()
        print("  결론: 이 테스트셋으로 PIERROT 의 수치 판정 정확도를 잴 수 없습니다.")
        print("        가상 세계와 실제 세계를 서로 다른 잣대로 재게 됩니다.")
        print("        수치와 무관한 구조적 함정만 비교할 수 있습니다")
        print("        (scripts/compare_sample_verdicts.py).")

    if args.show:
        print(f"\n{'=' * 100}\n[{args.show}] 상세\n{'=' * 100}")
        for sid, sample, answer, problems in rows:
            if answer["의도된_판정"] != args.show:
                continue
            print(f"\n[{sid}] {sample['종목명']} · 함정={sample['함정_유형']}")
            print(f"  문단: {sample['가상_문단']}")
            print(f"  근거: {answer['판정_근거']}")
            print(f"  수치 감사: {problems or '불일치 없음'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
