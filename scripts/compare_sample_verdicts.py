"""PIERROT 판정을 가상 정답표와 견줍니다. **비교 가능한 것만** 셉니다.

가장 중요한 전제
----------------
가상 문단의 수치는 **실제 공시와 무관**합니다(중앙값 5.6배, 최대 484배).
PIERROT 는 실제 DART 공시와 대조하므로, '수치상 참' 으로 라벨된 문단도
PIERROT 는 올바르게 '모순' 이라고 판정합니다. 삼성전자 매출을 1.75조라고
적어 놓으면 그건 **실제로 거짓**이기 때문입니다.

그래서 수치 라벨과 PIERROT 판정을 그대로 맞춰 정확도를 내면
**아무 의미 없는 숫자**가 나옵니다. 서로 다른 세계를 재고 있습니다.

그럼에도 비교할 수 있는 것
--------------------------
수치와 무관한 **구조적 함정**은 실제 데이터로도 그대로 시험됩니다.
숫자가 맞든 틀리든 검증기는 같은 행동을 해야 하기 때문입니다.

    회사명 없음   -> 회사를 지어내지 말고 확인 불가로 남겨야 함
    미래 전망·단정 -> 지금 참·거짓을 가릴 수 없으므로 확인 불가
    모호한 기간   -> '최근', '연말', '예전' 은 기간을 정할 수 없음
    정성 표현     -> '좋은 실적' 은 수치로 잴 대상이 아님
    인과관계      -> 방향이 맞아도 원인까지 확인한 것은 아님

정답표가 이들을 '보류' 로 둔 것과 PIERROT 가 '확인 불가' 로 두는 것은
**같은 기준**입니다. 여기서 어긋나면 그건 진짜 결함입니다.

사용:
    python scripts/compare_sample_verdicts.py result100.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_sample_answers import ANSWERS, SAMPLES, read_csv  # noqa: E402

# 수치와 무관하게 '보류해야 한다' 가 성립하는 함정들.
STRUCTURAL_TRAPS = (
    "회사명 없음", "미래 전망", "미래 단정", "모호한 기간",
    "정성 표현", "출처 미상", "근거 없는 비교",
)


def paragraph_verdict(claims: list[dict]) -> str:
    """문단 하나의 종합 판정. 하나라도 모순이면 모순입니다."""
    kinds = {c["verdict"] for c in claims}
    if "CONTRADICTED" in kinds:
        return "모순"
    if "SUPPORTED" in kinds:
        return "근거 있음"
    return "확인 불가"


def main() -> int:
    parser = argparse.ArgumentParser(description="PIERROT 판정과 가상 정답표 비교")
    parser.add_argument("result", type=Path, help="verify_batch.py --out 으로 만든 JSON")
    args = parser.parse_args()

    samples = {r["sample_id"]: r for r in read_csv(SAMPLES)}
    answers = {r["sample_id"]: r for r in read_csv(ANSWERS)}
    results = {r["label"]: r for r in json.loads(args.result.read_text(encoding="utf-8"))}

    missing = sorted(set(samples) - set(results))
    if missing:
        print(f"결과에 없는 샘플 {len(missing)}개: {missing[:10]}")

    print("=" * 100)
    print("검증 2 — 실제 공시 데이터로 100건 판정")
    print("=" * 100)

    total_claims = sum(len(r["claims"]) for r in results.values())
    print(f"\n문단 {len(results)}건 -> 주장 {total_claims}개")

    # ---- 1. 의도된 판정별 PIERROT 결과 ----------------------------------
    print("\n[참고] 의도된 판정별 PIERROT 종합 판정")
    print("  수치 라벨은 가상 기준값 기준이고 PIERROT 는 실제 공시 기준이라")
    print("  이 표의 일치/불일치로 정확도를 내면 안 됩니다.")
    grid = collections.Counter()
    for sid, answer in answers.items():
        if sid not in results:
            continue
        grid[(answer["의도된_판정"], paragraph_verdict(results[sid]["claims"]))] += 1
    verdicts = ["모순", "근거 있음", "확인 불가"]
    print(f"\n  {'의도된 판정':22}" + "".join(f"{v:>12}" for v in verdicts))
    print("  " + "-" * 60)
    for intended in ["수치상 참", "일부러 틀림", "보류", "일부 확인·나머지 보류", "보류/조건부"]:
        row = [grid[(intended, v)] for v in verdicts]
        if not sum(row):
            continue
        print(f"  {intended:22}" + "".join(f"{n:>12}" for n in row))

    # ---- 2. 비교 가능한 것: 구조적 함정 절은 보류해야 합니다 -------------
    #
    # **절 단위로 봅니다.** 문단 전체를 하나로 뭉뚱그리면 안 됩니다.
    # "2025년 매출은 X였다. 2027년에는 더 성장할 전망이다" 는 앞 절이
    # 실제 공시와 달라 모순이 나는 것이 맞고, 뒤 절만 보류되면 됩니다.
    # 문단 단위로 세면 앞 절의 정당한 모순을 결함으로 잘못 셉니다.
    print()
    print("=" * 100)
    print("비교 가능한 부분 — 구조적 함정이 걸린 '절' 은 보류가 정답입니다")
    print("=" * 100)

    # 함정별로 그 함정에 해당하는 주장을 골라내는 표지
    MARKERS = {
        "미래 전망": ("2027", "전망"),
        "미래 단정": ("2027", "것이다", "것입니다", "달성할", "달성한다"),
        "모호한 기간": ("최근", "연말", "예전"),
        # '알려졌' 은 넣지 않습니다. 정답표가 "좋은 실적은 평가 표현,
        # 순이익 증감은 수치로 검증 가능" 이라고 직접 적어 두었습니다.
        # 출처가 미상이어도 수치는 공시로 확인할 수 있습니다.
        "정성 표현": ("좋은 실적", "양호한 성과", "부진한 흐름"),
    }

    for trap, markers in MARKERS.items():
        ids = [sid for sid, s_ in samples.items()
               if trap in s_["함정_유형"].split(";") and sid in results]
        if not ids:
            continue
        declined, judged, vanished = [], [], []
        for sid in ids:
            res = results[sid]
            hits = [c for c in res["claims"]
                    if any(mk in c["text"] for mk in markers)]
            if not hits:
                # 주장으로 뽑히지도 않았다면 uncovered 에 사유와 함께 남았는지 봅니다.
                covered = any(any(mk in u["text"] for mk in markers)
                              for u in res.get("uncovered", []))
                (declined if covered else vanished).append(sid)
                continue
            if all(c["verdict"] == "UNVERIFIABLE" for c in hits):
                declined.append(sid)
            else:
                judged.append((sid, hits))
        bad = len(judged) + len(vanished)
        mark = "OK " if bad == 0 else "주의"
        print()
        print(f"  [{mark}] {trap:10} {len(ids)}건 | 보류 {len(declined)} / "
              f"판정해 버림 {len(judged)} / 흔적 없이 사라짐 {len(vanished)}")
        for sid, hits in judged:
            print(f"      - {sid} {samples[sid]['종목명']}")
            print(f"        문단: {samples[sid]['가상_문단'][:78]}")
            for c in hits:
                if c["verdict"] != "UNVERIFIABLE":
                    print(f"        [{c['verdict']}] ({c['confidence']}) {c['text'][:60]}")
                    print(f"          근거: {c['reason'][:82]}")
        for sid in vanished:
            print(f"      - {sid} {samples[sid]['종목명']}: 해당 절이 주장으로도 "
                  f"uncovered 로도 남지 않음")
            print(f"        문단: {samples[sid]['가상_문단'][:78]}")

    # ---- 3. 회사명 없음: 회사를 지어내지 않았는가 ------------------------
    print("\n" + "=" * 100)
    print("가장 중요한 확인 — 회사명이 없는 문단에 회사를 지어내지 않았는가")
    print("=" * 100)
    for sid, s in sorted(samples.items()):
        if "회사명 없음" not in s["함정_유형"].split(";") or sid not in results:
            continue
        print(f"\n  [{sid}] 원래 종목: {s['종목명']} (문단에는 회사명 없음)")
        print(f"      문단: {s['가상_문단'][:84]}")
        for c in results[sid]["claims"]:
            print(f"      [{c['verdict']}] {c['text'][:70]}")
            print(f"        근거: {c['reason'][:88]}")

    # ---- 4. 판정 성격 ---------------------------------------------------
    conf = collections.Counter(c["confidence"]
                               for r in results.values() for c in r["claims"])
    print("\n" + "=" * 100)
    print("판정의 성격")
    print("=" * 100)
    for k, v in conf.most_common():
        print(f"  {k:12} {v:4}개")
    cost = sum(r.get("cost_usd", 0) for r in results.values())
    print(f"\n  이번 실행 비용 ${cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
