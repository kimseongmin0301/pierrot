"""라벨된 테스트셋으로 세 방법의 성능을 측정합니다.

    도구 기반        DART 조회 + 주가 계산 (이 프로젝트의 주장)
    NLI 베이스라인    한국어 NLI + 생성된 근거 문장
    LLM-as-judge     gpt-4o-mini 에게 근거와 주장을 주고 직접 판정시키기

같은 테스트셋, 같은 근거를 줍니다. 한쪽에만 불리한 조건을 주면
"도구가 낫다"는 결론이 조작이 되기 때문입니다.

실행:
    .venv/Scripts/python.exe evaluation/run_eval.py
    .venv/Scripts/python.exe evaluation/run_eval.py --limit 30
    .venv/Scripts/python.exe evaluation/run_eval.py --methods tool nli
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

from evaluation.metrics import (  # noqa: E402
    confusion_matrix,
    evaluate,
    format_matrix,
    format_metrics,
)

TESTSET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "testset.jsonl"
RESULT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "eval_results.jsonl"
METRICS_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "eval_metrics.json"

METHODS = ("tool", "nli", "judge")


def load_testset(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"테스트셋이 없습니다: {path}\n"
            "먼저 evaluation/build_testset.py 를 실행해 주십시오."
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# 방법 1: 도구 기반
# ---------------------------------------------------------------------------


class ToolMethod:
    """주장 유형에 맞는 도구로 조회하고 계산해 판정합니다."""

    name = "도구 기반"

    def __init__(self, settings) -> None:
        from pierrot.tools.financials import FinancialSnapshot
        from pierrot.tools.numeric_claim import build_numeric_parser
        from pierrot.tools.price import PriceSnapshot
        from pierrot.tools.price_claim import build_price_parser

        self.settings = settings
        self.financials = FinancialSnapshot.load()
        self.prices = PriceSnapshot.load()
        self.numeric_parser = build_numeric_parser(settings=settings)
        self.price_parser = build_price_parser(settings=settings)

    def judge(self, claim: dict) -> tuple[str, str]:
        from pierrot.tools.numeric_claim import parse_numeric_claim
        from pierrot.tools.numeric_verify import verify_numeric_claim
        from pierrot.tools.price_claim import parse_price_claim
        from pierrot.tools.price_verify import verify_price_claim

        if claim["claim_type"] == "PRICE":
            query = parse_price_claim(
                claim["text"], parser=self.price_parser, settings=self.settings
            )
            result = verify_price_claim(query, self.prices)
        else:
            query = parse_numeric_claim(
                claim["text"], parser=self.numeric_parser, settings=self.settings
            )
            result = verify_numeric_claim(query, self.financials)
        return result.verdict.value, result.reason


# ---------------------------------------------------------------------------
# 방법 2: NLI 베이스라인
# ---------------------------------------------------------------------------


class NliMethod:
    """생성된 근거 문장과 주장을 NLI 모델로 대조합니다."""

    name = "NLI 베이스라인"

    def __init__(self, settings) -> None:
        from pierrot.tools.dart_documents import DocumentStore
        from pierrot.tools.financials import FinancialSnapshot
        from pierrot.tools.nli import NliModel
        from pierrot.tools.price import PriceSnapshot

        self.financials = FinancialSnapshot.load()
        self.prices = PriceSnapshot.load()
        try:
            self.documents = DocumentStore.load()
        except FileNotFoundError:
            self.documents = None
        self.model = NliModel(settings=settings).load()

    def judge(self, claim: dict) -> tuple[str, str]:
        from pierrot.tools.narrative_verify import verify_narrative_claim

        result = verify_narrative_claim(
            claim["text"],
            model=self.model,
            financials=self.financials,
            prices=self.prices,
            documents=self.documents,
        )
        return result.verdict.value, result.reason


# ---------------------------------------------------------------------------
# 방법 3: LLM-as-judge
# ---------------------------------------------------------------------------


JUDGE_SYSTEM = """당신은 금융 주장의 사실 여부를 판정하는 도구입니다.

근거 문장들과 주장 하나를 받아 다음 중 하나로 판정합니다.

- SUPPORTED: 근거가 주장을 뒷받침합니다.
- CONTRADICTED: 근거와 주장이 충돌합니다.
- UNVERIFIABLE: 근거만으로는 판단할 수 없습니다.

## 규칙

1. 근거에 없는 내용은 추측하지 마십시오. 모르면 UNVERIFIABLE 입니다.
2. 수치는 값으로 비교하십시오. 표기가 달라도 같은 값이면 같은 값입니다.
   "9조 1,800억원" 과 "9.18조원" 은 같습니다.
3. 어림수는 인정하십시오. 6조 5,669억원을 "약 7조원"이라고 하는 것은 맞습니다.
   다만 5,753억원을 "1조원"이라고 하면 과장이므로 CONTRADICTED 입니다.
4. 연결 기준과 별도 기준은 수치가 다릅니다. 주장이 기준을 밝히지 않았다면
   둘 중 하나라도 맞으면 SUPPORTED 입니다.
"""

JUDGE_USER = """근거:
{evidence}

주장: {claim}

위 근거만으로 주장을 판정하십시오."""


class JudgeMethod:
    """LLM 에게 근거와 주장을 주고 직접 판정시킵니다.

    "요즘은 LLM 에 직접 물어보면 되지 않나" 라는 가장 현실적인 반론에 답하는
    베이스라인입니다. NLI 와 같은 근거를 줍니다.
    """

    name = "LLM-as-judge"

    def __init__(self, settings) -> None:
        from enum import Enum

        from pydantic import BaseModel, Field

        from pierrot.llm import get_openai_chat
        from pierrot.tools.financials import FinancialSnapshot
        from pierrot.tools.price import PriceSnapshot

        class Verdict(str, Enum):
            SUPPORTED = "SUPPORTED"
            CONTRADICTED = "CONTRADICTED"
            UNVERIFIABLE = "UNVERIFIABLE"

        class JudgeResult(BaseModel):
            verdict: Verdict = Field(description="판정 결과")
            reason: str = Field(description="판정 근거를 한 문장으로")

        self.settings = settings
        self.financials = FinancialSnapshot.load()
        self.prices = PriceSnapshot.load()
        chat = get_openai_chat(
            "judge", temperature=0.0, max_tokens=512,
            source="llm_judge", settings=settings,
        )
        self.chain = chat.with_structured_output(
            JudgeResult, method="json_schema", strict=True
        )

    def judge(self, claim: dict) -> tuple[str, str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from pierrot.tools.narrative_verify import collect_evidence
        from pierrot.tools.retrieval import search

        pool, _, _ = collect_evidence(
            claim["text"], financials=self.financials, prices=self.prices
        )
        ranked = search(
            claim["text"], [item.text for item in pool], top_k=15, min_score=0.0
        )
        evidence = "\n".join(f"- {item.sentence}" for item in ranked)
        if not evidence:
            return "UNVERIFIABLE", "근거 문장을 모으지 못했습니다"

        result = self.chain.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM),
                HumanMessage(
                    content=JUDGE_USER.format(evidence=evidence, claim=claim["text"])
                ),
            ]
        )
        if isinstance(result, dict):
            return result["verdict"], result.get("reason", "")
        return result.verdict.value, result.reason


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="평가 실행")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=list(METHODS), choices=METHODS)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    from pierrot.config import get_settings
    from pierrot.llm import get_cost_logger

    settings = get_settings()
    logger = get_cost_logger(settings)
    spent_before = logger.month_total_usd()

    claims = load_testset(TESTSET_PATH, args.limit)

    print("=" * 96)
    print("평가 실행")
    print("=" * 96)
    print(f"테스트셋: {len(claims)}개 ({TESTSET_PATH.name})")
    print(f"방법    : {', '.join(args.methods)}")
    print()

    builders = {"tool": ToolMethod, "nli": NliMethod, "judge": JudgeMethod}
    results: list[dict] = []

    for key in args.methods:
        print(f"[{key}] 준비 중...")
        method = builders[key](settings)
        started = time.perf_counter()

        for index, claim in enumerate(claims, start=1):
            try:
                verdict, reason = method.judge(claim)
            except Exception as exc:  # 한 건 실패가 전체를 멈추면 안 됩니다
                verdict, reason = "UNVERIFIABLE", f"오류: {type(exc).__name__}: {exc}"
            results.append(
                {
                    "method": key,
                    "method_name": method.name,
                    "claim_id": claim["claim_id"],
                    "claim_type": claim["claim_type"],
                    "error_kind": claim["error_kind"],
                    "label": claim["label"],
                    "verdict": verdict,
                    "reason": reason[:300],
                    "claim_text": claim["text"],
                }
            )
            if index % 50 == 0:
                print(f"  {index}/{len(claims)}")

        elapsed = time.perf_counter() - started
        print(f"[{key}] 완료 {elapsed:.0f}초")
        print()

    print_report(results, args.methods, builders)

    spent = logger.month_total_usd() - spent_before
    print()
    print(f"이번 실행 비용: ${spent:.6f} "
          f"(이번 달 누적 ${logger.month_total_usd():.6f} / 상한 ${logger.monthly_budget_usd:.2f})")

    if not args.no_save:
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"판정 저장: {RESULT_PATH}")

        payload = {
            key: evaluate(
                [(r["label"], r["verdict"]) for r in results if r["method"] == key]
            ).to_dict()
            for key in args.methods
        }
        METRICS_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"지표 저장: {METRICS_PATH}")
    return 0


def print_report(results: list[dict], methods: list[str], builders: dict) -> None:
    print("=" * 96)
    print("결과")
    print("=" * 96)
    print()

    for key in methods:
        pairs = [(r["label"], r["verdict"]) for r in results if r["method"] == key]
        if not pairs:
            continue
        name = next(r["method_name"] for r in results if r["method"] == key)
        print(format_metrics(name, evaluate(pairs)))

    print()
    for key in methods:
        subset = [r for r in results if r["method"] == key]
        if not subset:
            continue
        name = subset[0]["method_name"]
        print(f"[{name}] 혼동행렬")
        for line in format_matrix(
            confusion_matrix([(r["label"], r["verdict"]) for r in subset])
        ):
            print(line)
        print()

    print("주장 유형별 재현율 (거짓 주장을 잡아낸 비율)")
    print(f"  {'방법':18}" + "".join(f"{t:>12}" for t in ("NUMERIC", "PRICE")))
    for key in methods:
        subset = [r for r in results if r["method"] == key]
        if not subset:
            continue
        cells = []
        for claim_type in ("NUMERIC", "PRICE"):
            pairs = [
                (r["label"], r["verdict"])
                for r in subset
                if r["claim_type"] == claim_type
            ]
            cells.append(f"{evaluate(pairs).recall:>11.1%}" if pairs else f"{'-':>12}")
        print(f"  {subset[0]['method_name']:18}" + "".join(cells))

    print()
    print("오류 유형별 재현율")
    kinds = sorted({r["error_kind"] for r in results if r["label"] == "CONTRADICTED"})
    print(f"  {'방법':18}" + "".join(f"{k:>16}" for k in kinds))
    for key in methods:
        subset = [r for r in results if r["method"] == key]
        if not subset:
            continue
        cells = []
        for kind in kinds:
            pairs = [
                (r["label"], r["verdict"])
                for r in subset
                if r["error_kind"] == kind and r["label"] == "CONTRADICTED"
            ]
            cells.append(f"{evaluate(pairs).recall:>15.1%}" if pairs else f"{'-':>16}")
        print(f"  {subset[0]['method_name']:18}" + "".join(cells))


if __name__ == "__main__":
    sys.exit(main())
