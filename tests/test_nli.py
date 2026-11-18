"""NLI 래퍼와 근거 문장 생성 테스트.

모델을 띄우는 테스트는 느리고 다운로드가 필요하므로 slow 로 표시했습니다.
기본 실행에서는 모델 없이 되는 것만 돕니다.

    pytest                  # 모델 없이
    pytest -m slow          # 모델을 띄우는 것까지
"""

from __future__ import annotations

import pytest

from pierrot.tools.evidence import financial_evidence, price_evidence
from pierrot.tools.financials import FinancialSnapshot
from pierrot.tools.narrative_verify import (
    Verdict,
    extract_years,
    verify_narrative_claim,
)
from pierrot.tools.nli import KLUE_FALLBACK_ORDER, NliLabel, NliResult, resolve_label_order
from pierrot.tools.price import PriceSnapshot


# ---------------------------------------------------------------------------
# 라벨 순서
#
# 순서를 틀리면 함의를 모순으로 읽어 판정이 통째로 뒤집히는데,
# 겉으로는 멀쩡해 보입니다. 그래서 따로 고정해 둡니다.
# ---------------------------------------------------------------------------


def test_label_order_from_config():
    """실제 모델(Huffon/klue-roberta-base-nli)의 설정값입니다."""
    order, source = resolve_label_order(
        {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
    )
    assert order == ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    assert source == "config"


def test_label_order_handles_different_arrangement():
    """다른 순서로 학습된 모델도 이름을 보고 맞춥니다."""
    order, source = resolve_label_order(
        {0: "contradiction", 1: "entailment", 2: "neutral"}
    )
    assert order == ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
    assert source == "config"


def test_label_order_falls_back_for_generic_names():
    """LABEL_0 처럼 이름이 없으면 KLUE 순서를 가정하고 그 사실을 남깁니다."""
    order, source = resolve_label_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    assert order == list(KLUE_FALLBACK_ORDER)
    assert "가정" in source


def test_label_order_rejects_non_nli_model():
    """3분류가 아니면 NLI 모델이 아닙니다. 조용히 넘어가면 안 됩니다."""
    with pytest.raises(ValueError):
        resolve_label_order({0: "POSITIVE", 1: "NEGATIVE"})


def test_nli_result_scores():
    result = NliResult(
        label=NliLabel.ENTAILMENT,
        scores={"ENTAILMENT": 0.9, "NEUTRAL": 0.08, "CONTRADICTION": 0.02},
        premise="전제",
        hypothesis="가설",
    )
    assert result.confidence == pytest.approx(0.9)
    assert result.score(NliLabel.CONTRADICTION) == pytest.approx(0.02)
    assert "entailment 0.900" in result.describe()


# ---------------------------------------------------------------------------
# 연도 추출
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("삼성전자의 2024년 3분기 영업이익", (2024,)),
        ("2023년에서 2024년으로", (2023, 2024)),
        ("영업이익이 증가했다", (2025,)),
    ],
)
def test_extract_years(text, expected):
    assert extract_years(text, fallback=(2025,)) == expected


# ---------------------------------------------------------------------------
# 근거 문장 생성
# ---------------------------------------------------------------------------


def financial_record(**kwargs) -> dict:
    base = {
        "code": "005930", "corp_code": "00126380", "bsns_year": 2024,
        "reprt_code": "11014", "period_label": "3분기", "quarter": 3,
        "period_months": 3, "fs_div": "CFS", "fs_nm": "연결재무제표",
        "account": "영업이익", "amount": 9_183_371_000_000,
        "period_start": "2024.01.01", "period_end": "2024.09.30",
        "currency": "KRW", "rcept_no": "20241114",
    }
    base.update(kwargs)
    return base


def test_financial_evidence_states_the_amount():
    snapshot = FinancialSnapshot.from_records([financial_record()])
    evidence = financial_evidence(snapshot, "005930", 2024, accounts=("영업이익",))
    assert len(evidence) == 1
    text = evidence[0].text
    assert "삼성전자" in text and "2024년 3분기" in text
    assert "9조 1,833억원" in text
    assert "연결" in text


def test_financial_evidence_includes_growth_sentence():
    """전년 수치가 있으면 증감 문장도 만듭니다."""
    snapshot = FinancialSnapshot.from_records(
        [financial_record(), financial_record(bsns_year=2023, amount=2_433_500_000_000)]
    )
    evidence = financial_evidence(snapshot, "005930", 2024, accounts=("영업이익",))
    kinds = {item.kind for item in evidence}
    assert kinds == {"amount", "growth"}
    growth = next(item for item in evidence if item.kind == "growth")
    assert "증가" in growth.text


def test_financial_evidence_without_previous_year_has_no_growth():
    snapshot = FinancialSnapshot.from_records([financial_record()])
    evidence = financial_evidence(snapshot, "005930", 2024, accounts=("영업이익",))
    assert all(item.kind == "amount" for item in evidence)


PRICE_RECORDS = [
    {"code": "005930", "date": "2024-01-02", "open": 79000, "high": 80000,
     "low": 78000, "close": 79600, "volume": 1, "change": 0.0},
    {"code": "005930", "date": "2024-03-29", "open": 82000, "high": 83000,
     "low": 81000, "close": 82400, "volume": 1, "change": 0.01},
    {"code": "005930", "date": "2024-12-30", "open": 53000, "high": 54000,
     "low": 52000, "close": 53200, "volume": 1, "change": -0.01},
]


def test_price_evidence_states_year_change():
    snapshot = PriceSnapshot.from_records(PRICE_RECORDS)
    evidence = price_evidence(snapshot, "005930", 2024)
    texts = " ".join(item.text for item in evidence)
    assert "79,600원" in texts and "53,200원" in texts
    assert "하락" in texts


def test_price_evidence_includes_high_and_low():
    snapshot = PriceSnapshot.from_records(PRICE_RECORDS)
    evidence = price_evidence(snapshot, "005930", 2024)
    levels = [item for item in evidence if item.kind == "price_level"]
    assert levels and "83,000원" in levels[0].text and "52,000원" in levels[0].text


def test_price_evidence_empty_for_missing_year():
    snapshot = PriceSnapshot.from_records(PRICE_RECORDS)
    assert price_evidence(snapshot, "005930", 2019) == []


# ---------------------------------------------------------------------------
# 판정 (모델 없이 되는 경우)
# ---------------------------------------------------------------------------


class FakeNliModel:
    """정해진 점수를 돌려주는 가짜 모델."""

    def __init__(self, entail: float, contra: float) -> None:
        self.entail = entail
        self.contra = contra
        self.calls = 0

    def predict_batch(self, pairs, **kwargs):
        self.calls += len(pairs)
        neutral = max(0.0, 1.0 - self.entail - self.contra)
        scores = {
            "ENTAILMENT": self.entail,
            "CONTRADICTION": self.contra,
            "NEUTRAL": neutral,
        }
        best = max(scores, key=scores.get)
        return [
            NliResult(NliLabel(best), scores, premise, hypothesis)
            for premise, hypothesis in pairs
        ]


def make_snapshots():
    financials = FinancialSnapshot.from_records([financial_record()])
    prices = PriceSnapshot.from_records(PRICE_RECORDS)
    return financials, prices


def test_unknown_company_is_unverifiable():
    """유니버스 밖 종목은 근거를 모을 수 없습니다."""
    financials, prices = make_snapshots()
    result = verify_narrative_claim(
        "애플의 실적이 개선되었다",
        model=FakeNliModel(0.9, 0.0),
        financials=financials,
        prices=prices,
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "종목을 찾지 못해" in result.reason


def test_high_entailment_is_supported():
    financials, prices = make_snapshots()
    result = verify_narrative_claim(
        "삼성전자의 2024년 영업이익이 늘었다",
        model=FakeNliModel(0.95, 0.0),
        financials=financials,
        prices=prices,
    )
    assert result.verdict is Verdict.SUPPORTED
    assert result.best_entailment == pytest.approx(0.95)


def test_high_contradiction_is_contradicted():
    financials, prices = make_snapshots()
    result = verify_narrative_claim(
        "삼성전자의 2024년 영업이익이 줄었다",
        model=FakeNliModel(0.0, 0.95),
        financials=financials,
        prices=prices,
    )
    assert result.verdict is Verdict.CONTRADICTED


def test_strongest_relation_wins():
    """둘 다 기준을 넘으면 확률이 높은 쪽을 택합니다.

    처음에는 "함의 우선"으로 만들었는데 실제로 돌려 보니 틀렸습니다.
    "SK하이닉스 2024년 매출 31.6% 증가"(실제 +102%)가 함의 0.712,
    모순 0.968 인데도 함의 우선 규칙 때문에 근거 있음으로 판정됐습니다.
    """
    financials, prices = make_snapshots()
    contradicted = verify_narrative_claim(
        "삼성전자의 2024년 영업이익",
        model=FakeNliModel(0.9, 0.95),
        financials=financials,
        prices=prices,
    )
    assert contradicted.verdict is Verdict.CONTRADICTED

    # 표기 차이로 생긴 모순(0.808)은 같은 값을 적은 근거의 함의(0.99)에 집니다.
    supported = verify_narrative_claim(
        "삼성전자의 2024년 영업이익",
        model=FakeNliModel(0.99, 0.808),
        financials=financials,
        prices=prices,
    )
    assert supported.verdict is Verdict.SUPPORTED


def test_below_threshold_is_unverifiable():
    financials, prices = make_snapshots()
    result = verify_narrative_claim(
        "삼성전자의 2024년 영업이익",
        model=FakeNliModel(0.4, 0.3),
        financials=financials,
        prices=prices,
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "기준" in result.reason


def test_threshold_is_recorded():
    financials, prices = make_snapshots()
    result = verify_narrative_claim(
        "삼성전자의 2024년 영업이익",
        model=FakeNliModel(0.95, 0.0),
        financials=financials,
        prices=prices,
        entailment_threshold=0.8,
    )
    assert result.threshold == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 실제 모델 (느림)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_label_order_is_correct():
    """라벨 순서가 실제로 맞는지 확인합니다.

    이것이 틀리면 모든 판정이 뒤집히므로 분명한 예시로 고정해 둡니다.
    """
    from pierrot.tools.nli import NliModel

    model = NliModel().load()
    premise = "삼성전자의 2024년 연간 영업이익은 32조 7,259억원이다."

    entail = model.predict(premise, "삼성전자의 2024년 영업이익은 32조원대이다.")
    contra = model.predict(premise, "삼성전자는 2024년에 영업손실을 기록했다.")
    neutral = model.predict(premise, "카카오는 2024년에 새 게임을 출시했다.")

    assert entail.label is NliLabel.ENTAILMENT
    assert contra.label is NliLabel.CONTRADICTION
    assert neutral.label is NliLabel.NEUTRAL


@pytest.mark.slow
def test_real_model_cannot_normalize_notation():
    """실측으로 확인한 NLI 의 약점을 고정합니다.

    같은 값인데 표기가 다르면 모순으로 판정합니다. 이것이 오탐의 주원인이고,
    도구 기반 검증이 필요한 이유입니다. 모델이 바뀌어 이 성질이 사라지면
    베이스라인 비교의 근거도 다시 써야 하므로 테스트로 감시합니다.
    """
    from pierrot.tools.nli import NliModel

    model = NliModel().load()
    result = model.predict(
        "삼성전자의 2024년 3분기 영업이익은 연결 기준 9조 1,833억원이다.",
        "삼성전자의 2024년 3분기 영업이익은 9.18조원이다.",
    )
    assert result.label is not NliLabel.ENTAILMENT
