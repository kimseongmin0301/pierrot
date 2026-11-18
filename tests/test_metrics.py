"""평가 지표 계산 테스트.

지표가 틀리면 이 프로젝트의 결론 전체가 틀립니다. 손으로 센 값과 대조합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (  # noqa: E402
    CONTRADICTED,
    SUPPORTED,
    UNVERIFIABLE,
    confusion_matrix,
    evaluate,
    format_matrix,
    format_metrics,
)


def test_perfect_verifier():
    """모두 맞히면 재현율 100%, 오탐률 0%."""
    pairs = [(SUPPORTED, SUPPORTED)] * 10 + [(CONTRADICTED, CONTRADICTED)] * 10
    m = evaluate(pairs)
    assert m.recall == 1.0
    assert m.false_positive_rate == 0.0
    assert m.precision == 1.0
    assert m.coverage == 1.0
    assert m.f1 == 1.0


def test_reject_everything_verifier():
    """전부 모순이라고 하는 검증기.

    재현율은 100% 지만 오탐률도 100% 입니다. 이 조합을 구분하지 못하면
    "전부 거부하는 검증기"를 좋은 검증기로 착각하게 됩니다.
    """
    pairs = [(SUPPORTED, CONTRADICTED)] * 10 + [(CONTRADICTED, CONTRADICTED)] * 10
    m = evaluate(pairs)
    assert m.recall == 1.0
    assert m.false_positive_rate == 1.0
    assert m.precision == 0.5


def test_accept_everything_verifier():
    """전부 인정하는 검증기. 오탐은 없지만 아무것도 못 잡습니다."""
    pairs = [(SUPPORTED, SUPPORTED)] * 10 + [(CONTRADICTED, SUPPORTED)] * 10
    m = evaluate(pairs)
    assert m.recall == 0.0
    assert m.false_positive_rate == 0.0
    assert m.missed == 10


def test_abstain_everything_verifier():
    """전부 보류하는 검증기. 틀리지는 않지만 쓸모가 없습니다."""
    pairs = [(SUPPORTED, UNVERIFIABLE)] * 10 + [(CONTRADICTED, UNVERIFIABLE)] * 10
    m = evaluate(pairs)
    assert m.recall == 0.0
    assert m.false_positive_rate == 0.0
    assert m.coverage == 0.0, "커버리지로 '아무것도 판정 안 함'이 드러나야 합니다"


def test_mixed_counts():
    """손으로 센 값과 대조합니다."""
    pairs = (
        [(SUPPORTED, SUPPORTED)] * 7
        + [(SUPPORTED, CONTRADICTED)] * 2
        + [(SUPPORTED, UNVERIFIABLE)] * 1
        + [(CONTRADICTED, CONTRADICTED)] * 8
        + [(CONTRADICTED, SUPPORTED)] * 1
        + [(CONTRADICTED, UNVERIFIABLE)] * 1
    )
    m = evaluate(pairs)
    assert m.true_claims == 10 and m.false_claims == 10
    assert m.caught == 8 and m.missed == 1 and m.skipped_false == 1
    assert m.confirmed == 7 and m.false_alarm == 2 and m.skipped_true == 1
    assert m.recall == pytest.approx(0.8)
    assert m.false_positive_rate == pytest.approx(0.2)
    assert m.precision == pytest.approx(8 / 10)
    assert m.coverage == pytest.approx(18 / 20)


def test_unverifiable_label_is_counted_separately():
    """확인 불가가 정답인 주장은 재현율·오탐률 분모에서 빠집니다."""
    pairs = [
        (SUPPORTED, SUPPORTED),
        (CONTRADICTED, CONTRADICTED),
        (UNVERIFIABLE, UNVERIFIABLE),
        (UNVERIFIABLE, CONTRADICTED),
    ]
    m = evaluate(pairs)
    assert m.true_claims == 1 and m.false_claims == 1
    assert m.correct_abstain == 1
    assert m.wrong_judgement == 1, "근거 없이 단정한 것을 따로 세야 합니다"
    assert m.recall == 1.0
    assert m.false_positive_rate == 0.0


def test_wrong_judgement_lowers_precision():
    """확인할 수 없는 주장을 모순이라고 하면 정밀도가 떨어져야 합니다."""
    clean = evaluate([(CONTRADICTED, CONTRADICTED)] * 5)
    noisy = evaluate(
        [(CONTRADICTED, CONTRADICTED)] * 5 + [(UNVERIFIABLE, CONTRADICTED)] * 5
    )
    assert clean.precision == 1.0
    assert noisy.precision == pytest.approx(0.5)


def test_empty_input():
    m = evaluate([])
    assert m.total == 0
    assert m.recall == 0.0 and m.precision == 0.0


@pytest.mark.parametrize(
    "pairs",
    [[("참", SUPPORTED)], [(SUPPORTED, "맞음")]],
)
def test_unknown_label_raises(pairs):
    """알 수 없는 라벨을 조용히 넘기면 지표가 틀립니다."""
    with pytest.raises(ValueError):
        evaluate(pairs)


def test_confusion_matrix():
    pairs = [(SUPPORTED, CONTRADICTED), (SUPPORTED, SUPPORTED), (CONTRADICTED, CONTRADICTED)]
    matrix = confusion_matrix(pairs)
    assert matrix[SUPPORTED][CONTRADICTED] == 1
    assert matrix[SUPPORTED][SUPPORTED] == 1
    assert matrix[CONTRADICTED][CONTRADICTED] == 1
    assert matrix[UNVERIFIABLE][UNVERIFIABLE] == 0


def test_formatters_produce_readable_output():
    m = evaluate([(SUPPORTED, SUPPORTED), (CONTRADICTED, CONTRADICTED)])
    line = format_metrics("도구 기반", m)
    assert "도구 기반" in line and "재현율" in line and "100.0%" in line

    lines = format_matrix(confusion_matrix([(SUPPORTED, SUPPORTED)]))
    assert len(lines) == 4
    assert "근거있음" in lines[0]


def test_metrics_dict_has_derived_values():
    m = evaluate([(SUPPORTED, SUPPORTED), (CONTRADICTED, CONTRADICTED)])
    data = m.to_dict()
    assert data["recall"] == 1.0
    assert data["false_positive_rate"] == 0.0
    assert data["coverage"] == 1.0
