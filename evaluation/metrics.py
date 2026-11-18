"""평가 지표 계산.

**이 파일은 외부 라이브러리에 의존하지 않습니다.** 표준 라이브러리만 씁니다.

이유: 이 프로젝트의 핵심 주장이 "검증기 성능을 숫자로 증명했다"입니다.
그 숫자를 만드는 코드가 LangChain 이나 pandas 버전에 따라 달라지면,
몇 달 뒤 같은 결과를 다시 뽑지 못합니다. 지표 계산만큼은 외부 의존성을
가장 적게 둡니다.

## 무엇을 재는가

환각 검증기에서 중요한 것은 두 가지입니다.

- **재현율(Recall)**: 거짓 주장을 얼마나 잡아내는가. 놓치면 환각이 통과합니다.
- **오탐률(FPR)**: 참인 주장을 틀렸다고 하는 비율. 높으면 아무도 안 씁니다.

정밀도(Precision)는 "모순이라고 한 것 중 실제로 거짓인 비율"입니다.
오탐이 늘면 정밀도가 떨어집니다.

UNVERIFIABLE 은 따로 셉니다. 틀린 판정이 아니라 "판단을 보류했다"이므로
맞고 틀림으로 나누면 지표가 왜곡됩니다. 대신 **커버리지**로 공개합니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNVERIFIABLE = "UNVERIFIABLE"

LABELS = (SUPPORTED, CONTRADICTED, UNVERIFIABLE)


@dataclass(frozen=True)
class Metrics:
    """판정 성능 지표 하나."""

    total: int

    true_claims: int
    false_claims: int
    unverifiable_claims: int

    caught: int
    """거짓 주장을 모순으로 잡은 수 (TP)."""

    missed: int
    """거짓 주장을 근거 있음으로 본 수 (FN, 환각 통과)."""

    skipped_false: int
    """거짓 주장을 판단 보류한 수."""

    false_alarm: int
    """참인 주장을 모순으로 본 수 (FP, 오탐)."""

    confirmed: int
    """참인 주장을 근거 있음으로 본 수 (TN)."""

    skipped_true: int
    """참인 주장을 판단 보류한 수."""

    correct_abstain: int
    """확인 불가 주장을 확인 불가로 본 수."""

    wrong_judgement: int
    """확인 불가 주장에 판정을 내린 수. 근거 없이 단정한 것입니다."""

    @property
    def recall(self) -> float:
        """거짓 주장을 잡아낸 비율. 보류는 못 잡은 것으로 셉니다."""
        return self.caught / self.false_claims if self.false_claims else 0.0

    @property
    def false_positive_rate(self) -> float:
        """참인 주장을 틀렸다고 한 비율."""
        return self.false_alarm / self.true_claims if self.true_claims else 0.0

    @property
    def precision(self) -> float:
        """모순이라고 한 것 중 실제로 거짓인 비율."""
        flagged = self.caught + self.false_alarm + self.wrong_judgement
        return self.caught / flagged if flagged else 0.0

    @property
    def coverage(self) -> float:
        """판정이 필요한 주장 중 실제로 판정한 비율.

        확인 불가가 정답인 주장은 분모에서 뺍니다.
        """
        decidable = self.true_claims + self.false_claims
        judged = decidable - self.skipped_true - self.skipped_false
        return judged / decidable if decidable else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def to_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = asdict(self)
        data.update(
            recall=round(self.recall, 4),
            false_positive_rate=round(self.false_positive_rate, 4),
            precision=round(self.precision, 4),
            coverage=round(self.coverage, 4),
            f1=round(self.f1, 4),
        )
        return data


def evaluate(pairs: list[tuple[str, str]]) -> Metrics:
    """(정답, 판정) 쌍 목록에서 지표를 계산합니다.

    Args:
        pairs: [(label, verdict), ...]. 둘 다 SUPPORTED/CONTRADICTED/UNVERIFIABLE.
    """
    for label, verdict in pairs:
        if label not in LABELS:
            raise ValueError(f"알 수 없는 정답 라벨: {label!r}")
        if verdict not in LABELS:
            raise ValueError(f"알 수 없는 판정: {verdict!r}")

    def count(label: str, verdict: str) -> int:
        return sum(1 for a, b in pairs if a == label and b == verdict)

    true_claims = sum(1 for a, _ in pairs if a == SUPPORTED)
    false_claims = sum(1 for a, _ in pairs if a == CONTRADICTED)
    unverifiable = sum(1 for a, _ in pairs if a == UNVERIFIABLE)

    return Metrics(
        total=len(pairs),
        true_claims=true_claims,
        false_claims=false_claims,
        unverifiable_claims=unverifiable,
        caught=count(CONTRADICTED, CONTRADICTED),
        missed=count(CONTRADICTED, SUPPORTED),
        skipped_false=count(CONTRADICTED, UNVERIFIABLE),
        false_alarm=count(SUPPORTED, CONTRADICTED),
        confirmed=count(SUPPORTED, SUPPORTED),
        skipped_true=count(SUPPORTED, UNVERIFIABLE),
        correct_abstain=count(UNVERIFIABLE, UNVERIFIABLE),
        wrong_judgement=count(UNVERIFIABLE, SUPPORTED) + count(UNVERIFIABLE, CONTRADICTED),
    )


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """정답 x 판정 표를 만듭니다."""
    matrix = {label: {verdict: 0 for verdict in LABELS} for label in LABELS}
    for label, verdict in pairs:
        matrix[label][verdict] += 1
    return matrix


def format_metrics(name: str, metrics: Metrics) -> str:
    """사람이 읽을 수 있는 한 줄 요약."""
    return (
        f"{name:22} "
        f"재현율 {metrics.recall:6.1%}  "
        f"오탐률 {metrics.false_positive_rate:6.1%}  "
        f"정밀도 {metrics.precision:6.1%}  "
        f"커버리지 {metrics.coverage:6.1%}"
    )


def format_matrix(matrix: dict[str, dict[str, int]]) -> list[str]:
    """혼동행렬을 표로 만듭니다."""
    short = {"SUPPORTED": "근거있음", "CONTRADICTED": "모순", "UNVERIFIABLE": "확인불가"}
    lines = [f"  {'정답 \\ 판정':14}" + "".join(f"{short[v]:>10}" for v in LABELS)]
    for label in LABELS:
        row = "".join(f"{matrix[label][verdict]:>10}" for verdict in LABELS)
        lines.append(f"  {short[label]:14}{row}")
    return lines
