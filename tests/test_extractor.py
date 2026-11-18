"""주장 분해의 원문 위치 계산과 결과 조립에 대한 단위 테스트.

API 호출은 하지 않습니다. LLM 자리에는 가짜 객체를 넣습니다.
"""

from __future__ import annotations

import pytest

from pierrot.extractor import (
    ClaimType,
    ExtractedClaim,
    ExtractedClaimList,
    MatchMode,
    assemble_result,
    extract_claims,
    find_span,
    resolve_span,
)

ANSWER = (
    "삼성전자의 2024년 3분기 영업이익은 9조 1,800억원을 기록했습니다. "
    "실적 발표 다음 날 주가는 5% 상승했습니다."
)


# ---------------------------------------------------------------------------
# 원문 위치 찾기
# ---------------------------------------------------------------------------


def test_exact_match_returns_span():
    """원문에 그대로 있는 문장은 정확한 위치를 찾습니다."""
    source = "실적 발표 다음 날 주가는 5% 상승했습니다"
    span = resolve_span(ANSWER, source)
    assert span is not None
    start, end = span
    assert ANSWER[start:end] == source


def test_whitespace_difference_is_tolerated():
    """LLM이 줄바꿈을 공백으로 바꿔 적어도 찾아냅니다."""
    answer = "삼성전자의 3분기\n영업이익은   9조원입니다."
    source = "삼성전자의 3분기 영업이익은 9조원입니다."
    span = resolve_span(answer, source)
    assert span is not None
    start, end = span
    # 원문 좌표로 되돌아왔는지 확인합니다.
    assert answer[start:end] == "삼성전자의 3분기\n영업이익은   9조원입니다."


def test_missing_source_returns_none():
    """원문에 없는 문장은 찾지 못했다고 알려야 합니다."""
    assert resolve_span(ANSWER, "카카오의 매출은 2조원입니다") is None


def test_empty_inputs_return_none():
    """빈 입력에서 예외가 나면 안 됩니다."""
    assert resolve_span("", "무언가") is None
    assert resolve_span(ANSWER, "") is None


# ---------------------------------------------------------------------------
# 위치 탐색 단계별 동작
#
# 아래 두 가지는 실제 샘플 10건에서 실패했던 사례를 그대로 테스트로 옮긴 것입니다.
# ---------------------------------------------------------------------------


def test_exact_match_reports_exact_mode():
    """글자 그대로 있으면 EXACT 입니다."""
    match = find_span(ANSWER, "실적 발표 다음 날 주가는 5% 상승했습니다")
    assert match is not None
    assert match.mode is MatchMode.EXACT
    assert match.ratio == 1.0


def test_whitespace_match_reports_whitespace_mode():
    """공백 차이만 있으면 WHITESPACE 입니다."""
    answer = "삼성전자의 3분기\n영업이익은   9조원입니다."
    match = find_span(answer, "삼성전자의 3분기 영업이익은 9조원입니다.")
    assert match is not None
    assert match.mode is MatchMode.WHITESPACE


def test_markdown_emphasis_is_ignored():
    """생성 모델이 넣은 마크다운 강조 기호 때문에 못 찾으면 안 됩니다.

    실제 사례: gemma3:4b 가 "**약 135%의 높은 상승률**" 처럼 답변에 볼드를 넣었는데,
    분해 모델이 source_text 를 옮기면서 별표를 빼고 적었습니다.
    """
    answer = "주가는 상승, **약 135%의 높은 상승률**을 기록하고 있습니다."
    match = find_span(answer, "주가는 상승, 약 135%의 높은 상승률을 기록하고 있습니다.")
    assert match is not None
    assert match.mode is MatchMode.NORMALIZED
    # 원문 좌표로 되돌아왔으므로 별표를 포함한 구간이 잡혀야 합니다.
    assert "**" in answer[match.start : match.end]


def test_clause_cut_with_changed_punctuation_is_found():
    """복문의 앞 절만 떼어내며 쉼표를 마침표로 바꾼 경우도 찾아야 합니다.

    실제 사례: 원문 "...증가한 수치이며, 매출은 68조원입니다" 에서
    분해 모델이 "...증가한 수치이며." 로 적었습니다.
    """
    answer = "영업이익은 6조원을 기록했습니다. 이는 전 분기 대비 15.7% 증가한 수치이며, 매출은 68조원입니다."
    match = find_span(answer, "이는 전 분기 대비 15.7% 증가한 수치이며.")
    assert match is not None
    assert match.mode is MatchMode.NORMALIZED


def test_altered_ending_falls_back_to_fuzzy():
    """어미까지 바꿔 적으면 FUZZY 로 찾되 검토 대상으로 남깁니다."""
    answer = "현재 삼성전자 주가는 71,800원이며, 3분기 실적 발표를 앞두고 있습니다."
    match = find_span(answer, "현재 삼성전자 주가는 71,800원이다.")
    assert match is not None
    assert match.mode is MatchMode.FUZZY
    assert match.ratio >= 0.85


def test_fabricated_text_is_not_force_matched():
    """지어낸 문장이 유사도로 억지 매칭되면 탐지 기능이 죽습니다."""
    answer = "삼성전자의 2024년 3분기 영업이익은 9조 1,800억원입니다."
    assert find_span(answer, "카카오의 2024년 매출은 2조원을 기록했습니다") is None


# ---------------------------------------------------------------------------
# 결과 조립
# ---------------------------------------------------------------------------


def make_raw() -> ExtractedClaimList:
    return ExtractedClaimList(
        claims=[
            ExtractedClaim(
                text="삼성전자의 2024년 3분기 영업이익은 9조 1,800억원이다",
                claim_type=ClaimType.NUMERIC,
                source_text="삼성전자의 2024년 3분기 영업이익은 9조 1,800억원을 기록했습니다",
            ),
            ExtractedClaim(
                text="삼성전자 주가는 2024년 3분기 실적 발표 다음 날 5% 상승했다",
                claim_type=ClaimType.PRICE,
                source_text="실적 발표 다음 날 주가는 5% 상승했습니다",
            ),
            ExtractedClaim(
                text="원문에 없는 내용을 지어낸 주장",
                claim_type=ClaimType.NARRATIVE,
                source_text="이 문장은 답변에 존재하지 않습니다",
            ),
        ]
    )


def test_claim_ids_are_assigned_in_order():
    """주장마다 순번 기반 식별자가 붙습니다."""
    result = assemble_result(ANSWER, make_raw(), model_name="gpt-4o-mini")
    assert [claim.claim_id for claim in result.claims] == ["c01", "c02", "c03"]


def test_span_is_attached_when_found():
    """원문에서 찾은 주장은 span_found 가 True 이고 위치가 원문과 일치합니다."""
    result = assemble_result(ANSWER, make_raw(), model_name="gpt-4o-mini")
    first = result.claims[0]
    assert first.span_found is True
    assert first.source_span is not None
    start, end = first.source_span
    assert ANSWER[start:end] == first.source_text


def test_unmatched_claim_is_flagged():
    """원문에 없는 source_text 는 분해 품질 문제로 표시됩니다."""
    result = assemble_result(ANSWER, make_raw(), model_name="gpt-4o-mini")
    unmatched = result.unmatched_claims
    assert len(unmatched) == 1
    assert unmatched[0].claim_id == "c03"
    assert unmatched[0].source_span is None
    assert unmatched[0].match_mode is MatchMode.NOT_FOUND
    assert unmatched[0].needs_review is True


def test_match_mode_is_recorded_on_claims():
    """주장마다 어떤 방식으로 위치를 찾았는지 남습니다."""
    result = assemble_result(ANSWER, make_raw(), model_name="gpt-4o-mini")
    assert result.claims[0].match_mode is MatchMode.EXACT
    assert result.claims[0].needs_review is False
    counts = result.count_by_match_mode()
    assert counts["EXACT"] == 2
    assert counts["NOT_FOUND"] == 1


def test_review_needed_claims_collects_fuzzy_and_missing():
    """검토 대상에는 못 찾은 것과 유사도로 찾은 것이 모두 들어갑니다."""
    result = assemble_result(ANSWER, make_raw(), model_name="gpt-4o-mini")
    assert [c.claim_id for c in result.review_needed_claims] == ["c03"]


def test_count_by_type():
    """유형별 집계가 맞아야 합니다."""
    result = assemble_result(ANSWER, make_raw(), model_name="gpt-4o-mini")
    assert result.count_by_type() == {"NUMERIC": 1, "PRICE": 1, "NARRATIVE": 1}


def test_dict_output_is_accepted():
    """LLM이 pydantic 객체 대신 dict 를 돌려줘도 처리합니다."""
    raw = {
        "claims": [
            {
                "text": "삼성전자 주가는 5% 상승했다",
                "claim_type": "PRICE",
                "source_text": "실적 발표 다음 날 주가는 5% 상승했습니다",
            }
        ]
    }
    result = assemble_result(ANSWER, raw, model_name="gpt-4o-mini")
    assert result.claim_count == 1
    assert result.claims[0].claim_type is ClaimType.PRICE


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


class FakeExtractor:
    """API를 호출하지 않는 가짜 체인."""

    def __init__(self, payload: ExtractedClaimList) -> None:
        self.payload = payload
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        self.last_messages = messages
        return self.payload


def test_extract_claims_uses_injected_extractor():
    """주입한 체인을 그대로 사용하고 결과를 조립합니다."""
    fake = FakeExtractor(make_raw())
    result = extract_claims(ANSWER, extractor=fake, model_name="gpt-4o-mini")
    assert fake.call_count == 1
    assert result.claim_count == 3
    assert result.model == "gpt-4o-mini"


def test_extract_claims_sends_answer_in_prompt():
    """답변 본문이 프롬프트에 실려 나가는지 확인합니다."""
    fake = FakeExtractor(ExtractedClaimList(claims=[]))
    extract_claims(ANSWER, extractor=fake, model_name="gpt-4o-mini")
    user_message = fake.last_messages[-1]
    assert ANSWER in user_message.content


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_answer_skips_llm_call(blank):
    """빈 답변에는 API를 호출하지 않습니다. 돈을 쓰지 않기 위해서입니다."""
    fake = FakeExtractor(make_raw())
    result = extract_claims(blank, extractor=fake, model_name="gpt-4o-mini")
    assert fake.call_count == 0
    assert result.claim_count == 0
