"""근거 문장 검색.

공시 본문에서 뽑은 문장이 종목·연도당 300개쯤 됩니다. 주장 하나를 판정하려고
그 전부와 NLI 를 돌리면 CPU 로는 너무 느리고, 관련 없는 문장이 많을수록
거짓 모순이 섞일 확률도 올라갑니다.

그래서 주장과 겹치는 낱말이 많은 문장만 추려서 NLI 에 넘깁니다.
실제 RAG 가 하는 일과 같은 단계입니다.

## 이 단계가 베이스라인의 성능을 좌우합니다

검색이 엉뚱한 문장을 가져오면 NLI 가 아무리 정확해도 판정할 수 없습니다.
그래서 "NLI 가 못 맞혔다"와 "검색이 못 찾았다"를 구분해서 봐야 합니다.
판정 결과에 검색된 문장 수와 최고 점수를 남기는 이유입니다.

## 낱말이 아니라 글자 2-gram 으로 비교합니다

처음에는 낱말 겹침으로 만들었는데 실제로 돌려 보니 너무 약했습니다.

    주장: 삼성전자의 반도체 사업 부문 실적이 크게 개선되었다
    문서: DS 부문의 제56기 매출은 111조 660억원으로 전년 대비 66.8% 증가하였으며...
    -> 겹치는 낱말 0개 (반도체 vs DS, 개선 vs 증가)

한국어는 조사와 어미가 붙어 같은 뜻도 표기가 달라집니다. 글자 2-gram 으로 보면
"증가하였" 과 "증가" 처럼 부분이라도 겹치는 것을 잡을 수 있습니다.

그래도 "반도체"와 "DS" 같은 어휘 차이는 메우지 못합니다. 이것은 검색의 한계이고,
임베딩 검색을 쓰면 나아집니다. 다만 베이스라인을 과도하게 튜닝하면
"도구가 낫다"는 결론이 검색 품질 차이로 바뀌어 버리므로 단순한 방법을 쓰고,
대신 **검색이 원인인지 NLI 가 원인인지 구분해서 기록**합니다.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# 조사와 흔한 접미사를 떼어 냅니다. 형태소 분석기를 쓰지 않는 이유는
# 의존성을 늘리지 않고도 겹침 계산에는 충분하기 때문입니다.
_PARTICLES = (
    "으로써", "으로서", "에서는", "에게서", "이라고", "라고는",
    "에서", "에게", "으로", "까지", "부터", "보다", "처럼", "마다",
    "이나", "거나", "라고", "하며", "하고", "되어", "되었",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로", "만",
)

_TOKEN = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+(?:[.,]\d+)?%?")

# 어느 문장에나 나오는 낱말은 변별력이 없습니다.
STOPWORDS = frozenset(
    {
        "당사", "회사", "사업", "경우", "관련", "대한", "통해", "위해", "있습니다",
        "입니다", "하였습니다", "있으며", "등은", "또한", "그리고", "따라", "현재",
        "기준", "부문", "수준", "상황", "내용", "참고", "바랍니다", "다음", "주요",
    }
)


@dataclass(frozen=True)
class ScoredSentence:
    sentence: str
    score: float
    overlap: tuple[str, ...]
    source: str = ""


def tokenize(text: str) -> list[str]:
    """비교에 쓸 낱말을 뽑습니다. 조사를 떼고 불용어를 거릅니다."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(text or ""):
        token = raw
        for particle in _PARTICLES:
            if len(token) > len(particle) + 1 and token.endswith(particle):
                token = token[: -len(particle)]
                break
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def bigrams(text: str) -> set[str]:
    """낱말에서 글자 2-gram 을 만듭니다.

    "증가하였으며" -> {증가, 가하, 하였, 였으, 으며} 가 되어
    "증가" 와 부분이 겹칩니다. 낱말 단위 비교로는 잡지 못하는 관계입니다.
    """
    result: set[str] = set()
    for token in tokenize(text):
        if len(token) < 2:
            continue
        result.add(token)
        result.update(token[index : index + 2] for index in range(len(token) - 1))
    return result


def score_sentence(claim_tokens: Counter[str], sentence: str) -> ScoredSentence:
    """주장과 문장이 얼마나 겹치는지 점수를 구합니다.

    낱말 겹침과 글자 2-gram 겹침을 함께 봅니다. 낱말이 그대로 겹치면 더 확실한
    신호이므로 가중치를 더 줍니다.
    """
    sentence_words = set(tokenize(sentence))
    word_overlap = tuple(
        sorted(token for token in claim_tokens if token in sentence_words)
    )
    if not claim_tokens:
        return ScoredSentence(sentence, 0.0, word_overlap)

    claim_grams = bigrams(" ".join(claim_tokens))
    sentence_grams = bigrams(sentence)
    gram_hits = len(claim_grams & sentence_grams)

    word_score = len(word_overlap) / len(claim_tokens)
    gram_score = gram_hits / len(claim_grams) if claim_grams else 0.0
    return ScoredSentence(
        sentence, word_score * 0.7 + gram_score * 0.3, word_overlap
    )


def search(
    claim: str,
    sentences: list[str],
    *,
    top_k: int = 40,
    min_score: float = 0.1,
) -> list[ScoredSentence]:
    """주장과 관련 있는 문장을 점수 높은 순으로 추립니다.

    Args:
        top_k: 최대 몇 개까지 넘길지.
        min_score: 이 점수에 못 미치면 버립니다. 0 이면 전부 통과합니다.
    """
    claim_tokens = Counter(tokenize(claim))
    if not claim_tokens or not sentences:
        return []

    scored = [score_sentence(claim_tokens, sentence) for sentence in sentences]
    scored = [item for item in scored if item.score >= min_score]
    scored.sort(key=lambda item: (-item.score, len(item.sentence)))
    return scored[:top_k]
