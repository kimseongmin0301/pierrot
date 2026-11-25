"""검토용 샘플 답변 10개 생성.

금융 질문에 대한 LLM 답변을 만듭니다. 이 답변이 곧 검증 대상입니다.

생성 모델을 둘 중에 고릅니다.

    --model local   로컬 Ollama(gemma3:4b). 공짜지만 4B 라 수치를 거의 다 지어냅니다.
    --model openai  OpenAI gpt-4o-mini. 맞는 말도 하고 틀린 말도 섞습니다.

**왜 둘 다 필요한가.** gemma3:4b 답변에서 뽑은 주장 94개 중 맞는 것이 0개였습니다.
참인 주장이 하나도 없으면 "맞는 문장을 틀렸다고 하지는 않는가"(오탐)를
**실제 글에서는 전혀 잴 수 없습니다.** 커버리지만 재고 끝납니다.
참과 거짓이 섞여 나오는 모델이라야 그 측정이 제 구실을 합니다.

의도적으로 근거 문서를 주지 않습니다. 4B급 모델이 아무 자료 없이 구체적인 수치를
말하게 하면 환각이 자연스럽게 섞여 나오고, 그것이 바로 이 프로젝트가 검증해야 할
입력이기 때문입니다. RAG 연결은 뒤 단계에서 붙입니다.

실행 전에 SSH 터널이 열려 있어야 합니다.

    ssh -N -L 11434:127.0.0.1:11434 ai-server

실행:
    .venv/Scripts/python.exe scripts/generate_samples.py
    .venv/Scripts/python.exe scripts/generate_samples.py --force   # 기존 파일 덮어쓰기
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

warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from pierrot.config import get_settings  # noqa: E402
from pierrot.llm import get_ollama_chat  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "sample_answers.jsonl"
OUTPUT_PATH_OPENAI = (
    PROJECT_ROOT / "evaluation" / "datasets" / "sample_answers_openai.jsonl"
)

# 답변을 자신 있게, 구체적인 수치를 들어 하게 만듭니다.
# 환각을 유도하는 것이 목적이므로 "모르면 모른다고 하라"는 지시를 넣지 않습니다.
SYSTEM_PROMPT = (
    "당신은 한국 주식시장을 담당하는 증권사 애널리스트입니다. "
    "질문에 대해 구체적인 수치와 날짜를 들어 4~6문장으로 간결하게 답변하십시오. "
    "영업이익, 매출, 주가 등락률 같은 숫자를 반드시 포함하십시오."
)

# gpt-4o-mini 전용 지시.
#
# 같은 프롬프트를 주면 2024년을 아직 오지 않은 때처럼 다뤄 거의 모든 문장을
# "예상됩니다 / 전망입니다" 로 씁니다(실측: 답변 10건에 전망 표현 21회).
# 그러면 검증기가 전부 전망 경로로 흘려보내 **커버리지만 0 에 가깝게 나오고**
# 정작 재려던 것을 못 잽니다.
#
# 검증기가 실제로 상대하는 글은 추정이 아니라 **확정적으로 말하는 글**입니다.
# 그래서 끝난 기간을 끝난 일로 쓰게 못박습니다. 틀리게 쓰라고는 하지 않습니다.
#
# 로컬용 프롬프트는 건드리지 않습니다. 그걸 바꾸면 이미 저장된
# sample_answers.jsonl 을 같은 조건으로 다시 만들 수 없습니다.
SYSTEM_PROMPT_OPENAI = (
    SYSTEM_PROMPT
    + " 질문에 나온 기간은 이미 끝난 기간입니다. "
    "'예상됩니다', '전망입니다', '계획입니다' 같은 추정 표현을 쓰지 말고 "
    "이미 발표된 실적을 보고하듯 과거형으로 확정해서 쓰십시오."
)

QUESTIONS: list[dict[str, str]] = [
    {"id": "q01", "company": "삼성전자",
     "question": "삼성전자의 2024년 3분기 영업이익과 실적 발표 이후 주가 흐름을 알려 주세요."},
    {"id": "q02", "company": "SK하이닉스",
     "question": "SK하이닉스의 2024년 연간 매출과 HBM 사업 현황을 설명해 주세요."},
    {"id": "q03", "company": "현대차",
     "question": "현대차의 2024년 영업이익률과 배당 정책을 알려 주세요."},
    {"id": "q04", "company": "LG에너지솔루션",
     "question": "LG에너지솔루션의 2024년 실적과 그해 주가 변동 폭을 설명해 주세요."},
    {"id": "q05", "company": "NAVER",
     "question": "NAVER의 2024년 광고 부문 매출과 주가 흐름을 요약해 주세요."},
    {"id": "q06", "company": "카카오",
     "question": "카카오의 2024년 실적 부진 원인과 주가 하락 폭을 설명해 주세요."},
    {"id": "q07", "company": "POSCO홀딩스",
     "question": "POSCO홀딩스의 2024년 철강 부문 영업이익과 주가 추이를 알려 주세요."},
    {"id": "q08", "company": "셀트리온",
     "question": "셀트리온의 2024년 매출과 합병 이후 주가 변화를 설명해 주세요."},
    {"id": "q09", "company": "KB금융",
     "question": "KB금융의 2024년 지배주주 순이익과 배당수익률을 알려 주세요."},
    {"id": "q10", "company": "삼성바이오로직스",
     "question": "삼성바이오로직스의 2024년 수주 실적과 주가 상승률을 요약해 주세요."},
    {"id": "q11", "company": "기아",
     "question": "기아의 2024년 연간 매출과 영업이익을 알려 주세요."},
    {"id": "q12", "company": "신한지주",
     "question": "신한지주의 2024년 당기순이익과 2024년 주가 흐름을 설명해 주세요."},
    {"id": "q13", "company": "하나금융지주",
     "question": "하나금융지주의 2023년과 2024년 순이익을 비교해 주세요."},
    {"id": "q14", "company": "우리금융지주",
     "question": "우리금융지주의 2024년 연간 실적을 요약해 주세요."},
    {"id": "q15", "company": "현대모비스",
     "question": "현대모비스의 2024년 매출과 영업이익을 알려 주세요."},
    {"id": "q16", "company": "LG화학",
     "question": "LG화학의 2024년 실적과 2024년 주가 등락을 설명해 주세요."},
    {"id": "q17", "company": "삼성SDI",
     "question": "삼성SDI의 2024년 매출이 전년 대비 어떻게 변했는지 알려 주세요."},
    {"id": "q18", "company": "삼성물산",
     "question": "삼성물산의 2024년 매출과 영업이익을 요약해 주세요."},
    {"id": "q19", "company": "한국전력",
     "question": "한국전력의 2024년 영업이익과 적자 해소 여부를 설명해 주세요."},
    {"id": "q20", "company": "SK텔레콤",
     "question": "SK텔레콤의 2024년 매출과 배당 정책을 알려 주세요."},
    {"id": "q21", "company": "삼성전자",
     "question": "삼성전자의 2023년 연간 영업이익과 2024년 회복 폭을 설명해 주세요."},
    {"id": "q22", "company": "SK하이닉스",
     "question": "SK하이닉스의 2024년 영업이익이 전년 대비 얼마나 늘었는지 알려 주세요."},
    {"id": "q23", "company": "현대차",
     "question": "현대차의 2023년과 2024년 연간 매출을 비교해 주세요."},
    {"id": "q24", "company": "기아",
     "question": "기아의 2024년 1분기 실적과 그 시점 주가를 알려 주세요."},
    {"id": "q25", "company": "NAVER",
     "question": "NAVER의 2024년 연간 매출과 영업이익을 설명해 주세요."},
    {"id": "q26", "company": "카카오",
     "question": "카카오의 2024년 연간 매출과 순이익을 알려 주세요."},
    {"id": "q27", "company": "LG에너지솔루션",
     "question": "LG에너지솔루션의 2023년 영업이익과 2024년 변화를 설명해 주세요."},
    {"id": "q28", "company": "셀트리온",
     "question": "셀트리온의 2023년 매출과 2024년 매출을 비교해 주세요."},
    {"id": "q29", "company": "POSCO홀딩스",
     "question": "POSCO홀딩스의 2024년 연간 매출과 영업이익을 알려 주세요."},
    {"id": "q30", "company": "삼성바이오로직스",
     "question": "삼성바이오로직스의 2024년 매출과 영업이익을 설명해 주세요."},
    {"id": "q31", "company": "KB금융",
     "question": "KB금융의 2024년 연간 당기순이익과 배당금을 알려 주세요."},
    {"id": "q32", "company": "한국전력",
     "question": "한국전력의 2023년 영업손실 규모를 설명해 주세요."},
    {"id": "q33", "company": "삼성전자",
     "question": "삼성전자의 2024년 주가 최고가와 최저가를 알려 주세요."},
    {"id": "q34", "company": "SK하이닉스",
     "question": "SK하이닉스의 2024년 주가 상승률을 설명해 주세요."},
    {"id": "q35", "company": "현대차",
     "question": "현대차의 2024년 주당 배당금과 배당 총액을 알려 주세요."},
    {"id": "q36", "company": "신한지주",
     "question": "신한지주의 2023년 대비 2024년 순이익 증감을 설명해 주세요."},
    {"id": "q37", "company": "LG화학",
     "question": "LG화학의 2024년 순이익이 흑자인지 적자인지 알려 주세요."},
    {"id": "q38", "company": "삼성물산",
     "question": "삼성물산의 2024년 주가 흐름을 요약해 주세요."},
    {"id": "q39", "company": "현대모비스",
     "question": "현대모비스의 2024년 순이익과 전년 대비 증감을 설명해 주세요."},
    {"id": "q40", "company": "SK텔레콤",
     "question": "SK텔레콤의 2024년 영업이익과 순이익을 알려 주세요."},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="검토용 샘플 답변 생성")
    parser.add_argument("--force", action="store_true", help="기존 파일을 덮어씁니다")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개만 생성합니다")
    parser.add_argument("--model", choices=["local", "openai"], default="local",
                        help="local=Ollama gemma3:4b(무료), openai=gpt-4o-mini(유료)")
    args = parser.parse_args()

    settings = get_settings()
    output = OUTPUT_PATH if args.model == "local" else OUTPUT_PATH_OPENAI

    if output.exists() and not args.force:
        print(f"이미 존재합니다: {output}")
        print("다시 생성하려면 --force 를 붙여 주십시오.")
        return 0

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    if args.model == "local":
        model_name = settings.ollama_generator_model
        chat = get_ollama_chat(num_predict=400)
        where = settings.ollama_base_url
    else:
        from pierrot.llm import get_openai_chat
        model_name = settings.openai_extractor_model
        chat = get_openai_chat("extractor", temperature=0.7, max_tokens=600,
                               source="generator_openai", settings=settings)
        where = "OpenAI API"
    system_prompt = SYSTEM_PROMPT if args.model == "local" else SYSTEM_PROMPT_OPENAI

    print(f"모델   : {model_name} ({where})")
    print(f"질문 수: {len(questions)}")
    print(f"출력   : {output}")
    print()
    rows: list[dict] = []

    for index, item in enumerate(questions, start=1):
        started = time.perf_counter()
        try:
            reply = chat.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=item["question"]),
                ]
            )
        except Exception as exc:
            print(f"[{index:2}/{len(questions)}] {item['id']} 실패: {type(exc).__name__}: {exc}")
            if args.model == "local":
                print("       SSH 터널이 열려 있는지 확인해 주십시오:")
                print("           ssh -N -L 11434:127.0.0.1:11434 ai-server")
            return 1

        elapsed = time.perf_counter() - started
        answer = (reply.content or "").strip()
        rows.append(
            {
                "id": item["id"],
                "company": item["company"],
                "question": item["question"],
                "answer": answer,
                "model": model_name,
            }
        )
        preview = answer.replace("\n", " ")[:60]
        print(f"[{index:2}/{len(questions)}] {item['id']} {elapsed:5.1f}초 "
              f"{len(answer):4}자  {preview}...")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"저장 완료: {output} ({len(rows)}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
