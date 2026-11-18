"""검토용 샘플 답변 10개 생성.

로컬 Ollama(gemma3:4b)로 금융 질문에 대한 답변을 만듭니다. 비용은 들지 않습니다.

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

# 답변을 자신 있게, 구체적인 수치를 들어 하게 만듭니다.
# 환각을 유도하는 것이 목적이므로 "모르면 모른다고 하라"는 지시를 넣지 않습니다.
SYSTEM_PROMPT = (
    "당신은 한국 주식시장을 담당하는 증권사 애널리스트입니다. "
    "질문에 대해 구체적인 수치와 날짜를 들어 4~6문장으로 간결하게 답변하십시오. "
    "영업이익, 매출, 주가 등락률 같은 숫자를 반드시 포함하십시오."
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
]


def main() -> int:
    parser = argparse.ArgumentParser(description="검토용 샘플 답변 생성")
    parser.add_argument("--force", action="store_true", help="기존 파일을 덮어씁니다")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개만 생성합니다")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        print(f"이미 존재합니다: {OUTPUT_PATH}")
        print("다시 생성하려면 --force 를 붙여 주십시오.")
        return 0

    settings = get_settings()
    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    print(f"모델   : {settings.ollama_generator_model} ({settings.ollama_base_url})")
    print(f"질문 수: {len(questions)}")
    print(f"출력   : {OUTPUT_PATH}")
    print()

    chat = get_ollama_chat(num_predict=400)
    rows: list[dict] = []

    for index, item in enumerate(questions, start=1):
        started = time.perf_counter()
        try:
            reply = chat.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=item["question"]),
                ]
            )
        except Exception as exc:
            print(f"[{index:2}/{len(questions)}] {item['id']} 실패: {type(exc).__name__}: {exc}")
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
                "model": settings.ollama_generator_model,
            }
        )
        preview = answer.replace("\n", " ")[:60]
        print(f"[{index:2}/{len(questions)}] {item['id']} {elapsed:5.1f}초 "
              f"{len(answer):4}자  {preview}...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"저장 완료: {OUTPUT_PATH} ({len(rows)}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
