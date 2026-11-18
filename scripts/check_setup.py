"""0단계 동작 확인 스크립트.

다음 네 가지를 실제로 실행해서 확인합니다. 하나라도 실패하면 종료 코드가 1입니다.

1. 설정 로딩         — .env 를 읽고 필요한 값이 채워져 있는가
2. Ollama (무료)     — SSH 터널이 열려 있고 gemma3:4b 가 응답하는가
3. OpenAI (유료)     — 실제 호출 1회가 성공하고 비용이 로그에 기록되는가
4. 응답 캐시         — 같은 프롬프트를 다시 보낼 때 과금 없이 캐시로 처리되는가

실행:
    .venv/Scripts/python.exe scripts/check_setup.py
    .venv/Scripts/python.exe scripts/check_setup.py --skip-ollama
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# langchain-community 의 캐시 관련 예고성 경고입니다. 동작에는 문제가 없고
# 확인 결과를 읽기 어렵게 만들기만 하므로 이 스크립트에서만 가립니다.
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

# src 레이아웃이라 설치 없이도 돌아가도록 경로를 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pierrot.config import get_settings  # noqa: E402
from pierrot.cost import summarize_log  # noqa: E402
from pierrot.llm import get_cost_logger, get_ollama_chat, get_openai_chat, setup_cache  # noqa: E402

OK = "[ OK ]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def section(title: str) -> None:
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


def read_log_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def check_settings() -> tuple[bool, object]:
    section("1. 설정 로딩")
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"{FAIL} 설정을 읽지 못했습니다: {exc}")
        return False, None

    masked = settings.masked()
    for key in (
        "openai_api_key",
        "openai_extractor_model",
        "openai_verifier_model",
        "openai_judge_model",
        "ollama_base_url",
        "ollama_generator_model",
        "dart_api_key",
        "nli_model",
        "monthly_budget_usd",
        "cache_path",
        "cost_log_path",
    ):
        print(f"       {key:26} = {masked[key]}")

    if not settings.openai_api_key:
        print(f"{FAIL} OPENAI_API_KEY 가 비어 있습니다. .env 를 확인해 주십시오.")
        return False, settings
    print(f"{OK} 설정 로딩 성공")
    return True, settings


def check_ollama(settings) -> bool:
    section("2. Ollama (로컬, 무료)")
    parsed = urlparse(settings.ollama_base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    sock = socket.socket()
    sock.settimeout(3)
    try:
        sock.connect((host, port))
    except OSError as exc:
        print(f"{FAIL} {host}:{port} 에 연결할 수 없습니다 ({exc}).")
        print("       SSH 터널이 열려 있는지 확인해 주십시오:")
        print("           ssh -N -L 11434:127.0.0.1:11434 ai-server")
        return False
    finally:
        sock.close()
    print(f"{OK} {host}:{port} 연결 확인")

    try:
        chat = get_ollama_chat(num_predict=24)
        started = time.perf_counter()
        reply = chat.invoke("한 문장으로 자기소개를 해 주세요.")
        elapsed = time.perf_counter() - started
    except Exception as exc:
        print(f"{FAIL} {settings.ollama_generator_model} 호출 실패: {type(exc).__name__}: {exc}")
        return False

    text = (reply.content or "").strip().replace("\n", " ")
    print(f"{OK} {settings.ollama_generator_model} 응답 ({elapsed:.1f}초): {text[:80]}")
    return True


def check_openai(settings) -> bool:
    section("3. OpenAI (유료) + 비용 기록")
    log_path = settings.cost_log_path
    before = len(read_log_lines(log_path))

    # 캐시에 걸리지 않도록 실행 시각을 프롬프트에 넣어 반드시 실제 호출이 나가게 합니다.
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        chat = get_openai_chat("extractor", max_tokens=16)
        reply = chat.invoke(f"Reply with exactly the word PONG. (run id: {stamp})")
    except Exception as exc:
        print(f"{FAIL} OpenAI 호출 실패: {type(exc).__name__}: {exc}")
        return False

    print(f"{OK} {settings.openai_extractor_model} 응답: {(reply.content or '').strip()!r}")

    rows = read_log_lines(log_path)
    if len(rows) <= before:
        print(f"{FAIL} 비용 로그에 새 기록이 남지 않았습니다: {log_path}")
        return False

    latest = rows[-1]
    print(f"       기록된 줄: {json.dumps(latest, ensure_ascii=False)}")
    if latest.get("cache_hit"):
        print(f"{FAIL} 실제 호출이어야 하는데 캐시 적중으로 기록되었습니다.")
        return False
    if (latest.get("cost_usd") or 0) <= 0:
        print(f"{FAIL} 비용이 0으로 기록되었습니다. 단가표를 확인해 주십시오.")
        return False

    print(f"{OK} 비용 기록 확인: ${latest['cost_usd']:.8f} "
          f"(입력 {latest['input_tokens']} / 출력 {latest['output_tokens']} 토큰)")
    return True


def check_cache(settings) -> bool:
    section("4. 응답 캐시 (같은 프롬프트 재호출 시 과금 없음)")
    setup_cache(settings)
    log_path = settings.cost_log_path
    prompt = "Reply with exactly the word CACHED."

    try:
        chat = get_openai_chat("extractor", max_tokens=16)
        chat.invoke(prompt)  # 첫 호출: 캐시에 없으면 실제 호출
        chat.invoke(prompt)  # 두 번째 호출: 캐시에 있어야 함
    except Exception as exc:
        print(f"{FAIL} 캐시 확인 중 호출 실패: {type(exc).__name__}: {exc}")
        return False

    rows = read_log_lines(log_path)
    if not rows:
        print(f"{FAIL} 비용 로그가 비어 있습니다.")
        return False

    latest = rows[-1]
    if not latest.get("cache_hit"):
        print(f"{FAIL} 두 번째 호출이 캐시로 처리되지 않았습니다: "
              f"{json.dumps(latest, ensure_ascii=False)}")
        return False

    print(f"{OK} 두 번째 호출이 캐시로 처리됨 "
          f"(비용 ${latest['cost_usd']:.8f}, 아낀 토큰 {latest['tokens_saved_by_cache']})")
    print(f"       캐시 파일: {settings.cache_path}")
    return True


def print_summary(settings) -> None:
    section("비용 요약")
    logger = get_cost_logger(settings)
    summary = logger.summary()
    print(f"       이번 달({summary['month']}) 사용액 : ${summary['spent_usd']:.6f}")
    print(f"       예산 상한                : ${summary['budget_usd']:.2f}")
    print(f"       남은 예산                : ${summary['remaining_usd']:.6f}")

    for month, bucket in sorted(summarize_log(settings.cost_log_path).items()):
        print(f"       [{month}] 호출 {bucket['calls']}회 "
              f"(캐시 적중 {bucket['cache_hits']}회), "
              f"입력 {bucket['input_tokens']} / 출력 {bucket['output_tokens']} 토큰, "
              f"비용 ${bucket['cost_usd']:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PIERROT 0단계 동작 확인")
    parser.add_argument("--skip-ollama", action="store_true", help="Ollama 확인을 건너뜁니다")
    parser.add_argument("--skip-openai", action="store_true", help="OpenAI 확인을 건너뜁니다 (과금 없음)")
    args = parser.parse_args()

    print("=" * 72)
    print("PIERROT 0단계 동작 확인")
    print("=" * 72)

    results: dict[str, bool | None] = {}

    passed, settings = check_settings()
    results["설정 로딩"] = passed
    if not passed or settings is None:
        _print_results(results)
        return 1

    if args.skip_ollama:
        section("2. Ollama (로컬, 무료)")
        print(f"{SKIP} --skip-ollama 지정")
        results["Ollama"] = None
    else:
        results["Ollama"] = check_ollama(settings)

    if args.skip_openai:
        section("3. OpenAI (유료) + 비용 기록")
        print(f"{SKIP} --skip-openai 지정")
        results["OpenAI + 비용 기록"] = None
        results["응답 캐시"] = None
    else:
        results["OpenAI + 비용 기록"] = check_openai(settings)
        results["응답 캐시"] = check_cache(settings)

    print_summary(settings)
    _print_results(results)
    return 0 if all(v is not False for v in results.values()) else 1


def _print_results(results: dict) -> None:
    section("결과")
    for name, value in results.items():
        mark = SKIP if value is None else (OK if value else FAIL)
        print(f"{mark} {name}")
    failed = [name for name, value in results.items() if value is False]
    print()
    if failed:
        print(f"실패한 항목: {', '.join(failed)}")
    else:
        print("모든 확인 항목을 통과했습니다.")


if __name__ == "__main__":
    sys.exit(main())
