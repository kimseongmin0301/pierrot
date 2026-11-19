"""검증 API 서버를 띄웁니다.

    .venv/Scripts/python.exe scripts/run_api.py
    .venv/Scripts/python.exe scripts/run_api.py --port 8080 --reload

띄운 뒤 문서는 http://127.0.0.1:8000/docs 에서 볼 수 있습니다.

워커를 하나만 두는 이유: 검증은 한 번에 하나씩 돌아갑니다
(`api/service.py` 설명 참고). 비용을 요청별로 정확히 재려면 겹치면 안 되고,
NLI 모델과 스냅샷도 스레드 안전하지 않습니다. 처리량이 필요하면
프로세스를 여러 개 띄우는 편이 낫습니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="PIERROT 검증 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="코드가 바뀌면 다시 올립니다")
    args = parser.parse_args()

    import uvicorn

    from pierrot.config import get_settings

    settings = get_settings()
    print("=" * 88)
    print("PIERROT 검증 API")
    print("=" * 88)
    print(f"  주소     : http://{args.host}:{args.port}")
    print(f"  문서     : http://{args.host}:{args.port}/docs")
    print(f"  스냅샷   : {settings.snapshot_dir}")
    print(f"  월 예산  : ${settings.monthly_budget_usd:.2f}")
    print()
    print("  시작할 때 스냅샷과 NLI 모델을 올립니다. 몇 초 걸립니다.")
    print()

    uvicorn.run(
        "pierrot.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
