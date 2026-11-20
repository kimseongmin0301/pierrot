"""측정 결과 페이지를 **직접 호스팅할 수 있는 독립 문서**로 감쌉니다.

왜 필요한가
-----------
`build/results.html` 은 완결된 HTML 문서가 아니라 **조각**입니다.
`<!doctype>` 도 `<html>` 도 `<head>` 도 없습니다. 아티팩트로 띄울 때는
플랫폼이 그 뼈대를 대신 씌워 주기 때문입니다.

그대로 서버에 올리면 머리말이 없는 문서가 되어 제목·언어·공유 카드가
전부 빠집니다. 그래서 같은 조각을 **독립 문서로 감싸** 따로 내보냅니다.

왜 조각을 그대로 두는가
-----------------------
`build/results.html` 하나만 고치면 두 곳이 같이 갱신되게 하려는 것입니다.
조각에 `<html>` 을 직접 넣으면 아티팩트에서는 태그가 중첩되어 깨집니다.
그래서 **조각을 원본으로 두고 서버용만 생성**합니다. 사본을 두 개 두고
손으로 맞추면 반드시 어긋납니다.

사용:
    python scripts/build_results_page.py
    python scripts/build_results_page.py --out build/results-page.html
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 확장자를 붙입니다. `/career/ax` 처럼 확장자 없는 주소는 전용 컨트롤러가
# 있을 때만 열립니다(`AxNoteController`). 이 페이지는 그런 매핑이 없어
# `/portfolio/pierrot` 은 404 입니다. canonical 과 og:url 이 404 를 가리키면
# 공유 카드가 깨집니다.
CANONICAL = "https://kimseongmin.co.kr/portfolio/pierrot.html"
TITLE = "PIERROT 검증 성능 측정"
DESCRIPTION = (
    "LLM 답변을 주장 단위로 쪼개 공시·주가와 대조하는 검증기 PIERROT 의 성능 측정 기록. "
    "정답 라벨 500건에서 재현율 100% / 오탐률 0%, 실제 LLM 답변 94건에서 판정 가능 비율 43.6%, "
    "함정을 심은 글 20건으로 찾아낸 결함 다섯 가지."
)
OG_DESCRIPTION = (
    "검증기를 만든 것이 아니라 그 검증기가 얼마나 맞히는지를 쟀습니다. "
    "베이스라인 두 종과 비교하고, 못 잡는 것이 무엇인지까지 숫자로 갈랐습니다."
)

# 본문 조각이 이미 자기 글꼴과 색을 전부 들고 있으므로 뼈대만 씌웁니다.
SKELETON = """<!doctype html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="{description}" />
    <title>{title}</title>
    <link rel="canonical" href="{canonical}" />
    <meta property="og:type" content="article" />
    <meta property="og:site_name" content="김성민" />
    <meta property="og:title" content="{title}" />
    <meta property="og:description" content="{og_description}" />
    <meta property="og:url" content="{canonical}" />
    <meta property="og:image" content="https://kimseongmin.co.kr/portfolio/images/og.jpg" />
    <meta property="og:locale" content="ko_KR" />
    <meta name="twitter:card" content="summary_large_image" />
    <link rel="icon" href="data:image/svg+xml,{favicon}" />
  </head>
  <body>
<!-- 이 파일은 scripts/build_results_page.py 가 build/results.html 을 감싸 만든 것입니다.
     직접 고치지 마십시오. 내용은 build/results.html 에서 고칩니다. ({stamp}) -->
{body}
  </body>
</html>
"""

# 돋보기. 외부 요청 없이 뜨도록 data URI 로 둡니다.
FAVICON = (
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
    "%3Ctext y='.9em' font-size='90'%3E%F0%9F%94%8D%3C/text%3E%3C/svg%3E"
)


def build(source: Path, out: Path) -> Path:
    fragment = source.read_text(encoding="utf-8")

    # 조각이 어느 날 완결 문서로 바뀌면 이중으로 감싸게 됩니다. 그때는 멈춥니다.
    lowered = fragment[:400].lower()
    if "<!doctype" in lowered or "<html" in lowered:
        raise SystemExit(
            f"{source} 가 이미 완결 문서로 보입니다. 두 번 감싸면 깨집니다.\n"
            "조각으로 되돌리거나 이 스크립트를 쓰지 마십시오."
        )

    page = SKELETON.format(
        title=TITLE,
        description=DESCRIPTION,
        og_description=OG_DESCRIPTION,
        canonical=CANONICAL,
        favicon=FAVICON,
        stamp=datetime.date.today().isoformat(),
        body=fragment.rstrip("\n"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="측정 결과 페이지를 독립 문서로 감쌉니다")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "build" / "results.html")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "build" / "results-page.html")
    args = parser.parse_args()

    written = build(args.source, args.out)
    print(f"{args.source.name} -> {written}  ({written.stat().st_size:,} bytes)")
    print(f"배포 위치: BOOT-INF/classes/static/portfolio/pierrot.html  ->  {CANONICAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
