"""DART 공시 본문 수집과 문장 추출.

NARRATIVE 주장("반도체 부문 실적이 개선되면서 영업이익 성장을 견인했다")은
수치만으로 판정할 수 없습니다. 서술이 담긴 문서가 있어야 합니다.

사업보고서의 아래 두 부분에 그런 내용이 들어 있습니다.

    II. 사업의 내용            사업 개요, 주요 제품, 매출 및 수주, 시장 상황
    이사의 경영진단 및 분석의견   실적 변동 원인에 대한 회사의 설명

실제로 삼성전자 2024년 사업보고서에는 "메모리 평균 판매가격은 전년 대비 약 63%
상승하였으며" 같은 문장이 있습니다. 수치 스냅샷으로는 만들 수 없는 근거입니다.

## 본문을 그대로 저장하지 않는 이유

사업보고서 한 건의 본체 XML 이 약 6MB 입니다. 20종목 x 3년이면 300MB 가 넘고,
대부분은 재무제표 표와 주석이라 NARRATIVE 판정에 쓸모가 없습니다.

그래서 받는 즉시 필요한 절만 잘라 내고, 태그를 벗기고, 문장으로 쪼개서
문장만 저장합니다. 저장소에 올라가는 것은 문장 parquet 하나입니다.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import Settings, get_settings
from .dart import DartError, _require_key

DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("사업의 내용", "II. 사업의 내용", "III. 재무에 관한 사항"),
    ("경영진단", "IV. 이사의 경영진단 및 분석의견", "V. 회계감사인의 감사의견 등"),
)
"""(절 이름, 시작 표지, 끝 표지).

DART 본문에는 목차와 본문에 같은 제목이 두 번 나옵니다. 뒤에 나오는 것(본문)을
쓰기 위해 마지막 출현 위치를 찾습니다.
"""

MIN_SENTENCE_LENGTH = 20
MAX_SENTENCE_LENGTH = 300
"""문장 길이 제한.

너무 짧으면 표의 조각이거나 제목이고, 너무 길면 표가 문장처럼 붙은 것입니다.
NLI 입력이 512 토큰이라 긴 문장은 잘리기도 합니다.
"""

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[다요음])\.\s+|(?<=[다요음])\.$")

# 표에서 뽑혀 나온 조각을 걸러 냅니다.
_NOISE = re.compile(r"^[\s\d,.\-()%]+$")


@dataclass(frozen=True)
class DocumentSentence:
    code: str
    bsns_year: int
    rcept_no: str
    section: str
    sentence: str


def strip_markup(text: str) -> str:
    """XML 태그를 벗기고 공백을 정리합니다."""
    without_tags = _TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", without_tags).strip()


_TITLE_TAG = re.compile(r"<TITLE[^>]*>(.*?)</TITLE>", re.S)

# 원문에 로마숫자가 ASCII(II)와 전각(Ⅱ)으로 섞여 있습니다.
_ROMAN = str.maketrans({
    "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V",
    "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X",
})


def normalize_title(text: str) -> str:
    """제목을 비교할 수 있는 형태로 맞춥니다."""
    plain = _TAG.sub("", text or "").translate(_ROMAN)
    return _WHITESPACE.sub(" ", plain).strip()


def extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """시작 표지와 끝 표지 사이를 잘라 냅니다.

    문자열을 그대로 찾으면 안 됩니다. 본문에는 "자세한 사항은 'II. 사업의 내용'을
    참고하시기 바랍니다" 같은 상호참조가 여러 번 나오고, 처음에 rfind 로 마지막
    출현을 잡았더니 그런 참조를 집어 절 내용을 통째로 놓쳤습니다.

    그래서 TITLE 태그의 위치를 기준으로 자릅니다. 제목으로 쓰인 것만 봅니다.
    """
    titles = [
        (match.start(), match.end(), normalize_title(match.group(1)))
        for match in _TITLE_TAG.finditer(text)
    ]
    if not titles:
        return ""

    start_key = normalize_title(start_marker)
    end_key = normalize_title(end_marker)

    begin_index = next(
        (index for index, (_, _, title) in enumerate(titles) if title == start_key),
        None,
    )
    if begin_index is None:
        begin_index = next(
            (
                index
                for index, (_, _, title) in enumerate(titles)
                if title.startswith(start_key)
            ),
            None,
        )
    if begin_index is None:
        return ""

    begin = titles[begin_index][1]
    for _, (position, _, title) in enumerate(titles[begin_index + 1 :], start=1):
        if title == end_key or title.startswith(end_key):
            return text[begin:position]
    return text[begin:]


def split_sentences(text: str) -> list[str]:
    """한국어 문장으로 쪼갭니다.

    종결어미(다/요/음) 뒤의 마침표를 기준으로 나눕니다. 완벽하지는 않지만
    NLI 입력으로 쓰기에는 충분하고, 규칙이라 재현됩니다.
    """
    if not text:
        return []
    pieces = _SENTENCE_END.split(text)
    result: list[str] = []
    for piece in pieces:
        if piece is None:
            continue
        candidate = piece.strip()
        if not candidate:
            continue
        if not candidate.endswith("."):
            candidate += "."
        result.append(candidate)
    return result


def clean_sentences(sentences: list[str]) -> list[str]:
    """쓸 수 없는 문장을 걸러 냅니다."""
    seen: set[str] = set()
    result: list[str] = []
    for sentence in sentences:
        text = sentence.strip()
        if not MIN_SENTENCE_LENGTH <= len(text) <= MAX_SENTENCE_LENGTH:
            continue
        if _NOISE.match(text):
            continue
        # 숫자와 기호만으로 이루어진 비율이 높으면 표 조각입니다.
        hangul = sum(1 for ch in text if "가" <= ch <= "힣")
        if hangul < len(text) * 0.3:
            continue
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def fetch_document_text(
    rcept_no: str, *, settings: Settings | None = None, pause: float = 0.5
) -> str:
    """공시 본문 XML 을 받아 본체 파일의 내용을 돌려줍니다.

    ZIP 안에는 본체와 감사보고서가 함께 들어 있습니다. 파일명에 밑줄이 없는 것이
    본체입니다(`{접수번호}.xml`). 처음에 감사보고서를 본문으로 착각한 적이 있습니다.
    """
    settings = settings or get_settings()
    key = _require_key(settings)

    response = requests.get(
        DOCUMENT_URL, params={"crtfc_key": key, "rcept_no": rcept_no}, timeout=300
    )
    if pause:
        time.sleep(pause)
    response.raise_for_status()

    if response.content[:2] != b"PK":
        raise DartError(f"ZIP 응답이 아닙니다: {response.text[:200]}")

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    main = next((name for name in names if "_" not in name), None)
    if main is None:
        main = max(names, key=lambda name: archive.getinfo(name).file_size)
    return archive.read(main).decode("utf-8", errors="replace")


class DocumentStore:
    """수집해 둔 공시 문장을 조회합니다.

    주가·재무 스냅샷과 같은 원칙입니다. 검증 중에는 DART 에 접속하지 않습니다.
    """

    def __init__(self, frame) -> None:
        self._frame = frame

    @classmethod
    def load(cls, path=None) -> "DocumentStore":
        import pandas as pd  # noqa: PLC0415

        from ..config import get_settings as _get_settings  # noqa: PLC0415

        if path is None:
            path = _get_settings().snapshot_dir / "dart_sentences.parquet"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"공시 문장 파일이 없습니다: {path}\n"
                "먼저 scripts/fetch_dart_documents.py 를 실행해 주십시오."
            )
        return cls(pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self._frame)

    def codes(self) -> list[str]:
        return sorted(self._frame["code"].unique())

    def year_range(self) -> tuple[int, int] | None:
        if self._frame.empty:
            return None
        return (int(self._frame["bsns_year"].min()), int(self._frame["bsns_year"].max()))

    def sentences(self, code: str, years: tuple[int, ...]) -> list[tuple[str, str]]:
        """(절 이름, 문장) 목록을 돌려줍니다."""
        selected = self._frame[
            (self._frame["code"] == code) & (self._frame["bsns_year"].isin(list(years)))
        ]
        return [
            (str(row["section"]), str(row["sentence"]))
            for _, row in selected.iterrows()
        ]

    def describe(self) -> str:
        span = self.year_range()
        if span is None:
            return "문장 없음"
        return (
            f"{len(self):,}문장 / {len(self.codes())}종목 / {span[0]}~{span[1]}년"
        )


def document_sentences(
    code: str, bsns_year: int, rcept_no: str, raw_xml: str
) -> list[DocumentSentence]:
    """본문에서 서술 절만 잘라 문장으로 만듭니다."""
    result: list[DocumentSentence] = []
    for section_name, start_marker, end_marker in SECTIONS:
        chunk = extract_section(raw_xml, start_marker, end_marker)
        if not chunk:
            continue
        sentences = clean_sentences(split_sentences(strip_markup(chunk)))
        result.extend(
            DocumentSentence(
                code=code,
                bsns_year=bsns_year,
                rcept_no=rcept_no,
                section=section_name,
                sentence=sentence,
            )
            for sentence in sentences
        )
    return result
