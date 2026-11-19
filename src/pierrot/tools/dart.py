"""DART 공시 API 클라이언트.

**수집 전용입니다.** 검증 도구와 평가 파이프라인은 저장된 스냅샷만 읽습니다.
주가와 같은 원칙입니다. 외부 의존성을 재현성 경계 밖으로 밀어냅니다.

## 알아 둬야 할 DART 특성 두 가지

### 1. 분기보고서의 기간 표기와 금액이 어긋납니다

`thstrm_dt` 는 누적 기간으로 적혀 있지만 `thstrm_amount` 는 **당분기 3개월치**입니다.

    1분기보고서  기간 2024.01.01~03.31  영업이익  6.61조
    반기보고서   기간 2024.01.01~06.30  영업이익 10.44조   <- 누적이면 17조여야 함
    3분기보고서  기간 2024.01.01~09.30  영업이익  9.18조   <- 누적이면 반기보다 커야 함
    사업보고서   기간 2024.01.01~12.31  영업이익 32.73조

    1~3분기 합 26.23조, 연간 32.73조 -> 4분기 6.49조. 실제와 맞습니다.

누적으로 오해하면 모든 분기 주장이 틀린 값과 비교됩니다. 그래서 스냅샷에
`period_months`(분기 3, 연간 12)를 명시해 둡니다.

### 2. 연결(CFS)과 별도(OFS)의 수치가 크게 다릅니다

    삼성전자 2024 영업이익: 연결 32.73조 / 별도 12.36조

LLM 답변은 보통 어느 쪽인지 밝히지 않습니다. 그래서 둘 다 저장하고,
판정 단계에서 어느 쪽과 맞았는지 기록합니다.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree

import requests

from ..config import Settings, get_settings

BASE_URL = "https://opendart.fss.or.kr/api"
CORP_CODE_URL = f"{BASE_URL}/corpCode.xml"
SINGLE_ACCOUNT_URL = f"{BASE_URL}/fnlttSinglAcnt.json"

REPORT_CODES: dict[str, dict[str, object]] = {
    "11013": {"label": "1분기", "quarter": 1, "months": 3},
    "11012": {"label": "2분기", "quarter": 2, "months": 3},
    "11014": {"label": "3분기", "quarter": 3, "months": 3},
    "11011": {"label": "연간", "quarter": 0, "months": 12},
}
"""보고서 코드.

11012 는 '반기보고서'지만 금액이 당분기(2분기 3개월)이므로 2분기로 다룹니다.
"""

# DART 계정명을 표준 이름으로 맞춥니다.
# 회사마다 표기가 조금씩 달라 판정 단계에서 일일이 대조할 수 없습니다.
ACCOUNT_ALIASES: dict[str, str] = {
    "매출액": "매출액",
    "수익(매출액)": "매출액",
    "영업수익": "매출액",
    "영업이익": "영업이익",
    "영업이익(손실)": "영업이익",
    "당기순이익(손실)": "당기순이익",
    "당기순이익": "당기순이익",
    "법인세차감전 순이익": "법인세차감전순이익",
    "자산총계": "자산총계",
    "부채총계": "부채총계",
    "자본총계": "자본총계",
    "유동자산": "유동자산",
    "유동부채": "유동부채",
    "이익잉여금": "이익잉여금",
    "자본금": "자본금",
    # 주요계정 API 밖에서 모은 것들 (scripts/fetch_dart_extra.py).
    # 여기 더하면 `numeric_claim.KNOWN_ACCOUNTS` 와 파서 프롬프트가 함께 늘어납니다.
    "배당금총액": "배당금총액",
    "현금배당금총액": "배당금총액",
    "주당배당금": "주당배당금",
    "주당 현금배당금": "주당배당금",
    "지배주주순이익": "지배주주순이익",
    "영업활동현금흐름": "영업활동현금흐름",
}


class DartError(RuntimeError):
    """DART API 호출이 실패했습니다."""


class DartKeyMissingError(DartError):
    """DART API 키가 설정되지 않았습니다."""


@dataclass(frozen=True)
class CorpCode:
    corp_code: str
    corp_name: str
    stock_code: str


def parse_amount(value: object) -> int | None:
    """DART 금액 문자열을 정수로. 빈 값이나 '-' 는 None 입니다."""
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text or text in {"-", "--"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_account(name: str) -> str | None:
    """DART 계정명을 표준 이름으로. 모르는 계정은 None 입니다."""
    return ACCOUNT_ALIASES.get((name or "").strip())


def _require_key(settings: Settings) -> str:
    if not settings.dart_api_key:
        raise DartKeyMissingError(
            "DART_API_KEY 가 설정되지 않았습니다.\n"
            "https://opendart.fss.or.kr 에서 무료 발급 후 .env 에 넣어 주십시오."
        )
    return settings.dart_api_key


def fetch_corp_codes(*, settings: Settings | None = None) -> list[CorpCode]:
    """상장사 고유번호 목록을 받습니다.

    DART 는 종목코드가 아니라 자체 corp_code 를 씁니다. 전체 목록이 ZIP 안의
    XML 로 오므로 풀어서 상장사(stock_code 가 있는 것)만 추립니다.
    """
    settings = settings or get_settings()
    key = _require_key(settings)

    response = requests.get(CORP_CODE_URL, params={"crtfc_key": key}, timeout=120)
    response.raise_for_status()
    if response.content[:2] != b"PK":
        raise DartError(f"ZIP 응답이 아닙니다: {response.text[:200]}")

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    xml_text = archive.read(archive.namelist()[0]).decode("utf-8")
    root = ElementTree.fromstring(xml_text)

    result: list[CorpCode] = []
    for node in root.findall("list"):
        stock_code = (node.findtext("stock_code") or "").strip()
        if not stock_code:
            continue
        result.append(
            CorpCode(
                corp_code=(node.findtext("corp_code") or "").strip(),
                corp_name=(node.findtext("corp_name") or "").strip(),
                stock_code=stock_code,
            )
        )
    return result


def fetch_financials(
    corp_code: str,
    bsns_year: int | str,
    reprt_code: str,
    *,
    settings: Settings | None = None,
    pause: float = 0.2,
) -> list[dict]:
    """단일회사 주요계정을 받습니다.

    응답에 연결(CFS)과 별도(OFS)가 함께 들어 있어 한 번만 호출하면 됩니다.

    Returns:
        정규화된 행 목록. 데이터가 없으면 빈 목록입니다.
        (DART 는 데이터 없음을 status 013 으로 알려 주며, 이는 오류가 아닙니다.)
    """
    settings = settings or get_settings()
    key = _require_key(settings)

    response = requests.get(
        SINGLE_ACCOUNT_URL,
        params={
            "crtfc_key": key,
            "corp_code": corp_code,
            "bsns_year": str(bsns_year),
            "reprt_code": reprt_code,
        },
        timeout=60,
    )
    if pause:
        time.sleep(pause)
    response.raise_for_status()
    payload = response.json()

    status = payload.get("status")
    if status == "013":  # 조회된 데이터가 없습니다
        return []
    if status != "000":
        raise DartError(f"DART 오류 {status}: {payload.get('message')}")

    meta = REPORT_CODES[reprt_code]
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for item in payload.get("list", []):
        account = normalize_account(item.get("account_nm", ""))
        if account is None:
            continue
        fs_div = (item.get("fs_div") or "").strip()
        # 같은 계정이 두 번 오는 경우가 있습니다(당기순이익 등). 첫 줄만 씁니다.
        key_pair = (fs_div, account)
        if key_pair in seen:
            continue
        amount = parse_amount(item.get("thstrm_amount"))
        if amount is None:
            continue
        seen.add(key_pair)

        period = (item.get("thstrm_dt") or "").strip()
        start, _, end = period.partition("~")
        rows.append(
            {
                "corp_code": corp_code,
                "stock_code": (item.get("stock_code") or "").strip(),
                "bsns_year": int(bsns_year),
                "reprt_code": reprt_code,
                "period_label": meta["label"],
                "quarter": meta["quarter"],
                "period_months": meta["months"],
                "fs_div": fs_div,
                "fs_nm": (item.get("fs_nm") or "").strip(),
                "sj_div": (item.get("sj_div") or "").strip(),
                "account": account,
                "account_nm_raw": (item.get("account_nm") or "").strip(),
                "amount": amount,
                "period_start": start.strip(),
                "period_end": end.strip(),
                "currency": (item.get("currency") or "").strip(),
                "rcept_no": (item.get("rcept_no") or "").strip(),
            }
        )
    return rows
