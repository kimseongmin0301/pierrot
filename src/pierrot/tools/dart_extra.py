"""DART 에서 주요계정 바깥의 수치를 가져옵니다.

왜 필요한가
-----------
3단계 실측에서 확인 불가 54건을 원인별로 가른 결과, 가장 큰 덩어리가
**"재무제표에 없는 계정" 13건**이었습니다. 배당금과 지배주주 순이익이 그 안에 있습니다.

    "2023년 현대차의 총수익에서 배당금 지급액은 4조 6천억 원이었다"
    "KB금융의 2024년 지배주주 순이익은 1조 원 규모이다"

지금 쓰는 `fnlttSinglAcnt`(단일회사 주요계정)는 계정이 **14개**뿐입니다.
DART 에는 두 가지 API 가 더 있고, 직접 호출해 확인했습니다.

    fnlttSinglAcntAll  전체 재무제표   계정 121개 (삼성전자 2024 기준)
    alotMatter         배당에 관한 사항  현금배당금총액 / 주당 현금배당금 등

왜 새 경로를 만들지 않는가
--------------------------
`numeric_claim.KNOWN_ACCOUNTS` 는 `dart.ACCOUNT_ALIASES` 의 값에서 자동으로
만들어지고, 파서 프롬프트도 그 목록을 그대로 씁니다. 그래서 **별칭에 계정을
더하고 스냅샷에 행을 더하면** 파서·판정기가 그대로 확장됩니다.
새 노드도 새 파서도 필요 없습니다. 부문별 매출(`segments`)은 표 구조가 달라
따로 만들었지만, 이쪽은 같은 모양의 수치입니다.

비중·수익률 같은 **비율 항목은 담지 않습니다.** 지금 판정기는 금액(원)과
증감률(%)만 다룹니다. 배당성향을 금액 자리에 넣으면 단위가 섞입니다.
비율 주장은 별도 작업이고, 그 전까지는 확인 불가로 두는 쪽이 맞습니다.
"""

from __future__ import annotations

import time

import requests

from ..config import Settings, get_settings
from .dart import BASE_URL, REPORT_CODES, DartError, _require_key

SINGLE_ACCOUNT_ALL_URL = f"{BASE_URL}/fnlttSinglAcntAll.json"
DIVIDEND_URL = f"{BASE_URL}/alotMatter.json"

# 전체 재무제표에서 추가로 가져올 계정.
#
# 주요계정 14개에 없으면서 실제 답변에 나온 것들입니다. 전부 가져오면
# 121개가 되는데, 파서 프롬프트에 121개를 늘어놓으면 분류가 나빠집니다.
# 실측에서 요구된 것만 더합니다.
#
# **이름이 아니라 XBRL account_id 로 집습니다.**
# 계정명으로 집었다가 틀렸습니다. 현대차 2023 전체 재무제표에는
# "지배기업소유주지분" 이라는 **같은 이름이 세 군데**에 있습니다.
#
#     sj=BS   ifrs-full_EquityAttributableToOwnersOfParent        92조 (자본)
#     sj=IS   ifrs-full_ProfitLossAttributableToOwnersOfParent    11.9조 (순이익)
#     sj=CIS  ifrs-full_ComprehensiveIncome...OwnersOfParent      12.2조 (포괄손익)
#
# 이름으로 집으면 먼저 나온 자본 92조를 "지배주주 순이익" 으로 기록합니다.
# account_id 는 국제회계기준 태그라 회사·연도가 달라도 같습니다.
EXTRA_ACCOUNT_IDS: dict[str, str] = {
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "지배주주순이익",
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "영업활동현금흐름",
}

# 배당 공시에서 가져올 항목. 금액으로 환산할 수 있는 것만 담습니다.
#
# `se`(구분) 표기가 보고서마다 조금씩 다릅니다. 공백을 지운 뒤 맞춥니다.
DIVIDEND_ACCOUNTS: dict[str, tuple[str, int]] = {
    # 표기 -> (계정 이름, 배수)
    "현금배당금총액(백만원)": ("배당금총액", 1_000_000),
    "주당현금배당금(원)": ("주당배당금", 1),
    # 주당순이익(EPS). 20종목을 훑어보니 표기가 모두 같았습니다.
    # 연결 기준만 공시되며 음수도 그대로 들어옵니다(LG에너지솔루션 2024년 -4,354원).
    "(연결)주당순이익(원)": ("주당순이익", 1),
}


def _get(url: str, params: dict, *, pause: float = 0.2) -> dict:
    response = requests.get(url, params=params, timeout=60)
    if pause:
        time.sleep(pause)
    response.raise_for_status()
    return response.json()


def _amount(text: str) -> int | None:
    """DART 금액 문자열을 정수로. 못 읽으면 None."""
    value = (text or "").strip().replace(",", "")
    if not value or value in {"-", "0"} and value != "0":
        return None
    negative = value.startswith("-") or (value.startswith("(") and value.endswith(")"))
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    number = int(digits)
    return -number if negative else number


def fetch_extra_accounts(
    corp_code: str,
    bsns_year: int | str,
    reprt_code: str,
    *,
    code: str,
    settings: Settings | None = None,
    pause: float = 0.2,
) -> list[dict]:
    """전체 재무제표에서 추가 계정만 골라 옵니다.

    주요계정 API 와 같은 모양의 행을 돌려줍니다. 그래야 같은 스냅샷에
    그대로 이어 붙일 수 있습니다.
    """
    settings = settings or get_settings()
    key = _require_key(settings)
    meta = REPORT_CODES[str(reprt_code)]

    rows: list[dict] = []
    for fs_div in ("CFS", "OFS"):
        payload = _get(
            SINGLE_ACCOUNT_ALL_URL,
            {
                "crtfc_key": key,
                "corp_code": corp_code,
                "bsns_year": str(bsns_year),
                "reprt_code": str(reprt_code),
                "fs_div": fs_div,
            },
            pause=pause,
        )
        status = payload.get("status")
        if status == "013":
            continue
        if status != "000":
            raise DartError(f"DART 오류 {status}: {payload.get('message')}")

        seen: set[str] = set()
        for item in payload.get("list", []):
            account_id = (item.get("account_id") or "").strip()
            account = EXTRA_ACCOUNT_IDS.get(account_id)
            if account is None or account in seen:
                continue
            amount = _amount(item.get("thstrm_amount", ""))
            if amount is None:
                continue
            seen.add(account)
            rows.append(
                {
                    "code": code,
                    "corp_code": corp_code,
                    "bsns_year": int(bsns_year),
                    "reprt_code": str(reprt_code),
                    "period_label": meta["label"],
                    "quarter": meta["quarter"],
                    "period_months": meta["months"],
                    "fs_div": fs_div,
                    "fs_nm": "연결재무제표" if fs_div == "CFS" else "재무제표",
                    "account": account,
                    "amount": amount,
                    "period_start": "",
                    "period_end": "",
                    "rcept_no": str(item.get("rcept_no", "")),
                }
            )
    return rows


def fetch_dividends(
    corp_code: str,
    bsns_year: int | str,
    reprt_code: str,
    *,
    code: str,
    settings: Settings | None = None,
    pause: float = 0.2,
) -> list[dict]:
    """배당에 관한 사항에서 금액 항목만 가져옵니다.

    응답은 `se`(구분) 가 행, `thstrm`(당기) 이 값인 모양입니다.
    주식 종류가 여럿이면 같은 구분이 여러 번 나오므로 **첫 행만** 씁니다
    (보통 보통주입니다).
    """
    settings = settings or get_settings()
    key = _require_key(settings)
    meta = REPORT_CODES[str(reprt_code)]

    payload = _get(
        DIVIDEND_URL,
        {
            "crtfc_key": key,
            "corp_code": corp_code,
            "bsns_year": str(bsns_year),
            "reprt_code": str(reprt_code),
        },
        pause=pause,
    )
    status = payload.get("status")
    if status == "013":
        return []
    if status != "000":
        raise DartError(f"DART 오류 {status}: {payload.get('message')}")

    rows: list[dict] = []
    seen: set[str] = set()
    for item in payload.get("list", []):
        label = (item.get("se") or "").replace(" ", "")
        mapping = DIVIDEND_ACCOUNTS.get(label)
        if mapping is None:
            continue
        account, scale = mapping
        if account in seen:
            continue
        amount = _amount(item.get("thstrm", ""))
        if amount is None:
            continue
        seen.add(account)
        rows.append(
            {
                "code": code,
                "corp_code": corp_code,
                "bsns_year": int(bsns_year),
                "reprt_code": str(reprt_code),
                "period_label": meta["label"],
                "quarter": meta["quarter"],
                "period_months": meta["months"],
                # 배당은 연결·별도 구분이 없습니다. 회사 단위 수치입니다.
                # 판정기가 두 기준을 모두 보므로 CFS 로 넣어 둡니다.
                "fs_div": "CFS",
                "fs_nm": "배당에 관한 사항",
                "account": account,
                "amount": amount * scale,
                "period_start": "",
                "period_end": "",
                "rcept_no": str(item.get("rcept_no", "")),
            }
        )
    return rows
