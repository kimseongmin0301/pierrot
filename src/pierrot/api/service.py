"""API 가 쓰는 작업 묶음.

여기서 하는 일은 셋입니다.

1. **무거운 것을 한 번만 올립니다.** 그래프, 스냅샷, NLI 모델, 파서.
   요청마다 올리면 주장 하나에 몇 초가 걸립니다.
2. **검증을 직렬화합니다.** 아래 설명 참고.
3. **요청별 비용과 지연을 잽니다.** 월 예산이 $5 인 프로젝트에서
   호출 하나가 얼마를 쓰는지 모르면 운영이 안 됩니다.

왜 직렬화하는가
---------------
FastAPI 는 동기 엔드포인트를 스레드풀에서 돌립니다. 그런데 이 시스템이
쓰는 것들은 스레드 안전하지 않습니다.

- NLI 모델(PyTorch) 추론을 동시에 부르면 결과가 섞일 수 있습니다.
- pandas 스냅샷은 읽기 전용이지만 조회 중 상태를 공유합니다.
- 무엇보다 **비용을 요청별로 가를 수 없습니다.** 비용은 월 누적액의
  차이로 재는데, 두 요청이 겹치면 서로의 비용을 가져갑니다.

동시에 처리하는 것보다 **얼마를 썼는지 정확히 아는 것**이 이 프로젝트에
더 중요합니다. 처리량이 필요해지면 그때 프로세스를 늘리는 편이 낫습니다.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..cost import CostLogger
from ..extractor import extract_claims
from ..reporter import Report, build_report
from ..verifier import VerifierTools, build_graph, verify_claim
from ..verifier.state import VerificationResult


@dataclass
class Usage:
    """요청 하나가 쓴 자원."""

    elapsed_seconds: float
    cost_usd: float
    month_spent_usd: float
    month_budget_usd: float
    llm_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "cost_usd": round(self.cost_usd, 6),
            "month_spent_usd": round(self.month_spent_usd, 6),
            "month_budget_usd": self.month_budget_usd,
            "llm_calls": self.llm_calls,
        }


class VerificationService:
    """검증 작업을 묶어 들고 있는 객체. 앱 수명 동안 하나만 둡니다."""

    def __init__(self, settings: Settings | None = None, *, graph: Any | None = None):
        self.settings = settings or get_settings()
        self.tools = VerifierTools(settings=self.settings)
        self.graph = graph if graph is not None else build_graph(self.tools)
        self.cost = CostLogger(
            log_path=self.settings.cost_log_path,
            monthly_budget_usd=self.settings.monthly_budget_usd,
        )
        # 검증 한 번에 하나씩. 이유는 모듈 설명 참고.
        self._lock = threading.Lock()
        self._extractor: Any | None = None
        self._decontextualizer: Any | None = None

    # -- 준비 상태 ---------------------------------------------------------

    def snapshot_status(self) -> dict[str, bool]:
        """스냅샷 파일이 있는지. 없으면 그 경로의 판정이 전부 확인 불가가 됩니다."""
        directory = Path(self.settings.snapshot_dir)
        return {
            "prices": (directory / "kospi_ohlcv.parquet").exists(),
            "financials": (directory / "dart_financials.parquet").exists(),
            "financials_extra": (directory / "dart_financials_extra.parquet").exists(),
            "segments": (directory / "dart_segments.parquet").exists(),
            "documents": (directory / "dart_sentences.parquet").exists(),
        }

    def health(self) -> dict[str, Any]:
        snapshots = self.snapshot_status()
        notes: list[str] = []
        for name, present in snapshots.items():
            if not present:
                notes.append(f"{name} 스냅샷이 없습니다. 해당 경로는 판정하지 못합니다.")
        has_key = bool(self.settings.openai_api_key)
        if not has_key:
            notes.append("OpenAI 키가 없어 주장 분해와 파싱을 할 수 없습니다.")

        remaining = self.cost.remaining_usd()
        if remaining <= 0:
            notes.append("이번 달 예산을 모두 썼습니다. 새 LLM 호출이 거부됩니다.")

        return {
            "status": "ok" if not notes else "degraded",
            "snapshots": snapshots,
            "openai_key": has_key,
            "budget_remaining_usd": round(remaining, 6),
            "notes": notes,
        }

    # -- 비용 측정 ---------------------------------------------------------

    def _log_lines(self) -> int:
        path = Path(self.settings.cost_log_path)
        if not path.exists():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _measure(self):
        """with 블록 동안의 비용·시간·호출 수를 재는 컨텍스트."""
        return _Measurement(self)

    def usage_by_source(self, limit: int = 5000) -> dict[str, dict[str, Any]]:
        """이번 달 비용을 단계별로 나눕니다.

        어느 단계가 돈을 쓰는지 알아야 줄일 곳을 정할 수 있습니다.
        """
        path = Path(self.settings.cost_log_path)
        if not path.exists():
            return {}
        month = self.cost.current_month()
        totals: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in list(handle)[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("month") != month:
                    continue
                source = row.get("source") or "(미상)"
                bucket = totals.setdefault(
                    source, {"calls": 0, "cache_hits": 0, "cost_usd": 0.0}
                )
                bucket["calls"] += 1
                bucket["cost_usd"] += row.get("cost_usd") or 0.0
                if row.get("cache_hit"):
                    bucket["cache_hits"] += 1
        for bucket in totals.values():
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        return dict(sorted(totals.items(), key=lambda item: -item[1]["cost_usd"]))

    # -- 검증 --------------------------------------------------------------

    def verify_answer(
        self,
        answer: str,
        *,
        allow_fallback: bool = False,
        refine: bool = True,
    ) -> tuple[Report, Usage, dict[str, Any]]:
        """답변 하나를 분해하고 주장마다 판정합니다."""
        with self._lock, self._measure() as measure:
            if self._extractor is None:
                from ..extractor.extractor import build_extractor

                self._extractor = build_extractor(settings=self.settings)
            if self._decontextualizer is None and refine:
                from ..extractor.decontextualize import build_decontextualizer

                self._decontextualizer = build_decontextualizer(settings=self.settings)

            extraction = extract_claims(
                answer,
                extractor=self._extractor,
                settings=self.settings,
                decontextualizer=self._decontextualizer,
                refine=refine,
            )
            results = [
                verify_claim(
                    claim.text,
                    claim_id=claim.claim_id,
                    declared_type=claim.claim_type.value
                    if hasattr(claim.claim_type, "value")
                    else str(claim.claim_type),
                    graph=self.graph,
                    allow_fallback=allow_fallback,
                )
                for claim in extraction.claims
            ]
            report = build_report(answer, extraction.claims, results)

        extractor_info = {
            "model": extraction.model,
            "prompt_version": extraction.prompt_version,
            "refined": extraction.refined,
            "refine_note": extraction.refine_note,
        }
        return report, measure.usage, extractor_info

    def verify_single_claim(
        self,
        claim: str,
        *,
        declared_type: str = "",
        allow_fallback: bool = False,
    ) -> tuple[VerificationResult, Usage]:
        """주장 문장 하나만 판정합니다. 분해 단계를 건너뜁니다."""
        with self._lock, self._measure() as measure:
            result = verify_claim(
                claim,
                claim_id="single",
                declared_type=declared_type,
                graph=self.graph,
                allow_fallback=allow_fallback,
            )
        return result, measure.usage


class _Measurement:
    """with 블록 동안 흐른 시간과 늘어난 비용을 잽니다.

    비용은 월 누적액의 **차이**로 잽니다. 이 방식이 맞으려면 블록이 겹치지
    않아야 하므로 호출하는 쪽에서 잠금을 잡습니다.
    """

    def __init__(self, service: VerificationService) -> None:
        self.service = service
        self.usage: Usage | None = None

    def __enter__(self) -> "_Measurement":
        self._started = time.perf_counter()
        self._cost_before = self.service.cost.month_total_usd()
        self._lines_before = self.service._log_lines()
        return self

    def __exit__(self, *exc_info) -> None:
        spent = self.service.cost.month_total_usd()
        self.usage = Usage(
            elapsed_seconds=time.perf_counter() - self._started,
            cost_usd=max(0.0, spent - self._cost_before),
            month_spent_usd=spent,
            month_budget_usd=self.service.settings.monthly_budget_usd,
            llm_calls=max(0, self.service._log_lines() - self._lines_before),
        )
