"""에이전트가 쓰는 도구와 데이터를 한 군데에 모읍니다.

왜 묶는가: 노드마다 스냅샷을 따로 읽으면 주장 하나 판정할 때마다 parquet 를
다시 읽게 됩니다. 53건이면 53번입니다. 한 번 읽어서 들고 다녀야 합니다.

왜 지연 로딩인가: NLI 모델은 올라오는 데 시간이 걸리고 메모리도 씁니다.
주가 주장만 판정하는 실행에서는 아예 필요 없습니다. 쓰는 순간에 올립니다.

왜 프로퍼티가 아니라 메서드인가 (중요)
--------------------------------------
처음에 `@property` 로 만들었더니 **그래프를 컴파일하는 것만으로 NLI 모델이
올라왔습니다.** 원인은 LangGraph 쪽에 있습니다.

    StateGraph.compile()
      -> PregelNode.__init__
        -> find_subgraph_pregel()
          -> get_function_nonlocals()
            -> getattr(value, instruction.argval, None)

LangGraph 는 노드 안에 중첩 그래프가 있는지 찾으려고 노드 함수의 바이트코드를
훑으면서, 클로저가 참조하는 속성 이름을 **실제로 getattr 합니다.** 노드 본문에
`tools.nli` 라고 적혀 있기만 하면 컴파일 시점에 그 프로퍼티가 평가됩니다.
지연 로딩이 통째로 무력화됩니다.

메서드로 두면 `getattr(tools, "nli")` 는 **바인딩된 메서드 객체**를 돌려줄 뿐
아무것도 실행하지 않습니다. 그래서 전부 메서드로 바꿨습니다.
호출하는 쪽이 `tools.nli()` 처럼 괄호를 붙여야 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings


@dataclass
class VerifierTools:
    """스냅샷·파서·모델을 담는 그릇. 전부 지연 로딩입니다.

    주의: 접근자는 전부 **메서드**입니다(`tools.prices()`). 이유는 모듈 설명 참고.
    """

    settings: Settings = field(default_factory=get_settings)

    _prices: Any = field(default=None, repr=False)
    _financials: Any = field(default=None, repr=False)
    _documents: Any = field(default=None, repr=False)
    _nli: Any = field(default=None, repr=False)
    _price_parser: Any = field(default=None, repr=False)
    _numeric_parser: Any = field(default=None, repr=False)
    _segment_parser: Any = field(default=None, repr=False)

    _segments: Any = field(default=None, repr=False)
    _documents_tried: bool = field(default=False, repr=False)
    _segments_tried: bool = field(default=False, repr=False)

    # -- 데이터 ------------------------------------------------------------

    def prices(self):
        """주가 스냅샷."""
        if self._prices is None:
            from ..tools.price import PriceSnapshot

            self._prices = PriceSnapshot.load()
        return self._prices

    def financials(self):
        """DART 재무 스냅샷."""
        if self._financials is None:
            from ..tools.financials import FinancialSnapshot

            self._financials = FinancialSnapshot.load()
        return self._financials

    def segments(self):
        """부문별 매출 스냅샷. 없으면 None(수집 전이어도 나머지는 돌아가야 합니다)."""
        if not self._segments_tried:
            self._segments_tried = True
            try:
                from ..tools.segments import SegmentSnapshot

                self._segments = SegmentSnapshot.load()
            except FileNotFoundError:
                self._segments = None
        return self._segments

    def documents(self):
        """공시 본문. 없으면 None 입니다(수집 전이어도 나머지는 돌아가야 합니다)."""
        if not self._documents_tried:
            self._documents_tried = True
            try:
                from ..tools.dart_documents import DocumentStore

                self._documents = DocumentStore.load()
            except FileNotFoundError:
                self._documents = None
        return self._documents

    # -- 모델 --------------------------------------------------------------

    def nli(self):
        """한국어 NLI 모델. 의미 비교 경로에서만 필요합니다."""
        if self._nli is None:
            from ..tools.nli import NliModel

            self._nli = NliModel(settings=self.settings).load()
        return self._nli

    def price_parser(self):
        """주가 주장 파서 (gpt-4o-mini)."""
        if self._price_parser is None:
            from ..tools.price_claim import build_price_parser

            self._price_parser = build_price_parser(settings=self.settings)
        return self._price_parser

    def numeric_parser(self):
        """재무 주장 파서 (gpt-4o-mini)."""
        if self._numeric_parser is None:
            from ..tools.numeric_claim import build_numeric_parser

            self._numeric_parser = build_numeric_parser(settings=self.settings)
        return self._numeric_parser

    def segment_parser(self):
        """부문 주장 파서 (gpt-4o-mini).

        재무 경로가 "주요계정에 없습니다" 로 끝났을 때만 호출합니다.
        """
        if self._segment_parser is None:
            from ..tools.segment_claim import build_segment_parser

            self._segment_parser = build_segment_parser(settings=self.settings)
        return self._segment_parser

    # -- 점검 --------------------------------------------------------------

    def loaded(self) -> dict[str, bool]:
        """무엇이 실제로 올라왔는지. 지연 로딩이 깨졌는지 확인할 때 씁니다."""
        return {
            "prices": self._prices is not None,
            "financials": self._financials is not None,
            "documents": self._documents is not None,
            "segments": self._segments is not None,
            "nli": self._nli is not None,
            "price_parser": self._price_parser is not None,
            "numeric_parser": self._numeric_parser is not None,
            "segment_parser": self._segment_parser is not None,
        }
