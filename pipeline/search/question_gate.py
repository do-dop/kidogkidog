"""질문이 반려동물 펫캠 영상 검색 서비스가 다루는 질문인지 판단한다.

CLIP 유사도는 "우주선이 날아가는 장면" 같은 무관한 질문에도 가장 가까운 프레임을 돌려주고, 점수로는 관련 있는
장면과 없는 장면을 가를 수 없다(무관한 질문 점수 0.200~0.227). 그래서 점수가 아니라 질문 내용으로 먼저 판단한다.

- 오탐(적합한 질문을 막는 것)이 더 나쁘다. 반려동물·행동 단어가 분명한 질문은 LLM을 거치지 않고 통과시키고,
  LLM도 "명백히 무관할 때만" 막도록 보수적으로 묻는다.
- LLM 호출이 실패하거나 키가 없으면 통과시킨다. 이 판단이 검색을 막으면 안 된다.
- QUESTION_GATE_MODE: off(기본, 아무것도 안 함) | shadow(판단만 하고 응답에 기록, 검색은 그대로) | on(부적합하면 검색하지 않음)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

# 반려동물·집·돌봄과 관련이 분명한 단어. 하나라도 있으면 LLM 없이 통과한다. 일부러 좁게 잡는다
# ("달리", "걷" 같은 일반 동작은 자동차·사람에도 쓰이므로 넣지 않는다).
DOMAIN_PATTERN = re.compile(
    r"강아지|고양이|반려|펫|냥이|멍멍|캣|(?<![가-힣])개(?:들)?(?![가-힣])|사료|밥그릇|물그릇|급식기|소파|침대|산책|목줄|장난감|"
    r"짖|꼬리|애완|코코|배변|간식|캣타워|켄넬|울타리|거실"
)

OUT_OF_SCOPE_ANSWER = (
    "반려동물 영상에서 찾을 수 있는 질문이 아니에요. "
    "'강아지가 밥 먹는 장면', '고양이가 소파에 올라간 장면'처럼 물어보세요."
)

_PROMPT = """너는 반려동물 펫캠 영상 검색 서비스의 질문 분류기다.

이 서비스는 집 안(거실, 침실, 마당 등)에 설치한 카메라로 찍은 반려동물(강아지, 고양이 등) 영상에서 장면을 찾아준다.
찾을 수 있는 것: 반려동물의 행동(먹기, 마시기, 걷기, 뛰기, 자기, 짖기, 놀기 등), 반려동물과 함께 나오는 사람·가구·물건,
집 안 물건이 쓰러지거나 움직이는 일, 전등이 꺼지는 것 같은 집 안 상황 변화.

규칙:
- 질문이 반려동물 영상에서 찾을 수 있는 일이면 fit=true.
- 애매하면 fit=true. 사람이 나오는 질문, 집 안 상황, 물건 질문도 fit=true.
- 반려동물 영상과 명백히 무관한 질문(우주선, 자동차 주행, 스포츠 경기, 결혼식, 요리 레시피, 코딩, 날씨, 일반 상식 등)만 fit=false.
- JSON 한 줄만 출력한다: {"fit": true 또는 false, "reason": "짧은 이유"}

질문: """

_CACHE_SIZE = 256
_cache: "OrderedDict[str, GateDecision]" = OrderedDict()


@dataclass(frozen=True)
class GateDecision:
    fit: bool
    source: str  # rules | llm | no_key | error | cache
    reason: str = ""
    latency_ms: int = 0
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_gate_mode() -> str:
    mode = os.getenv("QUESTION_GATE_MODE", "off").strip().lower()
    return mode if mode in {"off", "shadow", "on"} else "off"


def _model() -> str:
    return os.getenv("QUESTION_GATE_MODEL", "gpt-4o-mini")


def _parse(text: str) -> tuple[bool, str] | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    fit = data.get("fit")
    if not isinstance(fit, bool):
        return None
    return fit, str(data.get("reason") or "")[:120]


def _ask_llm(query: str) -> str:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=_model(), temperature=0, timeout=float(os.getenv("QUESTION_GATE_TIMEOUT", "6")), max_retries=0)
    return model.invoke(_PROMPT + query.strip()).content


def check_question_fit(query: str) -> GateDecision:
    """질문 적합성을 판단한다. 어떤 실패도 예외로 내보내지 않고 통과(fit=True)로 처리한다."""
    text = re.sub(r"\s+", " ", (query or "").strip())
    if not text or DOMAIN_PATTERN.search(text):
        return GateDecision(fit=True, source="rules", reason="반려동물·집 관련 단어")
    if text in _cache:
        _cache.move_to_end(text)
        cached = _cache[text]
        return GateDecision(cached.fit, "cache", cached.reason, 0, cached.model)
    if not os.getenv("OPENAI_API_KEY"):
        return GateDecision(fit=True, source="no_key", reason="OPENAI_API_KEY 없음")

    started = time.perf_counter()
    try:
        parsed = _parse(_ask_llm(text))
    except Exception as exc:
        return GateDecision(fit=True, source="error", reason=f"{type(exc).__name__}", latency_ms=int((time.perf_counter() - started) * 1000))
    latency = int((time.perf_counter() - started) * 1000)
    if parsed is None:
        return GateDecision(fit=True, source="error", reason="응답을 해석하지 못함", latency_ms=latency, model=_model())

    decision = GateDecision(fit=parsed[0], source="llm", reason=parsed[1], latency_ms=latency, model=_model())
    _cache[text] = decision
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return decision
