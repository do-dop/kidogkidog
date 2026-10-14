import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from pipeline.search.jev_client import JevClient, JevClientError


class QuestionType(str, Enum):
    STATIC = "static"
    EVENT = "event"
    SEQUENCE = "sequence"
    AGGREGATE = "aggregate"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RouteDecision:
    question_type: QuestionType
    confidence: float
    requires_multiple_frames: float
    requires_event_order: float
    requires_aggregation: float
    requires_time_filter: float
    source: str
    model_version: str | None = None
    latency_ms: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["question_type"] = self.question_type.value
        return result


AGGREGATE_PATTERNS = (
    r"몇\s*번",
    r"얼마나\s*(?:자주|많이|오래)",
    r"총\s*(?:몇|얼마)",
    r"가장\s*(?:많이|오래|자주)",
    r"평소보다",
    r"횟수",
    r"빈도",
)

SEQUENCE_PATTERNS = (
    r"(?:은|한)\s*뒤",
    r"하기\s*전",
    r"하고\s*나서",
    r"먼저",
    r"나중에",
    r"다가",
    r"이후",
    r"이전에",
)

EVENT_PATTERNS = (
    r"먹",
    r"마시|마셨",
    r"뛰",
    r"놀",
    r"이동",
    r"들어(?:갔|오)",
    r"나갔",
    r"접근",
    r"반복",
)

TIME_FILTER_PATTERNS = (
    r"오늘|어제|내일",
    r"오전|오후|저녁|밤|새벽|낮",
    r"\d{1,2}\s*시",
    r"\d{4}[-./년]\s*\d{1,2}",
)


def get_router_mode() -> str:
    mode = os.getenv("QUESTION_ROUTER_MODE", "off").strip().lower()
    return mode if mode in {"off", "shadow"} else "off"


def classify_with_rules(query: str) -> RouteDecision:
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    requires_time_filter = 1.0 if _matches_any(normalized, TIME_FILTER_PATTERNS) else 0.0

    if _matches_any(normalized, AGGREGATE_PATTERNS):
        return RouteDecision(
            question_type=QuestionType.AGGREGATE,
            confidence=0.95,
            requires_multiple_frames=1.0,
            requires_event_order=0.2,
            requires_aggregation=1.0,
            requires_time_filter=requires_time_filter,
            source="rule",
        )

    if _matches_any(normalized, SEQUENCE_PATTERNS):
        return RouteDecision(
            question_type=QuestionType.SEQUENCE,
            confidence=0.95,
            requires_multiple_frames=1.0,
            requires_event_order=1.0,
            requires_aggregation=0.0,
            requires_time_filter=requires_time_filter,
            source="rule",
        )

    if _matches_any(normalized, EVENT_PATTERNS):
        return RouteDecision(
            question_type=QuestionType.EVENT,
            confidence=0.75,
            requires_multiple_frames=0.8,
            requires_event_order=0.0,
            requires_aggregation=0.0,
            requires_time_filter=requires_time_filter,
            source="rule",
        )

    return RouteDecision(
        question_type=QuestionType.STATIC,
        confidence=0.55,
        requires_multiple_frames=0.2,
        requires_event_order=0.0,
        requires_aggregation=0.0,
        requires_time_filter=requires_time_filter,
        source="rule",
    )


def classify_question(query: str, client: JevClient | None = None) -> RouteDecision:
    rule_decision = classify_with_rules(query)
    client = client or JevClient()

    if not client.configured:
        return RouteDecision(
            **{
                **rule_decision.to_dict(),
                "question_type": rule_decision.question_type,
                "source": "rule_no_jev_key",
            }
        )

    started_at = time.perf_counter()
    try:
        response = client.classify_question(query)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        jev_decision = _parse_jev_response(response, latency_ms=latency_ms)
    except (JevClientError, KeyError, TypeError, ValueError) as exc:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return RouteDecision(
            **{
                **rule_decision.to_dict(),
                "question_type": rule_decision.question_type,
                "source": "rule_jev_fallback",
                "latency_ms": latency_ms,
                "error": str(exc),
            }
        )

    return _combine_decisions(rule_decision, jev_decision)


def run_shadow_classification(query: str, client: JevClient | None = None) -> RouteDecision | None:
    if get_router_mode() != "shadow":
        return None

    decision = classify_question(query, client=client)
    log_question_route(query, decision)
    return decision


def log_question_route(query: str, decision: RouteDecision) -> None:
    payload = {
        "event": "question_route_shadow",
        "query_hash": hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:12],
        "query_length": len(query),
        **decision.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _parse_jev_response(response: dict[str, Any], latency_ms: int) -> RouteDecision:
    answers = response["answers"]
    scope = answers["temporal_scope"]
    choice = str(scope["choice"]).lower()
    probabilities = scope.get("probabilities") or {}

    try:
        question_type = QuestionType(choice)
    except ValueError as exc:
        raise ValueError(f"지원하지 않는 Jev temporal_scope입니다: {choice}") from exc

    confidence = _probability(scope.get("confidence", probabilities.get(choice, 0.0)))

    return RouteDecision(
        question_type=question_type,
        confidence=confidence,
        requires_multiple_frames=_noul(answers, "requires_multiple_frames"),
        requires_event_order=_noul(answers, "requires_event_order"),
        requires_aggregation=_noul(answers, "requires_aggregation"),
        requires_time_filter=_noul(answers, "requires_time_filter"),
        source="jev",
        model_version=response.get("model"),
        latency_ms=latency_ms,
    )


def _combine_decisions(rule: RouteDecision, jev: RouteDecision) -> RouteDecision:
    high_threshold = float(os.getenv("QUESTION_ROUTER_CONFIDENCE_THRESHOLD", "0.80"))
    fallback_threshold = float(os.getenv("QUESTION_ROUTER_FALLBACK_THRESHOLD", "0.50"))

    if rule.confidence >= 0.9 and rule.question_type == jev.question_type:
        return RouteDecision(
            **{
                **jev.to_dict(),
                "question_type": jev.question_type,
                "confidence": max(rule.confidence, jev.confidence),
                "source": "rule+jev",
            }
        )

    if jev.confidence >= high_threshold:
        return jev

    if jev.confidence >= fallback_threshold:
        return RouteDecision(
            **{
                **jev.to_dict(),
                "question_type": jev.question_type,
                "source": "jev_with_fallback",
            }
        )

    return RouteDecision(
        question_type=QuestionType.HYBRID,
        confidence=jev.confidence,
        requires_multiple_frames=jev.requires_multiple_frames,
        requires_event_order=jev.requires_event_order,
        requires_aggregation=jev.requires_aggregation,
        requires_time_filter=jev.requires_time_filter,
        source="low_confidence_fallback",
        model_version=jev.model_version,
        latency_ms=jev.latency_ms,
    )


def _matches_any(query: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, query) for pattern in patterns)


def _noul(answers: dict[str, Any], key: str) -> float:
    return _probability(answers[key]["noul"])


def _probability(value: Any) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"확률 값이 범위를 벗어났습니다: {probability}")
    return probability
