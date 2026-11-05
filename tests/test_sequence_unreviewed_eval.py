"""검수 JSON이 적용되지 않는 '새 영상'에서 Sequence Search 동작을 평가한다.

- 검색 알고리즘·질문 분류 코드는 건드리지 않고 현재 동작만 측정한다.
- 모든 이벤트는 **합성 데이터**다. DB·GCS·VLM·원본 영상에 접근하지 않았으므로
  이 결과는 "이벤트가 주어졌을 때 검색 계층이 올바른가"만 말해 주며, 실제 영상 정확도가 아니다.
- 현재 동작이 기대와 다른 케이스는 `known_gap`에 원인 계층을 적어 `expectedFailure`로 등록한다.
  수정이 들어가 통과하면 '예상 밖 성공'으로 실패 처리되어 목록을 갱신하게 만든다.

실행:  python -m unittest tests.test_sequence_unreviewed_eval -v
표 출력: python -m tests.test_sequence_unreviewed_eval
"""

import sys
import unittest
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from pipeline.search.sequence_query import (
    _reviewed_events,
    is_sequence_query,
    run_sequence_query,
)

NEW_VIDEO = "new_home_cam_001"

# 실패 원인 계층
EXTRACTION = "Event Extraction"
ANCHOR = "Anchor Retrieval"
TEMPORAL = "Temporal Reasoning"
EVIDENCE = "Evidence/Playback"


def ev(event_id, chunk, second, text, end=None, frames=True, video=NEW_VIDEO, day="20261105"):
    """새 추출기가 만드는 형태: 이벤트 구간 안의 대표 프레임 1장."""
    event = {
        "id": event_id,
        "video_id": video,
        "start_time": second,
        "end_time": end if end is not None else second + 4,
        "summary": text,
        "confidence": 0.8,
        "source_frames": [],
    }
    if frames:
        event["source_frames"] = [{
            "frame_id": f"f{event_id}",
            "s3_key": f"frames/{video}/petcam_{day}_090000_{chunk:03d}_frame_{second}.00.jpg",
            "timestamp": float(second),
        }]
    return event


def base_events():
    return [
        ev(1, 0, 3, "반려동물이 소파 위에서 낮잠을 자고 있는 모습이 관찰됨."),
        ev(2, 1, 5, "반려동물이 물그릇에서 물을 마시는 모습을 보임."),
        ev(3, 2, 4, "반려동물이 물을 마시며 갈증을 해소하는 모습을 보임."),
        ev(4, 3, 6, "반려동물이 현관문 앞에서 짖는 모습이 보임."),
        ev(5, 4, 2, "반려동물이 사료 그릇에서 사료를 먹고 있는 모습입니다."),
        ev(6, 5, 8, "반려동물이 마당을 돌아다니며 탐색하는 모습이 관찰됨."),
    ]


def shuffled_events():
    events = base_events()
    return [events[i] for i in (4, 1, 5, 0, 3, 2)]


def drink_variant_events():
    """둘째 마심 이벤트에 '물'이 없어 앵커는 하나로 정해지지만, 같은 행동으로 묶이는 경우."""
    return [
        ev(1, 0, 3, "반려동물이 소파 위에서 낮잠을 자고 있는 모습이 관찰됨."),
        ev(2, 1, 5, "반려동물이 물그릇에서 물을 마시는 모습을 보임."),
        ev(3, 2, 4, "반려동물이 수분을 섭취하며 갈증을 해소하는 모습을 보임."),
        ev(4, 3, 6, "반려동물이 현관문 앞에서 짖는 모습이 보임."),
    ]


def repeated_far_events():
    return [
        ev(1, 0, 3, "반려동물이 물을 마시는 모습이 보임."),
        ev(2, 1, 3, "반려동물이 소파에서 쉬는 모습이 보임."),
        ev(3, 40, 3, "반려동물이 물을 마시는 모습이 보임."),
        ev(4, 41, 3, "반려동물이 현관에서 짖는 모습이 보임."),
    ]


def door_events():
    return [ev(1, 0, 2, "문 앞에 서 있음"), ev(2, 1, 2, "마당에서 걸음")]


def conjugation_events():
    return [
        ev(1, 0, 3, "반려동물이 거실을 뛰어다니는 모습이 보임."),
        ev(2, 1, 3, "반려동물이 하품하는 모습이 보임."),
    ]


def legacy_frameless_tail_events():
    return base_events() + [ev(7, 5, 9, "반려동물이 복도에서 움직임", frames=False)]


def over_limit_events():
    return [ev(i, i, 3, f"행동 {i}번 장면") for i in range(1, 601)]


def twelve_step_events():
    return [ev(i, i * 4, 3, f"반려동물이 장소{i}에서 서성임") for i in range(1, 13)]


def ids(result):
    return [item["source_event_id"] for item in result["results"]]


@dataclass
class Case:
    case_id: str
    qtype: str  # FULL / FIRST / LAST / BEFORE / AFTER
    query: str
    expect: str  # 사람이 읽는 기대 동작
    check: Callable[[Any], bool]
    events: Callable[[], list] = base_events
    video_id: str | None = NEW_VIDEO
    top_k: int = 10
    kind: str = "sequence"  # sequence | classify
    known_gap: tuple[str, str] | None = None  # (원인 계층, 한 줄 원인)
    kwargs: dict = field(default_factory=dict)

    def run(self):
        if self.kind == "classify":
            return is_sequence_query(self.query)
        return run_sequence_query(
            self.query, video_id=self.video_id, top_k=self.top_k, events=self.events(), **self.kwargs
        )


def playable_and_consistent(result):
    """모든 근거 카드가 재생 가능한지. 앞/뒤 질문은 기준 장면 카드를 함께 담는다."""
    cards = result["evidence_items"]
    if not result["results"] or any(card not in cards for card in result["results"]):
        return False
    if result["sequence_intent"] not in {"before", "after"} and cards != result["results"]:
        return False
    for item in cards:
        ts = item["timestamp"]
        if not item["s3_key"]:
            return False
        if not (item["event_start"] <= ts <= item["event_end"]):
            return False
        if item["playback_start"] != max(ts - 3, 0) or item["playback_end"] != ts + 5:
            return False
        if ts not in [frame["timestamp"] for frame in item["step_frames"]]:
            return False
    return True


ALL = [1, 2, 4, 5, 6]  # 2·3은 같은 '물 마심'으로 묶여 한 단계

CASES = [
    # ---------------- FULL ----------------
    Case("FULL-1", "FULL", "강아지가 어떤 순서로 행동했어?", "5단계, 물 마심 2건은 1단계로 병합",
         lambda r: ids(r) == ALL and [i["merged_count"] for i in r["results"]] == [1, 2, 1, 1, 1]),
    Case("FULL-2", "FULL", "반려동물 행동을 시간순으로 알려줘", "5단계 시간순",
         lambda r: ids(r) == ALL),
    Case("FULL-3", "FULL", "행동 순서 알려줘", "DB 순서가 뒤섞여도 시간순",
         lambda r: ids(r) == ALL, events=shuffled_events),
    Case("FULL-4", "FULL", "오늘 전체 행동 순서 알려줘", "전체 타임라인 5단계",
         lambda r: ids(r) == ALL),
    Case("FULL-5", "FULL", "하루 전체 흐름을 알려줘", "전체 타임라인 5단계",
         lambda r: ids(r) == ALL),
    Case("FULL-6", "FULL", "오후에 한 행동 순서 알려줘", "전체 타임라인(시간대 필터는 별도 인자)",
         lambda r: ids(r) == ALL),
    Case("FULL-7", "FULL", "행동 순서 알려줘", "12단계 중 10단계 + 잘림 고지",
         lambda r: len(r["results"]) == 10 and r["total_step_count"] == 12 and "처음 10개만" in r["answer"],
         events=twelve_step_events),
    Case("FULL-8", "FULL", "행동 순서 알려줘", "모든 카드가 재생 가능한 근거를 가짐",
         playable_and_consistent),
    # ---------------- FIRST ----------------
    Case("FIRST-1", "FIRST", "처음에 뭐했어?", "[1]", lambda r: ids(r) == [1]),
    Case("FIRST-2", "FIRST", "제일 먼저 한 행동은?", "[1]", lambda r: ids(r) == [1]),
    Case("FIRST-3", "FIRST", "처음에 뭐했어?", "DB 순서가 뒤섞여도 가장 이른 [1]",
         lambda r: ids(r) == [1], events=shuffled_events),
    Case("FIRST-4", "FIRST", "먼저 뭐했어?", "첫 행동 [1]만",
         lambda r: ids(r) == [1]),
    Case("FIRST-5", "FIRST", "맨 처음 물 마신 건 언제야?", "물 마신 첫 이벤트 [2]",
         lambda r: ids(r) == [2]),
    Case("FIRST-6", "FIRST", "처음에 뭐했어?", "재생 가능한 근거", playable_and_consistent),
    # ---------------- LAST ----------------
    Case("LAST-1", "LAST", "마지막에는 뭐했어?", "[6]", lambda r: ids(r) == [6]),
    Case("LAST-2", "LAST", "맨 마지막 행동은?", "[6]", lambda r: ids(r) == [6]),
    Case("LAST-3", "LAST", "마지막으로 한 행동 알려줘", "분류 통과(sequence)",
         lambda ok: ok is True, kind="classify"),
    Case("LAST-4", "LAST", "마지막으로 물 마신 건 언제야?", "물 마신 마지막 이벤트 [3]",
         lambda r: ids(r) == [3]),
    Case("LAST-5", "LAST", "끝에는 뭐했어?", "마지막 행동 [6]만",
         lambda r: ids(r) == [6]),
    Case("LAST-6", "LAST", "마지막에는 뭐했어?", "재생 가능한 근거를 가진 [6] (프레임 없는 레거시 행 제외)",
         lambda r: ids(r) == [6] and bool(r["results"][0]["s3_key"]),
         events=legacy_frameless_tail_events),
    Case("LAST-7", "LAST", "마지막에는 뭐했어?", "600건 중 실제 마지막 [600]",
         lambda r: ids(r) == [600], events=over_limit_events),
    Case("LAST-8", "LAST", "마지막에는 뭐했어?", "재생 가능한 근거", playable_and_consistent),
    # ---------------- BEFORE ----------------
    Case("BEFORE-1", "BEFORE", "현관문 앞에서 짖기 전에 뭐했어?", "직전 에피소드(물 마심 2건)의 대표 [2]",
         lambda r: ids(r) == [2] and r["results"][0]["merged_count"] == 2),
    Case("BEFORE-2", "BEFORE", "사료를 먹기 전에 뭐했어?", "[4]", lambda r: ids(r) == [4]),
    Case("BEFORE-3", "BEFORE", "소파에서 낮잠 자기 전에 뭐했어?", "앞선 장면 없음을 정직하게 답함",
         lambda r: r["results"] == [] and "앞선 장면이 없어요" in r["answer"]),
    Case("BEFORE-4", "BEFORE", "물 마시기 전에 뭐했어?", "묶인 물 마심 단계의 직전 [1]",
         lambda r: ids(r) == [1]),
    Case("BEFORE-5", "BEFORE", "사과 먹기 전에 뭐했어?", "없는 대상이므로 기준 못 찾음(결과 없음)",
         lambda r: r["results"] == []),
    Case("BEFORE-6", "BEFORE", "물 마시기 전에 뭐했어?", "먼 시각에 반복된 행동은 모호함을 고지(fail-closed)",
         lambda r: r["results"] == [] and "좁히지" in r["answer"], events=repeated_far_events),
    Case("BEFORE-7", "BEFORE", "현관문 앞에서 짖기 전에 뭐했어?", "재생 가능한 근거", playable_and_consistent),
    # ---------------- AFTER ----------------
    Case("AFTER-1", "AFTER", "현관문 앞에서 짖은 다음 뭐했어?", "[5]", lambda r: ids(r) == [5]),
    Case("AFTER-2", "AFTER", "사료 먹은 뒤에 뭐했어?", "[6]", lambda r: ids(r) == [6]),
    Case("AFTER-3", "AFTER", "마당 탐색 다음에는?", "이어지는 장면 없음을 정직하게 답함",
         lambda r: r["results"] == [] and "이어지는 장면이 없어요" in r["answer"]),
    Case("AFTER-4", "AFTER", "낮잠 잔 다음에 뭐했어?", "명사 '낮잠'으로 앵커 성공 [2]",
         lambda r: ids(r) == [2]),
    Case("AFTER-5", "AFTER", "물 마신 다음에 뭐했어?", "묶인 물 마심 단계의 다음 [4]",
         lambda r: ids(r) == [4]),
    Case("AFTER-6", "AFTER", "짖은 다음에 뭐했어?", "'은'이 조사로 제거돼 '짖'으로 우연히 성공 [5]",
         lambda r: ids(r) == [5]),
    Case("AFTER-6b", "AFTER", "뛰어다닌 다음에 뭐했어?", "[2]",
         lambda r: ids(r) == [2], events=conjugation_events),
    Case("AFTER-7", "AFTER", "물그릇에서 물 마신 다음 뭐했어?", "같은 행동 다음 단계 [4]",
         lambda r: ids(r) == [4], events=drink_variant_events),
    Case("AFTER-8", "AFTER", "물 마신 다음에 뭐했어?", "영상 미선택 시 영상 선택 요청, 근거 카드는 비어 있음",
         lambda r: r["results"] == [] and r["behavior_events"] == [] and "영상" in r["answer"], video_id=None),
    Case("AFTER-9", "AFTER", "문을 나간 다음 뭐했어?", "'나감'이 확인되지 않으면 이웃을 지어내지 않음",
         lambda r: r["results"] == [] and "행동까지는 보이지 않아요" in r["answer"], events=door_events),
    Case("AFTER-10", "AFTER", "현관문 앞에서 짖은 다음 뭐했어?", "답변이 인용한 기준 장면(짖음 [4])도 근거 카드로 반환",
         lambda r: 4 in [e["id"] for e in r["behavior_events"]]),
    Case("AFTER-11", "AFTER", "사료를 먹은 후에는 어땠어?", "분류 통과(sequence)",
         lambda ok: ok is True, kind="classify"),
    Case("AFTER-12", "AFTER", "사료를 먹은 뒤 어땠어?", "분류 통과(sequence)",
         lambda ok: ok is True, kind="classify"),
    Case("AFTER-13", "AFTER", "현관문 앞에서 짖은 다음 뭐했어?", "재생 가능한 근거", playable_and_consistent),
]


class UnreviewedPathTests(unittest.TestCase):
    """수기 검수 JSON이 새 영상에는 적용되지 않는지."""

    def test_fixture_video_is_not_in_reviewed_json(self):
        self.assertNotIn(NEW_VIDEO, _reviewed_events())

    def test_review_filter_does_not_leak_into_new_video_even_with_colliding_event_ids(self):
        reviewed_ids = {int(i) for video in _reviewed_events().values() for i in video}
        self.assertTrue(reviewed_ids)
        # 검수 JSON에 있는 이벤트 ID와 같은 번호를 새 영상 이벤트에 일부러 부여한다.
        events = [
            ev(min(reviewed_ids), 0, 3, "원문 그대로의 설명A 모습이 보임."),
            ev(max(reviewed_ids) + 1000, 1, 3, "원문 그대로의 설명B 모습이 보임."),
        ]
        result = run_sequence_query("행동 순서 알려줘", video_id=NEW_VIDEO, events=events)
        self.assertEqual(result["excluded_event_count"], 0)
        self.assertEqual(len(result["behavior_events"]), 2)
        self.assertTrue(all("review_status" not in e for e in result["behavior_events"]))
        self.assertIn("원문 그대로의 설명A", result["answer"])
        self.assertNotIn("이어 붙인", result["answer"])
        # 대표 프레임은 검수 timestamp가 아니라 저장된 프레임을 따른다.
        self.assertEqual(result["results"][0]["timestamp"], 3.0)


class CallGraphTests(unittest.TestCase):
    """rag_chain → sequence_query 호출 계약."""

    @classmethod
    def setUpClass(cls):
        from pipeline.search import rag_chain
        cls.rag = rag_chain

    def test_sequence_question_is_delegated_with_expanded_top_k_and_filters(self):
        with patch.object(self.rag, "run_sequence_query", return_value={"answer": "x"}) as seq:
            self.rag.run_rag_query(
                "행동 순서 알려줘", video_id=NEW_VIDEO, top_k=3,
                recording_date="2026-11-05", time_range={"start_hour": 6, "end_hour": 12},
            )
        seq.assert_called_once_with(
            query="행동 순서 알려줘", video_id=NEW_VIDEO, top_k=10,
            recording_date="2026-11-05", time_range={"start_hour": 6, "end_hour": 12},
        )

    def test_all_five_types_reach_sequence_path_and_skip_frame_search(self):
        fake_vector_store = MagicMock()
        queries = {
            "FULL": "행동 순서 알려줘", "FIRST": "처음에 뭐했어?", "LAST": "마지막에는 뭐했어?",
            "BEFORE": "사료를 먹기 전에 뭐했어?", "AFTER": "현관문 앞에서 짖은 다음 뭐했어?",
        }
        for qtype, query in queries.items():
            with self.subTest(qtype=qtype), \
                    patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=base_events()), \
                    patch.dict(sys.modules, {"pipeline.storage.vector_store": fake_vector_store}):
                result = self.rag.run_rag_query(query, video_id=NEW_VIDEO, top_k=3)
            self.assertEqual(result["query_type"], "sequence")
            self.assertEqual(result["sequence_intent"], {
                "FULL": "timeline", "FIRST": "first", "LAST": "last", "BEFORE": "before", "AFTER": "after",
            }[qtype])
        fake_vector_store.search_with_query_expansion.assert_not_called()

    def test_router_sequence_type_is_not_consulted_by_rag_chain(self):
        """rag_chain은 question_router의 AGGREGATE만 사용하고, 시퀀스 분기는 is_sequence_query가 독자적으로 결정한다."""
        with open(self.rag.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("QuestionType.AGGREGATE", source)
        self.assertNotIn("QuestionType.SEQUENCE", source)


class IntentParsingRegressionTests(unittest.TestCase):
    """'전/후/뒤'가 다른 낱말 안에 섞인 표현은 BEFORE/AFTER가 아니고, 자연스러운 표현은 처리된다."""

    def intent(self, query):
        return run_sequence_query(query, video_id=NEW_VIDEO, events=base_events())["sequence_intent"]

    def test_words_that_merely_contain_before_or_after_syllables_stay_timeline(self):
        for query in (
            "오늘 전체 행동 순서 알려줘", "하루 전체 흐름을 알려줘", "오후에 한 행동 순서 알려줘",
            "오전에 뭐했어?", "뒤섞여서 뭐했어?", "뒤뜰에서 뭐했어?", "전부 순서대로 알려줘",
            "안전하게 있었던 행동 순서", "처음부터 끝까지 흐름 알려줘", "처음과 마지막에 뭐했어?",
            "먼저 한 행동부터 순서대로 알려줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.intent(query), "timeline")

    def test_natural_markers_are_recognised(self):
        cases = {
            "사료를 먹은 후에는 어땠어?": "after", "사료를 먹은 뒤 어땠어?": "after", "짖은 이후엔 뭐했어?": "after",
            "물 마시고 나서 뭐했어?": "after", "물 마시고나서 뭐했어?": "after", "짖은 그 다음엔 뭐했어?": "after",
            "사료 먹기 전에 뭐했어?": "before", "사료 먹기전에 뭐했어?": "before", "짖기 직전엔 뭐했어?": "before",
            "짖기 이전에 뭐했어?": "before",
            "먼저 뭐 했어?": "first", "처음에는 어땠어?": "first", "제일 먼저 한 행동은?": "first",
            "끝에는 뭐했어?": "last", "맨 끝에는 어땠어?": "last", "마지막에 한 행동은?": "last",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.intent(query), expected)
                self.assertTrue(is_sequence_query(query))

    def test_classifier_does_not_hijack_unrelated_questions(self):
        for query in ("오전에 어디 있었어?", "책상 끝에 있어?", "전체 화면으로 보여줘", "물을 몇 번 마시고 나서 쉬었어?"):
            with self.subTest(query=query):
                self.assertFalse(is_sequence_query(query))


class FirstLastConditionTests(unittest.TestCase):
    """첫/마지막 질문의 행동 조건은 그 행동의 이벤트 안에서만 고르고, 없으면 엉뚱한 이벤트를 내지 않는다."""

    def ask(self, query, events=base_events):
        return run_sequence_query(query, video_id=NEW_VIDEO, events=events())

    def test_condition_picks_first_or_last_matching_event(self):
        self.assertEqual(ids(self.ask("마지막으로 물을 마신 건 언제야?")), [3])
        self.assertEqual(ids(self.ask("맨 처음 물 마신 건 언제야?")), [2])
        self.assertEqual(ids(self.ask("마지막으로 사료 먹은 건?")), [5])
        self.assertEqual(ids(self.ask("제일 먼저 짖은 건 언제야?")), [4])

    def test_question_filler_alone_is_not_a_condition(self):
        for query, expected in (
            ("마지막으로 한 행동 알려줘", [6]), ("마지막 행동이 뭐야?", [6]), ("마지막에는 뭐했어?", [6]),
            ("가장 처음 뭘 했어?", [1]), ("오늘 제일 먼저 한 일은?", [1]), ("맨 처음 행동은 뭐였어?", [1]),
        ):
            with self.subTest(query=query):
                self.assertEqual(ids(self.ask(query)), expected)

    def test_missing_action_returns_nothing_instead_of_an_unrelated_event(self):
        for query in ("마지막으로 수영한 건 언제야?", "처음으로 간식 먹은 건 언제야?"):
            with self.subTest(query=query):
                result = self.ask(query)
                self.assertEqual(result["results"], [])
                self.assertEqual(result["behavior_events"], [])
                self.assertIn("찾지 못했어요", result["answer"])

    def test_object_must_match_and_predicate_is_matched_by_stem(self):
        events = base_events() + [ev(7, 6, 3, "반려동물이 거실을 뛰어다니는 모습이 보임.")]
        # 대상('간식')이 없으면 '먹'만 겹치는 사료 이벤트를 내지 않는다.
        self.assertEqual(ids(run_sequence_query("마지막으로 간식 먹은 건?", video_id=NEW_VIDEO, events=events)), [])
        # 활용형('뛰어다닌' ↔ '뛰어다니는')은 어간으로 맞춘다.
        self.assertEqual(ids(run_sequence_query("마지막으로 뛰어다닌 건 언제야?", video_id=NEW_VIDEO, events=events)), [7])

    def test_place_match_without_the_asked_action_is_not_confirmed(self):
        # '물'이 '물그릇'의 일부라는 사실만으로 마셨다고 확정하지 않는다.
        events = [ev(1, 0, 3, "반려동물이 물그릇 앞에 서 있는 모습이 보임.")]
        result = run_sequence_query("마지막으로 물 마신 건 언제야?", video_id=NEW_VIDEO, events=events)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["evidence_items"], [])

    def test_nearby_place_does_not_confirm_the_asked_action(self):
        result = self.ask("마지막으로 문을 나간 건 언제야?", events=door_events)
        self.assertEqual(result["results"], [])
        self.assertIn("질문하신 행동까지는 보이지 않아요", result["answer"])

    def test_missing_action_on_reviewed_video_mentions_excluded_events_and_stays_isolated(self):
        approved = ev(59, 5, 2, "원본 설명", video="funny_moments", day="20261014")
        rejected = ev(60, 6, 2, "물이 튀는 장치와 상호작용", video="funny_moments", day="20261014")
        result = run_sequence_query("마지막으로 수영한 건 언제야?", video_id="funny_moments", events=[rejected, approved])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["excluded_event_count"], 1)
        self.assertIn("제외된 구간", result["answer"])
        # 새 영상의 같은 번호 이벤트는 검수 필터의 영향을 받지 않는다.
        fresh = run_sequence_query("마지막에는 뭐했어?", video_id=NEW_VIDEO, events=[ev(60, 6, 2, "새 영상 설명")])
        self.assertEqual((ids(fresh), fresh["excluded_event_count"]), ([60], 0))


class ChronologyTests(unittest.TestCase):
    """start_time은 60초 청크 안의 상대 시간이다. DB 순서가 아니라 청크 번호로 복원한 시각으로 정렬한다."""

    def test_db_order_by_chunk_relative_start_time_does_not_decide_the_last_event(self):
        # DB는 (video_id, start_time) 순으로 돌려준다 → 청크가 다른 이벤트가 start_time끼리 섞여 온다.
        events = [ev(i, i, 1 + (i % 50) if i < 600 else 58, f"행동 {i}번 장면") for i in range(1, 601)]
        db_order = sorted(events, key=lambda e: (e["video_id"], e["start_time"], e["end_time"], e["id"]))
        # 예전처럼 DB 순서 앞 500건만 읽으면 가장 늦은 이벤트(600)가 빠진다.
        self.assertNotIn(600, [e["id"] for e in db_order[:500]])
        result = run_sequence_query("마지막에는 뭐했어?", video_id=NEW_VIDEO, events=db_order)
        self.assertEqual(ids(result), [600])
        first = run_sequence_query("처음에 뭐했어?", video_id=NEW_VIDEO, events=db_order)
        self.assertEqual(ids(first), [1])

    def test_full_timeline_counts_every_event_beyond_500_and_is_not_marked_truncated(self):
        result = run_sequence_query("행동 순서 알려줘", video_id=NEW_VIDEO, events=over_limit_events(), top_k=10)
        self.assertEqual(result["total_event_count"], 600)
        self.assertFalse(result["truncated"])

    def test_relative_questions_see_events_beyond_500(self):
        events = over_limit_events()
        result = run_sequence_query("행동 599번 장면 다음에 뭐했어?", video_id=NEW_VIDEO, events=events)
        self.assertEqual(ids(result), [600])

    def test_default_db_path_reads_every_event_without_slicing(self):
        events = over_limit_events()
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=events):
            result = run_sequence_query("마지막에는 뭐했어?", video_id=NEW_VIDEO)
        self.assertEqual(ids(result), [600])


class AnchorMatchingTests(unittest.TestCase):
    """앞/뒤 질문의 기준 장면 매칭: FIRST/LAST와 같은 규칙(주어 제거, 어간, 조사, 부분 문자열 금지)."""

    def ask(self, query, events):
        return run_sequence_query(query, video_id=NEW_VIDEO, events=events)

    def test_subject_word_does_not_count_as_drinking(self):
        # '반려동물'의 '물'이 모든 이벤트에 맞아 동점이 되던 문제
        result = self.ask("물 마신 다음에 뭐했어?", base_events())
        self.assertEqual(ids(result), [4])

    def test_real_noun_with_particle_like_ending_is_kept(self):
        events = [ev(1, 0, 3, "반려동물이 사과를 먹는 모습이 보임."), ev(2, 1, 3, "반려동물이 하품하는 모습이 보임."),
                  ev(3, 2, 3, "반려동물이 사료를 먹는 모습이 보임.")]
        # '사과'는 '사'로 깎이지 않아 사과 이벤트를 정확히 찾는다.
        self.assertEqual(ids(self.ask("사과 먹은 다음 뭐했어?", events)), [2])
        # 없는 대상('사과')이 '사료'에 오매칭되지 않는다.
        self.assertEqual(self.ask("사과 먹은 다음 뭐했어?", events[1:])["results"], [])

    def test_object_must_match_so_a_shared_verb_is_not_enough(self):
        result = self.ask("간식 먹은 다음에 뭐했어?", base_events())
        self.assertEqual(result["results"], [])
        self.assertEqual(result["evidence_items"], [])
        self.assertIn("좁히지", result["answer"])

    def test_inflected_verb_matches_by_stem(self):
        self.assertEqual(ids(self.ask("뛰어다닌 다음에 뭐했어?", conjugation_events())), [2])
        self.assertEqual(ids(self.ask("하품하기 전에 뭐했어?", conjugation_events())), [1])  # -기 ↔ -는
        self.assertEqual(ids(self.ask("물 마시기 전에 뭐했어?", base_events())), [1])

    def test_single_syllable_term_needs_a_standalone_word(self):
        events = [ev(1, 0, 3, "반려동물이 물그릇 앞에 서 있는 모습이 보임."), ev(2, 1, 3, "반려동물이 마당에서 걷는 모습이 보임.")]
        result = self.ask("물 마신 다음에 뭐했어?", events)
        self.assertEqual(result["results"], [])


class EpisodeConsistencyTests(unittest.TestCase):
    """FULL과 앞/뒤 질문은 같은 에피소드 단위를 쓴다."""

    def test_every_after_answer_is_the_next_full_step(self):
        full = run_sequence_query("행동 순서 알려줘", video_id=NEW_VIDEO, events=base_events())
        steps = ids(full)
        anchors = {1: "낮잠 잔 다음에", 2: "물그릇에서 물 마신 다음에", 4: "현관문 앞에서 짖은 다음에", 5: "사료 먹은 다음에"}
        for index, step in enumerate(steps[:-1]):
            with self.subTest(step=step):
                result = run_sequence_query(f"{anchors[step]} 뭐했어?", video_id=NEW_VIDEO, events=base_events())
                self.assertEqual(ids(result), [steps[index + 1]])

    def test_before_answer_is_the_previous_full_step_and_carries_episode_size(self):
        result = run_sequence_query("현관문 앞에서 짖기 전에 뭐했어?", video_id=NEW_VIDEO, events=base_events())
        self.assertEqual(ids(result), [2])
        self.assertEqual(result["results"][0]["merged_count"], 2)
        self.assertEqual([e["id"] for e in result["behavior_events"]], [2, 3, 4])

    def test_different_actions_and_distant_repeats_are_not_merged(self):
        # 먹고 → 바로 물 마심: 다른 행동이므로 별개 에피소드
        events = [ev(1, 0, 3, "반려동물이 사료를 먹는 모습이 보임."), ev(2, 1, 3, "반려동물이 물을 마시는 모습이 보임."),
                  ev(3, 2, 3, "반려동물이 하품하는 모습이 보임.")]
        result = run_sequence_query("사료 먹은 다음에 뭐했어?", video_id=NEW_VIDEO, events=events)
        self.assertEqual((ids(result), result["results"][0]["merged_count"]), ([2], 1))
        # 같은 행동이어도 120초보다 멀면 별개 에피소드(→ 앵커가 모호)
        far = run_sequence_query("물 마신 다음에 뭐했어?", video_id=NEW_VIDEO, events=repeated_far_events())
        self.assertEqual(far["results"], [])
        self.assertIn("좁히지", far["answer"])
        # 불확실한 표현('마시려고')은 같은 행동으로 병합하지 않는다.
        trying = [ev(1, 0, 3, "반려동물이 물을 마시는 모습이 보임."), ev(2, 1, 3, "반려동물이 물을 마시려고 다가가는 모습이 보임."),
                  ev(3, 2, 3, "반려동물이 하품하는 모습이 보임.")]
        steps = run_sequence_query("행동 순서 알려줘", video_id=NEW_VIDEO, events=trying)
        self.assertEqual([c["merged_count"] for c in steps["results"]], [1, 1, 1])


class EvidenceGroundingTests(unittest.TestCase):
    def ask(self, query, events=base_events, video_id=NEW_VIDEO):
        return run_sequence_query(query, video_id=video_id, events=events())

    def test_before_and_after_return_anchor_and_answer_cards_in_time_order(self):
        after = self.ask("현관문 앞에서 짖은 다음 뭐했어?")
        self.assertEqual([c["source_event_id"] for c in after["evidence_items"]], [4, 5])
        self.assertEqual([c["evidence_role"] for c in after["evidence_items"]], ["anchor", "answer"])
        self.assertEqual(ids(after), [5])
        before = self.ask("현관문 앞에서 짖기 전에 뭐했어?")
        self.assertEqual([c["evidence_role"] for c in before["evidence_items"]], ["answer", "anchor"])
        self.assertEqual([c["source_event_id"] for c in before["evidence_items"]], [2, 4])

    def test_anchor_card_is_returned_even_when_nothing_follows(self):
        result = self.ask("마당 탐색 다음에는?")
        self.assertEqual(result["results"], [])
        self.assertEqual([c["source_event_id"] for c in result["evidence_items"]], [6])

    def test_no_evidence_when_anchor_is_ambiguous_or_unsupported(self):
        ambiguous = self.ask("문 다음 뭐했어?", events=lambda: [ev(1, 0, 2, "문 앞에 있음"), ev(2, 1, 2, "문을 통과함")])
        unsupported = self.ask("문을 나간 다음 뭐했어?", events=door_events)
        for result in (ambiguous, unsupported):
            self.assertEqual((result["results"], result["evidence_items"], result["behavior_events"]), ([], [], []))

    def test_without_a_selected_video_no_timeline_cards_come_back(self):
        for query in ("물 마신 다음에 뭐했어?", "현관문 앞에서 짖기 전에 뭐했어?"):
            result = self.ask(query, video_id=None)
            self.assertEqual((result["results"], result["evidence_items"], result["behavior_events"]), ([], [], []))
            self.assertIn("영상", result["answer"])

    def test_frameless_legacy_events_are_never_first_last_or_neighbors(self):
        legacy_head = [ev(0, 0, 1, "반려동물이 복도에서 움직임", frames=False)]
        events = lambda: legacy_head + base_events() + [ev(7, 5, 9, "반려동물이 복도에서 움직임", frames=False)]
        last = self.ask("마지막에는 뭐했어?", events=events)
        first = self.ask("처음에 뭐했어?", events=events)
        after = self.ask("마당 탐색 다음에는?", events=events)
        self.assertEqual((ids(last), ids(first), ids(after)), ([6], [1], []))
        self.assertTrue(all(c["s3_key"] for r in (last, first, after) for c in r["evidence_items"]))
        self.assertIn("재생할 수 있는 근거가 없는 장면 2개는 제외", last["answer"])

    def test_timeline_still_lists_frameless_events_and_all_frameless_is_reported(self):
        only_legacy = lambda: [ev(7, 5, 9, "반려동물이 복도에서 움직임", frames=False)]
        self.assertEqual(self.ask("행동 순서 알려줘", events=only_legacy)["total_step_count"], 1)
        result = self.ask("마지막에는 뭐했어?", events=only_legacy)
        self.assertEqual(result["results"], [])
        self.assertIn("재생할 수 있는 근거", result["answer"])


def _make_test(case: Case):
    def test(self):
        result = case.run()
        detail = result if case.kind == "classify" else result["answer"]
        self.assertTrue(case.check(result), f"[{case.case_id}] {case.query!r} 기대: {case.expect} / 실제: {detail}")
    test.__doc__ = f"{case.qtype}: {case.query} → {case.expect}"
    return unittest.expectedFailure(test) if case.known_gap else test


class SequenceEvalMatrix(unittest.TestCase):
    """FULL/FIRST/LAST/BEFORE/AFTER 케이스 표. known_gap 케이스는 expectedFailure."""


for _case in CASES:
    setattr(SequenceEvalMatrix, f"test_{_case.case_id.replace('-', '_').lower()}", _make_test(_case))


def report() -> None:
    by_type = defaultdict(Counter)
    by_layer = Counter()
    rows = []
    for case in CASES:
        passed = bool(case.check(case.run()))
        status = "PASS" if passed else "GAP " if case.known_gap else "FAIL"
        if case.known_gap and passed:
            status = "FIXED?"
        by_type[case.qtype]["pass" if passed else "fail"] += 1
        if not passed and case.known_gap:
            by_layer[case.known_gap[0]] += 1
        rows.append((case.case_id, status, case.query, case.known_gap[0] if case.known_gap and not passed else "", case.known_gap[1] if case.known_gap and not passed else ""))
    for row in rows:
        print(" | ".join(row))
    print()
    for qtype in ("FULL", "FIRST", "LAST", "BEFORE", "AFTER"):
        c = by_type[qtype]
        print(f"{qtype:7} pass {c['pass']:>2} / fail {c['fail']:>2}")
    print()
    for layer in (EXTRACTION, ANCHOR, TEMPORAL, EVIDENCE):
        print(f"{layer:20} {by_layer[layer]}")


if __name__ == "__main__":
    if "--report" in sys.argv or len(sys.argv) == 1:
        report()
    else:
        unittest.main()
