import unittest
from unittest.mock import patch

from pipeline.search import query_suggester
from pipeline.search.grounded_event_questions import grounded_event_question_items


def event(event_id, action, start):
    return {
        "id": event_id,
        "action": action,
        "start_time": start,
        "end_time": start + 1,
    }


class GroundedEventQuestionTests(unittest.TestCase):
    def test_observed_actions_keep_their_own_event_and_time(self):
        events = [
            event(1, "반려동물이 소파 쪽으로 이동함", 7),
            event(2, "반려동물이 문 쪽으로 이동함", 9),
            event(3, "반려동물이 소파 쪽으로 이동함", 12),
            event(4, "반려동물이 뒤를 바라보며 서 있음", 0),
        ]
        items = grounded_event_question_items(events, 3)
        self.assertEqual([item["question"] for item in items], [
            "소파 쪽으로 이동한 장면이 있어?",
            "문 쪽으로 이동한 장면이 있어?",
            "뒤를 바라보며 서 있는 장면이 있어?",
        ])
        self.assertEqual([item["source_event_id"] for item in items], [1, 2, 4])
        self.assertEqual([item["event_start"] for item in items], [7, 9, 0])

    def test_only_explicit_actions_are_promoted(self):
        events = [
            event(1, "반려동물이 나무에 매달린 물체를 향해 앞발을 올림", 2),
            event(2, "반려동물이 나무 위에서 앞쪽을 바라보며 몸을 움직임", 11),
            event(3, "반려동물이 나뭇가지를 향해 고개를 들어 올림", 21),
            event(4, "반려동물이 나무 위에서 몸을 움직이며 나가려 함", 17),
        ]
        questions = [item["question"] for item in grounded_event_question_items(events, 4)]
        self.assertEqual(questions, [
            "매달린 물체를 향해 앞발을 올린 장면이 있어?",
            "나무 위에서 몸을 움직인 장면이 있어?",
            "나뭇가지를 향해 고개를 든 장면이 있어?",
        ])

    def test_unlinked_or_uncertain_event_is_not_recommended(self):
        uncertain = event(1, "목줄이 떨어진 것으로 보임", 2)
        unlinked = event(None, "장난감을 바라봄", 3)
        self.assertEqual(grounded_event_question_items([uncertain, unlinked], 3), [])

    def test_existing_events_fill_suggestions_without_a_model_call(self):
        events = [
            event(191, "목줄이 벽에서 떨어져 아래로 늘어짐", 2),
            event(192, "반려동물이 장난감을 바라봄", 0),
            event(193, "반려동물이 장난감을 향해 몸을 숙임", 27),
        ]
        with patch.object(query_suggester, "get_top_behavior_events", return_value=events), \
             patch.object(query_suggester, "generate_question_items_from_behavior_events") as model:
            items = query_suggester.suggest_query_items(video_id="test", limit=3)
        model.assert_not_called()
        self.assertEqual([item["source_event_id"] for item in items], [191, 192, 193])
        self.assertEqual([item["question"] for item in items], [
            "목줄이 떨어진 장면이 있어?",
            "장난감을 바라본 장면이 있어?",
            "장난감을 향해 몸을 숙인 장면이 있어?",
        ])


if __name__ == "__main__":
    unittest.main()
