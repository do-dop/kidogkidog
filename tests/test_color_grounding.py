import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.search import color_grounding
from pipeline.search import rag_chain


class ColorGroundingTests(unittest.TestCase):
    def test_color_words_trigger_only_color_questions(self):
        self.assertTrue(color_grounding.asks_about_color("파란 통에 접근했어?"))
        self.assertTrue(color_grounding.asks_about_color("blue bowl visible?"))
        self.assertFalse(color_grounding.asks_about_color("통에 접근했어?"))
        self.assertEqual(color_grounding.without_color_words("파란 통에 접근했어?"), "통에 접근했어?")
        self.assertTrue(color_grounding.asks_about_colored_action("파란 통으로 접근한 적 있냐?"))

    def test_recorded_red_target_cannot_answer_blue_target(self):
        event = {"target_object": "빨간 통", "action": "반려동물이 빨간 통에 접근함"}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            color_grounding, "_image_data", side_effect=AssertionError("must not fetch image")
        ):
            self.assertFalse(color_grounding.verify_result_color(
                "파란 통에 접근했어?", {"s3_key": "frame.jpg"}, event,
            ))

    def test_visual_verifier_requires_true_boolean_and_uses_event_frames(self):
        event = {"action": "반려동물이 통에 접근함", "source_frames": [
            {"s3_key": "first.jpg"}, {"s3_key": "last.jpg"},
        ]}
        result = {"s3_key": "representative.jpg"}
        response = SimpleNamespace(output_text=json.dumps({"match": True}))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            color_grounding, "_image_data", side_effect=lambda key: f"data:{key}"
        ) as images, patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = response
            self.assertTrue(color_grounding.verify_result_color("파란 통에 접근했어?", result, event))
            images.assert_any_call("first.jpg")
            images.assert_any_call("last.jpg")
            self.assertEqual(client.return_value.responses.create.call_count, 1)

    def test_unverified_color_results_and_original_answer_are_removed(self):
        payload = {
            "answer": "빨간 통에 접근한 장면입니다",
            "results": [{"source_event_id": 1, "s3_key": "red.jpg", "timestamp": 2}],
            "behavior_events": [{"id": 1, "action": "빨간 통에 접근함"}],
            "aggregate_result": {"count": 1},
        }
        with patch.object(color_grounding, "verify_result_color", return_value=False):
            result = color_grounding.verify_color_results("파란 통에 접근했어?", payload, 5)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["behavior_events"], [])
        self.assertNotIn("aggregate_result", result)
        self.assertNotIn("빨간", result["answer"])

    def test_movement_needs_stored_action_before_visual_color_check(self):
        payload = {"results": [{"s3_key": "frame.jpg", "timestamp": 2}], "behavior_events": []}
        with patch.object(color_grounding, "verify_result_color") as verify, patch(
            "pipeline.storage.vector_store.get_indexed_frames", return_value=[]
        ):
            result = color_grounding.verify_color_results("파란 통에 접근했어?", payload, 5)
        verify.assert_not_called()
        self.assertEqual(result["results"], [])

    def test_nearby_frames_keep_the_same_chunk_and_temporal_order(self):
        result = {"s3_key": "frames/video/chunk_000_frame_2.jpg", "timestamp": 2}
        indexed = [
            {"s3_key": "frames/video/chunk_000_frame_3.jpg", "timestamp": 3},
            {"s3_key": "frames/video/chunk_001_frame_1.jpg", "timestamp": 1},
            {"s3_key": "frames/video/chunk_000_frame_1.jpg", "timestamp": 1},
            result,
        ]
        self.assertEqual(
            [frame["timestamp"] for frame in color_grounding._nearby_frames(result, indexed)],
            [1, 2, 3],
        )

    def test_color_query_verifies_wider_search_before_returning_results(self):
        payload = {"answer": "candidate", "results": [], "behavior_events": []}
        with patch.object(rag_chain, "_run_rag_query", return_value=payload) as search, patch.object(
            rag_chain, "include_explicit_color_events", return_value=payload
        ), patch.object(
            rag_chain, "verify_color_results", return_value={"answer": "verified"}
        ) as verify:
            result = rag_chain.run_rag_query("파란 통에 접근했어?", video_id="video", top_k=3)
        self.assertEqual(search.call_args.kwargs["top_k"], 12)
        verify.assert_called_once_with("파란 통에 접근했어?", payload, 3)
        self.assertEqual(result["answer"], "verified")

    def test_explicit_color_event_enters_shortlist_before_visual_check(self):
        event = {"id": 252, "video_id": "long", "start_time": 42, "end_time": 44,
                 "action": "반려동물이 빨간 물체에 다가감", "target_object": "빨간 물체",
                 "source_frames": [{"s3_key": "frame.jpg", "timestamp": 42}]}
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=[event]):
            payload = color_grounding.include_explicit_color_events(
                "빨간 물체에 다가간 장면 있어?", {"results": [], "behavior_events": []}, "long",
            )
        self.assertEqual(payload["results"][0]["source_event_id"], 252)

    def test_full_day_filter_keeps_color_event_but_restricted_time_does_not(self):
        event = {"id": 252, "video_id": "long", "start_time": 42, "end_time": 44,
                 "action": "반려동물이 빨간 물체에 다가감", "target_object": "빨간 물체",
                 "source_frames": [{"s3_key": "frames/long/chunk_frame_43.jpg", "timestamp": 43}]}
        empty = {"results": [], "behavior_events": []}
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=[event]):
            full_day = color_grounding.include_explicit_color_events(
                "빨간 통에 다가가는 장면", empty, "long",
                time_range={"start_hour": 0, "end_hour": 24},
            )
            restricted = color_grounding.include_explicit_color_events(
                "빨간 통에 다가가는 장면", empty, "long",
                time_range={"start_hour": 9, "end_hour": 10},
            )
        self.assertEqual([item["source_event_id"] for item in full_day["results"]], [252])
        self.assertEqual(restricted["results"], [])

    def test_movement_event_uses_neighbor_frames_for_color_check(self):
        event = {"id": 252, "action": "반려동물이 빨간 물체에 다가감",
                 "target_object": "빨간 물체", "source_frames": [
                     {"s3_key": "frames/long/chunk_frame_43.jpg", "timestamp": 43},
                 ]}
        result = {"source_event_id": 252, "video_id": "long", "s3_key": "frames/long/chunk_frame_43.jpg",
                  "timestamp": 43}
        indexed = [
            {"s3_key": "frames/long/chunk_frame_41.jpg", "timestamp": 41},
            result,
            {"s3_key": "frames/long/chunk_frame_45.jpg", "timestamp": 45},
        ]
        response = SimpleNamespace(output_text=json.dumps({"match": True}))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch(
            "pipeline.storage.vector_store.get_indexed_frames", return_value=indexed
        ), patch.object(color_grounding, "_image_data", side_effect=lambda key: f"data:{key}") as images, \
             patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = response
            verified = color_grounding.verify_color_results(
                "빨간 통에 다가가는 장면", {"results": [result], "behavior_events": [event]}, 3,
            )
        self.assertEqual(verified["match_level"], "verified_color")
        self.assertEqual([call.args[0] for call in images.call_args_list],
                         [item["s3_key"] for item in indexed])


if __name__ == "__main__":
    unittest.main()
