import unittest
from unittest.mock import patch

import numpy as np

from pipeline.search import grounded_action_search, species_verification, video_time


def event(event_id, chunk, start, end, action):
    return {
        "id": event_id,
        "video_id": "long",
        "start_time": start,
        "end_time": end,
        "action": action,
        "summary": action,
        "confidence": 0.8,
        "source_frames": [{
            "frame_id": f"long__petcam_20261119_171124_{chunk:03d}_frame_{start:.2f}",
            "s3_key": f"frames/long/petcam_20261119_171124_{chunk:03d}_frame_{start:.2f}.jpg",
            "timestamp": start,
        }],
    }


class LongVideoSearchFixes(unittest.TestCase):
    def test_imported_chunk_time_is_video_wide_but_camera_time_is_trusted(self):
        frame = event(1, 1, 30, 32, "젖을 먹음")["source_frames"][0]
        self.assertEqual(video_time.format_frame_time(frame, 30), "01:30")
        self.assertEqual(video_time.format_frame_time(frame, 32), "01:32")
        frame.update(capture_time_source="petcam", recorded_at="2026-11-05T07:35:30+09:00")
        self.assertEqual(video_time.format_frame_time(frame, 32), "07:35:32")

    def test_nursing_and_kibble_do_not_share_evidence(self):
        events = [
            event(1, 1, 29, 32, "새끼 강아지가 젖을 먹고 있음"),
            event(2, 1, 47, 49, "강아지가 그릇에서 사료를 먹고 있음"),
        ]
        def matching(query, candidates):
            return [row for row in candidates if ("젖" in row["action"] if "젖" in query else "사료" in row["action"])]
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events), \
             patch.object(grounded_action_search, "_select_matching_events", side_effect=matching):
            nursing = grounded_action_search.find_action_event_scenes(
                "새끼 강아지가 어미 젖을 먹는 장면은 언제야?", "long", 5)
            food = grounded_action_search.find_action_event_scenes(
                "강아지가 그릇에서 사료를 먹는 장면은 언제야?", "long", 5)
        self.assertEqual([row["source_event_id"] for row in nursing["results"]], [1])
        self.assertEqual([row["source_event_id"] for row in food["results"]], [2])
        self.assertIn("01:30", nursing["answer"])
        self.assertIn("01:48", food["answer"])

    def test_bathing_and_sleeping_return_their_own_chunk(self):
        events = [
            event(1, 4, 9, 12, "강아지가 욕조에서 목욕을 하고 있음"),
            event(2, 4, 42, 46, "강아지가 잠을 자고 있음"),
        ]
        def matching(query, candidates):
            return [row for row in candidates if ("목욕" in row["action"] if "목욕" in query else "잠" in row["action"])]
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events), \
             patch.object(grounded_action_search, "_select_matching_events", side_effect=matching):
            bath = grounded_action_search.find_action_event_scenes("강아지가 목욕하는 장면은 언제야?", "long", 5)
            sleep = grounded_action_search.find_action_event_scenes("강아지가 잠자는 장면은 언제야?", "long", 5)
        self.assertEqual(bath["results"][0]["source_event_id"], 1)
        self.assertEqual(sleep["results"][0]["source_event_id"], 2)
        self.assertIn("04:10", bath["answer"])
        self.assertIn("04:44", sleep["answer"])

    def test_unseen_action_can_be_grounded_without_adding_a_category(self):
        events = [
            event(1, 0, 45, 49, "사람이 강아지를 들어 올림"),
            event(2, 0, 52, 55, "강아지가 바닥에서 구름"),
        ]
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events), \
             patch.object(grounded_action_search, "_select_matching_events", return_value=[events[0]]):
            result = grounded_action_search.find_action_event_scenes("사람이 강아지를 들어 올리는 장면 보여줘", "long", 5)
        self.assertEqual([row["source_event_id"] for row in result["results"]], [1])
        self.assertIn("00:47", result["answer"])

    def test_species_and_direction_conflicts_are_not_presented_as_evidence(self):
        bath = event(1, 0, 10, 12, "반려동물이 욕조에서 목욕함")
        bath["source_frames"][0]["object_labels"] = ["dog"]
        leaving_screen = event(2, 0, 20, 22, "반려동물이 화면 밖으로 나감")
        leaving_screen["target_object"] = "문"
        events = [bath, leaving_screen]
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events), \
             patch.object(grounded_action_search, "_select_matching_events",
                          side_effect=lambda query, rows: [bath] if "목욕" in query else [leaving_screen]):
            cat = grounded_action_search.find_action_event_scenes("고양이가 목욕하는 장면 있어?", "long", 5)
            door = grounded_action_search.find_action_event_scenes("강아지가 문 밖으로 나가는 장면 있어?", "long", 5)
        self.assertEqual(cat["results"], [])
        self.assertEqual(door["results"], [])

    def test_core_action_with_unverified_detail_is_marked_related(self):
        nursing = event(1, 1, 29, 32, "반려동물이 다른 반려동물에게 젖을 먹고 있음")
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=[nursing]), \
             patch.object(grounded_action_search, "_select_matching_events",
                          return_value={"exact": [], "related": [nursing]}):
            result = grounded_action_search.find_action_event_scenes(
                "강아지가 어미 젖을 먹는 장면 보여줘", "long", 5)
        self.assertEqual(result["match_level"], "related_event")
        self.assertEqual(result["results"][0]["source_event_id"], 1)
        self.assertIn("세부 조건까지는 확인하지 못했어요", result["answer"])

    def test_default_ui_time_range_still_uses_action_index(self):
        shaking = event(1, 0, 11, 13, "강아지가 몸을 털음")
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=[shaking]), \
             patch("pipeline.search.action_event_index.semantic_event_candidates", return_value=[shaking]) as index, \
             patch.object(grounded_action_search, "_select_matching_events", return_value=[shaking]):
            result = grounded_action_search.find_action_event_scenes(
                "강아지가 몸을 터는 장면 보여줘", "long", 3,
                time_range={"start_hour": 0, "end_hour": 24})
        self.assertEqual(result["results"][0]["source_event_id"], 1)
        index.assert_called_once()

    def test_compound_question_returns_only_later_action_at_its_onset(self):
        entering = event(1, 0, 2, 2.1, "강아지가 수영장 물속으로 들어감")
        shaking = event(2, 0, 11, 12.25, "강아지가 몸을 털음")
        entering_again = event(3, 0, 20, 20.1, "강아지가 수영장 물속으로 다시 들어감")
        events = [entering, shaking, entering_again]
        def matching(question, rows):
            return {"exact": [shaking] if "몸털" in question else [entering, entering_again], "related": []}
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events), \
             patch("pipeline.search.action_event_index.semantic_event_candidates", return_value=None), \
             patch.object(grounded_action_search, "_select_matching_events", side_effect=matching):
            result = grounded_action_search.find_action_event_scenes("수영하고 몸털었어?", "long", 3)
        self.assertEqual([row["source_event_id"] for row in result["results"]], [2])
        self.assertEqual([row["timestamp"] for row in result["results"]], [11])
        self.assertEqual(result["results"][0]["playback_start"], 10)
        self.assertIn("00:11", result["answer"])

    def test_compound_question_does_not_claim_action_without_earlier_context(self):
        shaking = event(1, 0, 11, 12, "강아지가 몸을 털음")
        entering_later = event(2, 0, 20, 21, "강아지가 수영장 물속으로 들어감")
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation",
                          return_value=[shaking, entering_later]), \
             patch("pipeline.search.action_event_index.semantic_event_candidates", return_value=None), \
             patch.object(grounded_action_search, "_select_matching_events",
                          side_effect=lambda question, rows: {"exact": [shaking] if "몸털" in question
                                                                else [entering_later], "related": []}):
            result = grounded_action_search.find_action_event_scenes("수영하고 몸털었어?", "long", 3)
        self.assertEqual(result["results"], [])

    def test_global_action_search_can_find_a_new_video(self):
        shaking = event(1, 0, 11, 13, "강아지가 몸을 털음")
        shaking["video_id"] = "new-video"
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=[shaking]) as timeline, \
             patch("pipeline.search.action_event_index.semantic_event_candidates", return_value=[shaking]), \
             patch.object(grounded_action_search, "_select_matching_events", return_value=[shaking]):
            result = grounded_action_search.find_action_event_scenes("강아지가 몸을 터는 장면 보여줘", None, 3)
        timeline.assert_called_once_with(video_id=None)
        self.assertEqual(result["results"][0]["video_id"], "new-video")
        self.assertIn("new-video", result["answer"])

    def test_petcam_date_filter_uses_trusted_capture_time(self):
        shaking = event(1, 0, 11, 13, "강아지가 몸을 털음")
        shaking["source_frames"][0].update(
            source_type="petcam", captured_at="2026-11-24T09:30:11+09:00")
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=[shaking]), \
             patch("pipeline.search.action_event_index.semantic_event_candidates", return_value=[shaking]), \
             patch.object(grounded_action_search, "_select_matching_events", return_value=[shaking]):
            correct = grounded_action_search.find_action_event_scenes(
                "강아지가 몸을 터는 장면 보여줘", "long", 3,
                recording_date="2026-11-24", time_range={"start_hour": 9, "end_hour": 10})
            wrong = grounded_action_search.find_action_event_scenes(
                "강아지가 몸을 터는 장면 보여줘", "long", 3,
                recording_date="2026-11-23", time_range={"start_hour": 9, "end_hour": 10})
        self.assertEqual(correct["results"][0]["source_event_id"], 1)
        self.assertIsNone(wrong)

    def test_cat_action_search_drops_puppies_labelled_as_cats(self):
        puppy = event(1, 1, 47, 49, "반려동물이 그릇에서 사료를 먹고 있음")
        real = event(2, 2, 9, 12, "반려동물이 사료를 먹고 있음")
        named = event(3, 3, 4, 6, "고양이가 그릇에서 먹음")
        unlabelled = event(4, 4, 4, 6, "반려동물이 사료를 먹고 있음")
        for row in (puppy, real):
            row["source_frames"][0]["object_labels"] = ["bowl", "cat"]
        events = [puppy, real, named, unlabelled]
        cat_frame = real["source_frames"][0]["frame_id"]
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events), \
             patch.object(grounded_action_search, "_select_matching_events", return_value=events), \
             patch.object(grounded_action_search, "confirmed_cat_frame_ids", return_value={cat_frame}):
            result = grounded_action_search.find_action_event_scenes("고양이가 밥 먹는 장면 있어?", "long", 5)
        self.assertEqual([row["source_event_id"] for row in result["results"]], [2, 3])

    def test_cat_verification_rejects_dog_and_keeps_cat(self):
        class Collection:
            def get(self, ids, include):
                return {"ids": ids, "embeddings": np.asarray([[0.8, 0.2], [0.2, 0.8]])}

        with patch.object(species_verification, "get_collection", return_value=Collection()), \
             patch.object(species_verification, "_cat_dog_references", return_value=(np.asarray([0, 1]), np.asarray([1, 0]))):
            found = species_verification.confirmed_cat_frame_ids(["dog-as-cat", "real-cat"])
        self.assertEqual(found, {"real-cat"})


if __name__ == "__main__":
    unittest.main()
