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
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events):
            nursing = grounded_action_search.find_precise_action_scenes(
                "새끼 강아지가 어미 젖을 먹는 장면은 언제야?", "long", 5)
            food = grounded_action_search.find_precise_action_scenes(
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
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events):
            bath = grounded_action_search.find_precise_action_scenes("강아지가 목욕하는 장면은 언제야?", "long", 5)
            sleep = grounded_action_search.find_precise_action_scenes("강아지가 잠자는 장면은 언제야?", "long", 5)
        self.assertEqual(bath["results"][0]["source_event_id"], 1)
        self.assertEqual(sleep["results"][0]["source_event_id"], 2)
        self.assertIn("04:10", bath["answer"])
        self.assertIn("04:44", sleep["answer"])

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
