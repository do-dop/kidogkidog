import unittest
from unittest.mock import patch

from pipeline.search import object_presence, rag_chain
from pipeline.storage import vector_store


def frame(frame_id, timestamp, labels, chunk=0):
    return {
        "frame_id": frame_id,
        "frame_path": f"/tmp/{frame_id}.jpg",
        "s3_key": f"frames/video/video_{chunk:03d}_frame_{timestamp:.2f}.jpg",
        "timestamp": timestamp,
        "video_id": "video",
        "object_labels": labels,
    }


class ObjectPresenceTests(unittest.TestCase):
    def test_simple_visibility_question_uses_matching_playable_frames(self):
        frames = [frame("dog", 1, "dog"), frame("cat-1", 2, "cat"),
                  frame("cat-near", 3, "cat"), frame("cat-2", 9, "cat")]
        with patch.object(object_presence, "get_indexed_frames", return_value=frames) as indexed, \
             patch.object(vector_store, "search_with_query_expansion", side_effect=AssertionError("CLIP must not run")):
            result = rag_chain.run_rag_query("고양이가 보이는 장면 보여줘", video_id="video", top_k=3)

        indexed.assert_called_once_with(video_id="video", recording_date=None, time_range=None)
        self.assertEqual([item["frame_id"] for item in result["results"]], ["cat-1", "cat-2"])
        self.assertTrue(all(item["s3_key"] and item["match_level"] == "object_label" for item in result["results"]))
        self.assertIn("감지된 장면 후보", result["answer"])
        self.assertFalse(result["used_llm"])

    def test_place_or_action_question_is_not_reduced_to_object_presence(self):
        for query in ("고양이가 초인종 앞에 있는 장면 보여줘", "고양이가 움직인 장면 찾아줘"):
            self.assertIsNone(object_presence._presence_subject(query), query)

    def test_short_presence_questions_use_the_same_grounded_frames(self):
        for query in ("고양이가 있어?", "고양이 있니?", "강아지가 있어?", "강아지 보여?"):
            with self.subTest(query=query):
                subject = object_presence._presence_subject(query)
                self.assertIsNotNone(subject)
        with patch.object(object_presence, "get_indexed_frames", return_value=[frame("cat", 2, "cat")]):
            result = object_presence.find_presence_scenes("고양이가 있어?", "video", 3)
        self.assertEqual(result["results"][0]["frame_id"], "cat")

    def test_no_matching_label_does_not_invent_a_scene(self):
        with patch.object(object_presence, "get_indexed_frames", return_value=[frame("dog", 2, "dog")]):
            self.assertIsNone(object_presence.find_presence_scenes("고양이가 보이는 장면 보여줘", "video", 3))


if __name__ == "__main__":
    unittest.main()
