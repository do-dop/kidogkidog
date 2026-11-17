import json
import unittest
from pathlib import Path

from pipeline.search.sequence_query import run_sequence_query
from scripts.reindex_reviewed_sequence import build_events


REVIEW = Path(__file__).resolve().parents[1] / "notebooks/smarthome-bench-evidence/sequence-reviewed-0608.json"


class ReviewedSequenceReindexTests(unittest.TestCase):
    def test_pool_timeline_and_next_actions(self):
        review = json.loads(REVIEW.read_text(encoding="utf-8"))
        scenes = [{"start_time": t, "s3_key": f"frames/smartbench_0608/smartbench_0608_000_frame_{t:.2f}.jpg"}
                  for t in (2.0, 8.0, 12.0, 20.0)]
        events = build_events(review, scenes)
        for index, event in enumerate(events, 1):
            event["id"] = index

        timeline = run_sequence_query("수영장에 들어갔다 나온 순서대로 알려줘",
                                      video_id="smartbench_0608", top_k=10, events=events)
        after_exit = run_sequence_query("수영장에서 나온 다음에는 무엇을 했어?",
                                        video_id="smartbench_0608", events=events)
        after_shake = run_sequence_query("몸을 턴 다음에는 무엇을 했어?",
                                         video_id="smartbench_0608", events=events)

        self.assertEqual([card["source_event_id"] for card in timeline["results"]], [1, 2, 3, 4])
        self.assertEqual([card["source_event_id"] for card in after_exit["results"]], [3])
        self.assertEqual([card["source_event_id"] for card in after_shake["results"]], [4])
        self.assertIn("몸을 턴", after_exit["answer"])
        self.assertIn("다시 들어갔어요", after_shake["answer"])

    def test_review_requires_existing_frame(self):
        review = json.loads(REVIEW.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "저장 프레임"):
            build_events(review, [])


if __name__ == "__main__":
    unittest.main()
