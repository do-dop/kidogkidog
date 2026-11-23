import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from pipeline import tasks
from pipeline.search import grounded_action_search, query_suggester, rag_chain


class NewVideoActionFlowTests(unittest.TestCase):
    def test_new_chunk_commits_event_before_indexing_it(self):
        frame = {"frame_path": "/tmp/new_video_frame.jpg", "timestamp": 2.0}
        behavior = {"video_id": "new-video", "start_time": 1.0, "end_time": 3.0,
                    "action": "강아지가 몸을 털음", "summary": "강아지가 몸을 털고 있다",
                    "source_frames": [{"s3_key": "frames/new-video/frame_2.jpg", "timestamp": 2.0}]}
        committed = []

        def commit(_chunk, _video, writer, integrity):
            writer(object())
            committed.append(True)
            return True

        def index(events):
            self.assertTrue(committed)
            self.assertEqual([(row["id"], row["video_id"]) for row in events], [(42, "new-video")])
            return 1

        with ExitStack() as stack:
            for name in ("init_db", "mark_chunk_state", "insert_scene", "record_chunk_integrity"):
                stack.enter_context(patch.object(tasks, name))
            stack.enter_context(patch.object(tasks, "is_chunk_ready", return_value=False))
            stack.enter_context(patch.object(tasks, "get_object_metadata", return_value={"Metadata": {}}))
            stack.enter_context(patch.object(tasks, "download_video"))
            stack.enter_context(patch.object(tasks, "extract_frames", return_value=[frame]))
            stack.enter_context(patch.object(tasks, "upload_frame_to_object_storage",
                                      return_value="frames/new-video/frame_2.jpg"))
            stack.enter_context(patch.object(tasks, "detect_objects", return_value=["dog"]))
            stack.enter_context(patch.object(tasks, "index_frame_to_chromadb", return_value="frame-2"))
            stack.enter_context(patch.object(tasks, "verify_indexed_frame_ids", return_value=[]))
            stack.enter_context(patch.object(tasks, "extract_events_for_chunk", return_value=[behavior]))
            stack.enter_context(patch.object(tasks, "insert_behavior_events", return_value=[42]))
            stack.enter_context(patch.object(tasks, "commit_chunk_results", side_effect=commit))
            stack.enter_context(patch.object(tasks, "delete_local_frames", return_value=0))
            stack.enter_context(patch("pipeline.search.action_event_index.index_behavior_events",
                                      side_effect=index))
            capture = MagicMock()
            capture.get.return_value = 0
            stack.enter_context(patch("cv2.VideoCapture", return_value=capture))
            result = tasks.process_chunk.run("chunks/new-video/new_000.mp4")

            stored = {**behavior, "id": 42, "confidence": 0.9, "interestingness": 0.8}
            with patch.object(query_suggester, "get_top_behavior_events", return_value=[stored]), \
                 patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=[stored]), \
                 patch("pipeline.search.action_event_index.semantic_event_candidates", return_value=[stored]), \
                 patch.object(grounded_action_search, "_select_matching_events", return_value=[stored]):
                summary = query_suggester.get_suggestion_behavior_events(video_id="new-video")
                searched = rag_chain.run_rag_query(
                    "강아지가 몸을 터는 장면 보여줘", video_id="new-video",
                    time_range={"start_hour": 0, "end_hour": 24})

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["behavior_event_count"], 1)
        self.assertEqual(summary[0]["id"], 42)
        self.assertEqual(searched["results"][0]["source_event_id"], summary[0]["id"])


if __name__ == "__main__":
    unittest.main()
