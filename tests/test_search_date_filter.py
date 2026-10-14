import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from pipeline.search import rag_chain as rag
from pipeline.storage import vector_store as vs


SELECTED_DATE = "2026-10-14"
OTHER_DATE_KEY = "videos/petcam_20261010_090000_000_frame_1.0.jpg"
SELECTED_DATE_KEY = "videos/petcam_20261014_090000_000_frame_1.0.jpg"


def collection_with(*s3_keys):
    collection = MagicMock()
    collection.get.return_value = {
        "ids": [f"frame-{index}" for index in range(len(s3_keys))],
        "metadatas": [
            {"video_id": "video", "s3_key": key, "frame_path": key, "timestamp": 1.0}
            for key in s3_keys
        ],
    }
    return collection


def event(event_id, s3_key):
    return {
        "id": event_id,
        "video_id": "video",
        "start_time": 1.0,
        "end_time": 5.0,
        "action": "물을 마시는 것으로 보임",
        "source_frames": [{"timestamp": 1.0, "s3_key": s3_key}],
    }


class SearchDateFilterTests(unittest.TestCase):
    def run_query(self, collection, top_events=(), **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch.object(rag, "search_with_query_expansion", return_value=[]))
            stack.enter_context(patch.object(rag, "get_behavior_events_overlapping", return_value=[]))
            top = stack.enter_context(
                patch.object(rag, "get_top_behavior_events", return_value=list(top_events))
            )
            stack.enter_context(patch.object(vs, "get_collection", return_value=collection))
            stack.enter_context(patch.object(rag, "build_prompt_hint", return_value=""))
            stack.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": ""}))
            return rag.run_rag_query(query="물 마셨어?", video_id="video", **kwargs), top

    def test_indexed_frames_are_filtered_by_date(self):
        with patch.object(vs, "get_collection", return_value=collection_with(OTHER_DATE_KEY, SELECTED_DATE_KEY)):
            frames = vs.get_indexed_frames(video_id="video", recording_date=SELECTED_DATE)
        self.assertEqual([frame["s3_key"] for frame in frames], [SELECTED_DATE_KEY])

    def test_no_match_on_selected_date_returns_empty_instead_of_other_date(self):
        result, _ = self.run_query(
            collection_with(OTHER_DATE_KEY),
            top_events=[event(1, OTHER_DATE_KEY)],
            recording_date=SELECTED_DATE,
        )
        self.assertEqual(result["results"], [])
        self.assertEqual(result["behavior_events"], [])
        self.assertIn("선택한 날짜와 시간대", result["answer"])

    def test_selected_date_event_beyond_top_ten_is_found(self):
        other_events = [event(index, OTHER_DATE_KEY) for index in range(1, 16)]
        result, top = self.run_query(
            collection_with(),
            top_events=[*other_events, event(99, SELECTED_DATE_KEY)],
            recording_date=SELECTED_DATE,
        )
        self.assertEqual(top.call_args.kwargs["limit"], 200)
        self.assertEqual([item["id"] for item in result["behavior_events"]], [99])

    def test_without_date_filter_fallback_frames_are_kept(self):
        result, top = self.run_query(
            collection_with(OTHER_DATE_KEY),
            time_range={"start_hour": 0, "end_hour": 24},
        )
        self.assertEqual(top.call_args.kwargs["limit"], 10)
        self.assertEqual([frame["s3_key"] for frame in result["results"]], [OTHER_DATE_KEY])


if __name__ == "__main__":
    unittest.main()
