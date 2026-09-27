import unittest
from unittest.mock import Mock, patch

from pipeline import rag_chain as rag, vector_store as store


def indexed_frame(timestamp, chunk="000"):
    stem = f"petcam_20260926_173123_{chunk}_frame_{timestamp:.2f}"
    return {
        "frame_id": f"IMG_8450_2__{stem}",
        "frame_path": f"pipeline/frames/IMG_8450_2/{stem}.jpg",
        "s3_key": f"frames/IMG_8450_2/{stem}.jpg",
        "video_id": "IMG_8450_2",
        "timestamp": timestamp,
        "object_labels": "cat",
    }


class EventCandidateSearchTests(unittest.TestCase):
    def setUp(self):
        self.frames = [indexed_frame(t) for t in (2, 4.5, 6.5, 8.5, 12)]
        self.event = {
            "id": 3, "video_id": "IMG_8450_2", "start_time": 4.5,
            "end_time": 9, "confidence": 0.8, "source_frames": self.frames[1:4],
        }

    def run_event(self):
        with (
            patch.object(rag, "build_prompt_hint", return_value=""),
            patch.object(rag, "_generate_answer_with_langchain", return_value="answer"),
        ):
            return rag._run_event_grounded_query("시선", self.event, 3, None)

    def test_candidates_are_scoped_before_any_similarity_cutoff(self):
        # More than the previous global Top 20 limit, all from another chunk.
        distractors = [indexed_frame(2 + i / 100, "001") for i in range(30)]
        candidates = distractors + self.frames
        collection = Mock()
        collection.get.return_value = {
            "ids": [f["frame_id"] for f in candidates], "metadatas": candidates,
        }
        distances = [0.9, 0.3, 0.1, 0.2, 0.8]
        collection.query.return_value = {
            "ids": [[f["frame_id"] for f in self.frames]],
            "metadatas": [self.frames], "distances": [distances],
        }
        with (
            patch.object(store, "get_collection", return_value=collection),
            patch.object(store, "embed_text", return_value=[1.0, 0.0]),
            patch.object(rag, "search_with_query_expansion") as global_search,
        ):
            result = self.run_event()
        global_search.assert_not_called()
        collection.get.assert_called_once_with(
            where={"$and": [{"video_id": "IMG_8450_2"}, {"timestamp": {"$gte": 1.5}}, {"timestamp": {"$lte": 12.0}}]},
            include=["metadatas"],
        )
        self.assertEqual(collection.query.call_args.kwargs["ids"], [f["frame_id"] for f in self.frames])
        self.assertEqual(collection.query.call_args.kwargs["n_results"], len(self.frames))
        self.assertEqual([f["timestamp"] for f in result["results"]], [6.5, 8.5, 4.5])
        for item in result["results"]:
            self.assertEqual(item["retrieval_source"], "clip")
            self.assertEqual(item["clip_similarity"], item["score"])
            self.assertEqual(item["event_confidence"], 0.8)

    def test_expansions_merge_best_score_then_limit(self):
        collection = Mock()
        collection.query.side_effect = [
            {"ids": [[f["frame_id"] for f in self.frames]], "metadatas": [self.frames], "distances": [[0.9, 0.8, 0.7, 0.6, 0.5]]},
            {"ids": [[f["frame_id"] for f in self.frames]], "metadatas": [self.frames], "distances": [[0.1, 0.2, 0.3, 0.9, 0.9]]},
        ]
        with (
            patch.object(store, "get_collection", return_value=collection),
            patch.object(store, "embed_text", return_value=[1.0, 0.0]) as embed,
            patch.object(store, "_search_queries_with_original", return_value=["original", "expanded"]),
        ):
            results = store.search_within_frames("original", self.frames, top_k=3)
        self.assertEqual([f["timestamp"] for f in results], [2, 4.5, 6.5])
        self.assertEqual([c.args[0] for c in embed.call_args_list], ["original", "expanded"])
        self.assertEqual(len({f["frame_id"] for f in results}), 3)

    def test_empty_candidates_skip_embedding_and_query(self):
        with patch.object(store, "get_collection") as collection, patch.object(store, "embed_text") as embed:
            self.assertEqual(store.search_within_frames("시선", []), [])
        collection.assert_not_called()
        embed.assert_not_called()

    def test_missing_video_does_not_scan_all_videos(self):
        with patch.object(store, "get_collection") as collection:
            self.assertEqual(store.get_indexed_frames_in_time_window(None, 1.5, 12), [])
        collection.assert_not_called()

    def test_empty_index_or_metadata_failure_keeps_fallback(self):
        for outcome in ([], RuntimeError("metadata unavailable")):
            with self.subTest(outcome=outcome):
                mock_get = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
                with patch.object(rag, "get_indexed_frames_in_time_window", mock_get):
                    result = self.run_event()
                self.assertEqual([f["timestamp"] for f in result["results"]], [4.5, 6.5, 8.5])
                self.assertTrue(all(f["retrieval_source"] == "source_frames_fallback" and f["clip_similarity"] is None for f in result["results"]))

    def test_chroma_query_failure_keeps_fallback(self):
        collection = Mock()
        collection.query.side_effect = RuntimeError("query unavailable")
        with (
            patch.object(rag, "get_indexed_frames_in_time_window", return_value=self.frames),
            patch.object(store, "get_collection", return_value=collection),
            patch.object(store, "embed_text", return_value=[1.0, 0.0]),
        ):
            result = self.run_event()
        self.assertEqual([f["timestamp"] for f in result["results"]], [4.5, 6.5, 8.5])
        self.assertTrue(all(f["retrieval_source"] == "source_frames_fallback" for f in result["results"]))

    def test_general_search_does_not_use_event_candidate_path(self):
        expected = [{**self.frames[0], "score": 0.4}]
        with (
            patch.object(rag, "search_with_query_expansion", return_value=expected) as search,
            patch.object(rag, "get_indexed_frames_in_time_window") as candidates,
            patch.object(rag, "search_within_frames") as ranked,
            patch.object(rag, "_get_relevant_behavior_events", return_value=[]),
            patch.object(rag, "build_prompt_hint", return_value=""),
            patch.object(rag, "_generate_answer_with_langchain", return_value="answer"),
        ):
            result = rag.run_rag_query("시선", video_id="IMG_8450_2", top_k=3)
        self.assertEqual(result["results"], expected)
        search.assert_called_once_with(query="시선", top_k=3, video_id="IMG_8450_2", recording_date=None, time_range=None)
        candidates.assert_not_called()
        ranked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
