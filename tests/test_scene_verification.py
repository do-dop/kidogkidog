import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.search import rag_chain, scene_verification


def _result(second, start=None, end=None):
    item = {
        "video_id": "video",
        "s3_key": f"frames/video/video_000_frame_{second:.2f}.jpg",
        "timestamp": second,
        "score": 0.5,
    }
    if start is not None:
        item.update(scene_start=start, scene_end=end)
    return item


class SceneVerificationTests(unittest.TestCase):
    def test_intent_and_transition_states_share_one_request(self):
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = SimpleNamespace(
                output_text=json.dumps({"mode": "transition", "source": "inside",
                                        "intermediate": "on fence", "target": "outside"})
            )
            self.assertEqual(scene_verification._verification_plan("test transition"),
                             ("transition", {"source": "inside", "intermediate": "on fence", "target": "outside"}))
            client.return_value.responses.create.assert_called_once()

    def test_transition_without_clear_target_uses_motion_check(self):
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = SimpleNamespace(
                output_text=json.dumps({"mode": "transition", "source": "inside",
                                        "intermediate": "on fence", "target": None})
            )
            self.assertEqual(scene_verification._verification_plan("test unclear transition"), ("motion", None))

    def test_uncertain_event_and_visual_hit_both_enter_shortlist(self):
        payload = {"results": [_result(9, 3, 15)]}
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=[]), patch(
            "pipeline.storage.vector_store.search_with_query_expansion", return_value=[_result(26, 22, 31)]
        ):
            windows = scene_verification._candidate_windows("울타리 밖 장면", "video", payload)
        self.assertEqual([(item["start"], item["end"]) for item in windows], [(2, 17), (21, 33)])

    def test_visual_candidate_is_not_crowded_out_by_many_event_rows(self):
        events = [
            {"id": index, "video_id": "video", "start_time": index * 20.0,
             "end_time": index * 20.0 + 2, "action": "강아지가 움직임",
             "source_frames": [_result(index * 20.0)]}
            for index in range(1, 8)
        ]
        visual = _result(3)
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=events), patch(
            "pipeline.storage.vector_store.search_with_query_expansion", return_value=[visual]
        ):
            windows = scene_verification._candidate_windows("강아지 장면", "video", {"results": []})
        self.assertEqual(windows[0]["source"]["timestamp"], 3)

    def test_visual_verifier_requires_visible_timestamp_and_true_match(self):
        frames = [{"time": 3.0, "jpeg": b"image"}, {"time": 4.0, "jpeg": b"image"}]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch("openai.OpenAI") as client:
            for data, expected in (({"match": True, "time": 3.0}, 3.0),
                                   ({"match": False, "time": 3.0}, None),
                                   ({"match": True, "time": 12.0}, None)):
                client.return_value.responses.create.return_value = SimpleNamespace(output_text=json.dumps(data))
                self.assertEqual(scene_verification._verify_frames("강아지 두 마리 보여줘", frames), expected)

    def test_claimed_time_needs_visible_witness(self):
        frames = [{"time": 31.0, "jpeg": b"dog"}, {"time": 34.0, "jpeg": b"empty"}]
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = SimpleNamespace(
                output_text=json.dumps({"match": False})
            )
            self.assertFalse(scene_verification._verify_witness_frame("울타리 밖으로 나온 장면", frames, 34.0))
            content = client.return_value.responses.create.call_args.kwargs["input"][0]["content"]
            self.assertIn("ZW1wdHk=", content[1]["image_url"])

    def test_transition_requires_visible_final_state_after_initial_state(self):
        frames = [{"time": 16.0, "jpeg": b"a"}, {"time": 17.0, "jpeg": b"b"},
                  {"time": 18.0, "jpeg": b"c"}]
        schema = {"source": "inside", "intermediate": "on fence", "target": "outside"}
        with patch("openai.OpenAI") as client:
            for states, expected in ((["transition", "before", "before"], None),
                                     (["before", "transition", "after"], 18.0),
                                     (["before", "absent", "absent"], None)):
                client.return_value.responses.create.return_value = SimpleNamespace(
                    output_text=json.dumps({"frames": [
                        {"time": frame["time"], "state": state}
                        for frame, state in zip(frames, states)
                    ]})
                )
                self.assertEqual(scene_verification._verify_transition_states("밖으로 나오는 장면", frames, schema), expected)

    def test_verified_moment_controls_playback_and_negative_is_preserved(self):
        original = {"match_level": "related_event", "answer": "세부 조건 미확인",
                    "results": [_result(26, 22, 31)], "behavior_events": [{"id": 1}]}
        frames = [{"time": 31.0, "jpeg": b"image"}]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            scene_verification, "_candidate_windows",
            return_value=[scene_verification._window(original["results"][0])]
        ), patch.object(scene_verification, "_verification_plan", return_value=("motion", None)), patch(
            "pipeline.storage.gcs_uploader.download_video"
        ), patch(
            "pipeline.search.video_aggregate.video_duration", return_value=42
        ), patch("pipeline.search.video_aggregate.sample_frames", return_value=frames), patch.object(
            scene_verification, "_verify_frames", return_value=31.0
        ), patch.object(
            scene_verification, "_verify_witness_frame", return_value=True
        ), patch.object(
            scene_verification, "_nearest_indexed_frame", return_value=_result(31)
        ):
            result = scene_verification.verify_uncertain_scene("울타리 밖으로 나오는 장면", "video", original, 3)
            self.assertEqual(result["match_level"], "verified_video")
            self.assertEqual(result["results"][0]["timestamp"], 31.0)
            self.assertEqual(result["results"][0]["playback_start"], 29.5)
            self.assertEqual(result["results"][0]["s3_key"], _result(31)["s3_key"])
            self.assertEqual(result["behavior_events"], [])
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            scene_verification, "_candidate_windows",
            return_value=[scene_verification._window(original["results"][0])]
        ), patch.object(scene_verification, "_verification_plan", return_value=("motion", None)), patch(
            "pipeline.storage.gcs_uploader.download_video"
        ), patch(
            "pipeline.search.video_aggregate.video_duration", return_value=42
        ), patch("pipeline.search.video_aggregate.sample_frames", return_value=frames), patch.object(
            scene_verification, "_verify_frames", return_value=None
        ):
            self.assertIs(scene_verification.verify_uncertain_scene("울타리 밖으로 나오는 장면", "video", original, 3), original)

    def test_static_relation_uses_visible_frame_without_temporal_challenge(self):
        original = {"match_level": "unverified_action", "results": []}
        window = scene_verification._window(_result(3, 2, 4))
        frames = [{"time": 3.0, "jpeg": b"image"}]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            scene_verification, "_candidate_windows", return_value=[window]
        ), patch.object(scene_verification, "_verification_plan", return_value=("static", None)), patch(
            "pipeline.storage.gcs_uploader.download_video"
        ), patch(
            "pipeline.search.video_aggregate.video_duration", return_value=42
        ), patch("pipeline.search.video_aggregate.sample_frames", return_value=frames), patch.object(
            scene_verification, "_verify_frames", return_value=3.0
        ) as verify, patch.object(scene_verification, "_verify_witness_frame", return_value=True), patch.object(
            scene_verification, "_nearest_indexed_frame", return_value=None
        ):
            result = scene_verification.verify_uncertain_scene("강아지 두 마리 함께 보여줘", "video", original, 3)
        self.assertEqual(result["results"][0]["timestamp"], 3.0)
        verify.assert_called_once()

    def test_singular_scene_stops_at_first_verified_match(self):
        original = {"match_level": "unverified_action", "results": []}
        windows = [scene_verification._window(_result(second)) for second in (2, 15, 30)]
        frames = [{"time": 2.0, "jpeg": b"image"}]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            scene_verification, "_candidate_windows", return_value=windows
        ), patch.object(scene_verification, "_verification_plan", return_value=("static", None)), patch(
            "pipeline.storage.gcs_uploader.download_video"
        ), patch("pipeline.search.video_aggregate.video_duration", return_value=42), patch(
            "pipeline.search.video_aggregate.sample_frames", return_value=frames
        ), patch.object(scene_verification, "_verify_frames", return_value=2.0) as verify, patch.object(
            scene_verification, "_verify_witness_frame", return_value=True
        ), patch.object(scene_verification, "_nearest_indexed_frame", return_value=None):
            result = scene_verification.verify_uncertain_scene(
                "강아지 두 마리 함께 나오는 장면 보여줘", "video", original, 3,
            )
        self.assertEqual(len(result["results"]), 1)
        verify.assert_called_once()

    def test_explicit_multiple_scenes_keep_distinct_moments_only(self):
        self.assertEqual(scene_verification._requested_result_count("모든 장면 보여줘", 3), 3)
        self.assertEqual(scene_verification._requested_result_count("한 장면 보여줘", 3), 1)
        confirmed = [_result(2)]
        self.assertTrue(scene_verification._same_verified_scene(_result(3), confirmed))
        self.assertFalse(scene_verification._same_verified_scene(_result(9), confirmed))

        original = {"match_level": "unverified_action", "results": []}
        windows = [scene_verification._window(_result(second)) for second in (2, 3, 9)]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch.object(
            scene_verification, "_candidate_windows", return_value=windows
        ), patch.object(scene_verification, "_verification_plan", return_value=("static", None)), patch(
            "pipeline.storage.gcs_uploader.download_video"
        ), patch("pipeline.search.video_aggregate.video_duration", return_value=42), patch(
            "pipeline.search.video_aggregate.sample_frames", return_value=[{"time": 2.0, "jpeg": b"image"}]
        ), patch.object(scene_verification, "_verify_frames", side_effect=[2.0, 3.0, 9.0]), patch.object(
            scene_verification, "_verify_witness_frame", return_value=True
        ), patch.object(scene_verification, "_nearest_indexed_frame", return_value=None):
            result = scene_verification.verify_uncertain_scene("모든 장면 보여줘", "video", original, 3)
        self.assertEqual([item["timestamp"] for item in result["results"]], [2.0, 9.0])

    def test_rag_query_rechecks_only_uncertain_non_color_answer(self):
        payload = {"match_level": "unverified_action", "results": []}
        with patch.object(rag_chain, "_run_rag_query", return_value=payload), patch.object(
            rag_chain, "verify_uncertain_scene", return_value={"match_level": "verified_video"}
        ) as verify:
            result = rag_chain.run_rag_query("강아지 두 마리 같이 나온 장면 보여줘", video_id="video")
        self.assertEqual(result["match_level"], "verified_video")
        verify.assert_called_once_with("강아지 두 마리 같이 나온 장면 보여줘", "video", payload, 5)


if __name__ == "__main__":
    unittest.main()
