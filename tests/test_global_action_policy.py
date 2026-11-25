"""Exercise real scene selection with database/model boundaries mocked."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest import TestCase
from unittest.mock import patch


def event(event_id, video, start=0, action="강아지가 몸을 털음"):
    return {"id": event_id, "video_id": video, "start_time": start, "end_time": start + 2,
            "action": action, "summary": action, "confidence": 0.8,
            "source_frames": [{"timestamp": start, "s3_key": f"frames/{video}/{video}_000_frame_{start}.jpg"}]}


@contextmanager
def search_with(events, ranked=None):
    db = ModuleType("db.behavior_events")
    db.get_behavior_events_for_aggregation = lambda **kw: events
    species = ModuleType("pipeline.search.species_verification")
    species.confirmed_cat_frame_ids = lambda ids: set()
    presence = ModuleType("pipeline.search.object_presence")
    presence._presence_subject = lambda query: None
    index = ModuleType("pipeline.search.action_event_index")
    index.semantic_event_candidates = lambda *args, **kw: ranked if ranked is not None else events
    with patch.dict("sys.modules", {db.__name__: db, species.__name__: species,
                                     presence.__name__: presence, index.__name__: index}):
        spec = importlib.util.spec_from_file_location(
            "action_policy_under_test", Path(__file__).resolve().parents[1] / "pipeline/search/grounded_action_search.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "_select_matching_events", side_effect=lambda query, rows: rows):
            yield module


class GlobalActionPolicyTests(TestCase):
    def test_three_occurrences_do_not_hide_other_matching_videos(self):
        events = [event(i, "long", i * 10) for i in range(5)]
        events += [event(10, "second"), event(11, "third")]
        with search_with(events) as search:
            result = search.find_action_event_scenes("몸을 터는 장면 보여줘", None, 3)
        self.assertEqual({row["video_id"] for row in result["results"]}, {"long", "second", "third"})
        self.assertEqual(sum(row["video_id"] == "long" for row in result["results"]), 3)

    def test_global_semantic_shortlist_keeps_a_lead_from_other_videos(self):
        events = [event(i, "long", i * 10) for i in range(60)] + [event(100, "other")]
        with search_with(events, events[:48]) as search:
            result = search.find_action_event_scenes("몸을 터는 장면 보여줘", None, 3)
        self.assertIn("other", {row["video_id"] for row in result["results"]})

    def test_out_of_order_hits_do_not_merge_distant_episodes(self):
        events = [event(1, "video", 40), event(2, "video", 2)]
        with search_with(events) as search:
            result = search.find_action_event_scenes("몸을 터는 장면 보여줘", None, 3)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(row["playback_end"] - row["playback_start"] == 4 for row in result["results"]))

    def test_matcher_cannot_approve_an_explicit_opposite_action(self):
        events = [event(1, "food", action="강아지가 사료를 먹음")]
        with search_with(events) as search:
            result = search.find_action_event_scenes("강아지가 물 마시는 장면 보여줘", None, 3)
        self.assertEqual(result["results"], [])
