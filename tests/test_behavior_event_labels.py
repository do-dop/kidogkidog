import json
import unittest

from pipeline.search.sequence_query import run_sequence_query
from pipeline.events.behavior_event_extractor import (
    ACTION_TYPES,
    _build_vlm_prompt,
    _event_labels,
    _normalize_behavior_event,
)


def window():
    return [
        {
            "video_id": "clip", "timestamp": float(second), "motion_start": 0.0, "motion_end": 10.0,
            "frame_id": f"f{second}", "frame_path": f"/tmp/f{second}.jpg", "s3_key": f"frames/clip/f{second}.jpg",
            "object_labels": [],
        }
        for second in range(0, 11, 2)
    ]


class BehaviorEventLabelTests(unittest.TestCase):
    def test_labels_are_validated(self):
        self.assertEqual(_event_labels({"action_type": "Run", "continues_previous": True}),
                         {"action_type": "run", "continues_previous": True})
        self.assertEqual(_event_labels({"action_type": "dance", "continues_previous": "true"}), {})
        self.assertEqual(_event_labels({}), {})
        self.assertIn("other", ACTION_TYPES)

    def test_labels_are_stored_on_the_evidence_frame_without_a_schema_change(self):
        frames = window()
        event = _normalize_behavior_event(
            event={"action": "반려동물이 달려요", "start_time": 2.0, "end_time": 8.0,
                   "action_type": "run", "continues_previous": False},
            group=frames, representative_frames=frames,
        )
        labels = event["source_frames"][0]["event_labels"]
        self.assertEqual((labels["action_type"], labels["continues_previous"]), ("run", False))
        self.assertTrue(labels["call_id"].endswith("|0.0-10.0"))
        self.assertNotIn("_frame_", labels["call_id"])

    def test_events_without_valid_labels_store_nothing_extra(self):
        frames = window()
        event = _normalize_behavior_event(
            event={"action": "반려동물이 움직여요", "start_time": 2.0, "end_time": 8.0, "action_type": "moonwalk"},
            group=frames, representative_frames=frames,
        )
        self.assertNotIn("event_labels", event["source_frames"][0])

    def test_scene_change_events_are_not_labeled(self):
        frames = window()
        event = _normalize_behavior_event(
            event={"action": "소파가 넘어짐", "start_time": 2.0, "end_time": 8.0, "subject_type": "scene",
                   "target_object": "소파", "action_type": "other", "continues_previous": False},
            group=frames, representative_frames=frames,
        )
        self.assertNotIn("event_labels", event["source_frames"][0])

    def test_labels_survive_storage_and_drive_sequence_grouping(self):
        frames = [
            {**frame, "source_type": "petcam",
             "captured_at": f"2026-11-15T10:00:{int(frame['timestamp']):02d}+00:00",
             "frame_id": f"clip__petcam_20261115_100000_000_frame_{frame['timestamp']:.2f}",
             "s3_key": f"frames/clip/petcam_20261115_100000_000_frame_{frame['timestamp']:.2f}.jpg"}
            for frame in window()
        ]
        raw = [
            {"action": "반려동물이 연못에서 헤엄쳐요", "start_time": 0.0, "end_time": 4.0, "action_type": "swim", "continues_previous": False},
            {"action": "반려동물이 연못에서 계속 헤엄쳐요", "start_time": 4.0, "end_time": 8.0, "action_type": "swim", "continues_previous": True},
            {"action": "반려동물이 연못가로 올라와요", "start_time": 8.0, "end_time": 10.0, "action_type": "walk", "continues_previous": False},
        ]
        stored = []
        for index, item in enumerate(raw, start=1):
            normalized = _normalize_behavior_event(event=item, group=frames, representative_frames=frames)
            # DB 저장·조회를 JSON 왕복으로 흉내 낸다(behavior_events는 source_frames를 JSON 컬럼에 둔다).
            stored.append({
                "id": index, "video_id": "clip", "start_time": normalized["start_time"], "end_time": normalized["end_time"],
                "summary": normalized["summary"], "confidence": 0.8,
                "source_frames": json.loads(json.dumps(normalized["source_frames"])),
            })
        result = run_sequence_query("행동 순서 알려줘", events=stored, top_k=10)
        self.assertEqual([card["merged_count"] for card in result["results"]], [2, 1])

    def test_prompt_asks_for_the_label_fields(self):
        prompt = _build_vlm_prompt({"video_id": "clip", "frames": []})
        self.assertIn("action_type", prompt)
        self.assertIn("continues_previous", prompt)
        for name in ("eat", "swim", "leave_frame", "other"):
            self.assertIn(name, prompt)


if __name__ == "__main__":
    unittest.main()
