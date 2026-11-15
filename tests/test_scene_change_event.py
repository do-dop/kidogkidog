import unittest

from pipeline.events.behavior_event_extractor import _normalize_behavior_event
from pipeline.search.aggregate_query import event_matches_action, is_scene_event


def frames():
    return [
        {"video_id": "dog-video", "frame_id": f"f{n}", "frame_path": f"f{n}.jpg",
         "s3_key": f"frames/f{n}.jpg", "timestamp": float(n)}
        for n in (12, 13, 14)
    ]


def vlm_event(**overrides):
    base = {
        "start_time": 12.0,
        "end_time": 14.0,
        "subject": "소파",
        "subject_type": "scene",
        "action": "소파가 뒤로 젖혀짐",
        "target_object": "소파",
        "summary": "소파가 뒤로 젖혀지면서 반려동물도 같이 넘어갔어요",
    }
    return {**base, **overrides}


class SceneChangeEventTests(unittest.TestCase):
    def test_scene_change_keeps_the_object_as_subject_and_summary_untouched(self):
        saved = _normalize_behavior_event(vlm_event(), frames(), frames())

        self.assertEqual(saved["subject"], "소파")
        self.assertEqual(saved["target_object"], "소파")
        self.assertEqual(saved["summary"], "소파가 뒤로 젖혀지면서 반려동물도 같이 넘어갔어요")
        self.assertTrue(is_scene_event(saved))

    def test_pet_event_is_still_generalized(self):
        pet = vlm_event(subject_type="pet", subject="강아지", action="강아지가 소파에 올라감",
                        summary="강아지가 소파에 올라갔어요")
        saved = _normalize_behavior_event(pet, frames(), frames())

        self.assertEqual(saved["subject"], "반려동물")
        self.assertTrue(saved["summary"].startswith("반려동물이"))
        self.assertFalse(is_scene_event(saved))

    def test_scene_type_without_an_object_falls_back_to_pet_event(self):
        saved = _normalize_behavior_event(vlm_event(target_object=None), frames(), frames())

        self.assertEqual(saved["subject"], "반려동물")

    def test_scene_change_is_not_counted_as_a_pet_action(self):
        scene = {"subject": "소파", "target_object": "소파", "action": "소파에 올라가듯 젖혀짐",
                 "summary": "소파가 젖혀지면서 올라감"}

        self.assertFalse(event_matches_action(scene, "eat"))
        self.assertFalse(event_matches_action(scene, "gate_climb"))


if __name__ == "__main__":
    unittest.main()
