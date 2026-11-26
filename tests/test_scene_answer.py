import unittest

from pipeline.search.scene_answer import build_scene_answer


def scene(summary, **overrides):
    return {"video_id": "internal_video_001", "summary": summary, "timestamp": 3,
            "s3_key": "frames/video/video_003_frame_3.jpg", "match_level": "behavior_event", **overrides}


class SceneAnswerTests(unittest.TestCase):
    def test_food_answer_is_short_and_does_not_dump_ids_or_model_prose(self):
        answer = build_scene_answer([scene("반려동물이 사료를 섭취하고 있는 모습입니다.")])
        self.assertEqual(answer, "밥 먹는 장면을 찾았어요. 영상 03:03쯤부터 확인해 보세요.")
        self.assertNotIn("internal_video", answer)

    def test_partial_answer_keeps_the_uncertainty(self):
        answer = build_scene_answer([scene("반려동물이 젖을 먹음", match_level="related_event")])
        self.assertIn("젖 먹는 장면", answer)
        self.assertIn("세부 조건까지는 확인하지 못했어요", answer)

    def test_multiple_videos_do_not_get_one_ambiguous_timestamp(self):
        answer = build_scene_answer([scene("물을 마심"), scene("물을 마심", video_id="second")])
        self.assertIn("영상 2개", answer)
        self.assertNotIn("03:03", answer)

    def test_unfamiliar_action_is_kept_instead_of_inventing_a_category(self):
        answer = build_scene_answer([scene("사람이 강아지를 들어 올림")])
        self.assertIn("사람이 강아지를 들어 올림", answer)

    def test_mixed_actions_are_named(self):
        answer = build_scene_answer([scene("사료를 먹음"), scene("물을 마심", video_id="second"),
                                     scene("사료를 섭취함", video_id="third")])
        self.assertEqual(answer, "밥 먹는 장면, 물 마시는 장면을 영상 3개에서 찾았어요. 아래에서 보고 싶은 장면을 골라 보세요.")

    def test_unfamiliar_mixed_actions_are_quoted_and_shortened(self):
        answer = build_scene_answer([scene("강아지가 배식을 기다림."), scene("강아지가 급식기 앞에 앉음"),
                                     scene("강아지가 문 앞을 서성임")])
        self.assertTrue(answer.startswith("‘강아지가 배식을 기다림’ 장면, ‘강아지가 급식기 앞에 앉음’ 장면 등을 찾았어요."))
        self.assertNotIn("문 앞", answer)

    def test_rows_without_a_description_fall_back_to_a_generic_name(self):
        self.assertIn("관련 장면", build_scene_answer([scene("")]))

    def test_empty_result_does_not_claim_the_action_did_not_happen(self):
        self.assertIn("아직 확인하지 못했어요", build_scene_answer([]))

    def test_fractional_seconds_use_the_same_floor_as_video_cards(self):
        answer = build_scene_answer([scene("사료를 먹음", timestamp=2.8)])
        self.assertIn("03:02", answer)
        self.assertNotIn("03:03", answer)

    def test_answer_uses_representative_frame_and_full_video_not_capture_clock(self):
        answer = build_scene_answer([scene("사료를 먹음", timestamp=10,
            representative_timestamp=2.8, capture_time_source="petcam",
            recorded_at="2026-11-26T02:30:00+09:00")])
        self.assertIn("영상 03:02", answer)
        self.assertNotIn("02:30", answer)


if __name__ == "__main__":
    unittest.main()
