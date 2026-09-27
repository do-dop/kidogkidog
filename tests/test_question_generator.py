import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import question_generator as qg


def event(**kwargs):
    return {'id': 3, 'action': '사료 그릇에 접근하여 먹음',
            'summary': '사료 그릇에 접근하여 먹음', 'duration': 4.5,
            'repeat_count': 1, **kwargs}


class QuestionQualityTests(unittest.TestCase):
    def validate(self, question, source=None):
        return qg._validate_behavior_questions(
            [{'event_index': 1, 'question': question}], [source or event()], 3)

    def test_duration_modifiers_are_never_accepted(self):
        for duration in (None, 0, 4.5, 178):
            for word in ('오래', '계속', '한동안', '오랜 시간', '내내'):
                with self.subTest(duration=duration, word=word):
                    self.assertEqual(self.validate(f'그릇 근처에 {word} 머문 장면이 있어?', event(duration=duration)), [])

    def test_repeat_requires_valid_count(self):
        for count in (None, 0, 1, 'invalid', float('nan'), float('inf')):
            for word in ('반복해서', '반복된', '여러 번', '여러번', '자주'):
                self.assertEqual(self.validate(f'{word} 그릇에 접근한 장면이 있어?', event(repeat_count=count)), [])

    def test_repeat_allowed_with_evidence(self):
        question = '여러 번 그릇에 접근한 장면이 있어?'
        for count in (2, 3, '2'):
            self.assertEqual(self.validate(question, event(repeat_count=count)), [question])

    def test_interpretations_removed_without_rewriting(self):
        for phrase in ('불안해 보이는', '배식을 기다리는', '관심을 보인', '호기심을 보인',
                       '편안해 보이는', '행복해 보이는', '시선을 집중한', '먹으려는', '원하는 것 같은'):
            self.assertEqual(self.validate(f'{phrase} 장면이 있어?'), [])

    def test_waiting_fallback_uses_explicit_location(self):
        source = event(action='급식기 앞에서 기다리는 것으로 보임',
                       summary='자동 급식기 앞에서 배식을 기다리는 것으로 보임')
        self.assertEqual(qg._fallback_questions_from_behavior_events([source]),
                         ['자동 급식기 근처에 있는 장면이 있어?'])
        source.update(action='불안해 보인다', summary='불안해 보인다')
        self.assertEqual(qg._fallback_questions_from_behavior_events([source]), [])

    def test_fallback_does_not_turn_negation_or_desire_into_action(self):
        for text in ('물을 마시는 것은 아님. 마시지 않음', '물을 마시고 싶어함',
                     '움직임을 보인 것으로 추정되는 구간', '소파 아래에 들어가려는 모습'):
            self.assertEqual(qg._fallback_questions_from_behavior_events(
                [event(action=text, summary=text)]), [])

    def test_direct_observations_survive(self):
        for question in ('사료 그릇에 접근한 장면이 있어?', '물을 마신 장면이 있어?',
                         '소파 아래에 들어간 적이 있어?', '이동한 장면이 있어?', '사람이 함께 나온 장면이 있어?'):
            self.assertEqual(self.validate(question), [question])
        self.assertEqual(qg._fallback_questions_from_behavior_events([event()]),
                         ['사료 그릇에 접근한 장면이 있어?'])

    def test_one_question_per_event_and_diverse_events(self):
        events = [event(), event(id=4, action='사람이 등장함', summary='사람이 등장함')]
        candidates = [{'event_index': 1, 'question': '사료 그릇에 접근한 장면이 있어?'},
                      {'event_index': 1, 'question': '그릇에서 먹은 장면이 있어?'},
                      {'event_index': 2, 'question': '사람이 등장한 장면이 있어?'}]
        self.assertEqual(qg._validate_behavior_questions(candidates, events, 3),
                         [candidates[0]['question'], candidates[2]['question']])

    def test_linked_event_also_requires_repeat_evidence(self):
        events = [event(repeat_count=2), event(id=4, repeat_count=1)]
        with patch('pipeline.query_suggester._best_event_for_question', return_value=events[1]):
            self.assertEqual(qg._validate_behavior_questions(
                [{'event_index': 1, 'question': '여러 번 접근한 장면이 있어?'}], events, 3), [])

    def test_invalid_structured_output_is_rejected(self):
        items = ['그릇에 접근한 장면이 있어?', {'event_index': True, 'question': '접근?'},
                 {'event_index': 99, 'question': '접근?'}, {'event_index': 1, 'question': None}]
        self.assertEqual(qg._validate_behavior_questions(items, [event()], 3), [])
        self.assertEqual(qg._parse_behavior_questions('not json'), [])

    def test_llm_path_filters_before_return_and_does_not_fill(self):
        items = [{'event_index': 1, 'question': '오래 머문 장면이 있어?'},
                 {'event_index': 1, 'question': '반복해서 먹은 장면이 있어?'},
                 {'event_index': 1, 'question': '사료 그릇에 접근한 장면이 있어?'}]
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), patch('openai.OpenAI') as client:
            client.return_value.responses.create.return_value = SimpleNamespace(output_text=json.dumps(items))
            self.assertEqual(qg.generate_questions_from_behavior_events([event()], 3),
                             ['사료 그릇에 접근한 장면이 있어?'])
            client.return_value.responses.create.return_value.output_text = json.dumps(items[:2])
            self.assertEqual(qg.generate_questions_from_behavior_events([event()], 3), [])

    def test_no_key_and_api_failure_use_validated_fallback(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            self.assertEqual(qg.generate_questions_from_behavior_events([event()], 3),
                             ['사료 그릇에 접근한 장면이 있어?'])
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), patch('openai.OpenAI', side_effect=RuntimeError('test')):
            self.assertEqual(qg.generate_questions_from_behavior_events([event()], 3),
                             ['사료 그릇에 접근한 장면이 있어?'])
        self.assertEqual(qg.generate_questions_from_behavior_events([], 3), [])
        self.assertEqual(qg.generate_questions_from_behavior_events([event()], 0), [])


if __name__ == '__main__':
    unittest.main()
