import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers import search, suggestions
from pipeline import question_generator as qg, query_suggester as qs, rag_chain as rag


def event(event_id=1, **fields):
    return dict(id=event_id, video_id='video', action='물을 마시는 것으로 보임',
                summary='물 그릇에서 물을 마심', target_object='물 그릇',
                start_time=2, end_time=12, duration=10, repeat_count=1,
                interestingness=0.5, confidence=0.8) | fields


class QuestionEventLinkingTests(unittest.TestCase):
    def generate_final(self, events, llm_items, labels=None, limit=3):
        with ExitStack() as stack:
            stack.enter_context(patch.object(qs, 'get_top_behavior_events', return_value=events))
            stack.enter_context(patch.object(qs, '_generate_video_specific_questions', return_value=labels or []))
            stack.enter_context(patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}))
            client = stack.enter_context(patch('openai.OpenAI'))
            client.return_value.responses.create.return_value = SimpleNamespace(output_text=json.dumps(llm_items))
            return qs.suggest_query_items(video_id='video', limit=limit)

    def test_original_llm_event_survives_a_higher_keyword_score(self):
        original = event()
        other = event(2, action='소파에 앉음', summary='다른 행동',
                      evidence=['물을 마신 장면이 있어'], interestingness=1)
        question = '물을 마신 장면이 있어?'
        self.assertGreater(qs._question_event_score(question, other), qs._question_event_score(question, original))
        self.assertEqual(qs._best_event_for_question(question, [original, other])['id'], 2)
        item = self.generate_final([original, other], [{'event_index': 1, 'question': question}])[0]
        self.assertEqual(item['original_event_id'], 1)
        self.assertEqual(item['source_event_id'], 1)
        self.assertEqual((item['event_start'], item['event_end']), (2, 12))
        self.assertEqual(item['question_source'], 'LLM')

    def test_behavior_fallback_keeps_original_event(self):
        events = [event(), event(2, evidence=['물을 마신 장면이 있어'])]
        with patch.object(qs, 'get_top_behavior_events', return_value=events), \
             patch.object(qs, '_generate_video_specific_questions', return_value=[]), \
             patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            item = qs.suggest_query_items(video_id='video')[0]
        self.assertEqual(item['source_event_id'], 1)
        self.assertEqual(item['original_event_id'], 1)
        self.assertEqual(item['question_source'], 'behavior fallback')

    def test_label_without_original_event_uses_existing_matching(self):
        events = [event(), event(2, action='사람 등장', summary='사람 등장', target_object='사람')]
        item = self.generate_final(events, [], ['사람이 함께 나온 장면이 있어?'])[0]
        self.assertEqual(item['source_event_id'], 2)
        self.assertIsNone(item['original_event_id'])
        self.assertEqual(item['question_source'], 'label fallback')

    def test_zero_score_has_no_event_or_time_bounds(self):
        events = [event(interestingness=1000)]
        question = '문 앞에 있나요?'
        self.assertEqual(qs._question_event_score(question, events[0]), 0)
        self.assertIsNone(qs._best_event_for_question(question, events))
        item = self.generate_final(events, [], [question])[0]
        for field in ('source_event_id', 'event_start', 'event_end'):
            self.assertIsNone(item[field])

    def test_explicit_source_can_have_zero_score_or_zero_duration(self):
        original = event(action='문 앞에 있음', summary='문 앞에 있음', target_object='문',
                         start_time=6, end_time=6, duration=0)
        # '문'은 기존 키워드 목록에 없고 활용형도 토큰 일치가 되지 않는다.
        question = '문이 보이나요?'
        self.assertEqual(qs._question_event_score(question, original), 0)
        item = self.generate_final([original], [{'event_index': 1, 'question': question}])[0]
        self.assertEqual(item['source_event_id'], 1)
        self.assertEqual((item['event_start'], item['event_end']), (6, 6))

    def test_llm_priority_label_fill_and_three_limit(self):
        events = [event()]
        question = '물을 마신 장면이 있어?'
        items = self.generate_final(events, [{'event_index': 1, 'question': question}],
                                    ['사람이 나온 장면이 있어?', '소파가 보여?', '문이 보여?'])
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]['question'], question)
        self.assertEqual([i['question_source'] for i in items], ['LLM', 'label fallback', 'label fallback'])

    def test_frame_fallback_has_no_source(self):
        with patch.object(qs, 'get_top_behavior_events', return_value=[]), \
             patch.object(qs, '_generate_questions_from_indexed_frames', return_value=['문이 보여?']):
            item = qs.suggest_query_items(video_id='video')[0]
        self.assertEqual(item['question_source'], 'frame fallback')
        self.assertIsNone(item['original_event_id'])
        for field in ('source_event_id', 'event_start', 'event_end'):
            self.assertIsNone(item[field])

    def test_string_api_compatibility(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            self.assertEqual(qg.generate_questions_from_behavior_events([event()]), ['물을 마신 장면이 있어?'])

    def test_http_query_without_source_uses_general_search(self):
        app = FastAPI()
        app.include_router(search.router)
        with patch.object(rag, 'search_with_query_expansion', return_value=[]) as general, \
             patch.object(rag, '_run_event_grounded_query') as grounded, \
             patch.object(rag, 'get_behavior_event_by_id') as lookup, \
             patch.object(rag, '_get_relevant_behavior_events', return_value=[]), \
             patch.object(rag, '_fallback_indexed_frames', return_value=[]):
            response = TestClient(app).post('/query', json={
                'query': '문이 보여?', 'video_id': 'video', 'source_event_id': None,
                'event_start': None, 'event_end': None,
            })
        self.assertEqual(response.status_code, 200)
        general.assert_called_once()
        grounded.assert_not_called()
        lookup.assert_not_called()

    def test_suggestions_metadata_reaches_event_grounded_query(self):
        original = event()
        item = self.generate_final([original], [{'event_index': 1, 'question': '물을 마신 장면이 있어?'}])[0]
        app = FastAPI()
        app.include_router(suggestions.router)
        app.include_router(search.router)
        client = TestClient(app)
        with patch.object(suggestions, 'suggest_query_items', return_value=[item]), \
             patch.object(suggestions, 'get_suggestion_behavior_events', return_value=[original]), \
             patch.object(suggestions, 'get_indexed_frames', return_value=[]):
            payload = client.get('/suggestions', params={'video_id': 'video'}).json()
        source = payload['question_sources'][0]
        self.assertEqual(source, item)
        with patch.object(rag, 'get_behavior_event_by_id', return_value=original) as lookup, \
             patch.object(rag, '_run_event_grounded_query', return_value={'answer': 'ok', 'results': [], 'used_llm': False}) as grounded, \
             patch.object(rag, 'search_with_query_expansion') as general:
            response = client.post('/query', json={
                'query': source['question'], 'video_id': 'video',
                **{k: source[k] for k in ('source_event_id', 'event_start', 'event_end')},
            })
        self.assertEqual(response.status_code, 200)
        lookup.assert_called_once_with(1)
        self.assertEqual(grounded.call_args.kwargs['source_event'], original)
        general.assert_not_called()


if __name__ == '__main__':
    unittest.main()
