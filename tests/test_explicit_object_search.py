import unittest
from unittest.mock import Mock, patch

from pipeline import vector_store as store, rag_chain as rag


def frame(n, labels, **kwargs):
    stem = f'clip_000_frame_{n:.2f}'
    return dict(frame_id=stem, frame_path=f'{stem}.jpg', s3_key=f'{stem}.jpg',
                video_id='video', timestamp=n, object_labels=labels, **kwargs)


class ExplicitObjectSearchTests(unittest.TestCase):
    def test_explicit_names_and_boundaries(self):
        for text, expected in [
            ('고양이가 보이는 장면', {'cat'}), ('강아지가 보이는 장면', {'dog'}),
            ('개가 뛰는 장면', {'dog'}), ('사람이 함께 보이는 장면', {'person'}),
            ('Cats and dogs with humans', {'cat', 'dog', 'person'}),
            ('A person', {'person'}), ('물을 마신 장면', set()),
            ('반려동물이 그릇 근처에 있는 장면', set()),
            ('자동 급식기 근처에 있는 장면', set()), ('소개와 개인적인 catalog dogma', set()),
        ]:
            with self.subTest(text=text):
                self.assertEqual(store.explicit_object_labels(text), expected)

    def test_exact_label_tokens_and_multiple_objects(self):
        frames = [frame(0, 'dog'), frame(2, 'cat'), frame(4, 'cat,dog'),
                  frame(6, ['CAT', 'person']), frame(8, ''), frame(10, 'wildcat')]
        self.assertEqual([f['timestamp'] for f in store.filter_frames_by_objects(frames, {'cat'})], [2, 4, 6])
        self.assertEqual([f['timestamp'] for f in store.filter_frames_by_objects(frames, {'cat', 'dog'})], [4])
        self.assertIs(store.filter_frames_by_objects(frames, set()), frames)

    def test_scoped_search_filters_before_clip_and_top_k(self):
        frames = [frame(0, 'dog'), frame(2, 'cat'), frame(4, 'cat,dog')]
        collection = Mock()
        # Even a returned out-of-scope high-score dog cannot leak through.
        collection.query.return_value = {'ids': [[f['frame_id'] for f in frames]],
            'metadatas': [frames], 'distances': [[0.01, 0.4, 0.3]]}
        with patch.object(store, 'get_collection', return_value=collection), \
             patch.object(store, 'embed_text', return_value=[1]), \
             patch.object(store, '_search_queries_with_original', return_value=['고양이', 'pet']):
            results = store.search_within_frames('고양이가 보이는 장면', frames, 3)
        self.assertEqual([f['timestamp'] for f in results], [4, 2])
        self.assertEqual(collection.query.call_args.kwargs['ids'], [frames[1]['frame_id'], frames[2]['frame_id']])
        self.assertEqual([round(f['score'], 2) for f in results], [0.7, 0.6])

    def test_dog_and_person_filter(self):
        frames = [frame(0, 'cat'), frame(2, 'dog'), frame(4, 'person')]
        for query, timestamp in [('강아지가 보이는 장면', 2), ('사람이 함께 보이는 장면', 4)]:
            result = store.filter_frames_by_objects(frames, store.explicit_object_labels(query))
            self.assertEqual([f['timestamp'] for f in result], [timestamp])

    def test_no_matches_skips_embedding(self):
        with patch.object(store, 'embed_text') as embed, patch.object(store, 'get_collection') as collection:
            self.assertEqual(store.search_within_frames('고양이가 보여?', [frame(0, 'dog')]), [])
        embed.assert_not_called()
        collection.assert_not_called()

    def test_general_search_discovers_matches_beyond_old_top_k(self):
        frames = [frame(i*2, 'dog') for i in range(30)] + [frame(100, 'cat')]
        collection = Mock()
        collection.get.return_value = {'ids': [f['frame_id'] for f in frames], 'metadatas': frames}
        collection.query.return_value = {'ids': [[frames[-1]['frame_id']]], 'metadatas': [[frames[-1]]], 'distances': [[0.7]]}
        with patch.object(store, 'get_collection', return_value=collection), \
             patch.object(store, 'embed_text', return_value=[1]):
            result = store.search_with_query_expansion('고양이가 보여?', video_id='video')
        self.assertEqual(result[0]['timestamp'], 100)
        collection.get.assert_called_once_with(where={'video_id': 'video'}, include=['metadatas'])
        self.assertEqual(collection.query.call_args.kwargs['ids'], [frames[-1]['frame_id']])

    def test_general_date_scope_precedes_object_filter(self):
        frames = [frame(0, 'cat', recorded_at='2026-09-26T09:00:00'),
                  frame(2, 'cat', recorded_at='2026-09-27T09:00:00')]
        collection = Mock()
        collection.get.return_value = {'ids': [f['frame_id'] for f in frames], 'metadatas': frames}
        with patch.object(store, 'get_collection', return_value=collection), \
             patch.object(store, 'search_within_frames', return_value=[]) as ranked:
            store.search_with_query_expansion('고양이', video_id='video', recording_date='2026-09-27', time_range={'start_hour': 8, 'end_hour': 10})
        self.assertEqual([f['timestamp'] for f in ranked.call_args.args[1]], [2])

    def test_unqualified_query_keeps_existing_general_path(self):
        for query in ['물을 마신 장면', '반려동물이 그릇 근처에 있는 장면']:
            with patch.object(store, '_search_queries_with_original', return_value=[query]), \
                 patch.object(store, 'search', return_value=[frame(0, 'dog', score=0.7)]) as search, \
                 patch.object(store, '_object_candidate_frames') as candidates:
                result = store.search_with_query_expansion(query)
            self.assertEqual(result[0]['object_labels'], 'dog')
            search.assert_called_once()
            candidates.assert_not_called()

    def test_general_no_match_cannot_repopulate_unfiltered_fallback(self):
        with patch.object(rag, 'search_with_query_expansion', return_value=[]), \
             patch.object(rag, '_get_relevant_behavior_events') as events, \
             patch.object(rag, '_fallback_indexed_frames') as fallback, \
             patch.object(rag, '_generate_answer_with_langchain') as llm:
            result = rag.run_rag_query('고양이가 보여?', video_id='video')
        self.assertEqual(result['results'], [])
        self.assertEqual(result['evidence_items'], [])
        self.assertIn('검출 누락', result['answer'])
        events.assert_not_called(); fallback.assert_not_called(); llm.assert_not_called()

    def test_event_fallback_checks_all_sources_before_top_k(self):
        frames = [frame(i*2, 'dog') for i in range(4)] + [frame(8, 'person')]
        event = dict(id=1, video_id='video', start_time=0, end_time=10, source_frames=frames)
        with patch.object(rag, 'get_indexed_frames_in_time_window', return_value=[]), \
             patch.object(rag, 'build_prompt_hint', return_value=''), \
             patch.object(rag, '_generate_answer_with_langchain', return_value='answer'):
            result = rag._run_event_grounded_query('사람이 함께 보여?', event, 3, None)
        self.assertEqual([f['timestamp'] for f in result['results']], [8])
        self.assertIsNone(result['results'][0]['clip_similarity'])
        self.assertEqual(result['results'][0]['retrieval_source'], 'source_frames_fallback')

    def test_event_no_match_does_not_relax_or_leave_window(self):
        frames = [frame(0, 'dog'), frame(100, 'cat')]
        event = dict(id=1, video_id='video', start_time=0, end_time=10, source_frames=frames)
        with patch.object(rag, 'get_indexed_frames_in_time_window', return_value=[]), \
             patch.object(rag, '_generate_answer_with_langchain') as llm:
            result = rag._run_event_grounded_query('고양이가 보여?', event, 3, None)
        self.assertEqual(result['results'], [])
        self.assertIn('확인하지 못했어요', result['answer'])
        llm.assert_not_called()

    def test_direct_search_also_filters(self):
        frames = [frame(0, 'dog'), frame(2, 'cat')]
        collection = Mock()
        collection.get.return_value = {'ids': [f['frame_id'] for f in frames], 'metadatas': frames}
        collection.query.return_value = {'ids': [[f['frame_id'] for f in frames]], 'metadatas': [frames], 'distances': [[0.1, 0.4]]}
        with patch.object(store, 'get_collection', return_value=collection), patch.object(store, 'embed_text', return_value=[1]):
            result = store.search('고양이가 보여?', video_id='video')
        self.assertEqual([f['object_labels'] for f in result], ['cat'])
        self.assertEqual(collection.query.call_args.kwargs['ids'], [frames[1]['frame_id']])


if __name__ == '__main__':
    unittest.main()
