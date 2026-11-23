import unittest
from unittest.mock import patch

from pipeline.search import action_event_index


def event(event_id, action):
    return {"id": event_id, "video_id": "video", "start_time": float(event_id),
            "subject": "강아지", "action": action, "target_object": "", "summary": action}


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for key, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.rows[key] = (embedding, document, metadata)

    def get(self, ids, include):
        return {"ids": [key for key in ids if key in self.rows]}

    def query(self, query_embeddings, n_results, where, include):
        ranked = sorted(
            ((sum(a * b for a, b in zip(query_embeddings[0], value[0])), key)
             for key, value in self.rows.items() if value[2]["video_id"] == where["video_id"]),
            reverse=True,
        )
        return {"ids": [[key for _, key in ranked[:n_results]]]}


class ActionEventIndexTests(unittest.TestCase):
    def test_index_all_occurrences_and_rank_by_query(self):
        collection = FakeCollection()
        events = [event(1, "수영장에 들어감"), event(2, "수영장에서 나옴"),
                  event(3, "수영장에 다시 들어감")]

        def embed(_client, texts):
            return [[1.0, 0.0] if "들어" in text else [0.0, 1.0] for text in texts]

        with patch.object(action_event_index, "get_event_collection", return_value=collection), \
             patch.object(action_event_index, "_embed", side_effect=embed), \
             patch("openai.OpenAI"), patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            self.assertEqual(action_event_index.index_behavior_events(events), 3)
            self.assertEqual(action_event_index.index_behavior_events(events), 0)
            self.assertEqual(len(collection.rows), 3)
            ranked = action_event_index.semantic_event_candidates("들어가는 장면", events)
        self.assertEqual({row["id"] for row in ranked[:2]}, {1, 3})

    def test_incomplete_index_uses_existing_search_path(self):
        collection = FakeCollection()
        events = [event(1, "먹음"), event(2, "잠")]
        collection.upsert(["behavior-event-1"], [[1.0, 0.0]], ["먹음"],
                          [{"video_id": "video", "event_id": 1}])
        with patch.object(action_event_index, "get_event_collection", return_value=collection), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            self.assertIsNone(action_event_index.semantic_event_candidates("잠자는 장면", events))


if __name__ == "__main__":
    unittest.main()
