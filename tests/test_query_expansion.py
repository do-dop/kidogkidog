import unittest
from unittest.mock import patch

from pipeline import vector_store as store


class QueryExpansionTests(unittest.TestCase):
    def assertNeutral(self, queries):
        words = " ".join(queries).lower().split()
        self.assertNotIn("dog", words)
        self.assertNotIn("cat", words)

    def test_reported_questions_do_not_invent_door_sleep_or_species(self):
        for query in (
            "자동 급식기 주변에 오래 머문 구간은 언제야?",
            "자동 급식기 방향으로 시선을 집중한 장면이 있어?",
            "자동 급식기 앞에 있는 장면",
            "오래 머문 장면",
        ):
            with self.subTest(query=query):
                expanded = store.expand_search_query(query)
                self.assertNeutral(expanded)
                for wrong_word in ("door", "sleep", "lying"):
                    self.assertNotIn(wrong_word, " ".join(expanded))

    def test_door_with_korean_particles_and_punctuation(self):
        for query in ("문 앞에 있는 장면 보여줘", "문 근처에서 기다린 장면 있어?", "문을 보는 장면", "문, 근처에 있는 장면"):
            with self.subTest(query=query):
                self.assertTrue(any("door" in q for q in store.expand_search_query(query)))

    def test_sleep_conjugations(self):
        for query in ("반려동물이 자고 있는 장면 보여줘", "반려동물이 잠든 장면 있어?", "자는 모습", "잠자는 모습", "잠드는 모습", "잤던 장면"):
            with self.subTest(query=query):
                expanded = store.expand_search_query(query)
                self.assertTrue(any("sleep" in q for q in expanded))
                self.assertNeutral(expanded)

    def test_explicit_species_and_neutral_subject_with_bowl(self):
        for subject, expected in (("강아지가", "dog"), ("고양이가", "cat"), ("반려동물이", "pet")):
            with self.subTest(subject=subject):
                expanded = store.expand_search_query(f"{subject} 그릇 근처에 있는 장면")
                self.assertIn(f"{expected} near a bowl", expanded)
                if expected == "pet":
                    self.assertNeutral(expanded)

    def test_other_substrings_are_not_semantic_keywords(self):
        for query in ("반려동물이 공간에서 움직인 장면", "개수를 확인하는 장면", "질문을 확인해줘", "보호자가 자동차를 보여줘"):
            with self.subTest(query=query):
                self.assertEqual(store._infer_search_subject(query), "pet")
                self.assertNotIn(store._infer_search_target(query), ("water", "a ball", "a door"))
                self.assertIsNone(store._infer_search_action(query))
        for word in ("location", "catalog", "dogmatic", "balloon", "bedroom", "display", "walkway", "runningly"):
            with self.subTest(word=word):
                self.assertEqual(store.expand_search_query(word), [word])

    def test_existing_targets_and_compound_nouns(self):
        for noun, target in (("그릇", "a bowl"), ("밥그릇", "a bowl"), ("물그릇", "a bowl"), ("공", "a ball"), ("침대", "a bed"), ("소파", "a sofa"), ("사람", "a person"), ("물", "water"), ("장난감", "a toy"), ("가방", "a bag")):
            with self.subTest(noun=noun):
                self.assertEqual(store._infer_search_target(f"{noun}을 보는 장면"), target)

    def test_existing_action_conjugations(self):
        for phrase, action in (("만지는", "touching"), ("건드린", "touching"), ("먹는", "eating"), ("먹은", "eating"), ("마시는", "drinking"), ("마신", "drinking"), ("앉아", "sitting near"), ("달리는", "running near"), ("걷는", "walking near"), ("짖는", "barking near"), ("냄새를 맡는", "sniffing"), ("쳐다보는", "looking at"), ("누워 있는", "lying near")):
            with self.subTest(phrase=phrase):
                self.assertEqual(store._infer_search_action(f"{phrase} 장면"), action)

    def test_english_words_plural_and_inflections(self):
        for query, subject, target, action in (
            ("Cats touching bowls", "cat", "a bowl", "touching"),
            ("DOGS sleeping near beds", "dog", "a bed", "near"),
            ("pet drinking water", "pet", "water", "drinking"),
            ("pet looking at people", "pet", "a person", "looking at"),
        ):
            with self.subTest(query=query):
                self.assertEqual(store._infer_search_subject(query), subject)
                self.assertEqual(store._infer_search_target(query), target)
                self.assertEqual(store._infer_search_action(query), action)

    def test_original_query_limit_and_deduplication_are_preserved(self):
        query = "고양이가 공을 만지는 장면"
        queries = store._search_queries_with_original(query)
        self.assertEqual(queries[0], query)
        self.assertLessEqual(len(queries), 5)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertEqual(store.expand_search_query(""), [""])
        self.assertEqual(store.expand_search_query("알 수 없는 장면"), ["알 수 없는 장면"])
        with patch.object(store, "_infer_search_target", side_effect=ValueError("test failure")):
            self.assertEqual(store.expand_search_query(query), [query])


if __name__ == "__main__":
    unittest.main()
