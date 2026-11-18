import unittest
from unittest.mock import MagicMock, patch


class ChunkIntegrityTests(unittest.TestCase):
    def test_missing_chroma_ids_are_reported(self):
        from pipeline.storage import vector_store

        collection = MagicMock()
        collection.get.return_value = {"ids": ["frame-a"]}
        with patch.object(vector_store, "get_collection", return_value=collection):
            missing = vector_store.verify_indexed_frame_ids(["frame-a", "frame-b"])

        self.assertEqual(missing, ["frame-b"])
        collection.get.assert_called_once_with(ids=["frame-a", "frame-b"])

    def test_empty_frame_list_requires_no_vector_query(self):
        from pipeline.storage import vector_store

        with patch.object(vector_store, "get_collection") as collection:
            self.assertEqual(vector_store.verify_indexed_frame_ids([]), [])
        collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
