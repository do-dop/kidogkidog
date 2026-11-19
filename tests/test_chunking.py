import importlib
import unittest
from unittest.mock import patch

from pipeline import chunking


class ChunkSecondsTests(unittest.TestCase):
    def _load(self, value):
        env = {} if value is None else {"CHUNK_SECONDS": value}
        with patch.dict("os.environ", env, clear=True):
            return importlib.reload(chunking).CHUNK_SECONDS

    def tearDown(self):
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(chunking)

    def test_default_is_60(self):
        self.assertEqual(self._load(None), 60)

    def test_env_override(self):
        self.assertEqual(self._load("300"), 300)

    def test_invalid_values_fall_back(self):
        for bad in ("abc", "0", "-5", ""):
            self.assertEqual(self._load(bad), 60, bad)


if __name__ == "__main__":
    unittest.main()
