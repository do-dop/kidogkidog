import unittest
from unittest.mock import MagicMock

from db.connection import _Cursor


class CursorWrapperTests(unittest.TestCase):
    def test_fetchone_is_forwarded(self):
        raw = MagicMock()
        raw.fetchone.return_value = {"id": 7}

        self.assertEqual(_Cursor(raw).fetchone(), {"id": 7})

    def test_placeholders_are_converted(self):
        raw = MagicMock()
        _Cursor(raw).execute("SELECT * FROM t WHERE id = ?", (1,))

        raw.execute.assert_called_once_with("SELECT * FROM t WHERE id = %s", (1,))


if __name__ == "__main__":
    unittest.main()
