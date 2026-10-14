import unittest
from unittest.mock import MagicMock, patch

import pymysql

from db import processed_chunks


def make_conn(insert_error=None):
    cursor = MagicMock()
    if insert_error:
        cursor.execute.side_effect = insert_error
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


class CommitChunkResultsTests(unittest.TestCase):
    def test_commits_results_with_completion_marker(self):
        conn, cursor = make_conn()
        write = MagicMock()

        with patch.object(processed_chunks, "connect", return_value=conn):
            result = processed_chunks.commit_chunk_results("chunks/v/a.mp4", "v", write)

        self.assertTrue(result)
        self.assertIn("processed_chunks", cursor.execute.call_args.args[0])
        write.assert_called_once_with(conn)
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()
        conn.close.assert_called_once()

    def test_duplicate_marker_rolls_back_without_writing_results(self):
        conn, _ = make_conn(pymysql.err.IntegrityError(1062, "Duplicate entry"))
        write = MagicMock()

        with patch.object(processed_chunks, "connect", return_value=conn):
            result = processed_chunks.commit_chunk_results("chunks/v/a.mp4", "v", write)

        self.assertFalse(result)
        write.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()

    def test_failure_while_writing_rolls_back_everything(self):
        conn, _ = make_conn()
        write = MagicMock(side_effect=RuntimeError("db down"))

        with patch.object(processed_chunks, "connect", return_value=conn):
            with self.assertRaises(RuntimeError):
                processed_chunks.commit_chunk_results("chunks/v/a.mp4", "v", write)

        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_other_integrity_errors_are_not_treated_as_duplicates(self):
        conn, _ = make_conn(pymysql.err.IntegrityError(1452, "FK failure"))

        with patch.object(processed_chunks, "connect", return_value=conn):
            with self.assertRaises(pymysql.err.IntegrityError):
                processed_chunks.commit_chunk_results("chunks/v/a.mp4", "v", MagicMock())

    def test_is_chunk_processed_reflects_marker_presence(self):
        for rows, expected in (([(1,)], True), ([], False)):
            conn, cursor = make_conn()
            cursor.fetchall.return_value = rows
            with patch.object(processed_chunks, "connect", return_value=conn):
                self.assertEqual(processed_chunks.is_chunk_processed("chunks/v/a.mp4"), expected)
            conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
