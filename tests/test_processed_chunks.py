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
            result = processed_chunks.commit_chunk_results(
                "chunks/v/a.mp4", "v", write,
                {"scene_count": 2, "vector_frame_count": 2, "behavior_event_count": 1},
            )

        self.assertTrue(result)
        executed_sql = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("processed_chunks" in sql for sql in executed_sql))
        self.assertTrue(any("chunk_processing_status" in sql for sql in executed_sql))
        write.assert_called_once_with(conn)
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()
        conn.close.assert_called_once()

    def test_duplicate_marker_rolls_back_without_writing_results(self):
        conn, _ = make_conn(pymysql.err.IntegrityError(1062, "Duplicate entry"))
        write = MagicMock()

        with patch.object(processed_chunks, "connect", return_value=conn):
            result = processed_chunks.commit_chunk_results(
                "chunks/v/a.mp4", "v", write,
                {"scene_count": 2, "vector_frame_count": 2, "behavior_event_count": 1},
            )

        self.assertFalse(result)
        write.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()

    def test_failure_while_writing_rolls_back_everything(self):
        conn, _ = make_conn()
        write = MagicMock(side_effect=RuntimeError("db down"))

        with patch.object(processed_chunks, "connect", return_value=conn):
            with self.assertRaises(RuntimeError):
                processed_chunks.commit_chunk_results(
                    "chunks/v/a.mp4", "v", write,
                    {"scene_count": 2, "vector_frame_count": 2, "behavior_event_count": 1},
                )

        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_other_integrity_errors_are_not_treated_as_duplicates(self):
        conn, _ = make_conn(pymysql.err.IntegrityError(1452, "FK failure"))

        with patch.object(processed_chunks, "connect", return_value=conn):
            with self.assertRaises(pymysql.err.IntegrityError):
                processed_chunks.commit_chunk_results(
                    "chunks/v/a.mp4", "v", MagicMock(),
                    {"scene_count": 2, "vector_frame_count": 2, "behavior_event_count": 1},
                )

    def test_is_chunk_processed_reflects_marker_presence(self):
        for rows, expected in (([(1,)], True), ([], False)):
            conn, cursor = make_conn()
            cursor.fetchall.return_value = rows
            with patch.object(processed_chunks, "connect", return_value=conn):
                self.assertEqual(processed_chunks.is_chunk_processed("chunks/v/a.mp4"), expected)
            conn.close.assert_called_once()

    def test_is_chunk_ready_requires_integrity_status(self):
        for rows, expected in (([(1,)], True), ([], False)):
            conn, cursor = make_conn()
            cursor.fetchall.return_value = rows
            with patch.object(processed_chunks, "connect", return_value=conn):
                self.assertEqual(processed_chunks.is_chunk_ready("chunks/v/a.mp4"), expected)

    def test_record_chunk_integrity_does_not_overwrite_existing_ready_status(self):
        conn, cursor = make_conn()
        with patch.object(processed_chunks, "connect", return_value=conn):
            processed_chunks.record_chunk_integrity(
                "chunks/v/a.mp4", "v",
                {"scene_count": 2, "vector_frame_count": 2, "behavior_event_count": 0},
            )

        self.assertIn("ON DUPLICATE KEY UPDATE", cursor.execute.call_args.args[0])
        conn.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class MarkChunkStateTests(unittest.TestCase):
    def test_records_state_with_error_and_never_downgrades_ready_in_sql(self):
        conn, cursor = make_conn()
        with patch.object(processed_chunks, "connect", return_value=conn):
            ok = processed_chunks.mark_chunk_state("chunks/v/a.mp4", "v", "failed", error="RuntimeError: boom")

        self.assertTrue(ok)
        sql, params = cursor.execute.call_args.args
        self.assertIn("chunk_processing_status", sql)
        self.assertIn("IF(state = 'ready'", sql)
        self.assertEqual(params, ("chunks/v/a.mp4", "v", "failed", "RuntimeError: boom"))
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    def test_long_error_message_is_truncated(self):
        conn, cursor = make_conn()
        with patch.object(processed_chunks, "connect", return_value=conn):
            processed_chunks.mark_chunk_state("chunks/v/a.mp4", "v", "failed", error="x" * 5000)
        self.assertEqual(len(cursor.execute.call_args.args[1][3]), 1000)

    def test_status_write_failure_does_not_raise(self):
        conn, cursor = make_conn(RuntimeError("db down"))
        with patch.object(processed_chunks, "connect", return_value=conn):
            self.assertFalse(processed_chunks.mark_chunk_state("chunks/v/a.mp4", "v", "processing"))
        with patch.object(processed_chunks, "connect", side_effect=RuntimeError("no connection")):
            self.assertFalse(processed_chunks.mark_chunk_state("chunks/v/a.mp4", "v", "queued"))

    def test_ready_cannot_be_set_through_progress_marker(self):
        with self.assertRaises(ValueError):
            processed_chunks.mark_chunk_state("chunks/v/a.mp4", "v", "ready")
