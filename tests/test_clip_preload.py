import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from pipeline.models import clip_embedder


class LoadOnceTests(unittest.TestCase):
    def setUp(self):
        clip_embedder._model = None
        clip_embedder._preprocess = None

    def tearDown(self):
        clip_embedder._model = None
        clip_embedder._preprocess = None

    def test_concurrent_callers_load_the_model_only_once(self):
        # 앱 시작 시 미리 로드하는 스레드와 첫 요청이 동시에 들어오는 상황
        calls = []

        def slow_load(name, device=None):
            calls.append(name)
            time.sleep(0.2)
            return object(), object()

        results = []
        with patch.object(clip_embedder.clip, "load", side_effect=slow_load):
            threads = [
                threading.Thread(target=lambda: results.append(clip_embedder.load_clip_model()))
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result is results[0] or result == results[0] for result in results))

    def test_loaded_model_is_reused_without_taking_the_lock(self):
        model, preprocess = object(), object()
        clip_embedder._model, clip_embedder._preprocess = model, preprocess

        with patch.object(clip_embedder.clip, "load") as load:
            self.assertEqual(clip_embedder.load_clip_model(), (model, preprocess))

        load.assert_not_called()


class StartupPreloadTests(unittest.TestCase):
    def test_startup_starts_a_background_preload_by_default(self):
        from api import main

        with patch.object(main, "init_db"), \
                patch.object(main.threading, "Thread") as thread, \
                patch.dict("os.environ", {}, clear=False):
            main.os.environ.pop("PRELOAD_CLIP", None)
            main.startup()

        thread.assert_called_once()
        self.assertIs(thread.call_args.kwargs["target"], main._preload_clip)
        self.assertTrue(thread.call_args.kwargs["daemon"])
        thread.return_value.start.assert_called_once()

    def test_preload_can_be_disabled(self):
        from api import main

        with patch.object(main, "init_db"), \
                patch.object(main.threading, "Thread") as thread, \
                patch.dict("os.environ", {"PRELOAD_CLIP": "false"}):
            main.startup()

        thread.assert_not_called()

    def test_preload_failure_does_not_stop_the_service(self):
        from api import main

        with patch.object(clip_embedder, "load_clip_model", side_effect=RuntimeError("no weights")):
            main._preload_clip()  # 예외가 밖으로 나오지 않아야 한다

    def test_preload_loads_the_model(self):
        from api import main

        with patch.object(clip_embedder, "load_clip_model", return_value=(MagicMock(), MagicMock())) as load:
            main._preload_clip()

        load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
