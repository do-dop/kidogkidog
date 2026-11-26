import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "ci" / "deploy_target.sh"


class DeployTargetTests(unittest.TestCase):
    """연달아 병합한 상황에서 빠진 변경이 다음 배포에 포함되는지 임시 저장소로 확인한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.email", "t@example.com")
        self._git("config", "user.name", "t")
        self.base = self._commit("readme.md")

    def tearDown(self):
        self.tmp.cleanup()

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True, text=True).stdout.strip()

    def _commit(self, path, text="x"):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
        self._git("add", "-A")
        self._git("commit", "-qm", path)
        return self._git("rev-parse", "HEAD")

    def _decide(self, component, deployed, event="workflow_run"):
        output = subprocess.run(
            ["bash", str(SCRIPT), component, deployed, event],
            cwd=self.repo, check=True, capture_output=True, text=True, env={**os.environ},
        ).stdout
        return dict(line.split("=", 1) for line in output.splitlines())["deploy"]

    def test_skipped_web_change_is_picked_up_by_a_later_api_only_commit(self):
        self._commit("ui-react/src/App.css")  # 이 배포가 취소됐다고 가정
        self._commit("pipeline/search.py")
        self.assertEqual(self._decide("web", self.base), "true")
        self.assertEqual(self._decide("api", self.base), "true")

    def test_unrelated_changes_do_not_deploy(self):
        self._commit("docs/a.md")
        self.assertEqual(self._decide("web", self.base), "false")
        self.assertEqual(self._decide("api", self.base), "false")

    def test_only_the_changed_service_deploys(self):
        self._commit("requirements.txt")
        self.assertEqual(self._decide("api", self.base), "true")
        self.assertEqual(self._decide("web", self.base), "false")

    def test_same_commit_is_not_redeployed(self):
        self.assertEqual(self._decide("api", self.base), "false")

    def test_older_commit_never_replaces_a_newer_deployment(self):
        newer = self._commit("api/main.py")
        self._git("checkout", "-q", self.base)
        self.assertEqual(self._decide("api", newer), "false")

    def test_manual_run_always_deploys(self):
        self.assertEqual(self._decide("web", self.base, "workflow_dispatch"), "true")

    def test_unknown_deployed_version_deploys(self):
        for deployed in ("", "lab-901fae3-10041323", "0" * 40):
            self.assertEqual(self._decide("api", deployed), "true", deployed)

    def test_file_moved_out_of_a_service_path_still_deploys_it(self):
        deployed = self._commit("ui-react/src/note.ts", "same content\n")
        (self.repo / "docs").mkdir(exist_ok=True)
        self._git("mv", "ui-react/src/note.ts", "docs/note.ts")
        self._git("commit", "-qm", "move")
        self.assertEqual(self._decide("web", deployed), "true")

    def test_deployment_from_another_branch_is_compared_by_content(self):
        self._git("checkout", "-qb", "side")
        side = self._commit("ui-react/side.ts")
        self._git("checkout", "-q", "main")
        self._commit("docs/b.md")
        self.assertEqual(self._decide("web", side), "true")
        self.assertEqual(self._decide("api", side), "false")


if __name__ == "__main__":
    unittest.main()
