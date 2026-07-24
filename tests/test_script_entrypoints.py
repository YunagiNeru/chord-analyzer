from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_direct_script_help_commands_import_project_package(self) -> None:
        scripts = (
            "scripts/live_smoke_test.py",
            "scripts/live_schema_probe.py",
            "scripts/evaluate_accuracy.py",
            "scripts/tune_accuracy_profile.py",
            "scripts/rebuild_review_snapshot.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
