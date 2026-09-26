"""Small standard-library tests for the reproducible profile launcher."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "run_boundary_profile.py"
spec = importlib.util.spec_from_file_location("run_boundary_profile", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class BoundaryProfileTests(unittest.TestCase):
    def test_checked_in_profile_is_valid_and_uninstrumented(self):
        values = profile.load_profile(profile.DEFAULT_PROFILE)
        self.assertEqual(values["GLRMASK_BOUNDARY_NATIVE_SINGLE_CLASS"], "1")
        self.assertEqual(values["GLRMASK_BOUNDARY_BORROWED_QUERY"], "1")

    def test_environment_is_local_and_removes_stale_experiments(self):
        original = {"PATH": "/bin", "HOME": "/home/test", "GLRMASK_OLD": "1",
                    "PROBE_OLD": "1", "GLRMASK_VALIDATE_OLD": "1", "RAYON_NUM_THREADS": "3"}
        before = dict(original)
        child = profile.child_environment(original, {"GLRMASK_NEW": "1"}, 10)
        self.assertEqual(original, before)
        self.assertEqual(child, {"PATH": "/bin", "HOME": "/home/test",
                                 "GLRMASK_NEW": "1", "RAYON_NUM_THREADS": "10"})
        self.assertEqual(profile.child_environment(original, {})["RAYON_NUM_THREADS"], "3")

    def test_bad_profile_inputs_are_rejected(self):
        invalid = [[], {}, {"PATH": "bad"}, {"GLRMASK_X": 1}, {"GLRMASK_X": ""},
                   {"GLRMASK_X": "a\0b"}, {"GLRMASK_DUMP_X": "1"}, {"GLRMASK_PROFILE_X": "1"},
                   {"GLRMASK_VALIDATE_X": "1"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            for value in invalid:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        profile.load_profile(path)
        with self.assertRaises(ValueError):
            profile.child_environment({}, {}, 0)

    def test_dry_run_does_not_execute_and_preserves_arguments(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--threads", "1", "--dry-run",
                                 "--", "definitely-not-an-installed-command", "a b"],
                                capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["command"], ["definitely-not-an-installed-command", "a b"])
        self.assertEqual(data["environment"]["RAYON_NUM_THREADS"], "1")

    def test_command_receives_flags_without_a_shell(self):
        command = [sys.executable, "-c", "import os; print(os.environ['RAYON_NUM_THREADS']); print(os.environ['GLRMASK_BOUNDARY_NATIVE_SINGLE_CLASS'])"]
        result = subprocess.run([sys.executable, str(SCRIPT), "--threads", "2", "--"] + command,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), ["2", "1"])


if __name__ == "__main__":
    unittest.main()
