"""Regression checks for cross-platform distribution descriptions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "check-readme-archives.py"
spec = importlib.util.spec_from_file_location("archive_checker", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReadmeDescriptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_root = checker.ROOT
        checker.ROOT = Path(self.tmp.name)
        (checker.ROOT / "python").mkdir()
        self.readme = "# Example\n\nUnicode: \u03b2 and \u00e9.\n"
        (checker.ROOT / "python/PYTHON.md").write_text(self.readme, encoding="utf8")

    def tearDown(self):
        checker.ROOT = self.original_root
        self.tmp.cleanup()

    def metadata(self, body):
        return "Metadata-Version: 2.4\nDescription-Content-Type: text/markdown; charset=UTF-8\n\n" + body

    def test_lf(self):
        checker.description(self.metadata(self.readme))

    def test_crlf(self):
        checker.description(self.metadata(self.readme.replace("\n", "\r\n")))

    def test_real_content_change_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "description differs"):
            checker.description(self.metadata(self.readme.replace("Example", "Changed")))


if __name__ == "__main__":
    unittest.main()
