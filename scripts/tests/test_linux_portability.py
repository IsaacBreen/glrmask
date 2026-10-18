"""Unit tests for Linux portability: glibc baseline, musl targets, wheel classification, and archive verification."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

# Import check-python-wheel-set.py
wheel_spec = importlib.util.spec_from_file_location(
    "wheel_checker", ROOT / "scripts" / "check-python-wheel-set.py"
)
wheel_checker = importlib.util.module_from_spec(wheel_spec)
wheel_spec.loader.exec_module(wheel_checker)

# Import check-cpp-release-archives.py
cpp_spec = importlib.util.spec_from_file_location(
    "cpp_checker", ROOT / "scripts" / "check-cpp-release-archives.py"
)
cpp_checker = importlib.util.module_from_spec(cpp_spec)
cpp_spec.loader.exec_module(cpp_checker)

# Import check.py
ci_check_spec = importlib.util.spec_from_file_location(
    "ci_check", ROOT / "bindings" / "ci" / "check.py"
)
ci_check = importlib.util.module_from_spec(ci_check_spec)
ci_check_spec.loader.exec_module(ci_check)


class WheelMatrixClassificationTest(unittest.TestCase):
    def test_wheel_classification(self):
        samples = [
            ("glrmask-0.3.1-cp39-cp39-manylinux2014_x86_64.whl", "cp39", "manylinux-x86_64"),
            ("glrmask-0.3.1-cp310-cp310-manylinux2014_aarch64.whl", "cp310", "manylinux-aarch64"),
            ("glrmask-0.3.1-cp311-cp311-musllinux_1_2_x86_64.whl", "cp311", "musllinux-x86_64"),
            ("glrmask-0.3.1-cp312-cp312-musllinux_1_2_aarch64.whl", "cp312", "musllinux-aarch64"),
            ("glrmask-0.3.1-cp313-cp313-macosx_11_0_x86_64.whl", "cp313", "macos-x86_64"),
            ("glrmask-0.3.1-cp312-cp312-macosx_11_0_arm64.whl", "cp312", "macos-arm64"),
            ("glrmask-0.3.1-cp311-cp311-win_amd64.whl", "cp311", "windows-x86_64"),
        ]
        for name, expected_py, expected_plat in samples:
            py_tag, plat = wheel_checker.classify(Path(name))
            self.assertEqual(py_tag, expected_py)
            self.assertEqual(plat, expected_plat)

    def test_expected_platforms_count(self):
        expected_platforms = {
            "manylinux-x86_64",
            "manylinux-aarch64",
            "musllinux-x86_64",
            "musllinux-aarch64",
            "macos-x86_64",
            "macos-arm64",
            "windows-x86_64",
        }
        self.assertEqual(set(wheel_checker.PLATFORMS.keys()), expected_platforms)
        self.assertEqual(len(wheel_checker.PYTHONS), 5)
        # 7 platforms * 5 pythons = 35 total wheels
        self.assertEqual(len(wheel_checker.PLATFORMS) * len(wheel_checker.PYTHONS), 35)


class CppReleaseArchivesMatrixTest(unittest.TestCase):
    def test_targets_matrix(self):
        expected_targets = {
            "darwin-arm64",
            "darwin-x64",
            "linux-arm64-gnu",
            "linux-arm64-musl",
            "linux-x64-gnu",
            "linux-x64-musl",
            "win32-x64-msvc",
        }
        self.assertEqual(cpp_checker.TARGETS, expected_targets)
        self.assertEqual(cpp_checker.KINDS, {"static", "shared"})
        # 7 targets * 2 kinds = 14 archives
        self.assertEqual(len(cpp_checker.TARGETS) * len(cpp_checker.KINDS), 14)


class RuntimeDependenciesVerifierTest(unittest.TestCase):
    def test_portable_glibc_allowed(self):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            # Fake ELF with GLIBC_2.17 and GLIBC_2.28
            f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 40 + b"GLIBC_2.2.5\x00GLIBC_2.17\x00GLIBC_2.28\x00")
            f.flush()
            temp_path = Path(f.name)
        try:
            # Should succeed because max glibc is 2.28
            ci_check.verify_runtime_dependencies(temp_path, "linux-x64-gnu")
        finally:
            temp_path.unlink()

    def test_glibc_exceeding_baseline_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            # Fake ELF requiring GLIBC_2.35 (e.g. built on bare Ubuntu 22.04 without container)
            f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 40 + b"GLIBC_2.35\x00")
            f.flush()
            temp_path = Path(f.name)
        try:
            with self.assertRaisesRegex(RuntimeError, "exceeds 2.28 minimum portability baseline"):
                ci_check.verify_runtime_dependencies(temp_path, "linux-x64-gnu")
        finally:
            temp_path.unlink()

    def test_musl_binary_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            # Fake musl ELF
            f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 40 + b"libc.musl-x86_64.so.1\x00musl_alloc\x00")
            f.flush()
            temp_path = Path(f.name)
        try:
            ci_check.verify_runtime_dependencies(temp_path, "linux-x64-musl")
        finally:
            temp_path.unlink()

    def test_musl_binary_with_glibc_leak_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            # Fake musl ELF that accidentally links glibc
            f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 40 + b"libc.musl-x86_64.so.1\x00libc.so.6\x00")
            f.flush()
            temp_path = Path(f.name)
        try:
            with self.assertRaisesRegex(RuntimeError, "unexpectedly links to libc.so.6"):
                ci_check.verify_runtime_dependencies(temp_path, "linux-x64-musl")
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
