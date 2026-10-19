"""Unit tests for Linux portability: glibc baseline, musl targets, wheel classification, and archive verification."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from subprocess import CalledProcessError

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
            ("glrmask-0.3.1-cp39-abi3-manylinux2014_x86_64.whl", "cp39", "manylinux-x86_64"),
            ("glrmask-0.3.1-cp39-abi3-manylinux2014_aarch64.whl", "cp39", "manylinux-aarch64"),
            ("glrmask-0.3.1-cp39-abi3-musllinux_1_2_x86_64.whl", "cp39", "musllinux-x86_64"),
            ("glrmask-0.3.1-cp39-abi3-musllinux_1_2_aarch64.whl", "cp39", "musllinux-aarch64"),
            ("glrmask-0.3.1-cp39-abi3-macosx_11_0_x86_64.whl", "cp39", "macos-x86_64"),
            ("glrmask-0.3.1-cp39-abi3-macosx_11_0_arm64.whl", "cp39", "macos-arm64"),
            ("glrmask-0.3.1-cp39-abi3-win_amd64.whl", "cp39", "windows-x86_64"),
            ("glrmask-0.3.1-cp39-abi3-win_arm64.whl", "cp39", "windows-arm64"),
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
            "windows-arm64",
        }
        self.assertEqual(set(wheel_checker.PLATFORMS.keys()), expected_platforms)
        self.assertEqual(len(wheel_checker.PYTHONS), 1)
        # 8 platforms * 1 abi3 = 8 total wheels
        self.assertEqual(len(wheel_checker.PLATFORMS) * len(wheel_checker.PYTHONS), 8)


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
            "win32-arm64-msvc",
        }
        self.assertEqual(cpp_checker.TARGETS, expected_targets)
        self.assertEqual(cpp_checker.KINDS, {"static", "shared"})
        # 8 targets * 2 kinds = 16 archives
        self.assertEqual(len(cpp_checker.TARGETS) * len(cpp_checker.KINDS), 16)


class RuntimeDependenciesVerifierTest(unittest.TestCase):
    def test_portable_glibc_allowed(self):
        ci_check.validate_elf_runtime_report(
            "(NEEDED) Shared library: [libc.so.6]\n1: 0 0 FUNC GLOBAL DEFAULT UND old@GLIBC_2.17\n2: 0 0 FUNC GLOBAL DEFAULT UND recent@GLIBC_2.28",
            "linux-x64-gnu", "test.so")

    def test_glibc_exceeding_baseline_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds 2.28"):
            ci_check.validate_elf_runtime_report("1: 0 0 FUNC GLOBAL DEFAULT UND newer@GLIBC_2.35", "linux-x64-gnu", "test.so")

    def test_musl_binary_passes(self):
        ci_check.validate_elf_runtime_report(
            "(NEEDED) Shared library: [libc.musl-aarch64.so.1]", "linux-arm64-musl", "test.so")

    def test_musl_binary_with_glibc_leak_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unexpectedly links to libc.so.6"):
            ci_check.validate_elf_runtime_report(
                "(NEEDED) Shared library: [libc.so.6]", "linux-x64-musl", "test.so")

    def test_musl_versioned_glibc_import_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unexpectedly references GLIBC"):
            ci_check.validate_elf_runtime_report("1: 0 0 FUNC GLOBAL DEFAULT UND foreign@GLIBC_2.17", "linux-arm64-musl", "test.so")

    def test_embedded_lookup_strings_are_not_runtime_imports(self):
        with tempfile.TemporaryDirectory() as work:
            binary = Path(work) / "native.so"
            # A runtime may contain dlsym lookup strings without importing them.
            binary.write_bytes(b"\x7fELF" + b"GLIBC_2.99\x00libc.so.6\x00")
            result = SimpleNamespace(stdout="(NEEDED) Shared library: [libc.musl-x86_64.so.1]")
            with patch.object(ci_check.subprocess, "run", return_value=result) as readelf:
                ci_check.verify_runtime_dependencies(binary, "linux-x64-musl")
                self.assertEqual(readelf.call_args.args[0],
                    ["readelf", "--dynamic", "--dyn-syms", "--version-info", "--wide", str(binary)])
                self.assertTrue(readelf.call_args.kwargs["check"])

    def test_defined_glibc_compatibility_symbols_are_not_imports(self):
        report = """(NEEDED) Shared library: [libc.musl-aarch64.so.1]
  1: 00001000 4 FUNC GLOBAL DEFAULT 12 compat@GLIBC_2.0
Version definition section '.gnu.version_d':
  Name: GLIBC_2.0
"""
        ci_check.validate_elf_runtime_report(report, "linux-arm64-musl", "exports.so")

    def test_broken_elf_inspection_fails_closed(self):
        with tempfile.TemporaryDirectory() as work:
            binary = Path(work) / "broken.so"
            binary.write_bytes(b"\x7fELF")
            with patch.object(ci_check.subprocess, "run", side_effect=CalledProcessError(1, ["readelf"])):
                with self.assertRaises(CalledProcessError):
                    ci_check.verify_runtime_dependencies(binary, "linux-x64-gnu")


if __name__ == "__main__":
    unittest.main()
