#!/usr/bin/env python3
"""Offline regression tests for the release verifier; no GitHub/registry calls."""
import importlib.util
import io
import tarfile
import zipfile
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

release = load("release", "check-release-version.py")
wheels = load("wheels", "check-python-wheel-set.py")


class Versions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(release, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def fixture(self, rust, python, native="1.0.0"):
        self.write("Cargo.toml", f'[package]\nversion="{rust}"\n')
        self.write("python/Cargo.toml", f'[package]\nversion="{rust}"\n[dependencies]\nglrmask={{version="={rust}"}}\n')
        self.write("python/pyproject.toml", f'[project]\nversion="{python}"\n')
        self.write("bindings/cpp/CMakeLists.txt", f"project(glrmask VERSION {native})\n")
        self.write("bindings/node/package.json", json.dumps({
            "name": "@glrmask/node", "version": native,
            "optionalDependencies": {"@glrmask/node-linux-x64-gnu": native},
        }))
        for part in ("core", "cpp", "node"):
            self.write(f"bindings/{part}/Cargo.toml", f'[package]\nversion="{native}"\n')

    def test_stable(self):
        self.fixture("1.0.0", "1.0.0")
        self.assertEqual(release.verify("v1.0.0", check_ancestry=False, stable_only=True), "1.0.0")
        self.assertEqual(release.versions_for_tag("v1.0.0"), ("1.0.0", "1.0.0", False))

    def test_rc(self):
        self.fixture("1.0.0-rc.1", "1.0.0rc1", "0.3.1")
        self.assertEqual(release.verify("v1.0.0-rc.1", check_ancestry=False), "1.0.0-rc.1")
        self.assertEqual(release.versions_for_tag("v1.0.0-rc.1"), ("1.0.0-rc.1", "1.0.0rc1", True))
        with self.assertRaises(ValueError):
            release.verify("v1.0.0-rc.1", check_ancestry=False, stable_only=True)

    def test_bad_tags(self):
        for tag in ("1.0.0", "v01.0.0", "v1.0.0-rc.01", "v1.0.0-rc1", "v1.0.0rc1",
                    "v1.0.0-alpha.1", "v1.0.0+build", "v1.0.0-rc.1\n", "v1.0.0-rc.\u0661"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.versions_for_tag(tag)

    def test_mismatches(self):
        for rust, python, native, tag in (
            ("1.0.0", "1.0.1", "1.0.0", "v1.0.0"),
            ("1.0.0", "1.0.0", "0.3.1", "v1.0.0"),
            ("1.0.0-rc.1", "1.0.0-rc.1", "1.0.0", "v1.0.0-rc.1"),
            ("1.0.0-rc.1", "1.0.0rc2", "1.0.0", "v1.0.0-rc.1"),
            ("1.0.0-rc.2", "1.0.0rc1", "1.0.0", "v1.0.0-rc.1"),
        ):
            self.fixture(rust, python, native)
            with self.subTest(tag=tag, rust=rust, python=python, native=native), self.assertRaises(ValueError):
                release.verify(tag, check_ancestry=False)
        self.fixture("1.0.0-rc.1", "1.0.0rc1")
        self.write("python/Cargo.toml", '[package]\nversion="1.0.0-rc.1"\n[dependencies]\nglrmask={version="=1.0.0"}\n')
        with self.assertRaises(ValueError):
            release.verify("v1.0.0-rc.1", check_ancestry=False)

    def test_ancestry_not_bypassed_for_rc(self):
        self.fixture("1.0.0-rc.1", "1.0.0rc1")
        with patch.object(release.subprocess, "check_output", return_value="deadbeef\n"), patch.object(release.subprocess, "run") as run:
            release.verify("v1.0.0-rc.1", check_ancestry=True)
            run.assert_called_once_with(["git", "merge-base", "--is-ancestor", "deadbeef", "origin/main"], cwd=self.root, check=True)


class DistributionMetadata(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifacts(self, version, metadata_version, name="glrmask"):
        content = f"Metadata-Version: 2.4\nName: {name}\nVersion: {metadata_version}\n\n".encode()
        wheel = self.root / f"glrmask-{version}-cp39-abi3-win_amd64.whl"
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr(f"glrmask-{version}.dist-info/METADATA", content)
        sdist = self.root / f"glrmask-{version}.tar.gz"
        with tarfile.open(sdist, "w:gz") as t:
            info = tarfile.TarInfo(f"glrmask-{version}/PKG-INFO")
            info.size = len(content)
            t.addfile(info, io.BytesIO(content))
        return wheel, sdist

    def test_metadata(self):
        for version in ("1.0.0", "1.0.0rc1"):
            for path in self.artifacts(version, version):
                wheels.validate_distribution(path, version)
                with self.assertRaises(ValueError):
                    wheels.validate_distribution(path, "9.0.0")

    def test_filename_is_not_sufficient(self):
        for bad in ("1.0.0", "1.0.0rc2", "1.0.0-rc.1"):
            for path in self.artifacts("1.0.0rc1", bad):
                with self.assertRaises(ValueError):
                    wheels.validate_distribution(path, "1.0.0rc1")
        for path in self.artifacts("1.0.0rc1", "1.0.0rc1", name="other"):
            with self.assertRaises(ValueError):
                wheels.validate_distribution(path, "1.0.0rc1")



if __name__ == "__main__":
    unittest.main()
