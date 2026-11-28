#!/usr/bin/env python3
"""Verify stable or release-candidate metadata and tag ancestry."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-rc\.(0|[1-9]\d*))?\Z", re.ASCII)


def versions_for_tag(tag: str) -> tuple[str, str, bool]:
    match = VERSION.fullmatch(tag)
    if not match:
        raise ValueError(f"Expected vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rc.N; got {tag!r}")
    base = ".".join(match.groups()[:3])
    rc = match.group(4)
    return tag[1:], base if rc is None else f"{base}rc{rc}", rc is not None


def verify(tag: str, *, check_ancestry: bool) -> str:
    version, python_version, prerelease = versions_for_tag(tag)
    rust = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    python = tomllib.loads((ROOT / "python/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    python_crate = tomllib.loads((ROOT / "python/Cargo.toml").read_text(encoding="utf-8"))
    python_crate_version = python_crate["package"]["version"]
    python_crate_pin = python_crate["dependencies"]["glrmask"]["version"]
    if not python_crate_pin.startswith("="):
        raise ValueError("Unexpected glrmask requirement in python/Cargo.toml")
    versions = {
        "rust": rust,
        "python-crate": python_crate_version,
        "python-crate-pin": python_crate_pin[1:],
    }
    node_info = json.loads((ROOT / "bindings/node/package.json").read_text(encoding="utf-8"))
    if node_info["name"] != "@glrmask/node":
        raise ValueError("Unexpected npm package identity")
    cmake_match = re.search(r"^project\(glrmask VERSION ([^ )]+)",
                            (ROOT / "bindings/cpp/CMakeLists.txt").read_text(encoding="utf-8"), re.MULTILINE)
    if not cmake_match:
        raise ValueError("Unexpected CMake project declaration")
    numeric_version = version.split("-")[0]
    if cmake_match.group(1) != numeric_version:
        raise ValueError(f"CMake numeric version must equal {numeric_version}")
    for label, path, pattern in (
        ("cmake", "bindings/cpp/CMakeLists.txt", r'^set\(GLRMASK_VERSION_STRING "([^"]+)"\)'),
        ("c-header", "bindings/cpp/include/glrmask.h", r'^#define GLRMASK_VERSION_STRING "([^"]+)"'),
    ):
        match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE)
        if not match:
            raise ValueError(f"Missing full release version in {path}")
        versions[label] = match.group(1)
    versions["npm"] = node_info["version"]
    lock = json.loads((ROOT / "bindings/node/package-lock.json").read_text(encoding="utf-8"))
    versions["npm-lock"] = lock["version"]
    versions["npm-lock-root"] = lock["packages"][""]["version"]
    if lock["packages"][""]["optionalDependencies"] != node_info["optionalDependencies"]:
        raise ValueError("npm lockfile platform pins do not match package.json")
    for name, pin in node_info["optionalDependencies"].items():
        versions[f"npm-lock:{name}"] = lock["packages"][f"node_modules/{name}"]["version"]
    for name in ("core", "cpp", "node"):
        manifest = tomllib.loads((ROOT / f"bindings/{name}/Cargo.toml").read_text(encoding="utf-8"))
        versions[f"binding-{name}"] = manifest["package"]["version"]
    for name, pin in node_info["optionalDependencies"].items():
        versions[f"npm:{name}"] = pin
    if python != python_version or any(v != version for v in versions.values()):
        raise ValueError(f"Release {tag} disagrees with package versions: {versions}, Python={python!r} (expected {python_version!r})")

    if check_ancestry:
        ref = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, text=True).strip()
        try:
            subprocess.run(["git", "merge-base", "--is-ancestor", ref, "origin/main"],
                           cwd=ROOT, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError("Release ref is not an ancestor of origin/main") from e
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--skip-ancestry", action="store_true", help="Metadata-only PR checks")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()
    version = verify(args.tag, check_ancestry=not args.skip_ancestry)
    _, python_version, prerelease = versions_for_tag(args.tag)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"version={version}\npython_version={python_version}\nprerelease={str(prerelease).lower()}\n")
    print(f"Release metadata validated: {args.tag}, Rust={version}, Python={python_version}, native={version}")


if __name__ == "__main__":
    main()
