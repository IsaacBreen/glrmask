#!/usr/bin/env python3
"""Verify stable or Rust/Python release-candidate metadata and tag ancestry."""
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


def verify(tag: str, *, check_ancestry: bool, stable_only: bool = False) -> str:
    version, python_version, prerelease = versions_for_tag(tag)
    if stable_only and prerelease:
        raise ValueError("Native releases require a stable tag")
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
    if not prerelease:
        node_info = json.loads((ROOT / "bindings/node/package.json").read_text(encoding="utf-8"))
        if node_info["name"] != "@glrmask/node":
            raise ValueError("Unexpected npm package identity")
        cmake_match = re.search(r"^project\(glrmask VERSION ([^ )]+)",
                                (ROOT / "bindings/cpp/CMakeLists.txt").read_text(encoding="utf-8"), re.MULTILINE)
        if not cmake_match:
            raise ValueError("Unexpected CMake project declaration")
        versions.update(cmake=cmake_match.group(1), npm=node_info["version"])
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
    parser.add_argument("--stable-only", action="store_true", help="Reject prereleases on native publication paths")
    args = parser.parse_args()
    version = verify(args.tag, check_ancestry=not args.skip_ancestry, stable_only=args.stable_only)
    _, python_version, prerelease = versions_for_tag(args.tag)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"version={version}\npython_version={python_version}\nprerelease={str(prerelease).lower()}\n")
    print(f"Release metadata validated: {args.tag}, Rust={version}, Python={python_version}, native={'not checked (RC)' if prerelease else version}")


if __name__ == "__main__":
    main()
