#!/usr/bin/env python3
"""Fail closed if a GLRMask GitHub release tag disagrees with all package versions."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")


def verify(tag: str, *, check_ancestry: bool) -> str:
    match = VERSION.fullmatch(tag)
    if not match:
        raise ValueError(f"Expected a stable vMAJOR.MINOR.PATCH release tag; got {tag!r}")
    version = tag[1:]
    rust = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    python = tomllib.loads((ROOT / "python/pyproject.toml").read_text())["project"]["version"]
    node_info = json.loads((ROOT / "bindings/node/package.json").read_text())
    js_version = node_info["version"]
    if node_info["name"] != "@glrmask/node":
        raise ValueError("Unexpected npm package identity")
    versions = {"rust": rust, "python": python, "npm": js_version}
    if any(v != version for v in versions.values()):
        raise ValueError(f"Release {tag} disagrees with package versions: {versions}")

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
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"version={version}\n")
    print(f"Release metadata validated: {args.tag}, packages Rust/Python/npm all {version}")


if __name__ == "__main__":
    main()
