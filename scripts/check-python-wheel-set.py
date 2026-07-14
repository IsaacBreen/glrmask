#!/usr/bin/env python3
"""Verify that an aggregated release directory contains the intended wheel matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from packaging.utils import parse_wheel_filename

PYTHONS = {
    "cp39": "3.9",
    "cp310": "3.10",
    "cp311": "3.11",
    "cp312": "3.12",
    "cp313": "3.13",
}
PLATFORMS = {
    "manylinux-x86_64": lambda platform: "manylinux" in platform and platform.endswith("x86_64"),
    "manylinux-aarch64": lambda platform: "manylinux" in platform and platform.endswith("aarch64"),
    "macos-x86_64": lambda platform: platform.startswith("macosx_") and platform.endswith("x86_64"),
    "macos-arm64": lambda platform: platform.startswith("macosx_") and platform.endswith("arm64"),
    "windows-x86_64": lambda platform: platform == "win_amd64",
}


def classify(path: Path) -> tuple[str, str]:
    _, _, _, tags = parse_wheel_filename(path.name)
    python_tags = {tag.interpreter for tag in tags}
    if len(python_tags) != 1:
        raise SystemExit(f"unexpected Python tags in {path.name}: {sorted(python_tags)}")
    python_tag = next(iter(python_tags))
    if python_tag not in PYTHONS:
        raise SystemExit(f"unexpected Python tag in {path.name}: {python_tag}")

    platforms = {tag.platform for tag in tags}
    matches = [
        name
        for name, predicate in PLATFORMS.items()
        if any(predicate(platform) for platform in platforms)
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"could not uniquely classify platform for {path.name}: {sorted(platforms)} -> {matches}"
        )
    return python_tag, matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    wheels = sorted(args.directory.glob("*.whl"))
    sdists = sorted(args.directory.glob("*.tar.gz"))
    expected_wheels = len(PYTHONS) * len(PLATFORMS)
    if len(wheels) != expected_wheels:
        raise SystemExit(f"expected {expected_wheels} wheels, found {len(wheels)}")
    if len(sdists) != 1:
        raise SystemExit(f"expected exactly one sdist, found {len(sdists)}")

    observed = Counter(classify(wheel) for wheel in wheels)
    expected = Counter((python_tag, platform) for python_tag in PYTHONS for platform in PLATFORMS)
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise SystemExit(f"wheel matrix mismatch; missing={dict(missing)}, extra={dict(extra)}")

    print(f"validated {len(wheels)} wheels and {len(sdists)} sdist")


if __name__ == "__main__":
    main()
