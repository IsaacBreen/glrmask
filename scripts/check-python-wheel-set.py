#!/usr/bin/env python3
"""Verify that an aggregated release directory contains the intended wheel matrix."""

from __future__ import annotations

import argparse
import email.parser
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from packaging.utils import parse_wheel_filename, parse_sdist_filename, canonicalize_name
from packaging.version import Version

PYTHONS = {
    "cp39": "3.9",
}
PLATFORMS = {
    "manylinux-x86_64": lambda platform: "manylinux" in platform and platform.endswith("x86_64"),
    "manylinux-aarch64": lambda platform: "manylinux" in platform and platform.endswith("aarch64"),
    "musllinux-x86_64": lambda platform: "musllinux" in platform and platform.endswith("x86_64"),
    "musllinux-aarch64": lambda platform: "musllinux" in platform and platform.endswith("aarch64"),
    "macos-x86_64": lambda platform: platform.startswith("macosx_") and platform.endswith("x86_64"),
    "macos-arm64": lambda platform: platform.startswith("macosx_") and platform.endswith("arm64"),
    "windows-x86_64": lambda platform: platform == "win_amd64",
    "windows-arm64": lambda platform: platform == "win_arm64",
}


def classify(path: Path) -> tuple[str, str]:
    _, _, _, tags = parse_wheel_filename(path.name)
    python_tags = {tag.interpreter for tag in tags}
    abi_tags = {tag.abi for tag in tags}
    if len(python_tags) != 1:
        raise SystemExit(f"unexpected Python tags in {path.name}: {sorted(python_tags)}")
    if abi_tags != {"abi3"}:
        raise SystemExit(f"unexpected ABI tags in {path.name}: {sorted(abi_tags)}, expected abi3")
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


def validate_distribution(path: Path, expected: str) -> None:
    if path.suffix == ".whl":
        name, version, _, _ = parse_wheel_filename(path.name)
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
            if len(members) != 1:
                raise ValueError(f"{path}: expected exactly one wheel METADATA")
            raw = archive.read(members[0])
    else:
        name, version = parse_sdist_filename(path.name)
        with tarfile.open(path, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
            if len(members) != 1:
                raise ValueError(f"{path}: expected exactly one root PKG-INFO")
            with archive.extractfile(members[0]) as info:
                raw = info.read()
    metadata = email.parser.BytesParser().parsebytes(raw, headersonly=True)
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if (name != "glrmask" or version != Version(expected)
            or len(names) != 1 or canonicalize_name(names[0]) != "glrmask"
            or versions != [expected]):
        raise ValueError(f"{path}: expected glrmask {expected}, filename={name} {version}, metadata={names} {versions}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", help="Require this canonical Python version in every distribution")
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

    if args.version is not None:
        if str(Version(args.version)) != args.version:
            raise ValueError("Expected a canonical PEP 440 version")
        if set(args.directory.rglob("*")) - set(wheels) - set(sdists):
            raise ValueError("Unexpected files in release distribution directory")
        for path in wheels + sdists:
            validate_distribution(path, args.version)
    print(f"validated {len(wheels)} wheels and {len(sdists)} sdist")


if __name__ == "__main__":
    main()
