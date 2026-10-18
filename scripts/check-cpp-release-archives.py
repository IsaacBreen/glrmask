#!/usr/bin/env python3
"""Check tested C++ build archives and their checksums before upload to GitHub Releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

TARGETS = {"darwin-arm64", "darwin-x64", "linux-arm64-gnu",
           "linux-x64-gnu", "win32-arm64-msvc", "win32-x64-msvc"}
KINDS = {"static", "shared"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("version")
    a = p.parse_args()
    base = a.directory
    seen = set()
    for target in sorted(TARGETS):
        folder = base / f"cpp-native-{target}"
        checksum_file = folder / "checksums.json"
        records = json.loads(checksum_file.read_text(encoding="utf-8"))
        if len(records) != 2:
            raise RuntimeError(f"{target}: expected exactly 2 archives")
        for rec in records:
            name = rec["name"]
            kind = next((k for k in KINDS if name == f"glrmask-{a.version}-{target}-{k}.tar.gz"), None)
            if kind is None or (target, kind) in seen:
                raise RuntimeError(f"Unexpected/duplicate archive: {name}")
            seen.add((target, kind))
            file = folder / name
            actual = hashlib.sha256(file.read_bytes()).hexdigest()
            if rec["sha256"] != actual or rec["bytes"] != file.stat().st_size:
                raise RuntimeError(f"Archive checksum/size mismatch: {name}")
            expected_root = name.removesuffix(".tar.gz")
            with tarfile.open(file, "r:gz") as archive:
                members = archive.getmembers()
                if (not members or any(
                    m.issym() or m.islnk() or m.name.startswith("/")
                    or ".." in m.name.split("/")
                    or not (m.name == expected_root or m.name.startswith(expected_root + "/"))
                    for m in members
                )):
                    raise RuntimeError(f"Unsafe or misplaced archive members: {name}")
                manifest = archive.extractfile(f"{expected_root}/native-manifest.json")
                if manifest is None:
                    raise RuntimeError(f"Missing native-manifest.json: {name}")
                manifest_obj = json.load(manifest)
                if (manifest_obj.get("version"), manifest_obj.get("platform"),
                    manifest_obj.get("kind")) != (a.version, target, kind):
                    raise RuntimeError(f"Wrong embedded manifest: {name}")
                if not manifest_obj.get("files"):
                    raise RuntimeError(f"Empty native installation: {name}")
            print(f"Verified {name}: {file.stat().st_size} bytes, sha256:{actual}")
    if seen != {(t, k) for t in TARGETS for k in KINDS}:
        raise RuntimeError("Incomplete native release matrix")
    print(f"Complete 6-platform, static/shared C++ release set: {len(seen)} archives")


if __name__ == "__main__":
    main()
