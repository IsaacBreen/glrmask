#!/usr/bin/env python3
"""Partition normal Rust CI test targets without omitting or duplicating any."""
from pathlib import Path
import json
import subprocess
import sys

base = ["cargo", "test", "--locked", "--profile", "ci-test", "--features", "internal-api"]
metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], text=True))
packages = [p for p in metadata["packages"] if p["name"] in ("glrmask", "glrmask-grammar")]
assert {p["name"] for p in packages} == {"glrmask", "glrmask-grammar"}
targets = []
for package in packages:
    for target in package["targets"]:
        if "test" not in target["kind"]:
            continue
        assert set(target.get("required-features", [])) <= {"internal-api"}, target
        # Source size is only a deterministic balancing heuristic, not timing.
        targets.append((Path(target["src_path"]).stat().st_size, package["name"], target["name"]))
groups = [[], [], []]
sizes = [0, 0, 0]
for size, package, target in sorted(targets, reverse=True):
    index = min(range(len(groups)), key=lambda i: sizes[i])
    groups[index].append((package, target))
    sizes[index] += size
assert sum(map(len, groups)) == len(targets)
assert len(set(item for group in groups for item in group)) == len(targets)
for i, group in enumerate(groups):
    print(f"Shard {i}: {len(group)} targets, {sizes[i]} source bytes", flush=True)
    print(" ".join(f"{package}/{name}" for package, name in sorted(group)), flush=True)

def run(args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)

if len(sys.argv) != 2:
    raise SystemExit("usage: ci-rust-shard.py lib|integration-0|integration-1|integration-2|inventory")
group = sys.argv[1]
if group == "inventory":
    raise SystemExit(0)
if group == "lib":
    for kind in ("--lib", "--doc"):
        run(base + ["-p", "glrmask", "-p", "glrmask-grammar", kind])
elif group in ("integration-0", "integration-1", "integration-2"):
    for package in packages:
        names = sorted(name for pkg, name in groups[int(group[-1])] if pkg == package["name"])
        if names:
            run(base + ["-p", package["name"]] + [arg for name in names for arg in ("--test", name)])
else:
    raise SystemExit(f"unknown shard {group!r}")
