#!/usr/bin/env python3
"""Validate the GLRMask internal crate DAG and optionally its crates.io prerequisites.

This script never publishes or authenticates. Only --json-out writes a receipt.
The --require-published mode fails closed before *any* GitHub Release channel runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib
import urllib.error
import urllib.request


def load(path: Path) -> dict:
    with path.open('rb') as source:
        return tomllib.load(source)


def check_registry(name: str, version: str) -> str:
    req = urllib.request.Request(
        f'https://crates.io/api/v1/crates/{name}/{version}',
        headers={'User-Agent': 'GLRMask internal crate release preflight (https://github.com/IsaacBreen/glrmask)'},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.load(resp)
            v = payload.get('version', {})
            if v.get('num') != version:
                raise ValueError(f'{name} returned a different version')
            return 'YANKED' if v.get('yanked') else 'PUBLISHED'
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return 'MISSING'
            if error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(attempt + 1)
    raise AssertionError('unreachable')


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--require-published', action='store_true', help='require non-yanked public crates.io versions')
    cli.add_argument('--json-out', type=Path, help='optional evidence receipt')
    args = cli.parse_args()
    root = Path(__file__).resolve().parent.parent
    root_manifest = load(root / 'Cargo.toml')
    assert root_manifest['package']['name'] == 'glrmask'
    manifests = {p.parent.name: load(p) for p in sorted((root / 'crates').glob('*/Cargo.toml'))}
    crates = {manifest['package']['name']: manifest for manifest in manifests.values()}
    assert len(crates) == len(manifests) and len(crates) > 0, 'duplicate or absent local internal crates'
    errors: list[str] = []
    for owner, manifest in [('glrmask', root_manifest), *crates.items()]:
        for section in ('dependencies', 'build-dependencies'):
            for name, entry in manifest.get(section, {}).items():
                if name not in crates:
                    continue
                v = crates[name]['package']['version']
                if not isinstance(entry, dict) or entry.get('version') != f'={v}' or 'path' not in entry:
                    errors.append(f'{owner} must pin {name} = {v} using path and exact version, got {entry}')
    # All local crates will be packaged to crates.io from this workspace,
    # so the graph must not contain dependency cycles.
    remaining = set(crates)
    layers: list[list[str]] = []
    while remaining:
        layer = sorted(n for n in remaining if not (set(crates[n].get('dependencies', {})) & remaining))
        if not layer:
            errors.append(f'Cycle in internal crate graph: {sorted(remaining)}')
            break
        layers.append(layer)
        remaining.difference_update(layer)
    published = {}
    if args.require_published and not errors:
        for layer in layers:
            for n in layer:
                ver = crates[n]['package']['version']
                try:
                    published[n] = check_registry(n, ver)
                except Exception as exc:
                    errors.append(f'Unable to verify {n}@{ver} at crates.io: {type(exc).__name__}: {exc}')
                    published[n] = 'UNKNOWN'
                if published[n] != 'PUBLISHED':
                    errors.append(f'crates.io dependency unavailable: {n}@{ver} ({published[n]})')
    result = {
        'root': f"glrmask@{root_manifest['package']['version']}",
        'internal_versions': {n: crates[n]['package']['version'] for n in sorted(crates)},
        'publication_layers': layers,
        'registry_status': published,
        'errors': errors,
        'checks': 'All internal workspace pins exact, dependency graph acyclic; registry checked only if requested',
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if errors:
        print('INTERNAL_CRATE_RELEASE_PREFLIGHT_FAILED (no release action taken)', file=sys.stderr)
        return 1
    print('INTERNAL_CRATE_RELEASE_PREFLIGHT_OK (no package published)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())