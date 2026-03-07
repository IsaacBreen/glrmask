#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

OLD_ATTRS = (
    '#![allow(unused_imports, unused_variables, dead_code)]\n',
    '#![allow(unused_imports, unused_variables, unused_mut, dead_code)]\n',
)
NARROW_ATTRS = (
    '#![allow(unused_imports)]\n',
    '#![allow(unused_variables)]\n',
    '#![allow(unused_mut)]\n',
    '#![allow(dead_code)]\n',
)
OLD_UNIMP = 'unimplemented!("cargo-check-only stub")'
NEW_UNIMP = 'unimplemented!()'


def split_pre_test(text: str) -> tuple[str, str]:
    marker = '#[cfg(test)]'
    idx = text.find(marker)
    if idx == -1:
        return text, ''
    return text[:idx], text[idx:]


def insert_attr(pre: str, attr: str) -> str:
    if attr in pre:
        return pre
    lines = pre.splitlines(keepends=True)
    insert_at = 0
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith('//!') or s.startswith('/*!') or s.startswith('#![') or s == '\n':
            insert_at += len(lines[i])
            i += 1
            continue
        break
    return pre[:insert_at] + attr + pre[insert_at:]


def cleanup_file(path: Path) -> bool:
    original = path.read_text()
    pre, tail = split_pre_test(original)
    cleaned = pre
    for attr in OLD_ATTRS:
        cleaned = cleaned.replace(attr, '')
    for attr in NARROW_ATTRS:
        cleaned = cleaned.replace(attr, '')
    cleaned = cleaned.replace(OLD_UNIMP, NEW_UNIMP)
    attrs_to_add = [
        NARROW_ATTRS[1],
        NARROW_ATTRS[2],
        NARROW_ATTRS[3],
    ]
    if '\nuse ' in cleaned or cleaned.lstrip().startswith('use '):
        attrs_to_add.insert(0, NARROW_ATTRS[0])
    for attr in reversed(attrs_to_add):
        cleaned = insert_attr(cleaned, attr)
    rewritten = cleaned + tail
    if rewritten != original:
        path.write_text(rewritten)
        return True
    return False


def main() -> int:
    root = Path('src')
    changed = 0
    for path in sorted(root.rglob('*.rs')):
        if cleanup_file(path):
            changed += 1
            print(path)
    print(f'changed={changed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
