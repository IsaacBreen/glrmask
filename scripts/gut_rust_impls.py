#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ALLOW_ATTR = '#![allow(unused_imports, unused_variables, unused_mut, dead_code)]\n'
STUB_MSG = 'cargo-check-only stub'


def split_pre_test(text: str) -> tuple[str, str]:
    marker = '#[cfg(test)]'
    idx = text.find(marker)
    if idx == -1:
        return text, ''
    return text[:idx], text[idx:]


def ensure_allow_attr(pre: str) -> str:
    if ALLOW_ATTR in pre:
        return pre
    # Keep leading shebang/doc comments/inner attrs in place, then inject.
    insert_at = 0
    lines = pre.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith('//!') or s.startswith('/*!') or s.startswith('#![') or s == '\n':
            insert_at += len(lines[i])
            i += 1
            continue
        break
    return pre[:insert_at] + ALLOW_ATTR + pre[insert_at:]


def is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == '_'


def line_indent(text: str, pos: int) -> str:
    line_start = text.rfind('\n', 0, pos) + 1
    i = line_start
    while i < len(text) and text[i] in ' \t':
        i += 1
    return text[line_start:i]


def line_start_pos(text: str, pos: int) -> int:
    return text.rfind('\n', 0, pos) + 1


def match_block(text: str, open_brace: int) -> int:
    depth = 1
    i = open_brace + 1
    state = 'code'
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ''
        if state == 'code':
            if ch == '/' and nxt == '/':
                state = 'line_comment'
                i += 2
                continue
            if ch == '/' and nxt == '*':
                state = 'block_comment'
                i += 2
                continue
            if ch == '"':
                state = 'string'
                i += 1
                continue
            if ch == '\'':
                state = 'char'
                i += 1
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
        elif state == 'line_comment':
            if ch == '\n':
                state = 'code'
        elif state == 'block_comment':
            if ch == '*' and nxt == '/':
                state = 'code'
                i += 2
                continue
        elif state == 'string':
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                state = 'code'
        elif state == 'char':
            if ch == '\\':
                i += 2
                continue
            if ch == '\'':
                state = 'code'
        i += 1
    raise ValueError('unmatched brace')


def find_body_start_or_decl_end(text: str, fn_pos: int) -> tuple[int, str] | None:
    i = fn_pos
    state = 'code'
    paren = bracket = angle = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ''
        if state == 'code':
            if ch == '/' and nxt == '/':
                state = 'line_comment'
                i += 2
                continue
            if ch == '/' and nxt == '*':
                state = 'block_comment'
                i += 2
                continue
            if ch == '"':
                state = 'string'
                i += 1
                continue
            if ch == '\'':
                state = 'char'
                i += 1
                continue
            if ch == '(':
                paren += 1
            elif ch == ')':
                paren = max(0, paren - 1)
            elif ch == '[':
                bracket += 1
            elif ch == ']':
                bracket = max(0, bracket - 1)
            elif ch == '<':
                angle += 1
            elif ch == '>':
                angle = max(0, angle - 1)
            elif ch == ';' and paren == bracket == angle == 0:
                return i, 'decl'
            elif ch == '{' and paren == bracket == angle == 0:
                return i, 'body'
        elif state == 'line_comment':
            if ch == '\n':
                state = 'code'
        elif state == 'block_comment':
            if ch == '*' and nxt == '/':
                state = 'code'
                i += 2
                continue
        elif state == 'string':
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                state = 'code'
        elif state == 'char':
            if ch == '\\':
                i += 2
                continue
            if ch == '\'':
                state = 'code'
        i += 1
    return None


def rewrite_pre_test(pre: str) -> str:
    pre = ensure_allow_attr(pre)
    out: list[str] = []
    i = 0
    last = 0
    state = 'code'
    while i < len(pre):
        ch = pre[i]
        nxt = pre[i + 1] if i + 1 < len(pre) else ''
        if state == 'code':
            if ch == '/' and nxt == '/':
                state = 'line_comment'
                i += 2
                continue
            if ch == '/' and nxt == '*':
                state = 'block_comment'
                i += 2
                continue
            if ch == '"':
                state = 'string'
                i += 1
                continue
            if ch == '\'':
                state = 'char'
                i += 1
                continue
            if pre.startswith('fn', i) and (i == 0 or not is_ident_char(pre[i - 1])) and (i + 2 >= len(pre) or not is_ident_char(pre[i + 2])):
                found = find_body_start_or_decl_end(pre, i)
                if not found:
                    break
                pos, kind = found
                if kind == 'decl':
                    i = pos + 1
                    continue
                close = match_block(pre, pos)
                indent = line_indent(pre, i)
                sig_text = pre[line_start_pos(pre, i):pos]
                body = f'unimplemented!("{STUB_MSG}")'
                if 'const fn' in sig_text:
                    body = 'loop {}'
                elif '-> impl Iterator' in sig_text:
                    body = 'std::iter::empty()'
                out.append(pre[last:pos + 1])
                out.append(f"\n{indent}    {body}\n{indent}}}")
                last = close + 1
                i = close + 1
                continue
        elif state == 'line_comment':
            if ch == '\n':
                state = 'code'
        elif state == 'block_comment':
            if ch == '*' and nxt == '/':
                state = 'code'
                i += 2
                continue
        elif state == 'string':
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                state = 'code'
        elif state == 'char':
            if ch == '\\':
                i += 2
                continue
            if ch == '\'':
                state = 'code'
        i += 1
    out.append(pre[last:])
    return ''.join(out)


def rewrite_file(path: Path) -> bool:
    original = path.read_text()
    pre, tail = split_pre_test(original)
    rewritten = rewrite_pre_test(pre) + tail
    if rewritten != original:
        path.write_text(rewritten)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='src')
    parser.add_argument('--include', nargs='*', default=None, help='Optional relative .rs file paths to limit rewriting')
    args = parser.parse_args()

    root = Path(args.root)
    files: list[Path]
    if args.include:
        files = [root / rel if not Path(rel).is_absolute() else Path(rel) for rel in args.include]
    else:
        files = sorted(p for p in root.rglob('*.rs') if 'target/' not in str(p))

    changed = 0
    for path in files:
        if rewrite_file(path):
            changed += 1
            print(path)
    print(f'changed={changed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
