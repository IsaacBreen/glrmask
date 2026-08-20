#!/usr/bin/env python3
"""Render README.md from GUIDE.md.

The Guide body is authored once in GUIDE.md. README.md adds only the
repository title and README-only License section.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'GUIDE.md'
README = ROOT / 'README.md'
HEADER = '''# GLRMask\n\n<!-- Generated from GUIDE.md. Edit that file, then run `python scripts/sync-guide.py`. -->\n\n'''
LICENSE = '''\n## License\n\nLicensed under either the MIT License or the Apache License, Version 2.0, at your option.\n'''


def rendered() -> str:
    body = SOURCE.read_text().strip() + '\n'
    return HEADER + body + LICENSE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = rendered()
    if args.check:
        actual = README.read_text() if README.exists() else ''
        if actual != expected:
            print('README.md is stale; run: python scripts/sync-guide.py', file=sys.stderr)
            return 1
        return 0
    README.write_text(expected)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
