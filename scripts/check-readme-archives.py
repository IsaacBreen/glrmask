#!/usr/bin/env python3
"""Check README selection in local source archives or newly built wheels."""
import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import tomllib
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def description(text):
    metadata=Parser().parsestr(text)
    assert metadata['Description-Content-Type'].startswith('text/markdown')
    # Metadata generated on Windows may preserve CRLF in the description,
    # while Path.read_text uses universal newlines. Compare the same text
    # without treating a platform newline convention as README drift.
    actual = metadata.get_payload().replace('\r\n', '\n').strip()
    expected = (ROOT/'python/PYTHON.md').read_text(encoding='utf8').strip()
    assert actual == expected, 'Python metadata description differs'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crate',type=Path);parser.add_argument('--sdist',type=Path);parser.add_argument('--wheel',type=Path)
    args=parser.parse_args()
    assert any([args.crate,args.sdist,args.wheel]), 'Specify an archive to inspect'
    if args.crate:
        with tarfile.open(args.crate) as tar:
            prefix=tar.getnames()[0].split('/')[0]+'/'
            cargo=tomllib.loads(tar.extractfile(prefix+'Cargo.toml').read().decode())
            assert cargo['package']['readme']=='README.rust.md'
            assert tar.extractfile(prefix+'README.rust.md').read()==(ROOT/'README.rust.md').read_bytes()
        print('Cargo archive selects matching Rust README')
    if args.sdist:
        with tarfile.open(args.sdist) as tar:
            prefix=next(n for n in tar.getnames() if n.endswith('/PKG-INFO')).removesuffix('PKG-INFO')
            metadata=tar.extractfile(prefix+'PKG-INFO').read().decode();description(metadata)
            project=tomllib.loads(tar.extractfile(prefix+'pyproject.toml').read().decode())['project']
            assert tar.extractfile(prefix+project['readme']).read()==(ROOT/'python/PYTHON.md').read_bytes()
            assert tar.extractfile(prefix+'python/README.md').read()==(ROOT/'python/README.md').read_bytes()
            cargo=tomllib.loads(tar.extractfile(prefix+'Cargo.toml').read().decode())
            assert tar.extractfile(prefix+cargo['package']['readme']).read()==(ROOT/'README.rust.md').read_bytes()
        print('Python sdist selects matching Python description and includes root Rust README')
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as archive:
            metadata=next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            description(archive.read(metadata).decode())
        print('Wheel metadata contains matching Python description')


if __name__=='__main__':main()
