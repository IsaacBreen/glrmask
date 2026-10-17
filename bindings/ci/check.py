#!/usr/bin/env python3
"""Check native versions, licenses, documented API files and actual C exports."""
from __future__ import annotations
import argparse, ctypes, json, re, tomllib
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    version=tomllib.loads((root/'Cargo.toml').read_text(encoding='utf-8'))['package']['version']
    for name in ['core','cpp','node']:
        manifest=tomllib.loads((root/f'bindings/{name}/Cargo.toml').read_text(encoding='utf-8'))
        assert manifest['package']['version']==version,name
        assert manifest['package']['publish'] is False,name
    package=json.loads((root/'bindings/node/package.json').read_text(encoding='utf-8'))
    assert package['version']==version
    assert package['name']=='@glrmask/node'
    assert 'dependencies' not in package,'No runtime npm dependency is required'
    for license,source in [('LICENSE-MIT','LICENSE-MIT'),('LICENSE-APACHE','LICENSE-APACHE'),('NOTICE-LARK','crates/glrmask-grammar/NOTICE-LARK')]:
        assert (root/f'bindings/node/{license}').read_bytes().replace(b'\r\n',b'\n')==(root/source).read_bytes().replace(b'\r\n',b'\n'),license
    for name in ['cpp','node']:
        guide=(root/f'bindings/{name}/README.md').read_text(encoding='utf-8')
        assert len(guide)>5000,name
        for target in re.findall(r'\]\(([^)]+)\)',guide):
            if re.match(r'^\w+://',target) or target.startswith('#'):continue
            path=(root/f'bindings/{name}'/target.split('#')[0]).resolve()
            assert path.is_relative_to(root) and path.exists(),target
    header=(root/'bindings/cpp/include/glrmask.h').read_text(encoding='utf-8')
    symbols=set(re.findall(r'GLRMASK_API[^;]+?\b(glrmask_\w+)\(',header))
    assert len(symbols)>=60,len(symbols)
    if args.library:
        library=ctypes.CDLL(str(args.library.resolve()))
        for name in symbols:assert getattr(library,name),name
        library.glrmask_abi_version.restype=ctypes.c_uint32
        library.glrmask_version.restype=ctypes.c_char_p
        assert library.glrmask_abi_version()==1
        assert library.glrmask_version().decode()==version
    print(f'Native source/version/license/docs checks passed; {len(symbols)} declared C symbols'+(' verified in binary' if args.library else ''))

if __name__=='__main__':main()
