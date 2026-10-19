#!/usr/bin/env python3
"""Check native versions, licenses, documented API files and actual C exports."""
from __future__ import annotations
import argparse, ctypes, json, re, subprocess, tomllib
from pathlib import Path

def validate_elf_runtime_report(report: str, platform: str | None, name: str) -> None:
    """Validate dynamic dependencies, not arbitrary strings embedded in .rodata.

    Rust's runtime contains compatibility lookup strings even in musl builds;
    scanning every byte would mistake those strings for imported GLIBC symbols.
    readelf's dynamic and version tables describe the actual loader contract.
    """
    # The version table also lists versions DEFINED/exported by the binary.
    # Only undefined dynamic symbols impose a GLIBC requirement on its host.
    imported_symbols = "\n".join(line for line in report.splitlines()
                                 if re.search(r"\bUND\b", line))
    glibc_versions = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r'@GLIBC_([0-9]+)\.([0-9]+)(?:\.([0-9]+))?\b', imported_symbols)
    ]
    needed = re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', report)
    is_musl = (platform and 'musl' in platform) or any('musl' in lib for lib in needed)
    if is_musl:
        if glibc_versions:
            raise RuntimeError(f"{name}: Musl binary unexpectedly references GLIBC symbols: {glibc_versions}")
        if 'libc.so.6' in needed:
            raise RuntimeError(f"{name}: Musl binary unexpectedly links to libc.so.6")
        print(f"Verified musl runtime independence for {name}")
    else:
        if glibc_versions:
            max_v = max(glibc_versions)
            if max_v > (2, 28):
                raise RuntimeError(f"{name}: GLIBC requirement {max_v[0]}.{max_v[1]} exceeds 2.28 minimum portability baseline")
            print(f"Verified glibc runtime dependency baseline for {name} (max {max_v[0]}.{max_v[1]} <= 2.28)")
        else:
            print(f"Verified ELF binary {name} without versioned GLIBC symbols")

def verify_runtime_dependencies(path: Path, platform: str | None = None) -> None:
    with path.open('rb') as binary:
        if binary.read(4) != b'\x7fELF':
            return
    result = subprocess.run(
        ['readelf', '--dynamic', '--dyn-syms', '--version-info', '--wide', str(path)],
        check=True, capture_output=True, text=True,
    )
    try:
        validate_elf_runtime_report(result.stdout, platform, path.name)
    except RuntimeError:
        print(result.stdout, flush=True)
        raise

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--addon',type=Path)
    parser.add_argument('--platform')
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
    platform_cjs = (root/'bindings/node/platform.cjs').read_text(encoding='utf-8')
    for target in ['win32-x64-msvc', 'win32-arm64-msvc', 'darwin-x64', 'darwin-arm64', 'linux-x64-gnu', 'linux-arm64-gnu', 'linux-x64-musl', 'linux-arm64-musl']:
        assert f"'{target}'" in platform_cjs, f"Missing {target} in bindings/node/platform.cjs"
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
        verify_runtime_dependencies(args.library.resolve(), args.platform)
        # Library verification runs on the target runtime (inside its Linux
        # container where applicable). A loader failure is a failed check,
        # not evidence that the exports or runtime dependencies are correct.
        library=ctypes.CDLL(str(args.library.resolve()))
        for name in symbols:assert getattr(library,name),name
        library.glrmask_abi_version.restype=ctypes.c_uint32
        library.glrmask_version.restype=ctypes.c_char_p
        assert library.glrmask_abi_version()==1
        assert library.glrmask_version().decode()==version
    if args.addon:
        verify_runtime_dependencies(args.addon.resolve(), args.platform)
    print(f'Native source/version/license/docs checks passed; {len(symbols)} declared C symbols'+(' verified in binary' if args.library else ''))

if __name__=='__main__':main()
