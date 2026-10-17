#!/usr/bin/env python3
"""Validate static/shared installs and relocated C/C++ consumers without Rust rebuilds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

def run(*args: str, env: dict[str,str]|None=None) -> None:
    print('+', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), check=True, env=env)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-dir',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--config',default='Release')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    work=args.work_dir.resolve()
    # Only reuse/create explicitly owned destinations; never recursively delete
    # a user's old directory or silently consume a source-tree include path.
    work.mkdir(parents=True,exist_ok=True)
    reports=[]
    for shared in (False,True):
        name='shared' if shared else 'static'
        build=work/f'build-{name}'
        installed=work/f'install-{name}'
        moved=work/f'relocated-{name}'
        source=work/f'consumer-{name}'
        app=work/f'consumer-build-{name}'
        if moved.exists() or source.exists():
            raise RuntimeError(f'Validation destination already exists: {moved} or {source}; use a fresh --work-dir')
        run('cmake','-S',str(root/'bindings/cpp'),'-B',str(build),'-G','Ninja',
            f'-DCMAKE_BUILD_TYPE={args.config}',f'-DGLRMASK_LIBRARY_DIR={args.library_dir.resolve()}',
            f'-DGLRMASK_SHARED={"ON" if shared else "OFF"}', '-DGLRMASK_BUILD_TESTS=ON')
        run('cmake','--build',str(build),'--config',args.config)
        run('ctest','--test-dir',str(build),'-C',args.config,'--output-on-failure')
        run('cmake','--install',str(build),'--config',args.config,'--prefix',str(installed))
        installed.rename(moved)
        shutil.copytree(root/'bindings/cpp/tests/consumer',source)
        for config in moved.rglob('*.cmake'):
            content=config.read_text(encoding='utf-8')
            if str(root).replace('\\','/') in content.replace('\\','/'):
                raise RuntimeError(f'Installed config leaks the source directory: {config}')
        run('cmake','-S',str(source),'-B',str(app),'-G','Ninja',
            f'-DCMAKE_BUILD_TYPE={args.config}',f'-DCMAKE_PREFIX_PATH={moved}')
        run('cmake','--build',str(app),'--config',args.config)
        env=os.environ.copy()
        if shared and sys.platform=='win32':
            env['PATH']=str(moved/'bin')+os.pathsep+env.get('PATH','')
        run('ctest','--test-dir',str(app),'-C',args.config,'--output-on-failure',env=env)
        reports.append({'kind':name,'prefix':str(moved),'sourceTests':True,'relocatedConsumer':True})
    (work/'report.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    print('Static/shared source tests and relocated external C/C++ consumers passed.')

if __name__=='__main__':
    main()
