#!/usr/bin/env python3
"""Package already-tested native C/C++ installations, with content manifests."""
from __future__ import annotations
import argparse, hashlib, json, tarfile
from pathlib import Path
import tomllib

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-work',type=Path,required=True)
    p.add_argument('--platform',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    version=tomllib.loads((root/'Cargo.toml').read_text(encoding='utf-8'))['package']['version']
    if not all(c.isalnum() or c=='-' for c in args.platform): raise ValueError('Invalid platform label')
    report=json.loads((args.install_work/'report.json').read_text(encoding='utf-8'))
    if {v['kind'] for v in report}!={'static','shared'} or not all(v['sourceTests'] and v['relocatedConsumer'] for v in report):
        raise RuntimeError('Both tested installation kinds are required')
    args.output.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for kind in ['static','shared']:
        source=args.install_work/f'relocated-{kind}'
        files=sorted(p for p in source.rglob('*') if p.is_file())
        manifest={'schema':1,'version':version,'platform':args.platform,'kind':kind,'files':[
            {'path':p.relative_to(source).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
        ]}
        prefix=f'glrmask-{version}-{args.platform}-{kind}'
        (source/'native-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        archive=args.output/(prefix+'.tar.gz')
        with tarfile.open(archive,'w:gz') as t:t.add(source,arcname=prefix)
        artifacts.append({'name':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    (args.output/'checksums.json').write_text(json.dumps(artifacts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(artifacts,indent=2))

if __name__=='__main__':main()
