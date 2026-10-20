#!/usr/bin/env python3
"""Generate shared benchmark Markdown, SVGs and portable registry PNGs.

Reads reviewed aggregates only. Never runs a benchmark. --render-png requires
rsvg-convert; --check verifies source hashes and PNG integrity without it.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT/'docs/content'
MODES = [('static', 'GLRMask'), ('dynamic', 'GLRMask dynamic'), ('llguidance', 'llguidance')]
AWS_JSON_SHA256 = '770bcc50c47774b86f8b95ccb131c3bb5943d089bab3629e6075d6250ee58486'
AWS_JS_SHA256 = '8814aaa858ac642428d98b32fc874a39fb7ebfb6c79a15d222a8ef3e006aa87a'
PROVENANCE_SHA256 = '11f5b4704ecf9279d3d6b83c6a183e8ec7666a4eef8e7c58bf12256789e2cfbc'
JS_PROVENANCE_SHA256 = '3653f392561164fd306df5715ce9eafffef1dedeaa7034e1ef2e504560a92dbd'


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_historical():
    data = json.loads((ROOT/'docs/benchmarks/json-schema-2026-10-08.json').read_text(encoding="utf-8"))
    prov = {k:v for k,v in data.items() if k not in ['tail','modes']}
    prov['modes'] = {mode:{k:v for k,v in row.items() if k!='statistics_ns'} for mode,row in data['modes'].items()}
    assert digest(prov)==PROVENANCE_SHA256, 'JSON provenance changed; review report and prose before updating pin'
    js = json.loads((ROOT/'docs/benchmarks/javascript-2026-10-09.json').read_text(encoding="utf-8"))
    prov = {k:v for k,v in js.items() if k not in ['tail','tables','ttfm_single']}
    prov['ttfm_single'] = {k:v for k,v in js['ttfm_single'].items() if k!='values_ms'}
    assert digest(prov)==JS_PROVENANCE_SHA256, 'JavaScript provenance changed; review report and prose before updating pin'
    for tail in [data['tail'],js['tail']]:
        xs=tail['exceedance_pct']
        assert all(0<x<=100 for x in xs) and all(a<b for a,b in zip(xs,xs[1:]))
        for mode,_ in MODES:
            values=tail[mode+'_us']
            assert len(values)==len(xs) and all(math.isfinite(v) and v>0 for v in values)
            assert all(a>=b for a,b in zip(values,values[1:]))
    return data, js


def load():
    # Historical pins stay checked; current snapshots have separate identities.
    load_historical()
    data=json.loads((ROOT/'docs/benchmarks/json-schema-2026-10-14.json').read_text(encoding='utf-8'))
    js=json.loads((ROOT/'docs/benchmarks/javascript-2026-10-14.json').read_text(encoding='utf-8'))
    assert digest(data)==AWS_JSON_SHA256, 'Native JSON snapshot changed; verify native export and outcomes'
    assert digest(js)==AWS_JS_SHA256, 'Native JavaScript snapshot changed; verify native export and outcomes'
    for record in [data,js]:
        assert record['schema']=='glrmask-native-publication-v2'
        xs=record['tail']['exceedance_pct']
        assert all(0<x<=100 for x in xs) and all(a<b for a,b in zip(xs,xs[1:]))
        for mode in ['static','dynamic','balanced','llguidance']:
            ys=record['tail'][mode+'_us']
            assert len(xs)==len(ys) and all(math.isfinite(y) and y>0 for y in ys)
            assert all(a>=b for a,b in zip(ys,ys[1:]))
    return data,js


def table(rows, column='Metric', names=('GLRMask','llguidance')):
    return '\n'.join(['| '+column+' | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)] +
                     ['| '+label.replace(' ', '&nbsp;')+' | '+' | '.join(f'{v:,.1f}' for v in values)+' |' for label,values in rows])


def markdown(data, js):
    names=('Static','Dynamic','llguidance')
    modes=('static','dynamic','llguidance')
    def rows(record,metric,scale):
        return [(label,[record['modes'][m]['statistics_ns'][metric][key]/scale for m in modes])
                for key,label in [('mean','Mean'),('median','p50'),('p90','p90'),('p99','p99'),('p99_9','p99.9'),('max','Max')]]
    sections=[]
    for record,label,placeholder in [(data,'JSON Schema','json'),(js,'JavaScript','javascript')]:
        sections.append('### '+label+'\n\nTime between masks. Lower is better.\n\n{{'+placeholder+'-plot}}\n\n**TBM (µs)**\n\n'+table(rows(record,'tbm',1e3),'Statistic',names)+'\n\n**Time to first mask (ms)**\n\n'+table(rows(record,'ttfm',1e6),'Statistic',names)+f"\n\nStatic is `MaskStrategy.FAST_RUNTIME`; Dynamic is `MaskStrategy.FAST_BUILD`.\n\n14 October 2026 · {record['complete_valid_trajectories']:,} complete valid trajectories accepted by all measured engines · {record['tbm_positions']:,} TBM positions per engine · {record['common_schemas']:,} common-built cases for TTFM. One unpinned native wall-clock timing run on AWS c8i.8xlarge (Intel Xeon 6975P-C, 32 available CPUs). Vocabulary preparation is excluded.\n")
    return '\n'.join(sections)+"\n#### Dynamic mode\n\n{{dynamic}}\n\n{{json-dynamic-plot}}\n\n[Explore Dynamic JSON results](https://glrmask.com/?dataset=json&mode=dynamic#benchmarks).\n\nThere was no CPU pinning or SMT sibling reservation, and checkpoint publication overlapped the benchmark. Saved telemetry cannot isolate scheduling effects; latency stabilization is not certified. Successful builds do not establish semantic correctness. All 9,558 JSON schemas were attempted: build errors are Dynamic 232, Static 233 and llguidance 1,226. Incorrect examples are 460 for each GLRMask mode and 100 for llguidance; runtime-error examples are 0 and 6 respectively. JavaScript attempts all 55 fixtures: all builds succeed, with 1 incorrect example per GLRMask mode and 4 for llguidance. Headline TBM excludes incomplete or nonaccepted valid trajectories; all outcomes remain in the raw records. This finite corpus does not establish full-language equivalence or whole-corpus dominance.\n\n[Native AWS benchmark details](docs/benchmark-aws-2026-10-14.md). Historical [October 8 Mac JSON](docs/benchmark-cfa-2026-10-08.md) and [October 9 Mac JavaScript](docs/benchmark-javascript-2026-10-09.md) reports are retained with their original measurement protocols. These are source builds, not measurements of the published 0.2 packages.\n"


def svg(tail, spec, dark=False, label='JSON Schema', dataset='json', include_dynamic=False):
    w,h=spec['width'],spec['height']; g=spec['grid']
    left,right,top,bottom=g['left'],w-g['right'],g['top'],h-g['bottom']
    xmin=max(spec['exceedance']['min'],tail['exceedance_pct'][0]); xmax=spec['exceedance']['max']
    ymin,ymax=spec['latency_us']['min'],spec['latency_us']['max']
    palette=spec['palette']['dark' if dark else 'light']; text,rule=palette['text'],palette['rule']
    interpolation=spec['interpolation'][dataset]
    assert interpolation == 'linear'
    curve_description='Straight segments join observed tail samples; no smoothing or extrapolation.'
    series_description=('Measured GLRMask, GLRMask dynamic and llguidance curves. GLRMask is faded for context.'
                        if include_dynamic else 'Measured GLRMask Static and llguidance 1.8.0 curves.')
    x=lambda v:left+(math.log10(xmax)-math.log10(v))/(math.log10(xmax)-math.log10(xmin))*(right-left)
    y=lambda v:bottom-(math.log10(v)-math.log10(ymin))/(math.log10(ymax)-math.log10(ymin))*(bottom-top)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">',
         f'<title id="title">{label}: time between masks</title>',
         '<desc id="description">'+series_description+' Inverse logarithmic exceedance probability and logarithmic latency axes. Lower is better. '+curve_description+'</desc>',
         f'<defs><clipPath id="plot"><rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}"/></clipPath></defs>',
         f'<g font-family="{html.escape(spec["font_family"])}" font-size="14" fill="{text}">',
         f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="{rule}"/>']
    for value,label_y in [(1,'1 µs'),(10,'10 µs'),(100,'100 µs'),(1000,'1 ms'),(10000,'10 ms')]:
        yy=y(value)
        out += [f'<path d="M{left-4},{yy:.2f} H{left}" stroke="{rule}"/>',f'<text x="{left-9}" y="{yy+4:.2f}" text-anchor="end">{label_y}</text>']
    out.append(f'<text x="{left-9}" y="{top-10}" text-anchor="end" font-size="13">TBM</text>')
    ticks=[50,10,1,.1,.01,.001]
    if xmin>spec['exceedance']['min']:
        ticks=[v for v in ticks if v>=xmin and x(xmin)-x(v)>72]
        ticks.append(xmin)
    for value in ticks:
        if value<xmin:continue
        xx=x(value)
        anchor='end' if abs(xx-right)<1 else 'middle'
        out += [f'<path d="M{xx:.2f},{bottom} v4" stroke="{rule}"/>',f'<text x="{xx:.2f}" y="{bottom+23}" text-anchor="{anchor}">{value:.2g}%</text>']
    out.append(f'<text x="{(left+right)/2}" y="{h-14}" text-anchor="middle">Exceedance probability · log scale</text>')
    curves=[]
    for mode,name in MODES:
        display_modes=spec['dynamic_preview']['display_modes'] if include_dynamic else spec['display_modes']
        if mode not in display_modes:continue
        points=[(x(ex),y(us)) for ex,us in zip(tail['exceedance_pct'],tail[mode+'_us'])]
        path=' '.join(('M' if i==0 else 'L')+f'{xx:.3f},{yy:.3f}' for i,(xx,yy) in enumerate(points))
        fade=f' opacity="{spec["dynamic_preview"]["static_opacity"]}"' if include_dynamic and mode=='static' else ''
        out.append(f'<path d="{path}" fill="none" stroke="{palette[mode]}" stroke-width="{spec["line_width"]}" stroke-linejoin="round"{fade} clip-path="url(#plot)"/>')
        # Boundary is a log-space segment of the retained points, never extrapolated.
        visible=[p for p in points if p[0]<=right]
        end_y=visible[0][1]
        if points[0][0]>right:
            a=points[len(points)-len(visible)-1]; b=visible[0]
            end_y=a[1]+(b[1]-a[1])*(right-a[0])/(b[0]-a[0])
        curves.append((end_y,mode,name,points))
    previous=-100
    for end_y,mode,name,points in sorted(curves):
        baseline=end_y-10
        if baseline-15<previous+6:
            region=[yy for xx,yy in points if right-155<=xx<=right]
            baseline=max(previous+24,max(region)+21)
        baseline=min(bottom-8,max(top+16,baseline)); previous=baseline
        if abs(baseline-(end_y-10))>2:
            out.append(f'<path d="M{right},{end_y:.2f} V{baseline-6:.2f}" stroke="{palette[mode]}" stroke-width="0.8" opacity="0.65"/>')
        fade=f' opacity="{spec["dynamic_preview"]["static_opacity"]}"' if include_dynamic and mode=='static' else ''
        out.append(f'<text x="{right-6}" y="{baseline:.2f}" fill="{palette[mode]}" font-size="14" font-weight="600" text-anchor="end"{fade}>{name}</text>')
    return '\n'.join(out+['</g>','</svg>',''])


def png_chunks(data):
    assert data[:8]==b'\x89PNG\r\n\x1a\n', 'Invalid PNG signature'
    pos=8
    while pos<len(data):
        size=struct.unpack('>I',data[pos:pos+4])[0]; kind=data[pos+4:pos+8]; body=data[pos+8:pos+8+size]
        crc=struct.unpack('>I',data[pos+8+size:pos+12+size])[0]
        assert zlib.crc32(kind+body)&0xffffffff==crc, 'PNG CRC failure'
        yield kind,body
        pos+=12+size
    assert pos==len(data), 'Truncated PNG'


def render_png(source, path, spec):
    assert shutil.which('rsvg-convert'), 'Install librsvg / rsvg-convert to regenerate PNGs'
    png=subprocess.run(['rsvg-convert','--width',str(2*spec['width']),'--height',str(2*spec['height']),'--background-color','#ffffff'],input=source.encode(),stdout=subprocess.PIPE,check=True).stdout
    body=b'SourceSVG_SHA256\0'+hashlib.sha256(source.encode()).hexdigest().encode()
    chunk=struct.pack('>I',len(body))+b'tEXt'+body+struct.pack('>I',zlib.crc32(b'tEXt'+body)&0xffffffff)
    # IEND is the final 12-byte chunk. Keep raster source identity in the PNG.
    path.write_bytes(png[:-12]+chunk+png[-12:])


def check_website(data,js,root):
    assert data['tail']==json.loads((root/'src/data/glrmask-canonical-tail-20261014.json').read_text())
    actual=json.loads((root/'src/data/glrmask-canonical-20261014.json').read_text())
    for mode,row in data['modes'].items():
        assert row['statistics_ns']=={metric:value['interpolated'] for metric,value in actual['modes'][mode]['statistics']['current'].items()}
        assert row['outcomes']==actual['modes'][mode]['outcomes']['current']
    assert js['tail']==json.loads((root/'src/data/glrmask-javascript-native-20261014.json').read_text())['tail']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--render-png',action='store_true');parser.add_argument('--website-root',type=Path)
    args=parser.parse_args();data,js=load()
    if args.website_root:check_website(data,js,args.website_root)
    spec=json.loads((CONTENT/'tail-chart.json').read_text(encoding="utf-8"))
    files={CONTENT/'benchmarks.md':markdown(data,js)}
    for dataset,measured,tail,label in [('json','2026-10-14',data['tail'],'JSON Schema'),('json-dynamic','2026-10-14',data['tail'],'JSON Schema · Dynamic'),('javascript','2026-10-14',js['tail'],'JavaScript')]:
        stem=ROOT/f'docs/assets/benchmark-tbm-{dataset}-{measured}'
        for dark in [False,True]:files=files|{Path(str(stem)+('-dark' if dark else '')+'.svg'):svg(tail,spec,dark,label,'json' if dataset=='json-dynamic' else dataset,dataset=='json-dynamic')}
    stale=[str(p.relative_to(ROOT)) for p,body in files.items() if not p.exists() or p.read_text(encoding="utf-8")!=body]
    if not args.check:
        for p,body in files.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body, encoding="utf-8")
        stale=[]
    for path,body in files.items():
        if path.suffix!='.svg' or path.stem.endswith('-dark'):continue
        png=path.with_suffix('.png')
        if args.render_png and not args.check:render_png(body,png,spec)
        expected=b'SourceSVG_SHA256\0'+hashlib.sha256(body.encode()).hexdigest().encode()
        chunks=list(png_chunks(png.read_bytes())) if png.exists() else []
        valid=bool(chunks) and chunks[0][0]==b'IHDR' and len(chunks[0][1])==13 and struct.unpack('>II',chunks[0][1][:8])==(2*spec['width'],2*spec['height'])
        valid=valid and any(kind==b'IDAT' for kind,_ in chunks) and chunks[-1]==(b'IEND',b'')
        if not valid or not any(kind==b'tEXt' and value==expected for kind,value in chunks):stale.append(str(png.relative_to(ROOT)))
    if stale and args.check:
        print('Stale benchmark outputs: '+', '.join(stale),file=sys.stderr);return 1
    if stale and not args.render_png and not args.check:
        print('Run with --render-png to refresh portable registry PNGs.',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
