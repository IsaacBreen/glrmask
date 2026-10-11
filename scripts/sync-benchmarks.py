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
PROVENANCE_SHA256 = '11f5b4704ecf9279d3d6b83c6a183e8ec7666a4eef8e7c58bf12256789e2cfbc'
JS_PROVENANCE_SHA256 = '3653f392561164fd306df5715ce9eafffef1dedeaa7034e1ef2e504560a92dbd'


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load():
    data = json.loads((ROOT/'docs/benchmarks/json-schema-2026-10-08.json').read_text())
    prov = {k:v for k,v in data.items() if k not in ['tail','modes']}
    prov['modes'] = {mode:{k:v for k,v in row.items() if k!='statistics_ns'} for mode,row in data['modes'].items()}
    assert digest(prov)==PROVENANCE_SHA256, 'JSON provenance changed; review report and prose before updating pin'
    js = json.loads((ROOT/'docs/benchmarks/javascript-2026-10-09.json').read_text())
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


def table(rows, column='Metric', names=('GLRMask','llguidance')):
    return '\n'.join(['| '+column+' | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)] +
                     ['| '+label.replace(' ', '&nbsp;')+' | '+' | '.join(f'{v:,.1f}' for v in values)+' |' for label,values in rows])


def markdown(data, js):
    def json_rows(metric, scale, quantiles, modes=('static','llguidance')):
        return [(label,[data['modes'][m]['statistics_ns'][metric][key]/scale for m in modes])
                for label,key in quantiles]
    rows=json_rows('tbm',1e3,[('Mean','mean'),('p50','median'),('p99','p99'),('p99.9','p99_9'),('Max','max')])
    ttfm=json_rows('ttfm',1e6,[('Mean','mean'),('p50','median'),('p99','p99'),('Max','max')])
    def js_rows(metric, labels):
        return [(row['label'].capitalize() if row['label'] in ['mean','max'] else row['label'],
                 [row['values'][0],row['values'][2]]) for row in js['tables'][metric] if row['label'] in labels]
    js_tbm=js_rows('tbm',['mean','p50','p99','max'])
    js_ttfm=js_rows('ttfm',['mean'])+[('Selected',[js['ttfm_single']['values_ms'][0],js['ttfm_single']['values_ms'][2]])]+js_rows('ttfm',['max'])
    dynamic_ttfm=json_rows('ttfm',1e6,[('Mean','mean'),('p50','median'),('p99','p99'),('Max','max')],('static','dynamic','llguidance'))
    return '''### JSON Schema

Time between masks. Lower is better.

{{json-plot}}

**TBM (µs)**

'''+table(rows, 'Statistic')+'''

**Time to first mask (ms)**

'''+table(ttfm, 'Statistic')+'''

8 October 2026 · 2,925,983 aligned token positions · macOS 27 ARM64, 10 logical CPUs. [Benchmark details](docs/benchmark-cfa-2026-10-08.md).

#### Dynamic mode

Care about compile times? Try dynamic mode.

{{dynamic}}

{{json-dynamic-plot}}

**Time to first mask (ms)**

'''+table(dynamic_ttfm,'Stat',('GLRMask','GLRMask<br>dynamic','llguidance'))+'''

[Explore Dynamic JSON results](https://glrmask.com/?dataset=json&mode=dynamic#benchmarks).

### JavaScript

Time between masks. Lower is better.

{{javascript-plot}}

**TBM (µs)**

'''+table(js_tbm, 'Statistic')+'''

**Time to first mask (ms)**

'''+table(js_ttfm, 'Statistic')+'''

9 October 2026 · 31 examples, 4,099 token positions · macOS ARM64. [Benchmark details](docs/benchmark-javascript-2026-10-09.md).

Both datasets measure source builds, not the published 0.2 packages.
'''


def retained_indices(xs, ys, width, height, tolerance):
    """Website shared policy: initial screen-space chord selection."""
    xscale=width/(math.log(100)-xs[0]);yscale=height/(math.log(20000)-math.log(.5))
    points=[(x*xscale,y*yscale) for x,y in zip(xs,ys)]
    keep={0,len(xs)-1};stack=[(0,len(xs)-1)]
    while stack:
        a,b=stack.pop();px,py=points[a];qx,qy=points[b]
        dx,dy=qx-px,qy-py;den=dx*dx+dy*dy;best=tolerance;index=-1
        for i in range(a+1,b):
            rx,ry=points[i]
            t=max(0,min(1,((rx-px)*dx+(ry-py)*dy)/den)) if den else 0
            distance=math.hypot(rx-px-t*dx,ry-py-t*dy)
            if distance>best:best,index=distance,i
        if index>=0:keep.add(index);stack.extend([(a,index),(index,b)])
    anchors=[]
    for i in sorted(keep):
        if anchors and abs(ys[i]-ys[anchors[-1]])*yscale<=tolerance:
            if i==len(xs)-1:
                if len(anchors)>1:anchors.pop()
                anchors.append(i)
            continue
        anchors.append(i)
    return anchors


def fit_curve(raw_xs, raw_ys, indices):
    xs=[raw_xs[i] for i in indices];ys=[raw_ys[i] for i in indices]
    secants=[(b-a)/(x2-x1) for a,b,x1,x2 in zip(ys,ys[1:],xs,xs[1:])]
    slopes=[secants[0] if i==0 else secants[-1] if i==len(ys)-1 else
            0 if secants[i-1]*secants[i]<=0 else 2/(1/secants[i-1]+1/secants[i]) for i in range(len(ys))]
    for i,d in enumerate(secants):
        if d==0:slopes[i]=slopes[i+1]=0;continue
        a,b=slopes[i]/d,slopes[i+1]/d;radius=math.hypot(a,b)
        if radius>3:slopes[i],slopes[i+1]=3*a*d/radius,3*b*d/radius
    return {'xs':xs,'ys':ys,'slopes':slopes,'indices':indices}


def evaluate_curve(curve, x):
    from bisect import bisect_right
    xs,ys,m=curve['xs'],curve['ys'],curve['slopes']
    x=max(xs[0],min(xs[-1],x));lo=max(0,min(len(xs)-2,bisect_right(xs,x)-1));hi=lo+1
    h=xs[hi]-xs[lo];t=(x-xs[lo])/h
    return (2*t**3-3*t*t+1)*ys[lo]+(t**3-2*t*t+t)*h*m[lo]+(-2*t**3+3*t*t)*ys[hi]+(t**3-t*t)*h*m[hi]


def display_curve(probabilities, latencies, width, height, settings):
    """Monotone measured anchors, refined to the supplied-quantile error bound."""
    xs=[math.log(v) for v in probabilities];ys=[math.log(v) for v in latencies]
    assert len(xs)>=2 and len(xs)==len(ys) and all(a<b for a,b in zip(xs,xs[1:]))
    assert all(a>=b for a,b in zip(ys,ys[1:]))
    indices=retained_indices(xs,ys,width,height,settings['screen_tolerance_px'])
    curve=fit_curve(xs,ys,indices);limit=math.log(1+settings['latency_tolerance'])
    for _ in range(len(xs)):
        additions=[]
        for a,b in zip(curve['indices'],curve['indices'][1:]):
            worst=limit;index=-1
            for i in range(a+1,b):
                error=abs(evaluate_curve(curve,xs[i])-ys[i])
                if error>worst:worst,index=error,i
            if index>=0:additions.append(index)
        if not additions:break
        curve=fit_curve(xs,ys,sorted(set(curve['indices']+additions)))
    assert curve['indices'][0]==0 and curve['indices'][-1]==len(xs)-1
    assert max(abs(evaluate_curve(curve,x)-y) for x,y in zip(xs,ys))<=limit+1e-12
    return curve


def monotone_path(curve, project_x, project_y):
    xs,ys,m=curve['xs'],curve['ys'],curve['slopes']
    path=[f'M{project_x(xs[0]):.5f},{project_y(ys[0]):.5f}']
    for i in range(len(xs)-1):
        h=xs[i+1]-xs[i]
        controls=[(xs[i]+h/3,ys[i]+m[i]*h/3),(xs[i+1]-h/3,ys[i+1]-m[i+1]*h/3),(xs[i+1],ys[i+1])]
        lo,hi=sorted([ys[i],ys[i+1]])
        assert all(lo-1e-12<=y<=hi+1e-12 for _,y in controls), 'Interpolation overshoot'
        path.append('C'+' '.join(f'{project_x(x):.5f},{project_y(y):.5f}' for x,y in controls))
    return ' '.join(path)


def svg(tail, spec, dark=False, label='JSON Schema', dataset='json', include_dynamic=False):
    w,h=spec['width'],spec['height']; g=spec['grid']
    left,right,top,bottom=g['left'],w-g['right'],g['top'],h-g['bottom']
    xmin=max(spec['exceedance']['min'],tail['exceedance_pct'][0]); xmax=spec['exceedance']['max']
    ymin,ymax=spec['latency_us']['min'],spec['latency_us']['max']
    palette=spec['palette']['dark' if dark else 'light']; text,rule=palette['text'],palette['rule']
    interpolation=spec['interpolation'][dataset]
    assert interpolation in ['linear','adaptive-monotone']
    curve_description=('Adaptive monotone cubic display through selected measured anchors, within 2% latency of every supplied quantile; endpoints and extrema preserved, without extrapolation.'
                       if interpolation=='adaptive-monotone' else 'Straight segments join observed tail samples; no smoothing or extrapolation.')
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
        curve=(display_curve(tail['exceedance_pct'],tail[mode+'_us'],right-left,bottom-top,spec['javascript_display'])
               if interpolation=='adaptive-monotone' else None)
        path=(monotone_path(curve,lambda v:x(math.exp(v)),lambda v:y(math.exp(v))) if curve else
              ' '.join(('M' if i==0 else 'L')+f'{xx:.3f},{yy:.3f}' for i,(xx,yy) in enumerate(points)))
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


def check_website(data, js, root):
    website=json.loads((root/'src/data/glrmask-canonical-20261008.json').read_text()); tail=json.loads((root/'src/data/glrmask-canonical-tail-20261008.json').read_text())
    assert data['source_commits']==website['metadata']['current']['revisions']
    assert data['tail']==tail
    for mode,_ in MODES:
        actual,expected=data['modes'][mode],website['modes'][mode]
        for key in ['schemas','aligned_tbm_positions','membership_sha256','prefix_identity_sha256']:assert actual[key]==expected[key]
        assert actual['outcomes']==expected['outcomes']['current']
        for metric in ['tbm','ttfm','build']:assert actual['statistics_ns'][metric]==expected['statistics']['current'][metric]['nearest_rank']
    website_js=json.loads((root/'src/data/glrmask-javascript-measured-5f2609-20261009.json').read_text())
    for key in ['source_revision','cfa_revision','llguidance_version','examples','tokens','tail','tables','ttfm_single']:assert js[key]==website_js[key], 'JS '+key+' differs'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--render-png',action='store_true');parser.add_argument('--website-root',type=Path)
    args=parser.parse_args();data,js=load()
    if args.website_root:check_website(data,js,args.website_root)
    spec=json.loads((CONTENT/'tail-chart.json').read_text())
    files={CONTENT/'benchmarks.md':markdown(data,js)}
    for dataset,measured,tail,label in [('json','2026-10-08',data['tail'],'JSON Schema'),('json-dynamic','2026-10-08',data['tail'],'JSON Schema · Dynamic'),('javascript','2026-10-09',js['tail'],'JavaScript')]:
        stem=ROOT/f'docs/assets/benchmark-tbm-{dataset}-{measured}'
        for dark in [False,True]:files=files|{Path(str(stem)+('-dark' if dark else '')+'.svg'):svg(tail,spec,dark,label,'json' if dataset=='json-dynamic' else dataset,dataset=='json-dynamic')}
    stale=[str(p.relative_to(ROOT)) for p,body in files.items() if not p.exists() or p.read_text()!=body]
    if not args.check:
        for p,body in files.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
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
