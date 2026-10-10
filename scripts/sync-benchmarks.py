#!/usr/bin/env python3
"""Render the README benchmark blocks and SVGs from reviewed public aggregates.

--website-root checks that the site's canonical JSON data still matches this
snapshot. This reads aggregate data only; it never runs a benchmark.
"""
import argparse
from datetime import date
import html
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/benchmarks/json-schema-2026-10-08.json'
MODES = [('static', 'GLRMask Static'), ('dynamic', 'GLRMask Dynamic'), ('llguidance', 'llguidance 1.8.0')]
# This dated report has reviewed prose about a specific source and methodology.
# Changing provenance requires reviewing that prose and its companion report.
PROVENANCE_SHA256 = '11f5b4704ecf9279d3d6b83c6a183e8ec7666a4eef8e7c58bf12256789e2cfbc'


def validate_provenance(data):
    provenance = {key: value for key, value in data.items() if key not in ['tail', 'modes']}
    provenance['modes'] = {mode: {key: value for key, value in row.items() if key != 'statistics_ns'}
                           for mode, row in data['modes'].items()}
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != PROVENANCE_SHA256:
        raise ValueError('Benchmark provenance changed. Review the dated report and generator prose before updating PROVENANCE_SHA256; see docs/documentation-sync.md.')


def table(data):
    lines = ['| Metric | Static | Dynamic | llguidance 1.8.0 |', '|---|---:|---:|---:|']
    rows = [('TBM p50 (µs)', 'tbm', 'median', 1000),
            ('TBM p99 (µs)', 'tbm', 'p99', 1000),
            ('TBM p99.9 (µs)', 'tbm', 'p99_9', 1000),
            ('TBM maximum (µs)', 'tbm', 'max', 1000),
            ('Build p50 (ms)', 'build', 'median', 1000000),
            ('TTFM p50 (ms)', 'ttfm', 'median', 1000000),
            ('TTFM p99 (ms)', 'ttfm', 'p99', 1000000)]
    for label, metric, key, scale in rows:
        values = [data['modes'][mode]['statistics_ns'][metric][key] / scale for mode, _ in MODES]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:,.1f}' for v in values) + ' |')
    return '\n'.join(lines)


def overview(data):
    static = data['modes']['static']['statistics_ns']['tbm']
    llg = data['modes']['llguidance']['statistics_ns']['tbm']
    measured = date.fromisoformat(data['measurement_date'])
    measurement_date = f'{measured.day} {measured:%B %Y}'
    positions = data['modes']['static']['aligned_tbm_positions']
    return f'''Time between masks (TBM) measures token commitment through the next ready mask. On the measured JSON Schema cohort, GLRMask Static has a {static['median']/1000:.1f} µs median and {static['p99']/1000:.1f} µs p99, compared with {llg['median']/1000:.1f} µs and {llg['p99']/1000:.1f} µs for llguidance. See [Performance](#performance) for compilation costs and methodology.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-tbm-2026-10-08-dark.svg">
    <img src="docs/assets/benchmark-tbm-2026-10-08.svg" alt="JSON Schema TBM percentile curves for GLRMask Static, Dynamic and llguidance 1.8.0; logarithmic latency axis" width="500">
  </picture>
</p>

*JSONSchemaBench, {measurement_date}: {positions:,} aligned token positions; calling-thread CPU time, minimum of two native processes. Lower is better. See [measurement details](docs/benchmark-cfa-2026-10-08.md).*'''


def results(data):
    static = data['modes']['static']['statistics_ns']['build']['median']/1000000
    dynamic = data['modes']['dynamic']['statistics_ns']['build']['median']/1000000
    return f'''Static favors mask latency; Dynamic favors compilation time. In this run, Static’s median build cost is {static:.1f} ms and Dynamic’s is {dynamic:.1f} ms. The benchmark’s Static and Dynamic implementations correspond to the current `FastRuntime` and `FastBuild` choices, respectively.

''' + table(data) + '''

Measured on 8 October 2026 using native Rust APIs at GLRMask source `f9f5c84` and CFA `5d3feb4`, with official llguidance 1.8.0, on macOS 27 ARM64 (10 logical CPUs, Rust 1.95.0) and the Llama 3.1 8B Instruct vocabulary. These are source measurements, not a benchmark of the published 0.2.0 packages.

The full run contains 9,558 official JSONSchemaBench schemas. The matched build/TTFM cohorts contain 8,326 schemas for Static and 8,327 for Dynamic and llguidance; TBM uses 2,925,983 aligned error-free token positions, including valid and invalid examples up to their shared replay prefix. These conditional timing distributions do not establish full-corpus correctness or language equivalence. The full-run outcomes include 460 incorrect examples for each GLRMask mode, and 100 incorrect examples and six runtime-error examples for llguidance.

TBM measures commitment of a previously allowed token through the next ready mask, using calling-thread CPU time and the elementwise minimum of two independent native processes. Build and time to first mask (TTFM) use wall time from the first process; TTFM includes schema-dependent compilation. Vocabulary preparation is outside the timer. Production validation is enabled, with one schema at a time and a 60-second build timeout. See the [benchmark report](docs/benchmark-cfa-2026-10-08.md) and [website benchmarks](https://glrmask.com/#benchmarks) for provenance, outcomes and further results.'''


def svg(data, dark=False):
    # A modest portrait-leaning canvas keeps labels readable in a phone README.
    w, h = 500, 390
    left, right, top, bottom = 69, 475, 82, 324
    bg, ink, muted, rule = ('#0d1117', '#e6edf3', '#aeb8c4', '#30363d') if dark else ('#ffffff', '#20262e', '#57616e', '#e1e5ea')
    colors = ['#58a6cf', '#c5966b', '#c1c8d1'] if dark else ['#26789b', '#a66837', '#67717e']
    tail = data['tail']
    # Ordered percentile axis expands the upper tail using log exceedance.
    xmin, xmax = -math.log10(50), -math.log10(tail['exceedance_pct'][0])
    x = lambda exceedance: left + (-math.log10(exceedance) - xmin) / (xmax - xmin) * (right - left)
    y = lambda us: bottom - math.log10(us) / 4 * (bottom - top)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
           '<title id="title">Time between masks — JSON Schema</title>',
           '<desc id="desc">8 October 2026. Empirical TBM tail at 2,925,983 aligned positions. Log latency in microseconds and log exceedance percentile spacing. Minimum of two native timing processes.</desc>',
           f'<rect width="{w}" height="{h}" fill="{bg}"/>',
           f'<g font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{ink}">',
           f'<text x="{left}" y="27" font-size="18">Time between masks · JSON Schema</text>']
    for i, (_, label) in enumerate(MODES):
        lx = [left, 200, 340][i]
        dash = ['',' stroke-dasharray="7 4"',' stroke-dasharray="2 4"'][i]
        out.append(f'<path d="M{lx} 52h24" stroke="{colors[i]}" stroke-width="2.6"{dash}/><text x="{lx + 31}" y="57" font-size="15">{html.escape(label.replace("GLRMask ", ""))}</text>')
    out.append(f'<text x="{left}" y="76" fill="{muted}" font-size="14">TBM (µs) · log scale</text>')
    for value in [1, 10, 100, 1000, 10000]:
        yy = y(value)
        out.append(f'<path d="M{left} {yy:.2f}H{right}" stroke="{rule}"/><text x="{left - 10}" y="{yy + 5:.2f}" text-anchor="end" fill="{muted}">{value:,}</text>')
    out.append(f'<path d="M{left} {top}V{bottom}H{right}" fill="none" stroke="{muted}"/>')
    for exceedance, label in [(50,'50'),(10,'90'),(1,'99'),(.1,'99.9'),(.01,'99.99')]:
        xx = x(exceedance)
        out.append(f'<path d="M{xx:.2f} {bottom}v5" stroke="{muted}"/><text x="{xx:.2f}" y="{bottom+24}" text-anchor="middle" fill="{muted}">{label}</text>')
    out.append(f'<text x="{right}" y="{bottom+24}" text-anchor="end" fill="{muted}" font-size="14">max</text>')
    out.append(f'<text x="{(left+right)/2}" y="{bottom+52}" text-anchor="middle" fill="{muted}" font-size="15">Percentile · upper tail expanded</text>')
    for i, (mode, _) in enumerate(MODES):
        points = [(x(ex), y(us)) for ex, us in zip(tail['exceedance_pct'], tail[mode+'_us']) if ex <= 50]
        path = ' '.join(('M' if j == 0 else 'L') + f'{xx:.2f},{yy:.2f}' for j, (xx, yy) in enumerate(points))
        dash = ['', ' stroke-dasharray="7 4"', ' stroke-dasharray="2 4"'][i]
        out.append(f'<path d="{path}" fill="none" stroke="{colors[i]}" stroke-width="2.6" stroke-linejoin="round"{dash}/>')
    return '\n'.join(out + ['</g>', '</svg>', ''])


def check_website(data, root):
    website = json.loads((root / 'src/data/glrmask-canonical-20261008.json').read_text())
    tail = json.loads((root / 'src/data/glrmask-canonical-tail-20261008.json').read_text())
    assert data['source_commits'] == website['metadata']['current']['revisions'], 'Benchmark source revisions differ'
    assert data['tail'] == tail, 'Benchmark tail differs'
    for mode, _ in MODES:
        actual = data['modes'][mode]
        expected = website['modes'][mode]
        for key in ['schemas', 'aligned_tbm_positions', 'membership_sha256', 'prefix_identity_sha256']:
            assert actual[key] == expected[key], f'{mode}: {key} differs'
        assert actual['outcomes'] == expected['outcomes']['current'], f'{mode}: outcomes differ'
        for metric in ['tbm', 'ttfm', 'build']:
            assert actual['statistics_ns'][metric] == expected['statistics']['current'][metric]['nearest_rank'], f'{mode}: {metric} differs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--website-root', type=Path)
    args = parser.parse_args()
    data = json.loads(DATA.read_text())
    validate_provenance(data)
    if args.website_root:
        check_website(data, args.website_root)
    guide = ROOT / 'GUIDE.md'
    expected = guide.read_text()
    for block, body in [('overview', overview(data)), ('results', results(data))]:
        pattern = rf'<!-- benchmark-{block}:start -->[\s\S]*?<!-- benchmark-{block}:end -->'
        replacement = f'<!-- benchmark-{block}:start -->\n{body}\n<!-- benchmark-{block}:end -->'
        expected, count = re.subn(pattern, lambda _: replacement, expected)
        assert count == 1, f'Missing or duplicate benchmark {block} block'
    files = {guide: expected}
    for dark in [False, True]:
        path = ROOT / ('docs/assets/benchmark-tbm-2026-10-08' + ('-dark' if dark else '') + '.svg')
        files[path] = svg(data, dark)
    stale = [str(path.relative_to(ROOT)) for path, body in files.items() if not path.exists() or path.read_text() != body]
    if args.check:
        if stale:
            print('Stale benchmark outputs: ' + ', '.join(stale) + '; run python scripts/sync-benchmarks.py', file=sys.stderr)
            return 1
    else:
        for path, body in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
