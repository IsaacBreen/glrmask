#!/usr/bin/env python3
"""Compare canonical forcing in unchanged release executables; never build or install."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--candidate-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2000)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 20 or args.pairs < 2:
        parser.error('At least 20 repetitions and two alternating pairs are required')
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {name: getattr(args, name).resolve(strict=True) for name in ('baseline', 'candidate')}
    identities = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()}
    identity = {
        'started_at': datetime.now().astimezone().isoformat(),
        'baseline_ref': args.baseline_ref, 'candidate_ref': args.candidate_ref,
        'binaries': identities, 'python': platform.python_version(),
        'platform': platform.platform(), 'logical_processors': os.cpu_count(),
        'repetitions': args.repetitions, 'pairs': args.pairs,
        'controller_sha256': sha256(Path(__file__)),
        'notes': ['Alternating AB/BA fresh processes; unchanged compiled harness.',
                  'Canonical forced output, not historical strict-singleton forcing.',
                  'Percentiles are per-run observations; p99.99 is the maximum at small sample counts.',
                  'No build, model download or library modification by this controller.'],
    }
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    observations: dict[str, list[dict]] = {'baseline': [], 'candidate': []}
    semantic_reference = None
    for pair in range(args.pairs):
        for name in (('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')):
            label = f'pair-{pair:02d}-{name}'
            print('RUN', label, flush=True)
            before = time.perf_counter()
            with (args.output / f'{label}.json').open('wb') as out, (args.output / f'{label}.stderr').open('wb') as err:
                result = subprocess.run([str(paths[name]), str(args.repetitions)], stdout=out, stderr=err,
                                        timeout=1200, check=False)
            if result.returncode:
                raise RuntimeError(f'{label} exited {result.returncode}; inspect retained stderr')
            data = json.loads((args.output / f'{label}.json').read_text())
            if data.get('semantics') != 'canonical-forced-output' or data.get('repetitions') != args.repetitions:
                raise ValueError(f'{label}: wrong harness contract')
            rows = {(r['case'], r['mode'], r['warm_mask']): r for r in data['rows']}
            if len(rows) != 48 or len(data['rows']) != 48:
                raise ValueError(f'{label}: missing/duplicate scenario rows')
            semantics = {key: value['ids'] for key, value in rows.items()}
            if semantic_reference is None:
                semantic_reference = semantics
            if semantics != semantic_reference:
                raise ValueError(f'{label}: canonical original-ID outputs changed')
            if any(r['samples'] != args.repetitions for r in rows.values()):
                raise ValueError(f'{label}: wrong sample counts')
            observations[name].append(rows)
            print('PASS', label, f'{time.perf_counter() - before:.3f}s', flush=True)
    for name, path in paths.items():
        if sha256(path) != identities[name]['sha256']:
            raise RuntimeError(f'{name} executable changed during qualification')
    metrics = ('p50_ns', 'p90_ns', 'p95_ns', 'p99_ns', 'p999_ns', 'p9999_ns', 'max_ns', 'mean_ns')
    summary = []
    for key in sorted(semantic_reference):
        comparisons = {}
        for metric in metrics:
            b = [r[key][metric] for r in observations['baseline']]
            c = [r[key][metric] for r in observations['candidate']]
            ratios = [y / x if x > 0 else None for x, y in zip(b, c)]
            comparisons[metric] = {
                'baseline_runs': b, 'candidate_runs': c,
                'baseline_median': statistics.median(b), 'candidate_median': statistics.median(c),
                'paired_ratios': ratios,
                'median_ratio': statistics.median([r for r in ratios if r is not None]) if any(r is not None for r in ratios) else None,
                'all_pairs_over_1_10': all(r is not None and r > 1.1 for r in ratios),
            }
        summary.append({'case': key[0], 'mode': key[1], 'warm_mask': key[2],
                        'ids': semantic_reference[key], 'metrics': comparisons})
    report = {'semantics': 'canonical-forced-output', 'exact_outputs_equal': True,
              'scenarios': len(summary), 'completed_at': datetime.now().astimezone().isoformat(),
              'interpretation': 'Ratios and repeatability flags are observations, not automatic performance acceptance.',
              'rows': summary}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('All 48 canonical-output scenarios match across every run.', flush=True)


if __name__ == '__main__':
    main()
