"""Run unchanged native benchmark loops in balanced, separate processes.

This controller is external to the library. It never builds or edits a source
tree. Raw stdout, stderr, binary digests and every case are retained.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--candidate-ref', required=True)
    parser.add_argument('--repetitions', type=int, default=5000)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--vocab', type=int, default=8192)
    parser.add_argument('--reverse-first', action='store_true', help='Begin the first timing pair with candidate, then alternate')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')
    if args.repetitions < 20 or args.pairs < 1:
        raise SystemExit('Invalid repetition or pair count')
    args.output.mkdir(parents=True)
    bins = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    identity = {
        'started_at': dt.datetime.now().astimezone().isoformat(),
        'baseline_ref': args.baseline_ref,
        'candidate_ref': args.candidate_ref,
        'binaries': {k: {'path': str(v), 'sha256': digest(v)} for k, v in bins.items()},
        'repetitions': args.repetitions, 'vocab': args.vocab, 'pairs': args.pairs,
        'reverse_first': args.reverse_first,
        'notes': [
            'Separate processes, identical measured loops and inputs.',
            'AB then BA ordering alternates per pair.',
            'No building or source edits by this controller.',
            'Full original-ID words are compared, not only fingerprints.',
            'Ratios and repeatability flags are observations, not automatic performance acceptance.',
        ],
    }
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf8')

    def run(label: str, which: str, reps: int, capture: bool = False) -> dict:
        env = os.environ.copy()
        env.pop('GLRM_CORE_CAPTURE_MASKS', None)
        if capture:
            env['GLRM_CORE_CAPTURE_MASKS'] = '1'
        start = time.perf_counter()
        print(f'RUN {label} {which} reps={reps}', flush=True)
        with (args.output / f'{label}.json').open('wb') as out, \
                (args.output / f'{label}.stderr').open('wb') as err:
            proc = subprocess.run([str(bins[which]), str(args.vocab), str(reps)],
                                  env=env, stdout=out, stderr=err, timeout=1800)
        elapsed = time.perf_counter() - start
        receipt = {'binary': which, 'exit_code': proc.returncode, 'elapsed_seconds': elapsed,
                   'capture': capture, 'repetitions': reps}
        (args.output / f'{label}.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
        if proc.returncode:
            raise RuntimeError(f'{label} exited {proc.returncode}; inspect retained stderr')
        data = json.loads((args.output / f'{label}.json').read_text(encoding='utf8'))
        if len(data['rows']) != 15:
            raise AssertionError(f'Unexpected case count in {label}')
        print(f'DONE {label}: {elapsed:.3f}s', flush=True)
        return data

    def rows(data: dict) -> dict:
        indexed = {(r['mode'], r['case']): r for r in data['rows']}
        assert len(indexed) == len(data['rows'])
        return indexed

    base_capture = rows(run('capture-baseline', 'baseline', 20, True))
    cand_capture = rows(run('capture-candidate', 'candidate', 20, True))
    assert base_capture.keys() == cand_capture.keys()
    words = 0
    for key, base in base_capture.items():
        candidate = cand_capture[key]
        assert base['trace'] == candidate['trace'], (key, 'trace')
        assert base['mask_fingerprint'] == candidate['mask_fingerprint'], (key, 'fingerprint')
        assert base['captured_masks'] == candidate['captured_masks'], (key, 'full mask words')
        assert base['captured_masks'] is not None
        words += sum(len(m) for m in base['captured_masks'])
    (args.output / 'exact-mask-comparison.json').write_text(json.dumps({
        'cases': len(base_capture), 'original_u32_words_equal': words,
        'traces_equal': True, 'includes_final_mask': True,
    }, indent=2), encoding='utf8')

    trials: dict[str, list[dict]] = {'baseline': [], 'candidate': []}
    paired: list[dict] = []
    for pair in range(args.pairs):
        order = ['baseline', 'candidate'] if (pair + int(args.reverse_first)) % 2 == 0 else ['candidate', 'baseline']
        current = {}
        for which in order:
            data = run(f'pair-{pair + 1}-{which}', which, args.repetitions)
            current[which] = rows(data)
            trials[which].append(current[which])
            for key, result in current[which].items():
                assert result['trace'] == base_capture[key]['trace']
                assert result['mask_fingerprint'] == base_capture[key]['mask_fingerprint']
        paired.append(current)

    result_rows = []
    flags = []
    counts = {'baseline': 0, 'candidate': 0}
    for key in sorted(base_capture):
        result = {'mode': key[0], 'case': key[1], 'measurements': {}}
        for field in ['mask', 'commit', 'start', 'trace_total', 'warm_compile', 'warm_link']:
            details = {}
            for q in ['p50_ns', 'p95_ns', 'p99_ns', 'p999_ns', 'max_ns']:
                values = {side: [t[key][field][q] for t in trials[side]] for side in trials}
                medians = {side: statistics.median(v) for side, v in values.items()}
                ratios = [p['candidate'][key][field][q] / max(1, p['baseline'][key][field][q])
                          for p in paired]
                details[q] = {'baseline': medians['baseline'], 'candidate': medians['candidate'],
                              'ratio': medians['candidate'] / max(1, medians['baseline']),
                              'paired_ratios': ratios}
                if field == 'mask' and min(ratios) > 1.02 and statistics.median(ratios) > 1.05:
                    flags.append({'mode': key[0], 'case': key[1], 'quantile': q,
                                  'paired_ratios': ratios})
            result['measurements'][field] = details
        for field in ['first_compile_ns', 'prepare_with_warm_vocab_ns', 'first_link_ns',
                      'save_ns', 'load_ns', 'prepared_save_ns', 'prepared_load_ns',
                      'artifact_bytes', 'prepared_artifact_bytes', 'source_create_and_bind_ns']:
            a = statistics.median(t[key][field] for t in trials['baseline'])
            b = statistics.median(t[key][field] for t in trials['candidate'])
            result[field] = {'baseline': a, 'candidate': b, 'ratio': b / max(1, a)}
        for side in counts:
            counts[side] += sum(t[key]['mask']['n'] for t in trials[side])
        result_rows.append(result)
    summary = {'identity': identity, 'full_mask_words_compared': words,
               'timed_masks': counts, 'rows': result_rows,
               'repeatability_review_flags': flags,
               'completed_at': dt.datetime.now().astimezone().isoformat()}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    lines = ['# Matched native mask comparison', '',
             f'Exact original-ID words compared: {words:,}. Timed masks: {counts}.', '',
             '| Case | Mode | Mask p50 old/new (ns) | p50 ratio | p99.9 old/new (ns) | p99.9 ratio |',
             '|---|---|---:|---:|---:|---:|']
    for r in result_rows:
        m = r['measurements']['mask']; a = m['p50_ns']; b = m['p999_ns']
        lines.append(f"| {r['case']} | {r['mode']} | {a['baseline']:.1f}/{a['candidate']:.1f} | "
                     f"{a['ratio']:.3f} | {b['baseline']:.1f}/{b['candidate']:.1f} | {b['ratio']:.3f} |")
    lines += ['', f'Repeatability review flags: {len(flags)}. These require human/code review; '
                   'their absence does not prove universal zero regression.', '']
    (args.output / 'summary.md').write_text('\n'.join(lines), encoding='utf8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
