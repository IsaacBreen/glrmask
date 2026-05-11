#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFA_ROOT = REPO_ROOT.parent / "constraint-framework-analysis"


@dataclass
class PendingCase:
    score: int
    map_id: int
    problem: str
    example_index: int
    step: int
    internal_ids: list[int]
    expected_sparse_words: list[list[int]]


def _add_import_paths(cfa_root: Path) -> None:
    sys.path.insert(0, str(cfa_root))
    sys.path.insert(0, str(REPO_ROOT / "python"))


def _read_make_var(makefile: Path, name: str) -> list[str]:
    text = makefile.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\s*:=\s*(.*?)(?=^\S|\Z)", text, re.M | re.S)
    if not match:
        raise ValueError(f"could not find {name} in {makefile}")
    body = match.group(1).replace("\\\n", " ")
    return [part for part in body.split() if part.startswith("jsb/")]


def slow_example_problem_ids(cfa_root: Path) -> list[str]:
    makefile = cfa_root / "Makefile"
    latest = _read_make_var(makefile, "SLOW_EXAMPLE_PROBLEMS_LATEST_RUN")
    ttfm = _read_make_var(makefile, "SLOW_TTFM_PROBLEMS")
    tbm = _read_make_var(makefile, "SLOW_TBM_PROBLEMS")
    out: list[str] = []
    seen: set[str] = set()
    for problem_id in latest + sorted(set(ttfm + tbm)):
        if problem_id not in seen:
            out.append(problem_id)
            seen.add(problem_id)
    return out


def sparse_words_from_buf(buf: np.ndarray) -> list[list[int]]:
    sparse: list[list[int]] = []
    for idx, raw in enumerate(buf):
        word = int(raw) & 0xFFFFFFFF
        if word:
            sparse.append([idx, word])
    return sparse


def internal_ids_from_sparse(
    sparse_words: list[list[int]],
    original_to_internal: list[int],
) -> list[int]:
    ids: set[int] = set()
    for word_idx, word in sparse_words:
        base = word_idx * 32
        while word:
            bit = (word & -word).bit_length() - 1
            original = base + bit
            if original < len(original_to_internal):
                internal = int(original_to_internal[original])
                if internal != 0xFFFFFFFF:
                    ids.add(internal)
            word &= word - 1
    return sorted(ids)


def sparse_words_from_internal_ids(
    internal_ids: list[int],
    internal_to_original: list[list[int]],
    mask_words: int,
) -> list[list[int]]:
    words: dict[int, int] = {}
    for internal in internal_ids:
        if internal < 0 or internal >= len(internal_to_original):
            continue
        for original in internal_to_original[internal]:
            word_idx = int(original) // 32
            if word_idx >= mask_words:
                continue
            words[word_idx] = words.get(word_idx, 0) | (1 << (int(original) & 31))
    return [[idx, word] for idx, word in sorted(words.items()) if word]


def score_case(internal_ids: list[int], internal_to_original: list[list[int]], sparse_words: list[list[int]]) -> int:
    fanout = 0
    for internal in internal_ids:
        if 0 <= internal < len(internal_to_original):
            fanout += len(internal_to_original[internal])
    return fanout * 4 + len(sparse_words)


def map_hash(internal_to_original: list[list[int]]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for originals in internal_to_original:
        digest.update(len(originals).to_bytes(4, "little"))
        for original in originals:
            digest.update(int(original).to_bytes(4, "little"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfa-root", type=Path, default=DEFAULT_CFA_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "mask_game/data/example_slow_mask_game.json.gz")
    parser.add_argument("--cases-per-example", type=int, default=3)
    parser.add_argument("--max-examples-per-problem", type=int, default=-1)
    parser.add_argument("--max-problems", type=int, default=0)
    args = parser.parse_args()

    _add_import_paths(args.cfa_root)

    from cfa.adapters.glrmask_native_adapter import GlrMaskNativeAdapter
    from cfa.config import get_paths
    from cfa.registry import default_registry
    from cfa.tokenization import load_vocab_info
    from cfa.tokenizer import GreedyTokenizer
    from scripts.inspect_step_stabilized import _problem_from_spec
    from scripts.sweep import collect_examples_for_spec

    os.environ.setdefault("CFA_BUILD_TIMEOUT_SECONDS", "180")

    registry = default_registry(get_paths().data_dir)
    vocab = load_vocab_info()
    tokenizer = GreedyTokenizer(vocab.vocab)
    adapter = GlrMaskNativeAdapter()

    problem_ids = slow_example_problem_ids(args.cfa_root)
    if args.max_problems > 0:
        problem_ids = problem_ids[: args.max_problems]

    maps: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    hash_to_map_id: dict[str, int] = {}
    buf_words = 0
    skipped: list[dict[str, str]] = []

    for problem_index, problem_id in enumerate(problem_ids, start=1):
        spec = registry.get_spec(problem_id)
        if spec is None:
            skipped.append({"problem": problem_id, "reason": "not in registry"})
            continue

        examples, load_error = collect_examples_for_spec(
            spec,
            None if args.max_examples_per_problem < 0 else args.max_examples_per_problem,
        )
        if not examples:
            skipped.append({"problem": problem_id, "reason": load_error or "no examples"})
            continue

        print(f"[{problem_index}/{len(problem_ids)}] build {problem_id} examples={len(examples)}", flush=True)
        try:
            first_example = {"text": examples[0].text, "expected_valid": examples[0].expected_valid}
            problem = _problem_from_spec(spec, first_example)
            state = adapter.build(problem, vocab)
            internal_to_original, original_to_internal = state._constraint.mask_game_mapping()
        except BaseException as exc:
            skipped.append({"problem": problem_id, "reason": f"build failed: {exc}"})
            continue

        internal_to_original = [[int(x) for x in originals] for originals in internal_to_original]
        original_to_internal = [int(x) for x in original_to_internal]
        key = map_hash(internal_to_original)
        map_id = hash_to_map_id.get(key)
        if map_id is None:
            map_id = len(maps)
            hash_to_map_id[key] = map_id
            maps.append({
                "id": map_id,
                "problem": problem_id,
                "internal_to_original": internal_to_original,
            })

        mask_words = int(state._constraint.mask_len())
        buf_words = max(buf_words, mask_words)
        buf = np.zeros(mask_words, dtype=np.int32)

        for example_index, example in enumerate(examples):
            try:
                token_ids = tokenizer.encode(example.text.encode("utf-8"))
            except BaseException as exc:
                skipped.append({"problem": problem_id, "reason": f"tokenize example {example_index}: {exc}"})
                continue

            state.reset()
            best: list[PendingCase] = []
            for step, token_id in enumerate(token_ids):
                buf.fill(0)
                try:
                    state._constraint_state.fill_mask(buf)
                except BaseException as exc:
                    skipped.append({"problem": problem_id, "reason": f"fill example {example_index} step {step}: {exc}"})
                    break

                public_sparse = sparse_words_from_buf(buf)
                internal_ids = internal_ids_from_sparse(public_sparse, original_to_internal)
                expanded_sparse = sparse_words_from_internal_ids(
                    internal_ids,
                    internal_to_original,
                    mask_words,
                )
                score = score_case(internal_ids, internal_to_original, expanded_sparse)
                if expanded_sparse and internal_ids:
                    best.append(PendingCase(
                        score=score,
                        map_id=map_id,
                        problem=problem_id,
                        example_index=example_index,
                        step=step,
                        internal_ids=internal_ids,
                        expected_sparse_words=expanded_sparse,
                    ))
                    best.sort(key=lambda item: item.score, reverse=True)
                    del best[args.cases_per_example :]

                try:
                    state.commit(int(token_id))
                except BaseException as exc:
                    skipped.append({"problem": problem_id, "reason": f"commit example {example_index} step {step}: {exc}"})
                    break

            for item in best:
                cases.append({
                    "map_id": item.map_id,
                    "problem": item.problem,
                    "example_index": item.example_index,
                    "step": item.step,
                    "internal_ids": item.internal_ids,
                    "expected_sparse_words": item.expected_sparse_words,
                })

    payload = {
        "version": 1,
        "source": "constraint-framework-analysis make example-slow glrmask_native",
        "buf_words": buf_words,
        "maps": maps,
        "cases": cases,
        "skipped": skipped,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))

    size = args.output.stat().st_size
    print(f"wrote {args.output} ({size} bytes), maps={len(maps)}, cases={len(cases)}, skipped={len(skipped)}")
    if skipped:
        print("skipped:")
        for item in skipped[:20]:
            print(f"  {item['problem']}: {item['reason']}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
