#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
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
    token_id: int
    allowed_count: int
    internal_ids: list[int]
    expected_sparse_words: list[list[int]]


@dataclass(frozen=True)
class IncludeStep:
    problem: str
    example_index: int
    step: int


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


def parse_include_step(raw: str) -> IncludeStep:
    try:
        problem, example_index, step = raw.rsplit(":", 2)
        return IncludeStep(problem=problem, example_index=int(example_index), step=int(step))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "include steps must look like jsb/data/Problem---name:EXAMPLE:STEP"
        ) from exc


def sparse_words_from_buf(buf: np.ndarray) -> list[list[int]]:
    sparse: list[list[int]] = []
    for idx, raw in enumerate(buf):
        word = int(raw) & 0xFFFFFFFF
        if word:
            sparse.append([idx, word])
    return sparse


def fill_public_sparse_words(state: Any, buf: np.ndarray) -> list[list[int]]:
    """Capture the same original-token mask bits exposed by the native adapter."""
    buf.fill(0)
    state._constraint_state.fill_mask(buf)
    return sparse_words_from_buf(buf)


def fill_public_sparse_words_and_internal_ids(
    state: Any,
    buf: np.ndarray,
    original_to_internal: list[int],
) -> tuple[list[list[int]], list[int]]:
    """Capture production output plus the exact production internal dense mask."""
    from cfa.adapters.glrmask_internal import object_method

    buf.fill(0)
    fill_and_ids = object_method(state._constraint_state, "mask_game_fill_mask_and_internal_ids")
    if fill_and_ids is not None:
        internal_ids = [int(x) for x in fill_and_ids(buf)]
        return sparse_words_from_buf(buf), internal_ids

    state._constraint_state.fill_mask(buf)
    public_sparse = sparse_words_from_buf(buf)
    return public_sparse, internal_ids_from_sparse(public_sparse, original_to_internal)


def sparse_bit_count(sparse_words: list[list[int]]) -> int:
    return sum((int(word) & 0xFFFFFFFF).bit_count() for _, word in sparse_words)


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


def encode_internal_ids(internal_ids: list[int]) -> str:
    out = bytearray()
    prev = 0
    for internal in internal_ids:
        if internal < prev:
            raise ValueError("internal ids must be sorted before delta-varint encoding")
        value = internal - prev
        prev = internal
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return base64.b64encode(out).decode("ascii")


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
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--vocab", default="llama3")
    parser.add_argument("--include-step", type=parse_include_step, action="append", default=[])
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="emit every non-empty mask step instead of only the heaviest prefixes",
    )
    args = parser.parse_args()

    _add_import_paths(args.cfa_root)

    from cfa.adapters.glrmask_native_adapter import GlrMaskNativeAdapter
    from cfa.config import get_paths
    from cfa.registry import default_registry
    from cfa.tokenization import VOCAB_LOADERS
    from cfa.tokenizer import GreedyTokenizer
    from scripts.inspect_step_stabilized import _problem_from_spec
    from scripts.sweep import collect_examples_for_spec

    os.environ.setdefault("CFA_BUILD_TIMEOUT_SECONDS", "180")

    registry = default_registry(get_paths().data_dir)
    try:
        vocab_loader = VOCAB_LOADERS[args.vocab]
    except KeyError as exc:
        raise ValueError(f"unknown vocab {args.vocab!r}; choices: {sorted(VOCAB_LOADERS)}") from exc
    vocab = vocab_loader()
    tokenizer = GreedyTokenizer(vocab.vocab)
    adapter = GlrMaskNativeAdapter()

    problem_ids = args.problem or slow_example_problem_ids(args.cfa_root)
    for include in args.include_step:
        if include.problem not in problem_ids:
            problem_ids.append(include.problem)
    if args.max_problems > 0:
        problem_ids = problem_ids[: args.max_problems]
    include_steps = {(item.problem, item.example_index, item.step) for item in args.include_step}

    maps: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    seen_cases: set[tuple[str, int, int, int]] = set()
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
            from cfa.adapters.glrmask_internal import mask_game_mapping

            internal_to_original, original_to_internal = mask_game_mapping(state._constraint)
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

        def append_case(item: PendingCase) -> None:
            key = (item.problem, item.map_id, item.example_index, item.step)
            if key in seen_cases:
                return
            seen_cases.add(key)
            cases.append({
                "map_id": item.map_id,
                "problem": item.problem,
                "example_index": item.example_index,
                "step": item.step,
                "token_id": item.token_id,
                "allowed_count": item.allowed_count,
                "internal_ids_vb64": encode_internal_ids(item.internal_ids),
                "expected_sparse_words": item.expected_sparse_words,
            })

        for example_index, example in enumerate(examples):
            try:
                token_ids = tokenizer.encode(example.text.encode("utf-8"))
            except BaseException as exc:
                skipped.append({"problem": problem_id, "reason": f"tokenize example {example_index}: {exc}"})
                continue

            state.reset()
            best: list[PendingCase] = []
            for step, token_id in enumerate(token_ids):
                try:
                    public_sparse, internal_ids = fill_public_sparse_words_and_internal_ids(
                        state,
                        buf,
                        original_to_internal,
                    )
                except BaseException as exc:
                    skipped.append({"problem": problem_id, "reason": f"fill example {example_index} step {step}: {exc}"})
                    break

                expanded_sparse = sparse_words_from_internal_ids(
                    internal_ids,
                    internal_to_original,
                    mask_words,
                )
                score = score_case(internal_ids, internal_to_original, expanded_sparse)
                if expanded_sparse and internal_ids:
                    pending = PendingCase(
                        score=score,
                        map_id=map_id,
                        problem=problem_id,
                        example_index=example_index,
                        step=step,
                        token_id=int(token_id),
                        allowed_count=sparse_bit_count(public_sparse),
                        internal_ids=internal_ids,
                        expected_sparse_words=expanded_sparse,
                    )
                    if args.all_steps or (problem_id, example_index, step) in include_steps:
                        append_case(pending)
                    if not args.all_steps:
                        best.append(pending)
                        best.sort(key=lambda item: item.score, reverse=True)
                        del best[args.cases_per_example :]

                try:
                    state.commit(int(token_id))
                except BaseException as exc:
                    skipped.append({"problem": problem_id, "reason": f"commit example {example_index} step {step}: {exc}"})
                    break

            for item in best:
                append_case(item)

    payload = {
        "version": 1,
        "source": f"constraint-framework-analysis make example-slow glrmask_native vocab={args.vocab}",
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
