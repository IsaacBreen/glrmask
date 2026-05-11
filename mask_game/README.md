# Mask Game

This is an isolated micro-benchmark for the final mask-materialization problem.

The setup phase receives a real mapping:

```text
internal token id -> set of original token ids
```

Setup time is not counted. The timed phase receives:

```text
input:  &[u32]   // internal token ids
output: &mut [u32] // empty original-token bitset buffer
```

The candidate must OR every original token covered by every supplied internal id
into `output`. It may precompute anything from the mapping during setup, but it
must not cache completed timed cases, remember prior case sequences, or depend on
repeated benchmark invocations.

## Files

- `src/lib.rs`: data format, candidate trait, verifier, and evaluator.
- `src/candidate.rs`: example candidates, including a glrmask-like group-run
  candidate that exploits contiguous internal-id runs.
- `src/bin/evaluate.rs`: command-line evaluator.
- `scripts/generate_from_cfa.py`: builds real cases from CFA `example-slow`
  problems using glrmask-native constraints.
- `data/`: generated datasets.

## Generate Data

From the glrmask2 repo root:

```bash
python mask_game/scripts/generate_from_cfa.py \
  --output mask_game/data/example_slow_mask_game.json.gz
```

The generator traverses all `example-slow` problem/example prefixes, then keeps a
small number of the heaviest prefixes per example so the checked-in-style data is
real but not absurdly large.

## Evaluate

```bash
cargo run --release --manifest-path mask_game/Cargo.toml --bin evaluate -- \
  mask_game/data/example_slow_mask_game.json.gz 200 complement
```

The second argument is the number of repetitions. The third argument is the
candidate: `baseline`, `group`, `copy`, `complement`, `parallel`, or `all`.
The evaluator prepares each mapping once, clears the output buffer outside the
measured section, times only `Candidate::fill`, then verifies the produced bitset
after each timed call.
It reports both raw aggregate timing and `stabilized_max_ns`, the maximum over
per-case best repetitions. That stabilized max is the metric comparable to the
existing CFA/glrmask stabilized timing workflow.

## Candidate API

Implement this trait for a new strategy:

```rust
pub trait Candidate {
    type Prepared;

    fn name() -> &'static str;
    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared;
    fn fill(prepared: &Self::Prepared, internal_ids: &[u32], out: &mut [u32]);
}
```

`prepare` may build indexes from `mapping.internal_to_original`; its cost is not
timed. `fill` is the timed operation and must be a pure expansion of the supplied
`internal_ids` into the empty `out` bitset.
