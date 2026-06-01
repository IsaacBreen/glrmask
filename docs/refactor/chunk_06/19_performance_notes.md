# Performance notes

The Chunk 06 split is intended to be performance-neutral, but it clarifies where
future performance work belongs.

## Ordered vocabulary cache

Location: `ordered_vocab.rs`.

Performance role: avoid rebuilding byte-sorted vocab and prefix trie repeatedly
for the same vocabulary.

Potential improvement: move cache configuration into typed compile options so
benchmarks can vary it without environment variables.

## CanMatch vocabulary quotient

Location: `vocab_equivalence.rs`.

Performance role: shrink token space before expensive global relation building.

Potential improvement: expose profile counters for class count, average class
size, and max class size.

## Grouped interval collector

Location: `collector.rs`.

Performance role: avoid expanding terminal/token facts one terminal at a time.

Potential improvement: separate its timing profile from the general CanMatch
profile and document asymptotic costs.

## Sweep-line materializer

Location: `vocab_materialize.rs`.

Performance role: convert interval facts into signatures in roughly event-order
rather than token × terminal × state order.

Potential improvement: extract active-group signature caching into a small type
so collision handling and key verification are more obvious.

## Sparse root path

Location: `root_collect.rs`.

Performance role: use a simpler algorithm for tiny cases.

Potential improvement: make thresholds data-driven after benchmarking instead of
environment-driven.

## Runtime scan execution

Location: `scan/execution.rs`.

Performance role: tight loop over token bytes using flattened transitions.

Potential improvement: add specialized fast paths for one-byte tokens and tokens
with no possible terminal matches only if profiling shows this matters.
