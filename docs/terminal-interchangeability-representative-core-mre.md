# Minimal representative-core terminal-interchangeability witness

`tests/ti_representative_core_mre.rs` is a minimal diagnostic witness for one
specific claim:

> Building the terminal DWA with only one member of an interchangeable terminal
> class is not semantics-preserving on its own. The construction needs explicit
> reconstruction of the omitted member's outputs.

It is **not** a proof that every possible post-DWA reconstruction scheme fails.
For this tiny example, a reconstruction that knows to add the omitted initial
edge can repair the result. The normal direct-transport builder does exactly
that and agrees with the baseline.

## Reproducer

Vocabulary, with one token:

```text
0 = "c"
```

Grammar:

```lark
start: A B | B A
A: "ca"
B: "cb"
```

Terminal labels are `A = 0`, `B = 1`. The test forces the L2+ path and enables
terminal interchangeability.

## Raw scanner DFA

The raw tokenizer DFA has four states:

| Raw scanner state | Meaning | Transitions | Finalizers | Possible future terminals |
|---|---|---|---|---|
| `s0` | Initial scanner state | `c -> s1` | none | `{A, B}` |
| `s1` | After consuming the token byte `c` | `a -> s2`, `b -> s3` | none | `{A, B}` |
| `s2` | After `ca` | none | `{A}` | none |
| `s3` | After `cb` | none | `{B}` | none |

The token `c` does not complete either terminal. It leaves the scanner at
`s1`, where both `A` and `B` remain possible future terminal completions.

## Interchangeability plan

With only byte `c` relevant to the vocabulary, the detector picks `A` as the
representative and makes `B` its member:

```text
representative_for      = [A, A]
active_representatives  = [true, false]
```

The directed scanner transport for `A -> B` is a relation, not a chosen
one-to-one function:

```text
s0 -> {s0}
s1 -> {s1, s2, s3}
s2 -> {s1, s2, s3}
s3 -> {s1, s2, s3}
```

This is sufficient for the detector because the characterization is restricted
to the vocabulary byte `c`. State `s0` remains fixed; after that, `s1`, `s2`,
and `s3` cannot be distinguished by any further relevant-byte transition.

## TSIDs in the local DWA artifacts

A TSID is an internal **tokenizer-state class**, not a fixed global DFA-state
number. Compaction may renumber its classes independently in each artifact.

The representative-only core has:

```text
raw scanner state -> TSID: [0, 1, 1, 1]
TSID 0 = {s0}
TSID 1 = {s1, s2, s3}
```

The feature-suppressed baseline has the same partition but the opposite local
numbering:

```text
raw scanner state -> TSID: [1, 0, 0, 0]
TSID 0 = {s1, s2, s3}
TSID 1 = {s0}
```

The sole vocabulary token remains internal token `0` in both artifacts. The
only semantically relevant coordinate in this witness is therefore the
original pair `(s0, token 0)`.

## Terminal DWAs

All three artifacts minimize to two DWA states:

```text
q1 = start
q0 = accepting
```

`{s0:t0}` below means the edge/final weight admits original scanner state `s0`
and original token `t0 = "c"`.

### Baseline

```text
q1 -[A; {s0:t0}]-> q0
q1 -[B; {s0:t0}]-> q0
q0 final {s0:t0}
```

### Representative-only core

```text
q1 -[A; {s0:t0}]-> q0
q0 final {s0:t0}
```

The `B` edge is absent because `B` was removed from the active terminal mask
before the trie walk. The core never records the token `c` as a future `B`
terminal at `s0`.

### Normal direct transport

```text
q1 -[A; {s0:t0}]-> q0
q1 -[B; {s0:t0}]-> q0
q0 final {s0:t0}
```

Direct transport preserves the complete scanner alphabet and reconstructs `B`
through the transport mode while building the NWA. Its completed terminal DWA
therefore matches the baseline exactly.

## Exact failed query

The artifact comparator reports:

```text
partition=p2
original scanner state = s0
original token = t0 = "c"
terminal word = [1] = [B]
baseline accepts = true
representative-only core accepts = false
```

The DWA word `[B]` here means that after consuming token `c` from scanner state
`s0`, `B` is an admissible **future terminal completion**. It does not claim
that `c` itself fully matches `B`.

## Test command

```bash
cargo test --test ti_representative_core_mre
```

The test first catches the expected exact-comparator mismatch for the diagnostic
representative-only core. It then disables that diagnostic switch and checks
that the normal direct-transport construction compiles the identical grammar.
