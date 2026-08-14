# Exact batching and reuse of parser admission during token commit

## Setting

For a fixed parser GSS `G`, let

```text
can_G(t)
```

mean that the GLR table can admit terminal `t` from `G` under the table's exact
admission semantics.  This includes reduction/guard simulation when the table
uses `AdmissionPolicy::ExactSimulation`.

A tokenizer continuation state `q` carries an exact possible-future terminal
set `F(q)`.  The historical commit loop retained `q` exactly when

```text
q == tokenizer_initial || exists t in F(q): can_G(t).
```

When one tokenizer execution has many continuation states, this repeated the
same parser reduction closure once for every `F(q)`.

## Batched theorem

For continuation states `q_1 .. q_n`, define

```text
U = union_i F(q_i)
A = { t in U | can_G(t) }.
```

`stack_admissible_terminals(table, G, U)` computes exactly `A`; for a certified
direct-regular frontier the corresponding direct support is the same exact
predicate.

For every continuation state `q_i`,

```text
F(q_i) intersects A
iff
exists t in F(q_i): can_G(t).
```

Proof: by definition `A` contains exactly those members of `U` satisfying
`can_G`. Since `F(q_i) subseteq U`, intersection with `A` is nonempty exactly
when one member of `F(q_i)` satisfies `can_G`. Therefore replacing `n` exact
existential simulations by one exact admitted-set computation plus `n` bitset
intersections preserves every continuation decision.

The initial tokenizer state remains an unconditional continuation exactly as
before.  A single non-initial continuation keeps the old existential query,
because its early exit can be cheaper than constructing the full admitted set.

## Exact cache

For one exact `ParserGSS` object the bounded cache stores:

- `tested`: terminals whose pointwise admission has been computed;
- `admitted`: the subset of `tested` for which `can_G(t)` is true;
- a bounded set of exact existential results for complete future-terminal sets.

For a new candidate set `C`, only `C - tested` is sent through the exact
admitted-set computation.  The cache then unions that delta into `tested` and
its exact admitted subset into `admitted`.  By induction, after every update:

```text
admitted = { t in tested | can_G(t) }.
```

Thus a future set wholly contained in `tested` is answered exactly by a bitset
intersection.  Cached boolean queries are also exact because their stored value
is the result of the historical exact existential predicate.

Cache identity uses `ParserGSS::ptr_eq` and the entry holds a strong GSS clone,
so an allocator cannot recycle the identity while an entry exists.  Capacity
eviction can only discard facts and cause recomputation.

The persistent cache is relevant to multi-state lexer frontiers.  Before a
commit beginning with a single runtime parser-state entry, the cache is cleared,
so its strong references cannot interfere with uniquely-owned in-place parser
fast paths. `CommitBuffers::reset_all` also clears the cache; ordinary scratch
clears retain it so consecutive parser-inert lexer commits can reuse exact
facts.

## Consequence

The optimization changes only how the boolean continuation predicate is
computed.  Tokenizer transition/match semantics, parser advances, future
terminal disallow, GSS merging, and the resulting runtime state are unchanged.
Resource bounds and cache eviction are performance choices only: on any miss the
implementation falls back to the exact historical predicate.
