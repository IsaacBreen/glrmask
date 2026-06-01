# Mathematical contracts for Terminal DWA construction

## Denotation

Let `Q` be the tokenizer DFA state set, `V` the caller vocabulary, and `β(v)` the byte string of token `v`.  Let `T` be the grammar terminal alphabet.  The lexer induces a relation:

```text
Lex(q, β(v)) ⊆ T*
```

where `r ∈ Lex(q, β(v))` means: scanning the bytes of token `v` starting from lexer state `q` can complete exactly the terminal sequence `r` within the token bytes, with no incomplete terminal included in `r`.

The Terminal DWA is a weighted deterministic automaton over `T*` whose value is a Boolean mask over `Q × V`:

```text
[[TDWA]](r) = { (q, v) ∈ Q × V : r ∈ Lex(q, β(v)) }.
```

This is the only semantic object this subsystem is responsible for producing.  All other structures are auxiliary.

## Vocabulary partitioning is a coproduct construction

Suppose the vocabulary is partitioned into disjoint subsets:

```text
V = V_0 ⊔ V_1 ⊔ ... ⊔ V_{k-1}.
```

For each `i`, the local Terminal DWA denotes:

```text
[[TDWA_i]](r) = { (q, v) ∈ Q × V_i : r ∈ Lex(q, β(v)) }.
```

The global automaton must denote the union:

```text
[[TDWA]](r) = ⋃_i [[TDWA_i]](r).
```

The merge pass is therefore not an optimization afterthought; it is the categorical coproduct/reconciliation step that turns local id spaces back into a single global id space.  This is why `merge.rs` deserves first-class status.

## Tokenizer-state quotienting is not vocabulary partitioning

A tokenizer-state quotient is a map:

```text
π_Q : Q -> Q'
```

It may be sound for a construction phase if all states in each fibre of `π_Q` are indistinguishable with respect to the token/terminal behaviours the phase needs.  The global max-length state map is one such quotient.  It is shared by Terminal-DWA and scan-relation work, but it is not itself the Terminal DWA.

This chunk separates `global_state_map.rs` from `vocab_partition.rs` because these two quotients act on different axes:

- vocabulary partitioning splits `V`;
- state quotienting collapses `Q`.

Conflating them makes later correctness arguments almost impossible to read.

## Direct partition

The direct partition handles terminals whose relevant token paths are at most one completed-terminal step.  Its local DWA can be built directly because the terminal sequence dimension is shallow.  The proof obligation is roughly:

```text
for active direct terminal t:
  weight(t) = { (q, v) : scanning β(v) from q emits exactly t in the direct case }
```

The direct builder may compute exact state equivalence by token signatures, but that equivalence is local to the direct construction path.

## Pair partition

The pair partition handles multi-step terminal paths, where a token can complete several terminals.  Its construction goes through NWA/DWA machinery and more involved state/vocab equivalence.  Its proof obligation is the same denotation restricted to the multi-step terminal set, but its algorithmic route is different.

## Merge proof obligation

If direct and pair local builders are run for the same sub-vocabulary, their terminal masks are disjoint by construction of the terminal path-length masks.  The per-partition merge must therefore preserve:

```text
[[LocalTDWA]](r) = [[DirectTDWA]](r) ∪ [[PairTDWA]](r).
```

If multiple vocabulary partitions are then built, the global merge must preserve:

```text
[[GlobalTDWA]](r) = ⋃_partition [[LocalTDWA_partition]](r).
```

Both merges must also reconcile id maps.  The automaton language alone is insufficient; the weight coordinates must still refer to the intended original tokenizer states and caller token ids after every quotient.

## Why `options.rs` is mathematically important

It may look like mere cleanup, but isolating options protects the mathematical narrative.  Environment variables choose among construction strategies.  They are not part of `Lex`, not part of `[[TDWA]]`, and not part of the runtime semantics.  If those variables sit beside the denotation, readers can accidentally infer that a different option means a different object.  The intended invariant is stronger:

```text
same grammar + same tokenizer + same vocab => same Terminal-DWA denotation
```

for every sound option setting.  Options may change size, time, partition count, and profile lines.  They must not change the accepted relation.
