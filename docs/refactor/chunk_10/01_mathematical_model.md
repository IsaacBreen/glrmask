# Mathematical model of the runtime artifact

## 1. The artifact as a tuple

After compilation, the runtime object can be read as the tuple

```text
A = (P, G, L, C, Q_tok, Q_state, T, B, E)
```

where:

- `P` is the Parser DWA;
- `G` is the parser table;
- `L` is the tokenizer/lexer DFA;
- `C` is the CanMatch relation;
- `Q_tok` is the quotient from original vocabulary tokens to final internal
  token ids;
- `Q_state` is the quotient from original tokenizer states to final internal
  tokenizer-state ids;
- `T` is the family of terminal stack-effect template DFAs;
- `B` is the byte table for original and internal tokens;
- `E` is optional end-of-sequence and JSON escape prefix metadata.

The former source layout did not make this tuple visible.  The new layout does.

## 2. Semantic versus operational data

A semantic field participates in the denotation of the accepted token sequences.
A cache field is a deterministic function of semantic fields and chosen runtime
optimization policy.

For example:

```text
semantic: parser_dwa
semantic: tokenizer
semantic: can_match
semantic: original_token_to_internal
cache: word_group_sparse_masks
cache: weight_token_dense_masks
cache: tokenizer_fast_transitions
cache: all_tokens_buf_mask
```

The crucial property is:

```text
load(save(A))  =  A'  such that  L(A') = L(A)
```

where `L(A)` denotes the language of token sequences accepted by the compiled
constraint.  Cache fields may differ across versions if they preserve this
language.

## 3. Runtime finalization as a total function

Runtime finalization is a function:

```text
finalize : SemanticArtifact -> SemanticArtifact × RuntimeCaches
```

It must not change the language.  It may only add lookup tables and
materialization aids.

This chunk makes `Constraint::rebuild_runtime_caches` the single entry point for
that function.

## 4. Compile finalization as packaging, not cache construction

The compile pipeline now constructs `CompiledArtifactParts`.  That value is not
a runtime cache object.  It is the semantic output of compilation.

The compile finalizer then performs:

```text
parts -> Constraint::from_compiled_parts(parts) -> rebuild_runtime_caches()
```

This eliminates the old problem where compile finalization needed to enumerate
all empty cache fields directly.  That enumeration belongs to the runtime
artifact module.

## 5. Token-space quotient invariant

Let `orig` range over original token ids and `int` range over final internal
token ids.

The quotient maps must satisfy:

```text
original_token_to_internal[orig] = int
  iff
orig ∈ internal_token_to_tokens[int]
```

except that `u32::MAX` marks an original token that is excluded from the final
runtime-internal token space.

The quotient is final: it already includes every split needed to reconcile
Parser-DWA weights and CanMatch weights.  Runtime code must not treat this map
as a provisional Terminal-DWA compaction.

## 6. Tokenizer-state quotient invariant

Let `q` be an original tokenizer state and `q̂` be a final internal tokenizer
state id.

```text
state_to_internal_tsid[q] = q̂
```

and

```text
q ∈ internal_tsid_to_states[q̂]
```

CanMatch and Parser-DWA weights are interpreted over these final internal state
ids after reconciliation.

## 7. Cache correctness invariant

Every dense/sparse output-mask cache must materialize a set of original token
ids, not internal token ids.  Internal token ids are implementation coordinates;
the public mask is always over the original vocabulary.

Thus the cache builder is a homomorphism from internal-token sets to
original-token bitsets:

```text
materialize(S ∪ T) = materialize(S) OR materialize(T)
materialize(∅)     = zero mask
```

and complement shortcuts must be equivalent to starting from the universal
original-token mask and subtracting missing internal-token classes.
