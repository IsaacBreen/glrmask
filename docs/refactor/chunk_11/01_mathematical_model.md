# Mathematical model of the runtime

Let `C` be the immutable compiled constraint.  It contains the tokenizer, the
GLR table, the Parser DWA, the scan relation / CanMatch tables, terminal names,
token-space quotients, and runtime caches derived from those structures.

A runtime state is a finite map:

```text
S : TokenizerStateId -> ParserGSS
```

where each `ParserGSS` compactly represents a set of parser stacks plus delayed
longest-match exclusions.  This map is the semantic content of
`ConstraintState`.  Everything else in `ConstraintState` is operational:
allocation reuse, cached mask output, and generation counters.

Mask is a read-only function:

```text
Mask_C(S) -> Bitset(OriginalTokenId)
```

It is implemented by walking every active parser-stack prefix through the Parser
DWA, intersecting/combining encountered weights, and materializing the resulting
internal-token set into the caller's original token-id universe.

Commit is a transition relation:

```text
Commit_C(S, bytes) -> Result<S', Reject>
```

It scans bytes from every active tokenizer state, enumerates completed terminal
boundaries, advances parser stacks through the terminal sequence, stores partial
lexer states when a terminal is incomplete, and replaces the frontier with the
successor map.

This chunk encodes three principles in the file tree:

1. **Semantic state is small.** The frontier map and generation are the only
   live state that matter to the language recognized.
2. **Caches are discardable.** A correct runtime may delete mask caches and
   scratch buffers between calls.
3. **Debug oracles are not transitions.** The optional assertion that Commit
   agrees with Mask is deliberately separated from the Commit transition.

The paper talks about Mask and Commit as separate online operations.  The code
now mirrors that separation more strongly: Mask helper modules sit under
`runtime/mask`, Commit helper modules sit under `runtime/commit`, and the shared
live configuration sits under `runtime/state`.
