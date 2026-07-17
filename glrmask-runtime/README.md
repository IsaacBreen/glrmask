# glrmask-runtime

`glrmask-runtime` is the execution-only half of glrmask. It deliberately compiles no
schema, grammar, regular expression, tokenizer vocabulary, or parser automaton at
runtime. It loads a fully built, vocabulary-specific constraint artifact and exposes
an owned decoding session.

The runtime source graph contains only the modules needed to execute a compiled
constraint: lexer DFA, weighted parser DWA, GLR table/parser, GSS, mask materializer,
and token commit path. JSON Schema importing and the compiler pipeline remain in the
parent `glrmask` crate.

## Artifact boundary

`RuntimeArtifact` uses a versioned envelope. The current outer format version is
**5**, whose payload is the **RuntimePayloadV3** execution contract:

```text
GLRMASK\0 | u16 outer format version | u64 payload length | RuntimePayloadV3
```

RuntimePayloadV3 holds only persistent execution inputs: parser DWA, GLR table,
lexer/tokenizer, terminal matches, vocabulary maps, token bytes, and exact-token
terminals. All dense masks, lookup tables, and other acceleration caches are rebuilt
after load. Older artifact versions are rejected explicitly and must be recompiled.

## Loaded constraint and session API

```rust
let artifact = RuntimeArtifact::from_bytes(bytes)?;
let runtime = RuntimeConstraint::from_artifact(artifact)?;

// Cheap: each session shares the already-loaded immutable executor.
let mut session = runtime.start();

let mut words = vec![0; runtime.mask_len()];
session.fill_mask(&mut words);
session.commit_token(token_id)?;
let finished = session.is_finished();
session.reset();
```

`fill_mask` is allocation-free. `mask_words` remains available as a convenience
method when an owned vector is appropriate. Both use original vocabulary ID space: bit `id % 32` in word
`id / 32` is set exactly when token `id` is admissible at the current state.

The artifact is intentionally tokenizer/vocabulary-specific. A TinyStories artifact
must not be used with a different tokenizer, even one that happens to have a similarly
sized vocabulary.
