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
**2**, whose payload is the named **RuntimePayloadV1** execution contract:

```text
GLRMASK\0 | u16 outer format version | u64 payload length | RuntimePayloadV1
```

RuntimePayloadV1 holds only persistent execution inputs: parser DWA, GLR table,
lexer/tokenizer, terminal matches, vocabulary maps, token bytes, and EOS metadata.
All dense masks, lookup tables, and other acceleration caches are rebuilt after load.
The envelope makes version rejection explicit and lets a later artifact representation
change without changing the browser session API.

## Session API

```rust
let artifact = RuntimeArtifact::from_bytes(bytes)?;
let mut session = Session::from_artifact(artifact)?;

let words: Vec<u32> = session.mask_words();
session.commit_token(token_id)?;
let eos_ok = session.eos_allowed();
session.reset();
```

`mask_words` is packed in original vocabulary ID space: bit `id % 32` in word
`id / 32` is set exactly when token `id` is admissible at the current state.

The artifact is intentionally tokenizer/vocabulary-specific. A TinyStories artifact
must not be used with a different tokenizer, even one that happens to have a similarly
sized vocabulary.
