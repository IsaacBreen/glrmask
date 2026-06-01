# Runtime artifact module

This directory is the runtime-side boundary between compiled mathematics and
fast decoding execution.

The serialized part of a `Constraint` is the compiled artifact:

1. Parser DWA: admissible parser-stack prefixes to token masks.
2. GLR table: parser transition/reduction semantics.
3. Tokenizer DFA: byte-level lexer state machine.
4. CanMatch relation: lexer-state/token pairs that can complete partial scans.
5. Token-space quotients: maps between original model token ids and final
   runtime-internal token ids.
6. Terminal display names and token byte tables.

The nonserialized part is runtime cache:

1. Dense/sparse mask materialization tables.
2. Fast Parser-DWA transition arrays.
3. Fast tokenizer transition arrays.
4. Seed-state dense masks.
5. Weight-to-mask caches.
6. Heavy-token and word-group shortcuts.

The publication invariant is that compilation and deserialization may construct
the semantic artifact, but only artifact finalization may populate the runtime
cache fields.
