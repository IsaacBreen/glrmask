# Parser advance and stack-effect semantics

## Reference relation

The reference parser advance is still delegated to `parser_advance.rs`, which chooses between template-DFA acceleration and the reference GLR-table exploration. This chunk does not change that dispatch.

## Commit-specific shortcuts

`single_top.rs` contains shortcuts for the common case where a parser frontier has a single concrete top state or a single concrete stack path. These shortcuts are Commit-local because they are not a new parser semantics. They are specialized implementations of the same stack-effect relation.

## Review rule

A shortcut is acceptable only if it has the same denotation as `advance_parser_stacks`. When in doubt, the shortcut should be disabled or guarded by validation rather than treated as a semantic source of truth.
