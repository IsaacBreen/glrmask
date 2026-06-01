# `CompiledArtifactParts` contract

## Why this struct exists

A compile pipeline should not know the full memory layout of `Constraint`.
Compilation produces semantic objects: Parser DWA, GLR table, tokenizer, CanMatch,
quotient maps, token bytes, and template DFAs.  Runtime finalization decides how
many caches to allocate and how to rebuild them.

`CompiledArtifactParts` is the handoff object between those two worlds.

## Fields

The fields are:

```text
parser_dwa
```

The compiled Parser DWA.  Its weights are already in the final reconciled
internal token/state coordinate system.

```text
table
```

The GLR table used by Commit and diagnostics.

```text
terminal_display_names
```

Human-facing terminal names.  These do not change acceptance, but they are
semantic diagnostics attached to grammar terminals.

```text
tokenizer
```

The lexer/tokenizer DFA over bytes.

```text
ignore_terminal
```

Optional ignored terminal id, such as whitespace.

```text
can_match
```

CanMatch relation keyed by grammar terminal.  Its weights are already
reconciled with Parser-DWA weights.

```text
state_to_internal_tsid
internal_tsid_to_states
```

Final tokenizer-state quotient maps.

```text
template_dfas_by_terminal
```

Stack-effect recognizers used by Commit.  These are semantic accelerators: they
represent parser stack effects but are not needed to define the accepted token
language if the table is present.

```text
original_token_to_internal
internal_token_to_tokens
```

Final token quotient maps.

```text
eos_token_id
```

Optional EOS id in the original vocabulary.

```text
token_bytes
internal_token_bytes
```

Byte representatives for original and final internal tokens.

## Constructor semantics

`Constraint::from_compiled_parts(parts)` does exactly three conceptual things:

1. stores the semantic fields;
2. computes `json_u_prefix_token_id` from token bytes;
3. installs empty runtime cache fields via `RuntimeCaches::default()`.

It does not run the cache builder.  The caller explicitly invokes
`rebuild_runtime_caches` so that profiling can attribute finalization time.

## Why not construct `Constraint` directly?

Direct construction creates a dependency from compile finalization to every
optimization cache.  That is backwards.  Optimizations can change without
changing the mathematical artifact.  The handoff struct preserves that boundary.

## Future evolution

The next possible step is:

```rust
pub struct Constraint {
    artifact: Arc<CompiledArtifact>,
}
```

where `CompiledArtifact` stores the current semantic fields and owns a separate
`RuntimeCaches` value.  This chunk deliberately avoids that deeper migration so
that downstream code does not need to be rewritten before Mask/Commit cleanup.
