# Mathematical contract

This file states the contract that all Chunk 06 code is meant to preserve.

## Sets and symbols

Let:

- `Q` be lexer states.
- `T` be grammar terminals.
- `V` be vocabulary tokens.
- `β(v)` be the byte string of token `v`.
- `δ` be the lexer transition function over bytes.
- `M(q)` be the terminals that complete at lexer state `q`.
- `F(q)` be the terminals that could still be completed from state `q` by some
  future byte suffix.

The implementation names `F(q)` as `CanMatch(q)`.

## Scan of one byte fragment

Scanning a token byte string from a lexer state is not a Boolean question.  It
returns a structured outcome:

```text
Scan(q, β(v)) = blocked(completed)
             or complete(completed, q_end)
             or partial(completed, q_partial)
```

The `completed` component is the set or sequence of terminals whose ending
boundary occurred while consuming bytes of `β(v)`.  The `q_end` or `q_partial`
component is the lexer state reached exactly at the token boundary.

The runtime `TokenizerExecResult` is still the concrete encoding used by commit,
but the new `src/scan/relation.rs` names this conceptual shape explicitly.

## Complete-boundary case

If scanning ends at a complete boundary, parser admissibility is about the
completed terminal sequence alone:

```text
end(q, β(v)) is boundary
⇒ admit only if parser accepts completed(q, β(v)).
```

The Terminal DWA is the compiled object used to accelerate completed terminal
sequence behavior, but it is not enough for the partial-boundary case.

## Partial-boundary case

If scanning ends inside a terminal, there is a future-completion side condition:

```text
end(q, β(v)) = q'
q' is non-boundary
⇒ admit only if parser can accept at least one t ∈ CanMatch(q').
```

This is the precise reason `CanMatch` is stored in the runtime artifact.  Masking
must know which internal tokens can eventually complete which terminals from a
partial lexer state.

## CanMatch relation

For a lexer state `q'`:

```text
CanMatch(q') = { t ∈ T | ∃ bytes s such that scanning s from q' completes t }
```

The implementation computes a token-indexed form because the LLM chooses
vocabulary tokens, not arbitrary byte strings.  The runtime table is organized by
terminal:

```text
RuntimeCanMatchByTerminal[t] : tokenizer-state-class ↦ internal-token-set
```

That is why `RuntimeCanMatchByTerminal` is a `BTreeMap<TerminalID, Weight>`.

## Token quotient induced by CanMatch

Two vocabulary tokens may share a scan-relation internal id iff their active
state/terminal label sets agree:

```text
v₁ ≡can v₂
iff
∀ q ∈ Q . CanMatch-after-scan(q, β(v₁)) = CanMatch-after-scan(q, β(v₂))
```

The implementation does not always materialize this equation literally.  It uses
trie traversal, interval maps, and sweep-line signatures.  But the quotient is
still exactly this relation.

## Non-reuse theorem

It is unsound to reuse Terminal-DWA vocabulary equivalence for CanMatch.

Terminal-DWA equivalence is based on completed terminal sequences.  CanMatch
requires future completions from partial states.  Two tokens can complete the
same terminals while ending in different non-boundary lexer states whose future
completions differ.

Therefore:

```text
v₁ ≡terminal-dwa v₂ does not imply v₁ ≡can v₂.
```

Chunk 06 encodes this by giving CanMatch its own `vocab_equivalence.rs` and by
putting warning comments at the subsystem boundary.

## Runtime/compile separation

Compile-time scan-relation code may inspect the entire vocabulary trie and all
lexer states.  Runtime commit must not do that.  Runtime commit scans one chosen
byte fragment from one current lexer state.

The split is:

```text
src/scan/execution.rs
    one concrete runtime byte scan

src/compile/scan_relation/**
    global construction of CanMatch weights for all relevant states/tokens
```

This separation is a mathematical boundary, not just a performance boundary.
Compile-time code constructs a relation; runtime code applies a transition.
