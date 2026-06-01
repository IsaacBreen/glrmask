# Proof obligations introduced or clarified by this chunk

The chunk is mostly a refactor, so its proof obligations are preservation
obligations.

## State split preservation

For every method moved out of `state.rs`, show that the receiver and field
accesses are unchanged.  The modules are separate, but Rust `impl` blocks on the
same type preserve method lookup.

## Mask dense accumulator split preservation

`DenseMaskAcc` moved to `dense_acc.rs`.  Because the type is still private to the
`mask` module hierarchy via `pub(super)`, the external API is unchanged.  The
main proof obligation is that field visibility is sufficient only for the parent
phase graph and not exported outside `runtime/mask`.

## Mask bitset split preservation

The bitset functions operate only on caller-provided buffers and token ids.  No
state or constraint field moved with them.  Their behavior is syntactically
identical except for module qualification.

## Commit parser advance split preservation

The functions in `parser_advance.rs` still perform the same branch:

```text
if template DFA enabled and applicable:
    use template DFA result
    optionally validate against GLR table
else:
    use GLR table result
```

Thus the transition relation is unchanged if the template-DFA implementation is
unchanged.

## Commit mask assertion split preservation

The assertion functions are observational.  Moving them out cannot affect normal
Commit unless the import is wrong.  If disabled, they return `None` and do
nothing.  If enabled, they may panic on mismatch exactly as before.

## Rename preservation

`end_state_can_advance` is a naming correction only.  It still means: a tokenizer
end state is admissible if it is the tokenizer initial state or if the parser
stack can advance on some terminal accepted from that tokenizer state.
