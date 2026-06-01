# Chunk 05 review checklist

Use this as the acceptance checklist before starting Chunk 06.

## Boundary checks

- [ ] `src/compile/parser_dwa/mod.rs` states the denotation `[[PDWA]]`.
- [ ] `builder.rs` reads as phase orchestration, not implementation soup.
- [ ] Terminal-DWA grouping lives in `terminal_projection.rs`.
- [ ] Parser-NWA composition lives in `compose_nwa.rs`.
- [ ] Weighted subset construction lives in `determinize.rs`.
- [ ] Default/final-weight rewrites live in `optimize.rs`.
- [ ] Profile formatting lives in `profiling.rs`.
- [ ] Environment policy lives in `options.rs`.
- [ ] Raw label interpretation lives in `labels.rs`.

## Mathematical checks

- [ ] No file treats a `Weight` as anything other than a pair mask.
- [ ] `TerminalBundle` means terminals sharing a Terminal-DWA target.
- [ ] Productivity is computed by reverse reachability through accepting bundles.
- [ ] Template final states redirect to Terminal-DWA target continuations.
- [ ] Support sets are retained through first determinization.
- [ ] Default-edge optimization only covers possible parser-state labels.
- [ ] Final-weight subtraction subtracts only the source state's own final weight.
- [ ] Fallback determinization happens after default/final rewrites.
- [ ] Minimization is optional and semantics-preserving.

## Naming checks

- [ ] `rho` is used only in docs for parser stack prefix.
- [ ] `terminal` means grammar terminal, not vocabulary token.
- [ ] `token` means vocabulary token, not grammar terminal.
- [ ] `parser_state_label` means a raw automaton label interpreted as a GLR table
      state id.
- [ ] `continuation_state` means parser-NWA state corresponding to Terminal-DWA
      continuation.
- [ ] There are no new references to “mask game.”
- [ ] There are no comments saying “token loop” when they mean mask construction
      or LLM decoding loop.

## Static shape checks

- [ ] `builder.rs` is under 400 lines.
- [ ] `mod.rs` is not a dumping ground.
- [ ] No `eprintln!` occurs outside `profiling.rs` in `parser_dwa/`.
- [ ] Public crate API is unchanged by this chunk.
- [ ] The old wrapper name still exists for pipeline compatibility.

## Deferred compile-pass checks

These are intentionally deferred until the structural chunks are complete:

- [ ] import cleanup;
- [ ] rustfmt;
- [ ] cargo check;
- [ ] unit tests;
- [ ] performance regression check;
- [ ] profile-output compatibility check.
