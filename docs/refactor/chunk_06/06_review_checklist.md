# Review checklist

Use this checklist before accepting Chunk 06.

## Shape checks

- [ ] `src/compile/scan_relation/mod.rs` is a facade, not an algorithm file.
- [ ] No new file in `src/compile/scan_relation/` exceeds the old monolithic
      file size.
- [ ] `src/scan/` exists and is crate-private.
- [ ] `runtime/commit/tokenizer_scan.rs` delegates primitive scan execution.
- [ ] `legacy_materialize.rs` is not imported outside `compile::scan_relation`.

## Mathematical checks

- [ ] Documentation explicitly states that Terminal-DWA equivalence is not
      CanMatch equivalence.
- [ ] The partial lexer-state case is documented in source comments.
- [ ] Runtime commit does not construct global CanMatch tables.
- [ ] Compile-time scan-relation construction does not mutate runtime state.
- [ ] `CanMatch` vocabulary quotienting is local to `vocab_equivalence.rs`.

## Naming checks

Run:

```bash
rg -n 'constraint_possible_matches|PossibleMatches|possible_matches|possible_match|pmv|PMV' src
```

Expected: no source hits except explanatory comments that explicitly say the old
phrase is vague/deprecated.

Run:

```bash
rg -n 'Terminal-DWA equivalence|CanMatch equivalence|partial lexer' src/compile/scan_relation src/scan docs/refactor/chunk_06
```

Expected: hits in the subsystem boundary docs and mathematical contracts.

## Boundary checks

- [ ] `ordered_vocab.rs` does not import parser-DWA code.
- [ ] `vocab_equivalence.rs` does not import Terminal-DWA ID maps.
- [ ] `vocab_materialize.rs` does not know about runtime commit.
- [ ] `scan/execution.rs` does not import compile-pipeline modules.
- [ ] `compute.rs` is the only scan-relation file called directly by pipeline
      orchestration.

## Deferred compile checks

This chunk intentionally did not compile.  When compilation is resumed, expect
first-pass issues to be import visibility, unused imports, and module privacy.
Fix those mechanically without collapsing modules back together.
