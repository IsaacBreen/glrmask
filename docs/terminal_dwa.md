# Terminal DWA

The Terminal DWA is the weighted automaton over completed grammar-terminal
sequences.  It is the compiled object that connects lexer behavior to token
masks.

## Denotation

For a terminal sequence `r`, evaluating the Terminal DWA returns a Boolean mask
over `(lexer state, token)` pairs.  A pair appears in the output exactly when
scanning that token's bytes from that lexer state can emit the completed
terminal sequence `r`.

This object is independent of parser stacks.  Parser-stack information enters
later through the Parser DWA.

## Code ownership after chunk 02

```text
src/compile/terminal_dwa/
  mod.rs                 top-level builder and partition orchestration
  classify.rs            terminal path classification and partition-cost choices
  grammar_helpers.rs     terminal coloring and follow-set helpers
  direct_partition/      single-step construction path
  pair_partition/        multi-step construction path
  merge.rs               merge local partition artifacts
  partition.rs           per-vocab-partition orchestration
  types.rs               shared Terminal-DWA build types
```

The old path `src/compiler/stages/id_map_and_terminal_dwa/` was retired because
it named an implementation bundle rather than the paper object.

## Direct partition vs pair partition

The old names `l1` and `l2p` were local shorthand.  They are now replaced by
semantic names:

- **direct partition**: terminals whose relevant paths can be represented by the
  direct single-step construction;
- **pair partition**: terminals requiring the full multi-step NWA-based
  construction because token bytes can cross multiple terminal boundaries.

The words “direct” and “pair” are not part of the paper's core theorem, but they
are descriptive implementation subobjects.  They are preferable to level numbers
because they tell the reader what changes mathematically.

## Future cleanup

The Terminal-DWA subsystem still contains large implementation files.  Later
chunks should split:

```text
terminal_dwa/direct_partition/mod.rs
terminal_dwa/pair_partition/mod.rs
terminal_dwa/classify.rs
```

into smaller files organized around token quotient construction, vocabulary trie
walks, equivalence analysis, and DWA materialization.

## Chunk 04 source boundary update

Terminal-DWA construction is now split by mathematical role:

| File | Role |
|---|---|
| `src/compile/terminal_dwa/mod.rs` | Boundary and re-exports. |
| `src/compile/terminal_dwa/options.rs` | Environment-driven build policy. |
| `src/compile/terminal_dwa/vocab_partition.rs` | Caller-vocabulary partition selection. |
| `src/compile/terminal_dwa/global_state_map.rs` | Global tokenizer-state quotient shared by Terminal-DWA and scan-relation work. |
| `src/compile/terminal_dwa/builder.rs` | Top-level orchestration: choose partitions, build locals, merge. |
| `src/compile/terminal_dwa/partition.rs` | Build one local partition by combining direct and pair paths. |
| `src/compile/terminal_dwa/direct_partition/` | Direct/single-step construction. |
| `src/compile/terminal_dwa/pair_partition/` | Multi-step construction. |
| `src/compile/terminal_dwa/merge.rs` | Local/global id-map and DWA reconciliation. |

This split is meant to protect the denotation of the Terminal DWA from its construction heuristics.  Different partition strategies may change compile time and intermediate automaton size, but a sound strategy must not change `[[TDWA]]`.
