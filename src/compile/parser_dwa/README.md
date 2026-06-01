# Parser DWA subsystem

This directory contains the compile-time construction of the Parser DWA, the
weighted deterministic automaton over parser-stack prefixes used by runtime
`Mask`.

The central denotation is:

```text
[[PDWA]](rho)_(q,v) = 1  iff  rho ∈ E_(q,v)
```

where:

- `rho` is a parser-stack prefix, represented operationally as a word of parser
  state ids;
- `(q,v)` is a lexer-state / vocabulary-token pair;
- `E_(q,v)` is the stack-prefix language induced by all terminal sequences that
  can be scanned from lexer state `q` while reading token bytes `v`;
- weights are pair masks, so union/intersection/difference are set operations on
  `(q,v)` pairs.

## Reading order

1. `mod.rs`: denotation, boundary, and file guide.
2. `builder.rs`: phase ordering only. This is the best entrypoint.
3. `terminal_projection.rs`: how a Terminal-DWA state becomes terminal bundles.
4. `compose_nwa.rs`: the actual composition of Terminal-DWA branches with
   terminal stack-effect recognizers.
5. `determinize.rs`: two weighted subset constructions.
6. `optimize.rs`: default-edge and final-weight normalization.
7. `options.rs`: policy switches.
8. `profiling.rs`: construction profiles and the only profile-emission code.
9. `types.rs` and `labels.rs`: small local vocabulary.

## Current line counts

| file | lines |
| --- | ---: |
| `builder.rs` | 218 |
| `compose_nwa.rs` | 367 |
| `determinize/epsilon.rs` | 73 |
| `determinize/fallback.rs` | 211 |
| `determinize/mod.rs` | 18 |
| `determinize/outgoing.rs` | 99 |
| `determinize/support.rs` | 322 |
| `labels.rs` | 14 |
| `mod.rs` | 64 |
| `optimize.rs` | 252 |
| `options.rs` | 55 |
| `profiling.rs` | 220 |
| `terminal_projection.rs` | 157 |
| `types.rs` | 100 |

## Boundary rule

This directory owns Parser-DWA construction only. It may import:

- the GLR table and grammar analysis as parser-stack recognizer sources;
- the Terminal DWA as the terminal-language side of the construction;
- template automata as stack-effect recognizers;
- weighted automata and weights as representation machinery.

It must not own runtime Mask traversal, runtime Commit advancement, JSON Schema
lowering, tokenizer construction, or Terminal-DWA partitioning.
