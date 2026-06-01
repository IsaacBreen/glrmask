# Parser DWA

The Parser DWA is the compiled parser-side automaton used by `Mask`.  It reads
parser stack prefixes and returns a mask over lexer-state/token pairs.

```text
[[PDWA]](rho)_(q,v) = 1  iff  rho ∈ E_(q,v)
```

The important design choice is that the Parser DWA is not “the LR parser” or
“the GLR parser.”  It is the finite object obtained after composing parser
stack-effect recognizers with the Terminal DWA.  The current compiler obtains
those recognizers from a GLR table, but the runtime artifact only depends on the
recognizer language, not on the parser construction that produced it.

## Relationship to the Terminal DWA

The Terminal DWA answers a terminal-string question:

```text
[[TDWA]](r)_(q,v) = 1  iff  r ∈ Lex(q, beta(v)).
```

The Parser DWA answers a stack-prefix question:

```text
[[PDWA]](rho)_(q,v) = 1  iff  rho ∈ E_(q,v).
```

The connection is:

```text
E_(q,v) = ⋃_{r ∈ Lex(q,beta(v))} E_r
```

where `E_r` is the stack-prefix language accepted by the parser stack-effect
recognizer for terminal sequence `r`.  The Parser-DWA construction realizes this
union finitely: Terminal-DWA edges supply weighted terminal choices, while
terminal template automata supply stack-effect languages.

## Construction phases

1. **Terminal projection.** For each Terminal-DWA state, group outgoing terminal
   transitions by target state.  Each group is a terminal bundle: a finite map
   from terminal id to pair-mask weight.
2. **Productivity.** A Terminal-DWA state is productive if it can reach a final
   pair mask through accepting terminal templates.  Nonproductive states are
   omitted from the parser NWA.
3. **Composition.** Productive Terminal-DWA states become continuation states.
   Each branch is replaced by a template fragment.  Template final states are
   redirected by epsilon edges to the branch continuation.
4. **Negative-label resolution.** Template-local negative labels are resolved
   before deterministic parser-state labels are interpreted.
5. **Support determinization.** The composed NWA is determinized while retaining
   NWA support sets for each DWA state.
6. **Possible-outgoing computation.** Support sets determine which parser-state
   labels can legally receive default fallback behavior.
7. **Default optimization.** Repeated parser-state edges are compressed into
   default edges when this preserves semantics.
8. **Final-weight subtraction.** Final masks are lifted out of outgoing edges so
   runtime can account for terminal-boundary acceptance cleanly.
9. **Fallback determinization.** Default-edge semantics are made explicit in a
   deterministic weighted automaton.
10. **Optional minimization.** Weighted-DWA minimization may be run, but is
    currently skipped by default for compile-time performance.

## Source files

The implementation lives under `src/compile/parser_dwa/`:

- `builder.rs`: phase ordering and named build input/output structs.
- `terminal_projection.rs`: Terminal-DWA summaries and terminal bundle interning.
- `compose_nwa.rs`: composition of Terminal-DWA continuation states with parser
  stack-effect templates.
- `determinize.rs`: weighted subset construction and fallback determinization.
- `optimize.rs`: semantics-preserving default/final-weight rewrites.
- `profiling.rs`: profile records and all textual emission.
- `options.rs`: construction policy.
- `types.rs`: local data carriers.
- `labels.rs`: raw-label-to-parser-state interpretation.

## Naming constraints

Use these names consistently:

- `terminal_dwa`: the DWA over grammar-terminal sequences.
- `parser_dwa`: the DWA over parser-stack prefixes.
- `template`: a stack-effect recognizer for a terminal.
- `terminal_bundle`: terminals that share a Terminal-DWA source and target.
- `continuation_state`: the parser-NWA state corresponding to a Terminal-DWA
  state.
- `pair_mask` or `weight`: the set of `(lexer_state, token)` pairs carried by a
  weighted edge or final state.
- `parser_state_label`: a nonnegative automaton label interpreted as a parser
  stack-state id.

Avoid saying “token loop” when the algorithm is discussing runtime mask
construction.  Tokens are LLM outputs; Parser-DWA construction is a compile-time
composition over parser stack prefixes.
