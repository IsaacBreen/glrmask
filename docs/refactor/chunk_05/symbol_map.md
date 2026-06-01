# Parser-DWA symbol map

This map explains where old monolithic concepts live after Chunk 05.

| Concept | Old location | New location | Reason |
| --- | --- | --- | --- |
| Public build wrapper | `builder.rs` | `builder.rs` | Kept for compile-pipeline compatibility. |
| Named build inputs | none | `builder.rs` | Replaces long positional argument list for new code. |
| Named build output | none | `builder.rs` | Gives profile data a home. |
| Terminal bundle | `builder.rs` | `types.rs` | Local mathematical data carrier. |
| Target contributions | `builder.rs` | `types.rs` | Shared by determinization passes. |
| Parser-state label check | `builder.rs` | `labels.rs` | One place to interpret raw labels. |
| Minimize env override | `builder.rs` | `options.rs` | Policy separated from denotation. |
| Parser NWA build profile | `builder.rs` | `profiling.rs` | Profile record, not construction logic. |
| Compose detail profile | `builder.rs` | `profiling.rs` | Profile record, not construction logic. |
| Profile eprintln strings | `builder.rs` / composition | `profiling.rs` | Only one file formats profile output. |
| Terminal-DWA branch grouping | `builder.rs` | `terminal_projection.rs` | Projection from terminal object. |
| Bundle acceptance check | `builder.rs` | `terminal_projection.rs` | Terminal/template boundary. |
| Productive-state reverse search | `builder.rs` | `terminal_projection.rs` | Terminal continuation reachability. |
| Template splicing | `builder.rs` | `compose_nwa.rs` | Core composition step. |
| Bundle fragment cache | `builder.rs` | `compose_nwa.rs` | Parser-NWA construction detail. |
| Epsilon closure | `builder.rs` | `determinize.rs` | Weighted subset construction detail. |
| Support determinization | `builder.rs` | `determinize.rs` | Determinization phase. |
| Possible outgoing ids | `builder.rs` | `determinize.rs` + `types.rs` | Domain for fallback/defaults. |
| Fallback determinization | `builder.rs` | `determinize.rs` | Determinization phase. |
| Default optimization | `builder.rs` | `optimize.rs` | Semantics-preserving DWA rewrite. |
| Final-weight subtraction | `builder.rs` | `optimize.rs` | Semantics-preserving DWA rewrite. |

## Symbols whose names remain intentionally long

`build_parser_dwa_from_terminal_dwa_with_precomputed_templates`
: Kept as a compatibility wrapper.  It is verbose but stable for existing code.

`build_parser_dwa_from_terminal_dwa_with_templates`
: Preferred new entrypoint.  It says exactly which two finite objects are being
  composed.

`determinize_parser_dwa_with_fallbacks`
: Verbose but precise.  It is not ordinary DWA determinization; it bakes in
  fallback/default behavior.

`build_possible_outgoing_ids_by_state`
: Verbose but precise.  The result is not actual outgoing ids; it is the set of
  parser-state labels that may need default fallback coverage.
