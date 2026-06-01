# Exhaustive manual application notes for Chunk 05
This file is deliberately repetitive. It is meant for a basic implementer applying the refactor by hand.
## Step-by-step edits
### Step 1: Create parser_dwa/mod.rs
Replace the short module comment with a denotational header. Declare builder, compose_nwa, determinize, labels, optimize, options, profiling, terminal_projection, and types. Re-export only builder entrypoints and named input/output structs.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 2: Create types.rs
Move local type aliases and construction data carriers: TerminalBundle, BundleSignature, TargetContribs, Branch, StateSummary, StateSummaries, DeterminizedDwaWithSupports, CachedClosure, PossibleOutgoingIds, and contribution helpers.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 3: Create profiling.rs
Move elapsed_ms, ParserNwaBuildProfile, ParserDwaComposeDetailProfile, ParserDwaProfile, and profile emission helpers. Ensure eprintln appears only in this file.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 4: Create options.rs
Move the parser-DWA minimization environment override and wrap it in ParserDwaOptions.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 5: Create labels.rs
Move parser_state_label and document that it is the only raw-label-to-parser-state conversion.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 6: Create terminal_projection.rs
Move Terminal-DWA branch grouping, bundle signatures, template acceptance checks, state summary construction, and productivity analysis.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 7: Create compose_nwa.rs
Move DWA-to-NWA helper, template fragment appenders, bundle fragment cache logic, and build_parser_nwa_from_terminal_dwa.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 8: Create determinize.rs
Move PossibleOutgoingIds consumers, local epsilon closure, support-preserving determinization, and fallback determinization.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 9: Create optimize.rs
Move union_final_weight, default optimization, and final-weight subtraction.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 10: Rewrite builder.rs
Keep only phase orchestration. Add ParserDwaBuildInputs and ParserDwaBuildOutput. Keep old wrapper for compatibility.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 11: Add local README
Document reading order, denotation, and boundary rule in src/compile/parser_dwa/README.md.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

### Step 12: Update docs/parser_dwa.md
Record the mathematical relationship to Terminal DWA and list source files.
Checklist:
- Open the target file.
- Confirm the file owns only this concept.
- Confirm names distinguish terminal, token, parser state, lexer state, and pair-mask weight.
- Do not run cargo yet.

## Per-file import expectations
### `src/compile/parser_dwa/builder.rs`
Expected line count now: 219.

Imports currently present:

- `use std::time::Instant;`
- `use crate::Vocab;`
- `use crate::automata::weighted::dwa::DWA;`
- `use crate::automata::weighted::minimize::minimize;`
- `use crate::compiler::glr::analysis::AnalyzedGrammar;`
- `use crate::compiler::glr::table::GLRTable;`
- `use crate::compiler::stages::equiv_types::InternalIdMap;`
- `use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;`
- `use crate::compiler::stages::templates::Templates;`
- `use crate::compile::terminal_dwa::types::compile_profile_enabled;`
- `use super::compose_nwa::build_parser_nwa_from_terminal_dwa;`
- `use super::determinize::{`
- `use super::options::ParserDwaOptions;`
- `use super::optimize::{optimize_parser_dwa_defaults, subtract_final_weights_from_outgoing_dwa};`
- `use super::profiling::{elapsed_ms, ParserDwaProfile};`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/compose_nwa.rs`
Expected line count now: 367.

Imports currently present:

- `use std::sync::Arc;`
- `use std::time::Instant;`
- `use rustc_hash::FxHashMap;`
- `use crate::automata::weighted::dwa::DWA;`
- `use crate::automata::weighted::nwa::{NWA, NwaBody};`
- `use crate::compiler::glr::analysis::AnalyzedGrammar;`
- `use crate::compiler::stages::templates::Templates;`
- `use crate::ds::weight::Weight;`
- `use super::profiling::{`
- `use super::terminal_projection::{build_state_summaries, compute_productive_terminal_states};`
- `use super::types::StateSummaries;`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/determinize.rs`
Expected line count now: 668.

Imports currently present:

- `use std::collections::{hash_map::Entry, VecDeque};`
- `use rustc_hash::FxHashMap;`
- `use smallvec::SmallVec;`
- `use crate::automata::weighted::dwa::DWA;`
- `use crate::automata::weighted::nwa::NWA;`
- `use crate::compiler::glr::labels::DEFAULT_LABEL;`
- `use crate::ds::bitset::BitSet;`
- `use crate::ds::weight::Weight;`
- `use super::labels::parser_state_label;`
- `use super::types::{`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/labels.rs`
Expected line count now: 14.

Imports currently present:

- none

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/mod.rs`
Expected line count now: 64.

Imports currently present:

- none

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/optimize.rs`
Expected line count now: 252.

Imports currently present:

- `use std::collections::btree_map;`
- `use crate::automata::weighted::dwa::DWA;`
- `use crate::compiler::glr::labels::DEFAULT_LABEL;`
- `use crate::ds::bitset::BitSet;`
- `use crate::ds::weight::Weight;`
- `use super::labels::parser_state_label;`
- `use super::types::PossibleOutgoingIds;`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/options.rs`
Expected line count now: 55.

Imports currently present:

- none

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/profiling.rs`
Expected line count now: 220.

Imports currently present:

- `use std::time::Instant;`
- `use crate::compiler::stages::templates::BundleBuildProfile;`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/terminal_projection.rs`
Expected line count now: 157.

Imports currently present:

- `use std::collections::{BTreeMap, VecDeque};`
- `use rustc_hash::FxHashMap;`
- `use crate::automata::weighted::dwa::DWA;`
- `use crate::automata::weighted::nwa::NWA;`
- `use crate::compiler::glr::analysis::AnalyzedGrammar;`
- `use crate::compiler::stages::templates::Templates;`
- `use crate::ds::weight::Weight;`
- `use crate::grammar::flat::TerminalID;`
- `use super::types::{Branch, BundleSignature, StateSummaries, StateSummary, TerminalBundle};`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

### `src/compile/parser_dwa/types.rs`
Expected line count now: 100.

Imports currently present:

- `use std::collections::BTreeMap;`
- `use smallvec::SmallVec;`
- `use crate::automata::weighted::dwa::DWA;`
- `use crate::ds::weight::Weight;`
- `use crate::grammar::flat::TerminalID;`

Possible later compile-pass cleanup: remove any unused imports after the structural refactor sequence is finished.

## Do not accidentally change these semantics
1. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
2. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
3. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
4. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
5. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
6. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
7. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
8. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
9. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
10. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
11. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
12. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
13. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
14. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
15. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
16. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
17. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
18. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
19. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
20. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
21. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
22. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
23. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
24. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
25. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
26. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
27. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
28. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
29. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
30. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
31. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
32. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
33. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
34. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
35. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
36. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
37. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
38. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
39. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.
40. If two alternative paths produce the same pair `(q,v)`, the weight must be unioned, not overwritten.

## A basic non-mathematical sanity test for reviewers
Read `builder.rs` aloud. It should sound like a recipe, not like an algorithm textbook. Then read each callee file. Each callee file should sound like exactly one paragraph of the mathematical construction.
