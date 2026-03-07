//! Typed compiler debug artifacts.
//!
//! Captures intermediate automata from each stage of the compilation pipeline
//! without relying on env-var printing. Returned alongside the Constraint by
//! [`compile_with_debug`].
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::{EOF, AnalyzedGrammar};
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{GrammarDef, TerminalID};
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::compiler::stages::templates::compile::TemplateBundle;
use crate::compiler::terminal_dwa::TerminalDWA;

// ---------------------------------------------------------------------------
// Terminal-side debug stages
// ---------------------------------------------------------------------------

/// Snapshots of the terminal NWA at each stage of `build_terminal_dwa`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerminalDebug {
    /// Terminal NWA immediately after `build_terminal_dwa_nwa` (raw vocab walk),
    /// before any follow-path optimisations.
    pub nwa_after_build: NWA,

    /// Terminal NWA after `collapse_always_allowed` but before
    /// `prune_disallowed_follows`.
    pub nwa_after_collapse: NWA,

    // The final terminal NWA (after prune_disallowed_follows) lives in
    // `CompileDebug::terminal_dwa.nwa`.
}

// ---------------------------------------------------------------------------
// Automata-only debug (returned by build_parser_dwa_impl)
// ---------------------------------------------------------------------------

/// Intermediate automata captured during DWA construction.
///
/// This is the subset of debug data that `build_parser_dwa_with_debug`
/// can produce on its own. [`compile_with_debug`] combines this with
/// grammar-level metadata to form the full [`CompileDebug`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AutomataDebug {
    /// Terminal characterizations: terminal → parser characterization.
    pub characterizations: BTreeMap<TerminalID, TerminalCharacterization>,

    /// Terminal DWA (final, after collapse + prune).
    pub terminal_dwa: TerminalDWA,

    /// Terminal-side stage snapshots (raw → collapse → prune).
    pub terminal_debug: TerminalDebug,

    /// Template bundles grouping equivalent characterizations.
    pub template_bundles: Vec<TemplateBundle>,

    /// Composed parser NWA before resolve_negatives.
    pub parser_nwa_before_resolve: NWA,

    /// Composed parser NWA after resolve_negatives.
    pub parser_nwa_after_resolve: NWA,

    /// Parser DWA after determinization (before minimization).
    pub parser_dwa_pre_minimize: DWA,

    /// Final parser DWA (after minimization).
    pub parser_dwa: DWA,

    /// Compiler-side internal ID mappings.
    pub id_map: InternalIdMap,
}

// ---------------------------------------------------------------------------
// Full compilation debug bundle
// ---------------------------------------------------------------------------

/// Debug bundle capturing intermediate compilation artifacts.
///
/// Every field is public so tests and analysis tools can inspect freely.
///
/// # Interpretation metadata
///
/// The bundle carries enough context to interpret every automaton label
/// without recomputing hidden mappings:
///
/// - **`grammar_def`**: the original (user-facing) grammar, with terminal
///   names, patterns, and rules. Terminal IDs in `characterizations`,
///   `template_bundles`, and NWA weights map to `grammar_def.terminals[id]`.
/// - **`normalized_grammar_def`**: the grammar after `normalize_for_mask()`
///   (epsilon elimination, right-recursion rewrite). The GLR table and all
///   parser-side automata are built from this version. Compare with
///   `grammar_def` to see which rules were rewritten.
/// - **`glr_grammar`**: augmented GLR grammar built from the normalized def,
///   carrying FIRST/FOLLOW/nullable analysis.
/// - **`glr_table`**: the SLR(1) parse table. DWA labels are parser state
///   indices from this table. Use `table.actions(state, terminal)` and
///   `table.goto(state, nt)` to understand why a given state exists.
/// - **`id_map`**: compiler-side internal ID mappings. NWA weights encode
///   (tsid, token_range) pairs; use the tokenizer-state mapping to convert
///   between tokenizer DFA states and TSIDs, and the vocab-token mapping to
///   see which original token IDs collapse into a shared internal class.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompileDebug {
    // --- Interpretation metadata ---

    /// Original grammar definition (before normalization), with terminal
    /// names and regex patterns. `grammar_def.terminals[tid].name` gives
    /// the human-readable name for a terminal ID appearing anywhere in
    /// the bundle.
    pub grammar_def: GrammarDef,

    /// Normalized grammar definition (after `normalize_for_mask()`).
    /// This is the grammar the GLR table was actually built from.
    /// Rules may differ from `grammar_def` due to epsilon elimination
    /// and right-recursion rewriting.
    pub normalized_grammar_def: GrammarDef,

    /// Augmented GLR grammar (from normalized def). Carries FIRST/FOLLOW
    /// sets and nullable analysis for every nonterminal.
    pub glr_grammar: AnalyzedGrammar,

    /// SLR(1) parse table. Parser DWA labels are state indices in this
    /// table. Inspect with `table.actions(state, terminal)` and
    /// `table.goto(state, nt)`.
    pub glr_table: GLRTable,

    // --- Terminal side ---

    /// Terminal characterizations: terminal → parser characterization.
    pub characterizations: BTreeMap<TerminalID, TerminalCharacterization>,

    /// Terminal DWA (final, after collapse + prune).
    pub terminal_dwa: TerminalDWA,

    /// Terminal-side stage snapshots (raw → collapse → prune).
    pub terminal_debug: TerminalDebug,

    // --- Parser side ---

    /// Template bundles grouping equivalent characterizations.
    pub template_bundles: Vec<TemplateBundle>,

    /// Composed parser NWA before resolve_negatives.
    pub parser_nwa_before_resolve: NWA,

    /// Composed parser NWA after resolve_negatives.
    pub parser_nwa_after_resolve: NWA,

    /// Parser DWA after determinization (before minimization).
    pub parser_dwa_pre_minimize: DWA,

    /// Final parser DWA (after minimization).
    pub parser_dwa: DWA,

    // --- Vocab ---

    /// Compiler-side internal ID mappings.
    pub id_map: InternalIdMap,

    /// Raw vocabulary: (token_id, byte_sequence) pairs.
    /// Use this to map token IDs in weights back to their string form.
    pub vocab_entries: Vec<(u32, Vec<u8>)>,

    /// End-of-sequence token ID, if any.
    pub eos_token_id: Option<u32>,
}

impl CompileDebug {
    /// Assemble a full `CompileDebug` from grammar metadata and automata debug.
    pub fn from_parts(
        grammar_def: GrammarDef,
        normalized_grammar_def: GrammarDef,
        glr_grammar: AnalyzedGrammar,
        glr_table: GLRTable,
        automata: AutomataDebug,
        vocab_entries: Vec<(u32, Vec<u8>)>,
        eos_token_id: Option<u32>,
    ) -> Self {
        unimplemented!()
    }
}
