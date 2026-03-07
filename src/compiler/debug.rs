//! Typed debug bundle for compilation artifacts.
//!
//! Captures intermediate automata from each stage of the compilation pipeline
//! without relying on env-var printing. Returned alongside the Constraint by
//! [`compile_with_debug`].
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use std::collections::BTreeMap;

use crate::automata::weighted::dwa::CompDwa;
use crate::automata::weighted::nwa::Nwa;
use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::glr::table::GlrTable;
use crate::compiler::grammar_def::{GrammarDef, TerminalId};
use crate::compiler::parser_dwa::TerminalCharacterization;
use crate::compiler::template::TemplateBundle;
use crate::compiler::terminal_dwa::TerminalDwa;
use crate::compiler::vocab_pre::VocabPreprocessing;

// ---------------------------------------------------------------------------
// Terminal-side debug stages
// ---------------------------------------------------------------------------

/// Snapshots of the terminal NWA at each stage of `build_terminal_dwa`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerminalDebug {
    /// Terminal NWA immediately after `build_terminal_dwa_nwa` (raw vocab walk),
    /// before any follow-path optimisations.
    pub nwa_after_build: Nwa,

    /// Terminal NWA after `collapse_always_allowed` but before
    /// `prune_disallowed_follows`.
    pub nwa_after_collapse: Nwa,

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
    pub characterizations: BTreeMap<TerminalId, TerminalCharacterization>,

    /// Terminal DWA (final, after collapse + prune).
    pub terminal_dwa: TerminalDwa,

    /// Terminal-side stage snapshots (raw → collapse → prune).
    pub terminal_debug: TerminalDebug,

    /// Template bundles grouping equivalent characterizations.
    pub template_bundles: Vec<TemplateBundle>,

    /// Composed parser NWA before resolve_negatives.
    pub parser_nwa_before_resolve: Nwa,

    /// Composed parser NWA after resolve_negatives.
    pub parser_nwa_after_resolve: Nwa,

    /// Parser DWA after determinization (before minimization).
    pub parser_dwa_pre_minimize: CompDwa,

    /// Final parser DWA (after minimization).
    pub parser_dwa: CompDwa,

    /// Vocabulary preprocessing results (TSID mapping, possible_matches).
    pub vocab_pre: VocabPreprocessing,
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
/// - **`vocab_pre`**: vocabulary preprocessing with TSID mappings. NWA
///   weights encode (tsid, token_range) pairs; use `state_to_tsid` /
///   `tsid_to_state` to convert between tokenizer DFA states and TSIDs.
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
    pub glr_grammar: GlrGrammar,

    /// SLR(1) parse table. Parser DWA labels are state indices in this
    /// table. Inspect with `table.actions(state, terminal)` and
    /// `table.goto(state, nt)`.
    pub glr_table: GlrTable,

    // --- Terminal side ---

    /// Terminal characterizations: terminal → parser characterization.
    pub characterizations: BTreeMap<TerminalId, TerminalCharacterization>,

    /// Terminal DWA (final, after collapse + prune).
    pub terminal_dwa: TerminalDwa,

    /// Terminal-side stage snapshots (raw → collapse → prune).
    pub terminal_debug: TerminalDebug,

    // --- Parser side ---

    /// Template bundles grouping equivalent characterizations.
    pub template_bundles: Vec<TemplateBundle>,

    /// Composed parser NWA before resolve_negatives.
    pub parser_nwa_before_resolve: Nwa,

    /// Composed parser NWA after resolve_negatives.
    pub parser_nwa_after_resolve: Nwa,

    /// Parser DWA after determinization (before minimization).
    pub parser_dwa_pre_minimize: CompDwa,

    /// Final parser DWA (after minimization).
    pub parser_dwa: CompDwa,

    // --- Vocab ---

    /// Vocabulary preprocessing results (TSID mapping, possible_matches).
    pub vocab_pre: VocabPreprocessing,

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
        glr_grammar: GlrGrammar,
        glr_table: GlrTable,
        automata: AutomataDebug,
        vocab_entries: Vec<(u32, Vec<u8>)>,
        eos_token_id: Option<u32>,
    ) -> Self {
        unimplemented!("cargo-check-only stub")
    }

    // --- Display helpers (private) ---

    fn terminal_name(&self, id: TerminalId) -> &str {
        unimplemented!("cargo-check-only stub")
    }

    fn symbol_str(&self, sym: &crate::compiler::grammar_def::Symbol) -> String {
        unimplemented!("cargo-check-only stub")
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl std::fmt::Display for CompileDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::compiler::glr::grammar::EOF;

        // ===== 1. Grammar =====
        writeln!(f, "═══ GRAMMAR (original) ═══")?;
        writeln!(f, "Start: NT{}", self.grammar_def.start)?;
        writeln!(f, "Terminals:")?;
        for t in &self.grammar_def.terminals {
            writeln!(f, "  T{}: name={:?} pattern={:?}", t.id, t.name, t.pattern)?;
        }
        writeln!(f, "Rules:")?;
        for (i, r) in self.grammar_def.rules.iter().enumerate() {
            let rhs: Vec<String> = r.rhs.iter().map(|s| self.symbol_str(s)).collect();
            writeln!(f, "  [{}] NT{} → {}", i, r.lhs, rhs.join(" "))?;
        }

        // ===== 2. Normalized grammar =====
        writeln!(f, "\n═══ GRAMMAR (normalized for mask) ═══")?;
        writeln!(f, "Start: NT{}", self.normalized_grammar_def.start)?;
        if self.normalized_grammar_def.terminals.len() != self.grammar_def.terminals.len() {
            writeln!(f, "Terminals: {} (changed from {})",
                self.normalized_grammar_def.terminals.len(),
                self.grammar_def.terminals.len())?;
        }
        writeln!(f, "Rules:")?;
        for (i, r) in self.normalized_grammar_def.rules.iter().enumerate() {
            let rhs: Vec<String> = r.rhs.iter().map(|s| self.symbol_str(s)).collect();
            writeln!(f, "  [{}] NT{} → {}", i, r.lhs, rhs.join(" "))?;
        }

        // ===== 3. GLR Table =====
        writeln!(f, "\n═══ GLR PARSE TABLE ({} states, {} rules) ═══",
            self.glr_table.num_states, self.glr_table.num_rules)?;
        for state in 0..self.glr_table.num_states {
            let actions = &self.glr_table.action[state as usize];
            let gotos = &self.glr_table.goto[state as usize];
            if actions.is_empty() && gotos.is_empty() {
                continue;
            }
            writeln!(f, "  State {state}:")?;
            for (tid, acts) in actions {
                let tname = if *tid == EOF { "$".to_string() } else { self.terminal_name(*tid).to_string() };
                for a in acts {
                    let astr = match a {
                        crate::compiler::glr::table::Action::Shift(s) => format!("shift {s}"),
                        crate::compiler::glr::table::Action::Reduce(r) => {
                            let rule = &self.glr_table.rules[*r as usize];
                            format!("reduce r{r} (NT{} ← {} symbols)", rule.lhs, rule.rhs.len())
                        }
                        crate::compiler::glr::table::Action::Accept => "accept".to_string(),
                    };
                    writeln!(f, "    on '{tname}': {astr}")?;
                }
            }
            for (nt, tgt) in gotos {
                writeln!(f, "    goto NT{nt} → state {tgt}")?;
            }
        }

        // ===== 4. Vocab =====
        writeln!(f, "\n═══ VOCABULARY ({} tokens) ═══", self.vocab_entries.len())?;
        if let Some(eos) = self.eos_token_id {
            writeln!(f, "EOS token: {eos}")?;
        }
        for (id, bytes) in &self.vocab_entries {
            let repr = String::from_utf8_lossy(bytes);
            writeln!(f, "  tok{id}: {repr:?} ({} bytes)", bytes.len())?;
        }

        // ===== 5. TSID mapping =====
        writeln!(f, "\n═══ TSID MAPPING ({} TSIDs) ═══", self.vocab_pre.num_tsids)?;
        for (tsid, &dfa_state) in self.vocab_pre.tsid_to_state.iter().enumerate() {
            writeln!(f, "  TSID {tsid} ↔ DFA state {dfa_state}")?;
        }

        // ===== 6. Vocab preprocessing (possible_matches) =====
        writeln!(f, "\n═══ VOCAB PREPROCESSING ═══")?;
        for (tsid, matches) in self.vocab_pre.possible_matches.iter().enumerate() {
            if matches.is_empty() {
                continue;
            }
            writeln!(f, "  TSID {tsid}:")?;
            for (tid, tokens) in matches {
                writeln!(f, "    terminal '{}' (T{tid}): tokens {tokens}", self.terminal_name(*tid))?;
            }
            let pt = &self.vocab_pre.passthrough_tokens[tsid];
            if !pt.is_empty() {
                writeln!(f, "    passthrough tokens: {pt}")?;
            }
        }

        // ===== 7. Terminal characterizations =====
        writeln!(f, "\n═══ TERMINAL CHARACTERIZATIONS ═══")?;
        for (tid, tc) in &self.characterizations {
            writeln!(f, "  Terminal '{}' (T{tid}):", self.terminal_name(*tid))?;
            for (from, to) in &tc.shifts {
                writeln!(f, "    shift: state {from} → state {to}")?;
            }
            for (from, pop, nt) in &tc.reduces {
                writeln!(f, "    reduce: state {from}, pop {pop}, → NT{nt}")?;
            }
            for (nt, revealed, goto, shift) in &tc.nt_escapes {
                writeln!(f, "    nt_escape: NT{nt}, revealed={revealed}, goto={goto}, shift={shift}")?;
            }
            for (nt, revealed, pop, re_nt) in &tc.nt_rereduces {
                writeln!(f, "    nt_rereduce: NT{nt}, revealed={revealed}, pop={pop}, → NT{re_nt}")?;
            }
        }

        // ===== 8. Template bundles =====
        writeln!(f, "\n═══ TEMPLATE BUNDLES ({} bundles) ═══", self.template_bundles.len())?;
        for (i, b) in self.template_bundles.iter().enumerate() {
            let tnames: Vec<String> = b.terminals.iter().map(|t| format!("'{}'(T{t})", self.terminal_name(*t))).collect();
            writeln!(f, "  Bundle {i}: terminals=[{}]", tnames.join(", "))?;
            writeln!(f, "    Template DFA:")?;
            // Template DFA labels are parser state IDs — no symbol map needed
            write!(f, "{}", b.template_dfa.dfa)?;
        }

        // Build terminal symbol map for terminal-side automata.
        // Terminal NWA labels are terminal IDs.
        let terminal_symbols: std::collections::BTreeMap<i32, String> = self
            .grammar_def
            .terminals
            .iter()
            .map(|t| (t.id as i32, format!("'{}'", t.name)))
            .collect();

        // Build TSID and token name maps for symbolic weight display.
        let tsid_names: std::collections::BTreeMap<u32, String> = self
            .vocab_pre
            .tsid_to_state
            .iter()
            .enumerate()
            .map(|(tsid, &dfa_st)| (tsid as u32, format!("tsid{tsid}/dfa{dfa_st}")))
            .collect();
        let token_names: std::collections::BTreeMap<u32, String> = self
            .vocab_entries
            .iter()
            .map(|(id, bytes)| {
                let repr = String::from_utf8_lossy(bytes);
                (*id, format!("{repr:?}"))
            })
            .collect();

        // ===== 9. Terminal DWA stages =====
        writeln!(f, "\n═══ TERMINAL NWA — after build (raw) ═══")?;
        write!(f, "{}", self.terminal_debug.nwa_after_build.display_with_all_maps(&terminal_symbols, &tsid_names, &token_names))?;

        writeln!(f, "\n═══ TERMINAL NWA — after collapse_always_allowed ═══")?;
        write!(f, "{}", self.terminal_debug.nwa_after_collapse.display_with_all_maps(&terminal_symbols, &tsid_names, &token_names))?;

        writeln!(f, "\n═══ TERMINAL NWA — final (in terminal_dwa) ═══")?;
        write!(f, "{}", self.terminal_dwa.nwa.display_with_all_maps(&terminal_symbols, &tsid_names, &token_names))?;
        writeln!(f, "TSID roots: {:?}", self.terminal_dwa.tsid_roots)?;

        // ===== 10. Parser NWA stages =====
        // Parser NWA labels are GLR parser state IDs — keep as raw integers.
        writeln!(f, "\n═══ PARSER NWA — before resolve_negatives ═══")?;
        write!(f, "{}", self.parser_nwa_before_resolve)?;

        writeln!(f, "\n═══ PARSER NWA — after resolve_negatives ═══")?;
        write!(f, "{}", self.parser_nwa_after_resolve)?;

        // ===== 11. Parser DWA stages =====
        writeln!(f, "\n═══ PARSER DWA — pre-minimize ═══")?;
        write!(f, "{}", self.parser_dwa_pre_minimize)?;

        writeln!(f, "\n═══ PARSER DWA — final (post-minimize) ═══")?;
        write!(f, "{}", self.parser_dwa)?;

        Ok(())
    }
}
