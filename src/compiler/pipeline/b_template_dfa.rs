//! Template bundle construction.
//!
//! `sep1` does not build the parser automaton terminal-by-terminal. It first
//! groups equivalent terminal characterizations into reusable template bundles,
//! then composes those bundles into the final parser automaton. `glrmask`
//! still lacks the full tokenizer/template composition step, but it can and
//! should at least build the parser-side bundles explicitly instead of hiding
//! that structure inside `parser_dwa.rs`.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::automata::u32::unweighted::dfa::Dfa as UnweightedDfa;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::CompDwa;
use crate::automata::weighted::minimize::minimize_acyclic;
use crate::automata::weighted::nwa::{Nwa, NwaState};
use crate::automata::weighted::weight::Weight;
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::labels::{DEFAULT_LABEL, encode_negative_label, encode_positive_label};
use crate::compiler::parser_dwa::TerminalCharacterization;
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::terminal_dwa::TerminalDwa;

#[derive(Debug)]
struct TemplateFragment {
    start_states: Vec<u32>,
    final_states: Vec<(u32, Weight)>,
}

struct TemplateCompositionContext<'a> {
    bundles: &'a [TemplateBundle],
    terminal_dwa: &'a TerminalDwa,
    combined: Nwa,
    template_by_terminal: BTreeMap<TerminalId, usize>,
    blueprint_cache: HashMap<(usize, Weight), Nwa>,
    total_fragment_uses: usize,
    total_fragment_states: usize,
}

impl<'a> TemplateCompositionContext<'a> {
    fn new(
        bundles: &'a [TemplateBundle],
        terminal_dwa: &'a TerminalDwa,
        num_tsids: u32,
        max_token: u32,
    ) -> Self {
        let mut template_by_terminal = BTreeMap::new();
        for (bundle_idx, bundle) in bundles.iter().enumerate() {
            for &terminal in &bundle.terminals {
                template_by_terminal.insert(terminal, bundle_idx);
            }
        }

        Self {
            bundles,
            terminal_dwa,
            combined: Nwa::new(num_tsids, max_token),
            template_by_terminal,
            blueprint_cache: HashMap::new(),
            total_fragment_uses: 0,
            total_fragment_states: 0,
        }
    }

    fn fresh_fragment(
        &mut self,
        bundle_idx: usize,
        transition_weight: &Weight,
        num_tsids: u32,
        max_token: u32,
    ) -> TemplateFragment {
        // Cache the *blueprint* NWA per (bundle, weight) to avoid repeated
        // instantiate_template_dfa calls, but append a fresh copy to `combined`
        // for each use so that each transition gets independent NWA state IDs.
        //
        // This prevents cycles: sharing actual NWA state IDs between transitions
        // A→B and B→C creates cycle body_B → F.start →…→ F.final → body_B.
        // Fresh copies eliminate that back-edge opportunity while keeping build
        // cost proportional to (unique templates × uses) rather than (total transitions).
        let key = (bundle_idx, transition_weight.clone());
        if !self.blueprint_cache.contains_key(&key) {
            let blueprint = instantiate_template_dfa(
                &self.bundles[bundle_idx].template_dfa,
                transition_weight,
                num_tsids,
                max_token,
            );
            self.blueprint_cache.insert(key.clone(), blueprint);
        }
        let blueprint = &self.blueprint_cache[&key];
        self.total_fragment_uses += 1;
        self.total_fragment_states += blueprint.states.len();
        let offset = append_nwa(&mut self.combined, blueprint);

        // Diagnostic: verify append worked
        if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
            eprintln!("  [fragment] bundle={}, offset={}, blueprint_states={}", bundle_idx, offset, blueprint.states.len());
            for (i, state) in blueprint.states.iter().enumerate() {
                let trans_count: usize = state.transitions.values().map(|v| v.len()).sum();
                let has_final = state.final_weight.is_some();
                eprintln!("    bp_state {}: {} trans, {} eps, final={}", i, trans_count, state.epsilons.len(), has_final);
            }
            for i in 0..blueprint.states.len() {
                let combined_idx = offset as usize + i;
                let cs = &self.combined.states[combined_idx];
                let ct: usize = cs.transitions.values().map(|v| v.len()).sum();
                let has_final = cs.final_weight.is_some();
                eprintln!("    combined_state {}: {} trans, {} eps, final={}", combined_idx, ct, cs.epsilons.len(), has_final);
            }
        }

        let start_states = blueprint
            .start_states
            .iter()
            .map(|state| offset + *state)
            .collect();
        let mut final_states = Vec::new();
        for (fragment_sid, fragment_state) in blueprint.states.iter().enumerate() {
            let Some(final_weight) = &fragment_state.final_weight else {
                continue;
            };
            let combined_state = offset + fragment_sid as u32;
            self.combined.states[combined_state as usize].final_weight = None;
            final_states.push((combined_state, final_weight.clone()));
        }
        TemplateFragment {
            start_states,
            final_states,
        }
    }

    /// Compose a terminal NWA state into the combined parser NWA.
    ///
    /// Returns the **continuation body** — a non-final state with template
    /// fragment entries for processing the next stack element.  A separate
    /// **final node** (if the terminal NWA state is accepting) receives the
    /// token-specific weight from fragment exits.  This separation ensures
    /// that the token constraint from one template step does not bleed into
    /// subsequent steps through the determinizer's epsilon-closure
    /// intersection.
    fn compose_terminal_state(
        &mut self,
        state_id: u32,
        body_cache: &mut BTreeMap<u32, (u32, Option<u32>)>,
        num_tsids: u32,
        max_token: u32,
    ) -> u32 {
        unimplemented!("cargo-check-only stub")
    }
}

/// A compiled structural template DFA, independent of lexical weights.
#[derive(Debug, Clone)]
pub struct TemplateDfa {
    pub dfa: UnweightedDfa,
}

/// A parser-side template bundle shared by one or more terminals.
#[derive(Debug, Clone)]
pub struct TemplateBundle {
    /// Structural template DFA compiled from the shared characterization.
    pub template_dfa: TemplateDfa,
    /// All terminals that reuse the same parser-side template.
    pub terminals: Vec<TerminalId>,
}

/// Group terminals that share the same parser-side characterization.
///
/// This is the structural part of the `sep1` template-bundle design: parser
/// patterns are deduplicated before they are turned into automaton fragments.
/// The current `glrmask` rewrite still uses direct lexical weights instead of
/// true template/tokenizer composition, but bundling equivalent parser
/// templates keeps the architecture aligned with the intended direction.
pub(crate) fn build_template_bundles(
    characterizations: &BTreeMap<TerminalId, TerminalCharacterization>,
    used_terminals: &BTreeSet<TerminalId>,
) -> Vec<TemplateBundle> {
    unimplemented!("cargo-check-only stub")
}

fn ensure_nt_stack_state(
    nwa: &mut Nwa,
    nt_stacks: &mut BTreeMap<u32, Vec<u32>>,
    nt: u32,
    depth: usize,
    w_all: &Weight,
) -> u32 {
    unimplemented!("cargo-check-only stub")
}

fn build_template_structure_nwa(characterization: &TerminalCharacterization) -> Nwa {
    unimplemented!("cargo-check-only stub")
}

fn is_acyclic(dwa: &CompDwa) -> bool {
    unimplemented!("cargo-check-only stub")
}

fn build_template_dfa(characterization: &TerminalCharacterization) -> TemplateDfa {
    unimplemented!("cargo-check-only stub")
}

fn instantiate_template_dfa(
    template_dfa: &TemplateDfa,
    terminal_weight: &Weight,
    num_tsids: u32,
    max_token: u32,
) -> Nwa {
    unimplemented!("cargo-check-only stub")
}

fn append_nwa(target: &mut Nwa, fragment: &Nwa) -> u32 {
    unimplemented!("cargo-check-only stub")
}

/// Build the parser-side template NWA by unioning the per-bundle template NWAs.
pub(crate) fn build_template_nwa_from_bundles(
    bundles: &[TemplateBundle],
    terminal_dwa: &TerminalDwa,
    num_tsids: u32,
    max_token: u32,
) -> Nwa {
    unimplemented!("cargo-check-only stub")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn test_build_template_bundles_groups_equivalent_terminals() {
        let shared = TerminalCharacterization {
            shifts: vec![(0, 1)],
            reduces: vec![(2, 1, 7)],
            nt_escapes: vec![(7, 0, 3, 4)],
            nt_rereduces: vec![(7, 1, 2, 9)],
            all_nts: BTreeSet::from([7, 9]),
        };

        let characterizations = BTreeMap::from([(0, shared.clone()), (1, shared)]);
        let used_terminals = BTreeSet::from([0, 1]);
        let bundles = build_template_bundles(&characterizations, &used_terminals);

        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].terminals, vec![0, 1]);
        assert!(bundles[0].template_dfa.dfa.num_states() > 0);
    }
}
