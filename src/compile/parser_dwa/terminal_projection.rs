//! Projection from Terminal-DWA graph structure to Parser-DWA continuation data.
//!
//! The Terminal DWA is a weighted automaton over terminal strings.  Parser-DWA
//! construction needs to ask a different question: from this Terminal-DWA
//! state, which groups of terminals all flow to the same continuation state,
//! and can those groups actually be accepted by parser stack-effect templates?

use std::collections::{BTreeMap, VecDeque};

use rustc_hash::FxHashMap;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::compile::template_dfa::Templates;
use crate::sets::weight::Weight;
use crate::grammar::flat::TerminalID;

use super::types::{Branch, BundleSignature, StateSummaries, StateSummary, TerminalBundle};

fn group_terminal_edges_by_target(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, TerminalBundle> {
    let Some(state) = terminal_dwa.states().get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut bundles_by_target = BTreeMap::<u32, TerminalBundle>::new();
    for (&label, (target, weight)) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        bundles_by_target
            .entry(*target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }

    bundles_by_target
}

fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {
    bundle
        .iter()
        .map(|(&terminal, weight)| (terminal, weight.clone()))
        .collect()
}

fn terminal_template_has_acceptance(template: &NWA) -> bool {
    template.states().iter().any(|state| state.final_weight.is_some())
}

fn terminal_bundle_has_acceptance(bundle: &TerminalBundle, templates: &Templates) -> bool {
    bundle.iter().any(|(&terminal, weight)| {
        !weight.is_empty()
            && templates
                .by_terminal_nwa
                .get(&terminal)
                .is_some_and(terminal_template_has_acceptance)
    })
}

pub(crate) fn build_state_summaries(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> StateSummaries {
    let mut branches_by_state: Vec<Vec<Branch>> = Vec::with_capacity(terminal_dwa.states().len());
    let mut bundle_ids_by_signature: FxHashMap<BundleSignature, usize> = FxHashMap::default();
    let mut unique_bundles: Vec<TerminalBundle> = Vec::new();

    for (state_id, _state) in terminal_dwa.states().iter().enumerate() {
        let bundles_by_target = group_terminal_edges_by_target(terminal_dwa, grammar, state_id as u32);
        let mut branches = Vec::with_capacity(bundles_by_target.len());
        for (target, bundle) in bundles_by_target {
            let signature = bundle_signature(&bundle);
            let bundle_id = if let Some(&bundle_id) = bundle_ids_by_signature.get(&signature) {
                bundle_id
            } else {
                let bundle_id = unique_bundles.len();
                bundle_ids_by_signature.insert(signature, bundle_id);
                unique_bundles.push(bundle);
                bundle_id
            };
            branches.push(Branch { target, bundle_id });
        }
        branches_by_state.push(branches);
    }

    let bundle_accepts: Vec<bool> = unique_bundles
        .iter()
        .map(|bundle| terminal_bundle_has_acceptance(bundle, templates))
        .collect();

    let states = terminal_dwa
        .states()
        .iter()
        .enumerate()
        .map(|(state_id, state)| StateSummary {
            final_weight: state.final_weight.clone(),
            branches: std::mem::take(&mut branches_by_state[state_id]),
        })
        .collect();

    StateSummaries {
        states,
        unique_bundles,
        bundle_accepts,
    }
}

pub(crate) fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {
    let states = &summaries.states;
    let mut reverse_edges: Vec<Vec<u32>> = vec![Vec::new(); states.len()];
    let mut productive = vec![false; states.len()];
    let mut worklist = VecDeque::new();

    for (state_id, state) in states.iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            productive[state_id] = true;
            worklist.push_back(state_id as u32);
        }

        for branch in &state.branches {
            if (branch.target as usize) < states.len()
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                reverse_edges[branch.target as usize].push(state_id as u32);
            }
        }
    }

    while let Some(target) = worklist.pop_front() {
        for &source in &reverse_edges[target as usize] {
            let source_idx = source as usize;
            if !productive[source_idx] {
                productive[source_idx] = true;
                worklist.push_back(source);
            }
        }
    }

    productive
}
