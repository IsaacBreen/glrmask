//! Terminal automata accepted by the parser-DWA construction.
//!
//! A conventional terminal DWA has one target state per grammar label.  When
//! vocabulary source domains are disjoint, an NWA with several targets for a
//! label is still deterministic over `(label, token)`: each token belongs to
//! at most one branch weight.  Keeping that form avoids a product construction
//! whose only purpose is to encode the token-conditioned target selection.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::dwa::{DWA, DwaStats};
use super::nwa::NWA;
use crate::ds::weight::Weight;

#[derive(Debug, Clone)]
pub enum TerminalAutomaton {
    Dwa(DWA),
    /// An epsilon-free NWA that is deterministic over `(label, token)`.
    TokenDeterministicNwa(NWA),
}

impl TerminalAutomaton {
    pub fn num_states(&self) -> usize {
        match self {
            Self::Dwa(dwa) => dwa.states().len(),
            Self::TokenDeterministicNwa(nwa) => nwa.states().len(),
        }
    }

    pub fn start_states(&self) -> Vec<u32> {
        match self {
            Self::Dwa(dwa) => vec![dwa.start_state()],
            Self::TokenDeterministicNwa(nwa) => nwa.start_states().to_vec(),
        }
    }

    pub fn stats(&self) -> DwaStats {
        match self {
            Self::Dwa(dwa) => dwa.stats(),
            Self::TokenDeterministicNwa(nwa) => {
                let mut transition_pairs = 0usize;
                let mut seen_weight_ptrs = BTreeSet::new();
                let mut seen_rangeset_ptrs = BTreeSet::new();
                let mut total_outer_ranges = 0usize;
                let mut total_inner_ranges = 0usize;
                let mut process_weight = |weight: &Weight| {
                    let weight_ptr = Arc::as_ptr(&weight.0) as usize;
                    if seen_weight_ptrs.insert(weight_ptr) {
                        total_outer_ranges += weight.0.range_values().count();
                    }
                    for (_, tokens) in weight.0.range_values() {
                        let token_ptr = Arc::as_ptr(tokens) as usize;
                        if seen_rangeset_ptrs.insert(token_ptr) {
                            total_inner_ranges += tokens.ranges().count();
                        }
                    }
                };

                for state in nwa.states() {
                    let mut targets = BTreeSet::new();
                    if let Some(weight) = &state.final_weight {
                        process_weight(weight);
                    }
                    for branches in state.transitions.values() {
                        for (target, weight) in branches {
                            targets.insert(*target);
                            process_weight(weight);
                        }
                    }
                    for (target, weight) in &state.epsilons {
                        targets.insert(*target);
                        process_weight(weight);
                    }
                    transition_pairs += targets.len();
                }

                DwaStats {
                    states: nwa.states().len(),
                    transitions: nwa.num_transitions(),
                    transition_pairs,
                    interned_ranges: total_outer_ranges + total_inner_ranges,
                }
            }
        }
    }
}
