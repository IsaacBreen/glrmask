//! Simple equivalence analysis based on terminal behavior - v2.
//!
//! For each (token, initial_state) pair, we compute:
//! 1. The set of matches (terminal completions) at each position
//! 2. The final tokenizer state after consuming all bytes (if not dead)
//!
//! Two tokens are equivalent if they have the same (matches, final_state) for ALL initial states.
//!
//! Key insight: The DWA partition groups tokens by which transitions they contribute to.
//! A token contributes to:
//! - Transitions for each match (terminal completion)
//! - Transitions for each accessible terminal from the final state
//!
//! So the signature should capture: which terminals get matches at which positions,
//! and what the final tokenizer state is (which determines accessible terminals).

use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Result of simple equivalence analysis
pub struct SimpleEquivalenceResult {
    /// Equivalence classes: signature -> list of string indices
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Same as mask_classes for commit equivalence
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

/// Match info: (group_id, position)
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Match {
    group_id: usize,
    position: usize,
}

/// Outcome for one initial state
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct StateOutcome {
    /// Matches that occurred (sorted)
    matches: Vec<Match>,
    /// Final tokenizer state (if not dead end) - determines accessible terminals
    final_state: Option<usize>,
}

/// Compute outcome for a single initial state
fn compute_outcome_for_state(
    regex: &Regex,
    token: &[u8],
    initial_state: usize,
) -> StateOutcome {
    if token.is_empty() {
        return StateOutcome {
            matches: vec![],
            final_state: Some(initial_state),
        };
    }
    
    let mut current_state = initial_state;
    let mut matches: Vec<Match> = vec![];
    
    for (i, &byte) in token.iter().enumerate() {
        let sd = &regex.dfa.states[current_state];
        if let Some(&next_state) = sd.transitions.get(byte) {
            current_state = next_state;
            let next_data = &regex.dfa.states[current_state];
            for gid in next_data.finalizers.iter_indices() {
                matches.push(Match {
                    group_id: gid,
                    position: i + 1,
                });
            }
        } else {
            // Dead end
            return StateOutcome {
                matches,
                final_state: None,
            };
        }
    }
    
    // Sort matches for consistent ordering
    matches.sort();
    
    StateOutcome {
        matches,
        final_state: Some(current_state),
    }
}

/// Signature for a single token across all initial states
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct SimpleSignature {
    outcomes: Vec<(usize, StateOutcome)>,
}

/// Compute signature for a single token across all initial states
fn compute_simple_signature(
    regex: &Regex,
    token: &[u8],
    initial_states: &[usize],
) -> SimpleSignature {
    let mut outcomes = Vec::with_capacity(initial_states.len());
    
    for (state_idx, &init_state) in initial_states.iter().enumerate() {
        let outcome = compute_outcome_for_state(regex, token, init_state);
        // Include all outcomes (even empty ones are significant)
        outcomes.push((state_idx, outcome));
    }
    
    SimpleSignature { outcomes }
}

/// Main entry point
pub fn find_equivalence_classes_simple(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    crate::debug!(3, "Simple equivalence v2 analysis for {} strings, {} initial states",
                 strings.len(), initial_states.len());
    
    let t0 = std::time::Instant::now();
    
    // Compute signatures in parallel
    let signatures: Vec<SimpleSignature> = strings
        .par_iter()
        .map(|s| compute_simple_signature(regex, s, initial_states))
        .collect();
    
    crate::debug!(4, "Simple equiv v2: signatures computed in {:?}", t0.elapsed());
    
    // Group by signature
    let mut groups: HashMap<SimpleSignature, Vec<usize>> = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_default().push(idx);
    }
    
    // Convert to output format
    let mask_classes: BTreeMap<Vec<usize>, Vec<usize>> = groups
        .into_iter()
        .enumerate()
        .map(|(id, (_, indices))| (vec![id], indices))
        .collect();
    
    crate::debug!(3, "Simple equivalence v2: {} classes in {:?}", mask_classes.len(), t0.elapsed());
    
    SimpleEquivalenceResult {
        mask_classes: mask_classes.clone(),
        commit_classes: mask_classes,
    }
}
