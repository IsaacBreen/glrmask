//! Combined Equivalence Analysis
//!
//! This module orchestrates both state equivalence analysis and vocab equivalence
//! analysis in an efficient manner:
//!
//! 1. First, applies state equivalence analysis to reduce the number of unique
//!    tokenizer states that need to be considered.
//!
//! 2. Then, performs vocab equivalence analysis on the reduced state set.
//!
//! This combined approach significantly improves performance for grammars with
//! large DFAs by reducing the workload of the expensive vocab analysis.

use std::collections::{BTreeMap, BTreeSet};

use crate::finite_automata::Regex;
use crate::tokenizer::TokenizerStateID;

use super::state_equivalence_analysis;
use super::vocab_equivalence_analysis;
pub use super::vocab_equivalence_analysis::VocabEquivalenceResult;

/// Result of combined equivalence analysis.
pub struct CombinedEquivalenceResult {
    /// Vocab equivalence classes: sets of token indices that behave identically.
    pub vocab_classes: VocabEquivalenceResult,
    
    /// Mapping from original state ID to representative state ID.
    /// States with the same representative are equivalent under the analyzed vocabulary.
    pub state_to_representative: BTreeMap<TokenizerStateID, TokenizerStateID>,
    
    /// The set of representative states.
    pub representative_states: Vec<usize>,
}

/// Compute combined state and vocab equivalence analysis.
///
/// This function:
/// 1. Computes state equivalence classes to find representative states
/// 2. Runs vocab equivalence analysis only on representative states
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to analyze
/// * `initial_states` - Initial tokenizer state IDs to consider
/// * `state_reduction_threshold` - Minimum number of states before applying state reduction
///
/// # Returns
/// Combined result containing vocab classes, state-to-rep mapping, and representative states.
pub fn compute_combined_equivalence(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
    state_reduction_threshold: usize,
) -> CombinedEquivalenceResult {
    let start = std::time::Instant::now();
    
    // Step 1: State equivalence analysis (if beneficial)
    let (reduced_states, state_to_representative) = if initial_states.len() > state_reduction_threshold {
        let state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            regex,
            tokens,
            initial_states,
        );
        
        // Build state-to-rep mapping
        let mut state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        let mut rep_set: BTreeSet<usize> = BTreeSet::new();
        
        for (i, &rep) in state_reps.iter().enumerate() {
            let state_id = initial_states[i];
            state_to_rep.insert(TokenizerStateID(state_id), TokenizerStateID(rep));
            rep_set.insert(rep);
        }
        
        let reduced: Vec<usize> = rep_set.into_iter().collect();
        
        crate::debug!(
            3,
            "Combined equiv: state reduction {} -> {} states in {:?}",
            initial_states.len(),
            reduced.len(),
            start.elapsed(),
        );
        
        (reduced, state_to_rep)
    } else {
        // No reduction needed - use all states as their own representatives
        let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = initial_states
            .iter()
            .map(|&s| (TokenizerStateID(s), TokenizerStateID(s)))
            .collect();
        (initial_states.to_vec(), state_to_rep)
    };
    
    // Step 2: Vocab equivalence analysis on reduced states
    let vocab_start = std::time::Instant::now();
    let vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes(
        regex,
        tokens,
        &reduced_states,
    );
    
    crate::debug!(
        3,
        "Combined equiv: vocab analysis {} tokens -> {} classes in {:?}",
        tokens.len(),
        vocab_classes.len(),
        vocab_start.elapsed(),
    );
    
    crate::debug!(
        2,
        "Combined equivalence analysis complete: {} vocab classes, {} representative states (total {:?})",
        vocab_classes.len(),
        reduced_states.len(),
        start.elapsed(),
    );
    
    CombinedEquivalenceResult {
        vocab_classes,
        state_to_representative,
        representative_states: reduced_states,
    }
}

/// Simplified entry point that just returns vocab equivalence classes.
///
/// Use this when you don't need the state mapping information.
pub fn find_vocab_equivalence_classes_with_state_reduction(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
    state_reduction_threshold: usize,
) -> VocabEquivalenceResult {
    compute_combined_equivalence(regex, tokens, initial_states, state_reduction_threshold).vocab_classes
}
