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

use std::collections::BTreeSet;

use crate::finite_automata::Regex;

use super::state_equivalence_analysis_fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab_equivalence_analysis_fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};

/// Result of combined equivalence analysis.
pub struct CombinedEquivalenceResult {
    /// Vocab equivalence classes: sets of token indices that behave identically.
    pub vocab_classes: VocabEquivalenceResult,
    
    /// State equivalence classes: sets of state IDs that behave identically.
    pub state_classes: StateEquivalenceResult,
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
/// Combined result containing vocab classes and state classes.
pub fn compute_combined_equivalence(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
) -> CombinedEquivalenceResult {
    // Threshold for applying state equivalence analysis
    let state_reduction_threshold = 50;

    let start = std::time::Instant::now();
    
    // Check which state equivalence algorithm to use
    let use_trie = std::env::var("USE_TRIE_STATE_EQUIV").map(|v| v == "1").unwrap_or(false);
    let use_discriminating = std::env::var("USE_DISC_STATE_EQUIV").map(|v| v == "1").unwrap_or(false);
    let use_transposed = std::env::var("USE_TRANSPOSED_STATE_EQUIV").map(|v| v == "1").unwrap_or(false);
    let use_optimized = std::env::var("USE_OPTIMIZED_STATE_EQUIV").map(|v| v == "1").unwrap_or(false);
    let skip_state_equiv = std::env::var("SKIP_STATE_EQUIV").map(|v| v == "1").unwrap_or(false);
    
    // Step 1: State equivalence analysis (if beneficial)
    let (reduced_states, state_classes) = if skip_state_equiv {
        // Skip state equivalence - use all states directly
        crate::debug!(3, "Skipping state equivalence (SKIP_STATE_EQUIV=1)");
        let state_classes: StateEquivalenceResult = initial_states
            .iter()
            .map(|&s| std::iter::once(s).collect())
            .collect();
        (initial_states.to_vec(), state_classes)
    } else if initial_states.len() > state_reduction_threshold {
        let state_reps = if use_trie {
            crate::debug!(3, "Using trie-based state equivalence analysis");
            super::state_equivalence_trie::find_state_equivalence_classes_trie(
                regex,
                tokens,
                initial_states,
            )
        } else if use_discriminating {
            crate::debug!(3, "Using discriminating token state equivalence analysis");
            super::state_equivalence_discriminating::find_state_equivalence_classes_discriminating(
                regex,
                tokens,
                initial_states,
            )
        } else if use_transposed {
            crate::debug!(3, "Using transposed DFA state equivalence analysis");
            super::state_equivalence_transposed::find_state_equivalence_classes_transposed(
                regex,
                tokens,
                initial_states,
            )
        } else if use_optimized {
            crate::debug!(3, "Using optimized state equivalence analysis");
            super::state_equivalence_optimized::find_state_equivalence_classes_optimized(
                regex,
                tokens,
                initial_states,
            )
        } else if std::env::var("USE_U16_STATE_EQUIV").map(|v| v == "1").unwrap_or(false) {
            crate::debug!(3, "Using u16 state equivalence analysis");
            super::state_equivalence_u16::find_state_equivalence_classes_u16(
                regex,
                tokens,
                initial_states,
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes(
                regex,
                tokens,
                initial_states,
            )
        };
        
        // Build reduced state set
        let mut rep_set: BTreeSet<usize> = BTreeSet::new();
        for &rep in &state_reps {
            rep_set.insert(rep);
        }
        
        let reduced: Vec<usize> = rep_set.into_iter().collect();
        
        // Convert to StateEquivalenceResult format
        let state_classes = state_equivalence_analysis::mapping_to_equivalence_classes(initial_states, &state_reps);
        
        crate::debug!(
            3,
            "Combined equiv: state reduction {} -> {} states in {:?}",
            initial_states.len(),
            reduced.len(),
            start.elapsed(),
        );
        
        (reduced, state_classes)
    } else {
        // No reduction needed - use all states as their own representatives
        // Each state is its own equivalence class
        let state_classes: StateEquivalenceResult = initial_states
            .iter()
            .map(|&s| std::iter::once(s).collect())
            .collect();
        
        (initial_states.to_vec(), state_classes)
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
        state_classes,
    }
}

/// Simplified entry point that just returns vocab equivalence classes.
///
/// Use this when you don't need the state mapping information.
pub fn find_vocab_equivalence_classes_with_state_reduction(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    compute_combined_equivalence(regex, tokens, initial_states).vocab_classes
}
