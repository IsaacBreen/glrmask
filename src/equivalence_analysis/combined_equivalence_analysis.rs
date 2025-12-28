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
    
    // Step 1: State equivalence analysis (if beneficial)
    let (reduced_states, state_classes) = if initial_states.len() > state_reduction_threshold {
        let state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            regex,
            tokens,
            initial_states,
        );
        
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

    #[cfg(test)]
    {
        println!("Running combined equivalence analysis verification...");
        // VERIFICATION: Check against reference implementations
        let problem_size = initial_states.len() * tokens.len();
        let use_trellis_verification = problem_size < 1_000_000;
        
        // 1. Verify State Equivalence
        if initial_states.len() > state_reduction_threshold {
            let ref_mapping = super::state_equivalence_analysis_reference::find_state_equivalence_classes(
                regex,
                tokens,
                initial_states,
            );
            
            let ref_classes = super::state_equivalence_analysis_reference::mapping_to_equivalence_classes(
                initial_states,
                &ref_mapping,
            );
            
            if state_classes != ref_classes {
                panic!(
                    "State equivalence mismatch (fast vs reference)!\nFast: {:?}\nRef : {:?}",
                    state_classes, ref_classes
                );
            }
            
            // Trellis-based ground truth verification for small problems
            if use_trellis_verification {
                let trellis_mapping = super::trellis_equivalence_analysis::find_state_equivalence_classes_trellis(
                    regex,
                    tokens,
                    initial_states,
                );

                let trellis_classes = super::trellis_equivalence_analysis::mapping_to_equivalence_classes(
                    initial_states,
                    &trellis_mapping,
                );
                println!("Trellis classes: {:?}", trellis_classes);

                if state_classes != trellis_classes {
                    panic!(
                        "State equivalence mismatch (fast vs trellis ground truth)!\nFast   : {:?}\nTrellis: {:?}",
                        state_classes, trellis_classes
                    );
                }
            }
        }

        // 2. Verify Vocab Equivalence
        let ref_vocab_classes = super::vocab_equivalence_analysis_reference::find_vocab_equivalence_classes(
            regex,
            tokens,
            &reduced_states,
        );
        
        if vocab_classes != ref_vocab_classes {
             let in_fast_not_ref: Vec<_> = vocab_classes.difference(&ref_vocab_classes).collect();
             let in_ref_not_fast: Vec<_> = ref_vocab_classes.difference(&vocab_classes).collect();
             
             panic!(
                "Vocab equivalence mismatch (fast vs reference)!\nIn Fast Only: {:?}\nIn Ref Only : {:?}",
                in_fast_not_ref, in_ref_not_fast
            );
        }
        
        // Trellis-based ground truth verification for small problems
        if use_trellis_verification {
            let trellis_vocab_classes = super::trellis_equivalence_analysis::find_vocab_equivalence_classes_trellis(
                regex,
                tokens,
                &reduced_states,
            );
            println!("Trellis vocab classes: {:?}", trellis_vocab_classes);
            
            if vocab_classes != trellis_vocab_classes {
                let in_fast_not_trellis: Vec<_> = vocab_classes.difference(&trellis_vocab_classes).collect();
                let in_trellis_not_fast: Vec<_> = trellis_vocab_classes.difference(&vocab_classes).collect();
             
                panic!(
                    "Vocab equivalence mismatch (fast vs trellis ground truth)!\nIn Fast Only   : {:?}\nIn Trellis Only: {:?}",
                    in_fast_not_trellis, in_trellis_not_fast
                );
            }
        }
    }
    
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
