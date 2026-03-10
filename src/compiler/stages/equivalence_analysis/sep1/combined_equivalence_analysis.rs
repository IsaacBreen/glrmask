#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(clippy::all)]
#![allow(unreachable_code)]
#![allow(unused_assignments)]
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

use super::compat::{Sep1Tokenizer, FlatDfa, FlatDfaState, GroupID};

use super::state::fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab::fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};

/// Result of combined equivalence analysis.
pub struct CombinedEquivalenceResult {
    /// Vocab equivalence classes: sets of token indices that behave identically.
    pub vocab_classes: VocabEquivalenceResult,
    
    /// State equivalence classes: sets of state IDs that behave identically.
    pub state_classes: StateEquivalenceResult,
}

#[cfg(test)]
fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let trimmed = v.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn should_run_trellis_verification() -> bool {
    // Trellis checks are expensive and can panic on known mismatch classes.
    // We therefore gate them behind explicit pedantic/debug/test signals.
    let pedantic_mode = env_flag_enabled("SEP1_PEDANTIC");
    // let debug_gate = false;
    let debug_gate = false;
    let test_gate = cfg!(test) && env_flag_enabled("SEP1_TEST_TRELLIS_VERIFY");
    pedantic_mode || debug_gate || test_gate
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
pub fn compute_combined_equivalence<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    initial_states: &[usize],
    suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
) -> CombinedEquivalenceResult {
    // State equivalence reduction: groups initial states with identical tokenizer
    // behavior. The cost is O(V×S) token walks (same as vocab analysis), so it's
    // only beneficial when the reduction ratio is high (>50%). For most schemas
    // the reduction ratio is low (10-20%), making it a net loss. Only enable for
    // very large state counts where DFA/NWA cost dominates.
    let state_reduction_threshold = std::env::var("STATE_EQUIV_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5000);

    let start = std::time::Instant::now();
    let profile_equivalence = std::env::var("PROFILE_EQUIVALENCE").is_ok();
    let state_start = std::time::Instant::now();
    
    // Step 1: State equivalence analysis (if beneficial)
    let (reduced_states, state_classes) = if initial_states.len() > state_reduction_threshold {
        // Convert to owned tokens for state equivalence (cold path)
        let owned_tokens: Vec<Vec<u8>> = tokens.iter().map(|t| t.as_ref().to_vec()).collect();
        let state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            regex,
            &owned_tokens,
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
        
        // sep1_debug!(
            // 3,
            // "Combined equiv: state reduction {} -> {} states in {:?}",
            // initial_states.len(),
            // reduced.len(),
            // start.elapsed(),
        // );
        
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

    let state_time = state_start.elapsed();
    if profile_equivalence {
        // sep1_timing!(
            // "TIMING: equivalence.state {:?} ({} -> {} states)",
            // state_time,
            // initial_states.len(),
            // reduced_states.len(),
        // );
    }
    
    // Step 2: Vocab equivalence analysis on reduced states
    let vocab_start = std::time::Instant::now();
    
    let vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_follow(
        regex,
        tokens,
        &reduced_states,
        suffix_group_mask,
        ever_allowed_by_group,
        group_to_class,
    );

    if profile_equivalence {
        let vocab_time = vocab_start.elapsed();
        // sep1_timing!(
            // "TIMING: equivalence.vocab {:?} ({} tokens -> {} classes)",
            // vocab_time,
            // tokens.len(),
            // vocab_classes.len(),
        // );
        // sep1_timing!("TIMING: equivalence.total {:?}", start.elapsed());
    }
    
    // sep1_debug!(
        // 3,
        // "Combined equiv: vocab analysis {} tokens -> {} classes in {:?}",
        // tokens.len(),
        // vocab_classes.len(),
        // vocab_start.elapsed(),
    // );

    #[cfg(test)]
    {
        if std::env::var("SKIP_EQUIV_VERIFICATION").is_ok() {
            // Skipping verification
        } else {
        fn vocab_is_refinement(
            candidate: &VocabEquivalenceResult,
            target: &VocabEquivalenceResult,
        ) -> bool {
            candidate.iter().all(|candidate_class| {
                target.iter().any(|target_class| {
                    candidate_class
                        .iter()
                        .all(|token| target_class.contains(token))
                })
            })
        }

        fn vocab_is_comparable(
            a: &VocabEquivalenceResult,
            b: &VocabEquivalenceResult,
        ) -> bool {
            vocab_is_refinement(a, b) || vocab_is_refinement(b, a)
        }

        // Cross-validate: fast version vs trellis (slow) version
        let trellis_vocab_classes = super::vocab::slow::find_vocab_equivalence_classes_with_follow(
            regex,
            tokens,
            &reduced_states,
            suffix_group_mask,
            ever_allowed_by_group,
            group_to_class,
        );
        if !vocab_is_comparable(&vocab_classes, &trellis_vocab_classes) {
            panic!(
                "Vocab equivalence mismatch (fast vs trellis/slow not comparable)!\nFast ({} classes): {:?}\nTrellis ({} classes): {:?}",
                vocab_classes.len(), vocab_classes,
                trellis_vocab_classes.len(), trellis_vocab_classes
            );
        }

        // Cross-validate: flat (medium) version
        let flat_vocab_classes = super::vocab::medium::find_vocab_equivalence_classes_with_follow(
            regex,
            tokens,
            &reduced_states,
            suffix_group_mask,
            ever_allowed_by_group,
            group_to_class,
        );
        if !vocab_is_comparable(&vocab_classes, &flat_vocab_classes) {
            panic!(
                "Vocab equivalence mismatch (fast vs flat/medium not comparable)!\nFast ({} classes): {:?}\nFlat ({} classes): {:?}",
                vocab_classes.len(), vocab_classes,
                flat_vocab_classes.len(), flat_vocab_classes
            );
        }

        } // end of else (SKIP_EQUIV_VERIFICATION)
    }
    
    CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    }
}

/// Minimized entry point that just returns vocab equivalence classes.
///
/// Use this when you don't need the state mapping information.
pub fn find_vocab_equivalence_classes_with_state_reduction<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    compute_combined_equivalence(regex, tokens, initial_states, None, None, None).vocab_classes
}
