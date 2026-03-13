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

use super::compat::{FlatDfa, FlatDfaState, GroupID, Sep1Tokenizer};
use crate::ds::bitset::BitSet;

use super::state::fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab::fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};
use super::vocab::slow::{partitions_are_comparable, partition_is_at_least_as_fine};

const MEDIUM_VOCAB_EQUIV_VERIFICATION_ENV: &str = "MEDIUM_VOCAB_EQUIV_VERIFICATION";
const SLOW_VOCAB_EQUIV_VERIFICATION_ENV: &str = "SLOW_VOCAB_EQUIV_VERIFICATION";
const VERY_SLOW_VOCAB_EQUIV_VERIFICATION_ENV: &str = "VERY_SLOW_VOCAB_EQUIV_VERIFICATION";
const VERY_SLOW_VOCAB_EQUIV_PRIMARY_ENV: &str = "VERY_SLOW_VOCAB_EQUIV_PRIMARY";
const REFERENCE_EQUIV_VERIFICATION_ENV: &str = "REFERENCE_EQUIV_VERIFICATION";
const REFERENCE_VOCAB_EQUIV_PRIMARY_ENV: &str = "REFERENCE_VOCAB_EQUIV_PRIMARY";
const REFERENCE_STATE_EQUIV_PRIMARY_ENV: &str = "REFERENCE_STATE_EQUIV_PRIMARY";

/// Result of combined equivalence analysis.
pub struct CombinedEquivalenceResult {
    /// Vocab equivalence classes: sets of token indices that behave identically.
    pub vocab_classes: VocabEquivalenceResult,

    /// State equivalence classes: sets of state IDs that behave identically.
    pub state_classes: StateEquivalenceResult,
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let trimmed = v.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn verify_vocab_partition(
    label: &str,
    fast_vocab_classes: &VocabEquivalenceResult,
    candidate_vocab_classes: &VocabEquivalenceResult,
) {
    if !partitions_are_comparable(fast_vocab_classes, candidate_vocab_classes) {
        panic!(
            "Vocab equivalence mismatch (fast vs {label} not comparable)!\nFast ({} classes): {:?}\n{label} ({} classes): {:?}",
            fast_vocab_classes.len(),
            fast_vocab_classes,
            candidate_vocab_classes.len(),
            candidate_vocab_classes,
        );
    }
}

/// Verify that the reference analysis merges at least as aggressively as fast.
/// The reference is the ground truth — it must merge everything that fast merges,
/// and potentially more. Panics if fast merged tokens that reference kept separate.
fn verify_vocab_partition_reference(
    fast_vocab_classes: &VocabEquivalenceResult,
    reference_vocab_classes: &VocabEquivalenceResult,
) {
    if !partition_is_at_least_as_fine(fast_vocab_classes, reference_vocab_classes) {
        panic!(
            "Fast vocab equivalence merged tokens that reference kept separate!\n\
             Fast ({} classes): {:?}\n\
             Reference ({} classes): {:?}",
            fast_vocab_classes.len(),
            fast_vocab_classes,
            reference_vocab_classes.len(),
            reference_vocab_classes,
        );
    }
}

fn print_vocab_verification_stats(label: &str, vocab_classes: &VocabEquivalenceResult) {
    eprintln!(
        "[vocab equiv verification] {label}: {} classes",
        vocab_classes.len()
    );
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
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
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
        let state_classes =
            state_equivalence_analysis::mapping_to_equivalence_classes(initial_states, &state_reps);

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
    let vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_follow(
        regex,
        tokens,
        &reduced_states,
        disallowed_follows,
    );

    if env_flag_enabled(SLOW_VOCAB_EQUIV_VERIFICATION_ENV) {
        let slow_vocab_classes = super::vocab::slow::find_vocab_equivalence_classes_with_follow(
            regex,
            tokens,
            &reduced_states,
            disallowed_follows,
        );
        print_vocab_verification_stats("slow", &slow_vocab_classes);
        verify_vocab_partition("slow", &vocab_classes, &slow_vocab_classes);
    }

    if env_flag_enabled(MEDIUM_VOCAB_EQUIV_VERIFICATION_ENV) {
        let medium_vocab_classes = super::vocab::medium::find_vocab_equivalence_classes_with_follow(
            regex,
            tokens,
            &reduced_states,
            disallowed_follows,
        );
        print_vocab_verification_stats("medium", &medium_vocab_classes);
        verify_vocab_partition("medium", &vocab_classes, &medium_vocab_classes);
    }

    if env_flag_enabled(VERY_SLOW_VOCAB_EQUIV_VERIFICATION_ENV) {
        let very_slow_vocab_classes =
            super::vocab::very_slow::find_vocab_equivalence_classes_with_follow(
                regex,
                tokens,
                &reduced_states,
                disallowed_follows,
            );
        print_vocab_verification_stats("very_slow", &very_slow_vocab_classes);
        verify_vocab_partition("very_slow", &vocab_classes, &very_slow_vocab_classes);
    }

    // --- Reference analysis ---
    // Run once if any reference env var is enabled, reuse the result.
    let need_reference_verify = env_flag_enabled(REFERENCE_EQUIV_VERIFICATION_ENV);
    let need_reference_vocab = env_flag_enabled(REFERENCE_VOCAB_EQUIV_PRIMARY_ENV);
    let need_reference_state = env_flag_enabled(REFERENCE_STATE_EQUIV_PRIMARY_ENV);

    let reference_result = if need_reference_verify || need_reference_vocab || need_reference_state {
        Some(super::reference::find_equivalence_classes(
            regex,
            tokens,
            &reduced_states,
            disallowed_follows,
            ignore_terminal.map(|t| t as usize),
        ))
    } else {
        None
    };

    if need_reference_verify {
        let ref_result = reference_result.as_ref().unwrap();
        print_vocab_verification_stats("reference", &ref_result.vocab_classes);
        eprintln!(
            "[state equiv verification] reference: {} classes",
            ref_result.state_classes.len()
        );
        verify_vocab_partition_reference(&vocab_classes, &ref_result.vocab_classes);
    }

    // Replace vocab classes if reference or very_slow primary is requested
    let vocab_classes = if need_reference_vocab {
        let ref_result = reference_result.as_ref().unwrap();
        print_vocab_verification_stats("reference (primary)", &ref_result.vocab_classes);
        ref_result.vocab_classes.clone()
    } else if env_flag_enabled(VERY_SLOW_VOCAB_EQUIV_PRIMARY_ENV) {
        let very_slow_vocab_classes =
            super::vocab::very_slow::find_vocab_equivalence_classes_with_follow(
                regex,
                tokens,
                &reduced_states,
                disallowed_follows,
            );
        print_vocab_verification_stats("very_slow (primary)", &very_slow_vocab_classes);
        very_slow_vocab_classes
    } else {
        vocab_classes
    };

    // Replace state classes if reference state primary is requested
    let state_classes = if need_reference_state {
        let ref_result = reference_result.as_ref().unwrap();
        eprintln!(
            "[state equiv] reference (primary): {} classes",
            ref_result.state_classes.len()
        );
        ref_result.state_classes.clone()
    } else {
        state_classes
    };

    CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    }
}
