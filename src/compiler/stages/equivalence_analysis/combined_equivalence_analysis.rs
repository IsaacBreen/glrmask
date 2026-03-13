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

use super::compat::{FlatDfa, GroupID, Sep1Tokenizer};
use crate::ds::bitset::BitSet;

use super::state::fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab::fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};
use super::vocab::slow::{partitions_are_comparable, partition_is_at_least_as_fine};

const MEDIUM_VOCAB_EQUIV_VERIFICATION_ENV: &str = "MEDIUM_VOCAB_EQUIV_VERIFICATION";
const SLOW_VOCAB_EQUIV_VERIFICATION_ENV: &str = "SLOW_VOCAB_EQUIV_VERIFICATION";
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

fn format_token_preview(token: &[u8]) -> String {
    let escaped: String = token
        .iter()
        .flat_map(|byte| std::ascii::escape_default(*byte))
        .map(char::from)
        .collect();

    const MAX_PREVIEW_LEN: usize = 80;
    if escaped.len() <= MAX_PREVIEW_LEN {
        escaped
    } else {
        format!("{}...", &escaped[..MAX_PREVIEW_LEN])
    }
}

fn format_index_ranges(indices: &[usize]) -> String {
    if indices.is_empty() {
        return "[]".to_string();
    }

    const MAX_PARTS: usize = 8;
    let mut parts = Vec::new();
    let mut range_start = indices[0];
    let mut range_end = indices[0];

    for &index in &indices[1..] {
        if index == range_end + 1 {
            range_end = index;
            continue;
        }

        if range_start == range_end {
            parts.push(range_start.to_string());
        } else {
            parts.push(format!("{range_start}-{range_end}"));
        }
        range_start = index;
        range_end = index;
    }

    if range_start == range_end {
        parts.push(range_start.to_string());
    } else {
        parts.push(format!("{range_start}-{range_end}"));
    }

    let total_parts = parts.len();
    if total_parts > MAX_PARTS {
        parts.truncate(MAX_PARTS);
        parts.push("...".to_string());
    }

    format!("[{}] (n={})", parts.join(","), indices.len())
}

struct ReferenceMismatchWitness {
    left_token: usize,
    right_token: usize,
    summary: String,
}

fn find_reference_mismatch_witness<S: AsRef<[u8]>>(
    fast_vocab_classes: &VocabEquivalenceResult,
    reference_vocab_classes: &VocabEquivalenceResult,
    tokens: &[S],
) -> Option<ReferenceMismatchWitness> {
    let reference_classes: Vec<&Vec<usize>> = reference_vocab_classes.iter().collect();
    let mut reference_class_by_token = vec![usize::MAX; tokens.len()];
    for (class_idx, reference_class) in reference_classes.iter().enumerate() {
        for &token_idx in reference_class.iter() {
            if token_idx < reference_class_by_token.len() {
                reference_class_by_token[token_idx] = class_idx;
            }
        }
    }

    let mut best_witness: Option<((usize, usize, usize, usize), ReferenceMismatchWitness)> = None;

    for fast_class in fast_vocab_classes {
        if reference_vocab_classes
            .iter()
            .any(|reference_class| fast_class.iter().all(|token_idx| reference_class.contains(token_idx)))
        {
            continue;
        }

        let mut best_token_per_reference_class = BTreeMap::<usize, usize>::new();
        for &token_idx in fast_class {
            let reference_class_idx = reference_class_by_token[token_idx];
            if reference_class_idx == usize::MAX {
                continue;
            }

            let replace = best_token_per_reference_class
                .get(&reference_class_idx)
                .map(|&best_token_idx| {
                    let token = tokens[token_idx].as_ref();
                    let best_token = tokens[best_token_idx].as_ref();
                    (token.len(), token_idx) < (best_token.len(), best_token_idx)
                })
                .unwrap_or(true);
            if replace {
                best_token_per_reference_class.insert(reference_class_idx, token_idx);
            }
        }

        let mut representatives: Vec<(usize, usize)> = best_token_per_reference_class
            .into_iter()
            .map(|(reference_class_idx, token_idx)| (token_idx, reference_class_idx))
            .collect();
        representatives.sort_by_key(|&(token_idx, _)| (tokens[token_idx].as_ref().len(), token_idx));

        if representatives.len() < 2 {
            continue;
        }

        let (left_token, left_reference_class_idx) = representatives[0];
        let (right_token, right_reference_class_idx) = representatives[1];
        let left_reference_class = reference_classes[left_reference_class_idx];
        let right_reference_class = reference_classes[right_reference_class_idx];

        let score = (
            tokens[left_token].as_ref().len() + tokens[right_token].as_ref().len(),
            tokens[left_token].as_ref().len().max(tokens[right_token].as_ref().len()),
            left_token,
            right_token,
        );
        let witness = ReferenceMismatchWitness {
            left_token,
            right_token,
            summary: format!(
                "fast_class={} left_token={} \"{}\" right_token={} \"{}\" left_reference_class={} right_reference_class={}",
                format_index_ranges(fast_class),
                left_token,
                format_token_preview(tokens[left_token].as_ref()),
                right_token,
                format_token_preview(tokens[right_token].as_ref()),
                format_index_ranges(left_reference_class),
                format_index_ranges(right_reference_class),
            ),
        };

        if best_witness
            .as_ref()
            .map(|(best_score, _)| score < *best_score)
            .unwrap_or(true)
        {
            best_witness = Some((score, witness));
        }
    }

    best_witness.map(|(_, witness)| witness)
}

fn find_reference_distinguishing_state<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    left_token: usize,
    right_token: usize,
) -> Option<usize> {
    let pair_tokens = [tokens[left_token].as_ref(), tokens[right_token].as_ref()];
    let dfa = regex.dfa();

    fn state_complexity(dfa: &FlatDfa, start_state: usize) -> (usize, usize) {
        if start_state >= dfa.states.len() {
            return (usize::MAX, usize::MAX);
        }

        let mut visited = vec![false; dfa.states.len()];
        let mut stack = vec![start_state];
        visited[start_state] = true;
        let mut state_count = 0usize;
        let mut transition_count = 0usize;

        while let Some(state) = stack.pop() {
            state_count += 1;
            for &target in &dfa.states[state].transitions {
                if target == u32::MAX {
                    continue;
                }
                transition_count += 1;
                let target = target as usize;
                if target < dfa.states.len() && !visited[target] {
                    visited[target] = true;
                    stack.push(target);
                }
            }
        }

        (state_count, transition_count)
    }

    initial_states
        .iter()
        .copied()
        .filter(|&state| {
            let result = super::reference::find_equivalence_classes_with_progress(
                regex,
                &pair_tokens,
                &[state],
                disallowed_follows,
                ignore_terminal.map(|terminal| terminal as usize),
                false,
            );
            result.vocab_classes.len() > 1
        })
        .min_by_key(|&state| {
            let (reachable_states, reachable_transitions) = state_complexity(dfa, state);
            (reachable_states, reachable_transitions, state)
        })
}

/// Verify that the reference analysis merges at least as aggressively as fast.
/// The reference is the ground truth — it must merge everything that fast merges,
/// and potentially more. Panics if fast merged tokens that reference kept separate.
fn verify_vocab_partition_reference<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    fast_vocab_classes: &VocabEquivalenceResult,
    reference_vocab_classes: &VocabEquivalenceResult,
    tokens: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) {
    if !partition_is_at_least_as_fine(fast_vocab_classes, reference_vocab_classes) {
        let witness = find_reference_mismatch_witness(
            fast_vocab_classes,
            reference_vocab_classes,
            tokens,
        )
        .map(|witness| {
            let state = find_reference_distinguishing_state(
                regex,
                tokens,
                initial_states,
                disallowed_follows,
                ignore_terminal,
                witness.left_token,
                witness.right_token,
            );
            match state {
                Some(state) => format!("{} distinguishing_state={state}", witness.summary),
                None => format!("{} distinguishing_state=unavailable", witness.summary),
            }
        })
        .unwrap_or_else(|| "unavailable".to_string());
        eprintln!("[reference vocab mismatch] {witness}");
        panic!(
            "Fast vocab equivalence merged tokens that reference kept separate!\n\
             Witness: {witness}\n\
             Fast classes: {}\n\
             Reference classes: {}",
            fast_vocab_classes.len(),
            reference_vocab_classes.len(),
        );
    }
}

fn print_vocab_verification_stats(label: &str, vocab_classes: &VocabEquivalenceResult) {
    eprintln!(
        "[vocab equiv verification] {label}: {} classes",
        vocab_classes.len()
    );
}

pub(crate) fn repro_live_quote_witness_minimal_fineness_panic() {
    let comma_or_quote = crate::automata::lexer::ast::choice(vec![
        crate::automata::lexer::ast::bytes(b","),
        crate::automata::lexer::ast::bytes(b"'"),
    ]);
    let tokenizer = crate::compiler::compile::build_tokenizer_from_exprs(&[
        crate::automata::lexer::ast::star(comma_or_quote.clone()),
        crate::automata::lexer::ast::seq(vec![
            crate::automata::lexer::ast::star(comma_or_quote),
            crate::automata::lexer::ast::bytes(b","),
        ]),
    ]);
    let regex = Sep1Tokenizer::new(&tokenizer);

    let mut disallowed_follows = BTreeMap::new();
    let mut all_groups = BitSet::new(2);
    all_groups.set(0);
    all_groups.set(1);
    disallowed_follows.insert(0, all_groups.clone());
    disallowed_follows.insert(1, all_groups);

    let tokens = vec![b",\"".to_vec(), b",\'\"".to_vec()];
    let initial_states = [regex.initial_state_id()];

    let fast_vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_follow(
        &regex,
        &tokens,
        &initial_states,
        &disallowed_follows,
    );
    let reference = super::reference::find_equivalence_classes(
        &regex,
        &tokens,
        &initial_states,
        &disallowed_follows,
        None,
    );

    verify_vocab_partition_reference(
        &regex,
        &fast_vocab_classes,
        &reference.vocab_classes,
        &tokens,
        &initial_states,
        &disallowed_follows,
        None,
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
        verify_vocab_partition_reference(
            regex,
            &vocab_classes,
            &ref_result.vocab_classes,
            tokens,
            &reduced_states,
            disallowed_follows,
            ignore_terminal,
        );
    }

    // Replace vocab classes if reference primary is requested
    let vocab_classes = if need_reference_vocab {
        let ref_result = reference_result.as_ref().unwrap();
        print_vocab_verification_stats("reference (primary)", &ref_result.vocab_classes);
        ref_result.vocab_classes.clone()
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
