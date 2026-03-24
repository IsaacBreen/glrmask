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

use std::collections::{BTreeMap, BTreeSet, HashMap};

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
const SKIP_MAX_LENGTH_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_MAX_LENGTH_STATE_EQUIV";
const SKIP_TOKEN_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_TOKEN_STATE_EQUIV";

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

/// Dump a witness JSON file containing the two LLM tokens and the tokenizer DFA
/// pruned from the distinguishing state. Written to `witness.json` in the
/// current directory.
fn dump_witness_json<S: AsRef<[u8]>>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    left_token: usize,
    right_token: usize,
    distinguishing_state: usize,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) {
    let dfa = regex.dfa();
    let tokenizer_start = dfa.start_state;

    // Collect reachable states from BOTH the distinguishing state AND the
    // tokenizer's start state. The trellis DAG restarts segments from the
    // tokenizer start, so both regions are needed for a self-contained test.
    let mut visited = vec![false; dfa.states.len()];
    let mut stack = vec![distinguishing_state];
    if tokenizer_start != distinguishing_state && tokenizer_start < dfa.states.len() {
        stack.push(tokenizer_start);
        visited[tokenizer_start] = true;
    }
    visited[distinguishing_state] = true;
    let mut reachable_order: Vec<usize> = Vec::new();

    while let Some(state) = stack.pop() {
        reachable_order.push(state);
        for &target in &dfa.states[state].transitions {
            if target == u32::MAX {
                continue;
            }
            let t = target as usize;
            if t < dfa.states.len() && !visited[t] {
                visited[t] = true;
                stack.push(t);
            }
        }
    }
    reachable_order.sort_unstable();

    // Build old→new state id mapping.
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_id, &old_id) in reachable_order.iter().enumerate() {
        old_to_new.insert(old_id, new_id);
    }

    // Build pruned DFA states as JSON-serializable structures.
    let mut pruned_states = Vec::new();
    for &old_id in &reachable_order {
        let state = &dfa.states[old_id];
        // Compact transition repr: only non-dead entries, mapped to new ids.
        let mut transitions = serde_json::Map::new();
        for (byte, &target) in state.transitions.iter().enumerate() {
            if target == u32::MAX {
                continue;
            }
            let t = target as usize;
            if let Some(&new_target) = old_to_new.get(&t) {
                transitions.insert(byte.to_string(), serde_json::json!(new_target));
            }
        }
        pruned_states.push(serde_json::json!({
            "original_id": old_id,
            "finalizers": state.finalizers,
            "possible_future_group_ids": state.possible_future_group_ids,
            "transitions": transitions,
        }));
    }

    // Serialize disallowed_follows: map group_id -> list of group_ids that are disallowed.
    let disallowed_follows_json: BTreeMap<String, Vec<usize>> = disallowed_follows
        .iter()
        .map(|(&group_id, bits)| {
            (group_id.to_string(), bits.iter().collect())
        })
        .collect();

    let witness_json = serde_json::json!({
        "left_token_index": left_token,
        "left_token_bytes": tokens[left_token].as_ref(),
        "left_token_preview": format_token_preview(tokens[left_token].as_ref()),
        "right_token_index": right_token,
        "right_token_bytes": tokens[right_token].as_ref(),
        "right_token_preview": format_token_preview(tokens[right_token].as_ref()),
        "distinguishing_state": distinguishing_state,
        "num_groups": regex.dfa().states.iter()
            .flat_map(|s| s.finalizers.iter().chain(s.possible_future_group_ids.iter()))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0),
        "disallowed_follows": disallowed_follows_json,
        "pruned_dfa": {
            "start_state": old_to_new.get(&tokenizer_start).copied().unwrap_or(0),
            "distinguishing_state": old_to_new[&distinguishing_state],
            "original_start_state": tokenizer_start,
            "original_distinguishing_state": distinguishing_state,
            "num_states": pruned_states.len(),
            "states": pruned_states,
        },
    });

    let path = "witness.json";
    match std::fs::write(path, serde_json::to_string_pretty(&witness_json).unwrap_or_default()) {
        Ok(()) => eprintln!("[witness] Dumped to {path}"),
        Err(e) => eprintln!("[witness] Failed to write {path}: {e}"),
    }
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
            let state_str = match state {
                Some(state) => format!("{} distinguishing_state={state}", witness.summary),
                None => format!("{} distinguishing_state=unavailable", witness.summary),
            };
            // Dump witness JSON if a distinguishing state was found.
            if let Some(dist_state) = state {
                dump_witness_json(
                    regex,
                    tokens,
                    witness.left_token,
                    witness.right_token,
                    dist_state,
                    disallowed_follows,
                );
            }
            state_str
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

fn verify_state_partition_reference(
    fast_state_classes: &StateEquivalenceResult,
    reference_state_classes: &StateEquivalenceResult,
) {
    let fast_state_classes: BTreeSet<Vec<_>> = fast_state_classes.iter().map(|class| class.iter().copied().collect()).collect();
    let reference_state_classes: BTreeSet<Vec<_>> = reference_state_classes.iter().map(|class| class.iter().copied().collect()).collect();
    assert!(
        partition_is_at_least_as_fine(&fast_state_classes, &reference_state_classes),
        "Fast state equivalence merged tokens that reference kept separate!\n\
         Fast classes: {}\n\
         Reference classes: {}",
        fast_state_classes.len(),
        reference_state_classes.len(),
    );
}

fn print_vocab_verification_stats(label: &str, vocab_classes: &VocabEquivalenceResult) {
    eprintln!(
        "[vocab equiv verification] {label}: {} classes",
        vocab_classes.len()
    );
}

pub(crate) fn check_live_minimal_tokenizer_fineness() {
    let b_or_c = crate::automata::lexer::ast::class(crate::ds::u8set::U8Set::from_bytes(b"bc"));
    let tokenizer = crate::compiler::compile::build_tokenizer_from_exprs(&[
        crate::automata::lexer::ast::star(b_or_c.clone()),
        crate::automata::lexer::ast::seq(vec![
            crate::automata::lexer::ast::star(b_or_c),
            crate::automata::lexer::ast::bytes(b"b"),
        ]),
    ]);
    let regex = Sep1Tokenizer::new(&tokenizer);

    let mut disallowed_follows = BTreeMap::new();
    let mut all_groups = BitSet::new(2);
    all_groups.set(0);
    all_groups.set(1);
    disallowed_follows.insert(0, all_groups.clone());
    disallowed_follows.insert(1, all_groups);

    let tokens = vec![b"ba".to_vec(), b"bca".to_vec()];
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
    let skip_max_length = env_flag_enabled(SKIP_MAX_LENGTH_STATE_EQUIV_ENV);
    let skip_token_state = env_flag_enabled(SKIP_TOKEN_STATE_EQUIV_ENV);

    let pre_state_reps = if skip_max_length {
        initial_states.to_vec()
    } else {
        super::state::max_length::find_state_equivalence_classes(
            regex,
            tokens,
            initial_states,
        )
    };

    let mut rep_set: BTreeSet<usize> = BTreeSet::new();
    for &rep in &pre_state_reps {
        rep_set.insert(rep);
    }

    let pre_reduced_states: Vec<usize> = rep_set.into_iter().collect();
    let state_reps = if skip_token_state {
        pre_state_reps
    } else {
        let reduced_state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            regex,
            tokens,
            &pre_reduced_states,
        );
        let mut rep_to_final: HashMap<usize, usize> = HashMap::new();
        for (i, &rep_state) in pre_reduced_states.iter().enumerate() {
            rep_to_final.insert(rep_state, reduced_state_reps[i]);
        }
        pre_state_reps
            .iter()
            .map(|pre_rep| rep_to_final[pre_rep])
            .collect()
    };
    let state_classes =
        state_equivalence_analysis::mapping_to_equivalence_classes(initial_states, &state_reps);
    let mut final_rep_set: BTreeSet<usize> = BTreeSet::new();
    for &rep in &state_reps {
        final_rep_set.insert(rep);
    }
    let reduced_states: Vec<usize> = final_rep_set.into_iter().collect();

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
    let need_reference_verify = env_flag_enabled(REFERENCE_EQUIV_VERIFICATION_ENV) || cfg!(test);
    let need_reference_vocab = env_flag_enabled(REFERENCE_VOCAB_EQUIV_PRIMARY_ENV);
    let need_reference_state = env_flag_enabled(REFERENCE_STATE_EQUIV_PRIMARY_ENV);

    let reference_result = if need_reference_verify || need_reference_vocab || need_reference_state {
        Some(super::reference::find_equivalence_classes(
            regex,
            tokens,
            &initial_states,
            disallowed_follows,
            ignore_terminal.map(|t| t as usize),
        ))
    } else {
        None
    };

    if need_reference_verify {
        let ref_result = reference_result.as_ref().unwrap();
        print_vocab_verification_stats("reference", &ref_result.vocab_classes);
        verify_state_partition_reference(
            &state_classes,
            &ref_result.state_classes,
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::automata::lexer::ast::{bytes, star};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use crate::compiler::stages::equivalence_analysis::compat::Sep1Tokenizer;
    use crate::compiler::stages::equivalence_analysis::reference::find_equivalence_classes;
    use crate::compiler::stages::equivalence_analysis::state::fast as fast_state_equivalence;
    use crate::ds::bitset::BitSet;

    use super::verify_state_partition_reference;

    #[test]
    fn unrestricted_state_partition_refines_disallowed_follow_reference() {
        let exprs = [bytes(b"a"), star(bytes(b"b")), bytes(b"c")];
        let tokenizer = build_tokenizer_from_exprs(&exprs);
        let sep1 = Sep1Tokenizer::new(&tokenizer);

        let tokens: Vec<Vec<u8>> = vec![
            b"c".to_vec(),
            b"ca".to_vec(),
            b"cba".to_vec(),
            b"bb".to_vec(),
        ];
        let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

        let mut disallowed = BTreeMap::new();
        let mut bits = BitSet::new(3);
        bits.set(1);
        disallowed.insert(2u32, bits);

        let fast_mapping =
            fast_state_equivalence::find_state_equivalence_classes(&sep1, &tokens, &states);
        let fast_classes =
            fast_state_equivalence::mapping_to_equivalence_classes(&states, &fast_mapping);
        let reference = find_equivalence_classes(&sep1, &tokens, &states, &disallowed, None);

        verify_state_partition_reference(&fast_classes, &reference.state_classes);
    }
}
