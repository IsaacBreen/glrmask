//! State Equivalence Analysis
//!
//! Determines which tokenizer states behave identically for all tokens in a vocabulary.
//! States that are equivalent can be merged, reducing the workload for subsequent
//! vocab equivalence analysis.
//!
//! The algorithm uses a two-stage pipeline:
//! 1. max-length bounded path hashing to collapse obviously equivalent states.
//! 2. Full token-based analysis on the reduced representative set.
//!
//! This avoids scanning the full vocabulary for every state and collapses long
//! bounded-repeat chains efficiently.

use std::collections::BTreeSet;

#[cfg(test)]
use rayon::prelude::*;
#[cfg(test)]
use super::super::compat::Sep1Tokenizer;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

// -----------------------------------------------------------------------------
// Hashing Utilities (128-bit)
// -----------------------------------------------------------------------------

#[cfg(test)]
#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[cfg(test)]
fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn env_flag_enabled_any(names: &[&str]) -> bool {
    names.iter().find_map(|name| std::env::var(name).ok()).map_or(false, |value| {
        let trimmed = value.trim();
        !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
    })
}

#[cfg(test)]
fn profile_equivalence_enabled() -> bool {
    env_flag_enabled_any(&["GLRMASK_PROFILE_EQUIVALENCE", "PROFILE_EQUIVALENCE"])
        || env_flag_enabled("GLRMASK_PROFILE_COMPILE")
}

#[cfg(test)]
fn count_classes(mapping: &[usize]) -> usize {
    mapping.iter().copied().collect::<BTreeSet<_>>().len()
}

/// Find state equivalence classes for a tokenizer.
///
/// Uses a pre-filter + refinement approach:
/// 1. k-step inductive hashing to reduce the number of states.
/// 2. Full token-based analysis only on the reduced set.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to consider
/// * `states` - List of state IDs to analyze
///
/// # Returns
/// A vector where `result[i]` is the representative state for `states[i]`.
/// States with the same representative are equivalent.
#[cfg(test)]
pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    let profile_equivalence = profile_equivalence_enabled();

    if states.is_empty() {
        return Vec::new();
    }

    let prepass_started_at = std::time::Instant::now();
    let pre_mapping = super::max_length::find_state_equivalence_classes(regex, tokens, states);
    let prepass_time = prepass_started_at.elapsed();

    let owned_tokens: Vec<Vec<u8>> = tokens.iter().map(|t| t.as_ref().to_vec()).collect();

    use std::collections::HashMap;

    let mut rep_set: BTreeSet<usize> = BTreeSet::new();
    for &rep in &pre_mapping {
        rep_set.insert(rep);
    }
    let reduced_states: Vec<usize> = rep_set.into_iter().collect();
    let prepass_classes = reduced_states.len();

    if reduced_states.len() == states.len() {
        let refinement_started_at = std::time::Instant::now();
        let mapping = find_state_equivalence_classes_token_based(regex, &owned_tokens, states);
        let refinement_time = refinement_started_at.elapsed();

        if profile_equivalence {
            eprintln!(
                "[glrmask/profile][state_equiv] max_length_ms={:.3} states={}→{} ({:.2}x)",
                prepass_time.as_secs_f64() * 1000.0,
                states.len(),
                prepass_classes,
                states.len() as f64 / prepass_classes.max(1) as f64,
            );
            eprintln!(
                "[glrmask/profile][state_equiv] token_refine_ms={:.3} states={}→{} ({:.2}x)",
                refinement_time.as_secs_f64() * 1000.0,
                states.len(),
                count_classes(&mapping),
                states.len() as f64 / count_classes(&mapping).max(1) as f64,
            );
        }

        return mapping;
    }

    let refinement_started_at = std::time::Instant::now();
    let reduced_mapping = find_state_equivalence_classes_token_based(regex, &owned_tokens, &reduced_states);
    let refinement_time = refinement_started_at.elapsed();
    let mut rep_to_final: HashMap<usize, usize> = HashMap::new();
    for (i, &rep_state) in reduced_states.iter().enumerate() {
        rep_to_final.insert(rep_state, reduced_mapping[i]);
    }

    let mut mapping = vec![0usize; states.len()];
    for (i, &pre_rep) in pre_mapping.iter().enumerate() {
        mapping[i] = rep_to_final[&pre_rep];
    }

    if profile_equivalence {
        let final_classes = count_classes(&mapping);
        let refinement_classes = count_classes(&reduced_mapping);
        eprintln!(
            "[glrmask/profile][state_equiv] max_length_ms={:.3} states={}→{} ({:.2}x)",
            prepass_time.as_secs_f64() * 1000.0,
            states.len(),
            prepass_classes,
            states.len() as f64 / prepass_classes.max(1) as f64,
        );
        eprintln!(
            "[glrmask/profile][state_equiv] token_refine_ms={:.3} states={}→{} ({:.2}x) input={}→{} ({:.2}x)",
            refinement_time.as_secs_f64() * 1000.0,
            states.len(),
            final_classes,
            states.len() as f64 / final_classes.max(1) as f64,
            reduced_states.len(),
            refinement_classes,
            reduced_states.len() as f64 / refinement_classes.max(1) as f64,
        );
    }

    mapping
}

#[cfg(test)]
fn find_state_equivalence_classes_token_based(
    regex: &Sep1Tokenizer,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    use std::collections::HashMap;

    let dfa = regex.dfa();

    // Keep the full token pass here; sampled state equivalence was unsound.

    // Precompute packed transition tables and finalizers for cache efficiency
    const NONE_STATE: u32 = u32::MAX;
    let dfa_transitions: Vec<[u32; 256]> = dfa.states
        .iter()
        .map(|state| {
            let mut table = [NONE_STATE; 256];
            for (byte_idx, &target) in state.transitions.iter().enumerate() {
                table[byte_idx] = target;
            }
            table
        })
        .collect();

    let dfa_finalizers: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.finalizers.iter().copied().collect())
        .collect();

    let mut max_gid: Option<usize> = None;
    for finals in &dfa_finalizers {
        if let Some(m) = finals.iter().max() {
            max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
        }
    }
    let num_groups = max_gid.map(|m| m + 1).unwrap_or(0);

    // =========================================================================
    // PHASE 1: Token testing with early exit for singletons
    // =========================================================================
    // We test tokens in batches, but only on states that haven't been uniquely
    // identified yet. Once a state is in a singleton group, it stays there.

    // Precompute end state hashes
    // CRITICAL: Must include the COUNT of possible futures in the hash!
    // Otherwise sets like [0, 6] and [6] would hash the same because mix(0) = 0.
    let end_state_hashes: Vec<u128> = dfa.states
        .iter()
        .map(|state| {
            // Seed with the length to distinguish sets of different sizes
            let futures = &state.possible_future_group_ids;
            let mut h = mix_u128(futures.len() as u128 | (1u128 << 48));
            // Add (not XOR!) each element's hash for commutativity and collision resistance
            // NOTE: Match reference by NOT adding extra bits to gid
            for &gid in futures {
                h = h.wrapping_add(mix_u128(gid as u128));
            }
            // Flag to distinguish from dead state hash
            h | (1u128 << 127)
        })
        .collect();

    // Precomputed non-greedy flags for proper position tracking

    // Process tokens in lexicographic order and reuse prefix simulations per state.
    let mut sorted_indices: Vec<usize> = (0..tokens.len()).collect();
    sorted_indices.sort_by(|&a, &b| tokens[a].cmp(&tokens[b]));

    let mut sorted_tokens: Vec<&[u8]> = Vec::with_capacity(tokens.len());
    let mut sorted_weights: Vec<u128> = Vec::with_capacity(tokens.len());
    for &idx in &sorted_indices {
        sorted_tokens.push(tokens[idx].as_slice());
        sorted_weights.push(mix_u128((idx + 1) as u128));
    }

    let common_prefix_len = |a: &[u8], b: &[u8]| -> usize {
        let len = a.len().min(b.len());
        let mut i = 0usize;
        while i < len && a[i] == b[i] {
            i += 1;
        }
        i
    };

    let mut lcp_with_prev: Vec<usize> = Vec::with_capacity(sorted_tokens.len());
    let mut prev_token: Option<&[u8]> = None;
    for token in &sorted_tokens {
        let lcp = prev_token.map_or(0, |prev| common_prefix_len(prev, token));
        lcp_with_prev.push(lcp);
        prev_token = Some(token);
    }

    let total_tokens = sorted_tokens.len();

    let mut weight_prefix: Vec<u128> = vec![0u128; total_tokens + 1];
    for i in 0..total_tokens {
        weight_prefix[i + 1] = weight_prefix[i].wrapping_add(sorted_weights[i]);
    }

    let mut empty_end = 0usize;
    while empty_end < total_tokens && sorted_tokens[empty_end].is_empty() {
        empty_end += 1;
    }
    let empty_range = (0usize, empty_end);

    let mut first_byte_ranges: Vec<(usize, usize)> = vec![(0usize, 0usize); 256];
    let mut idx = empty_end;
    while idx < total_tokens {
        let byte = sorted_tokens[idx][0] as usize;
        let start = idx;
        idx += 1;
        while idx < total_tokens
            && !sorted_tokens[idx].is_empty()
            && sorted_tokens[idx][0] as usize == byte
        {
            idx += 1;
        }
        first_byte_ranges[byte] = (start, idx);
    }

    let dead_hash_depth1 = {
        let structure = mix_u128(1_u128 ^ 0xDEAD_DEAD_DEAD_DEAD);
        let end = mix_u128(0xDEADBEEF_u128);
        end.wrapping_add(structure)
    };

    let early_stop =
        env_flag_enabled_any(&["GLRMASK_STATE_EQUIV_EARLY_STOP", "STATE_EQUIV_EARLY_STOP"]);
    let batch_size = 5000usize;

    let mut group_ids: Vec<usize> = vec![0usize; states.len()];
    let mut next_group_ids: Vec<usize> = vec![0usize; states.len()];
    let mut active_indices: Vec<usize> = (0..states.len()).collect();
    let mut active_flags: Vec<bool> = vec![false; states.len()];
    let mut active_hashes: Vec<u128> = vec![0u128; states.len()];
    let mut prev_groups = 1usize;
    let mut stable_batches = 0usize;
    let mut tokens_tested;

    let mut batch_start = 0usize;
    while batch_start < total_tokens {
        if active_indices.is_empty() {
            break;
        }

        let batch_end = (batch_start + batch_size).min(total_tokens);

        let mut batch_hashes: Vec<(usize, u128)> = active_indices
            .par_iter()
            .map_init(
                || {
                    (
                        Vec::<u32>::new(),
                        Vec::<Option<usize>>::new(),
                        Vec::<usize>::new(),
                        vec![-1; num_groups],
                        Vec::<(usize, i32)>::new(),
                    )
                },
                |scratch, &state_idx| {
                let state = states[state_idx] as u32;
                let mut hash_delta: u128 = 0;
                let state_transitions = &dfa_transitions[state as usize];

                let mut dead_weight_sum: u128 = 0;
                let mut live_ranges: Vec<(usize, usize)> = Vec::new();

                if empty_range.0 < empty_range.1 {
                    let start = empty_range.0.max(batch_start);
                    let end = empty_range.1.min(batch_end);
                    if start < end {
                        live_ranges.push((start, end));
                    }
                }

                for byte in 0usize..256 {
                    let (range_start, range_end) = first_byte_ranges[byte];
                    if range_start >= range_end {
                        continue;
                    }
                    let start = range_start.max(batch_start);
                    let end = range_end.min(batch_end);
                    if start >= end {
                        continue;
                    }

                    if state_transitions[byte] == NONE_STATE {
                        let weight_sum = weight_prefix[end].wrapping_sub(weight_prefix[start]);
                        dead_weight_sum = dead_weight_sum.wrapping_add(weight_sum);
                    } else {
                        live_ranges.push((start, end));
                    }
                }

                if dead_weight_sum != 0 {
                    hash_delta = hash_delta.wrapping_add(
                        dead_hash_depth1.wrapping_mul(dead_weight_sum),
                    );
                }

                let (state_stack, dead_depth_stack, depth_marks, positions, changes) = scratch;
                let mut matches_len: usize;
                let mut matches_hash_sum: u128;

                for (range_start, range_end) in live_ranges {
                    if range_start >= range_end {
                        continue;
                    }

                    state_stack.clear();
                    state_stack.push(state);
                    dead_depth_stack.clear();
                    dead_depth_stack.push(None);
                    depth_marks.clear();
                    depth_marks.push(0);
                    if num_groups > 0 {
                        positions.fill(-1);
                    }
                    changes.clear();
                    matches_len = 0;
                    matches_hash_sum = 0;

                    for token_idx in range_start..range_end {
                        let token = sorted_tokens[token_idx];
                        let mut prefix_len = if token_idx == range_start {
                            0
                        } else {
                            lcp_with_prev[token_idx]
                        };
                        let max_prefix = state_stack.len().saturating_sub(1);
                        if prefix_len > max_prefix {
                            prefix_len = max_prefix;
                        }

                        if state_stack.len() > prefix_len + 1 {
                            let target_mark = depth_marks[prefix_len];
                            while changes.len() > target_mark {
                                let (gid, prev_pos) = changes.pop().unwrap();
                                let cur_pos = positions[gid];
                                if cur_pos >= 0 {
                                    let cur_pos_u = cur_pos as u32;
                                    matches_hash_sum = matches_hash_sum.wrapping_sub(mix_u128(
                                        (gid as u128) | ((cur_pos_u as u128) << 32),
                                    ));
                                    if prev_pos < 0 {
                                        matches_len -= 1;
                                        positions[gid] = -1;
                                    } else {
                                        let prev_pos_u = prev_pos as u32;
                                        matches_hash_sum = matches_hash_sum.wrapping_add(mix_u128(
                                            (gid as u128) | ((prev_pos_u as u128) << 32),
                                        ));
                                        positions[gid] = prev_pos;
                                    }
                                } else {
                                    positions[gid] = prev_pos;
                                }
                            }

                            state_stack.truncate(prefix_len + 1);
                            dead_depth_stack.truncate(prefix_len + 1);
                            depth_marks.truncate(prefix_len + 1);
                        }

                        let mut dead_at_depth = dead_depth_stack[prefix_len];

                        if dead_at_depth.is_none() {
                            let mut current = *state_stack.last().unwrap();
                            for (offset, &byte) in token[prefix_len..].iter().enumerate() {
                                if current == NONE_STATE {
                                    dead_at_depth = Some(prefix_len + offset);
                                    break;
                                }
                                let next = dfa_transitions[current as usize][byte as usize];
                                if next == NONE_STATE {
                                    dead_at_depth = Some(prefix_len + offset + 1);
                                    state_stack.push(NONE_STATE);
                                    dead_depth_stack.push(dead_at_depth);
                                    depth_marks.push(changes.len());
                                    break;
                                }
                                current = next;
                                state_stack.push(current);
                                let position = prefix_len + offset + 1;

                                if num_groups > 0 {
                                    for &gid in &dfa_finalizers[current as usize] {
                                        if gid >= num_groups {
                                            continue;
                                        }
                                        let pos_i32 = position as i32;
                                        let prev = positions[gid];
                                        if prev != pos_i32 {
                                            if prev < 0 {
                                                matches_len += 1;
                                                matches_hash_sum = matches_hash_sum.wrapping_add(mix_u128(
                                                    (gid as u128) | ((position as u128) << 32),
                                                ));
                                                changes.push((gid, -1));
                                            } else {
                                                matches_hash_sum = matches_hash_sum.wrapping_sub(mix_u128(
                                                    (gid as u128) | ((prev as u128) << 32),
                                                ));
                                                matches_hash_sum = matches_hash_sum.wrapping_add(mix_u128(
                                                    (gid as u128) | ((position as u128) << 32),
                                                ));
                                                changes.push((gid, prev));
                                            }
                                            positions[gid] = pos_i32;
                                        }
                                    }
                                }

                                dead_depth_stack.push(dead_at_depth);
                                depth_marks.push(changes.len());
                            }
                        }

                        let (structure_hash, end_hash) = if let Some(dead_depth) = dead_at_depth {
                            (
                                mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD),
                                mix_u128(0xDEADBEEF_u128),
                            )
                        } else {
                            let sh = mix_u128(matches_len as u128 | (1u128 << 48))
                                .wrapping_add(matches_hash_sum);
                            let current = *state_stack.last().unwrap();
                            (sh, end_state_hashes[current as usize])
                        };

                        let token_hash = end_hash.wrapping_add(structure_hash);
                        hash_delta = hash_delta.wrapping_add(
                            token_hash.wrapping_mul(sorted_weights[token_idx]),
                        );
                    }
                }

                (state_idx, hash_delta)
            })
            .collect();

        let all_active = active_indices.len() == states.len();

        let mut key_to_group: HashMap<(usize, u128), usize> = HashMap::new();
        let mut num_groups = 0usize;

        if all_active {
            for (state_idx, hash) in batch_hashes.drain(..) {
                active_hashes[state_idx] = hash;
            }

            for i in 0..states.len() {
                let key = (group_ids[i], active_hashes[i]);
                let entry = key_to_group.entry(key).or_insert_with(|| {
                    let id = num_groups;
                    num_groups += 1;
                    id
                });
                next_group_ids[i] = *entry;
            }
        } else {
            for &state_idx in &active_indices {
                active_flags[state_idx] = false;
            }
            for (state_idx, hash) in batch_hashes.drain(..) {
                active_flags[state_idx] = true;
                active_hashes[state_idx] = hash;
            }

            const FROZEN_SIG: u128 = u128::MAX;
            for i in 0..states.len() {
                let sig = if active_flags[i] {
                    active_hashes[i]
                } else {
                    FROZEN_SIG
                };
                let key = (group_ids[i], sig);
                let entry = key_to_group.entry(key).or_insert_with(|| {
                    let id = num_groups;
                    num_groups += 1;
                    id
                });
                next_group_ids[i] = *entry;
            }
        }
        std::mem::swap(&mut group_ids, &mut next_group_ids);

        let mut group_sizes = vec![0usize; num_groups];
        for &gid in &group_ids {
            group_sizes[gid] += 1;
        }

        active_indices.clear();
        for i in 0..states.len() {
            if group_sizes[group_ids[i]] > 1 {
                active_indices.push(i);
            }
        }

        tokens_tested = batch_end;
        if early_stop && tokens_tested * 2 >= total_tokens {
            if num_groups == prev_groups {
                stable_batches += 1;
            } else {
                stable_batches = 0;
            }
            if stable_batches >= 2 {
                break;
            }
        }

        prev_groups = num_groups;
        batch_start = batch_end;
    }

    let num_groups = group_ids.iter().copied().max().map(|v| v + 1).unwrap_or(0);
    let mut group_sizes = vec![0usize; num_groups];
    for &gid in &group_ids {
        group_sizes[gid] += 1;
    }

    let mut rep_for_group: Vec<usize> = vec![usize::MAX; num_groups];
    for (idx, &gid) in group_ids.iter().enumerate() {
        if rep_for_group[gid] == usize::MAX {
            rep_for_group[gid] = states[idx];
        }
    }

    let mut mapping = vec![0usize; states.len()];
    for (idx, &gid) in group_ids.iter().enumerate() {
        mapping[idx] = rep_for_group[gid];
    }

    mapping
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
///
/// # Arguments
/// * `states` - The original list of state IDs
/// * `mapping` - The mapping where `mapping[i]` is the representative for `states[i]`
///
/// # Returns
/// A set of equivalence classes, where each class is a set of state IDs.
pub fn mapping_to_equivalence_classes(states: &[usize], mapping: &[usize]) -> StateEquivalenceResult {
    let mut rep_to_class: std::collections::BTreeMap<usize, BTreeSet<usize>> = std::collections::BTreeMap::new();
    
    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }
    
    rep_to_class.into_values().collect()
}
