//! State equivalence analysis.
//!
//! Performs full token-based refinement over the supplied tokenizer states.
//! Any coarse max-length reduction happens in combined equivalence analysis.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use super::super::compat::TokenizerView;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

#[derive(Clone, Copy)]
struct WalkFrame {
    state: u32,
    dead_at_depth: Option<usize>,
    changes_len: usize,
}

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn profile_equivalence_enabled() -> bool {
    env_flag_enabled("PROFILE_EQUIVALENCE") || env_flag_enabled("GLRMASK_PROFILE_COMPILE")
}

fn count_classes(mapping: &[usize]) -> usize {
    mapping.iter().copied().collect::<BTreeSet<_>>().len()
}

#[inline(always)]
fn mix_tagged(hash: u128, tag: u128, value: u128) -> u128 {
    mix_u128(hash ^ tag.wrapping_add(value.rotate_left(17)))
}

fn hash_future_groups(future_groups: &[usize]) -> u128 {
    let mut hash = mix_u128(0xF0C7_F0C7_F0C7_F0C7 ^ future_groups.len() as u128);
    for &gid in future_groups {
        hash = mix_tagged(hash, 0x9E37_79B9_7F4A_7C15, gid as u128);
    }
    hash
}

fn hash_trellis_node_from_positions(
    end_state: Option<usize>,
    positions: &[i32],
    token_len: usize,
    suffix_hashes: &[u128],
    future_group_hashes: &[u128],
    skip_groups: &[bool],
) -> u128 {
    const DEAD_NODE_TAG: u128 = 0xDEAD_DEAD_DEAD_DEAD;
    const ACCEPT_SINK_HASH: u128 = 0xA11C_EA5E_A11C_EA5E;
    const EDGE_COUNT_TAG: u128 = 0xEDEC_EDEC_EDEC_EDEC;
    const EDGE_GID_TAG: u128 = 0xE001_E001_E001_E001;
    const EDGE_POS_TAG: u128 = 0xE002_E002_E002_E002;
    const EDGE_CHILD_TAG: u128 = 0xE003_E003_E003_E003;

    let mut edge_count = 0usize;
    let mut hash = match end_state {
        Some(state) => mix_tagged(
            0x51A7_E000_0000_0001,
            0xF070_F070_F070_F070,
            future_group_hashes[state],
        ),
        None => mix_u128(DEAD_NODE_TAG),
    };

    for (gid, &target_pos) in positions.iter().enumerate() {
        if target_pos < 0 {
            continue;
        }
        if !skip_groups.is_empty() && skip_groups[gid] {
            continue;
        }
        edge_count += 1;
        let target_pos = target_pos as usize;
        let child_hash = if target_pos >= token_len {
            ACCEPT_SINK_HASH
        } else {
            suffix_hashes[target_pos]
        };
        hash = mix_tagged(hash, EDGE_GID_TAG, gid as u128);
        hash = mix_tagged(hash, EDGE_POS_TAG, target_pos as u128);
        hash = mix_tagged(hash, EDGE_CHILD_TAG, child_hash);
    }

    mix_tagged(hash, EDGE_COUNT_TAG, edge_count as u128)
}

fn build_strided_batches(total_tokens: usize, target_batch_size: usize) -> Vec<Vec<usize>> {
    if total_tokens == 0 {
        return Vec::new();
    }

    let num_batches = total_tokens.div_ceil(target_batch_size.max(1));
    let mut batches = Vec::with_capacity(num_batches);
    for offset in 0..num_batches {
        let mut batch = Vec::with_capacity(total_tokens.div_ceil(num_batches));
        let mut idx = offset;
        while idx < total_tokens {
            batch.push(idx);
            idx += num_batches;
        }
        batches.push(batch);
    }
    batches
}

fn build_start_state_suffix_hashes(
    token: &[u8],
    tokenizer_start: usize,
    dfa_transitions: &[u32],
    dfa_finalizers: &[Vec<usize>],
    dfa_future_groups: &[Vec<usize>],
    future_group_hashes: &[u128],
    num_groups: usize,
    skip_groups: &[bool],
) -> Vec<u128> {
    let len = token.len();
    let mut suffix_hashes = vec![0u128; len];
    let mut positions = vec![-1i32; num_groups];

    for pos in (0..len).rev() {
        positions.fill(-1);

        let mut current = tokenizer_start;
        let mut done = dfa_future_groups[current].is_empty();

        for (offset, &byte) in token[pos..].iter().enumerate() {
            if done {
                break;
            }
            let next = dfa_transitions[current * 256 + byte as usize];
            if next == u32::MAX {
                done = true;
                break;
            }
            current = next as usize;
            let absolute_pos = (pos + offset + 1) as i32;
            for &gid in &dfa_finalizers[current] {
                if gid < num_groups {
                    positions[gid] = absolute_pos;
                }
            }
            if dfa_future_groups[current].is_empty() {
                done = true;
            }
        }

        let end_state = (!done).then_some(current);
        suffix_hashes[pos] = hash_trellis_node_from_positions(
            end_state,
            &positions,
            len,
            &suffix_hashes,
            future_group_hashes,
            skip_groups,
        );
    }

    suffix_hashes
}

/// Find state equivalence classes for a tokenizer.
pub fn find_state_equivalence_classes<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    find_state_equivalence_classes_ex(tokenizer, tokens, states, &[], None, None)
}

/// Find state equivalence classes with optional disallowed-follows filtering
/// and batch limit.
///
/// `skip_groups`: groups that are universally disallowed and can be ignored
///                in the trellis hash.
/// `max_batches`: if `Some(n)`, stop after processing `n` token batches
///                (useful for coarse pre-vocab reduction).
/// `batch_size`: override the default 5000-token batch size.
pub fn find_state_equivalence_classes_ex<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    skip_groups: &[bool],
    max_batches: Option<usize>,
    batch_size: Option<usize>,
) -> Vec<usize> {
    let profile_equivalence = profile_equivalence_enabled();

    if states.is_empty() {
        return Vec::new();
    }

    let refinement_started_at = std::time::Instant::now();
    let mapping = find_state_equivalence_classes_token_based(tokenizer, tokens, states, skip_groups, max_batches, batch_size);
    let refinement_time = refinement_started_at.elapsed();

    if profile_equivalence {
        let final_classes = count_classes(&mapping);
        eprintln!(
            "[glrmask/profile][state_equiv] token_refine_ms={:.3} states={}→{} ({:.2}x) skip_groups={} max_batches={:?}",
            refinement_time.as_secs_f64() * 1000.0,
            states.len(),
            final_classes,
            states.len() as f64 / final_classes.max(1) as f64,
            skip_groups.iter().filter(|&&b| b).count(),
            max_batches,
        );
    }

    mapping
}

fn find_state_equivalence_classes_token_based<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    skip_groups: &[bool],
    max_batches: Option<usize>,
    custom_batch_size: Option<usize>,
) -> Vec<usize> {
    use std::collections::{hash_map::Entry, HashMap};

    let dfa = tokenizer.dfa();

    const NONE_STATE: u32 = u32::MAX;
    // Use transitions directly from FlatDfa (shared via Arc, no redundant copy).
    let dfa_transitions: &[u32] = &dfa.transitions;

    let dfa_finalizers: Vec<Vec<usize>> = dfa
        .states
        .iter()
        .map(|state| state.finalizers.iter().copied().collect())
        .collect();
    let dfa_future_groups: Vec<Vec<usize>> = dfa
        .states
        .iter()
        .map(|state| state.possible_future_group_ids.iter().copied().collect())
        .collect();
    let future_group_hashes: Vec<u128> = dfa_future_groups
        .iter()
        .map(|future_groups| hash_future_groups(future_groups))
        .collect();

    let mut max_gid: Option<usize> = None;
    for finals in &dfa_finalizers {
        if let Some(m) = finals.iter().max() {
            max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
        }
    }
    let num_groups = max_gid.map(|m| m + 1).unwrap_or(0);
    let mut sorted_indices: Vec<usize> = (0..tokens.len()).collect();
    sorted_indices.par_sort_unstable_by(|&a, &b| tokens[a].as_ref().cmp(tokens[b].as_ref()));

    let mut sorted_tokens: Vec<&[u8]> = Vec::with_capacity(tokens.len());
    let mut sorted_weights: Vec<u128> = Vec::with_capacity(tokens.len());
    for &idx in &sorted_indices {
        sorted_tokens.push(tokens[idx].as_ref());
        sorted_weights.push(mix_u128((idx + 1) as u128));
    }

    let total_tokens = sorted_tokens.len();

    let tokenizer_start = tokenizer.initial_state_id();
    let suffix_hashes_by_token: Vec<Vec<u128>> = sorted_tokens
        .par_iter()
        .map(|token| {
            build_start_state_suffix_hashes(
                token,
                tokenizer_start,
                dfa_transitions,
                &dfa_finalizers,
                &dfa_future_groups,
                &future_group_hashes,
                num_groups,
                skip_groups,
            )
        })
        .collect();

    let common_prefix_len = |a: &[u8], b: &[u8]| -> usize {
        let len = a.len().min(b.len());
        let mut i = 0usize;
        while i < len && a[i] == b[i] {
            i += 1;
        }
        i
    };

    let early_stop = std::env::var("STATE_EQUIV_EARLY_STOP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let batch_size = custom_batch_size.unwrap_or(5000);
    let batches = build_strided_batches(total_tokens, batch_size);
    let dead_positions = vec![-1i32; num_groups];
    let fully_dead_token_hash = hash_trellis_node_from_positions(
        None,
        &dead_positions,
        0,
        &[],
        &future_group_hashes,
        skip_groups,
    );

    let mut group_ids: Vec<usize> = vec![0usize; states.len()];
    let mut group_sizes: Vec<usize> = vec![states.len()];
    let mut active_indices: Vec<usize> = (0..states.len()).collect();
    let mut touched_group_flags: Vec<bool> = vec![false; group_sizes.len()];
    let mut reused_group_flags: Vec<bool> = vec![false; group_sizes.len()];
    let mut prev_groups = 1usize;
    let mut stable_batches = 0usize;
    let mut tokens_tested = 0usize;
    let mut batches_processed = 0usize;
    for batch_indices in &batches {
        if active_indices.is_empty() {
            break;
        }
        if let Some(max) = max_batches {
            if batches_processed >= max {
                break;
            }
        }

        let batch_len = batch_indices.len();
        if batch_len == 0 {
            continue;
        }
        tokens_tested += batch_len;

        let mut batch_tokens: Vec<&[u8]> = Vec::with_capacity(batch_len);
        let mut batch_lcp_with_prev = Vec::with_capacity(batch_len);
        let mut batch_weight_prefix = vec![0u128; batch_len + 1];
        let mut prev_token: Option<&[u8]> = None;

        for (local_idx, &token_idx) in batch_indices.iter().enumerate() {
            let token = sorted_tokens[token_idx];
            let lcp = prev_token.map_or(0, |prev| common_prefix_len(prev, token));
            batch_tokens.push(token);
            batch_lcp_with_prev.push(lcp);
            batch_weight_prefix[local_idx + 1] =
                batch_weight_prefix[local_idx].wrapping_add(sorted_weights[token_idx]);
            prev_token = Some(token);
        }

        let batch_empty_end = batch_tokens
            .iter()
            .take_while(|token| token.is_empty())
            .count();
        let batch_empty_range = (0usize, batch_empty_end);

        let mut batch_first_byte_ranges = [(0usize, 0usize); 256];
        let mut batch_nonempty_first_bytes: Vec<usize> = Vec::new();
        let mut batch_pos = batch_empty_end;
        while batch_pos < batch_len {
            let byte = batch_tokens[batch_pos][0] as usize;
            let start = batch_pos;
            batch_pos += 1;
            while batch_pos < batch_len
                && !batch_tokens[batch_pos].is_empty()
                && batch_tokens[batch_pos][0] as usize == byte
            {
                batch_pos += 1;
            }
            batch_first_byte_ranges[byte] = (start, batch_pos);
            batch_nonempty_first_bytes.push(byte);
        }

        let mut batch_hashes: Vec<(usize, u128)> = active_indices
            .par_iter()
            .map_init(
                || {
                    (
                        Vec::<WalkFrame>::new(),
                        vec![-1; num_groups],
                        Vec::<(usize, i32)>::new(),
                    )
                },
                |scratch, &state_idx| {
                    let state = states[state_idx] as u32;
                    let mut hash_delta: u128 = 0;
                    let state_trans_base = (state as usize) * 256;

                    let mut live_ranges: Vec<(usize, usize)> = Vec::new();

                    if batch_empty_range.0 < batch_empty_range.1 {
                        live_ranges.push(batch_empty_range);
                    }

                    for &byte in &batch_nonempty_first_bytes {
                        let (range_start, range_end) = batch_first_byte_ranges[byte];
                        if range_start >= range_end {
                            continue;
                        }

                        if dfa_transitions[state_trans_base + byte] == NONE_STATE {
                            let weight_sum =
                                batch_weight_prefix[range_end].wrapping_sub(batch_weight_prefix[range_start]);
                            hash_delta = hash_delta
                                .wrapping_add(fully_dead_token_hash.wrapping_mul(weight_sum));
                        } else {
                            live_ranges.push((range_start, range_end));
                        }
                    }
                    let (walk_frames, positions, changes) = scratch;
                    let mut prev_groups_hash: u128;

                    for (range_start, range_end) in live_ranges {
                        if range_start >= range_end {
                            continue;
                        }

                        walk_frames.clear();
                        walk_frames.push(WalkFrame {
                            state,
                            dead_at_depth: None,
                            changes_len: 0,
                        });
                        if num_groups > 0 {
                            positions.fill(-1);
                        }
                        changes.clear();
                        prev_groups_hash = 0;

                        for token_idx in range_start..range_end {
                            let global_token_idx = batch_indices[token_idx];
                            let token = batch_tokens[token_idx];
                            let mut prefix_len = if token_idx == range_start {
                                0
                            } else {
                                batch_lcp_with_prev[token_idx]
                            };
                            let max_prefix = walk_frames.len().saturating_sub(1);
                            if prefix_len > max_prefix {
                                prefix_len = max_prefix;
                            }

                            if walk_frames.len() > prefix_len + 1 {
                                let target_mark = walk_frames[prefix_len].changes_len;
                                while changes.len() > target_mark {
                                    let (gid, prev_pos) = changes.pop().unwrap();
                                    let cur_pos = positions[gid];
                                    if cur_pos >= 0 {
                                        let cur_pos_u = cur_pos as u32;
                                        prev_groups_hash = prev_groups_hash.wrapping_sub(mix_u128(
                                            (gid as u128) | ((cur_pos_u as u128) << 32),
                                        ));
                                        if prev_pos < 0 {
                                            positions[gid] = -1;
                                        } else {
                                            let prev_pos_u = prev_pos as u32;
                                            prev_groups_hash = prev_groups_hash.wrapping_add(
                                                mix_u128(
                                                    (gid as u128)
                                                        | ((prev_pos_u as u128) << 32),
                                                ),
                                            );
                                            positions[gid] = prev_pos;
                                        }
                                    } else {
                                        positions[gid] = prev_pos;
                                    }
                                }

                                walk_frames.truncate(prefix_len + 1);
                            }

                            let mut dead_at_depth = walk_frames[prefix_len].dead_at_depth;

                            if dead_at_depth.is_none() {
                                let mut current = walk_frames.last().unwrap().state;
                                for (offset, &byte) in token[prefix_len..].iter().enumerate() {
                                    if current == NONE_STATE {
                                        dead_at_depth = Some(prefix_len + offset);
                                        break;
                                    }
                                    let next = dfa_transitions[current as usize * 256 + byte as usize];
                                    if next == NONE_STATE {
                                        dead_at_depth = Some(prefix_len + offset + 1);
                                        walk_frames.push(WalkFrame {
                                            state: NONE_STATE,
                                            dead_at_depth,
                                            changes_len: changes.len(),
                                        });
                                        break;
                                    }
                                    current = next;
                                    let position = prefix_len + offset + 1;

                                    if num_groups > 0 {
                                        for &gid in &dfa_finalizers[current as usize] {
                                            if gid >= num_groups {
                                                continue;
                                            }
                                            if !skip_groups.is_empty() && skip_groups[gid] {
                                                continue;
                                            }
                                            let pos_i32 = position as i32;
                                            let prev = positions[gid];
                                            if prev != pos_i32 {
                                                if prev < 0 {
                                                    prev_groups_hash = prev_groups_hash
                                                        .wrapping_add(mix_u128(
                                                            (gid as u128)
                                                                | ((position as u128) << 32),
                                                        ));
                                                    changes.push((gid, -1));
                                                } else {
                                                    prev_groups_hash = prev_groups_hash
                                                        .wrapping_sub(mix_u128(
                                                            (gid as u128) | ((prev as u128) << 32),
                                                        ));
                                                    prev_groups_hash = prev_groups_hash
                                                        .wrapping_add(mix_u128(
                                                            (gid as u128)
                                                                | ((position as u128) << 32),
                                                        ));
                                                    changes.push((gid, prev));
                                                }
                                                positions[gid] = pos_i32;
                                            }
                                        }
                                    }

                                    walk_frames.push(WalkFrame {
                                        state: current,
                                        dead_at_depth,
                                        changes_len: changes.len(),
                                    });
                                }
                            }

                            let token_hash = if dead_at_depth.is_some() {
                                hash_trellis_node_from_positions(
                                    None,
                                    positions,
                                    token.len(),
                                    &suffix_hashes_by_token[global_token_idx],
                                    &future_group_hashes,
                                    skip_groups,
                                )
                            } else {
                                let current = walk_frames.last().unwrap().state;
                                hash_trellis_node_from_positions(
                                    Some(current as usize),
                                    positions,
                                    token.len(),
                                    &suffix_hashes_by_token[global_token_idx],
                                    &future_group_hashes,
                                    skip_groups,
                                )
                            };
                            hash_delta = hash_delta.wrapping_add(
                                token_hash.wrapping_mul(sorted_weights[global_token_idx]),
                            );
                        }
                    }

                    (state_idx, hash_delta)
                },
            )
            .collect();

        batches_processed += 1;
        let previous_active_indices = std::mem::take(&mut active_indices);
        let all_active = previous_active_indices.len() == states.len();

        if all_active {
            let mut key_to_group: HashMap<(usize, u128), usize> =
                HashMap::with_capacity(states.len());
            group_sizes.clear();

            for (state_idx, hash) in batch_hashes.drain(..) {
                let key = (group_ids[state_idx], hash);
                let gid = *key_to_group.entry(key).or_insert_with(|| {
                    let id = group_sizes.len();
                    group_sizes.push(0);
                    id
                });
                group_ids[state_idx] = gid;
                group_sizes[gid] += 1;
            }

            touched_group_flags.clear();
            touched_group_flags.resize(group_sizes.len(), false);
            reused_group_flags.clear();
            reused_group_flags.resize(group_sizes.len(), false);
        } else {
            let mut key_to_group: HashMap<(usize, u128), usize> =
                HashMap::with_capacity(previous_active_indices.len());
            let mut touched_groups: Vec<usize> = Vec::new();

            for &state_idx in &previous_active_indices {
                let gid = group_ids[state_idx];
                if !touched_group_flags[gid] {
                    touched_group_flags[gid] = true;
                    reused_group_flags[gid] = false;
                    touched_groups.push(gid);
                    group_sizes[gid] = 0;
                }
            }

            for (state_idx, hash) in batch_hashes.drain(..) {
                let old_gid = group_ids[state_idx];
                let key = (old_gid, hash);
                let gid = match key_to_group.entry(key) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let gid = if !reused_group_flags[old_gid] {
                            reused_group_flags[old_gid] = true;
                            old_gid
                        } else {
                            let new_gid = group_sizes.len();
                            group_sizes.push(0);
                            touched_group_flags.push(false);
                            reused_group_flags.push(false);
                            new_gid
                        };
                        *entry.insert(gid)
                    }
                };
                group_ids[state_idx] = gid;
                group_sizes[gid] += 1;
            }

            for gid in touched_groups {
                touched_group_flags[gid] = false;
                reused_group_flags[gid] = false;
            }
        }

        let num_groups = group_sizes.len();
        active_indices.reserve(previous_active_indices.len());
        for state_idx in previous_active_indices {
            if group_sizes[group_ids[state_idx]] > 1 {
                active_indices.push(state_idx);
            }
        }

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
    }

    let num_groups = group_ids.iter().copied().max().map(|v| v + 1).unwrap_or(0);
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

/// Convert a state-to-representative mapping to `StateEquivalenceResult` format.
pub fn mapping_to_equivalence_classes(
    states: &[usize],
    mapping: &[usize],
) -> StateEquivalenceResult {
    let mut rep_to_class: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();

    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }

    rep_to_class.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::{
        find_state_equivalence_classes,
        find_state_equivalence_classes_token_based,
    };
    use crate::automata::lexer::ast::{bytes, choice, repeat, seq};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use super::super::super::compat::TokenizerView;

    #[test]
    fn full_refinement_matches_direct_token_analysis() {
        let exprs = vec![
            seq(vec![bytes(b"ab"), repeat(choice(vec![bytes(b"x"), bytes(b"y")]), 0, Some(3))]),
            seq(vec![bytes(b"ac"), repeat(bytes(b"z"), 0, Some(2))]),
        ];
        let tokenizer = build_tokenizer_from_exprs(&exprs);
        let tokenizer_view = TokenizerView::new(&tokenizer);
        let tokens: Vec<Vec<u8>> = vec![
            b"ab".to_vec(),
            b"abx".to_vec(),
            b"abyy".to_vec(),
            b"ac".to_vec(),
            b"aczz".to_vec(),
            b"zzz".to_vec(),
        ];
        let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

        let direct = find_state_equivalence_classes_token_based(&tokenizer_view, &tokens, &states, &[], None, None);
        let actual = find_state_equivalence_classes(&tokenizer_view, &tokens, &states);

        assert_eq!(
            actual, direct,
            "full refinement should be determined solely by direct token-based analysis"
        );
    }
}
