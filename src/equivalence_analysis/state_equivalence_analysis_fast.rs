//! State Equivalence Analysis
//!
//! Determines which tokenizer states behave identically for all tokens in a vocabulary.
//! States that are equivalent can be merged, reducing the workload for subsequent
//! vocab equivalence analysis.
//!
//! The algorithm uses a two-stage pipeline:
//! 1. k-step inductive hashing to collapse obviously equivalent states.
//! 2. Full token-based analysis on the reduced representative set.
//!
//! This avoids scanning the full vocabulary for every state and collapses long
//! bounded-repeat chains efficiently.

use std::collections::BTreeSet;
use rayon::prelude::*;
use crate::dfa_u8::Tokenizer;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

// -----------------------------------------------------------------------------
// Hashing Utilities (128-bit)
// -----------------------------------------------------------------------------

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[inline(always)]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

// -----------------------------------------------------------------------------
// State Equivalence Analysis (k-step inductive hashing)
// -----------------------------------------------------------------------------

#[inline(always)]
fn hash_sorted_set(values: &[usize], tag: u64) -> u64 {
    let mut h = mix_u64((values.len() as u64) ^ tag);
    for &v in values {
        h = h.wrapping_add(mix_u64((v as u64) ^ tag.rotate_left(17)));
    }
    h
}

#[inline(always)]
fn hash_state_label(finalizers: &[usize], possible_futures: &[usize]) -> u64 {
    const FINALIZER_TAG: u64 = 0xF11A_F11A_F11A_F11A;
    const FUTURE_TAG: u64 = 0xF0C7_F0C7_F0C7_F0C7;
    let finalizer_hash = hash_sorted_set(finalizers, FINALIZER_TAG);
    let future_hash = hash_sorted_set(possible_futures, FUTURE_TAG);
    mix_u64(finalizer_hash.wrapping_add(future_hash))
}

/// Find state equivalence classes using k-step inductive hashing.
///
/// # Proof (k-equivalence implies vocab-equivalence)
///
/// Let the DFA be D = (Q, Sigma, delta), with finalizers F(q) subset G and possible
/// futures P(q) subset G for each state q. For a state q and a string w in Sigma*,
/// define the run rho(q, w) = q_0, q_1, ..., q_|w| with q_0 = q and
/// q_{i+1} = delta(q_i, w_{i+1}) when the transition exists. If a transition is
/// missing, the run enters a distinguished dead state BOT and stays there for all
/// remaining input. For each group g in G, define the set of match positions
/// Occ(q, w, g) = { i | 0 <= i <= |w| and g in F(q_i) }.
/// The greedy match position is max Occ(q, w, g) and the non-greedy match position
/// is min Occ(q, w, g) (when the set is non-empty). The end-state semantic identity
/// for w is P(q_|w|) (or P(BOT) for dead).
///
/// Define a labeled unfolding hash by depth. Let the state label be
/// L(q) = (F(q), P(q)) and let L(BOT) be a unique dead label. Define
/// hash_0(q) = H(L(q)) and for d >= 1,
/// hash_d(q) = H(L(q), { (b, hash_{d-1}(delta(q,b))) : b in Sigma }),
/// where delta(q,b)=BOT if the transition is missing. Here H is a fixed
/// collision-resistant mixing function (128-bit). Two states are k-equivalent iff
/// hash_k is equal.
///
/// Lemma (Depth-d behavioral equivalence). If hash_d(q1)=hash_d(q2), then for every
/// string w with |w| <= d, the runs rho(q1,w) and rho(q2,w) visit states with
/// identical labels at every position, and their end-state possible futures are
/// identical.
///
/// Proof. By induction on d.
/// - Base d=0: hash_0 equality implies L(q1)=L(q2), so the empty string has identical
///   finalizers and identical P(q).
/// - Inductive step: assume the claim for d-1. Equality of hash_d implies equal root
///   labels and equal mapping from each byte b to the child hash hash_{d-1}(delta(q,b)).
///   Thus for any w = b w', the next states after b are (d-1)-equivalent, so by
///   induction their suffix runs match label-by-label and have identical end-state
///   futures. Prefix labels also match, so the full run matches for all positions.
/// QED.
///
/// Corollary. For every w with |w| <= d, all occurrences of each group g are at the
/// same positions in both runs, hence greedy (max) and non-greedy (min) choices are
/// identical, and P(q_|w|) is identical. Therefore k-equivalence implies identical
/// behavior for all strings of length <= k.
///
/// Since every vocabulary token has length <= k by construction, hash_k equality
/// implies vocabulary-state-equivalence. Hash collisions are possible but extremely
/// unlikely; the algorithm is a safe refinement that may over-split but will not
/// under-split absent collisions.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `states` - List of state IDs to analyze
/// * `k` - Maximum token length (bytes)
///
/// # Returns
/// A vector where `result[i]` is the representative state for `states[i]`.
/// States with the same representative are equivalent under k-equivalence.
pub fn find_state_equivalence_classes_kstep(
    regex: &Tokenizer,
    states: &[usize],
    k: usize,
) -> Vec<usize> {
    use std::collections::HashMap;

    if states.is_empty() {
        return Vec::new();
    }

    let instant = std::time::Instant::now();
    let dfa = regex.dfa();

    // Precompute transition lists (sparse) for each state.
    let transitions: Vec<Vec<(u8, usize)>> = dfa
        .states
        .iter()
        .map(|state| state.transitions.iter().map(|(b, &t)| (b, t)).collect())
        .collect();

    let profile_equivalence = std::env::var("PROFILE_EQUIVALENCE").is_ok();
    let mut class_for_state: Vec<usize> = Vec::new();
    let mut class_reps: Vec<usize> = Vec::new();
    let mut use_trans_class_cache = false;

    {
        let mut trans_pattern_to_class: HashMap<Vec<(u8, usize)>, usize> = HashMap::new();
        class_for_state = vec![0usize; transitions.len()];

        for (idx, trans) in transitions.iter().enumerate() {
            let class_id = match trans_pattern_to_class.get(trans) {
                Some(&existing) => existing,
                None => {
                    let new_id = class_reps.len();
                    trans_pattern_to_class.insert(trans.clone(), new_id);
                    class_reps.push(idx);
                    new_id
                }
            };
            class_for_state[idx] = class_id;
        }

        let num_classes = class_reps.len();
        let num_states = transitions.len();
        let shared = num_states.saturating_sub(num_classes);
        if profile_equivalence {
            eprintln!(
                "DFA transition classes: {} classes for {} states ({} shared)",
                num_classes,
                num_states,
                shared
            );
        }

        if num_states > 0 && shared * 2 >= num_states {
            use_trans_class_cache = true;
        } else {
            class_for_state.clear();
            class_reps.clear();
        }
    }

    // Precompute label hashes for each state (finalizers + possible futures).
    let label_hashes: Vec<u64> = dfa.states
        .iter()
        .map(|state| {
            let finalizers: Vec<usize> = state.finalizers.iter().collect();
            let futures: Vec<usize> = state.possible_future_group_ids.iter().copied().collect();
            hash_state_label(&finalizers, &futures)
        })
        .collect();

    // Dead transition hash is a unique constant that cannot collide with real labels.
    let dead_hash = mix_u64(0xDEAD_BEEF_DEAD_BEEF);
    let mut dead_byte_mix: Vec<u64> = vec![0u64; 256];
    let mut dead_base_sum: u64 = 0;
    for b in 0u8..=255u8 {
        let contrib = mix_u64(dead_hash ^ (((b as u64) << 1) | 1));
        dead_byte_mix[b as usize] = contrib;
        dead_base_sum = dead_base_sum.wrapping_add(contrib);
    }

    // Initialize hashes for depth 0 (empty string).
    let mut buf_a: Vec<u64> = label_hashes
        .iter()
        .map(|&h| mix_u64(h ^ 0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut buf_b: Vec<u64> = vec![0u64; dfa.states.len()];

    // Iteratively refine hashes for depths 1..=k.
    for iter in 0..k {
        let (src, dst) = if iter % 2 == 0 {
            (&buf_a, &mut buf_b)
        } else {
            (&buf_b, &mut buf_a)
        };

        let compute_trans_sum = |idx: usize, prev_hashes: &[u64]| -> u64 {
            let mut trans_sum = dead_base_sum;
            for &(byte, target) in &transitions[idx] {
                let b_idx = byte as usize;
                trans_sum = trans_sum.wrapping_sub(dead_byte_mix[b_idx]);
                let next_hash = prev_hashes[target];
                let contrib = mix_u64(next_hash ^ (((byte as u64) << 1) | 1));
                trans_sum = trans_sum.wrapping_add(contrib);
            }
            trans_sum
        };

        if use_trans_class_cache {
            let trans_sum_per_class: Vec<u64> = (0..class_reps.len())
                .into_par_iter()
                .map(|class_id| compute_trans_sum(class_reps[class_id], src))
                .collect();

            dst.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let trans_sum = trans_sum_per_class[class_for_state[idx]];
                let mut h = mix_u64(label_hashes[idx] ^ 0xC0DE_C0DE_C0DE_C0DE);
                h = h.wrapping_add(mix_u64(trans_sum ^ 0xA5A5_A5A5_5A5A_5A5A));
                *out = h;
            });
        } else {
            dst.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let trans_sum = compute_trans_sum(idx, src);
                let mut h = mix_u64(label_hashes[idx] ^ 0xC0DE_C0DE_C0DE_C0DE);
                h = h.wrapping_add(mix_u64(trans_sum ^ 0xA5A5_A5A5_5A5A_5A5A));
                *out = h;
            });
        }
    }

    let hashes = if k % 2 == 0 { &buf_a } else { &buf_b };

    // Group analyzed states by hash_k and pick representatives.
    let mut hash_to_rep: HashMap<u64, usize> = HashMap::new();
    let mut mapping = vec![0usize; states.len()];
    for (i, &state_id) in states.iter().enumerate() {
        let h = hashes[state_id];
        let rep = *hash_to_rep.entry(h).or_insert(state_id);
        mapping[i] = rep;
    }

    let num_representatives: usize = mapping.iter().collect::<std::collections::HashSet<_>>().len();
    crate::debug!(
        3,
        "State equiv k-hash: depth {} reduced {} states to {} in {:?}.",
        k,
        states.len(),
        num_representatives,
        instant.elapsed()
    );

    mapping
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
pub fn find_state_equivalence_classes(
    regex: &Tokenizer,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    use std::collections::HashMap;

    if states.is_empty() {
        return Vec::new();
    }

    let k = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
    let pre_mapping = find_state_equivalence_classes_kstep(regex, states, k);

    let mut rep_set: BTreeSet<usize> = BTreeSet::new();
    for &rep in &pre_mapping {
        rep_set.insert(rep);
    }
    let reduced_states: Vec<usize> = rep_set.into_iter().collect();

    crate::debug!(
        4,
        "State equiv prefilter: {} -> {} reps (k={})",
        states.len(),
        reduced_states.len(),
        k
    );

    if reduced_states.len() == states.len() {
        return find_state_equivalence_classes_token_based(regex, tokens, states);
    }

    let reduced_mapping = find_state_equivalence_classes_token_based(regex, tokens, &reduced_states);
    let mut rep_to_final: HashMap<usize, usize> = HashMap::new();
    for (i, &rep_state) in reduced_states.iter().enumerate() {
        rep_to_final.insert(rep_state, reduced_mapping[i]);
    }

    let mut mapping = vec![0usize; states.len()];
    for (i, &pre_rep) in pre_mapping.iter().enumerate() {
        mapping[i] = rep_to_final[&pre_rep];
    }

    mapping
}

fn find_state_equivalence_classes_token_based(
    regex: &Tokenizer,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    use std::collections::HashMap;

    let instant = std::time::Instant::now();
    let dfa = regex.dfa();

    // Note: Token sampling (STATE_EQUIV_MAX_TOKENS) was tested but causes correctness issues.
    // Sampled state equivalence doesn't fully capture distinguishing states,
    // leading to incorrect vocab class merging. Keep this disabled.
    //
    // let max_tokens = std::env::var("STATE_EQUIV_MAX_TOKENS")
    //     .ok()
    //     .and_then(|s| s.parse::<usize>().ok())
    //     .unwrap_or(tokens.len());

    // Precompute packed transition tables and finalizers for cache efficiency
    const NONE_STATE: u32 = u32::MAX;
    let dfa_transitions: Vec<[u32; 256]> = dfa.states
        .iter()
        .map(|state| {
            let mut table = [NONE_STATE; 256];
            for (byte, &target) in state.transitions.iter() {
                table[byte as usize] = target as u32;
            }
            table
        })
        .collect();

    let dfa_finalizers: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.finalizers.iter().collect())
        .collect();

    // Count states with finalizers for optimization insight
    let states_with_finalizers = dfa_finalizers.iter().filter(|f| !f.is_empty()).count();
    crate::debug!(
        5,
        "DFA stats: {} states, {} with finalizers ({:.1}%)",
        dfa.states.len(),
        states_with_finalizers,
        100.0 * states_with_finalizers as f64 / dfa.states.len() as f64
    );

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

    // Get non-greedy finalizers for proper position tracking
    let non_greedy_finalizers = &dfa.non_greedy_finalizers;

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

    let early_stop = std::env::var("STATE_EQUIV_EARLY_STOP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let batch_size = 5000usize;
    let total_tokens = sorted_tokens.len();

    let mut group_ids: Vec<usize> = vec![0usize; states.len()];
    let mut next_group_ids: Vec<usize> = vec![0usize; states.len()];
    let mut prev_groups = 1usize;
    let mut stable_batches = 0usize;
    let mut tokens_tested = 0usize;
    let mut early_stop_triggered = false;

    let mut batch_start = 0usize;
    while batch_start < total_tokens {
        let batch_end = (batch_start + batch_size).min(total_tokens);

        let batch_hashes: Vec<u128> = (0..states.len())
            .into_par_iter()
            .map(|i| {
                let state = states[i] as u32;
                let mut hash_delta: u128 = 0;
                let mut state_stack: Vec<u32> = vec![state];

                for token_idx in batch_start..batch_end {
                    let token = sorted_tokens[token_idx];
                    let mut prefix_len = if token_idx == batch_start {
                        0
                    } else {
                        lcp_with_prev[token_idx]
                    };
                    let max_prefix = state_stack.len().saturating_sub(1);
                    if prefix_len > max_prefix {
                        prefix_len = max_prefix;
                    }
                    state_stack.truncate(prefix_len + 1);

                    let mut matches: std::collections::BTreeMap<usize, usize> =
                        std::collections::BTreeMap::new();
                    let mut dead_at_depth: Option<usize> = None;

                    for depth in 1..=prefix_len {
                        let current = state_stack[depth];
                        if current == NONE_STATE {
                            dead_at_depth = Some(depth);
                            break;
                        }
                        let position = depth;
                        for &gid in &dfa_finalizers[current as usize] {
                            if non_greedy_finalizers.contains(&gid) {
                                matches.entry(gid).or_insert(position);
                            } else {
                                matches.insert(gid, position);
                            }
                        }
                    }

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
                                current = NONE_STATE;
                                break;
                            }
                            current = next;
                            state_stack.push(current);
                            let position = prefix_len + offset + 1;
                            for &gid in &dfa_finalizers[current as usize] {
                                if non_greedy_finalizers.contains(&gid) {
                                    matches.entry(gid).or_insert(position);
                                } else {
                                    matches.insert(gid, position);
                                }
                            }
                        }
                    }

                    let (structure_hash, end_hash) = if let Some(dead_depth) = dead_at_depth {
                        (
                            mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD),
                            mix_u128(0xDEADBEEF_u128),
                        )
                    } else {
                        let mut sh = mix_u128(matches.len() as u128 | (1u128 << 48));
                        for (&gid, &pos) in &matches {
                            sh = sh.wrapping_add(mix_u128((gid as u128) | ((pos as u128) << 32)));
                        }
                        let current = *state_stack.last().unwrap();
                        (sh, end_state_hashes[current as usize])
                    };

                    let token_hash = end_hash.wrapping_add(structure_hash);
                    hash_delta = hash_delta.wrapping_add(
                        token_hash.wrapping_mul(sorted_weights[token_idx]),
                    );
                }

                hash_delta
            })
            .collect();

        let mut key_to_group: HashMap<(usize, u128), usize> = HashMap::new();
        let mut num_groups = 0usize;
        for i in 0..states.len() {
            let key = (group_ids[i], batch_hashes[i]);
            let entry = key_to_group.entry(key).or_insert_with(|| {
                let id = num_groups;
                num_groups += 1;
                id
            });
            next_group_ids[i] = *entry;
        }
        std::mem::swap(&mut group_ids, &mut next_group_ids);

        tokens_tested = batch_end;
        if early_stop && tokens_tested * 2 >= total_tokens {
            if num_groups == prev_groups {
                stable_batches += 1;
            } else {
                stable_batches = 0;
            }
            if stable_batches >= 2 {
                early_stop_triggered = true;
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

    let phase1_time = instant.elapsed();
    let singleton_groups = group_sizes.iter().filter(|&&n| n == 1).count();
    let ambiguous_states: usize = group_sizes.iter().filter(|&&n| n > 1).sum();

    crate::debug!(
        4,
        "State equiv phase 1: {} groups ({} singletons, {} ambiguous) in {:?} ({} tokens)",
        num_groups,
        singleton_groups,
        ambiguous_states,
        phase1_time,
        tokens_tested
    );

    if early_stop_triggered {
        crate::debug!(
            4,
            "State equiv early stop enabled: processed {} of {} tokens",
            tokens_tested,
            total_tokens
        );
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

    let num_representatives = rep_for_group.iter().filter(|&&v| v != usize::MAX).count();

    crate::debug!(
        3,
        "State equivalence analysis took {:.2?}. Reduced {} states to {}.",
        instant.elapsed(),
        states.len(),
        num_representatives
    );

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
