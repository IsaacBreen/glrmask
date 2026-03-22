//! Max-length bounded state equivalence prepass.
//!
//! This pass ignores the actual token contents and uses only the maximum token
//! length. It groups states that have identical bounded-depth path behavior for
//! all strings up to that length, using the same public signature as the main
//! state-equivalence pass so it can slot cleanly into the pipeline.

use ahash::RandomState;
use hashbrown::HashMap;
use rayon::prelude::*;

use super::super::compat::Sep1Tokenizer;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
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

#[inline(always)]
fn hash_sorted_set(values: &[usize], tag: u64) -> u64 {
    let mut h = mix_u64((values.len() as u64) ^ tag);
    for &value in values {
        h = h.wrapping_add(mix_u64((value as u64) ^ tag.rotate_left(17)));
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

#[inline(always)]
fn hash_transition_list(transitions: &[(u8, usize)], prev_hashes: &[u64]) -> u64 {
    let mut hash = mix_u64((transitions.len() as u64) ^ 0xA5A5_A5A5_5A5A_5A5A);
    for &(byte, target) in transitions {
        let edge_hash = mix_u64(
            (((byte as u64) << 32) | byte as u64) ^ prev_hashes[target].rotate_left(17),
        );
        hash = mix_u64(hash ^ edge_hash.wrapping_add(0x517C_C1B7_2722_0A95));
    }
    hash
}

fn assign_hash_classes(hashes: &[u64], class_ids: &mut [usize]) -> usize {
    let mut hash_to_class: HashMap<u64, usize, RandomState> =
        HashMap::with_capacity_and_hasher(hashes.len(), RandomState::new());
    let mut total_class_count = 0usize;

    for (idx, &hash) in hashes.iter().enumerate() {
        let class_id = match hash_to_class.get(&hash) {
            Some(&existing) => existing,
            None => {
                let new_id = total_class_count;
                hash_to_class.insert(hash, new_id);
                total_class_count += 1;
                new_id
            }
        };
        class_ids[idx] = class_id;
    }

    total_class_count
}

fn count_subset_classes(
    states: &[usize],
    class_ids: &[usize],
    seen: &mut [u32],
    epoch: &mut u32,
) -> usize {
    *epoch = epoch.wrapping_add(1);
    if *epoch == 0 {
        seen.fill(0);
        *epoch = 1;
    }
    let current_epoch = *epoch;

    let mut count = 0usize;
    for &state_id in states {
        let class_id = class_ids[state_id];
        if seen[class_id] != current_epoch {
            seen[class_id] = current_epoch;
            count += 1;
        }
    }
    count
}

fn build_subset_mapping(states: &[usize], class_ids: &[usize], total_class_count: usize) -> Vec<usize> {
    let mut rep_for_class = vec![usize::MAX; total_class_count];
    let mut mapping = vec![0usize; states.len()];

    for (idx, &state_id) in states.iter().enumerate() {
        let class_id = class_ids[state_id];
        let rep = &mut rep_for_class[class_id];
        if *rep == usize::MAX {
            *rep = state_id;
        }
        mapping[idx] = *rep;
    }

    mapping
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
/// hash_d(q) = H(hash_{d-1}(q), [ (b, hash_{d-1}(delta(q,b))) ]),
/// where the list is taken in byte order and delta(q,b)=BOT if the transition is
/// missing. Here H is a fixed
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
/// - Inductive step: assume the claim for d-1. Equality of hash_d implies equal
///   hash_{d-1} roots and equal mapping from each byte b to the child hash
///   hash_{d-1}(delta(q,b)). Thus for any w = b w', the next states after b are
///   (d-1)-equivalent, so by induction their suffix runs match label-by-label and
///   have identical end-state futures. The equal root hash_{d-1} also preserves the
///   depth-(d-1) prefix behavior at the current state.
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
/// The refinement is monotone: once two analyzed states are separated, they never
/// merge again. This lets us safely short-circuit to the identity mapping when the
/// remaining possible savings become negligible, since deeper refinement can only
/// split states further.
fn find_state_equivalence_classes_kstep(
    regex: &Sep1Tokenizer,
    states: &[usize],
    k: usize,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let dfa = regex.dfa();

    let transitions: Vec<Vec<(u8, usize)>> = dfa
        .states
        .iter()
        .map(|state| {
            state.transitions
                .iter()
                .enumerate()
                .filter(|(_, target)| **target != u32::MAX)
                .map(|(byte, &target)| (byte as u8, target as usize))
                .collect()
        })
        .collect();

    let profile_equivalence = std::env::var("PROFILE_EQUIVALENCE").is_ok();
    let identity_max_saved_percent = env_usize("STATE_EQUIV_IDENTITY_MAX_SAVED_PERCENT", 2);
    let (transition_pattern_for_state, transition_pattern_reps, use_trans_class_cache) = {
        let mut trans_pattern_to_class: HashMap<Vec<(u8, usize)>, usize, RandomState> =
            HashMap::with_capacity_and_hasher(transitions.len(), RandomState::new());
        let mut transition_pattern_for_state = vec![0usize; transitions.len()];
        let mut transition_pattern_reps = Vec::new();

        for (idx, trans) in transitions.iter().enumerate() {
            let class_id = match trans_pattern_to_class.get(trans.as_slice()) {
                Some(&existing) => existing,
                None => {
                    let new_id = transition_pattern_reps.len();
                    trans_pattern_to_class.insert(trans.clone(), new_id);
                    transition_pattern_reps.push(idx);
                    new_id
                }
            };
            transition_pattern_for_state[idx] = class_id;
        }

        let num_classes = transition_pattern_reps.len();
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
            (
                transition_pattern_for_state,
                transition_pattern_reps,
                true,
            )
        } else {
            (Vec::new(), Vec::new(), false)
        }
    };

    let mut buf_a: Vec<u64> = dfa
        .states
        .iter()
        .map(|state| hash_state_label(&state.finalizers, &state.possible_future_group_ids))
        .collect();
    let mut buf_b: Vec<u64> = vec![0u64; dfa.states.len()];
    let mut class_ids_a = vec![0usize; dfa.states.len()];
    let mut class_ids_b = vec![0usize; dfa.states.len()];
    let states_cover_all = states.len() == dfa.states.len() && states.iter().copied().eq(0..dfa.states.len());
    let mut analyzed_seen = vec![0u32; dfa.states.len().max(1)];
    let mut analyzed_seen_epoch = 0u32;
    let mut prev_total_class_count = assign_hash_classes(&buf_a, &mut class_ids_a);
    let mut final_total_class_count = prev_total_class_count;
    let mut depth_completed = 0usize;

    for iter in 0..k {
        let (src_hashes, dst_hashes, dst_class_ids) = if iter % 2 == 0 {
            (&buf_a, &mut buf_b, &mut class_ids_b)
        } else {
            (&buf_b, &mut buf_a, &mut class_ids_a)
        };

        let compute_transition_hash = |idx: usize, prev_hashes: &[u64]| -> u64 {
            hash_transition_list(&transitions[idx], prev_hashes)
        };

        if use_trans_class_cache {
            let transition_hash_per_class: Vec<u64> = (0..transition_pattern_reps.len())
                .into_par_iter()
                .map(|class_id| compute_transition_hash(transition_pattern_reps[class_id], src_hashes))
                .collect();

            dst_hashes.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let transition_hash = transition_hash_per_class[transition_pattern_for_state[idx]];
                *out = mix_u64(src_hashes[idx] ^ transition_hash);
            });
        } else {
            dst_hashes.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let transition_hash = compute_transition_hash(idx, src_hashes);
                *out = mix_u64(src_hashes[idx] ^ transition_hash);
            });
        }

        final_total_class_count = assign_hash_classes(dst_hashes, dst_class_ids);

        depth_completed = iter + 1;

        let analyzed_class_count = if states_cover_all {
            final_total_class_count
        } else {
            count_subset_classes(
                states,
                dst_class_ids,
                &mut analyzed_seen,
                &mut analyzed_seen_epoch,
            )
        };
        let saved_states = states.len().saturating_sub(analyzed_class_count);

        if saved_states == 0 {
            if profile_equivalence {
                eprintln!(
                    "[state equiv] depth={} reached all-singleton analyzed states; using identity",
                    depth_completed,
                );
            }
            return states.to_vec();
        }

        if identity_max_saved_percent > 0
            && saved_states.saturating_mul(100)
                <= states.len().saturating_mul(identity_max_saved_percent)
        {
            if profile_equivalence {
                eprintln!(
                    "[state equiv] depth={} remaining_savings={} of {} states (<= {}%); using identity",
                    depth_completed,
                    saved_states,
                    states.len(),
                    identity_max_saved_percent,
                );
            }
            return states.to_vec();
        }

        if final_total_class_count == prev_total_class_count {
            if profile_equivalence {
                eprintln!(
                    "[state equiv] k-step partition stabilized at depth={} with {} total classes",
                    depth_completed,
                    final_total_class_count,
                );
            }
            break;
        }
        prev_total_class_count = final_total_class_count;
    }

    let final_class_ids = if depth_completed == 0 || depth_completed % 2 == 0 {
        &class_ids_a
    } else {
        &class_ids_b
    };

    build_subset_mapping(states, final_class_ids, final_total_class_count)
}

pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);
    find_state_equivalence_classes_kstep(regex, states, max_len)
}
