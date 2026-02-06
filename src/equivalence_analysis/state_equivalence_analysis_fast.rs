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

// -----------------------------------------------------------------------------
// State Equivalence Analysis (k-step inductive hashing)
// -----------------------------------------------------------------------------

#[inline(always)]
fn hash_sorted_set(values: &[usize], tag: u128) -> u128 {
    let mut h = mix_u128((values.len() as u128) ^ tag);
    for &v in values {
        h = h.wrapping_add(mix_u128((v as u128) ^ tag.rotate_left(17)));
    }
    h
}

#[inline(always)]
fn hash_state_label(finalizers: &[usize], possible_futures: &[usize]) -> u128 {
    const FINALIZER_TAG: u128 = 0xF11A_F11A_F11A_F11A;
    const FUTURE_TAG: u128 = 0xF0C7_F0C7_F0C7_F0C7;
    let finalizer_hash = hash_sorted_set(finalizers, FINALIZER_TAG);
    let future_hash = hash_sorted_set(possible_futures, FUTURE_TAG);
    mix_u128(finalizer_hash.wrapping_add(future_hash))
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
    let transitions: Vec<Vec<(u8, usize)>> = dfa.states
        .iter()
        .map(|state| state.transitions.iter().map(|(b, &t)| (b, t)).collect())
        .collect();

    // Precompute label hashes for each state (finalizers + possible futures).
    let label_hashes: Vec<u128> = dfa.states
        .iter()
        .map(|state| {
            let finalizers: Vec<usize> = state.finalizers.iter().collect();
            let futures: Vec<usize> = state.possible_future_group_ids.iter().copied().collect();
            hash_state_label(&finalizers, &futures)
        })
        .collect();

    // Dead transition hash is a unique constant that cannot collide with real labels.
    let dead_hash = mix_u128(0xDEAD_BEEF_DEAD_BEEF);
    let mut dead_byte_mix: Vec<u128> = vec![0u128; 256];
    let mut dead_base_sum: u128 = 0;
    for b in 0u8..=255u8 {
        let contrib = mix_u128(dead_hash ^ (((b as u128) << 1) | 1));
        dead_byte_mix[b as usize] = contrib;
        dead_base_sum = dead_base_sum.wrapping_add(contrib);
    }

    // Initialize hashes for depth 0 (empty string).
    let mut hashes: Vec<u128> = label_hashes
        .iter()
        .map(|&h| mix_u128(h ^ 0x9E37_79B9_7F4A_7C15))
        .collect();

    // Iteratively refine hashes for depths 1..=k.
    for _ in 0..k {
        let prev_hashes = &hashes;
        hashes = (0..dfa.states.len())
            .into_par_iter()
            .map(|idx| {
                let mut trans_sum = dead_base_sum;
                for &(byte, target) in &transitions[idx] {
                    let b_idx = byte as usize;
                    trans_sum = trans_sum.wrapping_sub(dead_byte_mix[b_idx]);
                    let next_hash = prev_hashes[target];
                    let contrib = mix_u128(next_hash ^ (((byte as u128) << 1) | 1));
                    trans_sum = trans_sum.wrapping_add(contrib);
                }
                let mut h = mix_u128(label_hashes[idx] ^ 0xC0DE_C0DE_C0DE_C0DE);
                h = h.wrapping_add(mix_u128(trans_sum ^ 0xA5A5_A5A5_5A5A_5A5A));
                h
            })
            .collect();
    }

    // Group analyzed states by hash_k and pick representatives.
    let mut hash_to_rep: HashMap<u128, usize> = HashMap::new();
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

    // Initialize state hashes to zero (like reference implementation)
    let mut state_hashes: Vec<u128> = vec![0u128; states.len()];

    // Get non-greedy finalizers for proper position tracking
    let non_greedy_finalizers = &dfa.non_greedy_finalizers;

    // Process tokens in batches for memory efficiency, but process ALL tokens for ALL states
    // to ensure correct equivalence (no early singleton exit optimization)
    let batch_size = if states.len() > 10000 {
        25000.min(tokens.len())
    } else {
        10000.min(tokens.len())
    };
    let mut tokens_tested = 0usize;

    while tokens_tested < tokens.len() {
        // Prepare next batch of tokens
        let batch_end = (tokens_tested + batch_size).min(tokens.len());
        let batch_tokens: Vec<&Vec<u8>> = (tokens_tested..batch_end)
            .map(|i| &tokens[i])
            .collect();

        // Precompute token weights based on global token index (for consistent hashing)
        let batch_weights: Vec<u128> = (tokens_tested..batch_end)
            .map(|i| mix_u128((i + 1) as u128)) // Match reference: mix(token_index + 1)
            .collect();

        // Update hashes for ALL states
        let updates: Vec<(usize, u128)> = (0..states.len())
            .into_par_iter()
            .map(|i| {
                let state = states[i];
                let mut hash_delta: u128 = 0;

                for (batch_idx, token) in batch_tokens.iter().enumerate() {
                    let mut current = state as u32;
                    let mut dead_at_depth: Option<usize> = None;

                    // Track (group_id, position) with proper greedy/non-greedy semantics
                    let mut matches: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();

                    for (depth, &byte) in (*token).iter().enumerate() {
                        // Safety check - should never trigger since we break after setting NONE_STATE
                        if current == NONE_STATE {
                            dead_at_depth = Some(depth);
                            break;
                        }
                        let next = dfa_transitions[current as usize][byte as usize];
                        if next == NONE_STATE {
                            dead_at_depth = Some(depth + 1);
                            current = NONE_STATE;
                            break;
                        }
                        current = next;
                        let position = depth + 1;

                        // Record matches with proper greedy/non-greedy semantics
                        for &gid in &dfa_finalizers[current as usize] {
                            if non_greedy_finalizers.contains(&gid) {
                                // Non-greedy: keep first
                                matches.entry(gid).or_insert(position);
                            } else {
                                // Greedy: keep last
                                matches.insert(gid, position);
                            }
                        }
                    }

                    // Hash structure: dead position OR (matches + end state)
                    let structure_hash: u128;
                    let end_hash: u128;

                    if let Some(dead_depth) = dead_at_depth {
                        // Token leads to dead state - hash the dead depth
                        structure_hash = mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD);
                        end_hash = mix_u128(0xDEADBEEF_u128);
                    } else {
                        // Token is valid - hash the (group_id, position) pairs
                        let mut sh = mix_u128(matches.len() as u128 | (1u128 << 48));
                        for (&gid, &pos) in &matches {
                            sh = sh.wrapping_add(mix_u128((gid as u128) | ((pos as u128) << 32)));
                        }
                        structure_hash = sh;

                        // Hash end state possible_futures (precomputed)
                        end_hash = end_state_hashes[current as usize];
                    }

                    let token_hash = end_hash.wrapping_add(structure_hash);
                    hash_delta = hash_delta.wrapping_add(token_hash.wrapping_mul(batch_weights[batch_idx]));
                }

                (i, hash_delta)
            })
            .collect();

        // Apply updates
        for (i, delta) in updates {
            state_hashes[i] = state_hashes[i].wrapping_add(delta);
        }

        tokens_tested = batch_end;
    }

    // Group by final hash
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &hash) in state_hashes.iter().enumerate() {
        groups.entry(hash).or_default().push(i);
    }

    let phase1_time = instant.elapsed();
    let num_groups = groups.len();
    let singleton_groups = groups.values().filter(|g| g.len() == 1).count();
    let ambiguous_states: usize = groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();

    crate::debug!(
        4,
        "State equiv phase 1: {} groups ({} singletons, {} ambiguous) in {:?} ({} tokens)",
        num_groups,
        singleton_groups,
        ambiguous_states,
        phase1_time,
        tokens_tested
    );

    // If all groups are singletons, we're done (no states are equivalent)
    if ambiguous_states == 0 {
        // Convert from state index to state ID
        let mapping: Vec<usize> = states.to_vec();
        crate::debug!(
            3,
            "State equivalence analysis took {:.2?}. Reduced {} states to {} (all unique).",
            instant.elapsed(),
            states.len(),
            states.len()
        );
        return mapping;
    }

    // If we've tested ALL tokens in phase 1, the groups are already correct.
    // Phase 2 would just recompute the same result.
    // All non-singleton states have seen all tokens (singletons stopped early but are already unique).
    if tokens_tested >= tokens.len() {
        // Build mapping from phase 1 groups
        // Note: groups contains state indices (positions in `states`), not state IDs
        let mut mapping = vec![0usize; states.len()];
        for group in groups.values() {
            let rep_state_id = states[group[0]]; // Convert representative index to state ID
            for &idx in group {
                mapping[idx] = rep_state_id;
            }
        }
        let num_representatives: usize = mapping.iter().collect::<std::collections::HashSet<_>>().len();
        crate::debug!(
            3,
            "State equivalence analysis took {:.2?}. Reduced {} states to {} (phase 1 complete).",
            instant.elapsed(),
            states.len(),
            num_representatives
        );
        return mapping;
    }

    // PHASE 2: Full token analysis for ambiguous states
    // =========================================================================
    // Only needed when phase 1 didn't test all tokens (early exit due to all singletons).
    // Use full token analysis for correctness.

    // Collect all ambiguous state indices (positions in `states` array)
    let ambiguous_idx_list: Vec<usize> = groups
        .values()
        .filter(|g| g.len() > 1)
        .flat_map(|g| g.iter().copied())
        .collect();

    // Full analysis: use all tokens
    let phase2_tokens: Vec<&Vec<u8>> = tokens.iter().collect();
    let phase2_token_weights: Vec<u128> = (0..tokens.len())
        .map(|i| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
        .collect();

    crate::debug!(
        4,
        "State equiv phase 2: analyzing {} states with {} tokens (full)",
        ambiguous_states,
        phase2_tokens.len()
    );
    // Compute signatures for all ambiguous states in parallel
    // Returns (state_index, signature) pairs
    let ambiguous_signatures: Vec<(usize, u128)> = ambiguous_idx_list
        .par_iter()
        .map(|&idx| {
            let state = states[idx]; // Convert index to actual state ID
            let mut hash: u128 = 0;

            for (token_idx, token) in phase2_tokens.iter().enumerate() {
                let mut current = state as u32;
                let mut dead_at_depth: Option<usize> = None;

                // Track (group_id, position) with proper greedy/non-greedy semantics
                let mut matches: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();

                const NONE_STATE: u32 = u32::MAX;
                for (depth, &byte) in (*token).iter().enumerate() {
                    let next = dfa_transitions[current as usize][byte as usize];
                    if next == NONE_STATE {
                        dead_at_depth = Some(depth + 1);
                        current = NONE_STATE;
                        break;
                    }
                    current = next;
                    let position = depth + 1;

                    // Record matches with proper greedy/non-greedy semantics
                    for &gid in &dfa_finalizers[current as usize] {
                        if non_greedy_finalizers.contains(&gid) {
                            // Non-greedy: keep first
                            matches.entry(gid).or_insert(position);
                        } else {
                            // Greedy: keep last
                            matches.insert(gid, position);
                        }
                    }
                }

                // Hash structure: dead position OR (matches + end state)
                let structure_hash: u128;
                let end_hash: u128;

                if let Some(dead_depth) = dead_at_depth {
                    // Token leads to dead state - hash the dead depth
                    structure_hash = mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD);
                    end_hash = mix_u128(0xDEADBEEF_u128);
                } else {
                    // Token is valid - hash the (group_id, position) pairs
                    let mut sh = mix_u128(matches.len() as u128 | (1u128 << 48));
                    for (&gid, &pos) in &matches {
                        sh = sh.wrapping_add(mix_u128((gid as u128) | ((pos as u128) << 32)));
                    }
                    structure_hash = sh;

                    // Hash end state possible_futures (precomputed)
                    end_hash = end_state_hashes[current as usize];
                }

                let token_hash = end_hash.wrapping_add(structure_hash);
                hash = hash.wrapping_add(token_hash.wrapping_mul(phase2_token_weights[token_idx]));
            }

            (idx, hash) // Return index, not state ID
        })
        .collect();

    // Build index -> signature map
    let idx_to_sig: HashMap<usize, u128> = ambiguous_signatures.iter().copied().collect();

    // Build final mapping: mapping[index] = representative state ID
    let mut mapping = vec![0usize; states.len()];

    // Singleton groups: state maps to itself
    for group in groups.values() {
        if group.len() == 1 {
            let idx = group[0];
            mapping[idx] = states[idx]; // Map to own state ID
        }
    }

    // Ambiguous groups: refine by full signature
    for group in groups.values() {
        if group.len() > 1 {
            // Group by full signature within this group
            let mut sig_to_rep: HashMap<u128, usize> = HashMap::new();
            for &idx in group {
                let sig = idx_to_sig[&idx];
                let rep_idx = *sig_to_rep.entry(sig).or_insert(idx);
                mapping[idx] = states[rep_idx]; // Map to representative's state ID
            }
        }
    }

    let num_representatives: usize = mapping.iter().collect::<std::collections::HashSet<_>>().len();

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
