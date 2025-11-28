use crate::finite_automata::{Regex, GroupID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeMap;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// If true, performs a brute-force verification step after hashing.
/// This guarantees 100% correctness at the cost of performance.
const VERIFY_RESULTS: bool = false;

// -----------------------------------------------------------------------------
// Hashing Utilities (128-bit)
// -----------------------------------------------------------------------------

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    const C1: u128 = 0x9e3779b97f4a7c15_bf58476d1ce4e5b9;
    const C2: u128 = 0x94d049bb133111eb_ff51afd7ed558ccd;

    x ^= x >> 33;
    x = x.wrapping_mul(C1);
    x ^= x >> 33;
    x = x.wrapping_mul(C2);
    x ^= x >> 33;
    x
}

#[inline(always)]
fn get_init_weight(idx: usize) -> u128 {
    mix_u128((idx as u128) << 1 | 1) | 1
}

const KIND_DEAD_END: u8 = 1;
const KIND_MATCH: u8 = 2;
const KIND_TERM: u8 = 3;

/// Hash for get_mask equivalence (uses state_sig which represents future group possibilities)
#[inline(always)]
fn hash_outcome_mask(
    kind: u8,
    match_group: u32,
    match_pos: u32,
    remainder_sig: u64,
    state_sig: u64,
) -> u128 {
    let flags = kind as u32;
    let packed = ((match_pos as u128) << 96)
        | ((match_group as u128) << 64)
        | ((flags as u128) << 32);
    mix_u128(packed ^ (remainder_sig as u128) ^ (state_sig as u128))
}

/// Hash for commit equivalence (uses actual final_state ID)
#[inline(always)]
fn hash_outcome_commit(
    kind: u8,
    match_group: u32,
    match_pos: u32,
    remainder_sig: u64,
    final_state: u32,
) -> u128 {
    let flags = (kind as u32) | (final_state << 4);
    let packed = ((match_pos as u128) << 96)
        | ((match_group as u128) << 64)
        | ((flags as u128) << 32);
    mix_u128(packed ^ (remainder_sig as u128))
}

// -----------------------------------------------------------------------------
// Trie Definition
// -----------------------------------------------------------------------------

#[derive(Clone, Default)]
struct TrieNode {
    // Use BTreeMap to maintain sorted order - avoids sorting during linearize
    transitions: std::collections::BTreeMap<u8, u32>,
    terminal_string_idx: Option<u32>,
    range_start: u32,
    range_end: u32,
}

struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn new() -> Self {
        Trie { nodes: vec![TrieNode::default()] }
    }

    fn insert(&mut self, s: &[u8], original_idx: u32) {
        let mut node_idx = 0;
        for &b in s {
            if let Some(&child) = self.nodes[node_idx].transitions.get(&b) {
                node_idx = child as usize;
            } else {
                let new_node_idx = self.nodes.len() as u32;
                self.nodes.push(TrieNode::default());
                self.nodes[node_idx].transitions.insert(b, new_node_idx);
                node_idx = new_node_idx as usize;
            }
        }
        self.nodes[node_idx].terminal_string_idx = Some(original_idx);
    }

    fn linearize(&mut self) -> Vec<usize> {
        let mut mapping = Vec::new();
        self.dfs_linearize(0, &mut mapping);
        mapping
    }

    fn dfs_linearize(&mut self, node_idx: usize, mapping: &mut Vec<usize>) {
        let start = mapping.len() as u32;
        if let Some(orig_idx) = self.nodes[node_idx].terminal_string_idx {
            mapping.push(orig_idx as usize);
        }
        // BTreeMap already maintains sorted order, no need to sort
        let children: Vec<_> = self.nodes[node_idx].transitions.values().copied().collect();
        for child_idx in children {
            self.dfs_linearize(child_idx as usize, mapping);
        }
        let end = mapping.len() as u32;
        self.nodes[node_idx].range_start = start;
        self.nodes[node_idx].range_end = end;
    }
}

// -----------------------------------------------------------------------------
// Combined Result
// -----------------------------------------------------------------------------

/// Result of combined equivalence analysis
pub struct CombinedEquivalenceResult {
    /// Equivalence classes for get_mask (uses state signatures)
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Equivalence classes for commit (uses actual final states)
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

// -----------------------------------------------------------------------------
// Combined Equivalence Analysis
// -----------------------------------------------------------------------------

/// Computes both get_mask and commit equivalence classes in a single pass.
/// This is more efficient than calling each separately because:
/// 1. We only build the trie once
/// 2. We only run symbolic execution once  
/// 3. We only call execute_from_state_fast once per position (the expensive part)
pub fn find_equivalence_classes_combined(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> CombinedEquivalenceResult {
    crate::debug!(3, "Combined equivalence analysis for {} strings.", strings.len());
    let pb = create_pb(4);

    let t0 = std::time::Instant::now();
    pb.set_message("Precomputing signatures...");
    let state_signatures = compute_state_signatures(regex);
    crate::debug!(3, "  Equiv: State signatures: {:?}", t0.elapsed());
    
    // Precompute remainder hashes for BOTH mask and commit in one pass
    let t1 = std::time::Instant::now();
    let (remainder_hashes_mask, remainder_hashes_commit) = 
        precompute_remainder_hashes_combined(regex, strings, &state_signatures);
    crate::debug!(3, "  Equiv: Remainder hashes (parallel): {:?}", t1.elapsed());
    pb.inc(1);

    let t2 = std::time::Instant::now();
    pb.set_message("Building Trie...");
    let mut trie = Trie::new();
    for (i, s) in strings.iter().enumerate() {
        trie.insert(s, i as u32);
    }
    let t_insert = t2.elapsed();
    let linearized_mapping = trie.linearize();
    crate::debug!(3, "  Equiv: Trie build: {:?} (insert: {:?}, linearize: {:?})", t2.elapsed(), t_insert, t2.elapsed() - t_insert);
    pb.inc(1);

    let t3 = std::time::Instant::now();
    pb.set_message("Symbolic Execution...");
    let mut accumulators_mask = vec![0u128; strings.len()];
    let mut diffs_mask = vec![0u128; strings.len() + 1];
    let mut accumulators_commit = vec![0u128; strings.len()];
    let mut diffs_commit = vec![0u128; strings.len() + 1];

    let mut root_states_map: HashMap<u32, u128> = HashMap::new();
    for (idx, &state) in initial_states.iter().enumerate() {
        *root_states_map.entry(state as u32).or_default() += get_init_weight(idx);
    }
    let mut root_active = root_states_map.into_iter().collect::<Vec<_>>();
    root_active.sort_unstable_by_key(|k| k.0);

    process_string_node_combined(
        regex, &trie, 0, root_active, 
        &remainder_hashes_mask, &remainder_hashes_commit,
        &state_signatures,
        &linearized_mapping, 
        &mut accumulators_mask, &mut diffs_mask,
        &mut accumulators_commit, &mut diffs_commit,
        0
    );
    crate::debug!(4, "  Symbolic execution: {:?}", t3.elapsed());
    crate::debug!(3, "  Equiv: Symbolic execution: {:?}", t3.elapsed());
    pb.inc(1);

    let t4 = std::time::Instant::now();
    pb.set_message("Grouping...");
    // Apply diffs for mask
    let mut current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs_mask[lin_idx]);
        accumulators_mask[orig_idx] = accumulators_mask[orig_idx].wrapping_add(current_diff);
    }
    
    // Apply diffs for commit
    current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs_commit[lin_idx]);
        accumulators_commit[orig_idx] = accumulators_commit[orig_idx].wrapping_add(current_diff);
    }

    let mask_classes = group_by_hash(&accumulators_mask);
    let commit_classes = group_by_hash(&accumulators_commit);
    crate::debug!(3, "  Equiv: Grouping: {:?}", t4.elapsed());

    pb.finish_with_message("Done");
    
    CombinedEquivalenceResult {
        mask_classes,
        commit_classes,
    }
}

#[allow(clippy::too_many_arguments)]
fn process_string_node_combined(
    regex: &Regex,
    trie: &Trie,
    node_idx: usize,
    active_states: Vec<(u32, u128)>,
    remainder_hashes_mask: &[Vec<u64>],
    remainder_hashes_commit: &[Vec<u64>],
    state_signatures: &[u64],
    linearized_mapping: &[usize],
    accumulators_mask: &mut Vec<u128>,
    diffs_mask: &mut Vec<u128>,
    accumulators_commit: &mut Vec<u128>,
    diffs_commit: &mut Vec<u128>,
    depth: u32,
) {
    let node = &trie.nodes[node_idx];

    // 1. Terminals - handle differently for mask vs commit
    if let Some(_orig_idx) = node.terminal_string_idx {
        let lin_idx = node.range_start as usize;
        
        for &(dfa_state, weight) in &active_states {
            // Mask uses state signature
            let h_mask = hash_outcome_mask(KIND_TERM, 0, 0, 0, state_signatures[dfa_state as usize]);
            let contrib_mask = weight.wrapping_mul(h_mask);
            diffs_mask[lin_idx] = diffs_mask[lin_idx].wrapping_add(contrib_mask);
            diffs_mask[lin_idx + 1] = diffs_mask[lin_idx + 1].wrapping_sub(contrib_mask);
            
            // Commit uses actual state
            let h_commit = hash_outcome_commit(KIND_TERM, 0, 0, 0, dfa_state);
            let contrib_commit = weight.wrapping_mul(h_commit);
            diffs_commit[lin_idx] = diffs_commit[lin_idx].wrapping_add(contrib_commit);
            diffs_commit[lin_idx + 1] = diffs_commit[lin_idx + 1].wrapping_sub(contrib_commit);
        }
    }

    // Check if node has any children
    if node.transitions.is_empty() { return; }

    // 2. Transitions
    let mut next_batch: Vec<(u32, u32, u128)> = Vec::with_capacity(active_states.len());

    for &(dfa_state, weight) in &active_states {
        let dfa_node = &regex.dfa.states[dfa_state as usize];
        for (&byte, &child_idx) in &node.transitions {
            if let Some(&next_state) = dfa_node.transitions.get(byte) {
                let next_data = &regex.dfa.states[next_state];
                if !next_data.finalizers.is_empty() {
                    let child = &trie.nodes[child_idx as usize];
                    for gid in &next_data.finalizers {
                        for lin_idx in (child.range_start as usize)..(child.range_end as usize) {
                            let orig = linearized_mapping[lin_idx];
                            
                            // Mask hash
                            let rem_mask = remainder_hashes_mask[orig].get((depth + 1) as usize).copied().unwrap_or(0);
                            let h_mask = hash_outcome_mask(KIND_MATCH, gid as u32, depth + 1, rem_mask, 0);
                            accumulators_mask[orig] = accumulators_mask[orig].wrapping_add(weight.wrapping_mul(h_mask));
                            
                            // Commit hash
                            let rem_commit = remainder_hashes_commit[orig].get((depth + 1) as usize).copied().unwrap_or(0);
                            let h_commit = hash_outcome_commit(KIND_MATCH, gid as u32, depth + 1, rem_commit, 0);
                            accumulators_commit[orig] = accumulators_commit[orig].wrapping_add(weight.wrapping_mul(h_commit));
                        }
                    }
                }
                next_batch.push((child_idx, next_state as u32, weight));
            } else {
                // Dead End - same for both
                let h_mask = hash_outcome_mask(KIND_DEAD_END, 0, 0, 0, 0);
                let h_commit = hash_outcome_commit(KIND_DEAD_END, 0, 0, 0, 0);
                let contrib_mask = weight.wrapping_mul(h_mask);
                let contrib_commit = weight.wrapping_mul(h_commit);
                let child = &trie.nodes[child_idx as usize];
                let r_start = child.range_start as usize;
                let r_end = child.range_end as usize;
                diffs_mask[r_start] = diffs_mask[r_start].wrapping_add(contrib_mask);
                diffs_mask[r_end] = diffs_mask[r_end].wrapping_sub(contrib_mask);
                diffs_commit[r_start] = diffs_commit[r_start].wrapping_add(contrib_commit);
                diffs_commit[r_end] = diffs_commit[r_end].wrapping_sub(contrib_commit);
            }
        }
    }

    if next_batch.is_empty() { return; }

    // 3. Merge & Recurse
    next_batch.sort_unstable_by_key(|k| k.0);
    let mut i = 0;
    while i < next_batch.len() {
        let child = next_batch[i].0;
        let start = i;
        while i < next_batch.len() && next_batch[i].0 == child { i += 1; }

        let mut temp: SmallVec<[(u32, u128); 16]> = SmallVec::new();
        for &(_, s, w) in &next_batch[start..i] { temp.push((s, w)); }
        temp.sort_unstable_by_key(|k| k.0);

        let mut merged = Vec::with_capacity(temp.len());
        if !temp.is_empty() {
            let (mut cs, mut cw) = (temp[0].0, temp[0].1);
            for &(s, w) in &temp[1..] {
                if s == cs { cw = cw.wrapping_add(w); }
                else { merged.push((cs, cw)); cs = s; cw = w; }
            }
            merged.push((cs, cw));
        }

        process_string_node_combined(
            regex, trie, child as usize, merged, 
            remainder_hashes_mask, remainder_hashes_commit,
            state_signatures,
            linearized_mapping, 
            accumulators_mask, diffs_mask,
            accumulators_commit, diffs_commit,
            depth + 1
        );
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn compute_state_signatures(regex: &Regex) -> Vec<u64> {
    regex.dfa.states.iter().map(|s| {
        let mut h = 0xcbf29ce484222325u64; // FNV-1a 64-bit init
        for &gid in &s.possible_future_group_ids {
            let k = (gid as u64).wrapping_add(1);
            h ^= k;
            h = h.wrapping_mul(0x100000001b3u64); // FNV prime
        }
        // Final avalanche
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccdu64);
        h ^= h >> 33;
        h
    }).collect()
}

/// Precompute remainder hashes for BOTH mask and commit in a single pass.
/// This is the key optimization - we only call execute_from_state_fast once per position.
fn precompute_remainder_hashes_combined(
    regex: &Regex, 
    strings: &[Vec<u8>],
    state_signatures: &[u64],
) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    // Process strings in parallel - each string is independent
    let results: Vec<(Vec<u64>, Vec<u64>)> = strings.par_iter().map(|s| {
        let len = s.len();
        let mut hashes_mask = vec![0u64; len + 1];
        let mut hashes_commit = vec![0u64; len + 1];

        // Base case: End of string is a Terminal outcome
        let term_hash_mask = hash_outcome_mask(KIND_TERM, 0, 0, 0, 0);
        hashes_mask[len] = (term_hash_mask ^ (term_hash_mask >> 64)) as u64;
        
        let term_hash_commit = hash_outcome_commit(KIND_TERM, 0, 0, 0, 0);
        hashes_commit[len] = (term_hash_commit ^ (term_hash_commit >> 64)) as u64;

        for i in (0..len).rev() {
            // THE KEY OPTIMIZATION: only one call to execute_from_state_fast
            let exec = regex.execute_from_state_fast(&s[i..], regex.dfa.start_state);
            
            let mut node_hash_mask = 0u128;
            let mut node_hash_commit = 0u128;

            // 1. Pass-through (consumed remaining string fully)
            if let Some(st) = exec.end_state {
                // Mask uses state signature
                let h_mask = hash_outcome_mask(KIND_TERM, 0, 0, 0, state_signatures[st]);
                node_hash_mask = node_hash_mask.wrapping_add(h_mask);
                
                // Commit uses actual state
                let h_commit = hash_outcome_commit(KIND_TERM, 0, 0, 0, st as u32);
                node_hash_commit = node_hash_commit.wrapping_add(h_commit);
            }

            // 2. Token Matches (Ambiguity: sum all valid greedy-per-group paths)
            if !exec.matches.is_empty() {
                let mut matches = exec.matches;
                matches.sort_unstable_by(|a, b| {
                    a.group_id.cmp(&b.group_id)
                        .then_with(|| b.position.cmp(&a.position))
                });

                let mut prev_gid = usize::MAX;
                for m in matches {
                    if m.position == 0 { continue; }
                    if m.group_id == prev_gid { continue; }
                    prev_gid = m.group_id;

                    let next_idx = i + m.position;
                    
                    // Mask hash
                    let next_h_mask = unsafe { *hashes_mask.get_unchecked(next_idx) };
                    let h_mask = hash_outcome_mask(KIND_MATCH, m.group_id as u32, m.position as u32, next_h_mask, 0);
                    node_hash_mask = node_hash_mask.wrapping_add(h_mask);
                    
                    // Commit hash
                    let next_h_commit = unsafe { *hashes_commit.get_unchecked(next_idx) };
                    let h_commit = hash_outcome_commit(KIND_MATCH, m.group_id as u32, m.position as u32, next_h_commit, 0);
                    node_hash_commit = node_hash_commit.wrapping_add(h_commit);
                }
            }
            
            hashes_mask[i] = (node_hash_mask ^ (node_hash_mask >> 64)) as u64;
            hashes_commit[i] = (node_hash_commit ^ (node_hash_commit >> 64)) as u64;
        }
        (hashes_mask, hashes_commit)
    }).collect();
    
    // Unzip the results
    let (result_mask, result_commit): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    (result_mask, result_commit)
}

fn group_by_hash(accumulators: &[u128]) -> BTreeMap<Vec<usize>, Vec<usize>> {
    // Optimized: First pass maps hashes to IDs, second pass creates result
    let mut hash_to_id: HashMap<u128, usize> = HashMap::new();
    let mut id_to_indices: Vec<Vec<usize>> = Vec::new();
    
    for (i, &h) in accumulators.iter().enumerate() {
        let id = *hash_to_id.entry(h).or_insert_with(|| {
            id_to_indices.push(Vec::new());
            id_to_indices.len() - 1
        });
        id_to_indices[id].push(i);
    }
    
    // Convert to final format
    id_to_indices
        .into_iter()
        .enumerate()
        .map(|(id, indices)| (vec![id], indices))
        .collect()
}

fn create_pb(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
    if !PROGRESS_BAR_ENABLED { pb.set_draw_target(ProgressDrawTarget::hidden()); }
    pb
}
