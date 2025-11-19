use crate::finite_automata::{ExecutionResult, GroupID, Match, Regex};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::collections::BTreeMap;
use hashbrown::HashMap;
use smallvec::SmallVec;

// -----------------------------------------------------------------------------
// Hashing Utilities
// -----------------------------------------------------------------------------

#[inline(always)]
fn mix_u128(x: u128) -> u128 {
    let mut x = x;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Generates a deterministic, odd pseudo-random weight for an initial state index.
#[inline(always)]
fn get_init_weight(idx: usize) -> u128 {
    mix_u128((idx as u128) << 1 | 1) | 1
}

/// Computes a hash for a specific outcome (Match or Dead/End).
///
/// `flags`:
///    Bit 0: is_match
///    Bit 1..32: final_state (if not match)
#[inline(always)]
fn hash_outcome(
    is_match: bool,
    match_group: u32,
    match_pos: u32,
    remainder_sig: u64,
    final_state: u32,
) -> u128 {
    let flags = if is_match { 1 } else { 0 } | (final_state << 1);

    // Layout: [ match_pos (32) | match_group (32) | flags (32) | padding (32) ]
    // We put high entropy stuff in low bits for wrapping arithmetic safety?
    // Actually, we mix the whole thing.
    let packed = ((match_pos as u128) << 96)
        | ((match_group as u128) << 64)
        | ((flags as u128) << 32);

    mix_u128(packed ^ (remainder_sig as u128))
}

// -----------------------------------------------------------------------------
// Trie Definition
// -----------------------------------------------------------------------------

#[derive(Default, Clone)]
struct TrieNode {
    // Sparse transitions: (byte, child_index)
    transitions: SmallVec<[(u8, u32); 4]>,
    // If this node terminates a string, which original string index is it?
    terminal_string_idx: Option<u32>,
    // Range of string indices (in the linearized DFS order) covered by this subtree
    range_start: u32,
    range_end: u32,
}

struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn new() -> Self {
        Trie {
            nodes: vec![TrieNode::default()], // Root is 0
        }
    }

    fn insert(&mut self, s: &[u8], original_idx: u32) {
        let mut node_idx = 0;
        for &b in s {
            let mut found = None;
            for &(byte, child) in &self.nodes[node_idx].transitions {
                if byte == b {
                    found = Some(child as usize);
                    break;
                }
            }
            match found {
                Some(child) => node_idx = child,
                None => {
                    let new_node_idx = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[node_idx].transitions.push((b, new_node_idx as u32));
                    node_idx = new_node_idx;
                }
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

        // Sort transitions for deterministic traversal
        self.nodes[node_idx].transitions.sort_unstable_by_key(|k| k.0);

        // Clone to avoid borrow checker issues during recursion
        let children = self.nodes[node_idx].transitions.clone();

        for &(_, child_idx) in &children {
            self.dfs_linearize(child_idx as usize, mapping);
        }

        let end = mapping.len() as u32;
        self.nodes[node_idx].range_start = start;
        self.nodes[node_idx].range_end = end;
    }
}

// -----------------------------------------------------------------------------
// Analysis Logic
// -----------------------------------------------------------------------------

/// Finds equivalence classes among a set of strings based on their tokenization
/// behavior with a given Regex, starting from a set of initial DFA states.
///
/// Optimized using a Trie + Difference Array approach.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<usize>, Vec<usize>> {
    crate::debug!(
        2,
        "Starting optimized equivalence analysis for {} strings and {} states.",
        strings.len(),
        initial_states.len()
    );

    let pb = ProgressBar::new(4);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    // 1. Precompute "Standard Signatures" (Remainder hashes)
    pb.set_message("Precomputing remainder signatures...");
    let remainder_hashes = precompute_remainder_hashes(regex, strings);
    pb.inc(1);

    // 2. Build and Linearize Trie
    pb.set_message("Building Trie...");
    let mut trie = Trie::new();
    for (i, s) in strings.iter().enumerate() {
        trie.insert(s, i as u32);
    }
    let linearized_mapping = trie.linearize();
    pb.inc(1);

    // 3. Symbolic Execution
    pb.set_message("Symbolic Execution...");

    // Accumulators for per-string hash sums (used for variable updates like matches)
    let mut accumulators = vec![0u128; strings.len()];
    // Difference array for range updates (used for constant updates like dead ends)
    let mut diffs = vec![0u128; strings.len() + 1];

    // Initial active states: Group by DFA state and sum weights.
    // Map: DFA_State -> TotalWeight
    let mut root_states_map: HashMap<u32, u128> = HashMap::new();
    for (idx, &state) in initial_states.iter().enumerate() {
        let weight = get_init_weight(idx);
        *root_states_map.entry(state as u32).or_default() += weight;
    }

    let mut root_active_states: Vec<(u32, u128)> = root_states_map.into_iter().collect();
    root_active_states.sort_unstable_by_key(|k| k.0);

    process_node(
        regex,
        &trie,
        0, // root
        root_active_states,
        &remainder_hashes,
        &linearized_mapping,
        &mut accumulators,
        &mut diffs,
        0, // depth
    );
    pb.inc(1);

    // 4. Finalize and Group
    pb.set_message("Grouping results...");

    // Apply difference array to accumulators
    let mut current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs[lin_idx]);
        accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
    }

    // Group by final hash
    let mut hash_to_sig_id: HashMap<u128, usize> = HashMap::new();
    let mut next_sig_id = 0;
    let mut classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

    for (str_idx, &h) in accumulators.iter().enumerate() {
        let sig_id = *hash_to_sig_id.entry(h).or_insert_with(|| {
            let id = next_sig_id;
            next_sig_id += 1;
            id
        });
        // Wrap in Vec to match original API
        classes.entry(vec![sig_id]).or_default().push(str_idx);
    }
    pb.finish_with_message("Done");

    classes
}

// Recursive function to traverse the Trie
#[allow(clippy::too_many_arguments)]
fn process_node(
    regex: &Regex,
    trie: &Trie,
    node_idx: usize,
    active_states: Vec<(u32, u128)>, // (CurrentDFAState, TotalWeight)
    remainder_hashes: &[Vec<u64>],
    linearized_mapping: &[usize],
    accumulators: &mut Vec<u128>,
    diffs: &mut Vec<u128>,
    depth: u32,
) {
    let node = &trie.nodes[node_idx];

    // 1. Handle Terminals (Strings ending here)
    if let Some(_orig_idx) = node.terminal_string_idx {
        // The string ends here.
        // For each active state, the outcome is "Ended at State X".
        // Since this node is a terminal, it occupies a slot in the linearized mapping.
        // The slot index is `node.range_start`.
        let lin_idx = node.range_start as usize;

        for &(dfa_state, weight) in &active_states {
            // Check if the state is "dead" (no transitions) or just stopped.
            // execute_from_state_fast returns None if no transitions, Some(state) if stopped by input end.
            // If transitions is empty, it's None? No, check finite_automata logic.
            // "End of input: return a continuation state only if further transitions are possible."
            // If transitions is empty, returns None. Else Some(state).
            let end_state_val = if regex.dfa.states[dfa_state as usize].transitions.is_empty() {
                0 // None -> 0
            } else {
                dfa_state + 1 // Some(s) -> s + 1
            };

            // If end_state_val is 0 (None), it means "Dead".
            // If > 0, it means "Stopped at state".
            // We hash this outcome.
            let h = hash_outcome(false, 0, 0, 0, end_state_val);

            // Apply weighted hash
            let contribution = weight.wrapping_mul(h);

            // Update diffs for this single position
            diffs[lin_idx] = diffs[lin_idx].wrapping_add(contribution);
            diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(contribution);
        }
    }

    if node.transitions.is_empty() {
        return;
    }

    // 2. Prepare for transitions
    // We collect next states for all children.
    // Structure: (ChildNodeIdx, NextDFAState, Weight)
    let mut next_batch: Vec<(u32, u32, u128)> = Vec::with_capacity(active_states.len());

    for &(dfa_state, weight) in &active_states {
        let dfa_state_node = &regex.dfa.states[dfa_state as usize];

        // Iterate over all outgoing bytes from this Trie node
        for &(byte, child_node_idx) in &node.transitions {
            if let Some(&next_dfa_state) = dfa_state_node.transitions.get(byte) {
                // Transition exists
                let next_state_data = &regex.dfa.states[next_dfa_state];

                // Check for Matches (Finalizers)
                if !next_state_data.finalizers.is_empty() {
                    // Match Event!
                    // Iterate all strings in the child's subtree
                    let child_node = &trie.nodes[child_node_idx as usize];
                    let r_start = child_node.range_start as usize;
                    let r_end = child_node.range_end as usize;

                    // For each finalizer (match)
                    for &group_id in &next_state_data.finalizers {
                        // Iterate subtree
                        for lin_idx in r_start..r_end {
                            let orig_idx = linearized_mapping[lin_idx];
                            // Remainder signature at depth + 1
                            // (depth is current node depth, child is depth+1)
                            let rem_sig = remainder_hashes[orig_idx][(depth + 1) as usize];

                            let h = hash_outcome(true, group_id as u32, depth + 1, rem_sig, 0);
                            let contribution = weight.wrapping_mul(h);
                            accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(contribution);
                        }
                    }
                }

                // Continue traversal
                next_batch.push((child_node_idx, next_dfa_state as u32, weight));
            } else {
                // No transition -> Dead End for this branch
                // Outcome: "Dead" (None)
                let h = hash_outcome(false, 0, 0, 0, 0); // 0 = None
                let contribution = weight.wrapping_mul(h);

                // Apply to child's subtree
                let child_node = &trie.nodes[child_node_idx as usize];
                let r_start = child_node.range_start as usize;
                let r_end = child_node.range_end as usize;

                diffs[r_start] = diffs[r_start].wrapping_add(contribution);
                diffs[r_end] = diffs[r_end].wrapping_sub(contribution);
            }
        }
    }

    // 3. Recurse
    if next_batch.is_empty() {
        return;
    }

    // Sort by ChildIdx to group for recursion
    next_batch.sort_unstable_by_key(|k| k.0);

    let mut i = 0;
    while i < next_batch.len() {
        let current_child = next_batch[i].0;
        let start_i = i;

        // Collect all states for this child
        // We need to merge weights for identical DFA states
        // Use a small local vector or map?
        // Since we need to pass `Vec<(u32, u128)>` sorted by DFA state.

        // Let's collect and sort/merge.
        // Optimization: If the batch for this child is small, simple sort is fine.

        // Find end of this child's batch
        while i < next_batch.len() && next_batch[i].0 == current_child {
            i += 1;
        }

        let child_states_raw = &next_batch[start_i..i];

        // Merge weights
        // If list is small, sort and linear scan merge is fastest.
        let mut merged_states: Vec<(u32, u128)> = Vec::with_capacity(child_states_raw.len());

        // Copy to temp vector to sort by DFA state
        let mut temp: SmallVec<[(u32, u128); 16]> = SmallVec::new();
        for &(_, state, w) in child_states_raw {
            temp.push((state, w));
        }
        temp.sort_unstable_by_key(|k| k.0);

        if !temp.is_empty() {
            let mut curr_state = temp[0].0;
            let mut curr_w = temp[0].1;

            for &(s, w) in &temp[1..] {
                if s == curr_state {
                    curr_w = curr_w.wrapping_add(w);
                } else {
                    merged_states.push((curr_state, curr_w));
                    curr_state = s;
                    curr_w = w;
                }
            }
            merged_states.push((curr_state, curr_w));
        }

        process_node(
            regex,
            trie,
            current_child as usize,
            merged_states,
            remainder_hashes,
            linearized_mapping,
            accumulators,
            diffs,
            depth + 1,
        );
    }
}

// Helper to precompute suffix hashes
fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>]) -> Vec<Vec<u64>> {
    let mut results = Vec::with_capacity(strings.len());
    for s in strings {
        let mut row = Vec::with_capacity(s.len() + 1);
        for i in 0..=s.len() {
            let suffix = &s[i..];
            // Run tokenizer (fast path)
            let exec = regex.execute_from_state_fast(suffix, regex.dfa.start_state);

            // Hash the matches and end state
            let mut h = 0u64;
            for m in exec.matches {
                // Mix (Group, Pos)
                // We use a simple mix here, distinct from the main hash_outcome
                let k = (m.group_id as u64).wrapping_mul(0x9E3779B97F4A7C15)
                      ^ ((m.position as u64).rotate_left(32));
                h = h.wrapping_mul(0xC6A4A7935BD1E995).wrapping_add(k);
            }

            // Mix end state
            let end_val = if let Some(fs) = exec.end_state {
                (fs as u64).wrapping_add(1)
            } else {
                0
            };
            h ^= end_val.rotate_left(17);

            row.push(h);
        }
        results.push(row);
    }
    results
}