use crate::finite_automata::{Regex, GroupID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use smallvec::SmallVec;
use std::collections::BTreeMap;

// -----------------------------------------------------------------------------
// Hashing Utilities
// -----------------------------------------------------------------------------

/// Mixes bits to generate a high-quality pseudo-random number.
#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
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

/// Computes a hash for a specific execution outcome.
///
/// This mixes the match details (if any) and the final state (if stopped)
/// with the precomputed signature of the remaining string.
#[inline(always)]
fn hash_outcome(
    is_match: bool,
    match_group: u32,
    match_pos: u32,
    remainder_sig: u64,
    final_state: u32,
) -> u128 {
    // Bit 0: is_match, Bits 1..32: final_state
    let flags = (if is_match { 1 } else { 0 }) | (final_state << 1);

    // Layout: [ match_pos (32) | match_group (32) | flags (32) | padding (32) ]
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
    /// Sparse transitions: (byte, child_index)
    transitions: SmallVec<[(u8, u32); 4]>,
    /// If this node terminates a string, which original string index is it?
    terminal_string_idx: Option<u32>,
    /// Range of string indices (in the linearized DFS order) covered by this subtree.
    /// Used for O(1) updates via the difference array.
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
            // Fast linear scan for transition (SmallVec makes this fast)
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

    /// Flattens the Trie into a linear range [0, N).
    /// Returns a mapping from [LinearIndex] -> [OriginalStringIndex].
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

        // Sort transitions to ensure deterministic traversal order
        self.nodes[node_idx].transitions.sort_unstable_by_key(|k| k.0);

        // Clone children to avoid borrow checker conflicts during recursion
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
/// This implementation uses a **Trie + Difference Array** approach.
/// It is optimized for cases where the number of active states is moderate,
/// using `SmallVec` and sorting instead of HashMaps for state merging.
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
    //    This allows us to hash the "future" of a string in O(1) during traversal.
    pb.set_message("Precomputing remainder signatures...");
    let remainder_hashes = precompute_remainder_hashes(regex, strings);
    pb.inc(1);

    // 2. Build and Linearize Trie
    //    This groups common prefixes and maps strings to a contiguous integer range.
    pb.set_message("Building Trie...");
    let mut trie = Trie::new();
    for (i, s) in strings.iter().enumerate() {
        trie.insert(s, i as u32);
    }
    let linearized_mapping = trie.linearize();
    pb.inc(1);

    // 3. Symbolic Execution
    pb.set_message("Symbolic Execution...");

    // Accumulators: Store the sum of hashes for events that happen specifically to a string (Matches).
    let mut accumulators = vec![0u128; strings.len()];
    // Difference Array: Stores updates for events that happen to a whole subtree (Dead ends / Terminals).
    let mut diffs = vec![0u128; strings.len() + 1];

    // Initial active states: Group by DFA state and sum weights.
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

    // Apply difference array to accumulators (Prefix Sum)
    let mut current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs[lin_idx]);
        accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
    }

    // Group strings by their final accumulated hash
    let mut hash_to_sig_id: HashMap<u128, usize> = HashMap::new();
    let mut next_sig_id = 0;
    let mut classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

    for (str_idx, &h) in accumulators.iter().enumerate() {
        let sig_id = *hash_to_sig_id.entry(h).or_insert_with(|| {
            let id = next_sig_id;
            next_sig_id += 1;
            id
        });
        // Wrap in Vec to match the expected API signature
        classes.entry(vec![sig_id]).or_default().push(str_idx);
    }
    pb.finish_with_message("Done");

    classes
}

/// Recursive function to traverse the Trie and update hash accumulators.
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

    // -------------------------------------------------------------------------
    // 1. Handle Terminals (Strings ending at this node)
    // -------------------------------------------------------------------------
    if let Some(_orig_idx) = node.terminal_string_idx {
        // The string ends here. We must record the state in which it ended.
        let lin_idx = node.range_start as usize;

        for &(dfa_state, weight) in &active_states {
            // If transitions are empty, it's a "Dead" state (0). Otherwise, it's a valid stop (state + 1).
            let end_state_val = if regex.dfa.states[dfa_state as usize].transitions.is_empty() {
                0
            } else {
                dfa_state + 1
            };

            let h = hash_outcome(false, 0, 0, 0, end_state_val);
            let contribution = weight.wrapping_mul(h);

            // Update diffs for this single position
            diffs[lin_idx] = diffs[lin_idx].wrapping_add(contribution);
            diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(contribution);
        }
    }

    if node.transitions.is_empty() {
        return;
    }

    // -------------------------------------------------------------------------
    // 2. Prepare Transitions
    // -------------------------------------------------------------------------
    // We collect all outgoing transitions from all active states.
    // Structure: (ChildNodeIdx, NextDFAState, Weight)
    let mut next_batch: Vec<(u32, u32, u128)> = Vec::with_capacity(active_states.len());

    for &(dfa_state, weight) in &active_states {
        let dfa_state_node = &regex.dfa.states[dfa_state as usize];

        // Iterate over all outgoing bytes from this Trie node
        for &(byte, child_node_idx) in &node.transitions {
            if let Some(&next_dfa_state) = dfa_state_node.transitions.get(byte) {
                // --- Transition Exists ---
                let next_state_data = &regex.dfa.states[next_dfa_state];

                // Check for Matches (Finalizers)
                if !next_state_data.finalizers.is_empty() {
                    // A match occurred. This is a variable update specific to the strings in this subtree.
                    let child_node = &trie.nodes[child_node_idx as usize];
                    let r_start = child_node.range_start as usize;
                    let r_end = child_node.range_end as usize;

                    for &group_id in &next_state_data.finalizers {
                        // Iterate subtree to apply specific remainder hashes
                        for lin_idx in r_start..r_end {
                            let orig_idx = linearized_mapping[lin_idx];
                            // Look ahead at the remainder signature
                            let rem_sig = remainder_hashes[orig_idx][(depth + 1) as usize];

                            let h = hash_outcome(true, group_id as u32, depth + 1, rem_sig, 0);
                            let contribution = weight.wrapping_mul(h);
                            accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(contribution);
                        }
                    }
                }

                // Queue for recursion
                next_batch.push((child_node_idx, next_dfa_state as u32, weight));
            } else {
                // --- No Transition (Dead End) ---
                // Apply "Dead" hash to the entire subtree via difference array.
                let h = hash_outcome(false, 0, 0, 0, 0); // 0 = None/Dead
                let contribution = weight.wrapping_mul(h);

                let child_node = &trie.nodes[child_node_idx as usize];
                let r_start = child_node.range_start as usize;
                let r_end = child_node.range_end as usize;

                diffs[r_start] = diffs[r_start].wrapping_add(contribution);
                diffs[r_end] = diffs[r_end].wrapping_sub(contribution);
            }
        }
    }

    if next_batch.is_empty() {
        return;
    }

    // -------------------------------------------------------------------------
    // 3. Merge and Recurse
    // -------------------------------------------------------------------------
    // Sort by ChildIdx to group states for the next recursion step.
    next_batch.sort_unstable_by_key(|k| k.0);

    let mut i = 0;
    while i < next_batch.len() {
        let current_child = next_batch[i].0;
        let start_i = i;

        // Find the range of states belonging to `current_child`
        while i < next_batch.len() && next_batch[i].0 == current_child {
            i += 1;
        }
        let child_states_raw = &next_batch[start_i..i];

        // Merge weights for identical DFA states.
        // Using SmallVec + Sort is faster than HashMap for small N.
        let mut temp: SmallVec<[(u32, u128); 16]> = SmallVec::new();
        for &(_, state, w) in child_states_raw {
            temp.push((state, w));
        }
        // Sort by DFA state to allow linear merge
        temp.sort_unstable_by_key(|k| k.0);

        let mut merged_states: Vec<(u32, u128)> = Vec::with_capacity(temp.len());
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

/// Precomputes a rolling hash of the "remainder" of every string at every position.
/// This allows us to know "what would happen if we restarted matching here" in O(1).
fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>]) -> Vec<Vec<u64>> {
    let mut results = Vec::with_capacity(strings.len());
    for s in strings {
        let mut row = Vec::with_capacity(s.len() + 1);
        for i in 0..=s.len() {
            let suffix = &s[i..];
            // Run tokenizer (fast path)
            let exec = regex.execute_from_state_fast(suffix, regex.dfa.start_state);

            // Hash the matches
            let mut h = 0u64;
            for m in exec.matches {
                // Mix (Group, Pos)
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