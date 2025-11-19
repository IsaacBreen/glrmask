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

/// Computes a hash for a specific outcome.
#[inline(always)]
fn hash_outcome(
    is_match: bool,
    match_group: u32,
    match_pos: u32,
    remainder_sig: u64,
    final_state: u32,
) -> u128 {
    let flags = if is_match { 1 } else { 0 } | (final_state << 1);
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
            nodes: vec![TrieNode::default()],
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
        self.nodes[node_idx].transitions.sort_unstable_by_key(|k| k.0);
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

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<usize>, Vec<usize>> {
    crate::debug!(
        2,
        "Starting sparse-wavefront equivalence analysis for {} strings and {} states.",
        strings.len(),
        initial_states.len()
    );

    let pb = ProgressBar::new(5);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    // 1. Precompute Remainder Signatures (Fast Path)
    pb.set_message("Precomputing remainder signatures...");
    let remainder_hashes = precompute_remainder_hashes(regex, strings);
    pb.inc(1);

    // 2. Build Inverted Index for DFA
    // Map: byte -> List of states that have a transition on this byte
    pb.set_message("Indexing DFA...");
    let mut states_with_transition: Vec<Vec<u32>> = vec![Vec::new(); 256];
    for (state_idx, state) in regex.dfa.states.iter().enumerate() {
        for (byte, _) in state.transitions.iter() {
            states_with_transition[byte as usize].push(state_idx as u32);
        }
    }
    pb.inc(1);

    // 3. Build Trie
    pb.set_message("Building Trie...");
    let mut trie = Trie::new();
    for (i, s) in strings.iter().enumerate() {
        trie.insert(s, i as u32);
    }
    let linearized_mapping = trie.linearize();
    pb.inc(1);

    // 4. Sparse Wavefront Traversal
    pb.set_message("Wavefront Traversal...");
    let mut accumulators = vec![0u128; strings.len()];
    let mut diffs = vec![0u128; strings.len() + 1];

    // Initial Active States
    let mut root_active: HashMap<u32, u128> = HashMap::new();
    for (idx, &state) in initial_states.iter().enumerate() {
        let w = get_init_weight(idx);
        *root_active.entry(state as u32).or_default() =
            root_active.entry(state as u32).or_default().wrapping_add(w);
    }

    process_node_sparse(
        regex,
        &trie,
        0,
        root_active,
        &states_with_transition,
        &remainder_hashes,
        &linearized_mapping,
        &mut accumulators,
        &mut diffs,
        0,
    );
    pb.inc(1);

    // 5. Finalize
    pb.set_message("Grouping...");
    let mut current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs[lin_idx]);
        accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
    }

    let mut hash_to_sig_id: HashMap<u128, usize> = HashMap::new();
    let mut next_sig_id = 0;
    let mut classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

    for (str_idx, &h) in accumulators.iter().enumerate() {
        let sig_id = *hash_to_sig_id.entry(h).or_insert_with(|| {
            let id = next_sig_id;
            next_sig_id += 1;
            id
        });
        classes.entry(vec![sig_id]).or_default().push(str_idx);
    }
    pb.finish_with_message("Done");

    classes
}

#[allow(clippy::too_many_arguments)]
fn process_node_sparse(
    regex: &Regex,
    trie: &Trie,
    node_idx: usize,
    active_states: HashMap<u32, u128>, // Sparse map: State -> Weight
    states_with_transition: &[Vec<u32>],
    remainder_hashes: &[Vec<u64>],
    linearized_mapping: &[usize],
    accumulators: &mut Vec<u128>,
    diffs: &mut Vec<u128>,
    depth: u32,
) {
    let node = &trie.nodes[node_idx];

    // Handle Terminal (String Ends Here)
    if let Some(_orig_idx) = node.terminal_string_idx {
        let lin_idx = node.range_start as usize;
        // Calculate hash for ending at these states
        // If active_states is huge (Root), this loop is slow.
        // But active_states is only huge at Root.
        // And only empty string ends at Root.
        // So this loop runs once for empty string.
        // For deeper nodes, active_states is small.
        let mut terminal_hash = 0u128;
        for (&state, &weight) in &active_states {
            // Logic: if transitions empty, Dead (0). Else state+1.
            let end_val = if regex.dfa.states[state as usize].transitions.is_empty() {
                0
            } else {
                state + 1
            };
            let h = hash_outcome(false, 0, 0, 0, end_val);
            terminal_hash = terminal_hash.wrapping_add(weight.wrapping_mul(h));
        }
        diffs[lin_idx] = diffs[lin_idx].wrapping_add(terminal_hash);
        diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(terminal_hash);
    }

    if node.transitions.is_empty() {
        return;
    }

    let total_weight: u128 = active_states.values().fold(0, |a, b| a.wrapping_add(*b));
    let dead_hash_const = hash_outcome(false, 0, 0, 0, 0); // Generic Dead Hash

    for &(byte, child_idx) in &node.transitions {
        // Identify survivors: Intersection of Active States and States with transition on 'byte'
        // Since Active States can be large (at Root), but StatesWithTransition is usually small (except for Start State),
        // we iterate the smaller set if possible.
        // However, HashMap lookup is O(1).
        // So iterating StatesWithTransition and looking up in Active is O(|Transitions|).
        // This is efficient.

        let candidates = &states_with_transition[byte as usize];
        let mut next_active: HashMap<u32, u128> = HashMap::new();
        let mut survivor_weight = 0u128;

        for &state in candidates {
            if let Some(&weight) = active_states.get(&state) {
                // This state survives
                survivor_weight = survivor_weight.wrapping_add(weight);

                let state_data = &regex.dfa.states[state as usize];
                // We know transition exists because it's in candidates
                let next_state = *state_data.transitions.get(byte).unwrap() as u32;

                // Accumulate weight for next state
                *next_active.entry(next_state).or_default() =
                    next_active.entry(next_state).or_default().wrapping_add(weight);

                // Check for Matches
                let next_state_data = &regex.dfa.states[next_state as usize];
                if !next_state_data.finalizers.is_empty() {
                    // Apply match hash to subtree
                    let child_node = &trie.nodes[child_idx as usize];
                    let r_start = child_node.range_start as usize;
                    let r_end = child_node.range_end as usize;

                    for &group_id in &next_state_data.finalizers {
                        // Iterate subtree for variable remainder hash
                        for lin_idx in r_start..r_end {
                            let orig_idx = linearized_mapping[lin_idx];
                            let rem_sig = remainder_hashes[orig_idx][(depth + 1) as usize];
                            let h = hash_outcome(true, group_id as u32, depth + 1, rem_sig, 0);
                            let contrib = weight.wrapping_mul(h);
                            accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(contrib);
                        }
                    }
                }
            }
        }

        // Apply Dead Hash to non-survivors
        let dead_weight = total_weight.wrapping_sub(survivor_weight);
        if dead_weight != 0 {
            let child_node = &trie.nodes[child_idx as usize];
            let r_start = child_node.range_start as usize;
            let r_end = child_node.range_end as usize;
            let contrib = dead_weight.wrapping_mul(dead_hash_const);
            diffs[r_start] = diffs[r_start].wrapping_add(contrib);
            diffs[r_end] = diffs[r_end].wrapping_sub(contrib);
        }

        // Recurse if there are survivors
        if !next_active.is_empty() {
            process_node_sparse(
                regex,
                trie,
                child_idx as usize,
                next_active,
                states_with_transition,
                remainder_hashes,
                linearized_mapping,
                accumulators,
                diffs,
                depth + 1,
            );
        }
    }
}

fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>]) -> Vec<Vec<u64>> {
    let mut results = Vec::with_capacity(strings.len());
    for s in strings {
        let mut row = Vec::with_capacity(s.len() + 1);
        for i in 0..=s.len() {
            let suffix = &s[i..];
            let exec = regex.execute_from_state_fast(suffix, regex.dfa.start_state);
            let mut h = 0u64;
            for m in exec.matches {
                let k = (m.group_id as u64).wrapping_mul(0x9E3779B97F4A7C15)
                      ^ ((m.position as u64).rotate_left(32));
                h = h.wrapping_mul(0xC6A4A7935BD1E995).wrapping_add(k);
            }
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