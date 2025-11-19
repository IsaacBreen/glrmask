use crate::finite_automata::{Regex, GroupID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use smallvec::SmallVec;
use std::collections::BTreeMap;

// -----------------------------------------------------------------------------
// Hashing Utilities (Fast Probabilistic Part)
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
fn get_init_weight(idx: usize) -> u128 {
    mix_u128((idx as u128) << 1 | 1) | 1
}

#[inline(always)]
fn hash_outcome(
    is_match: bool,
    match_group: u32,
    match_pos: u32,
    remainder_sig: u64,
    final_state: u32,
) -> u128 {
    let flags = (if is_match { 1 } else { 0 }) | (final_state << 1);
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
    transitions: SmallVec<[(u8, u32); 4]>,
    terminal_string_idx: Option<u32>,
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
// Main Analysis Logic
// -----------------------------------------------------------------------------

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<usize>, Vec<usize>> {
    crate::debug!(
        2,
        "Starting exact equivalence analysis (Hash+Verify) for {} strings.",
        strings.len()
    );

    let pb = ProgressBar::new(5);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    // --- Phase 1: Fast Hashing (The "Impl 1" Logic) ---

    pb.set_message("Precomputing signatures...");
    let remainder_hashes = precompute_remainder_hashes(regex, strings);
    pb.inc(1);

    pb.set_message("Building Trie...");
    let mut trie = Trie::new();
    for (i, s) in strings.iter().enumerate() {
        trie.insert(s, i as u32);
    }
    let linearized_mapping = trie.linearize();
    pb.inc(1);

    pb.set_message("Symbolic Execution...");
    let mut accumulators = vec![0u128; strings.len()];
    let mut diffs = vec![0u128; strings.len() + 1];

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
        0,
        root_active_states,
        &remainder_hashes,
        &linearized_mapping,
        &mut accumulators,
        &mut diffs,
        0,
    );
    pb.inc(1);

    pb.set_message("Grouping candidates...");
    let mut current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs[lin_idx]);
        accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
    }

    // Initial grouping by Hash
    let mut hash_groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (str_idx, &h) in accumulators.iter().enumerate() {
        hash_groups.entry(h).or_default().push(str_idx);
    }
    pb.inc(1);

    // --- Phase 2: Verification (The "Exact" Logic) ---
    // We iterate through the hash groups. If a group has > 1 item, we verify
    // that they are ACTUALLY identical by running the DFA.
    // Since collisions are 1 in 10^38, this loop almost never finds a split,
    // but it provides the mathematical guarantee.

    pb.set_message("Verifying groups...");

    let mut final_classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    let mut next_sig_id = 0;

    for (_, candidates) in hash_groups {
        if candidates.len() <= 1 {
            // Optimization: A group of 1 cannot have a collision.
            final_classes.entry(vec![next_sig_id]).or_default().extend(candidates);
            next_sig_id += 1;
            continue;
        }

        // Collision Resolution Strategy:
        // 1. Take the first string as a "Leader".
        // 2. Compare every other string to the Leader.
        // 3. If they match, they stay. If not, they form a new subgroup.

        // List of (LeaderStringIdx, ListOfMembers)
        let mut verified_subgroups: Vec<(usize, Vec<usize>)> = Vec::with_capacity(1);

        'candidate_loop: for &str_idx in &candidates {
            // Try to fit into an existing verified subgroup
            for (leader_idx, members) in &mut verified_subgroups {
                if are_strictly_equivalent(regex, strings, initial_states, *leader_idx, str_idx) {
                    members.push(str_idx);
                    continue 'candidate_loop;
                }
            }
            // If no match found, start a new subgroup
            verified_subgroups.push((str_idx, vec![str_idx]));
        }

        // Register all resulting subgroups
        for (_, members) in verified_subgroups {
            final_classes.entry(vec![next_sig_id]).or_default().extend(members);
            next_sig_id += 1;
        }
    }

    pb.finish_with_message("Done");
    final_classes
}

// -----------------------------------------------------------------------------
// Verification Logic
// -----------------------------------------------------------------------------

/// Returns true if two strings produce EXACTLY the same tokens for all initial states.
/// This is the "Ground Truth" check.
fn are_strictly_equivalent(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
    idx_a: usize,
    idx_b: usize,
) -> bool {
    let s_a = &strings[idx_a];
    let s_b = &strings[idx_b];

    // Optimization: If lengths differ, they might still be equivalent (if suffix is ignored),
    // but usually they aren't. We run the DFA to be sure.

    for &state in initial_states {
        // We run the DFA step-by-step or use the execute helper.
        // Since we need exact match of ALL tokens, we can just run the full execution.
        // Note: This is fast because we don't allocate vectors of matches, we just compare iterators.

        // However, `regex.execute` usually allocates. Let's do a lightweight check.
        // We can reuse the `execute_from_state_fast` logic but compare results immediately.

        let res_a = regex.execute_from_state_fast(s_a, state);
        let res_b = regex.execute_from_state_fast(s_b, state);

        // Compare End States
        if res_a.end_state != res_b.end_state {
            return false;
        }

        // Compare Matches (Count, Group, Position)
        if res_a.matches.len() != res_b.matches.len() {
            return false;
        }
        for (m_a, m_b) in res_a.matches.iter().zip(res_b.matches.iter()) {
            if m_a.group_id != m_b.group_id || m_a.position != m_b.position {
                return false;
            }
        }
    }
    true
}

// -----------------------------------------------------------------------------
// Recursive Trie Processing (Same as Impl 1)
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn process_node(
    regex: &Regex,
    trie: &Trie,
    node_idx: usize,
    active_states: Vec<(u32, u128)>,
    remainder_hashes: &[Vec<u64>],
    linearized_mapping: &[usize],
    accumulators: &mut Vec<u128>,
    diffs: &mut Vec<u128>,
    depth: u32,
) {
    let node = &trie.nodes[node_idx];

    if let Some(_orig_idx) = node.terminal_string_idx {
        let lin_idx = node.range_start as usize;
        for &(dfa_state, weight) in &active_states {
            let end_state_val = if regex.dfa.states[dfa_state as usize].transitions.is_empty() {
                0
            } else {
                dfa_state + 1
            };
            let h = hash_outcome(false, 0, 0, 0, end_state_val);
            let contribution = weight.wrapping_mul(h);
            diffs[lin_idx] = diffs[lin_idx].wrapping_add(contribution);
            diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(contribution);
        }
    }

    if node.transitions.is_empty() {
        return;
    }

    let mut next_batch: Vec<(u32, u32, u128)> = Vec::with_capacity(active_states.len());

    for &(dfa_state, weight) in &active_states {
        let dfa_state_node = &regex.dfa.states[dfa_state as usize];
        for &(byte, child_node_idx) in &node.transitions {
            if let Some(&next_dfa_state) = dfa_state_node.transitions.get(byte) {
                let next_state_data = &regex.dfa.states[next_dfa_state];
                if !next_state_data.finalizers.is_empty() {
                    let child_node = &trie.nodes[child_node_idx as usize];
                    let r_start = child_node.range_start as usize;
                    let r_end = child_node.range_end as usize;
                    for &group_id in &next_state_data.finalizers {
                        for lin_idx in r_start..r_end {
                            let orig_idx = linearized_mapping[lin_idx];
                            let rem_sig = remainder_hashes[orig_idx][(depth + 1) as usize];
                            let h = hash_outcome(true, group_id as u32, depth + 1, rem_sig, 0);
                            let contribution = weight.wrapping_mul(h);
                            accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(contribution);
                        }
                    }
                }
                next_batch.push((child_node_idx, next_dfa_state as u32, weight));
            } else {
                let h = hash_outcome(false, 0, 0, 0, 0);
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

    next_batch.sort_unstable_by_key(|k| k.0);

    let mut i = 0;
    while i < next_batch.len() {
        let current_child = next_batch[i].0;
        let start_i = i;
        while i < next_batch.len() && next_batch[i].0 == current_child {
            i += 1;
        }
        let child_states_raw = &next_batch[start_i..i];

        let mut temp: SmallVec<[(u32, u128); 16]> = SmallVec::new();
        for &(_, state, w) in child_states_raw {
            temp.push((state, w));
        }
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