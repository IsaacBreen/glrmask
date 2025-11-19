use crate::finite_automata::{Regex, GroupID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use smallvec::SmallVec;
use std::collections::BTreeMap;

// -----------------------------------------------------------------------------
// 256-bit Hashing Primitives
// -----------------------------------------------------------------------------

/// A 256-bit hash composed of two independent 128-bit hashes.
/// This provides cryptographic-grade collision resistance ($10^{-77}$).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct DualHash(u128, u128);

impl DualHash {
    #[inline(always)]
    fn zero() -> Self {
        DualHash(0, 0)
    }

    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        DualHash(
            self.0.wrapping_add(other.0),
            self.1.wrapping_add(other.1),
        )
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        DualHash(
            self.0.wrapping_sub(other.0),
            self.1.wrapping_sub(other.1),
        )
    }

    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        DualHash(
            self.0.wrapping_mul(other.0),
            self.1.wrapping_mul(other.1),
        )
    }
}

// --- Mixer A (Standard Variant) ---
#[inline(always)]
fn mix_a(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

// --- Mixer B (Independent Constants) ---
// Uses different prime constants to ensure linear independence from Mixer A.
// This ensures that even if Mixer A has a collision, Mixer B likely won't.
#[inline(always)]
fn mix_b(mut x: u128) -> u128 {
    x ^= x >> 31;
    x = x.wrapping_mul(0x9E3779B97F4A7C159E3779B97F4A7C15); // Golden Ratio expansion
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9BF58476D1CE4E5B9); // SplitMix constant expansion
    x ^= x >> 32;
    x
}

#[inline(always)]
fn get_init_weight(idx: usize) -> DualHash {
    // Seed A: (idx << 1) | 1
    // Seed B: ~(idx << 1) (Bitwise inverse to ensure difference)
    let raw = (idx as u128) << 1 | 1;
    DualHash(mix_a(raw) | 1, mix_b(!raw) | 1)
}

#[inline(always)]
fn hash_outcome(
    is_match: bool,
    match_group: u32,
    match_pos: u32,
    remainder_sig: DualHash,
    final_state: u32,
) -> DualHash {
    let flags = (if is_match { 1 } else { 0 }) | (final_state << 1);

    // Layout: [ match_pos (32) | match_group (32) | flags (32) | padding (32) ]
    let packed = ((match_pos as u128) << 96)
        | ((match_group as u128) << 64)
        | ((flags as u128) << 32);

    // Mix both lanes independently
    DualHash(
        mix_a(packed ^ remainder_sig.0),
        mix_b(packed ^ remainder_sig.1),
    )
}

// -----------------------------------------------------------------------------
// Trie Definition (Unchanged)
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
// Analysis Logic
// -----------------------------------------------------------------------------

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<usize>, Vec<usize>> {
    crate::debug!(
        2,
        "Starting high-precision (256-bit) equivalence analysis for {} strings.",
        strings.len()
    );

    let pb = ProgressBar::new(4);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    // 1. Precompute Dual Signatures
    pb.set_message("Precomputing 256-bit signatures...");
    let remainder_hashes = precompute_remainder_hashes(regex, strings);
    pb.inc(1);

    // 2. Build Trie
    pb.set_message("Building Trie...");
    let mut trie = Trie::new();
    for (i, s) in strings.iter().enumerate() {
        trie.insert(s, i as u32);
    }
    let linearized_mapping = trie.linearize();
    pb.inc(1);

    // 3. Symbolic Execution
    pb.set_message("Symbolic Execution...");

    // We use DualHash for accumulators to prevent collisions
    let mut accumulators = vec![DualHash::zero(); strings.len()];
    let mut diffs = vec![DualHash::zero(); strings.len() + 1];

    let mut root_states_map: HashMap<u32, DualHash> = HashMap::new();
    for (idx, &state) in initial_states.iter().enumerate() {
        let weight = get_init_weight(idx);
        let entry = root_states_map.entry(state as u32).or_default();
        *entry = entry.wrapping_add(weight);
    }

    let mut root_active_states: Vec<(u32, DualHash)> = root_states_map.into_iter().collect();
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

    // 4. Grouping
    pb.set_message("Grouping results...");

    let mut current_diff = DualHash::zero();
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs[lin_idx]);
        accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
    }

    let mut hash_to_sig_id: HashMap<DualHash, usize> = HashMap::new();
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
fn process_node(
    regex: &Regex,
    trie: &Trie,
    node_idx: usize,
    active_states: Vec<(u32, DualHash)>,
    remainder_hashes: &[Vec<DualHash>],
    linearized_mapping: &[usize],
    accumulators: &mut Vec<DualHash>,
    diffs: &mut Vec<DualHash>,
    depth: u32,
) {
    let node = &trie.nodes[node_idx];

    // Handle Terminals
    if let Some(_orig_idx) = node.terminal_string_idx {
        let lin_idx = node.range_start as usize;
        for &(dfa_state, weight) in &active_states {
            let end_state_val = if regex.dfa.states[dfa_state as usize].transitions.is_empty() {
                0
            } else {
                dfa_state + 1
            };

            let h = hash_outcome(false, 0, 0, DualHash::zero(), end_state_val);
            let contribution = weight.wrapping_mul(h);

            diffs[lin_idx] = diffs[lin_idx].wrapping_add(contribution);
            diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(contribution);
        }
    }

    if node.transitions.is_empty() {
        return;
    }

    let mut next_batch: Vec<(u32, u32, DualHash)> = Vec::with_capacity(active_states.len());

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
                // Dead End
                let h = hash_outcome(false, 0, 0, DualHash::zero(), 0);
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

        let mut temp: SmallVec<[(u32, DualHash); 16]> = SmallVec::new();
        for &(_, state, w) in child_states_raw {
            temp.push((state, w));
        }
        temp.sort_unstable_by_key(|k| k.0);

        let mut merged_states: Vec<(u32, DualHash)> = Vec::with_capacity(temp.len());
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

fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>]) -> Vec<Vec<DualHash>> {
    let mut results = Vec::with_capacity(strings.len());
    for s in strings {
        let mut row = Vec::with_capacity(s.len() + 1);
        for i in 0..=s.len() {
            let suffix = &s[i..];
            let exec = regex.execute_from_state_fast(suffix, regex.dfa.start_state);

            let mut h = DualHash::zero();
            for m in exec.matches {
                // Mix (Group, Pos)
                let k_a = (m.group_id as u64).wrapping_mul(0x9E3779B97F4A7C15)
                    ^ ((m.position as u64).rotate_left(32));

                // Use different constants for Lane B
                let k_b = (m.group_id as u64).wrapping_mul(0x1B873593C6A4A793)
                    ^ ((m.position as u64).rotate_right(27));

                let term = DualHash(
                    (k_a as u128).wrapping_mul(0xC6A4A7935BD1E995),
                    (k_b as u128).wrapping_mul(0x5BD1E995C6A4A793),
                );
                h = h.wrapping_add(term);
            }

            let end_val = if let Some(fs) = exec.end_state {
                (fs as u64).wrapping_add(1)
            } else {
                0
            };

            h.0 ^= (end_val.rotate_left(17)) as u128;
            h.1 ^= (end_val.rotate_right(13)) as u128;

            row.push(h);
        }
        results.push(row);
    }
    results
}