use crate::finite_automata::{Regex, GroupID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::ops::BitXor;
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
fn mix_hash(h: u128, val: u64) -> u128 {
    // Order-dependent mixing (rotation) but position-independent
    // (not dependent on depth).
    h.rotate_left(3) ^ (val as u128)
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
        Trie { nodes: vec![TrieNode::default()] }
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
// String Equivalence Analysis
// -----------------------------------------------------------------------------

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<usize>, Vec<usize>> {
    crate::debug!(3, "Analyzing string equivalence for {} strings.", strings.len());
    let pb = create_pb(4);

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

    // (state, path_hash) -> weight
    let mut root_states_map: HashMap<(u32, u128), u128> = HashMap::new();
    for (idx, &state) in initial_states.iter().enumerate() {
        *root_states_map.entry((state as u32, 0)).or_default() += get_init_weight(idx);
    }
    let mut root_active = root_states_map.into_iter().map(|((s, ph), w)| (s, ph, w)).collect::<Vec<_>>();
    root_active.sort_unstable_by_key(|k| k.0);

    process_string_node(
        regex, &trie, 0, root_active, &linearized_mapping, &mut accumulators, &mut diffs
    );
    pb.inc(1);

    pb.set_message("Grouping...");
    let mut current_diff = 0u128;
    for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
        current_diff = current_diff.wrapping_add(diffs[lin_idx]);
        accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
    }

    let mut classes = group_by_hash(&accumulators);

    if VERIFY_RESULTS {
        pb.set_message("Verifying...");
        verify_string_classes(regex, strings, initial_states, &mut classes);
    }

    pb.finish_with_message("Done");
    classes
}

#[allow(clippy::too_many_arguments)]
fn process_string_node(
    regex: &Regex,
    trie: &Trie,
    node_idx: usize,
    active_states: Vec<(u32, u128, u128)>, // state, path_hash, weight
    linearized_mapping: &[usize],
    accumulators: &mut Vec<u128>,
    diffs: &mut Vec<u128>,
) {
    let node = &trie.nodes[node_idx];

    // 1. Terminals: Add final hash to specific strings terminating here
    if let Some(orig_idx) = node.terminal_string_idx {
        for &(dfa_state, path_hash, weight) in &active_states {
            let future_sig = hash_future_groups(regex, dfa_state as usize);
            // Mix future groups into the path hash
            let final_h = mix_hash(path_hash, future_sig);
            let contrib = weight.wrapping_mul(final_h);
            accumulators[orig_idx as usize] = accumulators[orig_idx as usize].wrapping_add(contrib);
        }
    }

    if node.transitions.is_empty() { return; }

    // 2. Transitions: Propagate or Dead-End
    let mut next_batch: Vec<(u32, u32, u128, u128)> = Vec::with_capacity(active_states.len());

    for &(dfa_state, path_hash, weight) in &active_states {
        let dfa_node = &regex.dfa.states[dfa_state as usize];
        
        for &(byte, child_idx) in &node.transitions {
            if let Some(&next_state) = dfa_node.transitions.get(byte) {
                // Alive: update path hash with matches (if any)
                let mut new_ph = path_hash;
                let next_data = &regex.dfa.states[next_state];
                
                if !next_data.finalizers.is_empty() {
                    for &gid in &next_data.finalizers {
                        new_ph = mix_hash(new_ph, gid as u64);
                    }
                }
                next_batch.push((child_idx, next_state as u32, new_ph, weight));
            } else {
                // Dead End: Apply difference array to subtree
                // Use a constant to signify "dead" so it differs from valid paths
                let dead_ph = mix_hash(path_hash, 0xDEAD_BEEF);
                let contrib = weight.wrapping_mul(dead_ph);
                let child = &trie.nodes[child_idx as usize];
                let r_start = child.range_start as usize;
                let r_end = child.range_end as usize;
                
                diffs[r_start] = diffs[r_start].wrapping_add(contrib);
                diffs[r_end] = diffs[r_end].wrapping_sub(contrib);
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
        // next_batch item: (child_idx, next_state, new_ph, weight)
        // We group by (next_state, new_ph)
        let mut slice = next_batch[start..i].iter().map(|&(_, s, p, w)| (s, p, w)).collect::<Vec<_>>();
        slice.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut merged = Vec::with_capacity(slice.len());
        if !slice.is_empty() {
            let (mut cs, mut cp, mut cw) = (slice[0].0, slice[0].1, slice[0].2);
            for &(s, p, w) in &slice[1..] {
                if s == cs && p == cp { cw = cw.wrapping_add(w); }
                else { merged.push((cs, cp, cw)); cs = s; cp = p; cw = w; }
            }
            merged.push((cs, cp, cw));
        }

        process_string_node(regex, trie, child as usize, merged, linearized_mapping, accumulators, diffs);
    }
}

// -----------------------------------------------------------------------------
// Helpers & Verification
// -----------------------------------------------------------------------------

fn hash_future_groups(regex: &Regex, state_idx: usize) -> u64 {
    let mut h = 0u64;
    for &gid in &regex.dfa.states[state_idx].possible_future_group_ids {
        let k = (gid as u64).wrapping_mul(0x9E3779B97F4A7C15);
        h = h.rotate_left(3).bitxor(k);
    }
    h
}

fn group_by_hash(accumulators: &[u128]) -> BTreeMap<Vec<usize>, Vec<usize>> {
    let mut map = HashMap::new();
    let mut next_id = 0;
    let mut classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (i, &h) in accumulators.iter().enumerate() {
        let id = *map.entry(h).or_insert_with(|| { next_id += 1; next_id - 1 });
        classes.entry(vec![id]).or_default().push(i);
    }
    classes
}

fn create_pb(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
    if !PROGRESS_BAR_ENABLED { pb.set_draw_target(ProgressDrawTarget::hidden()); }
    pb
}

fn verify_string_classes(regex: &Regex, strings: &[Vec<u8>], initial_states: &[usize], classes: &mut BTreeMap<Vec<usize>, Vec<usize>>) {
    let mut new_classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    let mut next_id = 0;

    for (_, group) in classes.iter() {
        if group.len() <= 1 {
            new_classes.entry(vec![next_id]).or_default().extend(group);
            next_id += 1;
            continue;
        }
        let mut subgroups: Vec<(usize, Vec<usize>)> = Vec::new();
        'outer: for &idx in group {
            for (leader, members) in &mut subgroups {
                if are_strings_eq(regex, strings, initial_states, *leader, idx) {
                    members.push(idx);
                    continue 'outer;
                }
            }
            subgroups.push((idx, vec![idx]));
        }
        for (_, members) in subgroups {
            new_classes.entry(vec![next_id]).or_default().extend(members);
            next_id += 1;
        }
    }
    *classes = new_classes;
}

fn are_strings_eq(regex: &Regex, strings: &[Vec<u8>], states: &[usize], a: usize, b: usize) -> bool {
    let (sa, sb) = (&strings[a], &strings[b]);
    for &st in states {
        let (ra, rb) = (regex.execute_from_state_fast(sa, st), regex.execute_from_state_fast(sb, st));
        if ra.end_state != rb.end_state || ra.matches.len() != rb.matches.len() { return false; }
        for (ma, mb) in ra.matches.iter().zip(rb.matches.iter()) {
            if ma.group_id != mb.group_id || ma.position != mb.position { return false; }
        }
    }
    true
}
