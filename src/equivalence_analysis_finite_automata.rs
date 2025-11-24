use crate::finite_automata::{Regex, GroupID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
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
        *root_states_map.entry(state as u32).or_default() += get_init_weight(idx);
    }
    let mut root_active = root_states_map.into_iter().collect::<Vec<_>>();
    root_active.sort_unstable_by_key(|k| k.0);

    process_string_node(
        regex, &trie, 0, root_active, &remainder_hashes,
        &linearized_mapping, &mut accumulators, &mut diffs, 0
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
    active_states: Vec<(u32, u128)>,
    remainder_hashes: &[Vec<u64>],
    linearized_mapping: &[usize],
    accumulators: &mut Vec<u128>,
    diffs: &mut Vec<u128>,
    depth: u32,
) {
    let node = &trie.nodes[node_idx];

    // 1. Terminals
    if let Some(_orig_idx) = node.terminal_string_idx {
        let lin_idx = node.range_start as usize;
        for &(dfa_state, weight) in &active_states {
            let end_val = if regex.dfa.states[dfa_state as usize].transitions.is_empty() { 0 } else { dfa_state + 1 };
            let h = hash_outcome(false, 0, 0, 0, end_val);
            let contrib = weight.wrapping_mul(h);
            diffs[lin_idx] = diffs[lin_idx].wrapping_add(contrib);
            diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(contrib);
        }
    }

    if node.transitions.is_empty() { return; }

    // 2. Transitions
    let mut next_batch: Vec<(u32, u32, u128)> = Vec::with_capacity(active_states.len());

    for &(dfa_state, weight) in &active_states {
        let dfa_node = &regex.dfa.states[dfa_state as usize];
        for &(byte, child_idx) in &node.transitions {
            if let Some(&next_state) = dfa_node.transitions.get(byte) {
                let next_data = &regex.dfa.states[next_state];
                if !next_data.finalizers.is_empty() {
                    let child = &trie.nodes[child_idx as usize];
                    for gid in &next_data.finalizers {
                        for lin_idx in (child.range_start as usize)..(child.range_end as usize) {
                            let orig = linearized_mapping[lin_idx];
                            let rem = remainder_hashes[orig][(depth + 1) as usize];
                            let h = hash_outcome(true, gid as u32, depth + 1, rem, 0);
                            accumulators[orig] = accumulators[orig].wrapping_add(weight.wrapping_mul(h));
                        }
                    }
                }
                next_batch.push((child_idx, next_state as u32, weight));
            } else {
                // Dead End
                let h = hash_outcome(false, 0, 0, 0, 0);
                let contrib = weight.wrapping_mul(h);
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

        process_string_node(regex, trie, child as usize, merged, remainder_hashes, linearized_mapping, accumulators, diffs, depth + 1);
    }
}

// -----------------------------------------------------------------------------
// Helpers & Verification
// -----------------------------------------------------------------------------

fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>]) -> Vec<Vec<u64>> {
    strings.iter().map(|s| {
        (0..=s.len()).map(|i| {
            let exec = regex.execute_from_state_fast(&s[i..], regex.dfa.start_state);
            let mut h = 0u64;
            for m in exec.matches {
                let k = (m.group_id as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ ((m.position as u64).rotate_left(32));
                h = h.wrapping_mul(0xC6A4A7935BD1E995).wrapping_add(k);
            }
            let end_val = if let Some(fs) = exec.end_state { (fs as u64).wrapping_add(1) } else { 0 };
            h ^ end_val.rotate_left(17)
        }).collect()
    }).collect()
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
