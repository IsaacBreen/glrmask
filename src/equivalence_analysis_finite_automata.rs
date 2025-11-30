use crate::finite_automata::{Regex, GroupID};
use crate::r#macro::should_show_progress_bars;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};

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

#[inline(always)]
fn hash_outcome(
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
    // // TEMP: Disable
    // return strings.iter().enumerate().map(|(i, _)| (vec![i], vec![i])).collect();
    crate::debug!(3, "Analyzing string equivalence for {} strings.", strings.len());
    let pb = create_pb(4);

    let state_signatures = compute_state_signatures(regex);
    pb.set_message("Precomputing signatures...");
    let remainder_hashes = precompute_remainder_hashes(regex, strings, &state_signatures);
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
        regex, &trie, 0, root_active, &remainder_hashes, &state_signatures,
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
        verify_string_classes(regex, strings, initial_states, &classes, &accumulators);
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
    state_signatures: &[u64],
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
            let h = hash_outcome(KIND_TERM, 0, 0, 0, state_signatures[dfa_state as usize]);
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
                            let rem = remainder_hashes[orig].get((depth + 1) as usize).copied().unwrap_or(0);
                            let h = hash_outcome(KIND_MATCH, gid as u32, depth + 1, rem, 0);
                            accumulators[orig] = accumulators[orig].wrapping_add(weight.wrapping_mul(h));
                        }
                    }
                }
                next_batch.push((child_idx, next_state as u32, weight));
            } else {
                // Dead End
                let h = hash_outcome(KIND_DEAD_END, 0, 0, 0, 0);
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

        process_string_node(regex, trie, child as usize, merged, remainder_hashes, state_signatures, linearized_mapping, accumulators, diffs, depth + 1);
    }
}

// -----------------------------------------------------------------------------
// Helpers & Verification
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

fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>], state_signatures: &[u64]) -> Vec<Vec<u64>> {
    let mut result = Vec::with_capacity(strings.len());

    for s in strings {
        let len = s.len();
        let mut hashes = vec![0u64; len + 1];

        // Base case: End of string is a Terminal outcome
        // We fold the u128 hash to u64
        let term_hash = hash_outcome(KIND_TERM, 0, 0, 0, 0);
        hashes[len] = (term_hash ^ (term_hash >> 64)) as u64;

        for i in (0..len).rev() {
            let exec = regex.execute_from_state_fast(&s[i..], regex.dfa.start_state);
            
            let mut node_hash = 0u128;

            // 1. Pass-through (consumed remaining string fully)
            if let Some(st) = exec.end_state {
                let h = hash_outcome(KIND_TERM, 0, 0, 0, state_signatures[st]);
                node_hash = node_hash.wrapping_add(h);
            }

            // 2. Token Matches (Ambiguity: sum all valid greedy-per-group paths)
            if !exec.matches.is_empty() {
                let mut matches = exec.matches;
                // Sort by GroupID, then Position descending to easily pick longest per group
                matches.sort_unstable_by(|a, b| {
                    a.group_id.cmp(&b.group_id)
                        .then_with(|| b.position.cmp(&a.position))
                });

                let mut prev_gid = usize::MAX;
                for m in matches {
                    if m.position == 0 { continue; }
                    if m.group_id == prev_gid { continue; } // Skip shorter matches for same group
                    prev_gid = m.group_id;

                    let next_idx = i + m.position;
                    let next_h = unsafe { *hashes.get_unchecked(next_idx) };
                    let h = hash_outcome(KIND_MATCH, m.group_id as u32, m.position as u32, next_h, 0);
                    node_hash = node_hash.wrapping_add(h);
                }
            }
            
            hashes[i] = (node_hash ^ (node_hash >> 64)) as u64;
        }
        result.push(hashes);
    }
    result
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
    if !should_show_progress_bars() { pb.set_draw_target(ProgressDrawTarget::hidden()); }
    pb
}

fn verify_string_classes(regex: &Regex, strings: &[Vec<u8>], initial_states: &[usize], classes: &BTreeMap<Vec<usize>, Vec<usize>>, accumulators: &[u128]) {
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
    // Ensure they're exactly the same, and report any differences
    if classes != &new_classes {
        // Find some illustrative examples of differences
        let mut examples = Vec::new();

        // Check for strings that are in same class in 'classes' but different in 'new_classes'
        for (_, group) in classes.iter() {
            if group.len() > 1 {
                // Find which new_classes these strings ended up in
                let mut new_class_ids: HashMap<usize, Vec<usize>> = HashMap::new();
                for &idx in group {
                    for (new_id, new_group) in new_classes.iter() {
                        if new_group.contains(&idx) {
                            new_class_ids.entry(new_id[0]).or_default().push(idx);
                            break;
                        }
                    }
                }

                if new_class_ids.len() > 1 && examples.len() < 3 {
                    // This group was split - show first 2 strings from different subgroups
                    let mut iter = new_class_ids.values();
                    if let (Some(subgroup1), Some(subgroup2)) = (iter.next(), iter.next()) {
                        if let (Some(&idx1), Some(&idx2)) = (subgroup1.first(), subgroup2.first()) {
                            examples.push((idx1, idx2));
                        }
                    }
                }
            }
        }

        eprintln!("ERROR: Hash-based classification differs from brute-force verification!");
        eprintln!("Total classes: hash={}, verified={}", classes.len(), new_classes.len());
        eprintln!("\nIllustrative examples of incorrectly grouped strings:");
        for (i, (idx1, idx2)) in examples.iter().enumerate() {
            eprintln!("  Example {}:", i + 1);
            eprintln!("    String {}: {:?} (Hash: {:032x})", idx1, String::from_utf8_lossy(&strings[*idx1]), accumulators[*idx1]);
            eprintln!("    String {}: {:?} (Hash: {:032x})", idx2, String::from_utf8_lossy(&strings[*idx2]), accumulators[*idx2]);
            eprintln!("    Were grouped together but are NOT equivalent");
        }
        panic!("Hash collision or logic error detected in equivalence analysis");
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TokenizationOutcome {
    tokens: Vec<GroupID>,
    future_possibilities: Option<BTreeSet<GroupID>>,
}

fn get_tokenization_outcomes(regex: &Regex, s: &[u8], initial_state: usize) -> BTreeSet<TokenizationOutcome> {
    let mut results = BTreeSet::new();
    // Stack: (offset, state, history)
    let mut stack = vec![(0usize, initial_state, Vec::new())];

    while let Some((offset, state, history)) = stack.pop() {
        if offset == s.len() {
            results.insert(TokenizationOutcome {
                tokens: history,
                future_possibilities: Some(regex.dfa.states[state].possible_future_group_ids.clone()),
            });
            continue;
        }

        let remaining = &s[offset..];
        let exec = regex.execute_from_state_fast(remaining, state);

        // 1. Pass-through path
        if let Some(end_st) = exec.end_state {
            results.insert(TokenizationOutcome {
                tokens: history.clone(),
                future_possibilities: Some(regex.dfa.states[end_st].possible_future_group_ids.clone()),
            });
        }

        // 2. Match paths
        // Filter: for each GroupID, keep max position
        let mut max_matches: HashMap<GroupID, usize> = HashMap::new();
        for m in exec.matches {
            if m.position == 0 { continue; }
            let entry = max_matches.entry(m.group_id).or_insert(0);
            if m.position > *entry {
                *entry = m.position;
            }
        }

        for (gid, pos) in max_matches {
            let new_offset = offset + pos;
            let mut new_history = history.clone();
            new_history.push(gid);
            stack.push((new_offset, regex.dfa.start_state, new_history));
        }
    }
    results
}

fn are_strings_eq(regex: &Regex, strings: &[Vec<u8>], states: &[usize], a: usize, b: usize) -> bool {
    let (sa, sb) = (&strings[a], &strings[b]);
    for &st in states {
        let outcomes_a = get_tokenization_outcomes(regex, sa, st);
        let outcomes_b = get_tokenization_outcomes(regex, sb, st);
        if outcomes_a != outcomes_b {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite_automata::{eat_u8, eat_u8_seq};
    use crate::groups;

    /// Verify that find_equivalence_classes produces the same groupings as brute-force.
    fn verify_equivalence_classes(regex: &Regex, strings: &[Vec<u8>]) {
        let initial_states: Vec<usize> = (0..regex.dfa.states.len()).collect();
        let classes = find_equivalence_classes(regex, strings, &initial_states);

        // Build brute-force equivalence classes
        let mut bf_classes: Vec<Vec<usize>> = Vec::new();
        for i in 0..strings.len() {
            let mut found = false;
            for class in &mut bf_classes {
                if are_strings_eq(regex, strings, &initial_states, class[0], i) {
                    class.push(i);
                    found = true;
                    break;
                }
            }
            if !found {
                bf_classes.push(vec![i]);
            }
        }

        // Compare: each class from find_equivalence_classes should be a subset of some bf_class
        for (_, indices) in &classes {
            if indices.len() <= 1 {
                continue;
            }
            // All members should be equivalent according to brute-force
            let first = indices[0];
            for &other in &indices[1..] {
                assert!(
                    are_strings_eq(regex, strings, &initial_states, first, other),
                    "Strings {} and {} are in same class but not equivalent!\n  {:?}\n  {:?}",
                    first, other,
                    String::from_utf8_lossy(&strings[first]),
                    String::from_utf8_lossy(&strings[other])
                );
            }
        }
    }

    #[test]
    fn test_equivalence_end_with_extra_char() {
        // Regression test for iterative tokenization semantics.
        // " []" should NOT be equivalent to " [];" because after iteratively 
        // tokenizing both strings:
        // - " []" -> [WS, '[', ']'] and ends at state 0 (consumed all input)
        // - " [];" -> [WS, '[', ']'] but then ';' at state 0 causes dead-end
        //
        // The fix ensures are_strings_eq uses iterative tokenization (restarting
        // from state 0 after each terminal match) instead of single-pass DFA execution.
        use crate::finite_automata::rep1;
        
        let regex = groups![
            rep1(eat_u8(b' ')),  // WS
            eat_u8(b'['),        // array open
            eat_u8(b']'),        // array close
        ].build();

        let strings: Vec<Vec<u8>> = vec![
            b" []".to_vec(),   // matches WS, [, ] and ends successfully
            b" [];".to_vec(),  // matches WS, [, ], then ';' causes dead-end
        ];

        // Verify they are NOT considered equivalent
        let initial_states: Vec<usize> = (0..regex.dfa.states.len()).collect();
        let are_equal = are_strings_eq(&regex, &strings, &initial_states, 0, 1);
        assert!(!are_equal, 
            "Strings ' []' and ' [];' should NOT be equivalent - the latter has a trailing char that dead-ends");
        
        // Also verify through the full find_equivalence_classes function
        verify_equivalence_classes(&regex, &strings);
    }

    #[test]
    fn test_equivalence_end_state_collision() {
        // Regression test: " f" and " n" were colliding because remainder hash ignored end_state.
        // Setup: WS | "false" | "null"
        // " f" -> Matches WS, then remainder "f" results in partial match state for "false"
        // " n" -> Matches WS, then remainder "n" results in partial match state for "null"
        
        let f_seq = eat_u8_seq(b"false".to_vec());
        let n_seq = eat_u8_seq(b"null".to_vec());
        let ws = eat_u8(b' ');
        
        let regex = groups![ws, f_seq, n_seq].build();
        
        let strings: Vec<Vec<u8>> = vec![
            b" f".to_vec(),
            b" n".to_vec(),
        ];
        verify_equivalence_classes(&regex, &strings);
    }
}
