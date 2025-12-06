use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

struct PrecomputedDfa {
    start_state: usize,
    transitions: Vec<[Option<usize>; 256]>,
    finalizers: Vec<Vec<usize>>,
    possible_future: Vec<Vec<usize>>,
    has_transitions: Vec<bool>,
    non_greedy_flags: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

const NONE_POS: usize = usize::MAX;

struct Pos0Scratch {
    current_states: Vec<usize>,
    done: Vec<bool>,
    match_positions: Vec<usize>,
    touched_groups: Vec<Vec<usize>>,
}

fn precompute_dfa(regex: &Regex) -> PrecomputedDfa {
    let dfa = &regex.dfa;

    let mut max_gid: Option<usize> = None;
    for state in &dfa.states {
        if let Some(m) = state.finalizers.iter().max() {
            max_gid = Some(max_gid.map_or(m, |cur| cur.max(m)));
        }
        if let Some(m) = state.possible_future_group_ids.iter().max() {
            max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
        }
    }
    if let Some(m) = dfa.non_greedy_finalizers.iter().max() {
        max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
    }

    let num_groups = max_gid.map(|m| m + 1).unwrap_or(0);

    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut possible_future = Vec::with_capacity(dfa.states.len());
    let mut has_transitions = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [None; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = Some(target);
        }
        transitions.push(table);
        finalizers.push(state.finalizers.iter().collect());
        possible_future.push(state.possible_future_group_ids.iter().copied().collect());
        has_transitions.push(!state.transitions.is_empty());
    }

    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups {
            non_greedy_flags[gid] = true;
        }
    }

    let none_completion_hash = {
        let mut hasher = DefaultHasher::new();
        Option::<Vec<usize>>::None.hash(&mut hasher);
        hasher.finish()
    };

    let mut completion_hash = Vec::with_capacity(possible_future.len());
    for vec in &possible_future {
        let mut hasher = DefaultHasher::new();
        Some(vec).hash(&mut hasher);
        completion_hash.push(hasher.finish());
    }

    PrecomputedDfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        possible_future,
        has_transitions,
        non_greedy_flags,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

impl Pos0Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Pos0Scratch {
            current_states: vec![0; num_states],
            done: vec![false; num_states],
            match_positions: vec![NONE_POS; num_states.saturating_mul(num_groups)],
            touched_groups: vec![Vec::new(); num_states],
        }
    }

    fn reset(&mut self, initial_states: &[usize], num_groups: usize) {
        self.current_states.clone_from_slice(initial_states);
        self.done.fill(false);
        if !self.match_positions.is_empty() {
            self.match_positions.fill(NONE_POS);
        }
        for touched in &mut self.touched_groups {
            touched.clear();
        }
        if num_groups == 0 {
            return;
        }
    }

    #[inline]
    fn idx(num_groups: usize, state_idx: usize, gid: usize) -> usize {
        state_idx * num_groups + gid
    }
}

fn compute_pos0_results(
    pre: &PrecomputedDfa,
    scratch: &mut Pos0Scratch,
    slice: &[u8],
    initial_states: &[usize],
) -> (Vec<(Option<usize>, Vec<(usize, usize)>)>, Vec<usize>) {
    let num_states = initial_states.len();
    let num_groups = pre.num_groups;
    let len = slice.len();

    scratch.reset(initial_states, num_groups);

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let match_positions = &mut scratch.match_positions;
    let touched_groups = &mut scratch.touched_groups;

    for (i, &state) in initial_states.iter().enumerate() {
        let base = i * num_groups;
        for &gid in &pre.finalizers[state] {
            if gid < num_groups {
                let idx = base + gid;
                if match_positions[idx] == NONE_POS {
                    match_positions[idx] = 0;
                    touched_groups[i].push(gid);
                }
            }
        }
        if !pre.has_transitions[state] {
            done[i] = true;
        }
    }

    for (pos, &byte) in slice.iter().enumerate() {
        let position = pos + 1;

        for i in 0..num_states {
            if done[i] {
                continue;
            }

            let base = i * num_groups;
            let current = current_states[i];
            let next_state = pre.transitions[current][byte as usize];

            if let Some(next_state) = next_state {
                current_states[i] = next_state;

                for &gid in &pre.finalizers[next_state] {
                    if gid < num_groups {
                        let idx = base + gid;
                        let slot = &mut match_positions[idx];
                        if pre.non_greedy_flags[gid] {
                            if *slot == NONE_POS {
                                *slot = position;
                                touched_groups[i].push(gid);
                            }
                        } else {
                            let was_empty = *slot == NONE_POS;
                            *slot = position;
                            if was_empty {
                                touched_groups[i].push(gid);
                            }
                        }
                    }
                }

                let futures = &pre.possible_future[next_state];
                let mut terminate = true;
                for &gid in futures {
                    if gid >= num_groups {
                        terminate = false;
                        break;
                    }
                    let idx = base + gid;
                    if !(pre.non_greedy_flags[gid] && match_positions[idx] != NONE_POS) {
                        terminate = false;
                        break;
                    }
                }

                if terminate {
                    done[i] = true;
                }
            } else {
                done[i] = true;
            }
        }
    }

    let mut all_targets: Vec<usize> = Vec::new();
    let mut seen_target = vec![false; len + 1];
    let mut results: Vec<(Option<usize>, Vec<(usize, usize)>)> =
        Vec::with_capacity(num_states);

    for i in 0..num_states {
        let end_state = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };

        let mut edges: Vec<(usize, usize)> = Vec::new();
        if num_groups > 0 {
            let base = i * num_groups;
            for &gid in &touched_groups[i] {
                if gid >= num_groups {
                    continue;
                }
                let pos_val = match_positions[base + gid];
                if pos_val != NONE_POS && pos_val > 0 {
                    edges.push((gid, pos_val));
                    if pos_val <= len && !seen_target[pos_val] {
                        seen_target[pos_val] = true;
                        all_targets.push(pos_val);
                    }
                }
            }
        }

        edges.sort_unstable_by_key(|e| e.0);
        results.push((end_state, edges));
    }

    (results, all_targets)
}

fn compute_suffix_hashes(
    regex: &Regex,
    pre: &PrecomputedDfa,
    slice: &[u8],
    all_targets: &[usize],
) -> Vec<u64> {
    let len = slice.len();
    let mut visited = vec![false; len + 1];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut order: Vec<usize> = Vec::new();
    let mut nodes: Vec<Option<(Option<usize>, Vec<(usize, usize)>)>> =
        vec![None; len + 1];

    for &pos in all_targets {
        if pos > 0 && pos <= len && !visited[pos] {
            visited[pos] = true;
            queue.push_back(pos);
        }
    }

    while let Some(pos) = queue.pop_front() {
        let result = regex.execute_from_state_nonzero(&slice[pos..], pre.start_state);

        let mut edges: Vec<(usize, usize)> = result
            .matches
            .iter()
            .map(|m| {
                let target = pos + m.position;
                if target <= len && !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                }
                (m.group_id, target)
            })
            .collect();

        edges.sort_unstable_by_key(|e| e.0);
        nodes[pos] = Some((result.end_state, edges));
        order.push(pos);
    }

    order.sort_unstable_by(|a, b| b.cmp(a));
    let mut pos_hashes: Vec<u64> = vec![0; len + 1];

    for pos in order {
        if let Some((end_state, edges)) = &nodes[pos] {
            let mut hasher = DefaultHasher::new();
            let completion_hash = end_state
                .map(|id| pre.completion_hash[id])
                .unwrap_or(pre.none_completion_hash);
            completion_hash.hash(&mut hasher);

            for (group_id, target) in edges {
                let target_hash = pos_hashes[*target];
                (group_id, target_hash).hash(&mut hasher);
            }

            pos_hashes[pos] = hasher.finish();
        }
    }

    pos_hashes
}

fn compute_final_signature(
    pre: &PrecomputedDfa,
    pos0_results: &[(Option<usize>, Vec<(usize, usize)>)],
    pos_hashes: &[u64],
) -> u64 {
    let mut combined_hasher = DefaultHasher::new();

    for (end_state, edges) in pos0_results {
        let mut hasher = DefaultHasher::new();
        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        completion_hash.hash(&mut hasher);

        for (group_id, target) in edges {
            let target_hash = pos_hashes.get(*target).copied().unwrap_or(0);
            (group_id, target_hash).hash(&mut hasher);
        }

        hasher.finish().hash(&mut combined_hasher);
    }

    combined_hasher.finish()
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    use std::sync::atomic::{AtomicU64, Ordering};

    let pre = precompute_dfa(regex);
    let pos0_time = AtomicU64::new(0);
    let suffix_time = AtomicU64::new(0);
    let hash_time = AtomicU64::new(0);

    let signatures: Vec<u64> = strings
        .par_iter()
        .map_init(
            || Pos0Scratch::new(initial_states.len(), pre.num_groups),
            |scratch, s| {
            let t0 = std::time::Instant::now();
            let (pos0_results, all_targets) =
                compute_pos0_results(&pre, scratch, s, initial_states);
            let t1 = std::time::Instant::now();
            let pos_hashes = compute_suffix_hashes(regex, &pre, s, &all_targets);
            let t2 = std::time::Instant::now();
            let sig = compute_final_signature(&pre, &pos0_results, &pos_hashes);
            let t3 = std::time::Instant::now();

            pos0_time.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            suffix_time.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            hash_time.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);

            sig
            },
        )
        .collect();

    let total = pos0_time.load(Ordering::Relaxed)
        + suffix_time.load(Ordering::Relaxed)
        + hash_time.load(Ordering::Relaxed);

    if total > 0 {
        crate::debug!(
            3,
            "Time breakdown: pos0={:.0}% suffix={:.0}% hash={:.0}%",
            pos0_time.load(Ordering::Relaxed) as f64 / total as f64 * 100.0,
            suffix_time.load(Ordering::Relaxed) as f64 / total as f64 * 100.0,
            hash_time.load(Ordering::Relaxed) as f64 / total as f64 * 100.0
        );
    }

    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    groups.into_values().collect()
}
