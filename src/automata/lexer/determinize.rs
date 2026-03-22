//! sep1-style lexer NFA → DFA determinization, adapted to glrmask's leaner DFA types.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;
use crate::ds::compressed_state_set::{CompressedStateSet, SparseStateSet};
use crate::ds::u8set::U8Set;

use super::dfa::DFA;
use super::nfa::{CompactNFA, NFA};

#[derive(Default)]
struct LexerDeterminizeProfile {
    num_nfa_states: usize,
    num_classes: usize,
    distinct_transition_sets: usize,
    precomputed_closures: usize,
    subsets_processed: usize,
    subset_state_total: usize,
    fast_singleton_hits: usize,
    reachable_groups_ms: std::time::Duration,
    class_remap_ms: std::time::Duration,
    closure_precompute_ms: std::time::Duration,
    subset_construction_ms: std::time::Duration,
}

fn sparse_to_sorted_vec(set: &SparseStateSet) -> Vec<u32> {
    let mut states = Vec::new();
    for &word_idx in &set.dirty_words {
        let mut word = set.words[word_idx];
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= !(1u64 << bit);
            states.push((word_idx * 64 + bit) as u32);
        }
    }
    states.sort_unstable();
    states
}

fn compute_subset_metadata(
    nfa: &NFA,
    subset: &CompressedStateSet,
    group_count: usize,
    reachable_groups: &[BitSet],
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(group_count);
    let mut future = BitSet::new(group_count);

    for state in subset.iter() {
        for &group in &nfa.states[state].finalizers {
            finalizers.set(group as usize);
        }
        future.union_with(&reachable_groups[state]);
    }

    (finalizers, future)
}

fn compute_reachable_groups(nfa: &NFA, group_count: usize) -> Vec<BitSet> {
    let num_states = nfa.states.len();
    if num_states == 0 {
        return Vec::new();
    }

    let adj: Vec<Vec<usize>> = nfa
        .states
        .iter()
        .map(|state| {
            let mut targets: Vec<usize> = state
                .epsilon_transitions
                .iter()
                .map(|&target| target as usize)
                .collect();
            targets.extend(state.transitions.iter().map(|(_, target)| *target as usize));
            targets.sort_unstable();
            targets.dedup();
            targets
        })
        .collect();

    let mut scc_id = vec![u32::MAX; num_states];
    let mut scc_count: u32 = 0;
    let mut index_counter: u32 = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; num_states];
    let mut lowlink = vec![0u32; num_states];
    let mut disc = vec![u32::MAX; num_states];
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

    for root in 0..num_states {
        if disc[root] != u32::MAX {
            continue;
        }

        dfs_stack.push((root, 0));
        disc[root] = index_counter;
        lowlink[root] = index_counter;
        index_counter += 1;
        stack.push(root);
        on_stack[root] = true;

        while let Some(&mut (state, ref mut child_index)) = dfs_stack.last_mut() {
            if *child_index < adj[state].len() {
                let next = adj[state][*child_index];
                *child_index += 1;
                if disc[next] == u32::MAX {
                    disc[next] = index_counter;
                    lowlink[next] = index_counter;
                    index_counter += 1;
                    stack.push(next);
                    on_stack[next] = true;
                    dfs_stack.push((next, 0));
                } else if on_stack[next] {
                    lowlink[state] = lowlink[state].min(disc[next]);
                }
            } else {
                if lowlink[state] == disc[state] {
                    while let Some(member) = stack.pop() {
                        on_stack[member] = false;
                        scc_id[member] = scc_count;
                        if member == state {
                            break;
                        }
                    }
                    scc_count += 1;
                }
                dfs_stack.pop();
                if let Some(&mut (parent, _)) = dfs_stack.last_mut() {
                    lowlink[parent] = lowlink[parent].min(lowlink[state]);
                }
            }
        }
    }

    let mut scc_reachable: Vec<BitSet> = (0..scc_count as usize)
        .map(|_| BitSet::new(group_count))
        .collect();
    for (state_idx, state) in nfa.states.iter().enumerate() {
        let sid = scc_id[state_idx] as usize;
        for &group in &state.finalizers {
            scc_reachable[sid].set(group as usize);
        }
    }

    let mut scc_successors: Vec<Vec<u32>> = vec![Vec::new(); scc_count as usize];
    for (state_idx, targets) in adj.iter().enumerate() {
        let src_scc = scc_id[state_idx];
        for &target in targets {
            let dst_scc = scc_id[target];
            if src_scc != dst_scc {
                scc_successors[src_scc as usize].push(dst_scc);
            }
        }
    }
    for successors in &mut scc_successors {
        successors.sort_unstable();
        successors.dedup();
    }

    let mut scc_predecessors: Vec<Vec<u32>> = vec![Vec::new(); scc_count as usize];
    let mut remaining_successors = vec![0usize; scc_count as usize];
    for (sid, successors) in scc_successors.iter().enumerate() {
        remaining_successors[sid] = successors.len();
        for &succ in successors {
            scc_predecessors[succ as usize].push(sid as u32);
        }
    }

    let mut queue: VecDeque<u32> = remaining_successors
        .iter()
        .enumerate()
        .filter_map(|(sid, &count)| (count == 0).then_some(sid as u32))
        .collect();

    while let Some(sid) = queue.pop_front() {
        let sid = sid as usize;
        let sid_reachable = scc_reachable[sid].clone();

        for &pred in &scc_predecessors[sid] {
            let pred = pred as usize;
            scc_reachable[pred].union_with(&sid_reachable);
            remaining_successors[pred] -= 1;
            if remaining_successors[pred] == 0 {
                queue.push_back(pred as u32);
            }
        }
    }

    (0..num_states)
        .map(|state_idx| scc_reachable[scc_id[state_idx] as usize].clone())
        .collect()
}

impl NFA {
    pub fn to_dfa(&self) -> DFA {
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_LEXER_DETERMINIZE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let mut profile = profile_enabled.then(LexerDeterminizeProfile::default);

        let group_count = self
            .states
            .iter()
            .flat_map(|state| state.finalizers.iter())
            .max()
            .map(|group| *group as usize + 1)
            .unwrap_or(0);

        let phase_started_at = profile_enabled.then(std::time::Instant::now);
        let reachable_groups = compute_reachable_groups(self, group_count);
        if let (Some(profile), Some(started_at)) = (profile.as_mut(), phase_started_at) {
            profile.num_nfa_states = self.states.len();
            profile.reachable_groups_ms = started_at.elapsed();
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(group_count);

        let phase_started_at = profile_enabled.then(std::time::Instant::now);
        let (class_map, num_classes, class_members) = self.compute_equivalence_classes();
        let mut remapped_set_cache: FxHashMap<U8Set, U8Set> = FxHashMap::default();
        let remapped_transitions: Vec<Vec<(U8Set, u32)>> = self
            .states
            .iter()
            .map(|state| {
                state
                    .transitions
                    .iter()
                    .map(|(set, target)| {
                        let class_set = *remapped_set_cache.entry(*set).or_insert_with(|| {
                            let mut class_set = U8Set::empty();
                            for byte in set.iter() {
                                class_set.insert(class_map[byte as usize]);
                            }
                            class_set
                        });
                        (class_set, *target)
                    })
                    .collect()
            })
            .collect();
        if let (Some(profile), Some(started_at)) = (profile.as_mut(), phase_started_at) {
            profile.num_classes = num_classes;
            profile.distinct_transition_sets = remapped_set_cache.len();
            profile.class_remap_ms = started_at.elapsed();
        }

        let phase_started_at = profile_enabled.then(std::time::Instant::now);
        let compact_nfa = self.build_compact_nfa();
        let num_nfa_states = self.states.len();
        let mut stack: Vec<usize> = Vec::with_capacity(num_nfa_states);

        let out_degree: Vec<u32> = (0..num_nfa_states)
            .map(|state| compact_nfa.epsilon_offsets[state + 1] - compact_nfa.epsilon_offsets[state])
            .collect();

        let precompute_threshold: u32 = std::env::var("DFA_EPS_PRECOMPUTE_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(1);

        let mut high_degree_closures: Vec<Option<Vec<u32>>> = vec![None; num_nfa_states];
        let mut visited = vec![false; num_nfa_states];
        let mut post_order = Vec::with_capacity(num_nfa_states);

        fn dfs_postorder_selective(
            state: usize,
            compact_nfa: &CompactNFA,
            out_degree: &[u32],
            threshold: u32,
            visited: &mut [bool],
            post_order: &mut Vec<usize>,
        ) {
            if visited[state] {
                return;
            }
            visited[state] = true;

            let start = compact_nfa.epsilon_offsets[state] as usize;
            let end = compact_nfa.epsilon_offsets[state + 1] as usize;
            for &target in &compact_nfa.epsilon_targets[start..end] {
                dfs_postorder_selective(
                    target as usize,
                    compact_nfa,
                    out_degree,
                    threshold,
                    visited,
                    post_order,
                );
            }

            if out_degree[state] >= threshold {
                post_order.push(state);
            }
        }

        for state in 0..num_nfa_states {
            if out_degree[state] >= precompute_threshold {
                dfs_postorder_selective(
                    state,
                    &compact_nfa,
                    &out_degree,
                    precompute_threshold,
                    &mut visited,
                    &mut post_order,
                );
            }
        }

        let mut temp_set = SparseStateSet::new(num_nfa_states);
        for &state in &post_order {
            temp_set.clear();
            temp_set.insert(state);
            stack.push(state);

            while let Some(current) = stack.pop() {
                let start = compact_nfa.epsilon_offsets[current] as usize;
                let end = compact_nfa.epsilon_offsets[current + 1] as usize;
                for index in start..end {
                    let next = compact_nfa.epsilon_targets[index] as usize;
                    if temp_set.insert(next) {
                        if let Some(ref closure) = high_degree_closures[next] {
                            temp_set.insert_many(closure);
                        } else if out_degree[next] > 0 {
                            stack.push(next);
                        }
                    }
                }
            }

            high_degree_closures[state] = Some(sparse_to_sorted_vec(&temp_set));
        }
        if let Some(profile) = profile.as_mut() {
            profile.precomputed_closures = high_degree_closures
                .iter()
                .filter(|closure| closure.is_some())
                .count();
        }

        let mut start_closure = SparseStateSet::new(num_nfa_states);
        start_closure.insert(self.start_state as usize);
        if out_degree[self.start_state as usize] > 0 {
            stack.push(self.start_state as usize);
            while let Some(current) = stack.pop() {
                let start = compact_nfa.epsilon_offsets[current] as usize;
                let end = compact_nfa.epsilon_offsets[current + 1] as usize;
                for index in start..end {
                    let next = compact_nfa.epsilon_targets[index] as usize;
                    if start_closure.insert(next) {
                        if out_degree[next] > 0 {
                            stack.push(next);
                        }
                    }
                }
            }
        }
        if let (Some(profile), Some(started_at)) = (profile.as_mut(), phase_started_at) {
            profile.closure_precompute_ms = started_at.elapsed();
        }

        let start_key = CompressedStateSet::from_sparse(&start_closure);
        let (start_finalizers, start_future) =
            compute_subset_metadata(self, &start_key, group_count, &reachable_groups);
        dfa.overwrite_state_metadata(0, start_finalizers, start_future);

        let mut subset_map = FxHashMap::<CompressedStateSet, u32>::default();
        let mut worklist = Vec::new();
        subset_map.insert(start_key.clone(), 0);
        worklist.push(start_key);

        let mut transition_targets: Vec<SparseStateSet> = (0..num_classes)
            .map(|_| SparseStateSet::new(num_nfa_states))
            .collect();
        let mut used_classes = Vec::with_capacity(num_classes);
        let mut seen_class = vec![false; num_classes];
        let mut closure_set = SparseStateSet::new(num_nfa_states);
        let mut scratch_closure = CompressedStateSet::new();
        let mut sort_scratch = Vec::with_capacity(1024);

        let phase_started_at = profile_enabled.then(std::time::Instant::now);
        while let Some(current_set) = worklist.pop() {
            if let Some(profile) = profile.as_mut() {
                profile.subsets_processed += 1;
                profile.subset_state_total += current_set.len();
            }
            let current_dfa_state = subset_map[&current_set];

            for state in current_set.iter() {
                for (class_set, next_state) in &remapped_transitions[state] {
                    for class_id in class_set.iter() {
                        let idx = class_id as usize;
                        if !seen_class[idx] {
                            seen_class[idx] = true;
                            used_classes.push(idx);
                        }
                        transition_targets[idx].insert(*next_state as usize);
                    }
                }
            }

            let mut dfa_transitions_vec = Vec::with_capacity(used_classes.len() * 2);
            for &class_id in &used_classes {
                let target_set = &transition_targets[class_id];

                let mut fast_singleton_state = None;
                if target_set.dirty_words.len() == 1 {
                    let word_idx = target_set.dirty_words[0];
                    let word = target_set.words[word_idx];
                    if word != 0 && (word & (word - 1)) == 0 {
                        let bit = word.trailing_zeros() as usize;
                        let state = word_idx * 64 + bit;
                        if out_degree[state] == 0 && high_degree_closures[state].is_none() {
                            fast_singleton_state = Some(state);
                            if let Some(profile) = profile.as_mut() {
                                profile.fast_singleton_hits += 1;
                            }
                        }
                    }
                }

                if fast_singleton_state.is_none() {
                    closure_set.clear();
                    let mut needs_bfs = false;

                    for &word_idx in &target_set.dirty_words {
                        let mut word = target_set.words[word_idx];
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            word &= !(1u64 << bit);
                            let next_state = word_idx * 64 + bit;

                            if let Some(ref closure) = high_degree_closures[next_state] {
                                closure_set.insert_many(closure);
                            } else {
                                closure_set.insert(next_state);
                                if out_degree[next_state] > 0 {
                                    needs_bfs = true;
                                    stack.push(next_state);
                                }
                            }
                        }
                    }

                    if needs_bfs {
                        while let Some(current) = stack.pop() {
                            let start = compact_nfa.epsilon_offsets[current] as usize;
                            let end = compact_nfa.epsilon_offsets[current + 1] as usize;
                            for index in start..end {
                                let next = compact_nfa.epsilon_targets[index] as usize;
                                if closure_set.insert(next) {
                                    if let Some(ref closure) = high_degree_closures[next] {
                                        closure_set.insert_many(closure);
                                    } else if out_degree[next] > 0 {
                                        stack.push(next);
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(state) = fast_singleton_state {
                    scratch_closure.words.clear();
                    let word_idx = (state >> 6) as u32;
                    let mask = 1u64 << (state & 0x3f);
                    scratch_closure.words.push((word_idx, mask));
                    scratch_closure.hash = (word_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                        ^ mask.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                } else {
                    CompressedStateSet::reuse_from_sparse(
                        &closure_set,
                        &mut scratch_closure,
                        &mut sort_scratch,
                    );
                }

                let next_dfa_state = if let Some(&existing) = subset_map.get(&scratch_closure) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let key = scratch_closure.clone();
                    let (finalizers, future) =
                        compute_subset_metadata(self, &key, group_count, &reachable_groups);
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    subset_map.insert(key.clone(), new_state);
                    worklist.push(key);
                    new_state
                };

                for &byte in &class_members[class_id] {
                    dfa_transitions_vec.push((byte, next_dfa_state));
                }
            }

            if dfa_transitions_vec.len() > 1 {
                dfa_transitions_vec.sort_unstable_by_key(|entry| entry.0);
            }
            dfa.set_transitions_from_sorted_entries(current_dfa_state, dfa_transitions_vec);

            for &idx in &used_classes {
                seen_class[idx] = false;
                transition_targets[idx].clear();
            }
            used_classes.clear();
        }
        if let (Some(profile), Some(started_at)) = (profile.as_mut(), phase_started_at) {
            profile.subset_construction_ms = started_at.elapsed();
        }

        if let Some(profile) = profile {
            eprintln!(
                "[glrmask/profile][lexer_determinize] nfa_states={} classes={} distinct_transition_sets={} dfa_states={} precomputed_closures={} subsets={} avg_subset_states={:.2} fast_singleton_hits={} reachable_groups_ms={:.3} class_remap_ms={:.3} closure_precompute_ms={:.3} subset_construction_ms={:.3}",
                profile.num_nfa_states,
                profile.num_classes,
                profile.distinct_transition_sets,
                subset_map.len(),
                profile.precomputed_closures,
                profile.subsets_processed,
                if profile.subsets_processed == 0 {
                    0.0
                } else {
                    profile.subset_state_total as f64 / profile.subsets_processed as f64
                },
                profile.fast_singleton_hits,
                profile.reachable_groups_ms.as_secs_f64() * 1000.0,
                profile.class_remap_ms.as_secs_f64() * 1000.0,
                profile.closure_precompute_ms.as_secs_f64() * 1000.0,
                profile.subset_construction_ms.as_secs_f64() * 1000.0,
            );
        }

        dfa
    }
}
