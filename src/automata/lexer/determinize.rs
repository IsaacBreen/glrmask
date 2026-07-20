//! Lexer NFA -> DFA determinization for glrmask's byte-oriented DFA types.

use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;
use crate::ds::compressed_state_set::{CompressedStateSet, SparseStateSet};
use crate::ds::u8set::U8Set;

use super::dfa::DFA;
use super::nfa::{CompactNFA, NFA};

const EPSILON_CLOSURE_PRECOMPUTE_THRESHOLD: u32 = 1;

fn compute_subset_metadata(
    nfa: &NFA,
    subset: &CompressedStateSet,
    group_count: usize,
    reachable_groups: &[BitSet],
    single_group_finalizer_words: Option<&[u64]>,
    single_group_future_words: Option<&[u64]>,
) -> (BitSet, BitSet) {
    if group_count == 1 {
        let finalizer_words =
            single_group_finalizer_words.expect("single-group finalizer mask missing");
        let future_words = single_group_future_words.expect("single-group future mask missing");
        let mut has_finalizer = false;
        let mut has_future = false;
        for &(word_index, word) in &subset.words {
            let word_index = word_index as usize;
            has_finalizer |= word & finalizer_words[word_index] != 0;
            has_future |= word & future_words[word_index] != 0;
            if has_finalizer && has_future {
                break;
            }
        }
        let mut finalizers = BitSet::new(1);
        let mut future = BitSet::new(1);
        if has_finalizer {
            finalizers.set(0);
        }
        if has_future {
            future.set(0);
        }
        return (finalizers, future);
    }

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
    if group_count == 0 {
        return vec![BitSet::new(0); num_states];
    }
    if group_count == 1 {
        let mut reverse_adj = vec![Vec::new(); num_states];
        let mut reaches_final = vec![false; num_states];
        let mut queue = VecDeque::new();

        for (state_idx, state) in nfa.states.iter().enumerate() {
            for &target in &state.epsilon_transitions {
                reverse_adj[target as usize].push(state_idx);
            }
            for (_, target) in &state.transitions {
                reverse_adj[*target as usize].push(state_idx);
            }
            if !state.finalizers.is_empty() {
                reaches_final[state_idx] = true;
                queue.push_back(state_idx);
            }
        }

        while let Some(state) = queue.pop_front() {
            for &pred in &reverse_adj[state] {
                if !reaches_final[pred] {
                    reaches_final[pred] = true;
                    queue.push_back(pred);
                }
            }
        }

        return reaches_final
            .into_iter()
            .map(|reachable| {
                let mut bits = BitSet::new(1);
                if reachable {
                    bits.set(0);
                }
                bits
            })
            .collect();
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

fn is_epsilon_free_deterministic(nfa: &NFA) -> bool {
    nfa.states.iter().all(|state| {
        if !state.epsilon_transitions.is_empty() {
            return false;
        }

        let mut covered = U8Set::empty();
        for (set, _) in &state.transitions {
            if !covered.is_disjoint(set) {
                return false;
            }
            covered |= *set;
        }
        true
    })
}

fn determinize_epsilon_free_deterministic(nfa: &NFA, group_count: usize, reachable_groups: &[BitSet]) -> DFA {
    let num_states = nfa.states.len().max(1);
    let mut order = Vec::with_capacity(num_states);
    order.push(nfa.start_state as usize);
    for state_id in 0..nfa.states.len() {
        if state_id != nfa.start_state as usize {
            order.push(state_id);
        }
    }

    let mut remap = vec![0u32; nfa.states.len()];
    for (new_state, &old_state) in order.iter().enumerate() {
        remap[old_state] = new_state as u32;
    }

    let mut dfa = DFA::new(num_states);
    dfa.ensure_group_capacity(group_count);

    for (new_state, &old_state) in order.iter().enumerate() {
        let nfa_state = &nfa.states[old_state];
        let mut finalizers = BitSet::new(group_count);
        for &group in &nfa_state.finalizers {
            finalizers.set(group as usize);
        }

        let mut transitions = Vec::new();
        for (set, target) in &nfa_state.transitions {
            transitions.reserve(set.len());
            for byte in set.iter() {
                transitions.push((byte, remap[*target as usize]));
            }
        }
        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|entry| entry.0);
        }

        dfa.overwrite_state_metadata(new_state as u32, finalizers, reachable_groups[old_state].clone());
        dfa.set_transitions_from_sorted_entries(new_state as u32, transitions);
    }

    dfa
}

struct RemappedTransitions {
    offsets: Vec<u32>,
    class_sets: Vec<U8Set>,
    targets: Vec<u32>,
    source_word_masks: Vec<u64>,
}

fn build_remapped_transitions(nfa: &NFA, class_map: &[u8]) -> RemappedTransitions {
    let mut remapped_set_cache: FxHashMap<U8Set, U8Set> = FxHashMap::default();
    let mut offsets = Vec::with_capacity(nfa.states.len() + 1);
    let transition_count = nfa
        .states
        .iter()
        .map(|state| state.transitions.len())
        .sum();
    let mut class_sets = Vec::with_capacity(transition_count);
    let mut targets = Vec::with_capacity(transition_count);
    let mut source_word_masks = vec![0u64; nfa.states.len().div_ceil(64)];

    offsets.push(0);
    for (state_index, state) in nfa.states.iter().enumerate() {
        if !state.transitions.is_empty() {
            source_word_masks[state_index / 64] |= 1u64 << (state_index % 64);
        }
        for (set, target) in &state.transitions {
            let class_set = *remapped_set_cache.entry(*set).or_insert_with(|| {
                let mut class_set = U8Set::empty();
                for byte in set.iter() {
                    class_set.insert(class_map[byte as usize]);
                }
                class_set
            });
            class_sets.push(class_set);
            targets.push(*target);
        }
        offsets.push(class_sets.len() as u32);
    }

    RemappedTransitions {
        offsets,
        class_sets,
        targets,
        source_word_masks,
    }
}

fn dfs_selective_post_order(
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
        dfs_selective_post_order(
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

fn precompute_epsilon_closures(
    compact_nfa: &CompactNFA,
    out_degree: &[u32],
    retained_word_masks: &[u64],
) -> Vec<Option<CompressedStateSet>> {
    let num_states = out_degree.len();
    let mut high_degree_closures: Vec<Option<CompressedStateSet>> = vec![None; num_states];
    let mut visited = vec![false; num_states];
    let mut post_order = Vec::with_capacity(num_states);

    for state in 0..num_states {
        if out_degree[state] < EPSILON_CLOSURE_PRECOMPUTE_THRESHOLD {
            continue;
        }

        dfs_selective_post_order(
            state,
            compact_nfa,
            out_degree,
            EPSILON_CLOSURE_PRECOMPUTE_THRESHOLD,
            &mut visited,
            &mut post_order,
        );
    }

    let mut epsilon_stack = Vec::with_capacity(num_states);
    let mut closure = SparseStateSet::new(num_states);
    for &state in &post_order {
        closure.clear();
        closure.insert(state);
        epsilon_stack.push(state);

        while let Some(current) = epsilon_stack.pop() {
            let start = compact_nfa.epsilon_offsets[current] as usize;
            let end = compact_nfa.epsilon_offsets[current + 1] as usize;
            for index in start..end {
                let next = compact_nfa.epsilon_targets[index] as usize;
                if closure.insert(next) {
                    if let Some(ref precomputed) = high_degree_closures[next] {
                        closure.union_compressed(precomputed);
                    } else if out_degree[next] > 0 {
                        epsilon_stack.push(next);
                    }
                }
            }
        }

        high_degree_closures[state] = Some(CompressedStateSet::from_sparse_masked(
            &closure,
            retained_word_masks,
        ));
    }

    high_degree_closures
}

fn build_start_closure(
    start_state: usize,
    retained_states: &[bool],
    precomputed_closures: &[Option<CompressedStateSet>],
    num_states: usize,
) -> SparseStateSet {
    let mut closure = SparseStateSet::new(num_states);
    if let Some(precomputed) = &precomputed_closures[start_state] {
        closure.union_compressed(precomputed);
    } else if retained_states[start_state] {
        closure.insert(start_state);
    }
    closure
}

fn build_closed_state_targets(
    num_states: usize,
    retained_states: &[bool],
    precomputed_closures: &[Option<CompressedStateSet>],
) -> Vec<Option<CompressedStateSet>> {
    (0..num_states)
        .map(|state| {
            if let Some(closure) = &precomputed_closures[state] {
                Some(closure.clone())
            } else if retained_states[state] {
                let mut closure = CompressedStateSet::new();
                let word_index = (state / 64) as u32;
                let word = 1u64 << (state % 64);
                closure.words.push((word_index, word));
                closure.hash = (word_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                    ^ word.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                Some(closure)
            } else {
                None
            }
        })
        .collect()
}

fn collect_closed_transition_targets(
    subset: &CompressedStateSet,
    remapped_transitions: &RemappedTransitions,
    closed_state_targets: &[Option<CompressedStateSet>],
    transition_targets: &mut [SparseStateSet],
    used_classes: &mut Vec<usize>,
    seen_class: &mut [bool],
) {
    for &(word_index, subset_word) in &subset.words {
        let mut active_sources =
            subset_word & remapped_transitions.source_word_masks[word_index as usize];
        while active_sources != 0 {
            let bit = active_sources.trailing_zeros() as usize;
            active_sources &= active_sources - 1;
            let state = word_index as usize * 64 + bit;
            let start = remapped_transitions.offsets[state] as usize;
            let end = remapped_transitions.offsets[state + 1] as usize;
            for transition in start..end {
                let class_set = remapped_transitions.class_sets[transition];
                let next_state = remapped_transitions.targets[transition];
                let Some(closed_targets) = &closed_state_targets[next_state as usize] else {
                    continue;
                };
                for class_id in class_set.iter() {
                    let class_index = class_id as usize;
                    if !seen_class[class_index] {
                        seen_class[class_index] = true;
                        used_classes.push(class_index);
                    }
                    transition_targets[class_index].union_compressed(closed_targets);
                }
            }
        }
    }
}

impl NFA {
    pub(super) fn to_dfa(&self) -> DFA {
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let group_count = self
            .states
            .iter()
            .flat_map(|state| state.finalizers.iter())
            .max()
            .map(|group| *group as usize + 1)
            .unwrap_or(0);

        let reachable_started_at = profile.then(std::time::Instant::now);
        let reachable_groups = compute_reachable_groups(self, group_count);
        let reachable_ms = reachable_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let deterministic_no_epsilon = is_epsilon_free_deterministic(self);

        if deterministic_no_epsilon {
            return determinize_epsilon_free_deterministic(self, group_count, &reachable_groups);
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(group_count);

        let classes_started_at = profile.then(std::time::Instant::now);
        let (class_map, num_classes, class_members) = self.compute_equivalence_classes();
        let classes_ms = classes_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let remap_started_at = profile.then(std::time::Instant::now);
        let remapped_transitions = build_remapped_transitions(self, &class_map);
        let remap_ms = remap_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let compact_started_at = profile.then(std::time::Instant::now);
        let compact_nfa = self.build_compact_nfa();
        let compact_ms = compact_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let num_nfa_states = self.states.len();
        let out_degree: Vec<u32> = (0..num_nfa_states)
            .map(|state| compact_nfa.epsilon_offsets[state + 1] - compact_nfa.epsilon_offsets[state])
            .collect();

        let closure_precompute_started_at = profile.then(std::time::Instant::now);
        let retained_states = self
            .states
            .iter()
            .map(|state| !state.transitions.is_empty() || !state.finalizers.is_empty())
            .collect::<Vec<_>>();
        let mut retained_word_masks = vec![0u64; num_nfa_states.div_ceil(64)];
        for (state, &retained) in retained_states.iter().enumerate() {
            if retained {
                retained_word_masks[state / 64] |= 1u64 << (state % 64);
            }
        }
        let high_degree_closures =
            precompute_epsilon_closures(&compact_nfa, &out_degree, &retained_word_masks);
        let closed_state_targets = build_closed_state_targets(
            num_nfa_states,
            &retained_states,
            &high_degree_closures,
        );
        let closure_precompute_ms = closure_precompute_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let start_closure = build_start_closure(
            self.start_state as usize,
            &retained_states,
            &high_degree_closures,
            num_nfa_states,
        );

        let start_key = CompressedStateSet::from_sparse(&start_closure);
        let single_group_finalizer_words = (group_count == 1).then(|| {
            let mut words = vec![0u64; num_nfa_states.div_ceil(64)];
            for (state, nfa_state) in self.states.iter().enumerate() {
                if !nfa_state.finalizers.is_empty() {
                    words[state / 64] |= 1u64 << (state % 64);
                }
            }
            words
        });
        let single_group_future_words = (group_count == 1).then(|| {
            let mut words = vec![0u64; num_nfa_states.div_ceil(64)];
            for (state, groups) in reachable_groups.iter().enumerate() {
                if groups.contains(0) {
                    words[state / 64] |= 1u64 << (state % 64);
                }
            }
            words
        });
        let (start_finalizers, start_future) =
            compute_subset_metadata(
                self,
                &start_key,
                group_count,
                &reachable_groups,
                single_group_finalizer_words.as_deref(),
                single_group_future_words.as_deref(),
            );
        dfa.overwrite_state_metadata(0, start_finalizers, start_future);

        let start_key = Arc::new(start_key);
        let mut subset_map = FxHashMap::<Arc<CompressedStateSet>, u32>::default();
        subset_map.reserve(num_nfa_states);
        let mut subsets = Vec::<Arc<CompressedStateSet>>::with_capacity(num_nfa_states);
        let mut worklist = Vec::<u32>::with_capacity(num_nfa_states);
        subset_map.insert(Arc::clone(&start_key), 0);
        subsets.push(start_key);
        worklist.push(0);

        let mut transition_targets: Vec<SparseStateSet> = (0..num_classes)
            .map(|_| SparseStateSet::new(num_nfa_states))
            .collect();
        let mut used_classes = Vec::with_capacity(num_classes);
        let mut seen_class = vec![false; num_classes];
        let mut scratch_closure = CompressedStateSet::new();

        let loop_started_at = profile.then(std::time::Instant::now);
        let mut total_subset_members = 0usize;
        let mut max_subset_members = 0usize;
        let mut total_used_classes = 0usize;
        let mut total_target_words = 0usize;
        let mut max_target_words = 0usize;
        let mut total_raw_target_members = 0usize;
        let mut max_raw_target_members = 0usize;
        let mut collect_duration = std::time::Duration::ZERO;
        let mut compress_duration = std::time::Duration::ZERO;
        let mut lookup_duration = std::time::Duration::ZERO;
        let mut metadata_duration = std::time::Duration::ZERO;
        let mut transition_duration = std::time::Duration::ZERO;
        while let Some(current_dfa_state) = worklist.pop() {
            let current_set = Arc::clone(&subsets[current_dfa_state as usize]);
            if profile {
                let members = current_set.len();
                total_subset_members += members;
                max_subset_members = max_subset_members.max(members);
            }
            let collect_started_at = profile.then(std::time::Instant::now);
            collect_closed_transition_targets(
                &current_set,
                &remapped_transitions,
                &closed_state_targets,
                &mut transition_targets,
                &mut used_classes,
                &mut seen_class,
            );
            if let Some(started) = collect_started_at {
                collect_duration += started.elapsed();
            }

            if profile {
                total_used_classes += used_classes.len();
            }
            let mut dfa_transitions_vec = Vec::with_capacity(used_classes.len() * 2);
            for &class_id in &used_classes {
                let target_set = &transition_targets[class_id];

                let compress_started_at = profile.then(std::time::Instant::now);
                CompressedStateSet::reuse_from_sparse(target_set, &mut scratch_closure);
                if profile {
                    let members = target_set
                        .dirty_words
                        .iter()
                        .map(|&word| target_set.words[word].count_ones() as usize)
                        .sum::<usize>();
                    total_raw_target_members += members;
                    max_raw_target_members = max_raw_target_members.max(members);
                }
                if let Some(started) = compress_started_at {
                    compress_duration += started.elapsed();
                }
                if scratch_closure.words.is_empty() {
                    continue;
                }
                if profile {
                    total_target_words += scratch_closure.words.len();
                    max_target_words = max_target_words.max(scratch_closure.words.len());
                }

                let lookup_started_at = profile.then(std::time::Instant::now);
                let next_dfa_state = if let Some(&existing) = subset_map.get(&scratch_closure) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let key = Arc::new(scratch_closure.clone());
                    let metadata_started_at = profile.then(std::time::Instant::now);
                    let (finalizers, future) = compute_subset_metadata(
                        self,
                        &key,
                        group_count,
                        &reachable_groups,
                        single_group_finalizer_words.as_deref(),
                        single_group_future_words.as_deref(),
                    );
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    if let Some(started) = metadata_started_at {
                        metadata_duration += started.elapsed();
                    }
                    subset_map.insert(Arc::clone(&key), new_state);
                    subsets.push(key);
                    worklist.push(new_state);
                    new_state
                };
                if let Some(started) = lookup_started_at {
                    lookup_duration += started.elapsed();
                }

                let transition_started_at = profile.then(std::time::Instant::now);
                for &byte in &class_members[class_id] {
                    dfa_transitions_vec.push((byte, next_dfa_state));
                }
                if let Some(started) = transition_started_at {
                    transition_duration += started.elapsed();
                }
            }

            let transition_started_at = profile.then(std::time::Instant::now);
            if dfa_transitions_vec.len() > 1 {
                dfa_transitions_vec.sort_unstable_by_key(|entry| entry.0);
            }
            dfa.set_transitions_from_sorted_entries(current_dfa_state, dfa_transitions_vec);
            if let Some(started) = transition_started_at {
                transition_duration += started.elapsed();
            }

            for &idx in &used_classes {
                seen_class[idx] = false;
                transition_targets[idx].clear();
            }
            used_classes.clear();
        }

        if let Some(total_started_at) = total_started_at {
            let states = dfa.num_states();
            eprintln!(
                "[glrmask/profile][tokenizer] nfa_determinize_detail nfa_states={} nfa_transitions={} epsilon_edges={} groups={} classes={} dfa_states={} dfa_transitions={} avg_subset_members={:.2} max_subset_members={} avg_used_classes={:.2} avg_closed_target_members={:.2} max_closed_target_members={} avg_target_words={:.2} max_target_words={} reachable_ms={:.3} classes_ms={:.3} remap_ms={:.3} compact_ms={:.3} closure_precompute_ms={:.3} collect_closed_ms={:.3} compress_ms={:.3} lookup_ms={:.3} metadata_ms={:.3} transition_ms={:.3} loop_ms={:.3} total_ms={:.3}",
                self.states.len(),
                self.states.iter().map(|state| state.transitions.len()).sum::<usize>(),
                self.states.iter().map(|state| state.epsilon_transitions.len()).sum::<usize>(),
                group_count,
                num_classes,
                states,
                dfa.states().iter().map(|state| state.transitions.len()).sum::<usize>(),
                total_subset_members as f64 / states.max(1) as f64,
                max_subset_members,
                total_used_classes as f64 / states.max(1) as f64,
                total_raw_target_members as f64 / total_used_classes.max(1) as f64,
                max_raw_target_members,
                total_target_words as f64 / total_used_classes.max(1) as f64,
                max_target_words,
                reachable_ms,
                classes_ms,
                remap_ms,
                compact_ms,
                closure_precompute_ms,
                collect_duration.as_secs_f64() * 1000.0,
                compress_duration.as_secs_f64() * 1000.0,
                lookup_duration.as_secs_f64() * 1000.0,
                metadata_duration.as_secs_f64() * 1000.0,
                transition_duration.as_secs_f64() * 1000.0,
                loop_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        dfa
    }
}
