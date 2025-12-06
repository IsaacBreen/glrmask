use crate::finite_automata::Regex;
use crate::r#macro::is_debug_level_enabled;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;
type GroupList = SmallVec<[usize; 4]>;

#[derive(Clone, Copy)]
struct Finalizer {
    gid: usize,
    non_greedy: bool,
}

type FinalizerList = SmallVec<[Finalizer; 4]>;

#[derive(Clone)]
enum FutureMode {
    AlwaysTerminate,
    AlwaysContinue,
    Guarded(GroupList),
}

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE_STATE: u32 = u32::MAX;

#[inline]
fn new_hasher() -> AHasher {
    RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4).build_hasher()
}

#[inline]
fn hash_group_list(list: &[usize]) -> u64 {
    let mut hasher = new_hasher();
    hasher.write_u8(1);
    hasher.write_u64(list.len() as u64);
    for &value in list {
        hasher.write_u64(value as u64);
    }
    hasher.finish()
}

struct PrecomputedDfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<FinalizerList>,
    future_modes: Vec<FutureMode>,
    has_transitions: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

const NONE_POS: u32 = u32::MAX;

struct Pos0Scratch {
    current_states: Vec<usize>,
    done: Vec<bool>,
    match_positions: Vec<u32>,
    touched_groups: Vec<GroupList>,
    touched_positions: Vec<usize>,
    touched_states: Vec<usize>,
    active_states: Vec<usize>,
    next_active_states: Vec<usize>,
    base_offsets: Vec<usize>,
    results: Vec<(Option<usize>, EdgeList)>,
    seen_target: Vec<bool>,
    all_targets: Vec<usize>,
}

struct SuffixScratch {
    match_positions: Vec<u32>,
    touched_positions: GroupList,
}

fn precompute_dfa(regex: &Regex) -> PrecomputedDfa {
    let dfa = &regex.dfa;
    assert!(
        dfa.states.len() <= u32::MAX as usize,
        "DFA too large for packed transitions"
    );

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

    let mut transitions: Vec<[u32; 256]> = Vec::with_capacity(dfa.states.len());
    let mut finalizers: Vec<FinalizerList> = Vec::with_capacity(dfa.states.len());
    let mut possible_future: Vec<GroupList> = Vec::with_capacity(dfa.states.len());
    let mut has_transitions: Vec<bool> = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE_STATE; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = target as u32;
        }
        transitions.push(table);
        finalizers.push(
            state
                .finalizers
                .iter()
                .map(|gid| Finalizer {
                    gid,
                    non_greedy: false,
                })
                .collect::<FinalizerList>(),
        );
        possible_future.push(
            state
                .possible_future_group_ids
                .iter()
                .copied()
                .collect::<GroupList>(),
        );
        has_transitions.push(!state.transitions.is_empty());
    }

    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups {
            non_greedy_flags[gid] = true;
        }
    }

    for finals in &mut finalizers {
        for f in finals.iter_mut() {
            f.non_greedy = non_greedy_flags.get(f.gid).copied().unwrap_or(false);
        }
    }

    let future_modes: Vec<FutureMode> = possible_future
        .iter()
        .map(|future| {
            if future.is_empty() {
                return FutureMode::AlwaysTerminate;
            }

            let mut guard: GroupList = GroupList::new();
            for &gid in future {
                if gid >= num_groups {
                    return FutureMode::AlwaysContinue;
                }
                if !non_greedy_flags[gid] {
                    return FutureMode::AlwaysContinue;
                }
                guard.push(gid);
            }

            guard.sort_unstable();
            guard.dedup();

            FutureMode::Guarded(guard)
        })
        .collect();

    let none_completion_hash = {
        let mut hasher = new_hasher();
        hasher.write_u8(0);
        hasher.finish()
    };

    let mut completion_hash = Vec::with_capacity(possible_future.len());
    for vec in &possible_future {
        completion_hash.push(hash_group_list(vec));
    }

    PrecomputedDfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        future_modes,
        has_transitions,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

impl Pos0Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let base_offsets: Vec<usize> = (0..num_states)
            .map(|idx| idx.saturating_mul(num_groups))
            .collect();
        Pos0Scratch {
            current_states: vec![0; num_states],
            done: vec![false; num_states],
            match_positions: vec![NONE_POS; num_states.saturating_mul(num_groups)],
            touched_groups: vec![GroupList::new(); num_states],
            touched_positions: Vec::new(),
            touched_states: Vec::new(),
            active_states: Vec::with_capacity(num_states),
            next_active_states: Vec::with_capacity(num_states),
            base_offsets,
            results: Vec::with_capacity(num_states),
            seen_target: Vec::new(),
            all_targets: Vec::new(),
        }
    }

    fn reset(&mut self, initial_states: &[usize], num_groups: usize) {
        debug_assert_eq!(self.current_states.len(), initial_states.len());
        self.current_states.clone_from_slice(initial_states);
        self.done.fill(false);
        if !self.match_positions.is_empty() {
            self.match_positions.fill(NONE_POS);
        }
        self.touched_positions.clear();
        for groups in &mut self.touched_groups {
            groups.clear();
        }
        self.touched_states.clear();
        self.active_states.clear();
        self.next_active_states.clear();
        if num_groups == 0 {
            return;
        }

        if self.results.len() < self.current_states.len() {
            self.results
                .resize_with(self.current_states.len(), || (None, EdgeList::new()));
        }
    }
}

impl SuffixScratch {
    fn new(num_groups: usize) -> Self {
        SuffixScratch {
            match_positions: vec![NONE_POS; num_groups],
            touched_positions: GroupList::new(),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.match_positions.fill(NONE_POS);
        self.touched_positions.clear();
    }
}

fn compute_pos0_results<'a>(
    pre: &PrecomputedDfa,
    scratch: &'a mut Pos0Scratch,
    slice: &[u8],
    initial_states: &[usize],
) -> (&'a [(Option<usize>, EdgeList)], &'a [usize]) {
    let num_states = initial_states.len();
    let num_groups = pre.num_groups;
    let len = slice.len();

    scratch.reset(initial_states, num_groups);

    if scratch.results.len() < num_states {
        scratch
            .results
            .resize_with(num_states, || (None, EdgeList::new()));
    }
    for i in 0..num_states {
        scratch.results[i].0 = None;
        scratch.results[i].1.clear();
    }

    let all_targets = &mut scratch.all_targets;
    all_targets.clear();

    let seen_target = &mut scratch.seen_target;
    if seen_target.len() < len + 1 {
        seen_target.resize(len + 1, false);
    } else {
        for slot in seen_target.iter_mut().take(len + 1) {
            *slot = false;
        }
    }

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let match_positions = &mut scratch.match_positions;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let base_offsets = &scratch.base_offsets;
    let active_states = &mut scratch.active_states;
    let next_active_states = &mut scratch.next_active_states;

    for (i, &state) in initial_states.iter().enumerate() {
        let base = base_offsets[i];
        for f in &pre.finalizers[state] {
            let gid = f.gid;
            if gid < num_groups {
                let idx = base + gid;
                if match_positions[idx] == NONE_POS {
                    match_positions[idx] = 0;
                }
                let groups = &mut touched_groups[i];
                if !groups.contains(&gid) {
                    if groups.is_empty() {
                        touched_states.push(i);
                    }
                    groups.push(gid);
                }
                touched_positions.push(idx);
            }
        }
        if !pre.has_transitions[state] {
            if !done[i] {
                done[i] = true;
            }
        }
    }

    active_states.clear();
    next_active_states.clear();
    for i in 0..num_states {
        if !done[i] {
            active_states.push(i);
        }
    }

    for (pos, &byte) in slice.iter().enumerate() {
        if active_states.is_empty() {
            break;
        }
        let position = (pos + 1) as u32;

        next_active_states.clear();

        for &i in active_states.iter() {
            let base = base_offsets[i];
            let current = current_states[i];
            let next_state = pre.transitions[current][byte as usize];

            if next_state != NONE_STATE {
                let next_state = next_state as usize;
                current_states[i] = next_state;

                for f in &pre.finalizers[next_state] {
                    let gid = f.gid;
                    if gid < num_groups {
                        let idx = base + gid;
                        let slot = &mut match_positions[idx];
                        if f.non_greedy {
                            if *slot == NONE_POS {
                                *slot = position;
                            }
                        } else {
                            *slot = position;
                        }

                        let groups = &mut touched_groups[i];
                        if !groups.contains(&gid) {
                            if groups.is_empty() {
                                touched_states.push(i);
                            }
                            groups.push(gid);
                        }
                        touched_positions.push(idx);

                    }
                }

                let terminate = match &pre.future_modes[next_state] {
                    FutureMode::AlwaysTerminate => true,
                    FutureMode::AlwaysContinue => false,
                    FutureMode::Guarded(guard) => {
                        let mut all_met = true;
                        for &gid in guard.iter() {
                            let idx = base + gid;
                            if match_positions[idx] == NONE_POS {
                                all_met = false;
                                break;
                            }
                        }
                        all_met
                    }
                };

                if terminate {
                    done[i] = true;
                } else {
                    next_active_states.push(i);
                }
            } else {
                done[i] = true;
            }
        }

        std::mem::swap(active_states, next_active_states);
    }

    for i in 0..num_states {
        let end_state = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };

        let edges = &mut scratch.results[i].1;
        if num_groups > 0 {
            let base = base_offsets[i];
            for &gid in &touched_groups[i] {
                if gid >= num_groups {
                    continue;
                }
                let pos_val = match_positions[base + gid];
                if pos_val != NONE_POS && pos_val > 0 {
                    let pos_usize = pos_val as usize;
                    edges.push((gid, pos_usize));
                    if pos_usize <= len && !seen_target[pos_usize] {
                        seen_target[pos_usize] = true;
                        all_targets.push(pos_usize);
                    }
                }
            }
        }

        edges.sort_unstable_by_key(|e| e.0);
        scratch.results[i].0 = end_state;
    }

    (&scratch.results[..num_states], &scratch.all_targets)
}

#[inline]
fn execute_suffix(
    pre: &PrecomputedDfa,
    slice: &[u8],
    base_pos: usize,
    scratch: &mut SuffixScratch,
) -> (Option<usize>, EdgeList) {
    let num_groups = pre.num_groups;

    if num_groups > 0 {
        scratch.reset();
    }

    let match_positions = &mut scratch.match_positions;
    let touched = &mut scratch.touched_positions;

    let mut current = pre.start_state;
    let mut done = false;

    if num_groups > 0 {
        for f in &pre.finalizers[current] {
            let gid = f.gid;
            if gid < num_groups {
                let slot = &mut match_positions[gid];
                if f.non_greedy {
                    if *slot == NONE_POS {
                        *slot = 0;
                        touched.push(gid);
                    }
                } else {
                    let was_none = *slot == NONE_POS;
                    *slot = 0;
                    if was_none {
                        touched.push(gid);
                    }
                }
            }
        }
    }

    if !pre.has_transitions[current] {
        done = true;
    }

    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }

        let next_state = pre.transitions[current][byte as usize];
        if next_state != NONE_STATE {
            let next_state = next_state as usize;
            current = next_state;
            let position = (idx + 1) as u32;

            if num_groups > 0 {
                for f in &pre.finalizers[current] {
                    let gid = f.gid;
                    if gid < num_groups {
                        let slot = &mut match_positions[gid];
                        if f.non_greedy {
                            if *slot == NONE_POS {
                                *slot = position;
                            }
                        } else {
                            *slot = position;
                        }

                        if !touched.contains(&gid) {
                            touched.push(gid);
                        }

                    }
                }
            }

            let terminate = match &pre.future_modes[current] {
                FutureMode::AlwaysTerminate => true,
                FutureMode::AlwaysContinue => false,
                FutureMode::Guarded(guard) => {
                    guard.iter().all(|&gid| match_positions[gid] != NONE_POS)
                }
            };

            if terminate {
                done = true;
            }
        } else {
            done = true;
        }
    }

    let end_state = if done || !pre.has_transitions[current] {
        None
    } else {
        Some(current)
    };

    let mut edges: EdgeList = SmallVec::new();
    if num_groups > 0 {
        touched.sort_unstable();
        for &gid in touched.iter() {
            let pos_val = match_positions[gid];
            if pos_val != NONE_POS && pos_val != 0 {
                edges.push((gid, base_pos + pos_val as usize));
            }
        }
    }

    (end_state, edges)
}

fn compute_suffix_hashes(
    regex: &Regex,
    pre: &PrecomputedDfa,
    slice: &[u8],
    all_targets: &[usize],
) -> Vec<u64> {
    if std::env::var("EQ_SUFFIX_REF").is_ok() {
        use std::collections::VecDeque;

        let len = slice.len();
        let mut visited = vec![false; len + 1];
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut order: Vec<usize> = Vec::new();
        let mut nodes: Vec<Option<(Option<usize>, Vec<(usize, usize)>)>> = vec![None; len + 1];

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
                let mut hasher = new_hasher();
                let completion_hash = end_state
                    .map(|id| pre.completion_hash[id])
                    .unwrap_or(pre.none_completion_hash);
                hasher.write_u64(completion_hash);

                for (group_id, target) in edges {
                    let target_hash = pos_hashes[*target];
                    hasher.write_u64(*group_id as u64);
                    hasher.write_u64(target_hash);
                }

                pos_hashes[pos] = hasher.finish();
            }
        }

        return pos_hashes;
    }

    let len = slice.len();
    if all_targets.is_empty() {
        return vec![0; len + 1];
    }
    let mut visited = vec![false; len + 1];
    let mut queue: Vec<usize> = Vec::new();
    let mut cursor = 0usize;
    let mut order: Vec<usize> = Vec::new();
    let mut nodes: Vec<Option<(u64, EdgeList)>> = vec![None; len + 1];
    let mut scratch = SuffixScratch::new(pre.num_groups);

    for &pos in all_targets {
        if pos > 0 && pos <= len && !visited[pos] {
            visited[pos] = true;
            queue.push(pos);
        }
    }

    while cursor < queue.len() {
        let pos = queue[cursor];
        cursor += 1;
        let (end_state, edges) = execute_suffix(pre, &slice[pos..], pos, &mut scratch);

        for &(_, target) in &edges {
            if target <= len && !visited[target] {
                visited[target] = true;
                queue.push(target);
            }
        }

        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        nodes[pos] = Some((completion_hash, edges));
        order.push(pos);
    }

    order.sort_unstable_by(|a, b| b.cmp(a));
    let mut pos_hashes: Vec<u64> = vec![0; len + 1];

    for pos in order {
        if let Some((completion_hash, edges)) = nodes[pos].take() {
            let mut hasher = new_hasher();
            hasher.write_u64(completion_hash);

            for (group_id, target) in edges {
                let target_hash = pos_hashes[target];
                hasher.write_u64(group_id as u64);
                hasher.write_u64(target_hash);
            }

            pos_hashes[pos] = hasher.finish();
        }
    }

    pos_hashes
}

fn compute_final_signature(
    pre: &PrecomputedDfa,
    pos0_results: &[(Option<usize>, EdgeList)],
    pos_hashes: &[u64],
) -> u64 {
    let mut combined_hasher = new_hasher();

    for (end_state, edges) in pos0_results {
        let mut hasher = new_hasher();
        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        hasher.write_u64(completion_hash);

        for (group_id, target) in edges {
            let target_hash = *pos_hashes.get(*target).unwrap_or(&0);
            hasher.write_u64(*group_id as u64);
            hasher.write_u64(target_hash);
        }

        combined_hasher.write_u64(hasher.finish());
    }

    combined_hasher.finish()
}

fn compute_suffix_hashes_debug(
    regex: &Regex,
    slice: &[u8],
    all_targets: &[usize],
) -> Vec<u64> {
    use std::collections::VecDeque;

    let len = slice.len();
    if all_targets.is_empty() {
        return vec![0; len + 1];
    }

    let mut visited = vec![false; len + 1];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut order: Vec<usize> = Vec::new();
    let mut nodes: Vec<Option<(Option<usize>, EdgeList)>> = vec![None; len + 1];

    for &pos in all_targets {
        if pos > 0 && pos <= len && !visited[pos] {
            visited[pos] = true;
            queue.push_back(pos);
        }
    }

    while let Some(pos) = queue.pop_front() {
        let result = regex.execute_from_state_nonzero(&slice[pos..], regex.dfa.start_state);

        let mut edges: EdgeList = result
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
            let completion = end_state.map(|id| regex.dfa.states[id].possible_future_group_ids.clone());
            let mut hasher = DefaultHasher::new();
            completion.hash(&mut hasher);
            for (group_id, target) in edges {
                let target_hash = pos_hashes[*target];
                (group_id, target_hash).hash(&mut hasher);
            }
            pos_hashes[pos] = hasher.finish();
        }
    }

    pos_hashes
}

pub fn compute_signature_debug(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> Vec<u64> {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, all_targets) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    let pos_hashes = compute_suffix_hashes_debug(regex, slice, &all_targets);

    let mut signatures: Vec<u64> = Vec::with_capacity(initial_states.len());
    for (end_state, edges) in pos0_results.iter() {
        let completion = end_state.map(|id| regex.dfa.states[id].possible_future_group_ids.clone());
        let mut hasher = DefaultHasher::new();
        completion.hash(&mut hasher);
        for (group_id, target) in edges.iter() {
            let target_hash = *pos_hashes.get(*target).unwrap_or(&0);
            (group_id, target_hash).hash(&mut hasher);
        }
        signatures.push(hasher.finish());
    }

    signatures
}

pub fn debug_pos0_edges(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> Vec<EdgeList> {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, _) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    pos0_results
        .iter()
        .map(|(_, edges)| edges.clone())
        .collect()
}

pub fn compute_signature_actual(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> u64 {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, all_targets) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    let pos_hashes = compute_suffix_hashes(regex, &pre, slice, &all_targets);
    compute_final_signature(&pre, &pos0_results, &pos_hashes)
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    use std::sync::atomic::{AtomicU64, Ordering};

    let pre = precompute_dfa(regex);
    let track_timing = is_debug_level_enabled(3);
    if track_timing {
        crate::debug!(
            3,
            "fast equivalence: num_states={} num_groups={}",
            initial_states.len(),
            pre.num_groups
        );
    }
    let pos0_time = AtomicU64::new(0);
    let suffix_time = AtomicU64::new(0);
    let hash_time = AtomicU64::new(0);

    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);

            if track_timing {
                let t0 = std::time::Instant::now();
                let (pos0_results, all_targets) =
                    compute_pos0_results(&pre, &mut scratch, s, initial_states);
                let t1 = std::time::Instant::now();
                let pos_hashes = compute_suffix_hashes(regex, &pre, s, &all_targets);
                let t2 = std::time::Instant::now();
                let sig = compute_final_signature(&pre, &pos0_results, &pos_hashes);
                let t3 = std::time::Instant::now();

                pos0_time.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                suffix_time.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                hash_time.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);

                sig
            } else {
                let (pos0_results, all_targets) =
                    compute_pos0_results(&pre, &mut scratch, s, initial_states);
                let pos_hashes = compute_suffix_hashes(regex, &pre, s, &all_targets);
                compute_final_signature(&pre, &pos0_results, &pos_hashes)
            }
        })
        .collect();

    if let Ok(list) = std::env::var("EQ_DEBUG_COMPARE") {
        let indices: Vec<usize> = list
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        for &idx in &indices {
            if let Some(sig_par) = signatures.get(idx) {
                let sig_clean = compute_signature_actual(regex, &strings[idx], initial_states);
                if *sig_par != sig_clean {
                    eprintln!("EQ_DEBUG_COMPARE idx {} par_sig={} clean_sig={}", idx, sig_par, sig_clean);
                }
            }
        }
    }

    if track_timing {
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
    }

    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    groups.into_values().collect()
}
