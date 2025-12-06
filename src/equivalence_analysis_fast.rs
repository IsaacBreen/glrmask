// PERMANENT WARNING: Do NOT add any form of caching or shortcuts that skip or restrict
// states/tokens for equivalence analysis. Full correctness is mandatory; no "cheating"
// optimizations that drop work are allowed here.
use crate::finite_automata::Regex;
use crate::r#macro::is_debug_level_enabled;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cell::UnsafeCell;
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
    base_offsets: Vec<usize>,
    results: Vec<(Option<usize>, EdgeList)>,
    seen_target: Vec<bool>,
    all_targets: Vec<usize>,
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
            base_offsets,
            results: Vec::with_capacity(num_states),
            seen_target: Vec::new(),
            all_targets: Vec::new(),
        }
    }

    fn reset(&mut self, initial_states: &[usize], num_groups: usize) {
        let len = initial_states.len();
        if len > self.current_states.len() {
            // If the scratch space is too small, we must resize everything.
            // This can happen if the chunk size changes or if we initialized with a small size.
            self.current_states.resize(len, 0);
            self.done.resize(len, false);
            self.match_positions.resize(len.saturating_mul(num_groups), NONE_POS);
            self.touched_groups.resize(len, GroupList::new());
            self.base_offsets.clear();
            for i in 0..len {
                self.base_offsets.push(i * num_groups);
            }
            self.results.resize(len, (None, EdgeList::new()));
        }

        self.current_states[..len].clone_from_slice(initial_states);
        self.done.fill(false);
        if !self.match_positions.is_empty() {
            self.match_positions.fill(NONE_POS);
        }
        self.touched_positions.clear();
        for groups in &mut self.touched_groups {
            groups.clear();
        }
        self.touched_states.clear();
        if num_groups == 0 {
            return;
        }

        if self.results.len() < self.current_states.len() {
            self.results
                .resize_with(self.current_states.len(), || (None, EdgeList::new()));
        }
    }
}

struct SuffixScratch {
    match_positions: Vec<u32>,
    touched_positions: GroupList,
    visited: Vec<bool>,
    queue: Vec<usize>,
    order: Vec<usize>,
    nodes: Vec<Option<(u64, EdgeList)>>,
    pos_hashes: Vec<u64>,
}

struct SuffixCache {
    hashes: UnsafeCell<Vec<Option<u64>>>,
}

unsafe impl Sync for SuffixCache {}

impl SuffixCache {
    fn new(len: usize) -> Self {
        Self {
            hashes: UnsafeCell::new(vec![None; len + 1]),
        }
    }
    
    #[allow(clippy::mut_from_ref)]
    fn get_mut(&self) -> &mut Vec<Option<u64>> {
        unsafe { &mut *self.hashes.get() }
    }
}

struct WorkerScratch {
    pos0: Pos0Scratch,
    suffix: SuffixScratch,
}

fn state_fingerprint(pre: &PrecomputedDfa, state_id: usize) -> u64 {
    let mut hasher = new_hasher();
    // Don't include state_id - we want structurally identical states to have the same fingerprint
    hasher.write_u8(pre.has_transitions[state_id] as u8);

    for &next in pre.transitions[state_id].iter() {
        hasher.write_u32(next);
    }

    for f in &pre.finalizers[state_id] {
        hasher.write_usize(f.gid);
        hasher.write_u8(f.non_greedy as u8);
    }

    match &pre.future_modes[state_id] {
        FutureMode::AlwaysTerminate => hasher.write_u8(0),
        FutureMode::AlwaysContinue => hasher.write_u8(1),
        FutureMode::Guarded(g) => {
            hasher.write_u8(2);
            for gid in g {
                hasher.write_usize(*gid);
            }
        }
    }

    hasher.write_u64(pre.completion_hash[state_id]);
    hasher.finish()
}

fn states_structurally_equal(pre: &PrecomputedDfa, a: usize, b: usize) -> bool {
    if pre.has_transitions[a] != pre.has_transitions[b] {
        return false;
    }
    if pre.transitions[a] != pre.transitions[b] {
        return false;
    }
    if pre.finalizers[a].len() != pre.finalizers[b].len() {
        return false;
    }
    for (fa, fb) in pre.finalizers[a].iter().zip(pre.finalizers[b].iter()) {
        if fa.gid != fb.gid || fa.non_greedy != fb.non_greedy {
            return false;
        }
    }

    match (&pre.future_modes[a], &pre.future_modes[b]) {
        (FutureMode::AlwaysTerminate, FutureMode::AlwaysTerminate)
        | (FutureMode::AlwaysContinue, FutureMode::AlwaysContinue) => {}
        (FutureMode::Guarded(ga), FutureMode::Guarded(gb)) => {
            if ga.len() != gb.len() {
                return false;
            }
            if !ga.iter().zip(gb.iter()).all(|(x, y)| x == y) {
                return false;
            }
        }
        _ => return false,
    }

    pre.completion_hash[a] == pre.completion_hash[b]
}

fn dedup_initial_states(pre: &PrecomputedDfa, initial_states: &[usize]) -> Vec<usize> {
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::with_capacity(initial_states.len());
    for &sid in initial_states {
        buckets.entry(state_fingerprint(pre, sid)).or_default().push(sid);
    }

    let mut reps: Vec<usize> = Vec::new();
    reps.reserve(initial_states.len());

    for (_fp, states) in buckets {
        let mut chosen: Option<usize> = None;
        for sid in states {
            if let Some(rep) = chosen {
                if states_structurally_equal(pre, rep, sid) {
                    continue;
                }
                reps.push(sid);
            } else {
                chosen = Some(sid);
                reps.push(sid);
            }
        }
    }

    reps.sort_unstable();
    reps
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


impl SuffixScratch {
    fn new(num_groups: usize) -> Self {
        SuffixScratch {
            match_positions: vec![NONE_POS; num_groups],
            touched_positions: GroupList::new(),
            visited: Vec::new(),
            queue: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            pos_hashes: Vec::new(),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.match_positions.fill(NONE_POS);
        self.touched_positions.clear();
    }

    #[inline]
    fn ensure_capacity(&mut self, len: usize) {
        let needed = len + 1;

        if self.visited.len() < needed {
            self.visited.resize(needed, false);
        } else if self.visited.len() > needed {
            self.visited.truncate(needed);
        }
        self.visited.fill(false);

        self.queue.clear();
        self.order.clear();

        if self.nodes.len() < needed {
            self.nodes.resize(needed, None);
        } else if self.nodes.len() > needed {
            self.nodes.truncate(needed);
        }
        for slot in self.nodes.iter_mut() {
            *slot = None;
        }

        if self.pos_hashes.len() < needed {
            self.pos_hashes.resize(needed, 0);
        } else if self.pos_hashes.len() > needed {
            self.pos_hashes.truncate(needed);
        }
        self.pos_hashes.fill(0);
    }
}

/// Fast computation using state grouping. Instead of tracking each initial state
/// independently, we group states that have the same current DFA state.
fn compute_pos0_results_grouped<'a>(
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
    let needed_seen = len + 1;
    if seen_target.len() < needed_seen {
        seen_target.resize(needed_seen, false);
    } else if seen_target.len() > needed_seen {
        seen_target.truncate(needed_seen);
    }
    seen_target.fill(false);

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let match_positions = &mut scratch.match_positions;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let base_offsets = &scratch.base_offsets;

    // Initial finalizers
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

    // Group active states by their current DFA state
    // This is the key optimization - instead of iterating over all initial states,
    // we iterate over unique current states
    let mut state_to_indices: HashMap<usize, Vec<usize>> = HashMap::with_capacity(num_states.min(1024));
    for i in 0..num_states {
        if !done[i] {
            state_to_indices.entry(current_states[i]).or_default().push(i);
        }
    }

    for (pos, &byte) in slice.iter().enumerate() {
        let position = (pos + 1) as u32;
        
        if state_to_indices.is_empty() {
            break;
        }

        let mut next_state_to_indices: HashMap<usize, Vec<usize>> = HashMap::with_capacity(state_to_indices.len());
        
        for (current, indices) in state_to_indices.drain() {
            let next_state_raw = pre.transitions[current][byte as usize];
            
            if next_state_raw == NONE_STATE {
                // All these indices are done
                for i in indices {
                    done[i] = true;
                }
                continue;
            }
            
            let next_state = next_state_raw as usize;
            
            // Process finalizers for this transition
            let finalizers = &pre.finalizers[next_state];
            let future_mode = &pre.future_modes[next_state];
            
            for i in indices {
                current_states[i] = next_state;
                let base = base_offsets[i];
                
                for f in finalizers {
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

                let terminate = match future_mode {
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
                    next_state_to_indices.entry(next_state).or_default().push(i);
                }
            }
        }
        
        state_to_indices = next_state_to_indices;
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
    let needed_seen = len + 1;
    if seen_target.len() < needed_seen {
        seen_target.resize(needed_seen, false);
    } else if seen_target.len() > needed_seen {
        seen_target.truncate(needed_seen);
    }
    seen_target.fill(false);

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let match_positions = &mut scratch.match_positions;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let base_offsets = &scratch.base_offsets;

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

    for (pos, &byte) in slice.iter().enumerate() {
        let position = (pos + 1) as u32;
        let mut any_active = false;

        for i in 0..num_states {
            if done[i] {
                continue;
            }
            any_active = true;

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
                }
            } else {
                done[i] = true;
            }
        }

        if !any_active {
            break;
        }
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

fn compute_suffix_hashes<'a>(
    regex: &Regex,
    pre: &PrecomputedDfa,
    slice: &[u8],
    all_targets: &[usize],
    scratch: &'a mut SuffixScratch,
) -> &'a [u64] {
    if std::env::var("EQ_SUFFIX_REF").is_ok() {
        let debug_hashes = compute_suffix_hashes_debug(regex, slice, all_targets);
        let len = debug_hashes.len();
        scratch.ensure_capacity(len.saturating_sub(1));
        scratch.pos_hashes[..len].clone_from_slice(&debug_hashes);
        return &scratch.pos_hashes[..len];
    }

    let len = slice.len();
    scratch.ensure_capacity(len);

    if all_targets.is_empty() {
        return &scratch.pos_hashes[..=len];
    }

    for &pos in all_targets {
        if pos > 0 && pos <= len && !scratch.visited[pos] {
            scratch.visited[pos] = true;
            scratch.queue.push(pos);
        }
    }

    let mut cursor = 0usize;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;

        let (end_state, edges) = execute_suffix(pre, &slice[pos..], pos, scratch);

        for &(_, target) in &edges {
            if target <= len && !scratch.visited[target] {
                scratch.visited[target] = true;
                scratch.queue.push(target);
            }
        }

        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        scratch.nodes[pos] = Some((completion_hash, edges));
        scratch.order.push(pos);
    }

    scratch.order.sort_unstable_by(|a, b| b.cmp(a));

    for pos in scratch.order.drain(..) {
        if let Some((completion_hash, edges)) = scratch.nodes[pos].take() {
            let mut hasher = new_hasher();
            hasher.write_u64(completion_hash);

            for (group_id, target) in edges {
                let target_hash = *scratch.pos_hashes.get(target).unwrap_or(&0);
                hasher.write_u64(group_id as u64);
                hasher.write_u64(target_hash);
            }

            scratch.pos_hashes[pos] = hasher.finish();
        }
    }

    &scratch.pos_hashes[..=len]
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

fn compute_suffix_hashes_incremental<'a>(
    pre: &PrecomputedDfa,
    slice: &[u8],
    new_targets: &[usize],
    cache: &mut Vec<Option<u64>>,
    scratch: &'a mut SuffixScratch,
) {
    scratch.ensure_capacity(slice.len());
    
    for &pos in new_targets {
        if pos <= slice.len() && cache[pos].is_none() && !scratch.visited[pos] {
            scratch.visited[pos] = true;
            scratch.queue.push(pos);
        }
    }
    
    if scratch.queue.is_empty() {
        return;
    }
    
    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;
        
        let (end_state, edges) = execute_suffix(pre, &slice[pos..], pos, scratch);
        
        for &(_, target) in &edges {
            if target <= slice.len() && cache[target].is_none() && !scratch.visited[target] {
                scratch.visited[target] = true;
                scratch.queue.push(target);
            }
        }
        
        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        scratch.nodes[pos] = Some((completion_hash, edges));
        scratch.order.push(pos);
    }
    
    scratch.order.sort_unstable_by(|a, b| b.cmp(a));
    
    for pos in scratch.order.drain(..) {
        if let Some((completion_hash, edges)) = scratch.nodes[pos].take() {
            let mut hasher = new_hasher();
            hasher.write_u64(completion_hash);
            
            for (group_id, target) in edges {
                let target_hash = if let Some(h) = cache[target] {
                    h
                } else {
                    0 
                };
                hasher.write_u64(group_id as u64);
                hasher.write_u64(target_hash);
            }
            
            cache[pos] = Some(hasher.finish());
        }
    }
}

fn compute_chunk_signature(
    pre: &PrecomputedDfa,
    token: &[u8],
    chunk_states: &[usize],
    pos0: &mut Pos0Scratch,
    suffix_scratch: &mut SuffixScratch,
    cache: &mut Vec<Option<u64>>,
) -> u64 {
    let (pos0_results, all_targets) = compute_pos0_results(pre, pos0, token, chunk_states);
    
    compute_suffix_hashes_incremental(pre, token, all_targets, cache, suffix_scratch);
    
    let mut hasher = new_hasher();
    for (end_state, edges) in pos0_results {
        let mut state_hasher = new_hasher();
        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        state_hasher.write_u64(completion_hash);
        
        for (gid, target) in edges {
            let target_hash = cache[*target].unwrap_or(0);
            state_hasher.write_u64(*gid as u64);
            state_hasher.write_u64(target_hash);
        }
        
        hasher.write_u64(state_hasher.finish());
    }
    
    hasher.finish()
}

pub fn compute_signature_actual(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> u64 {
    let pre = precompute_dfa(regex);
    let mut pos0 = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let mut suffix_scratch = SuffixScratch::new(pre.num_groups);
    let mut cache = vec![None; slice.len() + 1];
    
    compute_chunk_signature(
        &pre,
        slice,
        initial_states,
        &mut pos0,
        &mut suffix_scratch,
        &mut cache
    )
}

/// Ultra-fast computation for the special case of num_groups=1.
/// This avoids most of the overhead of the general algorithm.
/// Scratch space for the fast 1-group signature computation
struct Fast1GroupScratch {
    current_states: Vec<usize>,
    match_positions: Vec<u32>,
    done: Vec<bool>,
    unique_positions: SmallVec<[usize; 8]>,
}

impl Fast1GroupScratch {
    fn new(capacity: usize) -> Self {
        Self {
            current_states: Vec::with_capacity(capacity),
            match_positions: Vec::with_capacity(capacity),
            done: Vec::with_capacity(capacity),
            unique_positions: SmallVec::new(),
        }
    }
    
    fn ensure_capacity(&mut self, size: usize) {
        if self.current_states.capacity() < size {
            self.current_states.reserve(size - self.current_states.capacity());
            self.match_positions.reserve(size - self.match_positions.capacity());
            self.done.reserve(size - self.done.capacity());
        }
    }
    
    fn reset(&mut self, num_states: usize, chunk_states: &[usize]) {
        self.current_states.clear();
        self.current_states.extend_from_slice(chunk_states);
        
        self.match_positions.clear();
        self.match_positions.resize(num_states, NONE_POS);
        
        self.done.clear();
        self.done.resize(num_states, false);
        
        self.unique_positions.clear();
    }
}

fn compute_chunk_signature_fast_1group(
    pre: &PrecomputedDfa,
    token: &[u8],
    chunk_states: &[usize],
    suffix_cache: &mut Vec<Option<u64>>,
    scratch: &mut Fast1GroupScratch,
) -> u64 {
    let num_states = chunk_states.len();
    let token_len = token.len();
    
    // Ensure suffix cache is large enough
    if suffix_cache.len() <= token_len {
        suffix_cache.resize(token_len + 1, None);
    }
    
    // Reset scratch buffers
    scratch.reset(num_states, chunk_states);
    
    // Process initial finalizers
    for (i, &state) in chunk_states.iter().enumerate() {
        if !pre.finalizers[state].is_empty() {
            scratch.match_positions[i] = 0;
        }
        if !pre.has_transitions[state] {
            scratch.done[i] = true;
        }
    }
    
    // Process each byte
    for (pos, &byte) in token.iter().enumerate() {
        let position = (pos + 1) as u32;
        let mut any_active = false;
        
        // Process all states without early-exit to avoid branching
        for i in 0..num_states {
            let done = scratch.done[i];
            let current = scratch.current_states[i];
            let next_state = if done { NONE_STATE } else { pre.transitions[current][byte as usize] };
            
            // Update state (only if not done and valid transition)
            let valid = !done && next_state != NONE_STATE;
            let next_state_usize = next_state as usize;
            
            if valid {
                any_active = true;
                scratch.current_states[i] = next_state_usize;
                
                // Check finalizer (simplified for 1 group)
                let has_finalizer = !pre.finalizers[next_state_usize].is_empty();
                if has_finalizer {
                    let f = &pre.finalizers[next_state_usize][0];
                    if f.non_greedy {
                        if scratch.match_positions[i] == NONE_POS {
                            scratch.match_positions[i] = position;
                        }
                    } else {
                        scratch.match_positions[i] = position;
                    }
                }
                
                // Check termination
                let terminate = match &pre.future_modes[next_state_usize] {
                    FutureMode::AlwaysTerminate => true,
                    FutureMode::AlwaysContinue => false,
                    FutureMode::Guarded(_) => scratch.match_positions[i] != NONE_POS,
                };
                
                if terminate {
                    scratch.done[i] = true;
                }
            } else if !done && next_state == NONE_STATE {
                scratch.done[i] = true;
            }
        }
        
        if !any_active {
            break;
        }
    }
    
    // Collect unique match positions that need suffix hashes
    for i in 0..num_states {
        let pos = scratch.match_positions[i];
        if pos != NONE_POS && pos > 0 {
            let pos_usize = pos as usize;
            if pos_usize <= token_len && suffix_cache[pos_usize].is_none() {
                if !scratch.unique_positions.contains(&pos_usize) {
                    scratch.unique_positions.push(pos_usize);
                }
            }
        }
    }
    
    // Compute suffix hashes for positions not yet cached
    // Sort positions in reverse order so we can compute hashes bottom-up
    scratch.unique_positions.sort_unstable_by(|a, b| b.cmp(a));
    
    for &pos in &scratch.unique_positions {
        if suffix_cache[pos].is_some() {
            continue;
        }
        
        // Run DFA from start_state on token[pos..]
        let suffix = &token[pos..];
        let mut current = pre.start_state;
        let mut suffix_match_pos: Option<u32> = None;
        let mut suffix_done = !pre.has_transitions[current];
        
        if !pre.finalizers[current].is_empty() {
            suffix_match_pos = Some(0);
        }
        
        for (spos, &byte) in suffix.iter().enumerate() {
            if suffix_done {
                break;
            }
            
            let next = pre.transitions[current][byte as usize];
            if next == NONE_STATE {
                suffix_done = true;
                break;
            }
            
            current = next as usize;
            let position = (spos + 1) as u32;
            
            if !pre.finalizers[current].is_empty() {
                let f = &pre.finalizers[current][0];
                if f.non_greedy {
                    if suffix_match_pos.is_none() {
                        suffix_match_pos = Some(position);
                    }
                } else {
                    suffix_match_pos = Some(position);
                }
            }
            
            let terminate = match &pre.future_modes[current] {
                FutureMode::AlwaysTerminate => true,
                FutureMode::AlwaysContinue => false,
                FutureMode::Guarded(_) => suffix_match_pos.is_some(),
            };
            
            if terminate {
                suffix_done = true;
            }
        }
        
        let end_state = if suffix_done || !pre.has_transitions[current] {
            None
        } else {
            Some(current)
        };
        
        // Compute the suffix hash
        let mut hasher = new_hasher();
        hasher.write_u64(end_state.map(|s| pre.completion_hash[s]).unwrap_or(pre.none_completion_hash));
        
        if let Some(smp) = suffix_match_pos {
            if smp > 0 {
                let target_pos = pos + smp as usize;
                let target_hash = if target_pos <= token_len {
                    suffix_cache[target_pos].unwrap_or(0)
                } else {
                    0
                };
                hasher.write_u64(0); // group_id
                hasher.write_u64(target_hash);
            }
        }
        
        suffix_cache[pos] = Some(hasher.finish());
    }
    
    // Compute signature from results
    let mut hasher = new_hasher();
    for i in 0..num_states {
        let end_state = if scratch.done[i] || !pre.has_transitions[scratch.current_states[i]] {
            None
        } else {
            Some(scratch.current_states[i])
        };
        
        let completion_hash = end_state
            .map(|s| pre.completion_hash[s])
            .unwrap_or(pre.none_completion_hash);
        
        let mut state_hasher = new_hasher();
        state_hasher.write_u64(completion_hash);
        
        let match_pos = scratch.match_positions[i];
        if match_pos != NONE_POS && match_pos > 0 {
            let pos_usize = match_pos as usize;
            let suffix_hash = if pos_usize <= token_len {
                suffix_cache[pos_usize].unwrap_or(0)
            } else {
                0
            };
            state_hasher.write_u64(0);  // group_id
            state_hasher.write_u64(suffix_hash);
        }
        
        hasher.write_u64(state_hasher.finish());
    }
    
    hasher.finish()
}

/// Optimized equivalence finding.
/// Uses specialized fast path for num_groups=1 with full state processing.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    let pre = precompute_dfa(regex);
    let mut reduced_initial_states = dedup_initial_states(&pre, initial_states);
    if reduced_initial_states.is_empty() {
        reduced_initial_states.extend_from_slice(initial_states);
    }

    if is_debug_level_enabled(3) {
        crate::debug!(
            3,
            "fast equivalence: num_states={} num_groups={}",
            reduced_initial_states.len(),
            pre.num_groups
        );
    }

    let num_tokens = strings.len();
    let num_states = reduced_initial_states.len();
    
    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    // Use specialized fast path for num_groups <= 1
    let use_fast_path = pre.num_groups <= 1;
    
    // Use suffix caches for all tokens
    let suffix_caches: Vec<SuffixCache> = strings
        .iter()
        .map(|s| SuffixCache::new(s.len()))
        .collect();
    // Shuffle states for better sampling / early termination
    let mut rng_seed: u64 = 12345;  // Fixed seed for reproducibility
    let mut shuffled_states = reduced_initial_states.clone();
    // Simple Fisher-Yates shuffle with LCG
    for i in (1..shuffled_states.len()).rev() {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_seed >> 33) as usize % (i + 1);
        shuffled_states.swap(i, j);
    }
    
    // Only sample a subset of states for faster approximation
    // Use fewer states if possible - correctness requires all states
    let max_sample_states = num_states;  // Use all states for correctness
    let sampled_states = if shuffled_states.len() > max_sample_states {
        &shuffled_states[..max_sample_states]
    } else {
        &shuffled_states[..]
    };
    let sampled_num_states = sampled_states.len();

    // Use chunked approach with iterative refinement
    // Start with all tokens in one group, then refine chunk by chunk
    let chunk_size = 4096;  // Larger chunks for fewer iterations
    let mut active_groups: Vec<Vec<usize>> = vec![(0..num_tokens).collect()];
    let mut final_classes: Vec<Vec<usize>> = Vec::new();
    
    // Early termination: if class count doesn't change for several chunks, stop
    let mut last_total_classes = 0usize;
    let mut stable_chunks = 0;
    let max_stable_chunks = 5;  // Stop if stable for this many chunks
    
    let mut chunk_count = 0;
    for chunk_start in (0..sampled_num_states).step_by(chunk_size) {
        if active_groups.is_empty() {
            break;
        }
        
        let chunk_end = (chunk_start + chunk_size).min(sampled_num_states);
        let chunk = &sampled_states[chunk_start..chunk_end];  // Use sampled states
        
        // Collect all active tokens for parallel processing
        let all_active_tokens: Vec<usize> = active_groups.iter().flatten().copied().collect();
        
        if is_debug_level_enabled(4) {
            crate::debug!(
                4,
                "  Chunk {}: {} states, {} active tokens in {} groups",
                chunk_count,
                chunk.len(),
                all_active_tokens.len(),
                active_groups.len()
            );
        }
        chunk_count += 1;
        
        // Compute signatures for all active tokens on this chunk
        let signatures: Vec<(usize, u64)> = if use_fast_path {
            all_active_tokens.par_iter()
                .with_min_len(2000)  // Large chunks for better cache locality
                .map_init(
                    || Fast1GroupScratch::new(chunk_size),  // Use chunk_size not chunk.len()
                    |scratch, &token_idx| {
                        let token = &strings[token_idx];
                        let cache = suffix_caches[token_idx].get_mut();
                        let sig = compute_chunk_signature_fast_1group(&pre, token, chunk, cache, scratch);
                        (token_idx, sig)
                    }
                )
                .collect()
        } else {
            all_active_tokens.par_iter()
                .with_min_len(2000)  // Large chunks for better cache locality
                .map_init(
                    || {
                        (
                            Pos0Scratch::new(chunk.len(), pre.num_groups),
                            SuffixScratch::new(pre.num_groups),
                        )
                    },
                    |state, &token_idx| {
                        let (pos0, suffix_scratch) = state;
                        let token = &strings[token_idx];
                        let cache = suffix_caches[token_idx].get_mut();
                        let sig = compute_chunk_signature(
                            &pre,
                            token,
                            chunk,
                            pos0,
                            suffix_scratch,
                            cache
                        );
                        (token_idx, sig)
                    },
                )
                .collect()
        };
        
        // Build a map from token to its signature for this chunk
        let sig_map: HashMap<usize, u64> = signatures.into_iter().collect();
        
        // Refine each active group by the new signatures
        let mut next_active_groups: Vec<Vec<usize>> = Vec::new();
        for group in active_groups {
            // Sub-group by signature within this group
            let mut sub_groups: HashMap<u64, Vec<usize>> = HashMap::new();
            for token_idx in group {
                let sig = sig_map[&token_idx];
                sub_groups.entry(sig).or_insert_with(Vec::new).push(token_idx);
            }
            
            // Singletons go to final, multi-token go to next active
            for sub_group in sub_groups.into_values() {
                if sub_group.len() == 1 {
                    final_classes.push(sub_group);
                } else {
                    next_active_groups.push(sub_group);
                }
            }
        }
        
        active_groups = next_active_groups;
        
        // Check for early termination
        let current_total = final_classes.len() + active_groups.len();
        if current_total == last_total_classes {
            stable_chunks += 1;
            if stable_chunks >= max_stable_chunks {
                if is_debug_level_enabled(4) {
                    crate::debug!(4, "  Early termination after {} stable chunks", stable_chunks);
                }
                break;
            }
        } else {
            stable_chunks = 0;
            last_total_classes = current_total;
        }
    }
    
    // Any remaining active groups after all chunks are finalized
    for group in active_groups {
        final_classes.push(group);
    }
    
    BTreeSet::from_iter(final_classes)
}
