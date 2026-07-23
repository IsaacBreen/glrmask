//! Byte-oriented lexer DFA used by the tokenizer and lexer compiler.

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::FxHashSet;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

use crate::ds::char_transitions::CharTransitions;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub(super) type GroupId = u32;
pub(super) const DEAD: u32 = u32::MAX;

fn resized_bitset(bits: &BitSet, num_groups: usize) -> BitSet {
    let mut resized = BitSet::new(num_groups);
    for bit in bits.iter() {
        resized.set(bit);
    }
    resized
}

fn project_bitset(bits: &BitSet, num_groups: usize) -> BitSet {
    let mut projected = BitSet::new(num_groups);
    for group_id in bits.iter().filter(|group_id| *group_id < num_groups) {
        projected.set(group_id);
    }
    projected
}

fn excluded_group_indices(
    finalizers: &BitSet,
    excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
) -> Vec<usize> {
    let mut to_clear = Vec::new();
    for (&group_id, blocked_by) in excludes {
        let group_index = group_id as usize;
        if !finalizers.contains(group_index) {
            continue;
        }
        if blocked_by
            .iter()
            .any(|blocked_by_id| finalizers.contains(*blocked_by_id as usize))
        {
            to_clear.push(group_index);
        }
    }
    to_clear
}

fn intersection_missing_group_indices(
    finalizers: &BitSet,
    intersections: &BTreeMap<GroupId, BTreeSet<GroupId>>,
) -> Vec<usize> {
    let mut to_clear = Vec::new();
    for (&group_id, required) in intersections {
        let group_index = group_id as usize;
        if !finalizers.contains(group_index) {
            continue;
        }
        if required
            .iter()
            .any(|required_id| !finalizers.contains(*required_id as usize))
        {
            to_clear.push(group_index);
        }
    }
    to_clear
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct DFAState {
    pub(super) transitions: CharTransitions<u32>,
    pub(super) finalizers: BitSet,
    possible_future_group_ids: BitSet,
    /// Epsilon transitions are the only source of lexer nondeterminism. Byte
    /// transitions remain deterministic within an individual physical state.
    pub(super) epsilon_transitions: Vec<u32>,
}

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct DFA {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
}

/// The historical persisted representation of a lexer state. Keep this exact
/// field order: existing bincode artifacts encode nested structs without field
/// boundaries, so adding even a trailing field would consume data belonging to
/// the next state or the enclosing DFA.
#[derive(Serialize, Deserialize)]
struct DfaStateWire {
    transitions: CharTransitions<u32>,
    finalizers: BitSet,
    possible_future_group_ids: BitSet,
}

/// The historical persisted representation of a lexer DFA. Epsilon metadata is
/// encoded as validated synthetic trailing states, leaving the outer wire shape
/// and every real state byte-for-byte compatible with pre-epsilon artifacts.
#[derive(Serialize, Deserialize)]
struct DfaWire {
    states: Vec<DfaStateWire>,
    group_id_to_u8set: Vec<U8Set>,
}

const EPSILON_WIRE_MARKER: u32 = u32::MAX;
const EPSILON_WIRE_EDGE: u32 = 0x4550_5345; // "EPSE"
const EPSILON_WIRE_TRAILER: u32 = 0x4550_5354; // "EPST"
const EPSILON_TARGETS_PER_WIRE_STATE: usize = 253;

impl From<&DFAState> for DfaStateWire {
    fn from(state: &DFAState) -> Self {
        Self {
            transitions: state.transitions.clone(),
            finalizers: state.finalizers.clone(),
            possible_future_group_ids: state.possible_future_group_ids.clone(),
        }
    }
}

impl From<DfaStateWire> for DFAState {
    fn from(state: DfaStateWire) -> Self {
        Self {
            transitions: state.transitions,
            finalizers: state.finalizers,
            possible_future_group_ids: state.possible_future_group_ids,
            epsilon_transitions: Vec::new(),
        }
    }
}

fn epsilon_wire_state(entries: Vec<(u8, u32)>) -> DfaStateWire {
    DfaStateWire {
        transitions: CharTransitions::from_sorted_entries(entries),
        finalizers: BitSet::new(0),
        possible_future_group_ids: BitSet::new(0),
    }
}

fn epsilon_wire_trailer(state: &DfaStateWire) -> Result<Option<(usize, usize)>, &'static str> {
    let marker = state.transitions.get(0).copied() == Some(EPSILON_WIRE_MARKER);
    let trailer = state.transitions.get(1).copied() == Some(EPSILON_WIRE_TRAILER);
    if !marker || !trailer {
        return Ok(None);
    }
    if !state.finalizers.is_empty()
        || !state.possible_future_group_ids.is_empty()
        || state.transitions.len() != 4
    {
        return Err("malformed lexer epsilon metadata trailer");
    }
    let real_states = state
        .transitions
        .get(2)
        .copied()
        .ok_or("lexer epsilon metadata trailer omitted real-state count")?
        as usize;
    let metadata_states = state
        .transitions
        .get(3)
        .copied()
        .ok_or("lexer epsilon metadata trailer omitted metadata-state count")?
        as usize;
    Ok(Some((real_states, metadata_states)))
}

fn decode_epsilon_wire_states(
    states: &mut Vec<DfaStateWire>,
) -> Result<Option<Vec<Vec<u32>>>, &'static str> {
    let Some(last) = states.last() else {
        return Ok(None);
    };
    let Some((real_state_count, metadata_state_count)) = epsilon_wire_trailer(last)? else {
        return Ok(None);
    };
    if metadata_state_count == 0
        || real_state_count
            .checked_add(metadata_state_count)
            .and_then(|count| count.checked_add(1))
            != Some(states.len())
    {
        return Err("lexer epsilon metadata counts do not match serialized state count");
    }

    let mut epsilon_transitions = vec![Vec::new(); real_state_count];
    let mut seen_targets = vec![None::<FxHashSet<u32>>; real_state_count];
    for state in &states[real_state_count..real_state_count + metadata_state_count] {
        if !state.finalizers.is_empty()
            || !state.possible_future_group_ids.is_empty()
            || state.transitions.len() < 4
            || state.transitions.get(0).copied() != Some(EPSILON_WIRE_MARKER)
            || state.transitions.get(1).copied() != Some(EPSILON_WIRE_EDGE)
        {
            return Err("malformed lexer epsilon metadata state");
        }
        let source = state
            .transitions
            .get(2)
            .copied()
            .ok_or("lexer epsilon metadata omitted source state")?
            as usize;
        if source >= real_state_count {
            return Err("lexer epsilon metadata source is out of range");
        }
        let mut targets_in_state = 0usize;
        for (index, (byte, &target)) in state.transitions.iter().skip(3).enumerate() {
            if byte as usize != index + 3 {
                return Err("lexer epsilon metadata target slots are not contiguous");
            }
            if target as usize >= real_state_count {
                return Err("lexer epsilon metadata target is out of range");
            }
            if !seen_targets[source]
                .get_or_insert_with(FxHashSet::default)
                .insert(target)
            {
                return Err("lexer epsilon metadata contains a duplicate edge");
            }
            epsilon_transitions[source].push(target);
            targets_in_state += 1;
        }
        if targets_in_state == 0 {
            return Err("lexer epsilon metadata state contains no targets");
        }
    }

    states.truncate(real_state_count);
    Ok(Some(epsilon_transitions))
}

impl Serialize for DFA {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;

        if self
            .states
            .iter()
            .flat_map(|state| state.transitions.values())
            .any(|&target| target == EPSILON_WIRE_MARKER)
        {
            return Err(S::Error::custom(
                "lexer DFA contains the reserved dead transition target",
            ));
        }

        let mut states = self.states.iter().map(DfaStateWire::from).collect::<Vec<_>>();
        let real_state_count = states.len();
        let mut metadata_state_count = 0usize;
        for (source, state) in self.states.iter().enumerate() {
            for targets in state
                .epsilon_transitions
                .chunks(EPSILON_TARGETS_PER_WIRE_STATE)
            {
                if targets.is_empty() {
                    continue;
                }
                let mut entries = Vec::with_capacity(targets.len() + 3);
                entries.push((0, EPSILON_WIRE_MARKER));
                entries.push((1, EPSILON_WIRE_EDGE));
                entries.push((2, source as u32));
                entries.extend(
                    targets
                        .iter()
                        .enumerate()
                        .map(|(index, &target)| ((index + 3) as u8, target)),
                );
                states.push(epsilon_wire_state(entries));
                metadata_state_count += 1;
            }
        }
        if metadata_state_count != 0 {
            states.push(epsilon_wire_state(vec![
                (0, EPSILON_WIRE_MARKER),
                (1, EPSILON_WIRE_TRAILER),
                (2, real_state_count as u32),
                (3, metadata_state_count as u32),
            ]));
        }

        DfaWire {
            states,
            group_id_to_u8set: self.group_id_to_u8set.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DFA {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut wire = DfaWire::deserialize(deserializer)?;
        let epsilon_transitions = decode_epsilon_wire_states(&mut wire.states)
            .map_err(serde::de::Error::custom)?;
        let mut states = wire
            .states
            .into_iter()
            .map(DFAState::from)
            .collect::<Vec<_>>();
        if let Some(epsilon_transitions) = epsilon_transitions {
            for (state, epsilon_transitions) in
                states.iter_mut().zip(epsilon_transitions.into_iter())
            {
                state.epsilon_transitions = epsilon_transitions;
            }
        }
        Ok(Self {
            states,
            group_id_to_u8set: wire.group_id_to_u8set,
        })
    }
}

impl std::fmt::Debug for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DFA { .. }")
    }
}

impl DFA {
    pub(super) fn new(num_states: usize) -> Self {
        Self {
            states: vec![DFAState::default(); num_states],
            group_id_to_u8set: Vec::new(),
        }
    }

    pub(crate) fn num_states(&self) -> usize {
        self.states.len()
    }

    pub(super) fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        let groups = self.group_id_to_u8set.len();
        self.states.push(DFAState {
            transitions: CharTransitions::default(),
            finalizers: BitSet::new(groups),
            possible_future_group_ids: BitSet::new(groups),
            epsilon_transitions: Vec::new(),
        });
        id
    }

    pub(super) fn ensure_group_capacity(&mut self, num_groups: usize) {
        if self.group_id_to_u8set.len() < num_groups {
            self.group_id_to_u8set.resize(num_groups, U8Set::empty());
        }
        for state in &mut self.states {
            Self::resize_state_group_bits(state, num_groups);
        }
    }

    pub(super) fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.insert(byte, to);
        }
    }

    pub(super) fn add_epsilon_transition(&mut self, from: u32, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            if !state.epsilon_transitions.contains(&to) {
                state.epsilon_transitions.push(to);
            }
        }
    }

    /// Move an independently compiled component into this DFA, rebasing its
    /// state targets and remapping local group IDs to the supplied global IDs.
    /// Transition buffers are retained rather than copied into newly allocated
    /// rows, which matters for very large exact runtime lexer components.
    pub(super) fn append_rebased_component(
        &mut self,
        mut component: DFA,
        global_group_ids: &[usize],
    ) -> u32 {
        assert_eq!(
            component.group_id_to_u8set.len(),
            global_group_ids.len(),
            "one global group ID is required per component group",
        );
        let offset = u32::try_from(self.states.len()).expect("lexer DFA state ID overflow");

        for (local_group, &global_group) in global_group_ids.iter().enumerate() {
            assert!(
                global_group < self.group_id_to_u8set.len(),
                "global lexer group ID is out of range",
            );
            self.group_id_to_u8set[global_group] = component.group_id_to_u8set[local_group];
        }

        let total_groups = self.group_id_to_u8set.len();
        for state in &mut component.states {
            for (_, target) in state.transitions.iter_mut() {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA transition target overflow");
            }
            for target in &mut state.epsilon_transitions {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA epsilon target overflow");
            }

            let mut finalizers = BitSet::new(total_groups);
            for local_group in state.finalizers.iter() {
                finalizers.set(global_group_ids[local_group]);
            }
            let mut futures = BitSet::new(total_groups);
            for local_group in state.possible_future_group_ids.iter() {
                futures.set(global_group_ids[local_group]);
            }
            state.finalizers = finalizers;
            state.possible_future_group_ids = futures;
        }

        self.states.append(&mut component.states);
        offset
    }

    pub(crate) fn has_epsilon_transitions(&self) -> bool {
        self.states
            .iter()
            .any(|state| !state.epsilon_transitions.is_empty())
    }

    /// Minimum number of consumed bytes needed to reach any accepting state.
    /// Epsilon transitions have zero cost and byte transitions have unit cost.
    pub(crate) fn min_match_byte_len(&self) -> Option<usize> {
        let mut distance = vec![usize::MAX; self.states.len()];
        let mut queue = std::collections::VecDeque::new();
        if self.states.is_empty() {
            return None;
        }
        distance[0] = 0;
        queue.push_front(0u32);

        while let Some(state_id) = queue.pop_front() {
            let state_index = state_id as usize;
            let current_distance = distance[state_index];
            let state = &self.states[state_index];
            if !state.finalizers.is_empty() {
                return Some(current_distance);
            }

            for &target in &state.epsilon_transitions {
                let target_index = target as usize;
                if current_distance < distance[target_index] {
                    distance[target_index] = current_distance;
                    queue.push_front(target);
                }
            }
            let next_distance = current_distance.saturating_add(1);
            for (_, &target) in state.transitions.iter() {
                let target_index = target as usize;
                if next_distance < distance[target_index] {
                    distance[target_index] = next_distance;
                    queue.push_back(target);
                }
            }
        }
        None
    }

    pub(super) fn epsilon_closure(&self, roots: &[u32]) -> SmallVec<[u32; 1]> {
        if roots.iter().all(|&root| {
            self.states
                .get(root as usize)
                .is_some_and(|state| state.epsilon_transitions.is_empty())
        }) {
            let mut closure = SmallVec::<[u32; 1]>::from_slice(roots);
            closure.sort_unstable();
            closure.dedup();
            return closure;
        }
        let mut closure = SmallVec::<[u32; 1]>::new();
        let mut seen = vec![false; self.states.len()];
        let mut stack = Vec::with_capacity(roots.len());
        for &root in roots {
            if (root as usize) < self.states.len() && !seen[root as usize] {
                seen[root as usize] = true;
                stack.push(root);
            }
        }
        while let Some(state) = stack.pop() {
            closure.push(state);
            for &target in &self.states[state as usize].epsilon_transitions {
                if !seen[target as usize] {
                    seen[target as usize] = true;
                    stack.push(target);
                }
            }
        }
        closure.sort_unstable();
        closure
    }

    /// Compute every singleton epsilon closure in one pass.
    ///
    /// Lexer partition unions and bounded adaptive frontiers produce an
    /// acyclic epsilon graph.  For that common case, reverse-topological
    /// dynamic programming reuses each target closure instead of allocating a
    /// fresh `seen` vector and walking the same tails once per source state.
    /// Persisted/debug automata may contain epsilon cycles; retain the exact
    /// scalar closure as a defensive fallback when Kahn's order detects one.
    pub(super) fn all_singleton_epsilon_closures(&self) -> Vec<Box<[u32]>> {
        let state_count = self.states.len();
        let mut indegree = vec![0usize; state_count];
        for state in &self.states {
            for &target in &state.epsilon_transitions {
                indegree[target as usize] += 1;
            }
        }
        let mut queue = std::collections::VecDeque::<u32>::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state as u32);
            }
        }
        let mut topo = Vec::with_capacity(state_count);
        while let Some(state) = queue.pop_front() {
            topo.push(state);
            for &target in &self.states[state as usize].epsilon_transitions {
                let degree = &mut indegree[target as usize];
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target);
                }
            }
        }
        if topo.len() != state_count {
            return (0..state_count)
                .map(|state| {
                    self.epsilon_closure(&[state as u32])
                        .into_vec()
                        .into_boxed_slice()
                })
                .collect();
        }

        let mut closures = vec![Box::<[u32]>::default(); state_count];
        let mut marks = vec![0u32; state_count];
        let mut generation = 0u32;
        for &state in topo.iter().rev() {
            generation = generation.wrapping_add(1);
            if generation == 0 {
                marks.fill(0);
                generation = 1;
            }
            let mut closure = Vec::<u32>::new();
            marks[state as usize] = generation;
            closure.push(state);
            for &target in &self.states[state as usize].epsilon_transitions {
                for &reachable in closures[target as usize].iter() {
                    let slot = &mut marks[reachable as usize];
                    if *slot != generation {
                        *slot = generation;
                        closure.push(reachable);
                    }
                }
            }
            closure.sort_unstable();
            closures[state as usize] = closure.into_boxed_slice();
        }
        closures
    }

    pub(super) fn step_all(&self, states: &[u32], byte: u8) -> SmallVec<[u32; 1]> {
        if states.len() == 1 {
            let state = states[0];
            if self.states[state as usize].epsilon_transitions.is_empty() {
                let Some(target) = self.step(state, byte) else {
                    return SmallVec::new();
                };
                if self.states[target as usize].epsilon_transitions.is_empty() {
                    return SmallVec::from_buf([target]);
                }
            }
        }
        let closure = self.epsilon_closure(states);
        let mut targets = SmallVec::<[u32; 1]>::new();
        for state in closure {
            if let Some(target) = self.step(state, byte) {
                targets.push(target);
            }
        }
        if targets.is_empty() {
            return targets;
        }
        targets.sort_unstable();
        targets.dedup();
        self.epsilon_closure(&targets)
    }

    pub(super) fn set_transitions_from_sorted_entries(
        &mut self,
        state: u32,
        entries: Vec<(u8, u32)>,
    ) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.transitions = CharTransitions::from_sorted_entries(entries);
        }
    }

    pub(super) fn clear_finalizers_for_state(&mut self, state: u32) -> BitSet {
        let num_groups = self.group_id_to_u8set.len();
        if let Some(entry) = self.state_mut(state) {
            std::mem::replace(&mut entry.finalizers, BitSet::new(num_groups))
        } else {
            BitSet::empty(0)
        }
    }

    pub(super) fn overwrite_state_metadata(
        &mut self,
        state: u32,
        finalizers: BitSet,
        possible_future_group_ids: BitSet,
    ) {
        if let Some(entry) = self.state_mut(state) {
            entry.finalizers = finalizers;
            entry.possible_future_group_ids = possible_future_group_ids;
        }
    }

    pub(super) fn set_group_u8set(&mut self, group_id: GroupId, set: U8Set) {
        if let Some(entry) = self.group_id_to_u8set.get_mut(group_id as usize) {
            *entry = set;
        }
    }

    pub(crate) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.states
            .get(state as usize)
            .and_then(|state| state.transitions.get(byte).copied())
    }

    pub(crate) fn transitions(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.states[state as usize]
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, target))
    }

    pub(super) fn get_u8set(&self, state: u32) -> U8Set {
        let mut out = U8Set::empty();
        if let Some(state) = self.states.get(state as usize) {
            for (byte, _) in state.transitions.iter() {
                out.insert(byte);
            }
        }
        out
    }

    pub(super) fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(DEAD)
    }

    pub(super) fn group_id_to_u8set(&self, group_id: GroupId) -> &U8Set {
        &self.group_id_to_u8set[group_id as usize]
    }

    pub(crate) fn finalizers(&self, state: u32) -> &BitSet {
        &self.states[state as usize].finalizers
    }

    pub(crate) fn possible_future_group_ids(&self, state: u32) -> &BitSet {
        &self.states[state as usize].possible_future_group_ids
    }

    pub(super) fn states(&self) -> &[DFAState] {
        &self.states
    }

    pub(super) fn states_mut(&mut self) -> &mut Vec<DFAState> {
        &mut self.states
    }

    pub(super) fn num_groups(&self) -> usize {
        self.group_id_to_u8set.len()
    }

    pub(super) fn set_possible_future_group_ids(&mut self, state: u32, ids: BitSet) {
        if let Some(entry) = self.state_mut(state) {
            entry.possible_future_group_ids = ids;
        }
    }
    /// Mask all states' possible_future_group_ids with the given bitset.
    pub(super) fn mask_possible_futures(&mut self, mask: &BitSet) {
        for state in &mut self.states {
            state.possible_future_group_ids.intersect_with(mask);
        }
    }
    /// Create a clone of an existing state (transitions, finalizers,
    /// possible_future_group_ids) and return the new state's id.
    pub(super) fn clone_state(&mut self, source: u32) -> u32 {
        let cloned = self.states[source as usize].clone();
        let id = self.states.len() as u32;
        self.states.push(cloned);
        id
    }

    /// Rewrite every transition that targets `old_target` so it targets
    /// `new_target` instead.
    /// Redirect every incoming edge to `old_target`, returning whether any
    /// edge changed. The caller may use this to speculatively clone a state
    /// and discard the clone when no incoming edge exists.
    pub(super) fn redirect_transitions(&mut self, old_target: u32, new_target: u32) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            for (_, target) in state.transitions.iter_mut() {
                if *target == old_target {
                    *target = new_target;
                    changed = true;
                }
            }
            for target in &mut state.epsilon_transitions {
                if *target == old_target {
                    *target = new_target;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Remove the final state when it is the expected freshly-created ID.
    pub(super) fn discard_last_state(&mut self, expected: u32) {
        debug_assert_eq!(self.states.len(), expected as usize + 1);
        self.states.pop();
    }

    pub(super) fn apply_group_exclusions(
        &mut self,
        excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            if state.finalizers.count_ones() < 2 {
                continue;
            }

            for group_index in excluded_group_indices(&state.finalizers, excludes) {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        changed
    }

    pub(super) fn apply_group_intersections(
        &mut self,
        intersections: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            for group_index in intersection_missing_group_indices(&state.finalizers, intersections) {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        changed
    }

    pub(super) fn project_groups(&self, num_groups: usize) -> DFA {
        let mut projected = DFA::new(self.num_states());
        projected.ensure_group_capacity(num_groups);

        for (state_index, state) in self.states.iter().enumerate() {
            let transitions = state
                .transitions
                .iter()
                .map(|(byte, &target)| (byte, target))
                .collect();
            projected.set_transitions_from_sorted_entries(state_index as u32, transitions);
            projected.states_mut()[state_index].epsilon_transitions =
                state.epsilon_transitions.clone();

            let finalizers = project_bitset(&state.finalizers, num_groups);
            let future = project_bitset(&state.possible_future_group_ids, num_groups);

            projected.overwrite_state_metadata(state_index as u32, finalizers, future);
        }

        for group_id in 0..num_groups {
            projected.set_group_u8set(group_id as u32, self.group_id_to_u8set[group_id]);
        }

        projected
    }

    fn state_mut(&mut self, state: u32) -> Option<&mut DFAState> {
        self.states.get_mut(state as usize)
    }

    fn resize_state_group_bits(state: &mut DFAState, num_groups: usize) {
        if state.finalizers.len() < num_groups {
            state.finalizers = resized_bitset(&state.finalizers, num_groups);
            state.possible_future_group_ids =
                resized_bitset(&state.possible_future_group_ids, num_groups);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::Serialize;

    use super::DFA;
    use crate::ds::char_transitions::CharTransitions;
    use crate::ds::bitset::BitSet;
    use crate::ds::u8set::U8Set;

    #[derive(Serialize)]
    struct LegacyDfaState {
        transitions: CharTransitions<u32>,
        finalizers: BitSet,
        possible_future_group_ids: BitSet,
    }

    #[derive(Serialize)]
    struct LegacyDfa {
        states: Vec<LegacyDfaState>,
        group_id_to_u8set: Vec<U8Set>,
    }

    #[test]
    fn legacy_dfa_layout_decodes_with_empty_epsilon_edges() {
        let mut transitions = CharTransitions::default();
        transitions.insert(b'a', 1);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        let mut future = BitSet::new(1);
        future.set(0);
        let legacy = LegacyDfa {
            states: vec![LegacyDfaState {
                transitions,
                finalizers,
                possible_future_group_ids: future,
            }],
            group_id_to_u8set: vec![U8Set::single(b'a')],
        };

        let bytes = bincode::serialize(&legacy).unwrap();
        let decoded: DFA = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.states.len(), 1);
        assert_eq!(decoded.states[0].transitions.get(b'a'), Some(&1));
        assert!(decoded.states[0].finalizers.contains(0));
        assert!(decoded.states[0].possible_future_group_ids.contains(0));
        assert!(decoded.states[0].epsilon_transitions.is_empty());
        assert_eq!(bincode::serialize(&decoded).unwrap(), bytes);

        // Epsilon-bearing DFAs use the compatible synthetic-state extension and
        // reconstruct the exact in-memory automaton.
        let mut current = decoded.clone();
        current.add_epsilon_transition(0, 0);
        let current_bytes = bincode::serialize(&current).unwrap();
        let current_decoded: DFA = bincode::deserialize(&current_bytes).unwrap();
        assert_eq!(current_decoded, current);
    }

    #[test]
    fn epsilon_closure_handles_fanout_chains_and_cycles() {
        let mut automaton = DFA::new(5);
        automaton.ensure_group_capacity(1);
        automaton.add_epsilon_transition(0, 1);
        automaton.add_epsilon_transition(0, 2);
        automaton.add_epsilon_transition(1, 3);
        automaton.add_epsilon_transition(2, 3);
        automaton.add_epsilon_transition(3, 1);
        automaton.add_transition(3, b'x', 4);

        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        automaton.overwrite_state_metadata(4, finalizers, BitSet::new(1));
        automaton.recompute_possible_futures();

        assert_eq!(automaton.epsilon_closure(&[0]).as_slice(), &[0, 1, 2, 3]);
        assert_eq!(automaton.step_all(&[0], b'x').as_slice(), &[4]);
        assert!(automaton.possible_future_group_ids(0).contains(0));
    }

    #[test]
    fn epsilon_wire_roundtrip_handles_more_than_one_metadata_chunk() {
        let mut automaton = DFA::new(302);
        for target in 1..=300 {
            automaton.add_epsilon_transition(0, target);
        }
        automaton.add_epsilon_transition(301, 0);

        let bytes = bincode::serialize(&automaton).unwrap();
        let decoded: DFA = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, automaton);
        assert_eq!(decoded.states[0].epsilon_transitions.len(), 300);
    }

    #[test]
    fn epsilon_possible_futures_match_bruteforce_reachability() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *seed
        }

        let mut seed = 0x8f31_6a2d_901c_447bu64;
        for case in 0..96 {
            let states = 1 + (next(&mut seed) as usize % 8);
            let groups = 3usize;
            let mut dfa = DFA::new(states);
            dfa.ensure_group_capacity(groups);

            for state in 0..states {
                let mut finalizers = BitSet::new(groups);
                for group in 0..groups {
                    if next(&mut seed).is_multiple_of(4) {
                        finalizers.set(group);
                    }
                }
                dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(groups));

                for byte in 0..3u8 {
                    if next(&mut seed).is_multiple_of(3) {
                        let target = (next(&mut seed) as usize % states) as u32;
                        dfa.add_transition(state as u32, byte, target);
                    }
                }
                for target in 0..states {
                    if next(&mut seed).is_multiple_of(9) {
                        dfa.add_epsilon_transition(state as u32, target as u32);
                    }
                }
            }

            dfa.recompute_possible_futures();

            for source in 0..states as u32 {
                let mut after_byte = Vec::new();
                for closure_state in dfa.epsilon_closure(&[source]) {
                    for (_, &target) in dfa.states[closure_state as usize].transitions.iter() {
                        after_byte.extend(dfa.epsilon_closure(&[target]));
                    }
                }
                after_byte.sort_unstable();
                after_byte.dedup();

                let mut seen = vec![false; states];
                let mut stack = after_byte;
                let mut expected = BitSet::new(groups);
                while let Some(state) = stack.pop() {
                    let state_index = state as usize;
                    if seen[state_index] {
                        continue;
                    }
                    seen[state_index] = true;
                    expected.union_with(&dfa.states[state_index].finalizers);
                    stack.extend(dfa.states[state_index].epsilon_transitions.iter().copied());
                    stack.extend(
                        dfa.states[state_index]
                            .transitions
                            .values()
                            .copied(),
                    );
                }

                assert_eq!(
                    dfa.possible_future_group_ids(source),
                    &expected,
                    "future mismatch in random case {case}, source {source}",
                );
            }
        }
    }
}
