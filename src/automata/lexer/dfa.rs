//! Byte-oriented lexer DFA used by the tokenizer and lexer compiler.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

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

const START_METADATA_SENTINEL: u32 = u32::MAX;

fn default_start_states() -> Vec<u32> {
    vec![0]
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct DFAState {
    pub(super) transitions: CharTransitions<u32>,
    pub(super) finalizers: BitSet,
    possible_future_group_ids: BitSet,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DFA {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
    /// Selectable entry states. The first entry is the default start state.
    ///
    /// These are not NFA-style union starts: callers select one concrete
    /// lexer state when beginning a scan. Keeping the full set here ensures
    /// pruning, minimization, and determinization preserve every supported
    /// entry point.
    start_states: Vec<u32>,
}

/// Deliberate two-field wire format compatible with lexer DFAs serialized
/// before selectable entry states existed. Multi-entry metadata is stored as
/// trailing invalid DFA states, so the enclosing bincode layout stays intact.
#[derive(Serialize, Deserialize)]
struct DfaWire {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
}

fn encode_start_metadata(starts: &[u32]) -> Vec<DFAState> {
    starts
        .chunks(255)
        .map(|chunk| {
            let mut entries: Vec<(u8, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(index, &state)| (index as u8, state))
                .collect();
            entries.push((u8::MAX, START_METADATA_SENTINEL));
            DFAState {
                transitions: CharTransitions::from_sorted_entries(entries),
                finalizers: BitSet::new(0),
                possible_future_group_ids: BitSet::new(0),
            }
        })
        .collect()
}

fn decode_start_metadata(states: &mut Vec<DFAState>) -> Result<Vec<u32>, String> {
    let mut reversed_chunks = Vec::<Vec<u32>>::new();
    while let Some(last) = states.last() {
        let is_metadata = last.finalizers.len() == 0
            && last.possible_future_group_ids.len() == 0
            && last.transitions.get(u8::MAX) == Some(&START_METADATA_SENTINEL);
        if !is_metadata {
            break;
        }

        let mut chunk = Vec::with_capacity(last.transitions.len().saturating_sub(1));
        for (byte, &target) in last.transitions.iter() {
            if byte == u8::MAX {
                continue;
            }
            if target == START_METADATA_SENTINEL {
                return Err("invalid lexer DFA start-state metadata".to_owned());
            }
            chunk.push(target);
        }
        reversed_chunks.push(chunk);
        states.pop();
    }

    if reversed_chunks.is_empty() {
        return Ok(default_start_states());
    }

    reversed_chunks.reverse();
    let starts: Vec<u32> = reversed_chunks.into_iter().flatten().collect();
    if starts.is_empty() {
        return Err("lexer DFA start-state metadata is empty".to_owned());
    }
    if starts.iter().any(|&state| state as usize >= states.len()) {
        return Err("lexer DFA start-state metadata is out of bounds".to_owned());
    }
    let mut seen = BTreeSet::new();
    if starts.iter().any(|&state| !seen.insert(state)) {
        return Err("lexer DFA start-state metadata contains duplicate entries".to_owned());
    }
    Ok(starts)
}

impl Serialize for DFA {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut states = self.states.clone();
        if self.start_states != default_start_states() {
            states.extend(encode_start_metadata(&self.start_states));
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
        D: serde::Deserializer<'de>,
    {
        let mut wire = DfaWire::deserialize(deserializer)?;
        let start_states = decode_start_metadata(&mut wire.states)
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            states: wire.states,
            group_id_to_u8set: wire.group_id_to_u8set,
            start_states,
        })
    }
}

impl Default for DFA {
    fn default() -> Self {
        Self::new(0)
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
            start_states: default_start_states(),
        }
    }

    /// The default entry state used by legacy single-start callers.
    pub(super) fn start_state(&self) -> u32 {
        self.start_states.first().copied().unwrap_or(0)
    }

    /// All selectable lexer entry states, with the default first.
    pub(super) fn start_states(&self) -> &[u32] {
        &self.start_states
    }

    /// Replace the selectable lexer entry states. The first state becomes the
    /// default. Duplicate entries are removed while preserving order.
    pub(super) fn set_start_states(&mut self, states: Vec<u32>) {
        assert!(!states.is_empty(), "lexer DFA must have a default start state");
        let mut deduplicated = Vec::with_capacity(states.len());
        for state in states {
            assert!(
                (state as usize) < self.states.len(),
                "lexer DFA start state {state} is out of bounds for {} states",
                self.states.len(),
            );
            if !deduplicated.contains(&state) {
                deduplicated.push(state);
            }
        }
        self.start_states = deduplicated;
    }

    /// Make an existing or new entry state the default while retaining every
    /// other selectable entry state as auxiliary.
    pub(super) fn set_default_start_state(&mut self, state: u32) {
        assert!(
            (state as usize) < self.states.len(),
            "lexer DFA start state {state} is out of bounds for {} states",
            self.states.len(),
        );
        let mut starts = Vec::with_capacity(self.start_states.len() + 1);
        starts.push(state);
        starts.extend(self.start_states.iter().copied().filter(|&entry| entry != state));
        self.set_start_states(starts);
    }

    /// Add an auxiliary selectable lexer entry state. It is appended after the
    /// default entry state unless it is already present.
    pub(super) fn add_start_state(&mut self, state: u32) {
        assert!(
            (state as usize) < self.states.len(),
            "lexer DFA start state {state} is out of bounds for {} states",
            self.states.len(),
        );
        if !self.start_states.contains(&state) {
            self.start_states.push(state);
        }
    }

    pub(super) fn num_states(&self) -> usize {
        self.states.len()
    }

    pub(super) fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        let groups = self.group_id_to_u8set.len();
        self.states.push(DFAState {
            transitions: CharTransitions::default(),
            finalizers: BitSet::new(groups),
            possible_future_group_ids: BitSet::new(groups),
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

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.states
            .get(state as usize)
            .and_then(|state| state.transitions.get(byte).copied())
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

    pub(super) fn finalizers(&self, state: u32) -> &BitSet {
        &self.states[state as usize].finalizers
    }

    pub(super) fn possible_future_group_ids(&self, state: u32) -> &BitSet {
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

            let finalizers = project_bitset(&state.finalizers, num_groups);
            let future = project_bitset(&state.possible_future_group_ids, num_groups);

            projected.overwrite_state_metadata(state_index as u32, finalizers, future);
        }

        for group_id in 0..num_groups {
            projected.set_group_u8set(group_id as u32, self.group_id_to_u8set[group_id]);
        }

        projected.set_start_states(self.start_states.clone());

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
    use super::*;

    #[derive(Serialize)]
    struct LegacyDfa {
        states: Vec<DFAState>,
        group_id_to_u8set: Vec<U8Set>,
    }

    #[derive(Serialize)]
    struct LegacyContainer {
        dfa: LegacyDfa,
        trailing_value: u32,
    }

    #[derive(Deserialize)]
    struct CurrentContainer {
        dfa: DFA,
        trailing_value: u32,
    }

    #[test]
    fn bincode_legacy_dfa_defaults_to_single_start_state() {
        let dfa = DFA::new(2);
        let legacy = LegacyDfa {
            states: dfa.states.clone(),
            group_id_to_u8set: dfa.group_id_to_u8set.clone(),
        };
        let bytes = bincode::serialize(&legacy).expect("legacy DFA serializes");
        let restored: DFA = bincode::deserialize(&bytes).expect("legacy DFA deserializes");
        assert_eq!(restored.start_states(), &[0]);
    }

    #[test]
    fn bincode_legacy_dfa_keeps_nested_stream_aligned() {
        let dfa = DFA::new(2);
        let legacy = LegacyContainer {
            dfa: LegacyDfa {
                states: dfa.states.clone(),
                group_id_to_u8set: dfa.group_id_to_u8set.clone(),
            },
            trailing_value: 0xA5A5_1234,
        };
        let bytes = bincode::serialize(&legacy).expect("legacy container serializes");
        let restored: CurrentContainer =
            bincode::deserialize(&bytes).expect("legacy container deserializes");
        assert_eq!(restored.dfa.start_states(), &[0]);
        assert_eq!(restored.trailing_value, legacy.trailing_value);
    }
}
