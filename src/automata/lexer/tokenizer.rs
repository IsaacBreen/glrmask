//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use smallvec::SmallVec;

use super::dfa::DFA;
use crate::automata::regex::Expr;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

#[derive(Debug, Clone, Deserialize)]
pub struct Tokenizer {
    pub(super) dfa: DFA,
    pub(super) num_terminals: u32,
    /// Runtime-only exact byte-class transition segments. The historical
    /// serialized tokenizer shape contains only `dfa` and `num_terminals`; the
    /// custom serializer expands these segments into that same DFA wire form.
    #[serde(default, skip)]
    pub(super) compressed_transition_segments: Arc<[CompressedTransitionSegment]>,
    /// Per-terminal regex expressions used to (re)build this tokenizer.
    /// Skipped during (de)serialization because they are only needed during
    /// compile-time simplification for active-terminal rebuilds.
    #[serde(default, skip)]
    pub(super) exprs: Option<Arc<[Expr]>>,
    /// Derived epsilon closures are shared by compile-time analyses.  A
    /// partitioned lexer is queried by many concurrent compiler lanes; without
    /// this cache each lane independently walks the same epsilon DAG for every
    /// raw state.
    #[serde(default, skip)]
    pub(super) singleton_epsilon_closures: OnceLock<Arc<[Box<[u32]>]>>,
}

/// Exact deterministic transition rows over a byte-equivalence-class alphabet.
/// Targets and row coordinates are local to one DFA component; `state_offset`
/// rebases both into the final partitioned runtime tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompressedTransitionSegment {
    pub(crate) state_offset: u32,
    pub(crate) state_count: u32,
    pub(crate) byte_to_class: Arc<[u8]>,
    pub(crate) class_members: Arc<[Box<[u8]>]>,
    pub(crate) row_offsets: Arc<[u32]>,
    pub(crate) entries: Arc<[(u8, u32)]>,
    pub(crate) expanded_transition_count: usize,
}

pub(crate) mod artifact_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize)]
    struct TokenizerArtifactRef<'a> {
        dfa: &'a DFA,
        num_terminals: u32,
        compressed_transition_segments: &'a [CompressedTransitionSegment],
    }

    #[derive(Deserialize)]
    struct TokenizerArtifact {
        dfa: DFA,
        num_terminals: u32,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    pub(crate) fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        TokenizerArtifactRef {
            dfa: &tokenizer.dfa,
            num_terminals: tokenizer.num_terminals,
            compressed_transition_segments: &tokenizer.compressed_transition_segments,
        }
        .serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        let artifact = TokenizerArtifact::deserialize(deserializer)?;
        Ok(Tokenizer {
            dfa: artifact.dfa,
            num_terminals: artifact.num_terminals,
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
        })
    }
}

impl CompressedTransitionSegment {
    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.state_count
    }

    #[inline]
    fn local_transition(&self, local_state: u32, byte: u8) -> Option<u32> {
        let class = self.byte_to_class[byte as usize];
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        self.entries[start..end]
            .binary_search_by_key(&class, |&(existing, _)| existing)
            .ok()
            .map(|index| self.entries[start + index].1)
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        self.local_transition(state - self.state_offset, byte)
            .map(|target| self.state_offset + target)
    }

    fn expanded_entries(&self, state: u32) -> Vec<(u8, u32)> {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        let row = &self.entries[start..end];
        let mut target_by_class = vec![u32::MAX; self.class_members.len()];
        let mut capacity = 0usize;
        for &(class, target) in row {
            target_by_class[class as usize] = target;
            capacity += self.class_members[class as usize].len();
        }
        let mut entries = Vec::with_capacity(capacity);
        for byte in 0u16..=255 {
            let class = self.byte_to_class[byte as usize] as usize;
            let target = target_by_class[class];
            if target != u32::MAX {
                entries.push((byte as u8, self.state_offset + target));
            }
        }
        entries
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        for &(class, target) in &self.entries[start..end] {
            let target = self.state_offset + target;
            for &byte in self.class_members[class as usize].iter() {
                row[byte as usize] = target;
            }
        }
    }

    fn transition_count(&self, state: u32) -> usize {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        self.entries[start..end]
            .iter()
            .map(|(class, _)| self.class_members[*class as usize].len())
            .sum()
    }
}

enum TokenizerTransitionsIterInner<'a> {
    Dense(crate::ds::char_transitions::CharTransitionsIter<'a, u32>),
    Compressed {
        segment: &'a CompressedTransitionSegment,
        state: u32,
        next_byte: u16,
    },
    Empty,
}

pub(crate) struct TokenizerTransitionsIter<'a> {
    inner: TokenizerTransitionsIterInner<'a>,
}

impl Iterator for TokenizerTransitionsIter<'_> {
    type Item = (u8, u32);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => {
                iter.next().map(|(byte, target)| (byte, *target))
            }
            TokenizerTransitionsIterInner::Compressed {
                segment,
                state,
                next_byte,
            } => {
                while *next_byte <= 255 {
                    let byte = *next_byte as u8;
                    *next_byte += 1;
                    if let Some(target) = segment.transition(*state, byte) {
                        return Some((byte, target));
                    }
                }
                None
            }
            TokenizerTransitionsIterInner::Empty => None,
        }
    }


    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => iter.size_hint(),
            TokenizerTransitionsIterInner::Compressed { segment, state, .. } => {
                let count = segment.transition_count(*state);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::Empty => (0, Some(0)),
        }
    }

    fn count(self) -> usize {
        match self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => iter.count(),
            TokenizerTransitionsIterInner::Compressed { segment, state, .. } => {
                segment.transition_count(state)
            }
            TokenizerTransitionsIterInner::Empty => 0,
        }
    }
}

impl Serialize for Tokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let materialized;
        let dfa = if self.compressed_transition_segments.is_empty() {
            &self.dfa
        } else {
            materialized = self.materialized_dfa();
            &materialized
        };
        // Match the historical derived-serialization field order exactly.
        let mut state = serializer.serialize_struct("Tokenizer", 2)?;
        state.serialize_field("dfa", dfa)?;
        state.serialize_field("num_terminals", &self.num_terminals)?;
        state.end()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
    pub end_state: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: TokenizerStateSet,
    pub matches: Vec<TokenizerMatch>,
}

pub type TokenizerStateSet = SmallVec<[u32; 1]>;

/// Exact disjoint union used only by cross-tokenizer compile-time analyses.
/// Source state `s` is represented by `left_offset + s` or
/// `right_offset + s`; state zero is a fresh epsilon dispatcher.
pub(crate) struct TokenizerAnalysisUnion {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) left_offset: u32,
    pub(crate) right_offset: u32,
}

pub(crate) trait Lexer {
    fn start_state(&self) -> u32;
    fn num_terminals(&self) -> u32;
    fn has_epsilon_transitions(&self) -> bool;
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_;

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    fn transition_count(&self) -> usize {
        (0..self.num_states())
            .map(|state| self.transitions_from(state).count())
            .sum()
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32>;
    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet;
    fn get_transition(&self, state: u32, byte: u8) -> u32;
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet;
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals(&self, state: u32) -> &BitSet;

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32;
    fn num_forced_minimized_states(&self) -> usize;
    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult;
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult;
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet;
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult;

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, TokenizerStateSet);
}

fn into_longest_matches(
    matches: FxHashMap<TerminalID, (usize, TokenizerStateSet)>,
) -> Vec<TokenizerMatch> {
    matches
        .into_iter()
        .flat_map(|(id, (width, end_states))| {
            end_states.into_iter().map(move |end_state| TokenizerMatch {
                id,
                width,
                end_state,
            })
        })
        .collect()
}

fn group_matches_by_width(matches: Vec<TokenizerMatch>) -> Vec<(usize, BTreeSet<TerminalID>)> {
    let mut grouped = std::collections::BTreeMap::<usize, BTreeSet<TerminalID>>::new();
    for matched in matches {
        grouped.entry(matched.width).or_default().insert(matched.id);
    }
    grouped.into_iter().collect()
}

impl Tokenizer {
    /// Materialize a deterministic compile-time analysis view as a tokenizer.
    /// The view may be a powerset of this tokenizers epsilon-NFA. State zero is
    /// reserved for the supplied start state, and the returned old-to-new map
    /// lets callers lift raw-start mappings into the materialized coordinate.
    pub(crate) fn materialize_deterministic_view(
        &self,
        start_state: usize,
        finalizers: &[Vec<usize>],
        futures: &[Vec<usize>],
        edge_offsets: &[u32],
        edges: &[(u8, u32)],
        active_terminals: &[bool],
    ) -> Option<(Tokenizer, Vec<u32>)> {
        let state_count = finalizers.len();
        if state_count == 0
            || futures.len() != state_count
            || edge_offsets.len() != state_count + 1
            || start_state >= state_count
            || active_terminals.len() != self.num_terminals as usize
        {
            return None;
        }
        let mut new_to_old = Vec::with_capacity(state_count);
        new_to_old.push(start_state);
        new_to_old.extend((0..state_count).filter(|&state| state != start_state));
        let mut old_to_new = vec![u32::MAX; state_count];
        for (new, &old) in new_to_old.iter().enumerate() {
            old_to_new[old] = new as u32;
        }

        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals as usize {
            if active_terminals[terminal] {
                dfa.set_group_u8set(
                    terminal as u32,
                    *self.dfa.group_id_to_u8set(terminal as u32),
                );
            }
        }
        for (new_state, &old_state) in new_to_old.iter().enumerate() {
            let start = *edge_offsets.get(old_state)? as usize;
            let end = *edge_offsets.get(old_state + 1)? as usize;
            let transitions = edges
                .get(start..end)?
                .iter()
                .map(|&(byte, target)| {
                    old_to_new
                        .get(target as usize)
                        .copied()
                        .filter(|&target| target != u32::MAX)
                        .map(|target| (byte, target))
                })
                .collect::<Option<Vec<_>>>()?;
            dfa.set_transitions_from_sorted_entries(new_state as u32, transitions);
            let to_bits = |groups: &[usize]| {
                let mut bits = BitSet::new(self.num_terminals as usize);
                for &group in groups {
                    if group >= active_terminals.len() || !active_terminals[group] {
                        return None;
                    }
                    bits.set(group);
                }
                Some(bits)
            };
            dfa.overwrite_state_metadata(
                new_state as u32,
                to_bits(&finalizers[old_state])?,
                to_bits(&futures[old_state])?,
            );
        }
        Some((
            Tokenizer {
                dfa,
                num_terminals: self.num_terminals,
                compressed_transition_segments: Arc::from([]),
                exprs: None,
                singleton_epsilon_closures: OnceLock::new(),
            },
            old_to_new,
        ))
    }

    /// Materialize an exact deterministic quotient for one compile-time branch.
    ///
    /// `original_to_quotient` must be a congruence for every vocabulary-relevant
    /// byte after filtering labels to `active_terminals`. The method verifies
    /// that property over every class member before constructing the smaller
    /// tokenizer, so callers can fail closed to the original tokenizer.
    pub(crate) fn materialize_active_quotient(
        &self,
        original_to_quotient: &[u32],
        representatives: &[u32],
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Option<Tokenizer> {
        if original_to_quotient.len() != self.num_states() as usize
            || active_terminals.len() != self.num_terminals as usize
            || representatives.is_empty()
            || original_to_quotient.get(self.start_state() as usize).copied() != Some(0)
        {
            return None;
        }
        let quotient_states = representatives.len();
        if original_to_quotient
            .iter()
            .any(|&state| state == u32::MAX || state as usize >= quotient_states)
        {
            return None;
        }

        let filtered = |bits: &BitSet| {
            let mut result = BitSet::new(self.num_terminals as usize);
            for terminal in bits.iter() {
                if active_terminals.get(terminal).copied().unwrap_or(false) {
                    result.set(terminal);
                }
            }
            result
        };

        // Verify output labels and every relevant transition for all members,
        // rather than trusting the refinement implementation as an implicit
        // construction contract.
        let mut class_members = vec![Vec::<u32>::new(); quotient_states];
        for (original, &quotient) in original_to_quotient.iter().enumerate() {
            class_members[quotient as usize].push(original as u32);
        }
        for (class, members) in class_members.iter().enumerate() {
            let representative = *representatives.get(class)?;
            if !members.contains(&representative) {
                return None;
            }
            let representative_finalizers = filtered(self.dfa.finalizers(representative));
            let representative_futures =
                filtered(self.dfa.possible_future_group_ids(representative));
            for &member in members {
                if filtered(self.dfa.finalizers(member)) != representative_finalizers
                    || filtered(self.dfa.possible_future_group_ids(member))
                        != representative_futures
                {
                    return None;
                }
                for byte in 0u16..=255 {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let mapped = self
                        .step(member, byte as u8)
                        .map(|target| original_to_quotient[target as usize]);
                    let representative_mapped = self
                        .step(representative, byte as u8)
                        .map(|target| original_to_quotient[target as usize]);
                    if mapped != representative_mapped {
                        return None;
                    }
                }
                let mapped_epsilon = |state: u32| {
                    let mut targets = self.dfa.states()[state as usize]
                        .epsilon_transitions
                        .iter()
                        .map(|&target| original_to_quotient[target as usize])
                        .collect::<Vec<_>>();
                    targets.sort_unstable();
                    targets.dedup();
                    targets
                };
                if mapped_epsilon(member) != mapped_epsilon(representative) {
                    return None;
                }
            }
        }

        let mut dfa = DFA::new(quotient_states);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals as usize {
            if active_terminals[terminal] {
                dfa.set_group_u8set(
                    terminal as u32,
                    *self.dfa.group_id_to_u8set(terminal as u32),
                );
            }
        }
        for (class, &representative) in representatives.iter().enumerate() {
            let transitions = (0u16..=255)
                .filter(|&byte| relevant_bytes[byte as usize])
                .filter_map(|byte| {
                    self.step(representative, byte as u8).map(|target| {
                        (byte as u8, original_to_quotient[target as usize])
                    })
                })
                .collect::<Vec<_>>();
            dfa.set_transitions_from_sorted_entries(class as u32, transitions);
            let mut epsilon_targets = self.dfa.states()[representative as usize]
                .epsilon_transitions
                .iter()
                .map(|&target| original_to_quotient[target as usize])
                .collect::<Vec<_>>();
            epsilon_targets.sort_unstable();
            epsilon_targets.dedup();
            for target in epsilon_targets {
                dfa.add_epsilon_transition(class as u32, target);
            }
            dfa.overwrite_state_metadata(
                class as u32,
                filtered(self.dfa.finalizers(representative)),
                filtered(self.dfa.possible_future_group_ids(representative)),
            );
        }
        Some(Tokenizer {
            dfa,
            num_terminals: self.num_terminals,
            compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
        })
    }

    /// Extend `self` with the source-only residual states that were appended to
    /// `source` after `rebuilt` was constructed.  `rebuilt_to_self` must be a
    /// structural state map from the rebuilt expression DFA into `self`.
    ///
    /// Protected residual synthesis appends externally-entered product states
    /// to otherwise identical deterministic dispatch components.  The original
    /// component states remain an exact prefix.  Verify that prefix relation
    /// state-for-state, then clone only the appended states while redirecting
    /// every edge through the completed source-to-self map.  The result is a
    /// transition homomorphism over the actual source tokenizer, not a bounded
    /// semantic approximation.
    pub(crate) fn augment_from_verified_component_prefixes(
        &mut self,
        source: &Tokenizer,
        rebuilt: &Tokenizer,
        rebuilt_to_self: &[u32],
    ) -> Option<Vec<u32>> {
        if source.num_terminals != rebuilt.num_terminals
            || source.num_terminals != self.num_terminals
            || rebuilt_to_self.len() != rebuilt.num_states() as usize
        {
            return None;
        }

        let source_components = source.disjoint_dispatch_components()?;
        let rebuilt_components = rebuilt.disjoint_dispatch_components()?;
        if source_components.len() != rebuilt_components.len() {
            return None;
        }

        let mut source_to_rebuilt = vec![u32::MAX; source.num_states() as usize];
        source_to_rebuilt[source.start_state() as usize] = rebuilt.start_state();
        for (source_states, rebuilt_states) in
            source_components.iter().zip(&rebuilt_components)
        {
            if rebuilt_states.len() > source_states.len() {
                return None;
            }
            for (&source_state, &rebuilt_state) in source_states.iter().zip(rebuilt_states) {
                source_to_rebuilt[source_state as usize] = rebuilt_state;
            }
        }

        // Verify that the mapped prefix is exactly the rebuilt DFA after state
        // renumbering.  This guards the append-only invariant rather than
        // relying on component construction order as an undocumented fact.
        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state == u32::MAX {
                continue;
            }
            let source_state = source_state as u32;
            if source.dfa.finalizers(source_state) != rebuilt.dfa.finalizers(rebuilt_state)
                || source.dfa.possible_future_group_ids(source_state)
                    != rebuilt.dfa.possible_future_group_ids(rebuilt_state)
                || source.state_has_epsilon_transitions(source_state)
                    != rebuilt.state_has_epsilon_transitions(rebuilt_state)
            {
                return None;
            }
            let source_epsilon = &source.dfa.states()[source_state as usize].epsilon_transitions;
            let rebuilt_epsilon = &rebuilt.dfa.states()[rebuilt_state as usize].epsilon_transitions;
            let mapped_epsilon = source_epsilon
                .iter()
                .map(|&target| *source_to_rebuilt.get(target as usize).unwrap_or(&u32::MAX))
                .collect::<Vec<_>>();
            if mapped_epsilon != *rebuilt_epsilon {
                return None;
            }
            let source_transitions = source
                .transitions_from(source_state)
                .map(|(byte, target)| {
                    Some((byte, *source_to_rebuilt.get(target as usize)?))
                })
                .collect::<Option<Vec<_>>>()?;
            if source_transitions.iter().any(|&(_, target)| target == u32::MAX)
                || source_transitions
                    != rebuilt.transitions_from(rebuilt_state).collect::<Vec<_>>()
            {
                return None;
            }
        }

        let original_self_states = self.num_states() as usize;
        let mut source_to_self = vec![u32::MAX; source.num_states() as usize];
        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state != u32::MAX {
                source_to_self[source_state] = *rebuilt_to_self.get(rebuilt_state as usize)?;
            }
        }
        for source_state in 0..source.num_states() as usize {
            if source_to_self[source_state] == u32::MAX {
                source_to_self[source_state] = self.dfa.add_state();
            }
        }

        for source_state in 0..source.num_states() as usize {
            if source_to_rebuilt[source_state] != u32::MAX {
                continue;
            }
            let target_state = source_to_self[source_state];
            let source_state_u32 = source_state as u32;
            if source.state_has_epsilon_transitions(source_state_u32) {
                return None;
            }
            let transitions = source
                .transitions_from(source_state_u32)
                .map(|(byte, target)| (byte, source_to_self[target as usize]))
                .collect::<Vec<_>>();
            self.dfa
                .set_transitions_from_sorted_entries(target_state, transitions);
            self.dfa.overwrite_state_metadata(
                target_state,
                source.dfa.finalizers(source_state_u32).clone(),
                source
                    .dfa
                    .possible_future_group_ids(source_state_u32)
                    .clone(),
            );
        }
        debug_assert_eq!(
            self.num_states() as usize - original_self_states,
            source_to_rebuilt
                .iter()
                .filter(|&&state| state == u32::MAX)
                .count(),
        );
        Some(source_to_self)
    }

    pub(super) fn from_parts(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
    ) -> Self {
        Self {
            dfa,
            num_terminals,
            compressed_transition_segments: Arc::from([]),
            exprs,
            singleton_epsilon_closures: OnceLock::new(),
        }
    }

    pub(crate) fn from_parts_with_compressed_transitions(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    ) -> Self {
        debug_assert!(compressed_transition_segments
            .windows(2)
            .all(|pair| pair[0].state_offset + pair[0].state_count <= pair[1].state_offset));
        Self {
            dfa,
            num_terminals,
            compressed_transition_segments: Arc::from(compressed_transition_segments),
            exprs,
            singleton_epsilon_closures: OnceLock::new(),
        }
    }

    fn compressed_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&CompressedTransitionSegment> {
        let index = self
            .compressed_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.compressed_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    pub(crate) fn has_compressed_transition_state(&self, state: u32) -> bool {
        self.compressed_segment_for_state(state).is_some()
    }

    fn materialized_dfa(&self) -> DFA {
        let mut dfa = self.dfa.clone();
        for segment in self.compressed_transition_segments.iter() {
            for local_state in 0..segment.state_count {
                let state = segment.state_offset + local_state;
                dfa.set_transitions_from_sorted_entries(state, segment.expanded_entries(state));
            }
        }
        dfa
    }

    /// Put two tokenizers with the same terminal-id domain under one fresh
    /// epsilon root without identifying any source states. This lets the exact
    /// state-equivalence machinery compare residual states across independently
    /// built full and synthesized lexers.
    pub(crate) fn disjoint_union_for_analysis(
        left: &Tokenizer,
        right: &Tokenizer,
    ) -> TokenizerAnalysisUnion {
        assert_eq!(
            left.num_terminals, right.num_terminals,
            "cross-tokenizer analysis requires one shared terminal-id domain",
        );

        let left_offset = 1u32;
        let right_offset = left_offset + left.dfa.num_states() as u32;
        let mut dfa = DFA::new(
            1usize
                .saturating_add(left.dfa.num_states())
                .saturating_add(right.dfa.num_states()),
        );
        let num_groups = left.num_terminals as usize;
        dfa.ensure_group_capacity(num_groups);

        for group in 0..num_groups {
            let left_set = *left.dfa.group_id_to_u8set(group as u32);
            let right_set = *right.dfa.group_id_to_u8set(group as u32);
            dfa.set_group_u8set(group as u32, left_set.union(&right_set));
        }

        let copy_source = |target: &mut DFA, source: &DFA, offset: u32| {
            for (state_index, state) in source.states().iter().enumerate() {
                let target_state = offset + state_index as u32;
                target.set_transitions_from_sorted_entries(
                    target_state,
                    state
                        .transitions
                        .iter()
                        .map(|(byte, &destination)| (byte, offset + destination))
                        .collect(),
                );
                for &destination in &state.epsilon_transitions {
                    target.add_epsilon_transition(target_state, offset + destination);
                }
                target.overwrite_state_metadata(
                    target_state,
                    state.finalizers.clone(),
                    source
                        .possible_future_group_ids(state_index as u32)
                        .clone(),
                );
            }
        };
        copy_source(&mut dfa, &left.dfa, left_offset);
        copy_source(&mut dfa, &right.dfa, right_offset);
        dfa.add_epsilon_transition(0, left_offset + left.start_state());
        dfa.add_epsilon_transition(0, right_offset + right.start_state());

        let mut root_futures = BitSet::new(num_groups);
        for terminal in left
            .possible_future_terminals_iter(left.start_state())
            .chain(right.possible_future_terminals_iter(right.start_state()))
        {
            root_futures.set(terminal as usize);
        }
        dfa.overwrite_state_metadata(0, BitSet::new(num_groups), root_futures);

        TokenizerAnalysisUnion {
            tokenizer: Tokenizer::from_parts(dfa, left.num_terminals, None),
            left_offset,
            right_offset,
        }
    }

    fn start_state(&self) -> u32 {
        0
    }

    fn num_terminals(&self) -> u32 {
        self.num_terminals
    }

    pub(crate) fn has_epsilon_transitions(&self) -> bool {
        self.dfa.has_epsilon_transitions()
    }

    #[inline]
    pub(crate) fn state_has_epsilon_transitions(&self, state: u32) -> bool {
        self.dfa
            .states()
            .get(state as usize)
            .is_some_and(|state| !state.epsilon_transitions.is_empty())
    }

    pub(crate) fn terminal_expr(&self, terminal: TerminalID) -> Option<&Expr> {
        self.exprs.as_deref()?.get(terminal as usize)
    }

    pub(crate) fn initial_epsilon_branch_count(&self) -> usize {
        self.dfa
            .states()
            .get(self.start_state() as usize)
            .map_or(0, |state| state.epsilon_transitions.len())
    }

    /// Return the deterministic scanner roots behind the special epsilon
    /// dispatch state produced by `build_regex_partitioned`.
    ///
    /// This is deliberately narrower than "has epsilon transitions".  The
    /// compiler can retain its scalar-state fast paths when the only live
    /// nondeterminism is a zero-byte fan-out from the global reset state into
    /// independently deterministic components.  Nullable-start isolation may
    /// leave an unreachable cloned dispatch state elsewhere in the DFA, so the
    /// predicate is based on the live reset shape rather than a whole-DFA scan.
    pub(crate) fn deterministic_dispatch_roots(&self) -> Option<&[u32]> {
        let start = self.dfa.states().get(self.start_state() as usize)?;
        if start.epsilon_transitions.len() < 2 || !start.transitions.is_empty() {
            return None;
        }
        if start.epsilon_transitions.iter().any(|&root| {
            self.dfa
                .states()
                .get(root as usize)
                .is_none_or(|state| !state.epsilon_transitions.is_empty())
        }) {
            return None;
        }
        Some(&start.epsilon_transitions)
    }

    #[inline]
    pub(crate) fn has_deterministic_dispatch(&self) -> bool {
        self.deterministic_dispatch_roots().is_some()
    }

    /// Return the closed, pairwise-disjoint state sets below the global
    /// epsilon dispatcher. Components may contain internal epsilon structure,
    /// but no byte or epsilon edge may cross between returned sets.
    pub(crate) fn disjoint_dispatch_components(&self) -> Option<Vec<Vec<u32>>> {
        let roots = self.deterministic_dispatch_roots()?;
        let mut owner = vec![usize::MAX; self.dfa.states().len()];
        owner[self.start_state() as usize] = roots.len();
        let mut components = Vec::with_capacity(roots.len());

        for (component_index, &root) in roots.iter().enumerate() {
            if owner.get(root as usize).copied().unwrap_or(roots.len()) != usize::MAX {
                return None;
            }
            let mut states = Vec::new();
            let mut stack = vec![root];
            while let Some(state) = stack.pop() {
                let slot = owner.get_mut(state as usize)?;
                if *slot == component_index {
                    continue;
                }
                if *slot != usize::MAX {
                    return None;
                }
                *slot = component_index;
                states.push(state);
                let dfa_state = self.dfa.states().get(state as usize)?;
                stack.extend(dfa_state.transitions.iter().map(|(_, &target)| target));
                stack.extend(dfa_state.epsilon_transitions.iter().copied());
            }
            if states.is_empty() {
                return None;
            }
            states.sort_unstable();
            components.push(states);
        }
        Some(components)
    }

    /// Scanner states to use after a terminal boundary.  A conventional DFA
    /// has one reset state.  A partitioned lexer has one deterministic reset
    /// state per component; keeping them separate avoids materializing their
    /// product while preserving cross-component terminal sequences inside one
    /// vocabulary token.
    pub(crate) fn deterministic_reset_states(&self) -> TokenizerStateSet {
        self.deterministic_dispatch_roots()
            .map(TokenizerStateSet::from_slice)
            .unwrap_or_else(|| TokenizerStateSet::from_buf([self.initial_state_id()]))
    }

    fn transitions_from(&self, state: u32) -> TokenizerTransitionsIter<'_> {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::Compressed {
                    segment,
                    state,
                    next_byte: 0,
                },
            };
        }
        TokenizerTransitionsIter {
            inner: self
                .dfa
                .states()
                .get(state as usize)
                .map_or(TokenizerTransitionsIterInner::Empty, |state| {
                    TokenizerTransitionsIterInner::Dense(state.transitions.iter())
                }),
        }
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            segment.fill_transition_row(state, row);
            return;
        }
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            let mut bytes = U8Set::empty();
            let local_state = state - segment.state_offset;
            let start = segment.row_offsets[local_state as usize] as usize;
            let end = segment.row_offsets[local_state as usize + 1] as usize;
            for &(class, target) in &segment.entries[start..end] {
                if target == local_state {
                    for &byte in segment.class_members[class as usize].iter() {
                        bytes.insert(byte);
                    }
                }
            }
            return bytes;
        }
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    fn transition_count(&self) -> usize {
        let compressed = self
            .compressed_transition_segments
            .iter()
            .map(|segment| segment.expanded_transition_count)
            .sum::<usize>();
        let dense = self
            .dfa
            .states()
            .iter()
            .enumerate()
            .filter(|(state, _)| {
                self.compressed_segment_for_state(*state as u32).is_none()
            })
            .map(|(_, state)| state.transitions.len())
            .sum::<usize>();
        compressed + dense
    }

    /// Detect nullable terminals (those that match the empty string) by
    /// inspecting start-state finalizers, remove them from the DFA, and return
    /// the set.  After this call the tokenizer no longer reports those
    /// terminals as matched at state 0.
    pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        self.singleton_epsilon_closures = OnceLock::new();
        let start = self.start_state();
        let initial_closure = self.dfa.epsilon_closure(&[start]);
        let mut nullable = BTreeSet::new();
        for &state in &initial_closure {
            nullable.extend(
                self.dfa
                    .finalizers(state)
                    .iter()
                    .map(|terminal| terminal as TerminalID),
            );
        }
        if nullable.is_empty() {
            return nullable;
        }

        // The whole initial epsilon closure represents the zero-byte scanner
        // configuration. A component root can also be reached later after a
        // byte transition (for example, a nullable `a*` terminal looping to its
        // root). Clearing its finalizers in place would then remove legitimate
        // non-empty matches. Clone the closure as the post-consumption version,
        // redirect byte entries and external epsilon entries to those clones,
        // and drain finalizers only from the original zero-byte closure.
        let original_state_count = self.dfa.num_states();
        let mut post_byte_state = vec![u32::MAX; original_state_count];
        for &state in &initial_closure {
            let clone = self.dfa.clone_state(state);
            post_byte_state[state as usize] = clone;
        }

        let in_initial_closure = |state: u32| {
            (state as usize) < post_byte_state.len()
                && post_byte_state[state as usize] != u32::MAX
        };

        // Rewrite the cloned closure so all of its internal epsilon structure
        // remains in the post-byte coordinate.
        for &state in &initial_closure {
            let clone = post_byte_state[state as usize];
            let clone_state = &mut self.dfa.states_mut()[clone as usize];
            for (_, target) in clone_state.transitions.iter_mut() {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
            for target in &mut clone_state.epsilon_transitions {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
        }

        // A byte edge always enters the post-byte coordinate. An epsilon edge
        // from outside the initial closure can only be traversed after input has
        // already been consumed, so it does too. Epsilon edges within the
        // original closure remain untouched for the initial zero-byte closure.
        for source in 0..original_state_count {
            let source_in_initial_closure = in_initial_closure(source as u32);
            let state = &mut self.dfa.states_mut()[source];
            for (_, target) in state.transitions.iter_mut() {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
            if !source_in_initial_closure {
                for target in &mut state.epsilon_transitions {
                    if in_initial_closure(*target) {
                        *target = post_byte_state[*target as usize];
                    }
                }
            }
        }

        for state in initial_closure {
            self.dfa.clear_finalizers_for_state(state);
        }
        self.dfa.recompute_possible_futures();
        nullable
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.compressed_segment_for_state(state)
            .map_or_else(|| self.dfa.step(state, byte), |segment| segment.transition(state, byte))
    }

    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet {
        if self.compressed_transition_segments.is_empty() {
            return self.dfa.step_all(states, byte);
        }
        if states.len() == 1 {
            let state = states[0];
            if self.dfa.states()[state as usize].epsilon_transitions.is_empty()
                && let Some(target) = self.step(state, byte)
                && self.dfa.states()[target as usize].epsilon_transitions.is_empty()
            {
                return TokenizerStateSet::from_buf([target]);
            }
        }
        let closure = self.dfa.epsilon_closure(states);
        let mut targets = TokenizerStateSet::new();
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
        self.dfa.epsilon_closure(&targets)
    }

    fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(u32::MAX)
    }

    pub fn run(&self, input: &[u8]) -> TokenizerStateSet {
        self.scan_input(input, self.start_state(), &mut (), |_, _, _, _| {})
    }

    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.dfa
            .epsilon_closure(&[state])
            .into_iter()
            .flat_map(|state| self.matched_terminals_iter(state))
            .collect()
    }

    pub(crate) fn all_singleton_epsilon_closures(&self) -> Arc<[Box<[u32]>]> {
        Arc::clone(self.singleton_epsilon_closures.get_or_init(|| {
            Arc::from(self.dfa.all_singleton_epsilon_closures())
        }))
    }

    pub(crate) fn singleton_epsilon_closure(&self, state: u32) -> Box<[u32]> {
        self.dfa.epsilon_closure(&[state]).into_boxed_slice()
    }

    fn matched_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn matched_terminal_bitset(&self, state: u32) -> &BitSet {
        self.dfa.finalizers(state)
    }

    fn possible_future_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .possible_future_group_ids(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.dfa.possible_future_group_ids(state)
    }

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32 {
        self.dfa.num_states() as u32
    }

    fn num_forced_minimized_states(&self) -> usize {
        self.dfa.minimize().num_states()
    }

    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult {
        let mut matches = Vec::new();
        let mut end_states = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_all_matches(matches, state, width);
        });
        end_states.retain(|state| !self.is_end(*state));

        TokenizerExecResult {
            end_state: end_states,
            matches,
        }
    }

    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, TokenizerStateSet)>::default();
        let end_states = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state: end_states,
            matches: into_longest_matches(matches),
        }
    }

    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_states = if exec.end_state.is_empty() {
            SmallVec::from_buf([start])
        } else {
            exec.end_state
        };
        TokenizerResult {
            end_state: end_states,
            matches: group_matches_by_width(exec.matches),
        }
    }

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    /// Scan input bytes and report which terminals of interest matched/finalized.
    ///
    /// Returns a bitset of matched terminals and an optional end state.
    ///
    /// Algorithm:
    /// 1. `remaining = terminals_of_interest`.
    /// 2. `matched = empty`.
    /// 3. For each byte:
    ///    - Check if current state's possible futures overlap `remaining`.
    ///      If not, return `(matched, None)`.
    ///    - Consume byte → next state.
    ///    - If no transition, return `(matched, None)`.
    ///    - Get finalizers at next state, intersect with `remaining`.
    ///    - Add intersection to `matched`, remove from `remaining`.
    /// 4. After all bytes, check futures at end state overlap `remaining`.
    ///    If not, return `(matched, None)`. Otherwise `(matched, Some(end_state))`.
    ///
    /// Important: initial-state finalizers are intentionally ignored.
    /// Only post-byte finalizers count.
    ///
    /// `terminals_of_interest` must have length equal to `self.num_terminals`.
    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, TokenizerStateSet) {
        debug_assert_eq!(terminals_of_interest.len(), self.num_terminals as usize);
        let mut remaining = terminals_of_interest.clone();
        let mut matched = BitSet::new(self.num_terminals as usize);
        let mut states = self.dfa.epsilon_closure(&[start]);

        for &byte in input {
            let any_future = states
                .iter()
                .any(|&state| !self.possible_future_terminals(state).is_disjoint(&remaining));
            if !any_future {
                return (matched, TokenizerStateSet::new());
            }

            states = self.step_all(&states, byte);
            if states.is_empty() {
                return (matched, states);
            }

            let mut finals = BitSet::new(self.num_terminals as usize);
            for &state in &states {
                finals.union_with(&self.dfa.finalizers(state).intersection(&remaining));
            }
            matched.union_with(&finals);
            remaining = remaining.difference(&finals);
        }

        states.retain(|state| !self.possible_future_terminals(*state).is_disjoint(&remaining));
        (matched, states)
    }

    fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: u32, width: usize) {
        matches.extend(self.matched_terminals_iter(state).map(|id| TokenizerMatch {
            id,
            width,
            end_state: state,
        }));
    }

    fn record_longest_matches(
        &self,
        matches: &mut FxHashMap<TerminalID, (usize, TokenizerStateSet)>,
        state: u32,
        width: usize,
    ) {
        for terminal in self.matched_terminals_iter(state) {
            let entry = matches
                .entry(terminal)
                .or_insert_with(|| (width, TokenizerStateSet::new()));
            if width > entry.0 {
                entry.0 = width;
                entry.1.clear();
            }
            if width == entry.0 && !entry.1.contains(&state) {
                entry.1.push(state);
            }
        }
    }

    fn scan_input<R>(
        &self,
        input: &[u8],
        start: u32,
        mut matches: &mut R,
        mut record_matches: impl FnMut(&Self, &mut R, u32, usize),
    ) -> TokenizerStateSet {
        let mut states = self.dfa.epsilon_closure(&[start]);
        for (index, &byte) in input.iter().enumerate() {
            states = self.step_all(&states, byte);
            if states.is_empty() {
                return states;
            }
            for &state in &states {
                record_matches(self, &mut matches, state, index + 1);
            }
        }
        states
    }


}

impl Lexer for Tokenizer {
    fn start_state(&self) -> u32 { self.start_state() }
    fn num_terminals(&self) -> u32 { self.num_terminals() }
    fn has_epsilon_transitions(&self) -> bool { self.has_epsilon_transitions() }
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ { self.transitions_from(state) }
    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) { self.fill_transition_row(state, row); }
    fn transition_row(&self, state: u32) -> Box<[u32; 256]> { self.transition_row(state) }
    fn self_loop_bytes(&self, state: u32) -> U8Set { self.self_loop_bytes(state) }
    fn transition_count(&self) -> usize { self.transition_count() }
    fn step(&self, state: u32, byte: u8) -> Option<u32> { self.step(state, byte) }
    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet { self.step_all(states, byte) }
    fn get_transition(&self, state: u32, byte: u8) -> u32 { self.get_transition(state, byte) }
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet { self.matched_terminal_bitset(state) }
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.matched_terminals_iter(state) }
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.possible_future_terminals_iter(state) }
    fn possible_future_terminals(&self, state: u32) -> &BitSet { self.possible_future_terminals(state) }
    fn is_end(&self, state: u32) -> bool { self.is_end(state) }
    fn num_states(&self) -> u32 { self.num_states() }
    fn num_forced_minimized_states(&self) -> usize { self.num_forced_minimized_states() }
    fn execute_from_state_all_widths(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state_all_widths(input, start) }
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state(input, start) }
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet { self.execute_from_state_end_only(input, start) }
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult { self.execute_all_matches(input, start) }
    fn initial_state(&self) -> u32 { self.initial_state() }
    fn initial_state_id(&self) -> u32 { self.initial_state_id() }
    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet { self.tokens_accessible_from_state(state) }
    fn scan_terminal_matches_from_state(&self, input: &[u8], start: u32, terminals_of_interest: &BitSet) -> (BitSet, TokenizerStateSet) {
        self.scan_terminal_matches_from_state(input, start, terminals_of_interest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: TokenizerStateSet,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

#[cfg(test)]
pub(crate) fn arbitrary_epsilon_l1_test_tokenizer() -> Tokenizer {
    let mut dfa = DFA::new(7);
    dfa.ensure_group_capacity(2);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(1, 2);
    dfa.add_epsilon_transition(1, 4);
    dfa.add_transition(2, b'a', 3);
    dfa.add_transition(4, b'a', 5);
    dfa.add_transition(2, b'b', 6);

    let mut terminal_zero = BitSet::new(2);
    terminal_zero.set(0);
    dfa.overwrite_state_metadata(3, terminal_zero.clone(), BitSet::new(2));
    dfa.overwrite_state_metadata(6, terminal_zero, BitSet::new(2));
    let mut terminal_one = BitSet::new(2);
    terminal_one.set(1);
    dfa.overwrite_state_metadata(5, terminal_one, BitSet::new(2));
    dfa.recompute_possible_futures();

    let tokenizer = Tokenizer::from_parts(dfa, 2, None);
    assert!(tokenizer.has_epsilon_transitions());
    assert!(!tokenizer.has_deterministic_dispatch());
    tokenizer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::dfa::DFA;

    fn dispatch_prefix_tokenizer(with_appended_residual: bool) -> Tokenizer {
        let mut dfa = DFA::new(if with_appended_residual { 5 } else { 4 });
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 3);
        dfa.add_transition(1, b'a', 2);
        dfa.add_transition(2, b'a', 2);
        dfa.add_transition(3, b'x', 3);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(2, accepting.clone(), BitSet::new(1));
        if with_appended_residual {
            // This state is deliberately not reset-reachable. It models the
            // externally-entered residuals appended by structural synthesis.
            dfa.add_transition(4, b'a', 2);
            dfa.add_transition(4, b'b', 4);
            dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
        }
        dfa.recompute_possible_futures();
        Tokenizer::from_parts(dfa, 1, None)
    }

    #[test]
    fn structural_prefix_augmentation_clones_only_appended_residuals() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();
        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

        let source_to_local = local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .expect("verified append-only component relation");

        assert_eq!(local.num_states(), source.num_states());
        assert_eq!(source_to_local, vec![0, 1, 2, 3, 4]);
        for input in [b"".as_slice(), b"a", b"b", b"ba", b"bba"] {
            let source_result = source.execute_from_state_all_widths(input, 4);
            let local_result = local.execute_from_state_all_widths(input, source_to_local[4]);
            assert_eq!(source_result.matches, local_result.matches, "input={input:?}");
            assert_eq!(source_result.end_state, local_result.end_state, "input={input:?}");
        }
    }

    #[test]
    fn execution_handles_epsilon_edges_before_and_after_a_byte() {
        let mut dfa = DFA::new(6);
        dfa.ensure_group_capacity(2);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(1, 2);
        dfa.add_epsilon_transition(2, 1);
        dfa.add_transition(1, b'a', 3);
        dfa.add_transition(2, b'a', 4);
        dfa.add_epsilon_transition(3, 5);

        let mut terminal_zero = BitSet::new(2);
        terminal_zero.set(0);
        dfa.overwrite_state_metadata(5, terminal_zero, BitSet::new(2));
        let mut terminal_one = BitSet::new(2);
        terminal_one.set(1);
        dfa.overwrite_state_metadata(4, terminal_one, BitSet::new(2));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 2, None);
        let execution = tokenizer.execute_from_state_all_widths(b"a", 0);
        let mut matches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        assert_eq!(matches, vec![(0, 1), (1, 1)]);
        assert!(execution.end_state.is_empty());
        let longest = tokenizer.execute_from_state(b"a", 0);
        assert_eq!(longest.end_state.as_slice(), &[3, 4, 5]);
        assert_eq!(tokenizer.matched_terminals(3), BTreeSet::from([0]));

        let interests = BitSet::all(2);
        let (matched, continuation) =
            tokenizer.scan_terminal_matches_from_state(b"a", 0, &interests);
        assert!(matched.contains(0));
        assert!(matched.contains(1));
        assert!(continuation.is_empty());
    }

    #[test]
    fn draining_nullable_initial_closure_preserves_later_root_matches() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_transition(1, b'a', 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert_eq!(tokenizer.matched_terminals(0), BTreeSet::from([0]));
        assert_eq!(
            tokenizer.isolate_start_state_and_drain_nullable_terminals(),
            BTreeSet::from([0]),
        );
        assert!(tokenizer.matched_terminals(0).is_empty());

        let one = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert!(one.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
        let two = tokenizer.execute_from_state(b"aa", tokenizer.initial_state());
        assert!(two.matches.iter().any(|matched| matched.id == 0 && matched.width == 2));
    }

    #[test]
    fn longest_match_preserves_every_accepting_end_state_for_one_terminal() {
        let mut dfa = DFA::new(5);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 2);
        dfa.add_transition(1, b'a', 3);
        dfa.add_transition(2, b'a', 4);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 1, None);
        let mut end_states = tokenizer
            .execute_from_state(b"a", 0)
            .matches
            .into_iter()
            .filter(|matched| matched.id == 0 && matched.width == 1)
            .map(|matched| matched.end_state)
            .collect::<Vec<_>>();
        end_states.sort_unstable();
        assert_eq!(end_states, vec![3, 4]);
    }
}
