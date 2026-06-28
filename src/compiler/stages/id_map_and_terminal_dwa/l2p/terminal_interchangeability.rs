//! Strict terminal interchangeability for the L2+ terminal-DWA reference path.
//!
//! For one vocabulary partition, interchangeability is computed on the tokenizer
//! DFA restricted to that partition's bytes. Only terminals active in this L2+
//! phase are observable. A pair is interchangeable when the original restricted
//! DFA and the DFA with those output labels swapped have a bijection between
//! their residual-state partitions.
//!
//! This is intentionally validation-first. For every accepted
//! representative/member swap, the builder constructs a restricted residual
//! scanner. During the trie walk it keeps the raw lexer state for token-boundary
//! semantics, while the union of residual scanners supplies completed terminal
//! labels. The resulting local DWA is checked exactly against the baseline.
//! Directed subsumption is deliberately excluded.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::grammar::flat::TerminalID;

use super::nwa_builder::{TerminalNwaTransportMachine, TerminalNwaTransportMode};

const NO_STATE: u32 = u32::MAX;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct OutputBits(Vec<u64>);

impl OutputBits {
    fn new(words: usize) -> Self {
        Self(vec![0; words])
    }

    fn set(&mut self, terminal: usize) {
        self.0[terminal / 64] |= 1u64 << (terminal % 64);
    }

    fn contains(&self, terminal: usize) -> bool {
        (self.0[terminal / 64] & (1u64 << (terminal % 64))) != 0
    }

    fn swap(&self, left: usize, right: usize) -> Self {
        if left == right {
            return self.clone();
        }
        let mut result = self.clone();
        let left_word = left / 64;
        let right_word = right / 64;
        let left_mask = 1u64 << (left % 64);
        let right_mask = 1u64 << (right % 64);
        if ((self.0[left_word] & left_mask) != 0) != ((self.0[right_word] & right_mask) != 0) {
            result.0[left_word] ^= left_mask;
            result.0[right_word] ^= right_mask;
        }
        result
    }
}

/// Bijection from the original minimized restricted-DFA states to the states
/// representing their swapped-output residuals.
#[derive(Clone, Debug)]
struct InterchangeMap {
    target_block_for_source_block: Vec<u32>,
}

/// The minimized restricted lexer DFA. A terminal-label swap only relabels this
/// machine's outputs; it does not change its transition graph or its intrinsic
/// state-equivalence partition.
#[derive(Clone, Debug)]
struct MinimizedResidualDfa {
    transitions: Vec<Vec<u32>>,
    outputs: Vec<OutputBits>,
}

impl MinimizedResidualDfa {
    fn state_count(&self) -> usize {
        self.outputs.len()
    }

    fn structural_colors(&self) -> Vec<u32> {
        let state_count = self.state_count();
        let cardinality = |state: usize| self.outputs[state]
            .0
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>();
        let mut colors = classify_keys((0..state_count).map(&cardinality));
        loop {
            let keys = (0..state_count)
                .map(|state| {
                    let successors = self.transitions[state]
                        .iter()
                        .map(|&target| colors[target as usize])
                        .collect::<Vec<_>>();
                    (cardinality(state), successors)
                })
                .collect::<Vec<_>>();
            let next = classify_keys(keys);
            if next == colors {
                return colors;
            }
            colors = next;
        }
    }

    /// Group terminals by a necessary structural signature for an output-label
    /// swap. Any swap automorphism preserves the cardinality-only transition
    /// colors, hence it preserves the multiset of colors on which each terminal
    /// appears. Different groups cannot be interchangeable.
    fn compatible_terminal_groups(
        &self,
        terminals: &[TerminalID],
    ) -> BTreeMap<Vec<u32>, Vec<TerminalID>> {
        let colors = self.structural_colors();
        let mut profiles = vec![Vec::<u32>::new(); terminals.len()];
        for (state, output) in self.outputs.iter().enumerate() {
            let color = colors[state];
            for (index, &terminal) in terminals.iter().enumerate() {
                if output.contains(terminal as usize) {
                    profiles[index].push(color);
                }
            }
        }
        let mut groups = BTreeMap::<Vec<u32>, Vec<TerminalID>>::new();
        for (terminal, mut profile) in terminals.iter().copied().zip(profiles) {
            profile.sort_unstable();
            groups.entry(profile).or_default().push(terminal);
        }
        groups
    }

    /// Minimize the disjoint union of the original-output and swapped-output
    /// copies of this already-minimized DFA. The result gives the exact
    /// cross-copy residual equivalence relation for one proposed swap.
    fn minimize_original_and_swapped(&self, left: usize, right: usize) -> Vec<u32> {
        let state_count = self.state_count();
        let combined_count = state_count * 2;
        let output = |combined: usize| {
            let copy = combined / state_count;
            let state = combined % state_count;
            if copy == 0 {
                self.outputs[state].clone()
            } else {
                self.outputs[state].swap(left, right)
            }
        };
        let mut blocks = classify_keys((0..combined_count).map(&output));
        loop {
            let keys = (0..combined_count)
                .map(|combined| {
                    let copy = combined / state_count;
                    let state = combined % state_count;
                    let successors = self.transitions[state]
                        .iter()
                        .map(|&target| blocks[copy * state_count + target as usize])
                        .collect::<Vec<_>>();
                    (output(combined), successors)
                })
                .collect::<Vec<_>>();
            let next = classify_keys(keys);
            if next == blocks {
                return blocks;
            }
            blocks = next;
        }
    }
}

struct RestrictedDfa<'a> {
    tokenizer: &'a Tokenizer,
    active_terminals: &'a [bool],
    bytes: Vec<u8>,
    real_state_count: usize,
    output_words: usize,
}

impl<'a> RestrictedDfa<'a> {
    fn new(
        tokenizer: &'a Tokenizer,
        active_terminals: &'a [bool],
        relevant_bytes: &[bool; 256],
    ) -> Self {
        Self {
            tokenizer,
            active_terminals,
            bytes: (0..=255u8)
                .filter(|&byte| relevant_bytes[byte as usize])
                .collect(),
            real_state_count: tokenizer.num_states() as usize,
            output_words: (tokenizer.num_terminals() as usize).div_ceil(64),
        }
    }

    fn state_count(&self) -> usize {
        self.real_state_count + 1
    }

    fn dead_state(&self) -> usize {
        self.real_state_count
    }

    fn output(&self, state: usize, swap: Option<(usize, usize)>) -> OutputBits {
        if state == self.dead_state() {
            return OutputBits::new(self.output_words);
        }
        let mut output = OutputBits::new(self.output_words);
        for terminal in self.tokenizer.matched_terminals_iter(state as u32) {
            if self
                .active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
            {
                output.set(terminal as usize);
            }
        }
        match swap {
            Some((left, right)) => output.swap(left, right),
            None => output,
        }
    }

    fn successor(&self, state: usize, byte_slot: usize) -> usize {
        if state == self.dead_state() {
            return state;
        }
        let next = self.tokenizer.get_transition(state as u32, self.bytes[byte_slot]);
        if next == NO_STATE {
            self.dead_state()
        } else {
            next as usize
        }
    }

    fn minimize(&self, swap: Option<(usize, usize)>) -> Vec<u32> {
        let state_count = self.state_count();
        let mut blocks = classify_keys((0..state_count).map(|state| self.output(state, swap)));
        loop {
            let keys = (0..state_count)
                .map(|state| {
                    let successors = (0..self.bytes.len())
                        .map(|slot| blocks[self.successor(state, slot)])
                        .collect::<Vec<_>>();
                    (self.output(state, swap), successors)
                })
                .collect::<Vec<_>>();
            let next = classify_keys(keys);
            if next == blocks {
                return blocks;
            }
            blocks = next;
        }
    }

    fn minimized_residual(&self, blocks: &[u32]) -> MinimizedResidualDfa {
        assert_eq!(blocks.len(), self.state_count());
        let count = blocks.iter().copied().max().map_or(0, |id| id as usize + 1);
        let mut representatives = vec![None::<usize>; count];
        for (state, &block) in blocks.iter().enumerate() {
            representatives[block as usize].get_or_insert(state);
        }
        let mut transitions = vec![vec![0u32; self.bytes.len()]; count];
        let mut outputs = Vec::with_capacity(count);
        for (block, state) in representatives.into_iter().enumerate() {
            let state = state.expect("every minimized residual block has a representative");
            for slot in 0..self.bytes.len() {
                transitions[block][slot] = blocks[self.successor(state, slot)];
            }
            outputs.push(self.output(state, None));
        }
        MinimizedResidualDfa {
            transitions,
            outputs,
        }
    }

    fn residual_machine(
        &self,
        blocks: &[u32],
        swap: Option<(usize, usize)>,
        emit: &[bool],
    ) -> TerminalNwaTransportMachine {
        assert_eq!(blocks.len(), self.state_count());
        let count = blocks.iter().copied().max().map_or(0, |id| id as usize + 1);
        let mut representative = vec![None::<usize>; count];
        for (state, &block) in blocks.iter().enumerate() {
            representative[block as usize].get_or_insert(state);
        }
        let mut byte_slot = Box::new([-1i16; 256]);
        for (slot, &byte) in self.bytes.iter().enumerate() {
            byte_slot[byte as usize] = slot as i16;
        }
        let mut transitions = vec![vec![0u32; self.bytes.len()]; count];
        let mut matched_terminals = vec![Vec::<TerminalID>::new(); count];
        for (block, state) in representative.into_iter().enumerate() {
            let state = state.expect("every residual block has a representative");
            for slot in 0..self.bytes.len() {
                transitions[block][slot] = blocks[self.successor(state, slot)];
            }
            let output = self.output(state, swap);
            for terminal in 0..emit.len() {
                if emit[terminal] && output.contains(terminal) {
                    matched_terminals[block].push(terminal as TerminalID);
                }
            }
        }
        TerminalNwaTransportMachine::new(byte_slot, transitions, matched_terminals)
    }

    fn interchange_map(
        &self,
        residual: &MinimizedResidualDfa,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let left = left as usize;
        let right = right as usize;
        let state_count = residual.state_count();
        if left == right {
            return Some(InterchangeMap {
                target_block_for_source_block: (0..state_count as u32).collect(),
            });
        }

        // A global output-label renaming preserves the DFA's state-equivalence
        // relation. Work on the minimized residual machine, then refine only
        // the relation between its original and swapped-output copies.
        let combined_blocks = residual.minimize_original_and_swapped(left, right);
        let mut target_for_source = vec![None::<u32>; state_count];
        for source in 0..state_count {
            let mut target = None;
            for candidate in 0..state_count {
                if combined_blocks[state_count + candidate] != combined_blocks[source] {
                    continue;
                }
                match target {
                    Some(existing) if existing != candidate as u32 => return None,
                    Some(_) => {}
                    None => target = Some(candidate as u32),
                }
            }
            target_for_source[source] = Some(target?);
        }
        let target_block_for_source_block = target_for_source
            .into_iter()
            .collect::<Option<Vec<_>>>()?;
        let mut seen = vec![false; state_count];
        for &target in &target_block_for_source_block {
            let slot = seen.get_mut(target as usize)?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        if seen.iter().any(|&seen| !seen) {
            return None;
        }
        Some(InterchangeMap {
            target_block_for_source_block,
        })
    }
}

fn classify_keys<K: Ord>(keys: impl IntoIterator<Item = K>) -> Vec<u32> {
    let mut ids = BTreeMap::<K, u32>::new();
    keys
        .into_iter()
        .map(|key| {
            let next = ids.len() as u32;
            *ids.entry(key).or_insert(next)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub(crate) struct TerminalInterchangeability {
    original_active: Vec<bool>,
    active_representatives: Vec<bool>,
    representative_for: Vec<TerminalID>,
    members_by_representative: Vec<Vec<TerminalID>>,
    maps_by_representative_member: BTreeMap<(TerminalID, TerminalID), InterchangeMap>,
    source_blocks: Vec<u32>,
}

impl TerminalInterchangeability {
    pub(crate) fn identity(active: &[bool]) -> Self {
        let terminal_count = active.len();
        Self {
            original_active: active.to_vec(),
            active_representatives: active.to_vec(),
            representative_for: (0..terminal_count as u32).collect(),
            members_by_representative: (0..terminal_count as u32).map(|terminal| vec![terminal]).collect(),
            maps_by_representative_member: BTreeMap::new(),
            source_blocks: Vec::new(),
        }
    }

    pub(crate) fn build(
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
        ignore_terminal: Option<TerminalID>,
    ) -> Self {
        let candidates = active_terminals.iter().enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if candidates.len() < 2 { return Self::identity(active_terminals); }

        let restricted = RestrictedDfa::new(tokenizer, active_terminals, relevant_bytes);
        let source_blocks = restricted.minimize(None);
        let residual = restricted.minimized_residual(&source_blocks);
        let candidate_groups = residual.compatible_terminal_groups(&candidates);
        let candidate_pairs = candidate_groups
            .values()
            .map(|group| group.len().saturating_sub(1) * group.len() / 2)
            .sum::<usize>();
        if std::env::var_os("GLRMASK_PROFILE_L2P_INTERCHANGEABILITY").is_some() {
            eprintln!(
                "[glrmask/profile][l2p_terminal_interchangeability] active_terminals={} residual_states={} structural_groups={} candidate_pairs={} total_pairs={}",
                candidates.len(),
                residual.state_count(),
                candidate_groups.len(),
                candidate_pairs,
                candidates.len().saturating_sub(1) * candidates.len() / 2,
            );
        }
        let mut accepted = BTreeMap::<(TerminalID, TerminalID), InterchangeMap>::new();
        let mut components = DisjointSet::new(active_terminals.len());
        for group in candidate_groups.values() {
            for (index, &left) in group.iter().enumerate() {
                for &right in &group[index + 1..] {
                    if let Some(left_to_right) =
                        restricted.interchange_map(&residual, left, right)
                    {
                        assert!(
                            restricted
                                .interchange_map(&residual, right, left)
                                .is_some(),
                            "terminal interchange map was not symmetric: {left} <-> {right}",
                        );
                        components.union(left as usize, right as usize);
                        accepted.insert((left, right), left_to_right);
                    }
                }
            }
        }

        let mut groups = BTreeMap::<usize, Vec<TerminalID>>::new();
        for &terminal in &candidates {
            groups.entry(components.find(terminal as usize)).or_default().push(terminal);
        }

        let mut result = Self::identity(active_terminals);
        result.source_blocks = source_blocks;
        for members in groups.into_values() {
            if members.len() < 2 { continue; }
            // The definition makes this an equivalence relation. Fail closed if
            // the implementation ever produces only a transitive DSU chain.
            for (index, &left) in members.iter().enumerate() {
                for &right in &members[index + 1..] {
                    assert!(
                        accepted.contains_key(&(left, right)),
                        "terminal interchangeability component was not a clique: {left} and {right}",
                    );
                }
            }
            let representative = *members.iter().min().expect("nonempty component");
            result.members_by_representative[representative as usize] = members.clone();
            for &member in &members {
                result.representative_for[member as usize] = representative;
                if member != representative {
                    result.active_representatives[member as usize] = false;
                    let map = accepted
                        .get(&(representative, member))
                        .expect("interchangeability clique pair missing a map")
                        .clone();
                    result.maps_by_representative_member.insert((representative, member), map);
                }
            }
        }
        if std::env::var_os("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY").is_some() {
            for (representative, members) in result.members_by_representative.iter().enumerate() {
                if members.len() < 2 {
                    continue;
                }
                eprintln!(
                    "[glrmask/debug][terminal_interchangeability] representative={} members={:?}",
                    representative,
                    members,
                );
                if std::env::var_os("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY_MAPS").is_some() {
                    for &member in members {
                        if member == representative as TerminalID {
                            continue;
                        }
                        let map = result
                            .maps_by_representative_member
                            .get(&(representative as TerminalID, member))
                            .expect("debug transport missing");
                        eprintln!(
                            "[glrmask/debug][terminal_interchangeability_transport] representative={} member={} map={:?}",
                            representative,
                            member,
                            map.target_block_for_source_block,
                        );
                    }
                }
            }
        }
        result
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.representative_for.iter().enumerate()
            .all(|(terminal, &representative)| terminal as TerminalID == representative)
    }

    pub(crate) fn active_representatives(&self) -> &[bool] { &self.active_representatives }
    pub(crate) fn active_terminal_count_before(&self) -> usize {
        self.original_active.iter().filter(|&&active| active).count()
    }
    pub(crate) fn active_terminal_count_after(&self) -> usize {
        self.active_representatives.iter().filter(|&&active| active).count()
    }

    pub(crate) fn terminal_nwa_transport_modes(
        &self,
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
    ) -> Option<Vec<TerminalNwaTransportMode>> {
        if self.is_identity() {
            return None;
        }
        let restricted = RestrictedDfa::new(tokenizer, &self.original_active, relevant_bytes);
        let source_blocks = &self.source_blocks;
        debug_assert_eq!(source_blocks.len(), restricted.state_count());
        let identity_machine = Arc::new(restricted.residual_machine(
            source_blocks,
            None,
            &self.active_representatives,
        ));
        let mut modes = vec![TerminalNwaTransportMode {
            logical_state_for_original: source_blocks[..restricted.real_state_count]
                .iter()
                .copied()
                .collect(),
            machine: identity_machine,
        }];

        for (representative, members) in self.members_by_representative.iter().enumerate() {
            let representative = representative as TerminalID;
            for &member in members {
                if member == representative {
                    continue;
                }
                let map = self.maps_by_representative_member.get(&(representative, member))?;
                let logical_state_for_original = source_blocks[..restricted.real_state_count]
                    .iter()
                    .map(|&source| map.target_block_for_source_block[source as usize])
                    .collect::<Vec<_>>();
                let mut emit_member = vec![false; self.original_active.len()];
                emit_member[member as usize] = true;
                modes.push(TerminalNwaTransportMode {
                    logical_state_for_original,
                    machine: Arc::new(restricted.residual_machine(
                        source_blocks,
                        Some((representative as usize, member as usize)),
                        &emit_member,
                    )),
                });
            }
        }
        Some(modes)
    }



}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect(), rank: vec![0; size] }
    }

    fn find(&mut self, item: usize) -> usize {
        if self.parent[item] != item {
            self.parent[item] = self.find(self.parent[item]);
        }
        self.parent[item]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right { return; }
        if self.rank[left] < self.rank[right] { std::mem::swap(&mut left, &mut right); }
        self.parent[right] = left;
        if self.rank[left] == self.rank[right] { self.rank[left] += 1; }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    #[test]
    fn rotated_residuals_form_an_l2p_terminal_partition() {
        let expressions = vec![
            Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
            Expr::Seq(vec![Expr::U8Seq(b"aaa".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert_eq!(plan.active_terminal_count_before(), 2);
        assert_eq!(plan.active_terminal_count_after(), 1);
        assert!(plan.terminal_nwa_transport_modes(&tokenizer, &[true; 256]).is_some());
    }

    #[test]
    fn residual_modes_follow_raw_transitions_and_restore_all_labels() {
        let expressions = vec![
            Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
            Expr::Seq(vec![Expr::U8Seq(b"aaa".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let active = [true, true];
        let plan = TerminalInterchangeability::build(&tokenizer, &active, &[true; 256], None);
        let modes = plan
            .terminal_nwa_transport_modes(&tokenizer, &[true; 256])
            .expect("rotated terminals must produce residual modes");

        for state in 0..tokenizer.num_states() {
            let mut expected = tokenizer.matched_terminals_iter(state).collect::<Vec<_>>();
            expected.sort_unstable();
            let mut actual = modes
                .iter()
                .flat_map(|mode| {
                    mode.machine
                        .matched_terminals(mode.logical_state_for_original[state as usize])
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            actual.sort_unstable();
            actual.dedup();
            assert_eq!(actual, expected, "residual labels differed at lexer state {state}");

            for byte in 0..=255u8 {
                let next = tokenizer.get_transition(state, byte);
                if next == NO_STATE {
                    continue;
                }
                for mode in &modes {
                    let logical = mode.logical_state_for_original[state as usize];
                    assert_eq!(
                        mode.machine.step(logical, byte),
                        Some(mode.logical_state_for_original[next as usize]),
                        "residual transport did not commute at state={state} byte={byte}",
                    );
                }
            }
        }
    }

    #[test]
    fn byte_preserving_swap_rejects_distinct_literal_bytes() {
        let expressions = vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        assert!({ let blocks = dfa.minimize(None); let residual = dfa.minimized_residual(&blocks); dfa.interchange_map(&residual, 0, 1) }.is_none());
        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert!(plan.is_identity());
    }

    #[test]
    fn metadata_only_terminal_filter_preserves_state_ids_and_byte_transitions() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"aba".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let active = [true, false, true];
        let filtered = tokenizer.deactivate_terminals_without_minimizing(&active);
        assert_eq!(filtered.num_states(), tokenizer.num_states());
        for state in 0..tokenizer.num_states() {
            for byte in 0..=255u8 {
                assert_eq!(
                    filtered.get_transition(state, byte),
                    tokenizer.get_transition(state, byte),
                );
            }
            let expected_matches = tokenizer
                .matched_terminals_iter(state)
                .filter(|&terminal| active[terminal as usize])
                .collect::<Vec<_>>();
            assert_eq!(filtered.matched_terminals_iter(state).collect::<Vec<_>>(), expected_matches);
            let expected_futures = tokenizer
                .possible_future_terminals_iter(state)
                .filter(|&terminal| active[terminal as usize])
                .collect::<Vec<_>>();
            assert_eq!(
                filtered.possible_future_terminals_iter(state).collect::<Vec<_>>(),
                expected_futures,
            );
        }
    }

    #[test]
    fn restricted_byte_alphabet_omits_unlisted_transitions() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"z".to_vec())),
                    min: 0,
                    max: Some(1),
                },
            ]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let after_a = tokenizer.get_transition(tokenizer.initial_state_id(), b'a');
        assert_ne!(tokenizer.get_transition(after_a, b'z'), NO_STATE);

        let mut only_a = [false; 256];
        only_a[b'a' as usize] = true;
        let restricted = RestrictedDfa::new(&tokenizer, &[true, true], &only_a);
        assert_eq!(restricted.bytes, vec![b'a']);
        assert_eq!(restricted.bytes.len(), 1);
        assert_ne!(restricted.successor(after_a as usize, 0), tokenizer.get_transition(after_a, b'z') as usize);

        let unrestricted = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        assert_eq!(unrestricted.bytes.len(), 256);
        assert_eq!(unrestricted.successor(after_a as usize, b'z' as usize), tokenizer.get_transition(after_a, b'z') as usize);
    }

    #[test]
    fn inactive_outputs_are_not_observed() {
        let expressions = vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec()), Expr::U8Seq(b"a".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, false, true], &[true; 256]);
        assert!({ let blocks = dfa.minimize(None); let residual = dfa.minimized_residual(&blocks); dfa.interchange_map(&residual, 0, 2) }.is_some());
    }
}
