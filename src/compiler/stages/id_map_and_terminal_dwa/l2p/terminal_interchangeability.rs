//! Strict terminal interchangeability for the L2+ terminal-DWA reference path.
//!
//! For one vocabulary partition, interchangeability is computed on the tokenizer
//! DFA restricted to that partition's bytes. Only terminals active in this L2+
//! phase are observable. A pair is interchangeable when the original restricted
//! DFA and the DFA with those output labels swapped have a bijection between
//! their residual-state partitions.
//!
//! This is intentionally a validation-first construction: hide
//! nonrepresentatives before id-map/DWA construction, restore noninitial labels,
//! make one transported copy per relevant initial replacement, and use the
//! existing local DWA/id-map merger. The simple restoration is used only when
//! the transport fixes the lexer reset residual class. Directed subsumption is
//! deliberately excluded.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use super::nwa_builder::TerminalNwaTransportMode;
use crate::grammar::flat::TerminalID;

const NO_STATE: u32 = u32::MAX;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct OutputBits(Vec<u64>);

impl OutputBits {
    fn new(words: usize) -> Self { Self(vec![0; words]) }
    fn set(&mut self, terminal: usize) { self.0[terminal / 64] |= 1u64 << (terminal % 64); }

    fn swap(&self, left: usize, right: usize) -> Self {
        if left == right { return self.clone(); }
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

/// Concrete representation of a bijection between residual-state partitions.
/// A source lexer state maps to every concrete state in its target partition;
/// the relation is bijective at the partition level.
#[derive(Clone, Debug)]
struct InterchangeMap {
    source_state_to_target_states: Vec<Vec<u32>>,
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

    fn output(&self, state: usize, swap: Option<(usize, usize)>) -> (OutputBits, OutputBits) {
        if state == self.dead_state() {
            return (
                OutputBits::new(self.output_words),
                OutputBits::new(self.output_words),
            );
        }
        let mut finalizers = OutputBits::new(self.output_words);
        for terminal in self.tokenizer.matched_terminals_iter(state as u32) {
            if self
                .active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
            {
                finalizers.set(terminal as usize);
            }
        }
        let mut future = OutputBits::new(self.output_words);
        for terminal in self.tokenizer.possible_future_terminals_iter(state as u32) {
            if self
                .active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
            {
                future.set(terminal as usize);
            }
        }
        match swap {
            Some((left, right)) => (finalizers.swap(left, right), future.swap(left, right)),
            None => (finalizers, future),
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

    fn minimize_original_and_swapped(&self, left: usize, right: usize) -> Vec<u32> {
        let state_count = self.state_count();
        let combined_count = state_count * 2;
        let mut blocks = classify_keys((0..combined_count).map(|combined| {
            let copy = combined / state_count;
            let state = combined % state_count;
            self.output(state, (copy == 1).then_some((left, right)))
        }));
        loop {
            let keys = (0..combined_count)
                .map(|combined| {
                    let copy = combined / state_count;
                    let state = combined % state_count;
                    let successors = (0..self.bytes.len())
                        .map(|slot| blocks[copy * state_count + self.successor(state, slot)])
                        .collect::<Vec<_>>();
                    (
                        self.output(state, (copy == 1).then_some((left, right))),
                        successors,
                    )
                })
                .collect::<Vec<_>>();
            let next = classify_keys(keys);
            if next == blocks {
                return blocks;
            }
            blocks = next;
        }
    }

    /// Return a concrete DFA automorphism for this label swap, if the
    /// residual partition map has a singleton, transition-commuting lift.
    ///
    /// The terminal-NWA transport executes concrete lexer states, so a
    /// quotient-level residual bijection alone is not sufficient here.
    #[cfg(test)]
    fn concrete_swap_automorphism(
        &self,
        map: &InterchangeMap,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<Vec<u32>> {
        let mapping = map
            .source_state_to_target_states
            .iter()
            .map(|targets| (targets.len() == 1).then_some(targets[0]))
            .collect::<Option<Vec<_>>>()?;
        let mut seen = vec![false; self.real_state_count];
        for &target in &mapping {
            let slot = seen.get_mut(target as usize)?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        if seen.iter().any(|&seen| !seen) {
            return None;
        }

        let swap = Some((left as usize, right as usize));
        for source in 0..self.real_state_count {
            let target = mapping[source] as usize;
            if self.output(source, None) != self.output(target, swap) {
                return None;
            }
            for slot in 0..self.bytes.len() {
                let source_next = self.successor(source, slot);
                let target_next = self.successor(target, slot);
                if source_next == self.dead_state() {
                    if target_next != self.dead_state() {
                        return None;
                    }
                } else if target_next != mapping[source_next] as usize {
                    return None;
                }
            }
        }
        Some(mapping)
    }

    fn interchange_map(&self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        let state_count = self.state_count();
        let left = left as usize;
        let right = right as usize;
        if left == right {
            return Some(InterchangeMap {
                source_state_to_target_states: (0..self.real_state_count)
                    .map(|state| vec![state as u32])
                    .collect(),
            });
        }

        let combined_blocks = self.minimize_original_and_swapped(left, right);
        let source_blocks = self.minimize(None);
        let swapped_blocks = self.minimize(Some((left, right)));

        let source_count = source_blocks
            .iter()
            .copied()
            .max()
            .map_or(0, |block| block as usize + 1);
        let target_count = swapped_blocks
            .iter()
            .copied()
            .max()
            .map_or(0, |block| block as usize + 1);
        let mut target_states_by_block = vec![Vec::<u32>::new(); target_count];
        for state in 0..self.real_state_count {
            target_states_by_block[swapped_blocks[state] as usize].push(state as u32);
        }
        let mut source_states_by_block = vec![Vec::<u32>::new(); source_count];
        for source in 0..self.real_state_count {
            source_states_by_block[source_blocks[source] as usize].push(source as u32);
        }
        let mut source_to_target = vec![None::<u32>; source_count];

        for source in 0..self.real_state_count {
            let mut target_block = None;
            for target in 0..self.real_state_count {
                if combined_blocks[state_count + target] != combined_blocks[source] {
                    continue;
                }
                let candidate = swapped_blocks[target];
                match target_block {
                    Some(existing) if existing != candidate => return None,
                    Some(_) => {}
                    None => target_block = Some(candidate),
                }
            }
            let target_block = target_block?;
            let source_block = source_blocks[source] as usize;
            match source_to_target[source_block] {
                Some(existing) if existing != target_block => return None,
                Some(_) => {}
                None => source_to_target[source_block] = Some(target_block),
            }
        }

        let mut target_to_source = vec![None::<u32>; target_count];
        for (source, target) in source_to_target.iter().enumerate() {
            if source_states_by_block[source].is_empty() {
                continue;
            }
            let target = (*target)? as usize;
            match target_to_source[target] {
                Some(existing) if existing != source as u32 => return None,
                Some(_) => {}
                None => target_to_source[target] = Some(source as u32),
            }
        }
        if target_to_source.iter().enumerate().any(|(block, source)| {
            !target_states_by_block[block].is_empty() && source.is_none()
        }) {
            return None;
        }

        let source_state_to_target_states = (0..self.real_state_count)
            .map(|source| {
                let target = source_to_target[source_blocks[source] as usize]
                    .expect("checked above") as usize;
                target_states_by_block[target].clone()
            })
            .collect::<Vec<_>>();
        if source_state_to_target_states.iter().any(Vec::is_empty) {
            return None;
        }
        Some(InterchangeMap {
            source_state_to_target_states,
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

        // The L2+ active mask chooses which terminals may be coalesced.  It
        // does *not* limit the terminal-word alphabet of the local DWA: other
        // terminal labels can still occur as continuations and must remain
        // fixed observables of the residual automaton.
        let observable_terminals = vec![true; active_terminals.len()];
        let restricted = RestrictedDfa::new(tokenizer, &observable_terminals, relevant_bytes);
        let mut accepted = BTreeMap::<(TerminalID, TerminalID), InterchangeMap>::new();
        let mut components = DisjointSet::new(active_terminals.len());
        for (index, &left) in candidates.iter().enumerate() {
            for &right in &candidates[index + 1..] {
                if let Some(left_to_right) = restricted.interchange_map(left, right) {
                    assert!(
                        restricted.interchange_map(right, left).is_some(),
                        "terminal interchange map was not symmetric: {left} <-> {right}",
                    );
                    components.union(left as usize, right as usize);
                    accepted.insert((left, right), left_to_right);
                }
            }
        }

        let mut groups = BTreeMap::<usize, Vec<TerminalID>>::new();
        for &terminal in &candidates {
            groups.entry(components.find(terminal as usize)).or_default().push(terminal);
        }

        let mut result = Self::identity(active_terminals);
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
                            map.source_state_to_target_states,
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
    pub(crate) fn visible_terminal_labels(&self) -> Vec<bool> {
        // The L2+ phase may only coalesce terminals selected by its active mask,
        // but its DWA remains over the complete terminal alphabet. Hide exactly
        // the concrete members replaced by a class representative; keep every
        // other terminal visible, including terminals assigned to L1.
        self.representative_for
            .iter()
            .enumerate()
            .map(|(terminal, &representative)| terminal as TerminalID == representative)
            .collect()
    }

    pub(crate) fn active_terminal_count_before(&self) -> usize {
        self.original_active.iter().filter(|&&active| active).count()
    }
    pub(crate) fn active_terminal_count_after(&self) -> usize {
        self.active_representatives.iter().filter(|&&active| active).count()
    }

    pub(crate) fn terminal_nwa_transport_modes(&self) -> Option<Vec<TerminalNwaTransportMode>> {
        if self.is_identity() {
            return None;
        }
        let terminal_count = self.original_active.len();
        let identity_states = (0..self
            .maps_by_representative_member
            .values()
            .next()
            .map(|map| map.source_state_to_target_states.len())? as u32)
            .collect::<Vec<_>>();
        let identity_labels = (0..terminal_count as u32).collect::<Vec<_>>();
        let mut modes = vec![TerminalNwaTransportMode {
            scanner_state_for_original: identity_states,
            terminal_map: identity_labels.clone(),
        }];

        for (representative, members) in self.members_by_representative.iter().enumerate() {
            let representative = representative as TerminalID;
            for &member in members {
                if member == representative {
                    continue;
                }
                let map = self.maps_by_representative_member.get(&(representative, member))?;
                let scanner_state_for_original = map
                    .source_state_to_target_states
                    .iter()
                    .map(|targets| targets.first().copied())
                    .collect::<Option<Vec<_>>>()?;
                let mut terminal_map = identity_labels.clone();
                terminal_map[representative as usize] = member;
                terminal_map[member as usize] = representative;
                modes.push(TerminalNwaTransportMode {
                    scanner_state_for_original,
                    terminal_map,
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
    fn strict_interchange_map_exists_for_rotated_residuals() {
        let expressions = vec![
            Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
            Expr::Seq(vec![Expr::U8Seq(b"aaa".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        let forward = dfa.interchange_map(0, 1).expect("rotated map must exist");
        assert!(dfa.concrete_swap_automorphism(&forward, 0, 1).is_some());
        assert!(dfa.interchange_map(1, 0).is_some());
    }

    #[test]
    fn transport_rejects_a_non_singleton_residual_target() {
        let expressions = vec![
            Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
            Expr::Seq(vec![Expr::U8Seq(b"aaa".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        let mut map = dfa.interchange_map(0, 1).expect("rotated map must exist");
        let duplicated_target = map.source_state_to_target_states[0][0];
        map.source_state_to_target_states[0].push(duplicated_target);
        assert!(dfa.concrete_swap_automorphism(&map, 0, 1).is_none());
    }

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
        assert!(plan.terminal_nwa_transport_modes().is_some());
    }

    #[test]
    fn byte_preserving_swap_rejects_distinct_literal_bytes() {
        let expressions = vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        assert!(dfa.interchange_map(0, 1).is_none());
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
        assert!(dfa.interchange_map(0, 2).is_some());
    }
}
