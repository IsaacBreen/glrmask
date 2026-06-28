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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_local_id_maps_and_terminal_dwas;
use crate::compiler::stages::id_map_and_terminal_dwa::types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile};
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

impl InterchangeMap {
    /// Reindex the candidate artifact by the *target* residual blocks and build
    /// the one-to-many map needed to push every source-TSID weight into that
    /// target coordinate system.
    ///
    /// This is deliberately block based.  A terminal interchange map is a
    /// bijection of residual partitions, not a concrete state permutation; a
    /// target block may contain several raw lexer states.  Assigning one TSID
    /// to that whole target block preserves exactly the information the map
    /// proves, without inventing a concrete-state automorphism.
    fn transport_id_map(&self, base: &InternalIdMap) -> (InternalIdMap, Vec<Vec<u32>>) {
        let state_count = base.tokenizer_states.original_to_internal.len();
        assert_eq!(
            self.source_state_to_target_states.len(),
            state_count,
            "interchange map and local id map have different raw-state domains"
        );
        assert!(
            base.tokenizer_states
                .original_to_internal
                .iter()
                .all(|&tsid| tsid != u32::MAX),
            "reference interchangeability transport requires every raw lexer state to have a TSID"
        );

        let mut target_block_ids = BTreeMap::<Vec<u32>, u32>::new();
        let mut target_state_to_internal = vec![u32::MAX; state_count];
        let mut target_internal_for_source = vec![u32::MAX; state_count];

        for (source, targets) in self.source_state_to_target_states.iter().enumerate() {
            assert!(
                !targets.is_empty(),
                "interchange map has an empty target residual block for source state {source}"
            );
            let next = target_block_ids.len() as u32;
            let target_internal = *target_block_ids.entry(targets.clone()).or_insert(next);
            target_internal_for_source[source] = target_internal;
            for &target in targets {
                let slot = target_state_to_internal
                    .get_mut(target as usize)
                    .unwrap_or_else(|| panic!("interchange target state {target} outside local id-map domain"));
                assert!(
                    *slot == u32::MAX || *slot == target_internal,
                    "interchange map assigned target state {target} to two distinct residual blocks"
                );
                *slot = target_internal;
            }
        }
        assert!(
            target_state_to_internal.iter().all(|&tsid| tsid != u32::MAX),
            "interchange map did not cover every raw lexer state"
        );

        let mut source_to_target = vec![BTreeSet::<u32>::new(); base.num_tsids() as usize];
        for (source, &source_tsid) in base
            .tokenizer_states
            .original_to_internal
            .iter()
            .enumerate()
        {
            source_to_target[source_tsid as usize]
                .insert(target_internal_for_source[source]);
        }
        let source_to_target = source_to_target
            .into_iter()
            .map(|targets| targets.into_iter().collect())
            .collect::<Vec<Vec<u32>>>();

        (
            InternalIdMap {
                tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                    target_state_to_internal,
                    target_block_ids.len() as u32,
                ),
                vocab_tokens: base.vocab_tokens.clone(),
            },
            source_to_target,
        )
    }
}

fn transport_all_weights(dwa: &mut DWA, source_to_target_tsids: &[Vec<u32>]) {
    let mut cache = HashMap::<usize, crate::ds::weight::Weight>::new();
    for state in dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = transport_weight(final_weight, source_to_target_tsids, &mut cache);
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }
        for (_, weight) in state.transitions.values_mut() {
            *weight = transport_weight(weight, source_to_target_tsids, &mut cache);
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
}

fn transport_weight(
    weight: &crate::ds::weight::Weight,
    source_to_target_tsids: &[Vec<u32>],
    cache: &mut HashMap<usize, crate::ds::weight::Weight>,
) -> crate::ds::weight::Weight {
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
    }
    let key = weight.ptr_key();
    if let Some(mapped) = cache.get(&key) {
        return mapped.clone();
    }

    let mut entries = Vec::new();
    for (start, end, tokens) in weight
        .compact_entries()
        .expect("non-full weights have compact entries")
    {
        for source_tsid in start..=end {
            let targets = source_to_target_tsids
                .get(source_tsid as usize)
                .unwrap_or_else(|| panic!("weight refers to out-of-range source TSID {source_tsid}"));
            for &target_tsid in targets {
                entries.push((target_tsid, tokens.clone()));
            }
        }
    }
    entries.sort_by_key(|(target_tsid, _)| *target_tsid);
    let mapped = crate::ds::weight::Weight::union_sorted_point_entries(entries);
    cache.insert(key, mapped.clone());
    mapped
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

        let restricted = RestrictedDfa::new(tokenizer, active_terminals, relevant_bytes);
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
    pub(crate) fn active_terminal_count_before(&self) -> usize {
        self.original_active.iter().filter(|&&active| active).count()
    }
    pub(crate) fn active_terminal_count_after(&self) -> usize {
        self.active_representatives.iter().filter(|&&active| active).count()
    }

    /// Restore the concrete terminal alphabet after building the representative
    /// artifact.  This intentionally follows the slow construction literally:
    ///
    /// 1. clone every noninitial representative edge to every class member;
    /// 2. make a whole-artifact branch for each replacement of each
    ///    representative initial edge;
    /// 3. for a nonrepresentative initial label, transport both the raw-state
    ///    id map and every DWA weight through the residual partition map; and
    /// 4. merge the branch-local `(DWA, id_map)` pairs with the ordinary local
    ///    merger used elsewhere for L1/L2P and vocab partitions.
    ///
    /// Each branch deliberately retains the whole DWA. The interchange map
    /// transports every weight and the id map together; the ordinary local
    /// merger reconciles the resulting coordinate systems exactly.
    pub(crate) fn restore_reference_artifact(
        &self,
        base: LocalIdMapTerminalDwa,
        num_tokenizer_states: usize,
        max_token_id: u32,
    ) -> LocalIdMapTerminalDwa {
        if self.is_identity() {
            return base;
        }

        let profile = base.profile.clone();
        let mut expanded = base.dwa.clone();
        self.clone_noninitial_edges(&mut expanded);

        let start = expanded.start_state() as usize;
        let initial_representatives = expanded.states()[start]
            .transitions
            .iter()
            .filter_map(|(&label, _)| TerminalID::try_from(label).ok())
            .filter(|&terminal| {
                self.representative_for
                    .get(terminal as usize)
                    .copied()
                    .is_some_and(|representative| representative == terminal)
            })
            .filter(|&representative| {
                self.members_by_representative[representative as usize].len() > 1
            })
            .collect::<Vec<_>>();
        let mut branches = Vec::<LocalIdMapTerminalDwa>::new();
        for representative in initial_representatives {
            for &replacement in &self.members_by_representative[representative as usize] {
                let mut branch = expanded.clone();
                let (id_map, tsid_targets) = if replacement == representative {
                    (base.id_map.clone(), None)
                } else {
                    let map = self
                        .maps_by_representative_member
                        .get(&(representative, replacement))
                        .unwrap_or_else(|| panic!(
                            "missing interchange map for representative {representative} and replacement {replacement}"
                        ));
                    let (id_map, tsid_targets) = map.transport_id_map(&base.id_map);
                    (id_map, Some(tsid_targets))
                };
                if let Some(tsid_targets) = tsid_targets.as_deref() {
                    transport_all_weights(&mut branch, tsid_targets);
                }

                let transitions = &mut branch.states_mut()[start].transitions;
                let (target, weight) = transitions
                    .remove(&(representative as i32))
                    .unwrap_or_else(|| {
                        panic!("missing representative initial edge during interchangeability restoration")
                    });
                transitions.insert(replacement as i32, (target, weight));
                branches.push(LocalIdMapTerminalDwa {
                    id_map,
                    dwa: branch,
                    profile: TerminalDwaPhaseProfile::default(),
                });
            }
        }

        if branches.is_empty() {
            return LocalIdMapTerminalDwa {
                id_map: base.id_map,
                dwa: expanded,
                profile,
            };
        }
        let mut merged =
            merge_local_id_maps_and_terminal_dwas(branches, num_tokenizer_states, max_token_id);
        merged.profile = profile;
        merged
    }

    fn clone_noninitial_edges(&self, dwa: &mut DWA) {
        let start = dwa.start_state();
        for (state_id, state) in dwa.states_mut().iter_mut().enumerate() {
            if state_id as u32 == start {
                continue;
            }
            let source_edges = state
                .transitions
                .iter()
                .map(|(&label, &(target, ref weight))| (label, target, weight.clone()))
                .collect::<Vec<_>>();
            for (label, target, weight) in source_edges {
                let Ok(terminal) = TerminalID::try_from(label) else {
                    continue;
                };
                let representative = self
                    .representative_for
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(terminal);
                if representative != terminal {
                    continue;
                }
                for &member in &self.members_by_representative[representative as usize] {
                    let member_label = member as i32;
                    if let Some((existing_target, existing_weight)) =
                        state.transitions.get(&member_label)
                    {
                        assert_eq!(
                            (*existing_target, existing_weight),
                            (target, &weight),
                            "incompatible noninitial interchangeability edge collision for terminal {member}"
                        );
                    } else {
                        state.transitions.insert(member_label, (target, weight.clone()));
                    }
                }
            }
        }
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
        assert!(dfa.interchange_map(0, 1).is_some());
        assert!(dfa.interchange_map(1, 0).is_some());
    }

    #[test]
    fn interchange_map_covers_a_target_residual_block_for_every_source_state() {
        let expressions = vec![
            Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
            Expr::Seq(vec![Expr::U8Seq(b"aaa".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        let map = dfa.interchange_map(0, 1).expect("rotated map must exist");
        assert!(map.source_state_to_target_states.iter().all(|targets| !targets.is_empty()));
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
        assert!(!plan.is_identity());
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
