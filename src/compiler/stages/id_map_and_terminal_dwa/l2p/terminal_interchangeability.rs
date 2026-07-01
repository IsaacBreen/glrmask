//! Strict terminal interchangeability for the L2+ terminal-DWA reference path.
//!
//! For one vocabulary partition, interchangeability is computed on the tokenizer
//! DFA restricted to that partition's bytes. Only terminals active in this L2+
//! phase are observable. A pair is interchangeable when the original restricted
//! DFA and the DFA with those output labels swapped have a bijection between
//! their residual-state partitions.
//!
//! This is intentionally a validation-first construction: hide
//! nonrepresentatives before id-map/DWA construction; expand surviving
//! noninitial representative edges in place; make one transported whole-DWA
//! copy per initial replacement; then use the existing local DWA/id-map merger.
//! Directed subsumption is deliberately excluded.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_local_id_maps_and_terminal_dwas;
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

/// A per-source-state map on raw tokenizer states used to transport a copied
/// terminal-DWA artifact from a representative terminal onto a class member.
///
/// Sound interchangeability (see [`RestrictedDfa::interchange_map`]) only ever
/// produces the identity, so `source_state_to_target_states[s] == [s]`. The
/// general `Vec<Vec<u32>>` shape is kept only so the transport helpers can stay
/// unchanged; a non-identity map would be unsound (see the discussion there).
#[derive(Clone, Debug)]
struct InterchangeMap {
    source_state_to_target_states: Vec<Vec<u32>>,
}

/// View of the tokenizer used to decide terminal interchangeability for one
/// vocabulary partition. Interchangeability depends only on the per-state
/// matched-terminal sets, so this is deliberately a thin wrapper over the
/// tokenizer rather than a materialized byte-restricted automaton.
struct RestrictedDfa<'a> {
    tokenizer: &'a Tokenizer,
    real_state_count: usize,
}

impl<'a> RestrictedDfa<'a> {
    fn new(
        tokenizer: &'a Tokenizer,
        _active_terminals: &'a [bool],
        _relevant_bytes: &[bool; 256],
    ) -> Self {
        Self {
            tokenizer,
            real_state_count: tokenizer.num_states() as usize,
        }
    }

    /// The set of tokenizer states at which `terminal` is a completed match.
    fn matched_states(&self, terminal: TerminalID) -> Vec<bool> {
        let mut matched = vec![false; self.real_state_count];
        for (state, slot) in matched.iter_mut().enumerate() {
            *slot = self
                .tokenizer
                .matched_terminals_iter(state as u32)
                .any(|candidate| candidate == terminal);
        }
        matched
    }

    /// Two terminals are interchangeable for this partition iff they are matched
    /// at exactly the same set of tokenizer states, in which case the interchange
    /// map is the identity on every state and label-swapping is a verbatim copy.
    ///
    /// This is the *only* sound reset-preserving interchange. Terminal-DWA edge
    /// weights are keyed by the raw incoming tokenizer state, so a valid swap
    /// map must be a concrete automorphism of the deterministic lexer DFA that
    /// fixes the reset state. Such an automorphism is necessarily the identity
    /// on all reachable states: it commutes with the transition function and
    /// fixes the root, so `phi(delta*(reset, w)) = delta*(reset, w)`. Any coarser
    /// relation - notably a bisimulation-block correspondence that merges or
    /// swaps distinct raw states while still fixing the reset *block* - is not a
    /// concrete automorphism. Transporting weights through it unions the
    /// distinct token sets of those raw states and over- or under-accepts, which
    /// is exactly the defect this replaces. Identity on reachable states is
    /// equivalent to the two terminals having identical matched-state sets.
    fn interchange_map(&self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        if left != right && self.matched_states(left) != self.matched_states(right) {
            return None;
        }
        Some(InterchangeMap {
            source_state_to_target_states: (0..self.real_state_count)
                .map(|state| vec![state as u32])
                .collect(),
        })
    }
}

impl InterchangeMap {
    /// Reindex a transported artifact by target residual blocks.
    ///
    /// `base` is the ordinary completed id map, possibly with non-singleton
    /// state classes and with unreachable original states left unmapped. A
    /// transported copy has one TSID per reachable target residual block. Its
    /// weights are the push-forward of the ordinary source TSIDs through this
    /// block correspondence.
    fn transport_id_map(&self, base: &InternalIdMap) -> (InternalIdMap, Vec<Vec<u32>>) {
        let state_count = base.tokenizer_states.original_to_internal.len();
        assert_eq!(
            self.source_state_to_target_states.len(),
            state_count,
            "interchange map and local ID map have different raw-state domains",
        );

        let mut target_block_ids = BTreeMap::<Vec<u32>, u32>::new();
        let mut target_state_to_internal = vec![u32::MAX; state_count];
        let mut source_to_target = vec![BTreeSet::<u32>::new(); base.num_tsids() as usize];

        for (source, &source_tsid) in base
            .tokenizer_states
            .original_to_internal
            .iter()
            .enumerate()
        {
            // Ordinary L2P intentionally leaves lexer states with no relevant
            // token behaviour unmapped. They carry no weight, so their target
            // residual block stays outside this copied artifact too.
            if source_tsid == u32::MAX {
                continue;
            }
            let targets = &self.source_state_to_target_states[source];
            if targets.is_empty() {
                continue;
            }
            let next = target_block_ids.len() as u32;
            let target_tsid = *target_block_ids.entry(targets.clone()).or_insert(next);
            source_to_target[source_tsid as usize].insert(target_tsid);
            for &target in targets {
                let slot = target_state_to_internal
                    .get_mut(target as usize)
                    .expect("interchange target state outside local ID-map domain");
                assert!(
                    *slot == u32::MAX || *slot == target_tsid,
                    "interchange map assigned one target state to distinct residual blocks",
                );
                *slot = target_tsid;
            }
        }

        let source_to_target = source_to_target
            .into_iter()
            .map(|targets| targets.into_iter().collect())
            .collect();

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

    /// A representative pair is coarsely always-allowed only if *every*
    /// concrete member pair has that relation. This is the conservative relation
    /// used before representative expansion.
    pub(crate) fn coalesced_always_allowed_follows(
        &self,
        concrete: &[Vec<TerminalID>],
    ) -> Vec<Vec<TerminalID>> {
        let terminal_count = self.original_active.len();
        let mut result = vec![Vec::new(); terminal_count];
        for representative in 0..terminal_count {
            if !self.active_representatives[representative] {
                continue;
            }
            let left_members = &self.members_by_representative[representative];
            for successor in 0..terminal_count {
                if !self.active_representatives[successor] {
                    continue;
                }
                let right_members = &self.members_by_representative[successor];
                if left_members.iter().all(|&left| {
                    concrete.get(left as usize).is_some_and(|follows| {
                        right_members.iter().all(|right| follows.contains(right))
                    })
                }) {
                    result[representative].push(successor as TerminalID);
                }
            }
        }
        result
    }

    /// Slow, validation-first undo of representative substitution.
    ///
    /// The representative DWA is already complete; only the representative of
    /// each interchangeability class appears in it. Expansion has two purely
    /// structural stages and applies no follow constraints (the caller runs the
    /// concrete disallowed-follow pass once, after this and determinize/minimize):
    ///
    /// 1. **Noninitial edges.** Edges that do not leave the start state describe
    ///    continuations after a terminal boundary, where the lexer has restarted
    ///    from its fixed initial state. They are independent of which member was
    ///    chosen, so every noninitial representative edge is cloned in place for
    ///    each class member (same destination and weight, different label).
    ///
    /// 2. **Initial edges.** Edges from the start state depend on the incoming
    ///    tokenizer state carried across the token boundary. For each member we
    ///    clone the whole (already noninitial-expanded) artifact, relabel the
    ///    representative's initial edge to the member, and transport the cloned
    ///    id map and every transition/final weight through that member's
    ///    interchange map.
    ///
    /// The existing local-artifact merger then unions the representative artifact
    /// and every transported copy in a common id-map space.
    pub(crate) fn expand_terminal_dwa_slow(
        &self,
        artifact: LocalIdMapTerminalDwa,
        num_tokenizer_states: u32,
        max_token_id: u32,
    ) -> LocalIdMapTerminalDwa {
        if self.is_identity() {
            return artifact;
        }

        let mut representative_artifact = artifact;
        self.expand_noninitial_edges(&mut representative_artifact.dwa);
        dump_dwa_if_requested("after_noninitial", &representative_artifact.dwa);

        let start = representative_artifact.dwa.start_state() as usize;
        let initial_transitions = representative_artifact.dwa.states()[start]
            .transitions
            .clone();
        let mut copies = vec![representative_artifact.clone()];

        for (representative, members) in self.members_by_representative.iter().enumerate() {
            if members.len() < 2 {
                continue;
            }
            let representative = representative as TerminalID;
            let Some((destination, weight)) = initial_transitions.get(&(representative as i32)) else {
                continue;
            };
            for &member in members {
                if member == representative {
                    continue;
                }
                let map = self
                    .maps_by_representative_member
                    .get(&(representative, member))
                    .expect("interchangeability member missing transport map");
                let mut copy = representative_artifact.clone();
                let (id_map, source_to_target_tsids) = map.transport_id_map(&copy.id_map);
                copy.id_map = id_map;
                let initial = &mut copy.dwa.states_mut()[start].transitions;
                let removed = initial.remove(&(representative as i32));
                assert_eq!(
                    removed.as_ref().map(|(target, _)| *target),
                    Some(*destination),
                    "representative initial transition changed before interchange expansion",
                );
                initial.insert(member as i32, (*destination, weight.clone()));
                transport_all_dwa_weights(&mut copy.dwa, &source_to_target_tsids);
                copies.push(copy);
            }
        }

        let merged = merge_local_id_maps_and_terminal_dwas(
            copies,
            num_tokenizer_states as usize,
            max_token_id,
        );
        dump_dwa_if_requested("after_initial_merge", &merged.dwa);
        merged
    }

    /// Stage 1 of expansion: clone every noninitial representative edge for each
    /// member of its interchangeability class, keeping the same destination and
    /// weight. Edges from the start state are left untouched (handled by the
    /// transported per-member copies in stage 2).
    fn expand_noninitial_edges(&self, dwa: &mut DWA) {
        let start = dwa.start_state() as usize;
        for (state_id, state) in dwa.states_mut().iter_mut().enumerate() {
            if state_id == start {
                continue;
            }
            let original = state.transitions.clone();
            for (&label, (destination, weight)) in &original {
                let Ok(terminal) = TerminalID::try_from(label) else {
                    continue;
                };
                let members = &self.members_by_representative[terminal as usize];
                if members.len() < 2 || members[0] != terminal {
                    continue;
                }
                for &member in members {
                    state
                        .transitions
                        .insert(member as i32, (*destination, weight.clone()));
                }
            }
        }
    }
}

pub(crate) fn dump_dwa_if_requested(stage: &str, dwa: &DWA) {
    if std::env::var_os("GLRMASK_DUMP_TI_EXPANSION").is_none() {
        return;
    }
    eprintln!("[terminal-interchangeability][{stage}] start={}", dwa.start_state());
    for (state_id, state) in dwa.states().iter().enumerate() {
        let transitions = state
            .transitions
            .iter()
            .map(|(&label, (target, weight))| format!("{label}->{target}:{weight:?}"))
            .collect::<Vec<_>>();
        eprintln!(
            "[terminal-interchangeability][{stage}] state={state_id} final={:?} transitions={transitions:?}",
            state.final_weight,
        );
    }
}

fn transport_all_dwa_weights(dwa: &mut DWA, source_to_target_tsids: &[Vec<u32>]) {
    let mut cache = HashMap::<usize, Weight>::new();
    for state in dwa.states_mut() {
        if let Some(final_weight) = &mut state.final_weight {
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
    weight: &Weight,
    source_to_target_tsids: &[Vec<u32>],
    cache: &mut HashMap<usize, Weight>,
) -> Weight {
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
    }
    if let Some(existing) = cache.get(&weight.ptr_key()) {
        return existing.clone();
    }

    let mut entries = Vec::new();
    for (start, end, tokens) in weight
        .compact_entries()
        .expect("non-full weights have compact entries")
    {
        for source_tsid in start..=end {
            let targets = source_to_target_tsids
                .get(source_tsid as usize)
                .expect("weight refers to source TSID outside transport domain");
            for &target_tsid in targets {
                entries.push((target_tsid, tokens.clone()));
            }
        }
    }
    entries.sort_by_key(|(target_tsid, _)| *target_tsid);
    let transported = Weight::union_sorted_point_entries(entries);
    cache.insert(weight.ptr_key(), transported.clone());
    transported
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
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use range_set_blaze::RangeSetBlaze;

    use super::*;
    use crate::ds::weight::shared_rangeset;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    #[test]
    fn rotated_residuals_have_no_interchange_map_because_reset_moves() {
        // A=/a(aaaa)*/ is matched at the residue-1 states, B=/aaa(aaaa)*/ at the
        // residue-3 states. Their matched-state sets differ, so swapping their
        // labels is not the identity on reachable states: A and B are NOT
        // interchangeable and the pair has no interchange map.
        let expressions = vec![
            Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
            Expr::Seq(vec![Expr::U8Seq(b"aaa".to_vec()), Expr::Repeat { expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())), min: 0, max: None }]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        assert!(dfa.interchange_map(0, 1).is_none());
        assert!(dfa.interchange_map(1, 0).is_none());
    }

    #[test]
    fn relation_weight_transport_unions_many_to_one_contributions() {
        let weight = Weight::from_per_tsid_shared([
            (
                0,
                shared_rangeset(RangeSetBlaze::from_iter([10u32..=10u32])),
            ),
            (
                1,
                shared_rangeset(RangeSetBlaze::from_iter([20u32..=20u32])),
            ),
        ]);
        let mut cache = HashMap::new();
        let transported = transport_weight(&weight, &[vec![7], vec![7]], &mut cache);
        let tokens = transported.tokens_for_tsid(7);
        assert!(tokens.contains(10));
        assert!(tokens.contains(20));
        assert_eq!(tokens.len(), 2);
    }

    fn two_by_two_partition_plan() -> TerminalInterchangeability {
        TerminalInterchangeability {
            original_active: vec![true; 4],
            active_representatives: vec![true, false, true, false],
            representative_for: vec![0, 0, 2, 2],
            members_by_representative: vec![vec![0, 1], vec![1], vec![2, 3], vec![3]],
            maps_by_representative_member: BTreeMap::new(),
        }
    }

    #[test]
    fn coarse_always_allowed_requires_every_concrete_member_pair() {
        let plan = two_by_two_partition_plan();

        let mut always_allowed = vec![Vec::new(); 4];
        for left in [0usize, 1] {
            always_allowed[left].extend([2, 3]);
        }
        assert!(plan.coalesced_always_allowed_follows(&always_allowed)[0].contains(&2));
        always_allowed[1].retain(|&terminal| terminal != 3);
        assert!(!plan.coalesced_always_allowed_follows(&always_allowed)[0].contains(&2));
    }

    #[test]
    fn rotated_residuals_do_not_form_an_interchangeable_partition() {
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
        // The reset-moving swap is not a valid interchange map, so the two
        // terminals stay in their own singleton classes (no substitution).
        assert_eq!(plan.active_terminal_count_after(), 2);
        assert!(plan.is_identity());
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
