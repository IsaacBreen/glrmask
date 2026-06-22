//! Partition-restricted L2P terminal equivalence.
//!
//! Two terminals may differ in the global lexer yet be indistinguishable on a
//! vocabulary partition. We minimize their residual scanners over that
//! partition's byte alphabet. A member terminal state is then mapped to a
//! representative terminal state in the same residual block.
//!
//! Every real terminal scanner remains active during raw NWA construction.
//! Only its emitted label is replaced by the terminal-class label. This retains
//! member-specific root topology through determinization; a later DWA pass
//! restores concrete labels and applies the exact grammar-follow relation.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHasher};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

const DEAD_BLOCK: u32 = 0;
const NO_PAIR: u32 = u32::MAX;

#[derive(Clone, Debug, Default)]
pub(crate) struct TerminalEquivalence {
    representative_for_terminal: Vec<TerminalID>,
    members_by_representative: Vec<Vec<TerminalID>>,
    active_representatives: Vec<bool>,
    /// Sorted `(original lexer state, restricted residual block)` pairs.
    /// A missing state belongs to the shared dead block.
    state_blocks_by_terminal: Vec<Vec<(u32, u32)>>,
    member_live_tsids: Vec<Option<BitSet>>,
    original_active_terminals: Vec<bool>,
    active_terminal_count: usize,
    class_count: usize,
    quotient_hits: usize,
    residual_pair_count: usize,
    residual_block_count: usize,
    active_byte_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TerminalEquivalenceProfile {
    pub(crate) active_terminals: usize,
    pub(crate) classes: usize,
    pub(crate) quotient_hits: usize,
    pub(crate) residual_pairs: usize,
    pub(crate) residual_blocks: usize,
    pub(crate) active_bytes: usize,
    pub(crate) expanded_transition_copies: usize,
    pub(crate) root_source_maps: usize,
}

impl TerminalEquivalence {
    pub(crate) fn identity(active_terminals: &[bool]) -> Self {
        let num_terminals = active_terminals.len();
        let representative_for_terminal = (0..num_terminals as u32).collect::<Vec<_>>();
        let members_by_representative = (0..num_terminals as u32)
            .map(|terminal| vec![terminal])
            .collect();
        let active_terminal_count = active_terminals.iter().filter(|&&active| active).count();
        Self {
            representative_for_terminal,
            members_by_representative,
            active_representatives: active_terminals.to_vec(),
            state_blocks_by_terminal: vec![Vec::new(); num_terminals],
            member_live_tsids: vec![None; num_terminals],
            original_active_terminals: active_terminals.to_vec(),
            active_terminal_count,
            class_count: active_terminal_count,
            quotient_hits: 0,
            residual_pair_count: 0,
            residual_block_count: 0,
            active_byte_count: 0,
        }
    }

    pub(crate) fn build(
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        ignore_terminal: Option<TerminalID>,
        relevant_bytes: &[bool; 256],
    ) -> Self {
        let num_terminals = tokenizer.num_terminals() as usize;
        assert_eq!(
            active_terminals.len(),
            num_terminals,
            "L2P terminal-equivalence mask must cover every tokenizer terminal"
        );
        let active_ids: Vec<TerminalID> = (0..num_terminals)
            .filter(|&terminal| active_terminals[terminal])
            .map(|terminal| terminal as TerminalID)
            .collect();
        if active_ids.len() < 2 {
            return Self::identity(active_terminals);
        }

        let active_bytes: Vec<u8> = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect();
        let quotient_active = |terminal: TerminalID| {
            active_terminals.get(terminal as usize).copied().unwrap_or(false)
                && Some(terminal) != ignore_terminal
        };

        let mut final_states = vec![Vec::<u32>::new(); num_terminals];
        let mut future_states = vec![Vec::<u32>::new(); num_terminals];
        for state in 0..tokenizer.num_states() {
            for terminal in tokenizer.matched_terminals_iter(state) {
                if quotient_active(terminal) {
                    final_states[terminal as usize].push(state);
                }
            }
            for terminal in tokenizer.possible_future_terminals_iter(state) {
                if quotient_active(terminal) {
                    future_states[terminal as usize].push(state);
                }
            }
        }

        let mut pair_ids_by_terminal = (0..num_terminals)
            .map(|_| FxHashMap::<u32, u32>::default())
            .collect::<Vec<_>>();
        let mut pair_terminals = Vec::<TerminalID>::new();
        let mut pair_states = Vec::<u32>::new();
        let mut outputs = Vec::<u8>::new();

        for &terminal in &active_ids {
            if Some(terminal) == ignore_terminal {
                continue;
            }
            let finals = &final_states[terminal as usize];
            let futures = &future_states[terminal as usize];
            let mut final_index = 0usize;
            let mut future_index = 0usize;
            while final_index < finals.len() || future_index < futures.len() {
                let next_final = finals.get(final_index).copied();
                let next_future = futures.get(future_index).copied();
                let state = match (next_final, next_future) {
                    (Some(left), Some(right)) => left.min(right),
                    (Some(left), None) => left,
                    (None, Some(right)) => right,
                    (None, None) => unreachable!(),
                };
                let is_final = next_final == Some(state);
                let is_future = next_future == Some(state);
                if is_final {
                    final_index += 1;
                }
                if is_future {
                    future_index += 1;
                }
                let pair = pair_terminals.len() as u32;
                pair_ids_by_terminal[terminal as usize].insert(state, pair);
                pair_terminals.push(terminal);
                pair_states.push(state);
                outputs.push((u8::from(is_final) << 1) | u8::from(is_future));
            }
        }

        if pair_terminals.is_empty() {
            return Self::identity(active_terminals);
        }

        let width = active_bytes.len();
        let mut transitions = vec![NO_PAIR; pair_terminals.len() * width];
        for pair_index in 0..pair_terminals.len() {
            let terminal = pair_terminals[pair_index] as usize;
            let state = pair_states[pair_index];
            let row = &mut transitions[pair_index * width..(pair_index + 1) * width];
            for (slot, &byte) in active_bytes.iter().enumerate() {
                let next = tokenizer.get_transition(state, byte);
                if next != u32::MAX {
                    row[slot] = pair_ids_by_terminal[terminal]
                        .get(&next)
                        .copied()
                        .unwrap_or(NO_PAIR);
                }
            }
        }

        let mut blocks = initial_blocks(&outputs);
        loop {
            let next = refine_blocks(&outputs, &transitions, width, &blocks);
            if next == blocks {
                break;
            }
            blocks = next;
        }
        let residual_block_count = blocks.iter().copied().max().unwrap_or(DEAD_BLOCK) as usize;

        let mut state_blocks_by_terminal = vec![Vec::<(u32, u32)>::new(); num_terminals];
        for pair_index in 0..pair_terminals.len() {
            state_blocks_by_terminal[pair_terminals[pair_index] as usize]
                .push((pair_states[pair_index], blocks[pair_index]));
        }

        let mut representative_for_terminal =
            (0..num_terminals as u32).collect::<Vec<TerminalID>>();
        let mut members_by_representative = (0..num_terminals as u32)
            .map(|terminal| vec![terminal])
            .collect::<Vec<_>>();
        let mut active_representatives = vec![false; num_terminals];
        let mut groups = FxHashMap::<Vec<u32>, Vec<TerminalID>>::default();

        for &terminal in &active_ids {
            if Some(terminal) == ignore_terminal {
                active_representatives[terminal as usize] = true;
                continue;
            }
            let mut inventory = state_blocks_by_terminal[terminal as usize]
                .iter()
                .map(|&(_, block)| block)
                .collect::<Vec<_>>();
            inventory.sort_unstable();
            inventory.dedup();
            groups.entry(inventory).or_default().push(terminal);
        }

        let mut class_count = 0usize;
        let mut quotient_hits = 0usize;
        for mut members in groups.into_values() {
            members.sort_unstable();
            let representative = *members
                .iter()
                .min_by_key(|&&terminal| {
                    (state_blocks_by_terminal[terminal as usize].len(), terminal)
                })
                .expect("terminal equivalence class must be non-empty");
            let representative_position = members
                .iter()
                .position(|&member| member == representative)
                .expect("chosen terminal representative must be a class member");
            members.swap(0, representative_position);
            quotient_hits += members.len().saturating_sub(1);
            class_count += 1;
            for &terminal in &members {
                representative_for_terminal[terminal as usize] = representative;
            }
            members_by_representative[representative as usize] = members;
            active_representatives[representative as usize] = true;
        }
        if let Some(ignore_terminal) = ignore_terminal {
            if active_terminals.get(ignore_terminal as usize).copied().unwrap_or(false) {
                class_count += 1;
            }
        }

        Self {
            representative_for_terminal,
            members_by_representative,
            active_representatives,
            state_blocks_by_terminal,
            member_live_tsids: vec![None; num_terminals],
            original_active_terminals: active_terminals.to_vec(),
            active_terminal_count: active_ids.len(),
            class_count,
            quotient_hits,
            residual_pair_count: pair_terminals.len(),
            residual_block_count,
            active_byte_count: active_bytes.len(),
        }
    }

    /// Refine classes against the runtime TSID map. First-terminal expansion
    /// later needs the exact set of TSIDs where every member is live. Terminal
    /// sequences after a completed match restart at the lexer initial state, so
    /// members whose initial residual differs from their representative cannot
    /// share a continuation class label.
    pub(crate) fn refine_for_tsid_map(
        &mut self,
        tokenizer_states: &ManyToOneIdMap,
        initial_state: u32,
    ) {
        let num_terminals = self.representative_for_terminal.len();
        let num_tsids = tokenizer_states.num_internal_ids() as usize;
        self.member_live_tsids = vec![None; num_terminals];
        for terminal in 0..num_terminals {
            if !self.original_active_terminals[terminal] {
                continue;
            }
            let mut live_tsids = BitSet::new(num_tsids);
            for (tsid, originals) in tokenizer_states.internal_to_originals.iter().enumerate() {
                if originals
                    .iter()
                    .any(|&state| self.block_for(terminal as TerminalID, state) != DEAD_BLOCK)
                {
                    live_tsids.set(tsid);
                }
            }
            self.member_live_tsids[terminal] = Some(live_tsids);
        }

        for representative_index in 0..num_terminals {
            let representative = representative_index as TerminalID;
            let members = self.members_by_representative[representative_index].clone();
            if members.len() <= 1 || members[0] != representative {
                continue;
            }
            let representative_initial_block = self.block_for(representative, initial_state);
            let mut retained = vec![representative];
            for member in members.into_iter().skip(1) {
                if self.block_for(member, initial_state) == representative_initial_block {
                    retained.push(member);
                } else {
                    self.make_singleton(member);
                }
            }
            self.members_by_representative[representative_index] = retained;
        }
        self.recompute_class_metadata();
    }

    /// Maps each real terminal to the class label emitted during raw NWA
    /// construction. The representative itself is a real terminal ID, so no
    /// downstream label format changes are needed.
    pub(crate) fn terminal_label_map(&self) -> &[TerminalID] {
        &self.representative_for_terminal
    }

    /// Conservative class-level disallowed follows. A class edge C→D is
    /// removed before determinization only when every member pair c∈C,d∈D is
    /// grammatically disallowed. The exact member relation is applied after
    /// class expansion.
    pub(crate) fn class_disallowed_follows(
        &self,
        disallowed_follows: &BTreeMap<u32, BitSet>,
        num_terminals: usize,
    ) -> BTreeMap<u32, BitSet> {
        let mut result = BTreeMap::new();
        for representative in 0..self.members_by_representative.len() {
            if !self.active_representatives[representative] {
                continue;
            }
            let members = &self.members_by_representative[representative];
            let mut disallowed = BitSet::new(num_terminals);
            for next_representative in 0..self.members_by_representative.len() {
                if !self.active_representatives[next_representative] {
                    continue;
                }
                let next_members = &self.members_by_representative[next_representative];
                let universally_disallowed = members.iter().all(|&previous| {
                    next_members.iter().all(|&next| {
                        disallowed_follows
                            .get(&previous)
                            .is_some_and(|bits| bits.contains(next as usize))
                    })
                });
                if universally_disallowed {
                    disallowed.set(next_representative);
                }
            }
            if !disallowed.is_zero() {
                result.insert(representative as u32, disallowed);
            }
        }
        result
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.quotient_hits == 0
    }

    pub(crate) fn profile(&self) -> TerminalEquivalenceProfile {
        TerminalEquivalenceProfile {
            active_terminals: self.active_terminal_count,
            classes: self.class_count,
            quotient_hits: self.quotient_hits,
            residual_pairs: self.residual_pair_count,
            residual_blocks: self.residual_block_count,
            active_bytes: self.active_byte_count,
            expanded_transition_copies: 0,
            root_source_maps: 0,
        }
    }

    /// Expand class labels on a deterministic DWA. `class` labels are
    /// continuation terminals; `first_label_offset + class` labels are first
    /// terminals and are restricted to each member's live source-TSID domain.
    pub(crate) fn expand_class_dwa(
        &self,
        dwa: &mut DWA,
        class_label_offset: u32,
        first_label_offset: u32,
    ) -> TerminalEquivalenceProfile {
        let mut profile = TerminalEquivalenceProfile {
            active_terminals: self.active_terminal_count,
            classes: self.class_count,
            quotient_hits: self.quotient_hits,
            residual_pairs: self.residual_pair_count,
            residual_blocks: self.residual_block_count,
            active_bytes: self.active_byte_count,
            ..TerminalEquivalenceProfile::default()
        };
        if self.is_identity() {
            return profile;
        }

        let all_tokens: RangeSetBlaze<u32> = std::iter::once(0..=u32::MAX).collect();
        let member_domain_weights = self
            .member_live_tsids
            .iter()
            .map(|live_tsids| {
                live_tsids.as_ref().map(|live_tsids| {
                    Weight::from_per_tsid_token_sets(
                        live_tsids
                            .iter_ones()
                            .map(|tsid| (tsid as u32, all_tokens.clone())),
                    )
                })
            })
            .collect::<Vec<_>>();

        for state in dwa.states_mut() {
            let original = state.transitions.clone();
            for (representative_index, members) in self.members_by_representative.iter().enumerate() {
                if !self.active_representatives[representative_index] {
                    continue;
                }
                let representative = representative_index as u32;
                let continuation = representative
                    .checked_add(class_label_offset)
                    .and_then(|label| original.get(&(label as i32)));
                let first = representative
                    .checked_add(first_label_offset)
                    .and_then(|label| original.get(&(label as i32)));
                for &member in members {
                    if let Some(&(target, ref weight)) = continuation {
                        insert_dwa_transition(
                            &mut state.transitions,
                            member as i32,
                            target,
                            weight.clone(),
                        );
                        profile.expanded_transition_copies += 1;
                    }
                    if let Some(&(target, ref weight)) = first {
                        let domain = member_domain_weights[member as usize]
                            .as_ref()
                            .expect("active terminal class member lacks live TSID domain");
                        let restricted = weight.intersection(domain);
                        if !restricted.is_empty() {
                            insert_dwa_transition(
                                &mut state.transitions,
                                member as i32,
                                target,
                                restricted,
                            );
                            profile.expanded_transition_copies += 1;
                        }
                    }
                }
            }
            state.transitions.retain(|&label, _| {
                if label < 0 {
                    return true;
                }
                let raw = label as u32;
                let encoded_class = if raw >= first_label_offset {
                    Some(raw - first_label_offset)
                } else if raw >= class_label_offset {
                    Some(raw - class_label_offset)
                } else {
                    None
                };
                !encoded_class.is_some_and(|class| {
                    self.active_representatives
                        .get(class as usize)
                        .copied()
                        .unwrap_or(false)
                })
            });
        }
        profile
    }

    fn block_for(&self, terminal: TerminalID, state: u32) -> u32 {
        self.state_blocks_by_terminal[terminal as usize]
            .binary_search_by_key(&state, |(candidate, _)| *candidate)
            .ok()
            .map(|index| self.state_blocks_by_terminal[terminal as usize][index].1)
            .unwrap_or(DEAD_BLOCK)
    }

    fn make_singleton(&mut self, terminal: TerminalID) {
        let index = terminal as usize;
        self.representative_for_terminal[index] = terminal;
        self.members_by_representative[index] = vec![terminal];
        self.active_representatives[index] = true;
    }

    fn recompute_class_metadata(&mut self) {
        self.active_representatives.fill(false);
        let mut classes = 0usize;
        let mut hits = 0usize;
        for (representative, members) in self.members_by_representative.iter().enumerate() {
            if !self.original_active_terminals[representative]
                || members.is_empty()
                || self.representative_for_terminal[representative] != representative as TerminalID
            {
                continue;
            }
            debug_assert_eq!(members[0], representative as TerminalID);
            self.active_representatives[representative] = true;
            classes += 1;
            hits += members.len().saturating_sub(1);
        }
        self.class_count = classes;
        self.quotient_hits = hits;
    }
}

fn initial_blocks(outputs: &[u8]) -> Vec<u32> {
    let mut order = (0..outputs.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&pair| outputs[pair]);
    let mut blocks = vec![DEAD_BLOCK; outputs.len()];
    let mut next_block = DEAD_BLOCK;
    let mut previous = None;
    for pair in order {
        if previous.map_or(true, |previous| outputs[pair] != outputs[previous]) {
            next_block += 1;
        }
        blocks[pair] = next_block;
        previous = Some(pair);
    }
    blocks
}

fn refine_blocks(
    outputs: &[u8],
    transitions: &[u32],
    width: usize,
    previous_blocks: &[u32],
) -> Vec<u32> {
    // The old implementation materialized every `(output, successor-blocks…)`
    // row then lexicographically sorted wide rows. On large partitions that is
    // dominated by comparisons, not by the actual DFA refinement. Bucket by a
    // stable hash and verify collisions against the source rows instead.
    let mut buckets = FxHashMap::<u64, Vec<(usize, u32)>>::default();
    let mut blocks = vec![DEAD_BLOCK; outputs.len()];
    let mut next_block = DEAD_BLOCK;

    for pair in 0..outputs.len() {
        let hash = residual_signature_hash(pair, outputs, transitions, width, previous_blocks);
        let candidates = buckets.entry(hash).or_default();
        if let Some(&(_, block)) = candidates.iter().find(|&&(candidate, _)| {
            same_residual_signature(
                pair,
                candidate,
                outputs,
                transitions,
                width,
                previous_blocks,
            )
        }) {
            blocks[pair] = block;
        } else {
            next_block += 1;
            candidates.push((pair, next_block));
            blocks[pair] = next_block;
        }
    }
    blocks
}

#[inline]
fn residual_successor_block(
    pair: usize,
    slot: usize,
    transitions: &[u32],
    width: usize,
    previous_blocks: &[u32],
) -> u32 {
    let target = transitions[pair * width + slot];
    if target == NO_PAIR {
        DEAD_BLOCK
    } else {
        previous_blocks[target as usize]
    }
}

fn residual_signature_hash(
    pair: usize,
    outputs: &[u8],
    transitions: &[u32],
    width: usize,
    previous_blocks: &[u32],
) -> u64 {
    let mut hasher = FxHasher::default();
    outputs[pair].hash(&mut hasher);
    for slot in 0..width {
        residual_successor_block(pair, slot, transitions, width, previous_blocks).hash(&mut hasher);
    }
    hasher.finish()
}

fn same_residual_signature(
    left: usize,
    right: usize,
    outputs: &[u8],
    transitions: &[u32],
    width: usize,
    previous_blocks: &[u32],
) -> bool {
    outputs[left] == outputs[right]
        && (0..width).all(|slot| {
            residual_successor_block(left, slot, transitions, width, previous_blocks)
                == residual_successor_block(right, slot, transitions, width, previous_blocks)
        })
}

fn insert_dwa_transition(
    transitions: &mut BTreeMap<i32, (u32, Weight)>,
    label: i32,
    target: u32,
    weight: Weight,
) {
    match transitions.entry(label) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert((target, weight));
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            let (existing_target, existing_weight) = entry.get_mut();
            assert_eq!(
                *existing_target,
                target,
                "class expansion requires distinct first/continuation DWA sources"
            );
            *existing_weight = existing_weight.union(&weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    fn all_bytes() -> [bool; 256] {
        [true; 256]
    }

    #[test]
    fn duplicate_terminal_scanners_share_one_representative() {
        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        let equivalence = TerminalEquivalence::build(
            &tokenizer,
            &[true, true, true],
            None,
            &all_bytes(),
        );

        assert_eq!(equivalence.representative_for_terminal[0], 0);
        assert_eq!(equivalence.representative_for_terminal[1], 0);
        assert_eq!(equivalence.members_by_representative[0], vec![0, 1]);
        assert_eq!(equivalence.profile().quotient_hits, 1);
    }

    #[test]
    fn restricted_partition_merges_distinct_literal_keys() {
        let expressions = vec![
            Expr::U8Seq(br#"\"x\": "#.to_vec()),
            Expr::U8Seq(br#"\"y\": "#.to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let mut relevant = [false; 256];
        for &byte in br#"\": "# {
            relevant[byte as usize] = true;
        }

        let equivalence = TerminalEquivalence::build(&tokenizer, &[true, true], None, &relevant);

        assert_eq!(equivalence.representative_for_terminal[0], 0);
        assert_eq!(equivalence.representative_for_terminal[1], 0);
        assert_eq!(equivalence.profile().quotient_hits, 1);
    }

    #[test]
    fn ignored_terminal_is_not_merged() {
        let expressions = vec![Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"ab".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        let equivalence = TerminalEquivalence::build(&tokenizer, &[true, true], Some(1), &all_bytes());

        assert!(equivalence.is_identity());
    }
}

#[cfg(test)]
mod class_follow_tests {
    use super::*;

    #[test]
    fn class_pre_follow_keeps_any_member_pair_that_is_legal() {
        // Class 0 = {0, 1}; class 2 = {2, 3}; class 4 = {4}. The pair 0→2
        // is legal, although 1→2 and both transitions to 3 are forbidden.
        // By contrast every pair into class 4 is forbidden. The class pre-pass
        // must retain class 0→2 but reject class 0→4.
        let equivalence = TerminalEquivalence {
            representative_for_terminal: vec![0, 0, 2, 2, 4],
            members_by_representative: vec![
                vec![0, 1],
                Vec::new(),
                vec![2, 3],
                Vec::new(),
                vec![4],
            ],
            active_representatives: vec![true, false, true, false, true],
            state_blocks_by_terminal: vec![Vec::new(); 5],
            member_live_tsids: vec![None; 5],
            original_active_terminals: vec![true; 5],
            active_terminal_count: 5,
            class_count: 3,
            quotient_hits: 2,
            residual_pair_count: 0,
            residual_block_count: 0,
            active_byte_count: 0,
        };
        let mut disallowed = BTreeMap::new();
        let mut zero = BitSet::new(5);
        zero.set(3);
        zero.set(4);
        disallowed.insert(0, zero);
        let mut one = BitSet::new(5);
        one.set(2);
        one.set(3);
        one.set(4);
        disallowed.insert(1, one);

        let class_disallowed = equivalence.class_disallowed_follows(&disallowed, 5);
        assert!(
            !class_disallowed
                .get(&0)
                .is_some_and(|bits| bits.contains(2)),
            "the legal member pair 0→2 must keep class 0→2"
        );
        assert!(
            class_disallowed
                .get(&0)
                .is_some_and(|bits| bits.contains(4)),
            "every member pair into class 4 is forbidden"
        );
    }
}
