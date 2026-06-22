//! Partition-restricted L2P terminal equivalence.
//!
//! Two terminals may differ in the global lexer yet be indistinguishable on a
//! vocabulary partition. We minimize their residual scanners over that
//! partition's byte alphabet. A member terminal state is then mapped to a
//! representative terminal state in the same residual block.
//!
//! The terminal NWA still needs a source state for every runtime TSID. Its raw
//! build therefore uses only representative labels, but preserves the full
//! state map. Expansion copies representative root transitions through the
//! member-to-representative residual-state map. Merely remapping edge weights
//! cannot create a member transition at a root where the representative was not
//! live.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
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
    /// For an active non-representative member, indexed by source TSID. Each
    /// entry contains every representative TSID corresponding to an original
    /// lexer state inside that source TSID. This relation is intentionally
    /// one-to-many: a merged runtime TSID may contain several member residuals
    /// that map to different, but equivalent, representative residuals.
    member_source_to_rep_tsids: Vec<Option<Vec<Vec<u32>>>>,
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
            member_source_to_rep_tsids: vec![None; num_terminals],
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
            member_source_to_rep_tsids: vec![None; num_terminals],
            original_active_terminals: active_terminals.to_vec(),
            active_terminal_count: active_ids.len(),
            class_count,
            quotient_hits,
            residual_pair_count: pair_terminals.len(),
            residual_block_count,
            active_byte_count: active_bytes.len(),
        }
    }

    /// Retain only members that can map every runtime source TSID to one
    /// representative TSID. The initial residual must also map to the
    /// representative's initial residual because post-match scans restart there.
    pub(crate) fn refine_for_tsid_map(
        &mut self,
        tokenizer_states: &ManyToOneIdMap,
        initial_state: u32,
    ) {
        self.member_source_to_rep_tsids.fill(None);
        let num_terminals = self.representative_for_terminal.len();
        let num_tsids = tokenizer_states.num_internal_ids() as usize;

        for representative_index in 0..num_terminals {
            let representative = representative_index as TerminalID;
            let members = self.members_by_representative[representative_index].clone();
            if members.len() <= 1 || members[0] != representative {
                continue;
            }

            let representative_initial_block = self.block_for(representative, initial_state);
            let mut retained = vec![representative];
            for member in members.into_iter().skip(1) {
                if self.block_for(member, initial_state) != representative_initial_block {
                    self.make_singleton(member);
                    continue;
                }

                let mut source_map = vec![Vec::new(); num_tsids];
                for (source_tsid, originals) in tokenizer_states.internal_to_originals.iter().enumerate() {
                    for &original in originals {
                        let block = self.block_for(member, original);
                        if let Some(representative_tsid) = self
                            .state_for_block(representative, block)
                            .and_then(|state| {
                                tokenizer_states
                                    .original_to_internal
                                    .get(state as usize)
                                    .copied()
                            })
                            .filter(|&tsid| tsid != u32::MAX)
                        {
                            source_map[source_tsid].push(representative_tsid);
                        }
                    }
                }
                for representative_tsids in &mut source_map {
                    representative_tsids.sort_unstable();
                    representative_tsids.dedup();
                }
                self.member_source_to_rep_tsids[member as usize] = Some(source_map);
                retained.push(member);
            }
            self.members_by_representative[representative_index] = retained;
        }
        self.recompute_class_metadata();
    }

    pub(crate) fn representative_active_terminals(&self) -> &[bool] {
        &self.active_representatives
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

    /// Restore member labels on the raw NWA. A member source TSID maps to one
    /// or more representative source TSIDs with the same partition-restricted
    /// residual. Copying those roots is essential: a representative-only NWA
    /// has no edge at member-only roots to repair by changing weights alone.
    pub(crate) fn expand_nwa(
        &self,
        nwa: &mut NWA,
        roots_by_tsid: &[u32],
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

        let raw_transitions = nwa
            .states()
            .iter()
            .map(|state| state.transitions.clone())
            .collect::<Vec<_>>();
        let mut is_root = vec![false; raw_transitions.len()];
        for &root in roots_by_tsid {
            if let Some(slot) = is_root.get_mut(root as usize) {
                *slot = true;
            }
        }

        // The first terminal of a token scans from a member-specific lexer
        // residual, so its source root must be mapped explicitly.
        for (representative_index, members) in self.members_by_representative.iter().enumerate() {
            if members.len() <= 1 {
                continue;
            }
            let representative = representative_index as TerminalID;
            for &member in members.iter().skip(1) {
                let source_map = self.member_source_to_rep_tsids[member as usize]
                    .as_ref()
                    .expect("retained terminal-equivalence member lacks source map");
                for (member_tsid, representative_tsids) in source_map.iter().enumerate() {
                    if representative_tsids.is_empty() {
                        continue;
                    }
                    let Some(&member_root) = roots_by_tsid.get(member_tsid) else {
                        continue;
                    };
                    for &representative_tsid in representative_tsids {
                        let Some(&representative_root) = roots_by_tsid.get(representative_tsid as usize) else {
                            continue;
                        };
                        let Some(targets) = raw_transitions[representative_root as usize]
                            .get(&(representative as i32))
                        else {
                            continue;
                        };
                        merge_labeled_targets(
                            &mut nwa.states_mut()[member_root as usize].transitions,
                            member as i32,
                            targets,
                        );
                        profile.expanded_transition_copies += targets.len();
                        profile.root_source_maps += 1;
                    }
                }
            }
        }

        // Every later terminal scan starts at the common tokenizer initial
        // state. Refinement required member and representative to share that
        // restricted residual, so the continuation topology is shared.
        for (state_id, transitions) in raw_transitions.iter().enumerate() {
            if is_root.get(state_id).copied().unwrap_or(false) {
                continue;
            }
            for (representative_index, members) in self.members_by_representative.iter().enumerate() {
                if members.len() <= 1 {
                    continue;
                }
                let representative = representative_index as TerminalID;
                let Some(targets) = transitions.get(&(representative as i32)) else {
                    continue;
                };
                for &member in members.iter().skip(1) {
                    merge_labeled_targets(
                        &mut nwa.states_mut()[state_id].transitions,
                        member as i32,
                        targets,
                    );
                    profile.expanded_transition_copies += targets.len();
                }
            }
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

    fn state_for_block(&self, terminal: TerminalID, block: u32) -> Option<u32> {
        if block == DEAD_BLOCK {
            return None;
        }
        self.state_blocks_by_terminal[terminal as usize]
            .iter()
            .find_map(|&(state, candidate)| (candidate == block).then_some(state))
    }

    fn make_singleton(&mut self, terminal: TerminalID) {
        let index = terminal as usize;
        self.representative_for_terminal[index] = terminal;
        self.members_by_representative[index] = vec![terminal];
        self.active_representatives[index] = true;
        self.member_source_to_rep_tsids[index] = None;
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
    let row_width = width + 1;
    let mut signatures = vec![DEAD_BLOCK; outputs.len() * row_width];
    for pair in 0..outputs.len() {
        let row = &mut signatures[pair * row_width..(pair + 1) * row_width];
        row[0] = outputs[pair] as u32;
        for slot in 0..width {
            let target = transitions[pair * width + slot];
            row[slot + 1] = if target == NO_PAIR {
                DEAD_BLOCK
            } else {
                previous_blocks[target as usize]
            };
        }
    }

    let mut order = (0..outputs.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        signatures[left * row_width..(left + 1) * row_width]
            .cmp(&signatures[right * row_width..(right + 1) * row_width])
    });
    let mut blocks = vec![DEAD_BLOCK; outputs.len()];
    let mut next_block = DEAD_BLOCK;
    let mut previous = None;
    for pair in order {
        let starts_new = previous.map_or(true, |previous| {
            signatures[pair * row_width..(pair + 1) * row_width]
                != signatures[previous * row_width..(previous + 1) * row_width]
        });
        if starts_new {
            next_block += 1;
        }
        blocks[pair] = next_block;
        previous = Some(pair);
    }
    blocks
}

fn merge_labeled_targets(
    transitions: &mut BTreeMap<i32, Vec<(u32, Weight)>>,
    label: i32,
    targets: &[(u32, Weight)],
) {
    let entry = transitions.entry(label).or_default();
    for (target, weight) in targets {
        if let Some((_, existing)) = entry.iter_mut().find(|(existing_target, _)| *existing_target == *target) {
            *existing = existing.union(weight);
        } else {
            entry.push((*target, weight.clone()));
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
        assert_eq!(equivalence.representative_active_terminals(), &[true, true]);
    }
}
