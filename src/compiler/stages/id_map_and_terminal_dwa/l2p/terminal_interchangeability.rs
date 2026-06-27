//! Exact terminal interchangeability for the L2+ terminal-DWA path.
//!
//! This uses the full-row swap definition. A complete row is represented by
//! the joint residual output function `w -> F(delta*(s,w))` over active bytes.
//! The expansion below is intentionally a slow reference construction: it
//! merges one full DWA view for every representative/member swap.

use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHasher, FxHashSet};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted_u32::dwa::{DWA, DWAState};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::weight::{SharedTokenSet, Weight};
use crate::grammar::flat::TerminalID;

const NO_STATE: u32 = u32::MAX;
const DEAD_RESIDUAL_BLOCK: u32 = 0;
const NO_RESIDUAL_PAIR: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OutputBits(Vec<u64>);

impl OutputBits {
    fn empty(words: usize) -> Self { Self(vec![0; words]) }

    fn swapped(&self, left: usize, right: usize) -> Self {
        if left == right { return self.clone(); }
        let mut result = self.clone();
        let lw = left / 64;
        let rw = right / 64;
        let lm = 1u64 << (left % 64);
        let rm = 1u64 << (right % 64);
        if ((self.0[lw] & lm) != 0) != ((self.0[rw] & rm) != 0) {
            result.0[lw] ^= lm;
            result.0[rw] ^= rm;
        }
        result
    }

    fn contains(&self, terminal: usize) -> bool {
        self.0[terminal / 64] & (1u64 << (terminal % 64)) != 0
    }

    fn clear(&mut self, terminal: usize) {
        self.0[terminal / 64] &= !(1u64 << (terminal % 64));
    }

    fn set_to(&mut self, terminal: usize, value: bool) {
        let mask = 1u64 << (terminal % 64);
        let word = &mut self.0[terminal / 64];
        if value { *word |= mask; } else { *word &= !mask; }
    }

    fn member_as_representative(&self, member: usize, representative: usize) -> Self {
        let member_value = self.contains(member);
        let mut result = self.clone();
        result.clear(member);
        result.set_to(representative, member_value);
        result
    }

    fn without(&self, terminal: usize) -> Self {
        let mut result = self.clone();
        result.clear(terminal);
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MooreKey { output: OutputBits, successors: Vec<u32> }

#[derive(Clone, Debug)]
struct RowMachine {
    width: usize,
    terminal_count: usize,
    class_for_state: Vec<u32>, // real tokenizer states + synthetic dead state
    class_outputs: Vec<OutputBits>,
    class_transitions: Vec<u32>,
    class_terminals: Vec<Vec<usize>>,
    terminal_classes: Vec<Vec<usize>>,
    class_has_real_state: Vec<bool>,
    class_original_states: Vec<Vec<u32>>,
    real_state_count: usize,
}

impl RowMachine {
    fn class_count(&self) -> usize { self.class_outputs.len() }
    fn transition(&self, class: u32, slot: usize) -> u32 {
        self.class_transitions[class as usize * self.width + slot]
    }
}

#[derive(Clone, Debug)]
struct SparseTerminalResidualProduct {
    state_count: usize,
    terminal_count: usize,
    pair_terminals: Vec<TerminalID>,
    pair_states: Vec<u32>,
    outputs: Vec<bool>,
    transitions: Vec<u32>,
    raw_width: usize,
    width: usize,
    live_states_ms: f64,
    transition_build_ms: f64,
    column_quotient_ms: f64,
}

impl SparseTerminalResidualProduct {
    fn build(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        let started_at = Instant::now();
        let terminal_count = tokenizer.num_terminals() as usize;
        let state_count = tokenizer.num_states() as usize;
        let active_bytes = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect::<Vec<_>>();
        let raw_width = active_bytes.len();

        let mut live_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        for state in 0..state_count as u32 {
            for terminal in tokenizer.possible_future_terminals_iter(state) {
                live_states_by_terminal[terminal as usize].push(state);
            }
            for terminal in tokenizer.matched_terminals_iter(state) {
                live_states_by_terminal[terminal as usize].push(state);
            }
        }
        for states in &mut live_states_by_terminal {
            states.sort_unstable();
            states.dedup();
        }
        let live_states_ms = started_at.elapsed().as_secs_f64() * 1000.0;

        let mut pair_ids_by_terminal = vec![Vec::<(u32, u32)>::new(); terminal_count];
        let mut pair_terminals = Vec::<TerminalID>::new();
        let mut pair_states = Vec::<u32>::new();
        let mut outputs = Vec::<bool>::new();
        for terminal in 0..terminal_count as TerminalID {
            let pair_ids = &mut pair_ids_by_terminal[terminal as usize];
            for &state in &live_states_by_terminal[terminal as usize] {
                let pair = pair_terminals.len() as u32;
                pair_ids.push((state, pair));
                pair_terminals.push(terminal);
                pair_states.push(state);
                outputs.push(tokenizer.matched_terminal_bitset(state).contains(terminal as usize));
            }
        }

        let mut transitions = vec![NO_RESIDUAL_PAIR; pair_terminals.len() * raw_width];
        for pair in 0..pair_terminals.len() {
            let terminal = pair_terminals[pair] as usize;
            let state = pair_states[pair];
            for (slot, &byte) in active_bytes.iter().enumerate() {
                let next = tokenizer.get_transition(state, byte);
                if next != NO_STATE {
                    if let Ok(index) = pair_ids_by_terminal[terminal]
                        .binary_search_by_key(&next, |(candidate, _)| *candidate)
                    {
                        transitions[pair * raw_width + slot] = pair_ids_by_terminal[terminal][index].1;
                    }
                }
            }
        }
        let transition_build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        let (transitions, width) =
            quotient_sparse_transition_columns(transitions, pair_terminals.len(), raw_width);
        let column_quotient_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        Self {
            state_count,
            terminal_count,
            pair_terminals,
            pair_states,
            outputs,
            transitions,
            raw_width,
            width,
            live_states_ms,
            transition_build_ms,
            column_quotient_ms,
        }
    }
}

impl SparseTerminalResiduals {
    fn build(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        let started_at = Instant::now();
        let product = SparseTerminalResidualProduct::build(tokenizer, relevant_bytes);
        let minimize_started_at = Instant::now();
        let (blocks, refinement_steps) =
            minimize_sparse_terminal_residuals(&product.outputs, &product.transitions, product.width);
        let residual_minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
        let raw_width = product.raw_width;
        let width = product.width;
        let live_states_ms = product.live_states_ms;
        let transition_build_ms = product.transition_build_ms;
        let column_quotient_ms = product.column_quotient_ms;
        let result = Self::from_product(&product, &blocks);
        if std::env::var_os("GLRMASK_PROFILE_L2P_SPARSE_RESIDUALS").is_some() {
            eprintln!(
                "[glrmask/profile][l2p_sparse_residual_build] terminals={} states={} pairs={} raw_bytes={} residual_bytes={} steps={} live_ms={:.3} transitions_ms={:.3} column_quotient_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
                result.terminal_count,
                result.state_count,
                result.pair_count,
                raw_width,
                width,
                refinement_steps,
                live_states_ms,
                transition_build_ms,
                column_quotient_ms,
                residual_minimize_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    fn from_product(product: &SparseTerminalResidualProduct, blocks: &[u32]) -> Self {
        let block_count = blocks.iter().copied().max().unwrap_or(DEAD_RESIDUAL_BLOCK) as usize + 1;
        let mut state_blocks_by_terminal = vec![Vec::<(u32, u32)>::new(); product.terminal_count];
        let mut rows_by_state = vec![Vec::<(TerminalID, u32)>::new(); product.state_count];
        let mut inventories_by_terminal = vec![Vec::<u32>::new(); product.terminal_count];
        for pair in 0..product.pair_terminals.len() {
            let terminal = product.pair_terminals[pair];
            let state = product.pair_states[pair];
            let block = blocks[pair];
            if block == DEAD_RESIDUAL_BLOCK {
                continue;
            }
            state_blocks_by_terminal[terminal as usize].push((state, block));
            rows_by_state[state as usize].push((terminal, block));
            inventories_by_terminal[terminal as usize].push(block);
        }
        for inventory in &mut inventories_by_terminal {
            inventory.sort_unstable();
            inventory.dedup();
        }
        let full_row_hashes = rows_by_state
            .iter()
            .map(|row| sparse_full_row_hash(row))
            .collect::<Vec<_>>();
        let mut states_by_full_row_hash = FxHashMap::<u64, Vec<u32>>::default();
        for (state, &hash) in full_row_hashes.iter().enumerate() {
            states_by_full_row_hash.entry(hash).or_default().push(state as u32);
        }
        Self {
            state_count: product.state_count,
            terminal_count: product.terminal_count,
            state_blocks_by_terminal,
            rows_by_state,
            full_row_hashes,
            states_by_full_row_hash,
            inventories_by_terminal,
            pair_count: product.pair_terminals.len(),
            block_count,
        }
    }

    fn block_for(&self, terminal: TerminalID, state: u32) -> u32 {
        self.state_blocks_by_terminal[terminal as usize]
            .binary_search_by_key(&state, |(candidate, _)| *candidate)
            .ok()
            .map(|index| self.state_blocks_by_terminal[terminal as usize][index].1)
            .unwrap_or(DEAD_RESIDUAL_BLOCK)
    }

    /// A terminal that may appear after another terminal within one token
    /// restarts at the lexer's fixed initial state. Its continuation label can
    /// therefore be expanded only when its complete row agrees there after the
    /// member→representative relabeling; unrooted subsumption alone is not
    /// enough.
    fn continuation_compatible(
        &self,
        initial_state: u32,
        member: TerminalID,
        representative: TerminalID,
    ) -> bool {
        sparse_row_member_subsumed_by(
            &self.rows_by_state[initial_state as usize],
            &self.rows_by_state[initial_state as usize],
            member,
            representative,
        )
    }

    fn inventory_is_subset_of(&self, member: TerminalID, representative: TerminalID) -> bool {
        let member_inventory = &self.inventories_by_terminal[member as usize];
        let representative_inventory = &self.inventories_by_terminal[representative as usize];
        let mut member_index = 0usize;
        let mut representative_index = 0usize;
        while member_index < member_inventory.len() {
            while representative_index < representative_inventory.len()
                && representative_inventory[representative_index] < member_inventory[member_index]
            {
                representative_index += 1;
            }
            if representative_index == representative_inventory.len()
                || representative_inventory[representative_index] != member_inventory[member_index]
            {
                return false;
            }
            member_index += 1;
        }
        true
    }

    /// Exact directed set-of-correlated-rows test. It checks
    /// `Sig(Q, T\{representative}, [member→representative]) ⊆
    ///  Sig(Q, T\{member})` using sparse canonical residual signatures.
    fn subsumption_transport(
        &self,
        member: TerminalID,
        representative: TerminalID,
    ) -> Option<SubsumptionTransport> {
        if member == representative || !self.inventory_is_subset_of(member, representative) {
            return None;
        }

        let mut relevant_sources = self.state_blocks_by_terminal[member as usize]
            .iter()
            .map(|&(state, _)| state)
            .chain(
                self.state_blocks_by_terminal[representative as usize]
                    .iter()
                    .map(|&(state, _)| state),
            )
            .collect::<Vec<_>>();
        relevant_sources.sort_unstable();
        relevant_sources.dedup();
        // First decide inclusion. On the overwhelmingly common failing pair we
        // need only one RHS witness per source row, not every duplicate state
        // carrying that row.
        for source_state in relevant_sources.iter().copied() {
            let source_row = &self.rows_by_state[source_state as usize];
            if self
                .matching_rhs_states(source_row, member, representative, true)
                .is_empty()
            {
                return None;
            }
        }

        // States outside the member/representative support have identical
        // projected rows on both sides and therefore witness themselves.
        let mut is_relevant = vec![false; self.state_count];
        for &state in &relevant_sources {
            is_relevant[state as usize] = true;
        }
        let mut representative_state_to_members = vec![Vec::<u32>::new(); self.state_count];
        for state in 0..self.state_count as u32 {
            if !is_relevant[state as usize] {
                representative_state_to_members[state as usize].push(state);
            }
        }
        for source_state in relevant_sources {
            let source_row = &self.rows_by_state[source_state as usize];
            for target_state in self.matching_rhs_states(source_row, member, representative, false) {
                representative_state_to_members[target_state as usize].push(source_state);
            }
        }
        for members in &mut representative_state_to_members {
            members.sort_unstable();
            members.dedup();
        }
        Some(SubsumptionTransport { representative_state_to_members })
    }

    fn matching_rhs_states(
        &self,
        source_row: &[(TerminalID, u32)],
        member: TerminalID,
        representative: TerminalID,
        stop_after_first: bool,
    ) -> Vec<u32> {
        let hash = sparse_member_to_representative_row_hash(source_row, member, representative);
        let mut matches = Vec::new();
        if let Some(candidates) = self.states_by_full_row_hash.get(&hash) {
            for &target_state in candidates {
                if self.block_for(member, target_state) != DEAD_RESIDUAL_BLOCK {
                    continue;
                }
                if sparse_row_member_subsumed_by(
                    source_row,
                    &self.rows_by_state[target_state as usize],
                    member,
                    representative,
                ) {
                    matches.push(target_state);
                    if stop_after_first {
                        return matches;
                    }
                }
            }
        }
        for &(target_state, member_block) in &self.state_blocks_by_terminal[member as usize] {
            if self.full_row_hashes[target_state as usize]
                ^ sparse_row_item_hash(member, member_block)
                != hash
            {
                continue;
            }
            if sparse_row_member_subsumed_by(
                source_row,
                &self.rows_by_state[target_state as usize],
                member,
                representative,
            ) {
                matches.push(target_state);
                if stop_after_first {
                    return matches;
                }
            }
        }
        matches
    }

    fn row_hash_index(&self) -> SparseRowHashIndex {
        let mut full_counts = FxHashMap::default();
        for &hash in &self.full_row_hashes {
            *full_counts.entry(hash).or_insert(0) += 1;
        }
        SparseRowHashIndex { full_counts }
    }

    fn member_rhs_row_hashes(&self, member: TerminalID) -> MemberRhsRowHashes {
        let mut removed_full_counts = FxHashMap::default();
        let mut relabelled_member_hashes = FxHashSet::default();
        for &(state, block) in &self.state_blocks_by_terminal[member as usize] {
            let full_hash = self.full_row_hashes[state as usize];
            *removed_full_counts.entry(full_hash).or_insert(0) += 1;
            relabelled_member_hashes.insert(full_hash ^ sparse_row_item_hash(member, block));
        }
        MemberRhsRowHashes {
            removed_full_counts,
            relabelled_member_hashes,
        }
    }

    fn hash_subsumption_possible(
        &self,
        index: &SparseRowHashIndex,
        member: TerminalID,
        representative: TerminalID,
        rhs: &MemberRhsRowHashes,
    ) -> bool {
        let member_states = &self.state_blocks_by_terminal[member as usize];
        let representative_states = &self.state_blocks_by_terminal[representative as usize];
        let mut member_index = 0usize;
        let mut representative_index = 0usize;
        while member_index < member_states.len() || representative_index < representative_states.len() {
            let next_member = member_states.get(member_index).map(|&(state, _)| state);
            let next_representative = representative_states
                .get(representative_index)
                .map(|&(state, _)| state);
            let state = match (next_member, next_representative) {
                (Some(left), Some(right)) => left.min(right),
                (Some(left), None) => left,
                (None, Some(right)) => right,
                (None, None) => break,
            };
            let member_block = if next_member == Some(state) {
                let block = member_states[member_index].1;
                member_index += 1;
                block
            } else {
                DEAD_RESIDUAL_BLOCK
            };
            let representative_block = if next_representative == Some(state) {
                let block = representative_states[representative_index].1;
                representative_index += 1;
                block
            } else {
                DEAD_RESIDUAL_BLOCK
            };
            let mut transformed_hash = self.full_row_hashes[state as usize];
            if member_block != DEAD_RESIDUAL_BLOCK {
                transformed_hash ^= sparse_row_item_hash(member, member_block);
                transformed_hash ^= sparse_row_item_hash(representative, member_block);
            }
            if representative_block != DEAD_RESIDUAL_BLOCK {
                transformed_hash ^= sparse_row_item_hash(representative, representative_block);
            }
            let unchanged_count = index.full_counts.get(&transformed_hash).copied().unwrap_or(0);
            let removed_count = rhs.removed_full_counts.get(&transformed_hash).copied().unwrap_or(0);
            if unchanged_count <= removed_count
                && !rhs.relabelled_member_hashes.contains(&transformed_hash)
            {
                return false;
            }
        }
        true
    }

    fn subsumption_transports(
        &self,
        candidates: &[(TerminalID, TerminalID)],
    ) -> FxHashMap<(TerminalID, TerminalID), SubsumptionTransport> {
        candidates
            .iter()
            .filter_map(|&(member, representative)| {
                self.subsumption_transport(member, representative)
                    .map(|transport| ((member, representative), transport))
            })
            .collect()
    }
}

fn sparse_row_item_hash(terminal: TerminalID, block: u32) -> u64 {
    sigma_mix(terminal as usize, block)
}

fn sparse_full_row_hash(row: &[(TerminalID, u32)]) -> u64 {
    row.iter().fold(0u64, |hash, &(terminal, block)| {
        hash ^ sparse_row_item_hash(terminal, block)
    })
}

fn sparse_rhs_row_hash(row: &[(TerminalID, u32)], member: TerminalID) -> u64 {
    row.iter().fold(0u64, |hash, &(terminal, block)| {
        if terminal == member { hash } else { hash ^ sparse_row_item_hash(terminal, block) }
    })
}

fn sparse_member_to_representative_row_hash(
    row: &[(TerminalID, u32)],
    member: TerminalID,
    representative: TerminalID,
) -> u64 {
    row.iter().fold(0u64, |hash, &(terminal, block)| {
        if terminal == representative {
            hash
        } else if terminal == member {
            hash ^ sparse_row_item_hash(representative, block)
        } else {
            hash ^ sparse_row_item_hash(terminal, block)
        }
    })
}

fn sparse_row_member_subsumed_by(
    member_row: &[(TerminalID, u32)],
    representative_row: &[(TerminalID, u32)],
    member: TerminalID,
    representative: TerminalID,
) -> bool {
    // Removing `representative` on the LHS and `member` on the RHS gives a
    // bijection of terminal labels, with `member` relabelled to
    // `representative`. Equal cardinality plus one exact lookup per LHS item
    // is therefore sufficient; no temporary sorted row is needed.
    let lhs_len = member_row.len()
        - usize::from(sparse_row_block(member_row, representative).is_some());
    let rhs_len = representative_row.len()
        - usize::from(sparse_row_block(representative_row, member).is_some());
    if lhs_len != rhs_len {
        return false;
    }
    for &(terminal, block) in member_row {
        if terminal == representative {
            continue;
        }
        let mapped_terminal = if terminal == member { representative } else { terminal };
        if sparse_row_block(representative_row, mapped_terminal) != Some(block) {
            return false;
        }
    }
    true
}

#[inline]
fn sparse_row_block(row: &[(TerminalID, u32)], terminal: TerminalID) -> Option<u32> {
    row.binary_search_by_key(&terminal, |&(candidate, _)| candidate)
        .ok()
        .map(|index| row[index].1)
}

/// Merge active byte columns only when their transition functions agree for
/// every live `(terminal, lexer-state)` pair. This is an exact alphabet quotient:
/// the residual DFA observes no distinction between the retained copies.
fn quotient_sparse_transition_columns(
    transitions: Vec<u32>,
    pairs: usize,
    width: usize,
) -> (Vec<u32>, usize) {
    if width <= 1 || pairs == 0 {
        return (transitions, width);
    }
    let mut buckets = FxHashMap::<u64, Vec<usize>>::default();
    let mut retained = Vec::<usize>::new();
    for slot in 0..width {
        let mut hasher = FxHasher::default();
        for pair in 0..pairs {
            transitions[pair * width + slot].hash(&mut hasher);
        }
        let hash = hasher.finish();
        let candidates = buckets.entry(hash).or_default();
        let duplicate = candidates.iter().copied().any(|other| {
            (0..pairs).all(|pair| {
                transitions[pair * width + slot] == transitions[pair * width + other]
            })
        });
        if !duplicate {
            candidates.push(slot);
            retained.push(slot);
        }
    }
    if retained.len() == width {
        return (transitions, width);
    }
    let mut reduced = vec![NO_RESIDUAL_PAIR; pairs * retained.len()];
    for pair in 0..pairs {
        for (reduced_slot, &slot) in retained.iter().enumerate() {
            reduced[pair * retained.len() + reduced_slot] = transitions[pair * width + slot];
        }
    }
    (reduced, retained.len())
}

#[derive(Clone, Debug)]
struct SparseInventoryPrune {
    candidate_counts: Vec<usize>,
    hash_candidate_counts: Vec<Option<usize>>,
    converged: bool,
}

impl SparseInventoryPrune {
    fn eliminated_all(&self) -> bool {
        self.candidate_counts.last().copied() == Some(0)
            || self.hash_candidate_counts.last().copied() == Some(Some(0))
    }
}

/// A necessary-condition filter for directed subsumption. At any finite residual
/// depth, every member residual signature must already occur for its proposed
/// representative. Once the candidate count reaches zero, exact subsumption is
/// impossible and no deeper refinement can restore it.
fn sparse_inventory_prune(
    product: &SparseTerminalResidualProduct,
    active_ids: &[TerminalID],
) -> SparseInventoryPrune {
    let mut blocks = initial_sparse_terminal_blocks(&product.outputs);
    let mut candidate_counts = Vec::new();
    let mut hash_candidate_counts = Vec::new();
    loop {
        let count = sparse_inventory_candidate_count(product, &blocks, active_ids);
        candidate_counts.push(count);
        if count == 0 {
            return SparseInventoryPrune {
                candidate_counts,
                hash_candidate_counts,
                converged: false,
            };
        }
        if count <= product.pair_terminals.len() {
            let candidates = sparse_inventory_candidate_pairs(product, &blocks, active_ids);
            let rows = SparseTerminalResiduals::from_product(product, &blocks);
            let survivors = sparse_row_hash_candidates(&rows, &candidates.pairs);
            hash_candidate_counts.push(Some(survivors.len()));
            if survivors.is_empty() {
                return SparseInventoryPrune {
                    candidate_counts,
                    hash_candidate_counts,
                    converged: false,
                };
            }
        } else {
            hash_candidate_counts.push(None);
        }
        let next = refine_sparse_terminal_blocks(
            &product.outputs,
            &product.transitions,
            product.width,
            &blocks,
        );
        if next == blocks {
            return SparseInventoryPrune {
                candidate_counts,
                hash_candidate_counts,
                converged: true,
            };
        }
        blocks = next;
    }
}

#[derive(Clone, Debug)]
struct SparseInventoryCandidates {
    count: usize,
    pairs: Vec<(TerminalID, TerminalID)>,
}

#[derive(Clone, Debug)]
struct SparseRowHashIndex {
    full_counts: FxHashMap<u64, usize>,
}

#[derive(Clone, Debug)]
struct MemberRhsRowHashes {
    removed_full_counts: FxHashMap<u64, usize>,
    relabelled_member_hashes: FxHashSet<u64>,
}

fn sparse_inventory_candidate_count(
    product: &SparseTerminalResidualProduct,
    blocks: &[u32],
    active_ids: &[TerminalID],
) -> usize {
    sparse_inventory_candidates(product, blocks, active_ids, false).count
}

fn sparse_inventory_candidate_pairs(
    product: &SparseTerminalResidualProduct,
    blocks: &[u32],
    active_ids: &[TerminalID],
) -> SparseInventoryCandidates {
    sparse_inventory_candidates(product, blocks, active_ids, true)
}

fn sparse_inventory_candidates(
    product: &SparseTerminalResidualProduct,
    blocks: &[u32],
    active_ids: &[TerminalID],
    materialize: bool,
) -> SparseInventoryCandidates {
    if active_ids.len() < 2 {
        return SparseInventoryCandidates { count: 0, pairs: Vec::new() };
    }
    let words = product.terminal_count.div_ceil(64);
    let mut is_active = vec![false; product.terminal_count];
    let mut active_mask = vec![0u64; words];
    for &terminal in active_ids {
        let terminal = terminal as usize;
        is_active[terminal] = true;
        active_mask[terminal / 64] |= 1u64 << (terminal % 64);
    }
    let max_block = blocks.iter().copied().max().unwrap_or(DEAD_RESIDUAL_BLOCK) as usize;
    let mut terminal_blocks = vec![Vec::<u32>::new(); product.terminal_count];
    for pair in 0..product.pair_terminals.len() {
        let terminal = product.pair_terminals[pair] as usize;
        let block = blocks[pair];
        if block != DEAD_RESIDUAL_BLOCK && is_active[terminal] {
            terminal_blocks[terminal].push(block);
        }
    }
    let mut block_terminals = vec![0u64; (max_block + 1) * words];
    for &terminal in active_ids {
        let terminal = terminal as usize;
        let inventory = &mut terminal_blocks[terminal];
        inventory.sort_unstable();
        inventory.dedup();
        for &block in inventory.iter() {
            block_terminals[block as usize * words + terminal / 64] |= 1u64 << (terminal % 64);
        }
    }

    let mut count = 0usize;
    let mut pairs = Vec::new();
    for &member in active_ids {
        let member = member as usize;
        let mut possible = active_mask.clone();
        possible[member / 64] &= !(1u64 << (member % 64));
        for &block in &terminal_blocks[member] {
            let membership = &block_terminals[block as usize * words..(block as usize + 1) * words];
            for (candidate_word, &members) in possible.iter_mut().zip(membership) {
                *candidate_word &= members;
            }
        }
        count += possible.iter().map(|word| word.count_ones() as usize).sum::<usize>();
        if materialize {
            for (word_index, word) in possible.iter().copied().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let offset = bits.trailing_zeros() as usize;
                    pairs.push((member as TerminalID, (word_index * 64 + offset) as TerminalID));
                    bits &= bits - 1;
                }
            }
        }
    }
    SparseInventoryCandidates { count, pairs }
}


fn sparse_row_hash_candidates(
    rows: &SparseTerminalResiduals,
    candidates: &[(TerminalID, TerminalID)],
) -> Vec<(TerminalID, TerminalID)> {
    let index = rows.row_hash_index();
    let mut by_member = BTreeMap::<TerminalID, Vec<TerminalID>>::new();
    for &(member, representative) in candidates {
        by_member.entry(member).or_default().push(representative);
    }
    let mut result = Vec::new();
    for (member, representatives) in by_member {
        let rhs = rows.member_rhs_row_hashes(member);
        for representative in representatives {
            if rows.hash_subsumption_possible(&index, member, representative, &rhs) {
                result.push((member, representative));
            }
        }
    }
    result
}

/// Exact DFA minimization for the sparse terminal-residual product.  This is
/// Hopcroft partition refinement over the live pairs plus one shared dead state.
/// The byte alphabet has already been quotient-ed exactly above.
fn minimize_sparse_terminal_residuals(
    outputs: &[bool],
    transitions: &[u32],
    width: usize,
) -> (Vec<u32>, usize) {
    let pairs = outputs.len();
    let dead = pairs;
    let states = pairs + 1;

    // For every byte, materialize reverse edges as a target-indexed contiguous
    // range. There is one outgoing edge per state/byte, including dead's loop.
    let mut offsets = vec![0usize; width * (states + 1)];
    for source in 0..states {
        for slot in 0..width {
            let target = if source == dead {
                dead
            } else {
                let target = transitions[source * width + slot];
                if target == NO_RESIDUAL_PAIR { dead } else { target as usize }
            };
            offsets[slot * (states + 1) + target + 1] += 1;
        }
    }
    for slot in 0..width {
        let base = slot * (states + 1);
        for target in 0..states {
            offsets[base + target + 1] += offsets[base + target];
        }
    }
    let mut cursors = offsets.clone();
    let mut reverse_sources = vec![0u32; width * states];
    for source in 0..states {
        for slot in 0..width {
            let target = if source == dead {
                dead
            } else {
                let target = transitions[source * width + slot];
                if target == NO_RESIDUAL_PAIR { dead } else { target as usize }
            };
            let base = slot * (states + 1);
            let position = cursors[base + target];
            reverse_sources[slot * states + position] = source as u32;
            cursors[base + target] += 1;
        }
    }
    drop(cursors);

    let mut nonaccepting = Vec::with_capacity(states);
    let mut accepting = Vec::new();
    for state in 0..states {
        if state < pairs && outputs[state] {
            accepting.push(state as u32);
        } else {
            nonaccepting.push(state as u32);
        }
    }
    let mut blocks = Vec::<Vec<u32>>::new();
    if !nonaccepting.is_empty() {
        blocks.push(nonaccepting);
    }
    if !accepting.is_empty() {
        blocks.push(accepting);
    }
    let mut block_of = vec![0usize; states];
    for (block, members) in blocks.iter().enumerate() {
        for &state in members {
            block_of[state as usize] = block;
        }
    }

    let mut in_work = vec![vec![false; width]; blocks.len()];
    let mut worklist = VecDeque::<(usize, usize)>::new();
    let seed = if blocks.len() == 2 && blocks[1].len() < blocks[0].len() {
        1
    } else {
        0
    };
    for slot in 0..width {
        enqueue_sparse_splitter(&mut worklist, &mut in_work, seed, slot);
    }

    let mut marked = vec![0usize; states];
    let mut touched_epoch = vec![0usize; blocks.len()];
    let mut epoch = 0usize;
    let mut processed_splitters = 0usize;
    while let Some((splitter, slot)) = worklist.pop_front() {
        if splitter >= blocks.len() || blocks[splitter].is_empty() {
            continue;
        }
        in_work[splitter][slot] = false;
        processed_splitters += 1;
        epoch += 1;
        if epoch == usize::MAX {
            marked.fill(0);
            touched_epoch.fill(0);
            epoch = 1;
        }
        let mut touched = Vec::new();
        let offset_base = slot * (states + 1);
        for &target in &blocks[splitter] {
            let start = offsets[offset_base + target as usize];
            let end = offsets[offset_base + target as usize + 1];
            for edge in start..end {
                let source = reverse_sources[slot * states + edge] as usize;
                if marked[source] == epoch {
                    continue;
                }
                marked[source] = epoch;
                let block = block_of[source];
                if touched_epoch[block] != epoch {
                    touched_epoch[block] = epoch;
                    touched.push(block);
                }
            }
        }

        for block in touched {
            let old = std::mem::take(&mut blocks[block]);
            let mut inside = Vec::new();
            let mut outside = Vec::new();
            for state in old {
                if marked[state as usize] == epoch {
                    inside.push(state);
                } else {
                    outside.push(state);
                }
            }
            if inside.is_empty() || outside.is_empty() {
                blocks[block] = if inside.is_empty() { outside } else { inside };
                continue;
            }
            // Keep the larger half under its old ID. This minimizes block_of
            // writes; the smaller half receives the fresh ID.
            let (keep, split) = if inside.len() >= outside.len() {
                (inside, outside)
            } else {
                (outside, inside)
            };
            blocks[block] = keep;
            let split_id = blocks.len();
            for &state in &split {
                block_of[state as usize] = split_id;
            }
            blocks.push(split);
            in_work.push(vec![false; width]);
            touched_epoch.push(0);
            for transition in 0..width {
                if in_work[block][transition] {
                    enqueue_sparse_splitter(&mut worklist, &mut in_work, split_id, transition);
                } else {
                    let smaller = if blocks[block].len() <= blocks[split_id].len() {
                        block
                    } else {
                        split_id
                    };
                    enqueue_sparse_splitter(&mut worklist, &mut in_work, smaller, transition);
                }
            }
        }
    }

    let dead_block = block_of[dead];
    let mut remap = vec![u32::MAX; blocks.len()];
    remap[dead_block] = DEAD_RESIDUAL_BLOCK;
    let mut next_block = DEAD_RESIDUAL_BLOCK + 1;
    for block in 0..blocks.len() {
        if block != dead_block {
            remap[block] = next_block;
            next_block += 1;
        }
    }
    (
        (0..pairs)
            .map(|pair| remap[block_of[pair]])
            .collect(),
        processed_splitters,
    )
}

fn enqueue_sparse_splitter(
    worklist: &mut VecDeque<(usize, usize)>,
    in_work: &mut [Vec<bool>],
    block: usize,
    slot: usize,
) {
    if !in_work[block][slot] {
        in_work[block][slot] = true;
        worklist.push_back((block, slot));
    }
}

fn initial_sparse_terminal_blocks(outputs: &[bool]) -> Vec<u32> {
    outputs.iter().map(|&accepting| u32::from(accepting)).collect()
}

/// One exact finite-depth residual refinement. Equal block IDs mean that the
/// terminal-restricted residuals agree through the current byte depth.
fn refine_sparse_terminal_blocks(
    outputs: &[bool],
    transitions: &[u32],
    width: usize,
    blocks: &[u32],
) -> Vec<u32> {
    let mut buckets = FxHashMap::<u64, Vec<(usize, u32)>>::default();
    let mut next = vec![DEAD_RESIDUAL_BLOCK; outputs.len()];
    let mut next_block = DEAD_RESIDUAL_BLOCK;
    for pair in 0..outputs.len() {
        let base = pair * width;
        let mut is_dead = !outputs[pair];
        let mut hash = if outputs[pair] {
            0x517c_c1b7_2722_0a95u64
        } else {
            0x6d0f_27bd_a2f3_11e9u64
        };
        for slot in 0..width {
            let target = transitions[base + slot];
            let successor = if target == NO_RESIDUAL_PAIR {
                DEAD_RESIDUAL_BLOCK
            } else {
                blocks[target as usize]
            };
            is_dead &= successor == DEAD_RESIDUAL_BLOCK;
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(7)
                ^ (successor as u64).wrapping_add(0x94d0_49bb_1331_11eb);
        }
        if is_dead {
            next[pair] = DEAD_RESIDUAL_BLOCK;
            continue;
        }
        let candidates = buckets.entry(hash).or_default();
        let mut found = None;
        for &(candidate, block) in candidates.iter() {
            if sparse_residual_blocks_equal(
                pair,
                candidate,
                outputs,
                transitions,
                width,
                blocks,
            ) {
                found = Some(block);
                break;
            }
        }
        let block = found.unwrap_or_else(|| {
            next_block += 1;
            candidates.push((pair, next_block));
            next_block
        });
        next[pair] = block;
    }
    next
}

fn sparse_residual_blocks_equal(
    left: usize,
    right: usize,
    outputs: &[bool],
    transitions: &[u32],
    width: usize,
    blocks: &[u32],
) -> bool {
    if outputs[left] != outputs[right] {
        return false;
    }
    let left_base = left * width;
    let right_base = right * width;
    for slot in 0..width {
        let left_target = transitions[left_base + slot];
        let right_target = transitions[right_base + slot];
        let left_block = if left_target == NO_RESIDUAL_PAIR {
            DEAD_RESIDUAL_BLOCK
        } else {
            blocks[left_target as usize]
        };
        let right_block = if right_target == NO_RESIDUAL_PAIR {
            DEAD_RESIDUAL_BLOCK
        } else {
            blocks[right_target as usize]
        };
        if left_block != right_block {
            return false;
        }
    }
    true
}

#[cfg(test)]
fn minimize_sparse_terminal_residuals_iterative(
    outputs: &[bool],
    transitions: &[u32],
    width: usize,
) -> (Vec<u32>, usize) {
    let mut blocks = initial_sparse_terminal_blocks(outputs);
    let mut rounds = 0usize;
    loop {
        rounds += 1;
        let next = refine_sparse_terminal_blocks(outputs, transitions, width, &blocks);
        if next == blocks {
            return (blocks, rounds);
        }
        blocks = next;
    }
}

#[derive(Clone, Debug)]
struct SwapTransport { class_map: Vec<u32> }

#[derive(Clone, Debug)]
struct SubsumptionTransport {
    /// Concrete representative lexer state → concrete member lexer states.
    /// Extra representative behaviours are allowed and therefore map to an
    /// empty set. This is the direct TSID transport used after weights are
    /// lifted back to original lexer-state IDs.
    representative_state_to_members: Vec<Vec<u32>>,
}

#[derive(Clone, Debug)]
struct TerminalResidualRefinement {
    blocks: Vec<u32>,
    surviving_pairs: Vec<(TerminalID, TerminalID)>,
    rounds: usize,
    fully_refined: bool,
}

/// The exact unrooted signature set from the terminal-interchangeability
/// definition. `rows` contains each distinct complete state row once; state
/// identity and multiplicity have deliberately been discarded.
#[derive(Clone, Debug)]
struct TerminalSigmaRows {
    terminal_count: usize,
    rows: Vec<Vec<u32>>,
    representative_class: Vec<u32>,
    class_to_row: Vec<Option<usize>>,
    rows_by_hash: FxHashMap<u64, Vec<usize>>,
}

/// Sparse canonical terminal residuals. `block_for(t, s)` is exactly
/// σ_A(s,t), with block zero reserved for the empty residual. The storage is
/// sparse in the terminal/state dimension: literal terminals contribute only
/// their live scanner states instead of one cell for every joint lexer state.
#[derive(Clone, Debug)]
struct SparseTerminalResiduals {
    state_count: usize,
    terminal_count: usize,
    state_blocks_by_terminal: Vec<Vec<(u32, u32)>>,
    rows_by_state: Vec<Vec<(TerminalID, u32)>>,
    full_row_hashes: Vec<u64>,
    states_by_full_row_hash: FxHashMap<u64, Vec<u32>>,
    inventories_by_terminal: Vec<Vec<u32>>,
    pair_count: usize,
    block_count: usize,
}

#[derive(Clone, Debug)]
struct SwapGenerator {
    representative: TerminalID,
    member: TerminalID,
    terminal_map: Vec<TerminalID>,
    class_map: Vec<u32>,
}

#[derive(Clone, Debug)]
struct SubsumptionGenerator {
    representative: TerminalID,
    member: TerminalID,
    transport: SubsumptionTransport,
}

#[derive(Clone, Debug)]
struct GroupElement { terminal_map: Vec<TerminalID>, class_map: Vec<u32> }

impl GroupElement {
    fn identity(num_terminals: usize, classes: usize) -> Self {
        Self {
            terminal_map: (0..num_terminals as u32).collect(),
            class_map: (0..classes as u32).collect(),
        }
    }

    fn compose_right(&self, swap: &SwapGenerator) -> Self {
        Self {
            terminal_map: self.terminal_map.iter().map(|&t| swap.terminal_map[t as usize]).collect(),
            class_map: self.class_map.iter().map(|&s| swap.class_map[s as usize]).collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TerminalInterchangeabilityProfile {
    pub(crate) active_terminals: usize,
    pub(crate) equivalence_classes: usize,
    pub(crate) inactive_members: usize,
    pub(crate) row_classes: usize,
    pub(crate) swap_generators: usize,
    pub(crate) subsumption_generators: usize,
    pub(crate) group_elements: usize,
    pub(crate) concrete_tsids_before: usize,
    pub(crate) concrete_tsids_after: usize,
    pub(crate) expanded_transition_copies: usize,
    pub(crate) initial_substitutions_applied: usize,
    pub(crate) initial_substitutions_missing: usize,
    pub(crate) continuation_initial_moved: usize,
    pub(crate) terminal_sigma_ms: f64,
    pub(crate) terminal_sigma_classes: usize,
    pub(crate) terminal_sigma_rounds: usize,
    pub(crate) terminal_sigma_pruned_early: bool,
    pub(crate) terminal_sigma_survivors: usize,
    pub(crate) weight_remap_ms: f64,
    pub(crate) expansion_ms: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct TerminalInterchangeability {
    original_active: Vec<bool>,
    active_representatives: Vec<bool>,
    members_by_representative: Vec<Vec<TerminalID>>,
    row_machine: Option<RowMachine>,
    generators: Vec<SwapGenerator>,
    subsumption_generators: Vec<SubsumptionGenerator>,
    original_state_count: usize,
    debug_target_state: Option<u32>,
    debug_token_id: Option<u32>,
    profile: TerminalInterchangeabilityProfile,
}

impl TerminalInterchangeability {
    pub(crate) fn identity(active: &[bool]) -> Self {
        let n = active.len();
        let count = active.iter().filter(|&&v| v).count();
        Self {
            original_active: active.to_vec(),
            active_representatives: active.to_vec(),
            members_by_representative: (0..n as u32).map(|t| vec![t]).collect(),
            row_machine: None,
            generators: Vec::new(),
            subsumption_generators: Vec::new(),
            original_state_count: 0,
            debug_target_state: None,
            debug_token_id: None,
            profile: TerminalInterchangeabilityProfile {
                active_terminals: count,
                equivalence_classes: count,
                ..TerminalInterchangeabilityProfile::default()
            },
        }
    }

    pub(crate) fn active_representatives(&self) -> &[bool] { &self.active_representatives }
    pub(crate) fn is_identity(&self) -> bool {
        self.generators.is_empty() && self.subsumption_generators.is_empty()
    }
    pub(crate) fn profile(&self) -> TerminalInterchangeabilityProfile { self.profile.clone() }

    pub(crate) fn nontrivial_classes(&self) -> Vec<Vec<TerminalID>> {
        self.members_by_representative
            .iter()
            .filter(|members| members.len() > 1)
            .cloned()
            .collect()
    }

    pub(crate) fn build(
        tokenizer: &Tokenizer,
        active: &[bool],
        ignore_terminal: Option<TerminalID>,
        relevant_bytes: &[bool; 256],
    ) -> Self {
        if std::env::var_os("GLRMASK_L2P_TERMINAL_SUBSUMPTION").is_some() {
            return Self::build_subsumption(tokenizer, active, ignore_terminal, relevant_bytes);
        }
        let active_ids: Vec<TerminalID> = active.iter().enumerate()
            .filter_map(|(t, &yes)| yes.then_some(t as TerminalID))
            .collect();
        if active_ids.len() < 2 {
            return Self::identity(active);
        }
        let total_started_at = Instant::now();
        if std::env::var_os("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_SPARSE_PREFILTER_ONLY").is_some() {
            let sparse_started_at = Instant::now();
            let sparse = SparseTerminalResiduals::build(tokenizer, relevant_bytes);
            let nonempty_rows = sparse
                .rows_by_state
                .iter()
                .filter(|row| !row.is_empty())
                .count();
            let active_ids = active
                .iter()
                .enumerate()
                .filter_map(|(terminal, &is_active)| is_active.then_some(terminal as TerminalID))
                .filter(|&terminal| Some(terminal) != ignore_terminal)
                .collect::<Vec<_>>();
            let mut inventory_candidates = 0usize;
            let mut exact_subsumption_edges = 0usize;
            let mut members_by_representative = vec![1usize; sparse.terminal_count];
            for &member in &active_ids {
                for &representative in &active_ids {
                    if member != representative
                        && sparse.inventory_is_subset_of(member, representative)
                    {
                        inventory_candidates += 1;
                        if std::env::var_os("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_SPARSE_EXACT_ONLY")
                            .is_some()
                            && sparse.subsumption_transport(member, representative).is_some()
                        {
                            exact_subsumption_edges += 1;
                            members_by_representative[representative as usize] += 1;
                        }
                    }
                }
            }
            let max_members_for_representative = members_by_representative.into_iter().max().unwrap_or(1);
            eprintln!(
                "[glrmask/profile][l2p_terminal_sparse_residuals] terminals={} raw_states={} residual_pairs={} residual_blocks={} nonempty_rows={} inventory_candidates={} exact_checked={} exact_subsumption_edges={} max_members_for_representative={} total_ms={:.3}",
                sparse.terminal_count,
                sparse.state_count,
                sparse.pair_count,
                sparse.block_count,
                nonempty_rows,
                inventory_candidates,
                std::env::var_os("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_SPARSE_EXACT_ONLY").is_some(),
                exact_subsumption_edges,
                max_members_for_representative,
                sparse_started_at.elapsed().as_secs_f64() * 1000.0,
            );
            return Self::identity(active);
        }
        let row_machine_started_at = Instant::now();
        // `relevant_bytes` is the global vocabulary byte universe. A terminal
        // can be continued by bytes carried by a later vocabulary partition,
        // so partition-local bytes are not sound here.
        let row_bytes = relevant_bytes;
        let machine = RowMachine::build(tokenizer, row_bytes);
        let row_machine_ms = row_machine_started_at.elapsed().as_secs_f64() * 1000.0;
        let debug_target_state = debug_state_after_env_prefix(tokenizer);
        let debug_token_id = std::env::var("GLRMASK_DEBUG_TOKEN_ID")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok());
        if let Some(state) = debug_target_state {
            eprintln!(
                "[glrmask/debug][terminal_interchangeability_prefix] state={} row_class={} token_id={:?}",
                state,
                machine.class_for_state[state as usize],
                debug_token_id,
            );
        }
        let candidates_started_at = Instant::now();
        let candidate_groups = terminal_candidate_groups(&machine, &active_ids, ignore_terminal);
        let candidate_pairs = candidate_groups
            .iter()
            .flat_map(|group| {
                group.iter().enumerate().flat_map(move |(index, &left)| {
                    group[index + 1..]
                        .iter()
                        .copied()
                        .map(move |right| (left, right))
                })
            })
            .collect::<Vec<_>>();
        let candidate_pair_count = candidate_pairs.len();
        let candidate_group_ms = candidates_started_at.elapsed().as_secs_f64() * 1000.0;
        let subsumption_candidates = std::env::var_os("GLRMASK_PROFILE_L2P_TERMINAL_SUBSUMPTION")
            .is_some()
            .then(|| subsumption_output_prefilter(&machine, &active_ids, ignore_terminal));
        if std::env::var_os("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_PREFILTER_ONLY").is_some() {
            eprintln!(
                "[glrmask/profile][l2p_terminal_interchangeability_prefilter] active_terminals={} active_bytes={} row_classes={} row_symbols={} candidate_groups={} candidate_pairs={} subsumption_candidates={} row_machine_ms={:.3} candidate_group_ms={:.3}",
                active_ids.len(),
                row_bytes.iter().filter(|&&byte| byte).count(),
                machine.class_count(),
                machine.width,
                candidate_groups.len(),
                candidate_pair_count,
                subsumption_candidates.as_ref().map_or(0, Vec::len),
                row_machine_ms,
                candidate_group_ms,
            );
            return Self::identity(active);
        }
        let sigma_started_at = Instant::now();
        let refinement = minimize_terminal_residuals_with_swap_pruning(&machine, &candidate_pairs);
        let terminal_sigma_ms = sigma_started_at.elapsed().as_secs_f64() * 1000.0;
        let terminal_sigma_classes = refinement
            .blocks
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let sigma_rows = TerminalSigmaRows::from_residual_blocks(&machine, &refinement.blocks);
        let profile_discovery = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        if profile_discovery {
            eprintln!(
                "[glrmask/profile][l2p_terminal_interchangeability_discovery_start] active_terminals={} active_bytes={} row_classes={} candidate_groups={} candidate_pairs={} row_machine_ms={:.3} terminal_sigma_ms={:.3} terminal_sigma_classes={} terminal_sigma_rounds={} terminal_sigma_pruned_early={} terminal_sigma_survivors={} candidate_group_ms={:.3}",
                active_ids.len(),
                row_bytes.iter().filter(|&&byte| byte).count(),
                machine.class_count(),
                candidate_groups.len(),
                candidate_pair_count,
                row_machine_ms,
                terminal_sigma_ms,
                terminal_sigma_classes,
                refinement.rounds,
                !refinement.fully_refined,
                refinement.surviving_pairs.len(),
                candidate_group_ms,
            );
        }
        let swap_checks_started_at = Instant::now();
        let mut passed = FxHashMap::<(TerminalID, TerminalID), SwapTransport>::default();
        for (left, right) in &refinement.surviving_pairs {
            if let Some(transport) = sigma_rows.swap_transport(&machine, *left, *right) {
                passed.insert((*left, *right), transport);
            }
        }

        let swap_checks_ms = swap_checks_started_at.elapsed().as_secs_f64() * 1000.0;
        if let Some(candidates) = subsumption_candidates {
            let subsumption_started_at = Instant::now();
            let eligible = active_ids
                .iter()
                .copied()
                .filter(|&terminal| Some(terminal) != ignore_terminal)
                .collect::<Vec<_>>();
            let mut directed_edges = 0usize;
            let mut max_members_for_rep = 1usize;
            if std::env::var_os("GLRMASK_PROFILE_L2P_TERMINAL_SUBSUMPTION_EXACT").is_some() {
                let mut members_by_representative = vec![1usize; machine.terminal_count];
                for (member, representative) in &candidates {
                    if subsumption_transport_pairwise(&machine, *member, *representative).is_some() {
                        directed_edges += 1;
                        members_by_representative[*representative as usize] += 1;
                    }
                }
                max_members_for_rep = members_by_representative.into_iter().max().unwrap_or(1);
            }
            eprintln!(
                "[glrmask/profile][l2p_terminal_subsumption] active_terminals={} output_candidates={} exact_checked={} directed_edges={} max_members_for_rep={} total_ms={:.3}",
                eligible.len(),
                candidates.len(),
                std::env::var_os("GLRMASK_PROFILE_L2P_TERMINAL_SUBSUMPTION_EXACT").is_some(),
                directed_edges,
                max_members_for_rep,
                subsumption_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if profile_discovery {
            eprintln!(
                "[glrmask/profile][l2p_terminal_interchangeability_discovery] active_terminals={} active_bytes={} row_classes={} candidate_groups={} candidate_pairs={} accepted_swaps={} row_machine_ms={:.3} terminal_sigma_ms={:.3} terminal_sigma_classes={} terminal_sigma_rounds={} terminal_sigma_pruned_early={} terminal_sigma_survivors={} candidate_group_ms={:.3} swap_checks_ms={:.3} total_ms={:.3}",
                active_ids.len(),
                row_bytes.iter().filter(|&&byte| byte).count(),
                machine.class_count(),
                candidate_groups.len(),
                candidate_pair_count,
                passed.len(),
                row_machine_ms,
                terminal_sigma_ms,
                terminal_sigma_classes,
                refinement.rounds,
                !refinement.fully_refined,
                refinement.surviving_pairs.len(),
                candidate_group_ms,
                swap_checks_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let n = active.len();
        let mut dsu = DisjointSet::new(n);
        for &(left, right) in passed.keys() {
            dsu.union(left as usize, right as usize);
        }
        let mut components = BTreeMap::<usize, Vec<TerminalID>>::new();
        for &terminal in &active_ids {
            if Some(terminal) != ignore_terminal {
                let root = dsu.find(terminal as usize);
                components.entry(root).or_default().push(terminal);
            }
        }

        let mut representatives = vec![false; n];
        let mut members = (0..n as u32).map(|t| vec![t]).collect::<Vec<_>>();
        let mut generators = Vec::new();
        for component in components.values_mut() {
            component.sort_unstable();
            let representative = component[0];
            representatives[representative as usize] = true;
            members[representative as usize] = component.clone();
            for &member in component.iter().skip(1) {
                let pair = (representative.min(member), representative.max(member));
                let transport = passed
                    .get(&pair)
                    .cloned()
                    .expect("equivalence component lacks representative transport");
                generators.push(SwapGenerator {
                    representative,
                    member,
                    terminal_map: terminal_swap_map(n, representative, member),
                    class_map: transport.class_map,
                });
            }
        }
        if let Some(ignore) = ignore_terminal {
            if active.get(ignore as usize).copied().unwrap_or(false) {
                representatives[ignore as usize] = true;
                members[ignore as usize] = vec![ignore];
            }
        }
        // Every non-ignore active terminal has already been placed in a
        // component.  Only component representatives remain active; members
        // are intentionally hidden during the quotient build.
        let class_count = representatives.iter().filter(|&&v| v).count();
        let active_count = active_ids.len();
        let lexer_initial_class = machine.class_for_state[tokenizer.initial_state_id() as usize] as usize;
        let continuation_initial_moved = generators
            .iter()
            .filter(|generator| generator.class_map[lexer_initial_class] as usize != lexer_initial_class)
            .count();
        Self {
            original_active: active.to_vec(),
            active_representatives: representatives,
            members_by_representative: members,
            row_machine: Some(machine.clone()),
            generators,
            subsumption_generators: Vec::new(),
            original_state_count: tokenizer.num_states() as usize,
            debug_target_state,
            debug_token_id,
            profile: TerminalInterchangeabilityProfile {
                active_terminals: active_count,
                equivalence_classes: class_count,
                inactive_members: active_count.saturating_sub(class_count),
                row_classes: machine.class_count(),
                swap_generators: 0,
                continuation_initial_moved,
                terminal_sigma_ms,
                terminal_sigma_classes,
                terminal_sigma_rounds: refinement.rounds,
                terminal_sigma_pruned_early: !refinement.fully_refined,
                terminal_sigma_survivors: refinement.surviving_pairs.len(),
                ..TerminalInterchangeabilityProfile::default()
            },
        }.with_generator_count()
    }

    fn build_subsumption(
        tokenizer: &Tokenizer,
        active: &[bool],
        ignore_terminal: Option<TerminalID>,
        relevant_bytes: &[bool; 256],
    ) -> Self {
        let active_ids = active
            .iter()
            .enumerate()
            .filter_map(|(terminal, &is_active)| is_active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if active_ids.len() < 2 {
            return Self::identity(active);
        }

        let started_at = Instant::now();
        let product = SparseTerminalResidualProduct::build(tokenizer, relevant_bytes);
        if std::env::var_os("GLRMASK_L2P_SUBSUMPTION_INVENTORY_PROBE_ONLY").is_some() {
            let probe_started_at = Instant::now();
            let probe = sparse_inventory_prune(&product, &active_ids);
            eprintln!(
                "[glrmask/profile][l2p_subsumption_inventory_probe] active_terminals={} pairs={} residual_bytes={} candidate_counts={:?} hash_candidate_counts={:?} converged={} probe_ms={:.3}",
                active_ids.len(),
                product.pair_terminals.len(),
                product.width,
                probe.candidate_counts,
                probe.hash_candidate_counts,
                probe.converged,
                probe_started_at.elapsed().as_secs_f64() * 1000.0,
            );
            return Self::identity(active);
        }
        let (blocks, _) = minimize_sparse_terminal_residuals(
            &product.outputs,
            &product.transitions,
            product.width,
        );
        let sparse_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        let sparse = SparseTerminalResiduals::from_product(&product, &blocks);
        let candidates_started_at = Instant::now();
        let mut candidates = Vec::new();
        for &member in &active_ids {
            for &representative in &active_ids {
                if member == representative
                    || !sparse.inventory_is_subset_of(member, representative)
                {
                    continue;
                }
                candidates.push((member, representative));
            }
        }
        let candidate_ms = candidates_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_started_at = Instant::now();
        let mut transports = sparse.subsumption_transports(&candidates);
        let exact_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L2P_SUBSUMPTION_PLAN").is_some() {
            eprintln!(
                "[glrmask/profile][l2p_subsumption_plan] active_terminals={} sparse_pairs={} candidates={} accepted={} sparse_ms={:.3} candidate_ms={:.3} exact_ms={:.3} total_ms={:.3}",
                active_ids.len(),
                sparse.pair_count,
                candidates.len(),
                transports.len(),
                sparse_ms,
                candidate_ms,
                exact_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let mut unassigned = vec![false; active.len()];
        for &terminal in &active_ids {
            unassigned[terminal as usize] = true;
        }
        let mut active_representatives = active.to_vec();
        for &terminal in &active_ids {
            active_representatives[terminal as usize] = false;
        }
        let mut members_by_representative = (0..active.len() as u32)
            .map(|terminal| vec![terminal])
            .collect::<Vec<_>>();
        let mut generators = Vec::new();

        while unassigned.iter().any(|&value| value) {
            let mut best = None::<(usize, TerminalID, Vec<TerminalID>)>;
            for &representative in &active_ids {
                if !unassigned[representative as usize] {
                    continue;
                }
                let mut covered = vec![representative];
                for &member in &active_ids {
                    if member != representative
                        && unassigned[member as usize]
                        && transports.contains_key(&(member, representative))
                    {
                        covered.push(member);
                    }
                }
                covered.sort_unstable();
                let candidate = (covered.len(), representative, covered);
                if best.as_ref().is_none_or(|best| {
                    candidate.0 > best.0 || (candidate.0 == best.0 && candidate.1 < best.1)
                }) {
                    best = Some(candidate);
                }
            }
            let (_, representative, covered) = best.expect("unassigned active terminal lacks self cover");
            active_representatives[representative as usize] = true;
            members_by_representative[representative as usize] = covered.clone();
            for member in covered {
                unassigned[member as usize] = false;
                if member != representative {
                    generators.push(SubsumptionGenerator {
                        representative,
                        member,
                        transport: transports
                            .remove(&(member, representative))
                            .expect("chosen subsumption edge lacks transport"),
                    });
                }
            }
        }
        if let Some(ignore) = ignore_terminal {
            if active.get(ignore as usize).copied().unwrap_or(false) {
                active_representatives[ignore as usize] = true;
                members_by_representative[ignore as usize] = vec![ignore];
            }
        }

        let active_count = active.iter().filter(|&&value| value).count();
        let representative_count = active_representatives.iter().filter(|&&value| value).count();
        Self {
            original_active: active.to_vec(),
            active_representatives,
            members_by_representative,
            row_machine: None,
            generators: Vec::new(),
            subsumption_generators: generators,
            original_state_count: tokenizer.num_states() as usize,
            debug_target_state: None,
            debug_token_id: None,
            profile: TerminalInterchangeabilityProfile {
                active_terminals: active_count,
                equivalence_classes: representative_count,
                inactive_members: active_count.saturating_sub(representative_count),
                row_classes: sparse.block_count,
                terminal_sigma_ms: sparse_ms,
                ..TerminalInterchangeabilityProfile::default()
            },
        }
        .with_generator_count()
    }

    fn with_generator_count(mut self) -> Self {
        self.profile.swap_generators = self.generators.len();
        self.profile.subsumption_generators = self.subsumption_generators.len();
        self
    }
    /// Lift the representative-only DWA to concrete states and return one
    /// separate complete DWA view for the identity plus each
    /// representative/member transposition.  The caller must union these as
    /// *disjoint* NWA branches before determinization; merging equal numeric
    /// DWA state IDs is unsound because the transformed view need not preserve
    /// the original deterministic state decomposition.
    pub(crate) fn expand_reference_dwa_views(
        &self,
        dwa: &mut DWA,
        state_map: &mut ManyToOneIdMap,
    ) -> (Vec<DWA>, TerminalInterchangeabilityProfile) {
        let started = Instant::now();
        let mut profile = self.profile();
        if self.is_identity() {
            return (vec![dwa.clone()], profile);
        }
        assert_eq!(state_map.original_to_internal.len(), self.original_state_count);
        profile.concrete_tsids_before = state_map.num_internal_ids() as usize;
        let remap_started = Instant::now();
        lift_dwa_weights_to_concrete_states(dwa, state_map);
        *state_map = concrete_state_map(self.original_state_count);
        profile.concrete_tsids_after = state_map.num_internal_ids() as usize;
        profile.weight_remap_ms = remap_started.elapsed().as_secs_f64() * 1000.0;

        // Direct semantic oracle: the identity DWA plus one full swapped view
        // per representative/member pair. The optional initial-only path is a
        // candidate shortcut and is not used by default while validating this.
        profile.group_elements = 1 + self.generators.len() + self.subsumption_generators.len();
        let original = dwa.clone();
        if std::env::var_os("GLRMASK_DEBUG_TERMINAL_SUBSUMPTION_DWA").is_some() {
            eprintln!("[glrmask/debug][terminal_subsumption_dwa] start={} states={}", original.start_state(), original.states().len());
            for (state_id, state) in original.states().iter().enumerate() {
                let labels = state.transitions.keys().copied().collect::<Vec<_>>();
                eprintln!("[glrmask/debug][terminal_subsumption_dwa_state] state={} labels={:?}", state_id, labels);
            }
        }
        let full_swap_reference = std::env::var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_REFERENCE_MODE")
            .map(|value| value.trim().eq_ignore_ascii_case("full_swap"))
            .unwrap_or(true);
        let only_member = std::env::var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_ONLY_MEMBER")
            .ok()
            .and_then(|value| value.trim().parse::<TerminalID>().ok());
        debug_assert!(
            self.generators.is_empty() || self.subsumption_generators.is_empty(),
            "swap and directed-subsumption reference expansion are distinct modes",
        );
        let subsumption_base = (!self.subsumption_generators.is_empty()).then(|| {
            expand_noninitial_terminal_labels(
                &original,
                &self.members_by_representative,
                &mut profile,
            )
        });
        let mut views = vec![subsumption_base.clone().unwrap_or_else(|| original.clone())];
        for generator in &self.generators {
            if only_member.is_some_and(|member| member != generator.member) {
                continue;
            }
            let element = GroupElement {
                terminal_map: generator.terminal_map.clone(),
                class_map: generator.class_map.clone(),
            };
            if full_swap_reference {
                views.push(transformed_dwa_view(
                    &original,
                    &element,
                    self.row_machine.as_ref().expect("swap expansion needs a row machine"),
                    &mut profile,
                ));
            } else {
                let base_view = expand_noninitial_terminal_labels(
                    &original,
                    &self.members_by_representative,
                    &mut profile,
                );
                views.push(substitute_initial_terminal(
                    &base_view,
                    generator.representative,
                    generator.member,
                    &generator.class_map,
                    self.row_machine.as_ref().expect("swap expansion needs a row machine"),
                    &mut profile,
                ));
            }
        }
        if let Some(base) = &subsumption_base {
            for generator in &self.subsumption_generators {
                views.push(subsumption_dwa_view(
                    base,
                    generator.representative,
                    generator.member,
                    &generator.transport,
                    &mut profile,
                ));
            }
        }
        *dwa = original;
        profile.expansion_ms = started.elapsed().as_secs_f64() * 1000.0;
        (views, profile)
    }
}

impl RowMachine {
    fn build(tokenizer: &Tokenizer, relevant: &[bool; 256]) -> Self {
        let real_count = tokenizer.num_states() as usize;
        let bytes: Vec<u8> = (0..=255u8).filter(|&b| relevant[b as usize]).collect();
        let width = bytes.len();
        // Complete rows always contain every tokenizer terminal.  The caller's
        // partition mask selects candidates only; it must not erase terminal
        // columns that provide contextual correlation.
        let words = (tokenizer.num_terminals() as usize).div_ceil(64);
        let dead = real_count as u32;
        let total = real_count + 1;
        let mut outputs = Vec::with_capacity(total);
        let mut transitions = vec![dead; total * width];
        for state in 0..real_count {
            let mut output = OutputBits::empty(words);
            for terminal in tokenizer.matched_terminals_iter(state as u32) {
                let index = terminal as usize;
                if index < tokenizer.num_terminals() as usize {
                    output.0[index / 64] |= 1u64 << (index % 64);
                }
            }
            outputs.push(output);
            let row = tokenizer.transition_row(state as u32);
            for (slot, &byte) in bytes.iter().enumerate() {
                let target = row[byte as usize];
                transitions[state * width + slot] = if target == NO_STATE { dead } else { target };
            }
        }
        outputs.push(OutputBits::empty(words));
        for slot in 0..width { transitions[real_count * width + slot] = dead; }

        let class_for_state = minimize_moore(&outputs, &transitions, width);
        let classes = class_for_state.iter().copied().max().map_or(0, |v| v as usize + 1);
        let mut class_outputs = vec![OutputBits::empty(words); classes];
        let mut class_transitions = vec![0; classes * width];
        let mut class_has_real_state = vec![false; classes];
        let mut class_original_states = vec![Vec::new(); classes];
        let mut representatives = vec![None; classes];
        for state in 0..total {
            let class = class_for_state[state] as usize;
            if representatives[class].is_none() {
                representatives[class] = Some(state);
                class_outputs[class] = outputs[state].clone();
            }
            if state < real_count {
                class_has_real_state[class] = true;
                class_original_states[class].push(state as u32);
            }
        }
        for class in 0..classes {
            let state = representatives[class].expect("empty Moore class");
            for slot in 0..width {
                let target = transitions[state * width + slot] as usize;
                class_transitions[class * width + slot] = class_for_state[target];
            }
        }

        let terminal_count = tokenizer.num_terminals() as usize;
        let mut class_terminals = vec![Vec::new(); classes];
        let mut terminal_classes = vec![Vec::new(); terminal_count];
        for class in 0..classes {
            for (word_index, &word) in class_outputs[class].0.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let offset = bits.trailing_zeros() as usize;
                    let terminal = word_index * 64 + offset;
                    if terminal < terminal_count {
                        class_terminals[class].push(terminal);
                        terminal_classes[terminal].push(class);
                    }
                    bits &= bits - 1;
                }
            }
        }
        let mut column_intern = FxHashMap::<Vec<u32>, usize>::default();
        let mut retained_slots = Vec::new();
        for slot in 0..width {
            let column = (0..classes)
                .map(|class| class_transitions[class * width + slot])
                .collect::<Vec<_>>();
            let id = column_intern.len();
            if column_intern.insert(column, id).is_none() {
                retained_slots.push(slot);
            }
        }
        let reduced_width = retained_slots.len();
        let mut reduced_transitions = vec![0; classes * reduced_width];
        for class in 0..classes {
            for (reduced_slot, &slot) in retained_slots.iter().enumerate() {
                reduced_transitions[class * reduced_width + reduced_slot] =
                    class_transitions[class * width + slot];
            }
        }
        Self {
            width: reduced_width,
            terminal_count,
            class_for_state,
            class_outputs,
            class_transitions: reduced_transitions,
            class_terminals,
            terminal_classes,
            class_has_real_state,
            class_original_states,
            real_state_count: real_count,
        }
    }
}


impl TerminalSigmaRows {
    fn build(machine: &RowMachine) -> (Self, usize) {
        let sigma = minimize_terminal_residuals(machine);
        let classes = sigma.iter().copied().max().map_or(0, |class| class as usize + 1);
        (Self::from_residual_blocks(machine, &sigma), classes)
    }

    fn from_residual_blocks(machine: &RowMachine, sigma: &[u32]) -> Self {
        let classes = machine.class_count();
        let terminals = machine.terminal_count;
        let mut rows = Vec::<Vec<u32>>::new();
        let mut representative_class = Vec::<u32>::new();
        let mut class_to_row = vec![None; classes];
        let mut unique_rows = FxHashMap::<Vec<u32>, usize>::default();

        for class in 0..classes {
            if !machine.class_has_real_state[class] {
                continue;
            }
            let row = (0..terminals)
                .map(|terminal| sigma[class * terminals + terminal])
                .collect::<Vec<_>>();
            let row_id = if let Some(&existing) = unique_rows.get(&row) {
                existing
            } else {
                let id = rows.len();
                unique_rows.insert(row.clone(), id);
                representative_class.push(class as u32);
                rows.push(row);
                id
            };
            class_to_row[class] = Some(row_id);
        }

        let mut rows_by_hash = FxHashMap::<u64, Vec<usize>>::default();
        for (row_id, row) in rows.iter().enumerate() {
            rows_by_hash.entry(sigma_row_hash(row)).or_default().push(row_id);
        }
        Self {
            terminal_count: terminals,
            rows,
            representative_class,
            class_to_row,
            rows_by_hash,
        }
    }

    /// Exact set-of-rows membership test, without choosing a state transport.
    /// This is also sound on bounded-depth residual approximations.
    fn has_swap_row_set(&self, left: TerminalID, right: TerminalID) -> bool {
        let left = left as usize;
        let right = right as usize;
        if left >= self.terminal_count || right >= self.terminal_count {
            return false;
        }
        self.rows.iter().all(|row| {
            let hash = sigma_swapped_row_hash(row, left, right);
            self.rows_by_hash.get(&hash).is_some_and(|candidates| {
                candidates.iter().any(|&candidate| {
                    sigma_row_equals_swapped(row, &self.rows[candidate], left, right)
                })
            })
        })
    }

    /// Exact set-of-rows swap test from the specification. The returned class
    /// map is the induced map on concrete minimized lexer-row classes; it is
    /// intentionally set-valued again when expanded through
    /// `class_original_states` during TSID transport.
    fn swap_transport(&self, machine: &RowMachine, left: TerminalID, right: TerminalID) -> Option<SwapTransport> {
        let left = left as usize;
        let right = right as usize;
        if left >= self.terminal_count || right >= self.terminal_count {
            return None;
        }
        let mut target_row_for_source = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let swapped_hash = sigma_swapped_row_hash(row, left, right);
            let candidates = self.rows_by_hash.get(&swapped_hash)?;
            let target = candidates.iter().copied().find(|&candidate| {
                sigma_row_equals_swapped(row, &self.rows[candidate], left, right)
            })?;
            target_row_for_source.push(target);
        }

        let mut class_map = (0..machine.class_count() as u32).collect::<Vec<_>>();
        for (class, row_id) in self.class_to_row.iter().enumerate() {
            let Some(source_row) = *row_id else { continue; };
            let target_row = target_row_for_source[source_row];
            class_map[class] = self.representative_class[target_row];
        }

        // On the duplicate-collapsed row set, a transposition is an involution.
        // This also catches any accidental representative mismatch before TSID
        // transport consumes the map.
        for (class, row_id) in self.class_to_row.iter().enumerate() {
            let Some(_) = row_id else { continue; };
            let target = class_map[class] as usize;
            if target >= class_map.len() || class_map[target] != class as u32 {
                return None;
            }
        }
        Some(SwapTransport { class_map })
    }

    /// Decide the directed relation `member ⪯ representative` directly from
    /// the unrooted set of complete rows. The returned relation is the inverse
    /// of the existential witness direction: it maps each representative class
    /// to every concrete member class that it can represent in a copied DWA
    /// view.
    fn subsumption_transport(
        &self,
        machine: &RowMachine,
        member: TerminalID,
        representative: TerminalID,
    ) -> Option<SubsumptionTransport> {
        let member = member as usize;
        let representative = representative as usize;
        if member >= self.terminal_count || representative >= self.terminal_count {
            return None;
        }
        if member == representative {
            let mut identity = vec![Vec::new(); machine.real_state_count];
            for state in 0..machine.real_state_count {
                identity[state].push(state as u32);
            }
            return Some(SubsumptionTransport { representative_state_to_members: identity });
        }

        // RHS rows omit the member column.  Index them once for this ordered
        // pair. A candidate hash is always followed by the exact row check.
        let mut rhs_by_hash = FxHashMap::<u64, Vec<usize>>::default();
        for (row_id, row) in self.rows.iter().enumerate() {
            rhs_by_hash
                .entry(sigma_row_hash(row) ^ sigma_mix(member, row[member]))
                .or_default()
                .push(row_id);
        }

        let mut representative_state_to_members = vec![Vec::<u32>::new(); machine.real_state_count];
        for (source_row_id, source) in self.rows.iter().enumerate() {
            let lhs_hash = sigma_member_to_representative_hash(source, member, representative);
            let candidates = rhs_by_hash.get(&lhs_hash)?;
            let mut matched = false;
            for &target_row_id in candidates {
                let target = &self.rows[target_row_id];
                if !sigma_row_member_subsumed_by(source, target, member, representative) {
                    continue;
                }
                let target_class = self.representative_class[target_row_id] as usize;
                let source_class = self.representative_class[source_row_id] as usize;
                for &target_state in &machine.class_original_states[target_class] {
                    representative_state_to_members[target_state as usize]
                        .extend(machine.class_original_states[source_class].iter().copied());
                }
                matched = true;
            }
            if !matched {
                return None;
            }
        }
        for members in &mut representative_state_to_members {
            members.sort_unstable();
            members.dedup();
        }
        Some(SubsumptionTransport { representative_state_to_members })
    }
}

fn terminal_output_bit(output: &OutputBits, terminal: usize) -> bool {
    output
        .0
        .get(terminal / 64)
        .is_some_and(|word| word & (1u64 << (terminal % 64)) != 0)
}

/// Simultaneously minimize the DFA family `(state, terminal)` where the
/// accepting bit says whether that terminal matches at the state. Equal result
/// IDs are exactly equal terminal-restricted residual languages.
fn minimize_terminal_residuals(machine: &RowMachine) -> Vec<u32> {
    let classes = machine.class_count();
    let terminals = machine.terminal_count;
    let total = classes * terminals;
    let mut blocks = vec![0u32; total];
    for class in 0..classes {
        for terminal in 0..terminals {
            blocks[class * terminals + terminal] =
                u32::from(terminal_output_bit(&machine.class_outputs[class], terminal));
        }
    }

    loop {
        // Hashes avoid allocating a width-sized successor vector for every
        // `(class, terminal)` product state. They are never trusted as proof:
        // every candidate bucket is checked slot-by-slot before interning.
        let mut intern = FxHashMap::<u64, Vec<(u32, usize)>>::default();
        let mut next = vec![0u32; total];
        let mut next_block_id = 0u32;
        for class in 0..classes {
            for terminal in 0..terminals {
                let product = class * terminals + terminal;
                let hash = terminal_residual_key_hash(machine, &blocks, class, terminal);
                let bucket = intern.entry(hash).or_default();
                let mut found = None;
                for &(block, exemplar) in bucket.iter() {
                    if terminal_residual_keys_equal(
                        machine,
                        &blocks,
                        product,
                        exemplar,
                    ) {
                        found = Some(block);
                        break;
                    }
                }
                let block = found.unwrap_or_else(|| {
                    let block = next_block_id;
                    next_block_id += 1;
                    bucket.push((block, product));
                    block
                });
                next[product] = block;
            }
        }
        if next == blocks {
            return blocks;
        }
        blocks = next;
    }
}

fn refine_terminal_residual_blocks(machine: &RowMachine, blocks: &[u32]) -> Vec<u32> {
    let classes = machine.class_count();
    let terminals = machine.terminal_count;
    let total = classes * terminals;
    let mut intern = FxHashMap::<u64, Vec<(u32, usize)>>::default();
    let mut next = vec![0u32; total];
    let mut next_block_id = 0u32;
    for class in 0..classes {
        for terminal in 0..terminals {
            let product = class * terminals + terminal;
            let hash = terminal_residual_key_hash(machine, blocks, class, terminal);
            let bucket = intern.entry(hash).or_default();
            let mut found = None;
            for &(block, exemplar) in bucket.iter() {
                if terminal_residual_keys_equal(machine, blocks, product, exemplar) {
                    found = Some(block);
                    break;
                }
            }
            let block = found.unwrap_or_else(|| {
                let block = next_block_id;
                next_block_id += 1;
                bucket.push((block, product));
                block
            });
            next[product] = block;
        }
    }
    next
}

/// Eliminate swaps as soon as they fail a finite residual approximation. This
/// cannot reject a real exact swap: exact residual equality implies equality at
/// every finite refinement depth.
fn minimize_terminal_residuals_with_swap_pruning(
    machine: &RowMachine,
    candidate_pairs: &[(TerminalID, TerminalID)],
) -> TerminalResidualRefinement {
    let classes = machine.class_count();
    let terminals = machine.terminal_count;
    let mut blocks = vec![0u32; classes * terminals];
    for class in 0..classes {
        for terminal in 0..terminals {
            blocks[class * terminals + terminal] =
                u32::from(terminal_output_bit(&machine.class_outputs[class], terminal));
        }
    }
    let mut surviving_pairs = candidate_pairs.to_vec();
    let mut rounds = 0usize;
    loop {
        let rows = TerminalSigmaRows::from_residual_blocks(machine, &blocks);
        surviving_pairs.retain(|&(left, right)| rows.has_swap_row_set(left, right));
        if surviving_pairs.is_empty() {
            return TerminalResidualRefinement {
                blocks,
                surviving_pairs,
                rounds,
                fully_refined: false,
            };
        }
        let next = refine_terminal_residual_blocks(machine, &blocks);
        rounds += 1;
        if next == blocks {
            return TerminalResidualRefinement {
                blocks,
                surviving_pairs,
                rounds,
                fully_refined: true,
            };
        }
        blocks = next;
    }
}

fn terminal_residual_key_hash(
    machine: &RowMachine,
    blocks: &[u32],
    class: usize,
    terminal: usize,
) -> u64 {
    let mut hash = if terminal_output_bit(&machine.class_outputs[class], terminal) {
        0x517c_c1b7_2722_0a95u64
    } else {
        0x6d0f_27bd_a2f3_11e9u64
    };
    let base = class * machine.width;
    for slot in 0..machine.width {
        let target = machine.class_transitions[base + slot] as usize;
        let successor = blocks[target * machine.terminal_count + terminal] as u64;
        hash = hash
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(7)
            ^ successor.wrapping_add(0x94d0_49bb_1331_11eb);
    }
    hash
}

fn terminal_residual_keys_equal(
    machine: &RowMachine,
    blocks: &[u32],
    left_product: usize,
    right_product: usize,
) -> bool {
    let terminals = machine.terminal_count;
    let left_class = left_product / terminals;
    let left_terminal = left_product % terminals;
    let right_class = right_product / terminals;
    let right_terminal = right_product % terminals;
    if terminal_output_bit(&machine.class_outputs[left_class], left_terminal)
        != terminal_output_bit(&machine.class_outputs[right_class], right_terminal)
    {
        return false;
    }
    let left_base = left_class * machine.width;
    let right_base = right_class * machine.width;
    for slot in 0..machine.width {
        let left_target = machine.class_transitions[left_base + slot] as usize;
        let right_target = machine.class_transitions[right_base + slot] as usize;
        if blocks[left_target * terminals + left_terminal]
            != blocks[right_target * terminals + right_terminal]
        {
            return false;
        }
    }
    true
}

fn sigma_mix(position: usize, value: u32) -> u64 {
    let mut z = ((position as u64) << 32) ^ value as u64 ^ 0x9e37_79b9_7f4a_7c15;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn sigma_row_hash(row: &[u32]) -> u64 {
    row.iter().enumerate().fold(0u64, |hash, (position, &value)| {
        hash ^ sigma_mix(position, value)
    })
}

fn sigma_swapped_row_hash(row: &[u32], left: usize, right: usize) -> u64 {
    if left == right {
        return sigma_row_hash(row);
    }
    sigma_row_hash(row)
        ^ sigma_mix(left, row[left])
        ^ sigma_mix(right, row[right])
        ^ sigma_mix(left, row[right])
        ^ sigma_mix(right, row[left])
}

fn sigma_row_equals_swapped(source: &[u32], target: &[u32], left: usize, right: usize) -> bool {
    if target[left] != source[right] || target[right] != source[left] {
        return false;
    }
    source.iter().enumerate().all(|(terminal, &value)| {
        terminal == left || terminal == right || target[terminal] == value
    })
}

fn sigma_member_to_representative_hash(row: &[u32], member: usize, representative: usize) -> u64 {
    sigma_row_hash(row)
        ^ sigma_mix(member, row[member])
        ^ sigma_mix(representative, row[representative])
        ^ sigma_mix(representative, row[member])
}

fn sigma_row_member_subsumed_by(
    member_row: &[u32],
    representative_row: &[u32],
    member: usize,
    representative: usize,
) -> bool {
    if representative_row[representative] != member_row[member] {
        return false;
    }
    member_row.iter().enumerate().all(|(terminal, &value)| {
        terminal == member
            || terminal == representative
            || representative_row[terminal] == value
    })
}

fn terminal_output_fingerprint(terminal: usize) -> u128 {
    let high = sigma_mix(terminal, 0x9e37_79b9) as u128;
    let low = sigma_mix(terminal, 0x85eb_ca6b) as u128;
    (high << 64) | low
}

fn output_row_fingerprints(machine: &RowMachine) -> Vec<u128> {
    machine
        .class_terminals
        .iter()
        .map(|terminals| {
            terminals.iter().fold(0u128, |fingerprint, &terminal| {
                fingerprint ^ terminal_output_fingerprint(terminal)
            })
        })
        .collect()
}

/// Safe directed filter for `member ⪯ representative` using only zero-byte
/// residual rows. A fingerprint miss proves that no matching row exists; a
/// collision can only retain an extra candidate for the later exact check.
fn subsumption_output_prefilter(
    machine: &RowMachine,
    active_ids: &[TerminalID],
    ignore: Option<TerminalID>,
) -> Vec<(TerminalID, TerminalID)> {
    let fingerprints = output_row_fingerprints(machine);
    let mut candidates = Vec::new();
    for &member in active_ids {
        if Some(member) == ignore { continue; }
        let member_index = member as usize;
        let member_fingerprint = terminal_output_fingerprint(member_index);
        let mut rhs_rows = FxHashSet::<u128>::default();
        for class in 0..machine.class_count() {
            if !machine.class_has_real_state[class] { continue; }
            let mut fingerprint = fingerprints[class];
            if terminal_output_bit(&machine.class_outputs[class], member_index) {
                fingerprint ^= member_fingerprint;
            }
            rhs_rows.insert(fingerprint);
        }

        for &representative in active_ids {
            if representative == member || Some(representative) == ignore { continue; }
            let representative_index = representative as usize;
            let representative_fingerprint = terminal_output_fingerprint(representative_index);
            let mut possible = true;
            for &class in machine.terminal_classes[member_index]
                .iter()
                .chain(machine.terminal_classes[representative_index].iter())
            {
                if !machine.class_has_real_state[class] { continue; }
                let member_set = terminal_output_bit(&machine.class_outputs[class], member_index);
                let representative_set = terminal_output_bit(
                    &machine.class_outputs[class],
                    representative_index,
                );
                let mut transformed = fingerprints[class];
                if member_set { transformed ^= member_fingerprint; }
                if representative_set { transformed ^= representative_fingerprint; }
                if member_set { transformed ^= representative_fingerprint; }
                if !rhs_rows.contains(&transformed) {
                    possible = false;
                    break;
                }
            }
            if possible {
                candidates.push((member, representative));
            }
        }
    }
    candidates
}

fn initial_blocks(outputs: &[OutputBits]) -> Vec<u32> {
    let mut intern = FxHashMap::<OutputBits, u32>::default();
    outputs.iter().map(|output| {
        let id = intern.len() as u32;
        *intern.entry(output.clone()).or_insert(id)
    }).collect()
}

fn minimize_moore(outputs: &[OutputBits], transitions: &[u32], width: usize) -> Vec<u32> {
    let mut blocks = initial_blocks(outputs);
    loop {
        let mut intern = FxHashMap::<MooreKey, u32>::default();
        let mut next = vec![0; outputs.len()];
        for state in 0..outputs.len() {
            let successors = (0..width)
                .map(|slot| blocks[transitions[state * width + slot] as usize])
                .collect();
            let key = MooreKey { output: outputs[state].clone(), successors };
            let id = intern.len() as u32;
            next[state] = *intern.entry(key).or_insert(id);
        }
        if next == blocks { return blocks; }
        blocks = next;
    }
}

// Color-refine the terminal/output-row incidence graph.  Equal colors are a
// necessary condition for a terminal swap, never a proof.
fn terminal_candidate_groups(
    machine: &RowMachine,
    active_ids: &[TerminalID],
    ignore: Option<TerminalID>,
) -> Vec<Vec<TerminalID>> {
    let classes = machine.class_count();
    let terminals = machine.terminal_count;
    let mut terminal_colors = (0..terminals)
        .map(|terminal| u32::from(Some(terminal as TerminalID) == ignore))
        .collect::<Vec<_>>();
    let mut state_colors = vec![0u32; classes];
    loop {
        let mut incoming = vec![Vec::<u64>::new(); classes];
        for source in 0..classes {
            for slot in 0..machine.width {
                let target = machine.transition(source as u32, slot) as usize;
                incoming[target].push(((slot as u64) << 32) | state_colors[source] as u64);
            }
        }
        for predecessors in &mut incoming {
            predecessors.sort_unstable();
        }

        let mut state_intern = FxHashMap::<Vec<u64>, u32>::default();
        let mut next_states = vec![0u32; classes];
        for class in 0..classes {
            let mut key = Vec::with_capacity(
                3 + machine.width + machine.class_terminals[class].len() + incoming[class].len(),
            );
            key.push(0x1000_0000_0000_0000 | state_colors[class] as u64);
            key.push(0x1000_0000_0000_0000 | u64::from(machine.class_has_real_state[class]));
            for slot in 0..machine.width {
                let target = machine.transition(class as u32, slot) as usize;
                key.push(0x2000_0000_0000_0000 | state_colors[target] as u64);
            }
            key.push(0x3000_0000_0000_0000);
            key.extend(
                machine.class_terminals[class]
                    .iter()
                    .map(|&terminal| 0x3000_0000_0000_0000 | terminal_colors[terminal] as u64),
            );
            let output_start = machine.width + 3;
            key[output_start..].sort_unstable();
            key.push(0x4000_0000_0000_0000);
            key.extend(
                incoming[class]
                    .iter()
                    .map(|&entry| 0x4000_0000_0000_0000 | entry),
            );
            let color = state_intern.len() as u32;
            next_states[class] = *state_intern.entry(key).or_insert(color);
        }

        let mut terminal_intern = FxHashMap::<Vec<u32>, u32>::default();
        let mut next_terminals = vec![0u32; terminals];
        for terminal in 0..terminals {
            let mut key = Vec::with_capacity(1 + machine.terminal_classes[terminal].len());
            key.push(terminal_colors[terminal]);
            key.extend(
                machine.terminal_classes[terminal]
                    .iter()
                    .map(|&class| next_states[class]),
            );
            key[1..].sort_unstable();
            let color = terminal_intern.len() as u32;
            next_terminals[terminal] = *terminal_intern.entry(key).or_insert(color);
        }

        if next_terminals == terminal_colors && next_states == state_colors { break; }
        terminal_colors = next_terminals;
        state_colors = next_states;
    }
    let mut groups = BTreeMap::<u32, Vec<TerminalID>>::new();
    for &terminal in active_ids {
        groups.entry(terminal_colors[terminal as usize]).or_default().push(terminal);
    }
    groups.into_values().collect()
}

// Compare the original row machine with an output-swapped copy.  Minimizing
// their disjoint union computes the exact residual relation, not a hash test.
fn swap_transport(
    machine: &RowMachine,
    left: TerminalID,
    right: TerminalID,
) -> Option<SwapTransport> {
    let left = left as usize;
    let right = right as usize;
    if left / 64 >= machine.class_outputs.first()?.0.len()
        || right / 64 >= machine.class_outputs.first()?.0.len()
    {
        return None;
    }
    let classes = machine.class_count();
    let mut outputs = machine.class_outputs.clone();
    outputs.extend(machine.class_outputs.iter().map(|o| o.swapped(left, right)));
    let mut transitions = vec![0; classes * 2 * machine.width];
    for copy in 0..2usize {
        for class in 0..classes {
            for slot in 0..machine.width {
                let target = machine.transition(class as u32, slot) as usize;
                transitions[(copy * classes + class) * machine.width + slot] =
                    (copy * classes + target) as u32;
            }
        }
    }
    let blocks = minimize_moore(&outputs, &transitions, machine.width);
    let mut swapped_block_to_class = FxHashMap::<u32, u32>::default();
    for class in 0..classes {
        let block = blocks[classes + class];
        if swapped_block_to_class.insert(block, class as u32).is_some() {
            return None;
        }
    }
    let mut class_map = Vec::with_capacity(classes);
    for class in 0..classes {
        let target = *swapped_block_to_class.get(&blocks[class])?;
        if machine.class_has_real_state[class] && !machine.class_has_real_state[target as usize] {
            return None;
        }
        class_map.push(target);
    }
    if class_map.iter().enumerate().any(|(s, &t)| class_map[t as usize] != s as u32) {
        return None;
    }
    Some(SwapTransport { class_map })
}

fn subsumption_transport_pairwise(
    machine: &RowMachine,
    member: TerminalID,
    representative: TerminalID,
) -> Option<SubsumptionTransport> {
    let member = member as usize;
    let representative = representative as usize;
    if member >= machine.terminal_count || representative >= machine.terminal_count {
        return None;
    }
    let classes = machine.class_count();
    let mut outputs = Vec::with_capacity(classes * 2);
    outputs.extend(machine.class_outputs.iter().map(|output| {
        output.member_as_representative(member, representative)
    }));
    outputs.extend(machine.class_outputs.iter().map(|output| output.without(member)));
    let mut transitions = vec![0u32; classes * 2 * machine.width];
    for copy in 0..2usize {
        for class in 0..classes {
            for slot in 0..machine.width {
                let target = machine.transition(class as u32, slot) as usize;
                transitions[(copy * classes + class) * machine.width + slot] =
                    (copy * classes + target) as u32;
            }
        }
    }
    let blocks = minimize_moore(&outputs, &transitions, machine.width);
    let mut representatives_by_block = FxHashMap::<u32, Vec<u32>>::default();
    for class in 0..classes {
        if machine.class_has_real_state[class] {
            representatives_by_block
                .entry(blocks[classes + class])
                .or_default()
                .push(class as u32);
        }
    }
    let mut representative_state_to_members = vec![Vec::new(); machine.real_state_count];
    for class in 0..classes {
        if !machine.class_has_real_state[class] { continue; }
        let block = blocks[class];
        let witnesses = representatives_by_block.get(&block)?;
        for &representative_class in witnesses {
            for &representative_state in &machine.class_original_states[representative_class as usize] {
                representative_state_to_members[representative_state as usize]
                    .extend(machine.class_original_states[class].iter().copied());
            }
        }
    }
    for members in &mut representative_state_to_members {
        members.sort_unstable();
        members.dedup();
    }
    Some(SubsumptionTransport { representative_state_to_members })
}

fn terminal_swap_map(n: usize, left: TerminalID, right: TerminalID) -> Vec<TerminalID> {
    let mut map = (0..n as u32).collect::<Vec<_>>();
    map[left as usize] = right;
    map[right as usize] = left;
    map
}

fn enumerate_group_elements(n: usize, classes: usize, generators: &[SwapGenerator]) -> Vec<GroupElement> {
    let identity = GroupElement::identity(n, classes);
    let mut elements = vec![identity.clone()];
    let mut seen = FxHashSet::<(Vec<TerminalID>, Vec<u32>)>::default();
    seen.insert((identity.terminal_map.clone(), identity.class_map.clone()));
    let mut worklist = VecDeque::from([identity]);
    while let Some(element) = worklist.pop_front() {
        for generator in generators {
            let next = element.compose_right(generator);
            let key = (next.terminal_map.clone(), next.class_map.clone());
            if seen.insert(key) {
                elements.push(next.clone());
                worklist.push_back(next);
            }
        }
    }
    elements
}

fn concrete_state_map(states: usize) -> ManyToOneIdMap {
    let ids = (0..states as u32).collect::<Vec<_>>();
    ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(ids.clone(), ids)
}

fn lift_weight_to_concrete_states(weight: &Weight, coarse: &ManyToOneIdMap) -> Weight {
    if weight.is_empty() || weight.is_full() { return weight.clone(); }
    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (start, end, tokens) in weight.compact_entries().expect("non-full weight must expose entries") {
        for tsid in start..=end {
            let originals = coarse.internal_to_originals.get(tsid as usize)
                .expect("weight TSID lies outside the coarse map");
            entries.extend(originals.iter().copied().map(|state| (state, tokens.clone())));
        }
    }
    entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    Weight::union_sorted_point_entries(entries)
}

fn lift_dwa_weights_to_concrete_states(dwa: &mut DWA, coarse: &ManyToOneIdMap) {
    let mut cache = FxHashMap::<usize, Weight>::default();
    let mut lift = |weight: &Weight| {
        cache.entry(weight.ptr_key())
            .or_insert_with(|| lift_weight_to_concrete_states(weight, coarse))
            .clone()
    };
    for state in dwa.states_mut() {
        if let Some(final_weight) = &mut state.final_weight { *final_weight = lift(final_weight); }
        for (_, weight) in state.transitions.values_mut() { *weight = lift(weight); }
    }
}

fn remap_weight_by_class_transport(
    weight: &Weight,
    class_map: &[u32],
    machine: &RowMachine,
) -> Weight {
    if weight.is_empty() || weight.is_full() { return weight.clone(); }
    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (start, end, tokens) in weight.compact_entries().expect("non-full weight must expose entries") {
        for source in start..=end {
            let source_class = machine.class_for_state[source as usize] as usize;
            let target_class = class_map[source_class] as usize;
            entries.extend(machine.class_original_states[target_class].iter().copied()
                .map(|target| (target, tokens.clone())));
        }
    }
    entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    Weight::union_sorted_point_entries(entries)
}

fn remap_weight_by_subsumption_transport(
    weight: &Weight,
    representative_state_to_members: &[Vec<u32>],
) -> Weight {
    if weight.is_empty() || weight.is_full() { return weight.clone(); }
    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (start, end, tokens) in weight.compact_entries().expect("non-full weight must expose entries") {
        for source in start..=end {
            entries.extend(representative_state_to_members[source as usize]
                .iter()
                .copied()
                .map(|member_state| (member_state, tokens.clone())));
        }
    }
    entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    Weight::union_sorted_point_entries(entries)
}

fn transformed_dwa_view(
    original: &DWA,
    element: &GroupElement,
    machine: &RowMachine,
    profile: &mut TerminalInterchangeabilityProfile,
) -> DWA {
    let mut states = Vec::with_capacity(original.states().len());
    for state in original.states() {
        let mut transitions = std::collections::BTreeMap::new();
        for (&label, &(target, ref weight)) in &state.transitions {
            let mapped_label = if label >= 0 { element.terminal_map[label as usize] as i32 } else { label };
            let mapped_weight = remap_weight_by_class_transport(
                weight,
                &element.class_map,
                machine,
            );
            if mapped_weight.is_empty() { continue; }
            transitions.insert(mapped_label, (target, mapped_weight));
            profile.expanded_transition_copies += 1;
        }
        let final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| remap_weight_by_class_transport(weight, &element.class_map, machine));
        states.push(DWAState {
            transitions,
            final_weight,
        });
    }
    DWA::from_parts(states, original.start_state())
}

/// Literal reference construction for one directed member substitution.
///
/// `base_view` has already expanded every continuation label.  This copy
/// transports the initial lexer-state coordinate in *every* transition/final
/// weight, replaces the representative's start label with the member label,
/// and is later unioned disjointly with the identity and other member copies
/// before a fresh determinization/minimization.
fn subsumption_dwa_view(
    base_view: &DWA,
    representative: TerminalID,
    member: TerminalID,
    transport: &SubsumptionTransport,
    profile: &mut TerminalInterchangeabilityProfile,
) -> DWA {
    let mut result = base_view.clone();
    for state in result.states_mut() {
        state.final_weight = state.final_weight.as_ref().map(|weight| {
            remap_weight_by_subsumption_transport(
                weight,
                &transport.representative_state_to_members,
            )
        });
        for (_, weight) in state.transitions.values_mut() {
            *weight = remap_weight_by_subsumption_transport(
                weight,
                &transport.representative_state_to_members,
            );
        }
    }

    let start = result.start_state() as usize;
    let transitions = &mut result.states_mut()[start].transitions;
    let Some((target, weight)) = transitions.remove(&(representative as i32)) else {
        profile.initial_substitutions_missing += 1;
        return result;
    };
    if !weight.is_empty() {
        assert!(
            transitions.insert(member as i32, (target, weight)).is_none(),
            "inactive subsumption member unexpectedly already present at DWA start",
        );
        profile.expanded_transition_copies += 1;
        profile.initial_substitutions_applied += 1;
    }
    result
}

fn expand_noninitial_terminal_labels(
    original: &DWA,
    members_by_representative: &[Vec<TerminalID>],
    profile: &mut TerminalInterchangeabilityProfile,
) -> DWA {
    let mut result = original.clone();
    let start = result.start_state();
    for (state_id, state) in result.states_mut().iter_mut().enumerate() {
        if state_id as u32 == start {
            continue;
        }
        let mut expanded = BTreeMap::new();
        for (&label, &(target, ref weight)) in &state.transitions {
            if label < 0 {
                expanded.insert(label, (target, weight.clone()));
                continue;
            }
            let members = &members_by_representative[label as usize];
            for id in members.iter().copied() {
                expanded.insert(id as i32, (target, weight.clone()));
                profile.expanded_transition_copies += 1;
            }
        }
        state.transitions = expanded;
    }
    result
}



fn transport_all_dwa_weights_in_place(
    dwa: &mut DWA,
    class_map: &[u32],
    machine: &RowMachine,
) {
    for state in dwa.states_mut() {
        state.final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| remap_weight_by_class_transport(weight, class_map, machine));
        for (_, weight) in state.transitions.values_mut() {
            *weight = remap_weight_by_class_transport(weight, class_map, machine);
        }
    }
}

fn substitute_initial_terminal(
    base_view: &DWA,
    representative: TerminalID,
    member: TerminalID,
    class_map: &[u32],
    machine: &RowMachine,
    profile: &mut TerminalInterchangeabilityProfile,
) -> DWA {
    let mut result = base_view.clone();
    let start = result.start_state() as usize;
    // TSID support is the token's initial lexer-state coordinate throughout
    // the DWA, including later static transition and final weights.  Transport
    // the complete copied view first; only the selected start label changes.
    transport_all_dwa_weights_in_place(&mut result, class_map, machine);
    let transitions = &mut result.states_mut()[start].transitions;
    let Some((target, mapped)) = transitions.remove(&(representative as i32)) else {
        profile.initial_substitutions_missing += 1;
        return result;
    };
    transitions.insert(member as i32, (target, mapped));
    profile.expanded_transition_copies += 1;
    profile.initial_substitutions_applied += 1;
    result
}

fn debug_state_after_env_prefix(tokenizer: &Tokenizer) -> Option<u32> {
    let hex = std::env::var("GLRMASK_DEBUG_LEXER_PREFIX_HEX").ok()?;
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut state = tokenizer.initial_state_id();
    for offset in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[offset..offset + 2], 16).ok()?;
        let next = tokenizer.transition_row(state)[byte as usize];
        if next == NO_STATE {
            return None;
        }
        state = next;
    }
    Some(state)
}

#[derive(Debug)]
struct DisjointSet { parent: Vec<usize>, rank: Vec<u8> }
impl DisjointSet {
    fn new(size: usize) -> Self { Self { parent: (0..size).collect(), rank: vec![0; size] } }
    fn find(&mut self, item: usize) -> usize {
        if self.parent[item] != item {
            let root = self.find(self.parent[item]);
            self.parent[item] = root;
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
    use range_set_blaze::RangeSetBlaze;

    fn tiny_machine(outputs: &[u8], transitions: &[u32], width: usize) -> RowMachine {
        let terminal_count = outputs
            .iter()
            .map(|&bits| (u8::BITS - bits.leading_zeros()) as usize)
            .max()
            .unwrap_or(0)
            .max(2);
        let outputs = outputs
            .iter()
            .map(|&bits| OutputBits(vec![bits as u64]))
            .collect::<Vec<_>>();
        let classes_for_state = minimize_moore(&outputs, transitions, width);
        let class_count = classes_for_state.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut class_outputs = vec![OutputBits(vec![0]); class_count];
        let mut class_transitions = vec![0; class_count * width];
        let mut originals = vec![Vec::new(); class_count];
        let mut representative = vec![None; class_count];
        for state in 0..outputs.len() {
            let class = classes_for_state[state] as usize;
            representative[class].get_or_insert(state);
            class_outputs[class] = outputs[state].clone();
            originals[class].push(state as u32);
        }
        for class in 0..class_count {
            let state = representative[class].unwrap();
            for slot in 0..width {
                class_transitions[class * width + slot] =
                    classes_for_state[transitions[state * width + slot] as usize];
            }
        }
        let mut class_terminals = vec![Vec::new(); class_count];
        let mut terminal_classes = vec![Vec::new(); terminal_count];
        for class in 0..class_count {
            for terminal in 0..terminal_count {
                if terminal_output_bit(&class_outputs[class], terminal) {
                    class_terminals[class].push(terminal);
                    terminal_classes[terminal].push(class);
                }
            }
        }
        RowMachine {
            width,
            terminal_count,
            class_for_state: classes_for_state,
            class_outputs,
            class_transitions,
            class_terminals,
            terminal_classes,
            class_has_real_state: vec![true; class_count],
            class_original_states: originals,
            real_state_count: outputs.len(),
        }
    }

    #[test]
    fn hopcroft_sparse_minimizer_matches_fixed_point_oracle() {
        for seed in 0u32..64 {
            let states = 9usize;
            let width = 5usize;
            let outputs = (0..states)
                .map(|state| ((seed.wrapping_mul(17) + state as u32 * 13) & 3) == 0)
                .collect::<Vec<_>>();
            let transitions = (0..states * width)
                .map(|index| {
                    let value = seed
                        .wrapping_mul(37)
                        .wrapping_add(index as u32 * 19)
                        .wrapping_add((index / width) as u32 * 7);
                    if value % 7 == 0 {
                        NO_RESIDUAL_PAIR
                    } else {
                        value % states as u32
                    }
                })
                .collect::<Vec<_>>();
            let (hopcroft, _) = minimize_sparse_terminal_residuals(&outputs, &transitions, width);
            let (iterative, _) =
                minimize_sparse_terminal_residuals_iterative(&outputs, &transitions, width);
            for left in 0..states {
                for right in 0..states {
                    assert_eq!(
                        hopcroft[left] == hopcroft[right],
                        iterative[left] == iterative[right],
                        "seed={seed}, states=({left}, {right})",
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_terminal_residuals_match_dense_joint_product() {
        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let bytes = [true; 256];
        let sparse = SparseTerminalResiduals::build(&tokenizer, &bytes);
        let dense = RowMachine::build(&tokenizer, &bytes);
        let dense_blocks = minimize_terminal_residuals(&dense);
        for left_state in 0..tokenizer.num_states() {
            for right_state in 0..tokenizer.num_states() {
                for left_terminal in 0..tokenizer.num_terminals() {
                    for right_terminal in 0..tokenizer.num_terminals() {
                        let sparse_equal = sparse.block_for(left_terminal, left_state)
                            == sparse.block_for(right_terminal, right_state);
                        let left_class = dense.class_for_state[left_state as usize] as usize;
                        let right_class = dense.class_for_state[right_state as usize] as usize;
                        let dense_equal = dense_blocks[left_class * dense.terminal_count + left_terminal as usize]
                            == dense_blocks[right_class * dense.terminal_count + right_terminal as usize];
                        assert_eq!(
                            sparse_equal,
                            dense_equal,
                            "states ({left_state}, {right_state}), terminals ({left_terminal}, {right_terminal})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_subsumption_matches_pairwise_oracle() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::make_choice(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())]),
            Expr::U8Seq(b"b".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let bytes = [true; 256];
        let sparse = SparseTerminalResiduals::build(&tokenizer, &bytes);
        let machine = RowMachine::build(&tokenizer, &bytes);
        for member in 0..tokenizer.num_terminals() {
            for representative in 0..tokenizer.num_terminals() {
                if member == representative {
                    continue;
                }
                assert_eq!(
                    sparse
                        .subsumption_transport(member, representative)
                        .is_some(),
                    subsumption_transport_pairwise(&machine, member, representative).is_some(),
                    "member={member}, representative={representative}",
                );
            }
        }
    }

    #[test]
    fn sparse_row_hash_filter_keeps_exact_subsumption() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::make_choice(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let sparse = SparseTerminalResiduals::build(&tokenizer, &[true; 256]);
        let candidates = (0..tokenizer.num_terminals())
            .flat_map(|member| {
                (0..tokenizer.num_terminals())
                    .filter(move |&representative| representative != member)
                    .map(move |representative| (member, representative))
            })
            .collect::<Vec<_>>();
        let retained = sparse_row_hash_candidates(&sparse, &candidates)
            .into_iter()
            .collect::<FxHashSet<_>>();
        for candidate in candidates {
            if sparse
                .subsumption_transport(candidate.0, candidate.1)
                .is_some()
            {
                assert!(retained.contains(&candidate), "hash filter rejected {candidate:?}");
            }
        }
    }

    #[test]
    fn sparse_subsumption_finds_prefix_embedded_terminal() {
        let expressions = vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"ba".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let bytes = [true; 256];
        let sparse = SparseTerminalResiduals::build(&tokenizer, &bytes);
        let transport = sparse
            .subsumption_transport(0, 1)
            .expect("'a' should be subsumed by 'ba'");
        assert!(transport.representative_state_to_members.iter().any(|members| !members.is_empty()));
        assert!(sparse.subsumption_transport(1, 0).is_none());
    }

    #[test]
    fn subsumption_planner_accepts_prefix_embedded_member() {
        let expressions = vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"ba".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let plan = TerminalInterchangeability::build_subsumption(
            &tokenizer,
            &[true, true],
            None,
            &[true; 256],
        );
        assert_eq!(plan.active_representatives, vec![false, true]);
        assert_eq!(plan.members_by_representative[1], vec![0, 1]);
        assert_eq!(plan.subsumption_generators.len(), 1);
        assert_eq!(plan.subsumption_generators[0].representative, 1);
        assert_eq!(plan.subsumption_generators[0].member, 0);
    }

    #[test]
    fn subsumption_planner_chooses_the_covering_representative() {
        let expressions = vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"a".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let plan = TerminalInterchangeability::build_subsumption(
            &tokenizer,
            &[true, true],
            None,
            &[true; 256],
        );
        assert_eq!(plan.active_representatives, vec![true, false]);
        assert_eq!(plan.members_by_representative[0], vec![0, 1]);
        assert_eq!(plan.subsumption_generators.len(), 1);
        assert_eq!(plan.subsumption_generators[0].representative, 0);
        assert_eq!(plan.subsumption_generators[0].member, 1);
    }

    #[test]
    fn directed_subsumption_keeps_extra_representative_behaviour() {
        // x has residuals {a*, ∅}; y has {a*, {ε}, ∅}. Hence x ⪯ y,
        // but not conversely. With only x/y columns, this is exactly the
        // directed row-inclusion definition.
        let machine = tiny_machine(&[0b11, 0b00, 0b10], &[0, 1, 1], 1);
        let (rows, _) = TerminalSigmaRows::build(&machine);
        let transport = rows
            .subsumption_transport(&machine, 0, 1)
            .expect("x should be subsumed by y");
        assert!(transport
            .representative_state_to_members
            .iter()
            .flatten()
            .any(|&state| state == 0));
        assert!(rows.subsumption_transport(&machine, 1, 0).is_none());
    }

    #[test]
    fn pairwise_subsumption_transport_has_the_same_direction() {
        let machine = tiny_machine(&[0b11, 0b00, 0b10], &[0, 1, 1], 1);
        let transport = subsumption_transport_pairwise(&machine, 0, 1)
            .expect("x should be subsumed by y");
        assert!(transport
            .representative_state_to_members
            .iter()
            .flatten()
            .any(|&state| state == 0));
        assert!(subsumption_transport_pairwise(&machine, 1, 0).is_none());
    }

    #[test]
    fn strict_discovery_finds_byte_preserving_swap() {
        let four = Expr::U8Seq(b"aaaa".to_vec());
        let expressions = vec![
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Repeat {
                    expr: Box::new(four.clone()),
                    min: 0,
                    max: None,
                },
            ]),
            Expr::Seq(vec![
                Expr::U8Seq(b"aaa".to_vec()),
                Expr::Repeat {
                    expr: Box::new(four),
                    min: 0,
                    max: None,
                },
            ]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let machine = RowMachine::build(&tokenizer, &[true; 256]);
        assert!(swap_transport(&machine, 0, 1).is_some());
        let plan = TerminalInterchangeability::build(
            &tokenizer,
            &[true, true],
            None,
            &[true; 256],
        );
        assert!(!plan.is_identity());
        assert_eq!(plan.active_representatives, vec![true, false]);
        assert_eq!(plan.members_by_representative[0], vec![0, 1]);
        assert_eq!(plan.generators.len(), 1);
        assert_eq!(plan.generators[0].representative, 0);
        assert_eq!(plan.generators[0].member, 1);
    }

    #[test]
    fn finite_refinement_prunes_only_impossible_swaps() {
        let asymmetric = tiny_machine(&[0b101, 0b010], &[0, 1], 1);
        let pruned = minimize_terminal_residuals_with_swap_pruning(&asymmetric, &[(0, 1)]);
        assert!(pruned.surviving_pairs.is_empty());
        assert!(!pruned.fully_refined);

        let symmetric = tiny_machine(&[0b001, 0b010], &[1, 0], 1);
        let retained = minimize_terminal_residuals_with_swap_pruning(&symmetric, &[(0, 1)]);
        assert_eq!(retained.surviving_pairs, vec![(0, 1)]);
        assert!(retained.fully_refined);
    }

    #[test]
    fn candidate_refinement_never_discards_an_exact_swap() {
        // Exhaust a small family of two-state, two-terminal, one-byte DFAs.
        // Every swap certified by the old disjoint-union oracle must remain in
        // one color class of the cheap transition-aware prefilter.
        for outputs in 0u8..16 {
            let output_bits = [outputs & 0b11, (outputs >> 2) & 0b11];
            for transitions in 0u32..4 {
                let machine = tiny_machine(
                    &output_bits,
                    &[transitions & 1, (transitions >> 1) & 1],
                    1,
                );
                if swap_transport(&machine, 0, 1).is_none() {
                    continue;
                }
                let groups = terminal_candidate_groups(&machine, &[0, 1], None);
                assert!(groups.iter().any(|group| group == &vec![0, 1]));
            }
        }
    }

    #[test]
    fn sigma_rows_match_moore_union_transport() {
        let symmetric = tiny_machine(&[0b001, 0b010], &[1, 0], 1);
        let (rows, _) = TerminalSigmaRows::build(&symmetric);
        assert_eq!(
            rows.swap_transport(&symmetric, 0, 1).unwrap().class_map,
            swap_transport(&symmetric, 0, 1).unwrap().class_map,
        );

        let contextual = tiny_machine(&[0b101, 0b010], &[0, 1], 1);
        let (rows, _) = TerminalSigmaRows::build(&contextual);
        assert!(rows.swap_transport(&contextual, 0, 1).is_none());
        assert!(swap_transport(&contextual, 0, 1).is_none());
    }

    #[test]
    fn swap_transport_is_involutive() {
        // x/y are exchanged by the only active byte.
        let machine = tiny_machine(&[0b01, 0b10], &[1, 0], 1);
        let transport = swap_transport(&machine, 0, 1).expect("symmetric rows swap");
        assert_eq!(transport.class_map.len(), 2);
        for (source, &target) in transport.class_map.iter().enumerate() {
            assert_eq!(transport.class_map[target as usize], source as u32);
        }
    }

    #[test]
    fn swap_rejects_changed_same_state_context() {
        // z remains associated with x but not y, preventing a full-row swap.
        let machine = tiny_machine(&[0b101, 0b010], &[0, 1], 1);
        assert!(swap_transport(&machine, 0, 1).is_none());
    }

    #[test]
    fn generated_swaps_close_under_composition() {
        let first = SwapGenerator {
            representative: 0,
            member: 1,
            terminal_map: terminal_swap_map(3, 0, 1),
            class_map: vec![1, 0, 2],
        };
        let second = SwapGenerator {
            representative: 1,
            member: 2,
            terminal_map: terminal_swap_map(3, 1, 2),
            class_map: vec![0, 2, 1],
        };
        assert_eq!(enumerate_group_elements(3, 3, &[first, second]).len(), 6);
    }

    #[test]
    fn initial_substitution_transports_the_entire_start_row() {
        let machine = tiny_machine(&[0b01, 0b10], &[1, 0], 1);
        let initial = Weight::from_per_tsid_token_sets([(0, RangeSetBlaze::from_iter([0..=0]))]);
        let unrelated = Weight::from_per_tsid_token_sets([(0, RangeSetBlaze::from_iter([2..=2]))]);
        let final_weight = Weight::from_per_tsid_token_sets([(0, RangeSetBlaze::from_iter([3..=3]))]);
        let original = DWA::from_parts(
            vec![DWAState {
                transitions: [(0, (0, initial)), (2, (0, unrelated))]
                    .into_iter()
                    .collect(),
                final_weight: Some(final_weight),
            }],
            0,
        );
        let mut profile = TerminalInterchangeabilityProfile::default();
        let substituted = substitute_initial_terminal(
            &original,
            0,
            1,
            &[1, 0],
            &machine,
            &mut profile,
        );
        let state = &substituted.states()[0];
        let (_, selected) = state.transitions.get(&1).unwrap();
        let (_, untouched_label) = state.transitions.get(&2).unwrap();
        assert!(selected.tokens_for_tsid(1).contains(0));
        assert!(untouched_label.tokens_for_tsid(1).contains(2));
        assert!(state.final_weight.as_ref().unwrap().tokens_for_tsid(1).contains(3));
    }

    #[test]
    fn transformed_view_transports_weight_support_on_every_edge() {
        let machine = tiny_machine(&[0b01, 0b10], &[1, 0], 1);
        let element = GroupElement {
            terminal_map: terminal_swap_map(2, 0, 1),
            class_map: vec![1, 0],
        };
        let initial = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=0])),
        ]);
        let later = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([1..=1])),
        ]);
        let original = DWA::from_parts(
            vec![
                DWAState {
                    transitions: [(0, (1, initial.clone()))].into_iter().collect(),
                    final_weight: None,
                },
                DWAState {
                    transitions: [(0, (1, later.clone()))].into_iter().collect(),
                    final_weight: None,
                },
            ],
            0,
        );
        let mut profile = TerminalInterchangeabilityProfile::default();
        let transformed = transformed_dwa_view(&original, &element, &machine, &mut profile);
        let (initial_target, mapped_initial) = transformed.states()[0].transitions.get(&1).unwrap();
        let (later_target, mapped_later) = transformed.states()[1].transitions.get(&1).unwrap();
        // A terminal swap never rewrites DWA control state. The target graph is
        // identical; only labels and the lexer-state coordinate in weights move.
        assert_eq!(*initial_target, 1);
        assert_eq!(*later_target, 1);
        assert!(mapped_initial.tokens_for_tsid(1).contains(0));
        assert!(!mapped_initial.tokens_for_tsid(0).contains(0));
        assert!(mapped_later.tokens_for_tsid(1).contains(1));
        assert!(!mapped_later.tokens_for_tsid(0).contains(1));
    }
}
