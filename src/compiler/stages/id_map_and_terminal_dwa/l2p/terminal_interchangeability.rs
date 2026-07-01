//! Exact directed terminal subsumption for the L2+ terminal-DWA path.
//!
//! For one vocabulary partition we compute joint terminal residual rows on the
//! tokenizer DFA restricted to that partition's bytes. A strict interchangeable
//! pair has a bijective relabelling of those rows. A directed representative
//! family is more general: one representative can transport its residual rows
//! onto each member, while source-only representative contexts contribute no
//! member weight. A family is certified simultaneously against the exact lexer
//! view in which all of its nonrepresentative members are hidden.
//!
//! Construction remains validation-first: build the ordinary
//! representatives-only artifact, expand noninitial labels, make one
//! transported whole-DWA copy per initial member substitution, then merge the
//! local artifacts through the existing exact DWA/id-map merger.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_local_id_maps_and_terminal_dwas;
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;
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

/// Exact partial transport from representative scanner states to member
/// scanner states. A source state maps to every concrete target state with the
/// same transported terminal-residual row. An empty target set denotes
/// representative-only behaviour; weight transport drops that coordinate.
#[derive(Clone, Debug)]
struct InterchangeMap {
    source_state_to_target_states: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TerminalResidualKey {
    accepting: bool,
    successor_blocks: Vec<u32>,
}

/// Exact correlated terminal residuals for one L2P byte partition.
///
/// A residual block names the full future match language of one
/// `(terminal, lexer-state)` pair. Blocks are minimized jointly across all
/// active terminals, so equal block IDs are comparable across terminal columns.
/// A scanner-state row is therefore a complete, exact terminal-residual view.
struct TerminalResidualRows {
    rows_by_state: Vec<Vec<(TerminalID, u32)>>,
    initial_state: usize,
    initial_block_by_terminal: Vec<Option<u32>>,
    terminal_count: usize,
}

impl TerminalResidualRows {
    fn build(restricted: &RestrictedDfa<'_>, terminals: &[TerminalID]) -> Self {
        let state_count = restricted.state_count();
        let real_state_count = restricted.real_state_count;
        let dead_state = restricted.dead_state();
        let terminal_count = restricted.tokenizer.num_terminals() as usize;
        let mut terminal_index = vec![usize::MAX; terminal_count];
        for (index, &terminal) in terminals.iter().enumerate() {
            terminal_index[terminal as usize] = index;
        }

        let pairs_per_terminal = state_count;
        let pair_count = terminals.len() * pairs_per_terminal;
        let mut accepting = vec![false; pair_count];
        for state in 0..real_state_count {
            for terminal in restricted.tokenizer.matched_terminals_iter(state as u32) {
                let index = terminal_index[terminal as usize];
                if index != usize::MAX {
                    accepting[index * pairs_per_terminal + state] = true;
                }
            }
        }
        let successors = (0..state_count)
            .map(|state| {
                (0..restricted.bytes.len())
                    .map(|slot| restricted.successor(state, slot))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut blocks = accepting
            .iter()
            .map(|&accepting| u32::from(accepting))
            .collect::<Vec<_>>();
        loop {
            let mut ids = HashMap::<TerminalResidualKey, u32>::with_capacity(pair_count);
            let mut next = vec![0u32; pair_count];
            for terminal_index in 0..terminals.len() {
                let base = terminal_index * pairs_per_terminal;
                for state in 0..state_count {
                    let key = TerminalResidualKey {
                        accepting: accepting[base + state],
                        successor_blocks: successors[state]
                            .iter()
                            .map(|&target| blocks[base + target])
                            .collect(),
                    };
                    let id = ids.len() as u32;
                    next[base + state] = *ids.entry(key).or_insert(id);
                }
            }
            if next == blocks {
                break;
            }
            blocks = next;
        }

        let initial = restricted.tokenizer.initial_state_id() as usize;
        let mut initial_block_by_terminal = vec![None; terminal_count];
        let mut rows_by_state = vec![Vec::<(TerminalID, u32)>::new(); real_state_count];
        for (terminal_index, &terminal) in terminals.iter().enumerate() {
            let base = terminal_index * pairs_per_terminal;
            let dead_block = blocks[base + dead_state];
            if blocks[base + initial] != dead_block {
                initial_block_by_terminal[terminal as usize] = Some(blocks[base + initial]);
            }
            for state in 0..real_state_count {
                let block = blocks[base + state];
                if block != dead_block {
                    rows_by_state[state].push((terminal, block));
                }
            }
        }
        for row in &mut rows_by_state {
            row.sort_unstable_by_key(|&(terminal, _)| terminal);
        }
        Self {
            rows_by_state,
            initial_state: initial,
            initial_block_by_terminal,
            terminal_count,
        }
    }

    fn reset_residual_groups(
        &self,
        candidates: &[TerminalID],
    ) -> BTreeMap<Option<u32>, Vec<TerminalID>> {
        let mut groups = BTreeMap::<Option<u32>, Vec<TerminalID>>::new();
        for &terminal in candidates {
            groups
                .entry(self.initial_block_by_terminal[terminal as usize])
                .or_default()
                .push(terminal);
        }
        groups
    }

    /// Return one partial transport result per nonrepresentative member in this
    /// exact family view.  `None` means that member cannot be restored from the
    /// supplied representative while every other member in `members` is hidden.
    fn family_member_maps(
        &self,
        representative: TerminalID,
        members: &[TerminalID],
    ) -> Vec<(TerminalID, Option<InterchangeMap>)> {
        assert!(members.contains(&representative));
        let mut hidden = vec![false; self.terminal_count];
        for &member in members {
            if member != representative {
                hidden[member as usize] = true;
            }
        }
        let mut base_states_by_row = HashMap::<Vec<(TerminalID, u32)>, Vec<u32>>::new();
        for (state, row) in self.rows_by_state.iter().enumerate() {
            let base_row = row
                .iter()
                .copied()
                .filter(|&(terminal, _)| !hidden[terminal as usize])
                .collect::<Vec<_>>();
            base_states_by_row
                .entry(base_row)
                .or_default()
                .push(state as u32);
        }

        members
            .iter()
            .copied()
            .filter(|&member| member != representative)
            .map(|member| {
                let mut base_state_to_member_states =
                    vec![Vec::<u32>::new(); self.rows_by_state.len()];
                let mut valid = true;
                for (member_state, row) in self.rows_by_state.iter().enumerate() {
                    let mut transformed = Vec::with_capacity(row.len());
                    let mut member_block = None;
                    for &(terminal, block) in row {
                        if terminal == representative {
                            continue;
                        }
                        if terminal == member {
                            member_block = Some(block);
                            continue;
                        }
                        if hidden[terminal as usize] {
                            continue;
                        }
                        transformed.push((terminal, block));
                    }
                    // A target lexer state in which this member has no residual
                    // contributes no member weight. It is intentionally outside
                    // the partial transport's image; the identity branch supplies
                    // all non-member behaviour there.
                    let Some(block) = member_block else {
                        continue;
                    };
                    let insertion = transformed
                        .binary_search_by_key(&representative, |&(terminal, _)| terminal)
                        .unwrap_or_else(|index| index);
                    transformed.insert(insertion, (representative, block));
                    let Some(base_states) = base_states_by_row.get(&transformed) else {
                        valid = false;
                        break;
                    };
                    for &base_state in base_states {
                        base_state_to_member_states[base_state as usize]
                            .push(member_state as u32);
                    }
                }
                if valid
                    && !base_state_to_member_states[self.initial_state]
                        .contains(&(self.initial_state as u32))
                {
                    valid = false;
                }
                (
                    member,
                    valid.then_some(InterchangeMap {
                        source_state_to_target_states: base_state_to_member_states,
                    }),
                )
            })
            .collect()
    }

    fn family_maps(
        &self,
        representative: TerminalID,
        members: &[TerminalID],
    ) -> Option<BTreeMap<TerminalID, InterchangeMap>> {
        if members.len() < 2 || !members.contains(&representative) {
            return None;
        }
        let mut maps = BTreeMap::new();
        for (member, map) in self.family_member_maps(representative, members) {
            maps.insert(member, map?);
        }
        Some(maps)
    }

    /// Return the largest exact family for a fixed representative within one
    /// reset-residual bucket.
    ///
    /// Start with every candidate hidden. Hiding an additional member deletes
    /// that same terminal column from both the representatives-only rows and
    /// each transformed member row, so an existing row equality cannot become
    /// false. Therefore a member that fails while *all* candidates are hidden
    /// cannot occur in any smaller valid family containing this representative.
    /// Peeling all such failures and repeating is exact, not a greedy heuristic.
    fn maximal_family_maps(
        &self,
        representative: TerminalID,
        pool: &[TerminalID],
    ) -> (Vec<TerminalID>, BTreeMap<TerminalID, InterchangeMap>) {
        let mut members = pool.to_vec();
        members.sort_unstable();
        assert!(members.contains(&representative));
        loop {
            if members.len() < 2 {
                return (vec![representative], BTreeMap::new());
            }
            let candidate_maps = self.family_member_maps(representative, &members);
            let failed = candidate_maps
                .iter()
                .filter_map(|&(member, ref map)| map.is_none().then_some(member))
                .collect::<Vec<_>>();
            if failed.is_empty() {
                let maps = candidate_maps
                    .into_iter()
                    .map(|(member, map)| (member, map.expect("checked nonempty")))
                    .collect();
                return (members, maps);
            }
            members.retain(|member| *member == representative || !failed.contains(member));
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
            for target in 0..state_count {
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

        // A valid interchange map must keep the lexer reset state in its own
        // image. Continuation scanning restarts from the fixed initial state
        // after every terminal boundary, so the start state must map to a
        // partition that still contains the start state. Maps that move the
        // reset residual (e.g. terminals that finalize at different byte
        // residues) are not valid interchange maps: the two terminals are not
        // interchangeable, and the construction falls back to an ordinary build.
        let initial = self.tokenizer.initial_state_id() as usize;
        if !source_state_to_target_states
            .get(initial)
            .is_some_and(|targets| targets.contains(&(initial as u32)))
        {
            return None;
        }

        Some(InterchangeMap {
            source_state_to_target_states,
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
                // Local-artifact merging requires every raw scanner state to
                // name a local coordinate.  Directed source-only behaviour has
                // no transported weight, so put those target states in a fresh
                // dead class rather than leaving an invalid u32::MAX mapping.
                tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                    target_state_to_internal,
                    target_block_ids.len() as u32,
                )
                .fill_unmapped_with_new_class(),
                vocab_tokens: base.vocab_tokens.clone(),
            },
            source_to_target,
        )
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
        let candidates = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Self::identity(active_terminals);
        }

        let restricted = RestrictedDfa::new(tokenizer, active_terminals, relevant_bytes);
        // All active terminals remain observable in the residual row, including
        // IGNORE. IGNORE is excluded only from representative selection; its
        // current/future matches still constrain every directed transport.
        let observable_terminals = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .collect::<Vec<_>>();
        let residuals = TerminalResidualRows::build(&restricted, &observable_terminals);
        let candidate_groups = residuals.reset_residual_groups(&candidates);
        let mut reset_group_for_terminal = vec![Vec::<TerminalID>::new(); active_terminals.len()];
        for group in candidate_groups.values() {
            for &terminal in group {
                reset_group_for_terminal[terminal as usize] = group.clone();
            }
        }

        // A family is valid only as a whole. A directed map is never inferred
        // by transitive closure or a collection of independently certified
        // pairwise maps. For each representative, `maximal_family_maps` returns
        // the largest exact family inside its reset-residual bucket.
        let mut result = Self::identity(active_terminals);
        let mut remaining = candidates.iter().copied().collect::<BTreeSet<_>>();
        let mut full_family_attempts = 0usize;
        let mut certified_members = 0usize;
        while let Some(&first_remaining) = remaining.iter().next() {
            let mut best_representative = first_remaining;
            let mut best_members = vec![first_remaining];
            let mut best_maps = BTreeMap::<TerminalID, InterchangeMap>::new();

            for &representative in &remaining {
                let mut pool = reset_group_for_terminal[representative as usize]
                    .iter()
                    .copied()
                    .filter(|member| remaining.contains(member))
                    .collect::<Vec<_>>();
                pool.sort_unstable();
                if pool.len() < 2 {
                    continue;
                }
                full_family_attempts += 1;
                let (members, maps) = residuals.maximal_family_maps(representative, &pool);
                if members.len() > best_members.len()
                    || (members.len() == best_members.len() && representative < best_representative)
                {
                    best_representative = representative;
                    best_members = members;
                    best_maps = maps;
                }
            }

            for &member in &best_members {
                remaining.remove(&member);
            }
            if best_members.len() < 2 {
                continue;
            }
            certified_members += best_members.len() - 1;
            result.members_by_representative[best_representative as usize] = best_members.clone();
            for &member in &best_members {
                result.representative_for[member as usize] = best_representative;
                if member != best_representative {
                    result.active_representatives[member as usize] = false;
                    result.maps_by_representative_member.insert(
                        (best_representative, member),
                        best_maps
                            .remove(&member)
                            .expect("certified family member missing its maximal-family transport"),
                    );
                }
            }
        }

        if std::env::var_os("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY").is_some() {
            eprintln!(
                "[glrmask/debug][terminal_subsumption] active={} reset_buckets={} representative_family_attempts={} certified_members={}",
                candidates.len(),
                candidate_groups.len(),
                full_family_attempts,
                certified_members,
            );
            for (representative, members) in result.members_by_representative.iter().enumerate() {
                if members.len() >= 2 {
                    eprintln!(
                        "[glrmask/debug][terminal_subsumption] representative={} members={:?}",
                        representative, members,
                    );
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
    /// member of its reset-equivalent representative family, keeping the same
    /// destination and weight. Edges from the start state are left untouched
    /// because their current scanner coordinates require the directed member
    /// transports built in stage 2.
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
    fn directed_subsumption_contains_strict_interchangeability_small_matrix() {
        fn language(words: &[&[u8]]) -> Expr {
            if words.len() == 1 {
                Expr::U8Seq(words[0].to_vec())
            } else {
                Expr::Choice(
                    words
                        .iter()
                        .map(|word| Expr::U8Seq(word.to_vec()))
                        .collect(),
                )
            }
        }

        let variants: &[&[&[u8]]] = &[
            &[b"ab"],
            &[b"ab", b"xa"],
            &[b"ab", b"yb"],
            &[b"ab", b"xa", b"yb"],
            &[b"aa"],
        ];
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let mut strict_pairs = 0usize;

        for left in variants {
            for right in variants {
                let expressions = vec![
                    language(left),
                    language(right),
                    Expr::U8Seq(b"z".to_vec()),
                ];
                let tokenizer = build_regex(&expressions).into_tokenizer(
                    expressions.len() as u32,
                    Some(Arc::from(expressions.into_boxed_slice())),
                );
                let dfa = RestrictedDfa::new(&tokenizer, &[true, true, true], &relevant);
                if dfa.interchange_map(0, 1).is_none() {
                    continue;
                }
                strict_pairs += 1;
                let residuals = TerminalResidualRows::build(&dfa, &[0, 1, 2]);
                assert!(
                    residuals.family_maps(0, &[0, 1]).is_some(),
                    "strict witness was not accepted in forward directed form: left={left:?} right={right:?}",
                );
                assert!(
                    residuals.family_maps(1, &[0, 1]).is_some(),
                    "strict witness was not accepted in reverse directed form: left={left:?} right={right:?}",
                );
            }
        }
        assert!(strict_pairs > 0, "matrix failed to exercise strict witnesses");
    }

    #[test]
    fn directed_subsumption_star_is_larger_than_strict_interchangeability() {
        // Every terminal recognizes `ab` from lexer reset on the current
        // `{a,b}` partition. R also carries two distinct incoming-context
        // residuals. M1 has only the x residual and M2 only the y residual.
        // The strict relation cannot swap R with either member, nor M1 with
        // M2, but R can represent the whole directed family.
        let expressions = vec![
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"xa".to_vec()),
                Expr::U8Seq(b"yb".to_vec()),
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"xa".to_vec()),
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"yb".to_vec()),
            ]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true, true], &relevant);
        assert!(dfa.interchange_map(0, 1).is_none());
        assert!(dfa.interchange_map(0, 2).is_none());
        assert!(dfa.interchange_map(1, 2).is_none());

        let residuals = TerminalResidualRows::build(&dfa, &[0, 1, 2]);
        let (members, maps) = residuals.maximal_family_maps(0, &[0, 1, 2]);
        assert_eq!(members, vec![0, 1, 2]);
        assert_eq!(maps.len(), 2, "R must simultaneously subsume M1 and M2");
        let plan = TerminalInterchangeability::build(
            &tokenizer,
            &[true, true, true],
            &relevant,
            None,
        );
        assert_eq!(plan.active_terminal_count_before(), 3);
        assert_eq!(plan.active_terminal_count_after(), 1);
        assert_eq!(plan.members_by_representative[0], vec![0, 1, 2]);
    }

    #[test]
    fn directed_family_certificate_accepts_the_nonmutual_pair() {
        let expressions = vec![
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"xa".to_vec()),
            ]),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"y".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true, true], &relevant);
        let residuals = TerminalResidualRows::build(&dfa, &[0, 1, 2]);
        let maps = residuals
            .family_maps(0, &[0, 1])
            .expect("the directed R/M family must have a simultaneous witness");
        assert_eq!(maps.len(), 1);
        assert!(maps.contains_key(&1));
    }

    #[test]
    fn directed_subsumption_can_project_an_extra_incoming_residual() {
        // On the current `{a,b}` vocabulary partition, `R` and `M` both
        // finalize exactly `ab` from lexer reset.  `R` nevertheless has the
        // additional incoming-context residual `a` after an earlier `x`.
        // Preserved terminal `Y` prevents the reverse projection from silently
        // collapsing the context, so this is strictly directed: R subsumes M,
        // but M does not subsume R.
        let expressions = vec![
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"xa".to_vec()),
            ]),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"y".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let dfa = RestrictedDfa::new(&tokenizer, &[true, true, true], &relevant);
        let residuals = TerminalResidualRows::build(&dfa, &[0, 1, 2]);
        let forward = residuals
            .family_maps(0, &[0, 1])
            .expect("R must cover M through an exact directed residual map");
        let mut seen_target_blocks = BTreeSet::<Vec<u32>>::new();
        assert!(
            forward[&1]
                .source_state_to_target_states
                .iter()
                .filter(|targets| !targets.is_empty())
                .any(|targets| !seen_target_blocks.insert(targets.clone())),
            "the witness must collapse distinct representative contexts",
        );
        assert!(
            residuals.family_maps(1, &[0, 1]).is_none(),
            "M cannot cover R's additional incoming-context residual",
        );

        let plan = TerminalInterchangeability::build(
            &tokenizer,
            &[true, true, true],
            &relevant,
            None,
        );
        assert_eq!(plan.active_terminal_count_before(), 3);
        assert_eq!(plan.active_terminal_count_after(), 2);
        assert_eq!(plan.members_by_representative[0], vec![0, 1]);
        assert!(!plan.active_representatives[1]);
    }

    #[test]
    fn rotated_residuals_have_no_interchange_map_because_reset_moves() {
        // A=/a(aaaa)*/ finalizes at residue 1, B=/aaa(aaaa)*/ at residue 3. The
        // only label swap that is a DFA automorphism is the +2 rotation, which
        // moves the reset state (0 -> 2). A valid interchange map must keep the
        // reset state in its own image, so A and B are NOT interchangeable.
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
