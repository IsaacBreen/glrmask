//! Rooted terminal interchangeability for the L2+ terminal-DWA reference path.
//!
//! For one vocabulary partition, the tokenizer DFA itself is left unchanged.
//! The partition only chooses which byte transitions `characterize` traverses.
//! In particular, the tokenizer's stored finalizer and future-finalizer metadata
//! is frozen: it is neither recomputed nor minimized after restricting bytes.
//!
//! For terminals `left` and `right`, characterize the tagged state `(map, s)` by
//! hashing, for every enabled byte, the destination's previous-round
//! characterization together with its mapped finalizers and mapped frozen future
//! finalizers. The identity and `left <-> right` sides share the same hashes.
//! Their stable hash classes are the two state partitions of the rooted terminal
//! interchange map. The initial lexer state must occur in the same class on both
//! sides.

use std::collections::BTreeMap;
use std::time::Instant;

use super::nwa_builder::TerminalNwaTransportMode;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::grammar::flat::TerminalID;

const NO_STATE: u32 = u32::MAX;
const CHARACTERIZATION_DOMAIN: &[u8] =
    b"glrmask terminal interchangeability characterize v1\0";
const CHARACTERIZATION_SEED: &[u8] =
    b"glrmask terminal interchangeability characterize seed v1\0";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct CharacterizationHash([u8; blake3::OUT_LEN]);

impl CharacterizationHash {
    fn seed() -> Self {
        Self(*blake3::hash(CHARACTERIZATION_SEED).as_bytes())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct OutputBits(Vec<u64>);

impl OutputBits {
    fn new(words: usize) -> Self { Self(vec![0; words]) }

    fn set(&mut self, terminal: usize) {
        self.0[terminal / 64] |= 1u64 << (terminal % 64);
    }

    #[cfg(test)]
    fn contains(&self, terminal: usize) -> bool {
        (self.0[terminal / 64] & (1u64 << (terminal % 64))) != 0
    }

    fn mapped(&self, swap: Option<(usize, usize)>) -> Self {
        let Some((left, right)) = swap else { return self.clone(); };
        if left == right { return self.clone(); }
        let left_word = left / 64;
        let right_word = right / 64;
        let left_mask = 1u64 << (left % 64);
        let right_mask = 1u64 << (right % 64);
        let left_present = (self.0[left_word] & left_mask) != 0;
        let right_present = (self.0[right_word] & right_mask) != 0;
        if left_present == right_present { return self.clone(); }
        let mut words = self.0.clone();
        words[left_word] ^= left_mask;
        words[right_word] ^= right_mask;
        Self(words)
    }

    fn append_to(&self, output: &mut Vec<u8>) {
        for &word in &self.0 { output.extend_from_slice(&word.to_le_bytes()); }
    }
}

/// A bijection between the stable terminal-specific state partitions. Every
/// source state maps to every raw DFA state in its corresponding target class.
#[derive(Clone, Debug)]
struct InterchangeMap {
    source_state_to_target_states: Vec<Vec<u32>>,
}

impl InterchangeMap {
    /// Select one concrete simulator state per target class. The class carrying
    /// the lexer reset state deliberately selects that reset state itself; every
    /// other class selects its smallest raw DFA state. This is only a concrete
    /// representative of the already established class-to-class transport.
    fn transport_representatives(&self, lexer_initial_state: u32) -> Option<Vec<u32>> {
        let root_targets = self
            .source_state_to_target_states
            .get(lexer_initial_state as usize)?;
        if !root_targets.contains(&lexer_initial_state) {
            return None;
        }

        self.source_state_to_target_states
            .iter()
            .map(|targets| {
                if targets == root_targets {
                    Some(lexer_initial_state)
                } else {
                    targets.first().copied()
                }
            })
            .collect()
    }
}

struct PairCharacterization {
    identity_hashes: Vec<CharacterizationHash>,
    swapped_hashes: Vec<CharacterizationHash>,
    rounds: usize,
}

struct RestrictedDfa {
    bytes: Vec<u8>,
    destinations: Vec<usize>,
    real_state_count: usize,
    initial_state: usize,
    empty_output: OutputBits,
    finalizers: Vec<OutputBits>,
    future_finalizers: Vec<OutputBits>,
    identity_rounds: Vec<Vec<CharacterizationHash>>,
    signature_capacity: usize,
}

impl RestrictedDfa {
    fn new(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Self {
        let bytes = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect::<Vec<_>>();
        let real_state_count = tokenizer.num_states() as usize;
        let state_count = real_state_count + 1;
        let output_words = (tokenizer.num_terminals() as usize).div_ceil(64);
        let terminal_bits = |terminals: Vec<TerminalID>| {
            let mut bits = OutputBits::new(output_words);
            for terminal in terminals {
                if observed_terminals
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    bits.set(terminal as usize);
                }
            }
            bits
        };
        let finalizers = (0..tokenizer.num_states())
            .map(|state| terminal_bits(tokenizer.matched_terminals_iter(state).collect()))
            .collect::<Vec<_>>();
        // These are the tokenizer's original, frozen future-finalizer sets. The
        // enabled-byte set below deliberately does not modify or recompute them.
        let future_finalizers = (0..tokenizer.num_states())
            .map(|state| {
                terminal_bits(tokenizer.possible_future_terminals_iter(state).collect())
            })
            .collect::<Vec<_>>();
        let destinations = (0..state_count)
            .flat_map(|state| {
                bytes.iter().map(move |&byte| {
                    if state == real_state_count {
                        return state;
                    }
                    let destination = tokenizer.get_transition(state as u32, byte);
                    if destination == NO_STATE {
                        real_state_count
                    } else {
                        destination as usize
                    }
                })
            })
            .collect::<Vec<_>>();
        let empty_output = OutputBits::new(output_words);
        let signature_capacity = CHARACTERIZATION_DOMAIN.len()
            + 4
            + bytes.len()
                * (1 + blake3::OUT_LEN + 8 + 2 * output_words * size_of::<u64>());
        let seed = CharacterizationHash::seed();

        Self {
            bytes,
            destinations,
            real_state_count,
            initial_state: tokenizer.initial_state_id() as usize,
            empty_output,
            finalizers,
            future_finalizers,
            identity_rounds: vec![vec![seed; state_count]],
            signature_capacity,
        }
    }

    fn state_count(&self) -> usize {
        self.real_state_count + 1
    }

    fn dead_state(&self) -> usize {
        self.real_state_count
    }

    /// This does not transform the lexer. It only supplies the absent
    /// destination while evaluating an enabled byte transition in
    /// `characterize`.
    fn destination_for_slot(&self, state: usize, byte_slot: usize) -> usize {
        self.destinations[state * self.bytes.len() + byte_slot]
    }

    fn output_at<'a>(&'a self, outputs: &'a [OutputBits], state: usize) -> &'a OutputBits {
        outputs.get(state).unwrap_or(&self.empty_output)
    }

    /// Hash a complete characterization tuple in one buffered BLAKE3 call.
    fn characterize_round(
        &self,
        previous: &[CharacterizationHash],
        finalizers: &[OutputBits],
        future_finalizers: &[OutputBits],
    ) -> Vec<CharacterizationHash> {
        debug_assert_eq!(previous.len(), self.state_count());
        let mut next = Vec::with_capacity(self.state_count());
        let mut tuple = Vec::with_capacity(self.signature_capacity);
        let output_word_count = self.empty_output.0.len() as u32;
        for state in 0..self.state_count() {
            tuple.clear();
            tuple.extend_from_slice(CHARACTERIZATION_DOMAIN);
            tuple.extend_from_slice(&(self.bytes.len() as u32).to_le_bytes());
            for byte_slot in 0..self.bytes.len() {
                let destination = self.destination_for_slot(state, byte_slot);
                tuple.push(self.bytes[byte_slot]);
                tuple.extend_from_slice(&previous[destination].0);
                tuple.extend_from_slice(&output_word_count.to_le_bytes());
                self.output_at(finalizers, destination).append_to(&mut tuple);
                tuple.extend_from_slice(&output_word_count.to_le_bytes());
                self.output_at(future_finalizers, destination)
                    .append_to(&mut tuple);
            }
            next.push(CharacterizationHash(*blake3::hash(&tuple).as_bytes()));
        }
        next
    }

    /// The identity-side recurrence is independent of the terminal pair, so it
    /// is cached by depth. This cache leaves the raw DFA and its frozen metadata
    /// untouched; it only avoids repeating the same recurrence.
    fn ensure_identity_round(&mut self, round: usize) {
        while self.identity_rounds.len() <= round {
            let previous_index = self.identity_rounds.len() - 1;
            let next = self.characterize_round(
                &self.identity_rounds[previous_index],
                &self.finalizers,
                &self.future_finalizers,
            );
            self.identity_rounds.push(next);
        }
    }

    /// Compute the terminal-specific partitions by iterating the supplied hash
    /// recurrence. The raw digests need not stabilize on a cycle; the induced
    /// equality partition over both tagged sides does.
    fn characterize_pair(&mut self, left: TerminalID, right: TerminalID) -> PairCharacterization {
        let state_count = self.state_count();
        let swap = Some((left as usize, right as usize));
        let swapped_finalizers = self
            .finalizers
            .iter()
            .map(|outputs| outputs.mapped(swap))
            .collect::<Vec<_>>();
        let swapped_future_finalizers = self
            .future_finalizers
            .iter()
            .map(|outputs| outputs.mapped(swap))
            .collect::<Vec<_>>();
        let mut swapped_previous = self.identity_rounds[0].clone();

        for rounds in 1..=state_count * 2 {
            self.ensure_identity_round(rounds);
            let swapped_next = self.characterize_round(
                &swapped_previous,
                &swapped_finalizers,
                &swapped_future_finalizers,
            );
            // A root mismatch or one-sided current class cannot be repaired by
            // later refinement: the common seed partition only ever splits.
            if !rooted_class_bijection_still_possible(
                &self.identity_rounds[rounds],
                &swapped_next,
                self.initial_state,
                self.real_state_count,
            ) {
                return PairCharacterization {
                    identity_hashes: self.identity_rounds[rounds].clone(),
                    swapped_hashes: swapped_next,
                    rounds,
                };
            }
            if same_equality_partition_pair(
                &self.identity_rounds[rounds - 1],
                &swapped_previous,
                &self.identity_rounds[rounds],
                &swapped_next,
            ) {
                return PairCharacterization {
                    identity_hashes: self.identity_rounds[rounds].clone(),
                    swapped_hashes: swapped_next,
                    rounds,
                };
            }
            swapped_previous = swapped_next;
        }

        panic!(
            "terminal interchangeability characterization did not stabilize within {} rounds",
            state_count * 2,
        );
    }

    fn interchange_map(&mut self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        let characterization = self.characterize_pair(left, right);
        self.interchange_map_from_characterization(&characterization)
    }

    fn interchange_map_from_characterization(
        &self,
        characterization: &PairCharacterization,
    ) -> Option<InterchangeMap> {
        if characterization.identity_hashes[self.initial_state]
            != characterization.swapped_hashes[self.initial_state]
        {
            return None;
        }

        let mut source_classes = BTreeMap::<CharacterizationHash, ()>::new();
        let mut target_states_by_class = BTreeMap::<CharacterizationHash, Vec<u32>>::new();
        for state in 0..self.real_state_count {
            source_classes.insert(characterization.identity_hashes[state], ());
            target_states_by_class
                .entry(characterization.swapped_hashes[state])
                .or_default()
                .push(state as u32);
        }
        if source_classes.len() != target_states_by_class.len()
            || source_classes
                .keys()
                .any(|hash| !target_states_by_class.contains_key(hash))
        {
            return None;
        }

        let source_state_to_target_states = (0..self.real_state_count)
            .map(|source| {
                target_states_by_class
                    .get(&characterization.identity_hashes[source])
                    .cloned()
            })
            .collect::<Option<Vec<_>>>()?;
        if source_state_to_target_states.iter().any(Vec::is_empty) {
            return None;
        }
        Some(InterchangeMap {
            source_state_to_target_states,
        })
    }

    fn debug_pair_summary(&mut self, left: TerminalID, right: TerminalID) -> String {
        let characterization = self.characterize_pair(left, right);
        let identity_classes = characterization.identity_hashes[..self.real_state_count]
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let swapped_classes = characterization.swapped_hashes[..self.real_state_count]
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let root_hash_equal = characterization.identity_hashes[self.initial_state]
            == characterization.swapped_hashes[self.initial_state];
        let map_exists = self
            .interchange_map_from_characterization(&characterization)
            .is_some();
        format!(
            "left={} right={} relevant_bytes={} rounds={} identity_classes={} swapped_classes={} root_hash_equal={} exact_map_exists={}",
            left,
            right,
            self.bytes.len(),
            characterization.rounds,
            identity_classes,
            swapped_classes,
            root_hash_equal,
            map_exists,
        )
    }
}

/// Equality of characterization digests represents a partition, not a required
/// fixed digest value. The partition is stable exactly when the two tagged sides
/// induce the same equivalence relation in consecutive rounds.
fn same_equality_partition_pair(
    identity_previous: &[CharacterizationHash],
    swapped_previous: &[CharacterizationHash],
    identity_next: &[CharacterizationHash],
    swapped_next: &[CharacterizationHash],
) -> bool {
    debug_assert_eq!(identity_previous.len(), swapped_previous.len());
    debug_assert_eq!(identity_previous.len(), identity_next.len());
    debug_assert_eq!(identity_previous.len(), swapped_next.len());
    let mut previous_to_next = BTreeMap::<CharacterizationHash, CharacterizationHash>::new();
    let mut next_to_previous = BTreeMap::<CharacterizationHash, CharacterizationHash>::new();
    for (&old, &new) in identity_previous
        .iter()
        .zip(identity_next)
        .chain(swapped_previous.iter().zip(swapped_next))
    {
        if previous_to_next
            .get(&old)
            .is_some_and(|&existing| existing != new)
            || next_to_previous
                .get(&new)
                .is_some_and(|&existing| existing != old)
        {
            return false;
        }
        previous_to_next.insert(old, new);
        next_to_previous.insert(new, old);
    }
    true
}

/// A valid eventual map needs its root class and every current left class to
/// have a matching right class. Since characterization starts from the common
/// seed and only refines, a failure here can never be repaired later.
fn rooted_class_bijection_still_possible(
    identity: &[CharacterizationHash],
    swapped: &[CharacterizationHash],
    initial_state: usize,
    real_state_count: usize,
) -> bool {
    if identity[initial_state] != swapped[initial_state] {
        return false;
    }
    let identity_classes = identity[..real_state_count]
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let swapped_classes = swapped[..real_state_count]
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    identity_classes == swapped_classes
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
            members_by_representative: (0..terminal_count as u32)
                .map(|terminal| vec![terminal])
                .collect(),
            maps_by_representative_member: BTreeMap::new(),
        }
    }

    pub(crate) fn build(
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
        ignore_terminal: Option<TerminalID>,
    ) -> Self {
        let started_at = Instant::now();
        let candidates = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Self::identity(active_terminals);
        }

        // Only L2+-active terminals are observable. The DFA and its stored
        // metadata remain untouched; the byte set merely bounds characterize.
        let mut restricted = RestrictedDfa::new(tokenizer, active_terminals, relevant_bytes);
        let mut accepted = BTreeMap::<(TerminalID, TerminalID), InterchangeMap>::new();
        let mut components = DisjointSet::new(active_terminals.len());
        let debug_pair = std::env::var("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY_PAIR")
            .ok()
            .and_then(|value| {
                let (left, right) = value.split_once(',')?;
                Some((
                    left.trim().parse::<TerminalID>().ok()?,
                    right.trim().parse::<TerminalID>().ok()?,
                ))
            });
        if let Some((left, right)) = debug_pair {
            if active_terminals.get(left as usize).copied().unwrap_or(false)
                && active_terminals.get(right as usize).copied().unwrap_or(false)
            {
                eprintln!(
                    "[glrmask/debug][terminal_interchangeability_pair] {}",
                    restricted.debug_pair_summary(left, right),
                );
            }
        }

        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] phase=start active={} pairs={} states={} bytes={}",
                candidates.len(),
                candidates.len() * (candidates.len() - 1) / 2,
                restricted.real_state_count,
                restricted.bytes.len(),
            );
        }

        let mut pair_count = 0usize;
        let mut skipped_transitive_pairs = 0usize;
        for (index, &left) in candidates.iter().enumerate() {
            for &right in &candidates[index + 1..] {
                // Rooted interchangeability maps are invertible and composable
                // while preserving the reset class. Once two terminals are in
                // one component, their direct pair map is already implied.
                if components.find(left as usize) == components.find(right as usize) {
                    skipped_transitive_pairs += 1;
                    continue;
                }
                pair_count += 1;
                if profile_enabled && pair_count % 256 == 0 {
                    eprintln!(
                        "[glrmask/profile][terminal_interchangeability] phase=pairs_done active={} checked_pairs={} skipped_transitive_pairs={} elapsed_ms={:.3}",
                        candidates.len(),
                        pair_count,
                        skipped_transitive_pairs,
                        started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
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

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] active={} checked_pairs={} skipped_transitive_pairs={} accepted_edges={} elapsed_ms={:.3}",
                candidates.len(),
                pair_count,
                skipped_transitive_pairs,
                accepted.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let mut groups = BTreeMap::<usize, Vec<TerminalID>>::new();
        for &terminal in &candidates {
            groups
                .entry(components.find(terminal as usize))
                .or_default()
                .push(terminal);
        }

        let mut result = Self::identity(active_terminals);
        for members in groups.into_values() {
            if members.len() < 2 {
                continue;
            }
            // This component is the terminal interchangeability class. Pair
            // maps compose while preserving the lexer-reset class, so skipped
            // internal pairs have derived maps and need not be recomputed.
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
                    result
                        .maps_by_representative_member
                        .insert((representative, member), map);
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
                    representative, members,
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
        self.representative_for
            .iter()
            .enumerate()
            .all(|(terminal, &representative)| terminal as TerminalID == representative)
    }

    pub(crate) fn active_representatives(&self) -> &[bool] {
        &self.active_representatives
    }

    /// Raw labels emitted by the transported trie walk. Nonrepresentatives
    /// remain visible to the scanner, where they can affect longest-match and
    /// frozen future-finalizer behavior, but their edges are reconstructed
    /// through the representative edge of the corresponding transport mode.
    pub(crate) fn visible_output_raw_labels(&self) -> Vec<bool> {
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

    pub(crate) fn nontrivial_classes(&self) -> impl Iterator<Item = (TerminalID, &[TerminalID])> {
        self.members_by_representative
            .iter()
            .enumerate()
            .filter(|(_, members)| members.len() > 1)
            .map(|(representative, members)| (representative as TerminalID, members.as_slice()))
    }

    pub(crate) fn active_assignments(&self) -> impl Iterator<Item = (TerminalID, TerminalID)> + '_ {
        self.original_active
            .iter()
            .enumerate()
            .filter(|&(_, &active)| active)
            .map(|(terminal, _)| {
                let terminal = terminal as TerminalID;
                (terminal, self.representative_for[terminal as usize])
            })
    }

    pub(crate) fn terminal_nwa_transport_modes(
        &self,
        lexer_initial_state: u32,
    ) -> Option<Vec<TerminalNwaTransportMode>> {
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
                let map = self
                    .maps_by_representative_member
                    .get(&(representative, member))?;
                let scanner_state_for_original =
                    map.transport_representatives(lexer_initial_state)?;
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
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
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
        if left == right {
            return;
        }
        if self.rank[left] < self.rank[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        if self.rank[left] == self.rank[right] {
            self.rank[left] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
        let terminal_count = expressions.len() as u32;
        build_regex(&expressions).into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    #[test]
    fn rooted_map_rejects_a_reset_moving_rotation() {
        let tokenizer = tokenizer(vec![
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                    min: 0,
                    max: None,
                },
            ]),
            Expr::Seq(vec![
                Expr::U8Seq(b"aaa".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                    min: 0,
                    max: None,
                },
            ]),
        ]);
        let mut dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        assert!(dfa.interchange_map(0, 1).is_none());
    }

    #[test]
    fn identical_literals_have_a_rooted_interchange_map() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
        ]);
        let mut dfa = RestrictedDfa::new(&tokenizer, &[true, true], &[true; 256]);
        let map = dfa.interchange_map(0, 1).expect("identical literals must transport");
        let root = tokenizer.initial_state_id() as usize;
        assert!(map.source_state_to_target_states[root].contains(&tokenizer.initial_state_id()));
        assert_eq!(
            map.transport_representatives(tokenizer.initial_state_id())
                .expect("rooted transport representatives")[root],
            tokenizer.initial_state_id(),
        );
        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert_eq!(plan.active_terminal_count_before(), 2);
        assert_eq!(plan.active_terminal_count_after(), 1);
    }

    #[test]
    fn alpha_interiors_are_ignored_when_only_punctuation_is_enabled() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"CREATE\"".to_vec()),
            Expr::U8Seq(b"CrossFit\"".to_vec()),
            Expr::U8Seq(b"DELETE\"".to_vec()),
            Expr::U8Seq(b"Drums\"".to_vec()),
        ]);
        let mut punctuation_only = [false; 256];
        punctuation_only[b'"' as usize] = true;
        let plan = TerminalInterchangeability::build(
            &tokenizer,
            &[true, true, true, true],
            &punctuation_only,
            None,
        );
        assert_eq!(plan.active_terminal_count_after(), 1);
        let members = plan
            .nontrivial_classes()
            .next()
            .expect("one literal class")
            .1;
        assert_eq!(members, &[0, 1, 2, 3]);
    }

    #[test]
    fn byte_restriction_does_not_recompute_frozen_future_finalizers() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"az".to_vec()),
        ]);
        let after_a = tokenizer.get_transition(tokenizer.initial_state_id(), b'a') as usize;
        let mut only_a = [false; 256];
        only_a[b'a' as usize] = true;
        let restricted = RestrictedDfa::new(&tokenizer, &[true, true], &only_a);
        assert_eq!(restricted.bytes, vec![b'a']);
        assert_eq!(restricted.destination_for_slot(after_a, 0), restricted.dead_state());
        assert!(restricted.output_at(&restricted.future_finalizers, after_a).contains(1));
    }

    #[test]
    fn inactive_outputs_are_not_observed() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ]);
        let mut dfa = RestrictedDfa::new(&tokenizer, &[true, false, true], &[true; 256]);
        assert!(dfa.interchange_map(0, 2).is_some());
    }

    #[test]
    fn equality_partition_stability_ignores_changing_digest_values() {
        let a = CharacterizationHash([1; blake3::OUT_LEN]);
        let b = CharacterizationHash([2; blake3::OUT_LEN]);
        let x = CharacterizationHash([9; blake3::OUT_LEN]);
        let y = CharacterizationHash([10; blake3::OUT_LEN]);
        assert!(same_equality_partition_pair(&[a, a, b], &[a, a, b], &[x, x, y], &[x, x, y]));
        assert!(!same_equality_partition_pair(&[a, a, b], &[a, a, b], &[x, y, y], &[x, y, y]));
    }
}
