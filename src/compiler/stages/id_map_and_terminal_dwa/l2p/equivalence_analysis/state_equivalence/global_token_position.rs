//! Scanner-state partitions induced by positions within vocabulary tokens.
//!
//! This module contains two deliberately different relations. Do not conflate
//! them:
//!
//! * `GlobalTokenBoundaryStatePartition` groups states by exact destinations on
//!   every possible token-first byte. It is exact only when substituting a
//!   state immediately before the first byte of a complete nonempty vocabulary
//!   token.
//! * `GlobalTokenPositionStatePartition` is partition C, a total safe packing
//!   of partial partitions induced by individual token-byte positions. For
//!   positions before a configurable tail cutoff, only states active at that
//!   position are partitioned by their exact positional observation. Position
//!   zero observes only destinations on bytes occurring there; later positions
//!   additionally observe the current finalizer and future-finalizer sets.
//!   Active frontiers are propagated position-wise: `A_0` is every tokenizer
//!   state and `A_{i+1}` is obtained by stepping every state in `A_i` on every
//!   byte in the union `B_i` of bytes occurring at vocab position `i`. This is
//!   deliberately a per-position cross product, not a walk of individual vocab
//!   tokens and not restricted to tokens that continue past position `i`.
//!   States active at the cutoff or later are pairwise incompatible (the tail
//!   identity constraint). Inactive states are wildcards, not members of an
//!   invented "inactive" class.
//!
//! Pairwise compatibility of partial signatures is not transitive, so merely
//! graph-colouring compatible states does not produce a safe raw-representative
//! quotient. C instead uses positional subsumption: state `s` may map to raw
//! representative `r` only when every positional class known for `s` is also
//! known, with the same value, for `r`. Thus `r`'s partial signature extends
//! `s`'s. Distinct maximal signatures are forced representatives; every other
//! signature maps to one deterministic maximal extension. This gives the
//! minimum number of raw-representative classes under that subsumption rule.
//!
//! This is a directional representative relation, not an equivalence relation:
//! the representative extends every positional observation known for each
//! member, but two members need not extend each other.

use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{GlobalScannerStateQuotient, ManyToOneIdMap};
use crate::ds::bitset::BitSet;

const DEFAULT_TAIL_START_POSITION: usize = 3;
const MIN_TAIL_START_POSITION: usize = 3;

fn configured_tail_start_position() -> usize {
    match std::env::var("GLRMASK_GLOBAL_TOKEN_POSITION_TAIL_START") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&position| position >= MIN_TAIL_START_POSITION)
            .unwrap_or_else(|| {
                panic!(
                    "GLRMASK_GLOBAL_TOKEN_POSITION_TAIL_START must be at least 3; got {value:?}"
                )
            }),
        Err(std::env::VarError::NotPresent) => DEFAULT_TAIL_START_POSITION,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("GLRMASK_GLOBAL_TOKEN_POSITION_TAIL_START must be valid UTF-8")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GlobalTokenPositionEquivalenceProfile {
    pub(crate) tail_start_position: usize,
    pub(crate) position_byte_counts: Vec<usize>,
    pub(crate) position_active_state_counts: Vec<usize>,
    pub(crate) position_class_counts: Vec<usize>,
    pub(crate) tail_byte_count: usize,
    pub(crate) tail_active_state_count: usize,
    pub(crate) class_count: usize,
    pub(crate) build_ms: f64,
    pub(crate) total_ms: f64,
}

/// First-byte-only token-boundary equivalence.
///
/// Two states are equivalent here when every byte that can start a nonempty
/// vocabulary token takes them to the same exact scanner state/configuration.
/// This relation is exact for replacing the start state of a complete token
/// scan. It is not partition C and is not a global token-position relation.
#[derive(Debug, Clone)]
pub(crate) struct GlobalTokenBoundaryStatePartition {
    map: ManyToOneIdMap,
}

impl GlobalTokenBoundaryStatePartition {
    #[inline]
    pub(crate) fn as_many_to_one(&self) -> &ManyToOneIdMap {
        &self.map
    }
}

/// Global token-position partition C.
///
/// This is a total safe packing of partial positional partitions plus a tail
/// identity constraint. It is still not, by itself, a raw-byte right
/// congruence or a frozen-output-preserving DFA quotient.
#[derive(Debug, Clone)]
pub(crate) struct GlobalTokenPositionStatePartition {
    map: ManyToOneIdMap,
}

impl GlobalTokenPositionStatePartition {
    #[inline]
    pub(crate) fn as_many_to_one(&self) -> &ManyToOneIdMap {
        &self.map
    }
}

#[derive(Debug)]
struct PartialPositionPartition {
    /// Zero-based byte position within a vocabulary token.
    position: usize,
    byte_count: usize,
    active_state_count: usize,
    class_count: usize,
    /// `u32::MAX` means inactive at this position and therefore unconstrained
    /// by this partial partition.
    class_by_state: Vec<u32>,
}

#[derive(Debug)]
struct PositionByteSets {
    /// Bytes occurring exactly at zero-based positions `0..tail_start`.
    position_bytes: Vec<Vec<u8>>,
    /// Union of bytes occurring at the tail cutoff or any later position.
    tail_bytes: Vec<u8>,
}

#[derive(Debug)]
struct SignatureGroup {
    signature: PositionSignature,
    members: Vec<u32>,
    representative: u32,
}

type PositionSignature = SmallVec<[u32; 4]>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PositionObservationKey {
    /// Position zero begins a fresh scan, so current-state outputs are not
    /// observed there. Later positions observe both frozen output families.
    finalizers: Option<BitSet>,
    future_finalizers: Option<BitSet>,
    destinations: Box<[u32]>,
}

struct NfaTokenPositionView<'a> {
    tokenizer: &'a Tokenizer,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    raw_start_configs: Vec<u32>,
    transitions: FxHashMap<(u32, u8), u32>,
}

impl<'a> NfaTokenPositionView<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        let mut view = Self {
            tokenizer,
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            raw_start_configs: Vec::with_capacity(tokenizer.num_states() as usize),
            transitions: FxHashMap::default(),
        };
        for raw_state in 0..tokenizer.num_states() {
            let closure = tokenizer
                .execute_from_state_end_only(&[], raw_state)
                .to_vec();
            let config = view.intern_config(closure);
            view.raw_start_configs.push(config);
        }
        view
    }

    fn intern_config(&mut self, states: Vec<u32>) -> u32 {
        if let Some(&config) = self.config_ids.get(&states) {
            return config;
        }
        let config = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), config);
        self.configs.push(states.into_boxed_slice());
        config
    }

    #[inline]
    fn raw_start_config(&self, raw_state: usize) -> u32 {
        self.raw_start_configs[raw_state]
    }

    fn step(&mut self, config: u32, byte: u8) -> u32 {
        if let Some(&target) = self.transitions.get(&(config, byte)) {
            return target;
        }
        let source = self.configs[config as usize].to_vec();
        let targets = self.tokenizer.step_all(&source, byte);
        let target = if targets.is_empty() {
            u32::MAX
        } else {
            self.intern_config(targets.to_vec())
        };
        self.transitions.insert((config, byte), target);
        target
    }

    #[inline]
    fn states(&self, config: u32) -> &[u32] {
        &self.configs[config as usize]
    }

    fn has_finalizer(&self, config: u32) -> bool {
        self.states(config)
            .iter()
            .any(|&state| !self.tokenizer.matched_terminal_bitset(state).is_empty())
    }

    fn output_pair(&self, config: u32) -> (BitSet, BitSet) {
        let terminal_count = self.tokenizer.matched_terminal_bitset(0).len();
        let mut finalizers = BitSet::new(terminal_count);
        let mut future_finalizers = BitSet::new(terminal_count);
        for &state in self.states(config) {
            finalizers.union_with(self.tokenizer.matched_terminal_bitset(state));
            future_finalizers.union_with(self.tokenizer.possible_future_terminals(state));
        }
        (finalizers, future_finalizers)
    }

}

fn destination_rows(tokenizer: &Tokenizer, bytes: &[u8]) -> Vec<Box<[u32]>> {
    if !tokenizer.has_epsilon_transitions() {
        return (0..tokenizer.num_states())
            .map(|state| {
                bytes
                    .iter()
                    .map(|&byte| tokenizer.get_transition(state, byte))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect();
    }

    let mut view = NfaTokenPositionView::new(tokenizer);
    (0..tokenizer.num_states() as usize)
        .map(|state| {
            let source = view.raw_start_config(state);
            bytes
                .iter()
                .map(|&byte| view.step(source, byte))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect()
}

fn output_rows(tokenizer: &Tokenizer) -> Vec<(BitSet, BitSet)> {
    if !tokenizer.has_epsilon_transitions() {
        return (0..tokenizer.num_states())
            .map(|state| {
                (
                    tokenizer.matched_terminal_bitset(state).clone(),
                    tokenizer.possible_future_terminals(state).clone(),
                )
            })
            .collect();
    }

    let view = NfaTokenPositionView::new(tokenizer);
    (0..tokenizer.num_states() as usize)
        .map(|state| view.output_pair(view.raw_start_config(state)))
        .collect()
}

fn selected_bytes(mask: &[bool; 256]) -> Vec<u8> {
    (0..=u8::MAX)
        .filter(|&byte| mask[byte as usize])
        .collect()
}

fn position_byte_sets(vocab: &Vocab, tail_start_position: usize) -> PositionByteSets {
    assert!(tail_start_position >= MIN_TAIL_START_POSITION);
    let max_token_length = vocab.entries.values().map(Vec::len).max().unwrap_or(0);
    // Position `i` describes states active after exactly `i` position-wise
    // propagation steps. We need the boundary position after the final vocab
    // byte as well, because positions > 0 observe current F/U even when B_i is
    // empty. Thus a max token length L has explicit positions 0..=L when the
    // tail cutoff does not intervene.
    let explicit_positions = tail_start_position.min(max_token_length.saturating_add(1));
    let mut position_masks = vec![[false; 256]; explicit_positions];
    let mut tail_mask = [false; 256];

    for token in vocab.entries.values() {
        for (index, &byte) in token.iter().enumerate() {
            if index < tail_start_position {
                position_masks[index][byte as usize] = true;
            } else {
                tail_mask[byte as usize] = true;
            }
        }
    }

    PositionByteSets {
        position_bytes: position_masks.iter().map(selected_bytes).collect(),
        tail_bytes: selected_bytes(&tail_mask),
    }
}

fn deterministic_position_zero_partition(
    tokenizer: &Tokenizer,
    active_states: &[bool],
    bytes: &[u8],
) -> PartialPositionPartition {
    let state_count = tokenizer.num_states() as usize;
    let mut classes_by_hash = FxHashMap::<u64, Vec<(Box<[u32]>, u32)>>::default();
    let mut class_by_state = vec![u32::MAX; state_count];
    let mut class_count = 0u32;
    let mut row = [u32::MAX; 256];

    for (state, &active) in active_states.iter().enumerate() {
        if !active {
            continue;
        }
        let mut hash = 0x517c_c1b7_2722_0a95u64;
        for (index, &byte) in bytes.iter().enumerate() {
            let destination = tokenizer.get_transition(state as u32, byte);
            row[index] = destination;
            hash = hash
                .rotate_left(9)
                .wrapping_mul(0x9e37_79b1_85eb_ca87)
                ^ destination as u64;
        }
        let bucket = classes_by_hash.entry(hash).or_default();
        let class = bucket
            .iter()
            .find_map(|(candidate, class)| {
                (candidate.as_ref() == &row[..bytes.len()]).then_some(*class)
            })
            .unwrap_or_else(|| {
                let class = class_count;
                class_count += 1;
                bucket.push((row[..bytes.len()].to_vec().into_boxed_slice(), class));
                class
            });
        class_by_state[state] = class;
    }

    PartialPositionPartition {
        position: 0,
        byte_count: bytes.len(),
        active_state_count: active_states.iter().filter(|&&active| active).count(),
        class_count: class_count as usize,
        class_by_state,
    }
}

fn partial_position_partition(
    tokenizer: &Tokenizer,
    position: usize,
    active_states: &[bool],
    bytes: &[u8],
) -> PartialPositionPartition {
    let state_count = tokenizer.num_states() as usize;
    assert_eq!(active_states.len(), state_count);
    if position == 0 && !tokenizer.has_epsilon_transitions() {
        return deterministic_position_zero_partition(tokenizer, active_states, bytes);
    }
    let mut key_to_class = FxHashMap::<PositionObservationKey, u32>::default();
    let mut class_by_state = vec![u32::MAX; state_count];
    let mut class_count = 0u32;

    if !tokenizer.has_epsilon_transitions() {
        for state in 0..state_count {
            if !active_states[state] {
                continue;
            }
            let (finalizers, future_finalizers) = if position > 0 {
                (
                    Some(tokenizer.matched_terminal_bitset(state as u32).clone()),
                    Some(tokenizer.possible_future_terminals(state as u32).clone()),
                )
            } else {
                (None, None)
            };
            let key = PositionObservationKey {
                finalizers,
                future_finalizers,
                destinations: bytes
                    .iter()
                    .map(|&byte| tokenizer.get_transition(state as u32, byte))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            };
            let next = class_count;
            let class = *key_to_class.entry(key).or_insert_with(|| {
                class_count += 1;
                next
            });
            class_by_state[state] = class;
        }
        return PartialPositionPartition {
            position,
            byte_count: bytes.len(),
            active_state_count: active_states.iter().filter(|&&active| active).count(),
            class_count: class_count as usize,
            class_by_state,
        };
    }

    let rows = destination_rows(tokenizer, bytes);
    let outputs = (position > 0).then(|| output_rows(tokenizer));
    for state in 0..state_count {
        if !active_states[state] {
            continue;
        }
        let (finalizers, future_finalizers) = outputs
            .as_ref()
            .map(|rows| {
                let (finalizers, future_finalizers) = &rows[state];
                (Some(finalizers.clone()), Some(future_finalizers.clone()))
            })
            .unwrap_or((None, None));
        let key = PositionObservationKey {
            finalizers,
            future_finalizers,
            destinations: rows[state].clone(),
        };
        let next = class_count;
        let class = *key_to_class.entry(key).or_insert_with(|| {
            class_count += 1;
            next
        });
        class_by_state[state] = class;
    }

    PartialPositionPartition {
        position,
        byte_count: bytes.len(),
        active_state_count: active_states.iter().filter(|&&active| active).count(),
        class_count: class_count as usize,
        class_by_state,
    }
}

/// Advance one positional frontier through the whole position-byte set `B_i`.
/// Every active state is crossed with every byte occurring at that vocab
/// position; token identity and whether a token ends at this position are not
/// tracked. A post-byte
/// finalizer contributes both the ordinary scanner destination and the lexer
/// reset roots, matching the two continuation branches in the terminal-NWA
/// builder.
fn advance_active_states_with_flat_trans(
    tokenizer: &Tokenizer,
    active_states: &[bool],
    bytes: &[u8],
    flat_trans: Option<&[u32]>,
) -> Vec<bool> {
    let state_count = tokenizer.num_states() as usize;
    let mut next = vec![false; state_count];
    if bytes.is_empty() {
        return next;
    }

    if tokenizer.has_epsilon_transitions() {
        let mut view = NfaTokenPositionView::new(tokenizer);
        let mut reset_needed = false;
        for (state, &active) in active_states.iter().enumerate() {
            if !active {
                continue;
            }
            let source = view.raw_start_config(state);
            for &byte in bytes {
                let destination = view.step(source, byte);
                if destination == u32::MAX {
                    continue;
                }
                reset_needed |= view.has_finalizer(destination);
                for &target in view.states(destination) {
                    next[target as usize] = true;
                }
            }
        }
        if reset_needed {
            for reset_state in tokenizer.deterministic_reset_states() {
                let reset = view.raw_start_config(reset_state as usize);
                for &state in view.states(reset) {
                    next[state as usize] = true;
                }
            }
        }
        return next;
    }

    let mut reset_needed = false;
    for (state, &active) in active_states.iter().enumerate() {
        if !active {
            continue;
        }
        for &byte in bytes {
            let destination = flat_trans.map_or_else(
                || tokenizer.get_transition(state as u32, byte),
                |transitions| transitions[state * 256 + byte as usize],
            );
            if destination == u32::MAX {
                continue;
            }
            next[destination as usize] = true;
            reset_needed |= !tokenizer.matched_terminal_bitset(destination).is_empty();
        }
    }
    if reset_needed {
        for reset_state in tokenizer.deterministic_reset_states() {
            next[reset_state as usize] = true;
        }
    }
    next
}

fn advance_active_states(
    tokenizer: &Tokenizer,
    active_states: &[bool],
    bytes: &[u8],
) -> Vec<bool> {
    advance_active_states_with_flat_trans(tokenizer, active_states, bytes, None)
}

/// Overapproximate the union of all active-state sets from `tail_start` onward.
/// The seed is the exact positional frontier computed up to the cutoff. Once
/// positions are collapsed into one tail coordinate, only bytes that occur at
/// the cutoff or later are traversed. Terminal matches add zero-byte reset
/// edges; reset changes scanner state, never the LLM-token position.
fn tail_active_states_with_flat_trans(
    tokenizer: &Tokenizer,
    seed: &[bool],
    tail_bytes: &[u8],
    flat_trans: Option<&[u32]>,
) -> Vec<bool> {
    let state_count = tokenizer.num_states() as usize;
    assert_eq!(seed.len(), state_count);
    let mut reached = seed.to_vec();
    let mut worklist = seed
        .iter()
        .enumerate()
        .filter_map(|(state, &active)| active.then_some(state as u32))
        .collect::<Vec<_>>();

    if tokenizer.has_epsilon_transitions() {
        let mut view = NfaTokenPositionView::new(tokenizer);
        while let Some(state) = worklist.pop() {
            let source = view.raw_start_config(state as usize);
            if view.has_finalizer(source) {
                for reset_state in tokenizer.deterministic_reset_states() {
                    let reset = view.raw_start_config(reset_state as usize);
                    for &target in view.states(reset) {
                        if !reached[target as usize] {
                            reached[target as usize] = true;
                            worklist.push(target);
                        }
                    }
                }
            }
            for &byte in tail_bytes {
                let destination = view.step(source, byte);
                if destination == u32::MAX {
                    continue;
                }
                let targets = view.states(destination).to_vec();
                for target in targets {
                    if !reached[target as usize] {
                        reached[target as usize] = true;
                        worklist.push(target);
                    }
                }
            }
        }
        return reached;
    }

    while let Some(state) = worklist.pop() {
        if !tokenizer.matched_terminal_bitset(state).is_empty() {
            for reset_state in tokenizer.deterministic_reset_states() {
                if !reached[reset_state as usize] {
                    reached[reset_state as usize] = true;
                    worklist.push(reset_state);
                }
            }
        }
        for &byte in tail_bytes {
            let destination = flat_trans.map_or_else(
                || tokenizer.get_transition(state, byte),
                |transitions| transitions[state as usize * 256 + byte as usize],
            );
            if destination == u32::MAX {
                continue;
            }
            if !reached[destination as usize] {
                reached[destination as usize] = true;
                worklist.push(destination);
            }
        }
    }
    reached
}

fn tail_active_states(
    tokenizer: &Tokenizer,
    seed: &[bool],
    tail_bytes: &[u8],
) -> Vec<bool> {
    tail_active_states_with_flat_trans(tokenizer, seed, tail_bytes, None)
}

fn signature_restrictions(
    signature: &[u32],
    include_self: bool,
) -> Vec<PositionSignature> {
    let known_coordinates = signature
        .iter()
        .enumerate()
        .filter_map(|(coordinate, &value)| (value != u32::MAX).then_some(coordinate))
        .collect::<Vec<_>>();
    let restriction_count = 1usize << known_coordinates.len();
    let first_drop_mask = usize::from(!include_self);
    let mut restrictions = Vec::with_capacity(restriction_count - first_drop_mask);
    for drop_mask in first_drop_mask..restriction_count {
        let mut restricted = PositionSignature::from_slice(signature);
        for (bit, &coordinate) in known_coordinates.iter().enumerate() {
            if (drop_mask & (1usize << bit)) != 0 {
                restricted[coordinate] = u32::MAX;
            }
        }
        restrictions.push(restricted);
    }
    restrictions
}

#[inline]
fn signature_is_restriction_of(restriction: &[u32], extension: &[u32]) -> bool {
    debug_assert_eq!(restriction.len(), extension.len());
    restriction
        .iter()
        .zip(extension)
        .all(|(&left, &right)| left == u32::MAX || left == right)
}

fn state_partial_signature(
    state: usize,
    later_partitions: &[PartialPositionPartition],
    tail_active: &[bool],
) -> PositionSignature {
    let mut signature = later_partitions
        .iter()
        .map(|partition| partition.class_by_state[state])
        .collect::<PositionSignature>();
    signature.push(
        tail_active[state]
            .then_some(state as u32)
            .unwrap_or(u32::MAX),
    );
    signature
}

/// Complete the partial positional constraints by raw-representative
/// subsumption.
///
/// Within one total position-one class, write `s <= r` when every known
/// coordinate of `s` is also known with the same value in `r`. Then `r` can
/// safely stand in for `s` at every token position where `s` is active. The
/// tail identity constraint is encoded as one extra partial coordinate whose
/// value is the raw state ID, so a tail-active state can only be represented by
/// itself.
///
/// Every finite partial-signature poset has maximal elements, and every
/// signature is below at least one maximal element. Every distinct maximal
/// signature is also a forced representative: by maximality no different
/// signature can represent it. Therefore one raw representative per maximal
/// signature is both sufficient and minimal under this subsumption rule.
fn pack_partial_partitions(
    partitions: &[PartialPositionPartition],
    tail_active: &[bool],
) -> ManyToOneIdMap {
    let first = partitions
        .first()
        .expect("position-one partition must exist");
    let state_count = first.class_by_state.len();
    assert_eq!(tail_active.len(), state_count);
    assert!(first.class_by_state.iter().all(|&class| class != u32::MAX));

    let later = &partitions[1..];
    let mut states_by_first_class = vec![Vec::<u32>::new(); first.class_count];
    for (state, &class) in first.class_by_state.iter().enumerate() {
        states_by_first_class[class as usize].push(state as u32);
    }

    let mut global_classes = Vec::<(u32, Vec<u32>)>::new();

    for first_block in states_by_first_class {
        if first_block.is_empty() {
            continue;
        }
        let mut signature_to_group = FxHashMap::<PositionSignature, usize>::default();
        let mut groups = Vec::<SignatureGroup>::new();
        for state in first_block {
            let signature = state_partial_signature(state as usize, later, tail_active);
            if let Some(&group) = signature_to_group.get(&signature) {
                groups[group].members.push(state);
                continue;
            }
            let group = groups.len();
            signature_to_group.insert(signature.clone(), group);
            groups.push(SignatureGroup {
                signature,
                members: vec![state],
                representative: state,
            });
        }

        let use_enumerated_restrictions = groups
            .iter()
            .map(|group| {
                group
                    .signature
                    .iter()
                    .filter(|&&coordinate| coordinate != u32::MAX)
                    .count()
            })
            .max()
            .unwrap_or(0)
            <= 8;

        let mut maximal_groups = if use_enumerated_restrictions {
            let mut strict_restrictions = FxHashSet::<PositionSignature>::default();
            for group in &groups {
                strict_restrictions.extend(signature_restrictions(&group.signature, false));
            }
            groups
                .iter()
                .enumerate()
                .filter_map(|(group, data)| {
                    (!strict_restrictions.contains(&data.signature)).then_some(group)
                })
                .collect::<Vec<_>>()
        } else {
            groups
                .iter()
                .enumerate()
                .filter_map(|(candidate, data)| {
                    let has_strict_extension = groups.iter().enumerate().any(|(other, extension)| {
                        candidate != other
                            && signature_is_restriction_of(&data.signature, &extension.signature)
                    });
                    (!has_strict_extension).then_some(candidate)
                })
                .collect::<Vec<_>>()
        };
        maximal_groups.sort_unstable_by_key(|&group| groups[group].representative);

        let maximal_extension_by_group = if use_enumerated_restrictions {
            let mut maximal_extension_for_restriction =
                FxHashMap::<PositionSignature, usize>::default();
            for &group in &maximal_groups {
                for restriction in signature_restrictions(&groups[group].signature, true) {
                    maximal_extension_for_restriction
                        .entry(restriction)
                        .or_insert(group);
                }
            }
            groups
                .iter()
                .map(|group| {
                    *maximal_extension_for_restriction
                        .get(&group.signature)
                        .expect("every partial signature must have a maximal extension")
                })
                .collect::<Vec<_>>()
        } else {
            groups
                .iter()
                .map(|group| {
                    maximal_groups
                        .iter()
                        .copied()
                        .find(|&maximal| {
                            signature_is_restriction_of(
                                &group.signature,
                                &groups[maximal].signature,
                            )
                        })
                        .expect("every partial signature must have a maximal extension")
                })
                .collect::<Vec<_>>()
        };

        let mut class_for_maximal_group = FxHashMap::<usize, usize>::default();
        for &group in &maximal_groups {
            let class = global_classes.len();
            class_for_maximal_group.insert(group, class);
            global_classes.push((groups[group].representative, Vec::new()));
        }
        for (group_index, group) in groups.into_iter().enumerate() {
            let maximal_group = maximal_extension_by_group[group_index];
            let class = class_for_maximal_group[&maximal_group];
            global_classes[class].1.extend(group.members);
        }
    }

    global_classes.sort_unstable_by_key(|(representative, _)| *representative);
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut representative_original_ids = Vec::with_capacity(global_classes.len());
    let mut internal_to_originals = Vec::with_capacity(global_classes.len());
    for (class, (representative, mut members)) in global_classes.into_iter().enumerate() {
        members.sort_unstable();
        representative_original_ids.push(representative);
        for &state in &members {
            original_to_internal[state as usize] = class as u32;
        }
        internal_to_originals.push(members);
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn first_destination_partition(
    tokenizer: &Tokenizer,
    first_bytes: &[u8],
) -> ManyToOneIdMap {
    let state_count = tokenizer.num_states() as usize;
    let first_destinations = destination_rows(tokenizer, first_bytes);
    let mut key_to_class = FxHashMap::<Box<[u32]>, u32>::default();
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for state in 0..state_count {
        let destinations = first_destinations[state].clone();
        let next = internal_to_originals.len() as u32;
        let class = *key_to_class.entry(destinations).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(state as u32);
            next
        });
        original_to_internal[state] = class;
        internal_to_originals[class as usize].push(state as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

/// Build first-byte-only token-boundary equivalence.
///
/// If two states agree here, every complete nonempty vocabulary token reaches
/// the same exact scanner state/configuration after byte one and therefore has
/// the same remaining scanner trajectory. This theorem is about substituting
/// states at token start; it does not make the relation global across positions
/// inside the token.
pub(crate) fn compute_global_token_boundary_state_partition(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> Option<GlobalTokenBoundaryStatePartition> {
    if vocab.entries.values().any(Vec::is_empty) {
        return None;
    }
    let first_bytes = position_byte_sets(vocab, MIN_TAIL_START_POSITION)
        .position_bytes
        .into_iter()
        .next()
        .unwrap_or_default();
    (!first_bytes.is_empty()).then(|| GlobalTokenBoundaryStatePartition {
        map: first_destination_partition(tokenizer, &first_bytes),
    })
}

fn compute_global_token_position_map_with_tail_start_and_flat_trans(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    tail_start_position: usize,
    flat_trans: Option<&[u32]>,
) -> Option<(ManyToOneIdMap, GlobalTokenPositionEquivalenceProfile)> {
    let started_at = Instant::now();
    if vocab.entries.values().any(Vec::is_empty) {
        return None;
    }
    let byte_sets = position_byte_sets(vocab, tail_start_position);
    if byte_sets
        .position_bytes
        .first()
        .is_none_or(Vec::is_empty)
    {
        return None;
    }
    let state_count = tokenizer.num_states() as usize;
    let flat_trans = flat_trans.filter(|transitions| {
        !tokenizer.has_epsilon_transitions() && transitions.len() == state_count * 256
    });
    let mut active_states = vec![true; state_count];
    let mut partitions = Vec::with_capacity(byte_sets.position_bytes.len());
    let profile_phases = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let mut partition_ms = Vec::with_capacity(byte_sets.position_bytes.len());
    let mut advance_ms = Vec::with_capacity(byte_sets.position_bytes.len());
    for (index, bytes) in byte_sets.position_bytes.iter().enumerate() {
        let partition_started_at = profile_phases.then(Instant::now);
        partitions.push(partial_position_partition(
            tokenizer,
            index,
            &active_states,
            bytes,
        ));
        partition_ms.push(
            partition_started_at.map_or(0.0, |started_at| {
                started_at.elapsed().as_secs_f64() * 1000.0
            }),
        );
        let advance_started_at = profile_phases.then(Instant::now);
        active_states = advance_active_states_with_flat_trans(
            tokenizer,
            &active_states,
            bytes,
            flat_trans,
        );
        advance_ms.push(
            advance_started_at.map_or(0.0, |started_at| {
                started_at.elapsed().as_secs_f64() * 1000.0
            }),
        );
    }
    let tail_started_at = profile_phases.then(Instant::now);
    let tail_active = tail_active_states_with_flat_trans(
        tokenizer,
        &active_states,
        &byte_sets.tail_bytes,
        flat_trans,
    );
    let tail_ms = tail_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    let pack_started_at = profile_phases.then(Instant::now);
    let map = pack_partial_partitions(&partitions, &tail_active);
    let pack_ms = pack_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let profile = GlobalTokenPositionEquivalenceProfile {
        tail_start_position,
        position_byte_counts: partitions
            .iter()
            .map(|partition| partition.byte_count)
            .collect(),
        position_active_state_counts: partitions
            .iter()
            .map(|partition| partition.active_state_count)
            .collect(),
        position_class_counts: partitions
            .iter()
            .map(|partition| partition.class_count)
            .collect(),
        tail_byte_count: byte_sets.tail_bytes.len(),
        tail_active_state_count: tail_active.iter().filter(|&&active| active).count(),
        class_count: map.num_internal_ids() as usize,
        build_ms,
        total_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    };
    if profile_phases {
        eprintln!(
            "[glrmask/profile][global_token_position_phases] position_bytes={:?} position_active_states={:?} position_classes={:?} position_partition_ms={:?} position_advance_ms={:?} tail_ms={:.3} pack_ms={:.3}",
            partitions
                .iter()
                .map(|partition| partition.byte_count)
                .collect::<Vec<_>>(),
            partitions
                .iter()
                .map(|partition| partition.active_state_count)
                .collect::<Vec<_>>(),
            partitions
                .iter()
                .map(|partition| partition.class_count)
                .collect::<Vec<_>>(),
            partition_ms,
            advance_ms,
            tail_ms,
            pack_ms,
        );
    }
    Some((map, profile))
}

fn compute_global_token_position_map_with_tail_start(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    tail_start_position: usize,
) -> Option<(ManyToOneIdMap, GlobalTokenPositionEquivalenceProfile)> {
    compute_global_token_position_map_with_tail_start_and_flat_trans(
        tokenizer,
        vocab,
        tail_start_position,
        None,
    )
}

/// Build global token-position partition C.
///
/// Zero-based positions before `GLRMASK_GLOBAL_TOKEN_POSITION_TAIL_START`
/// (default 3) contribute partial observation partitions over only their active
/// states. Position zero observes exact destinations only. Later explicit
/// positions additionally observe current finalizers and future-finalizers.
/// The cutoff and all later positions contribute one tail identity constraint:
/// every pair of distinct tail-active states is forbidden from merging.
/// Inactive states remain unconstrained at that position. Any cutoff >= 3 is
/// accepted; `usize::MAX` means every vocabulary position is explicit and the
/// conservative tail is empty.
pub(crate) fn compute_global_token_position_state_partition(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> Option<GlobalTokenPositionStatePartition> {
    compute_global_token_position_map_with_tail_start(
        tokenizer,
        vocab,
        configured_tail_start_position(),
    )
    .map(|(map, _)| GlobalTokenPositionStatePartition { map })
}

/// Build both production views of partition C from one positional analysis.
///
/// The quotient and token-position partition wrap the same total raw-state map.
/// Consumers that need both must use this entry point rather than independently
/// rebuilding C through the two legacy wrappers below.
pub(crate) fn compute_global_token_position_state_views(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    flat_trans: Option<&[u32]>,
) -> Option<(
    GlobalScannerStateQuotient,
    GlobalTokenPositionStatePartition,
    GlobalTokenPositionEquivalenceProfile,
)> {
    let started_at = Instant::now();
    let state_count = tokenizer.num_states() as usize;
    let (map, mut profile) = compute_global_token_position_map_with_tail_start_and_flat_trans(
        tokenizer,
        vocab,
        configured_tail_start_position(),
        flat_trans,
    )?;
    let partition = GlobalTokenPositionStatePartition { map: map.clone() };
    let quotient = GlobalScannerStateQuotient::from_total_raw_state_map(map, state_count);
    profile.total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][global_token_position_quotient] raw_states={} tail_start={} position_bytes={:?} position_active_states={:?} position_classes={:?} tail_bytes={} tail_active_states={} classes={} build_ms={:.3} total_ms={:.3}",
            state_count,
            profile.tail_start_position,
            profile.position_byte_counts,
            profile.position_active_state_counts,
            profile.position_class_counts,
            profile.tail_byte_count,
            profile.tail_active_state_count,
            profile.class_count,
            profile.build_ms,
            profile.total_ms,
        );
    }
    Some((quotient, partition, profile))
}
}

/// Wrap global token-position map C for the pre-TI pipeline.
pub(crate) fn compute_global_token_position_state_quotient(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> (GlobalScannerStateQuotient, GlobalTokenPositionEquivalenceProfile) {
    let started_at = Instant::now();
    let state_count = tokenizer.num_states() as usize;
    let (map, mut profile) = compute_global_token_position_map_with_tail_start(
        tokenizer,
        vocab,
        configured_tail_start_position(),
    )
    .expect("global token-position quotient requires nonempty vocabulary tokens");
    let quotient = GlobalScannerStateQuotient::from_total_raw_state_map(map, state_count);
    profile.total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][global_token_position_quotient] raw_states={} tail_start={} position_bytes={:?} position_active_states={:?} position_classes={:?} tail_bytes={} tail_active_states={} classes={} build_ms={:.3} total_ms={:.3}",
            state_count,
            profile.tail_start_position,
            profile.position_byte_counts,
            profile.position_active_state_counts,
            profile.position_class_counts,
            profile.tail_byte_count,
            profile.tail_active_state_count,
            profile.class_count,
            profile.build_ms,
            profile.total_ms,
        );
    }
    (quotient, profile)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

    fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
        let terminal_count = expressions.len() as u32;
        build_regex(&expressions).into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    fn vocab(entries: &[(u32, &[u8])]) -> Vocab {
        Vocab::new(
            entries
                .iter()
                .map(|&(token, bytes)| (token, bytes.to_vec()))
                .collect(),
            None,
        )
    }

    fn assert_positional_constraints_hold(
        tokenizer: &Tokenizer,
        vocab: &Vocab,
        tail_start: usize,
        map: &ManyToOneIdMap,
    ) {
        let byte_sets = position_byte_sets(vocab, tail_start);
        let mut active = vec![true; tokenizer.num_states() as usize];
        let mut partitions = Vec::new();
        for (index, bytes) in byte_sets.position_bytes.iter().enumerate() {
            partitions.push(partial_position_partition(
                tokenizer,
                index,
                &active,
                bytes,
            ));
            active = advance_active_states(tokenizer, &active, bytes);
        }
        let tail = tail_active_states(tokenizer, &active, &byte_sets.tail_bytes);

        for (class, members) in map.internal_to_originals.iter().enumerate() {
            let representative = map.representative_original_ids[class];
            assert!(members.contains(&representative));
            let tail_members = members
                .iter()
                .copied()
                .filter(|&state| tail[state as usize])
                .collect::<Vec<_>>();
            assert!(
                tail_members.len() <= 1,
                "one C class contained two distinct tail-active states: {members:?}",
            );
            if let Some(&tail_member) = tail_members.first() {
                assert_eq!(
                    representative, tail_member,
                    "a tail-active member can only be represented by itself",
                );
            }
            for partition in &partitions {
                let representative_class =
                    partition.class_by_state[representative as usize];
                for &state in members {
                    let class = partition.class_by_state[state as usize];
                    if class == u32::MAX {
                        continue;
                    }
                    assert_ne!(
                        representative_class,
                        u32::MAX,
                        "representative {representative} is inactive at position {} where member {state} is active",
                        partition.position,
                    );
                    assert_eq!(
                        class, representative_class,
                        "representative {representative} does not extend member {state} at position {}",
                        partition.position,
                    );
                }
            }
        }
    }

    #[test]
    fn token_boundary_classes_share_every_nonempty_token_trajectory() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"d".to_vec()),
            Expr::U8Seq(b"db".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"ab"), (1, b"ac"), (2, b"d"), (3, b"db")]);
        let partition = compute_global_token_boundary_state_partition(&tokenizer, &vocab)
            .expect("all fixture tokens are nonempty");
        let map = partition.as_many_to_one();

        for members in &map.internal_to_originals {
            let representative = *members.first().expect("total quotient class") as usize;
            for &state in members {
                for token in vocab.entries.values() {
                    let mut left = Some(state);
                    let mut right = Some(representative as u32);
                    for &byte in token {
                        left = left.and_then(|current| tokenizer.step(current, byte));
                        right = right.and_then(|current| tokenizer.step(current, byte));
                        assert_eq!(left, right, "boundary states diverged on token {:?}", token);
                    }
                }
            }
        }
    }

    #[test]
    fn global_c_keeps_genuine_tail_states_pairwise_distinct() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"abcd".to_vec()),
            Expr::U8Seq(b"x".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"abc")]);
        let initial = tokenizer.initial_state_id();
        let after_a = tokenizer.step(initial, b'a').expect("a transition");
        let after_ab = tokenizer.step(after_a, b'b').expect("ab transition");
        let after_abc = tokenizer.step(after_ab, b'c').expect("abc transition");

        let boundary = compute_global_token_boundary_state_partition(&tokenizer, &vocab)
            .expect("fixture token is nonempty");
        let boundary_map = boundary.as_many_to_one();
        assert_eq!(
            boundary_map.original_to_internal[after_ab as usize],
            boundary_map.original_to_internal[after_abc as usize],
            "first-byte-only boundary equivalence should merge states with the same 'a' destination",
        );

        let global = compute_global_token_position_state_partition(&tokenizer, &vocab)
            .expect("fixture token is nonempty");
        let global_map = global.as_many_to_one();
        assert_ne!(
            global_map.original_to_internal[after_ab as usize],
            global_map.original_to_internal[after_abc as usize],
            "distinct states active in the collapsed position-3+ tail are pairwise incompatible",
        );
    }

    #[test]
    fn inactive_position_is_a_wildcard_not_an_inactive_class() {
        let first = PartialPositionPartition {
            position: 0,
            byte_count: 1,
            active_state_count: 3,
            class_count: 1,
            class_by_state: vec![0, 0, 0],
        };
        let second = PartialPositionPartition {
            position: 1,
            byte_count: 1,
            active_state_count: 2,
            class_count: 2,
            class_by_state: vec![0, u32::MAX, 1],
        };
        let map = pack_partial_partitions(&[first, second], &[false; 3]);

        assert_ne!(map.original_to_internal[0], map.original_to_internal[2]);
        assert_eq!(
            map.original_to_internal[0], map.original_to_internal[1],
            "the position-2-inactive state should be free to join a compatible active class",
        );
        assert_eq!(map.num_internal_ids(), 2);
    }

    #[test]
    fn later_positions_observe_current_finalizers_and_futures() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"ax".to_vec()),
            Expr::U8Seq(b"bx".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"ax")]);
        let initial = tokenizer.initial_state_id();
        let after_a = tokenizer.step(initial, b'a').expect("a transition");
        let after_b = tokenizer.step(initial, b'b').expect("b transition");
        let after_ax = tokenizer.step(after_a, b'x').expect("ax transition");
        let after_bx = tokenizer.step(after_b, b'x').expect("bx transition");

        assert_ne!(
            tokenizer.matched_terminal_bitset(after_ax),
            tokenizer.matched_terminal_bitset(after_bx),
        );
        let active = vec![true; tokenizer.num_states() as usize];
        let position_zero = partial_position_partition(&tokenizer, 0, &active, b"x");
        let position_one = partial_position_partition(&tokenizer, 1, &active, b"x");

        assert_eq!(
            position_zero.class_by_state[after_ax as usize],
            position_zero.class_by_state[after_bx as usize],
            "position zero intentionally ignores current-state outputs",
        );
        assert_ne!(
            position_one.class_by_state[after_ax as usize],
            position_one.class_by_state[after_bx as usize],
            "later positions must preserve finalizer/future observation",
        );
    }

    #[test]
    fn positionwise_frontier_keeps_post_byte_finalizers_distinct() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"ax".to_vec()),
            Expr::U8Seq(b"bx".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"x")]);
        let initial = tokenizer.initial_state_id();
        let after_a = tokenizer.step(initial, b'a').expect("a transition");
        let after_b = tokenizer.step(initial, b'b').expect("b transition");
        let after_ax = tokenizer.step(after_a, b'x').expect("ax transition");
        let after_bx = tokenizer.step(after_b, b'x').expect("bx transition");

        let byte_sets = position_byte_sets(&vocab, 3);
        assert_eq!(byte_sets.position_bytes[0], b"x");
        let active_zero = vec![true; tokenizer.num_states() as usize];
        let active_one = advance_active_states(
            &tokenizer,
            &active_zero,
            &byte_sets.position_bytes[0],
        );
        assert!(active_one[after_ax as usize]);
        assert!(active_one[after_bx as usize]);
        let position_one = partial_position_partition(
            &tokenizer,
            1,
            &active_one,
            &byte_sets.position_bytes[1],
        );
        assert_ne!(
            position_one.class_by_state[after_ax as usize],
            position_one.class_by_state[after_bx as usize],
            "states reached through B0 must be compared at position 1 even when B1 is empty",
        );

        let c = compute_global_token_position_state_partition(&tokenizer, &vocab)
            .expect("nonempty vocab");
        assert_ne!(
            c.as_many_to_one().original_to_internal[after_ax as usize],
            c.as_many_to_one().original_to_internal[after_bx as usize],
            "position-wise propagation on B0={{x}} makes both final states active at position 1, where their F rows differ",
        );

        let (ti_quotient, _) =
            compute_global_token_position_state_quotient(&tokenizer, &vocab);
        assert_ne!(
            ti_quotient.as_many_to_one().original_to_internal[after_ax as usize],
            ti_quotient.as_many_to_one().original_to_internal[after_bx as usize],
            "the TI-facing C map must preserve the distinct position-1 classes",
        );
    }

    #[test]
    fn arbitrarily_late_tail_cutoff_uses_all_real_vocab_positions() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"abcdef".to_vec()),
            Expr::U8Seq(b"abcxef".to_vec()),
            Expr::U8Seq(b"x".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"abcdef"), (1, b"abcxef")]);
        let (map, profile) = compute_global_token_position_map_with_tail_start(
            &tokenizer,
            &vocab,
            usize::MAX,
        )
        .expect("nonempty vocab");

        assert_eq!(profile.tail_start_position, usize::MAX);
        assert_eq!(profile.position_byte_counts.len(), 7);
        assert_eq!(profile.tail_byte_count, 0);
        assert_eq!(profile.tail_active_state_count, 0);
        assert_positional_constraints_hold(&tokenizer, &vocab, usize::MAX, &map);
    }

    #[test]
    fn tail_reachability_follows_terminal_reset_after_byte_two() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"abc".to_vec()),
            Expr::U8Seq(b"d".to_vec()),
            Expr::U8Seq(b"x".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"abcd"), (1, b"abcx")]);
        let bytes = position_byte_sets(&vocab, 3);
        let mut active = vec![true; tokenizer.num_states() as usize];
        active = advance_active_states(&tokenizer, &active, &bytes.position_bytes[0]);
        active = advance_active_states(&tokenizer, &active, &bytes.position_bytes[1]);
        active = advance_active_states(&tokenizer, &active, &bytes.position_bytes[2]);
        let tail = tail_active_states(&tokenizer, &active, &bytes.tail_bytes);

        let reset = tokenizer.initial_state_id();
        let after_d = tokenizer.step(reset, b'd').expect("reset d transition");
        let after_x = tokenizer.step(reset, b'x').expect("reset x transition");
        assert!(tail[reset as usize], "position-two match must add reset to A3");
        assert!(tail[after_d as usize]);
        assert!(tail[after_x as usize]);

        let (map, profile) =
            compute_global_token_position_map_with_tail_start(&tokenizer, &vocab, 3)
                .expect("nonempty vocab");
        assert_eq!(profile.tail_start_position, 3);
        assert_ne!(
            map.original_to_internal[after_d as usize],
            map.original_to_internal[after_x as usize],
            "two distinct tail-active states must remain pairwise incompatible",
        );
    }

    #[test]
    fn tail_start_positions_three_four_and_five_are_supported() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"abcdef".to_vec()),
            Expr::U8Seq(b"abcxef".to_vec()),
            Expr::U8Seq(b"x".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"abcdef"), (1, b"abcxef")]);

        let mut tail_active_counts = Vec::new();
        let mut class_counts = Vec::new();
        for tail_start in 3..=5 {
            let (map, profile) =
                compute_global_token_position_map_with_tail_start(&tokenizer, &vocab, tail_start)
                    .expect("nonempty vocab");
            assert_eq!(profile.tail_start_position, tail_start);
            assert_eq!(map.original_to_internal.len(), tokenizer.num_states() as usize);
            assert!(map.original_to_internal.iter().all(|&class| class != u32::MAX));
            assert_positional_constraints_hold(&tokenizer, &vocab, tail_start, &map);
            tail_active_counts.push(profile.tail_active_state_count);
            class_counts.push(profile.class_count);
        }
        assert_eq!(tail_active_counts, vec![9, 7, 5]);
        assert!(class_counts.windows(2).all(|pair| pair[1] <= pair[0]));
    }

    #[test]
    fn epsilon_token_boundary_classes_share_exact_state_set_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let vocab = vocab(&[(0, b"a"), (1, b"b"), (2, b"aa"), (3, b"ba")]);
        let partition = compute_global_token_boundary_state_partition(&tokenizer, &vocab)
            .expect("all fixture tokens are nonempty");
        let map = partition.as_many_to_one();

        for members in &map.internal_to_originals {
            let representative = *members.first().expect("total quotient class");
            for &state in members {
                for token in vocab.entries.values() {
                    let mut left = tokenizer.execute_from_state_end_only(&[], state);
                    let mut right = tokenizer.execute_from_state_end_only(&[], representative);
                    for &byte in token {
                        left = tokenizer.step_all(&left, byte);
                        right = tokenizer.step_all(&right, byte);
                        assert_eq!(
                            left, right,
                            "epsilon boundary states diverged on token {token:?}",
                        );
                    }
                }
            }
        }

        let (quotient, _) = compute_global_token_position_state_quotient(&tokenizer, &vocab);
        assert_eq!(quotient.raw_state_count(), tokenizer.num_states() as usize);
    }
}
