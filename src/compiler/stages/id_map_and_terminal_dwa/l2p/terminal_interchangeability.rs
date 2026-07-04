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

use std::collections::{BTreeMap, hash_map::Entry};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use super::nwa_builder::TerminalNwaTransportMode;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::grammar::flat::TerminalID;

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

/// One observed frozen-output label. The reference only reads these labels
/// at enabled-byte destinations, so this type is used solely for the global
/// destination-output closure prefilter below.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct OutputPair {
    finalizers: OutputBits,
    future_finalizers: OutputBits,
}

/// One exact component of the sparse characterization tuple. The output id
/// names an immutable pair of frozen destination-output sets. Class ids are
/// only equality labels within one round; hash-map equality still compares the
/// complete tuple, so this avoids cryptographic hashing without relying on hash
/// collisions for semantics.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CanonicalComponent {
    byte: u8,
    previous_class: u32,
    output: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CanonicalSignature(Vec<CanonicalComponent>);

struct CanonicalRound {
    classes: Vec<u32>,
    signatures: FxHashMap<CanonicalSignature, u32>,
}

/// Stable identity characterization collapsed to its exact state quotient.
/// The quotient does not merge tokenizer states; it is only a cached view used
/// to certify a terminal-swap automorphism before materializing its raw map.
struct CanonicalQuotient {
    class_for_state: Vec<u32>,
    representative_by_class: Vec<u32>,
    class_by_signature: FxHashMap<CanonicalSignature, u32>,
    reverse_predecessors: Vec<Vec<u32>>,
    /// Identity class labels, projected onto stable quotient representatives,
    /// for rounds 0 through the identity fixed point.
    identity_classes_by_round: Vec<Vec<u32>>,
    /// Multiplicity of each identity class at the matching round. This lets a
    /// sparse swapped update prove class-set equality without scanning every
    /// quotient class.
    identity_class_counts_by_round: Vec<Vec<u32>>,
    /// At the identity fixed point, the preceding class partition maps
    /// bijectively to the next one. Sparse swapped rows must respect this map.
    stable_previous_to_next: Vec<u32>,
    stable_next_to_previous: Vec<u32>,
}

/// Per-swap output-label relabelling. The immutable base ids represent the
/// original frozen output pairs; ids allocated after `base_count` are local to
/// this swap and compare equal only to the same full mapped pair.
struct SwappedOutputIds<'a> {
    base_pairs: &'a [OutputPair],
    base_lookup: &'a FxHashMap<OutputPair, u32>,
    left: usize,
    right: usize,
    mapped: Vec<u32>,
    local: FxHashMap<OutputPair, u32>,
}

impl<'a> SwappedOutputIds<'a> {
    fn new(
        base_pairs: &'a [OutputPair],
        base_lookup: &'a FxHashMap<OutputPair, u32>,
        left: TerminalID,
        right: TerminalID,
    ) -> Self {
        Self {
            base_pairs,
            base_lookup,
            left: left as usize,
            right: right as usize,
            mapped: vec![u32::MAX; base_pairs.len()],
            local: FxHashMap::default(),
        }
    }

    fn id(&mut self, output: u32) -> u32 {
        let index = output as usize;
        let cached = self.mapped[index];
        if cached != u32::MAX {
            return cached;
        }
        let mapped = OutputPair {
            finalizers: self.base_pairs[index]
                .finalizers
                .mapped(Some((self.left, self.right))),
            future_finalizers: self.base_pairs[index]
                .future_finalizers
                .mapped(Some((self.left, self.right))),
        };
        let id = if let Some(&base) = self.base_lookup.get(&mapped) {
            base
        } else {
            let base_count = self.base_pairs.len() as u32;
            let next = base_count + self.local.len() as u32;
            *self.local.entry(mapped).or_insert(next)
        };
        self.mapped[index] = id;
        id
    }
}

/// Sparse topology of the byte-restricted raw tokenizer DFA. A missing enabled
/// transition has the same synthetic dead destination for every raw state.
/// Keeping only real edges is exact: all omitted bytes share that one default.
#[derive(Debug)]
struct RestrictedTopology {
    bytes: Vec<u8>,
    edge_offsets: Vec<u32>,
    edges: Vec<(u8, u32)>,
    real_state_count: usize,
    initial_state: usize,
    max_outdegree: usize,
}

impl RestrictedTopology {
    fn new(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        let bytes = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect::<Vec<_>>();
        let real_state_count = tokenizer.num_states() as usize;
        let mut edge_offsets = Vec::with_capacity(real_state_count + 2);
        let mut edges = Vec::new();
        let mut max_outdegree = 0usize;
        edge_offsets.push(0);
        for state in 0..real_state_count {
            let start = edges.len();
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                if relevant_bytes[byte as usize] {
                    edges.push((byte, target));
                }
            }
            max_outdegree = max_outdegree.max(edges.len() - start);
            edge_offsets.push(edges.len() as u32);
        }
        // Synthetic dead has no real edges: every enabled byte loops to itself.
        edge_offsets.push(edges.len() as u32);
        Self {
            bytes,
            edge_offsets,
            edges,
            real_state_count,
            initial_state: tokenizer.initial_state_id() as usize,
            max_outdegree,
        }
    }

    fn state_count(&self) -> usize {
        self.real_state_count + 1
    }

    fn dead_state(&self) -> usize {
        self.real_state_count
    }

    fn edges_from(&self, state: usize) -> &[(u8, u32)] {
        let start = self.edge_offsets[state] as usize;
        let end = self.edge_offsets[state + 1] as usize;
        &self.edges[start..end]
    }

    fn destination_for_byte(&self, state: usize, byte: u8) -> usize {
        if state == self.dead_state() {
            return state;
        }
        self.edges_from(state)
            .binary_search_by_key(&byte, |(edge_byte, _)| *edge_byte)
            .ok()
            .map(|index| self.edges_from(state)[index].1 as usize)
            .unwrap_or_else(|| self.dead_state())
    }

    fn observed_destinations(&self) -> Vec<bool> {
        let mut observed = vec![false; self.state_count()];
        for &(_, destination) in &self.edges {
            observed[destination as usize] = true;
        }
        observed
    }
}

/// The frozen-output observation made by the root-reachable part of the
/// byte-restricted DFA for one terminal. State IDs are deliberately kept rather
/// than quotienting or transforming the tokenizer: terminal interchangeability
/// uses the original DFA and its original metadata.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RootOutputSignature {
    finalizer_states: Box<[u32]>,
    future_finalizer_states: Box<[u32]>,
}

/// Partition candidate terminals by a necessary-and-sufficient condition for
/// the root part of the pair characterization.
///
/// Let `R` be the states reachable from the lexer initial state using enabled
/// bytes, and let `D = δ(R, bytes)` be the observed destination states. The
/// identity and swapped sides start at the same root and have the same byte
/// transition function, so induction on enabled-byte words makes each state in
/// `D` compare with itself. At such a state, swapping `left` and `right` leaves
/// either frozen output set unchanged exactly when the two terminals have equal
/// membership in that set. Thus the root hashes can agree at every refinement
/// depth exactly when both terminals have equal finalizer and frozen-future
/// membership over every state in `D`.
fn rooted_candidate_groups(
    tokenizer: &Tokenizer,
    candidates: &[TerminalID],
    topology: &RestrictedTopology,
) -> (Vec<Vec<TerminalID>>, usize) {
    let state_count = topology.real_state_count;
    let mut reachable = vec![false; state_count];
    reachable[topology.initial_state] = true;
    let mut worklist = vec![topology.initial_state];
    while let Some(state) = worklist.pop() {
        for &(_, destination) in topology.edges_from(state) {
            let destination = destination as usize;
            if !reachable[destination] {
                reachable[destination] = true;
                worklist.push(destination);
            }
        }
    }

    // `characterize` observes outputs on destinations, not current states.
    let mut observed = vec![false; state_count];
    for (state, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            for &(_, destination) in topology.edges_from(state) {
                observed[destination as usize] = true;
            }
        }
    }

    let mut is_candidate = vec![false; tokenizer.num_terminals() as usize];
    for &terminal in candidates {
        is_candidate[terminal as usize] = true;
    }
    let mut finalizer_states = vec![Vec::<u32>::new(); is_candidate.len()];
    let mut future_finalizer_states = vec![Vec::<u32>::new(); is_candidate.len()];
    for (state, &is_observed) in observed.iter().enumerate() {
        if !is_observed {
            continue;
        }
        for terminal in tokenizer.matched_terminals_iter(state as u32) {
            if is_candidate[terminal as usize] {
                finalizer_states[terminal as usize].push(state as u32);
            }
        }
        for terminal in tokenizer.possible_future_terminals_iter(state as u32) {
            if is_candidate[terminal as usize] {
                future_finalizer_states[terminal as usize].push(state as u32);
            }
        }
    }

    let mut groups = BTreeMap::<RootOutputSignature, Vec<TerminalID>>::new();
    for &terminal in candidates {
        let terminal = terminal as usize;
        groups
            .entry(RootOutputSignature {
                finalizer_states: std::mem::take(&mut finalizer_states[terminal])
                    .into_boxed_slice(),
                future_finalizer_states: std::mem::take(
                    &mut future_finalizer_states[terminal],
                )
                .into_boxed_slice(),
            })
            .or_default()
            .push(terminal as TerminalID);
    }
    (
        groups.into_values().collect(),
        observed.into_iter().filter(|&value| value).count(),
    )
}

/// A terminal's support across a terminal-name-independent structural
/// partition. It is only a rejection invariant; the full checker decides.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct StructuralOutputSignature {
    finalizer_support: Box<[u64]>,
    future_finalizer_support: Box<[u64]>,
}

const STRUCTURAL_REFINEMENT_ROUNDS: usize = 2;

/// Mix one invariant structural component into a deterministic 64-bit
/// fingerprint. Equal tuples always have equal fingerprints. A collision only
/// coarsens the prefilter and therefore cannot reject a valid pair.
#[inline]
fn mix_structural_fingerprint(mut state: u64, component: u64) -> u64 {
    state ^= component.wrapping_add(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    state = state.wrapping_mul(0x517c_c1b7_2722_0a95);
    state ^ (state >> 29)
}

/// Compute structural support signatures using a sparse canonical form of the
/// reference tuple. At a given round, every missing byte has the common dead
/// component `(hash(dead), 0, 0)`. The full tuple is therefore determined by
/// the enabled-byte keys whose component differs from that default. Omitting
/// default entries preserves tuple equality exactly.
fn structural_candidate_signatures(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    candidates: &[TerminalID],
    topology: &RestrictedTopology,
) -> (Vec<StructuralOutputSignature>, usize) {
    let state_count = topology.state_count();
    let dead_state = topology.dead_state();
    let mut finalizer_counts = vec![0u64; state_count];
    let mut future_finalizer_counts = vec![0u64; state_count];
    for state in 0..topology.real_state_count {
        finalizer_counts[state] = tokenizer
            .matched_terminals_iter(state as u32)
            .filter(|&terminal| active_terminals[terminal as usize])
            .count() as u64;
        future_finalizer_counts[state] = tokenizer
            .possible_future_terminals_iter(state as u32)
            .filter(|&terminal| active_terminals[terminal as usize])
            .count() as u64;
    }

    let mut fingerprints = vec![0x6a09_e667_f3bc_c909; state_count];
    for _ in 0..STRUCTURAL_REFINEMENT_ROUNDS {
        let default_fingerprint = fingerprints[dead_state];
        let mut next = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let mut fingerprint = mix_structural_fingerprint(
                0xbb67_ae85_84ca_a73b,
                topology.bytes.len() as u64,
            );
            for &(byte, destination) in topology.edges_from(state) {
                let destination = destination as usize;
                if fingerprints[destination] == default_fingerprint
                    && finalizer_counts[destination] == 0
                    && future_finalizer_counts[destination] == 0
                {
                    continue;
                }
                fingerprint = mix_structural_fingerprint(fingerprint, byte as u64);
                fingerprint = mix_structural_fingerprint(fingerprint, fingerprints[destination]);
                fingerprint = mix_structural_fingerprint(fingerprint, finalizer_counts[destination]);
                fingerprint = mix_structural_fingerprint(
                    fingerprint,
                    future_finalizer_counts[destination],
                );
            }
            next.push(fingerprint);
        }
        fingerprints = next;
    }

    let mut color_ids = FxHashMap::<u64, u32>::default();
    let mut colors = Vec::with_capacity(state_count);
    for fingerprint in fingerprints {
        let next = color_ids.len() as u32;
        colors.push(*color_ids.entry(fingerprint).or_insert(next));
    }
    let color_count = color_ids.len();
    let words = color_count.div_ceil(64);

    let mut candidate_index_by_terminal = vec![usize::MAX; active_terminals.len()];
    for (candidate_index, &terminal) in candidates.iter().enumerate() {
        candidate_index_by_terminal[terminal as usize] = candidate_index;
    }
    let mut finalizer_support = vec![vec![0u64; words]; candidates.len()];
    let mut future_finalizer_support = vec![vec![0u64; words]; candidates.len()];
    for (state, &is_observed) in topology.observed_destinations().iter().enumerate() {
        if !is_observed || state == dead_state {
            continue;
        }
        let color = colors[state] as usize;
        let word = color / 64;
        let mask = 1u64 << (color % 64);
        for terminal in tokenizer.matched_terminals_iter(state as u32) {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            if candidate_index != usize::MAX {
                finalizer_support[candidate_index][word] |= mask;
            }
        }
        for terminal in tokenizer.possible_future_terminals_iter(state as u32) {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            if candidate_index != usize::MAX {
                future_finalizer_support[candidate_index][word] |= mask;
            }
        }
    }

    (
        finalizer_support
            .into_iter()
            .zip(future_finalizer_support)
            .map(|(finalizer_support, future_finalizer_support)| StructuralOutputSignature {
                finalizer_support: finalizer_support.into_boxed_slice(),
                future_finalizer_support: future_finalizer_support.into_boxed_slice(),
            })
            .collect(),
        color_count,
    )
}

/// Refine root candidates by the global structural invariant. Singletons need
/// no direct terminal-pair check.
fn refine_candidate_groups_by_structure(
    root_groups: Vec<Vec<TerminalID>>,
    candidates: &[TerminalID],
    signatures: &[StructuralOutputSignature],
) -> Vec<Vec<TerminalID>> {
    let terminal_count = candidates
        .iter()
        .copied()
        .max()
        .map_or(0, |terminal| terminal as usize + 1);
    let mut candidate_index_by_terminal = vec![usize::MAX; terminal_count];
    for (candidate_index, &terminal) in candidates.iter().enumerate() {
        candidate_index_by_terminal[terminal as usize] = candidate_index;
    }

    let mut groups = Vec::new();
    for root_group in root_groups {
        let mut by_signature = BTreeMap::<StructuralOutputSignature, Vec<TerminalID>>::new();
        for terminal in root_group {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            debug_assert_ne!(candidate_index, usize::MAX);
            by_signature
                .entry(signatures[candidate_index].clone())
                .or_default()
                .push(terminal);
        }
        groups.extend(by_signature.into_values().filter(|group| group.len() >= 2));
    }
    groups
}

/// The selected raw tokenizer-state representative for every source state
/// under one terminal swap. The exact characterization establishes a mapped
/// target class; all downstream consumers have always selected only that
/// class's first raw state, so retaining the whole class is redundant.
#[derive(Clone, Debug, Eq, PartialEq)]
struct InterchangeMap {
    target_representative_for_source_state: Vec<u32>,
}

impl InterchangeMap {
    /// The scanner is implemented over raw tokenizer states, so each mapped
    /// target class needs one arbitrary raw representative at its entry point.
    /// This selection has no mathematical significance.
    fn arbitrary_target_representatives(&self) -> Vec<u32> {
        self.target_representative_for_source_state.clone()
    }
}

struct PairCharacterization {
    identity_hashes: Vec<CharacterizationHash>,
    swapped_hashes: Vec<CharacterizationHash>,
}

struct InterchangeabilityDfa {
    topology: RestrictedTopology,
    empty_output: OutputBits,
    finalizers: Vec<OutputBits>,
    future_finalizers: Vec<OutputBits>,
    observed_output_pairs: Vec<OutputPair>,
    observed_output_pair_lookup: FxHashMap<OutputPair, u32>,
    observed_output_pair_ids_by_terminal: Vec<Vec<u32>>,
    observed_output_pair_marks: Vec<u32>,
    observed_output_pair_mark_epoch: u32,
    /// Exact terminal columns over the frozen state outputs. When two columns
    /// coincide in both output families, swapping their terminal labels leaves
    /// every destination output unchanged and therefore needs no refinement.
    finalizer_states_by_terminal: Vec<Vec<u32>>,
    future_finalizer_states_by_terminal: Vec<Vec<u32>>,
    identity_rounds: Vec<Vec<CharacterizationHash>>,
    /// Canonical, collision-free characterization cache used by the hot path.
    /// It describes the same raw restricted topology as `identity_rounds` and
    /// never merges or rewrites tokenizer states.
    output_pairs: Vec<OutputPair>,
    output_pair_lookup: FxHashMap<OutputPair, u32>,
    output_pair_by_state: Vec<u32>,
    /// Reverse enabled-byte edges, used only by the exact first-round
    /// rejection prefilter. Each changed frozen output can affect only these
    /// source rows in the characterization tuple.
    reverse_predecessors: Vec<Vec<u32>>,
    canonical_rounds: Vec<CanonicalRound>,
    canonical_round_one_class_counts: Option<Vec<u32>>,
    canonical_round_one_source_marks: Vec<u32>,
    canonical_round_one_source_mark_epoch: u32,
    canonical_round_one_affected_sources: Vec<u32>,
    canonical_quotient: Option<CanonicalQuotient>,
    quotient_certified: usize,
    sparse_quotient_certified: usize,
    sparse_quotient_cone_classes_total: usize,
    sparse_quotient_cone_classes_max: usize,
    sparse_quotient_cone_ns: u64,
    sparse_quotient_refinement_ns: u64,
    sparse_quotient_map_ns: u64,
    canonical_stable_round: Option<usize>,
    canonical_identity_map: Option<InterchangeMap>,
    signature_capacity: usize,
}

impl InterchangeabilityDfa {
    fn new(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Self {
        Self::from_topology(
            tokenizer,
            observed_terminals,
            RestrictedTopology::new(tokenizer, relevant_bytes),
        )
    }

    fn from_topology(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        topology: RestrictedTopology,
    ) -> Self {
        let state_count = topology.state_count();
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
        // These are the tokenizer's original, frozen future-finalizer sets.
        let future_finalizers = (0..tokenizer.num_states())
            .map(|state| terminal_bits(tokenizer.possible_future_terminals_iter(state).collect()))
            .collect::<Vec<_>>();

        let observed_destinations = topology.observed_destinations();
        let mut observed_output_pairs = Vec::<OutputPair>::new();
        let mut observed_output_pair_lookup = FxHashMap::<OutputPair, u32>::default();
        for state in 0..topology.real_state_count {
            if !observed_destinations[state] {
                continue;
            }
            let pair = OutputPair {
                finalizers: finalizers[state].clone(),
                future_finalizers: future_finalizers[state].clone(),
            };
            if !observed_output_pair_lookup.contains_key(&pair) {
                let id = observed_output_pairs.len() as u32;
                observed_output_pair_lookup.insert(pair.clone(), id);
                observed_output_pairs.push(pair);
            }
        }
        let mut observed_output_pair_ids_by_terminal =
            vec![Vec::<u32>::new(); observed_terminals.len()];
        for (id, pair) in observed_output_pairs.iter().enumerate() {
            for outputs in [&pair.finalizers, &pair.future_finalizers] {
                for (word_index, &word) in outputs.0.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        let terminal = word_index * 64 + bit;
                        if observed_terminals.get(terminal).copied().unwrap_or(false) {
                            observed_output_pair_ids_by_terminal[terminal].push(id as u32);
                        }
                        word &= word - 1;
                    }
                }
            }
        }
        let mut finalizer_states_by_terminal = vec![Vec::<u32>::new(); observed_terminals.len()];
        let mut future_finalizer_states_by_terminal = vec![Vec::<u32>::new(); observed_terminals.len()];
        for (state, outputs) in finalizers.iter().enumerate() {
            for (word_index, &word) in outputs.0.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let terminal = word_index * 64 + bit;
                    if terminal < finalizer_states_by_terminal.len() {
                        finalizer_states_by_terminal[terminal].push(state as u32);
                    }
                    word &= word - 1;
                }
            }
        }
        for (state, outputs) in future_finalizers.iter().enumerate() {
            for (word_index, &word) in outputs.0.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let terminal = word_index * 64 + bit;
                    if terminal < future_finalizer_states_by_terminal.len() {
                        future_finalizer_states_by_terminal[terminal].push(state as u32);
                    }
                    word &= word - 1;
                }
            }
        }
        let empty_output = OutputBits::new(output_words);
        let empty_pair = OutputPair {
            finalizers: empty_output.clone(),
            future_finalizers: empty_output.clone(),
        };
        let mut output_pairs = vec![empty_pair.clone()];
        let mut output_pair_lookup = FxHashMap::<OutputPair, u32>::default();
        output_pair_lookup.insert(empty_pair, 0);
        let mut output_pair_by_state = Vec::with_capacity(state_count);
        for state in 0..topology.real_state_count {
            let pair = OutputPair {
                finalizers: finalizers[state].clone(),
                future_finalizers: future_finalizers[state].clone(),
            };
            let id = match output_pair_lookup.entry(pair) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let id = output_pairs.len() as u32;
                    output_pairs.push(entry.key().clone());
                    entry.insert(id);
                    id
                }
            };
            output_pair_by_state.push(id);
        }
        // The synthetic dead destination has the all-empty frozen output.
        output_pair_by_state.push(0);
        let mut reverse_predecessors = vec![Vec::<u32>::new(); topology.real_state_count];
        for source in 0..topology.real_state_count {
            for &(_, destination) in topology.edges_from(source) {
                reverse_predecessors[destination as usize].push(source as u32);
            }
        }
        let signature_capacity = CHARACTERIZATION_DOMAIN.len()
            + 4
            + topology.max_outdegree
                * (1 + blake3::OUT_LEN + 2 * output_words * size_of::<u64>());
        let seed = CharacterizationHash::seed();
        let observed_output_pair_count = observed_output_pair_lookup.len();
        Self {
            topology,
            empty_output,
            finalizers,
            future_finalizers,
            observed_output_pairs,
            observed_output_pair_lookup,
            observed_output_pair_ids_by_terminal,
            observed_output_pair_marks: vec![0; observed_output_pair_count],
            observed_output_pair_mark_epoch: 0,
            finalizer_states_by_terminal,
            future_finalizer_states_by_terminal,
            identity_rounds: vec![vec![seed; state_count]],
            output_pairs,
            output_pair_lookup,
            output_pair_by_state,
            reverse_predecessors,
            canonical_rounds: vec![CanonicalRound {
                classes: vec![0; state_count],
                signatures: FxHashMap::default(),
            }],
            canonical_round_one_class_counts: None,
            canonical_round_one_source_marks: vec![0; state_count - 1],
            canonical_round_one_source_mark_epoch: 0,
            canonical_round_one_affected_sources: Vec::new(),
            canonical_quotient: None,
            quotient_certified: 0,
            sparse_quotient_certified: 0,
            sparse_quotient_cone_classes_total: 0,
            sparse_quotient_cone_classes_max: 0,
            sparse_quotient_cone_ns: 0,
            sparse_quotient_refinement_ns: 0,
            sparse_quotient_map_ns: 0,
            canonical_stable_round: None,
            canonical_identity_map: None,
            signature_capacity,
        }
    }

    fn state_count(&self) -> usize {
        self.topology.state_count()
    }

    fn dead_state(&self) -> usize {
        self.topology.dead_state()
    }

    /// This does not transform the lexer. It supplies the absent destination
    /// while evaluating an enabled byte transition in `characterize`.
    fn destination_for_slot(&self, state: usize, byte_slot: usize) -> usize {
        self.topology
            .destination_for_byte(state, self.topology.bytes[byte_slot])
    }

    fn output_at<'a>(&'a self, outputs: &'a [OutputBits], state: usize) -> &'a OutputBits {
        outputs.get(state).unwrap_or(&self.empty_output)
    }

    /// Hash the canonical sparse form of one reference characterization tuple.
    /// Missing byte transitions have the common dead component. A byte is
    /// recorded exactly when its component differs from that default, so two
    /// full tuples are equal iff their sparse forms are equal.
    fn characterize_round(
        &self,
        previous: &[CharacterizationHash],
        finalizers: &[OutputBits],
        future_finalizers: &[OutputBits],
    ) -> Vec<CharacterizationHash> {
        debug_assert_eq!(previous.len(), self.state_count());
        let mut next = Vec::with_capacity(self.state_count());
        let mut tuple = Vec::with_capacity(self.signature_capacity);
        let dead_state = self.dead_state();
        let default_hash = previous[dead_state];
        for state in 0..self.state_count() {
            tuple.clear();
            tuple.extend_from_slice(CHARACTERIZATION_DOMAIN);
            tuple.extend_from_slice(&(self.topology.bytes.len() as u32).to_le_bytes());
            for &(byte, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                let finalizers = self.output_at(finalizers, destination);
                let future_finalizers = self.output_at(future_finalizers, destination);
                if previous[destination] == default_hash
                    && finalizers == &self.empty_output
                    && future_finalizers == &self.empty_output
                {
                    continue;
                }
                tuple.push(byte);
                tuple.extend_from_slice(&previous[destination].0);
                finalizers.append_to(&mut tuple);
                future_finalizers.append_to(&mut tuple);
            }
            next.push(CharacterizationHash(*blake3::hash(&tuple).as_bytes()));
        }
        next
    }

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

    fn canonical_identity_signature(
        &self,
        state: usize,
        previous: &[u32],
    ) -> CanonicalSignature {
        let default_class = previous[self.dead_state()];
        let mut components = Vec::with_capacity(self.topology.edges_from(state).len());
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = self.output_pair_by_state[destination];
            if previous[destination] == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class: previous[destination],
                output,
            });
        }
        CanonicalSignature(components)
    }

    fn canonical_swapped_signature(
        &self,
        state: usize,
        previous: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let default_class = previous[self.dead_state()];
        let mut components = Vec::with_capacity(self.topology.edges_from(state).len());
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            if previous[destination] == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class: previous[destination],
                output,
            });
        }
        CanonicalSignature(components)
    }

    fn canonical_identity_round(&self, previous: &[u32]) -> CanonicalRound {
        let mut signatures = FxHashMap::<CanonicalSignature, u32>::default();
        let mut classes = Vec::with_capacity(self.state_count());
        for state in 0..self.state_count() {
            let signature = self.canonical_identity_signature(state, previous);
            let next = signatures.len() as u32;
            let class = *signatures.entry(signature).or_insert(next);
            classes.push(class);
        }
        CanonicalRound {
            classes,
            signatures,
        }
    }

    fn ensure_canonical_identity_round(&mut self, round: usize) {
        while self.canonical_rounds.len() <= round {
            let started_at = Instant::now();
            let previous = self
                .canonical_rounds
                .last()
                .expect("round zero is always present")
                .classes
                .clone();
            let next = self.canonical_identity_round(&previous);
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
                eprintln!(
                    "[glrmask/profile][terminal_interchangeability] canonical_identity_round={} classes={} elapsed_ms={:.3}",
                    self.canonical_rounds.len(),
                    next.signatures.len(),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            self.canonical_rounds.push(next);
        }
    }

    fn ensure_canonical_identity_stable_round(&mut self) -> usize {
        if let Some(round) = self.canonical_stable_round {
            return round;
        }
        let state_count = self.state_count();
        for round in 1..=state_count * 2 {
            self.ensure_canonical_identity_round(round);
            if same_equality_partition_u32(
                &self.canonical_rounds[round - 1].classes,
                &self.canonical_rounds[round].classes,
            ) {
                self.canonical_stable_round = Some(round);
                return round;
            }
        }
        panic!(
            "canonical terminal interchangeability characterization did not stabilize within {} rounds",
            state_count * 2,
        );
    }

    fn ensure_canonical_quotient(&mut self) {
        if self.canonical_quotient.is_some() {
            return;
        }
        let stable_round = self.ensure_canonical_identity_stable_round();
        let class_for_state = self.canonical_rounds[stable_round].classes.clone();
        let class_count = self.canonical_rounds[stable_round].signatures.len();
        debug_assert_eq!(
            class_count,
            class_for_state.iter().copied().max().map_or(0, |class| class as usize + 1),
        );

        let mut representative_by_class = vec![u32::MAX; class_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            let representative = &mut representative_by_class[class as usize];
            if *representative == u32::MAX {
                *representative = state as u32;
            }
        }
        debug_assert!(
            representative_by_class
                .iter()
                .all(|&representative| representative != u32::MAX),
        );

        let mut class_by_signature = FxHashMap::<CanonicalSignature, u32>::default();
        let mut reverse_predecessors = vec![Vec::<u32>::new(); class_count];
        for (class, &representative) in representative_by_class.iter().enumerate() {
            let state = representative as usize;
            let signature = self.canonical_identity_signature(state, &class_for_state);
            if let Some(previous) = class_by_signature.insert(signature, class as u32) {
                assert_eq!(
                    previous, class as u32,
                    "stable terminal-interchangeability quotient must have unique class signatures",
                );
            }
            for &(_, destination) in self.topology.edges_from(state) {
                let destination_class = class_for_state[destination as usize] as usize;
                reverse_predecessors[destination_class].push(class as u32);
            }
        }
        for predecessors in &mut reverse_predecessors {
            predecessors.sort_unstable();
            predecessors.dedup();
        }
        debug_assert_eq!(class_by_signature.len(), class_count);

        let mut identity_classes_by_round = Vec::with_capacity(stable_round + 1);
        let mut identity_class_counts_by_round = Vec::with_capacity(stable_round + 1);
        for round in 0..=stable_round {
            let identity_classes = representative_by_class
                .iter()
                .map(|&state| self.canonical_rounds[round].classes[state as usize])
                .collect::<Vec<_>>();
            let class_count_at_round = self.canonical_rounds[round]
                .classes
                .iter()
                .copied()
                .max()
                .map_or(0, |class| class as usize + 1);
            let mut counts = vec![0u32; class_count_at_round];
            for (quotient_class, &class) in identity_classes.iter().enumerate() {
                if (representative_by_class[quotient_class] as usize) < self.topology.real_state_count {
                    counts[class as usize] += 1;
                }
            }
            identity_classes_by_round.push(identity_classes);
            identity_class_counts_by_round.push(counts);
        }

        let stable_previous = &identity_classes_by_round[stable_round - 1];
        let stable_next = &identity_classes_by_round[stable_round];
        let stable_previous_class_count = self.canonical_rounds[stable_round - 1]
            .classes
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let stable_next_class_count = self.canonical_rounds[stable_round]
            .classes
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let mut stable_previous_to_next = vec![u32::MAX; stable_previous_class_count];
        let mut stable_next_to_previous = vec![u32::MAX; stable_next_class_count];
        for (&previous, &next) in stable_previous.iter().zip(stable_next) {
            let previous_target = &mut stable_previous_to_next[previous as usize];
            if *previous_target == u32::MAX {
                *previous_target = next;
            } else {
                assert_eq!(*previous_target, next, "stable TI quotient split an identity class");
            }
            let next_source = &mut stable_next_to_previous[next as usize];
            if *next_source == u32::MAX {
                *next_source = previous;
            } else {
                assert_eq!(*next_source, previous, "stable TI quotient merged identity classes");
            }
        }
        assert!(
            stable_previous_to_next.iter().all(|&target| target != u32::MAX)
                && stable_next_to_previous.iter().all(|&source| source != u32::MAX),
            "stable TI quotient omitted an identity class",
        );

        self.canonical_quotient = Some(CanonicalQuotient {
            class_for_state,
            representative_by_class,
            class_by_signature,
            reverse_predecessors,
            identity_classes_by_round,
            identity_class_counts_by_round,
            stable_previous_to_next,
            stable_next_to_previous,
        });
    }

    fn canonical_quotient_swapped_signature(
        &self,
        quotient: &CanonicalQuotient,
        class: usize,
        previous: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let state = quotient.representative_by_class[class] as usize;
        let default_class = previous[quotient.class_for_state[self.dead_state()] as usize];
        let mut components = Vec::with_capacity(self.topology.edges_from(state).len());
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            let previous_class = previous[quotient.class_for_state[destination] as usize];
            if previous_class == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class,
                output,
            });
        }
        CanonicalSignature(components)
    }

    fn quotient_identity_classes_at_round<'a>(
        &self,
        quotient: &'a CanonicalQuotient,
        round: usize,
    ) -> &'a [u32] {
        quotient
            .identity_classes_by_round
            .get(round)
            .expect("identity quotient round must be cached")
    }

    fn quotient_rooted_class_set_still_possible(
        &self,
        quotient: &CanonicalQuotient,
        identity: &[u32],
        swapped: &[u32],
    ) -> bool {
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if identity[root_class] != swapped[root_class] {
            return false;
        }
        let mut identity_set = FxHashSet::<u32>::default();
        let mut swapped_set = FxHashSet::<u32>::default();
        for (class, &representative) in quotient.representative_by_class.iter().enumerate() {
            if representative as usize >= self.topology.real_state_count {
                continue;
            }
            identity_set.insert(identity[class]);
            swapped_set.insert(swapped[class]);
        }
        identity_set == swapped_set
    }

    /// Non-cone quotient classes retain their identity labels, so equality of
    /// complete identity and swapped class sets reduces to labels whose final
    /// occurrence lies inside the cone.
    fn sparse_quotient_rooted_class_set_still_possible(
        &self,
        quotient: &CanonicalQuotient,
        identity: &[u32],
        identity_counts: &[u32],
        swapped: &[u32],
        cone_classes: &[usize],
    ) -> bool {
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if identity[root_class] != swapped[root_class] {
            return false;
        }

        let mut changed_counts = FxHashMap::<u32, u32>::default();
        let mut introduced_or_retained = FxHashSet::<u32>::default();
        for &class in cone_classes {
            let before = identity[class];
            let after = swapped[class];
            if after as usize >= identity_counts.len() {
                return false;
            }
            *changed_counts.entry(before).or_default() += 1;
            introduced_or_retained.insert(after);
        }
        changed_counts.into_iter().all(|(class, changed)| {
            changed < identity_counts[class as usize]
                || introduced_or_retained.contains(&class)
        })
    }

    /// At the stable identity round the old-to-new class map is bijective.
    /// Outside the cone the paired rows are identity rows, so only the changed
    /// cone rows need checking to prove paired partition stability.
    fn sparse_quotient_pair_is_stable(
        &self,
        quotient: &CanonicalQuotient,
        swapped_previous: &[u32],
        swapped_next: &[u32],
        cone_classes: &[usize],
    ) -> bool {
        cone_classes.iter().all(|&class| {
            let previous = swapped_previous[class] as usize;
            let next = swapped_next[class] as usize;
            previous < quotient.stable_previous_to_next.len()
                && next < quotient.stable_next_to_previous.len()
                && quotient.stable_previous_to_next[previous] == next as u32
                && quotient.stable_next_to_previous[next] == previous as u32
        })
    }

    fn quotient_interchange_map_from_classes(
        &self,
        quotient: &CanonicalQuotient,
        swapped_classes: &[u32],
    ) -> Option<InterchangeMap> {
        let class_count = quotient.representative_by_class.len();
        if swapped_classes.len() != class_count {
            return None;
        }
        let mut target_representative_by_identity_class = vec![None; class_count];
        for state in 0..self.topology.real_state_count {
            let target_identity_class = swapped_classes[quotient.class_for_state[state] as usize] as usize;
            if target_identity_class >= class_count {
                return None;
            }
            target_representative_by_identity_class[target_identity_class]
                .get_or_insert(state as u32);
        }
        let target_representative_for_source_state = (0..self.topology.real_state_count)
            .map(|source| {
                target_representative_by_identity_class
                    [quotient.class_for_state[source] as usize]
            })
            .collect::<Option<Vec<_>>>()?;
        Some(InterchangeMap {
            target_representative_for_source_state,
        })
    }

    /// Evaluate the exact paired characterization on the already-stable
    /// identity quotient.  Stable quotient classes are congruent for every
    /// later swapped round, so one representative per class produces the same
    /// partition relation as the raw computation.  A failed quotient proof is
    /// inconclusive and falls back to the raw exact refinement below.
    fn canonical_quotient_affected_cone(
        &self,
        quotient: &CanonicalQuotient,
        left: TerminalID,
        right: TerminalID,
    ) -> (Vec<bool>, Vec<usize>) {
        let mut in_cone = vec![false; quotient.representative_by_class.len()];
        let mut cone_classes = Vec::<usize>::new();
        let mut worklist = Vec::<u32>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !in_cone[class] {
                        in_cone[class] = true;
                        cone_classes.push(class);
                        worklist.push(class as u32);
                    }
                }
            }
        }
        while let Some(class) = worklist.pop() {
            for &predecessor in &quotient.reverse_predecessors[class as usize] {
                let predecessor = predecessor as usize;
                if !in_cone[predecessor] {
                    in_cone[predecessor] = true;
                    cone_classes.push(predecessor);
                    worklist.push(predecessor as u32);
                }
            }
        }
        (in_cone, cone_classes)
    }

    /// Exact sparse quotient evaluation. Outside the backward cone of every
    /// relabelled frozen output, the swapped characterization is identical to
    /// the identity at every round. Only the cone needs recomputation.
    fn canonical_sparse_quotient_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let stable_round = self.ensure_canonical_identity_stable_round();
        self.ensure_canonical_quotient();
        let pair_started_at = profile_timing.then(Instant::now);
        let result = {
            let quotient = self
                .canonical_quotient
                .as_ref()
                .expect("canonical quotient initialized");
            debug_assert_eq!(
                quotient.class_by_signature.len(),
                quotient.representative_by_class.len(),
            );
            let cone_started_at = profile_timing.then(Instant::now);
            let (in_cone, cone_classes) =
                self.canonical_quotient_affected_cone(quotient, left, right);
            let cone_size = cone_classes.len();
            if let Some(started_at) = cone_started_at {
                self.sparse_quotient_cone_ns += started_at.elapsed().as_nanos() as u64;
            }
            if cone_size == 0 {
                let identity = (0..quotient.representative_by_class.len())
                    .map(|class| class as u32)
                    .collect::<Vec<_>>();
                self.quotient_interchange_map_from_classes(quotient, &identity)
                    .map(|map| (map, cone_size))
            } else {
                let mut swapped_previous = vec![0u32; quotient.representative_by_class.len()];
                let mut outputs = SwappedOutputIds::new(
                    &self.output_pairs,
                    &self.output_pair_lookup,
                    left,
                    right,
                );
                let mut found = None;
                for round in 1..=stable_round {
                    let identity_next = self.quotient_identity_classes_at_round(quotient, round);
                    let identity_counts = &quotient.identity_class_counts_by_round[round];
                    let identity = &self.canonical_rounds[round];
                    let mut local = FxHashMap::<CanonicalSignature, u32>::default();
                    let local_base = identity.signatures.len() as u32;
                    let mut swapped_next = identity_next.to_vec();
                    for &source_class in &cone_classes {
                        let signature = self.canonical_quotient_swapped_signature(
                            quotient,
                            source_class,
                            &swapped_previous,
                            &mut outputs,
                        );
                        let swapped_class =
                            if let Some(&identity_class) = identity.signatures.get(&signature) {
                                identity_class
                            } else {
                                let next = local_base + local.len() as u32;
                                *local.entry(signature).or_insert(next)
                            };
                        swapped_next[source_class] = swapped_class;
                    }
                    if !self.sparse_quotient_rooted_class_set_still_possible(
                        quotient,
                        identity_next,
                        identity_counts,
                        &swapped_next,
                        &cone_classes,
                    ) {
                        return None;
                    }
                    // Materialization uses stable identity class labels, so
                    // accept only at the identity partition fixed point.
                    if round == stable_round
                        && self.sparse_quotient_pair_is_stable(
                            quotient,
                            &swapped_previous,
                            &swapped_next,
                            &cone_classes,
                        )
                    {
                        let map_started_at = profile_timing.then(Instant::now);
                        found = self
                            .quotient_interchange_map_from_classes(quotient, &swapped_next)
                            .map(|map| (map, cone_size));
                        if let Some(started_at) = map_started_at {
                            self.sparse_quotient_map_ns += started_at.elapsed().as_nanos() as u64;
                        }
                        break;
                    }
                    swapped_previous = swapped_next;
                }
                found
            }
        };
        if profile_timing {
            if let Some(started_at) = pair_started_at {
                self.sparse_quotient_refinement_ns += started_at.elapsed().as_nanos() as u64;
            }
        }
        if let Some((map, cone_size)) = result {
            self.sparse_quotient_certified += 1;
            self.sparse_quotient_cone_classes_total += cone_size;
            self.sparse_quotient_cone_classes_max =
                self.sparse_quotient_cone_classes_max.max(cone_size);
            return Some(map);
        }
        None
    }

    fn canonical_quotient_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        self.canonical_sparse_quotient_interchange_map(left, right)
            .or_else(|| self.canonical_full_quotient_interchange_map(left, right))
    }

    fn canonical_full_quotient_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let stable_round = self.ensure_canonical_identity_stable_round();
        self.ensure_canonical_quotient();
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        let mut swapped_previous = vec![0u32; quotient.representative_by_class.len()];
        let mut outputs = SwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left,
            right,
        );
        for round in 1..=stable_round {
            let identity_previous = self.quotient_identity_classes_at_round(quotient, round - 1);
            let identity_next = self.quotient_identity_classes_at_round(quotient, round);
            let identity = &self.canonical_rounds[round];
            let mut local = FxHashMap::<CanonicalSignature, u32>::default();
            let local_base = identity.signatures.len() as u32;
            let mut swapped_next = Vec::with_capacity(quotient.representative_by_class.len());
            for class in 0..quotient.representative_by_class.len() {
                let signature = self.canonical_quotient_swapped_signature(
                    quotient,
                    class,
                    &swapped_previous,
                    &mut outputs,
                );
                let class = if let Some(&identity_class) = identity.signatures.get(&signature) {
                    identity_class
                } else {
                    let next = local_base + local.len() as u32;
                    *local.entry(signature).or_insert(next)
                };
                swapped_next.push(class);
            }
            if !self.quotient_rooted_class_set_still_possible(
                quotient,
                &identity_next,
                &swapped_next,
            ) {
                return None;
            }
            if same_equality_partition_pair_u32(
                &identity_previous,
                &swapped_previous,
                &identity_next,
                &swapped_next,
            ) {
                let map = self.quotient_interchange_map_from_classes(quotient, &swapped_next)?;
                self.quotient_certified += 1;
                return Some(map);
            }
            swapped_previous = swapped_next;
        }
        None
    }

    fn canonical_swapped_round(
        &self,
        previous: &[u32],
        identity: &CanonicalRound,
        outputs: &mut SwappedOutputIds<'_>,
    ) -> Vec<u32> {
        let mut local = FxHashMap::<CanonicalSignature, u32>::default();
        let local_base = identity.signatures.len() as u32;
        let mut classes = Vec::with_capacity(self.state_count());
        for state in 0..self.state_count() {
            let signature = self.canonical_swapped_signature(state, previous, outputs);
            let class = if let Some(&identity_class) = identity.signatures.get(&signature) {
                identity_class
            } else {
                let next = local_base + local.len() as u32;
                *local.entry(signature).or_insert(next)
            };
            classes.push(class);
        }
        classes
    }

    /// Exact necessary first-round condition for a terminal swap. Round one
    /// depends only on frozen destination outputs, so a swap can change only
    /// source rows with an enabled edge into a state mentioning either terminal.
    /// All other rows keep their cached identity class. This never accepts a
    /// pair by itself; it only rejects a pair whose first characterization
    /// partition cannot be rooted-bijective.
    fn canonical_round_one_still_possible(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        self.ensure_canonical_identity_round(1);
        if self.canonical_round_one_class_counts.is_none() {
            let class_count = self.canonical_rounds[1].signatures.len();
            let mut counts = vec![0u32; class_count];
            for &class in &self.canonical_rounds[1].classes[..self.topology.real_state_count] {
                counts[class as usize] += 1;
            }
            self.canonical_round_one_class_counts = Some(counts);
        }

        self.canonical_round_one_source_mark_epoch =
            self.canonical_round_one_source_mark_epoch.wrapping_add(1);
        if self.canonical_round_one_source_mark_epoch == 0 {
            self.canonical_round_one_source_marks.fill(0);
            self.canonical_round_one_source_mark_epoch = 1;
        }
        let epoch = self.canonical_round_one_source_mark_epoch;
        self.canonical_round_one_affected_sources.clear();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let source = source as usize;
                    if self.canonical_round_one_source_marks[source] != epoch {
                        self.canonical_round_one_source_marks[source] = epoch;
                        self.canonical_round_one_affected_sources.push(source as u32);
                    }
                }
            }
        }

        // Move the scratch list out while comparing exact cached signatures so
        // immutable borrows of the DFA do not conflict with scratch reuse.
        let affected_sources = std::mem::take(&mut self.canonical_round_one_affected_sources);
        let identity = &self.canonical_rounds[1];
        let identity_counts = self
            .canonical_round_one_class_counts
            .as_ref()
            .expect("first-round counts initialized");
        let mut changed_by_identity_class = FxHashMap::<u32, u32>::default();
        let mut added_identity_classes = FxHashSet::<u32>::default();
        let mut swapped_root_class = identity.classes[self.topology.initial_state];
        let mut outputs = SwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left,
            right,
        );
        for &source in &affected_sources {
            let source = source as usize;
            let identity_class = identity.classes[source];
            *changed_by_identity_class.entry(identity_class).or_default() += 1;
            let signature = self.canonical_swapped_signature(
                source,
                &self.canonical_rounds[0].classes,
                &mut outputs,
            );
            let Some(&swapped_class) = identity.signatures.get(&signature) else {
                self.canonical_round_one_affected_sources = affected_sources;
                return false;
            };
            added_identity_classes.insert(swapped_class);
            if source == self.topology.initial_state {
                swapped_root_class = swapped_class;
            }
        }
        self.canonical_round_one_affected_sources = affected_sources;

        if swapped_root_class != identity.classes[self.topology.initial_state] {
            return false;
        }
        changed_by_identity_class.into_iter().all(|(class, changed)| {
            changed < identity_counts[class as usize] || added_identity_classes.contains(&class)
        })
    }

    /// Collision-free exact refinement. This deliberately recomputes every
    /// raw restricted state each round: an earlier incremental cone shortcut
    /// was not a sufficient proof of cross-side partition stabilization.
    fn canonical_interchange_map(&mut self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        if let Some(map) = self.canonical_quotient_interchange_map(left, right) {
            return Some(map);
        }
        let pair_started_at = Instant::now();
        let profile_pair = std::env::var_os("GLRMASK_PROFILE_L2P_TI_CANONICAL_PAIRS").is_some();
        let stable_round = self.ensure_canonical_identity_stable_round();
        let mut outputs = SwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left,
            right,
        );
        let mut swapped_previous = self.canonical_rounds[0].classes.clone();
        for round in 1..=stable_round {
            let identity_previous = &self.canonical_rounds[round - 1].classes;
            let identity = &self.canonical_rounds[round];
            let swapped_next = self.canonical_swapped_round(
                &swapped_previous,
                identity,
                &mut outputs,
            );
            if !rooted_class_set_still_possible_u32(
                &identity.classes,
                &swapped_next,
                self.topology.initial_state,
                self.topology.real_state_count,
            ) {
                if profile_pair {
                    eprintln!(
                        "[glrmask/profile][terminal_interchangeability] canonical_pair={}<>{} outcome=class_set_mismatch round={} elapsed_ms={:.3}",
                        left,
                        right,
                        round,
                        pair_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                return None;
            }
            if same_equality_partition_pair_u32(
                identity_previous,
                &swapped_previous,
                &identity.classes,
                &swapped_next,
            ) {
                if profile_pair {
                    eprintln!(
                        "[glrmask/profile][terminal_interchangeability] canonical_pair={}<>{} outcome=stable round={} elapsed_ms={:.3}",
                        left,
                        right,
                        round,
                        pair_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                return self.interchange_map_from_classes(&identity.classes, &swapped_next);
            }
            swapped_previous = swapped_next;
        }
        drop(outputs);
        self.reference_interchange_map(left, right)
    }

    fn canonical_identity_map(&mut self) -> InterchangeMap {
        if let Some(map) = &self.canonical_identity_map {
            return map.clone();
        }
        let round = self.ensure_canonical_identity_stable_round();
        let map = self
            .interchange_map_from_classes(
                &self.canonical_rounds[round].classes,
                &self.canonical_rounds[round].classes,
            )
            .expect("identity canonical characterization must map every class to itself");
        self.canonical_identity_map = Some(map.clone());
        map
    }

    fn interchange_map_from_classes(
        &self,
        identity_classes: &[u32],
        swapped_classes: &[u32],
    ) -> Option<InterchangeMap> {
        debug_assert_eq!(identity_classes.len(), self.state_count());
        debug_assert_eq!(swapped_classes.len(), self.state_count());
        if identity_classes[self.topology.initial_state]
            != swapped_classes[self.topology.initial_state]
        {
            return None;
        }

        let mut target_representative_by_class = BTreeMap::<u32, u32>::new();
        let mut source_classes = BTreeMap::<u32, ()>::new();
        for state in 0..self.topology.real_state_count {
            source_classes.insert(identity_classes[state], ());
            target_representative_by_class
                .entry(swapped_classes[state])
                .or_insert(state as u32);
        }
        if source_classes.len() != target_representative_by_class.len()
            || source_classes
                .keys()
                .any(|class| !target_representative_by_class.contains_key(class))
        {
            return None;
        }

        let target_representative_for_source_state = (0..self.topology.real_state_count)
            .map(|source| target_representative_by_class.get(&identity_classes[source]).copied())
            .collect::<Option<Vec<_>>>()?;
        Some(InterchangeMap {
            target_representative_for_source_state,
        })
    }

    /// The set of output pairs visible on enabled-byte destinations is closed
    /// under every valid interchange. This filter is exact as a rejection
    /// condition: it does not accept a pair, it only avoids an impossible full
    /// characterization.
    fn observed_output_pair_set_is_swap_closed(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        let left = left as usize;
        let right = right as usize;
        self.observed_output_pair_mark_epoch = self.observed_output_pair_mark_epoch.wrapping_add(1);
        if self.observed_output_pair_mark_epoch == 0 {
            self.observed_output_pair_marks.fill(0);
            self.observed_output_pair_mark_epoch = 1;
        }
        let epoch = self.observed_output_pair_mark_epoch;
        let swap = Some((left, right));
        for ids in [
            &self.observed_output_pair_ids_by_terminal[left],
            &self.observed_output_pair_ids_by_terminal[right],
        ] {
            for &id in ids {
                let id = id as usize;
                if self.observed_output_pair_marks[id] == epoch {
                    continue;
                }
                self.observed_output_pair_marks[id] = epoch;
                let pair = &self.observed_output_pairs[id];
                let swapped = OutputPair {
                    finalizers: pair.finalizers.mapped(swap),
                    future_finalizers: pair.future_finalizers.mapped(swap),
                };
                if !self.observed_output_pair_lookup.contains_key(&swapped) {
                    return false;
                }
            }
        }
        true
    }

    fn swap_preserves_all_frozen_outputs(&self, left: TerminalID, right: TerminalID) -> bool {
        let left = left as usize;
        let right = right as usize;
        self.finalizer_states_by_terminal[left] == self.finalizer_states_by_terminal[right]
            && self.future_finalizer_states_by_terminal[left]
                == self.future_finalizer_states_by_terminal[right]
    }

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
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() && rounds % 256 == 0 {
                eprintln!(
                    "[glrmask/profile][terminal_interchangeability] exact_pair={}<> {} rounds={} identity_rounds={}",
                    left,
                    right,
                    rounds,
                    self.identity_rounds.len(),
                );
            }
            let swapped_next = self.characterize_round(
                &swapped_previous,
                &swapped_finalizers,
                &swapped_future_finalizers,
            );
            if !rooted_class_bijection_still_possible(
                &self.identity_rounds[rounds],
                &swapped_next,
                self.topology.initial_state,
                self.topology.real_state_count,
            ) {
                return PairCharacterization {
                    identity_hashes: self.identity_rounds[rounds].clone(),
                    swapped_hashes: swapped_next,
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
        if self.swap_preserves_all_frozen_outputs(left, right) {
            return Some(self.canonical_identity_map());
        }
        self.canonical_interchange_map(left, right)
    }

    fn reference_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let characterization = self.characterize_pair(left, right);
        self.interchange_map_from_characterization(&characterization)
    }

    fn interchange_map_from_characterization(
        &self,
        characterization: &PairCharacterization,
    ) -> Option<InterchangeMap> {
        if characterization.identity_hashes[self.topology.initial_state]
            != characterization.swapped_hashes[self.topology.initial_state]
        {
            return None;
        }

        let mut source_classes = BTreeMap::<CharacterizationHash, ()>::new();
        let mut target_representative_by_class = BTreeMap::<CharacterizationHash, u32>::new();
        for state in 0..self.topology.real_state_count {
            source_classes.insert(characterization.identity_hashes[state], ());
            target_representative_by_class
                .entry(characterization.swapped_hashes[state])
                .or_insert(state as u32);
        }
        if source_classes.len() != target_representative_by_class.len()
            || source_classes
                .keys()
                .any(|hash| !target_representative_by_class.contains_key(hash))
        {
            return None;
        }

        let target_representative_for_source_state = (0..self.topology.real_state_count)
            .map(|source| {
                target_representative_by_class
                    .get(&characterization.identity_hashes[source])
                    .copied()
            })
            .collect::<Option<Vec<_>>>()?;
        Some(InterchangeMap {
            target_representative_for_source_state,
        })
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
    let mut previous_to_next = FxHashMap::<CharacterizationHash, CharacterizationHash>::default();
    let mut next_to_previous = FxHashMap::<CharacterizationHash, CharacterizationHash>::default();
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

/// Equality of integer class ids denotes a partition within a single
/// refinement side. The concrete ids may change between rounds; only the
/// induced equality relation matters.
fn same_equality_partition_u32(previous: &[u32], next: &[u32]) -> bool {
    debug_assert_eq!(previous.len(), next.len());
    let mut previous_to_next = FxHashMap::<u32, u32>::default();
    let mut next_to_previous = FxHashMap::<u32, u32>::default();
    for (&old, &new) in previous.iter().zip(next) {
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

/// Integer-class counterpart of `same_equality_partition_pair`. Class ids are
/// shared across the identity and relabelled sides exactly when their complete
/// sparse signatures are equal, so the combined refinement must induce one
/// coherent old-to-new partition map across both sides.
fn same_equality_partition_pair_u32(
    identity_previous: &[u32],
    swapped_previous: &[u32],
    identity_next: &[u32],
    swapped_next: &[u32],
) -> bool {
    debug_assert_eq!(identity_previous.len(), swapped_previous.len());
    debug_assert_eq!(identity_previous.len(), identity_next.len());
    debug_assert_eq!(identity_previous.len(), swapped_next.len());
    let mut previous_to_next = FxHashMap::<u32, u32>::default();
    let mut next_to_previous = FxHashMap::<u32, u32>::default();
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

/// Canonical class ids are shared with the identity side exactly when their
/// full sparse signatures are equal. A missing root match or class-set member
/// can never be restored by a later refinement because partitions only split.
fn rooted_class_set_still_possible_u32(
    identity: &[u32],
    swapped: &[u32],
    initial_state: usize,
    real_state_count: usize,
) -> bool {
    if identity[initial_state] != swapped[initial_state] {
        return false;
    }
    let mut identity_classes = FxHashSet::<u32>::default();
    let mut swapped_classes = FxHashSet::<u32>::default();
    identity_classes.extend(identity[..real_state_count].iter().copied());
    swapped_classes.extend(swapped[..real_state_count].iter().copied());
    identity_classes == swapped_classes
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
    let mut identity_classes = FxHashSet::<CharacterizationHash>::default();
    let mut swapped_classes = FxHashSet::<CharacterizationHash>::default();
    identity_classes.extend(identity[..real_state_count].iter().copied());
    swapped_classes.extend(swapped[..real_state_count].iter().copied());
    identity_classes == swapped_classes
}

#[derive(Clone, Debug)]
pub(crate) struct TerminalInterchangeability {
    active_representatives: Vec<bool>,
    representative_for: Vec<TerminalID>,
    map_for_representative_member: BTreeMap<(TerminalID, TerminalID), InterchangeMap>,
}

impl TerminalInterchangeability {
    pub(crate) fn identity(active_terminals: &[bool]) -> Self {
        Self {
            active_representatives: active_terminals.to_vec(),
            representative_for: (0..active_terminals.len() as TerminalID).collect(),
            map_for_representative_member: BTreeMap::new(),
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

        let started_at = Instant::now();
        let topology = RestrictedTopology::new(tokenizer, relevant_bytes);
        let topology_edge_count = topology.edges.len();
        let topology_max_outdegree = topology.max_outdegree;
        let topology_byte_count = topology.bytes.len();
        let (root_candidate_groups, root_observed_states) =
            rooted_candidate_groups(tokenizer, &candidates, &topology);
        let root_candidate_pairs = root_candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        if root_candidate_pairs == 0 {
            return Self::identity(active_terminals);
        }

        let (structural_signatures, structural_color_count) = structural_candidate_signatures(
            tokenizer,
            active_terminals,
            &candidates,
            &topology,
        );
        let candidate_groups = refine_candidate_groups_by_structure(
            root_candidate_groups,
            &candidates,
            &structural_signatures,
        );
        let exact_candidate_pairs = candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            let mut group_size_histogram = BTreeMap::<usize, usize>::new();
            for group in &candidate_groups {
                *group_size_histogram.entry(group.len()).or_default() += 1;
            }
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] active={} selected_bytes={} sparse_edges={} max_outdegree={} root_observed_states={} root_candidate_pairs={} structural_colors={} structural_candidate_groups={} exact_candidate_pairs={} group_size_histogram={:?} filter_ms={:.3}",
                candidates.len(),
                topology_byte_count,
                topology_edge_count,
                topology_max_outdegree,
                root_observed_states,
                root_candidate_pairs,
                structural_color_count,
                candidate_groups.len(),
                exact_candidate_pairs,
                group_size_histogram,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if exact_candidate_pairs == 0 {
            return Self::identity(active_terminals);
        }

        let mut dfa = InterchangeabilityDfa::from_topology(tokenizer, active_terminals, topology);
        let mut result = Self::identity(active_terminals);
        let mut output_pair_rejections = 0usize;
        let mut output_invariant_checks = 0usize;
        let mut first_round_rejections = 0usize;
        let mut direct_exact_checks = 0usize;
        let mut accepted_representative_members = 0usize;

        // Accepted terminal swaps are automorphisms. Therefore (a b) and
        // (b c) imply (a c) by conjugation, so interchangeability is an
        // equivalence relation. Partition each candidate group by pivots,
        // keeping only the representative-to-member maps transport requires.
        for initial_group in candidate_groups {
            let mut unresolved = initial_group;
            while !unresolved.is_empty() {
                let representative = unresolved[0];
                let mut next_unresolved = Vec::with_capacity(unresolved.len().saturating_sub(1));
                for &terminal in &unresolved[1..] {
                    if !dfa.observed_output_pair_set_is_swap_closed(representative, terminal) {
                        output_pair_rejections += 1;
                        next_unresolved.push(terminal);
                        continue;
                    }
                    let map = if dfa.swap_preserves_all_frozen_outputs(representative, terminal) {
                        output_invariant_checks += 1;
                        Some(dfa.canonical_identity_map())
                    } else if !dfa.canonical_round_one_still_possible(representative, terminal) {
                        first_round_rejections += 1;
                        None
                    } else {
                        direct_exact_checks += 1;
                        dfa.interchange_map(representative, terminal)
                    };
                    if let Some(map) = map {
                        accepted_representative_members += 1;
                        result.representative_for[terminal as usize] = representative;
                        result.active_representatives[terminal as usize] = false;
                        result.map_for_representative_member.insert((representative, terminal), map);
                    } else {
                        next_unresolved.push(terminal);
                    }
                }
                unresolved = next_unresolved;
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] output_pair_rejections={} output_invariant_checks={} first_round_rejections={} direct_exact_checks={} quotient_certified={} sparse_quotient_certified={} sparse_cone_avg={:.1} sparse_cone_max={} sparse_cone_ms={:.3} sparse_refinement_ms={:.3} sparse_map_ms={:.3} accepted_representative_members={} total_ms={:.3}",
                output_pair_rejections,
                output_invariant_checks,
                first_round_rejections,
                direct_exact_checks,
                dfa.quotient_certified,
                dfa.sparse_quotient_certified,
                dfa.sparse_quotient_cone_classes_total as f64
                    / dfa.sparse_quotient_certified.max(1) as f64,
                dfa.sparse_quotient_cone_classes_max,
                dfa.sparse_quotient_cone_ns as f64 / 1_000_000.0,
                dfa.sparse_quotient_refinement_ns as f64 / 1_000_000.0,
                dfa.sparse_quotient_map_ns as f64 / 1_000_000.0,
                accepted_representative_members,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.map_for_representative_member.is_empty()
    }

    pub(crate) fn active_representatives(&self) -> &[bool] {
        &self.active_representatives
    }

    /// Per-terminal representative assignment: `representative_for[t]` is the
    /// terminal whose scanner behavior absorbs `t` under interchangeability
    /// (equal to `t` for representatives / identity terminals). Exposed for
    /// diagnostic dumps.
    pub(crate) fn representative_for(&self) -> &[TerminalID] {
        &self.representative_for
    }

    /// Scanner metadata remains visible for every raw terminal. Only edges for
    /// nonrepresentative active terminals are reconstructed through a transport
    /// mode rather than emitted directly.
    pub(crate) fn visible_output_raw_labels(&self) -> Vec<bool> {
        self.representative_for
            .iter()
            .enumerate()
            .map(|(terminal, &representative)| terminal as TerminalID == representative)
            .collect()
    }

    /// Replace each mapped raw scanner state with the representative of its
    /// ordinary exact terminal-DWA class. The transport map's label permutation
    /// is unchanged, and ordinary-equivalent scanner states admit exactly the
    /// same vocabulary continuations under that fixed permutation. This makes
    /// context sharing depend on semantic scanner destinations rather than
    /// incidental raw DFA state identities.
    pub(crate) fn canonicalize_transport_mode_states(
        &self,
        modes: &mut [TerminalNwaTransportMode],
        ordinary_state_map: &ManyToOneIdMap,
    ) {
        let state_count = ordinary_state_map.original_to_internal.len();
        for mode in modes {
            assert_eq!(
                mode.scanner_state_for_original.len(),
                state_count,
                "transport mode state domain must match ordinary state quotient",
            );
            for scanner_state in &mut mode.scanner_state_for_original {
                let raw = *scanner_state as usize;
                let internal = ordinary_state_map
                    .original_to_internal
                    .get(raw)
                    .copied()
                    .unwrap_or(u32::MAX);
                if internal != u32::MAX {
                    *scanner_state = ordinary_state_map
                        .representative_original_id_for_internal(internal)
                        .expect("ordinary state quotient missing representative");
                }
            }
        }
    }

    /// Refine an exact ordinary terminal-DWA state quotient only where a
    /// transport mode observes a different quotient destination. For a raw
    /// state `s`, the signature is
    /// `(Q(m_0(s)), Q(m_1(s)), …)`, where `Q` is the ordinary exact state map
    /// and `m_i` is a scanner transport. States with equal signatures remain
    /// interchangeable for every transported output: each mode starts from
    /// ordinary-equivalent scanner states and then applies the same fixed label
    /// permutation.
    ///
    /// The signature is *not* materialized. Starting from one class, each mode
    /// exactly refines the current partition by the next `Q(m_i(s))` value. The
    /// resulting class ID is therefore an exact canonical encoding of the full
    /// vector, while scratch space stays O(number of raw lexer states) rather
    /// than O(states × modes).
    pub(crate) fn transport_coordinate_quotient(
        &self,
        ordinary_state_map: &ManyToOneIdMap,
        modes: &[TerminalNwaTransportMode],
    ) -> ManyToOneIdMap {
        assert!(!modes.is_empty(), "transport coordinate quotient needs a mode");
        let state_count = ordinary_state_map.original_to_internal.len();
        let mut class_for_state = vec![0u32; state_count];
        let mut next_class_for_state = vec![0u32; state_count];
        let mut class_for_pair = FxHashMap::<(u32, u64), u32>::default();

        for mode in modes {
            assert_eq!(
                mode.scanner_state_for_original.len(),
                state_count,
                "transport mode state domain must match ordinary state quotient",
            );
            class_for_pair.clear();
            let mut next_class_count = 0u32;

            for (source_state, (&prior_class, &target_state)) in class_for_state
                .iter()
                .zip(mode.scanner_state_for_original.iter())
                .enumerate()
            {
                let target_state = target_state as usize;
                let mapped = ordinary_state_map
                    .original_to_internal
                    .get(target_state)
                    .copied()
                    .unwrap_or(u32::MAX);
                // Unmapped targets are outside the ordinary quotient's proof
                // domain. Keep their raw identity distinct rather than merging
                // them by accident.
                let target_key = if mapped == u32::MAX {
                    (1u64 << 32) | target_state as u64
                } else {
                    mapped as u64
                };
                let class = match class_for_pair.entry((prior_class, target_key)) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let class = next_class_count;
                        next_class_count += 1;
                        entry.insert(class);
                        class
                    }
                };
                next_class_for_state[source_state] = class;
            }
            std::mem::swap(&mut class_for_state, &mut next_class_for_state);
        }

        let class_count = class_for_state
            .iter()
            .copied()
            .max()
            .map_or(0usize, |class| class as usize + 1);
        let mut representatives = vec![u32::MAX; class_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            let representative = &mut representatives[class as usize];
            if *representative == u32::MAX {
                *representative = state as u32;
            }
        }

        ManyToOneIdMap::from_original_to_internal_with_representatives(
            class_for_state,
            class_count as u32,
            representatives,
        )
    }

    pub(crate) fn terminal_nwa_transport_modes(&self) -> Option<Vec<TerminalNwaTransportMode>> {
        let state_count = self
            .map_for_representative_member
            .values()
            .next()
            .map(|map| map.target_representative_for_source_state.len())?;
        let identity_labels = (0..self.representative_for.len() as TerminalID).collect::<Vec<_>>();
        let mut modes = vec![TerminalNwaTransportMode {
            scanner_state_for_original: (0..state_count as u32).collect(),
            terminal_map: identity_labels.clone(),
        }];

        for (&(representative, member), map) in &self.map_for_representative_member {
            let mut terminal_map = identity_labels.clone();
            terminal_map[representative as usize] = member;
            terminal_map[member as usize] = representative;
            modes.push(TerminalNwaTransportMode {
                scanner_state_for_original: map.arbitrary_target_representatives(),
                terminal_map,
            });
        }
        Some(modes)
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
    fn transport_coordinate_quotient_matches_full_mode_signature() {
        let ordinary = ManyToOneIdMap::from_original_to_internal_with_representatives(
            vec![0, 0, 1, 1, 2, 2, 3],
            4,
            vec![0, 2, 4, 6],
        );
        let modes = vec![
            TerminalNwaTransportMode {
                scanner_state_for_original: vec![0, 1, 2, 3, 4, 5, 6],
                terminal_map: vec![0],
            },
            TerminalNwaTransportMode {
                scanner_state_for_original: vec![1, 0, 3, 2, 5, 4, 6],
                terminal_map: vec![0],
            },
            TerminalNwaTransportMode {
                scanner_state_for_original: vec![2, 3, 0, 1, 6, 6, 4],
                terminal_map: vec![0],
            },
        ];
        let plan = TerminalInterchangeability::identity(&[true]);
        let quotient = plan.transport_coordinate_quotient(&ordinary, &modes);

        let signatures = (0..ordinary.original_to_internal.len())
            .map(|source| {
                modes
                    .iter()
                    .map(|mode| {
                        ordinary.original_to_internal
                            [mode.scanner_state_for_original[source] as usize]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for left in 0..signatures.len() {
            for right in 0..signatures.len() {
                assert_eq!(
                    quotient.original_to_internal[left] == quotient.original_to_internal[right],
                    signatures[left] == signatures[right],
                    "signature quotient disagreed for states {left} and {right}",
                );
            }
        }

        let mut canonical_modes = modes.clone();
        plan.canonicalize_transport_mode_states(&mut canonical_modes, &ordinary);
        let canonical_quotient = plan.transport_coordinate_quotient(&ordinary, &canonical_modes);
        for left in 0..signatures.len() {
            for right in 0..signatures.len() {
                assert_eq!(
                    quotient.original_to_internal[left] == quotient.original_to_internal[right],
                    canonical_quotient.original_to_internal[left]
                        == canonical_quotient.original_to_internal[right],
                    "canonical transport changed the quotient for states {left} and {right}",
                );
            }
        }
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
        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, true], &[true; 256]);
        assert!(dfa.interchange_map(0, 1).is_none());
    }

    #[test]
    fn identical_literals_have_a_rooted_interchange_map() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
        ]);
        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, true], &[true; 256]);
        let map = dfa.interchange_map(0, 1).expect("identical literals must transport");
        let root = tokenizer.initial_state_id() as usize;
        assert_eq!(map.target_representative_for_source_state[root], tokenizer.initial_state_id());
        let representatives = map.arbitrary_target_representatives();
        assert_eq!(map.target_representative_for_source_state[root], representatives[root]);
        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert_eq!(plan.active_representatives.iter().filter(|&&active| active).count(), 1);
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
        assert_eq!(plan.active_representatives.iter().filter(|&&active| active).count(), 1);
        assert_eq!(plan.representative_for, vec![0, 0, 0, 0]);
    }

    #[test]
    fn unobserved_byte_nonidentity_map_matches_hash_reference() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ]);
        let mut relevant_bytes = [false; 256];
        relevant_bytes[b'c' as usize] = true;
        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, true], &relevant_bytes);
        let optimized = dfa
            .interchange_map(0, 1)
            .expect("unobserved terminals must transport");
        let reference = dfa
            .reference_interchange_map(0, 1)
            .expect("hash reference must transport the same pair");
        assert_eq!(optimized, reference);
        assert!(
            optimized
                .target_representative_for_source_state
                .iter()
                .enumerate()
                .any(|(state, &target)| target != state as u32),
            "the MRE requires a nonidentity raw scanner map",
        );
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
        let restricted = InterchangeabilityDfa::new(&tokenizer, &[true, true], &only_a);
        assert_eq!(restricted.topology.bytes, vec![b'a']);
        assert_eq!(restricted.destination_for_slot(after_a, 0), restricted.dead_state());
        assert!(restricted.output_at(&restricted.future_finalizers, after_a).contains(1));
    }

    #[test]
    fn unobserved_outputs_do_not_split_structural_prefilter() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bb".to_vec()),
        ]);
        let active = vec![true, true];
        let candidates = vec![0, 1];
        let mut only_x = [false; 256];
        only_x[b'x' as usize] = true;

        // No raw terminal-output state is an enabled-byte destination. The
        // reference consequently observes only the synthetic dead output.
        let topology = RestrictedTopology::new(&tokenizer, &only_x);
        let (root_groups, _) = rooted_candidate_groups(&tokenizer, &candidates, &topology);
        let (structural_signatures, _) =
            structural_candidate_signatures(&tokenizer, &active, &candidates, &topology);
        let filtered_groups =
            refine_candidate_groups_by_structure(root_groups, &candidates, &structural_signatures);
        assert!(group_contains_pair(&filtered_groups, 0, 1));

        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &active, &only_x);
        assert!(dfa.interchange_map(0, 1).is_some());
        let plan = TerminalInterchangeability::build(&tokenizer, &active, &only_x, None);
        assert_eq!(plan.active_representatives.iter().filter(|&&active| active).count(), 1);
    }

    #[test]
    fn inactive_outputs_are_not_observed() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ]);
        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, false, true], &[true; 256]);
        assert!(dfa.interchange_map(0, 2).is_some());
    }

    fn group_contains_pair(groups: &[Vec<TerminalID>], left: TerminalID, right: TerminalID) -> bool {
        groups.iter().any(|group| {
            group.contains(&left) && group.contains(&right)
        })
    }

    #[test]
    fn exact_prefilters_never_reject_a_reference_interchange_pair() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"different".to_vec()),
            Expr::U8Seq(b"differs".to_vec()),
        ]);
        let active = vec![true; 4];
        let candidates = (0..4).collect::<Vec<TerminalID>>();
        let relevant_bytes = [true; 256];
        let topology = RestrictedTopology::new(&tokenizer, &relevant_bytes);
        let (root_groups, _) = rooted_candidate_groups(&tokenizer, &candidates, &topology);
        let (structural_signatures, _) = structural_candidate_signatures(
            &tokenizer,
            &active,
            &candidates,
            &topology,
        );
        let filtered_groups = refine_candidate_groups_by_structure(
            root_groups.clone(),
            &candidates,
            &structural_signatures,
        );
        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &active, &relevant_bytes);
        for (index, &left) in candidates.iter().enumerate() {
            for &right in &candidates[index + 1..] {
                if let Some(left_to_right) = dfa.interchange_map(left, right) {
                    let right_to_left = dfa
                        .interchange_map(right, left)
                        .expect("the same transposition must produce the same map");
                    assert!(
                        group_contains_pair(&root_groups, left, right),
                        "root prefilter rejected exact pair {left} <-> {right}",
                    );
                    assert!(
                        group_contains_pair(&filtered_groups, left, right),
                        "structural prefilter rejected exact pair {left} <-> {right}",
                    );
                    assert!(
                        dfa.canonical_round_one_still_possible(left, right),
                        "first-round prefilter rejected exact pair {left} <-> {right}",
                    );
                    assert_eq!(
                        left_to_right.target_representative_for_source_state,
                        right_to_left.target_representative_for_source_state,
                        "the reversed pair call must be operationally identical",
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_sparse_quotient_matches_hash_reference_for_all_pairs() {
        let tokenizer = tokenizer(vec![Expr::U8Seq(b"same".to_vec()), Expr::U8Seq(b"same".to_vec()), Expr::U8Seq(b"sample".to_vec()), Expr::U8Seq(b"simple".to_vec()), Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"b".to_vec()), Expr::U8Seq(b"ba".to_vec())]);
        let active = vec![true; 8];
        for left in 0..active.len() as TerminalID {
            for right in left + 1..active.len() as TerminalID {
                let mut canonical = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
                let mut reference = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
                assert_eq!(canonical.interchange_map(left, right), reference.reference_interchange_map(left, right), "canonical refinement disagreed with hash reference for {left} <-> {right}");
            }
        }
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

    #[test]
    fn combined_integer_partition_stability_rejects_cross_side_split() {
        assert!(same_equality_partition_pair_u32(
            &[0, 0, 1],
            &[0, 0, 1],
            &[4, 4, 5],
            &[4, 4, 5],
        ));
        // Each side is individually stable, but the shared old class `0`
        // refines differently across sides. This cannot certify a transport.
        assert!(!same_equality_partition_pair_u32(
            &[0, 0],
            &[0, 0],
            &[4, 4],
            &[4, 5],
        ));
    }
}
