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

use std::collections::{BTreeMap, BTreeSet, VecDeque, hash_map::Entry};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use super::nwa_builder::{TerminalNwaTransportMode, TransportScannerStateMap};
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::automata::weighted::dwa::{DWAState, DWA};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::weight::{SharedTokenSet, Weight, shared_rangeset};
use crate::ds::bitset::BitSet;
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

/// Sparse, sorted active-terminal labels for one frozen output family.
///
/// TI partitions typically observe only a handful of terminals at a lexer
/// state. Keeping a full active-terminal bitmap for every state therefore
/// spends most planning time allocating and copying zero words. Raw terminal
/// ids remain exact labels: disabled terminals are simply absent for the
/// current iterative round.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct OutputBits(SmallVec<[TerminalID; 4]>);

impl OutputBits {
    fn new(_words: usize) -> Self { Self(SmallVec::new()) }

    fn from_active(terminals: &[TerminalID], active_terminals: &[bool]) -> Self {
        let mut output = SmallVec::<[TerminalID; 4]>::new();
        for &terminal in terminals {
            if active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
            {
                output.push(terminal);
            }
        }
        Self(output)
    }

    fn contains(&self, terminal: usize) -> bool {
        self.0.binary_search(&(terminal as TerminalID)).is_ok()
    }

    fn mapped(&self, swap: Option<(usize, usize)>) -> Self {
        let Some((left, right)) = swap else { return self.clone(); };
        if left == right { return self.clone(); }
        let left = left as TerminalID;
        let right = right as TerminalID;
        let left_present = self.0.binary_search(&left).ok();
        let right_present = self.0.binary_search(&right).ok();
        if left_present.is_some() == right_present.is_some() { return self.clone(); }
        let (source_index, replacement) = if let Some(index) = left_present {
            (index, right)
        } else {
            (right_present.expect("one swapped terminal must be present"), left)
        };
        let mut terminals = self.0.clone();
        terminals.remove(source_index);
        let insertion = terminals
            .binary_search(&replacement)
            .expect_err("the replacement terminal must be absent");
        terminals.insert(insertion, replacement);
        Self(terminals)
    }

    fn append_to(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&(self.0.len() as u32).to_le_bytes());
        for &terminal in &self.0 {
            output.extend_from_slice(&(terminal as u32).to_le_bytes());
        }
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
struct CanonicalSignature(SmallVec<[CanonicalComponent; 8]>);

struct CanonicalRound {
    classes: Vec<u32>,
    /// One raw-state representative for every exact class in this round.
    /// Identity-round construction hashes the sparse row directly, then
    /// compares it against these representatives on every hash match.  This
    /// avoids allocating a `CanonicalSignature` for every raw state while
    /// retaining collision-free equality semantics.
    representative_by_class: Vec<u32>,
    classes_by_signature_hash: FxHashMap<u64, SmallVec<[u32; 1]>>,
}

/// Stable identity characterization collapsed to its exact state quotient.
/// The quotient does not merge tokenizer states; it is only a cached view used
/// to certify a terminal-swap automorphism before materializing its raw map.
struct CanonicalQuotient {
    class_for_state: Arc<[u32]>,
    representative_by_class: Arc<[u32]>,
    reverse_predecessors: Arc<[Vec<u32>]>,
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

/// The stable identity quotient data needed by the common support-transport
/// proof.  It deliberately omits canonical row signatures and historical
/// round projections, which are required only by the slower generic fallback.
struct SupportQuotient {
    class_for_state: Arc<[u32]>,
    representative_by_class: Arc<[u32]>,
    reverse_predecessors: Arc<[Vec<u32>]>,
}

/// Per-swap output-label relabelling. The immutable base ids represent the
/// original frozen output pairs; ids allocated after `base_count` are local to
/// this swap and compare equal only to the same full mapped pair.
struct SwappedOutputIds<'a> {
    base_pairs: &'a [OutputPair],
    base_lookup: &'a FxHashMap<OutputPair, u32>,
    left: usize,
    right: usize,
    mapped: FxHashMap<u32, u32>,
    local: FxHashMap<OutputPair, u32>,
}

impl<'a> SwappedOutputIds<'a> {
    fn new(
        base_pairs: &'a [OutputPair],
        base_lookup: &'a FxHashMap<OutputPair, u32>,
        left: usize,
        right: usize,
    ) -> Self {
        Self {
            base_pairs,
            base_lookup,
            left,
            right,
            mapped: FxHashMap::default(),
            local: FxHashMap::default(),
        }
    }

    fn id(&mut self, output: u32) -> u32 {
        if let Some(&cached) = self.mapped.get(&output) {
            return cached;
        }
        let index = output as usize;
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
        self.mapped.insert(output, id);
        id
    }
}

/// The support-transposition proof touches only a handful of output-pair ids
/// per candidate. Keep its relabelling cache inline; unrestricted fallback
/// refinement retains `SwappedOutputIds` above.
struct SparseSwappedOutputIds<'a> {
    base_pairs: &'a [OutputPair],
    base_lookup: &'a FxHashMap<OutputPair, u32>,
    left: usize,
    right: usize,
    mapped: SmallVec<[(u32, u32); 16]>,
    local: SmallVec<[(OutputPair, u32); 4]>,
}

impl<'a> SparseSwappedOutputIds<'a> {
    fn new(
        base_pairs: &'a [OutputPair],
        base_lookup: &'a FxHashMap<OutputPair, u32>,
        left: usize,
        right: usize,
    ) -> Self {
        Self {
            base_pairs,
            base_lookup,
            left,
            right,
            mapped: SmallVec::new(),
            local: SmallVec::new(),
        }
    }

    fn id(&mut self, output: u32) -> u32 {
        if let Some((_, id)) = self.mapped.iter().find(|&&(seen, _)| seen == output) {
            return *id;
        }
        let mapped = OutputPair {
            finalizers: self.base_pairs[output as usize]
                .finalizers
                .mapped(Some((self.left, self.right))),
            future_finalizers: self.base_pairs[output as usize]
                .future_finalizers
                .mapped(Some((self.left, self.right))),
        };
        let id = if let Some(&base) = self.base_lookup.get(&mapped) {
            base
        } else if let Some((_, existing)) = self.local.iter().find(|(seen, _)| *seen == mapped) {
            *existing
        } else {
            let next = self.base_pairs.len() as u32 + self.local.len() as u32;
            self.local.push((mapped, next));
            next
        };
        self.mapped.push((output, id));
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
    raw_state_count: usize,
    raw_to_restricted_state: Arc<[u32]>,
    raw_representative_by_restricted_state: Arc<[u32]>,
    real_state_count: usize,
    initial_state: usize,
    max_outdegree: usize,
}

impl RestrictedTopology {
    fn new(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        preexisting_state_map: Option<&ManyToOneIdMap>,
    ) -> Self {
        let bytes = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect::<Vec<_>>();
        let raw_state_count = tokenizer.num_states() as usize;
        let (raw_to_restricted_state, raw_representative_by_restricted_state):
            (Arc<[u32]>, Arc<[u32]>) =
            if let Some(state_map) = preexisting_state_map {
                assert_eq!(
                    state_map.original_to_internal.len(),
                    raw_state_count,
                    "preexisting TI state map must cover every raw tokenizer state",
                );
                assert!(
                    state_map.original_to_internal.iter().all(|&state| {
                        (state as usize) < state_map.representative_original_ids.len()
                    }),
                    "preexisting TI state map must not contain unmapped raw states",
                );
                (
                    Arc::from(state_map.original_to_internal.clone()),
                    Arc::from(state_map.representative_original_ids.clone()),
                )
            } else {
                (
                    Arc::from((0..raw_state_count as u32).collect::<Vec<_>>()),
                    Arc::from((0..raw_state_count as u32).collect::<Vec<_>>()),
                )
            };
        let real_state_count = raw_representative_by_restricted_state.len();
        let mut edge_offsets = Vec::with_capacity(real_state_count + 2);
        let mut edges = Vec::new();
        let mut max_outdegree = 0usize;
        edge_offsets.push(0);
        for state in 0..real_state_count {
            let start = edges.len();
            let raw_state = raw_representative_by_restricted_state[state];
            for (byte, target) in tokenizer.transitions_from(raw_state) {
                if relevant_bytes[byte as usize] {
                    edges.push((byte, raw_to_restricted_state[target as usize]));
                }
            }
            max_outdegree = max_outdegree.max(edges.len() - start);
            edge_offsets.push(edges.len() as u32);
        }
        // Synthetic dead has no real edges: every enabled byte loops to itself.
        edge_offsets.push(edges.len() as u32);
        let initial_state =
            raw_to_restricted_state[tokenizer.initial_state_id() as usize] as usize;
        Self {
            bytes,
            edge_offsets,
            edges,
            raw_state_count,
            raw_to_restricted_state,
            raw_representative_by_restricted_state,
            real_state_count,
            initial_state,
            max_outdegree,
        }
    }

    fn raw_state_for_restricted(&self, state: usize) -> u32 {
        self.raw_representative_by_restricted_state[state]
    }

    fn raw_class_for_restricted_classes(&self, class_for_restricted: &[u32]) -> Arc<[u32]> {
        Arc::from(
            self.raw_to_restricted_state
                .iter()
                .map(|&state| class_for_restricted[state as usize])
                .collect::<Vec<_>>(),
        )
    }

    fn raw_representatives_for_classes(
        &self,
        representative_restricted: &[u32],
    ) -> Arc<[u32]> {
        Arc::from(
            representative_restricted
                .iter()
                .map(|&state| {
                    if state as usize == self.real_state_count {
                        self.raw_state_count as u32
                    } else {
                        self.raw_representative_by_restricted_state[state as usize]
                    }
                })
                .collect::<Vec<_>>(),
        )
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExternalOutputSignature {
    entries: Box<[(u8, Box<[TerminalID]>, Box<[TerminalID]>)]>,
}

/// Immutable raw lexer evidence that is independent of a historical TI round's
/// active-terminal mask.  The projected `OutputBits` remain round-local, but
/// their source columns and the reverse restricted topology are reusable.
struct TiRawDiscoveryData {
    finalizer_terminals_by_state: Vec<Box<[TerminalID]>>,
    future_finalizer_terminals_by_state: Vec<Box<[TerminalID]>>,
    raw_output_pair_by_state: Arc<[u32]>,
    raw_output_pair_representatives: Arc<[u32]>,
    finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    future_finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    reverse_predecessors: Arc<[Vec<u32>]>,
    reverse_edges: Arc<[Vec<(u8, u32)>]>,
    observed_destinations: Arc<[bool]>,
}

impl TiRawDiscoveryData {
    fn new(
        tokenizer: &Tokenizer,
        topology: &RestrictedTopology,
        active_terminals: Option<&[bool]>,
    ) -> Self {
        let terminal_count = tokenizer.num_terminals() as usize;
        let mut finalizer_terminals_by_state = Vec::with_capacity(topology.real_state_count);
        let mut future_finalizer_terminals_by_state =
            Vec::with_capacity(topology.real_state_count);
        let mut finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        let mut future_finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];

        for state in 0..topology.real_state_count {
            let raw_state = topology.raw_state_for_restricted(state);
            let finalizers = tokenizer
                .matched_terminals_iter(raw_state)
                .filter(|&terminal| {
                    active_terminals
                        .map_or(true, |active| active[terminal as usize])
                })
                .collect::<Vec<_>>();
            for &terminal in &finalizers {
                finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
            finalizer_terminals_by_state.push(finalizers.into_boxed_slice());

            let future_finalizers = tokenizer
                .possible_future_terminals_iter(raw_state)
                .filter(|&terminal| {
                    active_terminals
                        .map_or(true, |active| active[terminal as usize])
                })
                .collect::<Vec<_>>();
            for &terminal in &future_finalizers {
                future_finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
            future_finalizer_terminals_by_state.push(future_finalizers.into_boxed_slice());
        }

        let mut raw_output_pair_lookup =
            FxHashMap::<(&[TerminalID], &[TerminalID]), u32>::default();
        let mut raw_output_pair_by_state = Vec::with_capacity(topology.real_state_count);
        let mut raw_output_pair_representatives = Vec::new();
        for state in 0..topology.real_state_count {
            let key = (
                finalizer_terminals_by_state[state].as_ref(),
                future_finalizer_terminals_by_state[state].as_ref(),
            );
            let id = if let Some(&id) = raw_output_pair_lookup.get(&key) {
                id
            } else {
                let id = raw_output_pair_representatives.len() as u32;
                raw_output_pair_lookup.insert(key, id);
                raw_output_pair_representatives.push(state as u32);
                id
            };
            raw_output_pair_by_state.push(id);
        }

        let mut reverse_predecessors = vec![Vec::<u32>::new(); topology.real_state_count];
        let mut reverse_edges = vec![Vec::<(u8, u32)>::new(); topology.real_state_count];
        for source in 0..topology.real_state_count {
            for &(byte, destination) in topology.edges_from(source) {
                reverse_predecessors[destination as usize].push(source as u32);
                reverse_edges[destination as usize].push((byte, source as u32));
            }
        }
        for edges in &mut reverse_edges {
            edges.sort_unstable();
        }

        Self {
            finalizer_terminals_by_state,
            future_finalizer_terminals_by_state,
            raw_output_pair_by_state: raw_output_pair_by_state.into(),
            raw_output_pair_representatives: raw_output_pair_representatives.into(),
            finalizer_states_by_terminal: finalizer_states_by_terminal.into(),
            future_finalizer_states_by_terminal: future_finalizer_states_by_terminal.into(),
            reverse_predecessors: reverse_predecessors.into(),
            reverse_edges: reverse_edges.into(),
            observed_destinations: topology.observed_destinations().into(),
        }
    }

    fn project_from(base: &TiRawDiscoveryData, topology: &RestrictedTopology) -> Self {
        let terminal_count = base.finalizer_states_by_terminal.len();
        let mut finalizer_terminals_by_state = Vec::with_capacity(topology.real_state_count);
        let mut future_finalizer_terminals_by_state =
            Vec::with_capacity(topology.real_state_count);
        let mut finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        let mut future_finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];

        for state in 0..topology.real_state_count {
            let raw_state = topology.raw_state_for_restricted(state) as usize;
            let finalizers = base.finalizer_terminals_by_state[raw_state].clone();
            for &terminal in finalizers.iter() {
                finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
            finalizer_terminals_by_state.push(finalizers);

            let future_finalizers = base.future_finalizer_terminals_by_state[raw_state].clone();
            for &terminal in future_finalizers.iter() {
                future_finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
            future_finalizer_terminals_by_state.push(future_finalizers);
        }

        let mut raw_output_pair_lookup =
            FxHashMap::<(&[TerminalID], &[TerminalID]), u32>::default();
        let mut raw_output_pair_by_state = Vec::with_capacity(topology.real_state_count);
        let mut raw_output_pair_representatives = Vec::new();
        for state in 0..topology.real_state_count {
            let key = (
                finalizer_terminals_by_state[state].as_ref(),
                future_finalizer_terminals_by_state[state].as_ref(),
            );
            let id = if let Some(&id) = raw_output_pair_lookup.get(&key) {
                id
            } else {
                let id = raw_output_pair_representatives.len() as u32;
                raw_output_pair_lookup.insert(key, id);
                raw_output_pair_representatives.push(state as u32);
                id
            };
            raw_output_pair_by_state.push(id);
        }

        let mut reverse_predecessors = vec![Vec::<u32>::new(); topology.real_state_count];
        let mut reverse_edges = vec![Vec::<(u8, u32)>::new(); topology.real_state_count];
        for source in 0..topology.real_state_count {
            for &(byte, destination) in topology.edges_from(source) {
                reverse_predecessors[destination as usize].push(source as u32);
                reverse_edges[destination as usize].push((byte, source as u32));
            }
        }
        for edges in &mut reverse_edges {
            edges.sort_unstable();
        }

        Self {
            finalizer_terminals_by_state,
            future_finalizer_terminals_by_state,
            raw_output_pair_by_state: raw_output_pair_by_state.into(),
            raw_output_pair_representatives: raw_output_pair_representatives.into(),
            finalizer_states_by_terminal: finalizer_states_by_terminal.into(),
            future_finalizer_states_by_terminal: future_finalizer_states_by_terminal.into(),
            reverse_predecessors: reverse_predecessors.into(),
            reverse_edges: reverse_edges.into(),
            observed_destinations: topology.observed_destinations().into(),
        }
    }
}

/// Static per-L2P-partition TI data.  The restricted raw lexer topology and
/// root observation depend only on vocabulary bytes, not on the historical
/// active-terminal mask of an iterative TI round.
#[derive(Clone)]
pub(crate) struct TiDiscoveryContext {
    topology: Arc<RestrictedTopology>,
    raw: Arc<TiRawDiscoveryData>,
    root_output_signatures: Vec<RootOutputSignature>,
    root_observed_states: usize,
}

impl TiDiscoveryContext {
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        preexisting_state_map: Option<&ManyToOneIdMap>,
    ) -> Self {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let topology = Arc::new(RestrictedTopology::new(
            tokenizer,
            relevant_bytes,
            preexisting_state_map,
        ));
        let topology_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let raw = Arc::new(TiRawDiscoveryData::new(tokenizer, &topology, None));
        let raw_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - topology_ms)
            .unwrap_or(0.0);
        let (root_output_signatures, root_observed_states) =
            root_output_signatures(tokenizer, &topology);
        let result = Self {
            topology,
            raw,
            root_output_signatures,
            root_observed_states,
        };
        if let Some(started_at) = started_at {
            eprintln!(
                "[glrmask/profile][ti_context] states={} selected_bytes={} topology_ms={:.3} raw_ms={:.3} root_ms={:.3} total_ms={:.3}",
                result.topology.real_state_count,
                result.topology.bytes.len(),
                topology_ms,
                raw_ms,
                started_at.elapsed().as_secs_f64() * 1000.0 - topology_ms - raw_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    pub(crate) fn from_base_with_state_map(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        preexisting_state_map: &ManyToOneIdMap,
        base: &TiDiscoveryContext,
    ) -> Self {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let topology = Arc::new(RestrictedTopology::new(
            tokenizer,
            relevant_bytes,
            Some(preexisting_state_map),
        ));
        let topology_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let raw = Arc::new(TiRawDiscoveryData::project_from(&base.raw, &topology));
        let raw_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - topology_ms)
            .unwrap_or(0.0);
        let (root_output_signatures, root_observed_states) =
            root_output_signatures(tokenizer, &topology);
        let result = Self {
            topology,
            raw,
            root_output_signatures,
            root_observed_states,
        };
        if profile_timing {
            eprintln!(
                "[glrmask/profile][ti_context_projected] states={} selected_bytes={} topology_ms={:.3} raw_ms={:.3} root_ms={:.3} total_ms={:.3}",
                result.topology.real_state_count,
                result.topology.bytes.len(),
                topology_ms,
                raw_ms,
                started_at
                    .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - topology_ms - raw_ms)
                    .unwrap_or(0.0),
                started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0),
            );
        }
        result
    }
}

/// Build the strongest globally safe scanner quotient available from the
/// complete frozen terminal labelling on one selected-byte alphabet.  The
/// result is a total right congruence and therefore can be used by TI without
/// changing raw-coordinate transport semantics.
pub(crate) fn full_frozen_scanner_state_quotient(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
) -> ManyToOneIdMap {
    let context = TiDiscoveryContext::new(tokenizer, relevant_bytes, None);
    let active = vec![true; tokenizer.num_terminals() as usize];
    let mut dfa = InterchangeabilityDfa::from_context(&active, &context);
    dfa.raw_stable_identity_state_map()
}

pub(crate) fn validate_full_frozen_scanner_state_quotient(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    state_map: &ManyToOneIdMap,
) {
    assert_eq!(
        state_map.original_to_internal.len(),
        tokenizer.num_states() as usize,
        "global frozen scanner quotient must cover every tokenizer state",
    );
    for raw_state in 0..tokenizer.num_states() as usize {
        let class = state_map.original_to_internal[raw_state] as usize;
        let representative = state_map.representative_original_ids[class];
        assert_eq!(
            tokenizer
                .matched_terminals_iter(raw_state as u32)
                .collect::<Vec<_>>(),
            tokenizer
                .matched_terminals_iter(representative)
                .collect::<Vec<_>>(),
            "global frozen scanner quotient changed finalizers at raw state {raw_state}",
        );
        assert_eq!(
            tokenizer
                .possible_future_terminals_iter(raw_state as u32)
                .collect::<Vec<_>>(),
            tokenizer
                .possible_future_terminals_iter(representative)
                .collect::<Vec<_>>(),
            "global frozen scanner quotient changed future finalizers at raw state {raw_state}",
        );
        for byte in 0..=255u8 {
            if !relevant_bytes[byte as usize] {
                continue;
            }
            let target = tokenizer
                .step(raw_state as u32, byte)
                .map(|target| state_map.original_to_internal[target as usize]);
            let representative_target = tokenizer
                .step(representative, byte)
                .map(|target| state_map.original_to_internal[target as usize]);
            assert_eq!(
                target,
                representative_target,
                "global frozen scanner quotient is not a right congruence at raw state {raw_state}, byte {byte}",
            );
        }
    }
}

/// Conservative literal-family buckets.  Equal selected-byte projections only
/// nominate a family; the quotient-level fiber proof below is the acceptance
/// condition.
fn literal_projection_groups(
    tokenizer: &Tokenizer,
    active: &[bool],
    context: &TiDiscoveryContext,
    ignore: Option<TerminalID>,
) -> Vec<Vec<TerminalID>> {
    let mut selected = [false; 256];
    for &byte in &context.topology.bytes {
        selected[byte as usize] = true;
    }
    let mut groups = BTreeMap::<Vec<u8>, Vec<TerminalID>>::new();
    for (terminal, &is_active) in active.iter().enumerate() {
        let terminal = terminal as TerminalID;
        if !is_active || Some(terminal) == ignore {
            continue;
        }
        let Some(bytes) = tokenizer.literal_terminal_bytes(terminal) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let projection = bytes
            .iter()
            .copied()
            .filter(|byte| selected[*byte as usize])
            .collect::<Vec<_>>();
        groups.entry(projection).or_default().push(terminal);
    }
    let groups = groups
        .into_values()
        .filter(|group| group.len() >= 2)
        .collect::<Vec<_>>();
    if std::env::var_os("GLRMASK_PROFILE_L2P_LITERAL_PREFIX_SPLIT").is_none() {
        return groups;
    }

    let mut terminals_by_literal = BTreeMap::<Vec<u8>, Vec<TerminalID>>::new();
    let mut literal_by_terminal = vec![None::<Vec<u8>>; active.len()];
    for (terminal, &is_active) in active.iter().enumerate() {
        let terminal = terminal as TerminalID;
        if !is_active || Some(terminal) == ignore {
            continue;
        }
        if let Some(bytes) = tokenizer.literal_terminal_bytes(terminal) {
            literal_by_terminal[terminal as usize] = Some(bytes.clone());
            terminals_by_literal.entry(bytes).or_default().push(terminal);
        }
    }
    let literals = terminals_by_literal.into_iter().collect::<Vec<_>>();
    let mut refined = Vec::new();
    for group in groups {
        let mut member = vec![false; active.len()];
        let group_bytes = group
            .iter()
            .map(|&terminal| {
                member[terminal as usize] = true;
                literal_by_terminal[terminal as usize]
                    .as_deref()
                    .expect("literal projection group must retain literal bytes")
            })
            .collect::<Vec<_>>();
        let common_prefix_len = group_bytes.iter().skip(1).fold(
            group_bytes[0].len(),
            |prefix_len, bytes| {
                prefix_len.min(
                    group_bytes[0]
                        .iter()
                        .zip(bytes.iter())
                        .take_while(|(left, right)| left == right)
                        .count(),
                )
            },
        );
        let mut buckets = BTreeMap::<bool, Vec<TerminalID>>::new();
        for terminal in group {
            let bytes = literal_by_terminal[terminal as usize]
                .as_deref()
                .expect("literal projection group must retain literal bytes");
            let max_outside_prefix = literals
                .iter()
                .filter(|(_, terminals)| !terminals.iter().any(|&other| member[other as usize]))
                .map(|(other, _)| {
                    bytes
                        .iter()
                        .zip(other.iter())
                        .take_while(|(left, right)| left == right)
                        .count()
                })
                .max()
                .unwrap_or(0);
            buckets
                .entry(max_outside_prefix > common_prefix_len)
                .or_default()
                .push(terminal);
        }
        refined.extend(buckets.into_values().filter(|group| group.len() >= 2));
    }
    refined
}

/// States immediately before and after each byte relevant to the current
/// restricted topology while recognizing one pure literal terminal.  The
/// unselected literal bytes are traversed to find the real DFA states, but do
/// not themselves impose a transport condition.
#[derive(Clone, Debug)]
struct LiteralSelectedTrace {
    transitions: Box<[(u8, u32, u32)]>,
}

fn literal_selected_trace(
    tokenizer: &Tokenizer,
    topology: &RestrictedTopology,
    terminal: TerminalID,
) -> Option<LiteralSelectedTrace> {
    let bytes = tokenizer.literal_terminal_bytes(terminal)?;
    let mut state = tokenizer.initial_state();
    let mut transitions = Vec::new();
    for byte in bytes {
        let next = tokenizer.step(state, byte)?;
        if topology.bytes.binary_search(&byte).is_ok() {
            transitions.push((
                byte,
                topology.raw_to_restricted_state[state as usize],
                topology.raw_to_restricted_state[next as usize],
            ));
        }
        state = next;
    }
    Some(LiteralSelectedTrace {
        transitions: transitions.into_boxed_slice(),
    })
}

/// Attempt one sparse raw-state transport family from literal selected-byte
/// milestones. The first literal containing an unselected byte avoids using a
/// degenerate punctuation-only sentinel such as `""` as the pivot. Every
/// accepted member is independently proved by `raw_literal_trace_interchange_map`.
fn raw_literal_trace_group_witnesses(
    tokenizer: &Tokenizer,
    group: &[TerminalID],
    dfa: &mut InterchangeabilityDfa,
) -> Vec<(TerminalID, TerminalID, InterchangeMap)> {
    const PROBE_MEMBERS: usize = 8;
    let traces = group
        .iter()
        .filter_map(|&terminal| {
            let bytes = tokenizer.literal_terminal_bytes(terminal)?;
            let trace = literal_selected_trace(tokenizer, &dfa.topology, terminal)?;
            Some((terminal, bytes.len(), trace))
        })
        .collect::<Vec<_>>();
    if traces.len() != group.len() {
        return Vec::new();
    }
    let pivot_index = traces
        .iter()
        .position(|(_, length, trace)| *length > trace.transitions.len())
        .unwrap_or(0);
    let (representative, _, representative_trace) = &traces[pivot_index];
    let mut maps = Vec::new();
    let mut attempted = vec![false; traces.len()];
    attempted[pivot_index] = true;
    let mut probe_count = 0usize;
    for (index, (member, _, member_trace)) in traces.iter().enumerate() {
        if index == pivot_index || probe_count == PROBE_MEMBERS {
            continue;
        }
        attempted[index] = true;
        probe_count += 1;
        if let Some(map) = dfa.raw_literal_trace_interchange_map(
            *representative,
            *member,
            representative_trace,
            member_trace,
        ) {
            maps.push((*representative, *member, map));
        }
    }
    if maps.is_empty() {
        return maps;
    }
    for (index, (member, _, member_trace)) in traces.iter().enumerate() {
        if attempted[index] {
            continue;
        }
        if let Some(map) = dfa.raw_literal_trace_interchange_map(
            *representative,
            *member,
            representative_trace,
            member_trace,
        ) {
            maps.push((*representative, *member, map));
        }
    }
    maps
}

fn literal_support_color_trace_orbit_witnesses(
    tokenizer: &Tokenizer,
    group: &[TerminalID],
    dfa: &mut InterchangeabilityDfa,
) -> Vec<(TerminalID, TerminalID, InterchangeMap)> {
    if std::env::var_os("GLRMASK_PROFILE_L2P_COLOR_TRACE_CERTIFICATE").is_none() {
        return Vec::new();
    }
    const MIN_ORBIT_MEMBERS: usize = 8;
    let mut buckets = BTreeMap::<Box<[(u8, u64, u64)]>, Vec<TerminalID>>::new();
    for &terminal in group {
        let Some(signature) = dfa.literal_support_color_trace_signature(tokenizer, terminal) else {
            continue;
        };
        buckets.entry(signature).or_default().push(terminal);
    }
    if std::env::var_os("GLRMASK_PROFILE_L2P_COLOR_TRACE_DIAGNOSTIC").is_some() {
        let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
        sizes.sort_unstable_by(|left, right| right.cmp(left));
        eprintln!(
            "[glrmask/profile][ti_literal_color_trace] representative={} members={} bucket_sizes={:?}",
            group[0],
            group.len(),
            sizes,
        );
    }
    let mut witnesses = Vec::new();
    for family in buckets.into_values() {
        if family.len() < MIN_ORBIT_MEMBERS {
            continue;
        }
        let started_at = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some().then(Instant::now);
        let maps = dfa.symmetric_support_orbit_witnesses(&family);
        if let Some(started_at) = started_at {
            eprintln!(
                "[glrmask/profile][ti_literal_color_orbit] representative={} members={} certified={} total_ms={:.3}",
                family[0],
                family.len(),
                maps.is_some(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let Some(maps) = maps else {
            continue;
        };
        for (member, map) in maps {
            witnesses.push((family[0], member, map));
        }
    }
    witnesses
}

/// Pair the non-shared support coordinates of two literal-family members.
/// This is only a candidate construction.  Failure falls through to the
/// existing generic exact solver; acceptance requires the quotient fiber proof.
fn literal_pivot_support_pairs_by_class(
    left: &[(u32, u8)],
    right: &[(u32, u8)],
) -> Option<Vec<(u32, u32)>> {
    let mut pairs = Vec::with_capacity(left.len() + right.len());
    // This coarse ordering is not the proof.  It is merely a stable seed for
    // the later complete quotient automorphism check, and has proved useful
    // for literal cones whose final/future tracks interleave differently.
    for mask in 1..=3u8 {
        let left_only = left
            .iter()
            .filter_map(|&(class, bits)| (bits == mask).then_some(class))
            .filter(|class| right.binary_search_by_key(class, |&(other, _)| other).is_err())
            .collect::<Vec<_>>();
        let right_only = right
            .iter()
            .filter_map(|&(class, bits)| (bits == mask).then_some(class))
            .filter(|class| left.binary_search_by_key(class, |&(other, _)| other).is_err())
            .collect::<Vec<_>>();
        if left_only.len() != right_only.len() {
            return None;
        }
        pairs.extend(left_only.into_iter().zip(right_only));
    }
    pairs.sort_unstable_by_key(|&(source, _)| source);
    pairs
        .windows(2)
        .all(|pair| pair[0].0 != pair[1].0)
        .then_some(pairs)
}

fn literal_support_seeded(
    supports: &[Vec<(u32, u8)>],
    member_count: usize,
) -> Option<BTreeMap<u32, Vec<u32>>> {
    let mut seeded = BTreeMap::<u32, Vec<u32>>::new();
    let mut expected_sources = Vec::<u32>::new();
    for member_index in 1..member_count {
        let pairs = literal_pivot_support_pairs_by_class(&supports[0], &supports[member_index])?;
        if pairs.is_empty() {
            return None;
        }
        let sources = pairs.iter().map(|&(source, _)| source).collect::<Vec<_>>();
        if member_index == 1 {
            expected_sources = sources;
            for &(source, target) in &pairs {
                let mut tuple = vec![u32::MAX; member_count];
                tuple[0] = source;
                tuple[member_index] = target;
                if seeded.insert(source, tuple).is_some() {
                    return None;
                }
            }
        } else {
            if sources != expected_sources {
                return None;
            }
            for (source, target) in pairs {
                let tuple = seeded.get_mut(&source)?;
                if tuple[member_index] != u32::MAX {
                    return None;
                }
                tuple[member_index] = target;
            }
        }
    }
    (!seeded.is_empty() && !seeded.values().any(|tuple| tuple.contains(&u32::MAX)))
        .then_some(seeded)
}


/// Install only exact symmetric-orbit witnesses for blocks produced by the
/// canonical-quotient WL rejection filter. The WL split merely nominates a
/// block; complete quotient generator proof remains the acceptance condition.
fn quotient_wl_pre_certificate_maps(
    dfa: &mut InterchangeabilityDfa,
    groups: &[Vec<TerminalID>],
) -> BTreeMap<(TerminalID, TerminalID), Arc<TransportScannerStateMap>> {
    let min_orbit_members = std::env::var("GLRMASK_PROFILE_L2P_GLOBAL_WL_CERT_MIN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(8);
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let started_at = profile_timing.then(Instant::now);
    let mut maps = BTreeMap::new();
    let mut certified_groups = 0usize;
    for group in groups {
        if group.len() < min_orbit_members {
            continue;
        }
        let group_started_at = profile_timing.then(Instant::now);
        let profile_local = std::env::var_os("GLRMASK_PROFILE_L2P_LOCAL_FIBER_ORBIT").is_some();
        let local_started_at = profile_local.then(Instant::now);
        let local_witnesses = if profile_local {
            dfa.uniform_fiber_orbit_witnesses_local(group)
        } else {
            None
        };
        let local_ms = local_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let full_started_at = profile_timing.then(Instant::now);
        let witnesses = dfa.symmetric_support_orbit_witnesses(group);
        let full_ms = full_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if let Some(started_at) = group_started_at {
            eprintln!(
                "[glrmask/profile][ti_global_wl_orbit] representative={} members={} local_certified={} certified={} local_ms={:.3} full_ms={:.3} total_ms={:.3}",
                group[0],
                group.len(),
                local_witnesses.is_some(),
                witnesses.is_some(),
                local_ms,
                full_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let Some(witnesses) = witnesses else {
            continue;
        };
        certified_groups += 1;
        for (member, map) in witnesses {
            let previous = maps.insert((group[0], member), Arc::new(map.scanner_state_map()));
            debug_assert!(previous.is_none());
        }
    }
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][ti_global_wl_certificate] candidate_groups={} certified_groups={} certified_members={} total_ms={:.3}",
            groups.iter().filter(|group| group.len() >= min_orbit_members).count(),
            certified_groups,
            maps.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    maps
}

/// Exact support-shape certificates for arbitrary root/output candidate
/// buckets. The support colors merely nominate families; every returned map
/// still comes from the full symmetric-orbit proof.
fn support_shape_pre_certificate_maps(
    dfa: &mut InterchangeabilityDfa,
    groups: &[Vec<TerminalID>],
) -> BTreeMap<(TerminalID, TerminalID), Arc<TransportScannerStateMap>> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let started_at = profile_timing.then(Instant::now);
    let mut maps = BTreeMap::new();
    let mut groups_with_maps = 0usize;
    for group in groups {
        let witnesses = dfa.literal_support_shape_orbit_witnesses(group);
        if witnesses.is_empty() {
            continue;
        }
        groups_with_maps += 1;
        for (representative, member, map) in witnesses {
            let previous = maps.insert((representative, member), Arc::new(map.scanner_state_map()));
            debug_assert!(previous.is_none());
        }
    }
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][ti_support_shape_certificate] candidate_groups={} groups_with_maps={} certified_members={} total_ms={:.3}",
            groups.len(),
            groups_with_maps,
            maps.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    maps
}

/// Build exact literal/fiber witnesses from groups already filtered by the
/// ordinary root and frozen-output candidate filters.  Returned maps are
/// proved under `active` and can therefore be installed directly into that
/// same historical TI round.  An inconclusive group returns no map and is
/// handled by the unchanged generic path.
fn literal_fiber_pre_certificate_maps(
    tokenizer: &Tokenizer,
    active: &[bool],
    context: &TiDiscoveryContext,
    dfa: &mut InterchangeabilityDfa,
    generic_candidate_groups: &[Vec<TerminalID>],
    ignore: Option<TerminalID>,
) -> BTreeMap<(TerminalID, TerminalID), Arc<TransportScannerStateMap>> {
    const MIN_FIBER_GROUP_MEMBERS: usize = 8;
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let started_at = profile_timing.then(Instant::now);
    let groups = literal_projection_groups(tokenizer, active, context, ignore)
        .into_iter()
        .filter(|group| group.len() >= MIN_FIBER_GROUP_MEMBERS)
        .collect::<Vec<_>>();

    let mut maps = BTreeMap::<(TerminalID, TerminalID), Arc<TransportScannerStateMap>>::new();
    // Certificate groups are independently nominated by two necessary filters.
    // Keep the resulting seeded partition a true partition even if a terminal
    // appears in more than one nomination. A terminal may be a representative
    // of many members, but never both a representative and a member.
    let mut member_owner = vec![None::<TerminalID>; active.len()];
    let mut representative_used = vec![false; active.len()];
    let mut orbit_certified_groups = 0usize;
    let mut direct_certified_groups = 0usize;
    for group in &groups {
        if std::env::var_os("GLRMASK_PROFILE_L2P_LITERAL_BUCKET_DIAGNOSTIC").is_some() {
            let mut lengths = BTreeMap::<usize, usize>::new();
            for &terminal in group {
                let length = tokenizer
                    .literal_terminal_bytes(terminal)
                    .map_or(0, |bytes| bytes.len());
                *lengths.entry(length).or_default() += 1;
            }
            eprintln!(
                "[glrmask/profile][ti_literal_bucket] representative={} members={} literal_lengths={:?}",
                group[0],
                group.len(),
                lengths,
            );
        }

        let color_maps = literal_support_color_trace_orbit_witnesses(tokenizer, group, dfa);
        if !color_maps.is_empty() {
            for (representative, member, map) in color_maps {
                if member_owner[representative as usize].is_some()
                    || representative_used[member as usize]
                {
                    continue;
                }
                if let Some(owner) = member_owner[member as usize] {
                    if owner != representative {
                        continue;
                    }
                }
                let previous = maps.insert((representative, member), Arc::new(map.scanner_state_map()));
                debug_assert!(previous.is_none());
                member_owner[member as usize] = Some(representative);
                representative_used[representative as usize] = true;
            }
            orbit_certified_groups += 1;
            continue;
        }
        let raw_started_at = profile_timing.then(Instant::now);
        let raw_maps = if std::env::var_os("GLRMASK_PROFILE_L2P_SKIP_RAW_LITERAL_CERTIFICATE").is_some() {
            Vec::new()
        } else {
            raw_literal_trace_group_witnesses(tokenizer, group, dfa)
        };
        if let Some(started_at) = raw_started_at {
            eprintln!(
                "[glrmask/profile][ti_literal_raw_attempt] representative={} members={} certified_members={} total_ms={:.3}",
                group[0],
                group.len(),
                raw_maps.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if raw_maps.len() >= MIN_FIBER_GROUP_MEMBERS - 1 {
            if profile_timing {
                eprintln!(
                    "[glrmask/profile][ti_literal_raw_group] representative={} members={} certified_members={}",
                    raw_maps[0].0,
                    group.len(),
                    raw_maps.len(),
                );
            }
            for (representative, member, map) in raw_maps {
                if member_owner[representative as usize].is_some()
                    || representative_used[member as usize]
                {
                    continue;
                }
                if let Some(owner) = member_owner[member as usize] {
                    if owner != representative {
                        continue;
                    }
                }
                let previous = maps.insert((representative, member), Arc::new(map.scanner_state_map()));
                debug_assert!(previous.is_none());
                member_owner[member as usize] = Some(representative);
                representative_used[representative as usize] = true;
            }
            direct_certified_groups += 1;
            continue;
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_RAW_ONLY_CERTIFICATE").is_some() {
            continue;
        }

        let mut orbit_maps = dfa.support_orbit_first_bucket_witnesses(group);
        if orbit_maps.is_empty() {
            orbit_maps = dfa.literal_support_shape_orbit_witnesses(group);
        }
        if !orbit_maps.is_empty() {
            if profile_timing {
                eprintln!(
                    "[glrmask/profile][ti_literal_orbit_group] representative={} members={} certified_members={}",
                    group[0],
                    group.len(),
                    orbit_maps.len(),
                );
            }
            for (representative, member, map) in orbit_maps {
                if member_owner[representative as usize].is_some()
                    || representative_used[member as usize]
                {
                    continue;
                }
                if let Some(owner) = member_owner[member as usize] {
                    if owner != representative {
                        continue;
                    }
                }
                let previous = maps.insert((representative, member), Arc::new(map.scanner_state_map()));
                debug_assert!(previous.is_none());
                member_owner[member as usize] = Some(representative);
                representative_used[representative as usize] = true;
            }
            orbit_certified_groups += 1;
            continue;
        }

        let Some(group_maps) = dfa.literal_fiber_group_witnesses(group) else {
            if profile_timing {
                eprintln!(
                    "[glrmask/profile][ti_literal_fiber_group] representative={} members={} certified=false",
                    group[0],
                    group.len(),
                );
            }
            continue;
        };
        if profile_timing {
            eprintln!(
                "[glrmask/profile][ti_literal_fiber_group] representative={} members={} certified=true",
                group[0],
                group.len(),
            );
        }
        for (member, map) in group_maps {
            if member_owner[group[0] as usize].is_some()
                || representative_used[member as usize]
            {
                continue;
            }
            if let Some(owner) = member_owner[member as usize] {
                if owner != group[0] {
                    continue;
                }
            }
            let previous = maps.insert((group[0], member), Arc::new(map.scanner_state_map()));
            debug_assert!(previous.is_none());
            member_owner[member as usize] = Some(group[0]);
            representative_used[group[0] as usize] = true;
        }
        direct_certified_groups += 1;
    }

    if profile_timing {
        let mut group_size_histogram = BTreeMap::<usize, usize>::new();
        for group in &groups {
            *group_size_histogram.entry(group.len()).or_default() += 1;
        }
        eprintln!(
            "[glrmask/profile][ti_literal_fiber_certificate] groups={} group_size_histogram={:?} orbit_certified_groups={} direct_certified_groups={} certified_members={} total_ms={:.3}",
            groups.len(),
            group_size_histogram,
            orbit_certified_groups,
            direct_certified_groups,
            maps.len(),
            started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        );
    }
    maps
}

/// Test-only/diagnostic wrapper around the same certificate maps that
/// production injects into the initial generic TI round.
pub(crate) fn discover_literal_fiber_pre_certificate_round(
    tokenizer: &Tokenizer,
    active: &[bool],
    context: &TiDiscoveryContext,
    ignore: Option<TerminalID>,
) -> Option<TiRoundTransportWitnesses> {
    let candidates = active
        .iter()
        .enumerate()
        .filter_map(|(terminal, &is_active)| is_active.then_some(terminal as TerminalID))
        .filter(|&terminal| Some(terminal) != ignore)
        .collect::<Vec<_>>();
    let root_groups = rooted_candidate_groups_from_signatures(
        &candidates,
        &context.root_output_signatures,
    );
    let mut dfa = InterchangeabilityDfa::from_context(active, context);
    let shape_groups = refine_candidate_groups_by_observed_output_pair_shape(
        root_groups,
        &dfa.observed_output_pair_support_shapes_by_terminal,
    );
    let (candidate_groups, _) = refine_candidate_groups_by_observed_output_hypergraph(
        shape_groups,
        &dfa.observed_output_pair_ids,
        &dfa.output_pairs,
        active.len(),
    );
    let maps = literal_fiber_pre_certificate_maps(
        tokenizer,
        active,
        context,
        &mut dfa,
        &candidate_groups,
        ignore,
    );
    if maps.is_empty() {
        return None;
    }
    let mut partition = singleton_partition(active);
    for &(representative, member) in maps.keys() {
        partition
            .get_mut(&representative)
            .expect("literal TI representative must remain active")
            .insert(member);
        let removed = partition.remove(&member);
        debug_assert!(removed.is_some());
    }
    assert_partition_invariants(&partition, active);
    Some(TiRoundTransportWitnesses {
        active_before_round: active.to_vec(),
        partition,
        maps,
        next_round_state_map: None,
    })
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
fn root_output_signatures(
    tokenizer: &Tokenizer,
    topology: &RestrictedTopology,
) -> (Vec<RootOutputSignature>, usize) {
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

    let terminal_count = tokenizer.num_terminals() as usize;
    let mut finalizer_states = vec![Vec::<u32>::new(); terminal_count];
    let mut future_finalizer_states = vec![Vec::<u32>::new(); terminal_count];
    for (state, &is_observed) in observed.iter().enumerate() {
        if !is_observed {
            continue;
        }
        let raw_state = topology.raw_state_for_restricted(state);
        for terminal in tokenizer.matched_terminals_iter(raw_state) {
            finalizer_states[terminal as usize].push(state as u32);
        }
        for terminal in tokenizer.possible_future_terminals_iter(raw_state) {
            future_finalizer_states[terminal as usize].push(state as u32);
        }
    }

    let signatures = finalizer_states
        .into_iter()
        .zip(future_finalizer_states)
        .map(|(finalizer_states, future_finalizer_states)| RootOutputSignature {
            finalizer_states: finalizer_states.into_boxed_slice(),
            future_finalizer_states: future_finalizer_states.into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    (
        signatures,
        observed.into_iter().filter(|&value| value).count(),
    )
}

fn rooted_candidate_groups_from_signatures(
    candidates: &[TerminalID],
    signatures: &[RootOutputSignature],
) -> Vec<Vec<TerminalID>> {
    let mut groups = BTreeMap::<&RootOutputSignature, Vec<TerminalID>>::new();
    for &terminal in candidates {
        groups
            .entry(
                signatures
                    .get(terminal as usize)
                    .expect("TI candidate terminal must have a root signature"),
            )
            .or_default()
            .push(terminal);
    }
    groups.into_values().collect()
}

fn rooted_candidate_groups(
    tokenizer: &Tokenizer,
    candidates: &[TerminalID],
    topology: &RestrictedTopology,
) -> (Vec<Vec<TerminalID>>, usize) {
    let (signatures, observed_states) = root_output_signatures(tokenizer, topology);
    (
        rooted_candidate_groups_from_signatures(candidates, &signatures),
        observed_states,
    )
}

/// A terminal's support across a terminal-name-independent structural
/// partition. It is only a rejection invariant; the full checker decides.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct StructuralOutputSignature {
    finalizer_support: Box<[u64]>,
    future_finalizer_support: Box<[u64]>,
}

/// Counts of quotient source classes whose successor observations mention a
/// terminal in finalizers only, future-finalizers only, or both. An accepted
/// terminal transposition maps these source classes bijectively, preserving
/// each category; unequal shapes are therefore an exact rejection invariant.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct SupportTrackShape {
    finalizer_only: u32,
    future_only: u32,
    both: u32,
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
    refinement_rounds: usize,
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
    for _ in 0..refinement_rounds {
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

/// Raw-row variant of `structural_candidate_signatures`.  TI discovery has
/// already materialized the restricted scanner's terminal rows, so rescanning
/// the tokenizer here is pure duplicate work.  The returned invariant is
/// byte-for-byte equivalent to the tokenizer-scanning version above.
fn structural_candidate_signatures_from_raw(
    active_terminals: &[bool],
    candidates: &[TerminalID],
    topology: &RestrictedTopology,
    raw: &TiRawDiscoveryData,
    refinement_rounds: usize,
) -> (Vec<StructuralOutputSignature>, usize) {
    let state_count = topology.state_count();
    let dead_state = topology.dead_state();
    let mut finalizer_counts = vec![0u64; state_count];
    let mut future_finalizer_counts = vec![0u64; state_count];
    for state in 0..topology.real_state_count {
        finalizer_counts[state] = raw.finalizer_terminals_by_state[state]
            .iter()
            .filter(|&&terminal| active_terminals[terminal as usize])
            .count() as u64;
        future_finalizer_counts[state] = raw.future_finalizer_terminals_by_state[state]
            .iter()
            .filter(|&&terminal| active_terminals[terminal as usize])
            .count() as u64;
    }

    let mut fingerprints = vec![0x6a09_e667_f3bc_c909; state_count];
    for _ in 0..refinement_rounds {
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
    for (state, &is_observed) in raw.observed_destinations.iter().enumerate() {
        if !is_observed || state == dead_state {
            continue;
        }
        let color = colors[state] as usize;
        let word = color / 64;
        let mask = 1u64 << (color % 64);
        for &terminal in &raw.finalizer_terminals_by_state[state] {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            if candidate_index != usize::MAX {
                finalizer_support[candidate_index][word] |= mask;
            }
        }
        for &terminal in &raw.future_finalizer_terminals_by_state[state] {
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

/// Refine candidates by an exact necessary invariant of the observed frozen
/// output-pair relation.  A valid transposition permutes that relation, hence
/// it must preserve how many distinct observed pairs mention each terminal in
/// finalizer-only, future-finalizer-only, and both positions.  This is only a
/// rejection filter; all surviving pairs still pass the full exact checker.
fn refine_candidate_groups_by_observed_output_pair_shape(
    groups: Vec<Vec<TerminalID>>,
    shapes: &[SupportTrackShape],
) -> Vec<Vec<TerminalID>> {
    let mut refined = Vec::new();
    for group in groups {
        let mut by_shape = BTreeMap::<SupportTrackShape, Vec<TerminalID>>::new();
        for terminal in group {
            let shape = shapes
                .get(terminal as usize)
                .copied()
                .expect("TI candidate terminal must have an observed-output support shape");
            by_shape.entry(shape).or_default().push(terminal);
        }
        refined.extend(by_shape.into_values().filter(|group| group.len() >= 2));
    }
    refined
}

/// Sound terminal-side color refinement of the set of observed frozen output
/// pairs.  Treat each pair `(finalizers, future_finalizers)` as a directed
/// hyperedge, with a three-way incidence tag for final-only, future-only, and
/// both.  A valid terminal transposition maps this set of hyperedges to itself,
/// so it preserves every refinement color.  Different resulting colors can
/// therefore reject a pair without affecting exact TI semantics.
///
/// Terminals already excluded from a candidate group are fixed under every
/// remaining binary swap.  Giving them individual colors strengthens the
/// invariant: their raw labels are not accidentally treated as exchangeable
/// context.  The compact commutative fingerprints may collide only by merging
/// colors, which is conservative.
fn refine_candidate_groups_by_observed_output_hypergraph(
    groups: Vec<Vec<TerminalID>>,
    observed_pair_ids: &[u32],
    output_pairs: &[OutputPair],
    terminal_count: usize,
) -> (Vec<Vec<TerminalID>>, usize) {
    if groups.is_empty() {
        return (groups, 0);
    }

    let mut candidate_group_by_terminal = vec![usize::MAX; terminal_count];
    for (group_index, group) in groups.iter().enumerate() {
        for &terminal in group {
            candidate_group_by_terminal[terminal as usize] = group_index;
        }
    }

    // Fixed terminals receive raw-id colors. Candidate terminals receive their
    // already sound prefilter group color, ensuring an accepted swap starts in
    // one common color class.
    let mut colors = (0..terminal_count as u32).collect::<Vec<_>>();
    for (group_index, group) in groups.iter().enumerate() {
        let color = terminal_count as u32 + group_index as u32;
        for &terminal in group {
            colors[terminal as usize] = color;
        }
    }

    let mut rounds = 0usize;
    loop {
        rounds += 1;
        let mut sum = vec![0u64; terminal_count];
        let mut xor = vec![0u64; terminal_count];
        let mut count = vec![0u32; terminal_count];

        for &pair_id in observed_pair_ids {
            let pair = &output_pairs[pair_id as usize];
            let color_multiset_fingerprint = |terminals: &OutputBits| {
                let mut total = 0u64;
                let mut parity = 0u64;
                for &terminal in &terminals.0 {
                    let component = mix_structural_fingerprint(
                        0x6d2b_79f5_aa99_5a71,
                        colors[terminal as usize] as u64,
                    );
                    total = total.wrapping_add(component);
                    parity ^= component.rotate_left((colors[terminal as usize] & 63) as u32);
                }
                let mut fingerprint = mix_structural_fingerprint(
                    0xa076_1d64_78bd_642f,
                    terminals.0.len() as u64,
                );
                fingerprint = mix_structural_fingerprint(fingerprint, total);
                mix_structural_fingerprint(fingerprint, parity)
            };
            let finalizer_fingerprint = color_multiset_fingerprint(&pair.finalizers);
            let future_fingerprint = color_multiset_fingerprint(&pair.future_finalizers);
            let mut edge_fingerprint = mix_structural_fingerprint(
                0xe703_7ed1_a0b4_28db,
                finalizer_fingerprint,
            );
            edge_fingerprint = mix_structural_fingerprint(edge_fingerprint, future_fingerprint);

            let mut finalizer_index = 0usize;
            let mut future_index = 0usize;
            while finalizer_index < pair.finalizers.0.len()
                || future_index < pair.future_finalizers.0.len()
            {
                let (terminal, category) = match (
                    pair.finalizers.0.get(finalizer_index),
                    pair.future_finalizers.0.get(future_index),
                ) {
                    (Some(&finalizer), Some(&future)) if finalizer == future => {
                        finalizer_index += 1;
                        future_index += 1;
                        (finalizer as usize, 2u64)
                    }
                    (Some(&finalizer), Some(&future)) if finalizer < future => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u64)
                    }
                    (Some(_), Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u64)
                    }
                    (Some(&finalizer), None) => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u64)
                    }
                    (None, Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u64)
                    }
                    (None, None) => unreachable!("nonempty observed-output hyperedge merge"),
                };
                let component = mix_structural_fingerprint(edge_fingerprint, category);
                sum[terminal] = sum[terminal].wrapping_add(component);
                xor[terminal] ^= component.rotate_left((category * 17) as u32);
                count[terminal] += 1;
            }
        }

        let mut class_for_signature = FxHashMap::<u64, u32>::default();
        let mut next_colors = colors.clone();
        for (terminal, &group_index) in candidate_group_by_terminal.iter().enumerate() {
            if group_index == usize::MAX {
                continue;
            }
            let mut fingerprint = mix_structural_fingerprint(
                0x8ebc_6af0_9c88_c6e3,
                colors[terminal] as u64,
            );
            fingerprint = mix_structural_fingerprint(fingerprint, sum[terminal]);
            fingerprint = mix_structural_fingerprint(fingerprint, xor[terminal]);
            fingerprint = mix_structural_fingerprint(fingerprint, count[terminal] as u64);
            let next = terminal_count as u32 + class_for_signature.len() as u32;
            next_colors[terminal] = *class_for_signature.entry(fingerprint).or_insert(next);
        }
        let stable = same_equality_partition_u32(&colors, &next_colors);
        colors = next_colors;
        if stable || rounds == terminal_count {
            break;
        }
    }

    let mut refined = Vec::new();
    for group in groups {
        let mut by_color = BTreeMap::<u32, Vec<TerminalID>>::new();
        for terminal in group {
            by_color
                .entry(colors[terminal as usize])
                .or_default()
                .push(terminal);
        }
        refined.extend(by_color.into_values().filter(|group| group.len() >= 2));
    }
    (refined, rounds)
}

/// The selected raw tokenizer-state representative for every source state
/// under one terminal swap. The exact characterization establishes a mapped
/// target class; all downstream consumers have always selected only that
/// class's first raw state, so retaining the whole class is redundant.
#[derive(Clone, Debug)]
struct InterchangeMap {
    scanner_state_map: TransportScannerStateMap,
}

/// One transient exact TI round.  This is deliberately build-local: the only
/// durable TI result remains the final flat terminal partition.  Keeping the
/// already-certified witnesses avoids re-running terminal discovery after the
/// representative core has been built.
#[derive(Clone, Debug)]
pub(crate) struct TiRoundTransportWitnesses {
    active_before_round: Vec<bool>,
    pub(crate) partition: BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    /// Certified maps for precisely the accepted representative/member pairs
    /// of this historical round.  The key is `(new_representative,
    /// old_representative)`.
    maps: BTreeMap<(TerminalID, TerminalID), Arc<TransportScannerStateMap>>,
    pub(crate) next_round_state_map: Option<ManyToOneIdMap>,
}

impl TiRoundTransportWitnesses {
    fn singleton(active_terminals: &[bool]) -> Self {
        Self {
            active_before_round: active_terminals.to_vec(),
            partition: singleton_partition(active_terminals),
            maps: BTreeMap::new(),
            next_round_state_map: None,
        }
    }
}

impl PartialEq for InterchangeMap {
    fn eq(&self, other: &Self) -> bool {
        self.scanner_state_map.len() == other.scanner_state_map.len()
            && (0..self.scanner_state_map.len()).all(|state| {
                self.scanner_state_map.scanner_state(state as u32)
                    == other.scanner_state_map.scanner_state(state as u32)
            })
    }
}

impl Eq for InterchangeMap {}

impl InterchangeMap {
    fn scanner_state_map(&self) -> TransportScannerStateMap {
        self.scanner_state_map.clone()
    }

    #[cfg(test)]
    fn materialized_scanner_states(&self) -> Arc<[u32]> {
        self.scanner_state_map.materialized()
    }
}

struct PairCharacterization {
    identity_hashes: Vec<CharacterizationHash>,
    swapped_hashes: Vec<CharacterizationHash>,
}

struct InterchangeabilityDfa {
    topology: Arc<RestrictedTopology>,
    empty_output: OutputBits,
    finalizers: Vec<OutputBits>,
    future_finalizers: Vec<OutputBits>,
    /// Canonical output-pair IDs visible on enabled-byte destinations.
    observed_output_pair_ids: Vec<u32>,
    observed_output_pair_present: Vec<bool>,
    observed_output_pair_ids_by_terminal: Vec<Vec<u32>>,
    /// For each active terminal, its exact membership counts across the
    /// deduplicated observed frozen-output pairs. A valid terminal swap maps
    /// those pairs bijectively, preserving the final-only, future-only, and
    /// both categories. This is therefore an inexpensive rejection invariant
    /// for candidate pairs before the per-pair closure check.
    observed_output_pair_support_shapes_by_terminal: Vec<SupportTrackShape>,
    observed_output_pair_marks: Vec<u32>,
    observed_output_pair_mark_epoch: u32,
    /// Exact terminal columns over the frozen state outputs. When two columns
    /// coincide in both output families, swapping their terminal labels leaves
    /// every destination output unchanged and therefore needs no refinement.
    finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    future_finalizer_states_by_terminal: Arc<[Vec<u32>]>,
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
    reverse_predecessors: Arc<[Vec<u32>]>,
    reverse_edges: Arc<[Vec<(u8, u32)>]>,
    canonical_rounds: Vec<CanonicalRound>,
    canonical_hopcroft_identity: Option<CanonicalRound>,
    canonical_incremental_identity: Option<CanonicalRound>,
    canonical_dag_identity: Option<CanonicalRound>,
    canonical_propagated_identity: Option<CanonicalRound>,
    canonical_kahn_identity: Option<CanonicalRound>,
    canonical_round_one_class_counts: Option<Vec<u32>>,
    canonical_round_two_class_counts: Option<Vec<u32>>,
    canonical_round_one_source_marks: Vec<u32>,
    canonical_round_one_source_mark_epoch: u32,
    canonical_round_one_affected_sources: Vec<u32>,
    support_quotient: Option<SupportQuotient>,
    canonical_quotient: Option<CanonicalQuotient>,
    /// Per raw terminal, the canonical quotient classes whose representative
    /// frozen output mentions that terminal.  This is build-local discovery
    /// scratch, used to propose a tiny support-transposition witness before
    /// falling back to exact refinement.
    terminal_quotient_output_supports: Option<Vec<Option<Vec<(u32, u8)>>>>,
    terminal_erased_support_colors: Option<Arc<[u64]>>,
    /// Shared identity coordinate system for the raw-state local literal
    /// certificate. Accepted maps change only the sparse listed deviations.
    raw_identity_classes: Option<Arc<[u32]>>,
    quotient_certified: usize,
    support_transposition_certified: usize,
    support_transposition_no_template: usize,
    support_transposition_outside_cone: usize,
    support_transposition_root_rejected: usize,
    support_transposition_signature_rejected: usize,
    support_transposition_support_setup_ns: u64,
    support_transposition_template_ns: u64,
    support_transposition_cone_ns: u64,
    support_transposition_verify_ns: u64,
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
            Arc::new(RestrictedTopology::new(tokenizer, relevant_bytes, None)),
        )
    }

    fn from_topology(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        topology: Arc<RestrictedTopology>,
    ) -> Self {
        let raw = Arc::new(TiRawDiscoveryData::new(tokenizer, &topology, None));
        Self::from_raw_discovery_data(observed_terminals, topology, raw)
    }

    fn from_context(observed_terminals: &[bool], context: &TiDiscoveryContext) -> Self {
        Self::from_raw_discovery_data(
            observed_terminals,
            Arc::clone(&context.topology),
            Arc::clone(&context.raw),
        )
    }

    fn quotient_transport_state_map(
        &self,
        class_for_restricted: Arc<[u32]>,
        representative_restricted: Arc<[u32]>,
        source_class_for_target_deviations: Box<[(u32, u32)]>,
    ) -> TransportScannerStateMap {
        if self.topology.raw_state_count == self.topology.real_state_count {
            return TransportScannerStateMap::Quotient {
                state_count: self.topology.raw_state_count,
                class_for_original: class_for_restricted,
                representative_for_class: representative_restricted,
                source_class_for_target_deviations,
            };
        }
        TransportScannerStateMap::Quotient {
            state_count: self.topology.raw_state_count,
            class_for_original: self
                .topology
                .raw_class_for_restricted_classes(&class_for_restricted),
            representative_for_class: self
                .topology
                .raw_representatives_for_classes(&representative_restricted),
            source_class_for_target_deviations,
        }
    }

    fn from_raw_discovery_data(
        observed_terminals: &[bool],
        topology: Arc<RestrictedTopology>,
        raw: Arc<TiRawDiscoveryData>,
    ) -> Self {
        let profile_setup = true;
        let setup_started_at = profile_setup.then(Instant::now);
        let state_count = topology.state_count();
        let terminal_bits = |terminals: &[TerminalID]| {
            OutputBits::from_active(terminals, observed_terminals)
        };
        let finalizers = raw
            .finalizer_terminals_by_state
            .iter()
            .map(|terminals| terminal_bits(terminals))
            .collect::<Vec<_>>();
        let output_filter_ms = setup_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        // These are the tokenizer's original, frozen future-finalizer sets.
        let future_finalizers = raw
            .future_finalizer_terminals_by_state
            .iter()
            .map(|terminals| terminal_bits(terminals))
            .collect::<Vec<_>>();
        let empty_output = OutputBits::new(0);
        let empty_pair = OutputPair {
            finalizers: empty_output.clone(),
            future_finalizers: empty_output.clone(),
        };
        let mut output_pairs = vec![empty_pair.clone()];
        let mut output_pair_lookup = FxHashMap::<OutputPair, u32>::default();
        output_pair_lookup.insert(empty_pair, 0);
        let mut output_pair_id_by_raw_pair =
            vec![u32::MAX; raw.raw_output_pair_representatives.len()];
        let mut output_pair_by_state = Vec::with_capacity(state_count);
        for state in 0..topology.real_state_count {
            let raw_pair = raw.raw_output_pair_by_state[state] as usize;
            let id = if output_pair_id_by_raw_pair[raw_pair] != u32::MAX {
                output_pair_id_by_raw_pair[raw_pair]
            } else {
                let representative = raw.raw_output_pair_representatives[raw_pair] as usize;
                let pair = OutputPair {
                    finalizers: finalizers[representative].clone(),
                    future_finalizers: future_finalizers[representative].clone(),
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
                output_pair_id_by_raw_pair[raw_pair] = id;
                id
            };
            output_pair_by_state.push(id);
        }
        let output_pair_ms = setup_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - output_filter_ms)
            .unwrap_or(0.0);
        let mut observed_output_pair_ids = Vec::<u32>::new();
        let mut observed_output_pair_present = vec![false; output_pairs.len()];
        let mut observed_output_pair_ids_by_terminal =
            vec![Vec::<u32>::new(); observed_terminals.len()];
        let mut observed_output_pair_support_shapes_by_terminal =
            vec![SupportTrackShape::default(); observed_terminals.len()];
        for state in 0..topology.real_state_count {
            if !raw.observed_destinations[state] {
                continue;
            }
            let id = output_pair_by_state[state] as usize;
            if observed_output_pair_present[id] {
                continue;
            }
            observed_output_pair_present[id] = true;
            observed_output_pair_ids.push(id as u32);
            let pair = &output_pairs[id];
            let mut finalizer_index = 0usize;
            let mut future_index = 0usize;
            while finalizer_index < pair.finalizers.0.len()
                || future_index < pair.future_finalizers.0.len()
            {
                let (terminal, category) = match (
                    pair.finalizers.0.get(finalizer_index),
                    pair.future_finalizers.0.get(future_index),
                ) {
                    (Some(&finalizer), Some(&future)) if finalizer == future => {
                        finalizer_index += 1;
                        future_index += 1;
                        (finalizer as usize, 2u8)
                    }
                    (Some(&finalizer), Some(&future)) if finalizer < future => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u8)
                    }
                    (Some(_), Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u8)
                    }
                    (Some(&finalizer), None) => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u8)
                    }
                    (None, Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u8)
                    }
                    (None, None) => unreachable!("nonempty sparse output merge"),
                };
                observed_output_pair_ids_by_terminal[terminal].push(id as u32);
                let shape = &mut observed_output_pair_support_shapes_by_terminal[terminal];
                match category {
                    0 => shape.finalizer_only += 1,
                    1 => shape.future_only += 1,
                    2 => shape.both += 1,
                    _ => unreachable!("known observed-output support category"),
                }
            }
        }
        let observed_pair_ms = setup_started_at
            .map(|started_at| {
                started_at.elapsed().as_secs_f64() * 1000.0 - output_filter_ms - output_pair_ms
            })
            .unwrap_or(0.0);
        // The synthetic dead destination has the all-empty frozen output.
        output_pair_by_state.push(0);
        let signature_capacity = CHARACTERIZATION_DOMAIN.len()
            + 4
            + topology.max_outdegree
                * (1 + blake3::OUT_LEN + 2 * (size_of::<u32>() + 4 * size_of::<TerminalID>()));
        let seed = CharacterizationHash::seed();
        let observed_output_pair_count = output_pairs.len();
        let result = Self {
            topology,
            empty_output,
            finalizers,
            future_finalizers,
            observed_output_pair_ids,
            observed_output_pair_present,
            observed_output_pair_ids_by_terminal,
            observed_output_pair_support_shapes_by_terminal,
            observed_output_pair_marks: vec![0; observed_output_pair_count],
            observed_output_pair_mark_epoch: 0,
            finalizer_states_by_terminal: Arc::clone(&raw.finalizer_states_by_terminal),
            future_finalizer_states_by_terminal: Arc::clone(
                &raw.future_finalizer_states_by_terminal,
            ),
            identity_rounds: vec![vec![seed; state_count]],
            output_pairs,
            output_pair_lookup,
            output_pair_by_state,
            reverse_predecessors: Arc::clone(&raw.reverse_predecessors),
            reverse_edges: Arc::clone(&raw.reverse_edges),
            canonical_rounds: vec![CanonicalRound {
                classes: vec![0; state_count],
                representative_by_class: vec![0],
                classes_by_signature_hash: FxHashMap::default(),
            }],
            canonical_hopcroft_identity: None,
            canonical_incremental_identity: None,
            canonical_dag_identity: None,
            canonical_propagated_identity: None,
            canonical_kahn_identity: None,
            canonical_round_one_class_counts: None,
            canonical_round_two_class_counts: None,
            canonical_round_one_source_marks: vec![0; state_count - 1],
            canonical_round_one_source_mark_epoch: 0,
            canonical_round_one_affected_sources: Vec::new(),
            support_quotient: None,
            canonical_quotient: None,
            terminal_quotient_output_supports: None,
            terminal_erased_support_colors: None,
            raw_identity_classes: None,
            quotient_certified: 0,
            support_transposition_certified: 0,
            support_transposition_no_template: 0,
            support_transposition_outside_cone: 0,
            support_transposition_root_rejected: 0,
            support_transposition_signature_rejected: 0,
            support_transposition_support_setup_ns: 0,
            support_transposition_template_ns: 0,
            support_transposition_cone_ns: 0,
            support_transposition_verify_ns: 0,
            sparse_quotient_certified: 0,
            sparse_quotient_cone_classes_total: 0,
            sparse_quotient_cone_classes_max: 0,
            sparse_quotient_cone_ns: 0,
            sparse_quotient_refinement_ns: 0,
            sparse_quotient_map_ns: 0,
            canonical_stable_round: None,
            canonical_identity_map: None,
            signature_capacity,
        };
        if let Some(started_at) = setup_started_at {
            eprintln!(
                "[glrmask/profile][ti_dfa_setup] states={} observed_pairs={} output_pairs={} output_filter_ms={:.3} observed_pair_ms={:.3} output_pair_ms={:.3} remainder_ms={:.3} total_ms={:.3}",
                result.topology.real_state_count,
                result.observed_output_pair_ids.len(),
                result.output_pairs.len(),
                output_filter_ms,
                observed_pair_ms,
                output_pair_ms,
                started_at.elapsed().as_secs_f64() * 1000.0 - output_filter_ms - observed_pair_ms - output_pair_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    fn state_count(&self) -> usize {
        self.topology.state_count()
    }

    fn profile_topology_sccs(&self) {
        let state_count = self.state_count();
        let mut nonforward_edges = 0usize;
        for source in 0..self.topology.real_state_count {
            for &(_, destination) in self.topology.edges_from(source) {
                nonforward_edges += (destination as usize <= source) as usize;
            }
        }
        let mut visited = vec![false; state_count];
        let mut order = Vec::with_capacity(state_count);
        for start in 0..state_count {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut stack = vec![(start, 0usize)];
            while let Some((state, edge_index)) = stack.pop() {
                let edges = self.topology.edges_from(state);
                if edge_index < edges.len() {
                    stack.push((state, edge_index + 1));
                    let destination = edges[edge_index].1 as usize;
                    if !visited[destination] {
                        visited[destination] = true;
                        stack.push((destination, 0));
                    }
                } else {
                    order.push(state);
                }
            }
        }
        let mut component = vec![u32::MAX; state_count];
        let mut component_sizes = Vec::new();
        for &start in order.iter().rev() {
            if component[start] != u32::MAX {
                continue;
            }
            let id = component_sizes.len() as u32;
            let mut size = 0usize;
            let mut stack = vec![start];
            component[start] = id;
            while let Some(state) = stack.pop() {
                size += 1;
                if state < self.topology.real_state_count {
                    for &predecessor in &self.reverse_predecessors[state] {
                        let predecessor = predecessor as usize;
                        if component[predecessor] == u32::MAX {
                            component[predecessor] = id;
                            stack.push(predecessor);
                        }
                    }
                }
            }
            component_sizes.push(size);
        }
        let cyclic_components = component_sizes.iter().filter(|&&size| size > 1).count();
        let cyclic_states = component_sizes.iter().filter(|&&size| size > 1).sum::<usize>();
        let largest_component = component_sizes.iter().copied().max().unwrap_or(0);
        eprintln!(
            "[glrmask/profile][ti_topology_scc] states={} components={} cyclic_components={} cyclic_states={} largest_component={} nonforward_edges={}",
            state_count,
            component_sizes.len(),
            cyclic_components,
            cyclic_states,
            largest_component,
            nonforward_edges,
        );
    }

    fn profile_topology_kahn(&self) {
        let real_state_count = self.topology.real_state_count;
        let mut pending = vec![0u32; real_state_count];
        for state in 0..real_state_count {
            pending[state] = self
                .topology
                .edges_from(state)
                .iter()
                .filter(|&&(_, destination)| destination as usize != self.dead_state())
                .count() as u32;
        }
        let mut queue = VecDeque::<u32>::new();
        for (state, &count) in pending.iter().enumerate() {
            if count == 0 {
                queue.push_back(state as u32);
            }
        }
        let mut processed = 0usize;
        while let Some(state) = queue.pop_front() {
            processed += 1;
            for &source in &self.reverse_predecessors[state as usize] {
                let source = source as usize;
                pending[source] -= 1;
                if pending[source] == 0 {
                    queue.push_back(source as u32);
                }
            }
        }
        eprintln!(
            "[glrmask/profile][ti_topology_kahn] real_states={} initially_resolved={} residual={}",
            real_state_count,
            processed,
            real_state_count - processed,
        );
    }

    fn topology_kahn_order(&self) -> (Vec<usize>, Vec<usize>) {
        let real_state_count = self.topology.real_state_count;
        let mut pending = vec![0u32; real_state_count];
        for state in 0..real_state_count {
            pending[state] = self
                .topology
                .edges_from(state)
                .iter()
                .filter(|&&(_, destination)| destination as usize != self.dead_state())
                .count() as u32;
        }
        let mut queue = VecDeque::<u32>::new();
        for (state, &count) in pending.iter().enumerate() {
            if count == 0 {
                queue.push_back(state as u32);
            }
        }
        let mut order = Vec::<usize>::with_capacity(real_state_count);
        while let Some(state) = queue.pop_front() {
            let state = state as usize;
            order.push(state);
            for &source in &self.reverse_predecessors[state] {
                let source = source as usize;
                pending[source] -= 1;
                if pending[source] == 0 {
                    queue.push_back(source as u32);
                }
            }
        }
        let residual = pending
            .iter()
            .enumerate()
            .filter_map(|(state, &count)| (count > 0).then_some(state))
            .collect::<Vec<_>>();
        (order, residual)
    }

    fn residual_cyclic_core(&self, residual: &[usize]) -> Vec<u32> {
        let real_state_count = self.topology.real_state_count;
        let mut in_residual = vec![false; real_state_count];
        for &state in residual {
            in_residual[state] = true;
        }
        let mut incoming = vec![0u32; real_state_count];
        let mut outgoing = vec![0u32; real_state_count];
        for &state in residual {
            for &(_, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                if destination < real_state_count && in_residual[destination] {
                    outgoing[state] += 1;
                    incoming[destination] += 1;
                }
            }
        }
        let mut removed = vec![false; real_state_count];
        let mut queue = VecDeque::<u32>::new();
        for &state in residual {
            if incoming[state] == 0 || outgoing[state] == 0 {
                removed[state] = true;
                queue.push_back(state as u32);
            }
        }
        while let Some(state) = queue.pop_front() {
            let state = state as usize;
            for &(_, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                if destination < real_state_count
                    && in_residual[destination]
                    && !removed[destination]
                {
                    incoming[destination] -= 1;
                    if incoming[destination] == 0 || outgoing[destination] == 0 {
                        removed[destination] = true;
                        queue.push_back(destination as u32);
                    }
                }
            }
            for &predecessor in &self.reverse_predecessors[state] {
                let predecessor = predecessor as usize;
                if in_residual[predecessor] && !removed[predecessor] {
                    outgoing[predecessor] -= 1;
                    if incoming[predecessor] == 0 || outgoing[predecessor] == 0 {
                        removed[predecessor] = true;
                        queue.push_back(predecessor as u32);
                    }
                }
            }
        }
        residual
            .iter()
            .copied()
            .filter(|&state| !removed[state])
            .map(|state| state as u32)
            .collect()
    }

    fn topology_scc_components(&self) -> (Vec<u32>, Vec<Vec<u32>>) {
        let state_count = self.state_count();
        let mut visited = vec![false; state_count];
        let mut order = Vec::with_capacity(state_count);
        for start in 0..state_count {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut stack = vec![(start, 0usize)];
            while let Some((state, edge_index)) = stack.pop() {
                let edges = self.topology.edges_from(state);
                if edge_index < edges.len() {
                    stack.push((state, edge_index + 1));
                    let destination = edges[edge_index].1 as usize;
                    if !visited[destination] {
                        visited[destination] = true;
                        stack.push((destination, 0));
                    }
                } else {
                    order.push(state);
                }
            }
        }
        let mut component_for_state = vec![u32::MAX; state_count];
        let mut components = Vec::<Vec<u32>>::new();
        for &start in order.iter().rev() {
            if component_for_state[start] != u32::MAX {
                continue;
            }
            let component = components.len() as u32;
            let mut states = Vec::new();
            let mut stack = vec![start];
            component_for_state[start] = component;
            while let Some(state) = stack.pop() {
                states.push(state as u32);
                if state < self.topology.real_state_count {
                    for &predecessor in &self.reverse_predecessors[state] {
                        let predecessor = predecessor as usize;
                        if component_for_state[predecessor] == u32::MAX {
                            component_for_state[predecessor] = component;
                            stack.push(predecessor);
                        }
                    }
                }
            }
            components.push(states);
        }
        (component_for_state, components)
    }

    fn raw_stable_identity_state_map(&mut self) -> ManyToOneIdMap {
        let stable_round = self.ensure_canonical_identity_stable_round();
        let classes = &self.canonical_rounds[stable_round].classes;
        let raw_state_count = self.topology.raw_state_count;
        let mut compact_class = FxHashMap::<(u32, u32), u32>::default();
        let mut original_to_internal = Vec::with_capacity(raw_state_count);
        let mut representative_original_ids = Vec::new();
        for raw_state in 0..raw_state_count {
            let restricted_state = self.topology.raw_to_restricted_state[raw_state] as usize;
            let canonical_class = classes[restricted_state];
            let output_pair = self.output_pair_by_state[restricted_state];
            let next = compact_class.len() as u32;
            let internal = *compact_class
                .entry((canonical_class, output_pair))
                .or_insert_with(|| {
                representative_original_ids.push(raw_state as u32);
                next
            });
            original_to_internal.push(internal);
        }
        ManyToOneIdMap::from_original_to_internal_with_representatives(
            original_to_internal,
            representative_original_ids.len() as u32,
            representative_original_ids,
        )
    }

    fn dead_state(&self) -> usize {
        self.topology.dead_state()
    }

    fn ensure_support_quotient(&mut self) {
        if self.support_quotient.is_some() {
            return;
        }
        if let Some(quotient) = self.canonical_quotient.as_ref() {
            self.support_quotient = Some(SupportQuotient {
                class_for_state: Arc::clone(&quotient.class_for_state),
                representative_by_class: Arc::clone(&quotient.representative_by_class),
                reverse_predecessors: Arc::clone(&quotient.reverse_predecessors),
            });
            return;
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TOPOLOGY_SCC").is_some() {
            self.profile_topology_sccs();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TOPOLOGY_KAHN").is_some() {
            self.profile_topology_kahn();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_KAHN_IDENTITY_DIAGNOSTIC").is_some() {
            self.ensure_canonical_kahn_identity();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_HOPCROFT_IDENTITY_DIAGNOSTIC").is_some() {
            self.ensure_canonical_hopcroft_identity();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_PROPAGATED_IDENTITY_DIAGNOSTIC").is_some() {
            self.ensure_canonical_propagated_identity();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_INCREMENTAL_IDENTITY_DIAGNOSTIC").is_some() {
            self.ensure_canonical_incremental_identity();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_DAG_IDENTITY_DIAGNOSTIC").is_some() {
            self.ensure_canonical_dag_identity();
        }
        let stable_round = self.ensure_canonical_identity_stable_round();
        let class_for_state = self.canonical_rounds[stable_round].classes.clone();
        let class_count = self.canonical_rounds[stable_round]
            .representative_by_class
            .len();
        let representative_by_class = self.canonical_rounds[stable_round]
            .representative_by_class
            .clone();
        let mut reverse_predecessors = vec![Vec::<u32>::new(); class_count];
        for (class, &representative) in representative_by_class.iter().enumerate() {
            let state = representative as usize;
            for &(_, destination) in self.topology.edges_from(state) {
                let destination_class = class_for_state[destination as usize] as usize;
                reverse_predecessors[destination_class].push(class as u32);
            }
        }
        for predecessors in &mut reverse_predecessors {
            predecessors.sort_unstable();
            predecessors.dedup();
        }
        self.support_quotient = Some(SupportQuotient {
            class_for_state: class_for_state.into(),
            representative_by_class: representative_by_class.into(),
            reverse_predecessors: reverse_predecessors.into(),
        });
    }

    fn ensure_terminal_erased_support_colors(&mut self) {
        if self.terminal_erased_support_colors.is_some() {
            return;
        }
        self.ensure_support_quotient();
        let quotient = self.support_quotient.as_ref().expect("support quotient initialized");
        let class_count = quotient.representative_by_class.len();
        let dead_class = quotient.class_for_state[self.dead_state()] as usize;
        let compact_colors = |hashes: Vec<u64>| {
            let mut class_for_hash = FxHashMap::<u64, u64>::default();
            hashes
                .into_iter()
                .map(|hash| {
                    let next = class_for_hash.len() as u64;
                    *class_for_hash.entry(hash).or_insert(next)
                })
                .collect::<Vec<_>>()
        };
        let mut initial_hashes = Vec::with_capacity(class_count);
        for class in 0..class_count {
            let state = quotient.representative_by_class[class] as usize;
            let observed = !quotient.reverse_predecessors[class].is_empty();
            let (finalizers, future_finalizers) = if observed {
                let pair = &self.output_pairs[self.output_pair_by_state[state] as usize];
                (pair.finalizers.0.len() as u64, pair.future_finalizers.0.len() as u64)
            } else { (0, 0) };
            let mut color = mix_structural_fingerprint(0x8c3c_010c_7d13_1f5b, finalizers);
            color = mix_structural_fingerprint(color, future_finalizers);
            initial_hashes.push(color);
        }
        let mut colors = compact_colors(initial_hashes);
        for _ in 0..64 {
            let default = colors[dead_class];
            let mut next_hashes = Vec::with_capacity(class_count);
            for class in 0..class_count {
                let state = quotient.representative_by_class[class] as usize;
                let mut color = mix_structural_fingerprint(0x9e37_79b9_7f4a_7c15, colors[class]);
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination_class = quotient.class_for_state[destination as usize] as usize;
                    if colors[destination_class] == default { continue; }
                    color = mix_structural_fingerprint(color, byte as u64);
                    color = mix_structural_fingerprint(color, colors[destination_class]);
                }
                next_hashes.push(color);
            }
            let next = compact_colors(next_hashes);
            if next == colors {
                break;
            }
            colors = next;
        }
        self.terminal_erased_support_colors = Some(colors.into());
    }

    /// Materialize exact quotient output-support columns for many terminals in
    /// one destination scan. This is equivalent to repeatedly calling
    /// `ensure_terminal_quotient_output_support`, but avoids rewalking the
    /// same predecessor lists and improves cache locality for family proofs.
    fn ensure_terminal_quotient_output_supports_bulk(&mut self, terminals: &[TerminalID]) {
        let mut requested = vec![false; self.finalizer_states_by_terminal.len()];
        let mut any_missing = false;
        for &terminal in terminals {
            let terminal = terminal as usize;
            if terminal >= requested.len() {
                continue;
            }
            requested[terminal] = true;
            any_missing |= !self
                .terminal_quotient_output_supports
                .as_ref()
                .is_some_and(|supports| supports.get(terminal).is_some_and(Option::is_some));
        }
        if !any_missing {
            return;
        }
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        self.ensure_support_quotient();
        let class_for_state = Arc::clone(
            &self
                .support_quotient
                .as_ref()
                .expect("support quotient initialized")
                .class_for_state,
        );
        let terminal_count = requested.len();
        let mut columns = vec![Vec::<(u32, u8)>::new(); terminal_count];
        for destination in 0..self.topology.real_state_count {
            let predecessors = &self.reverse_predecessors[destination];
            if predecessors.is_empty() {
                continue;
            }
            let output = &self.output_pairs[self.output_pair_by_state[destination] as usize];
            for (terminals, mask) in [(&output.finalizers.0, 1u8), (&output.future_finalizers.0, 2u8)] {
                for terminal in terminals {
                    let terminal = *terminal as usize;
                    if !requested[terminal] {
                        continue;
                    }
                    let column = &mut columns[terminal];
                    for &source in predecessors {
                        column.push((class_for_state[source as usize], mask));
                    }
                }
            }
        }
        let supports = self.terminal_quotient_output_supports.get_or_insert_with(|| {
            vec![None; terminal_count]
        });
        for (terminal, mut column) in columns.into_iter().enumerate() {
            if !requested[terminal] || supports[terminal].is_some() {
                continue;
            }
            column.sort_unstable_by_key(|&(class, _)| class);
            let mut write = 0usize;
            for read in 0..column.len() {
                if write > 0 && column[write - 1].0 == column[read].0 {
                    column[write - 1].1 |= column[read].1;
                } else {
                    column[write] = column[read];
                    write += 1;
                }
            }
            column.truncate(write);
            supports[terminal] = Some(column);
        }
        if let Some(started_at) = started_at {
            self.support_transposition_support_setup_ns += started_at.elapsed().as_nanos() as u64;
        }
    }

    /// Ordered terminal-erased support colors along a literal's selected-byte
    /// trace. Every valid restricted scanner transport preserves this sequence,
    /// so it is a sound candidate splitter before the complete orbit proof.
    fn literal_support_color_trace_signature(
        &mut self,
        tokenizer: &Tokenizer,
        terminal: TerminalID,
    ) -> Option<Box<[(u8, u64, u64)]>> {
        self.ensure_support_quotient();
        self.ensure_terminal_erased_support_colors();
        let class_for_state = Arc::clone(&self.support_quotient.as_ref()?.class_for_state);
        let colors = Arc::clone(self.terminal_erased_support_colors.as_ref()?);
        let trace = literal_selected_trace(tokenizer, &self.topology, terminal)?;
        Some(
            trace
                .transitions
                .iter()
                .map(|&(byte, source, destination)| {
                    (
                        byte,
                        colors[class_for_state[source as usize] as usize],
                        colors[class_for_state[destination as usize] as usize],
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }


    fn ensure_terminal_quotient_output_support(&mut self, terminal: TerminalID) {
        let terminal = terminal as usize;
        if self
            .terminal_quotient_output_supports
            .as_ref()
            .is_some_and(|supports| supports.get(terminal).is_some_and(Option::is_some))
        {
            return;
        }
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        self.ensure_support_quotient();
        let class_for_state = Arc::clone(
            &self
                .support_quotient
                .as_ref()
                .expect("support quotient initialized")
                .class_for_state,
        );
        let mut support = Vec::<(u32, u8)>::new();
        for (destinations, mask) in [
            (&self.finalizer_states_by_terminal[terminal], 1u8),
            (&self.future_finalizer_states_by_terminal[terminal], 2u8),
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    support.push((class_for_state[source as usize], mask));
                }
            }
        }
        support.sort_unstable_by_key(|&(class, _)| class);
        let mut write = 0usize;
        for read in 0..support.len() {
            if write > 0 && support[write - 1].0 == support[read].0 {
                support[write - 1].1 |= support[read].1;
            } else {
                support[write] = support[read];
                write += 1;
            }
        }
        support.truncate(write);
        let supports = self.terminal_quotient_output_supports.get_or_insert_with(|| {
            vec![None; self.finalizer_states_by_terminal.len()]
        });
        supports[terminal] = Some(support);
        if let Some(started_at) = started_at {
            self.support_transposition_support_setup_ns += started_at.elapsed().as_nanos() as u64;
        }
    }

    /// Propose the quotient action of swapping two terminal labels by pairing
    /// their terminal-specific observable support tracks.  It is only a
    /// candidate: `support_transposition_interchange_map` proves the complete
    /// local automorphism before returning it, otherwise discovery falls back
    /// to the ordinary exact checker.
    fn support_transposition_deviations(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<Vec<(u32, u32)>> {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        self.ensure_terminal_quotient_output_support(left);
        self.ensure_terminal_quotient_output_support(right);
        let supports = self
            .terminal_quotient_output_supports
            .as_ref()
            .expect("terminal quotient output supports initialized");
        let left_support = supports.get(left as usize)?.as_ref()?;
        let right_support = supports.get(right as usize)?.as_ref()?;
        let mut deviations = Vec::with_capacity(left_support.len() + right_support.len());
        for mask in 1..=3u8 {
            let mut left_only = SmallVec::<[u32; 8]>::new();
            let mut right_only = SmallVec::<[u32; 8]>::new();
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            loop {
                while left_index < left_support.len() && left_support[left_index].1 != mask {
                    left_index += 1;
                }
                while right_index < right_support.len() && right_support[right_index].1 != mask {
                    right_index += 1;
                }
                match (left_support.get(left_index), right_support.get(right_index)) {
                    (Some(&(left_class, _)), Some(&(right_class, _))) if left_class == right_class => {
                        left_index += 1;
                        right_index += 1;
                    }
                    (Some(&(left_class, _)), Some(&(right_class, _))) if left_class < right_class => {
                        left_only.push(left_class);
                        left_index += 1;
                    }
                    (Some(_), Some(&(right_class, _))) => {
                        right_only.push(right_class);
                        right_index += 1;
                    }
                    (Some(&(left_class, _)), None) => {
                        left_only.push(left_class);
                        left_index += 1;
                    }
                    (None, Some(&(right_class, _))) => {
                        right_only.push(right_class);
                        right_index += 1;
                    }
                    (None, None) => break,
                }
            }
            if left_only.len() != right_only.len() {
                return None;
            }
            for (left_class, right_class) in left_only.into_iter().zip(right_only) {
                deviations.push((left_class, right_class));
                deviations.push((right_class, left_class));
            }
        }
        deviations.sort_unstable_by_key(|&(source, _)| source);
        let result = deviations
            .windows(2)
            .all(|pair| pair[0].0 != pair[1].0)
            .then_some(deviations);
        if let Some(started_at) = started_at {
            self.support_transposition_template_ns += started_at.elapsed().as_nanos() as u64;
        }
        result
    }

    fn symmetric_fiber_product_witnesses(
        &mut self,
        members: &[TerminalID],
    ) -> Option<Vec<(TerminalID, InterchangeMap)>> {
        if members.len() < 3 {
            return None;
        }
        let tuple_rows = self.uniform_support_fiber_tuples(members)?;
        let member_count = members.len();
        let quotient = self.support_quotient.as_ref()?;
        let mut member_index_by_terminal = vec![usize::MAX; self.finalizer_states_by_terminal.len()];
        for (member, &terminal) in members.iter().enumerate() {
            member_index_by_terminal[terminal as usize] = member;
        }
        let coordinate_count = tuple_rows.first()?.len();
        let dead_class = quotient.class_for_state[self.dead_state()];
        let mut coordinates = Vec::<Vec<u32>>::new();
        let mut class_cell = vec![None; quotient.representative_by_class.len()];
        let mut coordinate_by_tuple = FxHashMap::<Vec<u32>, usize>::default();
        let mut worklist = VecDeque::<usize>::new();
        for coordinate in 0..coordinate_count {
            let classes = tuple_rows
                .iter()
                .map(|tuple| tuple[coordinate])
                .collect::<Vec<_>>();
            Self::register_literal_fiber_coordinate(
                classes,
                dead_class,
                &mut coordinates,
                &mut class_cell,
                &mut coordinate_by_tuple,
                &mut worklist,
            )?;
        }
        let root = quotient.class_for_state[self.topology.initial_state] as usize;
        if class_cell[root].is_some() {
            return None;
        }
        while let Some(coordinate) = worklist.pop_front() {
            let source_classes = coordinates[coordinate].clone();
            for &byte in &self.topology.bytes {
                let mut destination_states = Vec::with_capacity(member_count);
                let mut destination_classes = Vec::with_capacity(member_count);
                for &source_class in &source_classes {
                    let source = quotient.representative_by_class[source_class as usize] as usize;
                    let destination = self.topology.destination_for_byte(source, byte);
                    destination_states.push(destination);
                    destination_classes.push(quotient.class_for_state[destination]);
                }
                let expected = self.normalized_fiber_output_pair(
                    destination_states[0],
                    &member_index_by_terminal,
                    Some(0),
                    member_count,
                )?;
                for member in 1..member_count {
                    if self.normalized_fiber_output_pair(
                        destination_states[member],
                        &member_index_by_terminal,
                        Some(member),
                        member_count,
                    )? != expected {
                        return None;
                    }
                }
                if destination_classes.iter().all(|&class| class == destination_classes[0]) {
                    continue;
                }
                if let Some(&existing) = coordinate_by_tuple.get(&destination_classes) {
                    if coordinates[existing] != destination_classes {
                        return None;
                    }
                } else {
                    Self::register_literal_fiber_coordinate(
                        destination_classes,
                        dead_class,
                        &mut coordinates,
                        &mut class_cell,
                        &mut coordinate_by_tuple,
                        &mut worklist,
                    )?;
                }
            }
        }
        let supports = self.terminal_quotient_output_supports.as_ref()?;
        let mut sensitive_classes = BTreeSet::<u32>::new();
        for &terminal in members {
            for &(class, _) in supports.get(terminal as usize)?.as_ref()? {
                sensitive_classes.insert(class);
            }
        }
        for class in sensitive_classes {
            let class = class as usize;
            if class_cell[class].is_some() {
                continue;
            }
            let source = quotient.representative_by_class[class] as usize;
            for &byte in &self.topology.bytes {
                let destination = self.topology.destination_for_byte(source, byte);
                let destination_class = quotient.class_for_state[destination] as usize;
                if class_cell[destination_class].is_some() {
                    return None;
                }
                self.normalized_fiber_output_pair(
                    destination,
                    &member_index_by_terminal,
                    None,
                    member_count,
                )?;
            }
        }
        for (class, cell) in class_cell.iter().enumerate() {
            let Some((_, member)) = *cell else {
                continue;
            };
            for &predecessor in &quotient.reverse_predecessors[class] {
                match class_cell[predecessor as usize] {
                    Some((_, predecessor_member)) if predecessor_member == member => {}
                    _ => return None,
                }
            }
        }
        let mut witnesses = Vec::with_capacity(members.len().saturating_sub(1));
        for member in 1..members.len() {
            let mut permutation = (0..members.len()).collect::<Vec<_>>();
            permutation.swap(0, member);
            let deviations = Self::deviations_for_terminal_permutation(
                &coordinates,
                &permutation,
            )?;
            witnesses.push((
                members[member],
                InterchangeMap {
                    scanner_state_map: self.quotient_transport_state_map(
                        Arc::clone(&quotient.class_for_state),
                        Arc::clone(&quotient.representative_by_class),
                        deviations.into_boxed_slice(),
                    ),
                },
            ));
        }
        Some(witnesses)
    }

    #[inline]
    fn mapped_support_class(deviations: &[(u32, u32)], class: usize) -> u32 {
        deviations
            .binary_search_by_key(&(class as u32), |&(source, _)| source)
            .ok()
            .map(|index| deviations[index].1)
            .unwrap_or(class as u32)
    }

    fn canonical_quotient_support_permuted_signature(
        &self,
        quotient: &CanonicalQuotient,
        class: usize,
        deviations: &[(u32, u32)],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let state = quotient.representative_by_class[class] as usize;
        let default_class = Self::mapped_support_class(
            deviations,
            quotient.class_for_state[self.dead_state()] as usize,
        );
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            let previous_class = Self::mapped_support_class(
                deviations,
                quotient.class_for_state[destination] as usize,
            );
            if previous_class == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class,
                output,
            });
        }
        CanonicalSignature(components.into())
    }

    fn raw_identity_classes(&mut self) -> Arc<[u32]> {
        if self.raw_identity_classes.is_none() {
            self.raw_identity_classes = Some(
                (0..self.topology.real_state_count as u32)
                    .collect::<Vec<_>>()
                    .into(),
            );
        }
        Arc::clone(
            self.raw_identity_classes
                .as_ref()
                .expect("raw identity classes initialized"),
        )
    }

    fn insert_raw_involution_pair(
        mapping: &mut [u32],
        worklist: &mut VecDeque<(u32, u32)>,
        left: u32,
        right: u32,
    ) -> bool {
        const UNMAPPED: u32 = u32::MAX;
        if left == right {
            return mapping[left as usize] == UNMAPPED || mapping[left as usize] == left;
        }
        if (mapping[left as usize] != UNMAPPED && mapping[left as usize] != right)
            || (mapping[right as usize] != UNMAPPED && mapping[right as usize] != left)
        {
            return false;
        }
        let is_new = mapping[left as usize] == UNMAPPED;
        mapping[left as usize] = right;
        mapping[right as usize] = left;
        if is_new {
            worklist.push_back((left, right));
            worklist.push_back((right, left));
        }
        true
    }

    /// Prove a binary TI transport directly on the raw restricted topology.
    /// Aligned selected-byte literal milestones seed the swap; closure under
    /// selected forward edges and uniquely determined reverse predecessors
    /// proves that identity outside the resulting cone is sound. Ambiguity is
    /// an exact fallback condition, never a guessed pairing.
    fn raw_literal_trace_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
        left_trace: &LiteralSelectedTrace,
        right_trace: &LiteralSelectedTrace,
    ) -> Option<InterchangeMap> {
        if left_trace.transitions.len() != right_trace.transitions.len() {
            return None;
        }
        let real_state_count = self.topology.real_state_count;
        let dead_state = self.topology.dead_state();
        let mut mapping = vec![u32::MAX; real_state_count];
        let mut worklist = VecDeque::<(u32, u32)>::new();
        for (&(left_byte, left_source, left_destination), &(right_byte, right_source, right_destination))
            in left_trace.transitions.iter().zip(&right_trace.transitions)
        {
            if left_byte != right_byte
                || !Self::insert_raw_involution_pair(
                    &mut mapping,
                    &mut worklist,
                    left_source,
                    right_source,
                )
                || !Self::insert_raw_involution_pair(
                    &mut mapping,
                    &mut worklist,
                    left_destination,
                    right_destination,
                )
            {
                return None;
            }
        }
        if worklist.is_empty() {
            return None;
        }

        let mut swapped_outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        while let Some((source, target)) = worklist.pop_front() {
            debug_assert_ne!(source, target);
            let source_edges = self.topology.edges_from(source as usize);
            let target_edges = self.topology.edges_from(target as usize);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_destination as usize, target_destination as usize)
                    }
                    _ => return None,
                };
                if swapped_outputs.id(self.output_pair_by_state[source_destination])
                    != self.output_pair_by_state[target_destination]
                    || !Self::insert_raw_involution_pair(
                        &mut mapping,
                        &mut worklist,
                        source_destination as u32,
                        target_destination as u32,
                    )
                {
                    return None;
                }
            }

            let source_reverse = &self.reverse_edges[source as usize];
            let target_reverse = &self.reverse_edges[target as usize];
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_reverse.len() || target_index < target_reverse.len() {
                let byte = match (source_reverse.get(source_index), target_reverse.get(target_index)) {
                    (Some(&(source_byte, _)), Some(&(target_byte, _))) if source_byte == target_byte => source_byte,
                    _ => return None,
                };
                let source_end = source_index
                    + source_reverse[source_index..]
                        .iter()
                        .take_while(|&&(edge_byte, _)| edge_byte == byte)
                        .count();
                let target_end = target_index
                    + target_reverse[target_index..]
                        .iter()
                        .take_while(|&&(edge_byte, _)| edge_byte == byte)
                        .count();
                let source_predecessors = &source_reverse[source_index..source_end];
                let target_predecessors = &target_reverse[target_index..target_end];
                if source_predecessors.len() != target_predecessors.len() {
                    return None;
                }
                let mut unmatched_source = Vec::new();
                let mut unmatched_target = Vec::new();
                for &(_, predecessor) in source_predecessors {
                    let mapped = mapping[predecessor as usize];
                    if mapped == u32::MAX {
                        unmatched_source.push(predecessor);
                    } else if target_predecessors
                        .binary_search_by_key(&mapped, |&(_, source)| source)
                        .is_err()
                    {
                        return None;
                    }
                }
                for &(_, predecessor) in target_predecessors {
                    let mapped = mapping[predecessor as usize];
                    if mapped == u32::MAX {
                        unmatched_target.push(predecessor);
                    } else if source_predecessors
                        .binary_search_by_key(&mapped, |&(_, source)| source)
                        .is_err()
                    {
                        return None;
                    }
                }
                if unmatched_source.len() != unmatched_target.len() {
                    return None;
                }
                match unmatched_source.len() {
                    0 => {}
                    1 => {
                        if !Self::insert_raw_involution_pair(
                            &mut mapping,
                            &mut worklist,
                            unmatched_source[0],
                            unmatched_target[0],
                        ) {
                            return None;
                        }
                    }
                    _ => return None,
                }
                source_index = source_end;
                target_index = target_end;
            }
        }

        let initial = self.topology.initial_state;
        if mapping[initial] != u32::MAX && mapping[initial] != initial as u32 {
            return None;
        }
        for states in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &state in states {
                let state = state as usize;
                if self.reverse_predecessors[state].is_empty() {
                    continue;
                }
                let target = mapping[state];
                if target == u32::MAX
                    || swapped_outputs.id(self.output_pair_by_state[state])
                        != self.output_pair_by_state[target as usize]
                {
                    return None;
                }
            }
        }

        let deviations = mapping
            .into_iter()
            .enumerate()
            .filter_map(|(source, target)| {
                (target != u32::MAX && target != source as u32).then_some((source as u32, target))
            })
            .collect::<Vec<_>>();
        if deviations.is_empty() {
            return None;
        }
        let identity = self.raw_identity_classes();
        Some(InterchangeMap {
            scanner_state_map: self.quotient_transport_state_map(
                Arc::clone(&identity),
                identity,
                deviations.into_boxed_slice(),
            ),
        })
    }

    fn support_transposition_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let Some(deviations) = self.support_transposition_deviations(left, right) else {
            self.support_transposition_no_template += 1;
            return None;
        };
        self.support_transposition_interchange_map_from_deviations(left, right, deviations)
    }

    fn support_transposition_interchange_map_from_deviations(
        &mut self,
        left: TerminalID,
        right: TerminalID,
        deviations: Vec<(u32, u32)>,
    ) -> Option<InterchangeMap> {
        let quotient = self
            .support_quotient
            .as_ref()
            .expect("support quotient initialized");
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let cone_started_at = profile_timing.then(Instant::now);
        let cone_classes =
            self.support_quotient_affected_cone_small(quotient, left, right);
        if let Some(started_at) = cone_started_at {
            self.support_transposition_cone_ns += started_at.elapsed().as_nanos() as u64;
        }
        debug_assert!(deviations.iter().all(|&(source, target)| {
            (source as usize) < quotient.representative_by_class.len()
                && (target as usize) < quotient.representative_by_class.len()
                && cone_classes.contains(&(source as usize))
                && cone_classes.contains(&(target as usize))
        }));
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(&deviations, root_class) != root_class as u32 {
            self.support_transposition_root_rejected += 1;
            return None;
        }
        let verify_started_at = profile_timing.then(Instant::now);
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        for &class in &cone_classes {
            let target = Self::mapped_support_class(&deviations, class);
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target as usize] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (byte, source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_byte, source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                        if source_byte < target_byte =>
                    {
                        source_index += 1;
                        (source_byte, source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(target_byte, target_destination))) => {
                        target_index += 1;
                        (target_byte, self.dead_state(), target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), None) => {
                        source_index += 1;
                        (source_byte, source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(target_byte, target_destination))) => {
                        target_index += 1;
                        (target_byte, self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("sparse edge union loop is nonempty"),
                };
                let expected_destination = Self::mapped_support_class(
                    &deviations,
                    quotient.class_for_state[source_destination] as usize,
                );
                if quotient.class_for_state[target_destination] != expected_destination
                    || outputs.id(self.output_pair_by_state[source_destination])
                        != self.output_pair_by_state[target_destination]
                {
                    self.support_transposition_signature_rejected += 1;
                    if let Some(started_at) = verify_started_at {
                        self.support_transposition_verify_ns +=
                            started_at.elapsed().as_nanos() as u64;
                    }
                    return None;
                }
            }
        }
        if let Some(started_at) = verify_started_at {
            self.support_transposition_verify_ns += started_at.elapsed().as_nanos() as u64;
        }
        self.support_transposition_certified += 1;
        Some(InterchangeMap {
            scanner_state_map: self.quotient_transport_state_map(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                deviations.into_boxed_slice(),
            ),
        })
    }

    fn support_difference_by_mask(
        left: &[(u32, u8)],
        right: &[(u32, u8)],
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut all_left = Vec::new();
        let mut all_right = Vec::new();
        for mask in 1..=3u8 {
            let mut left_only = SmallVec::<[u32; 8]>::new();
            let mut right_only = SmallVec::<[u32; 8]>::new();
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            loop {
                while left_index < left.len() && left[left_index].1 != mask {
                    left_index += 1;
                }
                while right_index < right.len() && right[right_index].1 != mask {
                    right_index += 1;
                }
                match (left.get(left_index), right.get(right_index)) {
                    (Some(&(left_class, _)), Some(&(right_class, _))) if left_class == right_class => {
                        left_index += 1;
                        right_index += 1;
                    }
                    (Some(&(left_class, _)), Some(&(right_class, _))) if left_class < right_class => {
                        left_only.push(left_class);
                        left_index += 1;
                    }
                    (Some(_), Some(&(right_class, _))) => {
                        right_only.push(right_class);
                        right_index += 1;
                    }
                    (Some(&(left_class, _)), None) => {
                        left_only.push(left_class);
                        left_index += 1;
                    }
                    (None, Some(&(right_class, _))) => {
                        right_only.push(right_class);
                        right_index += 1;
                    }
                    (None, None) => break,
                }
            }
            if left_only.len() != right_only.len() {
                return None;
            }
            all_left.extend(left_only);
            all_right.extend(right_only);
        }
        Some((all_left, all_right))
    }

    fn uniform_support_fiber_tuples(&mut self, members: &[TerminalID]) -> Option<Vec<Vec<u32>>> {
        let (&representative, rest) = members.split_first()?;
        if rest.is_empty() {
            return None;
        }
        self.ensure_support_quotient();
        self.ensure_terminal_quotient_output_supports_bulk(members);
        let supports = self.terminal_quotient_output_supports.as_ref()?;
        let representative_support = supports.get(representative as usize)?.as_ref()?;
        let mut tuples = Vec::<Vec<u32>>::with_capacity(members.len());
        let mut base_tuple = None::<Vec<u32>>;
        for &terminal in rest {
            let support = supports.get(terminal as usize)?.as_ref()?;
            let (left_only, right_only) = Self::support_difference_by_mask(representative_support, support)?;
            if left_only.is_empty() {
                return None;
            }
            if let Some(base) = &base_tuple {
                if base != &left_only {
                    return None;
                }
            } else {
                base_tuple = Some(left_only);
            }
            tuples.push(right_only);
        }
        let base_tuple = base_tuple?;
        let coordinate_count = base_tuple.len();
        if coordinate_count == 0 || tuples.iter().any(|tuple| tuple.len() != coordinate_count) {
            return None;
        }
        tuples.insert(0, base_tuple);
        let quotient = self.support_quotient.as_ref()?;
        let mut owner = vec![false; quotient.representative_by_class.len()];
        for tuple in &tuples {
            for &class in tuple {
                let slot = owner.get_mut(class as usize)?;
                if std::mem::replace(slot, true) {
                    return None;
                }
            }
        }
        Some(tuples)
    }

    #[inline]
    fn group_output_kind(
        bits: &OutputBits,
        group_member: &[bool],
        owner_terminal: Option<TerminalID>,
        group_len: usize,
    ) -> Option<(u8, Box<[TerminalID]>)> {
        let mut group_count = 0usize;
        let mut contains_owner = false;
        let mut outside = Vec::new();
        for &terminal in &bits.0 {
            if group_member[terminal as usize] {
                group_count += 1;
                contains_owner |= owner_terminal == Some(terminal);
            } else {
                outside.push(terminal);
            }
        }
        let kind = match (group_count, contains_owner) {
            (0, _) => 0,
            (count, _) if count == group_len => 1,
            (1, true) => 2,
            (count, false) if count + 1 == group_len => 3,
            _ => return None,
        };
        Some((kind, outside.into_boxed_slice()))
    }

    /// Exact full-fibre certificate, evaluated only on fibre rows, incoming
    /// fibre boundaries, and core states whose frozen output mentions the
    /// candidate group. This is equivalent to the saved full-quotient theorem
    /// but avoids rewalking the fixed unaffected core for every family.
    fn uniform_fiber_orbit_witnesses_local(
        &mut self,
        members: &[TerminalID],
    ) -> Option<Vec<(TerminalID, InterchangeMap)>> {
        if members.len() < 2 {
            return None;
        }
        let tuples = self.uniform_support_fiber_tuples(members)?;
        let quotient = self.support_quotient.as_ref()?;
        let class_count = quotient.representative_by_class.len();
        let mut owner = vec![None::<(usize, usize)>; class_count];
        for (member, tuple) in tuples.iter().enumerate() {
            for (coordinate, &class) in tuple.iter().enumerate() {
                let slot = owner.get_mut(class as usize)?;
                if slot.replace((member, coordinate)).is_some() {
                    return None;
                }
            }
        }
        let root = quotient.class_for_state[self.topology.initial_state] as usize;
        if owner[root].is_some() {
            return None;
        }

        let mut member_index_by_terminal = vec![usize::MAX; self.finalizer_states_by_terminal.len()];
        for (member, &terminal) in members.iter().enumerate() {
            member_index_by_terminal[terminal as usize] = member;
        }
        let coordinate_count = tuples.first()?.len();
        type NormalizedOutput = (SmallVec<[TerminalID; 4]>, u8);
        type FibreRow = Vec<(u8, u64, NormalizedOutput, NormalizedOutput)>;
        let mut templates = vec![None::<FibreRow>; coordinate_count];
        let mut fibre_classes = Vec::<(usize, usize, usize)>::new();
        for (member, tuple) in tuples.iter().enumerate() {
            for (coordinate, &class) in tuple.iter().enumerate() {
                let class = class as usize;
                let state = quotient.representative_by_class[class] as usize;
                let mut row = Vec::new();
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination_state = destination as usize;
                    let destination_class = quotient.class_for_state[destination_state] as usize;
                    let target = match owner[destination_class] {
                        None => destination_class as u64,
                        Some((target_member, target_coordinate)) if target_member == member => {
                            (1u64 << 63) | target_coordinate as u64
                        }
                        Some(_) => return None,
                    };
                    let (finalizers, futures) = self.normalized_fiber_output_pair(
                        destination_state,
                        &member_index_by_terminal,
                        Some(member),
                        members.len(),
                    )?;
                    row.push((byte, target, finalizers, futures));
                }
                if let Some(existing) = &templates[coordinate] {
                    if existing != &row {
                        return None;
                    }
                } else {
                    templates[coordinate] = Some(row);
                }
                fibre_classes.push((class, member, coordinate));
            }
        }
        if templates.iter().any(Option::is_none) {
            return None;
        }

        for (class, member, _) in &fibre_classes {
            for &predecessor in &quotient.reverse_predecessors[*class] {
                match owner[predecessor as usize] {
                    Some((predecessor_member, _)) if predecessor_member == *member => {}
                    _ => return None,
                }
            }
        }

        let supports = self.terminal_quotient_output_supports.as_ref()?;
        let mut core_output_seen = vec![false; class_count];
        for &terminal in members {
            for &(class, _) in supports[terminal as usize].as_ref()? {
                let class = class as usize;
                if owner[class].is_some() || std::mem::replace(&mut core_output_seen[class], true) {
                    continue;
                }
                let state = quotient.representative_by_class[class] as usize;
                for &(_, destination) in self.topology.edges_from(state) {
                    let (finalizers, futures) = self.normalized_fiber_output_pair(
                        destination as usize,
                        &member_index_by_terminal,
                        None,
                        members.len(),
                    )?;
                    if finalizers.1 > 1 || futures.1 > 1 {
                        return None;
                    }
                }
            }
        }

        let mut witnesses = Vec::with_capacity(members.len().saturating_sub(1));
        for member in 1..members.len() {
            let deviations = Self::deviations_for_representative_swap(&tuples, member)?;
            witnesses.push((
                members[member],
                InterchangeMap {
                    scanner_state_map: self.quotient_transport_state_map(
                        Arc::clone(&quotient.class_for_state),
                        Arc::clone(&quotient.representative_by_class),
                        deviations.into_boxed_slice(),
                    ),
                },
            ));
        }
        Some(witnesses)
    }

    fn deviations_for_terminal_permutation(
        tuples: &[Vec<u32>],
        permutation: &[usize],
    ) -> Option<Vec<(u32, u32)>> {
        if tuples.len() != permutation.len()
            || permutation.iter().any(|&member| member >= tuples.len())
        {
            return None;
        }
        let mut seen = vec![false; tuples.len()];
        for &member in permutation {
            if std::mem::replace(&mut seen[member], true) {
                return None;
            }
        }
        let coordinate_count = tuples.first()?.len();
        if tuples.iter().any(|tuple| tuple.len() != coordinate_count) {
            return None;
        }
        let mut deviations = Vec::with_capacity(tuples.len() * coordinate_count);
        for (source_member, tuple) in tuples.iter().enumerate() {
            let target = &tuples[permutation[source_member]];
            for (&source, &target) in tuple.iter().zip(target) {
                if source != target {
                    deviations.push((source, target));
                }
            }
        }
        deviations.sort_unstable_by_key(|&(source, _)| source);
        deviations
            .windows(2)
            .all(|pair| pair[0].0 != pair[1].0)
            .then_some(deviations)
    }

    fn deviations_for_representative_swap(
        tuples: &[Vec<u32>],
        member: usize,
    ) -> Option<Vec<(u32, u32)>> {
        let representative = tuples.first()?;
        let target = tuples.get(member)?;
        if member == 0 || representative.len() != target.len() {
            return None;
        }
        let mut deviations = Vec::with_capacity(representative.len() * 2);
        for (&source, &destination) in representative.iter().zip(target) {
            if source != destination {
                deviations.push((source, destination));
                deviations.push((destination, source));
            }
        }
        deviations.sort_unstable_by_key(|&(source, _)| source);
        deviations
            .windows(2)
            .all(|pair| pair[0].0 != pair[1].0)
            .then_some(deviations)
    }

    fn permuted_group_output_bits(
        bits: &OutputBits,
        member_index_by_terminal: &[usize],
        members: &[TerminalID],
        permutation: &[usize],
    ) -> OutputBits {
        let mut terminals = SmallVec::<[TerminalID; 4]>::new();
        for &terminal in &bits.0 {
            let member = member_index_by_terminal
                .get(terminal as usize)
                .copied()
                .unwrap_or(usize::MAX);
            terminals.push(if member == usize::MAX {
                terminal
            } else {
                members[permutation[member]]
            });
        }
        terminals.sort_unstable();
        if terminals.windows(2).any(|pair| pair[0] == pair[1]) {
            return OutputBits(SmallVec::new());
        }
        OutputBits(terminals)
    }

    /// Normalize a frozen output relative to one member of a symmetric fibre.
    /// Group labels may be absent, contain every member, or be exactly the
    /// current fibre member. Any other subset is not invariant under the full
    /// symmetric action and therefore cannot use this certificate.
    fn normalized_fiber_output_bits(
        bits: &OutputBits,
        member_index_by_terminal: &[usize],
        current_member: Option<usize>,
        member_count: usize,
    ) -> Option<(SmallVec<[TerminalID; 4]>, u8)> {
        let mut nongroup = SmallVec::<[TerminalID; 4]>::new();
        let mut group_count = 0usize;
        let mut contains_current = false;
        for &terminal in &bits.0 {
            let member = member_index_by_terminal
                .get(terminal as usize)
                .copied()
                .unwrap_or(usize::MAX);
            if member == usize::MAX {
                nongroup.push(terminal);
            } else {
                group_count += 1;
                contains_current |= current_member == Some(member);
            }
        }
        let kind = if group_count == 0 {
            0
        } else if group_count == member_count {
            1
        } else if group_count == 1 && contains_current {
            2
        } else if group_count + 1 == member_count && !contains_current {
            3
        } else {
            return None;
        };
        Some((nongroup, kind))
    }

    fn normalized_fiber_output_pair(
        &self,
        state: usize,
        member_index_by_terminal: &[usize],
        current_member: Option<usize>,
        member_count: usize,
    ) -> Option<((SmallVec<[TerminalID; 4]>, u8), (SmallVec<[TerminalID; 4]>, u8))> {
        let output = &self.output_pairs[self.output_pair_by_state[state] as usize];
        Some((
            Self::normalized_fiber_output_bits(
                &output.finalizers,
                member_index_by_terminal,
                current_member,
                member_count,
            )?,
            Self::normalized_fiber_output_bits(
                &output.future_finalizers,
                member_index_by_terminal,
                current_member,
                member_count,
            )?,
        ))
    }

    fn support_quotient_group_affected_cone(
        &self,
        quotient: &SupportQuotient,
        members: &[TerminalID],
    ) -> Vec<usize> {
        let mut in_cone = vec![false; quotient.representative_by_class.len()];
        let mut cone = Vec::<usize>::new();
        let mut worklist = Vec::<usize>::new();
        for &terminal in members {
            for destinations in [
                &self.finalizer_states_by_terminal[terminal as usize],
                &self.future_finalizer_states_by_terminal[terminal as usize],
            ] {
                for &destination in destinations {
                    for &source in &self.reverse_predecessors[destination as usize] {
                        let class = quotient.class_for_state[source as usize] as usize;
                        if !in_cone[class] {
                            in_cone[class] = true;
                            cone.push(class);
                            worklist.push(class);
                        }
                    }
                }
            }
        }
        while let Some(class) = worklist.pop() {
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !in_cone[predecessor] {
                    in_cone[predecessor] = true;
                    cone.push(predecessor);
                    worklist.push(predecessor);
                }
            }
        }
        cone
    }

    fn mapped_output_pair_ids_for_terminal_permutation(
        &self,
        member_index_by_terminal: &[usize],
        members: &[TerminalID],
        permutation: &[usize],
    ) -> Option<Vec<u32>> {
        let mut pair_ids = FxHashMap::<OutputPair, u32>::default();
        for (id, pair) in self.output_pairs.iter().enumerate() {
            pair_ids.insert(pair.clone(), id as u32);
        }
        self.output_pairs.iter().map(|pair| {
            pair_ids.get(&OutputPair {
                finalizers: Self::permuted_group_output_bits(
                    &pair.finalizers, member_index_by_terminal, members, permutation,
                ),
                future_finalizers: Self::permuted_group_output_bits(
                    &pair.future_finalizers, member_index_by_terminal, members, permutation,
                ),
            }).copied()
        }).collect()
    }

    fn support_quotient_permutation_is_automorphism(
        &self,
        quotient: &SupportQuotient,
        deviations: &[(u32, u32)],
        members: &[TerminalID],
        member_index_by_terminal: &[usize],
        permutation: &[usize],
    ) -> bool {
        let diagnostics = std::env::var_os("GLRMASK_PROFILE_L2P_ORBIT_REJECTION_DIAGNOSTIC")
            .is_some();
        let root = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(deviations, root) != root as u32 {
            if diagnostics {
                eprintln!("[glrmask/profile][ti_orbit_reject_detail] kind=root class={}", root);
            }
            return false;
        }
        let cone = self.support_quotient_group_affected_cone(quotient, members);
        for class in cone {
            let target = Self::mapped_support_class(deviations, class) as usize;
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                        if source_byte < target_byte =>
                    {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (Some(&(_, source_destination)), None) => {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("nonempty sparse edge union loop"),
                };
                let expected = Self::mapped_support_class(
                    deviations,
                    quotient.class_for_state[source_destination] as usize,
                );
                if quotient.class_for_state[target_destination] != expected {
                    if diagnostics {
                        eprintln!(
                            "[glrmask/profile][ti_orbit_reject_detail] kind=transition class={} target={} expected={} actual={}",
                            class,
                            target,
                            expected,
                            quotient.class_for_state[target_destination],
                        );
                    }
                    return false;
                }
                let source_pair = &self.output_pairs[self.output_pair_by_state[source_destination] as usize];
                let target_pair = &self.output_pairs[self.output_pair_by_state[target_destination] as usize];
                if Self::permuted_group_output_bits(
                    &source_pair.finalizers,
                    member_index_by_terminal,
                    members,
                    permutation,
                ) != target_pair.finalizers
                    || Self::permuted_group_output_bits(
                        &source_pair.future_finalizers,
                        member_index_by_terminal,
                        members,
                        permutation,
                    ) != target_pair.future_finalizers
                {
                    if diagnostics {
                        eprintln!(
                            "[glrmask/profile][ti_orbit_reject_detail] kind=output class={} target={} source_destination_class={} target_destination_class={}",
                            class,
                            target,
                            quotient.class_for_state[source_destination],
                            quotient.class_for_state[target_destination],
                        );
                    }
                    return false;
                }
            }
        }
        true
    }

    /// Exact orbit verification with one dense output-pair relabelling table
    /// per generator. This has the same transition/cone test as the ordinary
    /// verifier, but each edge compares integer pair IDs instead of rebuilding
    /// two mapped terminal sets.
    fn support_quotient_permutation_is_automorphism_dense_outputs(
        &self,
        quotient: &SupportQuotient,
        cone: &[usize],
        deviations: &[(u32, u32)],
        members: &[TerminalID],
        member_index_by_terminal: &[usize],
        permutation: &[usize],
    ) -> bool {
        let root = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(deviations, root) != root as u32 {
            return false;
        }
        const UNMAPPED: u32 = u32::MAX;
        const INVALID: u32 = u32::MAX - 1;
        let mut mapped_output_pair_ids = vec![UNMAPPED; self.output_pairs.len()];
        for &class in cone {
            let target = Self::mapped_support_class(deviations, class) as usize;
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                        if source_byte < target_byte =>
                    {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (Some(&(_, source_destination)), None) => {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("nonempty sparse edge union loop"),
                };
                let expected = Self::mapped_support_class(
                    deviations,
                    quotient.class_for_state[source_destination] as usize,
                );
                if quotient.class_for_state[target_destination] != expected {
                    return false;
                }
                let source_pair = self.output_pair_by_state[source_destination] as usize;
                if mapped_output_pair_ids[source_pair] == UNMAPPED {
                    let source_pair_value = &self.output_pairs[source_pair];
                    let mapped = OutputPair {
                        finalizers: Self::permuted_group_output_bits(
                            &source_pair_value.finalizers,
                            member_index_by_terminal,
                            members,
                            permutation,
                        ),
                        future_finalizers: Self::permuted_group_output_bits(
                            &source_pair_value.future_finalizers,
                            member_index_by_terminal,
                            members,
                            permutation,
                        ),
                    };
                    mapped_output_pair_ids[source_pair] = self
                        .output_pair_lookup
                        .get(&mapped)
                        .copied()
                        .unwrap_or(INVALID);
                }
                if mapped_output_pair_ids[source_pair] == INVALID
                    || self.output_pairs[mapped_output_pair_ids[source_pair] as usize]
                    != self.output_pairs[self.output_pair_by_state[target_destination] as usize]
                {
                    return false;
                }
            }
        }
        true
    }

    /// Exact counterpart of the ordinary orbit verifier that interns the
    /// terminal-permuted frozen output once per source output pair. The
    /// transition and cone checks are intentionally identical to the ordinary
    /// verifier; only repeated set remapping on equal output pairs is removed.
    fn support_quotient_permutation_is_automorphism_cached(
        &self,
        quotient: &SupportQuotient,
        cone: &[usize],
        deviations: &[(u32, u32)],
        members: &[TerminalID],
        member_index_by_terminal: &[usize],
        permutation: &[usize],
    ) -> bool {
        let root = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(deviations, root) != root as u32 {
            return false;
        }
        let mut mapped_pairs = FxHashMap::<u32, OutputPair>::default();
        for &class in cone {
            let target = Self::mapped_support_class(deviations, class) as usize;
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                        if source_byte < target_byte =>
                    {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (Some(&(_, source_destination)), None) => {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("nonempty sparse edge union loop"),
                };
                let expected = Self::mapped_support_class(
                    deviations,
                    quotient.class_for_state[source_destination] as usize,
                );
                if quotient.class_for_state[target_destination] != expected {
                    return false;
                }
                let source_pair_id = self.output_pair_by_state[source_destination];
                let mapped_pair = mapped_pairs.entry(source_pair_id).or_insert_with(|| {
                    let source_pair = &self.output_pairs[source_pair_id as usize];
                    OutputPair {
                        finalizers: Self::permuted_group_output_bits(
                            &source_pair.finalizers,
                            member_index_by_terminal,
                            members,
                            permutation,
                        ),
                        future_finalizers: Self::permuted_group_output_bits(
                            &source_pair.future_finalizers,
                            member_index_by_terminal,
                            members,
                            permutation,
                        ),
                    }
                });
                if mapped_pair
                    != &self.output_pairs[self.output_pair_by_state[target_destination] as usize]
                {
                    return false;
                }
            }
        }
        true
    }

    fn support_quotient_permutation_is_automorphism_fast(
        &self,
        quotient: &SupportQuotient,
        deviations: &[(u32, u32)],
        members: &[TerminalID],
        member_index_by_terminal: &[usize],
        permutation: &[usize],
    ) -> bool {
        let class_count = quotient.representative_by_class.len();
        let mut mapped_class = (0..class_count)
            .map(|class| class as u32)
            .collect::<Vec<_>>();
        let mut seen_target = vec![false; class_count];
        for &(source, target) in deviations {
            let source = source as usize;
            let target = target as usize;
            if source >= class_count || target >= class_count || seen_target[target] {
                return false;
            }
            seen_target[target] = true;
            mapped_class[source] = target as u32;
        }
        let mut image_seen = vec![false; class_count];
        for &target in &mapped_class {
            let target = target as usize;
            if target >= class_count || std::mem::replace(&mut image_seen[target], true) {
                return false;
            }
        }
        let root = quotient.class_for_state[self.topology.initial_state] as usize;
        let dead = quotient.class_for_state[self.dead_state()] as usize;
        if mapped_class[root] != root as u32 || mapped_class[dead] != dead as u32 {
            return false;
        }
        let Some(mapped_output_pair) = self.mapped_output_pair_ids_for_terminal_permutation(
            member_index_by_terminal,
            members,
            permutation,
        ) else {
            return false;
        };
        let cone = self.support_quotient_group_affected_cone(quotient, members);
        let mut in_cone = vec![false; class_count];
        for &class in &cone {
            in_cone[class] = true;
        }
        for &class in &cone {
            if !in_cone[mapped_class[class] as usize] {
                return false;
            }
        }
        for &class in &cone {
            let target_class = mapped_class[class] as usize;
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target_class] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination))) if source_byte == target_byte => {
                        source_index += 1;
                        target_index += 1;
                        (source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _))) if source_byte < target_byte => {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (Some(&(_, source_destination)), None) => {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("nonempty sparse edge union loop"),
                };
                let source_class = quotient.class_for_state[source_destination] as usize;
                let target_class = quotient.class_for_state[target_destination] as usize;
                if mapped_class[source_class] != target_class as u32 {
                    return false;
                }
                let source_pair = self.output_pair_by_state[source_destination] as usize;
                let target_pair = self.output_pair_by_state[target_destination] as usize;
                if mapped_output_pair[source_pair] != target_pair as u32 {
                    return false;
                }
            }
        }
        true
    }

    fn symmetric_support_orbit_witnesses(
        &mut self,
        members: &[TerminalID],
    ) -> Option<Vec<(TerminalID, InterchangeMap)>> {
        if members.len() < 3 {
            return None;
        }
        let profile_breakdown = std::env::var_os("GLRMASK_PROFILE_L2P_ORBIT_BREAKDOWN").is_some();
        let started_at = profile_breakdown.then(Instant::now);
        let tuples = self.uniform_support_fiber_tuples(members)?;
        let tuples_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let quotient = self.support_quotient.as_ref()?;
        let mut member_index_by_terminal = vec![usize::MAX; self.finalizer_states_by_terminal.len()];
        for (member, &terminal) in members.iter().enumerate() {
            member_index_by_terminal[terminal as usize] = member;
        }
        let cycle = (0..members.len())
            .map(|member| (member + 1) % members.len())
            .collect::<Vec<_>>();
        let mut adjacent = (0..members.len()).collect::<Vec<_>>();
        adjacent.swap(0, 1);
        let cycle_deviations = Self::deviations_for_terminal_permutation(&tuples, &cycle)?;
        let adjacent_deviations = Self::deviations_for_terminal_permutation(&tuples, &adjacent)?;
        if std::env::var_os("GLRMASK_PROFILE_L2P_ORBIT_CONE_DIAGNOSTIC").is_some() {
            let cone = self.support_quotient_group_affected_cone(quotient, members);
            eprintln!(
                "[glrmask/profile][ti_orbit_cone] representative={} members={} quotient_classes={} cone_classes={}",
                members[0],
                members.len(),
                quotient.representative_by_class.len(),
                cone.len(),
            );
        }
        let use_dense_orbit = std::env::var_os("GLRMASK_PROFILE_L2P_DENSE_ORBIT").is_some();
        let use_fast_orbit = std::env::var_os("GLRMASK_PROFILE_L2P_FAST_ORBIT").is_some();
        let use_cached_orbit = std::env::var_os("GLRMASK_PROFILE_L2P_CACHED_ORBIT").is_some();
        let cone = (use_cached_orbit || use_dense_orbit)
            .then(|| self.support_quotient_group_affected_cone(quotient, members));
        let cycle_ok = if use_dense_orbit {
            self.support_quotient_permutation_is_automorphism_dense_outputs(
                quotient,
                cone.as_deref().expect("dense orbit cone initialized"),
                &cycle_deviations,
                members,
                &member_index_by_terminal,
                &cycle,
            )
        } else if let Some(cone) = cone.as_deref() {
            self.support_quotient_permutation_is_automorphism_cached(
                quotient,
                cone,
                &cycle_deviations,
                members,
                &member_index_by_terminal,
                &cycle,
            )
        } else if use_fast_orbit {
            self.support_quotient_permutation_is_automorphism_fast(
                quotient,
                &cycle_deviations,
                members,
                &member_index_by_terminal,
                &cycle,
            )
        } else {
            self.support_quotient_permutation_is_automorphism(
                quotient,
                &cycle_deviations,
                members,
                &member_index_by_terminal,
                &cycle,
            )
        };
        let adjacent_ok = if use_dense_orbit {
            self.support_quotient_permutation_is_automorphism_dense_outputs(
                quotient,
                cone.as_deref().expect("dense orbit cone initialized"),
                &adjacent_deviations,
                members,
                &member_index_by_terminal,
                &adjacent,
            )
        } else if let Some(cone) = cone.as_deref() {
            self.support_quotient_permutation_is_automorphism_cached(
                quotient,
                cone,
                &adjacent_deviations,
                members,
                &member_index_by_terminal,
                &adjacent,
            )
        } else if use_fast_orbit {
            self.support_quotient_permutation_is_automorphism_fast(
                quotient,
                &adjacent_deviations,
                members,
                &member_index_by_terminal,
                &adjacent,
            )
        } else {
            self.support_quotient_permutation_is_automorphism(
                quotient,
                &adjacent_deviations,
                members,
                &member_index_by_terminal,
                &adjacent,
            )
        };
        let verify_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - tuples_ms)
            .unwrap_or(0.0);
        if std::env::var_os("GLRMASK_PROFILE_L2P_ORBIT_REJECTION_DIAGNOSTIC").is_some()
            && (!cycle_ok || !adjacent_ok)
        {
            eprintln!(
                "[glrmask/profile][ti_orbit_reject] representative={} members={} cycle_ok={} adjacent_ok={}",
                members[0],
                members.len(),
                cycle_ok,
                adjacent_ok,
            );
        }
        if !cycle_ok || !adjacent_ok {
            return None;
        }
        let mut witnesses = Vec::with_capacity(members.len().saturating_sub(1));
        for member in 1..members.len() {
            let deviations = Self::deviations_for_representative_swap(&tuples, member)?;
            witnesses.push((
                members[member],
                InterchangeMap {
                    scanner_state_map: self.quotient_transport_state_map(
                        Arc::clone(&quotient.class_for_state),
                        Arc::clone(&quotient.representative_by_class),
                        deviations.into_boxed_slice(),
                    ),
                },
            ));
        }
        Some(witnesses)
    }

    /// Split a broad literal bucket into candidate support fibers around a
    /// succession of pivots. The support-difference key is only a nomination
    /// device: every returned class is accepted solely after the complete
    /// symmetric-orbit automorphism proof.
    /// Split a lexical family by exact support-mask cardinalities. The full
    /// quotient orbit verifier remains the acceptance condition.
    fn external_output_signature(
        &self,
        terminal: TerminalID,
        bucket_member: &[bool],
    ) -> ExternalOutputSignature {
        let mut entries = Vec::new();
        for &pair_id in &self.observed_output_pair_ids_by_terminal[terminal as usize] {
            let pair = &self.output_pairs[pair_id as usize];
            let mut role = 0u8;
            role |= pair
                .finalizers
                .0
                .binary_search(&terminal)
                .is_ok()
                .then_some(1)
                .unwrap_or(0);
            role |= pair
                .future_finalizers
                .0
                .binary_search(&terminal)
                .is_ok()
                .then_some(2)
                .unwrap_or(0);
            debug_assert_ne!(role, 0);
            let finalizers = pair
                .finalizers
                .0
                .iter()
                .copied()
                .filter(|&other| !bucket_member[other as usize])
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let future_finalizers = pair
                .future_finalizers
                .0
                .iter()
                .copied()
                .filter(|&other| !bucket_member[other as usize])
                .collect::<Vec<_>>()
                .into_boxed_slice();
            entries.push((role, finalizers, future_finalizers));
        }
        entries.sort_unstable();
        entries.dedup();
        ExternalOutputSignature {
            entries: entries.into_boxed_slice(),
        }
    }

    fn refine_literal_groups_by_external_outputs(
        &self,
        groups: Vec<Vec<TerminalID>>,
    ) -> Vec<Vec<TerminalID>> {
        let terminal_count = self.finalizer_states_by_terminal.len();
        let mut refined = Vec::new();
        for group in groups {
            let mut bucket_member = vec![false; terminal_count];
            for &terminal in &group {
                bucket_member[terminal as usize] = true;
            }
            let mut buckets = BTreeMap::<ExternalOutputSignature, Vec<TerminalID>>::new();
            for &terminal in &group {
                buckets
                    .entry(self.external_output_signature(terminal, &bucket_member))
                    .or_default()
                    .push(terminal);
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_EXTERNAL_CONTEXT_DIAGNOSTIC").is_some() {
                let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
                sizes.sort_unstable_by(|left, right| right.cmp(left));
                eprintln!(
                    "[glrmask/profile][ti_literal_external_context] representative={} members={} bucket_sizes={:?}",
                    group[0],
                    group.len(),
                    sizes,
                );
            }
            refined.extend(buckets.into_values());
        }
        refined
    }

    /// Refine broad terminal candidates using 1-WL on the combined restricted
    /// scanner/output-incidence graph.  Terminals outside each input group are
    /// fixed by unique initial colors; terminals inside begin with one color.
    /// Every exact TI transport preserves every refinement round, so distinct
    /// terminal colors are a sound rejection condition.  Equal colors remain
    /// candidates and still require the ordinary exact transport proof.
    /// 1-WL on the stable canonical quotient and its byte-labelled
    /// destination-output incidence.  Unlike raw-state refinement, this uses
    /// exactly the quotient on which every generic TI witness is a class
    /// permutation. Terminals outside an input group are fixed individually;
    /// group terminals initially share one color. Distinct final terminal
    /// colors therefore rule out a TI swap, while equal colors remain only a
    /// candidate for the unchanged exact proof.
    /// One global quotient-WL refinement. Root-output candidate groups seed
    /// terminal colors; singleton/noncandidate terminals remain fixed unique
    /// colors. A valid TI swap preserves this initial coloring, so terminal
    /// color splits are exact necessary conditions. Unlike the per-group form,
    /// this refines the canonical quotient once for the entire partition.
    /// Allocation-free conservative global quotient refinement. Each round is
    /// an isomorphism invariant hash of the exact quotient-WL tuple. Hash
    /// collisions can only leave non-equivalent terminals together; they can
    /// never split a valid TI class. A small fixed round count avoids needing
    /// canonical color renumbering while retaining the useful deep structure.
    fn refine_candidate_groups_by_global_quotient_hash_wl(
        &mut self,
        groups: Vec<Vec<TerminalID>>,
    ) -> Vec<Vec<TerminalID>> {
        let profile_breakdown = std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_HASH_BREAKDOWN").is_some();
        let started_at = profile_breakdown.then(Instant::now);
        let rounds = std::env::var("GLRMASK_PROFILE_L2P_GLOBAL_HASH_WL_ROUNDS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&rounds| rounds > 0)
            .unwrap_or(1);
        let terminal_count = self.finalizer_states_by_terminal.len();
        self.ensure_canonical_quotient();
        let quotient_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        let class_count = quotient.representative_by_class.len();
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let dead_class = quotient.class_for_state[self.dead_state()] as usize;
        let mut group_for_terminal = vec![usize::MAX; terminal_count];
        let mut mutable_terminal = vec![false; terminal_count];
        for (group_index, group) in groups.iter().enumerate() {
            for &terminal in group {
                let terminal = terminal as usize;
                group_for_terminal[terminal] = group_index;
                mutable_terminal[terminal] = group.len() >= 2;
            }
        }
        let mut terminal_colors = (0..terminal_count)
            .map(|terminal| {
                let seed = if mutable_terminal[terminal] {
                    0x9ec0_4f31_7a3d_2119u64
                } else {
                    0x2c1b_3c6d_4e5f_7081u64
                };
                let component = if mutable_terminal[terminal] {
                    group_for_terminal[terminal] as u64
                } else {
                    terminal as u64
                };
                mix_structural_fingerprint(seed, component)
            })
            .collect::<Vec<_>>();
        let mut class_colors = (0..class_count)
            .map(|class| {
                mix_structural_fingerprint(
                    0x517c_c1b7_2722_0a95,
                    if class == root_class { 1 } else if class == dead_class { 2 } else { 0 },
                )
            })
            .collect::<Vec<_>>();
        let output_hash = |bits: &OutputBits, terminal_colors: &[u64]| {
            let mut sum = 0u64;
            let mut xor = 0u64;
            let mut sum_rot = 0u64;
            for &terminal in &bits.0 {
                let color = terminal_colors[terminal as usize];
                sum = sum.wrapping_add(color);
                xor ^= color;
                sum_rot = sum_rot.wrapping_add(color.rotate_left(23));
            }
            let mut hash = mix_structural_fingerprint(0x7f4a_7c15_9e37_79b9, bits.0.len() as u64);
            hash = mix_structural_fingerprint(hash, sum);
            hash = mix_structural_fingerprint(hash, xor);
            mix_structural_fingerprint(hash, sum_rot)
        };

        for _ in 0..rounds {
            let round_started_at = profile_breakdown.then(Instant::now);
            let mut finalizer_hash_by_output_pair = Vec::with_capacity(self.output_pairs.len());
            let mut future_hash_by_output_pair = Vec::with_capacity(self.output_pairs.len());
            let mut mutable_members_by_output_pair =
                Vec::<SmallVec<[(usize, u8); 4]>>::with_capacity(self.output_pairs.len());
            for output in &self.output_pairs {
                finalizer_hash_by_output_pair.push(output_hash(&output.finalizers, &terminal_colors));
                future_hash_by_output_pair.push(output_hash(&output.future_finalizers, &terminal_colors));
                let mut members = SmallVec::<[(usize, u8); 4]>::new();
                let mut finalizer_index = 0usize;
                let mut future_index = 0usize;
                while finalizer_index < output.finalizers.0.len()
                    || future_index < output.future_finalizers.0.len()
                {
                    let (terminal, roles) = match (
                        output.finalizers.0.get(finalizer_index),
                        output.future_finalizers.0.get(future_index),
                    ) {
                        (Some(&finalizer), Some(&future)) if finalizer == future => {
                            finalizer_index += 1;
                            future_index += 1;
                            (finalizer as usize, 3u8)
                        }
                        (Some(&finalizer), Some(&future)) if finalizer < future => {
                            finalizer_index += 1;
                            (finalizer as usize, 1u8)
                        }
                        (Some(_), Some(&future)) => {
                            future_index += 1;
                            (future as usize, 2u8)
                        }
                        (Some(&finalizer), None) => {
                            finalizer_index += 1;
                            (finalizer as usize, 1u8)
                        }
                        (None, Some(&future)) => {
                            future_index += 1;
                            (future as usize, 2u8)
                        }
                        (None, None) => unreachable!("nonempty frozen output merge"),
                    };
                    if mutable_terminal[terminal] {
                        members.push((terminal, roles));
                    }
                }
                mutable_members_by_output_pair.push(members);
            }
            let output_cache_ms = round_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let mut next_class_colors = Vec::with_capacity(class_count);
            for class in 0..class_count {
                let state = quotient.representative_by_class[class] as usize;
                let mut hash = mix_structural_fingerprint(0xd1b5_4a32_d192_ed03, class_colors[class]);
                hash = mix_structural_fingerprint(hash, self.topology.edges_from(state).len() as u64);
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination = destination as usize;
                    let destination_class = quotient.class_for_state[destination] as usize;
                    let output_pair = self.output_pair_by_state[destination] as usize;
                    hash = mix_structural_fingerprint(hash, byte as u64);
                    hash = mix_structural_fingerprint(hash, class_colors[destination_class]);
                    hash = mix_structural_fingerprint(hash, finalizer_hash_by_output_pair[output_pair]);
                    hash = mix_structural_fingerprint(hash, future_hash_by_output_pair[output_pair]);
                }
                next_class_colors.push(hash);
            }
            let class_pass_ms = round_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - output_cache_ms)
                .unwrap_or(0.0);
            let mut counts = vec![0u32; terminal_count];
            let mut sums = vec![0u64; terminal_count];
            let mut xors = vec![0u64; terminal_count];
            let mut sums_rot = vec![0u64; terminal_count];
            for class in 0..class_count {
                let state = quotient.representative_by_class[class] as usize;
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination = destination as usize;
                    let destination_class = quotient.class_for_state[destination] as usize;
                    let output_pair = self.output_pair_by_state[destination] as usize;
                    for &(terminal, roles) in &mutable_members_by_output_pair[output_pair] {
                        for (bit, role) in [(1u8, 1u64), (2u8, 2u64)] {
                            if roles & bit == 0 {
                                continue;
                            }
                            let mut relation = mix_structural_fingerprint(
                                0x94d0_49bb_1331_11ebu64,
                                next_class_colors[class],
                            );
                            relation = mix_structural_fingerprint(relation, byte as u64);
                            relation = mix_structural_fingerprint(relation, next_class_colors[destination_class]);
                            relation = mix_structural_fingerprint(relation, role);
                            counts[terminal] += 1;
                            sums[terminal] = sums[terminal].wrapping_add(relation);
                            xors[terminal] ^= relation.rotate_left((byte & 63) as u32);
                            sums_rot[terminal] = sums_rot[terminal].wrapping_add(relation.rotate_left(29));
                        }
                    }
                }
            }
            let incidence_ms = round_started_at
                .map(|started_at| {
                    started_at.elapsed().as_secs_f64() * 1000.0 - output_cache_ms - class_pass_ms
                })
                .unwrap_or(0.0);
            let mut next_terminal_colors = terminal_colors.clone();
            for terminal in 0..terminal_count {
                if !mutable_terminal[terminal] {
                    continue;
                }
                let mut hash = mix_structural_fingerprint(0x243f_6a88_85a3_08d3, terminal_colors[terminal]);
                hash = mix_structural_fingerprint(hash, counts[terminal] as u64);
                hash = mix_structural_fingerprint(hash, sums[terminal]);
                hash = mix_structural_fingerprint(hash, xors[terminal]);
                next_terminal_colors[terminal] = mix_structural_fingerprint(hash, sums_rot[terminal]);
            }
            class_colors = next_class_colors;
            terminal_colors = next_terminal_colors;
            if let Some(started_at) = round_started_at {
                eprintln!(
                    "[glrmask/profile][ti_global_hash_round] output_cache_ms={:.3} class_pass_ms={:.3} incidence_ms={:.3} terminal_color_ms={:.3} total_ms={:.3}",
                    output_cache_ms,
                    class_pass_ms,
                    incidence_ms,
                    started_at.elapsed().as_secs_f64() * 1000.0 - output_cache_ms - class_pass_ms - incidence_ms,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }

        let mut refined = Vec::new();
        for group in groups {
            if group.len() < 2 {
                refined.push(group);
                continue;
            }
            let mut buckets = BTreeMap::<u64, Vec<TerminalID>>::new();
            for terminal in group {
                buckets.entry(terminal_colors[terminal as usize]).or_default().push(terminal);
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_HASH_WL_DIAGNOSTIC").is_some() {
                let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
                sizes.sort_unstable_by(|left, right| right.cmp(left));
                if sizes.len() > 1 {
                    eprintln!(
                        "[glrmask/profile][ti_global_hash_wl] representative={} members={} bucket_sizes={:?}",
                        buckets.values().next().and_then(|bucket| bucket.first()).copied().unwrap_or_default(),
                        sizes.iter().sum::<usize>(),
                        sizes,
                    );
                }
            }
            refined.extend(buckets.into_values());
        }
        if let Some(started_at) = started_at {
            eprintln!(
                "[glrmask/profile][ti_global_hash_total] quotient_ms={:.3} hash_ms={:.3} total_ms={:.3}",
                quotient_ms,
                started_at.elapsed().as_secs_f64() * 1000.0 - quotient_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        refined
    }

    fn refine_candidate_groups_by_global_quotient_wl(
        &mut self,
        groups: Vec<Vec<TerminalID>>,
    ) -> Vec<Vec<TerminalID>> {
        const MAX_ROUNDS: usize = 64;
        let terminal_count = self.finalizer_states_by_terminal.len();
        self.ensure_canonical_quotient();
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        let class_count = quotient.representative_by_class.len();
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let dead_class = quotient.class_for_state[self.dead_state()] as usize;

        let mut group_for_terminal = vec![usize::MAX; terminal_count];
        let mut mutable_terminal = vec![false; terminal_count];
        for (group_index, group) in groups.iter().enumerate() {
            for &terminal in group {
                let terminal = terminal as usize;
                group_for_terminal[terminal] = group_index;
                mutable_terminal[terminal] = group.len() >= 2;
            }
        }
        let candidate_color_base = terminal_count as u32 + 1;
        let mut terminal_colors = (0..terminal_count)
            .map(|terminal| match group_for_terminal[terminal] {
                group_index if mutable_terminal[terminal] => {
                    candidate_color_base + group_index as u32
                }
                _ => terminal as u32 + 1,
            })
            .collect::<Vec<_>>();
        let mut class_colors = (0..class_count)
            .map(|class| {
                if class == root_class {
                    1u32
                } else if class == dead_class {
                    2u32
                } else {
                    0u32
                }
            })
            .collect::<Vec<_>>();

        for _ in 0..MAX_ROUNDS {
            let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();
            let mut next_class_colors = Vec::with_capacity(class_count);
            for class in 0..class_count {
                let state = quotient.representative_by_class[class] as usize;
                let mut signature = Vec::<u32>::new();
                signature.push(class_colors[class]);
                signature.push(self.topology.edges_from(state).len() as u32);
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination = destination as usize;
                    let destination_class = quotient.class_for_state[destination] as usize;
                    signature.push(byte as u32);
                    signature.push(class_colors[destination_class]);
                    let output = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                    let mut finalizer_colors = output
                        .finalizers
                        .0
                        .iter()
                        .map(|&terminal| terminal_colors[terminal as usize])
                        .collect::<SmallVec<[u32; 4]>>();
                    finalizer_colors.sort_unstable();
                    signature.push(finalizer_colors.len() as u32);
                    signature.extend(finalizer_colors);
                    let mut future_colors = output
                        .future_finalizers
                        .0
                        .iter()
                        .map(|&terminal| terminal_colors[terminal as usize])
                        .collect::<SmallVec<[u32; 4]>>();
                    future_colors.sort_unstable();
                    signature.push(future_colors.len() as u32);
                    signature.extend(future_colors);
                }
                let next = class_ids.len() as u32;
                next_class_colors.push(*class_ids.entry(signature).or_insert(next));
            }

            let mut terminal_incidence = vec![Vec::<[u32; 4]>::new(); terminal_count];
            for class in 0..class_count {
                let state = quotient.representative_by_class[class] as usize;
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination = destination as usize;
                    let destination_class = quotient.class_for_state[destination] as usize;
                    let output = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                    for &terminal in &output.finalizers.0 {
                        let terminal = terminal as usize;
                        if mutable_terminal[terminal] {
                            terminal_incidence[terminal].push([
                                next_class_colors[class],
                                byte as u32,
                                next_class_colors[destination_class],
                                1,
                            ]);
                        }
                    }
                    for &terminal in &output.future_finalizers.0 {
                        let terminal = terminal as usize;
                        if mutable_terminal[terminal] {
                            terminal_incidence[terminal].push([
                                next_class_colors[class],
                                byte as u32,
                                next_class_colors[destination_class],
                                2,
                            ]);
                        }
                    }
                }
            }
            let mut terminal_ids = FxHashMap::<Vec<u32>, u32>::default();
            let mut next_terminal_colors = terminal_colors.clone();
            for terminal in 0..terminal_count {
                if !mutable_terminal[terminal] {
                    continue;
                }
                let incidence = &mut terminal_incidence[terminal];
                incidence.sort_unstable();
                let mut signature = Vec::<u32>::with_capacity(1 + incidence.len() * 4);
                signature.push(terminal_colors[terminal]);
                for relation in incidence.iter() {
                    signature.extend(relation);
                }
                let next = candidate_color_base + terminal_ids.len() as u32;
                next_terminal_colors[terminal] = *terminal_ids.entry(signature).or_insert(next);
            }
            if next_class_colors == class_colors && next_terminal_colors == terminal_colors {
                break;
            }
            class_colors = next_class_colors;
            terminal_colors = next_terminal_colors;
        }

        let mut refined = Vec::new();
        for group in groups {
            if group.len() < 2 {
                refined.push(group);
                continue;
            }
            let mut buckets = BTreeMap::<u32, Vec<TerminalID>>::new();
            for terminal in group {
                buckets
                    .entry(terminal_colors[terminal as usize])
                    .or_default()
                    .push(terminal);
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_QUOTIENT_WL_DIAGNOSTIC").is_some() {
                let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
                sizes.sort_unstable_by(|left, right| right.cmp(left));
                if sizes.len() > 1 {
                    eprintln!(
                        "[glrmask/profile][ti_global_quotient_wl] representative={} members={} bucket_sizes={:?}",
                        buckets.values().next().and_then(|bucket| bucket.first()).copied().unwrap_or_default(),
                        sizes.iter().sum::<usize>(),
                        sizes,
                    );
                }
            }
            refined.extend(buckets.into_values());
        }
        refined
    }

    fn refine_candidate_groups_by_quotient_wl(
        &mut self,
        groups: Vec<Vec<TerminalID>>,
    ) -> Vec<Vec<TerminalID>> {
        const MIN_GROUP_SIZE: usize = 32;
        const MAX_ROUNDS: usize = 64;
        let terminal_count = self.finalizer_states_by_terminal.len();
        let mut refined = Vec::new();
        self.ensure_canonical_quotient();
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        let class_count = quotient.representative_by_class.len();
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let dead_class = quotient.class_for_state[self.dead_state()] as usize;

        for group in groups {
            if group.len() < MIN_GROUP_SIZE {
                refined.push(group);
                continue;
            }
            let mut member = vec![false; terminal_count];
            for &terminal in &group {
                member[terminal as usize] = true;
            }
            let candidate_color_base = terminal_count as u32 + 1;
            let mut terminal_colors = (0..terminal_count)
                .map(|terminal| {
                    if member[terminal] {
                        candidate_color_base
                    } else {
                        terminal as u32 + 1
                    }
                })
                .collect::<Vec<_>>();
            let mut class_colors = (0..class_count)
                .map(|class| {
                    if class == root_class {
                        1u32
                    } else if class == dead_class {
                        2u32
                    } else {
                        0u32
                    }
                })
                .collect::<Vec<_>>();

            for _ in 0..MAX_ROUNDS {
                let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();
                let mut next_class_colors = Vec::with_capacity(class_count);
                for class in 0..class_count {
                    let state = quotient.representative_by_class[class] as usize;
                    let mut signature = Vec::<u32>::new();
                    signature.push(class_colors[class]);
                    signature.push(self.topology.edges_from(state).len() as u32);
                    for &(byte, destination) in self.topology.edges_from(state) {
                        let destination = destination as usize;
                        let destination_class = quotient.class_for_state[destination] as usize;
                        signature.push(byte as u32);
                        signature.push(class_colors[destination_class]);
                        let output = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                        let mut finalizer_colors = output
                            .finalizers
                            .0
                            .iter()
                            .map(|&terminal| terminal_colors[terminal as usize])
                            .collect::<SmallVec<[u32; 4]>>();
                        finalizer_colors.sort_unstable();
                        signature.push(finalizer_colors.len() as u32);
                        signature.extend(finalizer_colors);
                        let mut future_colors = output
                            .future_finalizers
                            .0
                            .iter()
                            .map(|&terminal| terminal_colors[terminal as usize])
                            .collect::<SmallVec<[u32; 4]>>();
                        future_colors.sort_unstable();
                        signature.push(future_colors.len() as u32);
                        signature.extend(future_colors);
                    }
                    let next = class_ids.len() as u32;
                    next_class_colors.push(*class_ids.entry(signature).or_insert(next));
                }

                let mut terminal_incidence = vec![Vec::<[u32; 4]>::new(); terminal_count];
                for class in 0..class_count {
                    let state = quotient.representative_by_class[class] as usize;
                    for &(byte, destination) in self.topology.edges_from(state) {
                        let destination = destination as usize;
                        let destination_class = quotient.class_for_state[destination] as usize;
                        let output = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                        for &terminal in &output.finalizers.0 {
                            if member[terminal as usize] {
                                terminal_incidence[terminal as usize].push([
                                    next_class_colors[class],
                                    byte as u32,
                                    next_class_colors[destination_class],
                                    1,
                                ]);
                            }
                        }
                        for &terminal in &output.future_finalizers.0 {
                            if member[terminal as usize] {
                                terminal_incidence[terminal as usize].push([
                                    next_class_colors[class],
                                    byte as u32,
                                    next_class_colors[destination_class],
                                    2,
                                ]);
                            }
                        }
                    }
                }
                let mut terminal_ids = FxHashMap::<Vec<u32>, u32>::default();
                let mut next_terminal_colors = terminal_colors.clone();
                for &terminal in &group {
                    let terminal = terminal as usize;
                    let incidence = &mut terminal_incidence[terminal];
                    incidence.sort_unstable();
                    let mut signature = Vec::<u32>::with_capacity(1 + incidence.len() * 4);
                    signature.push(terminal_colors[terminal]);
                    for relation in incidence.iter() {
                        signature.extend(relation);
                    }
                    let next = candidate_color_base + terminal_ids.len() as u32;
                    next_terminal_colors[terminal] =
                        *terminal_ids.entry(signature).or_insert(next);
                }
                if next_class_colors == class_colors && next_terminal_colors == terminal_colors {
                    break;
                }
                class_colors = next_class_colors;
                terminal_colors = next_terminal_colors;
            }

            let mut buckets = BTreeMap::<u32, Vec<TerminalID>>::new();
            for &terminal in &group {
                buckets
                    .entry(terminal_colors[terminal as usize])
                    .or_default()
                    .push(terminal);
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_QUOTIENT_WL_DIAGNOSTIC").is_some() {
                let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
                sizes.sort_unstable_by(|left, right| right.cmp(left));
                eprintln!(
                    "[glrmask/profile][ti_quotient_wl] representative={} members={} bucket_sizes={:?}",
                    group[0],
                    group.len(),
                    sizes,
                );
            }
            refined.extend(buckets.into_values());
        }
        refined
    }

    fn refine_candidate_groups_by_joint_wl(
        &self,
        groups: Vec<Vec<TerminalID>>,
    ) -> Vec<Vec<TerminalID>> {
        const MIN_GROUP_SIZE: usize = 16;
        const MAX_ROUNDS: usize = 64;
        let terminal_count = self.finalizer_states_by_terminal.len();
        let state_count = self.topology.state_count();
        let root = self.topology.initial_state;
        let dead = self.dead_state();
        let mut refined = Vec::new();

        for group in groups {
            if group.len() < MIN_GROUP_SIZE {
                refined.push(group);
                continue;
            }
            let mut member = vec![false; terminal_count];
            for &terminal in &group {
                member[terminal as usize] = true;
            }
            let candidate_color_base = terminal_count as u32 + 1;
            let mut terminal_colors = (0..terminal_count)
                .map(|terminal| {
                    if member[terminal] {
                        candidate_color_base
                    } else {
                        terminal as u32 + 1
                    }
                })
                .collect::<Vec<_>>();
            let mut state_colors = (0..state_count)
                .map(|state| {
                    if state == root {
                        1u32
                    } else if state == dead {
                        2u32
                    } else {
                        0u32
                    }
                })
                .collect::<Vec<_>>();

            for _ in 0..MAX_ROUNDS {
                let mut state_ids = FxHashMap::<Vec<u32>, u32>::default();
                let mut next_state_colors = Vec::with_capacity(state_count);
                for state in 0..state_count {
                    let mut signature = Vec::<u32>::new();
                    signature.push(state_colors[state]);
                    signature.push(self.topology.edges_from(state).len() as u32);
                    for &(byte, destination) in self.topology.edges_from(state) {
                        signature.push(byte as u32);
                        signature.push(state_colors[destination as usize]);
                    }
                    let finalizers = self.finalizers.get(state).unwrap_or(&self.empty_output);
                    signature.push(finalizers.0.len() as u32);
                    for &terminal in &finalizers.0 {
                        signature.push(terminal_colors[terminal as usize]);
                    }
                    let futures = self.future_finalizers.get(state).unwrap_or(&self.empty_output);
                    signature.push(futures.0.len() as u32);
                    for &terminal in &futures.0 {
                        signature.push(terminal_colors[terminal as usize]);
                    }
                    let next = state_ids.len() as u32;
                    next_state_colors.push(*state_ids.entry(signature).or_insert(next));
                }

                let mut terminal_ids = FxHashMap::<Vec<u32>, u32>::default();
                let mut next_terminal_colors = terminal_colors.clone();
                for &terminal in &group {
                    let terminal_index = terminal as usize;
                    let mut signature = Vec::<u32>::new();
                    signature.push(terminal_colors[terminal_index]);
                    let finalizer_states = &self.finalizer_states_by_terminal[terminal_index];
                    signature.push(finalizer_states.len() as u32);
                    let mut finalizer_colors = finalizer_states
                        .iter()
                        .map(|&state| next_state_colors[state as usize])
                        .collect::<Vec<_>>();
                    finalizer_colors.sort_unstable();
                    signature.extend(finalizer_colors);
                    let future_states = &self.future_finalizer_states_by_terminal[terminal_index];
                    signature.push(future_states.len() as u32);
                    let mut future_colors = future_states
                        .iter()
                        .map(|&state| next_state_colors[state as usize])
                        .collect::<Vec<_>>();
                    future_colors.sort_unstable();
                    signature.extend(future_colors);
                    let next = candidate_color_base + terminal_ids.len() as u32;
                    next_terminal_colors[terminal_index] =
                        *terminal_ids.entry(signature).or_insert(next);
                }
                if next_state_colors == state_colors && next_terminal_colors == terminal_colors {
                    break;
                }
                state_colors = next_state_colors;
                terminal_colors = next_terminal_colors;
            }

            let mut buckets = BTreeMap::<u32, Vec<TerminalID>>::new();
            for &terminal in &group {
                buckets
                    .entry(terminal_colors[terminal as usize])
                    .or_default()
                    .push(terminal);
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_JOINT_WL_DIAGNOSTIC").is_some() {
                let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
                sizes.sort_unstable_by(|left, right| right.cmp(left));
                eprintln!(
                    "[glrmask/profile][ti_joint_wl] representative={} members={} bucket_sizes={:?}",
                    group[0],
                    group.len(),
                    sizes,
                );
            }
            refined.extend(buckets.into_values());
        }
        refined
    }

    fn refine_literal_groups_by_external_quotient(
        &mut self,
        groups: Vec<Vec<TerminalID>>,
    ) -> Vec<Vec<TerminalID>> {
        let terminal_count = self.finalizer_states_by_terminal.len();
        let mut refined = Vec::new();
        for group in groups {
            if group.len() < 2 {
                refined.push(group);
                continue;
            }
            self.ensure_support_quotient();
            let quotient = self
                .support_quotient
                .as_ref()
                .expect("support quotient initialized");
            let class_count = quotient.representative_by_class.len();
            let dead_class = quotient.class_for_state[self.dead_state()] as usize;
            let mut bucket_member = vec![false; terminal_count];
            for &terminal in &group {
                bucket_member[terminal as usize] = true;
            }
            let compact_colors = |hashes: Vec<u64>| {
                let mut class_for_hash = FxHashMap::<u64, u64>::default();
                hashes
                    .into_iter()
                    .map(|hash| {
                        let next = class_for_hash.len() as u64;
                        *class_for_hash.entry(hash).or_insert(next)
                    })
                    .collect::<Vec<_>>()
            };
            let mut initial_hashes = Vec::with_capacity(class_count);
            for class in 0..class_count {
                let state = quotient.representative_by_class[class] as usize;
                let mut color = 0x6a09_e667_f3bc_c909u64;
                if !quotient.reverse_predecessors[class].is_empty() {
                    let pair = &self.output_pairs[self.output_pair_by_state[state] as usize];
                    color = mix_structural_fingerprint(color, 0xf1);
                    for &terminal in &pair.finalizers.0 {
                        if !bucket_member[terminal as usize] {
                            color = mix_structural_fingerprint(color, terminal as u64);
                        }
                    }
                    color = mix_structural_fingerprint(color, 0xf2);
                    for &terminal in &pair.future_finalizers.0 {
                        if !bucket_member[terminal as usize] {
                            color = mix_structural_fingerprint(color, terminal as u64);
                        }
                    }
                }
                initial_hashes.push(color);
            }
            let mut colors = compact_colors(initial_hashes);
            for _ in 0..64 {
                let default = colors[dead_class];
                let mut next_hashes = Vec::with_capacity(class_count);
                for class in 0..class_count {
                    let state = quotient.representative_by_class[class] as usize;
                    let mut color = mix_structural_fingerprint(0xbb67_ae85_84ca_a73b, colors[class]);
                    for &(byte, destination) in self.topology.edges_from(state) {
                        let destination_class = quotient.class_for_state[destination as usize] as usize;
                        if colors[destination_class] == default {
                            continue;
                        }
                        color = mix_structural_fingerprint(color, byte as u64);
                        color = mix_structural_fingerprint(color, colors[destination_class]);
                    }
                    next_hashes.push(color);
                }
                let next = compact_colors(next_hashes);
                if next == colors {
                    break;
                }
                colors = next;
            }
            self.ensure_terminal_quotient_output_supports_bulk(&group);
            let supports = self
                .terminal_quotient_output_supports
                .as_ref()
                .expect("terminal support columns initialized");
            let mut buckets = BTreeMap::<Vec<(u8, u64)>, Vec<TerminalID>>::new();
            for &terminal in &group {
                let Some(support) = supports
                    .get(terminal as usize)
                    .and_then(Option::as_ref)
                else {
                    continue;
                };
                let mut signature = support
                    .iter()
                    .map(|&(class, mask)| (mask, colors[class as usize]))
                    .collect::<Vec<_>>();
                signature.sort_unstable();
                buckets.entry(signature).or_default().push(terminal);
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_EXTERNAL_QUOTIENT_DIAGNOSTIC").is_some() {
                let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
                sizes.sort_unstable_by(|left, right| right.cmp(left));
                eprintln!(
                    "[glrmask/profile][ti_literal_external_quotient] representative={} members={} bucket_sizes={:?}",
                    group[0],
                    group.len(),
                    sizes,
                );
            }
            refined.extend(buckets.into_values());
        }
        refined
    }

    fn literal_support_shape_orbit_witnesses(
        &mut self,
        candidates: &[TerminalID],
    ) -> Vec<(TerminalID, TerminalID, InterchangeMap)> {
        const MIN_ORBIT_MEMBERS: usize = 8;
        if candidates.len() < MIN_ORBIT_MEMBERS {
            return Vec::new();
        }
        self.ensure_support_quotient();
        self.ensure_terminal_erased_support_colors();
        self.ensure_terminal_quotient_output_supports_bulk(candidates);
        let colors = Arc::clone(
            self.terminal_erased_support_colors
                .as_ref()
                .expect("terminal-erased support colors initialized"),
        );
        let supports = self
            .terminal_quotient_output_supports
            .as_ref()
            .expect("terminal support columns initialized");
        let mut buckets = BTreeMap::<Vec<(u8, u64)>, Vec<TerminalID>>::new();
        for &terminal in candidates {
            let Some(support) = supports
                .get(terminal as usize)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let mut signature = support
                .iter()
                .map(|&(class, mask)| (mask, colors[class as usize]))
                .collect::<Vec<_>>();
            signature.sort_unstable();
            buckets.entry(signature).or_default().push(terminal);
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_LITERAL_SUPPORT_SHAPES").is_some() {
            let mut sizes = buckets
                .iter()
                .map(|(shape, members)| (shape.clone(), members.len()))
                .collect::<Vec<_>>();
            sizes.sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
            eprintln!(
                "[glrmask/profile][ti_literal_support_shapes] representative={} members={} buckets={:?}",
                candidates[0],
                candidates.len(),
                sizes,
            );
        }
        let mut witnesses = Vec::new();
        for group in buckets.into_values() {
            if group.len() < MIN_ORBIT_MEMBERS {
                continue;
            }
            let pivot_maps = self.support_orbit_first_bucket_witnesses(&group);
            if !pivot_maps.is_empty() {
                witnesses.extend(pivot_maps);
                continue;
            }
            let started_at = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some().then(Instant::now);
            let maps = self.symmetric_support_orbit_witnesses(&group);
            if let Some(started_at) = started_at {
                eprintln!(
                    "[glrmask/profile][ti_literal_orbit_attempt] representative={} members={} certified={} total_ms={:.3}",
                    group[0],
                    group.len(),
                    maps.is_some(),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let Some(maps) = maps else {
                continue;
            };
            for (member, map) in maps {
                witnesses.push((group[0], member, map));
            }
        }
        witnesses
    }

    /// One bounded support-fibre attempt for an already output-filtered
    /// candidate bucket. The support key only nominates a family; the orbit
    /// proof remains the acceptance condition.
    fn support_orbit_first_bucket_witnesses(
        &mut self,
        candidates: &[TerminalID],
    ) -> Vec<(TerminalID, TerminalID, InterchangeMap)> {
        const MIN_ORBIT_MEMBERS: usize = 8;
        if candidates.len() < MIN_ORBIT_MEMBERS {
            return Vec::new();
        }
        let representative = candidates[0];
        self.ensure_support_quotient();
        self.ensure_terminal_quotient_output_support(representative);
        let representative_support = match self
            .terminal_quotient_output_supports
            .as_ref()
            .and_then(|supports| supports.get(representative as usize))
            .and_then(Option::as_ref)
        {
            Some(support) => support.clone(),
            None => return Vec::new(),
        };
        let mut buckets = BTreeMap::<Vec<u32>, Vec<TerminalID>>::new();
        for &member in &candidates[1..] {
            self.ensure_terminal_quotient_output_support(member);
            let support = self
                .terminal_quotient_output_supports
                .as_ref()
                .and_then(|supports| supports.get(member as usize))
                .and_then(Option::as_ref);
            let Some(support) = support else {
                continue;
            };
            let Some((representative_only, _)) =
                Self::support_difference_by_mask(&representative_support, support)
            else {
                continue;
            };
            if !representative_only.is_empty() {
                buckets.entry(representative_only).or_default().push(member);
            }
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_PIVOT_BUCKETS").is_some() {
            let mut sizes = buckets.values().map(Vec::len).collect::<Vec<_>>();
            sizes.sort_unstable_by(|left, right| right.cmp(left));
            eprintln!(
                "[glrmask/profile][ti_pivot_buckets] representative={} candidates={} bucket_sizes={:?}",
                representative,
                candidates.len(),
                sizes,
            );
        }
        let Some((_, members)) = buckets.into_iter().max_by(|(left_key, left_members), (right_key, right_members)| {
            left_members
                .len()
                .cmp(&right_members.len())
                .then_with(|| right_key.cmp(left_key))
        }) else {
            return Vec::new();
        };
        if members.len() + 1 < MIN_ORBIT_MEMBERS {
            return Vec::new();
        }
        let mut family = Vec::with_capacity(members.len() + 1);
        family.push(representative);
        family.extend(members);
        let witnesses = self.symmetric_support_orbit_witnesses(&family);
        if std::env::var_os("GLRMASK_PROFILE_L2P_PIVOT_BUCKETS").is_some() {
            eprintln!(
                "[glrmask/profile][ti_pivot_orbit] representative={} members={} certified={}",
                representative,
                family.len(),
                witnesses.is_some(),
            );
        }
        let Some(witnesses) = witnesses else {
            return Vec::new();
        };
        witnesses
            .into_iter()
            .map(|(member, map)| (representative, member, map))
            .collect()
    }

    #[inline]
    fn literal_output_matches_after_swap(
        &self,
        source_state: usize,
        target_state: usize,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        let source = &self.output_pairs[self.output_pair_by_state[source_state] as usize];
        let target = &self.output_pairs[self.output_pair_by_state[target_state] as usize];
        let swap = Some((left as usize, right as usize));
        source.finalizers.mapped(swap) == target.finalizers
            && source.future_finalizers.mapped(swap) == target.future_finalizers
    }

    fn register_literal_fiber_coordinate(
        classes: Vec<u32>,
        dead_class: u32,
        coordinates: &mut Vec<Vec<u32>>,
        class_cell: &mut [Option<(usize, usize)>],
        coordinate_by_tuple: &mut FxHashMap<Vec<u32>, usize>,
        worklist: &mut VecDeque<usize>,
    ) -> Option<usize> {
        if classes.len() < 2 || classes.iter().any(|&class| class == dead_class) {
            return None;
        }
        if let Some(&coordinate) = coordinate_by_tuple.get(&classes) {
            return Some(coordinate);
        }
        let mut seen = FxHashSet::default();
        for &class in &classes {
            if !seen.insert(class) || class_cell.get(class as usize)?.is_some() {
                return None;
            }
        }
        let coordinate = coordinates.len();
        for (member, &class) in classes.iter().enumerate() {
            class_cell[class as usize] = Some((coordinate, member));
        }
        coordinate_by_tuple.insert(classes.clone(), coordinate);
        coordinates.push(classes);
        worklist.push_back(coordinate);
        Some(coordinate)
    }

    /// Exact literal-family fiber certificate on the stable output-aware
    /// quotient.  It has a fixed core and discovered terminal-indexed fiber
    /// tuples.  Every pivot/member transposition is proved in one shared
    /// closure rather than by independently refining every pair.
    fn literal_fiber_group_witnesses(
        &mut self,
        group: &[TerminalID],
    ) -> Option<Vec<(TerminalID, InterchangeMap)>> {
        let (&pivot, members) = group.split_first()?;
        if members.is_empty() {
            return None;
        }

        // If all frozen output columns already agree exactly, the ordinary
        // identity quotient is an exact witness for each literal member.
        if members
            .iter()
            .copied()
            .all(|member| self.swap_preserves_all_frozen_outputs(pivot, member))
        {
            let map = self.canonical_identity_map();
            return Some(
                members
                    .iter()
                    .copied()
                    .map(|member| (member, map.clone()))
                    .collect(),
            );
        }

        self.ensure_support_quotient();
        for &terminal in group {
            self.ensure_terminal_quotient_output_support(terminal);
        }
        let supports = {
            let supports = self.terminal_quotient_output_supports.as_ref()?;
            group
                .iter()
                .map(|&terminal| supports.get(terminal as usize)?.as_ref().cloned())
                .collect::<Option<Vec<_>>>()?
        };
        let quotient = self.support_quotient.as_ref()?;
        let class_count = quotient.representative_by_class.len();
        let dead_class = quotient.class_for_state[self.dead_state()];

        // Support tracks nominate only an initial fiber alignment. The full
        // quotient transition/output proof below is the acceptance condition.
        let seeded = literal_support_seeded(&supports, group.len())?;

        let mut coordinates = Vec::<Vec<u32>>::new();
        let mut class_cell = vec![None; class_count];
        let mut coordinate_by_tuple = FxHashMap::<Vec<u32>, usize>::default();
        let mut worklist = VecDeque::<usize>::new();
        for tuple in seeded.into_values() {
            Self::register_literal_fiber_coordinate(
                tuple,
                dead_class,
                &mut coordinates,
                &mut class_cell,
                &mut coordinate_by_tuple,
                &mut worklist,
            )?;
        }

        while let Some(coordinate) = worklist.pop_front() {
            let source_classes = coordinates[coordinate].clone();
            for &byte in &self.topology.bytes {
                let mut destination_states = Vec::with_capacity(group.len());
                let mut destination_classes = Vec::with_capacity(group.len());
                for &source_class in &source_classes {
                    let source_state = quotient.representative_by_class[source_class as usize] as usize;
                    let destination_state = self.topology.destination_for_byte(source_state, byte);
                    destination_classes.push(quotient.class_for_state[destination_state]);
                    destination_states.push(destination_state);
                }
                for member_index in 1..group.len() {
                    if !self.literal_output_matches_after_swap(
                        destination_states[0],
                        destination_states[member_index],
                        pivot,
                        group[member_index],
                    ) {
                        return None;
                    }
                }

                if destination_classes
                    .iter()
                    .all(|&class| class == destination_classes[0])
                {
                    if class_cell[destination_classes[0] as usize].is_some() {
                        return None;
                    }
                    continue;
                }

                let destination_coordinate = if let Some(&existing) =
                    coordinate_by_tuple.get(&destination_classes)
                {
                    existing
                } else {
                    Self::register_literal_fiber_coordinate(
                        destination_classes.clone(),
                        dead_class,
                        &mut coordinates,
                        &mut class_cell,
                        &mut coordinate_by_tuple,
                        &mut worklist,
                    )?
                };
                for (member_index, &class) in destination_classes.iter().enumerate() {
                    if class_cell[class as usize] != Some((destination_coordinate, member_index)) {
                        return None;
                    }
                }
            }
        }

        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if class_cell[root_class].is_some() {
            return None;
        }

        // A core row cannot enter a moved fiber.  For every quotient source
        // whose observed outputs mention this family, also prove the frozen
        // outputs are unchanged by every emitted pivot/member swap.  Sources
        // outside this union observe no family terminal and are invariant.
        let sensitive_classes = supports
            .iter()
            .flat_map(|support| support.iter().map(|&(class, _)| class))
            .collect::<BTreeSet<_>>();
        for class in sensitive_classes {
            let class = class as usize;
            if class_cell[class].is_some() {
                continue;
            }
            let source_state = quotient.representative_by_class[class] as usize;
            for &byte in &self.topology.bytes {
                let destination_state = self.topology.destination_for_byte(source_state, byte);
                let destination_class = quotient.class_for_state[destination_state] as usize;
                if class_cell[destination_class].is_some() {
                    return None;
                }
                for member_index in 1..group.len() {
                    if !self.literal_output_matches_after_swap(
                        destination_state,
                        destination_state,
                        pivot,
                        group[member_index],
                    ) {
                        return None;
                    }
                }
            }
        }

        // The reverse boundary completes the fixed-core proof.  Any core
        // predecessor of a fiber destination would violate equivariance of a
        // fixed source row; a cross-member predecessor would violate the
        // direct pivot/member transpositions.
        for (class, cell) in class_cell.iter().enumerate() {
            let Some((_, member)) = *cell else {
                continue;
            };
            for &predecessor in &quotient.reverse_predecessors[class] {
                match class_cell[predecessor as usize] {
                    Some((_, predecessor_member)) if predecessor_member == member => {}
                    _ => return None,
                }
            }
        }

        let mut witnesses = Vec::with_capacity(members.len());
        for member_index in 1..group.len() {
            let mut deviations = Vec::with_capacity(coordinates.len() * 2);
            for coordinate in &coordinates {
                deviations.push((coordinate[0], coordinate[member_index]));
                deviations.push((coordinate[member_index], coordinate[0]));
            }
            deviations.sort_unstable_by_key(|&(source, _)| source);
            if !deviations.windows(2).all(|pair| pair[0].0 != pair[1].0) {
                return None;
            }
            witnesses.push((
                group[member_index],
                InterchangeMap {
                    scanner_state_map: self.quotient_transport_state_map(
                        Arc::clone(&quotient.class_for_state),
                        Arc::clone(&quotient.representative_by_class),
                        deviations.into_boxed_slice(),
                    ),
                },
            ));
        }
        Some(witnesses)
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

    #[inline]
    fn canonical_signature_hash_component(
        mut hash: u64,
        byte: u8,
        previous_class: u32,
        output: u32,
    ) -> u64 {
        hash = mix_structural_fingerprint(hash, byte as u64);
        hash = mix_structural_fingerprint(hash, previous_class as u64);
        mix_structural_fingerprint(hash, output as u64)
    }

    fn canonical_signature_hash_components(components: &[CanonicalComponent]) -> u64 {
        let mut hash = 0x2d35_83d7_4f1a_6e9b;
        for component in components {
            hash = Self::canonical_signature_hash_component(
                hash,
                component.byte,
                component.previous_class,
                component.output,
            );
        }
        mix_structural_fingerprint(hash, components.len() as u64)
    }

    fn canonical_identity_signature_hash(&self, state: usize, previous: &[u32]) -> u64 {
        let default_class = previous[self.dead_state()];
        let mut hash = 0x2d35_83d7_4f1a_6e9b;
        let mut component_count = 0usize;
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = self.output_pair_by_state[destination];
            let previous_class = previous[destination];
            if previous_class == default_class && output == 0 {
                continue;
            }
            hash = Self::canonical_signature_hash_component(hash, byte, previous_class, output);
            component_count += 1;
        }
        mix_structural_fingerprint(hash, component_count as u64)
    }

    fn canonical_identity_signatures_equal(
        &self,
        left_state: usize,
        right_state: usize,
        previous: &[u32],
    ) -> bool {
        let default_class = previous[self.dead_state()];
        let left_edges = self.topology.edges_from(left_state);
        let right_edges = self.topology.edges_from(right_state);
        let mut left_index = 0usize;
        let mut right_index = 0usize;

        loop {
            while let Some(&(_, destination)) = left_edges.get(left_index) {
                let destination = destination as usize;
                if previous[destination] != default_class
                    || self.output_pair_by_state[destination] != 0
                {
                    break;
                }
                left_index += 1;
            }
            while let Some(&(_, destination)) = right_edges.get(right_index) {
                let destination = destination as usize;
                if previous[destination] != default_class
                    || self.output_pair_by_state[destination] != 0
                {
                    break;
                }
                right_index += 1;
            }

            match (left_edges.get(left_index), right_edges.get(right_index)) {
                (None, None) => return true,
                (Some(_), None) | (None, Some(_)) => return false,
                (Some(&(left_byte, left_destination)), Some(&(right_byte, right_destination))) => {
                    let left_destination = left_destination as usize;
                    let right_destination = right_destination as usize;
                    if left_byte != right_byte
                        || previous[left_destination] != previous[right_destination]
                        || self.output_pair_by_state[left_destination]
                            != self.output_pair_by_state[right_destination]
                    {
                        return false;
                    }
                    left_index += 1;
                    right_index += 1;
                }
            }
        }
    }

    fn canonical_identity_intern_state(
        &self,
        state: usize,
        classes: &[u32],
        representative_by_class: &mut Vec<u32>,
        classes_by_signature_hash: &mut FxHashMap<u64, SmallVec<[u32; 1]>>,
    ) -> u32 {
        let hash = self.canonical_identity_signature_hash(state, classes);
        if let Some(class) = classes_by_signature_hash.get(&hash).and_then(|candidates| {
            candidates.iter().copied().find(|&class| {
                self.canonical_identity_signatures_equal(
                    state,
                    representative_by_class[class as usize] as usize,
                    classes,
                )
            })
        }) {
            return class;
        }
        let class = representative_by_class.len() as u32;
        representative_by_class.push(state as u32);
        classes_by_signature_hash.entry(hash).or_default().push(class);
        class
    }

    fn canonical_signature_matches_identity_state(
        &self,
        signature: &CanonicalSignature,
        state: usize,
        identity_previous: &[u32],
    ) -> bool {
        let default_class = identity_previous[self.dead_state()];
        let mut signature_index = 0usize;
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let previous_class = identity_previous[destination];
            let output = self.output_pair_by_state[destination];
            if previous_class == default_class && output == 0 {
                continue;
            }
            let Some(component) = signature.0.get(signature_index) else {
                return false;
            };
            if component.byte != byte
                || component.previous_class != previous_class
                || component.output != output
            {
                return false;
            }
            signature_index += 1;
        }
        signature_index == signature.0.len()
    }

    fn canonical_round_identity_class_for_signature(
        &self,
        round: &CanonicalRound,
        identity_previous: &[u32],
        signature: &CanonicalSignature,
    ) -> Option<u32> {
        let hash = Self::canonical_signature_hash_components(&signature.0);
        round.classes_by_signature_hash.get(&hash)?.iter().copied().find(|&class| {
            let representative = round.representative_by_class[class as usize] as usize;
            self.canonical_signature_matches_identity_state(signature, representative, identity_previous)
        })
    }

    fn canonical_swapped_signature(
        &self,
        state: usize,
        previous: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let default_class = previous[self.dead_state()];
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
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
        CanonicalSignature(components.into())
    }

    fn canonical_swapped_signature_sparse(
        &self,
        state: usize,
        previous: &[u32],
        outputs: &mut SparseSwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let default_class = previous[self.dead_state()];
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
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
        CanonicalSignature(components.into())
    }

    fn canonical_identity_round(&self, previous: &[u32]) -> CanonicalRound {
        let mut representative_by_class = Vec::<u32>::new();
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let mut classes = Vec::with_capacity(self.state_count());
        for state in 0..self.state_count() {
            let hash = self.canonical_identity_signature_hash(state, previous);
            let existing = classes_by_signature_hash.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&class| {
                    self.canonical_identity_signatures_equal(
                        state,
                        representative_by_class[class as usize] as usize,
                        previous,
                    )
                })
            });
            let class = existing.unwrap_or_else(|| {
                let class = representative_by_class.len() as u32;
                representative_by_class.push(state as u32);
                classes_by_signature_hash.entry(hash).or_default().push(class);
                class
            });
            classes.push(class);
        }
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash,
        }
    }

    fn canonical_identity_round_residual(
        &self,
        previous: &[u32],
        fixed_representatives: &[u32],
        residual: &[usize],
    ) -> CanonicalRound {
        let mut representative_by_class = fixed_representatives.to_vec();
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        for (class, &representative) in fixed_representatives.iter().enumerate() {
            let hash = self.canonical_identity_signature_hash(representative as usize, previous);
            classes_by_signature_hash
                .entry(hash)
                .or_default()
                .push(class as u32);
        }
        let mut classes = previous.to_vec();
        for &state in residual {
            let hash = self.canonical_identity_signature_hash(state, previous);
            let existing = classes_by_signature_hash.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&class| {
                    self.canonical_identity_signatures_equal(
                        state,
                        representative_by_class[class as usize] as usize,
                        previous,
                    )
                })
            });
            let class = existing.unwrap_or_else(|| {
                let class = representative_by_class.len() as u32;
                representative_by_class.push(state as u32);
                classes_by_signature_hash.entry(hash).or_default().push(class);
                class
            });
            classes[state] = class;
        }
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash,
        }
    }

    fn canonical_kahn_identity_round(&self) -> (CanonicalRound, usize, usize, usize) {
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let (order, residual) = self.topology_kahn_order();
        let core = self.residual_cyclic_core(&residual);
        let mut in_core = vec![false; state_count];
        for &state in &core {
            in_core[state as usize] = true;
        }

        let mut classes = vec![u32::MAX; state_count];
        classes[dead_state] = 0;
        let mut representative_by_class = vec![dead_state as u32];
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let dead_hash = self.canonical_identity_signature_hash(dead_state, &classes);
        classes_by_signature_hash.insert(dead_hash, SmallVec::from_slice(&[0]));

        let mut classified = vec![false; state_count];
        classified[dead_state] = true;
        let mut recomputed = 0usize;
        for &state in &order {
            let class = self.canonical_identity_intern_state(
                state,
                &classes,
                &mut representative_by_class,
                &mut classes_by_signature_hash,
            );
            classes[state] = class;
            classified[state] = true;
            recomputed += 1;
        }

        let mut rounds = 0usize;
        if !core.is_empty() {
            rounds = 1;
            recomputed += core.len();
            self.canonical_process_cyclic_component(
                &core,
                &mut classes,
                &mut representative_by_class,
                &mut classes_by_signature_hash,
            );
            for &state in &core {
                classified[state as usize] = true;
            }
        }

        let mut pending = vec![0u32; self.topology.real_state_count];
        let mut queue = VecDeque::<u32>::new();
        for state in 0..self.topology.real_state_count {
            if classified[state] {
                continue;
            }
            pending[state] = self
                .topology
                .edges_from(state)
                .iter()
                .filter(|&&(_, destination)| !classified[destination as usize])
                .count() as u32;
            if pending[state] == 0 {
                queue.push_back(state as u32);
            }
        }
        while let Some(state) = queue.pop_front() {
            let state = state as usize;
            if classified[state] {
                continue;
            }
            let class = self.canonical_identity_intern_state(
                state,
                &classes,
                &mut representative_by_class,
                &mut classes_by_signature_hash,
            );
            classes[state] = class;
            classified[state] = true;
            recomputed += 1;
            for &source in &self.reverse_predecessors[state] {
                let source = source as usize;
                if !classified[source] {
                    pending[source] -= 1;
                    if pending[source] == 0 {
                        queue.push_back(source as u32);
                    }
                }
            }
        }
        assert!(
            classified.iter().all(|&done| done),
            "Kahn TI identity left a state unclassified",
        );
        (
            CanonicalRound {
                classes,
                representative_by_class,
                classes_by_signature_hash,
            },
            rounds,
            recomputed,
            order.len() + core.len(),
        )
    }

    fn ensure_canonical_kahn_identity(&mut self) {
        if self.canonical_kahn_identity.is_some() {
            return;
        }
        let started_at = Instant::now();
        let (kahn, rounds, recomputed, acyclic_states) = self.canonical_kahn_identity_round();
        if std::env::var_os("GLRMASK_PROFILE_L2P_KAHN_IDENTITY_DIAGNOSTIC").is_some() {
            let stable_round = self.ensure_canonical_identity_stable_round();
            assert!(
                same_equality_partition_u32(&kahn.classes, &self.canonical_rounds[stable_round].classes),
                "Kahn TI identity quotient disagrees with canonical refinement",
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] canonical_kahn_identity classes={} rounds={} acyclic_states={} recomputed_states={} elapsed_ms={:.3}",
                kahn.representative_by_class.len(),
                rounds,
                acyclic_states,
                recomputed,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.canonical_kahn_identity = Some(kahn);
    }

    /// Exact Hopcroft minimization of the canonical identity transducer.
    /// Each materialized edge is labelled by its selected byte together with
    /// the frozen output pair of its destination. An absent byte edge remains
    /// the common implicit `(byte, empty-output)` edge to synthetic dead.
    fn canonical_hopcroft_identity_round(&self) -> CanonicalRound {
        let state_count = self.state_count();
        let mut symbol_for = FxHashMap::<(u8, u32), u32>::default();
        let mut inverse = vec![Vec::<(u32, u32)>::new(); state_count];
        for source in 0..state_count {
            let edges = self.topology.edges_from(source);
            let mut edge_index = 0usize;
            for &byte in &self.topology.bytes {
                let destination = match edges.get(edge_index) {
                    Some(&(edge_byte, destination)) if edge_byte == byte => {
                        edge_index += 1;
                        destination as usize
                    }
                    Some(&(edge_byte, _)) if edge_byte < byte => {
                        unreachable!("restricted topology edges must be byte-sorted")
                    }
                    _ => self.dead_state(),
                };
                let next_symbol = symbol_for.len() as u32;
                let symbol = *symbol_for
                    .entry((byte, self.output_pair_by_state[destination]))
                    .or_insert(next_symbol);
                inverse[destination].push((symbol, source as u32));
            }
        }

        let mut classes = vec![0u32; state_count];
        let mut blocks = vec![(0..state_count as u32).collect::<Vec<_>>()];
        let mut worklist = VecDeque::from([0u32]);
        let mut in_worklist = vec![true];
        let mut source_marked = vec![false; state_count];
        let mut sources_to_clear = Vec::<u32>::with_capacity(state_count.min(10_000));
        let mut touched_blocks = Vec::<u32>::with_capacity(1024);
        let mut block_touched = vec![false];
        let mut block_sources = vec![Vec::<u32>::new()];
        let mut symbol_sources = vec![Vec::<u32>::new(); symbol_for.len()];
        let mut touched_symbols = Vec::<u32>::with_capacity(64);

        while let Some(splitter_block) = worklist.pop_front() {
            let splitter = splitter_block as usize;
            if splitter >= blocks.len() || blocks[splitter].is_empty() {
                continue;
            }
            in_worklist[splitter] = false;
            touched_symbols.clear();
            for &target in &blocks[splitter] {
                for &(symbol, source) in &inverse[target as usize] {
                    let sources = &mut symbol_sources[symbol as usize];
                    if sources.is_empty() {
                        touched_symbols.push(symbol);
                    }
                    sources.push(source);
                }
            }

            for &symbol in &touched_symbols {
                sources_to_clear.clear();
                let sources = &mut symbol_sources[symbol as usize];
                for &source in sources.iter() {
                    if source_marked[source as usize] {
                        continue;
                    }
                    source_marked[source as usize] = true;
                    sources_to_clear.push(source);
                    let block = classes[source as usize] as usize;
                    if !block_touched[block] {
                        block_touched[block] = true;
                        touched_blocks.push(block as u32);
                    }
                    block_sources[block].push(source);
                }
                sources.clear();

                for &block_id in &touched_blocks {
                    let block = block_id as usize;
                    let block_len = blocks[block].len();
                    let marked_len = block_sources[block].len();
                    if block_len <= 1 || marked_len == 0 || marked_len == block_len {
                        continue;
                    }
                    let new_block = blocks.len();
                    let move_marked = marked_len <= block_len - marked_len;
                    let old_members = std::mem::take(&mut blocks[block]);
                    let (remaining, moved) = if move_marked {
                        let mut remaining = Vec::with_capacity(block_len - marked_len);
                        for state in old_members {
                            if !source_marked[state as usize] {
                                remaining.push(state);
                            }
                        }
                        (remaining, std::mem::take(&mut block_sources[block]))
                    } else {
                        let mut moved = Vec::with_capacity(block_len - marked_len);
                        for state in old_members {
                            if !source_marked[state as usize] {
                                moved.push(state);
                            }
                        }
                        (std::mem::take(&mut block_sources[block]), moved)
                    };
                    for &state in &moved {
                        classes[state as usize] = new_block as u32;
                    }
                    blocks[block] = remaining;
                    blocks.push(moved);
                    in_worklist.push(false);
                    block_touched.push(false);
                    block_sources.push(Vec::new());
                    if in_worklist[block] {
                        in_worklist[new_block] = true;
                        worklist.push_back(new_block as u32);
                    } else if blocks[block].len() <= blocks[new_block].len() {
                        in_worklist[block] = true;
                        worklist.push_back(block as u32);
                    } else {
                        in_worklist[new_block] = true;
                        worklist.push_back(new_block as u32);
                    }
                }

                for &source in &sources_to_clear {
                    source_marked[source as usize] = false;
                }
                for &block_id in &touched_blocks {
                    let block = block_id as usize;
                    block_touched[block] = false;
                    block_sources[block].clear();
                }
                touched_blocks.clear();
            }
        }

        let mut representative_by_class = vec![u32::MAX; blocks.len()];
        for (state, &class) in classes.iter().enumerate() {
            let representative = &mut representative_by_class[class as usize];
            if *representative == u32::MAX {
                *representative = state as u32;
            }
        }
        debug_assert!(representative_by_class.iter().all(|&state| state != u32::MAX));
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash: FxHashMap::default(),
        }
    }

    fn ensure_canonical_hopcroft_identity(&mut self) {
        if self.canonical_hopcroft_identity.is_some() {
            return;
        }
        let started_at = Instant::now();
        let hopcroft = self.canonical_hopcroft_identity_round();
        if std::env::var_os("GLRMASK_PROFILE_L2P_HOPCROFT_IDENTITY_DIAGNOSTIC").is_some() {
            let stable_round = self.ensure_canonical_identity_stable_round();
            assert!(
                same_equality_partition_u32(
                    &hopcroft.classes,
                    &self.canonical_rounds[stable_round].classes,
                ),
                "Hopcroft TI identity quotient disagrees with canonical refinement",
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] canonical_hopcroft_identity classes={} elapsed_ms={:.3}",
                hopcroft.representative_by_class.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.canonical_hopcroft_identity = Some(hopcroft);
    }

    /// Exact incremental form of the canonical identity refinement. A class
    /// can split only when one of its destination classes split, so after the
    /// first full pass only reverse predecessors of moved states are revisited.
    fn canonical_incremental_identity_round(&self) -> (CanonicalRound, usize, usize) {
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let mut classes = vec![0u32; state_count];
        let mut representative_by_class = vec![dead_state as u32];
        let mut dirty = Vec::with_capacity(state_count);
        dirty.push(dead_state);
        dirty.extend((0..state_count).filter(|&state| state != dead_state));
        let mut dirty_marks = vec![false; state_count];
        let mut rounds = 0usize;
        let mut recomputed = 0usize;

        while !dirty.is_empty() {
            rounds += 1;
            recomputed += dirty.len();
            for &state in &dirty {
                dirty_marks[state] = true;
            }
            let old_class_count = representative_by_class.len();
            let mut members_by_class = vec![Vec::<u32>::new(); old_class_count];
            for (state, &class) in classes.iter().enumerate() {
                members_by_class[class as usize].push(state as u32);
            }
            let mut dirty_by_class = vec![Vec::<u32>::new(); old_class_count];
            for &state in &dirty {
                dirty_by_class[classes[state] as usize].push(state as u32);
            }

            let mut next_classes = classes.clone();
            let mut next_representative_by_class = representative_by_class.clone();
            let mut class_has_signature = vec![false; old_class_count];
            let mut signature_classes = FxHashMap::<(u32, u64), SmallVec<[u32; 1]>>::default();

            for old_class in 0..old_class_count {
                if dirty_by_class[old_class].is_empty() {
                    continue;
                }
                let non_dirty = members_by_class[old_class]
                    .iter()
                    .copied()
                    .find(|&state| !dirty_marks[state as usize]);
                let Some(state) = non_dirty else {
                    continue;
                };
                let state = state as usize;
                let hash = self.canonical_identity_signature_hash(state, &classes);
                signature_classes
                    .entry((old_class as u32, hash))
                    .or_default()
                    .push(old_class as u32);
                next_representative_by_class[old_class] = state as u32;
                class_has_signature[old_class] = true;
            }

            let mut moved_states = Vec::new();
            for &state in &dirty {
                let old_class = classes[state] as usize;
                let hash = self.canonical_identity_signature_hash(state, &classes);
                let target = signature_classes
                    .get(&(old_class as u32, hash))
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            self.canonical_identity_signatures_equal(
                                state,
                                next_representative_by_class[candidate as usize] as usize,
                                &classes,
                            )
                        })
                    });
                let target = target.unwrap_or_else(|| {
                    let target = if !class_has_signature[old_class] {
                        class_has_signature[old_class] = true;
                        next_representative_by_class[old_class] = state as u32;
                        old_class as u32
                    } else {
                        let next = next_representative_by_class.len() as u32;
                        next_representative_by_class.push(state as u32);
                        class_has_signature.push(true);
                        next
                    };
                    signature_classes
                        .entry((old_class as u32, hash))
                        .or_default()
                        .push(target);
                    target
                });
                next_classes[state] = target;
                if target != old_class as u32 {
                    moved_states.push(state);
                }
            }

            for &state in &dirty {
                dirty_marks[state] = false;
            }
            if moved_states.is_empty() {
                classes = next_classes;
                representative_by_class = next_representative_by_class;
                break;
            }
            let mut next_dirty = Vec::new();
            for &state in &moved_states {
                if state == dead_state {
                    for source in 0..state_count {
                        if !dirty_marks[source] {
                            dirty_marks[source] = true;
                            next_dirty.push(source);
                        }
                    }
                } else {
                    for &source in &self.reverse_predecessors[state] {
                        let source = source as usize;
                        if !dirty_marks[source] {
                            dirty_marks[source] = true;
                            next_dirty.push(source);
                        }
                    }
                }
            }
            for &state in &next_dirty {
                dirty_marks[state] = false;
            }
            classes = next_classes;
            representative_by_class = next_representative_by_class;
            dirty = next_dirty;
        }

        (
            CanonicalRound {
                classes,
                representative_by_class,
                classes_by_signature_hash: FxHashMap::default(),
            },
            rounds,
            recomputed,
        )
    }

    fn ensure_canonical_incremental_identity(&mut self) {
        if self.canonical_incremental_identity.is_some() {
            return;
        }
        let started_at = Instant::now();
        let (incremental, rounds, recomputed) = self.canonical_incremental_identity_round();
        if std::env::var_os("GLRMASK_PROFILE_L2P_INCREMENTAL_IDENTITY_DIAGNOSTIC").is_some() {
            let stable_round = self.ensure_canonical_identity_stable_round();
            assert!(
                same_equality_partition_u32(
                    &incremental.classes,
                    &self.canonical_rounds[stable_round].classes,
                ),
                "incremental TI identity quotient disagrees with canonical refinement",
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] canonical_incremental_identity classes={} rounds={} recomputed_states={} elapsed_ms={:.3}",
                incremental.representative_by_class.len(),
                rounds,
                recomputed,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.canonical_incremental_identity = Some(incremental);
    }

    fn canonical_propagated_identity_round(&self) -> (CanonicalRound, usize, usize) {
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let first = self.canonical_identity_round(&vec![0; state_count]);
        let mut classes = first.classes;
        let mut representatives = first.representative_by_class;
        let mut members = vec![Vec::<u32>::new(); representatives.len()];
        for (state, &class) in classes.iter().enumerate() {
            members[class as usize].push(state as u32);
        }
        let mut marked = vec![false; state_count];
        let mut dirty = Vec::<u32>::new();
        if classes[dead_state] != 0 {
            dirty.extend((0..state_count).map(|state| state as u32));
        } else {
            for destination in 0..self.topology.real_state_count {
                if classes[destination] == 0 {
                    continue;
                }
                for &source in &self.reverse_predecessors[destination] {
                    let source = source as usize;
                    if !marked[source] {
                        marked[source] = true;
                        dirty.push(source as u32);
                    }
                }
            }
        }
        for &state in &dirty {
            marked[state as usize] = false;
        }
        let mut passes = 1usize;
        let mut recomputed = state_count;
        while !dirty.is_empty() {
            passes += 1;
            recomputed += dirty.len();
            for &state in &dirty {
                marked[state as usize] = true;
            }
            let previous = classes.clone();
            let old_class_count = representatives.len();
            let mut dirty_by_class = vec![Vec::<u32>::new(); old_class_count];
            for &state in &dirty {
                dirty_by_class[previous[state as usize] as usize].push(state);
            }
            let mut assignments = Vec::<(u32, u32)>::with_capacity(dirty.len());
            for old_class in 0..old_class_count {
                if dirty_by_class[old_class].is_empty() {
                    continue;
                }
                let base = members[old_class].iter().copied().find(|&state| {
                    previous[state as usize] == old_class as u32 && !marked[state as usize]
                });
                let mut retained_old = base.is_some();
                let mut candidates = FxHashMap::<u64, SmallVec<[(u32, u32); 1]>>::default();
                if let Some(base) = base {
                    let hash = self.canonical_identity_signature_hash(base as usize, &previous);
                    candidates.entry(hash).or_default().push((old_class as u32, base));
                    representatives[old_class] = base;
                }
                for &state in &dirty_by_class[old_class] {
                    let hash = self.canonical_identity_signature_hash(state as usize, &previous);
                    let existing = candidates.get(&hash).and_then(|classes| {
                        classes.iter().copied().find(|&(_, representative)| {
                            self.canonical_identity_signatures_equal(
                                state as usize,
                                representative as usize,
                                &previous,
                            )
                        })
                    });
                    let class = existing.map(|(class, _)| class).unwrap_or_else(|| {
                        let class = if !retained_old {
                            retained_old = true;
                            representatives[old_class] = state;
                            old_class as u32
                        } else {
                            let class = representatives.len() as u32;
                            representatives.push(state);
                            members.push(Vec::new());
                            class
                        };
                        candidates.entry(hash).or_default().push((class, state));
                        class
                    });
                    assignments.push((state, class));
                }
            }
            for &state in &dirty {
                marked[state as usize] = false;
            }
            let mut changed = Vec::<u32>::new();
            for (state, class) in assignments {
                let state = state as usize;
                if previous[state] != class {
                    classes[state] = class;
                    members[class as usize].push(state as u32);
                    changed.push(state as u32);
                }
            }
            if changed.is_empty() {
                break;
            }
            dirty.clear();
            if changed.iter().any(|&state| state as usize == dead_state) {
                dirty.extend((0..state_count).map(|state| state as u32));
            } else {
                for &state in &changed {
                    if state as usize >= self.topology.real_state_count {
                        continue;
                    }
                    for &source in &self.reverse_predecessors[state as usize] {
                        let source = source as usize;
                        if !marked[source] {
                            marked[source] = true;
                            dirty.push(source as u32);
                        }
                    }
                }
            }
            for &state in &dirty {
                marked[state as usize] = false;
            }
        }
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        for (class, &representative) in representatives.iter().enumerate() {
            let hash = self.canonical_identity_signature_hash(representative as usize, &classes);
            classes_by_signature_hash.entry(hash).or_default().push(class as u32);
        }
        (
            CanonicalRound {
                classes,
                representative_by_class: representatives,
                classes_by_signature_hash,
            },
            passes,
            recomputed,
        )
    }

    fn ensure_canonical_propagated_identity(&mut self) {
        if self.canonical_propagated_identity.is_some() {
            return;
        }
        let started_at = Instant::now();
        let (round, passes, recomputed) = self.canonical_propagated_identity_round();
        if std::env::var_os("GLRMASK_PROFILE_L2P_PROPAGATED_IDENTITY_DIAGNOSTIC").is_some() {
            let stable_round = self.ensure_canonical_identity_stable_round();
            assert!(
                same_equality_partition_u32(
                    &round.classes,
                    &self.canonical_rounds[stable_round].classes,
                ),
                "propagated TI identity quotient disagrees with canonical refinement",
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] canonical_propagated_identity classes={} passes={} recomputed_states={} elapsed_ms={:.3}",
                round.representative_by_class.len(),
                passes,
                recomputed,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.canonical_propagated_identity = Some(round);
    }

    fn canonical_dag_identity_round(&self) -> CanonicalRound {
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let (component_for_state, components) = self.topology_scc_components();
        let component_count = components.len();
        let dead_component = component_for_state[dead_state] as usize;
        let mut outgoing = vec![Vec::<u32>::new(); component_count];
        let mut incoming = vec![Vec::<u32>::new(); component_count];
        for source in 0..state_count {
            let source_component = component_for_state[source] as usize;
            for &(_, destination) in self.topology.edges_from(source) {
                let destination_component = component_for_state[destination as usize] as usize;
                if source_component != destination_component {
                    outgoing[source_component].push(destination_component as u32);
                    incoming[destination_component].push(source_component as u32);
                }
            }
        }
        for edges in &mut outgoing {
            edges.sort_unstable();
            edges.dedup();
        }
        for edges in &mut incoming {
            edges.sort_unstable();
            edges.dedup();
        }
        let mut remaining_outgoing = outgoing.iter().map(Vec::len).collect::<Vec<_>>();
        let mut processed = vec![false; component_count];
        let mut ready = VecDeque::<u32>::new();
        processed[dead_component] = true;
        for &predecessor in &incoming[dead_component] {
            let predecessor = predecessor as usize;
            remaining_outgoing[predecessor] -= 1;
        }
        for component in 0..component_count {
            if !processed[component] && remaining_outgoing[component] == 0 {
                ready.push_back(component as u32);
            }
        }

        let mut classes = vec![u32::MAX; state_count];
        classes[dead_state] = 0;
        let mut representative_by_class = vec![dead_state as u32];
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let dead_hash = self.canonical_identity_signature_hash(dead_state, &classes);
        classes_by_signature_hash.insert(dead_hash, SmallVec::from_slice(&[0]));

        while let Some(component) = ready.pop_front() {
            let component = component as usize;
            if processed[component] {
                continue;
            }
            let states = &components[component];
            let cyclic = states.len() > 1
                || self.topology.edges_from(states[0] as usize).iter().any(|&(_, destination)| {
                    destination == states[0]
                });
            if cyclic {
                self.canonical_process_cyclic_component(
                    states,
                    &mut classes,
                    &mut representative_by_class,
                    &mut classes_by_signature_hash,
                );
            } else {
                let state = states[0] as usize;
                let class = self.canonical_identity_intern_state(
                    state,
                    &classes,
                    &mut representative_by_class,
                    &mut classes_by_signature_hash,
                );
                classes[state] = class;
            }
            processed[component] = true;
            for &predecessor in &incoming[component] {
                let predecessor = predecessor as usize;
                remaining_outgoing[predecessor] -= 1;
                if !processed[predecessor] && remaining_outgoing[predecessor] == 0 {
                    ready.push_back(predecessor as u32);
                }
            }
        }
        assert!(processed.iter().all(|&done| done), "TI condensation order omitted a component");
        assert!(classes.iter().all(|&class| class != u32::MAX), "TI DAG identity left a state unclassified");
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash,
        }
    }

    fn canonical_process_cyclic_component(
        &self,
        states: &[u32],
        classes: &mut [u32],
        representative_by_class: &mut Vec<u32>,
        classes_by_signature_hash: &mut FxHashMap<u64, SmallVec<[u32; 1]>>,
    ) {
        let temporary_base = representative_by_class.len() as u32;
        for &state in states {
            classes[state as usize] = 0;
        }
        let mut previous = states
            .iter()
            .map(|&state| classes[state as usize])
            .collect::<Vec<_>>();
        let mut temporary_representatives = Vec::<u32>::new();
        for _ in 0..states.len() * 2 + 2 {
            let mut next = Vec::with_capacity(states.len());
            let mut temporary_by_signature = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
            let mut temporary_state_for_class = Vec::<u32>::new();
            for &state in states {
                let state = state as usize;
                let hash = self.canonical_identity_signature_hash(state, classes);
                let global = classes_by_signature_hash.get(&hash).and_then(|candidates| {
                    candidates.iter().copied().find(|&class| {
                        self.canonical_identity_signatures_equal(
                            state,
                            representative_by_class[class as usize] as usize,
                            classes,
                        )
                    })
                });
                let class = if let Some(class) = global {
                    class
                } else if let Some(class) = temporary_by_signature.get(&hash).and_then(|candidates| {
                    candidates.iter().copied().find(|&class| {
                        self.canonical_identity_signatures_equal(
                            state,
                            temporary_state_for_class[(class - temporary_base) as usize] as usize,
                            classes,
                        )
                    })
                }) {
                    class
                } else {
                    let class = temporary_base + temporary_state_for_class.len() as u32;
                    temporary_state_for_class.push(state as u32);
                    temporary_by_signature.entry(hash).or_default().push(class);
                    class
                };
                next.push(class);
            }
            let stable = same_equality_partition_u32(&previous, &next);
            for (&state, &class) in states.iter().zip(&next) {
                classes[state as usize] = class;
            }
            temporary_representatives = temporary_state_for_class;
            previous = next;
            if stable {
                break;
            }
        }
        let mut temporary_to_global = FxHashMap::<u32, u32>::default();
        for (index, &representative) in temporary_representatives.iter().enumerate() {
            let temporary = temporary_base + index as u32;
            let global = representative_by_class.len() as u32;
            representative_by_class.push(representative);
            temporary_to_global.insert(temporary, global);
        }
        for &state in states {
            if let Some(&global) = temporary_to_global.get(&classes[state as usize]) {
                classes[state as usize] = global;
            }
        }
        let first_new_class = temporary_base as usize;
        for class in first_new_class..representative_by_class.len() {
            let representative = representative_by_class[class] as usize;
            let hash = self.canonical_identity_signature_hash(representative, classes);
            classes_by_signature_hash
                .entry(hash)
                .or_default()
                .push(class as u32);
        }
    }

    fn ensure_canonical_dag_identity(&mut self) {
        if self.canonical_dag_identity.is_some() {
            return;
        }
        let started_at = Instant::now();
        let dag = self.canonical_dag_identity_round();
        if std::env::var_os("GLRMASK_PROFILE_L2P_DAG_IDENTITY_DIAGNOSTIC").is_some() {
            let stable_round = self.ensure_canonical_identity_stable_round();
            assert!(
                same_equality_partition_u32(&dag.classes, &self.canonical_rounds[stable_round].classes),
                "DAG TI identity quotient disagrees with canonical refinement",
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] canonical_dag_identity classes={} elapsed_ms={:.3}",
                dag.representative_by_class.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.canonical_dag_identity = Some(dag);
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
                    next.representative_by_class.len(),
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
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        if std::env::var_os("GLRMASK_PROFILE_L2P_KAHN_IDENTITY_DIAGNOSTIC").is_some() {
            self.ensure_canonical_kahn_identity();
        }
        let stable_round = self.ensure_canonical_identity_stable_round();
        let identity_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let class_for_state = self.canonical_rounds[stable_round].classes.clone();
        let class_count = self.canonical_rounds[stable_round]
            .representative_by_class
            .len();
        debug_assert_eq!(
            class_count,
            class_for_state.iter().copied().max().map_or(0, |class| class as usize + 1),
        );

        let representative_by_class = self.canonical_rounds[stable_round]
            .representative_by_class
            .clone();

        let mut reverse_predecessors = vec![Vec::<u32>::new(); class_count];
        for (class, &representative) in representative_by_class.iter().enumerate() {
            let state = representative as usize;
            for &(_, destination) in self.topology.edges_from(state) {
                let destination_class = class_for_state[destination as usize] as usize;
                reverse_predecessors[destination_class].push(class as u32);
            }
        }
        for predecessors in &mut reverse_predecessors {
            predecessors.sort_unstable();
            predecessors.dedup();
        }
        let reverse_ms = started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0 - identity_ms)
            .unwrap_or(0.0);

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
            class_for_state: class_for_state.into(),
            representative_by_class: representative_by_class.into(),
            reverse_predecessors: reverse_predecessors.into(),
            identity_classes_by_round,
            identity_class_counts_by_round,
            stable_previous_to_next,
            stable_next_to_previous,
        });
        if let Some(started_at) = started_at {
            eprintln!(
                "[glrmask/profile][ti_canonical_quotient] classes={} identity_ms={:.3} reverse_ms={:.3} remainder_ms={:.3} total_ms={:.3}",
                class_count,
                identity_ms,
                reverse_ms,
                started_at.elapsed().as_secs_f64() * 1000.0 - identity_ms - reverse_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
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
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
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
        CanonicalSignature(components.into())
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
        swapped_cone: &[u32],
        cone_classes: &[usize],
        cone_index_by_class: &FxHashMap<usize, usize>,
    ) -> bool {
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let swapped_root_class = cone_index_by_class
            .get(&root_class)
            .map(|&index| swapped_cone[index])
            .unwrap_or(identity[root_class]);
        if identity[root_class] != swapped_root_class {
            return false;
        }

        let mut changed_counts = FxHashMap::<u32, u32>::default();
        let mut introduced_or_retained = FxHashSet::<u32>::default();
        for (cone_index, &class) in cone_classes.iter().enumerate() {
            let before = identity[class];
            let after = swapped_cone[cone_index];
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
        swapped_previous_cone: &[u32],
        swapped_next_cone: &[u32],
    ) -> bool {
        swapped_previous_cone
            .iter()
            .zip(swapped_next_cone)
            .all(|(&previous, &next)| {
                let previous = previous as usize;
                let next = next as usize;
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
        // Invert the stable quotient-level class permutation rather than
        // scanning every raw state once merely to rediscover its first class
        // representative. The later raw-state expansion remains exact.
        let mut source_class_for_target_class = vec![u32::MAX; class_count];
        for source_class in 0..class_count {
            // The synthetic dead state may have its own quotient class. The
            // old raw-state scan never visited it, so keep that same domain.
            if quotient.representative_by_class[source_class] as usize
                >= self.topology.real_state_count
            {
                continue;
            }
            let target_class = swapped_classes[source_class] as usize;
            if target_class >= class_count
                || quotient.representative_by_class[target_class] as usize
                    >= self.topology.real_state_count
                || source_class_for_target_class[target_class] != u32::MAX
            {
                return None;
            }
            source_class_for_target_class[target_class] = source_class as u32;
        }
        let source_class_for_target_deviations = source_class_for_target_class
            .iter()
            .enumerate()
            .filter_map(|(target_class, &source_class)| {
                (source_class != u32::MAX && source_class != target_class as u32)
                    .then_some((target_class as u32, source_class))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Some(InterchangeMap {
            scanner_state_map: self.quotient_transport_state_map(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                source_class_for_target_deviations,
            ),
        })
    }

    #[inline]
    fn quotient_identity_map(&self, quotient: &CanonicalQuotient) -> InterchangeMap {
        InterchangeMap {
            scanner_state_map: self.quotient_transport_state_map(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                Box::default(),
            ),
        }
    }

    fn sparse_quotient_interchange_map_from_cone(
        &self,
        quotient: &CanonicalQuotient,
        cone_classes: &[usize],
        swapped_cone: &[u32],
        cone_index_by_class: &FxHashMap<usize, usize>,
    ) -> Option<InterchangeMap> {
        if cone_classes.len() != swapped_cone.len() {
            return None;
        }
        let mut targets = FxHashSet::<usize>::default();
        let mut deviations = Vec::new();
        for (cone_index, &source_class) in cone_classes.iter().enumerate() {
            let target_class = swapped_cone[cone_index] as usize;
            if source_class >= quotient.representative_by_class.len()
                || target_class >= quotient.representative_by_class.len()
                || quotient.representative_by_class[source_class] as usize
                    >= self.topology.real_state_count
                || quotient.representative_by_class[target_class] as usize
                    >= self.topology.real_state_count
                || !cone_index_by_class.contains_key(&target_class)
                || !targets.insert(target_class)
            {
                return None;
            }
            if source_class != target_class {
                deviations.push((target_class as u32, source_class as u32));
            }
        }
        if targets.len() != cone_classes.len() {
            return None;
        }
        deviations.sort_unstable_by_key(|&(target, _)| target);
        Some(InterchangeMap {
            scanner_state_map: self.quotient_transport_state_map(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                deviations.into_boxed_slice(),
            ),
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

    /// Allocation-free exact reverse closure for the support-transposition
    /// proof. The normal sparse fallback also needs a membership bit-vector;
    /// this fast path only iterates the resulting cone, which is tiny for the
    /// proposed transport actions.
    fn support_quotient_affected_cone_small(
        &self,
        quotient: &SupportQuotient,
        left: TerminalID,
        right: TerminalID,
    ) -> SmallVec<[usize; 16]> {
        let mut cone = SmallVec::<[usize; 16]>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !cone.contains(&class) {
                        cone.push(class);
                    }
                }
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !cone.contains(&predecessor) {
                    cone.push(predecessor);
                }
            }
        }
        cone
    }

    fn canonical_quotient_affected_cone_small(
        &self,
        quotient: &CanonicalQuotient,
        left: TerminalID,
        right: TerminalID,
    ) -> SmallVec<[usize; 16]> {
        let mut cone = SmallVec::<[usize; 16]>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !cone.contains(&class) {
                        cone.push(class);
                    }
                }
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !cone.contains(&predecessor) {
                    cone.push(predecessor);
                }
            }
        }
        cone
    }

    fn canonical_sparse_quotient_swapped_signature(
        &self,
        quotient: &CanonicalQuotient,
        class: usize,
        identity_previous: &[u32],
        cone_index_by_class: &FxHashMap<usize, usize>,
        swapped_previous_cone: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let state = quotient.representative_by_class[class] as usize;
        let previous_class_for = |quotient_class: usize| {
            cone_index_by_class
                .get(&quotient_class)
                .map(|&index| swapped_previous_cone[index])
                .unwrap_or(identity_previous[quotient_class])
        };
        let default_class = previous_class_for(
            quotient.class_for_state[self.dead_state()] as usize,
        );
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            let previous_class = previous_class_for(
                quotient.class_for_state[destination] as usize,
            );
            if previous_class == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class,
                output,
            });
        }
        CanonicalSignature(components.into())
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
            let cone_started_at = profile_timing.then(Instant::now);
            let (_in_cone, cone_classes) =
                self.canonical_quotient_affected_cone(quotient, left, right);
            let cone_size = cone_classes.len();
            if let Some(started_at) = cone_started_at {
                self.sparse_quotient_cone_ns += started_at.elapsed().as_nanos() as u64;
            }
            if cone_size == 0 {
                Some((self.quotient_identity_map(quotient), cone_size))
            } else {
                let cone_index_by_class = cone_classes
                    .iter()
                    .enumerate()
                    .map(|(index, &class)| (class, index))
                    .collect::<FxHashMap<_, _>>();
                let mut swapped_previous_cone = cone_classes
                    .iter()
                    .map(|&class| self.quotient_identity_classes_at_round(quotient, 0)[class])
                    .collect::<Vec<_>>();
                let mut outputs = SwappedOutputIds::new(
                    &self.output_pairs,
                    &self.output_pair_lookup,
                    left as usize,
                    right as usize,
                );
                let mut found = None;
                for round in 1..=stable_round {
                    let identity_previous =
                        self.quotient_identity_classes_at_round(quotient, round - 1);
                    let identity_next = self.quotient_identity_classes_at_round(quotient, round);
                    let identity_counts = &quotient.identity_class_counts_by_round[round];
                    let identity = &self.canonical_rounds[round];
                    let identity_raw_previous = &self.canonical_rounds[round - 1].classes;
                    let mut local = FxHashMap::<CanonicalSignature, u32>::default();
                    let local_base = identity.representative_by_class.len() as u32;
                    let mut swapped_next_cone = Vec::with_capacity(cone_size);
                    for &source_class in &cone_classes {
                        let signature = self.canonical_sparse_quotient_swapped_signature(
                            quotient,
                            source_class,
                            identity_previous,
                            &cone_index_by_class,
                            &swapped_previous_cone,
                            &mut outputs,
                        );
                        let swapped_class = if let Some(identity_class) = self
                            .canonical_round_identity_class_for_signature(
                                identity,
                                identity_raw_previous,
                                &signature,
                            )
                        {
                            identity_class
                        } else {
                                let next = local_base + local.len() as u32;
                                *local.entry(signature).or_insert(next)
                            };
                        swapped_next_cone.push(swapped_class);
                    }
                    if !self.sparse_quotient_rooted_class_set_still_possible(
                        quotient,
                        identity_next,
                        identity_counts,
                        &swapped_next_cone,
                        &cone_classes,
                        &cone_index_by_class,
                    ) {
                        return None;
                    }
                    // Materialization uses stable identity class labels, so
                    // accept only at the identity partition fixed point.
                    if round == stable_round
                        && self.sparse_quotient_pair_is_stable(
                            quotient,
                            &swapped_previous_cone,
                            &swapped_next_cone,
                        )
                    {
                        let map_started_at = profile_timing.then(Instant::now);
                        found = self
                            .sparse_quotient_interchange_map_from_cone(
                                quotient,
                                &cone_classes,
                                &swapped_next_cone,
                                &cone_index_by_class,
                            )
                            .map(|map| (map, cone_size));
                        if let Some(started_at) = map_started_at {
                            self.sparse_quotient_map_ns += started_at.elapsed().as_nanos() as u64;
                        }
                        break;
                    }
                    swapped_previous_cone = swapped_next_cone;
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
            left as usize,
            right as usize,
        );
        for round in 1..=stable_round {
            let identity_previous = self.quotient_identity_classes_at_round(quotient, round - 1);
            let identity_next = self.quotient_identity_classes_at_round(quotient, round);
            let identity = &self.canonical_rounds[round];
            let identity_raw_previous = &self.canonical_rounds[round - 1].classes;
            let mut local = FxHashMap::<CanonicalSignature, u32>::default();
            let local_base = identity.representative_by_class.len() as u32;
            let mut swapped_next = Vec::with_capacity(quotient.representative_by_class.len());
            for class in 0..quotient.representative_by_class.len() {
                let signature = self.canonical_quotient_swapped_signature(
                    quotient,
                    class,
                    &swapped_previous,
                    &mut outputs,
                );
                let class = if let Some(identity_class) = self
                    .canonical_round_identity_class_for_signature(
                        identity,
                        identity_raw_previous,
                        &signature,
                    )
                {
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
        identity_previous: &[u32],
        identity: &CanonicalRound,
        outputs: &mut SwappedOutputIds<'_>,
    ) -> Vec<u32> {
        let mut local = FxHashMap::<CanonicalSignature, u32>::default();
        let local_base = identity.representative_by_class.len() as u32;
        let mut classes = Vec::with_capacity(self.state_count());
        for state in 0..self.state_count() {
            let signature = self.canonical_swapped_signature(state, previous, outputs);
            let class = if let Some(identity_class) = self
                .canonical_round_identity_class_for_signature(
                    identity,
                    identity_previous,
                    &signature,
                )
            {
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
            let class_count = self.canonical_rounds[1].representative_by_class.len();
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
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        for &source in &affected_sources {
            let source = source as usize;
            let identity_class = identity.classes[source];
            *changed_by_identity_class.entry(identity_class).or_default() += 1;
            let signature = self.canonical_swapped_signature_sparse(
                source,
                &self.canonical_rounds[0].classes,
                &mut outputs,
            );
            let Some(swapped_class) = self.canonical_round_identity_class_for_signature(
                identity,
                &self.canonical_rounds[0].classes,
                &signature,
            ) else {
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

    /// Exact necessary two-round condition for a terminal swap.  Starting from
    /// the sparse first-round output changes, only their reverse predecessors
    /// can acquire a different round-two signature.  All other rows retain
    /// the cached identity round-two class.  This is still rejection-only.
    fn canonical_round_two_still_possible(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        self.ensure_canonical_identity_round(2);
        if self.canonical_round_two_class_counts.is_none() {
            let class_count = self.canonical_rounds[2].representative_by_class.len();
            let mut counts = vec![0u32; class_count];
            for &class in &self.canonical_rounds[2].classes[..self.topology.real_state_count] {
                counts[class as usize] += 1;
            }
            self.canonical_round_two_class_counts = Some(counts);
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
        let first_sources = std::mem::take(&mut self.canonical_round_one_affected_sources);
        let identity_zero = &self.canonical_rounds[0].classes;
        let identity_one = &self.canonical_rounds[1];
        let identity_two = &self.canonical_rounds[2];
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        let mut swapped_one = identity_one.classes.clone();
        let mut changed_one = Vec::<u32>::new();
        for &source in &first_sources {
            let source = source as usize;
            let signature = self.canonical_swapped_signature_sparse(source, identity_zero, &mut outputs);
            let Some(class) = self.canonical_round_identity_class_for_signature(
                identity_one,
                identity_zero,
                &signature,
            ) else {
                self.canonical_round_one_affected_sources = first_sources;
                return false;
            };
            if class != identity_one.classes[source] {
                swapped_one[source] = class;
                changed_one.push(source as u32);
            }
        }

        self.canonical_round_one_source_mark_epoch =
            self.canonical_round_one_source_mark_epoch.wrapping_add(1);
        if self.canonical_round_one_source_mark_epoch == 0 {
            self.canonical_round_one_source_marks.fill(0);
            self.canonical_round_one_source_mark_epoch = 1;
        }
        let epoch = self.canonical_round_one_source_mark_epoch;
        let mut second_sources = Vec::<u32>::new();
        let mut add_source = |source: usize| {
            if self.canonical_round_one_source_marks[source] != epoch {
                self.canonical_round_one_source_marks[source] = epoch;
                second_sources.push(source as u32);
            }
        };
        for &source in &first_sources {
            add_source(source as usize);
        }
        for &changed in &changed_one {
            for &source in &self.reverse_predecessors[changed as usize] {
                add_source(source as usize);
            }
        }

        let identity_counts = self
            .canonical_round_two_class_counts
            .as_ref()
            .expect("second-round counts initialized");
        let mut changed_by_identity_class = FxHashMap::<u32, u32>::default();
        let mut added_identity_classes = FxHashSet::<u32>::default();
        let mut swapped_root_class = identity_two.classes[self.topology.initial_state];
        for &source in &second_sources {
            let source = source as usize;
            let identity_class = identity_two.classes[source];
            *changed_by_identity_class.entry(identity_class).or_default() += 1;
            let signature = self.canonical_swapped_signature_sparse(source, &swapped_one, &mut outputs);
            let Some(class) = self.canonical_round_identity_class_for_signature(
                identity_two,
                &identity_one.classes,
                &signature,
            ) else {
                self.canonical_round_one_affected_sources = first_sources;
                return false;
            };
            added_identity_classes.insert(class);
            if source == self.topology.initial_state {
                swapped_root_class = class;
            }
        }
        self.canonical_round_one_affected_sources = first_sources;
        if swapped_root_class != identity_two.classes[self.topology.initial_state] {
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
            left as usize,
            right as usize,
        );
        let mut swapped_previous = self.canonical_rounds[0].classes.clone();
        for round in 1..=stable_round {
            let identity_previous = &self.canonical_rounds[round - 1].classes;
            let identity = &self.canonical_rounds[round];
            let swapped_next = self.canonical_swapped_round(
                &swapped_previous,
                identity_previous,
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
        self.ensure_canonical_identity_stable_round();
        self.ensure_canonical_quotient();
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        let map = self.quotient_identity_map(quotient);
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
            scanner_state_map: TransportScannerStateMap::Explicit(target_representative_for_source_state.into()),
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
        let left_terminal = left as usize;
        let right_terminal = right as usize;
        self.observed_output_pair_mark_epoch = self.observed_output_pair_mark_epoch.wrapping_add(1);
        if self.observed_output_pair_mark_epoch == 0 {
            self.observed_output_pair_marks.fill(0);
            self.observed_output_pair_mark_epoch = 1;
        }
        let epoch = self.observed_output_pair_mark_epoch;
        let swap = Some((
            left as usize,
            right as usize,
        ));
        for ids in [
            &self.observed_output_pair_ids_by_terminal[left_terminal],
            &self.observed_output_pair_ids_by_terminal[right_terminal],
        ] {
            for &id in ids {
                let id = id as usize;
                if self.observed_output_pair_marks[id] == epoch {
                    continue;
                }
                self.observed_output_pair_marks[id] = epoch;
                let pair = &self.output_pairs[id];
                let swapped = OutputPair {
                    finalizers: pair.finalizers.mapped(swap),
                    future_finalizers: pair.future_finalizers.mapped(swap),
                };
                let Some(&swapped_id) = self.output_pair_lookup.get(&swapped) else {
                    return false;
                };
                if !self.observed_output_pair_present[swapped_id as usize] {
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
        let swap = Some((
            left as usize,
            right as usize,
        ));
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
            scanner_state_map: TransportScannerStateMap::Explicit(target_representative_for_source_state.into()),
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

pub(crate) fn discover_one_round(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
) -> BTreeMap<TerminalID, BTreeSet<TerminalID>> {
    discover_one_round_with_transport_witnesses(
        tokenizer,
        active_terminals,
        relevant_bytes,
        ignore_terminal,
    )
    .partition
}

/// Discover one exact TI round and retain only the accepted binary witnesses
/// for immediate build-local transport reconstruction.  Each witness is
/// derived under this round's exact pre-merge active mask; no later final-mask
/// re-derivation is permitted.
pub(crate) fn discover_one_round_with_transport_witnesses(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
) -> TiRoundTransportWitnesses {
    let context = TiDiscoveryContext::new(tokenizer, relevant_bytes, None);
    discover_one_round_with_transport_witnesses_in_context(
        tokenizer,
        active_terminals,
        &context,
        ignore_terminal,
    )
}

/// As `discover_one_round_with_transport_witnesses`, but reuses the immutable
/// topology/root-observation context across iterative historical masks.
pub(crate) fn discover_one_round_with_transport_witnesses_in_context(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    context: &TiDiscoveryContext,
    ignore_terminal: Option<TerminalID>,
) -> TiRoundTransportWitnesses {
    discover_one_round_with_transport_witnesses_in_context_impl(
        tokenizer,
        active_terminals,
        context,
        ignore_terminal,
        false,
    )
}

/// Run the ordinary first historical TI round, but install only fully proved
/// literal/fiber witnesses before the generic candidate loop. The remaining
/// terminals retain the same active mask and follow the generic exact path.
pub(crate) fn discover_one_round_with_literal_fiber_certificate_in_context(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    context: &TiDiscoveryContext,
    ignore_terminal: Option<TerminalID>,
) -> TiRoundTransportWitnesses {
    discover_one_round_with_transport_witnesses_in_context_impl(
        tokenizer,
        active_terminals,
        context,
        ignore_terminal,
        true,
    )
}

fn discover_one_round_with_transport_witnesses_in_context_impl(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    context: &TiDiscoveryContext,
    ignore_terminal: Option<TerminalID>,
    seed_literal_fiber_certificate: bool,
) -> TiRoundTransportWitnesses {
        let candidates = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let topology = Arc::clone(&context.topology);
        let topology_ms = 0.0;
        let topology_edge_count = topology.edges.len();
        let topology_max_outdegree = topology.max_outdegree;
        let topology_byte_count = topology.bytes.len();
        let candidate_filter_started_at = profile_timing.then(Instant::now);
        let root_candidate_groups = rooted_candidate_groups_from_signatures(
            &candidates,
            &context.root_output_signatures,
        );
        let root_observed_states = context.root_observed_states;
        let root_candidate_pairs = root_candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        if root_candidate_pairs == 0 {
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }

        // The old structural support pass is only a rejection prefilter. Once
        // the output-pair hypergraph refinement is in place it costs more than
        // the additional exact checks it avoids on p0/p1, so production goes
        // straight from rooted candidates to the exact-safe output filters.
        let structural_filter_started_at = profile_timing.then(Instant::now);
        let structural_candidate_groups = root_candidate_groups
            .into_iter()
            .filter(|group| group.len() >= 2)
            .collect::<Vec<_>>();
        let use_first_round_global_hash_filter = seed_literal_fiber_certificate
            && std::env::var_os("GLRMASK_PROFILE_L2P_FIRST_ROUND_GLOBAL_HASH_FILTER").is_some();
        let use_first_round_global_hash_certificate = seed_literal_fiber_certificate
            && std::env::var_os("GLRMASK_PROFILE_L2P_FIRST_ROUND_GLOBAL_HASH_CERTIFICATE")
                .is_some();
        let use_global_quotient_wl = use_first_round_global_hash_filter
            || use_first_round_global_hash_certificate
            || std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_QUOTIENT_WL").is_some()
            || std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_HASH_WL").is_some()
            || std::env::var_os("GLRMASK_PROFILE_L2P_DENSE_ORBIT").is_some();
        let structural_color_count = 0usize;
        let structural_candidate_pairs = structural_candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        let structural_candidate_group_count = structural_candidate_groups.len();
        let structural_filter_ms = structural_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if structural_candidate_pairs == 0 {
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }

        let dfa_setup_started_at = profile_timing.then(Instant::now);
        let mut dfa = InterchangeabilityDfa::from_context(active_terminals, context);
        let dfa_setup_ms = dfa_setup_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let output_shape_filter_started_at = profile_timing.then(Instant::now);
        let output_shape_candidate_groups = if use_global_quotient_wl {
            structural_candidate_groups.clone()
        } else {
            refine_candidate_groups_by_observed_output_pair_shape(
                structural_candidate_groups,
                &dfa.observed_output_pair_support_shapes_by_terminal,
            )
        };
        let output_shape_filter_ms = output_shape_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let output_hypergraph_filter_started_at = profile_timing.then(Instant::now);
        let (candidate_groups, output_hypergraph_rounds) = if use_global_quotient_wl {
            (output_shape_candidate_groups, 0)
        } else {
            refine_candidate_groups_by_observed_output_hypergraph(
                output_shape_candidate_groups,
                &dfa.observed_output_pair_ids,
                &dfa.output_pairs,
                active_terminals.len(),
            )
        };
        let candidate_groups = if std::env::var_os("GLRMASK_PROFILE_L2P_JOINT_WL").is_some() {
            dfa.refine_candidate_groups_by_joint_wl(candidate_groups)
        } else {
            candidate_groups
        };
        let candidate_groups = if std::env::var_os("GLRMASK_PROFILE_L2P_QUOTIENT_WL").is_some() {
            dfa.refine_candidate_groups_by_quotient_wl(candidate_groups)
        } else {
            candidate_groups
        };
        let candidate_groups = if use_global_quotient_wl {
            if use_first_round_global_hash_filter
                || use_first_round_global_hash_certificate
                || std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_HASH_WL").is_some()
            {
                dfa.refine_candidate_groups_by_global_quotient_hash_wl(candidate_groups)
            } else {
                dfa.refine_candidate_groups_by_global_quotient_wl(candidate_groups)
            }
        } else {
            candidate_groups
        };
        let exact_candidate_pairs = candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        let output_hypergraph_filter_ms = output_hypergraph_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let candidate_filter_ms = candidate_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if profile_timing {
            let mut group_size_histogram = BTreeMap::<usize, usize>::new();
            for group in &candidate_groups {
                *group_size_histogram.entry(group.len()).or_default() += 1;
            }
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] active={} selected_bytes={} sparse_edges={} max_outdegree={} root_observed_states={} root_candidate_pairs={} structural_colors={} structural_candidate_groups={} structural_candidate_pairs={} observed_output_candidate_groups={} output_hypergraph_rounds={} exact_candidate_pairs={} group_size_histogram={:?} topology_ms={:.3} candidate_filter_total_ms={:.3} structural_filter_ms={:.3} dfa_setup_ms={:.3} output_shape_filter_ms={:.3} output_hypergraph_filter_ms={:.3}",
                candidates.len(),
                topology_byte_count,
                topology_edge_count,
                topology_max_outdegree,
                root_observed_states,
                root_candidate_pairs,
                structural_color_count,
                structural_candidate_group_count,
                structural_candidate_pairs,
                candidate_groups.len(),
                output_hypergraph_rounds,
                exact_candidate_pairs,
                group_size_histogram,
                topology_ms,
                candidate_filter_ms,
                structural_filter_ms,
                dfa_setup_ms,
                output_shape_filter_ms,
                output_hypergraph_filter_ms,
            );
        }
        if exact_candidate_pairs == 0 {
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }
        let mut accepted_maps = if std::env::var_os("GLRMASK_PROFILE_L2P_SUPPORT_SHAPE_CERTIFICATE")
            .is_some()
        {
            support_shape_pre_certificate_maps(&mut dfa, &candidate_groups)
        } else if use_first_round_global_hash_certificate
            || std::env::var_os("GLRMASK_PROFILE_L2P_GLOBAL_QUOTIENT_WL_CERTIFICATE").is_some()
        {
            quotient_wl_pre_certificate_maps(&mut dfa, &candidate_groups)
        } else if seed_literal_fiber_certificate
            && std::env::var_os("GLRMASK_PROFILE_L2P_DISABLE_LITERAL_CERTIFICATE").is_none()
        {
            literal_fiber_pre_certificate_maps(
                tokenizer,
                active_terminals,
                context,
                &mut dfa,
                &candidate_groups,
                ignore_terminal,
            )
        } else {
            BTreeMap::new()
        };
        let literal_fiber_certified_members = accepted_maps.len();
        let literal_fiber_representatives = accepted_maps
            .keys()
            .map(|&(representative, _)| representative)
            .collect::<BTreeSet<_>>();
        let mut result = singleton_partition(active_terminals);
        for &(representative, member) in accepted_maps.keys() {
            result
                .get_mut(&representative)
                .expect("literal TI representative must remain active")
                .insert(member);
            let removed = result.remove(&member);
            debug_assert!(removed.is_some(), "literal TI member must remain active");
        }
        let mut output_pair_rejections = 0usize;
        let mut output_invariant_checks = 0usize;
        let mut first_round_rejections = 0usize;
        let mut second_round_rejections = 0usize;
        let mut support_transposition_checks = 0usize;
        let mut direct_exact_checks = 0usize;
        let mut accepted_representative_members = 0usize;
        let mut output_pair_filter_ns = 0u64;
        let mut frozen_output_ns = 0u64;
        let mut first_round_ns = 0u64;
        let mut support_transposition_ns = 0u64;
        let mut exact_map_ns = 0u64;
        let mut accepted_map_storage_ns = 0u64;

        // Accepted terminal swaps are automorphisms. Therefore (a b) and
        // (b c) imply (a c) by conjugation, so interchangeability is an
        // equivalence relation. Partition each candidate group by pivots,
        // keeping only the representative-to-member maps transport requires.
        for initial_group in candidate_groups {
            let group_representative = initial_group.first().copied().unwrap_or_default();
            let group_member_count = initial_group.len();
            let mut group_attempts = 0usize;
            let mut group_output_rejections = 0usize;
            let mut group_first_round_rejections = 0usize;
            let mut group_accepted = 0usize;
            let mut unresolved = initial_group
                .into_iter()
                .filter(|terminal| result.contains_key(terminal))
                .collect::<Vec<_>>();
            if std::env::var_os("GLRMASK_PROFILE_L2P_BULK_GROUP_SUPPORTS").is_some()
                && unresolved.len() >= 8
            {
                dfa.ensure_terminal_quotient_output_supports_bulk(&unresolved);
            }
            while !unresolved.is_empty() {
                let representative = unresolved[0];
                let mut next_unresolved = Vec::with_capacity(unresolved.len().saturating_sub(1));
                for &terminal in &unresolved[1..] {
                    if literal_fiber_representatives.contains(&terminal) {
                        next_unresolved.push(terminal);
                        continue;
                    }
                    group_attempts += 1;
                    let output_pair_started_at = profile_timing.then(Instant::now);
                    let output_pair_is_closed =
                        dfa.observed_output_pair_set_is_swap_closed(representative, terminal);
                    if let Some(started_at) = output_pair_started_at {
                        output_pair_filter_ns += started_at.elapsed().as_nanos() as u64;
                    }
                    if !output_pair_is_closed {
                        output_pair_rejections += 1;
                        group_output_rejections += 1;
                        next_unresolved.push(terminal);
                        continue;
                    }
                    let frozen_output_started_at = profile_timing.then(Instant::now);
                    let preserves_frozen_outputs =
                        dfa.swap_preserves_all_frozen_outputs(representative, terminal);
                    if let Some(started_at) = frozen_output_started_at {
                        frozen_output_ns += started_at.elapsed().as_nanos() as u64;
                    }
                    let map = if preserves_frozen_outputs {
                        output_invariant_checks += 1;
                        Some(dfa.canonical_identity_map())
                    } else {
                        let first_round_started_at = profile_timing.then(Instant::now);
                        let first_round_possible =
                            dfa.canonical_round_one_still_possible(representative, terminal);
                        if let Some(started_at) = first_round_started_at {
                            first_round_ns += started_at.elapsed().as_nanos() as u64;
                        }
                        if !first_round_possible {
                            first_round_rejections += 1;
                            group_first_round_rejections += 1;
                            None
                        } else if std::env::var_os("GLRMASK_PROFILE_L2P_SECOND_ROUND_FILTER").is_some()
                            && !dfa.canonical_round_two_still_possible(representative, terminal)
                        {
                            second_round_rejections += 1;
                            None
                        } else if std::env::var_os("GLRMASK_PROFILE_L2P_DISABLE_SUPPORT_TRANSPOSITION").is_some() {
                            direct_exact_checks += 1;
                            let exact_map_started_at = profile_timing.then(Instant::now);
                            let map = dfa.interchange_map(representative, terminal);
                            if let Some(started_at) = exact_map_started_at {
                                exact_map_ns += started_at.elapsed().as_nanos() as u64;
                            }
                            map
                        } else {
                            support_transposition_checks += 1;
                            let support_transposition_started_at = profile_timing.then(Instant::now);
                            let support_map = dfa
                                .support_transposition_interchange_map(representative, terminal);
                            if let Some(started_at) = support_transposition_started_at {
                                support_transposition_ns += started_at.elapsed().as_nanos() as u64;
                            }
                            if let Some(support_map) = support_map {
                                Some(support_map)
                            } else {
                                direct_exact_checks += 1;
                                let exact_map_started_at = profile_timing.then(Instant::now);
                                let map = dfa.interchange_map(representative, terminal);
                                if let Some(started_at) = exact_map_started_at {
                                    exact_map_ns += started_at.elapsed().as_nanos() as u64;
                                }
                                map
                            }
                        }
                    };
                    if let Some(map) = map {
                        let storage_started_at = profile_timing.then(Instant::now);
                        accepted_representative_members += 1;
                        group_accepted += 1;
                        result
                            .get_mut(&representative)
                            .expect("TI representative must retain its singleton partition entry")
                            .insert(terminal);
                        let removed = result.remove(&terminal);
                        debug_assert!(removed.is_some(), "TI member must retain its singleton partition entry");
                        let replaced = accepted_maps.insert(
                            (representative, terminal),
                            Arc::new(map.scanner_state_map()),
                        );
                        debug_assert!(replaced.is_none(), "TI round must accept each pair once");
                        if let Some(started_at) = storage_started_at {
                            accepted_map_storage_ns += started_at.elapsed().as_nanos() as u64;
                        }
                    } else {
                        next_unresolved.push(terminal);
                    }
                }
                unresolved = next_unresolved;
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_GROUP_DIAGNOSTIC").is_some()
                && group_member_count >= 8
            {
                eprintln!(
                    "[glrmask/profile][ti_candidate_group] representative={} members={} attempts={} output_rejections={} first_round_rejections={} accepted={}",
                    group_representative,
                    group_member_count,
                    group_attempts,
                    group_output_rejections,
                    group_first_round_rejections,
                    group_accepted,
                );
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_LITERAL_DIFFS").is_some() {
            let literal_groups = literal_projection_groups(
                tokenizer,
                active_terminals,
                context,
                ignore_terminal,
            );
            for group in literal_groups.into_iter().filter(|group| group.len() >= 8) {
                let group_set = group.iter().copied().collect::<BTreeSet<_>>();
                let Some((&class_representative, class_members)) = result
                    .iter()
                    .max_by_key(|(_, members)| members.intersection(&group_set).count())
                else {
                    continue;
                };
                let overlap = class_members.intersection(&group_set).count();
                let extras = group_set
                    .difference(class_members)
                    .copied()
                    .map(|terminal| {
                        let literal = tokenizer
                            .literal_terminal_bytes(terminal)
                            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                            .unwrap_or_else(|| "<nonliteral>".to_owned());
                        format!("{}:{:?}", terminal, literal)
                    })
                    .collect::<Vec<_>>();
                let missing = class_members.difference(&group_set).copied().collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][ti_literal_group_diff] group_rep={} group_members={} class_rep={} class_members={} overlap={} extras={:?} missing={:?}",
                    group[0],
                    group.len(),
                    class_representative,
                    class_members.len(),
                    overlap,
                    extras,
                    missing,
                );
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_CLASS_MEMBERS").is_some() {
            for (&representative, members) in &result {
                if members.len() < 16 {
                    continue;
                }
                let preview = members
                    .iter()
                    .take(24)
                    .map(|&terminal| {
                        let literal = tokenizer
                            .literal_terminal_bytes(terminal)
                            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                            .unwrap_or_else(|| "<nonliteral>".to_owned());
                        format!("{}:{:?}", terminal, literal)
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][ti_class_members] representative={} members={} preview={:?}",
                    representative,
                    members.len(),
                    preview,
                );
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_ORBIT_DIAGNOSTIC").is_some() {
            for (&representative, members) in &result {
                if members.len() < 16 {
                    continue;
                }
                let group = members.iter().copied().collect::<Vec<_>>();
                let witness_count = dfa
                    .symmetric_support_orbit_witnesses(&group)
                    .map(|witnesses| witnesses.len());
                eprintln!(
                    "[glrmask/profile][ti_symmetric_orbit_diagnostic] representative={} members={} witnesses={:?}",
                    representative,
                    group.len(),
                    witness_count,
                );
            }
        }

        if profile_timing {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] literal_fiber_certified_members={} output_pair_rejections={} output_invariant_checks={} first_round_rejections={} second_round_rejections={} support_transposition_checks={} support_transposition_certified={} support_transposition_no_template={} support_transposition_outside_cone={} support_transposition_root_rejected={} support_transposition_signature_rejected={} direct_exact_checks={} output_pair_filter_ms={:.3} frozen_output_ms={:.3} first_round_ms={:.3} support_transposition_ms={:.3} support_setup_ms={:.3} support_template_ms={:.3} support_cone_ms={:.3} support_verify_ms={:.3} exact_map_ms={:.3} accepted_map_storage_ms={:.3} quotient_certified={} sparse_quotient_certified={} sparse_cone_avg={:.1} sparse_cone_max={} sparse_cone_ms={:.3} sparse_refinement_ms={:.3} sparse_map_ms={:.3} accepted_representative_members={} total_ms={:.3}",
                literal_fiber_certified_members,
                output_pair_rejections,
                output_invariant_checks,
                first_round_rejections,
                second_round_rejections,
                support_transposition_checks,
                dfa.support_transposition_certified,
                dfa.support_transposition_no_template,
                dfa.support_transposition_outside_cone,
                dfa.support_transposition_root_rejected,
                dfa.support_transposition_signature_rejected,
                direct_exact_checks,
                output_pair_filter_ns as f64 / 1_000_000.0,
                frozen_output_ns as f64 / 1_000_000.0,
                first_round_ns as f64 / 1_000_000.0,
                support_transposition_ns as f64 / 1_000_000.0,
                dfa.support_transposition_support_setup_ns as f64 / 1_000_000.0,
                dfa.support_transposition_template_ns as f64 / 1_000_000.0,
                dfa.support_transposition_cone_ns as f64 / 1_000_000.0,
                dfa.support_transposition_verify_ns as f64 / 1_000_000.0,
                exact_map_ns as f64 / 1_000_000.0,
                accepted_map_storage_ns as f64 / 1_000_000.0,
                dfa.quotient_certified,
                dfa.sparse_quotient_certified,
                dfa.sparse_quotient_cone_classes_total as f64
                    / dfa.sparse_quotient_certified.max(1) as f64,
                dfa.sparse_quotient_cone_classes_max,
                dfa.sparse_quotient_cone_ns as f64 / 1_000_000.0,
                dfa.sparse_quotient_refinement_ns as f64 / 1_000_000.0,
                dfa.sparse_quotient_map_ns as f64 / 1_000_000.0,
                accepted_representative_members,
                started_at
                    .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0),
            );
        }
        assert_partition_invariants(&result, active_terminals);
        let next_round_state_map = result
            .values()
            .any(|members| members.len() > 1)
            .then(|| dfa.raw_stable_identity_state_map());
        TiRoundTransportWitnesses {
            active_before_round: active_terminals.to_vec(),
            partition: result,
            maps: accepted_maps,
            next_round_state_map,
        }
    }


/// Return the singleton-inclusive one-round partition for the current active
/// terminal mask. Inactive terminals are deliberately absent: this same mask
/// controls both candidates and frozen F/U observation for the round.
pub(crate) fn singleton_partition(
    active_terminals: &[bool],
) -> BTreeMap<TerminalID, BTreeSet<TerminalID>> {
    active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| {
            active.then_some((terminal as TerminalID, BTreeSet::from([terminal as TerminalID])))
        })
        .collect()
}

fn assert_partition_invariants(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    active_terminals: &[bool],
) {
    let mut seen = vec![false; active_terminals.len()];
    for (&representative, members) in partition {
        assert!(members.contains(&representative), "TI partition class must contain its key");
        for &member in members {
            assert!(
                active_terminals.get(member as usize).copied().unwrap_or(false),
                "TI partition contains an inactive terminal",
            );
            assert!(
                !std::mem::replace(&mut seen[member as usize], true),
                "TI partition contains a terminal in multiple classes",
            );
        }
    }
    assert_eq!(seen, active_terminals, "TI partition must cover exactly active terminals");
}

/// Fold one transient discovery round into a flat original-terminal
/// partition. `round` names only current visible representatives; its values
/// are expanded through the existing flat classes. No history is retained.
pub(crate) fn fold_one_round_partition(
    classes: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    round: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
) -> BTreeMap<TerminalID, BTreeSet<TerminalID>> {
    let mut next_classes = BTreeMap::new();
    for (&new_representative, old_representatives) in round {
        let mut members = BTreeSet::new();
        for &old_representative in old_representatives {
            let old_members = classes
                .get(&old_representative)
                .expect("TI round must refer only to visible representatives");
            members.extend(old_members.iter().copied());
        }
        assert!(members.contains(&new_representative));
        next_classes.insert(new_representative, members);
    }
    next_classes
}

pub(crate) fn active_terminals_for_partition(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    terminal_count: usize,
) -> Vec<bool> {
    let mut active = vec![false; terminal_count];
    for &representative in partition.keys() {
        active[representative as usize] = true;
    }
    active
}

pub(crate) fn partition_has_merges(partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>) -> bool {
    partition.values().any(|members| members.len() > 1)
}

pub(crate) fn visible_output_raw_labels(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    terminal_count: usize,
) -> Vec<bool> {
    // The partition contains only TI-active terminals. Terminals outside it
    // were never compressed and must remain ordinary visible outputs. Start
    // from the full raw alphabet, then hide only actual nonrepresentative
    // partition members.
    let mut visible = vec![true; terminal_count];
    for (&representative, members) in partition {
        for &member in members {
            if member != representative {
                visible[member as usize] = false;
            }
        }
    }
    visible
}

fn representatives_for_partition(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    terminal_count: usize,
) -> Vec<TerminalID> {
    let mut representative_for = (0..terminal_count as TerminalID).collect::<Vec<_>>();
    for (&representative, members) in partition {
        for &member in members {
            representative_for[member as usize] = representative;
        }
    }
    representative_for
}

/// Coalesce grammar follows for a compact representative alphabet. A class pair
/// is disallowed only when every concrete member pair is disallowed; original
/// follows are retained for the later raw-member transport construction.
pub(crate) fn coalesced_disallowed_follows(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    original: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
) -> BTreeMap<u32, BitSet> {
    // A compact predecessor class is disallowed from a compact successor class
    // precisely when every raw predecessor member forbids every raw successor
    // member.  For one predecessor class, first intersect its raw rows; the
    // resulting bitset contains exactly the raw successors forbidden by all
    // of its members.  Testing each successor class against that intersection
    // avoids a hash-map update for every raw edge, while retaining the exact
    // all-member-pair relation.
    let classes = partition
        .iter()
        .map(|(&representative, members)| (representative, members))
        .collect::<Vec<_>>();
    let empty = BitSet::new(num_terminals);
    let mut coalesced = BTreeMap::new();

    for &(representative, members) in &classes {
        let mut members = members.iter();
        let Some(&first_member) = members.next() else {
            continue;
        };
        let mut common_successors = original
            .get(&(first_member as u32))
            .cloned()
            .unwrap_or_else(|| empty.clone());
        for &member in members {
            let Some(successors) = original.get(&(member as u32)) else {
                common_successors.clear_all();
                break;
            };
            common_successors.intersect_with(successors);
            if common_successors.is_empty() {
                break;
            }
        }

        let mut row = BitSet::new(num_terminals);
        for &(successor_representative, successor_members) in &classes {
            if successor_members
                .iter()
                .all(|&member| common_successors.contains(member as usize))
            {
                row.set(successor_representative as usize);
            }
        }
        coalesced.insert(representative as u32, row);
    }

    coalesced
}

/// Re-run the same accepted-pair decision used by discovery for one replayed
/// round. Replay must derive the scanner coordinate under that round's
/// pre-merge active-terminal mask.
fn replay_accepted_interchange_map(
    dfa: &mut InterchangeabilityDfa,
    representative: TerminalID,
    member: TerminalID,
) -> Option<InterchangeMap> {
    if !dfa.observed_output_pair_set_is_swap_closed(representative, member) {
        return None;
    }
    if dfa.swap_preserves_all_frozen_outputs(representative, member) {
        return Some(dfa.canonical_identity_map());
    }
    if !dfa.canonical_round_one_still_possible(representative, member) {
        return None;
    }
    dfa.interchange_map(representative, member)
}

/// Replay deterministic TI discovery while materializing temporary transport
/// witnesses. The stored result remains only the flat final partition.
///
/// Each accepted pair uses the active mask that existed before its round
/// merged anything. When a later round folds `old_representative` into
/// `representative`, it composes that round-local map outside the prior map for
/// every original member. Reconstructing final pairs under only final
/// representatives, or final representatives plus one member, is unsound.
pub(crate) fn binary_transport_modes(
    tokenizer: &Tokenizer,
    original_active_terminals: &[bool],
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
) -> Vec<TerminalNwaTransportMode> {
    let terminal_count = tokenizer.num_terminals() as usize;
    assert_eq!(original_active_terminals.len(), terminal_count);

    let mut active = original_active_terminals.to_vec();
    let mut classes = singleton_partition(&active);
    let mut rounds = Vec::new();

    loop {
        let round = discover_one_round_with_transport_witnesses(
            tokenizer,
            &active,
            relevant_bytes,
            ignore_terminal,
        );
        let next_classes = fold_one_round_partition(&classes, &round.partition);
        let next_active = active_terminals_for_partition(&round.partition, active.len());
        rounds.push(round);
        if next_active == active {
            assert_eq!(
                &next_classes,
                partition,
                "deterministic TI replay must reproduce the retained final partition",
            );
            break;
        }
        classes = next_classes;
        active = next_active;
    }

    binary_transport_modes_from_witnesses(tokenizer, original_active_terminals, partition, &rounds)
}

/// Build temporary member reconstruction modes from the witnesses produced by
/// the original discovery loop.  This is the exact iterative replay relation,
/// but it never repeats discovery: each historical active mask and its
/// accepted pair map were captured at the instant that round was certified.
pub(crate) fn binary_transport_modes_from_witnesses(
    tokenizer: &Tokenizer,
    original_active_terminals: &[bool],
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    rounds: &[TiRoundTransportWitnesses],
) -> Vec<TerminalNwaTransportMode> {
    let state_count = tokenizer.num_states() as usize;
    let terminal_count = tokenizer.num_terminals() as usize;
    assert_eq!(original_active_terminals.len(), terminal_count);

    let mut active = original_active_terminals.to_vec();
    let mut classes = singleton_partition(&active);
    let mut map_for_original = vec![None::<Arc<TransportScannerStateMap>>; terminal_count];

    for round in rounds {
        assert_eq!(
            round.active_before_round, active,
            "TI transport witness must use its historical pre-merge active mask",
        );
        for (&representative, old_representatives) in &round.partition {
            for &old_representative in old_representatives {
                if old_representative == representative {
                    continue;
                }
                let round_map = round
                    .maps
                    .get(&(representative, old_representative))
                    .unwrap_or_else(|| {
                        panic!(
                            "TI replay lost an accepted round-local transport witness: rep={} member={} active_before_round={:?}",
                            representative,
                            old_representative,
                            active,
                        )
                    });
                let original_members = classes
                    .get(&old_representative)
                    .expect("TI replay round must refer only to current representatives");
                for &original_member in original_members {
                    let original_member = original_member as usize;
                    map_for_original[original_member] = Some(
                        map_for_original[original_member]
                            .take()
                            .map(|prior_map| {
                                TransportScannerStateMap::compose(
                                    Arc::clone(round_map),
                                    prior_map,
                                )
                            })
                            .unwrap_or_else(|| Arc::clone(round_map)),
                    );
                }
            }
        }

        let next_classes = fold_one_round_partition(&classes, &round.partition);
        let next_active = active_terminals_for_partition(&round.partition, active.len());
        classes = next_classes;
        active = next_active;
    }

    assert_eq!(
        &classes, partition,
        "captured TI replay rounds must reproduce the retained final partition",
    );

    let mut modes = vec![TerminalNwaTransportMode::ordinary(state_count)];
    for (&representative, members) in partition {
        for &member in members {
            if member == representative {
                continue;
            }
            let map = map_for_original[member as usize].as_ref().unwrap_or_else(|| {
                panic!(
                    "TI replay produced no composed transport witness: rep={} member={}",
                    representative, member,
                )
            });
            modes.push(TerminalNwaTransportMode::member(
                map.as_ref().clone(),
                representative,
                member,
            ));
        }
    }
    modes
}

/// Validate the common raw scanner domain of temporary transport modes.
///
/// Older code eagerly rewrote every mode into an explicit raw-state vector of
/// ordinary quotient representatives here.  That does not change either later
/// consumer: both need exactly `Q(mode(source))`, and applying `Q` directly is
/// equivalent to first replacing the raw target by its `Q` representative.
/// Keeping the maps lazy avoids `raw lexer states × TI members` work.
pub(crate) fn canonicalize_transport_mode_states(
    modes: &[TerminalNwaTransportMode],
    ordinary_state_map: &ManyToOneIdMap,
) {
    let state_count = ordinary_state_map.original_to_internal.len();
    for mode in modes {
        assert_eq!(
            mode.scanner_state_for_original.len(),
            state_count,
            "transport mode state domain must match ordinary state quotient",
        );
    }
}

/// Refine an ordinary terminal-DWA quotient by the scanner destinations of
/// every temporary transport mode. A source raw state `s` is assigned a compact
/// transport TSID from the exact signature
/// `(Q(m_0(s)), Q(m_1(s)), …)`, where `m_0` is the ordinary mode and each later
/// mode is one target-only member reconstruction coordinate.
///
/// The full states-by-modes signature matrix is deliberately never
/// materialized.  A composed transport is constant on the source quotient of
/// its innermost round-local map.  Therefore every mode sharing that source
/// quotient is evaluated once per source class, then each raw state is refined
/// only by the small tuple of its per-round group ids.  This is the same exact
/// signature vector while avoiding the normal `raw states × TI members` path.
pub(crate) fn transport_coordinate_quotient(
    ordinary_state_map: &ManyToOneIdMap,
    modes: &[TerminalNwaTransportMode],
) -> ManyToOneIdMap {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
    assert!(
        !modes.is_empty(),
        "transport coordinate quotient needs the ordinary mode",
    );
    let state_count = ordinary_state_map.original_to_internal.len();
    let ordinary_coordinate_key = |target_state: usize| {
        let mapped = ordinary_state_map
            .original_to_internal
            .get(target_state)
            .copied()
            .unwrap_or(u32::MAX);
        // A target outside the ordinary proof domain must retain its raw
        // identity; merging all unmapped targets would be unsound.
        if mapped == u32::MAX {
            (1u64 << 32) | target_state as u64
        } else {
            mapped as u64
        }
    };

    struct ModeGroup {
        domain_key: usize,
        domain_mode: usize,
        modes: Vec<usize>,
        component_for_source_class: Vec<u32>,
    }

    #[derive(Eq, Hash, PartialEq)]
    struct SparseModeSignature {
        defaults: SmallVec<[u64; 4]>,
        deviations: SmallVec<[(u32, u64); 4]>,
    }

    struct TailGroup<'a> {
        tail: Vec<&'a TransportScannerStateMap>,
        modes: Vec<(usize, &'a TransportScannerStateMap)>,
        default_for_source_class: Vec<u64>,
    }

    // The ordinary mode is represented directly by the first coordinate below.
    // Group every target-only reconstruction by the partition on which its
    // complete composed transform is constant.
    let mut group_index_by_domain = FxHashMap::<usize, usize>::default();
    let mut groups = Vec::<ModeGroup>::new();
    for (mode_index, mode) in modes.iter().enumerate().skip(1) {
        assert_eq!(
            mode.scanner_state_for_original.len(),
            state_count,
            "transport mode state domain must match ordinary state quotient",
        );
        let domain_key = mode.scanner_state_for_original.innermost_source_domain_key();
        let group_index = match group_index_by_domain.entry(domain_key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = groups.len();
                groups.push(ModeGroup {
                    domain_key,
                    domain_mode: mode_index,
                    modes: Vec::new(),
                    component_for_source_class: Vec::new(),
                });
                entry.insert(index);
                index
            }
        };
        groups[group_index].modes.push(mode_index);
    }

    let components_started_at = profile_timing.then(Instant::now);
    let mut source_class_mode_evaluations = 0usize;
    for group in &mut groups {
        let domain = &modes[group.domain_mode].scanner_state_for_original;
        let source_class_count = domain.innermost_source_class_count();
        let mut programs = Vec::<(usize, Vec<&TransportScannerStateMap>)>::new();
        let mut supports_sparse_program = true;
        for &mode_index in &group.modes {
            let mut parts = Vec::new();
            modes[mode_index]
                .scanner_state_for_original
                .append_atomic_transforms(&mut parts);
            if parts
                .first()
                .and_then(|inner| inner.quotient_deviations())
                .is_none()
            {
                supports_sparse_program = false;
                break;
            }
            programs.push((mode_index, parts));
        }

        if !supports_sparse_program {
            let mut component_by_signature = FxHashMap::<Vec<u64>, u32>::default();
            group.component_for_source_class = Vec::with_capacity(source_class_count);
            for source_class in 0..source_class_count {
                let source_state = domain.innermost_source_representative(source_class);
                let mut signature = Vec::with_capacity(group.modes.len());
                for &mode_index in &group.modes {
                    let mode = &modes[mode_index].scanner_state_for_original;
                    let target_state = mode.scanner_state(source_state) as usize;
                    signature.push(ordinary_coordinate_key(target_state));
                }
                source_class_mode_evaluations += group.modes.len();
                let next_component = component_by_signature.len() as u32;
                let component = *component_by_signature
                    .entry(signature)
                    .or_insert(next_component);
                group.component_for_source_class.push(component);
            }
            continue;
        }

        let mut tail_group_index_by_key = FxHashMap::<Vec<usize>, usize>::default();
        let mut tail_groups = Vec::<TailGroup>::new();
        for (mode_index, parts) in programs {
            let inner = parts[0];
            debug_assert_eq!(
                inner.innermost_source_domain_key(),
                group.domain_key,
                "the first transport program atom must own this source domain",
            );
            let tail = parts[1..].to_vec();
            let tail_key = tail
                .iter()
                .map(|part| std::ptr::from_ref(*part) as usize)
                .collect::<Vec<_>>();
            let tail_group_index = match tail_group_index_by_key.entry(tail_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let index = tail_groups.len();
                    tail_groups.push(TailGroup {
                        tail,
                        modes: Vec::new(),
                        default_for_source_class: Vec::new(),
                    });
                    entry.insert(index);
                    index
                }
            };
            tail_groups[tail_group_index].modes.push((mode_index, inner));
        }

        let mut deviations_by_source_class =
            vec![SmallVec::<[(u32, u64); 4]>::new(); source_class_count];
        for tail_group in &mut tail_groups {
            tail_group.default_for_source_class = Vec::with_capacity(source_class_count);
            for source_class in 0..source_class_count {
                let mut target_state = domain.innermost_source_representative(source_class);
                for transform in &tail_group.tail {
                    target_state = transform.scanner_state(target_state);
                }
                tail_group
                    .default_for_source_class
                    .push(ordinary_coordinate_key(target_state as usize));
                source_class_mode_evaluations += 1 + tail_group.tail.len();
            }
            for &(mode_index, inner) in &tail_group.modes {
                let deviations = inner
                    .quotient_deviations()
                    .expect("sparse transport program was checked above");
                for &(input_class, output_class) in deviations {
                    let input_class = input_class as usize;
                    let output_class = output_class as usize;
                    assert!(
                        input_class < source_class_count && output_class < source_class_count,
                        "TI transport deviation must stay within its source quotient",
                    );
                    // The tail default table already contains the exact
                    // coordinate reached from every source-class representative.
                    // A sparse inner deviation merely selects a different
                    // source class, so re-running the same tail here is
                    // redundant.
                    let coordinate = tail_group.default_for_source_class[output_class];
                    if coordinate != tail_group.default_for_source_class[input_class] {
                        deviations_by_source_class[input_class]
                            .push((mode_index as u32, coordinate));
                    }
                }
            }
        }

        let mut component_by_signature = FxHashMap::<SparseModeSignature, u32>::default();
        group.component_for_source_class = Vec::with_capacity(source_class_count);
        for source_class in 0..source_class_count {
            let mut deviations = std::mem::take(&mut deviations_by_source_class[source_class]);
            deviations.sort_unstable_by_key(|&(mode, _)| mode);
            let signature = SparseModeSignature {
                defaults: tail_groups
                    .iter()
                    .map(|tail_group| tail_group.default_for_source_class[source_class])
                    .collect(),
                deviations,
            };
            let next_component = component_by_signature.len() as u32;
            let component = *component_by_signature
                .entry(signature)
                .or_insert(next_component);
            group.component_for_source_class.push(component);
        }
    }
    let component_build_ms = components_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let final_refinement_started_at = profile_timing.then(Instant::now);
    let mut class_for_signature = FxHashMap::<SmallVec<[u64; 8]>, u32>::default();
    let mut class_for_state = Vec::with_capacity(state_count);
    for source_state in 0..state_count {
        let mut signature = SmallVec::<[u64; 8]>::with_capacity(groups.len() + 1);
        signature.push(ordinary_coordinate_key(source_state));
        for group in &groups {
            let source_class = modes[group.domain_mode]
                .scanner_state_for_original
                .innermost_source_class(source_state as u32);
            signature.push(group.component_for_source_class[source_class] as u64);
        }
        let next_class = class_for_signature.len() as u32;
        let class = *class_for_signature.entry(signature).or_insert(next_class);
        class_for_state.push(class);
    }
    let final_refinement_ms = final_refinement_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

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

    let quotient = ManyToOneIdMap::from_original_to_internal_with_representatives(
        class_for_state,
        class_count as u32,
        representatives,
    );
    if profile_timing {
        let group_source_class_counts = groups
            .iter()
            .map(|group| {
                modes[group.domain_mode]
                    .scanner_state_for_original
                    .innermost_source_class_count()
            })
            .collect::<Vec<_>>();
        let group_mode_counts = groups
            .iter()
            .map(|group| group.modes.len())
            .collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][transport_coordinate_quotient] modes={} groups={} group_source_class_counts={:?} group_mode_counts={:?} source_class_mode_evaluations={} raw_state_group_lookups={} component_build_ms={:.3} final_refinement_ms={:.3} total_ms={:.3}",
            modes.len(),
            groups.len(),
            group_source_class_counts,
            group_mode_counts,
            source_class_mode_evaluations,
            state_count * groups.len(),
            component_build_ms,
            final_refinement_ms,
            total_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        );
    }
    quotient
}

/// Sparse final-coordinate lifting for post-DWA TI expansion.
///
/// The ordinary coordinate lift is shared by every member mode. A mode differs
/// only at the final transport-coordinate classes whose representative scanner
/// state lands in a different ordinary core class. BFCL p0 has thousands of
/// members but only a handful of such deviations per mode, so lifting the full
/// final-coordinate weight for every `(weight, mode)` pair is the wrong shape.
struct PostDwaWeightLifter<'a> {
    core_state_map: &'a ManyToOneIdMap,
    final_sources: Vec<u32>,
    ordinary_coordinates: Vec<u32>,
    mode_deviations: Vec<Vec<(u32, u32)>>,
    base_lifts: FxHashMap<usize, Weight>,
    mode_lifts: FxHashMap<(usize, usize), Weight>,
    // (core-weight body, exact mode-set id, entry-domain body).
    group_lifts: FxHashMap<(usize, usize, usize), Weight>,
    // Coordinate plans depend only on the exact transport-mode set, not on
    // the core state whose entry domain happens to reach that set.
    group_coordinate_plans: FxHashMap<Vec<usize>, GroupCoordinatePlan>,
    profile_base_lift_ms: f64,
    profile_plan_ms: f64,
    profile_signature_ms: f64,
    profile_override_ms: f64,
    profile_apply_ms: f64,
}

#[derive(Clone)]
struct GroupCoordinateSignature {
    base_coordinate_index: usize,
    alternate_coordinate_indices: Box<[usize]>,
}

struct GroupCoordinatePlan {
    overrides: Vec<(u32, u32)>,
    signatures: Vec<GroupCoordinateSignature>,
    coordinates: Vec<u32>,
}

fn finite_weight_token_cardinality(weight: &Weight) -> Option<u128> {
    if weight.is_full() {
        return None;
    }
    Some(
        weight
            .range_entries()
            .map(|(start, end, tokens)| {
                let tsid_count = u128::from(end) - u128::from(start) + 1;
                let token_count: u128 = tokens
                    .ranges()
                    .map(|range| u128::from(*range.end()) - u128::from(*range.start()) + 1)
                    .sum();
                tsid_count * token_count
            })
            .sum(),
    )
}

impl<'a> PostDwaWeightLifter<'a> {
    fn new(
        core_state_map: &'a ManyToOneIdMap,
        final_state_map: &ManyToOneIdMap,
        modes: &[TerminalNwaTransportMode],
        active_mode_indices: &[usize],
    ) -> Self {
        let final_sources: Vec<u32> = final_state_map.iter_representative_ids().collect();
        let ordinary_coordinates: Vec<u32> = final_sources
            .iter()
            .map(|&source| Self::core_coordinate(core_state_map, source))
            .collect();
        let mut mode_deviations = vec![Vec::new(); modes.len()];
        let mut sparse_mode_ready = vec![false; modes.len()];
        let mut direct_modes_by_default = FxHashMap::<(usize, usize), Vec<usize>>::default();
        for &mode_index in active_mode_indices {
            if mode_index == 0 {
                continue;
            }
            if let Some(default_key) = modes[mode_index]
                .scanner_state_for_original
                .direct_quotient_default_key()
            {
                direct_modes_by_default
                    .entry(default_key)
                    .or_default()
                    .push(mode_index);
            }
        }

        // A direct quotient has one shared default representative per source
        // class and only sparse member-specific class deviations. Verify that
        // the shared default preserves every final coordinate before using the
        // deviation-only path; all other transport shapes retain the full scan.
        for mode_indices in direct_modes_by_default.into_values() {
            let domain = &modes[mode_indices[0]].scanner_state_for_original;
            let default_preserves_ordinary = final_sources
                .iter()
                .enumerate()
                .all(|(final_tsid, &source)| {
                    let source_class = domain.innermost_source_class(source);
                    Self::core_coordinate(
                        core_state_map,
                        domain.innermost_source_representative(source_class),
                    ) == ordinary_coordinates[final_tsid]
                });
            if !default_preserves_ordinary {
                continue;
            }

            let mut final_entries_by_source_class =
                vec![Vec::<(u32, u32)>::new(); domain.innermost_source_class_count()];
            for (final_tsid, &source) in final_sources.iter().enumerate() {
                final_entries_by_source_class[domain.innermost_source_class(source)].push((
                    final_tsid as u32,
                    ordinary_coordinates[final_tsid],
                ));
            }
            for mode_index in mode_indices {
                let deviations = &mut mode_deviations[mode_index];
                for &(input_class, output_class) in modes[mode_index]
                    .scanner_state_for_original
                    .quotient_deviations()
                    .expect("direct quotient transport must expose deviations")
                {
                    let Some(entries) = final_entries_by_source_class.get(input_class as usize) else {
                        continue;
                    };
                    let coordinate = Self::core_coordinate(
                        core_state_map,
                        domain.innermost_source_representative(output_class as usize),
                    );
                    deviations.extend(entries.iter().filter_map(
                        |&(final_tsid, ordinary_coordinate)| {
                            (coordinate != ordinary_coordinate)
                                .then_some((final_tsid, coordinate))
                        },
                    ));
                }
                deviations.sort_unstable_by_key(|&(final_tsid, _)| final_tsid);
                sparse_mode_ready[mode_index] = true;
            }
        }

        for &mode_index in active_mode_indices {
            if mode_index == 0 || sparse_mode_ready[mode_index] {
                continue;
            }
            let mode = &modes[mode_index];
            let deviations = &mut mode_deviations[mode_index];
            for (final_tsid, &source) in final_sources.iter().enumerate() {
                let coordinate = Self::core_coordinate(
                    core_state_map,
                    mode.scanner_state_for_original.scanner_state(source),
                );
                if coordinate != ordinary_coordinates[final_tsid] {
                    deviations.push((final_tsid as u32, coordinate));
                }
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            let mut nonempty_modes = 0usize;
            let mut total_deviations = 0usize;
            let mut max_deviations = 0usize;
            for &mode_index in active_mode_indices {
                if mode_index == 0 {
                    continue;
                }
                let count = mode_deviations[mode_index].len();
                nonempty_modes += usize::from(count != 0);
                total_deviations += count;
                max_deviations = max_deviations.max(count);
            }
            eprintln!(
                "[glrmask/profile][ti_post_dwa_lifter] final_tsids={} active_modes={} nonempty_modes={} total_deviations={} max_deviations={}",
                final_sources.len(),
                active_mode_indices.len(),
                nonempty_modes,
                total_deviations,
                max_deviations,
            );
        }

        Self {
            core_state_map,
            final_sources,
            ordinary_coordinates,
            mode_deviations,
            base_lifts: FxHashMap::default(),
            mode_lifts: FxHashMap::default(),
            group_lifts: FxHashMap::default(),
            group_coordinate_plans: FxHashMap::default(),
            profile_base_lift_ms: 0.0,
            profile_plan_ms: 0.0,
            profile_signature_ms: 0.0,
            profile_override_ms: 0.0,
            profile_apply_ms: 0.0,
        }
    }

    #[inline]
    fn core_coordinate(core_state_map: &ManyToOneIdMap, raw_state: u32) -> u32 {
        core_state_map
            .original_to_internal
            .get(raw_state as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    #[inline]
    fn tokens_for_coordinate(weight: &Weight, coordinate: u32) -> SharedTokenSet {
        if coordinate == u32::MAX {
            shared_rangeset(range_set_blaze::RangeSetBlaze::new())
        } else {
            weight.shared_tokens_for_tsid(coordinate)
        }
    }

    fn base_lift(&mut self, weight: &Weight) -> Weight {
        if weight.is_empty() || weight.is_full() {
            return weight.clone();
        }
        let key = weight.ptr_key();
        if let Some(existing) = self.base_lifts.get(&key) {
            return existing.clone();
        }

        let lifted = Weight::from_per_tsid_shared(
            self.final_sources
                .iter()
                .enumerate()
                .filter_map(|(final_tsid, &source)| {
                    let coordinate = Self::core_coordinate(self.core_state_map, source);
                    (coordinate != u32::MAX).then(|| {
                        (
                            final_tsid as u32,
                            weight.shared_tokens_for_tsid(coordinate),
                        )
                    })
                }),
        );
        self.base_lifts.insert(key, lifted.clone());
        lifted
    }

    fn lift_for_mode(&mut self, weight: &Weight, mode_index: usize) -> Weight {
        if weight.is_empty() || weight.is_full() || mode_index == 0 {
            return self.base_lift(weight);
        }
        let key = (weight.ptr_key(), mode_index);
        if let Some(existing) = self.mode_lifts.get(&key) {
            return existing.clone();
        }

        let base = self.base_lift(weight);
        let overrides: Vec<(u32, SharedTokenSet)> = self.mode_deviations[mode_index]
            .iter()
            .map(|&(final_tsid, coordinate)| {
                (final_tsid, Self::tokens_for_coordinate(weight, coordinate))
            })
            .collect();
        let lifted = base.with_sparse_tsid_overrides(&overrides);
        self.mode_lifts.insert(key, lifted.clone());
        lifted
    }

    fn prepare_group_coordinate_plan(&mut self, mode_indices: &[usize]) {
        if mode_indices.len() <= 1 || self.group_coordinate_plans.contains_key(mode_indices) {
            return;
        }

        let mut alternates_by_final_tsid = vec![Vec::<u32>::new(); self.final_sources.len()];
        for &mode_index in mode_indices {
            for &(final_tsid, coordinate) in &self.mode_deviations[mode_index] {
                let alternates = &mut alternates_by_final_tsid[final_tsid as usize];
                if !alternates.contains(&coordinate) {
                    alternates.push(coordinate);
                }
            }
        }

        let mut signature_for_key = FxHashMap::<(u32, Vec<u32>), u32>::default();
        let mut signatures = Vec::<GroupCoordinateSignature>::new();
        let mut overrides = Vec::<(u32, u32)>::new();
        for (final_tsid, alternates) in alternates_by_final_tsid.iter_mut().enumerate() {
            if alternates.is_empty() {
                continue;
            }
            alternates.sort_unstable();
            alternates.dedup();
            let base_coordinate = self.ordinary_coordinates[final_tsid];
            let key = (base_coordinate, alternates.clone());
            let signature = match signature_for_key.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let signature = signatures.len() as u32;
                    let (base_coordinate, alternate_coordinates) = entry.key();
                    signatures.push(GroupCoordinateSignature {
                        base_coordinate_index: *base_coordinate as usize,
                        alternate_coordinate_indices: alternate_coordinates.iter().map(|&coordinate| coordinate as usize).collect::<Vec<_>>().into_boxed_slice(),
                    });
                    entry.insert(signature);
                    signature
                }
            };
            overrides.push((final_tsid as u32, signature));
        }

        let mut coordinates = signatures
            .iter()
            .flat_map(|signature| {
                std::iter::once(signature.base_coordinate_index as u32).chain(
                    signature
                        .alternate_coordinate_indices
                        .iter()
                        .map(|&coordinate| coordinate as u32),
                )
            })
            .collect::<Vec<_>>();
        coordinates.sort_unstable();
        coordinates.dedup();
        for signature in &mut signatures {
            let raw_base = signature.base_coordinate_index as u32;
            signature.base_coordinate_index = coordinates
                .binary_search(&raw_base)
                .expect("group-plan base coordinate must be retained");
            for coordinate_index in signature.alternate_coordinate_indices.iter_mut() {
                let raw_coordinate = *coordinate_index as u32;
                *coordinate_index = coordinates
                    .binary_search(&raw_coordinate)
                    .expect("group-plan alternate coordinate must be retained");
            }
        }

        self.group_coordinate_plans.insert(
            mode_indices.to_vec(),
            GroupCoordinatePlan {
                overrides,
                signatures,
                coordinates,
            },
        );
    }

    /// Union the exact lifted weight over one proven-disjoint transport group.
    fn lift_over_disjoint_group(
        &mut self,
        weight: &Weight,
        mode_set_id: usize,
        mode_indices: &[usize],
        entry_domain: &Weight,
    ) -> Weight {
        if weight.is_empty() || weight.is_full() {
            return weight.intersection(entry_domain);
        }
        if mode_indices.len() == 1 {
            return self
                .lift_for_mode(weight, mode_indices[0])
                .intersection(entry_domain);
        }
        let key = (weight.ptr_key(), mode_set_id, entry_domain.ptr_key());
        if let Some(existing) = self.group_lifts.get(&key) {
            return existing.clone();
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let base = self.base_lift(weight);
        if let Some(started_at) = started_at {
            self.profile_base_lift_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profile_timing.then(Instant::now);
        self.prepare_group_coordinate_plan(mode_indices);
        if let Some(started_at) = started_at {
            self.profile_plan_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profile_timing.then(Instant::now);
        let overrides = {
            let plan = self
                .group_coordinate_plans
                .get(mode_indices)
                .expect("prepared group coordinate plan must be retained");
            let mut union_for_token_signature = FxHashMap::<Vec<usize>, SharedTokenSet>::default();
            let coordinate_tokens = plan
                .coordinates
                .iter()
                .map(|&coordinate| Self::tokens_for_coordinate(weight, coordinate))
                .collect::<Vec<_>>();
            let transformed_tokens: Vec<SharedTokenSet> = plan
                .signatures
                .iter()
                .map(|signature| {
                    let mut token_sets = Vec::with_capacity(signature.alternate_coordinate_indices.len() + 1);
                    token_sets.push(Arc::clone(&coordinate_tokens[signature.base_coordinate_index]));
                    for &coordinate_index in &signature.alternate_coordinate_indices {
                        token_sets.push(Arc::clone(&coordinate_tokens[coordinate_index]));
                    }
                    let mut token_signature: Vec<usize> = token_sets
                        .iter()
                        .filter(|tokens| !tokens.is_empty())
                        .map(|tokens| Arc::as_ptr(tokens) as usize)
                        .collect();
                    token_signature.sort_unstable();
                    token_signature.dedup();
                    if let Some(tokens) = union_for_token_signature.get(&token_signature) {
                        return Arc::clone(tokens);
                    }

                    let mut tokens = shared_rangeset(range_set_blaze::RangeSetBlaze::new());
                    for candidate in token_sets {
                        if candidate.is_empty() || candidate.as_ref() == tokens.as_ref() {
                            continue;
                        }
                        tokens = shared_rangeset(tokens.as_ref() | candidate.as_ref());
                    }
                    union_for_token_signature.insert(token_signature, Arc::clone(&tokens));
                    tokens
                })
                .collect();
            plan.overrides
                .iter()
                .filter_map(|&(final_tsid, signature)| {
                    let base_tokens = base.shared_tokens_for_tsid(final_tsid);
                    let tokens = &transformed_tokens[signature as usize];
                    (tokens.as_ref() != base_tokens.as_ref())
                        .then(|| (final_tsid, Arc::clone(tokens)))
                })
                .collect::<Vec<_>>()
        };
        if let Some(started_at) = started_at {
            self.profile_signature_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let started_at = profile_timing.then(Instant::now);
        let lifted = base.with_sparse_tsid_overrides_intersection(&overrides, entry_domain);
        if let Some(started_at) = started_at {
            self.profile_apply_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        self.group_lifts.insert(key, lifted.clone());
        lifted
    }
}

/// Expand a minimized representative DWA directly into a raw-terminal DWA.
///
/// The previous implementation allocated one full suffix graph per replayed
/// member mode and asked generic DWA minimization to rediscover the common
/// representative topology. The direct construction shares that topology from
/// the start. It restores raw member labels at the dispatcher, then lifts each
/// shared suffix weight to the exact union of its reachable transported
/// final-coordinate behaviors.
pub(crate) fn expand_representative_dwa_after_minimization(
    core_dwa: &DWA,
    core_state_map: &ManyToOneIdMap,
    final_state_map: &ManyToOneIdMap,
    modes: &[TerminalNwaTransportMode],
) -> DWA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let coordinate_setup_started_at = profile_timing.then(Instant::now);
    assert!(
        !modes.is_empty(),
        "post-DWA TI expansion needs the ordinary transport mode",
    );
    assert_eq!(
        modes[0].member_reconstruction(),
        None,
        "the first TI transport mode must be ordinary",
    );

    let mut member_modes_by_representative = BTreeMap::<TerminalID, Vec<(TerminalID, usize)>>::new();
    for (mode_index, mode) in modes.iter().enumerate().skip(1) {
        let (representative, member) = mode
            .member_reconstruction()
            .expect("non-ordinary TI transport mode must reconstruct one member");
        member_modes_by_representative
            .entry(representative)
            .or_default()
            .push((member, mode_index));
    }

    let core_start = core_dwa.start_state() as usize;
    let core_states = core_dwa.states();
    let core_start_transitions = &core_states[core_start].transitions;

    let mut candidate_mode_indices = vec![0usize];
    for (mode_index, mode) in modes.iter().enumerate().skip(1) {
        let (representative, _) = mode
            .member_reconstruction()
            .expect("non-ordinary TI transport mode must reconstruct one member");
        if core_start_transitions.contains_key(&(representative as i32)) {
            candidate_mode_indices.push(mode_index);
        }
    }

    // Drop modes whose first raw-member edge has no viable token flow. These
    // clones were unreachable in the old graph-copy construction and therefore
    // must not participate in a shared suffix-weight union.
    let mut lifter = PostDwaWeightLifter::new(
        core_state_map,
        final_state_map,
        modes,
        &candidate_mode_indices,
    );
    let coordinate_setup_ms = coordinate_setup_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let active_filter_started_at = profile_timing.then(Instant::now);
    let active_mode_indices: Vec<usize> = candidate_mode_indices
        .iter()
        .copied()
        .filter(|&mode_index| {
            if mode_index == 0 {
                return true;
            }
            let (representative, _) = modes[mode_index]
                .member_reconstruction()
                .expect("non-ordinary TI transport mode must reconstruct one member");
            let (_, weight) = core_start_transitions
                .get(&(representative as i32))
                .expect("active TI member mode must have its representative start edge");
            !lifter.lift_for_mode(weight, mode_index).is_empty()
        })
        .collect();
    if active_mode_indices != candidate_mode_indices {
        lifter = PostDwaWeightLifter::new(
            core_state_map,
            final_state_map,
            modes,
            &active_mode_indices,
        );
    }
    let active_filter_ms = active_filter_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let grouping_started_at = profile_timing.then(Instant::now);
    // Modes can share one suffix graph only when their first-edge domains are
    // disjoint. That domain is retained by DWA weight intersection after the
    // dispatcher, so merging their suffix weights by union cannot leak one
    // member's transported behavior into another member's execution.
    // The ordinary representative mode remains a singleton deliberately: it
    // has multiple possible first labels rather than one stable gate.
    let mut mode_groups = vec![vec![0usize]];
    let mut entry_union_by_group = vec![Weight::empty()];
    // Exact entry unions computed while proving disjointness can be reused
    // later for suffix states with the same complete member-mode set.
    let mut known_member_entry_domain_by_mode_set = FxHashMap::<Vec<usize>, Weight>::default();
    let mut group_for_mode = vec![None::<usize>; modes.len()];
    group_for_mode[0] = Some(0);
    let member_entry_weights: Vec<(usize, u32, Weight)> = active_mode_indices
        .iter()
        .copied()
        .skip(1)
        .map(|mode_index| {
            let (representative, _) = modes[mode_index]
                .member_reconstruction()
                .expect("non-ordinary TI transport mode must reconstruct one member");
            let (target, start_weight) = core_start_transitions
                .get(&(representative as i32))
                .expect("active TI member mode must have its representative start edge");
            (mode_index, *target, lifter.lift_for_mode(start_weight, mode_index))
        })
        .collect();

    let all_member_entry_union =
        Weight::union_all(member_entry_weights.iter().map(|(_, _, weight)| weight));
    let member_entries_are_pairwise_disjoint = finite_weight_token_cardinality(&all_member_entry_union)
        .zip(
            member_entry_weights
                .iter()
                .map(|(_, _, weight)| finite_weight_token_cardinality(weight))
                .collect::<Option<Vec<_>>>()
                .map(|counts| counts.into_iter().sum()),
        )
        .is_some_and(|(union_count, member_count)| union_count == member_count);
    let ordinary_entry_weights: Vec<Weight> = core_start_transitions
        .values()
        .map(|(_, weight)| lifter.lift_for_mode(weight, 0))
        .collect();
    let ordinary_entry_union = Weight::union_all(ordinary_entry_weights.iter());

    let ordinary_and_members_are_disjoint = member_entries_are_pairwise_disjoint
        && ordinary_entry_union.is_disjoint(&all_member_entry_union);
    if ordinary_and_members_are_disjoint {
        let modes_in_group: Vec<usize> = member_entry_weights
            .iter()
            .map(|(mode_index, _, _)| *mode_index)
            .collect();
        for &mode_index in &modes_in_group {
            group_for_mode[mode_index] = Some(0);
        }
        mode_groups[0].extend(modes_in_group);
        entry_union_by_group[0] = ordinary_entry_union.union(&all_member_entry_union);
    } else if member_entries_are_pairwise_disjoint {
        let group_index = mode_groups.len();
        let modes_in_group: Vec<usize> = member_entry_weights
            .iter()
            .map(|(mode_index, _, _)| *mode_index)
            .collect();
        for &mode_index in &modes_in_group {
            group_for_mode[mode_index] = Some(group_index);
        }
        known_member_entry_domain_by_mode_set
            .insert(modes_in_group.clone(), all_member_entry_union.clone());
        mode_groups.push(modes_in_group);
        entry_union_by_group.push(all_member_entry_union);
    } else {
        for (mode_index, _, entry_weight) in &member_entry_weights {
            let group_index = (1..mode_groups.len())
                .find(|&group_index| entry_weight.is_disjoint(&entry_union_by_group[group_index]))
                .unwrap_or_else(|| {
                    let group_index = mode_groups.len();
                    mode_groups.push(Vec::new());
                    entry_union_by_group.push(Weight::empty());
                    group_index
                });
            entry_union_by_group[group_index] = entry_union_by_group[group_index].union(entry_weight);
            mode_groups[group_index].push(*mode_index);
            group_for_mode[*mode_index] = Some(group_index);
        }
        entry_union_by_group[0] = ordinary_entry_union;
    }

    let mut core_reachable_from = vec![vec![false; core_states.len()]; core_states.len()];
    for source in 0..core_states.len() {
        let mut stack = vec![source];
        while let Some(state) = stack.pop() {
            if core_reachable_from[source][state] {
                continue;
            }
            core_reachable_from[source][state] = true;
            for (target, _) in core_states[state].transitions.values() {
                if (*target as usize) < core_states.len() {
                    stack.push(*target as usize);
                }
            }
        }
    }

    let mut mode_indices_at_core_state =
        vec![vec![Vec::<usize>::new(); core_states.len()]; mode_groups.len()];
    let mut entry_weights_at_core_state =
        vec![vec![Vec::<Weight>::new(); core_states.len()]; mode_groups.len()];

    for (_, (target, weight)) in core_start_transitions {
        let entry_weight = lifter.lift_for_mode(weight, 0);
        if entry_weight.is_empty() {
            continue;
        }
        for (core_state, reachable) in core_reachable_from[*target as usize].iter().enumerate() {
            if *reachable {
                entry_weights_at_core_state[0][core_state].push(entry_weight.clone());
            }
        }
    }
    for (mode_index, target, entry_weight) in &member_entry_weights {
        let group_index = group_for_mode[*mode_index]
            .expect("active TI member mode must belong to one direct suffix group");
        for (core_state, reachable) in core_reachable_from[*target as usize].iter().enumerate() {
            if *reachable {
                mode_indices_at_core_state[group_index][core_state].push(*mode_index);
                entry_weights_at_core_state[group_index][core_state].push(entry_weight.clone());
            }
        }
    }
    for core_state in 0..core_states.len() {
        if !entry_weights_at_core_state[0][core_state].is_empty() {
            mode_indices_at_core_state[0][core_state].push(0);
        }
    }
    // Member entry domains are determined solely by the exact active mode
    // set. Several p0 suffix states share the complete 1,609-member set, so
    // retain one exact union rather than rebuilding it per core state.
    let mut member_entry_domain_by_mode_set = known_member_entry_domain_by_mode_set;
    let entry_domain_at_core_state: Vec<Vec<Weight>> = entry_weights_at_core_state
        .iter()
        .enumerate()
        .map(|(group_index, by_core_state)| {
            by_core_state
                .iter()
                .enumerate()
                .map(|(core_state, weights)| {
                    if group_index == 0 {
                        return Weight::union_all(weights.iter());
                    }
                    let mode_set = &mode_indices_at_core_state[group_index][core_state];
                    if mode_set.is_empty() {
                        return Weight::empty();
                    }
                    member_entry_domain_by_mode_set
                        .entry(mode_set.clone())
                        .or_insert_with(|| Weight::union_all(weights.iter()))
                        .clone()
                })
                .collect()
        })
        .collect();

    let mut mode_set_id_by_contents = FxHashMap::<Vec<usize>, usize>::default();
    let mode_set_id_at_core_state: Vec<Vec<Option<usize>>> = mode_indices_at_core_state
        .iter()
        .map(|by_core_state| {
            by_core_state
                .iter()
                .map(|mode_set| {
                    if mode_set.is_empty() {
                        return None;
                    }
                    let next = mode_set_id_by_contents.len();
                    Some(*mode_set_id_by_contents
                        .entry(mode_set.clone())
                        .or_insert(next))
                })
                .collect()
        })
        .collect();

    let grouping_ms = grouping_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if profile_timing {
        let member_group_count = mode_groups.len().saturating_sub(1);
        let largest_member_group = mode_groups
            .iter()
            .skip(1)
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][ti_post_dwa_direct_groups] core_states={} active_modes={} member_groups={} largest_member_group={} direct_states_before_follow={}",
            core_states.len(),
            active_mode_indices.len(),
            member_group_count,
            largest_member_group,
            1 + mode_groups.len() * core_states.len(),
        );
    }

    let state_for = |group_index: usize, core_state_index: usize| -> u32 {
        (1 + group_index * core_states.len() + core_state_index) as u32
    };
    let shared_build_started_at = profile_timing.then(Instant::now);
    let mut lift_ms = 0.0;
    let mut raw_label_insert_ms = 0.0;
    let mut lift_calls = 0usize;
    let mut raw_label_inserts = 0usize;
    let mut states = vec![DWAState::default(); 1 + mode_groups.len() * core_states.len()];
    for group_index in 0..mode_groups.len() {
        for (core_state_index, core_state) in core_states.iter().enumerate() {
            let mode_indices = &mode_indices_at_core_state[group_index][core_state_index];
            let entry_domain = &entry_domain_at_core_state[group_index][core_state_index];
            if mode_indices.is_empty() || entry_domain.is_empty() {
                continue;
            }
            let mode_set_id = mode_set_id_at_core_state[group_index][core_state_index]
                .expect("active suffix state must have a mode-set id");
            let final_weight = core_state.final_weight.as_ref().map(|weight| {
                let started_at = profile_timing.then(Instant::now);
                let lifted = lifter.lift_over_disjoint_group(weight, mode_set_id, mode_indices, entry_domain);
                if let Some(started_at) = started_at {
                    lift_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                }
                lift_calls += 1;
                lifted
            });
            let mut transitions = BTreeMap::new();
            for (&label, (target, weight)) in &core_state.transitions {
                let lift_started_at = profile_timing.then(Instant::now);
                let lifted_weight = lifter.lift_over_disjoint_group(
                    weight,
                    mode_set_id,
                    mode_indices,
                    entry_domain,
                );
                if let Some(started_at) = lift_started_at {
                    lift_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                }
                lift_calls += 1;
                if lifted_weight.is_empty() {
                    continue;
                }
                let destination = state_for(group_index, *target as usize);
                assert!(
                    transitions
                        .insert(label, (destination, lifted_weight.clone()))
                        .is_none(),
                    "representative DWA must be deterministic before TI expansion",
                );

                // First terminal selection is handled only by the dispatcher:
                // its member-specific transport determines the entire suffix.
                if core_state_index == core_start || label < 0 {
                    continue;
                }
                let representative = label as TerminalID;
                for &(member, _) in member_modes_by_representative
                    .get(&representative)
                    .into_iter()
                    .flatten()
                {
                    let raw_label_started_at = profile_timing.then(Instant::now);
                    assert!(
                        transitions
                            .insert(member as i32, (destination, lifted_weight.clone()))
                            .is_none(),
                        "one raw member must belong to exactly one TI representative class",
                    );
                    if let Some(started_at) = raw_label_started_at {
                        raw_label_insert_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    raw_label_inserts += 1;
                }
            }
            states[state_for(group_index, core_state_index) as usize] = DWAState {
                transitions,
                final_weight,
            };
        }
    }

    let ordinary_start = &core_states[core_start];
    let mut dispatcher_transitions = BTreeMap::new();
    for (&label, (target, weight)) in &ordinary_start.transitions {
        let ordinary_weight = lifter.lift_for_mode(weight, 0);
        if ordinary_weight.is_empty() {
            continue;
        }
        assert!(
            dispatcher_transitions
                .insert(label, (state_for(0, *target as usize), ordinary_weight))
                .is_none(),
            "representative initial DWA edge labels must be unique",
        );
    }
    for (representative, member_modes) in &member_modes_by_representative {
        let Some((target, weight)) = core_start_transitions.get(&(*representative as i32)) else {
            continue;
        };
        for &(member, mode_index) in member_modes {
            let Some(group_index) = group_for_mode[mode_index] else {
                continue;
            };
            let member_weight = lifter.lift_for_mode(weight, mode_index);
            if member_weight.is_empty() {
                continue;
            }
            assert!(
                dispatcher_transitions
                    .insert(
                        member as i32,
                        (state_for(group_index, *target as usize), member_weight),
                    )
                    .is_none(),
                "one raw member must belong to exactly one TI representative class",
            );
        }
    }
    states[0] = DWAState {
        transitions: dispatcher_transitions,
        final_weight: ordinary_start
            .final_weight
            .as_ref()
            .map(|weight| lifter.lift_for_mode(weight, 0)),
    };

    if let Some(started_at) = shared_build_started_at {
        eprintln!(
            "[glrmask/profile][ti_post_dwa_direct_detail] coordinate_setup_ms={:.3} active_filter_ms={:.3} grouping_ms={:.3} shared_build_ms={:.3} lift_calls={} lift_ms={:.3} raw_label_inserts={} raw_label_insert_ms={:.3} base_lifts={} mode_lifts={} group_lifts={} coordinate_plans={} group_base_ms={:.3} group_plan_ms={:.3} group_signature_ms={:.3} group_apply_ms={:.3}",
            coordinate_setup_ms,
            active_filter_ms,
            grouping_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
            lift_calls,
            lift_ms,
            raw_label_inserts,
            raw_label_insert_ms,
            lifter.base_lifts.len(),
            lifter.mode_lifts.len(),
            lifter.group_lifts.len(),
            lifter.group_coordinate_plans.len(),
            lifter.profile_base_lift_ms,
            lifter.profile_plan_ms,
            lifter.profile_signature_ms,
            lifter.profile_apply_ms,
        );
    }

    DWA::from_parts(states, 0)
}

/// Restore the original raw-terminal follow relation after building and
/// minimizing a representative-only core. This is the deterministic product of
/// the expanded DWA with the same one-previous-terminal guard used by the NWA
/// postprocess. It deliberately performs no further NWA construction,
/// determinization, or minimization.
pub(crate) struct RawFollowRestoration {
    pub(crate) dwa: DWA,
    pub(crate) used_follow_row_quotient: bool,
}

pub(crate) fn restore_raw_follow_constraints_after_expansion(
    expanded_dwa: &DWA,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
    ignore_terminal: Option<TerminalID>,
) -> RawFollowRestoration {
    // Every query below is for an in-range raw terminal. Borrow the original
    // rows instead of normalizing by cloning `num_terminals` full bitsets. The
    // former normalization was several milliseconds by itself on p0/p1 even
    // when the ensuing product was tiny.
    let rows_for_terminal: Vec<Option<&BitSet>> = (0..num_terminals)
        .map(|terminal| {
            disallowed_follows
                .get(&(terminal as u32))
                .filter(|row| !row.is_zero())
        })
        .collect();
    if !rows_for_terminal.iter().flatten().any(|row| {
        row.iter().any(|disallowed| disallowed < num_terminals)
    }) {
        return RawFollowRestoration {
            dwa: expanded_dwa.clone(),
            used_follow_row_quotient: false,
        };
    }

    // A previous terminal is observed only through its follow row. On the
    // tiny direct suffix shapes this exact quotient avoids constructing one
    // otherwise identical product state per raw terminal. Retain raw
    // predecessor identity on larger shapes because pointwise minimization's
    // canonical representative choice is intentionally shape-sensitive there.
    let follow_row_class_count = rows_for_terminal
        .iter()
        .copied()
        .collect::<FxHashSet<_>>()
        .len();
    // This exact quotient is particularly valuable for compact direct suffix
    // graphs.  Bound the product shape so large general TI artifacts retain
    // their historical canonicalization path.
    let use_follow_row_quotient = expanded_dwa.states().len() <= 24
        && follow_row_class_count <= 64
        && expanded_dwa.states().len() * follow_row_class_count <= 1200;
    let (previous_key_for_terminal, follow_rows) = if use_follow_row_quotient {
        let mut class_for_row = FxHashMap::<Option<&BitSet>, u32>::default();
        let mut rows = Vec::<Option<&BitSet>>::new();
        let keys = rows_for_terminal
            .iter()
            .copied()
            .map(|row| {
                let next = rows.len() as u32;
                *class_for_row.entry(row).or_insert_with(|| {
                    rows.push(row);
                    next
                })
            })
            .collect::<Vec<_>>();
        (keys, rows)
    } else {
        ((0..num_terminals as u32).collect(), rows_for_terminal)
    };

    type ProductState = (u32, Option<u32>);
    let mut state_ids = FxHashMap::<ProductState, u32>::default();
    let mut worklist = VecDeque::<ProductState>::new();
    let mut states = Vec::<DWAState>::new();

    let get_or_create = |product: ProductState,
                         state_ids: &mut FxHashMap<ProductState, u32>,
                         worklist: &mut VecDeque<ProductState>,
                         states: &mut Vec<DWAState>| {
        if let Some(&id) = state_ids.get(&product) {
            return id;
        }
        let id = states.len() as u32;
        state_ids.insert(product, id);
        worklist.push_back(product);
        states.push(DWAState::default());
        id
    };

    let start = get_or_create(
        (expanded_dwa.start_state(), None),
        &mut state_ids,
        &mut worklist,
        &mut states,
    );
    while let Some((dwa_state, previous_key)) = worklist.pop_front() {
        let result_state = state_ids[&(dwa_state, previous_key)] as usize;
        let source = &expanded_dwa.states()[dwa_state as usize];
        states[result_state].final_weight = source.final_weight.clone();

        for (&label, (target, weight)) in &source.transitions {
            let next_previous_key = if label < 0
                || ignore_terminal.is_some_and(|ignore| label as TerminalID == ignore)
            {
                previous_key
            } else if (label as usize) < previous_key_for_terminal.len() {
                let terminal = label as usize;
                if previous_key.is_some_and(|previous| {
                    follow_rows[previous as usize].is_some_and(|row| row.contains(terminal))
                }) {
                    continue;
                }
                Some(previous_key_for_terminal[terminal])
            } else {
                None
            };
            let destination = get_or_create(
                (*target, next_previous_key),
                &mut state_ids,
                &mut worklist,
                &mut states,
            );
            let previous = states[result_state]
                .transitions
                .insert(label, (destination, weight.clone()));
            assert!(
                previous.is_none(),
                "expanded DWA and raw follow product must remain deterministic",
            );
        }
    }

    RawFollowRestoration {
        dwa: DWA::from_parts(states, start),
        used_follow_row_quotient: use_follow_row_quotient,
    }
}


/// Return a topological state order when every in-range transition is acyclic.
///
/// This small local helper intentionally mirrors the minimizer's Kahn pass
/// instead of exposing a general graph API just for post-DWA normalization.
fn forward_domain_topological_order(dwa: &DWA) -> Option<Vec<usize>> {
    let state_count = dwa.states().len();
    let mut indegree = vec![0u32; state_count];
    for state in dwa.states() {
        for (_, (target, _)) in &state.transitions {
            let target = *target as usize;
            if target < state_count {
                indegree[target] = indegree[target].saturating_add(1);
            }
        }
    }

    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state))
        .collect::<Vec<_>>();
    let mut head = 0usize;
    let mut order = Vec::with_capacity(state_count);
    while head < queue.len() {
        let source = queue[head];
        head += 1;
        order.push(source);
        for (_, (target, _)) in &dwa.states()[source].transitions {
            let target = *target as usize;
            if target >= state_count {
                continue;
            }
            indegree[target] -= 1;
            if indegree[target] == 0 {
                queue.push(target);
            }
        }
    }
    (order.len() == state_count).then_some(order)
}

/// Exact intersection cache scoped to one forward-domain normalization pass.
///
/// The global Weight operation memo protects correctness and reuse across the
/// compiler, but this tight pass repeatedly intersects the same weight bodies
/// with the same per-state domains. Keeping strong local results avoids weak
/// interner/memo traffic and also shares the final restriction rewrite.
fn forward_domain_intersection(
    cache: &mut FxHashMap<(usize, usize), Weight>,
    left: &Weight,
    right: &Weight,
) -> Weight {
    if left.is_empty() || right.is_empty() {
        return Weight::empty();
    }
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() || left.ptr_key() == right.ptr_key() {
        return left.clone();
    }
    let (left_key, right_key) = if left.ptr_key() <= right.ptr_key() {
        (left.ptr_key(), right.ptr_key())
    } else {
        (right.ptr_key(), left.ptr_key())
    };
    cache
        .entry((left_key, right_key))
        .or_insert_with(|| left.intersection(right))
        .clone()
}

/// Collapse a target's pending domain contributions once, at the point where a
/// topological traversal proves all predecessors have contributed. Pointer
/// deduplication is exact because each Weight body is immutable.
fn union_forward_domain_parts(parts: &mut Vec<Weight>) -> Weight {
    if parts.is_empty() {
        return Weight::empty();
    }
    if parts.iter().any(Weight::is_full) {
        parts.clear();
        return Weight::all();
    }
    parts.retain(|weight| !weight.is_empty());
    if parts.is_empty() {
        return Weight::empty();
    }
    if parts.len() == 1 {
        return parts.pop().expect("one retained forward-domain part");
    }
    parts.sort_unstable_by_key(Weight::ptr_key);
    parts.dedup_by_key(|weight| weight.ptr_key());
    if parts.len() == 1 {
        return parts.pop().expect("one deduplicated forward-domain part");
    }
    Weight::union_all(parts.iter())
}

/// Propagate forward domains and rewrite the product DWA in one exact
/// topological pass. Every incoming contribution is the same intersection
/// that becomes the normalized transition weight, so retaining it avoids
/// computing `source_domain ∩ transition_weight` twice.
fn normalize_weights_to_forward_domains_acyclic_in_place(dwa: &mut DWA) -> bool {
    let state_count = dwa.states().len();
    let Some(order) = forward_domain_topological_order(dwa) else {
        return false;
    };
    let start = dwa.start_state() as usize;
    if start >= state_count {
        return true;
    }

    let mut pending = vec![Vec::<Weight>::new(); state_count];
    pending[start].push(Weight::all());
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();

    for source in order {
        let source_domain = union_forward_domain_parts(&mut pending[source]);
        let state = &mut dwa.states_mut()[source];
        if source_domain.is_empty() {
            state.final_weight = None;
            state.transitions.clear();
            continue;
        }

        state.final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| forward_domain_intersection(&mut intersection_cache, weight, &source_domain))
            .filter(|weight| !weight.is_empty());
        state.transitions.retain(|_, (target, weight)| {
            let normalized = forward_domain_intersection(
                &mut intersection_cache,
                weight,
                &source_domain,
            );
            let target = *target as usize;
            if target < state_count && !normalized.is_empty() {
                pending[target].push(normalized.clone());
            }
            *weight = normalized;
            !weight.is_empty()
        });
    }

    true
}

fn forward_domains_fixed_point(dwa: &DWA) -> Vec<Weight> {
    let state_count = dwa.states().len();
    let mut domains = vec![Weight::empty(); state_count];
    let mut worklist = VecDeque::new();
    let start = dwa.start_state() as usize;
    domains[start] = Weight::all();
    worklist.push_back(start);
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();
    while let Some(source) = worklist.pop_front() {
        let source_domain = domains[source].clone();
        for (target, weight) in dwa.states()[source].transitions.values() {
            let target = *target as usize;
            if target >= state_count {
                continue;
            }
            let incoming = forward_domain_intersection(
                &mut intersection_cache,
                &source_domain,
                weight,
            );
            if incoming.is_empty() {
                continue;
            }
            let merged = domains[target].union(&incoming);
            if merged != domains[target] {
                domains[target] = merged;
                worklist.push_back(target);
            }
        }
    }
    domains
}

/// Restrict each final/transition weight to coordinates that can reach its
/// source state from the DWA start, mutating an already-owned product DWA.
/// This preserves every completed path while dropping unreachable
/// transport-factor fragments before minimization without cloning the full
/// raw-follow state vector a second time.
pub(crate) fn restrict_weights_to_forward_domains_in_place(dwa: &mut DWA) {
    let state_count = dwa.states().len();
    if state_count == 0 || (dwa.start_state() as usize) >= state_count {
        return;
    }

    if normalize_weights_to_forward_domains_acyclic_in_place(dwa) {
        return;
    }

    let domains = forward_domains_fixed_point(dwa);
    let mut restriction_cache = FxHashMap::<(usize, usize), Weight>::default();
    for (state, domain) in dwa.states_mut().iter_mut().zip(domains.iter()) {
        state.final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| forward_domain_intersection(&mut restriction_cache, weight, domain))
            .filter(|weight| !weight.is_empty());
        state.transitions.retain(|_, (_, weight)| {
            *weight = forward_domain_intersection(&mut restriction_cache, weight, domain);
            !weight.is_empty()
        });
    }
}

/// Owned convenience wrapper for tests and callers that need to retain the
/// unnormalized input DWA.
pub(crate) fn restrict_weights_to_forward_domains(dwa: &DWA) -> DWA {
    let mut normalized = dwa.clone();
    restrict_weights_to_forward_domains_in_place(&mut normalized);
    normalized
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
    fn partition_invariants_hide_only_merged_active_members() {
        let active = [true, false, true, true, false];
        let partition = BTreeMap::from([
            (0, BTreeSet::from([0, 2])),
            (3, BTreeSet::from([3])),
        ]);
        assert_partition_invariants(&partition, &active);
        assert_eq!(active_terminals_for_partition(&partition, active.len()), [true, false, false, true, false]);
        assert_eq!(visible_output_raw_labels(&partition, active.len()), [true, true, false, true, true]);
    }

    #[test]
    fn sparse_coalesced_disallowed_follows_matches_direct_member_pair_predicate() {
        fn direct(
            partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
            original: &BTreeMap<u32, BitSet>,
            num_terminals: usize,
        ) -> BTreeMap<u32, BitSet> {
            let mut result = BTreeMap::new();
            for (&predecessor_representative, predecessors) in partition {
                let mut bits = BitSet::new(num_terminals);
                for (&successor_representative, successors) in partition {
                    if predecessors.iter().all(|&predecessor| {
                        successors.iter().all(|&successor| {
                            original
                                .get(&(predecessor as u32))
                                .is_some_and(|bits| bits.contains(successor as usize))
                        })
                    }) {
                        bits.set(successor_representative as usize);
                    }
                }
                result.insert(predecessor_representative as u32, bits);
            }
            result
        }

        let partition = BTreeMap::from([
            (0, BTreeSet::from([0, 1])),
            (2, BTreeSet::from([2, 3])),
            (4, BTreeSet::from([4])),
            (5, BTreeSet::from([5])),
        ]);
        let mut original = BTreeMap::new();
        let row = |successors: &[usize]| {
            let mut bits = BitSet::new(6);
            for &successor in successors {
                bits.set(successor);
            }
            bits
        };
        original.insert(0, row(&[2, 3, 4]));
        original.insert(1, row(&[2, 3]));
        original.insert(2, row(&[0, 1]));
        original.insert(3, row(&[0]));

        assert_eq!(
            coalesced_disallowed_follows(&partition, &original, 6),
            direct(&partition, &original, 6),
        );
    }

    #[test]
    fn folding_memberships_requires_only_current_classes_and_round() {
        let active = [true, true, true, true];
        let initial = singleton_partition(&active);
        let first_round = BTreeMap::from([
            (0, BTreeSet::from([0, 1])),
            (2, BTreeSet::from([2])),
            (3, BTreeSet::from([3])),
        ]);
        let after_first = fold_one_round_partition(&initial, &first_round);
        let second_round = BTreeMap::from([
            (0, BTreeSet::from([0, 2])),
            (3, BTreeSet::from([3])),
        ]);
        let final_classes = fold_one_round_partition(&after_first, &second_round);

        assert_eq!(
            final_classes,
            BTreeMap::from([
                (0, BTreeSet::from([0, 1, 2])),
                (3, BTreeSet::from([3])),
            ]),
        );
        assert_partition_invariants(&final_classes, &active);
    }

    #[test]
    fn transport_coordinate_quotient_matches_target_only_mode_signature() {
        let ordinary = ManyToOneIdMap::from_original_to_internal_with_representatives(
            vec![0, 0, 1, 1, 2, 2, 3],
            4,
            vec![0, 2, 4, 6],
        );
        let mut modes = vec![
            TerminalNwaTransportMode::ordinary(7),
            TerminalNwaTransportMode::member(
                TransportScannerStateMap::Explicit(vec![1, 0, 3, 2, 5, 4, 6].into()),
                0,
                1,
            ),
            TerminalNwaTransportMode::member(
                TransportScannerStateMap::Explicit(vec![2, 3, 0, 1, 6, 6, 4].into()),
                2,
                3,
            ),
        ];

        let expected_signatures = (0..ordinary.original_to_internal.len())
            .map(|source| {
                modes
                    .iter()
                    .map(|mode| {
                        ordinary.original_to_internal
                            [mode.scanner_state_for_original.scanner_state(source as u32) as usize]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quotient = transport_coordinate_quotient(&ordinary, &modes);
        for left in 0..expected_signatures.len() {
            for right in 0..expected_signatures.len() {
                assert_eq!(
                    quotient.original_to_internal[left] == quotient.original_to_internal[right],
                    expected_signatures[left] == expected_signatures[right],
                    "signature quotient disagreed for states {left} and {right}",
                );
            }
        }

        canonicalize_transport_mode_states(&mut modes, &ordinary);
        let canonical_quotient = transport_coordinate_quotient(&ordinary, &modes);
        for left in 0..expected_signatures.len() {
            for right in 0..expected_signatures.len() {
                assert_eq!(
                    canonical_quotient.original_to_internal[left]
                        == canonical_quotient.original_to_internal[right],
                    expected_signatures[left] == expected_signatures[right],
                    "canonical target-only transport changed the quotient for states {left} and {right}",
                );
            }
        }
    }

    #[test]
    fn post_dwa_member_expansion_reads_representative_weight_at_transport_target() {
        let state_map = ManyToOneIdMap::from_original_to_internal_with_representatives(
            vec![0, 1],
            2,
            vec![0, 1],
        );
        let member_map = TransportScannerStateMap::Explicit(vec![1, 0].into());
        let modes = vec![
            TerminalNwaTransportMode::ordinary(2),
            TerminalNwaTransportMode::member(member_map, 0, 1),
        ];
        let start_weight = Weight::from_uniform(
            0..=1,
            range_set_blaze::RangeSetBlaze::from_iter([10..=10, 20..=20]),
        );
        let suffix_weight = Weight::from_per_tsid_token_sets([
            (0, range_set_blaze::RangeSetBlaze::from_iter([10..=10])),
            (1, range_set_blaze::RangeSetBlaze::from_iter([20..=20])),
        ]);
        let core = DWA::from_parts(
            vec![
                DWAState {
                    transitions: BTreeMap::from([(0, (1, start_weight))]),
                    final_weight: None,
                },
                DWAState {
                    transitions: BTreeMap::from([(2, (2, suffix_weight))]),
                    final_weight: None,
                },
                DWAState {
                    transitions: BTreeMap::new(),
                    final_weight: Some(Weight::all()),
                },
            ],
            0,
        );

        let expanded = expand_representative_dwa_after_minimization(
            &core,
            &state_map,
            &state_map,
            &modes,
        );
        let member_word = expanded.eval_word(&[1, 2]);
        let ordinary_word = expanded.eval_word(&[0, 2]);
        assert!(member_word.tokens_for_tsid(0).contains(20));
        assert!(member_word.tokens_for_tsid(1).contains(10));
        assert!(
            !member_word.tokens_for_tsid(0).contains(10),
            "the member's suffix must use its transported representative coordinate",
        );
        assert!(ordinary_word.tokens_for_tsid(0).contains(10));
        assert!(!ordinary_word.tokens_for_tsid(0).contains(20));
    }

    #[test]
    fn forward_domain_normalization_removes_unreachable_weight_coordinates() {
        let reachable = Weight::from_uniform(
            0..=0,
            range_set_blaze::RangeSetBlaze::from_iter([7..=7]),
        );
        let source = DWAState {
            transitions: BTreeMap::from([(10, (1, reachable.clone()))]),
            final_weight: None,
        };
        let middle = DWAState {
            transitions: BTreeMap::from([(11, (2, Weight::all()))]),
            final_weight: Some(Weight::all()),
        };
        let final_state = DWAState {
            transitions: BTreeMap::new(),
            final_weight: Some(Weight::all()),
        };
        let before = DWA::from_parts(vec![source, middle, final_state], 0);
        let after = restrict_weights_to_forward_domains(&before);

        assert_eq!(after.eval_word(&[10]), before.eval_word(&[10]));
        assert_eq!(after.eval_word(&[10, 11]), before.eval_word(&[10, 11]));
        assert_eq!(after.states()[1].final_weight.as_ref(), Some(&reachable));
        assert_eq!(after.states()[1].transitions.get(&11).unwrap().1, reachable);
    }

    #[test]
    fn forward_domain_normalization_converges_on_cycles() {
        let reachable = Weight::from_uniform(
            0..=0,
            range_set_blaze::RangeSetBlaze::from_iter([7..=7]),
        );
        let before = DWA::from_parts(
            vec![
                DWAState {
                    transitions: BTreeMap::from([(10, (1, reachable.clone()))]),
                    final_weight: None,
                },
                DWAState {
                    transitions: BTreeMap::from([(11, (1, Weight::all()))]),
                    final_weight: Some(Weight::all()),
                },
            ],
            0,
        );
        let after = restrict_weights_to_forward_domains(&before);

        assert_eq!(after.eval_word(&[10]), before.eval_word(&[10]));
        assert_eq!(after.eval_word(&[10, 11, 11]), before.eval_word(&[10, 11, 11]));
        assert_eq!(after.states()[1].final_weight.as_ref(), Some(&reachable));
        assert_eq!(after.states()[1].transitions.get(&11).unwrap().1, reachable);
    }

    #[test]
    fn iterative_discovery_stops_at_the_first_stable_round() {
        // Distinct literals whose alphabetic interiors are unobserved in this
        // punctuation-only L2P byte partition. The first exact round merges
        // them; the next single-representative round is the fixed point.
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"CREATE\"".to_vec()),
            Expr::U8Seq(b"CrossFit\"".to_vec()),
            Expr::U8Seq(b"DELETE\"".to_vec()),
            Expr::U8Seq(b"Drums\"".to_vec()),
        ]);
        let mut active = vec![true; 4];
        let mut classes = singleton_partition(&active);
        let mut rounds = 0usize;
        let mut punctuation_only = [false; 256];
        punctuation_only[b'"' as usize] = true;
        loop {
            let round = discover_one_round(&tokenizer, &active, &punctuation_only, None);
            let next_active = active_terminals_for_partition(&round, active.len());
            classes = fold_one_round_partition(&classes, &round);
            rounds += 1;
            if next_active == active {
                break;
            }
            active = next_active;
        }

        assert_eq!(rounds, 2);
        assert_eq!(active, vec![true, false, false, false]);
        assert_eq!(classes, BTreeMap::from([(0, BTreeSet::from([0, 1, 2, 3]))]));
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
        assert_eq!(map.scanner_state_map.scanner_state(root as u32), tokenizer.initial_state_id());
        let representatives = map.materialized_scanner_states();
        assert_eq!(map.scanner_state_map.scanner_state(root as u32), representatives[root]);
        let partition = discover_one_round(&tokenizer, &[true, true], &[true; 256], None);
        assert_eq!(partition, BTreeMap::from([(0, BTreeSet::from([0, 1]))]));
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
        let partition = discover_one_round(
            &tokenizer,
            &[true, true, true, true],
            &punctuation_only,
            None,
        );
        assert_eq!(partition, BTreeMap::from([(0, BTreeSet::from([0, 1, 2, 3]))]));
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
                .materialized_scanner_states()
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
        let topology = RestrictedTopology::new(&tokenizer, &only_x, None);
        let (root_groups, _) = rooted_candidate_groups(&tokenizer, &candidates, &topology);
        let (structural_signatures, _) =
            structural_candidate_signatures(
                &tokenizer,
                &active,
                &candidates,
                &topology,
                STRUCTURAL_REFINEMENT_ROUNDS,
            );
        let filtered_groups =
            refine_candidate_groups_by_structure(root_groups, &candidates, &structural_signatures);
        assert!(group_contains_pair(&filtered_groups, 0, 1));

        let mut dfa = InterchangeabilityDfa::new(&tokenizer, &active, &only_x);
        assert!(dfa.interchange_map(0, 1).is_some());
        let partition = discover_one_round(&tokenizer, &active, &only_x, None);
        assert_eq!(partition, BTreeMap::from([(0, BTreeSet::from([0, 1]))]));
    }

    #[test]
    fn literal_fiber_certificate_matches_generic_for_duplicate_literals() {
        let tokenizer = tokenizer((0..8).map(|_| Expr::U8Seq(b"same".to_vec())).collect());
        let active = vec![true; 8];
        let relevant = [true; 256];
        let context = TiDiscoveryContext::new(&tokenizer, &relevant, None);
        let certificate = discover_literal_fiber_pre_certificate_round(
            &tokenizer,
            &active,
            &context,
            None,
        )
        .expect("duplicate literal family must certify");
        let generic = discover_one_round_with_transport_witnesses_in_context(
            &tokenizer,
            &active,
            &context,
            None,
        );
        assert_eq!(certificate.partition, generic.partition);
        assert_eq!(certificate.maps.len(), 7);
        for member in 1..8 {
            assert!(certificate.maps.contains_key(&(0, member)));
        }
    }

    #[test]
    fn literal_fiber_certificate_rejects_mixed_finalizer_shapes() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ]);
        let active = vec![true; 8];
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        let context = TiDiscoveryContext::new(&tokenizer, &relevant, None);
        assert!(discover_literal_fiber_pre_certificate_round(
            &tokenizer,
            &active,
            &context,
            None,
        )
        .is_none());
        let generic = discover_one_round_with_transport_witnesses_in_context(
            &tokenizer,
            &active,
            &context,
            None,
        );
        assert!(generic.partition.values().any(|class| class.len() > 1));
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
        let topology = RestrictedTopology::new(&tokenizer, &relevant_bytes, None);
        let (root_groups, _) = rooted_candidate_groups(&tokenizer, &candidates, &topology);
        let (structural_signatures, _) = structural_candidate_signatures(
            &tokenizer,
            &active,
            &candidates,
            &topology,
            STRUCTURAL_REFINEMENT_ROUNDS,
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
                        left_to_right.materialized_scanner_states(),
                        right_to_left.materialized_scanner_states(),
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
