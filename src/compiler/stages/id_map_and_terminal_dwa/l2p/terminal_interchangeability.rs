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
use crate::compiler::stages::equiv_types::{GlobalScannerStateQuotient, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::state_equivalence::global_token_position::GlobalTokenPositionStatePartition;
use crate::ds::weight::{SharedTokenSet, Weight, shared_rangeset};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

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

/// The stable identity quotient data needed by the common support-transport
/// proof.  It deliberately omits canonical row signatures and historical
/// round projections, which are required only by the slower generic fallback.
struct SupportQuotient {
    class_for_state: Arc<[u32]>,
    representative_by_class: Arc<[u32]>,
    reverse_predecessors: Vec<Vec<u32>>,
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

/// Sparse topology of the byte-restricted tokenizer DFA. Its real-state domain
/// is either the raw tokenizer state domain or a global scanner-state
/// quotient. A missing enabled transition has the same synthetic dead
/// destination for every real state. Keeping only real edges is exact: all
/// omitted bytes share that one default.
#[derive(Debug)]
struct RestrictedTopology {
    bytes: Vec<u8>,
    edge_offsets: Vec<u32>,
    edges: Vec<(u8, u32)>,
    /// Number of raw tokenizer states. Transport witnesses must retain this
    /// external scanner coordinate even when discovery runs on a quotient.
    raw_state_count: usize,
    /// Raw scanner state selected for each discovery-domain state.
    raw_representative_by_state: Option<Arc<[u32]>>,
    /// Discovery-domain state for every raw scanner state.
    state_for_raw: Option<Arc<[u32]>>,
    real_state_count: usize,
    initial_state: usize,
    max_outdegree: usize,
}

impl RestrictedTopology {
    fn new(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        Self::new_inner(tokenizer, relevant_bytes, None)
    }

    /// Build a restricted topology over a global scanner-state quotient.
    /// The quotient must be a total right congruence for the selected bytes
    /// and must preserve both frozen output families.
    fn new_with_global_state_quotient(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: &GlobalScannerStateQuotient,
    ) -> Self {
        Self::new_inner(tokenizer, relevant_bytes, Some(global_state_quotient))
    }

    fn new_inner(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: Option<&GlobalScannerStateQuotient>,
    ) -> Self {
        let bytes = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect::<Vec<_>>();
        let raw_state_count = tokenizer.num_states() as usize;
        let (state_for_raw, raw_representative_by_state): (
            Option<Arc<[u32]>>,
            Option<Arc<[u32]>>,
        ) =
            match global_state_quotient {
                Some(global_state_quotient) => {
                    assert_eq!(
                        global_state_quotient.raw_state_count(),
                        raw_state_count,
                        "global scanner-state quotient must match the tokenizer state domain",
                    );
                    let map = global_state_quotient.as_many_to_one();
                    assert_eq!(
                        map.original_to_internal.len(),
                        raw_state_count,
                        "global scanner-state quotient must cover every tokenizer state",
                    );
                    assert!(
                        !map.representative_original_ids.is_empty() || raw_state_count == 0,
                        "global scanner-state quotient must retain one representative per class",
                    );
                    for (raw_state, &class) in map.original_to_internal.iter().enumerate() {
                        assert!(
                            class != u32::MAX
                                && (class as usize) < map.representative_original_ids.len(),
                            "global scanner-state quotient omitted raw tokenizer state {raw_state}",
                        );
                    }
                    for (class, &representative) in
                        map.representative_original_ids.iter().enumerate()
                    {
                        let representative = representative as usize;
                        assert!(
                            representative < raw_state_count
                                && map.original_to_internal[representative] == class as u32,
                            "global scanner-state quotient representative must belong to its class",
                        );
                    }

                    (
                        Some(Arc::from(map.original_to_internal.clone())),
                        Some(Arc::from(map.representative_original_ids.clone())),
                    )
                }
                None => (None, None),
            };
        let real_state_count = raw_representative_by_state
            .as_ref()
            .map_or(raw_state_count, |representatives| representatives.len());
        let mut edge_offsets = Vec::with_capacity(real_state_count + 2);
        let mut edges = Vec::new();
        let mut max_outdegree = 0usize;
        edge_offsets.push(0);
        for state in 0..real_state_count {
            let start = edges.len();
            let raw_state = raw_representative_by_state
                .as_ref()
                .map_or(state as u32, |representatives| representatives[state]);
            for (byte, target) in tokenizer.transitions_from(raw_state) {
                if relevant_bytes[byte as usize] {
                    let target = state_for_raw
                        .as_ref()
                        .map_or(target, |classes| classes[target as usize]);
                    edges.push((byte, target));
                }
            }
            max_outdegree = max_outdegree.max(edges.len() - start);
            edge_offsets.push(edges.len() as u32);
        }
        // Synthetic dead has no real edges: every enabled byte loops to itself.
        edge_offsets.push(edges.len() as u32);
        let initial_state = state_for_raw
            .as_ref()
            .map_or(tokenizer.initial_state_id() as usize, |classes| {
                classes[tokenizer.initial_state_id() as usize] as usize
            });
        Self {
            bytes,
            edge_offsets,
            edges,
            raw_state_count,
            raw_representative_by_state,
            state_for_raw,
            real_state_count,
            initial_state,
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

    #[inline]
    fn raw_state_for_state(&self, state: usize) -> u32 {
        self.raw_representative_by_state
            .as_ref()
            .map_or(state as u32, |representatives| representatives[state])
    }

    fn scanner_map_from_state_targets(&self, state_targets: &[u32]) -> TransportScannerStateMap {
        assert_eq!(state_targets.len(), self.real_state_count);
        if self.state_for_raw.is_none() {
            return TransportScannerStateMap::Explicit(Arc::from(state_targets.to_vec()));
        }
        let state_for_raw = self
            .state_for_raw
            .as_ref()
            .expect("quotient topology must retain raw-to-state classes");
        let raw_representative_by_state = self
            .raw_representative_by_state
            .as_ref()
            .expect("quotient topology must retain raw representatives");
        let raw_targets = (0..self.raw_state_count)
            .map(|raw_state| {
                let state = state_for_raw[raw_state] as usize;
                raw_representative_by_state[state_targets[state] as usize]
            })
            .collect::<Vec<_>>();
        TransportScannerStateMap::Explicit(raw_targets.into())
    }

    fn scanner_map_from_internal_quotient(
        &self,
        class_for_state: Arc<[u32]>,
        representative_for_class: Arc<[u32]>,
        source_class_for_target_deviations: Box<[(u32, u32)]>,
    ) -> TransportScannerStateMap {
        if self.state_for_raw.is_none() {
            return TransportScannerStateMap::Quotient {
                state_count: self.real_state_count,
                class_for_original: class_for_state,
                representative_for_class,
                source_class_for_target_deviations,
            };
        }
        let state_for_raw = self
            .state_for_raw
            .as_ref()
            .expect("quotient topology must retain raw-to-state classes");
        let raw_representative_by_state = self
            .raw_representative_by_state
            .as_ref()
            .expect("quotient topology must retain raw representatives");
        let class_for_original = (0..self.raw_state_count)
            .map(|raw_state| class_for_state[state_for_raw[raw_state] as usize])
            .collect::<Vec<_>>();
        let unused_dead_representative = raw_representative_by_state
            .first()
            .copied()
            .unwrap_or(0);
        let raw_representative_for_class = representative_for_class
            .iter()
            .map(|&state| {
                raw_representative_by_state
                    .get(state as usize)
                    .copied()
                    // The synthetic dead class has no raw scanner source. Its
                    // representative is therefore never queried, but the
                    // public transport-map shape still needs one in-range ID.
                    .unwrap_or(unused_dead_representative)
            })
            .collect::<Vec<_>>();
        TransportScannerStateMap::Quotient {
            state_count: self.raw_state_count,
            class_for_original: class_for_original.into(),
            representative_for_class: raw_representative_for_class.into(),
            source_class_for_target_deviations,
        }
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

/// Immutable raw lexer evidence that is independent of a historical TI round's
/// active-terminal mask.  The projected `OutputBits` remain round-local, but
/// their source columns and the reverse restricted topology are reusable.
struct TiRawDiscoveryData {
    finalizer_terminals_by_state: Vec<Box<[TerminalID]>>,
    future_finalizer_terminals_by_state: Vec<Box<[TerminalID]>>,
    finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    future_finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    reverse_predecessors: Arc<[Vec<u32>]>,
    observed_destinations: Arc<[bool]>,
}

impl TiRawDiscoveryData {
    fn new(tokenizer: &Tokenizer, topology: &RestrictedTopology) -> Self {
        let terminal_count = tokenizer.num_terminals() as usize;
        let mut finalizer_terminals_by_state = Vec::with_capacity(topology.real_state_count);
        let mut future_finalizer_terminals_by_state =
            Vec::with_capacity(topology.real_state_count);
        let mut finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        let mut future_finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];

        for state in 0..topology.real_state_count {
            let raw_state = topology.raw_state_for_state(state);
            let finalizers = tokenizer
                .matched_terminals_iter(raw_state)
                .collect::<Vec<_>>();
            for &terminal in &finalizers {
                finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
            finalizer_terminals_by_state.push(finalizers.into_boxed_slice());

            let future_finalizers = tokenizer
                .possible_future_terminals_iter(raw_state)
                .collect::<Vec<_>>();
            for &terminal in &future_finalizers {
                future_finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
            future_finalizer_terminals_by_state.push(future_finalizers.into_boxed_slice());
        }

        let mut reverse_predecessors = vec![Vec::<u32>::new(); topology.real_state_count];
        for source in 0..topology.real_state_count {
            for &(_, destination) in topology.edges_from(source) {
                reverse_predecessors[destination as usize].push(source as u32);
            }
        }

        Self {
            finalizer_terminals_by_state,
            future_finalizer_terminals_by_state,
            finalizer_states_by_terminal: finalizer_states_by_terminal.into(),
            future_finalizer_states_by_terminal: future_finalizer_states_by_terminal.into(),
            reverse_predecessors: reverse_predecessors.into(),
            observed_destinations: topology.observed_destinations().into(),
        }
    }
}

/// Static per-L2P-partition TI data.  The restricted raw lexer topology and
/// root observation depend only on vocabulary bytes, not on the historical
/// active-terminal mask of an iterative TI round.
pub(crate) struct TiDiscoveryContext {
    topology: Arc<RestrictedTopology>,
    raw: Arc<TiRawDiscoveryData>,
    root_output_signatures: Vec<RootOutputSignature>,
    root_observed_states: usize,
}

impl TiDiscoveryContext {
    pub(crate) fn new(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        let topology = Arc::new(RestrictedTopology::new(tokenizer, relevant_bytes));
        let raw = Arc::new(TiRawDiscoveryData::new(tokenizer, &topology));
        let (root_output_signatures, root_observed_states) =
            root_output_signatures(tokenizer, &topology);
        Self {
            topology,
            raw,
            root_output_signatures,
            root_observed_states,
        }
    }

    /// Build immutable discovery evidence over a total global scanner-state
    /// quotient. This is intentionally separate from an `initial_state_map`
    /// used to seed later id-map analysis.
    pub(crate) fn new_with_global_state_quotient(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: &GlobalScannerStateQuotient,
    ) -> Self {
        let topology = Arc::new(RestrictedTopology::new_with_global_state_quotient(
            tokenizer,
            relevant_bytes,
            global_state_quotient,
        ));
        let raw = Arc::new(TiRawDiscoveryData::new(tokenizer, &topology));
        let (root_output_signatures, root_observed_states) =
            root_output_signatures(tokenizer, &topology);
        Self {
            topology,
            raw,
            root_output_signatures,
            root_observed_states,
        }
    }
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
        let raw_state = topology.raw_state_for_state(state);
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
        let raw_state = topology.raw_state_for_state(state);
        finalizer_counts[state] = tokenizer
            .matched_terminals_iter(raw_state)
            .filter(|&terminal| active_terminals[terminal as usize])
            .count() as u64;
        future_finalizer_counts[state] = tokenizer
            .possible_future_terminals_iter(raw_state)
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
        let raw_state = topology.raw_state_for_state(state);
        for terminal in tokenizer.matched_terminals_iter(raw_state) {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            if candidate_index != usize::MAX {
                finalizer_support[candidate_index][word] |= mask;
            }
        }
        for terminal in tokenizer.possible_future_terminals_iter(raw_state) {
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
    observed_pairs: &[OutputPair],
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

        for pair in observed_pairs {
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


/// Exact token-macro observer used with the global C partition.
///
/// The existing TI observer follows arbitrary selected bytes.  C is instead a
/// relation at vocabulary-token boundaries: equivalent sources share the raw
/// destination of every token-first byte, after which the complete remaining
/// token trajectory is identical.  This topology therefore records the raw
/// frozen-output trace for every whole vocabulary token and only transitions
/// between C classes at token boundaries.
#[derive(Clone)]
struct TokenMacroEdge {
    destination: u32,
    output_states: Box<[u32]>,
}

#[derive(Clone)]
struct TokenMacroScan {
    endpoint: Option<u32>,
    longest_matches: Box<[(TerminalID, usize)]>,
}

struct TokenMacroTopology {
    state_map: ManyToOneIdMap,
    class_for_original: Arc<[u32]>,
    representative_for_source_class: Arc<[u32]>,
    tokens: Vec<Box<[u8]>>,
    edges: Vec<TokenMacroEdge>,
    root_scans: Vec<TokenMacroScan>,
    reset_suffix_scans: Vec<Vec<TokenMacroScan>>,
    reachable: Vec<bool>,
    initial_state: usize,
    initial_raw_state: u32,
    dead_state: usize,
}

pub(crate) struct TokenMacroDiscoveryContext {
    topology: Arc<TokenMacroTopology>,
    /// Round-invariant per-terminal candidate signatures, computed once with
    /// every terminal active and reused for every round's candidate grouping.
    candidate_signatures: std::sync::OnceLock<Vec<Vec<u32>>>,
}

impl TokenMacroDiscoveryContext {
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        vocab: &Vocab,
        partition: &GlobalTokenPositionStatePartition,
    ) -> Option<Self> {
        TokenMacroTopology::new(tokenizer, vocab, partition)
            .map(|topology| Self {
                topology: Arc::new(topology),
                candidate_signatures: std::sync::OnceLock::new(),
            })
    }
}

impl TokenMacroTopology {
    fn scan_suffix(tokenizer: &Tokenizer, start: u32, token: &[u8], offset: usize) -> TokenMacroScan {
        let mut state = Some(start);
        let mut matches = BTreeMap::<TerminalID, usize>::new();
        for (relative, &byte) in token[offset..].iter().enumerate() {
            state = state.and_then(|current| tokenizer.step(current, byte));
            match state {
                Some(current) => {
                    for terminal in tokenizer.matched_terminals_iter(current) {
                        matches.insert(terminal, offset + relative + 1);
                    }
                }
                None => break,
            }
        }
        TokenMacroScan {
            endpoint: state,
            longest_matches: matches.into_iter().collect::<Vec<_>>().into(),
        }
    }

    fn new(
        tokenizer: &Tokenizer,
        vocab: &Vocab,
        partition: &GlobalTokenPositionStatePartition,
    ) -> Option<Self> {
        let state_map = partition.as_many_to_one().clone();
        let raw_state_count = tokenizer.num_states() as usize;
        if state_map.original_to_internal.len() != raw_state_count
            || state_map.representative_original_ids.is_empty()
        {
            return None;
        }
        let mut tokens = vocab.entries.values().cloned().collect::<Vec<_>>();
        if tokens.iter().any(Vec::is_empty) {
            return None;
        }
        tokens.sort_unstable();
        tokens.dedup();
        if tokens.is_empty() {
            return None;
        }
        let tokens = tokens
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect::<Vec<_>>();
        let class_count = state_map.representative_original_ids.len();
        let class_for_original: Arc<[u32]> = Arc::from(state_map.original_to_internal.clone());
        let dead_state = class_count;
        let mut edges = Vec::with_capacity(class_count * tokens.len());
        let mut root_scans = Vec::with_capacity(class_count * tokens.len());
        let initial_raw_state = tokenizer.initial_state_id();
        let initial_class = state_map.original_to_internal[initial_raw_state as usize] as usize;
        let mut representative_for_source_class = state_map.representative_original_ids.clone();
        representative_for_source_class[initial_class] = initial_raw_state;
        for (class, &representative) in state_map.representative_original_ids.iter().enumerate() {
            let representative = if class == initial_class {
                initial_raw_state as usize
            } else {
                representative as usize
            };
            if representative >= raw_state_count {
                return None;
            }
            for token in &tokens {
                let mut state = Some(representative as u32);
                let mut output_states = Vec::with_capacity(token.len());
                for &byte in token.iter() {
                    state = state.and_then(|current| tokenizer.step(current, byte));
                    output_states.push(state.unwrap_or(u32::MAX));
                }
                let destination = state
                    .map(|state| state_map.original_to_internal[state as usize])
                    .unwrap_or(dead_state as u32);
                if destination != dead_state as u32 && destination as usize >= class_count {
                    return None;
                }
                edges.push(TokenMacroEdge {
                    destination,
                    output_states: output_states.into_boxed_slice(),
                });
                root_scans.push(Self::scan_suffix(
                    tokenizer,
                    representative as u32,
                    token,
                    0,
                ));
            }
        }
        let reset_suffix_scans = tokens
            .iter()
            .map(|token| {
                (0..token.len())
                    .map(|offset| Self::scan_suffix(tokenizer, initial_raw_state, token, offset))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let initial_state = state_map.original_to_internal[initial_raw_state as usize] as usize;
        if initial_state >= class_count {
            return None;
        }
        let mut reachable = vec![false; class_count];
        reachable[initial_state] = true;
        let mut worklist = vec![initial_state];
        while let Some(state) = worklist.pop() {
            for action in 0..tokens.len() {
                let destination = edges[state * tokens.len() + action].destination as usize;
                if destination != dead_state && !reachable[destination] {
                    reachable[destination] = true;
                    worklist.push(destination);
                }
            }
        }
        Some(Self {
            state_map,
            class_for_original,
            representative_for_source_class: representative_for_source_class.into(),
            tokens,
            edges,
            root_scans,
            reset_suffix_scans,
            reachable,
            initial_state,
            initial_raw_state,
            dead_state,
        })
    }

    #[inline]
    fn edge(&self, state: usize, action: usize) -> &TokenMacroEdge {
        &self.edges[state * self.tokens.len() + action]
    }

    #[inline]
    fn scan(&self, raw_start: u32, token: usize, offset: usize) -> &TokenMacroScan {
        if offset == 0 {
            let c_state = self.state_map.original_to_internal[raw_start as usize] as usize;
            &self.root_scans[c_state * self.tokens.len() + token]
        } else {
            debug_assert_eq!(raw_start, self.initial_raw_state);
            &self.reset_suffix_scans[token][offset]
        }
    }
}

struct TokenMacroRoundOutputs {
    by_raw_state: Vec<OutputPair>,
    empty: OutputPair,
    active: Vec<bool>,
}

impl TokenMacroRoundOutputs {
    fn new(tokenizer: &Tokenizer, active_terminals: &[bool]) -> Self {
        let by_raw_state = (0..tokenizer.num_states() as usize)
            .map(|state| OutputPair {
                finalizers: OutputBits::from_active(
                    &tokenizer
                        .matched_terminals_iter(state as u32)
                        .collect::<Vec<_>>(),
                    active_terminals,
                ),
                future_finalizers: OutputBits::from_active(
                    &tokenizer
                        .possible_future_terminals_iter(state as u32)
                        .collect::<Vec<_>>(),
                    active_terminals,
                ),
            })
            .collect();
        Self {
            by_raw_state,
            empty: OutputPair {
                finalizers: OutputBits::new(0),
                future_finalizers: OutputBits::new(0),
            },
            active: active_terminals.to_vec(),
        }
    }

    #[inline]
    fn at(&self, raw_state: u32) -> &OutputPair {
        self.by_raw_state.get(raw_state as usize).unwrap_or(&self.empty)
    }

    #[inline]
    fn terminal_is_active(&self, terminal: TerminalID) -> bool {
        self.active.get(terminal as usize).copied().unwrap_or(false)
    }
}

fn macro_output_pair_matches_after_swap(
    identity: &OutputPair,
    swapped: &OutputPair,
    left: TerminalID,
    right: TerminalID,
) -> bool {
    identity.finalizers == swapped.finalizers.mapped(Some((left as usize, right as usize)))
        && identity.future_finalizers
            == swapped
                .future_finalizers
                .mapped(Some((left as usize, right as usize)))
}

fn macro_edge_matches_after_swap(
    identity: &TokenMacroEdge,
    swapped: &TokenMacroEdge,
    outputs: &TokenMacroRoundOutputs,
    left: TerminalID,
    right: TerminalID,
) -> bool {
    identity.output_states.len() == swapped.output_states.len()
        && identity
            .output_states
            .iter()
            .zip(&swapped.output_states)
            .all(|(&identity_state, &swapped_state)| {
                macro_output_pair_matches_after_swap(
                    outputs.at(identity_state),
                    outputs.at(swapped_state),
                    left,
                    right,
                )
            })
}

/// Per-terminal exact rejection signatures for token-macro TI candidate
/// grouping.  A terminal's signature counts only its *own* occurrences at each
/// (action, position) slot, so it is invariant across TI rounds for any
/// terminal that stays active: shrinking the active set never changes an active
/// terminal's counts.  The signatures are therefore computed once (with every
/// terminal active) and reused for every round's grouping.
fn compute_candidate_signatures(
    topology: &TokenMacroTopology,
    outputs: &TokenMacroRoundOutputs,
    num_terminals: usize,
) -> Vec<Vec<u32>> {
    let output_slot_count = topology
        .tokens
        .iter()
        .map(|token| token.len() * 2)
        .sum::<usize>();
    let root_match_slot_count = topology.tokens.iter().map(|token| token.len()).sum::<usize>();
    let reset_match_slot_count = topology
        .tokens
        .iter()
        .map(|token| token.len().saturating_sub(1) * token.len())
        .sum::<usize>();
    let reset_endpoint_slot_count = topology
        .tokens
        .iter()
        .map(|token| token.len().saturating_sub(1) * 2)
        .sum::<usize>();
    let all_output_slot_count = output_slot_count;
    let all_root_match_slot_count = root_match_slot_count;
    let slot_count = output_slot_count
        + root_match_slot_count
        + reset_match_slot_count
        + reset_endpoint_slot_count
        + all_output_slot_count
        + all_root_match_slot_count;
    let mut signatures = vec![vec![0u32; slot_count]; num_terminals];

    let mut output_token_offset = Vec::with_capacity(topology.tokens.len());
    let mut root_match_token_offset = Vec::with_capacity(topology.tokens.len());
    let mut reset_match_token_offset = Vec::with_capacity(topology.tokens.len());
    let mut reset_endpoint_token_offset = Vec::with_capacity(topology.tokens.len());
    let mut all_output_token_offset = Vec::with_capacity(topology.tokens.len());
    let mut all_root_match_token_offset = Vec::with_capacity(topology.tokens.len());
    let mut output_offset = 0usize;
    let mut root_match_offset = output_slot_count;
    let mut reset_match_offset = output_slot_count + root_match_slot_count;
    let mut reset_endpoint_offset = output_slot_count + root_match_slot_count + reset_match_slot_count;
    let mut all_output_offset = reset_endpoint_offset + reset_endpoint_slot_count;
    let mut all_root_match_offset = all_output_offset + all_output_slot_count;
    for token in &topology.tokens {
        output_token_offset.push(output_offset);
        root_match_token_offset.push(root_match_offset);
        reset_match_token_offset.push(reset_match_offset);
        reset_endpoint_token_offset.push(reset_endpoint_offset);
        all_output_token_offset.push(all_output_offset);
        all_root_match_token_offset.push(all_root_match_offset);
        output_offset += token.len() * 2;
        root_match_offset += token.len();
        reset_match_offset += token.len().saturating_sub(1) * token.len();
        reset_endpoint_offset += token.len().saturating_sub(1) * 2;
        all_output_offset += token.len() * 2;
        all_root_match_offset += token.len();
    }

    for (state, &reachable) in topology.reachable.iter().enumerate() {
        if !reachable {
            continue;
        }
        for action in 0..topology.tokens.len() {
            let edge = topology.edge(state, action);
            let output_offset = output_token_offset[action];
            for (byte_index, &raw_state) in edge.output_states.iter().enumerate() {
                let output = outputs.at(raw_state);
                for &terminal in &output.finalizers.0 {
                    signatures[terminal as usize][output_offset + byte_index * 2] += 1;
                }
                for &terminal in &output.future_finalizers.0 {
                    signatures[terminal as usize][output_offset + byte_index * 2 + 1] += 1;
                }
            }

            // A rooted token-macro isomorphism preserves the token action and
            // the reset suffix offset reached by each terminal edge.  Counting
            // active terminal matches at these positions is therefore a cheap
            // exact rejection invariant before constructing the full macro DAG.
            let root_offset = root_match_token_offset[action];
            for &(terminal, next_offset) in topology.root_scans[state * topology.tokens.len() + action]
                .longest_matches
                .iter()
            {
                if outputs.terminal_is_active(terminal) {
                    signatures[terminal as usize][root_offset + next_offset - 1] += 1;
                }
            }
        }
    }

    // Root reachability is the strongest cheap root observation, but accepted
    // macro transports also have to map the whole scanner-coordinate domain.
    // A second aggregate over every C source class cheaply exposes many pairs
    // that only differ in unreachable macro classes. It is intentionally still
    // only a prefilter: all surviving pairs pass the exact certificate below.
    for state in 0..topology.state_map.representative_original_ids.len() {
        for action in 0..topology.tokens.len() {
            let edge = topology.edge(state, action);
            let output_offset = all_output_token_offset[action];
            for (byte_index, &raw_state) in edge.output_states.iter().enumerate() {
                let output = outputs.at(raw_state);
                for &terminal in &output.finalizers.0 {
                    signatures[terminal as usize][output_offset + byte_index * 2] += 1;
                }
                for &terminal in &output.future_finalizers.0 {
                    signatures[terminal as usize][output_offset + byte_index * 2 + 1] += 1;
                }
            }
            let root_offset = all_root_match_token_offset[action];
            for &(terminal, next_offset) in topology.root_scans[state * topology.tokens.len() + action]
                .longest_matches
                .iter()
            {
                if outputs.terminal_is_active(terminal) {
                    signatures[terminal as usize][root_offset + next_offset - 1] += 1;
                }
            }
        }
    }

    for (action, suffix_scans) in topology.reset_suffix_scans.iter().enumerate() {
        let token_len = topology.tokens[action].len();
        let token_match_offset = reset_match_token_offset[action];
        let token_endpoint_offset = reset_endpoint_token_offset[action];
        for offset in 1..token_len {
            let scan = &suffix_scans[offset];
            for &(terminal, next_offset) in scan.longest_matches.iter() {
                if outputs.terminal_is_active(terminal) {
                    signatures[terminal as usize]
                        [token_match_offset + (offset - 1) * token_len + next_offset - 1] += 1;
                }
            }
            if let Some(endpoint) = scan.endpoint {
                let output = outputs.at(endpoint);
                for &terminal in &output.finalizers.0 {
                    signatures[terminal as usize]
                        [token_endpoint_offset + (offset - 1) * 2] += 1;
                }
                for &terminal in &output.future_finalizers.0 {
                    signatures[terminal as usize]
                        [token_endpoint_offset + (offset - 1) * 2 + 1] += 1;
                }
            }
        }
    }

    signatures
}

/// Group active terminals by identical precomputed rejection signature.  Only
/// groups of 2+ terminals can possibly interchange, so singletons are dropped.
fn token_macro_candidate_groups_from_signatures(
    signatures: &[Vec<u32>],
    active_terminals: &[bool],
    ignore_terminal: Option<TerminalID>,
) -> Vec<Vec<TerminalID>> {
    let mut groups = FxHashMap::<&[u32], Vec<TerminalID>>::default();
    for (terminal, &active) in active_terminals.iter().enumerate() {
        if active && Some(terminal as TerminalID) != ignore_terminal {
            groups
                .entry(signatures[terminal].as_slice())
                .or_default()
                .push(terminal as TerminalID);
        }
    }
    groups
        .into_values()
        .filter(|group| group.len() >= 2)
        .collect()
}


#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TokenMacroSignature(Box<[(u32, u32)]>);

struct TokenMacroIdentityRound {
    classes: Vec<u32>,
    representative_by_class: Vec<u32>,
    class_by_signature: FxHashMap<TokenMacroSignature, u32>,
}

struct TokenMacroDfa {
    topology: Arc<TokenMacroTopology>,
    outputs: TokenMacroRoundOutputs,
    identity_trace_ids: Vec<u32>,
    dead_trace_ids: Vec<u32>,
    identity_trace_lookup: FxHashMap<Box<[OutputPair]>, u32>,
    identity_rounds: Vec<TokenMacroIdentityRound>,
    stable_round: usize,
}

impl TokenMacroDfa {
    fn output_trace_for_edge(&self, edge: &TokenMacroEdge) -> Box<[OutputPair]> {
        edge.output_states
            .iter()
            .map(|&state| self.outputs.at(state).clone())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn dead_output_trace(&self, action: usize) -> Box<[OutputPair]> {
        vec![self.outputs.empty.clone(); self.topology.tokens[action].len()].into_boxed_slice()
    }

    fn new(topology: Arc<TokenMacroTopology>, outputs: TokenMacroRoundOutputs) -> Self {
        let mut trace_lookup = FxHashMap::<Box<[OutputPair]>, u32>::default();
        let mut next_trace_id = 0u32;
        let mut intern = |trace: Box<[OutputPair]>| {
            if let Some(&id) = trace_lookup.get(&trace) {
                id
            } else {
                let id = next_trace_id;
                next_trace_id += 1;
                trace_lookup.insert(trace, id);
                id
            }
        };
        let identity_trace_ids = topology
            .edges
            .iter()
            .map(|edge| {
                let trace = edge
                    .output_states
                    .iter()
                    .map(|&state| outputs.at(state).clone())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                intern(trace)
            })
            .collect::<Vec<_>>();
        let dead_trace_ids = topology
            .tokens
            .iter()
            .map(|token| intern(vec![outputs.empty.clone(); token.len()].into_boxed_slice()))
            .collect::<Vec<_>>();
        let mut dfa = Self {
            topology,
            outputs,
            identity_trace_ids,
            dead_trace_ids,
            identity_trace_lookup: trace_lookup,
            identity_rounds: Vec::new(),
            stable_round: 0,
        };
        dfa.build_identity_rounds();
        dfa
    }

    fn state_count(&self) -> usize {
        self.topology.state_map.representative_original_ids.len() + 1
    }

    fn signature(&self, state: usize, previous: &[u32], trace_ids: &[u32], dead_trace_ids: &[u32]) -> TokenMacroSignature {
        let mut components = Vec::with_capacity(self.topology.tokens.len());
        if state == self.topology.dead_state {
            for (action, &trace) in dead_trace_ids.iter().enumerate() {
                let _ = action;
                components.push((previous[self.topology.dead_state], trace));
            }
        } else {
            for action in 0..self.topology.tokens.len() {
                let edge = self.topology.edge(state, action);
                components.push((previous[edge.destination as usize], trace_ids[state * self.topology.tokens.len() + action]));
            }
        }
        TokenMacroSignature(components.into_boxed_slice())
    }

    fn identity_round(&self, previous: &[u32]) -> TokenMacroIdentityRound {
        let mut classes = Vec::with_capacity(self.state_count());
        let mut representative_by_class = Vec::<u32>::new();
        let mut class_by_signature = FxHashMap::<TokenMacroSignature, u32>::default();
        for state in 0..self.state_count() {
            let signature = self.signature(
                state,
                previous,
                &self.identity_trace_ids,
                &self.dead_trace_ids,
            );
            let next = representative_by_class.len() as u32;
            let class = *class_by_signature.entry(signature).or_insert_with(|| {
                representative_by_class.push(state as u32);
                next
            });
            classes.push(class);
        }
        TokenMacroIdentityRound {
            classes,
            representative_by_class,
            class_by_signature,
        }
    }

    fn build_identity_rounds(&mut self) {
        let initial = vec![0u32; self.state_count()];
        self.identity_rounds.push(TokenMacroIdentityRound {
            classes: initial,
            representative_by_class: vec![0],
            class_by_signature: FxHashMap::default(),
        });
        for round in 1..=self.state_count() {
            let next = self.identity_round(&self.identity_rounds[round - 1].classes);
            let stable = same_equality_partition_u32(
                &self.identity_rounds[round - 1].classes,
                &next.classes,
            );
            self.identity_rounds.push(next);
            if stable {
                self.stable_round = round;
                return;
            }
        }
        panic!("token-macro TI identity refinement did not stabilize");
    }

    fn swapped_trace_ids(&self, left: TerminalID, right: TerminalID) -> (Vec<u32>, Vec<u32>) {
        let mut local = FxHashMap::<Box<[OutputPair]>, u32>::default();
        let base = self.identity_trace_lookup.len() as u32;
        let mut intern = |trace: Box<[OutputPair]>| {
            if let Some(&id) = self.identity_trace_lookup.get(&trace) {
                id
            } else {
                let next = base + local.len() as u32;
                *local.entry(trace).or_insert(next)
            }
        };
        let edge_ids = self
            .topology
            .edges
            .iter()
            .map(|edge| {
                let trace = edge
                    .output_states
                    .iter()
                    .map(|&state| {
                        let output = self.outputs.at(state);
                        OutputPair {
                            finalizers: output.finalizers.mapped(Some((left as usize, right as usize))),
                            future_finalizers: output
                                .future_finalizers
                                .mapped(Some((left as usize, right as usize))),
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                intern(trace)
            })
            .collect::<Vec<_>>();
        let dead_ids = self
            .topology
            .tokens
            .iter()
            .map(|token| intern(vec![self.outputs.empty.clone(); token.len()].into_boxed_slice()))
            .collect::<Vec<_>>();
        (edge_ids, dead_ids)
    }

    fn class_counts(classes: &[u32], class_count: usize) -> Option<Vec<u32>> {
        let mut counts = vec![0u32; class_count];
        for &class in classes {
            let count = counts.get_mut(class as usize)?;
            *count += 1;
        }
        Some(counts)
    }

    fn interchange_map(&self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        let (swapped_trace_ids, swapped_dead_trace_ids) = self.swapped_trace_ids(left, right);
        let mut swapped_previous = vec![0u32; self.state_count()];
        for round in 1..=self.stable_round {
            let identity = &self.identity_rounds[round];
            let mut swapped_next = Vec::with_capacity(self.state_count());
            for state in 0..self.state_count() {
                let signature = self.signature(
                    state,
                    &swapped_previous,
                    &swapped_trace_ids,
                    &swapped_dead_trace_ids,
                );
                let identity_class = *identity.class_by_signature.get(&signature)?;
                swapped_next.push(identity_class);
            }
            let identity_counts = Self::class_counts(
                &identity.classes,
                identity.representative_by_class.len(),
            )?;
            if Self::class_counts(&swapped_next, identity.representative_by_class.len())? != identity_counts {
                return None;
            }
            let root_identity = identity.classes[self.topology.initial_state];
            if swapped_next[self.topology.initial_state] != root_identity {
                return None;
            }
            if round == self.stable_round {
                if !same_equality_partition_pair_u32(
                    &self.identity_rounds[round - 1].classes,
                    &swapped_previous,
                    &identity.classes,
                    &swapped_next,
                ) {
                    return None;
                }
                let class_count = identity.representative_by_class.len();
                let mut target_class_for_source_class = vec![u32::MAX; class_count];
                for state in 0..self.topology.dead_state {
                    let source_class = identity.classes[state] as usize;
                    let target_class = swapped_next[state];
                    let slot = &mut target_class_for_source_class[source_class];
                    if *slot == u32::MAX {
                        *slot = target_class;
                    } else if *slot != target_class {
                        return None;
                    }
                }
                if target_class_for_source_class.iter().any(|&class| class == u32::MAX) {
                    return None;
                }
                let mut target_for_c_state = vec![0usize; self.topology.dead_state];
                for c_state in 0..self.topology.dead_state {
                    let source_class = identity.classes[c_state] as usize;
                    let target_class = target_class_for_source_class[source_class] as usize;
                    target_for_c_state[c_state] = identity.representative_by_class[target_class] as usize;
                }
                target_for_c_state[self.topology.initial_state] = self.topology.initial_state;
                let mut representative_for_class =
                    self.topology.state_map.representative_original_ids.clone();
                representative_for_class[self.topology.initial_state] = self.topology.initial_raw_state;
                let source_class_for_target_deviations = target_for_c_state
                    .iter()
                    .enumerate()
                    .filter_map(|(source_class, &target_class)| {
                        (source_class != target_class)
                            .then_some((source_class as u32, target_class as u32))
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                return Some(InterchangeMap {
                    scanner_state_map: TransportScannerStateMap::Quotient {
                        state_count: self.topology.state_map.original_to_internal.len(),
                        class_for_original: Arc::clone(&self.topology.class_for_original),
                        representative_for_class: representative_for_class.into(),
                        source_class_for_target_deviations,
                    },
                });
            }
            swapped_previous = swapped_next;
        }
        None
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactTokenMacroNode {
    endpoint: Option<(u32, OutputPair)>,
    edges: Box<[(TerminalID, u32)]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactTokenMacroNodeKey {
    endpoint: Option<(u32, u32)>,
    edges: Box<[(TerminalID, u32)]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactTokenMacroStateSignature(SmallVec<[u32; 64]>);

struct ExactTokenMacroRound {
    classes: Vec<u32>,
    representative_by_class: Vec<usize>,
    node_ids: FxHashMap<ExactTokenMacroNodeKey, u32>,
    nodes_by_id: Vec<ExactTokenMacroNode>,
    action_nodes_by_class: Vec<Box<[u32]>>,
    state_classes: FxHashMap<ExactTokenMacroStateSignature, u32>,
}

/// Result of the fast macro-automorphism proposal. `Impossible` is a proof
/// from a necessary root-support cardinality invariant; `Inconclusive` keeps
/// the legacy exact refinement available for ambiguous support buckets.
enum DirectMacroInterchangeResult {
    Proven(InterchangeMap),
    Impossible,
    Inconclusive,
}

/// Exact reset-aware token observer for C-assisted TI.  Boundary states are C
/// representatives; each byte within a token remains a raw tokenizer state.
/// A terminal match produces a labelled edge to a suffix scanned from the
/// actual lexer initial state, matching `TerminalNwaBuilder::process_child_segment`.
struct ExactTokenMacroDfa {
    topology: Arc<TokenMacroTopology>,
    outputs: TokenMacroRoundOutputs,
    output_pair_ids: FxHashMap<OutputPair, u32>,
    output_pair_id_by_raw_state: Vec<u32>,
    empty_output_pair_id: u32,
    identity_rounds: Vec<ExactTokenMacroRound>,
    root_terminal_supports: Vec<Vec<(usize, Box<[u8]>)>>,
    node_parents: Vec<Vec<u32>>,
    nodes_by_terminal: Vec<Vec<u32>>,
    nodes_by_previous_class: Vec<Vec<u32>>,
    macro_class_for_source_class: Arc<[u32]>,
    representative_for_macro_class: Arc<[u32]>,
    stable_round: usize,
    identity_setup_ms: f64,
}

impl ExactTokenMacroDfa {
    const ACCEPT_SINK: u32 = u32::MAX;

    fn new(topology: Arc<TokenMacroTopology>, outputs: TokenMacroRoundOutputs) -> Self {
        let started_at = Instant::now();
        let mut output_pair_ids = FxHashMap::<OutputPair, u32>::default();
        let empty_output_pair_id = {
            let id = output_pair_ids.len() as u32;
            output_pair_ids.insert(outputs.empty.clone(), id);
            id
        };
        let mut output_pair_id_by_raw_state = Vec::with_capacity(outputs.by_raw_state.len());
        for output in &outputs.by_raw_state {
            let id = match output_pair_ids.get(output) {
                Some(&id) => id,
                None => {
                    let next = output_pair_ids.len() as u32;
                    output_pair_ids.insert(output.clone(), next);
                    next
                }
            };
            output_pair_id_by_raw_state.push(id);
        }

        let mut dfa = Self {
            topology,
            outputs,
            output_pair_ids,
            output_pair_id_by_raw_state,
            empty_output_pair_id,
            identity_rounds: Vec::new(),
            root_terminal_supports: Vec::new(),
            node_parents: Vec::new(),
            nodes_by_terminal: Vec::new(),
            nodes_by_previous_class: Vec::new(),
            macro_class_for_source_class: Arc::from([]),
            representative_for_macro_class: Arc::from([]),
            stable_round: 0,
            identity_setup_ms: 0.0,
        };
        dfa.build_identity_rounds();
        dfa.build_macro_transport_quotient();
        dfa.root_terminal_supports = dfa.build_root_terminal_supports();
        dfa.build_node_reverse_indices();
        dfa.identity_setup_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        dfa
    }

    #[inline]
    fn c_state_count(&self) -> usize {
        self.topology.state_map.representative_original_ids.len()
    }

    fn map_terminal(terminal: TerminalID, swap: Option<(TerminalID, TerminalID)>) -> TerminalID {
        match swap {
            Some((left, right)) if terminal == left => right,
            Some((left, right)) if terminal == right => left,
            _ => terminal,
        }
    }

    fn map_output(&self, raw_state: u32, swap: Option<(TerminalID, TerminalID)>) -> OutputPair {
        let output = self.outputs.at(raw_state);
        let swap = swap.map(|(left, right)| (left as usize, right as usize));
        OutputPair {
            finalizers: output.finalizers.mapped(swap),
            future_finalizers: output.future_finalizers.mapped(swap),
        }
    }

    fn swapped_node_id(
        &self,
        raw_start: u32,
        token_index: usize,
        offset: usize,
        previous: &[u32],
        swap: (TerminalID, TerminalID),
        identity: &ExactTokenMacroRound,
        cache: &mut FxHashMap<(u32, usize, usize), u32>,
    ) -> Option<u32> {
        if let Some(&id) = cache.get(&(raw_start, token_index, offset)) {
            return Some(id);
        }
        let scan = self.topology.scan(raw_start, token_index, offset);

        let endpoint = if let Some(state) = scan.endpoint {
            let c_state = self.topology.state_map.original_to_internal[state as usize] as usize;
            let mapped_output = self.map_output(state, Some(swap));
            let output_id = *self.output_pair_ids.get(&mapped_output)?;
            Some((previous[c_state], output_id))
        } else {
            None
        };
        let token_len = self.topology.tokens[token_index].len();
        let mut edges = Vec::with_capacity(scan.longest_matches.len());
        for &(terminal, next_offset) in scan.longest_matches.iter() {
            if !self.outputs.terminal_is_active(terminal) {
                continue;
            }
            let child = if next_offset == token_len {
                Self::ACCEPT_SINK
            } else {
                self.swapped_node_id(
                    self.topology.initial_raw_state,
                    token_index,
                    next_offset,
                    previous,
                    swap,
                    identity,
                    cache,
                )?
            };
            edges.push((Self::map_terminal(terminal, Some(swap)), child));
        }
        let node = ExactTokenMacroNodeKey {
            endpoint,
            edges: edges.into_boxed_slice(),
        };
        let id = *identity.node_ids.get(&node)?;
        cache.insert((raw_start, token_index, offset), id);
        Some(id)
    }

    #[inline]
    fn dense_node_cache_index(
        &self,
        raw_start: u32,
        token_index: usize,
        offset: usize,
        cache_stride: usize,
    ) -> usize {
        ((raw_start as usize * self.topology.tokens.len() + token_index) * cache_stride) + offset
    }

    fn identity_node_id_dense(
        &self,
        raw_start: u32,
        token_index: usize,
        offset: usize,
        previous: &[u32],
        interned_nodes: &mut FxHashMap<ExactTokenMacroNodeKey, u32>,
        output_pair_id_by_raw_state: &[u32],
        empty_output_pair_id: u32,
        nodes_by_id: &mut Vec<ExactTokenMacroNode>,
        cache: &mut [u32],
        cache_stride: usize,
    ) -> u32 {
        let cache_index = self.dense_node_cache_index(raw_start, token_index, offset, cache_stride);
        let cached = cache[cache_index];
        if cached != u32::MAX {
            return cached;
        }
        let scan = self.topology.scan(raw_start, token_index, offset);

        let endpoint = scan.endpoint.map(|state| {
            let c_state = self.topology.state_map.original_to_internal[state as usize] as usize;
            (previous[c_state], self.map_output(state, None))
        });
        let endpoint_key = scan.endpoint.map(|state| {
            let c_state = self.topology.state_map.original_to_internal[state as usize] as usize;
            let output_id = output_pair_id_by_raw_state
                .get(state as usize)
                .copied()
                .unwrap_or(empty_output_pair_id);
            (previous[c_state], output_id)
        });
        let token_len = self.topology.tokens[token_index].len();
        let mut edges = Vec::with_capacity(scan.longest_matches.len());
        for &(terminal, next_offset) in scan.longest_matches.iter() {
            if !self.outputs.terminal_is_active(terminal) {
                continue;
            }
            let child = if next_offset == token_len {
                Self::ACCEPT_SINK
            } else {
                self.identity_node_id_dense(
                    self.topology.initial_raw_state,
                    token_index,
                    next_offset,
                    previous,
                    interned_nodes,
                    output_pair_id_by_raw_state,
                    empty_output_pair_id,
                    nodes_by_id,
                    cache,
                    cache_stride,
                )
            };
            edges.push((terminal, child));
        }
        let edges = edges.into_boxed_slice();
        let node = ExactTokenMacroNode {
            endpoint,
            edges: edges.clone(),
        };
        let key = ExactTokenMacroNodeKey {
            endpoint: endpoint_key,
            edges,
        };
        let id = match interned_nodes.entry(key) {
            Entry::Vacant(entry) => {
                let id = nodes_by_id.len() as u32;
                nodes_by_id.push(node);
                entry.insert(id);
                id
            }
            Entry::Occupied(entry) => *entry.get(),
        };
        cache[cache_index] = id;
        id
    }

    fn identity_round(&self, previous: &[u32]) -> ExactTokenMacroRound {
        let mut node_ids = FxHashMap::<ExactTokenMacroNodeKey, u32>::default();
        let mut nodes_by_id = Vec::<ExactTokenMacroNode>::new();
        let cache_stride = self
            .topology
            .tokens
            .iter()
            .map(|token| token.len())
            .max()
            .unwrap_or(0)
            + 1;
        let mut cache = vec![
            u32::MAX;
            self.topology.state_map.original_to_internal.len()
                * self.topology.tokens.len()
                * cache_stride
        ];
        let mut state_classes = FxHashMap::<ExactTokenMacroStateSignature, u32>::default();
        let mut classes = Vec::with_capacity(self.c_state_count());
        let mut representative_by_class = Vec::new();
        let mut action_nodes_by_class = Vec::<Box<[u32]>>::new();
        for c_state in 0..self.c_state_count() {
            let raw_start = if c_state == self.topology.initial_state {
                self.topology.initial_raw_state
            } else {
                self.topology.state_map.representative_original_ids[c_state]
            };
            let signature = {
                let mut signature = SmallVec::<[u32; 64]>::with_capacity(self.topology.tokens.len());
                for token_index in 0..self.topology.tokens.len() {
                    signature.push(
                    self.identity_node_id_dense(
                        raw_start,
                        token_index,
                        0,
                        previous,
                        &mut node_ids,
                        &self.output_pair_id_by_raw_state,
                        self.empty_output_pair_id,
                        &mut nodes_by_id,
                        &mut cache,
                        cache_stride,
                    )
                    );
                }
                signature
            };
            let next = representative_by_class.len() as u32;
            let class = match state_classes.entry(ExactTokenMacroStateSignature(signature)) {
                Entry::Vacant(entry) => {
                    action_nodes_by_class.push(entry.key().0.iter().copied().collect());
                    representative_by_class.push(c_state);
                    entry.insert(next);
                    next
                }
                Entry::Occupied(entry) => *entry.get(),
            };
            classes.push(class);
        }
        ExactTokenMacroRound {
            classes,
            representative_by_class,
            node_ids,
            nodes_by_id,
            action_nodes_by_class,
            state_classes,
        }
    }

    fn swapped_round(
        &self,
        previous: &[u32],
        identity: &ExactTokenMacroRound,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<Vec<u32>> {
        let mut cache = FxHashMap::<(u32, usize, usize), u32>::default();
        let mut classes = Vec::with_capacity(self.c_state_count());
        for c_state in 0..self.c_state_count() {
            let raw_start = if c_state == self.topology.initial_state {
                self.topology.initial_raw_state
            } else {
                self.topology.state_map.representative_original_ids[c_state]
            };
            let signature = {
                let mut signature = SmallVec::<[u32; 64]>::with_capacity(self.topology.tokens.len());
                for token_index in 0..self.topology.tokens.len() {
                    signature.push(self.swapped_node_id(
                        raw_start,
                        token_index,
                        0,
                        previous,
                        (left, right),
                        identity,
                        &mut cache,
                    )?);
                }
                signature
            };
            classes.push(*identity
                .state_classes
                .get(&ExactTokenMacroStateSignature(signature))?);
        }
        Some(classes)
    }

    /// Evaluate one representative of every exact identity macro class.  A
    /// class is a complete structural equality of reset-aware token behavior,
    /// so applying a global label swap cannot split its members.  This avoids
    /// repeating the same swapped macro construction for all C members.
    fn swapped_round_by_identity_class(
        &self,
        previous: &[u32],
        identity: &ExactTokenMacroRound,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<Vec<u32>> {
        let mut cache = FxHashMap::<(u32, usize, usize), u32>::default();
        let mut classes = Vec::with_capacity(identity.representative_by_class.len());
        for &c_state in &identity.representative_by_class {
            let raw_start = if c_state == self.topology.initial_state {
                self.topology.initial_raw_state
            } else {
                self.topology.state_map.representative_original_ids[c_state]
            };
            let signature = {
                let mut signature = SmallVec::<[u32; 64]>::with_capacity(self.topology.tokens.len());
                for token_index in 0..self.topology.tokens.len() {
                    signature.push(self.swapped_node_id(
                        raw_start,
                        token_index,
                        0,
                        previous,
                        (left, right),
                        identity,
                        &mut cache,
                    )?);
                }
                signature
            };
            classes.push(*identity
                .state_classes
                .get(&ExactTokenMacroStateSignature(signature))?);
        }
        Some(classes)
    }

    fn build_identity_rounds(&mut self) {
        let mut previous = vec![0u32; self.c_state_count()];
        for round in 1..=self.c_state_count().saturating_mul(2).max(1) {
            let next = self.identity_round(&previous);
            let stable = same_equality_partition_u32(&previous, &next.classes);
            self.identity_rounds.push(next);
            if stable {
                self.stable_round = round;
                return;
            }
            previous = self.identity_rounds[round - 1].classes.clone();
        }
        panic!("exact token-macro TI identity refinement did not stabilize");
    }

    fn build_macro_transport_quotient(&mut self) {
        let identity = &self.identity_rounds[self.stable_round - 1];
        self.macro_class_for_source_class = Arc::from(identity.classes.clone());
        self.representative_for_macro_class = identity
            .representative_by_class
            .iter()
            .map(|&source_class| self.topology.representative_for_source_class[source_class])
            .collect::<Vec<_>>()
            .into();
    }

    fn build_root_terminal_supports(&self) -> Vec<Vec<(usize, Box<[u8]>)>> {
        let identity = &self.identity_rounds[self.stable_round - 1];
        let mut supports = vec![Vec::new(); self.outputs.active.len()];
        for (class, action_nodes) in identity.action_nodes_by_class.iter().enumerate() {
            let mut per_terminal = FxHashMap::<TerminalID, Vec<u8>>::default();
            for (action, &node_id) in action_nodes.iter().enumerate() {
                let node = &identity.nodes_by_id[node_id as usize];
                for &(terminal, _) in node.edges.iter() {
                    let bits = per_terminal
                        .entry(terminal)
                        .or_insert_with(|| vec![0; action_nodes.len()]);
                    bits[action] |= 1;
                }
                if let Some((_, output)) = &node.endpoint {
                    for &terminal in output.finalizers.0.iter() {
                        let bits = per_terminal
                            .entry(terminal)
                            .or_insert_with(|| vec![0; action_nodes.len()]);
                        bits[action] |= 2;
                    }
                    for &terminal in output.future_finalizers.0.iter() {
                        let bits = per_terminal
                            .entry(terminal)
                            .or_insert_with(|| vec![0; action_nodes.len()]);
                        bits[action] |= 4;
                    }
                }
            }
            for (terminal, bits) in per_terminal {
                supports[terminal as usize].push((class, bits.into_boxed_slice()));
            }
        }
        supports
    }

    fn build_node_reverse_indices(&mut self) {
        let identity = &self.identity_rounds[self.stable_round - 1];
        let previous_class_count = if self.stable_round == 1 {
            1
        } else {
            self.identity_rounds[self.stable_round - 2]
                .representative_by_class
                .len()
        };
        self.node_parents = vec![Vec::new(); identity.nodes_by_id.len()];
        self.nodes_by_terminal = vec![Vec::new(); self.outputs.active.len()];
        self.nodes_by_previous_class = vec![Vec::new(); previous_class_count];
        for (node_id, node) in identity.nodes_by_id.iter().enumerate() {
            if let Some((previous_class, output)) = &node.endpoint {
                self.nodes_by_previous_class[*previous_class as usize].push(node_id as u32);
                for &terminal in output.finalizers.0.iter() {
                    self.nodes_by_terminal[terminal as usize].push(node_id as u32);
                }
                for &terminal in output.future_finalizers.0.iter() {
                    self.nodes_by_terminal[terminal as usize].push(node_id as u32);
                }
            }
            for &(terminal, child) in node.edges.iter() {
                self.nodes_by_terminal[terminal as usize].push(node_id as u32);
                if child != Self::ACCEPT_SINK {
                    self.node_parents[child as usize].push(node_id as u32);
                }
            }
        }
        for nodes in &mut self.node_parents {
            nodes.sort_unstable();
            nodes.dedup();
        }
        for nodes in &mut self.nodes_by_terminal {
            nodes.sort_unstable();
            nodes.dedup();
        }
        for nodes in &mut self.nodes_by_previous_class {
            nodes.sort_unstable();
            nodes.dedup();
        }
    }

    fn class_counts(classes: &[u32], count: usize) -> Option<Vec<u32>> {
        let mut result = vec![0u32; count];
        for &class in classes {
            *result.get_mut(class as usize)? += 1;
        }
        Some(result)
    }

    /// Candidate macro permutation from direct root support.  This is only a
    /// proposal. `verify_macro_permutation` proves every endpoint and reset
    /// suffix node before it can be used.
    fn candidate_macro_permutation(
        &self,
        identity: &ExactTokenMacroRound,
        left: TerminalID,
        right: TerminalID,
    ) -> Result<Vec<usize>, ()> {
        let left_supports = self.root_terminal_supports.get(left as usize).ok_or(())?;
        let right_supports = self.root_terminal_supports.get(right as usize).ok_or(())?;
        let empty_signature = vec![0u8; self.topology.tokens.len()];
        let mut buckets = BTreeMap::<(Box<[u8]>, Box<[u8]>), Vec<usize>>::new();
        let mut left_index = 0usize;
        let mut right_index = 0usize;
        while left_index < left_supports.len() || right_index < right_supports.len() {
            let left_class = left_supports.get(left_index).map(|&(class, _)| class);
            let right_class = right_supports.get(right_index).map(|&(class, _)| class);
            let class = match (left_class, right_class) {
                (Some(left_class), Some(right_class)) => left_class.min(right_class),
                (Some(left_class), None) => left_class,
                (None, Some(right_class)) => right_class,
                (None, None) => unreachable!("nonempty support merge must have a class"),
            };
            let left_signature = if left_class == Some(class) {
                let signature = &left_supports[left_index].1;
                left_index += 1;
                signature.as_ref()
            } else {
                empty_signature.as_slice()
            };
            let right_signature = if right_class == Some(class) {
                let signature = &right_supports[right_index].1;
                right_index += 1;
                signature.as_ref()
            } else {
                empty_signature.as_slice()
            };
            if left_signature != right_signature {
                buckets
                    .entry((left_signature.into(), right_signature.into()))
                    .or_default()
                    .push(class);
            }
        }

        let mut permutation: Vec<usize> = (0..identity.representative_by_class.len()).collect();
        for (signature, sources) in &buckets {
            let reverse = (signature.1.clone(), signature.0.clone());
            if signature > &reverse {
                continue;
            }
            let targets = buckets.get(&reverse).ok_or(())?;
            if sources.len() != targets.len() {
                return Err(());
            }
            if signature == &reverse {
                continue;
            }
            for (&source, &target) in sources.iter().zip(targets) {
                permutation[source] = target;
                permutation[target] = source;
            }
        }
        (permutation[self.identity_rounds[self.stable_round - 1].classes[self.topology.initial_state]
            as usize]
            == self.identity_rounds[self.stable_round - 1].classes[self.topology.initial_state]
                as usize)
            .then_some(permutation)
            .ok_or(())
    }

    fn previous_permutation(
        &self,
        identity: &ExactTokenMacroRound,
        permutation: &[usize],
    ) -> Option<Vec<usize>> {
        if self.stable_round == 1 {
            return Some(vec![0]);
        }
        let previous = &self.identity_rounds[self.stable_round - 2];
        let class_count = previous.representative_by_class.len();
        let mut result = vec![usize::MAX; class_count];
        for (source_final, &target_final) in permutation.iter().enumerate() {
            let source_c_state = identity.representative_by_class[source_final];
            let target_c_state = identity.representative_by_class[target_final];
            let source_previous = previous.classes[source_c_state] as usize;
            let target_previous = previous.classes[target_c_state] as usize;
            let slot = &mut result[source_previous];
            if *slot == usize::MAX {
                *slot = target_previous;
            } else if *slot != target_previous {
                return None;
            }
        }
        result.iter().all(|&class| class != usize::MAX).then_some(result)
    }

    fn impacted_nodes(
        &self,
        previous_permutation: &[usize],
        left: TerminalID,
        right: TerminalID,
    ) -> Vec<bool> {
        let mut impacted = vec![false; self.node_parents.len()];
        let mut worklist = Vec::<u32>::new();
        let seed = |node: u32, impacted: &mut [bool], worklist: &mut Vec<u32>| {
            if !impacted[node as usize] {
                impacted[node as usize] = true;
                worklist.push(node);
            }
        };
        for &node in self
            .nodes_by_terminal
            .get(left as usize)
            .into_iter()
            .flatten()
            .chain(
                self.nodes_by_terminal
                    .get(right as usize)
                    .into_iter()
                    .flatten(),
            )
        {
            seed(node, &mut impacted, &mut worklist);
        }
        for (previous_class, &target_class) in previous_permutation.iter().enumerate() {
            if previous_class == target_class {
                continue;
            }
            for &node in &self.nodes_by_previous_class[previous_class] {
                seed(node, &mut impacted, &mut worklist);
            }
        }
        while let Some(node) = worklist.pop() {
            for &parent in &self.node_parents[node as usize] {
                seed(parent, &mut impacted, &mut worklist);
            }
        }
        impacted
    }

    fn node_pair_matches_permutation(
        identity: &ExactTokenMacroRound,
        source_node: u32,
        target_node: u32,
        previous_permutation: &[usize],
        left: TerminalID,
        right: TerminalID,
        memo: &mut FxHashMap<(u32, u32), bool>,
    ) -> bool {
        if let Some(&result) = memo.get(&(source_node, target_node)) {
            return result;
        }
        let source = &identity.nodes_by_id[source_node as usize];
        let target = &identity.nodes_by_id[target_node as usize];
        let mut result = source.edges.len() == target.edges.len();
        if result {
            result = match (&source.endpoint, &target.endpoint) {
                (None, None) => true,
                (Some((source_previous, source_output)), Some((target_previous, target_output))) => {
                    *target_previous as usize
                        == previous_permutation[*source_previous as usize]
                        && *target_output
                            == OutputPair {
                                finalizers: source_output
                                    .finalizers
                                    .mapped(Some((left as usize, right as usize))),
                                future_finalizers: source_output
                                    .future_finalizers
                                    .mapped(Some((left as usize, right as usize))),
                            }
                }
                _ => false,
            };
        }
        if result {
            for &(source_label, source_child) in source.edges.iter() {
                let mapped_label = Self::map_terminal(source_label, Some((left, right)));
                let Ok(target_index) = target
                    .edges
                    .binary_search_by_key(&mapped_label, |&(label, _)| label)
                else {
                    result = false;
                    break;
                };
                let target_child = target.edges[target_index].1;
                if source_child == Self::ACCEPT_SINK || target_child == Self::ACCEPT_SINK {
                    if source_child != target_child {
                        result = false;
                        break;
                    }
                } else if !Self::node_pair_matches_permutation(
                    identity,
                    source_child,
                    target_child,
                    previous_permutation,
                    left,
                    right,
                    memo,
                ) {
                    result = false;
                    break;
                }
            }
        }
        memo.insert((source_node, target_node), result);
        result
    }

    fn verify_macro_permutation(
        &self,
        identity: &ExactTokenMacroRound,
        permutation: &[usize],
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        // The proposal starts from identity and swaps disjoint reverse
        // signature buckets, so it is an involutive bijection by construction.
        if permutation.len() != identity.representative_by_class.len() {
            return false;
        }
        let Some(previous_permutation) = self.previous_permutation(identity, permutation) else {
            return false;
        };
        let impacted = self.impacted_nodes(&previous_permutation, left, right);
        let mut memo = FxHashMap::<(u32, u32), bool>::default();
        for source_class in 0..permutation.len() {
            let target_class = permutation[source_class];
            for (&source_node, &target_node) in identity.action_nodes_by_class[source_class]
                .iter()
                .zip(identity.action_nodes_by_class[target_class].iter())
            {
                if source_class == target_class && !impacted[source_node as usize] {
                    continue;
                }
                if !Self::node_pair_matches_permutation(
                    identity,
                    source_node,
                    target_node,
                    &previous_permutation,
                    left,
                    right,
                    &mut memo,
                ) {
                    return false;
                }
            }
        }
        true
    }

    fn interchange_map_from_macro_permutation(
        &self,
        identity: &ExactTokenMacroRound,
        permutation: &[usize],
    ) -> Option<InterchangeMap> {
        let root_class = identity.classes[self.topology.initial_state] as usize;
        if permutation.get(root_class).copied()? != root_class {
            return None;
        }
        let deviations = permutation
            .iter()
            .enumerate()
            .filter_map(|(source, &target)| (source != target).then_some((source as u32, target as u32)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Some(InterchangeMap {
            scanner_state_map: TransportScannerStateMap::MacroQuotient {
                state_count: self.topology.state_map.original_to_internal.len(),
                class_for_original: Arc::clone(&self.topology.class_for_original),
                representative_for_source_class: Arc::clone(
                    &self.topology.representative_for_source_class,
                ),
                macro_class_for_source_class: Arc::clone(&self.macro_class_for_source_class),
                representative_for_macro_class: Arc::clone(&self.representative_for_macro_class),
                source_macro_for_target_deviations: deviations,
            },
        })
    }

    fn direct_interchange_attempt(
        &self,
        left: TerminalID,
        right: TerminalID,
    ) -> DirectMacroInterchangeResult {
        let stable_identity = &self.identity_rounds[self.stable_round - 1];
        let permutation = match self.candidate_macro_permutation(stable_identity, left, right) {
            Ok(permutation) => permutation,
            Err(()) => return DirectMacroInterchangeResult::Impossible,
        };
        if !self.verify_macro_permutation(stable_identity, &permutation, left, right) {
            return DirectMacroInterchangeResult::Inconclusive;
        }
        self.interchange_map_from_macro_permutation(stable_identity, &permutation)
            .map(DirectMacroInterchangeResult::Proven)
            .unwrap_or(DirectMacroInterchangeResult::Inconclusive)
    }

    fn refinement_interchange_map(&self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        let mut swapped_previous_by_class = vec![0u32; 1];
        for round in 1..=self.stable_round {
            let identity = &self.identity_rounds[round - 1];
            let previous = if round == 1 {
                vec![0u32; self.c_state_count()]
            } else {
                let identity_previous = &self.identity_rounds[round - 2];
                identity_previous
                    .classes
                    .iter()
                    .map(|&class| swapped_previous_by_class[class as usize])
                    .collect::<Vec<_>>()
            };
            let swapped_next = self.swapped_round_by_identity_class(
                &previous,
                identity,
                left,
                right,
            )?;
            let counts = Self::class_counts(&identity.classes, identity.representative_by_class.len())?;
            let mut swapped_counts = vec![0u32; counts.len()];
            for (target_class, &source_class) in swapped_next.iter().enumerate() {
                swapped_counts[source_class as usize] += counts[target_class];
            }
            if swapped_counts != counts {
                return None;
            }
            let root_class = identity.classes[self.topology.initial_state] as usize;
            if swapped_next[root_class] != root_class as u32 {
                return None;
            }
            if round == self.stable_round {
                if round > 1 && swapped_next != swapped_previous_by_class {
                    return None;
                }
                let mut target_for_identity_class = vec![None; identity.representative_by_class.len()];
                for (candidate_class, &identity_class) in swapped_next.iter().enumerate() {
                    target_for_identity_class[identity_class as usize]
                        .get_or_insert(identity.representative_by_class[candidate_class]);
                }
                target_for_identity_class[root_class] = Some(self.topology.initial_state);
                let target_for_c_state = identity
                    .classes
                    .iter()
                    .map(|&identity_class| target_for_identity_class[identity_class as usize])
                    .collect::<Option<Vec<_>>>()?;
                let mut representatives = self.topology.state_map.representative_original_ids.clone();
                representatives[self.topology.initial_state] = self.topology.initial_raw_state;
                let deviations = target_for_c_state
                    .iter()
                    .enumerate()
                    .filter_map(|(source, &target)| (source != target).then_some((source as u32, target as u32)))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                return Some(InterchangeMap {
                    scanner_state_map: TransportScannerStateMap::Quotient {
                        state_count: self.topology.state_map.original_to_internal.len(),
                        class_for_original: Arc::clone(&self.topology.class_for_original),
                        representative_for_class: representatives.into(),
                        source_class_for_target_deviations: deviations,
                    },
                });
            }
            swapped_previous_by_class = swapped_next;
        }
        None
    }
}

impl ExactTokenMacroDfa {
    fn interchange_map(&self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        match self.direct_interchange_attempt(left, right) {
            DirectMacroInterchangeResult::Proven(map) => Some(map),
            DirectMacroInterchangeResult::Impossible => None,
            DirectMacroInterchangeResult::Inconclusive => self.refinement_interchange_map(left, right),
        }
    }
}

fn token_macro_interchange_map(
    topology: &TokenMacroTopology,
    outputs: &TokenMacroRoundOutputs,
    left: TerminalID,
    right: TerminalID,
) -> Option<InterchangeMap> {
    let class_count = topology.state_map.representative_original_ids.len();
    let mut target_for_source = vec![u32::MAX; class_count];
    let mut source_for_target = vec![u32::MAX; class_count];
    target_for_source[topology.initial_state] = topology.initial_state as u32;
    source_for_target[topology.initial_state] = topology.initial_state as u32;
    let mut worklist = vec![topology.initial_state];

    while let Some(source) = worklist.pop() {
        let target = target_for_source[source] as usize;
        for action in 0..topology.tokens.len() {
            let identity_edge = topology.edge(source, action);
            let swapped_edge = topology.edge(target, action);
            if !macro_edge_matches_after_swap(identity_edge, swapped_edge, outputs, left, right) {
                return None;
            }
            let identity_destination = identity_edge.destination as usize;
            let swapped_destination = swapped_edge.destination as usize;
            if identity_destination == topology.dead_state || swapped_destination == topology.dead_state {
                if identity_destination != swapped_destination {
                    return None;
                }
                continue;
            }
            match target_for_source[identity_destination] {
                u32::MAX => {
                    if source_for_target[swapped_destination] != u32::MAX {
                        return None;
                    }
                    target_for_source[identity_destination] = swapped_destination as u32;
                    source_for_target[swapped_destination] = identity_destination as u32;
                    worklist.push(identity_destination);
                }
                mapped if mapped as usize != swapped_destination => return None,
                _ => {}
            }
        }
    }

    let initial_class = topology.initial_state;
    let initial_raw = topology.initial_raw_state;
    let raw_targets = topology
        .state_map
        .original_to_internal
        .iter()
        .map(|&source_class| {
            let source_class = source_class as usize;
            let mapped = target_for_source[source_class];
            let target_class = (mapped != u32::MAX)
                .then_some(mapped as usize)
                .unwrap_or(source_class);
            if target_class == initial_class {
                initial_raw
            } else {
                topology.state_map.representative_original_ids[target_class]
            }
        })
        .collect::<Vec<_>>();
    Some(InterchangeMap {
        scanner_state_map: TransportScannerStateMap::Explicit(raw_targets.into()),
    })
}

/// Discover one exact token-macro TI round using the global C partition.  The
/// certificate is over complete vocabulary tokens, which is the coordinate at
/// which transport maps are consumed by the terminal NWA builder.
pub(crate) fn discover_one_round_with_token_macro_context(
    tokenizer: &Tokenizer,
    context: &TokenMacroDiscoveryContext,
    active_terminals: &[bool],
    ignore_terminal: Option<TerminalID>,
) -> TiRoundTransportWitnesses {
    let topology = Arc::clone(&context.topology);
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
    let signatures = context.candidate_signatures.get_or_init(|| {
        // Signatures are round-invariant, so compute them once with every
        // terminal active and reuse for every round. Building the full-active
        // outputs here also keeps the no-op rounds from paying for a fresh
        // per-round `TokenMacroRoundOutputs` scan of all raw states.
        let all_active = vec![true; active_terminals.len()];
        let full_outputs = TokenMacroRoundOutputs::new(tokenizer, &all_active);
        compute_candidate_signatures(&topology, &full_outputs, active_terminals.len())
    });
    let candidate_groups =
        token_macro_candidate_groups_from_signatures(signatures, active_terminals, ignore_terminal);
    // `token_macro_candidate_groups_from_signatures` keeps only groups of 2+
    // terminals whose exact rejection signatures collide. The signature is a
    // necessary condition for interchangeability, so an empty result proves no
    // merge is possible this round. Skip both the per-round outputs scan and
    // the expensive `ExactTokenMacroDfa` build (and its identity-round
    // refinement) entirely; this eliminates the no-op fixed-point round that
    // otherwise rebuilds the full macro DAG.
    if candidate_groups.is_empty() {
        if profile_timing {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability_token_macro] active={} c_classes={} macro_identity_classes=0 macro_rounds=0 identity_setup_ms=0.000 reachable_c_classes={} token_actions={} candidate_groups=0 candidate_pairs=0 exact_checks=0 direct_certificates=0 direct_impossible_rejections=0 refinement_fallbacks=0 accepted_members=0 total_ms={:.3}",
                candidates.len(),
                topology.state_map.representative_original_ids.len(),
                topology.reachable.iter().filter(|&&reachable| reachable).count(),
                topology.tokens.len(),
                started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0),
            );
        }
        return TiRoundTransportWitnesses::singleton(active_terminals);
    }
    let dfa_build_started = profile_timing.then(Instant::now);
    let outputs = TokenMacroRoundOutputs::new(tokenizer, active_terminals);
    let macro_dfa = ExactTokenMacroDfa::new(topology, outputs);
    let dfa_build_ms = dfa_build_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let macro_identity_classes = macro_dfa
        .identity_rounds
        .last()
        .map_or(0, |round| round.representative_by_class.len());
    let candidate_pairs = candidate_groups
        .iter()
        .map(|group| group.len() * group.len().saturating_sub(1) / 2)
        .sum::<usize>();
    let mut result = singleton_partition(active_terminals);
    let mut maps = BTreeMap::<(TerminalID, TerminalID), Arc<TransportScannerStateMap>>::new();
    let mut exact_checks = 0usize;
    let mut accepted = 0usize;
    let mut direct_certificates = 0usize;
    let mut direct_impossible_rejections = 0usize;
    let mut refinement_fallbacks = 0usize;
    let mut direct_ms = 0.0f64;
    let mut refine_ms = 0.0f64;
    for initial_group in candidate_groups {
        let mut unresolved = initial_group;
        while !unresolved.is_empty() {
            let representative = unresolved[0];
            let mut next_unresolved = Vec::with_capacity(unresolved.len().saturating_sub(1));
            for &terminal in &unresolved[1..] {
                exact_checks += 1;
                let direct_started = profile_timing.then(Instant::now);
                let direct_result = macro_dfa.direct_interchange_attempt(representative, terminal);
                if let Some(started) = direct_started {
                    direct_ms += started.elapsed().as_secs_f64() * 1000.0;
                }
                let map = match direct_result {
                    DirectMacroInterchangeResult::Proven(map) => {
                        direct_certificates += 1;
                        Some(map)
                    }
                    DirectMacroInterchangeResult::Impossible => {
                        direct_impossible_rejections += 1;
                        None
                    }
                    DirectMacroInterchangeResult::Inconclusive => {
                        refinement_fallbacks += 1;
                        let refine_started = profile_timing.then(Instant::now);
                        let refine_result =
                            macro_dfa.refinement_interchange_map(representative, terminal);
                        if let Some(started) = refine_started {
                            refine_ms += started.elapsed().as_secs_f64() * 1000.0;
                        }
                        refine_result
                    }
                };
                if let Some(map) = map {
                    accepted += 1;
                    result
                        .get_mut(&representative)
                        .expect("token-macro TI representative must exist")
                        .insert(terminal);
                    result.remove(&terminal);
                    maps.insert((representative, terminal), Arc::new(map.scanner_state_map()));
                } else {
                    next_unresolved.push(terminal);
                }
            }
            unresolved = next_unresolved;
        }
    }
    if profile_timing {
        eprintln!(
            "[glrmask/profile][terminal_interchangeability_token_macro] active={} c_classes={} macro_identity_classes={} macro_rounds={} identity_setup_ms={:.3} dfa_build_ms={:.3} direct_ms={:.3} refine_ms={:.3} reachable_c_classes={} token_actions={} candidate_groups={} candidate_pairs={} exact_checks={} direct_certificates={} direct_impossible_rejections={} refinement_fallbacks={} accepted_members={} total_ms={:.3}",
            candidates.len(),
            macro_dfa.topology.state_map.representative_original_ids.len(),
            macro_identity_classes,
            macro_dfa.stable_round,
            macro_dfa.identity_setup_ms,
            dfa_build_ms,
            direct_ms,
            refine_ms,
            macro_dfa.topology.reachable.iter().filter(|&&reachable| reachable).count(),
            macro_dfa.topology.tokens.len(),
            result.values().filter(|members| members.len() >= 2).count(),
            candidate_pairs,
            exact_checks,
            direct_certificates,
            direct_impossible_rejections,
            refinement_fallbacks,
            accepted,
            started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0),
        );
    }
    assert_partition_invariants(&result, active_terminals);
    TiRoundTransportWitnesses {
        active_before_round: active_terminals.to_vec(),
        partition: result,
        maps,
    }
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
}

impl TiRoundTransportWitnesses {
    fn singleton(active_terminals: &[bool]) -> Self {
        Self {
            active_before_round: active_terminals.to_vec(),
            partition: singleton_partition(active_terminals),
            maps: BTreeMap::new(),
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
    observed_output_pairs: Vec<OutputPair>,
    observed_output_pair_lookup: FxHashMap<OutputPair, u32>,
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
    canonical_rounds: Vec<CanonicalRound>,
    canonical_round_one_class_counts: Option<Vec<u32>>,
    canonical_round_one_source_marks: Vec<u32>,
    canonical_round_one_source_mark_epoch: u32,
    canonical_round_one_affected_sources: Vec<u32>,
    canonical_round_one_changed_scratch: FxHashMap<u32, u32>,
    canonical_round_one_added_scratch: FxHashSet<u32>,
    support_quotient: Option<SupportQuotient>,
    canonical_quotient: Option<CanonicalQuotient>,
    /// Per raw terminal, the canonical quotient classes whose representative
    /// frozen output mentions that terminal.  This is build-local discovery
    /// scratch, used to propose a tiny support-transposition witness before
    /// falling back to exact refinement.
    terminal_quotient_output_supports: Option<Vec<Option<Vec<(u32, u8)>>>>,
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
            Arc::new(RestrictedTopology::new(tokenizer, relevant_bytes)),
        )
    }

    fn from_topology(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        topology: Arc<RestrictedTopology>,
    ) -> Self {
        let raw = Arc::new(TiRawDiscoveryData::new(tokenizer, &topology));
        Self::from_raw_discovery_data(observed_terminals, topology, raw)
    }

    fn from_context(observed_terminals: &[bool], context: &TiDiscoveryContext) -> Self {
        Self::from_raw_discovery_data(
            observed_terminals,
            Arc::clone(&context.topology),
            Arc::clone(&context.raw),
        )
    }

    fn from_raw_discovery_data(
        observed_terminals: &[bool],
        topology: Arc<RestrictedTopology>,
        raw: Arc<TiRawDiscoveryData>,
    ) -> Self {
        let state_count = topology.state_count();
        let terminal_bits = |terminals: &[TerminalID]| {
            OutputBits::from_active(terminals, observed_terminals)
        };
        let finalizers = raw
            .finalizer_terminals_by_state
            .iter()
            .map(|terminals| terminal_bits(terminals))
            .collect::<Vec<_>>();
        // These are the tokenizer's original, frozen future-finalizer sets.
        let future_finalizers = raw
            .future_finalizer_terminals_by_state
            .iter()
            .map(|terminals| terminal_bits(terminals))
            .collect::<Vec<_>>();
        let observed_destinations = &raw.observed_destinations;
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
        let mut observed_output_pair_support_shapes_by_terminal =
            vec![SupportTrackShape::default(); observed_terminals.len()];
        for (id, pair) in observed_output_pairs.iter().enumerate() {
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
        let empty_output = OutputBits::new(0);
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
        let signature_capacity = CHARACTERIZATION_DOMAIN.len()
            + 4
            + topology.max_outdegree
                * (1 + blake3::OUT_LEN + 2 * (size_of::<u32>() + 4 * size_of::<TerminalID>()));
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
            canonical_rounds: vec![CanonicalRound {
                classes: vec![0; state_count],
                representative_by_class: vec![0],
                classes_by_signature_hash: FxHashMap::default(),
            }],
            canonical_round_one_class_counts: None,
            canonical_round_one_source_marks: vec![0; state_count - 1],
            canonical_round_one_source_mark_epoch: 0,
            canonical_round_one_affected_sources: Vec::new(),
            canonical_round_one_changed_scratch: FxHashMap::default(),
            canonical_round_one_added_scratch: FxHashSet::default(),
            support_quotient: None,
            canonical_quotient: None,
            terminal_quotient_output_supports: None,
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
        }
    }

    fn state_count(&self) -> usize {
        self.topology.state_count()
    }

    fn dead_state(&self) -> usize {
        self.topology.dead_state()
    }

    fn ensure_support_quotient(&mut self) {
        if self.support_quotient.is_some() {
            return;
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
            reverse_predecessors,
        });
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
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                deviations.into_boxed_slice(),
            ),
        })
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
        let stable_round = self.ensure_canonical_identity_stable_round();
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
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                source_class_for_target_deviations,
            ),
        })
    }

    #[inline]
    fn quotient_identity_map(&self, quotient: &CanonicalQuotient) -> InterchangeMap {
        InterchangeMap {
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
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
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
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
        let mut changed_by_identity_class =
            std::mem::take(&mut self.canonical_round_one_changed_scratch);
        changed_by_identity_class.clear();
        let mut added_identity_classes =
            std::mem::take(&mut self.canonical_round_one_added_scratch);
        added_identity_classes.clear();
        let identity = &self.canonical_rounds[1];
        let identity_counts = self
            .canonical_round_one_class_counts
            .as_ref()
            .expect("first-round counts initialized");
        let mut swapped_root_class = identity.classes[self.topology.initial_state];
        let mut outputs = SwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
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
            let Some(swapped_class) = self.canonical_round_identity_class_for_signature(
                identity,
                &self.canonical_rounds[0].classes,
                &signature,
            ) else {
                self.canonical_round_one_affected_sources = affected_sources;
                self.canonical_round_one_changed_scratch = changed_by_identity_class;
                self.canonical_round_one_added_scratch = added_identity_classes;
                return false;
            };
            added_identity_classes.insert(swapped_class);
            if source == self.topology.initial_state {
                swapped_root_class = swapped_class;
            }
        }
        self.canonical_round_one_affected_sources = affected_sources;

        let root_matches = swapped_root_class == identity.classes[self.topology.initial_state];
        let accepted = root_matches
            && changed_by_identity_class.iter().all(|(&class, &changed)| {
                changed < identity_counts[class as usize]
                    || added_identity_classes.contains(&class)
            });
        self.canonical_round_one_changed_scratch = changed_by_identity_class;
        self.canonical_round_one_added_scratch = added_identity_classes;
        accepted
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
            scanner_state_map: self
                .topology
                .scanner_map_from_state_targets(&target_representative_for_source_state),
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
            scanner_state_map: self
                .topology
                .scanner_map_from_state_targets(&target_representative_for_source_state),
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
    let context = TiDiscoveryContext::new(tokenizer, relevant_bytes);
    discover_one_round_with_transport_witnesses_in_context(
        tokenizer,
        active_terminals,
        &context,
        ignore_terminal,
    )
}

/// As `discover_one_round_with_transport_witnesses`, but evaluates the exact
/// TI relation over a total global scanner-state quotient.
pub(crate) fn discover_one_round_with_transport_witnesses_with_global_state_quotient(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
    global_state_quotient: &GlobalScannerStateQuotient,
) -> TiRoundTransportWitnesses {
    let context = TiDiscoveryContext::new_with_global_state_quotient(
        tokenizer,
        relevant_bytes,
        global_state_quotient,
    );
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
        let output_shape_candidate_groups = refine_candidate_groups_by_observed_output_pair_shape(
            structural_candidate_groups,
            &dfa.observed_output_pair_support_shapes_by_terminal,
        );
        let output_shape_filter_ms = output_shape_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let output_hypergraph_filter_started_at = profile_timing.then(Instant::now);
        let (candidate_groups, output_hypergraph_rounds) =
            refine_candidate_groups_by_observed_output_hypergraph(
                output_shape_candidate_groups,
                &dfa.observed_output_pairs,
                active_terminals.len(),
            );
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
                "[glrmask/profile][terminal_interchangeability] active={} raw_states={} discovery_states={} selected_bytes={} sparse_edges={} max_outdegree={} root_observed_states={} root_candidate_pairs={} structural_colors={} structural_candidate_groups={} structural_candidate_pairs={} observed_output_candidate_groups={} output_hypergraph_rounds={} exact_candidate_pairs={} group_size_histogram={:?} topology_ms={:.3} candidate_filter_total_ms={:.3} structural_filter_ms={:.3} dfa_setup_ms={:.3} output_shape_filter_ms={:.3} output_hypergraph_filter_ms={:.3}",
                candidates.len(),
                topology.raw_state_count,
                topology.real_state_count,
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
        let mut result = singleton_partition(active_terminals);
        let mut accepted_maps = BTreeMap::<(TerminalID, TerminalID), Arc<TransportScannerStateMap>>::new();
        let mut output_pair_rejections = 0usize;
        let mut output_invariant_checks = 0usize;
        let mut first_round_rejections = 0usize;
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
            let mut unresolved = initial_group;
            while !unresolved.is_empty() {
                let representative = unresolved[0];
                let mut next_unresolved = Vec::with_capacity(unresolved.len().saturating_sub(1));
                for &terminal in &unresolved[1..] {
                    let output_pair_started_at = profile_timing.then(Instant::now);
                    let output_pair_is_closed =
                        dfa.observed_output_pair_set_is_swap_closed(representative, terminal);
                    if let Some(started_at) = output_pair_started_at {
                        output_pair_filter_ns += started_at.elapsed().as_nanos() as u64;
                    }
                    if !output_pair_is_closed {
                        output_pair_rejections += 1;
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
                            None
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
        }

        if profile_timing {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] output_pair_rejections={} output_invariant_checks={} first_round_rejections={} support_transposition_checks={} support_transposition_certified={} support_transposition_no_template={} support_transposition_outside_cone={} support_transposition_root_rejected={} support_transposition_signature_rejected={} direct_exact_checks={} output_pair_filter_ms={:.3} frozen_output_ms={:.3} first_round_ms={:.3} support_transposition_ms={:.3} support_setup_ms={:.3} support_template_ms={:.3} support_cone_ms={:.3} support_verify_ms={:.3} exact_map_ms={:.3} accepted_map_storage_ms={:.3} quotient_certified={} sparse_quotient_certified={} sparse_cone_avg={:.1} sparse_cone_max={} sparse_cone_ms={:.3} sparse_refinement_ms={:.3} sparse_map_ms={:.3} accepted_representative_members={} total_ms={:.3}",
                output_pair_rejections,
                output_invariant_checks,
                first_round_rejections,
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
        TiRoundTransportWitnesses {
            active_before_round: active_terminals.to_vec(),
            partition: result,
            maps: accepted_maps,
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
    let mut macro_transport_groups = 0usize;
    // A transport mode with no quotient deviations is a pure structural
    // relabeling that preserves the ordinary coordinate, so its component is a
    // function of the ordinary coordinate and cannot refine the ordinary
    // partition. Such groups are redundant: we skip building their component
    // table and omit them from the final per-state signature. Because the
    // absolute final-class numbering never reaches the digest (proven: the
    // digest is invariant under class renumbering), dropping redundant groups
    // is exact.
    let mut group_is_redundant = vec![false; groups.len()];
    for (group_index, group) in groups.iter_mut().enumerate() {
        let domain = &modes[group.domain_mode].scanner_state_for_original;
        let source_class_count = domain.innermost_source_class_count();

        let group_quotient_deviations: usize = group
            .modes
            .iter()
            .map(|&mode_index| {
                modes[mode_index]
                    .scanner_state_for_original
                    .quotient_deviations()
                    .map_or(0, |deviations| deviations.len())
            })
            .sum();
        if group_quotient_deviations == 0 {
            group_is_redundant[group_index] = true;
            continue;
        }

        // A direct macro quotient is constant on the shared macro class of a
        // source C class. Classify those macro classes once, then project the
        // component id back to C instead of scanning every C class per mode.
        if let Some(first_parts) = domain.direct_macro_quotient_parts() {
            let shared_macro_classes = first_parts.macro_class_for_source_class;
            let shared_macro_representatives = first_parts.representative_for_macro_class;
            let mut mode_deviations = Vec::with_capacity(group.modes.len());
            let mut all_direct_macro = true;
            for &mode_index in &group.modes {
                let Some(parts) = modes[mode_index]
                    .scanner_state_for_original
                    .direct_macro_quotient_parts()
                else {
                    all_direct_macro = false;
                    break;
                };
                if parts.macro_class_for_source_class.as_ptr() != shared_macro_classes.as_ptr()
                    || parts.representative_for_macro_class.as_ptr()
                        != shared_macro_representatives.as_ptr()
                {
                    all_direct_macro = false;
                    break;
                }
                mode_deviations.push(parts.source_macro_for_target_deviations);
            }
            if all_direct_macro {
                assert_eq!(shared_macro_classes.len(), source_class_count);
                let macro_coordinates = shared_macro_representatives
                    .iter()
                    .map(|&state| ordinary_coordinate_key(state as usize))
                    .collect::<Vec<_>>();
                let mut component_by_signature = FxHashMap::<Vec<u64>, u32>::default();
                let mut component_for_macro = Vec::with_capacity(macro_coordinates.len());
                for source_macro in 0..macro_coordinates.len() {
                    let signature = mode_deviations
                        .iter()
                        .map(|deviations| {
                            let target_macro = deviations
                                .binary_search_by_key(&(source_macro as u32), |&(source, _)| source)
                                .map(|index| deviations[index].1 as usize)
                                .unwrap_or(source_macro);
                            macro_coordinates[target_macro]
                        })
                        .collect::<Vec<_>>();
                    let next_component = component_by_signature.len() as u32;
                    let component = *component_by_signature
                        .entry(signature)
                        .or_insert(next_component);
                    component_for_macro.push(component);
                }
                source_class_mode_evaluations += macro_coordinates.len() * group.modes.len();
                group.component_for_source_class = shared_macro_classes
                    .iter()
                    .map(|&source_macro| component_for_macro[source_macro as usize])
                    .collect();
                macro_transport_groups += 1;
                continue;
            }
        }

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
    let refining_groups: Vec<&ModeGroup> = groups
        .iter()
        .enumerate()
        .filter(|&(group_index, _)| !group_is_redundant[group_index])
        .map(|(_, group)| group)
        .collect();
    let mut class_for_signature = FxHashMap::<SmallVec<[u64; 8]>, u32>::default();
    let mut class_for_state = Vec::with_capacity(state_count);
    for source_state in 0..state_count {
        let mut signature = SmallVec::<[u64; 8]>::with_capacity(refining_groups.len() + 1);
        signature.push(ordinary_coordinate_key(source_state));
        for group in &refining_groups {
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
            "[glrmask/profile][transport_coordinate_quotient] modes={} groups={} macro_transport_groups={} group_source_class_counts={:?} group_mode_counts={:?} source_class_mode_evaluations={} raw_state_group_lookups={} component_build_ms={:.3} final_refinement_ms={:.3} total_ms={:.3}",
            modes.len(),
            groups.len(),
            macro_transport_groups,
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
    // Maximal `(start_tsid, end_tsid, coordinate)` runs of consecutive final
    // TSIDs sharing one non-MAX core coordinate. Precomputed once so `base_lift`
    // can emit one range per run instead of iterating every final TSID.
    coordinate_runs: Vec<(u32, u32, u32)>,
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
        // Precompute maximal contiguous runs of equal, non-MAX coordinates so
        // `base_lift` emits one range per run instead of iterating every final
        // TSID. MAX coordinates break runs (they are skipped when lifting).
        let mut coordinate_runs: Vec<(u32, u32, u32)> = Vec::new();
        {
            let mut idx = 0usize;
            let n = ordinary_coordinates.len();
            while idx < n {
                let coordinate = ordinary_coordinates[idx];
                if coordinate == u32::MAX {
                    idx += 1;
                    continue;
                }
                let start = idx;
                idx += 1;
                while idx < n && ordinary_coordinates[idx] == coordinate {
                    idx += 1;
                }
                coordinate_runs.push((start as u32, (idx - 1) as u32, coordinate));
            }
        }
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
            coordinate_runs,
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

        // `ordinary_coordinates[final_tsid]` is precomputed and identical to
        // `core_coordinate(core_state_map, final_sources[final_tsid])`. Thousands
        // of final TSIDs collapse onto only a few core coordinates, so iterate
        // the precomputed contiguous coordinate runs and emit one range per run,
        // caching each coordinate's token set to avoid redundant range lookups.
        let mut coordinate_tokens_cache = FxHashMap::<u32, SharedTokenSet>::default();
        let lifted = Weight::from_tsid_runs_shared(self.coordinate_runs.iter().map(
            |&(start, end, coordinate)| {
                let tokens = coordinate_tokens_cache
                    .entry(coordinate)
                    .or_insert_with(|| weight.shared_tokens_for_tsid(coordinate))
                    .clone();
                (start, end, tokens)
            },
        ));
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

    // `entry_union_by_group[0]` (group 0 = ordinary + any co-disjoint members)
    // is written below but never read: the disjointness search only scans the
    // member-derived groups (indices `1..`). We therefore avoid materializing
    // the expensive `union_all` of every ordinary entry weight. The ordinary
    // union being disjoint from the member union is exactly equivalent to every
    // individual ordinary entry weight being disjoint from that member union.
    let ordinary_and_members_are_disjoint = member_entries_are_pairwise_disjoint
        && ordinary_entry_weights
            .iter()
            .all(|weight| weight.is_disjoint(&all_member_entry_union));
    if ordinary_and_members_are_disjoint {
        let modes_in_group: Vec<usize> = member_entry_weights
            .iter()
            .map(|(mode_index, _, _)| *mode_index)
            .collect();
        for &mode_index in &modes_in_group {
            group_for_mode[mode_index] = Some(0);
        }
        mode_groups[0].extend(modes_in_group);
        // group-0 union is unused downstream; keep it cheap.
        entry_union_by_group[0] = Weight::empty();
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
        // group-0 union is unused downstream; keep it cheap.
        entry_union_by_group[0] = Weight::empty();
    }

    let mut core_reachable_from = vec![vec![false; core_states.len()]; core_states.len()];
    let grouping_pre_reach_ms = grouping_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
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
    let grouping_reach_ms = grouping_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
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
        let grouping_entryw_ms = grouping_ms - grouping_reach_ms;
        let grouping_reach_only_ms = grouping_reach_ms - grouping_pre_reach_ms;
        eprintln!(
            "[glrmask/profile][ti_post_dwa_grouping_split] pre_reach_ms={:.3} reach_ms={:.3} entryw_ms={:.3} total_ms={:.3}",
            grouping_pre_reach_ms,
            grouping_reach_only_ms,
            grouping_entryw_ms,
            grouping_ms,
        );
    }

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
    fn global_scanner_state_quotient_preserves_exact_ti_round() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
        ]);
        let state_count = tokenizer.num_states() as usize;
        let identity = ManyToOneIdMap::from_original_to_internal_with_representatives(
            (0..state_count as u32).collect(),
            state_count as u32,
            (0..state_count as u32).collect(),
        );
        let global_state_quotient =
            GlobalScannerStateQuotient::from_total_raw_state_map(identity, state_count);

        let raw = discover_one_round_with_transport_witnesses(
            &tokenizer,
            &[true, true],
            &[true; 256],
            None,
        );
        let quotient = discover_one_round_with_transport_witnesses_with_global_state_quotient(
            &tokenizer,
            &[true, true],
            &[true; 256],
            None,
            &global_state_quotient,
        );

        assert_eq!(quotient.partition, raw.partition);
        assert!(quotient.maps.values().all(|map| map.len() == state_count));
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
        let topology = RestrictedTopology::new(&tokenizer, &only_x);
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
