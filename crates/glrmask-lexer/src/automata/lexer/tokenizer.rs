//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::cell::Cell;
use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap;
use rayon::prelude::*;
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use smallvec::SmallVec;

use super::dfa::DFA;
pub use super::dfa::SingletonEpsilonClosures;
use crate::automata::regex::Expr;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

thread_local! {
    /// Static constraint artifacts before v14 use the historical dense DFA
    /// serializer. v14+ may opt into the exact sparse/CSR tokenizer wire form
    /// while still reconstructing the same runtime Tokenizer type.
    static COMPACT_ARTIFACT_SERDE: Cell<bool> = const { Cell::new(false) };
    /// Current sectioned Constraint artifacts can carry the tokenizer in its
    /// own independently decodable section. In that mode the Constraint core
    /// contains only a one-byte placeholder.
    static EXTERNAL_ARTIFACT_SERDE: Cell<bool> = const { Cell::new(false) };
}

pub fn set_compact_artifact_serde(enabled: bool) -> bool {
    COMPACT_ARTIFACT_SERDE.with(|mode| mode.replace(enabled))
}

pub fn set_external_artifact_serde(enabled: bool) -> bool {
    EXTERNAL_ARTIFACT_SERDE.with(|mode| mode.replace(enabled))
}

fn external_artifact_serde_enabled() -> bool {
    EXTERNAL_ARTIFACT_SERDE.with(Cell::get)
}

fn compact_artifact_serde_enabled() -> bool {
    COMPACT_ARTIFACT_SERDE.with(Cell::get)
}

#[derive(Debug, Clone)]
pub(super) struct MatchedTerminalLists {
    offsets: Arc<[usize]>,
    entries: Arc<[TerminalID]>,
}

impl MatchedTerminalLists {
    #[inline]
    fn for_state(&self, state: u32) -> &[TerminalID] {
        let index = state as usize;
        let Some(&start) = self.offsets.get(index) else {
            return &[];
        };
        let Some(&end) = self.offsets.get(index + 1) else {
            return &[];
        };
        &self.entries[start..end]
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tokenizer {
    pub(super) dfa: DFA,
    pub(super) num_terminals: u32,
    /// Current static artifacts may retain their canonical packed transition
    /// topology directly. The DFA then carries only state metadata/epsilon
    /// structure; scalar byte lookup and transition iteration use this sidecar.
    #[serde(default, skip)]
    pub(super) packed_runtime_transitions: Option<Arc<PackedRuntimeTransitions>>,
    /// Rebased borrowed packed transition blocks retained by structural
    /// tokenizer composition. Loaded tokenizers keep ordinary sparse byte
    /// transitions in `packed_runtime_transitions` while their DFA rows are
    /// metadata-only stubs; dropping that sidecar during composition silently
    /// deletes lexer edges. Segments preserve those blocks without expanding
    /// them into dense DFA rows.
    #[serde(default, skip)]
    pub(super) packed_runtime_transition_segments: Arc<[PackedRuntimeTransitionSegment]>,
    /// Runtime-only exact byte-class transition segments. The historical
    /// serialized tokenizer shape contains only `dfa` and `num_terminals`; the
    /// custom serializer expands these segments into that same DFA wire form.
    #[serde(default, skip)]
    pub(super) compressed_transition_segments: Arc<[CompressedTransitionSegment]>,
    /// Current giant-tokenizer artifacts keep per-state labels/epsilon rows in
    /// a compact dictionary-backed sidecar instead of allocating one DFAState
    /// for every serialized state. Freshly compiled tokenizers leave this
    /// empty and continue to use the ordinary DFA representation.
    #[serde(default, skip)]
    pub(super) packed_runtime_metadata: Option<Arc<PackedTokenizerMetadata>>,
    /// Rebased packed metadata blocks retained by structural composition of
    /// loaded tokenizers. Like packed transition segments, these keep TKS3
    /// finalizer/future/epsilon metadata zero-copy instead of expanding one DFA
    /// state record per serialized state.
    #[serde(default, skip)]
    pub(super) packed_runtime_metadata_segments: Arc<[PackedTokenizerMetadataSegment]>,
    /// Dictionary-backed byte-class transition segments used by the compact
    /// giant-tokenizer wire. These are semantically identical to
    /// `compressed_transition_segments`, but intern repeated rows and encode
    /// targets as source-relative deltas.
    #[serde(default, skip)]
    pub(super) packed_compressed_transition_segments: Arc<[PackedCompressedTransitionSegment]>,
    /// Per-terminal regex expressions used to (re)build this tokenizer.
    /// Skipped during (de)serialization because they are only needed during
    /// compile-time simplification for active-terminal rebuilds.
    #[serde(default, skip)]
    pub(super) exprs: Option<Arc<[Expr]>>,
    /// Derived epsilon closures are shared by compile-time analyses.  A
    /// partitioned lexer is queried by many concurrent compiler lanes; without
    /// this cache each lane independently walks the same epsilon DAG for every
    /// raw state.
    #[serde(default, skip)]
    pub(super) singleton_epsilon_closures: OnceLock<Arc<SingletonEpsilonClosures>>,
    /// Sparse accepting-terminal labels per raw state. Runtime commit scans
    /// iterate these lists instead of rescanning a terminal-domain-sized bitset.
    #[serde(default, skip)]
    pub(super) matched_terminals_cache: OnceLock<Arc<MatchedTerminalLists>>,
    /// Exact epsilon-closed frontier after consuming one byte from tokenizer reset.
    #[serde(default, skip)]
    pub(super) initial_byte_frontiers: OnceLock<Arc<[TokenizerStateSet]>>,
    /// Per-state byte sets whose transitions loop to the same raw tokenizer
    /// state. Compiler partitions reuse this table instead of rescanning every
    /// transition row independently.
    #[serde(default, skip)]
    pub(super) all_self_loop_bytes_cache: OnceLock<Arc<[U8Set]>>,
    /// Exact expanded byte-transition count, including compressed segments.
    /// This is immutable after construction except in the two structural
    /// tokenizer transforms, which invalidate all derived caches together.
    #[serde(default, skip)]
    pub(super) transition_count_cache: OnceLock<usize>,
    /// State count after a forced full DFA minimization. Used only by explicit
    /// diagnostics/regression tests; the expensive computation is cached.
    #[serde(default, skip)]
    pub(super) forced_minimized_state_count_cache: OnceLock<usize>,
    /// Whether reset dispatch is followed only by scalar byte transitions.
    /// Several compiler paths query this structural invariant repeatedly; cache
    /// the full reachable-graph proof rather than walking the tokenizer per
    /// vocabulary node or build stage.
    #[serde(default, skip)]
    pub(super) scalar_deterministic_dispatch_cache: OnceLock<bool>,
}

#[derive(Debug, Clone)]
pub struct PackedRuntimeTransitions {
    byte_offsets: Arc<[u32]>,
    bytes: PackedRuntimeBytes,
    targets: PackedRuntimeTargets,
}

#[derive(Debug, Clone)]
pub(super) struct PackedRuntimeTransitionSegment {
    state_offset: u32,
    transitions: Arc<PackedRuntimeTransitions>,
}

impl PackedRuntimeTransitionSegment {
    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset
            && (state - self.state_offset) < self.transitions.state_count() as u32
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        self.transitions
            .transition(state - self.state_offset, byte)
            .and_then(|target| self.state_offset.checked_add(target))
    }

    #[inline]
    fn row(&self, state: u32) -> Option<(&[u8], PackedRuntimeTargetSlice<'_>)> {
        self.transitions.row(state - self.state_offset)
    }
}

#[derive(Debug, Clone)]
enum PackedRowIds {
    U8(Arc<[u8]>),
    U16(Arc<[u16]>),
    U32(Arc<[u32]>),
    BackedU8 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    BackedU16 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    BackedU32 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl PackedRowIds {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::U8(ids) => ids.len(),
            Self::U16(ids) => ids.len(),
            Self::U32(ids) => ids.len(),
            Self::BackedU8 { len, .. }
            | Self::BackedU16 { len, .. }
            | Self::BackedU32 { len, .. } => *len,
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Option<usize> {
        match self {
            Self::U8(ids) => ids.get(index).copied().map(usize::from),
            Self::U16(ids) => ids.get(index).copied().map(usize::from),
            Self::U32(ids) => ids.get(index).copied().map(|value| value as usize),
            Self::BackedU8 {
                backing,
                start,
                len,
            } => (index < *len)
                .then(|| backing[*start + index] as usize),
            Self::BackedU16 {
                backing,
                start,
                len,
            } => {
                if index >= *len {
                    return None;
                }
                let offset = *start + index * 2;
                let bytes = backing.get(offset..offset + 2)?;
                Some(u16::from_le_bytes([bytes[0], bytes[1]]) as usize)
            }
            Self::BackedU32 {
                backing,
                start,
                len,
            } => {
                if index >= *len {
                    return None;
                }
                let offset = *start + index * 4;
                let bytes = backing.get(offset..offset + 4)?;
                Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
            }
        }
    }

    fn all_lt(&self, limit: usize) -> bool {
        match self {
            Self::U8(ids) => ids.iter().all(|&id| (id as usize) < limit),
            Self::U16(ids) => ids.iter().all(|&id| (id as usize) < limit),
            Self::U32(ids) => ids.iter().all(|&id| (id as usize) < limit),
            Self::BackedU8 {
                backing,
                start,
                len,
            } => backing
                .get(*start..*start + *len)
                .is_some_and(|ids| ids.iter().all(|&id| (id as usize) < limit)),
            Self::BackedU16 {
                backing,
                start,
                len,
            } => backing
                .get(*start..*start + *len * 2)
                .is_some_and(|bytes| {
                    bytes
                        .chunks_exact(2)
                        .all(|b| (u16::from_le_bytes([b[0], b[1]]) as usize) < limit)
                }),
            Self::BackedU32 {
                backing,
                start,
                len,
            } => backing
                .get(*start..*start + *len * 4)
                .is_some_and(|bytes| {
                    bytes.chunks_exact(4).all(|b| {
                        (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize) < limit
                    })
                }),
        }
    }
}

#[derive(Debug, Clone)]
enum PackedI16Values {
    Owned(Arc<[i16]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl PackedI16Values {
    #[inline]
    fn get(&self, index: usize) -> Option<i16> {
        match self {
            Self::Owned(values) => values.get(index).copied(),
            Self::Backed {
                backing,
                start,
                len,
            } => {
                if index >= *len {
                    return None;
                }
                let offset = *start + index * 2;
                let bytes = backing.get(offset..offset + 2)?;
                Some(i16::from_le_bytes([bytes[0], bytes[1]]))
            }
        }
    }

    #[inline]
    fn backed_le_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Owned(_) => None,
            Self::Backed {
                backing,
                start,
                len,
            } => backing.get(*start..start.checked_add(len.checked_mul(2)?)?),
        }
    }

    #[inline]
    fn owned_values(&self) -> Option<&[i16]> {
        match self {
            Self::Owned(values) => Some(values),
            Self::Backed { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PackedTokenizerMetadata {
    state_count: u32,
    finalizer_row_ids: PackedRowIds,
    finalizer_rows: Arc<[BitSet]>,
    finalizer_lists: Arc<[Box<[TerminalID]>]>,
    future_row_ids: PackedRowIds,
    future_rows: Arc<[BitSet]>,
    epsilon_states: Arc<[u32]>,
    epsilon_offsets: Arc<[u32]>,
    epsilon_targets: Arc<[u32]>,
}

#[derive(Debug, Clone)]
pub(super) struct PackedTokenizerMetadataSegment {
    state_offset: u32,
    metadata: Arc<PackedTokenizerMetadata>,
}

impl PackedTokenizerMetadataSegment {
    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.metadata.state_count
    }

    #[inline]
    fn local_state(&self, state: u32) -> u32 {
        state - self.state_offset
    }
}

impl PackedTokenizerMetadata {
    #[inline]
    fn finalizers(&self, state: u32) -> Option<&BitSet> {
        let row = self.finalizer_row_ids.get(state as usize)?;
        self.finalizer_rows.get(row)
    }

    #[inline]
    fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        let row = self.finalizer_row_ids.get(state as usize)?;
        self.finalizer_lists.get(row).map(Box::as_ref)
    }

    #[inline]
    fn futures(&self, state: u32) -> Option<&BitSet> {
        let row = self.future_row_ids.get(state as usize)?;
        self.future_rows.get(row)
    }

    #[inline]
    fn epsilon_targets(&self, state: u32) -> &[u32] {
        let Ok(index) = self.epsilon_states.binary_search(&state) else {
            return &[];
        };
        let Some(&start) = self.epsilon_offsets.get(index) else {
            return &[];
        };
        let Some(&end) = self.epsilon_offsets.get(index + 1) else {
            return &[];
        };
        self.epsilon_targets
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }

    #[inline]
    fn has_epsilon_transitions(&self) -> bool {
        !self.epsilon_states.is_empty()
    }

    fn rebased_terminals(
        &self,
        terminal_offset: TerminalID,
        total_terminals: u32,
    ) -> Arc<Self> {
        let offset = terminal_offset as usize;
        let total = total_terminals as usize;
        let rebase_bits = |bits: &BitSet| {
            let mut rebased = BitSet::new(total);
            for terminal in bits.iter() {
                let target = offset
                    .checked_add(terminal)
                    .expect("packed tokenizer terminal offset overflow");
                assert!(target < total, "rebased packed tokenizer terminal exceeds merged domain");
                rebased.set(target);
            }
            rebased
        };
        let finalizer_rows = self
            .finalizer_rows
            .iter()
            .map(rebase_bits)
            .collect::<Vec<_>>();
        let finalizer_lists = self
            .finalizer_lists
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&terminal| {
                        terminal_offset
                            .checked_add(terminal)
                            .expect("packed tokenizer terminal offset overflow")
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let future_rows = self.future_rows.iter().map(rebase_bits).collect::<Vec<_>>();
        Arc::new(Self {
            state_count: self.state_count,
            finalizer_row_ids: self.finalizer_row_ids.clone(),
            finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids: self.future_row_ids.clone(),
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::clone(&self.epsilon_states),
            epsilon_offsets: Arc::clone(&self.epsilon_offsets),
            epsilon_targets: Arc::clone(&self.epsilon_targets),
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct PackedCompressedTransitionSegment {
    state_offset: u32,
    state_count: u32,
    byte_to_class: PackedRuntimeBytes,
    class_members: Arc<[Box<[u8]>]>,
    row_ids: PackedRowIds,
    row_offsets: Arc<[u32]>,
    classes: PackedRuntimeBytes,
    deltas: PackedI16Values,
    overflow_indices: Arc<[u32]>,
    overflow_deltas: Arc<[i32]>,
    expanded_transition_count: usize,
}

impl PackedCompressedTransitionSegment {
    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.state_count
    }

    #[inline]
    fn row_range(&self, state: u32) -> Option<(usize, usize)> {
        let local = (state - self.state_offset) as usize;
        let row = self.row_ids.get(local)?;
        let start = *self.row_offsets.get(row)? as usize;
        let end = *self.row_offsets.get(row + 1)? as usize;
        Some((start, end))
    }

    #[inline]
    fn delta(&self, index: usize) -> Option<i32> {
        let delta = self.deltas.get(index)?;
        if delta != i16::MIN {
            return Some(delta as i32);
        }
        let overflow = self.overflow_indices.binary_search(&(index as u32)).ok()?;
        self.overflow_deltas.get(overflow).copied()
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        let class = *self.byte_to_class.as_slice().get(byte as usize)?;
        let (start, end) = self.row_range(state)?;
        let local = self.classes.as_slice().get(start..end)?;
        let in_row = local.binary_search(&class).ok()?;
        let delta = self.delta(start + in_row)? as i64;
        let target = state as i64 + delta;
        (target >= self.state_offset as i64
            && target < (self.state_offset + self.state_count) as i64)
            .then_some(target as u32)
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        let Some((start, end)) = self.row_range(state) else {
            return;
        };
        for index in start..end {
            let Some(&class) = self.classes.as_slice().get(index) else {
                continue;
            };
            let Some(delta) = self.delta(index) else {
                continue;
            };
            let target = (state as i64 + delta as i64) as u32;
            for &byte in self.class_members[class as usize].iter() {
                row[byte as usize] = target;
            }
        }
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        let Some((start, end)) = self.row_range(state) else {
            return bytes;
        };
        for index in start..end {
            if self.delta(index) != Some(0) {
                continue;
            }
            let class = self.classes.as_slice()[index] as usize;
            for &byte in self.class_members[class].iter() {
                bytes.insert(byte);
            }
        }
        bytes
    }

    fn transition_count(&self, state: u32) -> usize {
        let Some((start, end)) = self.row_range(state) else {
            return 0;
        };
        self.classes.as_slice()[start..end]
            .iter()
            .map(|&class| self.class_members[class as usize].len())
            .sum()
    }
}

#[derive(Debug, Clone)]
enum PackedRuntimeBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl PackedRuntimeBytes {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(values) => values,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }
}

#[derive(Debug, Clone)]
enum PackedRuntimeTargets {
    U16(Arc<[u16]>),
    U32(Arc<[u32]>),
    BackedU16 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    BackedU32 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

#[derive(Clone, Copy)]
enum PackedRuntimeTargetSlice<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
    BackedU16(&'a [u8]),
    BackedU32(&'a [u8]),
}

impl PackedRuntimeTargetSlice<'_> {
    #[inline]
    fn len(self) -> usize {
        match self {
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::BackedU16(bytes) => bytes.len() / 2,
            Self::BackedU32(bytes) => bytes.len() / 4,
        }
    }

    #[inline]
    fn get(self, index: usize) -> Option<u32> {
        match self {
            Self::U16(values) => values.get(index).map(|&value| value as u32),
            Self::U32(values) => values.get(index).copied(),
            Self::BackedU16(bytes) => {
                let start = index.checked_mul(2)?;
                let pair = bytes.get(start..start + 2)?;
                Some(u16::from_le_bytes([pair[0], pair[1]]) as u32)
            }
            Self::BackedU32(bytes) => {
                let start = index.checked_mul(4)?;
                let word = bytes.get(start..start + 4)?;
                Some(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            }
        }
    }
}

impl PackedRuntimeTransitions {
    #[inline]
    fn row(&self, state: u32) -> Option<(&[u8], PackedRuntimeTargetSlice<'_>)> {
        let state = state as usize;
        let byte_start = *self.byte_offsets.get(state)? as usize;
        let byte_end = *self.byte_offsets.get(state + 1)? as usize;
        let bytes = self.bytes.as_slice().get(byte_start..byte_end)?;
        let targets = match &self.targets {
            PackedRuntimeTargets::U16(values) => {
                PackedRuntimeTargetSlice::U16(values.get(byte_start..byte_end)?)
            }
            PackedRuntimeTargets::U32(values) => {
                PackedRuntimeTargetSlice::U32(values.get(byte_start..byte_end)?)
            }
            PackedRuntimeTargets::BackedU16 {
                backing,
                start,
                len,
            } => {
                let all = backing.get(*start..*start + *len * 2)?;
                PackedRuntimeTargetSlice::BackedU16(
                    all.get(byte_start * 2..byte_end * 2)?,
                )
            }
            PackedRuntimeTargets::BackedU32 {
                backing,
                start,
                len,
            } => {
                let all = backing.get(*start..*start + *len * 4)?;
                PackedRuntimeTargetSlice::BackedU32(
                    all.get(byte_start * 4..byte_end * 4)?,
                )
            }
        };
        (bytes.len() == targets.len()).then_some((bytes, targets))
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        let (bytes, targets) = self.row(state)?;
        let index = bytes.binary_search(&byte).ok()?;
        targets.get(index)
    }

    #[inline]
    fn state_count(&self) -> usize {
        self.byte_offsets.len().saturating_sub(1)
    }
}

pub struct FullTokenizerDeterminization {
    pub tokenizer: Tokenizer,
    /// Exact epsilon-closed source-state subset represented by each new state.
    pub source_subsets: Vec<Box<[u32]>>,
    /// First state of an appended exact copy of the source tokenizer.  The
    /// copy is a correctness fallback for parser histories that cease to be
    /// uniform across one determinized subset.
    pub source_state_offset: u32,
    /// A source state whose singleton epsilon closure is exactly the product
    /// subset, or `u32::MAX` when no such scalar representative exists.
    pub exact_source_states: Vec<u32>,
}

/// Exact deterministic transition rows over a byte-equivalence-class alphabet.
/// Targets and row coordinates are local to one DFA component; `state_offset`
/// rebases both into the final partitioned runtime tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTransitionSegment {
    pub state_offset: u32,
    pub state_count: u32,
    pub byte_to_class: Arc<[u8]>,
    pub class_members: Arc<[Box<[u8]>]>,
    pub row_offsets: Arc<[u32]>,
    pub entries: CompressedTransitionEntries,
    pub expanded_transition_count: usize,
}

/// Structure-of-arrays storage for compressed transition entries. Rust pads a
/// `(u8, u32)` tuple to eight bytes; keeping the fields separately uses five
/// bytes per entry while retaining the historical serialized sequence of
/// `(class, target)` pairs.
#[derive(Debug, Clone, Default)]
pub struct CompressedTransitionEntries {
    classes: Arc<[u8]>,
    targets: Arc<[u32]>,
}

impl CompressedTransitionEntries {
    pub fn from_parts(classes: Vec<u8>, targets: Vec<u32>) -> Self {
        assert_eq!(classes.len(), targets.len());
        Self {
            classes: Arc::from(classes.into_boxed_slice()),
            targets: Arc::from(targets.into_boxed_slice()),
        }
    }

    #[inline]
    fn class_slice(&self, start: usize, end: usize) -> &[u8] {
        &self.classes[start..end]
    }

    #[inline]
    fn target(&self, index: usize) -> u32 {
        self.targets[index]
    }

    #[inline]
    fn iter_range(&self, start: usize, end: usize) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.classes[start..end]
            .iter()
            .copied()
            .zip(self.targets[start..end].iter().copied())
    }

    #[inline]
    fn len(&self) -> usize {
        self.classes.len()
    }
}

impl Serialize for CompressedTransitionEntries {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for entry in self
            .classes
            .iter()
            .copied()
            .zip(self.targets.iter().copied())
        {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for CompressedTransitionEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let entries = Vec::<(u8, u32)>::deserialize(deserializer)?;
        let mut classes = Vec::with_capacity(entries.len());
        let mut targets = Vec::with_capacity(entries.len());
        for (class, target) in entries {
            classes.push(class);
            targets.push(target);
        }
        Ok(Self::from_parts(classes, targets))
    }
}

pub mod artifact_serde {
    use super::*;

    #[inline]
    fn u16_values_all_below(bytes: &[u8], limit: usize) -> bool {
        debug_assert_eq!(bytes.len() % 2, 0);
        debug_assert!(limit <= u16::MAX as usize + 1);
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 guarantees SSE2. TKF2 target slabs are not necessarily
            // aligned (the preceding byte-transition slab has arbitrary
            // length), so use unaligned vector loads. Convert the unsigned
            // comparison to a signed one by flipping the sign bit, and test
            // `value > limit - 1` eight targets at a time.
            if limit != 0 && limit <= u16::MAX as usize {
                unsafe {
                    use std::arch::x86_64::{
                        __m128i, _mm_cmpgt_epi16, _mm_loadu_si128, _mm_movemask_epi8,
                        _mm_set1_epi16, _mm_xor_si128,
                    };
                    let sign = _mm_set1_epi16(i16::MIN);
                    let threshold = _mm_xor_si128(
                        _mm_set1_epi16((limit as u16 - 1) as i16),
                        sign,
                    );
                    let mut pos = 0usize;
                    while pos + 16 <= bytes.len() {
                        let values = _mm_loadu_si128(bytes.as_ptr().add(pos).cast::<__m128i>());
                        let unsigned_ordered = _mm_xor_si128(values, sign);
                        let invalid = _mm_cmpgt_epi16(unsigned_ordered, threshold);
                        if _mm_movemask_epi8(invalid) != 0 {
                            return false;
                        }
                        pos += 16;
                    }
                    return bytes[pos..].chunks_exact(2).all(|word| {
                        (u16::from_le_bytes([word[0], word[1]]) as usize) < limit
                    });
                }
            }
        }
        bytes
            .chunks_exact(2)
            .all(|word| (u16::from_le_bytes([word[0], word[1]]) as usize) < limit)
    }
    use serde::{Deserializer, Serializer};

    #[derive(Serialize)]
    struct TokenizerArtifactRef<'a> {
        dfa: &'a DFA,
        num_terminals: u32,
        compressed_transition_segments: &'a [CompressedTransitionSegment],
    }

    #[derive(Deserialize)]
    struct TokenizerArtifact {
        dfa: DFA,
        num_terminals: u32,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    pub fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if external_artifact_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        if compact_artifact_serde_enabled() {
            return packed_artifact_serde::serialize(tokenizer, serializer);
        }
        TokenizerArtifactRef {
            dfa: &tokenizer.dfa,
            num_terminals: tokenizer.num_terminals,
            compressed_transition_segments: &tokenizer.compressed_transition_segments,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        if external_artifact_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom(
                    "invalid external tokenizer placeholder",
                ));
            }
            return Ok(Tokenizer {
                dfa: DFA::new(1),
                num_terminals: 0,
                packed_runtime_transitions: None,
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: None,
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                exprs: None,
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
                initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache: OnceLock::new(),
            });
        }
        if compact_artifact_serde_enabled() {
            return packed_artifact_serde::deserialize(deserializer);
        }
        let artifact = TokenizerArtifact::deserialize(deserializer)?;
        Ok(Tokenizer {
            dfa: artifact.dfa,
            num_terminals: artifact.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        })
    }

    #[derive(Clone, Copy)]
    #[doc(hidden)]
    pub struct FastLayout {
        state_count: usize,
        transition_count: usize,
        epsilon_count: usize,
        finalizer_count: usize,
        future_count: usize,
        terminal_count: usize,
        state_id_width: usize,
        terminal_id_width: usize,
        len: usize,
    }

    impl FastLayout {
        #[inline]
        pub fn len(self) -> usize {
            self.len
        }
    }

    fn fast_layout(tokenizer: &Tokenizer, dfa: &DFA) -> FastLayout {
        const HEADER_LEN: usize = 32;
        let states = dfa.states();
        let state_count = states.len();
        let transition_count = states.iter().map(|state| state.transitions.len()).sum::<usize>();
        let epsilon_count = states.iter().map(|state| state.epsilon_transitions.len()).sum::<usize>();
        let finalizer_count = states
            .iter()
            .map(|state| state.finalizers.iter().count())
            .sum::<usize>();
        let future_count = states
            .iter()
            .map(|state| state.possible_future_group_ids.iter().count())
            .sum::<usize>();
        let terminal_count = tokenizer.num_terminals as usize;
        let state_id_width = if state_count <= u16::MAX as usize + 1 {
            2usize
        } else {
            4usize
        };
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 {
            2usize
        } else {
            4usize
        };
        let offsets_bytes = (state_count + 1) * 4;
        let len = HEADER_LEN
            + terminal_count * 32
            + offsets_bytes + transition_count
            + transition_count * state_id_width
            + offsets_bytes + epsilon_count * state_id_width
            + offsets_bytes + finalizer_count * terminal_id_width
            + offsets_bytes + future_count * terminal_id_width;
        FastLayout {
            state_count,
            transition_count,
            epsilon_count,
            finalizer_count,
            future_count,
            terminal_count,
            state_id_width,
            terminal_id_width,
            len,
        }
    }

    /// Estimated eventual TKF2 wire length without materializing compressed
    /// transition segments. The estimate is exact because compressed segments
    /// retain their expanded transition counts while all state metadata stays
    /// resident in the DFA.
    pub fn estimated_fast_len(tokenizer: &Tokenizer) -> usize {
        const HEADER_LEN: usize = 32;
        let states = tokenizer.dfa.states();
        let state_count = states.len();
        let transition_count = tokenizer.transition_count();
        let epsilon_count = states
            .iter()
            .map(|state| state.epsilon_transitions.len())
            .sum::<usize>();
        let finalizer_count = states
            .iter()
            .map(|state| state.finalizers.iter().count())
            .sum::<usize>();
        let future_count = states
            .iter()
            .map(|state| state.possible_future_group_ids.iter().count())
            .sum::<usize>();
        let terminal_count = tokenizer.num_terminals as usize;
        let state_id_width = if state_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let offsets_bytes = (state_count + 1) * 4;
        HEADER_LEN
            + terminal_count * 32
            + offsets_bytes
            + transition_count
            + transition_count * state_id_width
            + offsets_bytes
            + epsilon_count * state_id_width
            + offsets_bytes
            + finalizer_count * terminal_id_width
            + offsets_bytes
            + future_count * terminal_id_width
    }

    /// Exact TKF2 wire length without constructing the wire buffer. Returns
    /// `None` only for the uncommon compressed-segment representation, whose
    /// direct writer requires materialized transition rows.
    pub fn fast_layout_for_write(tokenizer: &Tokenizer) -> Option<FastLayout> {
        if !tokenizer.compressed_transition_segments.is_empty()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
        {
            return None;
        }
        if tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
        {
            let materialized = tokenizer.materialized_dfa();
            return Some(fast_layout(tokenizer, &materialized));
        }
        Some(fast_layout(tokenizer, &tokenizer.dfa))
    }

    pub fn fast_len(tokenizer: &Tokenizer) -> Option<usize> {
        fast_layout_for_write(tokenizer).map(FastLayout::len)
    }

    fn write_fast_bytes_for_dfa(
        tokenizer: &Tokenizer,
        dfa: &DFA,
        layout: FastLayout,
        out: &mut [u8],
    ) -> Result<(), String> {
        if out.len() != layout.len {
            return Err(format!(
                "fast tokenizer output has length {}, expected {}",
                out.len(), layout.len
            ));
        }
        let states = dfa.states();
        let mut pos = 0usize;
        let put = |out: &mut [u8], pos: &mut usize, bytes: &[u8]| {
            let end = *pos + bytes.len();
            out[*pos..end].copy_from_slice(bytes);
            *pos = end;
        };
        let put_u32 = |out: &mut [u8], pos: &mut usize, value: u32| {
            put(out, pos, &value.to_le_bytes());
        };
        let put_id = |out: &mut [u8], pos: &mut usize, value: u32, width: usize| match width {
            2 => put(out, pos, &(value as u16).to_le_bytes()),
            4 => put(out, pos, &value.to_le_bytes()),
            _ => unreachable!(),
        };
        let put_offsets =
            |out: &mut [u8], pos: &mut usize, lengths: &mut dyn Iterator<Item = usize>| {
                let mut end = 0u32;
                put_u32(out, pos, end);
                for len in lengths {
                    end = end.saturating_add(len as u32);
                    put_u32(out, pos, end);
                }
            };

        put(out, &mut pos, b"TKF2");
        for value in [
            tokenizer.num_terminals,
            layout.state_count as u32,
            layout.transition_count as u32,
            layout.epsilon_count as u32,
            layout.finalizer_count as u32,
            layout.future_count as u32,
        ] {
            put_u32(out, &mut pos, value);
        }
        put(
            out,
            &mut pos,
            &[layout.state_id_width as u8, layout.terminal_id_width as u8, 0, 0],
        );
        for terminal in 0..layout.terminal_count {
            for word in dfa.group_id_to_u8set(terminal as u32).to_words() {
                put(out, &mut pos, &word.to_le_bytes());
            }
        }
        let packed_transitions = tokenizer
            .packed_runtime_transitions
            .as_deref()
            .filter(|packed| {
                packed.state_count() == states.len()
                    && packed.bytes.as_slice().len() == layout.transition_count
            });
        if let Some(packed) = packed_transitions {
            for &offset in packed.byte_offsets.iter() {
                put_u32(out, &mut pos, offset);
            }
            put(out, &mut pos, packed.bytes.as_slice());
            match (&packed.targets, layout.state_id_width) {
                (PackedRuntimeTargets::U16(values), 2) if cfg!(target_endian = "little") => {
                    // SAFETY: u16 has no padding, the source allocation remains
                    // alive for the copy, and TKF2 is explicitly little-endian.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            values.len() * std::mem::size_of::<u16>(),
                        )
                    };
                    put(out, &mut pos, bytes);
                }
                (PackedRuntimeTargets::U32(values), 4) if cfg!(target_endian = "little") => {
                    // SAFETY: u32 has no padding; see the U16 branch above.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            values.len() * std::mem::size_of::<u32>(),
                        )
                    };
                    put(out, &mut pos, bytes);
                }
                (
                    PackedRuntimeTargets::BackedU16 {
                        backing,
                        start,
                        len,
                    },
                    2,
                ) => put(out, &mut pos, &backing[*start..*start + *len * 2]),
                (
                    PackedRuntimeTargets::BackedU32 {
                        backing,
                        start,
                        len,
                    },
                    4,
                ) => put(out, &mut pos, &backing[*start..*start + *len * 4]),
                _ => {
                    for state in 0..states.len() as u32 {
                        let (_, targets) = packed
                            .row(state)
                            .expect("packed runtime transition rows must cover every state");
                        for index in 0..targets.len() {
                            put_id(
                                out,
                                &mut pos,
                                targets
                                    .get(index)
                                    .expect("packed runtime target row length was validated"),
                                layout.state_id_width,
                            );
                        }
                    }
                }
            }
        } else {
            put_offsets(
                out,
                &mut pos,
                &mut states.iter().map(|state| state.transitions.len()),
            );
            for state in states {
                for (byte, _) in state.transitions.iter() {
                    out[pos] = byte;
                    pos += 1;
                }
            }
            for state in states {
                for &target in state.transitions.values() {
                    debug_assert!(layout.state_id_width == 4 || target <= u16::MAX as u32);
                    put_id(out, &mut pos, target, layout.state_id_width);
                }
            }
        }
        put_offsets(
            out,
            &mut pos,
            &mut states.iter().map(|state| state.epsilon_transitions.len()),
        );
        for state in states {
            for &target in &state.epsilon_transitions {
                debug_assert!(layout.state_id_width == 4 || target <= u16::MAX as u32);
                put_id(out, &mut pos, target, layout.state_id_width);
            }
        }
        put_offsets(
            out,
            &mut pos,
            &mut states.iter().map(|state| state.finalizers.iter().count()),
        );
        for state in states {
            for terminal in state.finalizers.iter() {
                let terminal = terminal as u32;
                debug_assert!(layout.terminal_id_width == 4 || terminal <= u16::MAX as u32);
                put_id(out, &mut pos, terminal, layout.terminal_id_width);
            }
        }
        put_offsets(
            out,
            &mut pos,
            &mut states
                .iter()
                .map(|state| state.possible_future_group_ids.iter().count()),
        );
        for state in states {
            for terminal in state.possible_future_group_ids.iter() {
                let terminal = terminal as u32;
                debug_assert!(layout.terminal_id_width == 4 || terminal <= u16::MAX as u32);
                put_id(out, &mut pos, terminal, layout.terminal_id_width);
            }
        }
        debug_assert_eq!(pos, out.len());
        Ok(())
    }

    /// Write TKF2 directly into an exactly-sized destination. Used by the
    /// constraint serializer to overlap tokenizer encoding with independent
    /// final-section copies.
    pub fn write_fast_bytes(tokenizer: &Tokenizer, out: &mut [u8]) -> Result<(), String> {
        if !tokenizer.compressed_transition_segments.is_empty() {
            return Err("direct fast tokenizer write requires materialized transitions".to_owned());
        }
        let layout = fast_layout(tokenizer, &tokenizer.dfa);
        write_fast_bytes_for_dfa(tokenizer, &tokenizer.dfa, layout, out)
    }

    /// Write TKF2 using an exact layout already computed by the caller. This
    /// avoids rescanning every DFA state after `fast_layout_for_write()` was
    /// used to size the final artifact section.
    pub fn write_fast_bytes_with_layout(
        tokenizer: &Tokenizer,
        layout: FastLayout,
        out: &mut [u8],
    ) -> Result<(), String> {
        if !tokenizer.compressed_transition_segments.is_empty()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
        {
            return Err("direct fast tokenizer write requires materialized transitions".to_owned());
        }
        if tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
        {
            let materialized = tokenizer.materialized_dfa();
            return write_fast_bytes_for_dfa(tokenizer, &materialized, layout, out);
        }
        write_fast_bytes_for_dfa(tokenizer, &tokenizer.dfa, layout, out)
    }

    /// Runtime-native current-format tokenizer wire. Unlike the older packed
    /// serializer this does no row hashing and no per-row varint target
    /// encoding: the compiler already owns the exact DFA, so save is a linear
    /// fixed-width copy and load reconstructs the existing packed runtime
    /// transition sidecar directly.
    pub fn to_fast_bytes(tokenizer: &Tokenizer) -> Vec<u8> {
        let materialized;
        let dfa = if tokenizer.compressed_transition_segments.is_empty()
            && tokenizer.packed_compressed_transition_segments.is_empty()
            && tokenizer.packed_runtime_transitions.is_none()
            && tokenizer.packed_runtime_transition_segments.is_empty()
        {
            &tokenizer.dfa
        } else {
            materialized = tokenizer.materialized_dfa();
            &materialized
        };
        let layout = fast_layout(tokenizer, dfa);
        let len = layout.len;
        let mut out = Vec::<u8>::with_capacity(len);
        unsafe {
            out.set_len(len);
        }
        write_fast_bytes_for_dfa(tokenizer, dfa, layout, &mut out)
            .expect("exact fast tokenizer layout should always write successfully");
        out
    }

    const PACKED_WIRE_MAGIC: &[u8; 4] = b"TKP1";
    const SEGMENT_WIRE_MAGIC: &[u8; 4] = b"TKS2";

    struct PackedWireRef<'a>(&'a Tokenizer);

    impl Serialize for PackedWireRef<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            packed_artifact_serde::serialize(self.0, serializer)
        }
    }

    struct PackedWireOwned(Tokenizer);

    impl<'de> Deserialize<'de> for PackedWireOwned {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            packed_artifact_serde::deserialize(deserializer).map(Self)
        }
    }

    /// Compact current tokenizer wire for exceptionally large DFAs. The
    /// historical packed codec interns repeated byte rows and varint-encodes
    /// targets, avoiding TKF2's fixed-width expansion when state ids require
    /// four bytes.
    pub fn to_packed_bytes(tokenizer: &Tokenizer) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PACKED_WIRE_MAGIC);
        bincode::serialize_into(&mut out, &PackedWireRef(tokenizer))
            .expect("packed tokenizer serialization should succeed");
        out
    }

    fn from_packed_bytes(input: &[u8]) -> Result<Tokenizer, String> {
        let body = input
            .strip_prefix(PACKED_WIRE_MAGIC)
            .ok_or_else(|| "invalid packed tokenizer header".to_owned())?;
        bincode::deserialize::<PackedWireOwned>(body)
            .map(|wire| wire.0)
            .map_err(|err| err.to_string())
    }

    /// Persist an already-compressed tokenizer without expanding its byte-class
    /// transition segments. State metadata uses the same flat CSR shape as the
    /// fast tokenizer wire, while only transition rows outside compressed
    /// segments are stored explicitly.
    pub fn to_segment_bytes(tokenizer: &Tokenizer) -> Vec<u8> {
        const HEADER_LEN: usize = 36;
        let states = tokenizer.dfa.states();
        let state_count = states.len();
        let state_id_width = if state_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let terminal_count = tokenizer.num_terminals as usize;
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 { 2 } else { 4 };

        let mut compressed = vec![false; state_count];
        for segment in tokenizer.compressed_transition_segments.iter() {
            let start = segment.state_offset as usize;
            let end = start.saturating_add(segment.state_count as usize).min(state_count);
            compressed[start..end].fill(true);
        }
        let residual_transition_count = states
            .iter()
            .enumerate()
            .filter(|(state, _)| !compressed[*state])
            .map(|(_, state)| state.transitions.len())
            .sum::<usize>();
        let epsilon_count = states.iter().map(|state| state.epsilon_transitions.len()).sum::<usize>();
        let finalizer_count = states
            .iter()
            .map(|state| state.finalizers.iter().count())
            .sum::<usize>();
        let future_count = states
            .iter()
            .map(|state| state.possible_future_group_ids.iter().count())
            .sum::<usize>();
        let segment_blob = bincode::serialize(tokenizer.compressed_transition_segments.as_ref())
            .expect("compressed tokenizer segments should serialize");
        let offsets_bytes = (state_count + 1) * 4;
        let len = HEADER_LEN
            + terminal_count * 32
            + offsets_bytes
            + residual_transition_count
            + residual_transition_count * state_id_width
            + offsets_bytes
            + epsilon_count * state_id_width
            + offsets_bytes
            + finalizer_count * terminal_id_width
            + offsets_bytes
            + future_count * terminal_id_width
            + segment_blob.len();
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(SEGMENT_WIRE_MAGIC);
        for value in [
            tokenizer.num_terminals,
            state_count as u32,
            residual_transition_count as u32,
            epsilon_count as u32,
            finalizer_count as u32,
            future_count as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(segment_blob.len() as u64).to_le_bytes());
        debug_assert_eq!(out.len(), HEADER_LEN);
        for terminal in 0..terminal_count {
            for word in tokenizer.dfa.group_id_to_u8set(terminal as u32).to_words() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        let put_id = |out: &mut Vec<u8>, value: u32, width: usize| match width {
            2 => out.extend_from_slice(&(value as u16).to_le_bytes()),
            4 => out.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!(),
        };
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for (index, state) in states.iter().enumerate() {
            if !compressed[index] {
                end += state.transitions.len() as u32;
            }
            out.extend_from_slice(&end.to_le_bytes());
        }
        for (index, state) in states.iter().enumerate() {
            if !compressed[index] {
                out.extend(state.transitions.iter().map(|(byte, _)| byte));
            }
        }
        for (index, state) in states.iter().enumerate() {
            if !compressed[index] {
                for &target in state.transitions.values() {
                    put_id(&mut out, target, state_id_width);
                }
            }
        }
        let mut put_sparse = |counts: &mut dyn Iterator<Item = usize>, ids: &mut dyn Iterator<Item = u32>, width: usize| {
            let mut end = 0u32;
            out.extend_from_slice(&end.to_le_bytes());
            for count in counts {
                end += count as u32;
                out.extend_from_slice(&end.to_le_bytes());
            }
            for id in ids {
                put_id(&mut out, id, width);
            }
        };
        put_sparse(
            &mut states.iter().map(|state| state.epsilon_transitions.len()),
            &mut states.iter().flat_map(|state| state.epsilon_transitions.iter().copied()),
            state_id_width,
        );
        put_sparse(
            &mut states.iter().map(|state| state.finalizers.iter().count()),
            &mut states.iter().flat_map(|state| state.finalizers.iter().map(|id| id as u32)),
            terminal_id_width,
        );
        put_sparse(
            &mut states.iter().map(|state| state.possible_future_group_ids.iter().count()),
            &mut states
                .iter()
                .flat_map(|state| state.possible_future_group_ids.iter().map(|id| id as u32)),
            terminal_id_width,
        );
        out.extend_from_slice(&segment_blob);
        debug_assert_eq!(out.len(), len);
        out
    }

    fn from_segment_bytes(input: &[u8]) -> Result<Tokenizer, String> {
        const HEADER_LEN: usize = 36;
        if input.len() < HEADER_LEN || !input.starts_with(SEGMENT_WIRE_MAGIC) {
            return Err("invalid compressed tokenizer header".to_owned());
        }
        let mut pos = 4usize;
        let take_u32 = |input: &[u8], pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or_else(|| "compressed tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated compressed tokenizer".to_owned())?;
            *pos = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let num_terminals = take_u32(input, &mut pos)?;
        let state_count = take_u32(input, &mut pos)? as usize;
        let residual_transition_count = take_u32(input, &mut pos)? as usize;
        let epsilon_count = take_u32(input, &mut pos)? as usize;
        let finalizer_count = take_u32(input, &mut pos)? as usize;
        let future_count = take_u32(input, &mut pos)? as usize;
        let segment_len_end = pos + 8;
        let segment_blob_len = u64::from_le_bytes(
            input.get(pos..segment_len_end)
                .ok_or_else(|| "truncated compressed tokenizer segment length".to_owned())?
                .try_into().unwrap(),
        ) as usize;
        pos = segment_len_end;
        if state_count == 0 {
            return Err("compressed tokenizer has no states".to_owned());
        }
        let state_id_width = if state_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let terminal_count = num_terminals as usize;
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let mut group_id_to_u8set = Vec::with_capacity(terminal_count);
        for _ in 0..terminal_count {
            let mut words = [0u64; 4];
            for word in &mut words {
                let end = pos + 8;
                *word = u64::from_le_bytes(
                    input.get(pos..end)
                        .ok_or_else(|| "truncated compressed tokenizer groups".to_owned())?
                        .try_into().unwrap(),
                );
                pos = end;
            }
            group_id_to_u8set.push(U8Set::from_words(words));
        }
        let read_u32_vec = |input: &[u8], pos: &mut usize, count: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count.checked_mul(4).ok_or_else(|| "compressed tokenizer vector overflow".to_owned())?;
            let end = pos.checked_add(bytes_len).ok_or_else(|| "compressed tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated compressed tokenizer vector".to_owned())?;
            let mut out = Vec::with_capacity(count);
            out.extend(bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
            *pos = end;
            Ok(out)
        };
        let read_ids = |input: &[u8], pos: &mut usize, count: usize, width: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count.checked_mul(width).ok_or_else(|| "compressed tokenizer id vector overflow".to_owned())?;
            let end = pos.checked_add(bytes_len).ok_or_else(|| "compressed tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated compressed tokenizer ids".to_owned())?;
            let out = match width {
                2 => bytes.chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap()) as u32).collect(),
                4 => bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect(),
                _ => unreachable!(),
            };
            *pos = end;
            Ok(out)
        };
        let validate_offsets = |offsets: &[u32], count: usize, label: &str| -> Result<(), String> {
            if offsets.len() != state_count + 1
                || offsets.first().copied() != Some(0)
                || offsets.last().copied() != Some(count as u32)
                || offsets.windows(2).any(|pair| pair[0] > pair[1])
            {
                return Err(format!("invalid compressed tokenizer {label} offsets"));
            }
            Ok(())
        };
        let transition_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&transition_offsets, residual_transition_count, "transition")?;
        let transition_bytes_end = pos + residual_transition_count;
        let transition_bytes = input.get(pos..transition_bytes_end)
            .ok_or_else(|| "truncated compressed tokenizer transition bytes".to_owned())?;
        pos = transition_bytes_end;
        let transition_targets = read_ids(input, &mut pos, residual_transition_count, state_id_width)?;
        if transition_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("compressed tokenizer transition target is out of range".to_owned());
        }
        let epsilon_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&epsilon_offsets, epsilon_count, "epsilon")?;
        let epsilon_targets = read_ids(input, &mut pos, epsilon_count, state_id_width)?;
        if epsilon_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("compressed tokenizer epsilon target is out of range".to_owned());
        }
        let finalizer_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&finalizer_offsets, finalizer_count, "finalizer")?;
        let finalizers = read_ids(input, &mut pos, finalizer_count, terminal_id_width)?;
        let future_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&future_offsets, future_count, "future")?;
        let futures = read_ids(input, &mut pos, future_count, terminal_id_width)?;
        if finalizers.iter().chain(&futures).any(|&id| id as usize >= terminal_count) {
            return Err("compressed tokenizer terminal id is out of range".to_owned());
        }
        let segment_end = pos.checked_add(segment_blob_len)
            .ok_or_else(|| "compressed tokenizer segment length overflow".to_owned())?;
        let segment_blob = input.get(pos..segment_end)
            .ok_or_else(|| "truncated compressed tokenizer segments".to_owned())?;
        if segment_end != input.len() {
            return Err("trailing bytes in compressed tokenizer".to_owned());
        }
        let compressed_transition_segments =
            bincode::deserialize::<Vec<CompressedTransitionSegment>>(segment_blob)
                .map_err(|err| err.to_string())?;
        let mut previous_end = 0usize;
        for segment in &compressed_transition_segments {
            let start = segment.state_offset as usize;
            let end = start.checked_add(segment.state_count as usize)
                .ok_or_else(|| "compressed tokenizer segment state range overflow".to_owned())?;
            if start < previous_end || end > state_count || segment.row_offsets.len() != segment.state_count as usize + 1 {
                return Err("invalid compressed tokenizer segment state range".to_owned());
            }
            previous_end = end;
        }
        let mut dfa = DFA::new_from_sparse_metadata(
            group_id_to_u8set,
            &epsilon_offsets,
            &epsilon_targets,
            &finalizer_offsets,
            &finalizers,
            &future_offsets,
            &futures,
        );
        for state in 0..state_count {
            let a = transition_offsets[state] as usize;
            let b = transition_offsets[state + 1] as usize;
            if a != b {
                dfa.set_transitions_from_sorted_entries(
                    state as u32,
                    transition_bytes[a..b]
                        .iter()
                        .copied()
                        .zip(transition_targets[a..b].iter().copied())
                        .collect(),
                );
            }
        }
        Ok(Tokenizer {
            dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(compressed_transition_segments.into_boxed_slice()),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        })
    }


    const HUGE_WIRE_MAGIC: &[u8; 4] = b"TKS3";
    const HUGE_WIRE_HEADER_LEN: usize = 52;

    struct PackedSegmentBuild {
        row_ids: Vec<u32>,
        row_offsets: Vec<u32>,
        classes: Vec<u8>,
        deltas: Vec<i16>,
        overflow_indices: Vec<u32>,
        overflow_deltas: Vec<i32>,
    }

    #[inline]
    fn packed_row_id_width(row_count: usize) -> usize {
        if row_count <= u8::MAX as usize + 1 {
            1
        } else if row_count <= u16::MAX as usize + 1 {
            2
        } else {
            4
        }
    }

    fn write_packed_row_ids(out: &mut Vec<u8>, ids: &[u32], width: usize) {
        match width {
            1 => out.extend(ids.iter().map(|&id| id as u8)),
            2 => {
                for &id in ids {
                    out.extend_from_slice(&(id as u16).to_le_bytes());
                }
            }
            4 => {
                for &id in ids {
                    out.extend_from_slice(&id.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
    }

    fn read_packed_row_ids(
        input: &[u8],
        pos: &mut usize,
        count: usize,
        width: usize,
        backing: Option<(&Arc<Vec<u8>>, usize)>,
    ) -> Result<PackedRowIds, String> {
        let local_start = *pos;
        let bytes_len = count
            .checked_mul(width)
            .ok_or_else(|| "packed tokenizer row-id length overflow".to_owned())?;
        let end = pos
            .checked_add(bytes_len)
            .ok_or_else(|| "packed tokenizer row-id offset overflow".to_owned())?;
        let bytes = input
            .get(*pos..end)
            .ok_or_else(|| "truncated packed tokenizer row ids".to_owned())?;
        *pos = end;
        if let Some((artifact, section_start)) = backing {
            let start = section_start
                .checked_add(local_start)
                .ok_or_else(|| "packed tokenizer backing offset overflow".to_owned())?;
            return match width {
                1 => Ok(PackedRowIds::BackedU8 {
                    backing: Arc::clone(artifact),
                    start,
                    len: count,
                }),
                2 => Ok(PackedRowIds::BackedU16 {
                    backing: Arc::clone(artifact),
                    start,
                    len: count,
                }),
                4 => Ok(PackedRowIds::BackedU32 {
                    backing: Arc::clone(artifact),
                    start,
                    len: count,
                }),
                _ => Err("invalid packed tokenizer row-id width".to_owned()),
            };
        }
        match width {
            1 => Ok(PackedRowIds::U8(Arc::from(bytes))),
            2 => Ok(PackedRowIds::U16(Arc::from(
                bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))),
            4 => Ok(PackedRowIds::U32(Arc::from(
                bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))),
            _ => Err("invalid packed tokenizer row-id width".to_owned()),
        }
    }

    fn packed_segment_build(segment: &CompressedTransitionSegment) -> Option<PackedSegmentBuild> {
        use std::hash::{Hash, Hasher};
        if segment.state_count > i32::MAX as u32 || segment.class_members.len() > u8::MAX as usize {
            return None;
        }
        let mut by_hash = FxHashMap::<u64, Vec<(u32, u32)>>::default();
        let mut row_ids = Vec::with_capacity(segment.state_count as usize);
        let mut row_offsets = vec![0u32];
        let mut classes = Vec::<u8>::new();
        let mut deltas = Vec::<i16>::new();
        let mut overflow_indices = Vec::<u32>::new();
        let mut overflow_deltas = Vec::<i32>::new();

        let rows_equal = |left: u32, right: u32| {
            let la = segment.row_offsets[left as usize] as usize;
            let lb = segment.row_offsets[left as usize + 1] as usize;
            let ra = segment.row_offsets[right as usize] as usize;
            let rb = segment.row_offsets[right as usize + 1] as usize;
            if lb - la != rb - ra || segment.entries.class_slice(la, lb) != segment.entries.class_slice(ra, rb) {
                return false;
            }
            (0..lb - la).all(|offset| {
                let lhs = segment.entries.target(la + offset) as i64 - left as i64;
                let rhs = segment.entries.target(ra + offset) as i64 - right as i64;
                lhs == rhs
            })
        };

        for local_state in 0..segment.state_count {
            let start = segment.row_offsets[local_state as usize] as usize;
            let end = segment.row_offsets[local_state as usize + 1] as usize;
            let mut hasher = rustc_hash::FxHasher::default();
            (end - start).hash(&mut hasher);
            for index in start..end {
                segment.entries.class_slice(index, index + 1)[0].hash(&mut hasher);
                let delta = segment.entries.target(index) as i64 - local_state as i64;
                delta.hash(&mut hasher);
            }
            let hash = hasher.finish();
            let existing = by_hash.get(&hash).and_then(|candidates| {
                candidates
                    .iter()
                    .find(|(_, representative)| rows_equal(local_state, *representative))
                    .map(|(row, _)| *row)
            });
            let row = if let Some(row) = existing {
                row
            } else {
                let row = u32::try_from(row_offsets.len() - 1).ok()?;
                for index in start..end {
                    let class = segment.entries.class_slice(index, index + 1)[0];
                    let delta64 = segment.entries.target(index) as i64 - local_state as i64;
                    let delta = i32::try_from(delta64).ok()?;
                    let entry_index = u32::try_from(classes.len()).ok()?;
                    classes.push(class);
                    if let Ok(short) = i16::try_from(delta) {
                        if short != i16::MIN {
                            deltas.push(short);
                            continue;
                        }
                    }
                    deltas.push(i16::MIN);
                    overflow_indices.push(entry_index);
                    overflow_deltas.push(delta);
                }
                row_offsets.push(u32::try_from(classes.len()).ok()?);
                by_hash.entry(hash).or_default().push((row, local_state));
                row
            };
            row_ids.push(row);
        }
        Some(PackedSegmentBuild {
            row_ids,
            row_offsets,
            classes,
            deltas,
            overflow_indices,
            overflow_deltas,
        })
    }

    fn metadata_rows(tokenizer: &Tokenizer) -> Option<(Vec<[u64; 2]>, Vec<u32>, Vec<[u64; 2]>, Vec<u32>)> {
        if tokenizer.num_terminals > 128 {
            return None;
        }
        fn words2(bits: &BitSet) -> [u64; 2] {
            let words = bits.words();
            [words.first().copied().unwrap_or(0), words.get(1).copied().unwrap_or(0)]
        }
        let mut final_map = FxHashMap::<[u64; 2], u32>::default();
        let mut final_rows = Vec::<[u64; 2]>::new();
        let mut final_ids = Vec::<u32>::with_capacity(tokenizer.dfa.num_states());
        let mut future_map = FxHashMap::<[u64; 2], u32>::default();
        let mut future_rows = Vec::<[u64; 2]>::new();
        let mut future_ids = Vec::<u32>::with_capacity(tokenizer.dfa.num_states());
        for state in tokenizer.dfa.states() {
            let final_key = words2(&state.finalizers);
            let final_id = if let Some(&id) = final_map.get(&final_key) {
                id
            } else {
                let id = u32::try_from(final_rows.len()).ok()?;
                final_rows.push(final_key);
                final_map.insert(final_key, id);
                id
            };
            final_ids.push(final_id);
            let future_key = words2(&state.possible_future_group_ids);
            let future_id = if let Some(&id) = future_map.get(&future_key) {
                id
            } else {
                let id = u32::try_from(future_rows.len()).ok()?;
                future_rows.push(future_key);
                future_map.insert(future_key, id);
                id
            };
            future_ids.push(future_id);
        }
        Some((final_rows, final_ids, future_rows, future_ids))
    }

    fn packed_row_ids_from_u32(ids: Vec<u32>, row_count: usize) -> PackedRowIds {
        match packed_row_id_width(row_count) {
            1 => PackedRowIds::U8(Arc::from(
                ids.into_iter()
                    .map(|id| id as u8)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )),
            2 => PackedRowIds::U16(Arc::from(
                ids.into_iter()
                    .map(|id| id as u16)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )),
            4 => PackedRowIds::U32(Arc::from(ids.into_boxed_slice())),
            _ => unreachable!(),
        }
    }

    fn bitset_from_words2(words: [u64; 2], terminal_count: usize) -> BitSet {
        let mut bits = BitSet::new(terminal_count);
        for (slot, value) in bits.words_mut().iter_mut().zip(words) {
            *slot = value;
        }
        bits
    }

    /// Convert a large compressed-suffix tokenizer into the same packed form
    /// used by freshly loaded TKS3 artifacts. This is a runtime storage change,
    /// not a serialization cache: subsequent tokenization reads packed metadata
    /// and packed compressed transition rows directly.
    pub fn compact_large_runtime(tokenizer: &mut Tokenizer) -> bool {
        const MIN_COMPRESSED_STATES: usize = 100_000;
        if tokenizer.compressed_transition_segments.is_empty()
            || tokenizer.packed_runtime_metadata.is_some()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
            || tokenizer.num_terminals > 128
        {
            return false;
        }
        let compressed_states = tokenizer
            .compressed_transition_segments
            .iter()
            .map(|segment| segment.state_count as usize)
            .sum::<usize>();
        if compressed_states < MIN_COMPRESSED_STATES {
            return false;
        }

        let packed_builds = match tokenizer
            .compressed_transition_segments
            .iter()
            .map(packed_segment_build)
            .collect::<Option<Vec<_>>>()
        {
            Some(value) => value,
            None => return false,
        };
        let (final_rows_raw, final_ids, future_rows_raw, future_ids) =
            match metadata_rows(tokenizer) {
                Some(value) => value,
                None => return false,
            };

        let terminal_count = tokenizer.num_terminals as usize;
        let finalizer_rows = final_rows_raw
            .into_iter()
            .map(|words| bitset_from_words2(words, terminal_count))
            .collect::<Vec<_>>();
        let finalizer_lists = finalizer_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let future_rows = future_rows_raw
            .into_iter()
            .map(|words| bitset_from_words2(words, terminal_count))
            .collect::<Vec<_>>();

        let mut epsilon_states = Vec::<u32>::new();
        let mut epsilon_offsets = vec![0u32];
        let mut epsilon_targets = Vec::<u32>::new();
        for (state, row) in tokenizer.dfa.states().iter().enumerate() {
            if row.epsilon_transitions.is_empty() {
                continue;
            }
            epsilon_states.push(state as u32);
            epsilon_targets.extend_from_slice(&row.epsilon_transitions);
            let Some(end) = u32::try_from(epsilon_targets.len()).ok() else {
                return false;
            };
            epsilon_offsets.push(end);
        }

        let metadata = Arc::new(PackedTokenizerMetadata {
            state_count: tokenizer.dfa.num_states() as u32,
            finalizer_row_ids: packed_row_ids_from_u32(final_ids, finalizer_rows.len()),
            finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids: packed_row_ids_from_u32(future_ids, future_rows.len()),
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::from(epsilon_states.into_boxed_slice()),
            epsilon_offsets: Arc::from(epsilon_offsets.into_boxed_slice()),
            epsilon_targets: Arc::from(epsilon_targets.into_boxed_slice()),
        });

        let packed_segments = tokenizer
            .compressed_transition_segments
            .iter()
            .zip(packed_builds)
            .map(|(segment, packed)| PackedCompressedTransitionSegment {
                state_offset: segment.state_offset,
                state_count: segment.state_count,
                byte_to_class: PackedRuntimeBytes::Owned(Arc::clone(&segment.byte_to_class)),
                class_members: Arc::clone(&segment.class_members),
                row_ids: packed_row_ids_from_u32(
                    packed.row_ids,
                    packed.row_offsets.len().saturating_sub(1),
                ),
                row_offsets: Arc::from(packed.row_offsets.into_boxed_slice()),
                classes: PackedRuntimeBytes::Owned(Arc::from(packed.classes.into_boxed_slice())),
                deltas: PackedI16Values::Owned(Arc::from(packed.deltas.into_boxed_slice())),
                overflow_indices: Arc::from(packed.overflow_indices.into_boxed_slice()),
                overflow_deltas: Arc::from(packed.overflow_deltas.into_boxed_slice()),
                expanded_transition_count: segment.expanded_transition_count,
            })
            .collect::<Vec<_>>();

        tokenizer.packed_runtime_metadata = Some(metadata);
        tokenizer.packed_compressed_transition_segments =
            Arc::from(packed_segments.into_boxed_slice());
        tokenizer.compressed_transition_segments = Arc::from([]);
        tokenizer.invalidate_derived_caches();
        true
    }

    /// Add contiguous runtime transition storage for large ordinary DFAs.  The
    /// full DFA is retained for compiler/analysis consumers, but runtime byte
    /// stepping immediately prefers this sidecar.  This is therefore a real
    /// runtime representation, not serialized-wire precomputation.
    pub fn compact_large_fast_runtime(tokenizer: &mut Tokenizer) -> bool {
        const MIN_TRANSITIONS: usize = 1_000_000;
        if tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
            || !tokenizer.compressed_transition_segments.is_empty()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
        {
            return false;
        }
        let transition_count = tokenizer.dfa.transition_count();
        if transition_count < MIN_TRANSITIONS {
            return false;
        }
        let states = tokenizer.dfa.states();
        let state_count = states.len();
        let mut offsets = Vec::<u32>::with_capacity(state_count + 1);
        let mut bytes = Vec::<u8>::with_capacity(transition_count);
        offsets.push(0);
        if state_count <= u16::MAX as usize + 1 {
            let mut targets = Vec::<u16>::with_capacity(transition_count);
            for state in states {
                bytes.extend(state.transitions.iter().map(|(byte, _)| byte));
                targets.extend(state.transitions.values().map(|&target| {
                    u16::try_from(target).expect("u16-sized DFA must have u16 transition targets")
                }));
                offsets.push(
                    u32::try_from(bytes.len())
                        .expect("runtime tokenizer transition count must fit u32"),
                );
            }
            tokenizer.packed_runtime_transitions = Some(Arc::new(PackedRuntimeTransitions {
                byte_offsets: Arc::from(offsets.into_boxed_slice()),
                bytes: PackedRuntimeBytes::Owned(Arc::from(bytes.into_boxed_slice())),
                targets: PackedRuntimeTargets::U16(Arc::from(targets.into_boxed_slice())),
            }));
        } else {
            let mut targets = Vec::<u32>::with_capacity(transition_count);
            for state in states {
                bytes.extend(state.transitions.iter().map(|(byte, _)| byte));
                targets.extend(state.transitions.values().copied());
                offsets.push(
                    u32::try_from(bytes.len())
                        .expect("runtime tokenizer transition count must fit u32"),
                );
            }
            tokenizer.packed_runtime_transitions = Some(Arc::new(PackedRuntimeTransitions {
                byte_offsets: Arc::from(offsets.into_boxed_slice()),
                bytes: PackedRuntimeBytes::Owned(Arc::from(bytes.into_boxed_slice())),
                targets: PackedRuntimeTargets::U32(Arc::from(targets.into_boxed_slice())),
            }));
        }
        true
    }

    fn write_packed_row_ids_any(out: &mut Vec<u8>, ids: &PackedRowIds, width: usize) {
        match (ids, width) {
            (PackedRowIds::U8(values), 1) => out.extend_from_slice(values),
            (PackedRowIds::U16(values), 2) if cfg!(target_endian = "little") => {
                // SAFETY: u16 has no padding and TKS3 is little-endian.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        values.len() * std::mem::size_of::<u16>(),
                    )
                };
                out.extend_from_slice(bytes);
            }
            (PackedRowIds::U32(values), 4) if cfg!(target_endian = "little") => {
                // SAFETY: u32 has no padding and TKS3 is little-endian.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        values.len() * std::mem::size_of::<u32>(),
                    )
                };
                out.extend_from_slice(bytes);
            }
            _ => {
                for index in 0..ids.len() {
                    let id = ids
                        .get(index)
                        .expect("packed row id was validated to cover every state")
                        as u32;
                    match width {
                        1 => out.push(id as u8),
                        2 => out.extend_from_slice(&(id as u16).to_le_bytes()),
                        4 => out.extend_from_slice(&id.to_le_bytes()),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    #[inline]
    fn extend_u32_le(out: &mut Vec<u8>, values: &[u32]) {
        if cfg!(target_endian = "little") {
            // SAFETY: u32 has no padding and the source slice outlives the copy.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    #[inline]
    fn extend_i16_le(out: &mut Vec<u8>, values: &[i16]) {
        if cfg!(target_endian = "little") {
            // SAFETY: i16 has no padding and the source slice outlives the copy.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    #[inline]
    fn extend_i32_le(out: &mut Vec<u8>, values: &[i32]) {
        if cfg!(target_endian = "little") {
            // SAFETY: i32 has no padding and the source slice outlives the copy.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    fn build_huge_bytes_from_packed_runtime(tokenizer: &Tokenizer) -> Option<Vec<u8>> {
        let metadata = tokenizer.packed_runtime_metadata.as_deref()?;
        let segments = tokenizer.packed_compressed_transition_segments.as_ref();
        if segments.is_empty()
            || tokenizer.num_terminals > 128
            || tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
            || !tokenizer.packed_runtime_metadata_segments.is_empty()
        {
            return None;
        }
        let state_count = tokenizer.dfa.num_states();
        if metadata.state_count as usize != state_count
            || metadata.finalizer_row_ids.len() != state_count
            || metadata.future_row_ids.len() != state_count
        {
            return None;
        }
        let prefix_state_count = segments.first()?.state_offset as usize;
        let mut expected = prefix_state_count;
        for segment in segments {
            if segment.state_offset as usize != expected {
                return None;
            }
            expected = expected.checked_add(segment.state_count as usize)?;
        }
        if expected != state_count {
            return None;
        }

        let prefix_states = &tokenizer.dfa.states()[..prefix_state_count];
        let residual_transition_count = prefix_states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        let expanded_transition_count = residual_transition_count
            + segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
        let final_width = packed_row_id_width(metadata.finalizer_rows.len());
        let future_width = packed_row_id_width(metadata.future_rows.len());

        let metadata_word_count = (tokenizer.num_terminals as usize).div_ceil(64);
        let mut exact_len = HUGE_WIRE_HEADER_LEN
            + tokenizer.num_terminals as usize * 32
            + (prefix_state_count + 1) * 4
            + residual_transition_count
            + residual_transition_count * 4
            + metadata.finalizer_rows.len() * metadata_word_count * 8
            + metadata.future_rows.len() * metadata_word_count * 8
            + state_count * final_width
            + state_count * future_width
            + metadata.epsilon_states.len() * 4
            + metadata.epsilon_offsets.len() * 4
            + metadata.epsilon_targets.len() * 4;
        for segment in segments {
            let row_count = segment.row_offsets.len().checked_sub(1)?;
            let entry_count = segment.classes.as_slice().len();
            let row_width = packed_row_id_width(row_count);
            exact_len = exact_len
                .checked_add(28)?
                .checked_add(segment.byte_to_class.as_slice().len())?
                .checked_add(segment.row_ids.len().checked_mul(row_width)?)?
                .checked_add(segment.row_offsets.len().checked_mul(4)?)?
                .checked_add(entry_count)?
                .checked_add(entry_count.checked_mul(2)?)?
                .checked_add(segment.overflow_indices.len().checked_mul(4)?)?
                .checked_add(segment.overflow_deltas.len().checked_mul(4)?)?;
        }
        let mut out = Vec::<u8>::with_capacity(exact_len);
        out.extend_from_slice(HUGE_WIRE_MAGIC);
        for value in [
            tokenizer.num_terminals,
            state_count as u32,
            prefix_state_count as u32,
            residual_transition_count as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(expanded_transition_count as u64).to_le_bytes());
        for value in [
            segments.len() as u32,
            metadata.finalizer_rows.len() as u32,
            metadata.future_rows.len() as u32,
            metadata.epsilon_states.len() as u32,
            metadata.epsilon_targets.len() as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&[final_width as u8, future_width as u8, 0, 0]);
        debug_assert_eq!(out.len(), HUGE_WIRE_HEADER_LEN);
        for terminal in 0..tokenizer.num_terminals {
            for word in tokenizer.dfa.group_id_to_u8set(terminal).to_words() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for state in prefix_states {
            end += state.transitions.len() as u32;
            out.extend_from_slice(&end.to_le_bytes());
        }
        for state in prefix_states {
            out.extend(state.transitions.iter().map(|(byte, _)| byte));
        }
        for state in prefix_states {
            for &target in state.transitions.values() {
                out.extend_from_slice(&target.to_le_bytes());
            }
        }
        for row in metadata.finalizer_rows.iter() {
            for word in row.words().iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        for row in metadata.future_rows.iter() {
            for word in row.words().iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        write_packed_row_ids_any(&mut out, &metadata.finalizer_row_ids, final_width);
        write_packed_row_ids_any(&mut out, &metadata.future_row_ids, future_width);
        extend_u32_le(&mut out, &metadata.epsilon_states);
        extend_u32_le(&mut out, &metadata.epsilon_offsets);
        extend_u32_le(&mut out, &metadata.epsilon_targets);
        for segment in segments {
            let class_count = u16::try_from(segment.class_members.len()).ok()?;
            let row_count = segment.row_offsets.len().checked_sub(1)?;
            let entry_count = segment.classes.as_slice().len();
            let row_width = packed_row_id_width(row_count);
            for value in [
                segment.state_offset,
                segment.state_count,
                u32::try_from(segment.expanded_transition_count).ok()?,
                u32::try_from(row_count).ok()?,
                u32::try_from(entry_count).ok()?,
                u32::try_from(segment.overflow_indices.len()).ok()?,
            ] {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&class_count.to_le_bytes());
            out.extend_from_slice(&[row_width as u8, 0]);
            out.extend_from_slice(segment.byte_to_class.as_slice());
            write_packed_row_ids_any(&mut out, &segment.row_ids, row_width);
            extend_u32_le(&mut out, &segment.row_offsets);
            out.extend_from_slice(segment.classes.as_slice());
            if let Some(bytes) = segment.deltas.backed_le_bytes() {
                out.extend_from_slice(bytes);
            } else if let Some(values) = segment.deltas.owned_values() {
                extend_i16_le(&mut out, values);
            } else {
                for index in 0..entry_count {
                    out.extend_from_slice(
                        &segment
                            .deltas
                            .get(index)
                            .expect("packed delta row must cover every entry")
                            .to_le_bytes(),
                    );
                }
            }
            extend_u32_le(&mut out, &segment.overflow_indices);
            extend_i32_le(&mut out, &segment.overflow_deltas);
        }
        debug_assert_eq!(out.len(), exact_len);
        Some(out)
    }

    /// Build the compact giant-tokenizer artifact. This is intentionally gated
    /// to the already-compressed suffix representation; normal tokenizers stay
    /// on TKF2 byte-for-byte.
    pub fn build_huge_bytes(tokenizer: &Tokenizer) -> Option<Vec<u8>> {
        if tokenizer.compressed_transition_segments.is_empty() {
            return build_huge_bytes_from_packed_runtime(tokenizer);
        }
        let segments = tokenizer.compressed_transition_segments.as_ref();
        if segments.is_empty() || tokenizer.num_terminals > 128 {
            return None;
        }
        let state_count = tokenizer.dfa.num_states();
        let prefix_state_count = segments.first()?.state_offset as usize;
        let mut expected = prefix_state_count;
        for segment in segments {
            if segment.state_offset as usize != expected {
                return None;
            }
            expected = expected.checked_add(segment.state_count as usize)?;
        }
        if expected != state_count {
            return None;
        }
        let packed_segments = segments
            .iter()
            .map(packed_segment_build)
            .collect::<Option<Vec<_>>>()?;
        let (final_rows, final_ids, future_rows, future_ids) = metadata_rows(tokenizer)?;
        let final_width = packed_row_id_width(final_rows.len());
        let future_width = packed_row_id_width(future_rows.len());

        let prefix_states = &tokenizer.dfa.states()[..prefix_state_count];
        let residual_transition_count = prefix_states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        let expanded_transition_count = residual_transition_count
            + segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
        let mut epsilon_states = Vec::<u32>::new();
        let mut epsilon_offsets = vec![0u32];
        let mut epsilon_targets = Vec::<u32>::new();
        for (state, row) in tokenizer.dfa.states().iter().enumerate() {
            if row.epsilon_transitions.is_empty() {
                continue;
            }
            epsilon_states.push(state as u32);
            epsilon_targets.extend_from_slice(&row.epsilon_transitions);
            epsilon_offsets.push(u32::try_from(epsilon_targets.len()).ok()?);
        }

        let mut out = Vec::<u8>::new();
        out.extend_from_slice(HUGE_WIRE_MAGIC);
        for value in [
            tokenizer.num_terminals,
            state_count as u32,
            prefix_state_count as u32,
            residual_transition_count as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(expanded_transition_count as u64).to_le_bytes());
        for value in [
            segments.len() as u32,
            final_rows.len() as u32,
            future_rows.len() as u32,
            epsilon_states.len() as u32,
            epsilon_targets.len() as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&[final_width as u8, future_width as u8, 0, 0]);
        debug_assert_eq!(out.len(), HUGE_WIRE_HEADER_LEN);
        for terminal in 0..tokenizer.num_terminals {
            for word in tokenizer.dfa.group_id_to_u8set(terminal).to_words() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for state in prefix_states {
            end += state.transitions.len() as u32;
            out.extend_from_slice(&end.to_le_bytes());
        }
        for state in prefix_states {
            out.extend(state.transitions.iter().map(|(byte, _)| byte));
        }
        for state in prefix_states {
            for &target in state.transitions.values() {
                out.extend_from_slice(&target.to_le_bytes());
            }
        }
        let metadata_word_count = (tokenizer.num_terminals as usize).div_ceil(64);
        for row in &final_rows {
            for word in row.iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        for row in &future_rows {
            for word in row.iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        write_packed_row_ids(&mut out, &final_ids, final_width);
        write_packed_row_ids(&mut out, &future_ids, future_width);
        for &state in &epsilon_states {
            out.extend_from_slice(&state.to_le_bytes());
        }
        for &offset in &epsilon_offsets {
            out.extend_from_slice(&offset.to_le_bytes());
        }
        for &target in &epsilon_targets {
            out.extend_from_slice(&target.to_le_bytes());
        }
        for (segment, packed) in segments.iter().zip(&packed_segments) {
            let class_count = u16::try_from(segment.class_members.len()).ok()?;
            let row_count = packed.row_offsets.len().checked_sub(1)?;
            let row_width = packed_row_id_width(row_count);
            for value in [
                segment.state_offset,
                segment.state_count,
                u32::try_from(segment.expanded_transition_count).ok()?,
                row_count as u32,
                packed.classes.len() as u32,
                packed.overflow_indices.len() as u32,
            ] {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&class_count.to_le_bytes());
            out.push(row_width as u8);
            out.push(0);
            out.extend_from_slice(&segment.byte_to_class);
            write_packed_row_ids(&mut out, &packed.row_ids, row_width);
            for &offset in &packed.row_offsets {
                out.extend_from_slice(&offset.to_le_bytes());
            }
            out.extend_from_slice(&packed.classes);
            for &delta in &packed.deltas {
                out.extend_from_slice(&delta.to_le_bytes());
            }
            for &index in &packed.overflow_indices {
                out.extend_from_slice(&index.to_le_bytes());
            }
            for &delta in &packed.overflow_deltas {
                out.extend_from_slice(&delta.to_le_bytes());
            }
        }
        Some(out)
    }

    fn from_huge_bytes(
        input: &[u8],
        backing: Option<(Arc<Vec<u8>>, usize)>,
    ) -> Result<Tokenizer, String> {
        use rayon::prelude::*;
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let huge_started = profile.then(std::time::Instant::now);
        if input.len() < HUGE_WIRE_HEADER_LEN || !input.starts_with(HUGE_WIRE_MAGIC) {
            return Err("invalid giant tokenizer header".to_owned());
        }
        let mut pos = 4usize;
        let take_u32 = |input: &[u8], pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated giant tokenizer".to_owned())?;
            *pos = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let take_u64 = |input: &[u8], pos: &mut usize| -> Result<u64, String> {
            let end = pos.checked_add(8).ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated giant tokenizer".to_owned())?;
            *pos = end;
            Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
        };
        let num_terminals = take_u32(input, &mut pos)?;
        let state_count = take_u32(input, &mut pos)? as usize;
        let prefix_state_count = take_u32(input, &mut pos)? as usize;
        let residual_transition_count = take_u32(input, &mut pos)? as usize;
        let expanded_transition_count = take_u64(input, &mut pos)? as usize;
        let segment_count = take_u32(input, &mut pos)? as usize;
        let final_row_count = take_u32(input, &mut pos)? as usize;
        let future_row_count = take_u32(input, &mut pos)? as usize;
        let epsilon_state_count = take_u32(input, &mut pos)? as usize;
        let epsilon_target_count = take_u32(input, &mut pos)? as usize;
        let final_width = *input.get(pos).ok_or_else(|| "truncated giant tokenizer widths".to_owned())? as usize;
        let future_width = *input.get(pos + 1).ok_or_else(|| "truncated giant tokenizer widths".to_owned())? as usize;
        if input.get(pos + 2..pos + 4) != Some(&[0, 0]) || !matches!(final_width, 1 | 2 | 4) || !matches!(future_width, 1 | 2 | 4) {
            return Err("invalid giant tokenizer row-id widths".to_owned());
        }
        pos += 4;
        if pos != HUGE_WIRE_HEADER_LEN || state_count == 0 || prefix_state_count > state_count || num_terminals > 128 {
            return Err("invalid giant tokenizer dimensions".to_owned());
        }
        let terminal_count = num_terminals as usize;
        let mut groups = Vec::with_capacity(terminal_count);
        for _ in 0..terminal_count {
            let mut words = [0u64; 4];
            for word in &mut words {
                let end = pos + 8;
                let bytes = input.get(pos..end).ok_or_else(|| "truncated giant tokenizer groups".to_owned())?;
                *word = u64::from_le_bytes(bytes.try_into().unwrap());
                pos = end;
            }
            groups.push(U8Set::from_words(words));
        }
        fn read_u32_vec_at(
            input: &[u8],
            pos: &mut usize,
            count: usize,
        ) -> Result<Vec<u32>, String> {
            let end = pos
                .checked_add(
                    count
                        .checked_mul(4)
                        .ok_or_else(|| "giant vector overflow".to_owned())?,
                )
                .ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated giant tokenizer vector".to_owned())?;
            *pos = end;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        }
        let prefix_offsets = read_u32_vec_at(input, &mut pos, prefix_state_count + 1)?;
        if prefix_offsets.first().copied() != Some(0)
            || prefix_offsets.last().copied() != Some(residual_transition_count as u32)
            || prefix_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid giant tokenizer residual offsets".to_owned());
        }
        let bytes_end = pos.checked_add(residual_transition_count).ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
        let prefix_bytes = Arc::<[u8]>::from(input.get(pos..bytes_end).ok_or_else(|| "truncated giant tokenizer residual bytes".to_owned())?);
        pos = bytes_end;
        let prefix_targets = read_u32_vec_at(input, &mut pos, residual_transition_count)?;
        if prefix_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("giant tokenizer residual target out of range".to_owned());
        }
        let mut read_metadata_rows = |count: usize| -> Result<Vec<BitSet>, String> {
            let mut rows = Vec::with_capacity(count);
            for _ in 0..count {
                let mut bits = BitSet::new(terminal_count);
                for word in bits.words_mut() {
                    let end = pos + 8;
                    let bytes = input.get(pos..end).ok_or_else(|| "truncated giant tokenizer metadata".to_owned())?;
                    *word = u64::from_le_bytes(bytes.try_into().unwrap());
                    pos = end;
                }
                rows.push(bits);
            }
            Ok(rows)
        };
        let final_rows = read_metadata_rows(final_row_count)?;
        let future_rows = read_metadata_rows(future_row_count)?;
        let backed = backing.as_ref().map(|(artifact, start)| (artifact, *start));
        let final_row_ids =
            read_packed_row_ids(input, &mut pos, state_count, final_width, backed)?;
        let future_row_ids =
            read_packed_row_ids(input, &mut pos, state_count, future_width, backed)?;
        let meta_validate_started = profile.then(std::time::Instant::now);
        let (final_ids_valid, future_ids_valid) = if state_count >= 100_000
            && rayon::current_num_threads() > 1
        {
            rayon::join(
                || final_row_ids.all_lt(final_rows.len()),
                || future_row_ids.all_lt(future_rows.len()),
            )
        } else {
            (
                final_row_ids.all_lt(final_rows.len()),
                future_row_ids.all_lt(future_rows.len()),
            )
        };
        if !final_ids_valid || !future_ids_valid {
            return Err("giant tokenizer metadata row id out of range".to_owned());
        }
        if let Some(started) = meta_validate_started {
            eprintln!("[glrmask/profile][tks3] meta_ids_validate_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        let epsilon_states = read_u32_vec_at(input, &mut pos, epsilon_state_count)?;
        let epsilon_offsets = read_u32_vec_at(input, &mut pos, epsilon_state_count + 1)?;
        let epsilon_targets = read_u32_vec_at(input, &mut pos, epsilon_target_count)?;
        if epsilon_offsets.first().copied() != Some(0)
            || epsilon_offsets.last().copied() != Some(epsilon_target_count as u32)
            || epsilon_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || epsilon_states.windows(2).any(|pair| pair[0] >= pair[1])
            || epsilon_states.iter().any(|&state| state as usize >= state_count)
            || epsilon_targets.iter().any(|&state| state as usize >= state_count)
        {
            return Err("invalid giant tokenizer epsilon metadata".to_owned());
        }
        let finalizer_lists = final_rows
            .iter()
            .map(|row| row.iter().map(|id| id as TerminalID).collect::<Vec<_>>().into_boxed_slice())
            .collect::<Vec<_>>();
        let metadata = Arc::new(PackedTokenizerMetadata {
            state_count: state_count as u32,
            finalizer_row_ids: final_row_ids,
            finalizer_rows: Arc::from(final_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids,
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::from(epsilon_states.into_boxed_slice()),
            epsilon_offsets: Arc::from(epsilon_offsets.into_boxed_slice()),
            epsilon_targets: Arc::from(epsilon_targets.into_boxed_slice()),
        });
        let mut packed_segments = Vec::with_capacity(segment_count);
        let mut expected_state = prefix_state_count as u32;
        let mut segment_expanded = 0usize;
        for _ in 0..segment_count {
            let segment_started = profile.then(std::time::Instant::now);
            let state_offset = take_u32(input, &mut pos)?;
            let segment_state_count = take_u32(input, &mut pos)?;
            let segment_expanded_count = take_u32(input, &mut pos)? as usize;
            let row_count = take_u32(input, &mut pos)? as usize;
            let entry_count = take_u32(input, &mut pos)? as usize;
            let overflow_count = take_u32(input, &mut pos)? as usize;
            let class_end = pos + 2;
            let class_count = u16::from_le_bytes(input.get(pos..class_end).ok_or_else(|| "truncated giant segment class count".to_owned())?.try_into().unwrap()) as usize;
            pos = class_end;
            let row_width = *input.get(pos).ok_or_else(|| "truncated giant segment row width".to_owned())? as usize;
            let reserved = *input.get(pos + 1).ok_or_else(|| "truncated giant segment row width".to_owned())?;
            pos += 2;
            if state_offset != expected_state || segment_state_count == 0 || class_count == 0 || class_count > 255 || reserved != 0 || !matches!(row_width, 1 | 2 | 4) {
                return Err("invalid giant tokenizer segment header".to_owned());
            }
            expected_state = expected_state.checked_add(segment_state_count).ok_or_else(|| "giant tokenizer state range overflow".to_owned())?;
            let map_end = pos + 256;
            let byte_to_class_slice = input
                .get(pos..map_end)
                .ok_or_else(|| "truncated giant tokenizer byte classes".to_owned())?;
            let byte_to_class = if let Some((artifact, section_start)) = backed {
                PackedRuntimeBytes::Backed {
                    backing: Arc::clone(artifact),
                    start: section_start + pos,
                    len: 256,
                }
            } else {
                PackedRuntimeBytes::Owned(Arc::from(byte_to_class_slice))
            };
            pos = map_end;
            if byte_to_class
                .as_slice()
                .iter()
                .any(|&class| class != u8::MAX && class as usize >= class_count)
            {
                return Err("giant tokenizer byte class out of range".to_owned());
            }
            let row_ids = read_packed_row_ids(
                input,
                &mut pos,
                segment_state_count as usize,
                row_width,
                backed,
            )?;
            let row_offsets = read_u32_vec_at(input, &mut pos, row_count + 1)?;
            if row_offsets.first().copied() != Some(0)
                || row_offsets.last().copied() != Some(entry_count as u32)
                || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
            {
                return Err("invalid giant tokenizer transition dictionary offsets".to_owned());
            }
            let classes_end = pos + entry_count;
            let classes_slice = input
                .get(pos..classes_end)
                .ok_or_else(|| "truncated giant tokenizer transition classes".to_owned())?;
            let classes = if let Some((artifact, section_start)) = backed {
                PackedRuntimeBytes::Backed {
                    backing: Arc::clone(artifact),
                    start: section_start + pos,
                    len: entry_count,
                }
            } else {
                PackedRuntimeBytes::Owned(Arc::from(classes_slice))
            };
            pos = classes_end;
            if classes
                .as_slice()
                .iter()
                .any(|&class| class as usize >= class_count)
            {
                return Err("giant tokenizer transition class out of range".to_owned());
            }
            let delta_end = pos.checked_add(entry_count * 2).ok_or_else(|| "giant tokenizer delta overflow".to_owned())?;
            let delta_bytes = input.get(pos..delta_end).ok_or_else(|| "truncated giant tokenizer deltas".to_owned())?;
            let deltas = if let Some((artifact, section_start)) = backed {
                PackedI16Values::Backed {
                    backing: Arc::clone(artifact),
                    start: section_start + pos,
                    len: entry_count,
                }
            } else {
                PackedI16Values::Owned(Arc::from(
                    delta_bytes
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                ))
            };
            pos = delta_end;
            let overflow_indices = read_u32_vec_at(input, &mut pos, overflow_count)?;
            let overflow_raw = read_u32_vec_at(input, &mut pos, overflow_count)?;
            let overflow_deltas = Arc::<[i32]>::from(overflow_raw.into_iter().map(|v| v as i32).collect::<Vec<_>>().into_boxed_slice());
            let mut class_members = vec![Vec::<u8>::new(); class_count];
            for (byte, &class) in byte_to_class.as_slice().iter().enumerate() {
                if class != u8::MAX {
                    class_members[class as usize].push(byte as u8);
                }
            }
            let class_members = Arc::from(
                class_members
                    .into_iter()
                    .map(Vec::into_boxed_slice)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            let segment = PackedCompressedTransitionSegment {
                state_offset,
                state_count: segment_state_count,
                byte_to_class,
                class_members,
                row_ids,
                row_offsets: Arc::from(row_offsets.into_boxed_slice()),
                classes,
                deltas,
                overflow_indices: Arc::from(overflow_indices.into_boxed_slice()),
                overflow_deltas,
                expanded_transition_count: segment_expanded_count,
            };
            // Validate row ids and target bounds together. The historical
            // decoder first scanned every physical state only to validate the
            // row id, then scanned all of them again for target bounds.
            let mut min_delta = vec![0i32; row_count];
            let mut max_delta = vec![0i32; row_count];
            let delta_validate_started = profile.then(std::time::Instant::now);
            if let Some(delta_bytes) = segment.deltas.backed_le_bytes() {
                // The section length and row-offset coverage have already been
                // validated. Walk the fixed-width slab once, resolving the
                // sparse overflow table monotonically instead of calling
                // `segment.delta()` for every entry (which performs bounds
                // checks and binary-searches overflow sentinels).
                let scan_rows = |row_base: usize,
                                 min_chunk: &mut [i32],
                                 max_chunk: &mut [i32]|
                 -> Result<(), String> {
                    let first_entry = segment.row_offsets[row_base] as usize;
                    let end_entry = segment.row_offsets[row_base + min_chunk.len()] as usize;
                    let mut overflow = segment
                        .overflow_indices
                        .partition_point(|&index| (index as usize) < first_entry);
                    let expected_overflow_end = segment
                        .overflow_indices
                        .partition_point(|&index| (index as usize) < end_entry);
                    for offset in 0..min_chunk.len() {
                        let row = row_base + offset;
                        let start = segment.row_offsets[row] as usize;
                        let end = segment.row_offsets[row + 1] as usize;
                        if start == end {
                            continue;
                        }
                        let mut min = i32::MAX;
                        let mut max = i32::MIN;
                        for index in start..end {
                            let byte = index * 2;
                            let raw = i16::from_le_bytes([delta_bytes[byte], delta_bytes[byte + 1]]);
                            let delta = if raw != i16::MIN {
                                if segment
                                    .overflow_indices
                                    .get(overflow)
                                    .is_some_and(|&overflow_index| overflow_index as usize == index)
                                {
                                    return Err("invalid giant tokenizer overflow table".to_owned());
                                }
                                raw as i32
                            } else {
                                let Some(&overflow_index) = segment.overflow_indices.get(overflow) else {
                                    return Err("missing giant tokenizer overflow record".to_owned());
                                };
                                if overflow_index as usize != index {
                                    return Err("invalid giant tokenizer overflow table".to_owned());
                                }
                                let Some(&delta) = segment.overflow_deltas.get(overflow) else {
                                    return Err("missing giant tokenizer overflow delta".to_owned());
                                };
                                overflow += 1;
                                delta
                            };
                            min = min.min(delta);
                            max = max.max(delta);
                        }
                        min_chunk[offset] = min;
                        max_chunk[offset] = max;
                    }
                    if overflow != expected_overflow_end {
                        return Err("unused giant tokenizer overflow record".to_owned());
                    }
                    Ok(())
                };

                if entry_count >= 100_000 && rayon::current_num_threads() > 1 {
                    let workers = rayon::current_num_threads().min(8).max(1);
                    let chunk_rows = row_count.div_ceil(workers);
                    min_delta
                        .par_chunks_mut(chunk_rows)
                        .zip(max_delta.par_chunks_mut(chunk_rows))
                        .enumerate()
                        .try_for_each(|(chunk, (min_chunk, max_chunk))| {
                            scan_rows(chunk * chunk_rows, min_chunk, max_chunk)
                        })?;
                } else {
                    scan_rows(0, &mut min_delta, &mut max_delta)?;
                }

                if segment.overflow_indices.len() != segment.overflow_deltas.len() {
                    return Err("unused giant tokenizer overflow record".to_owned());
                }
            } else {
                for row in 0..row_count {
                    let start = segment.row_offsets[row] as usize;
                    let end = segment.row_offsets[row + 1] as usize;
                    if start == end {
                        continue;
                    }
                    let first = segment
                        .delta(start)
                        .ok_or_else(|| "invalid giant tokenizer row delta".to_owned())?;
                    let mut min = first;
                    let mut max = first;
                    for index in start + 1..end {
                        let delta = segment
                            .delta(index)
                            .ok_or_else(|| "invalid giant tokenizer row delta".to_owned())?;
                        min = min.min(delta);
                        max = max.max(delta);
                    }
                    min_delta[row] = min;
                    max_delta[row] = max;
                }
            }
            if let Some(started) = delta_validate_started {
                eprintln!("[glrmask/profile][tks3] segment={} delta_extrema_ms={:.3}", state_offset, started.elapsed().as_secs_f64() * 1000.0);
            }
            let row_validate_started = profile.then(std::time::Instant::now);
            let validate_local = |local: usize| -> bool {
                let Some(row) = segment.row_ids.get(local) else {
                    return false;
                };
                if row >= row_count {
                    return false;
                }
                let lo = local as i64 + min_delta[row] as i64;
                let hi = local as i64 + max_delta[row] as i64;
                lo >= 0 && hi < segment_state_count as i64
            };
            let rows_valid = if segment_state_count as usize >= 100_000
                && rayon::current_num_threads() > 1
            {
                (0..segment_state_count as usize)
                    .into_par_iter()
                    .all(validate_local)
            } else {
                (0..segment_state_count as usize).all(validate_local)
            };
            if !rows_valid {
                return Err("giant tokenizer transition row or target out of range".to_owned());
            }
            if let Some(started) = row_validate_started {
                eprintln!("[glrmask/profile][tks3] segment={} row_ids_validate_ms={:.3}", state_offset, started.elapsed().as_secs_f64() * 1000.0);
            }
            segment_expanded += segment_expanded_count;
            packed_segments.push(segment);
            if let Some(started) = segment_started {
                eprintln!("[glrmask/profile][tks3] segment={} total_segment_ms={:.3}", state_offset, started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        if expected_state as usize != state_count
            || residual_transition_count + segment_expanded != expanded_transition_count
            || pos != input.len()
        {
            return Err("invalid giant tokenizer coverage or trailing bytes".to_owned());
        }
        let mut dfa = DFA::new(prefix_state_count);
        dfa.ensure_group_capacity(terminal_count);
        for (terminal, group) in groups.into_iter().enumerate() {
            dfa.set_group_u8set(terminal as u32, group);
        }
        for state in 0..prefix_state_count as u32 {
            dfa.overwrite_state_metadata(
                state,
                metadata.finalizers(state).unwrap().clone(),
                metadata.futures(state).unwrap().clone(),
            );
            for &target in metadata.epsilon_targets(state) {
                dfa.add_epsilon_transition(state, target);
            }
        }
        let prefix_transitions = PackedRuntimeTransitions {
            byte_offsets: Arc::from(prefix_offsets.into_boxed_slice()),
            bytes: PackedRuntimeBytes::Owned(prefix_bytes),
            targets: PackedRuntimeTargets::U32(Arc::from(prefix_targets.into_boxed_slice())),
        };
        let transition_count_cache = OnceLock::new();
        let _ = transition_count_cache.set(expanded_transition_count);
        let tokenizer = Tokenizer {
            dfa,
            num_terminals,
            packed_runtime_transitions: Some(Arc::new(prefix_transitions)),
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: Some(metadata),
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from(packed_segments.into_boxed_slice()),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache,
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        };
        if let Some(started) = huge_started {
            eprintln!("[glrmask/profile][tks3] total_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(tokenizer)
    }

    pub fn from_fast_bytes(input: &[u8]) -> Result<Tokenizer, String> {
        from_fast_bytes_impl(input, None)
    }

    /// Decode a current tokenizer section while retaining transition payloads
    /// directly in the owned constraint artifact. The caller must supply the
    /// exact byte offset of `input` inside `backing`; this is validated before
    /// any backed view is installed.
    pub fn from_fast_bytes_backed(
        input: &[u8],
        backing: Arc<Vec<u8>>,
        section_start: usize,
    ) -> Result<Tokenizer, String> {
        let section_end = section_start
            .checked_add(input.len())
            .ok_or_else(|| "fast tokenizer backing range overflow".to_owned())?;
        let backed = backing
            .get(section_start..section_end)
            .ok_or_else(|| "fast tokenizer section is outside artifact backing".to_owned())?;
        if backed.as_ptr() != input.as_ptr() || backed.len() != input.len() {
            return Err("fast tokenizer section does not match artifact backing".to_owned());
        }
        from_fast_bytes_impl(input, Some((backing, section_start)))
    }

    fn from_fast_bytes_impl(
        input: &[u8],
        backing: Option<(Arc<Vec<u8>>, usize)>,
    ) -> Result<Tokenizer, String> {
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        if input.starts_with(HUGE_WIRE_MAGIC) {
            return from_huge_bytes(input, backing);
        }
        if input.starts_with(PACKED_WIRE_MAGIC) {
            return from_packed_bytes(input);
        }
        if input.starts_with(SEGMENT_WIRE_MAGIC) {
            return from_segment_bytes(input);
        }
        let tkf2 = input.starts_with(b"TKF2");
        if input.len() < 28 || (!tkf2 && !input.starts_with(b"TKF1")) {
            return Err("invalid fast tokenizer header".to_owned());
        }
        let mut pos = 4usize;
        let take_u32 = |input: &[u8], pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated fast tokenizer".to_owned())?;
            *pos = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let num_terminals = take_u32(input, &mut pos)?;
        let state_count = take_u32(input, &mut pos)? as usize;
        let transition_count = take_u32(input, &mut pos)? as usize;
        let epsilon_count = take_u32(input, &mut pos)? as usize;
        let finalizer_count = take_u32(input, &mut pos)? as usize;
        let future_count = take_u32(input, &mut pos)? as usize;
        if state_count == 0 {
            return Err("fast tokenizer has no states".to_owned());
        }
        let (state_id_width, terminal_id_width) = if tkf2 {
            let state_width = *input
                .get(pos)
                .ok_or_else(|| "truncated fast tokenizer state-id width".to_owned())?
                as usize;
            let terminal_width = *input
                .get(pos + 1)
                .ok_or_else(|| "truncated fast tokenizer terminal-id width".to_owned())?
                as usize;
            let reserved = input
                .get(pos + 2..pos + 4)
                .ok_or_else(|| "truncated fast tokenizer width header".to_owned())?;
            pos += 4;
            if !matches!(state_width, 2 | 4)
                || !matches!(terminal_width, 2 | 4)
                || reserved != [0, 0]
            {
                return Err("invalid fast tokenizer id widths".to_owned());
            }
            if state_width == 2 && state_count > u16::MAX as usize + 1 {
                return Err("fast tokenizer u16 state ids cannot address all states".to_owned());
            }
            if terminal_width == 2 && num_terminals as usize > u16::MAX as usize + 1 {
                return Err("fast tokenizer u16 terminal ids cannot address all terminals".to_owned());
            }
            (state_width, terminal_width)
        } else {
            (4usize, 4usize)
        };
        let dfa_alloc_started = profile.then(std::time::Instant::now);
        let mut group_id_to_u8set = Vec::with_capacity(num_terminals as usize);
        for _ in 0..num_terminals as usize {
            let mut words = [0u64; 4];
            for word in &mut words {
                let end = pos.checked_add(8).ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
                let bytes = input.get(pos..end).ok_or_else(|| "truncated fast tokenizer groups".to_owned())?;
                *word = u64::from_le_bytes(bytes.try_into().unwrap());
                pos = end;
            }
            group_id_to_u8set.push(U8Set::from_words(words));
        }
        let dfa_alloc_groups_ms = dfa_alloc_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let read_u32_vec = |input: &[u8], pos: &mut usize, count: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count.checked_mul(4).ok_or_else(|| "fast tokenizer vector overflow".to_owned())?;
            let end = pos.checked_add(bytes_len).ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated fast tokenizer vector".to_owned())?;
            let mut out = Vec::<u32>::with_capacity(count);
            if cfg!(target_endian = "little") {
                unsafe {
                    out.set_len(count);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes_len);
                }
            } else {
                out.extend(bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
            }
            *pos = end;
            Ok(out)
        };
        let read_u16_as_u32_vec =
            |input: &[u8], pos: &mut usize, count: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count
                .checked_mul(2)
                .ok_or_else(|| "fast tokenizer u16 vector overflow".to_owned())?;
            let end = pos
                .checked_add(bytes_len)
                .ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast tokenizer u16 vector".to_owned())?;
            let mut out = Vec::<u32>::with_capacity(count);
            out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]]))),
            );
            *pos = end;
            Ok(out)
        };
        let read_ids_as_u32 =
            |input: &[u8], pos: &mut usize, count: usize, width: usize| -> Result<Vec<u32>, String> {
                match width {
                    2 => read_u16_as_u32_vec(input, pos, count),
                    4 => read_u32_vec(input, pos, count),
                    _ => Err("invalid fast tokenizer id width".to_owned()),
                }
            };
        let transitions_started = profile.then(std::time::Instant::now);
        let transition_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        if transition_offsets.first().copied() != Some(0)
            || transition_offsets.last().copied() != Some(transition_count as u32)
            || transition_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err("invalid fast tokenizer transition offsets".to_owned());
        }
        let transition_bytes_start = pos;
        let transition_bytes_end = pos
            .checked_add(transition_count)
            .ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
        let transition_bytes_slice = input
            .get(pos..transition_bytes_end)
            .ok_or_else(|| "truncated fast tokenizer transition bytes".to_owned())?;
        let transition_bytes = if let Some((artifact, section_start)) = &backing {
            PackedRuntimeBytes::Backed {
                backing: Arc::clone(artifact),
                start: section_start + transition_bytes_start,
                len: transition_count,
            }
        } else {
            PackedRuntimeBytes::Owned(Arc::from(transition_bytes_slice))
        };
        pos = transition_bytes_end;
        let transition_targets = match state_id_width {
            2 => {
                let bytes_len = transition_count
                    .checked_mul(2)
                    .ok_or_else(|| "fast tokenizer target vector overflow".to_owned())?;
                let start = pos;
                let end = pos
                    .checked_add(bytes_len)
                    .ok_or_else(|| "fast tokenizer target offset overflow".to_owned())?;
                let bytes = input
                    .get(start..end)
                    .ok_or_else(|| "truncated fast tokenizer targets".to_owned())?;
                if !u16_values_all_below(bytes, state_count) {
                    return Err("fast tokenizer transition target out of range".to_owned());
                }
                pos = end;
                if let Some((artifact, section_start)) = &backing {
                    PackedRuntimeTargets::BackedU16 {
                        backing: Arc::clone(artifact),
                        start: section_start + start,
                        len: transition_count,
                    }
                } else {
                    let mut targets = Vec::with_capacity(transition_count);
                    targets.extend(
                        bytes
                            .chunks_exact(2)
                            .map(|word| u16::from_le_bytes([word[0], word[1]])),
                    );
                    PackedRuntimeTargets::U16(Arc::from(targets.into_boxed_slice()))
                }
            }
            4 => {
                let bytes_len = transition_count
                    .checked_mul(4)
                    .ok_or_else(|| "fast tokenizer target vector overflow".to_owned())?;
                let start = pos;
                let end = pos
                    .checked_add(bytes_len)
                    .ok_or_else(|| "fast tokenizer target offset overflow".to_owned())?;
                let bytes = input
                    .get(start..end)
                    .ok_or_else(|| "truncated fast tokenizer targets".to_owned())?;
                if bytes.chunks_exact(4).any(|word| {
                    u32::from_le_bytes([word[0], word[1], word[2], word[3]]) as usize
                        >= state_count
                }) {
                    return Err("fast tokenizer transition target out of range".to_owned());
                }
                pos = end;
                if let Some((artifact, section_start)) = &backing {
                    PackedRuntimeTargets::BackedU32 {
                        backing: Arc::clone(artifact),
                        start: section_start + start,
                        len: transition_count,
                    }
                } else {
                    let mut targets = Vec::with_capacity(transition_count);
                    targets.extend(bytes.chunks_exact(4).map(|word| {
                        u32::from_le_bytes([word[0], word[1], word[2], word[3]])
                    }));
                    PackedRuntimeTargets::U32(Arc::from(targets.into_boxed_slice()))
                }
            }
            _ => return Err("invalid fast tokenizer state-id width".to_owned()),
        };
        let transitions_ms = transitions_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let metadata_wire_started = profile.then(std::time::Instant::now);
        let mut read_state_values =
            |count: usize, width: usize| -> Result<(Vec<u32>, Vec<u32>), String> {
            let offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
            if offsets.first().copied() != Some(0)
                || offsets.last().copied() != Some(count as u32)
                || offsets.windows(2).any(|w| w[0] > w[1])
            {
                return Err("invalid fast tokenizer metadata offsets".to_owned());
            }
            let values = read_ids_as_u32(input, &mut pos, count, width)?;
            Ok((offsets, values))
        };
        let (epsilon_offsets, epsilon_targets) = read_state_values(epsilon_count, state_id_width)?;
        let (finalizer_offsets, finalizers) =
            read_state_values(finalizer_count, terminal_id_width)?;
        let (future_offsets, futures) = read_state_values(future_count, terminal_id_width)?;
        let metadata_wire_ms = metadata_wire_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if pos != input.len() {
            return Err("trailing bytes in fast tokenizer".to_owned());
        }
        if epsilon_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("fast tokenizer epsilon target out of range".to_owned());
        }
        if finalizers
            .iter()
            .chain(&futures)
            .any(|&terminal| terminal >= num_terminals)
        {
            return Err("fast tokenizer terminal id out of range".to_owned());
        }
        let metadata_build_started = profile.then(std::time::Instant::now);
        let dfa = DFA::new_from_sparse_metadata(
            group_id_to_u8set,
            &epsilon_offsets,
            &epsilon_targets,
            &finalizer_offsets,
            &finalizers,
            &future_offsets,
            &futures,
        );
        let metadata_build_ms = metadata_build_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        // Static-constraint finalization always primes this cache before the
        // constraint is returned. Build it here while tokenizer decode is one
        // branch of the outer parallel section fan-out, rather than paying the
        // same work serially after every section has joined.
        let singleton_epsilon_closures = {
            let cache = OnceLock::new();
            let _ = cache.set(Arc::new(dfa.all_singleton_epsilon_closures()));
            cache
        };
        if let Some(total_started) = total_started {
            let epsilon_rows = epsilon_offsets
                .windows(2)
                .filter(|row| row[0] != row[1])
                .count();
            let finalizer_rows = finalizer_offsets
                .windows(2)
                .filter(|row| row[0] != row[1])
                .count();
            let future_rows = future_offsets
                .windows(2)
                .filter(|row| row[0] != row[1])
                .count();
            eprintln!(
                "[glrmask/profile][tokenizer_fast_decode] states={} transitions={} epsilon={} epsilon_rows={} finalizers={} finalizer_rows={} futures={} future_rows={} dfa_groups_ms={:.3} transitions_ms={:.3} metadata_wire_ms={:.3} metadata_build_ms={:.3} total_ms={:.3}",
                state_count,
                transition_count,
                epsilon_count,
                epsilon_rows,
                finalizer_count,
                finalizer_rows,
                future_count,
                future_rows,
                dfa_alloc_groups_ms,
                transitions_ms,
                metadata_wire_ms,
                metadata_build_ms,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(Tokenizer {
            dfa,
            num_terminals,
            packed_runtime_transitions: Some(Arc::new(PackedRuntimeTransitions {
                byte_offsets: Arc::from(transition_offsets.into_boxed_slice()),
                bytes: transition_bytes,
                targets: transition_targets,
            })),
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures,
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        })
    }
}


/// Compact tokenizer wire form for dynamic artifacts.
///
/// The ordinary historical `Tokenizer` serializer stores one dense terminal
/// bitset for every lexer state.  That is appropriate for backwards-compatible
/// static artifacts, but source-specialized grammars can have hundreds of
/// thousands of states and thousands of terminals while setting only a handful
/// of metadata bits per state.  This wire form stores exactly the set bits and
/// the actual graph edges.
pub mod compact_artifact_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    struct CompactTokenizerArtifact {
        num_terminals: u32,
        state_count: u32,
        group_id_to_u8set: Vec<U8Set>,
        transition_offsets: Vec<u32>,
        transition_bytes: Vec<u8>,
        transition_targets: Vec<u32>,
        epsilon_offsets: Vec<u32>,
        epsilon_targets: Vec<u32>,
        finalizer_offsets: Vec<u32>,
        finalizers: Vec<u32>,
        future_offsets: Vec<u32>,
        futures: Vec<u32>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    fn offset(value: usize, label: &str) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| format!("{label} count exceeds u32"))
    }

    fn validate_offsets(
        offsets: &[u32],
        state_count: usize,
        entry_count: usize,
        label: &str,
    ) -> Result<(), String> {
        if offsets.len() != state_count.saturating_add(1) {
            return Err(format!(
                "{label} offsets have length {}, expected {}",
                offsets.len(),
                state_count.saturating_add(1),
            ));
        }
        if offsets.first().copied() != Some(0) {
            return Err(format!("{label} offsets do not start at zero"));
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(format!("{label} offsets are not monotonic"));
        }
        if offsets.last().copied().map(|value| value as usize) != Some(entry_count) {
            return Err(format!(
                "{label} offsets end at {:?}, expected {entry_count}",
                offsets.last(),
            ));
        }
        Ok(())
    }

    pub fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error as _;

        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);

        let state_count = tokenizer.dfa.num_states();
        let mut transition_offsets = Vec::with_capacity(state_count + 1);
        let mut transition_bytes = Vec::new();
        let mut transition_targets = Vec::new();
        let mut epsilon_offsets = Vec::with_capacity(state_count + 1);
        let mut epsilon_targets = Vec::new();
        let mut finalizer_offsets = Vec::with_capacity(state_count + 1);
        let mut finalizers = Vec::new();
        let mut future_offsets = Vec::with_capacity(state_count + 1);
        let mut futures = Vec::new();
        transition_offsets.push(0);
        epsilon_offsets.push(0);
        finalizer_offsets.push(0);
        future_offsets.push(0);

        for state in tokenizer.dfa.states() {
            for (byte, &target) in state.transitions.iter() {
                transition_bytes.push(byte);
                transition_targets.push(target);
            }
            transition_offsets.push(
                offset(transition_bytes.len(), "tokenizer transition")
                    .map_err(S::Error::custom)?,
            );

            epsilon_targets.extend_from_slice(&state.epsilon_transitions);
            epsilon_offsets.push(
                offset(epsilon_targets.len(), "tokenizer epsilon transition")
                    .map_err(S::Error::custom)?,
            );

            finalizers.extend(state.finalizers.iter().map(|terminal| terminal as u32));
            finalizer_offsets.push(
                offset(finalizers.len(), "tokenizer finalizer")
                    .map_err(S::Error::custom)?,
            );

            futures.extend(
                state
                    .possible_future_group_ids
                    .iter()
                    .map(|terminal| terminal as u32),
            );
            future_offsets.push(
                offset(futures.len(), "tokenizer future")
                    .map_err(S::Error::custom)?,
            );
        }

        let group_id_to_u8set = (0..tokenizer.dfa.num_groups())
            .map(|group| *tokenizer.dfa.group_id_to_u8set(group as u32))
            .collect();
        CompactTokenizerArtifact {
            num_terminals: tokenizer.num_terminals,
            state_count: u32::try_from(state_count).map_err(S::Error::custom)?,
            group_id_to_u8set,
            transition_offsets,
            transition_bytes,
            transition_targets,
            epsilon_offsets,
            epsilon_targets,
            finalizer_offsets,
            finalizers,
            future_offsets,
            futures,
            compressed_transition_segments: tokenizer.compressed_transition_segments.to_vec(),
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        let artifact = CompactTokenizerArtifact::deserialize(deserializer)?;
        let state_count = artifact.state_count as usize;
        let terminal_count = artifact.num_terminals as usize;
        if artifact.group_id_to_u8set.len() != terminal_count {
            return Err(D::Error::custom(format!(
                "compact tokenizer has {} group byte sets for {terminal_count} terminals",
                artifact.group_id_to_u8set.len(),
            )));
        }
        if artifact.transition_bytes.len() != artifact.transition_targets.len() {
            return Err(D::Error::custom(
                "compact tokenizer transition byte/target lengths differ",
            ));
        }
        validate_offsets(
            &artifact.transition_offsets,
            state_count,
            artifact.transition_targets.len(),
            "transition",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.epsilon_offsets,
            state_count,
            artifact.epsilon_targets.len(),
            "epsilon",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.finalizer_offsets,
            state_count,
            artifact.finalizers.len(),
            "finalizer",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.future_offsets,
            state_count,
            artifact.futures.len(),
            "future",
        )
        .map_err(D::Error::custom)?;

        for &target in artifact
            .transition_targets
            .iter()
            .chain(&artifact.epsilon_targets)
        {
            if target as usize >= state_count {
                return Err(D::Error::custom(format!(
                    "compact tokenizer transition target {target} is out of range for {state_count} states",
                )));
            }
        }
        for &terminal in artifact.finalizers.iter().chain(&artifact.futures) {
            if terminal as usize >= terminal_count {
                return Err(D::Error::custom(format!(
                    "compact tokenizer terminal {terminal} is out of range for {terminal_count} terminals",
                )));
            }
        }
        for segment in &artifact.compressed_transition_segments {
            let end = segment
                .state_offset
                .checked_add(segment.state_count)
                .ok_or_else(|| D::Error::custom("compressed tokenizer segment state range overflow"))?;
            if end as usize > state_count {
                return Err(D::Error::custom(format!(
                    "compressed tokenizer segment ends at {end}, beyond {state_count} states",
                )));
            }
        }

        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(terminal_count);
        for (group, set) in artifact.group_id_to_u8set.into_iter().enumerate() {
            dfa.set_group_u8set(group as u32, set);
        }
        for state in 0..state_count {
            let start = artifact.transition_offsets[state] as usize;
            let end = artifact.transition_offsets[state + 1] as usize;
            let entries = artifact.transition_bytes[start..end]
                .iter()
                .copied()
                .zip(artifact.transition_targets[start..end].iter().copied())
                .collect();
            dfa.set_transitions_from_sorted_entries(state as u32, entries);
        }
        {
            let states = dfa.states_mut();
            for (state_index, state) in states.iter_mut().enumerate() {
                let start = artifact.epsilon_offsets[state_index] as usize;
                let end = artifact.epsilon_offsets[state_index + 1] as usize;
                state.epsilon_transitions = artifact.epsilon_targets[start..end].to_vec();

                let start = artifact.finalizer_offsets[state_index] as usize;
                let end = artifact.finalizer_offsets[state_index + 1] as usize;
                for &terminal in &artifact.finalizers[start..end] {
                    state.finalizers.set(terminal as usize);
                }

                let start = artifact.future_offsets[state_index] as usize;
                let end = artifact.future_offsets[state_index + 1] as usize;
                for &terminal in &artifact.futures[start..end] {
                    state.possible_future_group_ids.set(terminal as usize);
                }
            }
        }

        Ok(Tokenizer {
            dfa,
            num_terminals: artifact.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        })
    }
}

/// Packed tokenizer wire form used only by current static-constraint
/// artifacts. Runtime and compiler representations remain unchanged.
///
/// The byte labels of lexer transition rows are highly repetitive (large
/// schema tokenizers commonly have only ~100 distinct byte patterns across
/// thousands of states), while target ids are small. Store each byte pattern
/// once and encode the matching target sequence as varints. The old
/// `compact_artifact_serde` remains untouched because dynamic-constraint
/// artifacts use it as a persisted format.
mod packed_artifact_serde {
    use super::*;
    use rustc_hash::FxHashMap;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    struct PackedTokenizerArtifact {
        num_terminals: u32,
        state_count: u32,
        group_id_to_u8set: Vec<U8Set>,
        transition_byte_rows: Vec<Vec<u8>>,
        transition_byte_row_ids: Vec<u32>,
        transition_target_offsets: Vec<u32>,
        transition_targets: Vec<u8>,
        epsilon_offsets: Vec<u32>,
        epsilon_targets: Vec<u32>,
        finalizer_offsets: Vec<u32>,
        finalizers: Vec<u32>,
        future_offsets: Vec<u32>,
        futures: Vec<u32>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    #[inline]
    fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[inline]
    fn put_var_i64(out: &mut Vec<u8>, value: i64) {
        let zigzag = ((value << 1) ^ (value >> 63)) as u64;
        let mut value = zigzag;
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[inline]
    fn var_u32_len(mut value: u32) -> usize {
        let mut len = 1usize;
        while value >= 0x80 {
            value >>= 7;
            len += 1;
        }
        len
    }

    #[inline]
    fn var_i64_len(value: i64) -> usize {
        let mut value = ((value << 1) ^ (value >> 63)) as u64;
        let mut len = 1usize;
        while value >= 0x80 {
            value >>= 7;
            len += 1;
        }
        len
    }

    #[inline]
    fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
        let mut value = 0u32;
        let mut shift = 0u32;
        for _ in 0..5 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed tokenizer target".to_owned())?;
            *pos += 1;
            if shift == 28 && byte > 0x0f {
                return Err("overflowing packed tokenizer target".to_owned());
            }
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err("overflowing packed tokenizer target".to_owned())
    }

    #[inline]
    fn take_var_i64(input: &[u8], pos: &mut usize) -> Result<i64, String> {
        let mut value = 0u64;
        let mut shift = 0u32;
        for index in 0..10 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed tokenizer target delta".to_owned())?;
            *pos += 1;
            if index == 9 && byte > 1 {
                return Err("overflowing packed tokenizer target delta".to_owned());
            }
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(((value >> 1) as i64) ^ (-((value & 1) as i64)));
            }
            shift += 7;
        }
        Err("overflowing packed tokenizer target delta".to_owned())
    }

    fn encode_targets(targets: &[u32], out: &mut Vec<u8>) {
        let absolute_len = targets.iter().map(|&target| var_u32_len(target)).sum::<usize>();
        let mut previous = 0i64;
        let delta_len = targets
            .iter()
            .map(|&target| {
                let target = target as i64;
                let len = var_i64_len(target - previous);
                previous = target;
                len
            })
            .sum::<usize>();
        let use_delta = delta_len < absolute_len;
        out.push(u8::from(use_delta));
        if use_delta {
            let mut previous = 0i64;
            for &target in targets {
                let target = target as i64;
                put_var_i64(out, target - previous);
                previous = target;
            }
        } else {
            for &target in targets {
                put_var_u32(out, target);
            }
        }
    }

    fn decode_targets_into(
        body: &[u8],
        count: usize,
        state_count: usize,
        targets: &mut Vec<u32>,
    ) -> Result<(), String> {
        let (&mode, rest) = body
            .split_first()
            .ok_or_else(|| "missing packed tokenizer target mode".to_owned())?;
        let start_len = targets.len();
        let mut pos = 0usize;
        match mode {
            0 => {
                for _ in 0..count {
                    targets.push(take_var_u32(rest, &mut pos)?);
                }
            }
            1 => {
                let mut previous = 0i64;
                for _ in 0..count {
                    let delta = take_var_i64(rest, &mut pos)?;
                    let target = previous
                        .checked_add(delta)
                        .ok_or_else(|| "overflowing packed tokenizer target".to_owned())?;
                    let target = u32::try_from(target)
                        .map_err(|_| "invalid packed tokenizer target".to_owned())?;
                    targets.push(target);
                    previous = target as i64;
                }
            }
            _ => return Err("invalid packed tokenizer target mode".to_owned()),
        }
        if pos != rest.len() {
            return Err("trailing bytes in packed tokenizer target row".to_owned());
        }
        if targets[start_len..]
            .iter()
            .any(|&target| target as usize >= state_count)
        {
            return Err("packed tokenizer transition target is out of range".to_owned());
        }
        Ok(())
    }

    fn offset(value: usize, label: &str) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| format!("{label} count exceeds u32"))
    }

    fn validate_offsets(
        offsets: &[u32],
        state_count: usize,
        entry_count: usize,
        label: &str,
    ) -> Result<(), String> {
        if offsets.len() != state_count.saturating_add(1) {
            return Err(format!(
                "{label} offsets have length {}, expected {}",
                offsets.len(),
                state_count.saturating_add(1),
            ));
        }
        if offsets.first().copied() != Some(0) {
            return Err(format!("{label} offsets do not start at zero"));
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(format!("{label} offsets are not monotonic"));
        }
        if offsets.last().copied().map(|value| value as usize) != Some(entry_count) {
            return Err(format!(
                "{label} offsets end at {:?}, expected {entry_count}",
                offsets.last(),
            ));
        }
        Ok(())
    }

    pub fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error as _;

        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);

        // Current static artifacts should already have materialized rows, but
        // preserve exact behavior for the uncommon compressed-segment case.
        let materialized;
        let dfa = if tokenizer.compressed_transition_segments.is_empty() {
            &tokenizer.dfa
        } else {
            materialized = tokenizer.materialized_dfa();
            &materialized
        };
        let state_count = dfa.num_states();
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_RAW").is_some() {
            let started = std::time::Instant::now();
            let transition_count = dfa
                .states()
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>();
            let mut raw = Vec::with_capacity(
                12usize
                    .saturating_add((state_count + 1).saturating_mul(4))
                    .saturating_add(transition_count.saturating_mul(5)),
            );
            raw.extend_from_slice(b"TRW1");
            raw.extend_from_slice(&(state_count as u32).to_le_bytes());
            raw.extend_from_slice(&(transition_count as u32).to_le_bytes());
            let mut end = 0u32;
            raw.extend_from_slice(&end.to_le_bytes());
            for state in dfa.states() {
                end = end.saturating_add(state.transitions.len() as u32);
                raw.extend_from_slice(&end.to_le_bytes());
            }
            for state in dfa.states() {
                for (byte, &target) in state.transitions.iter() {
                    raw.push(byte);
                    raw.extend_from_slice(&target.to_le_bytes());
                }
            }
            eprintln!(
                "[glrmask/profile][tokenizer_raw_candidate] states={} transitions={} bytes={} ms={:.3}",
                state_count,
                transition_count,
                raw.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let mut byte_rows = Vec::<Vec<u8>>::new();
        let mut byte_row_ids = Vec::with_capacity(state_count);
        let mut byte_row_by_value = FxHashMap::<Vec<u8>, u32>::default();
        let mut target_offsets = Vec::with_capacity(state_count + 1);
        let mut target_bytes = Vec::<u8>::new();
        let mut epsilon_offsets = Vec::with_capacity(state_count + 1);
        let mut epsilon_targets = Vec::new();
        let mut finalizer_offsets = Vec::with_capacity(state_count + 1);
        let mut finalizers = Vec::new();
        let mut future_offsets = Vec::with_capacity(state_count + 1);
        let mut futures = Vec::new();
        target_offsets.push(0);
        epsilon_offsets.push(0);
        finalizer_offsets.push(0);
        future_offsets.push(0);

        for state in dfa.states() {
            let bytes = state.transitions.iter().map(|(byte, _)| byte).collect::<Vec<_>>();
            let byte_row = if let Some(&row) = byte_row_by_value.get(&bytes) {
                row
            } else {
                let row = u32::try_from(byte_rows.len()).map_err(S::Error::custom)?;
                byte_row_by_value.insert(bytes.clone(), row);
                byte_rows.push(bytes);
                row
            };
            byte_row_ids.push(byte_row);
            let targets = state.transitions.values().copied().collect::<Vec<_>>();
            encode_targets(&targets, &mut target_bytes);
            target_offsets.push(
                offset(target_bytes.len(), "tokenizer target byte").map_err(S::Error::custom)?,
            );

            epsilon_targets.extend_from_slice(&state.epsilon_transitions);
            epsilon_offsets.push(
                offset(epsilon_targets.len(), "tokenizer epsilon transition")
                    .map_err(S::Error::custom)?,
            );
            finalizers.extend(state.finalizers.iter().map(|terminal| terminal as u32));
            finalizer_offsets.push(
                offset(finalizers.len(), "tokenizer finalizer").map_err(S::Error::custom)?,
            );
            futures.extend(
                state
                    .possible_future_group_ids
                    .iter()
                    .map(|terminal| terminal as u32),
            );
            future_offsets.push(
                offset(futures.len(), "tokenizer future").map_err(S::Error::custom)?,
            );
        }

        let artifact = PackedTokenizerArtifact {
            num_terminals: tokenizer.num_terminals,
            state_count: u32::try_from(state_count).map_err(S::Error::custom)?,
            group_id_to_u8set: (0..dfa.num_groups())
                .map(|group| *dfa.group_id_to_u8set(group as u32))
                .collect(),
            transition_byte_rows: byte_rows,
            transition_byte_row_ids: byte_row_ids,
            transition_target_offsets: target_offsets,
            transition_targets: target_bytes,
            epsilon_offsets,
            epsilon_targets,
            finalizer_offsets,
            finalizers,
            future_offsets,
            futures,
            compressed_transition_segments: Vec::new(),
        };
        let result = artifact.serialize(serializer);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][tokenizer_encode] states={} transition_rows={} target_bytes={} ms={:.3}",
                state_count,
                artifact.transition_byte_rows.len(),
                artifact.transition_targets.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let artifact_started = profile.then(std::time::Instant::now);
        let artifact = PackedTokenizerArtifact::deserialize(deserializer)?;
        let artifact_ms = artifact_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        let state_count = artifact.state_count as usize;
        let terminal_count = artifact.num_terminals as usize;
        if artifact.group_id_to_u8set.len() != terminal_count {
            return Err(D::Error::custom(format!(
                "packed tokenizer has {} group byte sets for {terminal_count} terminals",
                artifact.group_id_to_u8set.len(),
            )));
        }
        if artifact.transition_byte_row_ids.len() != state_count {
            return Err(D::Error::custom("packed tokenizer byte-row id count differs from state count"));
        }
        validate_offsets(
            &artifact.transition_target_offsets,
            state_count,
            artifact.transition_targets.len(),
            "transition target",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.epsilon_offsets,
            state_count,
            artifact.epsilon_targets.len(),
            "epsilon",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.finalizer_offsets,
            state_count,
            artifact.finalizers.len(),
            "finalizer",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.future_offsets,
            state_count,
            artifact.futures.len(),
            "future",
        )
        .map_err(D::Error::custom)?;
        if artifact
            .epsilon_targets
            .iter()
            .any(|&target| target as usize >= state_count)
        {
            return Err(D::Error::custom("packed tokenizer epsilon target is out of range"));
        }
        if artifact
            .finalizers
            .iter()
            .chain(&artifact.futures)
            .any(|&terminal| terminal as usize >= terminal_count)
        {
            return Err(D::Error::custom("packed tokenizer terminal is out of range"));
        }
        if artifact
            .transition_byte_rows
            .iter()
            .any(|bytes| bytes.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(D::Error::custom(
                "packed tokenizer byte row is not sorted unique",
            ));
        }

        let transitions_started = profile.then(std::time::Instant::now);
        let mut target_count = 0usize;
        for &byte_row in &artifact.transition_byte_row_ids {
            target_count = target_count
                .checked_add(
                    artifact
                        .transition_byte_rows
                        .get(byte_row as usize)
                        .ok_or_else(|| D::Error::custom("packed tokenizer byte-row id is out of range"))?
                        .len(),
                )
                .ok_or_else(|| D::Error::custom("packed tokenizer target count overflow"))?;
        }
        let mut runtime_byte_offsets = Vec::with_capacity(state_count + 1);
        let mut runtime_bytes = Vec::with_capacity(target_count);
        let mut runtime_target_offsets = Vec::with_capacity(state_count + 1);
        let mut runtime_targets = Vec::with_capacity(target_count);
        runtime_byte_offsets.push(0u32);
        runtime_target_offsets.push(0u32);
        for state in 0..state_count {
            let byte_row = artifact.transition_byte_row_ids[state] as usize;
            let bytes = &artifact.transition_byte_rows[byte_row];
            let count = bytes.len();
            runtime_bytes.extend_from_slice(bytes);
            runtime_byte_offsets.push(
                u32::try_from(runtime_bytes.len())
                    .map_err(|_| D::Error::custom("packed tokenizer runtime byte count exceeds u32"))?,
            );
            let start = artifact.transition_target_offsets[state] as usize;
            let end = artifact.transition_target_offsets[state + 1] as usize;
            decode_targets_into(
                &artifact.transition_targets[start..end],
                count,
                state_count,
                &mut runtime_targets,
            )
            .map_err(D::Error::custom)?;
            runtime_target_offsets.push(
                u32::try_from(runtime_targets.len())
                    .map_err(|_| D::Error::custom("packed tokenizer runtime target count exceeds u32"))?,
            );
        }
        let transitions_ms = transitions_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

        let dfa_started = profile.then(std::time::Instant::now);
        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(terminal_count);
        for (group, set) in artifact.group_id_to_u8set.into_iter().enumerate() {
            dfa.set_group_u8set(group as u32, set);
        }
        {
            let states = dfa.states_mut();
            for (state_index, state) in states.iter_mut().enumerate() {
                let start = artifact.epsilon_offsets[state_index] as usize;
                let end = artifact.epsilon_offsets[state_index + 1] as usize;
                state.epsilon_transitions = artifact.epsilon_targets[start..end].to_vec();
                let start = artifact.finalizer_offsets[state_index] as usize;
                let end = artifact.finalizer_offsets[state_index + 1] as usize;
                for &terminal in &artifact.finalizers[start..end] {
                    state.finalizers.set(terminal as usize);
                }
                let start = artifact.future_offsets[state_index] as usize;
                let end = artifact.future_offsets[state_index + 1] as usize;
                for &terminal in &artifact.futures[start..end] {
                    state.possible_future_group_ids.set(terminal as usize);
                }
            }
        }
        let packed_runtime_transitions = Arc::new(PackedRuntimeTransitions {
            byte_offsets: Arc::from(runtime_byte_offsets.into_boxed_slice()),
            bytes: PackedRuntimeBytes::Owned(Arc::from(runtime_bytes.into_boxed_slice())),
            targets: PackedRuntimeTargets::U32(Arc::from(runtime_targets.into_boxed_slice())),
        });
        let dfa_ms = dfa_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][tokenizer_decode] states={} artifact_ms={artifact_ms:.3} transitions_ms={transitions_ms:.3} dfa_ms={dfa_ms:.3} total_ms={:.3}",
                state_count,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Ok(Tokenizer {
            dfa,
            num_terminals: artifact.num_terminals,
            packed_runtime_transitions: Some(packed_runtime_transitions),
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        })
    }
}

impl CompressedTransitionSegment {
    #[inline]
    pub(super) fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.state_count
    }

    #[inline]
    fn local_transition(&self, local_state: u32, byte: u8) -> Option<u32> {
        let class = self.byte_to_class[byte as usize];
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        self.entries
            .class_slice(start, end)
            .binary_search(&class)
            .ok()
            .map(|index| self.entries.target(start + index))
    }

    #[inline]
    pub(super) fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        self.local_transition(state - self.state_offset, byte)
            .map(|target| self.state_offset + target)
    }

    fn expanded_entries(&self, state: u32) -> Vec<(u8, u32)> {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        let mut target_by_class = vec![u32::MAX; self.class_members.len()];
        let mut capacity = 0usize;
        for (class, target) in self.entries.iter_range(start, end) {
            target_by_class[class as usize] = target;
            capacity += self.class_members[class as usize].len();
        }
        let mut entries = Vec::with_capacity(capacity);
        for byte in 0u16..=255 {
            let class = self.byte_to_class[byte as usize] as usize;
            let target = target_by_class[class];
            if target != u32::MAX {
                entries.push((byte as u8, self.state_offset + target));
            }
        }
        entries
    }

    pub fn materialize_into_dfa(&self, dfa: &mut DFA) {
        for local_state in 0..self.state_count {
            let state = self.state_offset + local_state;
            dfa.set_transitions_from_sorted_entries(state, self.expanded_entries(state));
        }
    }

    pub(super) fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        for (class, target) in self.entries.iter_range(start, end) {
            let target = self.state_offset + target;
            for &byte in self.class_members[class as usize].iter() {
                row[byte as usize] = target;
            }
        }
    }

    pub(super) fn transition_count(&self, state: u32) -> usize {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        self.entries
            .class_slice(start, end)
            .iter()
            .map(|class| self.class_members[*class as usize].len())
            .sum()
    }

    pub(super) fn transitions_satisfy(
        &self,
        state: u32,
        mut predicate: impl FnMut(u8, u32) -> bool,
    ) -> bool {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        for (class, target) in self.entries.iter_range(start, end) {
            let target = self.state_offset + target;
            for &byte in self.class_members[class as usize].iter() {
                if !predicate(byte, target) {
                    return false;
                }
            }
        }
        true
    }
}

enum TokenizerTransitionsIterInner<'a> {
    Dense(crate::ds::char_transitions::CharTransitionsIter<'a, u32>),
    Packed {
        bytes: &'a [u8],
        targets: PackedRuntimeTargetSlice<'a>,
        next: usize,
    },
    PackedSegment {
        bytes: &'a [u8],
        targets: PackedRuntimeTargetSlice<'a>,
        target_offset: u32,
        next: usize,
    },
    Compressed {
        segment: &'a CompressedTransitionSegment,
        state: u32,
        next_byte: u16,
    },
    PackedCompressed {
        segment: &'a PackedCompressedTransitionSegment,
        state: u32,
        next_byte: u16,
    },
    Empty,
}

pub struct TokenizerTransitionsIter<'a> {
    inner: TokenizerTransitionsIterInner<'a>,
}

impl Iterator for TokenizerTransitionsIter<'_> {
    type Item = (u8, u32);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => {
                iter.next().map(|(byte, target)| (byte, *target))
            }
            TokenizerTransitionsIterInner::Packed {
                bytes,
                targets,
                next,
            } => {
                let index = *next;
                let byte = *bytes.get(index)?;
                let target = targets.get(index)?;
                *next += 1;
                Some((byte, target))
            }
            TokenizerTransitionsIterInner::PackedSegment {
                bytes,
                targets,
                target_offset,
                next,
            } => {
                let index = *next;
                let byte = *bytes.get(index)?;
                let target = targets.get(index)?.checked_add(*target_offset)?;
                *next += 1;
                Some((byte, target))
            }
            TokenizerTransitionsIterInner::Compressed {
                segment,
                state,
                next_byte,
            } => {
                while *next_byte <= 255 {
                    let byte = *next_byte as u8;
                    *next_byte += 1;
                    if let Some(target) = segment.transition(*state, byte) {
                        return Some((byte, target));
                    }
                }
                None
            }
            TokenizerTransitionsIterInner::PackedCompressed {
                segment,
                state,
                next_byte,
            } => {
                while *next_byte <= 255 {
                    let byte = *next_byte as u8;
                    *next_byte += 1;
                    if let Some(target) = segment.transition(*state, byte) {
                        return Some((byte, target));
                    }
                }
                None
            }
            TokenizerTransitionsIterInner::Empty => None,
        }
    }


    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => iter.size_hint(),
            TokenizerTransitionsIterInner::Packed { bytes, next, .. }
            | TokenizerTransitionsIterInner::PackedSegment { bytes, next, .. } => {
                let count = bytes.len().saturating_sub(*next);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::Compressed { segment, state, .. } => {
                let count = segment.transition_count(*state);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::PackedCompressed { segment, state, .. } => {
                let count = segment.transition_count(*state);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::Empty => (0, Some(0)),
        }
    }

    fn count(self) -> usize {
        match self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => iter.count(),
            TokenizerTransitionsIterInner::Packed { bytes, next, .. }
            | TokenizerTransitionsIterInner::PackedSegment { bytes, next, .. } => {
                bytes.len().saturating_sub(next)
            }
            TokenizerTransitionsIterInner::Compressed { segment, state, .. } => {
                segment.transition_count(state)
            }
            TokenizerTransitionsIterInner::PackedCompressed { segment, state, .. } => {
                segment.transition_count(state)
            }
            TokenizerTransitionsIterInner::Empty => 0,
        }
    }
}

impl Serialize for Tokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let materialized;
        let dfa = if self.compressed_transition_segments.is_empty() {
            &self.dfa
        } else {
            materialized = self.materialized_dfa();
            &materialized
        };
        // Match the historical derived-serialization field order exactly.
        let mut state = serializer.serialize_struct("Tokenizer", 2)?;
        state.serialize_field("dfa", dfa)?;
        state.serialize_field("num_terminals", &self.num_terminals)?;
        state.end()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
    pub end_state: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: TokenizerStateSet,
    pub matches: Vec<TokenizerMatch>,
}

pub type TokenizerStateSet = SmallVec<[u32; 1]>;

/// Exact disjoint union used only by cross-tokenizer compile-time analyses.
/// Source state `s` is represented by `left_offset + s` or
/// `right_offset + s`; state zero is a fresh epsilon dispatcher.
pub struct TokenizerAnalysisUnion {
    pub tokenizer: Tokenizer,
    pub left_offset: u32,
    pub right_offset: u32,
}

pub trait Lexer {
    fn start_state(&self) -> u32;
    fn num_terminals(&self) -> u32;
    fn has_epsilon_transitions(&self) -> bool;
    fn state_has_epsilon_transitions(&self, state: u32) -> bool;
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_;

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    /// Largest H<=`max_horizon` such that every byte string over `bytes` of
    /// length at most H preserves the lexer observation of `source`.
    ///
    /// This is an exact local proof for scalar deterministic states. It follows
    /// only states actually reachable from `source` under the requested byte
    /// alphabet, deduplicating the frontier at every depth. Bounded-repeat
    /// chains therefore cost O(H) states rather than an O(all-tokenizer-states)
    /// prepass. If the reachable frontier becomes too wide, return the already
    /// proved shorter horizon and let the ordinary exact runtime walk handle
    /// the subtree.
    fn bounded_observation_safe_horizon_from_state(
        &self,
        source: u32,
        bytes: U8Set,
        active_terminals: &BitSet,
        max_horizon: u8,
    ) -> u8 {
        const MAX_FRONTIER_STATES: usize = 4_096;

        #[inline]
        fn equal_under_mask(left: &BitSet, right: &BitSet, mask: &BitSet) -> bool {
            debug_assert_eq!(left.len(), right.len());
            debug_assert_eq!(left.len(), mask.len());
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        if max_horizon == 0
            || bytes.is_empty()
            || source >= self.num_states()
            || self.state_has_epsilon_transitions(source)
        {
            return 0;
        }

        let mut frontier = vec![source];
        for depth in 1..=max_horizon {
            let mut next = Vec::<u32>::new();
            for &state in &frontier {
                if self.state_has_epsilon_transitions(state) {
                    return depth - 1;
                }
                for byte in bytes.iter() {
                    let target = self.get_transition(state, byte);
                    if target == u32::MAX || self.state_has_epsilon_transitions(target) {
                        return depth - 1;
                    }
                    if !equal_under_mask(
                        self.matched_terminal_bitset(target),
                        self.matched_terminal_bitset(source),
                        active_terminals,
                    ) || !equal_under_mask(
                        self.possible_future_terminals(target),
                        self.possible_future_terminals(source),
                        active_terminals,
                    )
                    {
                        return depth - 1;
                    }
                    next.push(target);
                }
            }
            next.sort_unstable();
            next.dedup();
            if next.len() > MAX_FRONTIER_STATES {
                return depth - 1;
            }
            // The same closed deterministic frontier has reappeared. Every
            // requested byte remains inside this already-validated set, so the
            // certificate is valid for every longer horizon too.
            if next == frontier {
                return max_horizon;
            }
            frontier = next;
        }
        max_horizon
    }

    fn transition_count(&self) -> usize {
        (0..self.num_states())
            .map(|state| self.transitions_from(state).count())
            .sum()
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32>;
    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet;
    fn get_transition(&self, state: u32, byte: u8) -> u32;
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet;
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals(&self, state: u32) -> &BitSet;

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32;
    fn compute_forced_minimized_state_count(&self) -> usize;
    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult;
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult;
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet;
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult;

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, TokenizerStateSet);
}

fn into_longest_matches(
    matches: FxHashMap<TerminalID, (usize, TokenizerStateSet)>,
) -> Vec<TokenizerMatch> {
    matches
        .into_iter()
        .flat_map(|(id, (width, end_states))| {
            end_states.into_iter().map(move |end_state| TokenizerMatch {
                id,
                width,
                end_state,
            })
        })
        .collect()
}

fn group_matches_by_width(matches: Vec<TokenizerMatch>) -> Vec<(usize, BTreeSet<TerminalID>)> {
    let mut grouped = std::collections::BTreeMap::<usize, BTreeSet<TerminalID>>::new();
    for matched in matches {
        grouped.entry(matched.width).or_default().insert(matched.id);
    }
    grouped.into_iter().collect()
}

impl Tokenizer {
    /// Exact finite-horizon quotient for the Boolean observation
    /// `terminal is still a possible future terminal`.
    ///
    /// Class zero is the dead class: states from which `terminal` is already
    /// impossible, epsilon-bearing states, and missing byte transitions all
    /// behave identically for the no-finalization continuation proof used by
    /// dynamic trie projections.  For each subsequent round we partition live
    /// states by the byte -> previous-class map, omitting edges into class zero.
    /// After H rounds, equal class ids therefore imply equal terminal-liveness
    /// for every byte string of length at most H.
    pub fn bounded_terminal_future_partition(
        &self,
        terminal: TerminalID,
        horizon: u8,
    ) -> Box<[u32]> {
        use rustc_hash::FxHashMap;

        let state_count = self.num_states() as usize;
        let mut classes = vec![0u32; state_count];
        for state in 0..state_count {
            let state_u32 = state as u32;
            if !self.state_has_epsilon_transitions(state_u32)
                && self
                    .possible_future_terminals(state_u32)
                    .contains(terminal as usize)
            {
                classes[state] = 1;
            }
        }
        if horizon == 0 {
            return classes.into_boxed_slice();
        }

        // Most synthesized bounded-repeat rows have roughly 90 live bytes.
        // Keep those signatures inline; only unusually broad rows allocate.
        type Signature = SmallVec<[(u8, u32); 128]>;
        let mut next = vec![0u32; state_count];
        for _ in 0..horizon {
            let mut ids = FxHashMap::<Signature, u32>::default();
            let mut next_id = 1u32;
            for state in 0..state_count {
                if classes[state] == 0 {
                    continue;
                }
                let mut signature = Signature::new();
                for (byte, target) in self.transitions_from(state as u32) {
                    let target_class = classes.get(target as usize).copied().unwrap_or(0);
                    if target_class != 0 {
                        signature.push((byte, target_class));
                    }
                }
                let id = if let Some(&id) = ids.get(&signature) {
                    id
                } else {
                    let id = next_id;
                    next_id = next_id.saturating_add(1);
                    ids.insert(signature, id);
                    id
                };
                next[state] = id;
            }
            std::mem::swap(&mut classes, &mut next);
            next.fill(0);
        }
        classes.into_boxed_slice()
    }

    fn merged_terminal_exprs(
        tokenizers: &[(&Tokenizer, TerminalID)],
        total_terminals: TerminalID,
    ) -> Option<Arc<[Expr]>> {
        let mut merged = vec![None::<Expr>; total_terminals as usize];
        for &(tokenizer, terminal_offset) in tokenizers {
            let exprs = tokenizer.exprs.as_deref()?;
            if exprs.len() != tokenizer.num_terminals as usize {
                return None;
            }
            for (terminal, expr) in exprs.iter().enumerate() {
                let slot = merged.get_mut(terminal_offset as usize + terminal)?;
                if slot.is_some() {
                    return None;
                }
                *slot = Some(expr.clone());
            }
        }
        merged
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(Arc::from)
    }

    pub fn canonicalize_terminal_aliases(
        &mut self,
        canonical: TerminalID,
        aliases: &[TerminalID],
    ) {
        let aliases = aliases
            .iter()
            .copied()
            .filter(|&alias| alias != canonical)
            .map(|alias| alias as usize)
            .collect::<Vec<_>>();
        self.dfa
            .canonicalize_group_aliases(canonical as usize, &aliases);
        self.invalidate_derived_caches();
    }

    /// Form an exact disjoint union of independently compiled tokenizers while
    /// keeping every terminal ID distinct.
    ///
    /// `terminal_offsets[i]` is added to every terminal/group ID in
    /// `tokenizers[i]`. A fresh epsilon root dispatches to each source start
    /// state. No DFA states or terminals are identified across inputs.
    ///
    /// The returned state offsets map each source raw tokenizer state into the
    /// merged raw-state domain.
    /// Form one ordinary flattened tokenizer by consuming the parent as the
    /// destination and appending borrowed child components. Parent state IDs and
    /// transition buffers remain unchanged; only child states are cloned and
    /// rebased. Terminal IDs in the parent remain identity-mapped from zero.
    pub fn disjoint_union_with_owned_parent(
        mut parent: Tokenizer,
        parent_terminal_offset: TerminalID,
        children: &[(&Tokenizer, TerminalID)],
    ) -> (Tokenizer, Vec<u32>) {
        assert_eq!(
            parent_terminal_offset, 0,
            "owned-parent tokenizer composition requires the parent terminal domain to start at zero",
        );
        // Fast-loaded tokenizers may keep finalizer/future/epsilon metadata in
        // packed runtime storage while retaining only the structural DFA
        // skeleton.  This function mutates the parent's epsilon graph in
        // place, so leaving packed metadata authoritative would hide every new
        // child edge (and can make the start state look like a pure dispatcher
        // even when its byte transitions still live in packed storage).
        // Materialize metadata only; byte-transition rows remain packed.
        parent.materialize_runtime_metadata_for_structural_mutation();
        let total_terminals = std::iter::once((&parent, parent_terminal_offset))
            .chain(children.iter().copied())
            .map(|(tokenizer, terminal_offset)| {
                terminal_offset
                    .checked_add(tokenizer.num_terminals)
                    .expect("merged tokenizer terminal ID overflow")
            })
            .max()
            .unwrap_or(0);
        let expr_components = std::iter::once((&parent, parent_terminal_offset))
            .chain(children.iter().copied())
            .collect::<Vec<_>>();
        let merged_exprs = Self::merged_terminal_exprs(&expr_components, total_terminals);

        let mut merged_byte_transition_count = parent.transition_count();
        let mut merged_epsilon_transition_count = parent.dfa.epsilon_transition_count();
        let mut merged_has_self_loops = parent.dfa.has_self_loops();
        for &(tokenizer, _) in children {
            merged_byte_transition_count = merged_byte_transition_count
                .checked_add(tokenizer.transition_count())
                .expect("merged tokenizer byte-transition count overflow");
            merged_epsilon_transition_count = merged_epsilon_transition_count
                .checked_add(tokenizer.dfa.epsilon_transition_count())
                .and_then(|count| count.checked_add(1))
                .expect("merged tokenizer epsilon-transition count overflow");
            merged_has_self_loops |= tokenizer.dfa.has_self_loops();
        }

        parent
            .dfa
            .ensure_group_mapping_capacity(total_terminals as usize);
        let parent_closures = std::mem::take(&mut parent.singleton_epsilon_closures)
            .into_inner()
            .and_then(|closures| Arc::try_unwrap(closures).ok());
        let mut child_closures = Vec::with_capacity(children.len());
        let mut state_offsets = Vec::with_capacity(children.len() + 1);
        state_offsets.push(0);
        let mut compressed_segments = parent
            .compressed_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut packed_runtime_segments = parent
            .packed_runtime_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut packed_metadata_segments = parent
            .packed_runtime_metadata_segments
            .iter()
            .map(|segment| PackedTokenizerMetadataSegment {
                state_offset: segment.state_offset,
                metadata: segment
                    .metadata
                    .rebased_terminals(parent_terminal_offset, total_terminals),
            })
            .collect::<Vec<_>>();
        let mut packed_compressed_segments = parent
            .packed_compressed_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut root_finalizers = BitSet::new(total_terminals as usize);
        for terminal in parent.matched_terminals_iter(parent.start_state()) {
            root_finalizers.set(terminal as usize);
        }
        let mut root_futures = BitSet::new(total_terminals as usize);
        for terminal in parent.possible_future_terminals_iter(parent.start_state()) {
            root_futures.set(terminal as usize);
        }

        for &(tokenizer, terminal_offset) in children {
            let global_groups = (0..tokenizer.num_terminals)
                .map(|terminal| (terminal_offset + terminal) as usize)
                .collect::<Vec<_>>();
            let state_offset = parent
                .dfa
                .append_rebased_component_ref(&tokenizer.dfa, &global_groups);
            state_offsets.push(state_offset);
            if let Some(closures) = tokenizer.cached_singleton_epsilon_closures() {
                child_closures.push((Arc::clone(closures), state_offset, tokenizer.start_state()));
            }
            parent
                .dfa
                .add_epsilon_transition(parent.start_state(), state_offset + tokenizer.start_state());
            if let Some(transitions) = tokenizer.packed_runtime_transitions.as_ref() {
                packed_runtime_segments.push(PackedRuntimeTransitionSegment {
                    state_offset,
                    transitions: Arc::clone(transitions),
                });
            }
            packed_runtime_segments.extend(
                tokenizer
                    .packed_runtime_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer state offset overflow");
                        segment
                    }),
            );
            if let Some(metadata) = tokenizer.packed_runtime_metadata.as_ref() {
                packed_metadata_segments.push(PackedTokenizerMetadataSegment {
                    state_offset,
                    metadata: metadata.rebased_terminals(terminal_offset, total_terminals),
                });
            }
            packed_metadata_segments.extend(
                tokenizer
                    .packed_runtime_metadata_segments
                    .iter()
                    .map(|segment| {
                        let rebased_state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer metadata state offset overflow");
                        PackedTokenizerMetadataSegment {
                            state_offset: rebased_state_offset,
                            metadata: segment
                                .metadata
                                .rebased_terminals(terminal_offset, total_terminals),
                        }
                    }),
            );
            packed_compressed_segments.extend(
                tokenizer
                    .packed_compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed-compressed tokenizer state offset overflow");
                        segment
                    }),
            );
            for terminal in tokenizer.matched_terminals_iter(tokenizer.start_state()) {
                root_finalizers.set(terminal_offset as usize + terminal as usize);
            }
            for terminal in tokenizer.possible_future_terminals_iter(tokenizer.start_state()) {
                root_futures.set(terminal_offset as usize + terminal as usize);
            }
            compressed_segments.extend(
                tokenizer
                    .compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged compressed tokenizer state offset overflow");
                        segment
                    }),
            );
        }
        parent
            .dfa
            .overwrite_state_metadata(parent.start_state(), root_finalizers, root_futures);
        parent.dfa.set_derived_stats(
            merged_byte_transition_count,
            merged_epsilon_transition_count,
            merged_has_self_loops,
        );
        parent.num_terminals = total_terminals;
        parent.compressed_transition_segments =
            Arc::from(compressed_segments.into_boxed_slice());
        packed_runtime_segments.sort_unstable_by_key(|segment| segment.state_offset);
        parent.packed_runtime_transition_segments =
            Arc::from(packed_runtime_segments.into_boxed_slice());
        packed_metadata_segments.sort_unstable_by_key(|segment| segment.state_offset);
        parent.packed_runtime_metadata_segments =
            Arc::from(packed_metadata_segments.into_boxed_slice());
        packed_compressed_segments.sort_unstable_by_key(|segment| segment.state_offset);
        parent.packed_compressed_transition_segments =
            Arc::from(packed_compressed_segments.into_boxed_slice());
        parent.exprs = merged_exprs;
        parent.invalidate_derived_caches();
        if child_closures.len() == children.len()
            && let Some(closures) = parent_closures.and_then(|closures| {
            closures.append_rebased_children(
                parent.start_state(),
                &child_closures,
            )
        }) {
            let _ = parent
                .singleton_epsilon_closures
                .set(Arc::new(closures));
        }
        (parent, state_offsets)
    }

    pub fn disjoint_union_with_terminal_offsets(
        tokenizers: &[(&Tokenizer, TerminalID)],
    ) -> (Tokenizer, Vec<u32>) {
        let total_terminals = tokenizers
            .iter()
            .map(|(tokenizer, terminal_offset)| {
                terminal_offset
                    .checked_add(tokenizer.num_terminals)
                    .expect("merged tokenizer terminal ID overflow")
            })
            .max()
            .unwrap_or(0);
        let merged_exprs = Self::merged_terminal_exprs(tokenizers, total_terminals);

        let total_states = 1usize.saturating_add(
            tokenizers
                .iter()
                .map(|(tokenizer, _)| tokenizer.num_states() as usize)
                .sum::<usize>(),
        );
        let mut merged = DFA::new(total_states.min(1));
        merged.ensure_group_capacity(total_terminals as usize);
        let mut state_offsets = Vec::with_capacity(tokenizers.len());
        let mut compressed_segments = Vec::<CompressedTransitionSegment>::new();
        let mut packed_runtime_segments = Vec::<PackedRuntimeTransitionSegment>::new();
        let mut packed_metadata_segments = Vec::<PackedTokenizerMetadataSegment>::new();
        let mut packed_compressed_segments = Vec::<PackedCompressedTransitionSegment>::new();
        let mut root_finalizers = BitSet::new(total_terminals as usize);
        let mut root_futures = BitSet::new(total_terminals as usize);

        for &(tokenizer, terminal_offset) in tokenizers {
            // Preserve compact runtime transition segments. Materializing them
            // here expands every byte-class row before immediately rebuilding
            // equivalent runtime caches, which is catastrophic for million-state
            // composed tokenizers. Segment targets are local to the segment, so
            // rebasing only its state offset is exact.
            let component = tokenizer.dfa.clone();
            let global_groups = (0..tokenizer.num_terminals)
                .map(|terminal| (terminal_offset + terminal) as usize)
                .collect::<Vec<_>>();
            let state_offset = merged.append_rebased_component(component, &global_groups);
            if let Some(transitions) = tokenizer.packed_runtime_transitions.as_ref() {
                packed_runtime_segments.push(PackedRuntimeTransitionSegment {
                    state_offset,
                    transitions: Arc::clone(transitions),
                });
            }
            packed_runtime_segments.extend(
                tokenizer
                    .packed_runtime_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer state offset overflow");
                        segment
                    }),
            );
            if let Some(metadata) = tokenizer.packed_runtime_metadata.as_ref() {
                packed_metadata_segments.push(PackedTokenizerMetadataSegment {
                    state_offset,
                    metadata: metadata.rebased_terminals(terminal_offset, total_terminals),
                });
            }
            packed_metadata_segments.extend(
                tokenizer
                    .packed_runtime_metadata_segments
                    .iter()
                    .map(|segment| {
                        let rebased_state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer metadata state offset overflow");
                        PackedTokenizerMetadataSegment {
                            state_offset: rebased_state_offset,
                            metadata: segment
                                .metadata
                                .rebased_terminals(terminal_offset, total_terminals),
                        }
                    }),
            );
            compressed_segments.extend(
                tokenizer
                    .compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged compressed tokenizer state offset overflow");
                        segment
                    }),
            );
            packed_compressed_segments.extend(
                tokenizer
                    .packed_compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed-compressed tokenizer state offset overflow");
                        segment
                    }),
            );
            state_offsets.push(state_offset);
            merged.add_epsilon_transition(0, state_offset + tokenizer.start_state());

            for terminal in tokenizer.matched_terminals_iter(tokenizer.start_state()) {
                root_finalizers.set(terminal_offset as usize + terminal as usize);
            }
            for terminal in tokenizer.possible_future_terminals_iter(tokenizer.start_state()) {
                root_futures.set(terminal_offset as usize + terminal as usize);
            }
        }
        merged.overwrite_state_metadata(0, root_finalizers, root_futures);
        packed_runtime_segments.sort_unstable_by_key(|segment| segment.state_offset);
        packed_metadata_segments.sort_unstable_by_key(|segment| segment.state_offset);
        packed_compressed_segments.sort_unstable_by_key(|segment| segment.state_offset);

        let mut tokenizer = Tokenizer::from_parts_with_compressed_transitions(
            merged,
            total_terminals,
            merged_exprs,
            compressed_segments,
        );
        tokenizer.packed_runtime_transition_segments =
            Arc::from(packed_runtime_segments.into_boxed_slice());
        tokenizer.packed_runtime_metadata_segments =
            Arc::from(packed_metadata_segments.into_boxed_slice());
        tokenizer.packed_compressed_transition_segments =
            Arc::from(packed_compressed_segments.into_boxed_slice());
        (tokenizer, state_offsets)
    }

    #[inline]
    fn invalidate_derived_caches(&mut self) {
        let _ = self.singleton_epsilon_closures.take();
        let _ = self.matched_terminals_cache.take();
        let _ = self.initial_byte_frontiers.take();
        let _ = self.all_self_loop_bytes_cache.take();
        let _ = self.transition_count_cache.take();
        let _ = self.forced_minimized_state_count_cache.take();
        let _ = self.scalar_deterministic_dispatch_cache.take();
    }

    /// Fully determinize the current runtime tokenizer by exact subset
    /// construction.  Each returned DFA state carries the epsilon-closed set of
    /// source states it represents so callers can transport the already-final
    /// tokenizer-state ID map without rebuilding compiler analyses.
    pub fn try_full_determinization(
        &self,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<FullTokenizerDeterminization> {
        if state_limit == 0 || transition_limit == 0 || !self.has_epsilon_transitions() {
            return None;
        }

        let mut start = self.dfa.epsilon_closure(&[self.initial_state_id()]);
        start.sort_unstable();
        start.dedup();
        if start.is_empty() {
            return None;
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals {
            dfa.set_group_u8set(
                terminal,
                *self.dfa.group_id_to_u8set(terminal),
            );
        }

        let metadata = |subset: &[u32]| {
            let mut finalizers = BitSet::new(self.num_terminals as usize);
            let mut futures = BitSet::new(self.num_terminals as usize);
            for &state in subset {
                // Structural tokenizer composition may append DFA components
                // whose per-state metadata bitsets were created in a smaller
                // local terminal domain. Their set bits are already rebased to
                // global group IDs, but the backing BitSet width need not have
                // been eagerly widened on every historical state. Product
                // construction needs set union, not identical storage widths;
                // materialize that union into the known global domain.
                for terminal in self.dfa.finalizers(state).iter() {
                    finalizers.set(terminal);
                }
                for terminal in self.dfa.possible_future_group_ids(state).iter() {
                    futures.set(terminal);
                }
            }
            (finalizers, futures)
        };
        let (start_finalizers, start_futures) = metadata(&start);
        dfa.overwrite_state_metadata(0, start_finalizers, start_futures);

        let start: Box<[u32]> = start.into_vec().into_boxed_slice();
        let mut source_subsets = vec![start.clone()];
        let mut state_by_subset = FxHashMap::<Box<[u32]>, u32>::default();
        state_by_subset.insert(start, 0);
        let mut worklist = VecDeque::from([0u32]);
        let mut transitions_built = 0usize;

        while let Some(determinized_state) = worklist.pop_front() {
            let subset = source_subsets[determinized_state as usize].clone();
            let mut transitions = Vec::<(u8, u32)>::new();
            for byte in 0u16..=255 {
                let mut targets = SmallVec::<[u32; 8]>::new();
                for &source_state in subset.iter() {
                    if let Some(target) = self.step(source_state, byte as u8) {
                        targets.push(target);
                    }
                }
                if targets.is_empty() {
                    continue;
                }
                targets.sort_unstable();
                targets.dedup();
                let mut closed = self.dfa.epsilon_closure(&targets);
                closed.sort_unstable();
                closed.dedup();
                if closed.is_empty() {
                    continue;
                }
                transitions_built = transitions_built.saturating_add(1);
                if transitions_built > transition_limit {
                    return None;
                }
                let closed: Box<[u32]> = closed.into_vec().into_boxed_slice();
                let target = if let Some(&existing) = state_by_subset.get(&closed) {
                    existing
                } else {
                    if source_subsets.len() >= state_limit {
                        return None;
                    }
                    let new_state = dfa.add_state();
                    debug_assert_eq!(new_state as usize, source_subsets.len());
                    let (finalizers, futures) = metadata(&closed);
                    dfa.overwrite_state_metadata(new_state, finalizers, futures);
                    state_by_subset.insert(closed.clone(), new_state);
                    source_subsets.push(closed);
                    worklist.push_back(new_state);
                    new_state
                };
                transitions.push((byte as u8, target));
            }
            dfa.set_transitions_from_sorted_entries(determinized_state, transitions);
        }

        let closures = self.all_singleton_epsilon_closures();
        let mut source_by_closure = FxHashMap::<Box<[u32]>, u32>::default();
        // Prefer the true initial state when another raw state happens to have
        // the same closure: accumulator state keys are observable at commit.
        let initial = self.initial_state_id();
        source_by_closure.insert(
            closures[initial as usize].to_vec().into_boxed_slice(),
            initial,
        );
        for (state, closure) in closures.iter().enumerate() {
            source_by_closure
                .entry(closure.to_vec().into_boxed_slice())
                .or_insert(state as u32);
        }
        let exact_source_states = source_subsets
            .iter()
            .map(|subset| source_by_closure.get(subset).copied().unwrap_or(u32::MAX))
            .collect();

        Some(FullTokenizerDeterminization {
            tokenizer: Tokenizer {
                dfa,
                num_terminals: self.num_terminals,
                packed_runtime_transitions: None,
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: None,
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                exprs: self.exprs.clone(),
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache: OnceLock::new(),
            },
            source_subsets,
            source_state_offset: u32::MAX,
            exact_source_states,
        })
    }

    /// Move the exact source tokenizer behind a completed subset tokenizer.
    ///
    /// Product states are safe only while one parser language is uniformly
    /// associated with every source state in their subset. Runtime commit can
    /// expand such a state into this appended source coordinate, execute the
    /// historical NFA semantics unchanged, and re-coalesce only exact uniform
    /// subsets afterward. The product start state remains state zero.
    pub fn finish_full_determinization_with_source_fallback(
        &mut self,
        mut built: FullTokenizerDeterminization,
    ) -> FullTokenizerDeterminization {
        debug_assert_eq!(built.source_subsets.len(), built.tokenizer.num_states() as usize);
        debug_assert_eq!(built.exact_source_states.len(), built.source_subsets.len());

        let mut source_dfa = std::mem::replace(&mut self.dfa, DFA::new(0));
        // Immutable disjoint tokenizer composition deliberately permits old
        // component states to retain shorter finalizer/future bitsets: group
        // membership beyond that shorter local domain is simply false. Once
        // those states are appended behind a deterministic runtime product,
        // however, product and fallback states coexist in one live tokenizer
        // frontier and runtime admission unions their metadata directly. Make
        // the source fallback use the product tokenizer's one canonical
        // terminal domain before publishing it.
        source_dfa.ensure_group_capacity(self.num_terminals as usize);
        let global_groups = (0..self.num_terminals as usize).collect::<Vec<_>>();
        built.source_state_offset = built
            .tokenizer
            .dfa
            .append_rebased_component(source_dfa, &global_groups);
        let mut compressed_segments = built
            .tokenizer
            .compressed_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        compressed_segments.extend(self.compressed_transition_segments.iter().cloned().map(
            |mut segment| {
                segment.state_offset = segment
                    .state_offset
                    .checked_add(built.source_state_offset)
                    .expect("runtime tokenizer compressed state offset overflow");
                segment
            },
        ));
        built.tokenizer.compressed_transition_segments = Arc::from(compressed_segments);
        self.compressed_transition_segments = Arc::from([]);
        self.invalidate_derived_caches();
        built.tokenizer.invalidate_derived_caches();
        built
    }

    /// Materialize a deterministic compile-time analysis view as a tokenizer.
    /// The view may be a powerset of this tokenizers epsilon-NFA. State zero is
    /// reserved for the supplied start state, and the returned old-to-new map
    /// lets callers lift raw-start mappings into the materialized coordinate.
    pub fn materialize_deterministic_view(
        &self,
        start_state: usize,
        finalizers: &[Vec<usize>],
        futures: &[Vec<usize>],
        edge_offsets: &[u32],
        edges: &[(u8, u32)],
        active_terminals: &[bool],
    ) -> Option<(Tokenizer, Vec<u32>)> {
        let state_count = finalizers.len();
        if state_count == 0
            || futures.len() != state_count
            || edge_offsets.len() != state_count + 1
            || start_state >= state_count
            || active_terminals.len() != self.num_terminals as usize
        {
            return None;
        }
        let mut new_to_old = Vec::with_capacity(state_count);
        new_to_old.push(start_state);
        new_to_old.extend((0..state_count).filter(|&state| state != start_state));
        let mut old_to_new = vec![u32::MAX; state_count];
        for (new, &old) in new_to_old.iter().enumerate() {
            old_to_new[old] = new as u32;
        }

        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals as usize {
            if active_terminals[terminal] {
                dfa.set_group_u8set(
                    terminal as u32,
                    *self.dfa.group_id_to_u8set(terminal as u32),
                );
            }
        }
        for (new_state, &old_state) in new_to_old.iter().enumerate() {
            let start = *edge_offsets.get(old_state)? as usize;
            let end = *edge_offsets.get(old_state + 1)? as usize;
            let transitions = edges
                .get(start..end)?
                .iter()
                .map(|&(byte, target)| {
                    old_to_new
                        .get(target as usize)
                        .copied()
                        .filter(|&target| target != u32::MAX)
                        .map(|target| (byte, target))
                })
                .collect::<Option<Vec<_>>>()?;
            dfa.set_transitions_from_sorted_entries(new_state as u32, transitions);
            let to_bits = |groups: &[usize]| {
                let mut bits = BitSet::new(self.num_terminals as usize);
                for &group in groups {
                    if group >= active_terminals.len() || !active_terminals[group] {
                        return None;
                    }
                    bits.set(group);
                }
                Some(bits)
            };
            dfa.overwrite_state_metadata(
                new_state as u32,
                to_bits(&finalizers[old_state])?,
                to_bits(&futures[old_state])?,
            );
        }
        Some((
            Tokenizer {
                dfa,
                num_terminals: self.num_terminals,
                packed_runtime_transitions: None,
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: None,
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                exprs: None,
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache: OnceLock::new(),
            },
            old_to_new,
        ))
    }

    /// Materialize an exact deterministic quotient for one compile-time branch.
    ///
    /// `original_to_quotient` must be a congruence for every vocabulary-relevant
    /// byte after filtering labels to `active_terminals`. The method verifies
    /// that property over every class member before constructing the smaller
    /// tokenizer, so callers can fail closed to the original tokenizer.
    pub fn materialize_active_quotient(
        &self,
        original_to_quotient: &[u32],
        representatives: &[u32],
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Option<Tokenizer> {
        if original_to_quotient.len() != self.num_states() as usize
            || active_terminals.len() != self.num_terminals as usize
            || representatives.is_empty()
            || original_to_quotient.get(self.start_state() as usize).copied() != Some(0)
        {
            return None;
        }
        let quotient_states = representatives.len();
        if original_to_quotient
            .iter()
            .any(|&state| state == u32::MAX || state as usize >= quotient_states)
        {
            return None;
        }

        let filtered = |bits: &BitSet| {
            let mut result = BitSet::new(self.num_terminals as usize);
            for terminal in bits.iter() {
                if active_terminals.get(terminal).copied().unwrap_or(false) {
                    result.set(terminal);
                }
            }
            result
        };

        // Verify output labels and every relevant transition for all members,
        // rather than trusting the refinement implementation as an implicit
        // construction contract.
        let mut class_members = vec![Vec::<u32>::new(); quotient_states];
        for (original, &quotient) in original_to_quotient.iter().enumerate() {
            class_members[quotient as usize].push(original as u32);
        }
        for (class, members) in class_members.iter().enumerate() {
            let representative = *representatives.get(class)?;
            if !members.contains(&representative) {
                return None;
            }
            let representative_finalizers = filtered(self.dfa.finalizers(representative));
            let representative_futures =
                filtered(self.dfa.possible_future_group_ids(representative));
            for &member in members {
                if filtered(self.dfa.finalizers(member)) != representative_finalizers
                    || filtered(self.dfa.possible_future_group_ids(member))
                        != representative_futures
                {
                    return None;
                }
                for byte in 0u16..=255 {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let mapped = self
                        .step(member, byte as u8)
                        .map(|target| original_to_quotient[target as usize]);
                    let representative_mapped = self
                        .step(representative, byte as u8)
                        .map(|target| original_to_quotient[target as usize]);
                    if mapped != representative_mapped {
                        return None;
                    }
                }
                let mapped_epsilon = |state: u32| {
                    let mut targets = self.dfa.states()[state as usize]
                        .epsilon_transitions
                        .iter()
                        .map(|&target| original_to_quotient[target as usize])
                        .collect::<Vec<_>>();
                    targets.sort_unstable();
                    targets.dedup();
                    targets
                };
                if mapped_epsilon(member) != mapped_epsilon(representative) {
                    return None;
                }
            }
        }

        let mut dfa = DFA::new(quotient_states);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals as usize {
            if active_terminals[terminal] {
                dfa.set_group_u8set(
                    terminal as u32,
                    *self.dfa.group_id_to_u8set(terminal as u32),
                );
            }
        }
        for (class, &representative) in representatives.iter().enumerate() {
            let transitions = (0u16..=255)
                .filter(|&byte| relevant_bytes[byte as usize])
                .filter_map(|byte| {
                    self.step(representative, byte as u8).map(|target| {
                        (byte as u8, original_to_quotient[target as usize])
                    })
                })
                .collect::<Vec<_>>();
            dfa.set_transitions_from_sorted_entries(class as u32, transitions);
            let mut epsilon_targets = self.dfa.states()[representative as usize]
                .epsilon_transitions
                .iter()
                .map(|&target| original_to_quotient[target as usize])
                .collect::<Vec<_>>();
            epsilon_targets.sort_unstable();
            epsilon_targets.dedup();
            for target in epsilon_targets {
                dfa.add_epsilon_transition(class as u32, target);
            }
            dfa.overwrite_state_metadata(
                class as u32,
                filtered(self.dfa.finalizers(representative)),
                filtered(self.dfa.possible_future_group_ids(representative)),
            );
        }
        Some(Tokenizer {
            dfa,
            num_terminals: self.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        })
    }

    /// Verify that `source_to_self` is a raw tokenizer homomorphism.
    ///
    /// Whole-vocabulary token equivalence is not sufficient here: callers use
    /// the target tokenizer as the byte-transition coordinate while building a
    /// partition-local terminal DWA. Every mapped source state must therefore
    /// preserve labels, epsilon successors, and every byte successor exactly.
    fn certifies_mapped_prefix_homomorphism_from(
        &self,
        source: &Tokenizer,
        source_to_self: &[u32],
        source_to_rebuilt: &[u32],
    ) -> bool {
        if source.num_terminals != self.num_terminals
            || source_to_self.len() != source.num_states() as usize
            || source_to_rebuilt.len() != source.num_states() as usize
            || source_to_self
                .iter()
                .any(|&state| state as usize >= self.num_states() as usize)
            || source_to_self.get(source.start_state() as usize).copied()
                != Some(self.start_state())
        {
            return false;
        }

        let mut mapped_epsilon = Vec::<u32>::new();
        let mut target_epsilon = Vec::<u32>::new();
        for source_state in 0..source.num_states() {
            // States absent from the rebuilt prefix were appended below by
            // copying their exact metadata and transition rows through the
            // completed source-to-self map. Only states represented by the
            // synthesized prefix can violate the raw homomorphism.
            if source_to_rebuilt[source_state as usize] == u32::MAX {
                continue;
            }
            let target_state = source_to_self[source_state as usize];
            if source.dfa.finalizers(source_state) != self.dfa.finalizers(target_state)
                || source.dfa.possible_future_group_ids(source_state)
                    != self.dfa.possible_future_group_ids(target_state)
            {
                return false;
            }

            mapped_epsilon.clear();
            mapped_epsilon.extend(
                source.dfa.states()[source_state as usize]
                    .epsilon_transitions
                    .iter()
                    .map(|&next| source_to_self[next as usize]),
            );
            mapped_epsilon.sort_unstable();
            mapped_epsilon.dedup();
            target_epsilon.clear();
            target_epsilon.extend_from_slice(
                &self.dfa.states()[target_state as usize].epsilon_transitions,
            );
            target_epsilon.sort_unstable();
            target_epsilon.dedup();
            if mapped_epsilon != target_epsilon {
                return false;
            }

            let mut source_transitions = source.transitions_from(source_state);
            let mut target_transitions = self.transitions_from(target_state);
            loop {
                match (source_transitions.next(), target_transitions.next()) {
                    (None, None) => break,
                    (
                        Some((source_byte, source_next)),
                        Some((target_byte, target_next)),
                    ) if source_byte == target_byte
                        && source_to_self[source_next as usize] == target_next => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Extend `self` with the source-only residual states that were appended to
    /// `source` after `rebuilt` was constructed.  `rebuilt_to_self` must be a
    /// structural state map from the rebuilt expression DFA into `self`.
    ///
    /// Protected residual synthesis appends externally-entered product states
    /// to otherwise identical deterministic dispatch components.  The original
    /// component states remain an exact prefix.  Verify that prefix relation
    /// state-for-state, then clone only the appended states while redirecting
    /// every edge through the completed source-to-self map.  The result is a
    /// transition homomorphism over the actual source tokenizer, not a bounded
    /// semantic approximation.
    pub fn augment_from_verified_component_prefixes(
        &mut self,
        source: &Tokenizer,
        rebuilt: &Tokenizer,
        rebuilt_to_self: &[u32],
    ) -> Option<Vec<u32>> {
        if source.num_terminals != rebuilt.num_terminals
            || source.num_terminals != self.num_terminals
            || rebuilt_to_self.len() != rebuilt.num_states() as usize
        {
            return None;
        }

        let source_components = source.disjoint_dispatch_components()?;
        let rebuilt_components = rebuilt.disjoint_dispatch_components()?;
        if source_components.len() != rebuilt_components.len() {
            return None;
        }

        let mut source_to_rebuilt = vec![u32::MAX; source.num_states() as usize];
        source_to_rebuilt[source.start_state() as usize] = rebuilt.start_state();
        for (source_states, rebuilt_states) in
            source_components.iter().zip(&rebuilt_components)
        {
            if rebuilt_states.len() > source_states.len() {
                return None;
            }
            for (&source_state, &rebuilt_state) in source_states.iter().zip(rebuilt_states) {
                source_to_rebuilt[source_state as usize] = rebuilt_state;
            }
        }

        // Verify that the mapped prefix is exactly the rebuilt DFA after state
        // renumbering.  This guards the append-only invariant rather than
        // relying on component construction order as an undocumented fact.
        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state == u32::MAX {
                continue;
            }
            let source_state = source_state as u32;
            if source.dfa.finalizers(source_state) != rebuilt.dfa.finalizers(rebuilt_state)
                || source.dfa.possible_future_group_ids(source_state)
                    != rebuilt.dfa.possible_future_group_ids(rebuilt_state)
                || source.state_has_epsilon_transitions(source_state)
                    != rebuilt.state_has_epsilon_transitions(rebuilt_state)
            {
                return None;
            }
            let source_epsilon = &source.dfa.states()[source_state as usize].epsilon_transitions;
            let rebuilt_epsilon = &rebuilt.dfa.states()[rebuilt_state as usize].epsilon_transitions;
            let mapped_epsilon = source_epsilon
                .iter()
                .map(|&target| *source_to_rebuilt.get(target as usize).unwrap_or(&u32::MAX))
                .collect::<Vec<_>>();
            if mapped_epsilon != *rebuilt_epsilon {
                return None;
            }
            let source_transitions = source
                .transitions_from(source_state)
                .map(|(byte, target)| {
                    Some((byte, *source_to_rebuilt.get(target as usize)?))
                })
                .collect::<Option<Vec<_>>>()?;
            if source_transitions.iter().any(|&(_, target)| target == u32::MAX)
                || source_transitions
                    != rebuilt.transitions_from(rebuilt_state).collect::<Vec<_>>()
            {
                return None;
            }
        }

        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state == u32::MAX
                && source.state_has_epsilon_transitions(source_state as u32)
            {
                return None;
            }
        }

        self.invalidate_derived_caches();
        let original_self_states = self.num_states() as usize;
        let mut source_to_self = vec![u32::MAX; source.num_states() as usize];
        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state != u32::MAX {
                source_to_self[source_state] = *rebuilt_to_self.get(rebuilt_state as usize)?;
            }
        }
        for source_state in 0..source.num_states() as usize {
            if source_to_self[source_state] == u32::MAX {
                source_to_self[source_state] = self.dfa.add_state();
            }
        }

        for source_state in 0..source.num_states() as usize {
            if source_to_rebuilt[source_state] != u32::MAX {
                continue;
            }
            let target_state = source_to_self[source_state];
            let source_state_u32 = source_state as u32;
            let transitions = source
                .transitions_from(source_state_u32)
                .map(|(byte, target)| (byte, source_to_self[target as usize]))
                .collect::<Vec<_>>();
            self.dfa
                .set_transitions_from_sorted_entries(target_state, transitions);
            self.dfa.overwrite_state_metadata(
                target_state,
                source.dfa.finalizers(source_state_u32).clone(),
                source
                    .dfa
                    .possible_future_group_ids(source_state_u32)
                    .clone(),
            );
        }
        debug_assert_eq!(
            self.num_states() as usize - original_self_states,
            source_to_rebuilt
                .iter()
                .filter(|&&state| state == u32::MAX)
                .count(),
        );
        if !self.certifies_mapped_prefix_homomorphism_from(
            source,
            &source_to_self,
            &source_to_rebuilt,
        ) {
            return None;
        }
        Some(source_to_self)
    }

    pub(super) fn from_parts(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
    ) -> Self {
        Self {
            dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        }
    }

    pub fn from_parts_with_compressed_transitions(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    ) -> Self {
        debug_assert!(compressed_transition_segments
            .windows(2)
            .all(|pair| pair[0].state_offset + pair[0].state_count <= pair[1].state_offset));
        Self {
            dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(compressed_transition_segments),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            exprs,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
        }
    }

    fn compressed_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&CompressedTransitionSegment> {
        let index = self
            .compressed_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.compressed_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    fn packed_compressed_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&PackedCompressedTransitionSegment> {
        let index = self
            .packed_compressed_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.packed_compressed_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    fn packed_runtime_transition_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&PackedRuntimeTransitionSegment> {
        let index = self
            .packed_runtime_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.packed_runtime_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    fn packed_runtime_metadata_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&PackedTokenizerMetadataSegment> {
        let index = self
            .packed_runtime_metadata_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.packed_runtime_metadata_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    pub fn has_packed_runtime_metadata(&self) -> bool {
        self.packed_runtime_metadata.is_some() || !self.packed_runtime_metadata_segments.is_empty()
    }

    pub fn has_compressed_transition_state(&self, state: u32) -> bool {
        self.compressed_segment_for_state(state).is_some()
            || self.packed_compressed_segment_for_state(state).is_some()
    }

    #[inline]
    pub fn has_packed_runtime_transitions(&self) -> bool {
        self.packed_runtime_transitions.is_some()
            || !self.packed_runtime_transition_segments.is_empty()
    }

    /// Move packed observation/epsilon metadata back into the structural DFA
    /// without expanding packed byte-transition rows. This is needed before a
    /// tokenizer is structurally mutated: packed metadata is otherwise the
    /// authoritative source for epsilon closure and would shadow newly added
    /// DFA epsilon edges.
    fn materialize_runtime_metadata_for_structural_mutation(&mut self) {
        if let Some(metadata) = self.packed_runtime_metadata.take() {
            for state in 0..metadata.state_count {
                let finalizers = metadata
                    .finalizers(state)
                    .expect("packed tokenizer metadata covers every state")
                    .clone();
                let futures = metadata
                    .futures(state)
                    .expect("packed tokenizer metadata covers every state")
                    .clone();
                self.dfa.overwrite_state_metadata(state, finalizers, futures);
                for &target in metadata.epsilon_targets(state) {
                    let already_present = self
                        .dfa
                        .states()
                        .get(state as usize)
                        .is_some_and(|row| row.epsilon_transitions.contains(&target));
                    if !already_present {
                        self.dfa.add_epsilon_transition(state, target);
                    }
                }
            }
        }

        let segments = std::mem::take(&mut self.packed_runtime_metadata_segments);
        for segment in segments.iter() {
            for local_state in 0..segment.metadata.state_count {
                let state = segment.state_offset + local_state;
                let finalizers = segment
                    .metadata
                    .finalizers(local_state)
                    .expect("packed tokenizer metadata segment covers every state")
                    .clone();
                let futures = segment
                    .metadata
                    .futures(local_state)
                    .expect("packed tokenizer metadata segment covers every state")
                    .clone();
                self.dfa.overwrite_state_metadata(state, finalizers, futures);
                for &local_target in segment.metadata.epsilon_targets(local_state) {
                    let target = segment
                        .state_offset
                        .checked_add(local_target)
                        .expect("packed tokenizer epsilon target overflow");
                    let already_present = self
                        .dfa
                        .states()
                        .get(state as usize)
                        .is_some_and(|row| row.epsilon_transitions.contains(&target));
                    if !already_present {
                        self.dfa.add_epsilon_transition(state, target);
                    }
                }
            }
        }
    }

    fn materialized_dfa(&self) -> DFA {
        let mut dfa = self.dfa.clone();
        if let Some(packed) = &self.packed_runtime_transitions {
            for state in 0..packed.state_count() as u32 {
                if let Some((bytes, targets)) = packed.row(state) {
                    dfa.set_transitions_from_sorted_entries(
                        state,
                        bytes
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(index, byte)| {
                                (
                                    byte,
                                    targets
                                        .get(index)
                                        .expect("packed tokenizer row lengths were validated"),
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        for segment in self.packed_runtime_transition_segments.iter() {
            for local_state in 0..segment.transitions.state_count() as u32 {
                let state = segment.state_offset + local_state;
                if let Some((bytes, targets)) = segment.transitions.row(local_state) {
                    dfa.set_transitions_from_sorted_entries(
                        state,
                        bytes
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(index, byte)| {
                                (
                                    byte,
                                    segment.state_offset
                                        + targets
                                            .get(index)
                                            .expect("packed tokenizer row lengths were validated"),
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        for segment in self.compressed_transition_segments.iter() {
            for local_state in 0..segment.state_count {
                let state = segment.state_offset + local_state;
                dfa.set_transitions_from_sorted_entries(state, segment.expanded_entries(state));
            }
        }
        for segment in self.packed_compressed_transition_segments.iter() {
            for local_state in 0..segment.state_count {
                let state = segment.state_offset + local_state;
                let mut row = [u32::MAX; 256];
                segment.fill_transition_row(state, &mut row);
                dfa.set_transitions_from_sorted_entries(
                    state,
                    row.iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(byte, target)| {
                            (target != u32::MAX).then_some((byte as u8, target))
                        })
                        .collect(),
                );
            }
        }
        dfa
    }

    /// Put two tokenizers with the same terminal-id domain under one fresh
    /// epsilon root without identifying any source states. This lets the exact
    /// state-equivalence machinery compare residual states across independently
    /// built full and synthesized lexers.
    pub fn disjoint_union_for_analysis(
        left: &Tokenizer,
        right: &Tokenizer,
    ) -> TokenizerAnalysisUnion {
        assert_eq!(
            left.num_terminals, right.num_terminals,
            "cross-tokenizer analysis requires one shared terminal-id domain",
        );

        let left_offset = 1u32;
        let right_offset = left_offset + left.dfa.num_states() as u32;
        let mut dfa = DFA::new(
            1usize
                .saturating_add(left.dfa.num_states())
                .saturating_add(right.dfa.num_states()),
        );
        let num_groups = left.num_terminals as usize;
        dfa.ensure_group_capacity(num_groups);

        for group in 0..num_groups {
            let left_set = *left.dfa.group_id_to_u8set(group as u32);
            let right_set = *right.dfa.group_id_to_u8set(group as u32);
            dfa.set_group_u8set(group as u32, left_set.union(&right_set));
        }

        let copy_source = |target: &mut DFA, source: &DFA, offset: u32| {
            for (state_index, state) in source.states().iter().enumerate() {
                let target_state = offset + state_index as u32;
                target.set_transitions_from_sorted_entries(
                    target_state,
                    state
                        .transitions
                        .iter()
                        .map(|(byte, &destination)| (byte, offset + destination))
                        .collect(),
                );
                for &destination in &state.epsilon_transitions {
                    target.add_epsilon_transition(target_state, offset + destination);
                }
                target.overwrite_state_metadata(
                    target_state,
                    state.finalizers.clone(),
                    source
                        .possible_future_group_ids(state_index as u32)
                        .clone(),
                );
            }
        };
        copy_source(&mut dfa, &left.dfa, left_offset);
        copy_source(&mut dfa, &right.dfa, right_offset);
        dfa.add_epsilon_transition(0, left_offset + left.start_state());
        dfa.add_epsilon_transition(0, right_offset + right.start_state());

        let mut root_futures = BitSet::new(num_groups);
        for terminal in left
            .possible_future_terminals_iter(left.start_state())
            .chain(right.possible_future_terminals_iter(right.start_state()))
        {
            root_futures.set(terminal as usize);
        }
        dfa.overwrite_state_metadata(0, BitSet::new(num_groups), root_futures);

        TokenizerAnalysisUnion {
            tokenizer: Tokenizer::from_parts(dfa, left.num_terminals, None),
            left_offset,
            right_offset,
        }
    }

    fn start_state(&self) -> u32 {
        0
    }

    fn num_terminals(&self) -> u32 {
        self.num_terminals
    }

    pub fn has_epsilon_transitions(&self) -> bool {
        self.packed_runtime_metadata
            .as_deref()
            .is_some_and(PackedTokenizerMetadata::has_epsilon_transitions)
            || self
                .packed_runtime_metadata_segments
                .iter()
                .any(|segment| segment.metadata.has_epsilon_transitions())
            || self.dfa.has_epsilon_transitions()
    }

    #[inline]
    pub fn state_has_epsilon_transitions(&self, state: u32) -> bool {
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return !metadata.epsilon_targets(state).is_empty();
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return !segment
                .metadata
                .epsilon_targets(segment.local_state(state))
                .is_empty();
        }
        self.dfa
            .states()
            .get(state as usize)
            .is_some_and(|state| !state.epsilon_transitions.is_empty())
    }

    #[inline]
    fn state_finalizers(&self, state: u32) -> &BitSet {
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return metadata
                .finalizers(state)
                .expect("packed tokenizer finalizer row must cover every state");
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return segment
                .metadata
                .finalizers(segment.local_state(state))
                .expect("packed tokenizer metadata segment must cover every state");
        }
        self.dfa.finalizers(state)
    }

    #[inline]
    fn state_futures(&self, state: u32) -> &BitSet {
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return metadata
                .futures(state)
                .expect("packed tokenizer future row must cover every state");
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return segment
                .metadata
                .futures(segment.local_state(state))
                .expect("packed tokenizer metadata segment must cover every state");
        }
        self.dfa.possible_future_group_ids(state)
    }

    fn epsilon_closure_states(&self, roots: &[u32]) -> SmallVec<[u32; 1]> {
        if self.packed_runtime_metadata.is_none() && self.packed_runtime_metadata_segments.is_empty() {
            return self.dfa.epsilon_closure(roots);
        }
        if roots.iter().all(|&state| !self.state_has_epsilon_transitions(state)) {
            let mut result = SmallVec::from_slice(roots);
            result.sort_unstable();
            result.dedup();
            return result;
        }
        let mut result = SmallVec::<[u32; 1]>::new();
        let mut stack = SmallVec::<[u32; 8]>::from_slice(roots);
        while let Some(state) = stack.pop() {
            if result.contains(&state) {
                continue;
            }
            result.push(state);
            if let Some(metadata) = self
                .packed_runtime_metadata
                .as_deref()
                .filter(|metadata| state < metadata.state_count)
            {
                stack.extend_from_slice(metadata.epsilon_targets(state));
            } else if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
                stack.extend(
                    segment
                        .metadata
                        .epsilon_targets(segment.local_state(state))
                        .iter()
                        .map(|target| segment.state_offset + *target),
                );
            } else if let Some(dfa_state) = self.dfa.states().get(state as usize) {
                stack.extend_from_slice(&dfa_state.epsilon_transitions);
            }
        }
        result.sort_unstable();
        result
    }

    pub fn terminal_expr(&self, terminal: TerminalID) -> Option<&Expr> {
        self.exprs.as_deref()?.get(terminal as usize)
    }

    /// Compile-time terminal expressions, when retained by the artifact.
    pub fn terminal_exprs(&self) -> Option<&[Expr]> {
        self.exprs.as_deref()
    }

    /// Restore terminal expressions carried by a versioned outer artifact.
    /// Invalid-length metadata is rejected rather than silently associating
    /// expressions with the wrong terminal IDs.
    pub fn restore_terminal_exprs(&mut self, exprs: Option<Vec<Expr>>) -> Result<(), String> {
        let Some(exprs) = exprs else {
            self.exprs = None;
            return Ok(());
        };
        if exprs.len() != self.num_terminals as usize {
            return Err(format!(
                "serialized tokenizer has {} terminal expressions for {} terminals",
                exprs.len(), self.num_terminals,
            ));
        }
        self.exprs = Some(Arc::from(exprs.into_boxed_slice()));
        Ok(())
    }

    /// Exact syntactic byte support retained for one terminal.
    ///
    /// This is a cheap necessary condition for terminal-language equality. It
    /// is not itself used as an equivalence proof.
    pub fn terminal_byte_support(&self, terminal: TerminalID) -> Option<U8Set> {
        (terminal < self.num_terminals)
            .then(|| *self.dfa.group_id_to_u8set(terminal))
    }

    #[inline]
    fn state_live_for_terminal(&self, state: u32, terminal: TerminalID) -> bool {
        self.state_finalizers(state).contains(terminal as usize)
            || self.state_futures(state).contains(terminal as usize)
    }

    fn terminal_projected_epsilon_closure(
        &self,
        states: &[u32],
        terminal: TerminalID,
    ) -> Box<[u32]> {
        let mut closure = self.epsilon_closure_states(states);
        closure.retain(|state| self.state_live_for_terminal(*state, terminal));
        closure.sort_unstable();
        closure.dedup();
        closure.into_vec().into_boxed_slice()
    }

    fn terminal_projected_subset_accepting(
        &self,
        states: &[u32],
        terminal: TerminalID,
    ) -> bool {
        states
            .iter()
            .any(|&state| self.state_finalizers(state).contains(terminal as usize))
    }

    /// Return the unique scalar deterministic reset branch containing this
    /// terminal, when the tokenizer's structural certificate proves such a
    /// branch exists. This avoids powerset construction for the common
    /// partitioned-lexer representation.
    fn terminal_scalar_dispatch_root(&self, terminal: TerminalID) -> Option<u32> {
        let start = self.start_state();
        if self.state_finalizers(start).contains(terminal as usize) {
            // A nullable terminal can accept before entering a dispatch root;
            // keep it on the general epsilon-NFA proof path.
            return None;
        }
        if !self.has_epsilon_transitions() {
            return Some(start);
        }
        if !self.has_scalar_deterministic_dispatch() {
            return None;
        }
        let mut live_roots = self
            .deterministic_dispatch_roots()?
            .iter()
            .copied()
            .filter(|&root| self.state_live_for_terminal(root, terminal));
        let root = live_roots.next()?;
        live_roots.next().is_none().then_some(root)
    }

    #[inline]
    fn terminal_projected_scalar_step(
        &self,
        state: u32,
        terminal: TerminalID,
        byte: u8,
    ) -> Option<u32> {
        self.step(state, byte)
            .filter(|&target| self.state_live_for_terminal(target, terminal))
    }

    fn terminal_scalar_prefix_fingerprint_from_state(
        &self,
        state: u32,
        terminal: TerminalID,
        depth: u8,
        memo: &mut FxHashMap<(u32, u8), u64>,
    ) -> u64 {
        if let Some(&fingerprint) = memo.get(&(state, depth)) {
            return fingerprint;
        }
        // Fixed deterministic mixer. Hash collisions merely admit an extra
        // exact equivalence check; this fingerprint is never a proof.
        #[inline]
        fn mix(mut hash: u64, value: u64) -> u64 {
            hash ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
            hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            hash ^ (hash >> 29)
        }

        let accepting = self.state_finalizers(state).contains(terminal as usize);
        let mut hash = if accepting {
            0x6a09_e667_f3bc_c909
        } else {
            0xbb67_ae85_84ca_a73b
        };
        if depth > 0 {
            let mut transitions = self
                .transitions_from(state)
                .filter_map(|(byte, target)| {
                    self.state_live_for_terminal(target, terminal)
                        .then_some((byte, target))
                })
                .collect::<Vec<_>>();
            transitions.sort_unstable_by_key(|&(byte, _)| byte);
            for (byte, target) in transitions {
                let child = self.terminal_scalar_prefix_fingerprint_from_state(
                    target,
                    terminal,
                    depth - 1,
                    memo,
                );
                hash = mix(hash, byte as u64 + 1);
                hash = mix(hash, child);
            }
            hash = mix(hash, 0x100 + depth as u64);
        }
        memo.insert((state, depth), hash);
        hash
    }

    /// A bounded language-observation fingerprint for candidate indexing.
    ///
    /// For scalar deterministic terminal branches this recursively records the
    /// accepting bit and every live byte derivative to `depth`. Equal terminal
    /// languages necessarily produce the same value. Hash collisions or a
    /// shallow horizon only create extra exact checks; equality of the value is
    /// never used as an equivalence certificate.
    pub fn terminal_scalar_prefix_fingerprint(
        &self,
        terminal: TerminalID,
        depth: u8,
    ) -> Option<u64> {
        let root = self.terminal_scalar_dispatch_root(terminal)?;
        let mut memo = FxHashMap::default();
        Some(self.terminal_scalar_prefix_fingerprint_from_state(
            root,
            terminal,
            depth,
            &mut memo,
        ))
    }

    /// Canonical rooted-graph certificate for one scalar terminal DFA.
    ///
    /// States are renumbered by deterministic BFS from the terminal's unique
    /// dispatch root, following live transitions in byte order. The returned
    /// sparse serialization records each state's accepting bit and every
    /// byte-labelled edge to the canonical target ID. Equality of two returned
    /// vectors is therefore an exact rooted labelled-DFA isomorphism proof and
    /// implies terminal-language equality. Non-isomorphic but language-
    /// equivalent DFAs may produce different certificates; this is an
    /// intentionally sufficient, not complete, proof.
    pub fn terminal_scalar_structural_certificate(
        &self,
        terminal: TerminalID,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<Vec<u64>> {
        if state_limit == 0 || transition_limit == 0 {
            return None;
        }
        let root = self.terminal_scalar_dispatch_root(terminal)?;
        let mut canonical = FxHashMap::<u32, u32>::default();
        canonical.insert(root, 0);
        let mut queue = VecDeque::from([root]);
        let mut encoded = Vec::<u64>::new();
        let mut transition_count = 0usize;

        while let Some(state) = queue.pop_front() {
            if canonical.len() > state_limit {
                return None;
            }
            let accepting = self.state_finalizers(state).contains(terminal as usize);
            let mut transitions = self
                .transitions_from(state)
                .filter_map(|(byte, target)| {
                    self.state_live_for_terminal(target, terminal)
                        .then_some((byte, target))
                })
                .collect::<Vec<_>>();
            transitions.sort_unstable_by_key(|&(byte, _)| byte);
            transition_count = transition_count.saturating_add(transitions.len());
            if transition_count > transition_limit {
                return None;
            }
            encoded.push(
                ((accepting as u64) << 63)
                    | u64::try_from(transitions.len()).ok()?.min((1u64 << 63) - 1),
            );
            for (byte, target) in transitions {
                let next_id = if let Some(&existing) = canonical.get(&target) {
                    existing
                } else {
                    let next = canonical.len() as u32;
                    canonical.insert(target, next);
                    queue.push_back(target);
                    next
                };
                encoded.push(((byte as u64) << 32) | next_id as u64);
            }
        }
        Some(encoded)
    }

    fn terminal_scalar_language_equivalent_bounded(
        &self,
        terminal: TerminalID,
        left_root: u32,
        other: &Tokenizer,
        other_terminal: TerminalID,
        right_root: u32,
        pair_limit: usize,
        transition_work_limit: usize,
    ) -> Option<bool> {
        let mut seen = rustc_hash::FxHashSet::<(Option<u32>, Option<u32>)>::default();
        let mut queue = VecDeque::from([(Some(left_root), Some(right_root))]);
        let mut work = 0usize;

        while let Some(pair @ (left, right)) = queue.pop_front() {
            if !seen.insert(pair) {
                continue;
            }
            if seen.len() > pair_limit {
                return None;
            }
            let left_accepting =
                left.is_some_and(|state| self.state_finalizers(state).contains(terminal as usize));
            let right_accepting = right.is_some_and(|state| {
                other.state_finalizers(state).contains(other_terminal as usize)
            });
            if left_accepting != right_accepting {
                return Some(false);
            }

            let mut bytes = U8Set::empty();
            if let Some(state) = left {
                for (byte, _) in self.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            if let Some(state) = right {
                for (byte, _) in other.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            for byte in bytes.iter() {
                work = work.saturating_add(1);
                if work > transition_work_limit {
                    return None;
                }
                let next = (
                    left.and_then(|state| {
                        self.terminal_projected_scalar_step(state, terminal, byte)
                    }),
                    right.and_then(|state| {
                        other.terminal_projected_scalar_step(state, other_terminal, byte)
                    }),
                );
                if next != (None, None) && !seen.contains(&next) {
                    queue.push_back(next);
                }
            }
        }
        Some(true)
    }

    fn terminal_projected_step(
        &self,
        states: &[u32],
        terminal: TerminalID,
        byte: u8,
        work: &mut usize,
        work_limit: usize,
    ) -> Option<Box<[u32]>> {
        let mut targets = SmallVec::<[u32; 8]>::new();
        for &state in states {
            *work = work.saturating_add(1);
            if *work > work_limit {
                return None;
            }
            if let Some(target) = self.step(state, byte) {
                targets.push(target);
            }
        }
        if targets.is_empty() {
            return Some(Box::new([]));
        }
        targets.sort_unstable();
        targets.dedup();
        Some(self.terminal_projected_epsilon_closure(&targets, terminal))
    }

    /// Prove whether two compiled terminals denote the same byte language.
    ///
    /// This works directly from the serialized tokenizer automata and does not
    /// require the compile-time `Expr` sidecar. The proof is exact: each
    /// epsilon-NFA is projected to the chosen terminal, then the symmetric
    /// difference of their on-the-fly subset constructions is searched by BFS.
    /// `Some(true)` means equivalence was proved; `Some(false)` means a concrete
    /// distinguishing byte word exists. `None` means the supplied resource
    /// bounds were exhausted, in which case callers must conservatively decline
    /// any optimization that requires equivalence.
    ///
    /// `possible_future_group_ids` is used only to remove states from which the
    /// selected terminal can neither finalize now nor in the future. By its
    /// tokenizer invariant such states cannot contribute to the selected
    /// terminal's language.
    pub fn terminal_language_equivalent_bounded(
        &self,
        terminal: TerminalID,
        other: &Tokenizer,
        other_terminal: TerminalID,
        pair_limit: usize,
        transition_work_limit: usize,
    ) -> Option<bool> {
        if terminal >= self.num_terminals
            || other_terminal >= other.num_terminals
            || pair_limit == 0
            || transition_work_limit == 0
        {
            return None;
        }

        // Equal languages necessarily consume the same set of byte values.
        // This metadata is a cheap rejection filter only; equality here is not
        // treated as a proof.
        if self.terminal_byte_support(terminal)? != other.terminal_byte_support(other_terminal)? {
            return Some(false);
        }

        if let (Some(left_root), Some(right_root)) = (
            self.terminal_scalar_dispatch_root(terminal),
            other.terminal_scalar_dispatch_root(other_terminal),
        ) {
            return self.terminal_scalar_language_equivalent_bounded(
                terminal,
                left_root,
                other,
                other_terminal,
                right_root,
                pair_limit,
                transition_work_limit,
            );
        }

        let left_start =
            self.terminal_projected_epsilon_closure(&[self.start_state()], terminal);
        let right_start = other
            .terminal_projected_epsilon_closure(&[other.start_state()], other_terminal);

        type SubsetPair = (Box<[u32]>, Box<[u32]>);
        let mut seen = rustc_hash::FxHashSet::<SubsetPair>::default();
        let mut queue = VecDeque::<SubsetPair>::from([(left_start, right_start)]);
        let mut work = 0usize;

        while let Some((left, right)) = queue.pop_front() {
            if !seen.insert((left.clone(), right.clone())) {
                continue;
            }
            if seen.len() > pair_limit {
                return None;
            }
            if self.terminal_projected_subset_accepting(&left, terminal)
                != other.terminal_projected_subset_accepting(&right, other_terminal)
            {
                return Some(false);
            }

            // Explore exactly the bytes that have an outgoing transition from
            // either current subset. This is equivalent to scanning all 256
            // bytes while avoiding dead/dead product edges.
            let mut bytes = U8Set::empty();
            for &state in left.iter() {
                for (byte, _) in self.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            for &state in right.iter() {
                for (byte, _) in other.transitions_from(state) {
                    bytes.insert(byte);
                }
            }

            for byte in bytes.iter() {
                let next_left = self.terminal_projected_step(
                    &left,
                    terminal,
                    byte,
                    &mut work,
                    transition_work_limit,
                )?;
                let next_right = other.terminal_projected_step(
                    &right,
                    other_terminal,
                    byte,
                    &mut work,
                    transition_work_limit,
                )?;
                if next_left.is_empty() && next_right.is_empty() {
                    continue;
                }
                if !seen.contains(&(next_left.clone(), next_right.clone())) {
                    queue.push_back((next_left, next_right));
                }
            }
        }
        Some(true)
    }

    pub fn initial_epsilon_branch_count(&self) -> usize {
        self.dfa
            .states()
            .get(self.start_state() as usize)
            .map_or(0, |state| state.epsilon_transitions.len())
    }

    /// Return the deterministic scanner roots behind the special epsilon
    /// dispatch state produced by `build_regex_partitioned`.
    ///
    /// This is deliberately narrower than "has epsilon transitions".  The
    /// compiler can retain its scalar-state fast paths when the only live
    /// nondeterminism is a zero-byte fan-out from the global reset state into
    /// independently deterministic components.  Nullable-start isolation may
    /// leave an unreachable cloned dispatch state elsewhere in the DFA, so the
    /// predicate is based on the live reset shape rather than a whole-DFA scan.
    pub fn deterministic_dispatch_roots(&self) -> Option<&[u32]> {
        let start = self.dfa.states().get(self.start_state() as usize)?;
        if start.epsilon_transitions.len() < 2
            || self.transitions_from(self.start_state()).next().is_some()
        {
            return None;
        }
        if start
            .epsilon_transitions
            .iter()
            .any(|&root| self.state_has_epsilon_transitions(root))
        {
            return None;
        }
        Some(&start.epsilon_transitions)
    }

    #[inline]
    pub fn has_deterministic_dispatch(&self) -> bool {
        self.deterministic_dispatch_roots().is_some()
    }

    /// Return whether selecting one reset-dispatch root leaves a genuinely
    /// scalar scanner for the rest of the byte stream.
    ///
    /// `deterministic_dispatch_roots()` only certifies the shape at reset: a
    /// zero-byte fan-out whose immediate roots have no epsilon edges.  A
    /// depth-limited adaptive product can have that same reset shape while
    /// introducing epsilon fan-out later, at its determinization frontier.
    /// Whole-token flat-transition walkers are sound only under the stronger
    /// condition checked here: no state byte-reachable from a dispatch root has
    /// an epsilon transition.
    pub fn has_scalar_deterministic_dispatch(&self) -> bool {
        *self.scalar_deterministic_dispatch_cache.get_or_init(|| {
            let Some(roots) = self.deterministic_dispatch_roots() else {
                return false;
            };
            let states = self.dfa.states();
            let mut seen = vec![false; states.len()];
            let mut pending = roots.to_vec();
            while let Some(state) = pending.pop() {
                let Some(slot) = seen.get_mut(state as usize) else {
                    return false;
                };
                if *slot {
                    continue;
                }
                *slot = true;
                let Some(dfa_state) = states.get(state as usize) else {
                    return false;
                };
                if !dfa_state.epsilon_transitions.is_empty() {
                    return false;
                }
                pending.extend(dfa_state.transitions.iter().map(|(_, &target)| target));
            }
            true
        })
    }

    /// Return the closed, pairwise-disjoint state sets below the global
    /// epsilon dispatcher. Components may contain internal epsilon structure,
    /// but no byte or epsilon edge may cross between returned sets.
    pub fn disjoint_dispatch_components(&self) -> Option<Vec<Vec<u32>>> {
        let roots = self.deterministic_dispatch_roots()?;
        let mut owner = vec![usize::MAX; self.dfa.states().len()];
        owner[self.start_state() as usize] = roots.len();
        let mut components = Vec::with_capacity(roots.len());

        for (component_index, &root) in roots.iter().enumerate() {
            if owner.get(root as usize).copied().unwrap_or(roots.len()) != usize::MAX {
                return None;
            }
            let mut states = Vec::new();
            let mut stack = vec![root];
            while let Some(state) = stack.pop() {
                let slot = owner.get_mut(state as usize)?;
                if *slot == component_index {
                    continue;
                }
                if *slot != usize::MAX {
                    return None;
                }
                *slot = component_index;
                states.push(state);
                let dfa_state = self.dfa.states().get(state as usize)?;
                stack.extend(dfa_state.transitions.iter().map(|(_, &target)| target));
                stack.extend(dfa_state.epsilon_transitions.iter().copied());
            }
            if states.is_empty() {
                return None;
            }
            states.sort_unstable();
            components.push(states);
        }
        Some(components)
    }

    /// Scanner states to use after a terminal boundary.  A conventional DFA
    /// has one reset state.  A partitioned lexer has one deterministic reset
    /// state per component; keeping them separate avoids materializing their
    /// product while preserving cross-component terminal sequences inside one
    /// vocabulary token.
    pub fn deterministic_reset_states(&self) -> TokenizerStateSet {
        self.deterministic_dispatch_roots()
            .map(TokenizerStateSet::from_slice)
            .unwrap_or_else(|| TokenizerStateSet::from_buf([self.initial_state_id()]))
    }

    fn transitions_from(&self, state: u32) -> TokenizerTransitionsIter<'_> {
        if let Some(packed) = &self.packed_runtime_transitions {
            if let Some((bytes, targets)) = packed.row(state) {
                return TokenizerTransitionsIter {
                    inner: TokenizerTransitionsIterInner::Packed {
                        bytes,
                        targets,
                        next: 0,
                    },
                };
            }
        }
        if let Some(segment) = self.packed_runtime_transition_segment_for_state(state)
            && let Some((bytes, targets)) = segment.row(state)
        {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::PackedSegment {
                    bytes,
                    targets,
                    target_offset: segment.state_offset,
                    next: 0,
                },
            };
        }
        if let Some(segment) = self.compressed_segment_for_state(state) {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::Compressed {
                    segment,
                    state,
                    next_byte: 0,
                },
            };
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::PackedCompressed {
                    segment,
                    state,
                    next_byte: 0,
                },
            };
        }
        TokenizerTransitionsIter {
            inner: self
                .dfa
                .states()
                .get(state as usize)
                .map_or(TokenizerTransitionsIterInner::Empty, |state| {
                    TokenizerTransitionsIterInner::Dense(state.transitions.iter())
                }),
        }
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            segment.fill_transition_row(state, row);
            return;
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            segment.fill_transition_row(state, row);
            return;
        }
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            let mut bytes = U8Set::empty();
            let local_state = state - segment.state_offset;
            let start = segment.row_offsets[local_state as usize] as usize;
            let end = segment.row_offsets[local_state as usize + 1] as usize;
            for (class, target) in segment.entries.iter_range(start, end) {
                if target == local_state {
                    for &byte in segment.class_members[class as usize].iter() {
                        bytes.insert(byte);
                    }
                }
            }
            return bytes;
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            return segment.self_loop_bytes(state);
        }
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    /// Optimized exact finite-horizon observation certificate for one scalar
    /// tokenizer state. This shadows the generic [`Lexer`] default for the
    /// concrete runtime tokenizer so compressed transition segments can be
    /// consumed in their native byte-class form.
    pub fn bounded_observation_safe_horizon_from_state(
        &self,
        source: u32,
        bytes: U8Set,
        active_terminals: &BitSet,
        max_horizon: u8,
    ) -> u8 {
        self.bounded_observation_safe_horizon_with_witnesses(
            source,
            bytes,
            active_terminals,
            max_horizon,
        )
        .0
    }

    /// Precompute one canonical full-observation-stable byte alphabet per raw
    /// tokenizer state for the two runtime horizons used by dynamic masking.
    ///
    /// `safe_h[state]` is a conservative alphabet B such that every B-string
    /// of length at most H keeps both the complete finalizer set and complete
    /// possible-future-terminal set equal to their values at `state`.
    ///
    /// A single alphabet cannot represent every safe subset exactly (safe
    /// alphabets are not union-closed), so this computes one deterministic
    /// closed alphabet by refinement.  Round 1 retains all observation-
    /// preserving byte transitions.  Each later round intersects that set with
    /// the previous-round safe alphabet of every destination reachable by a
    /// currently retained byte.  The result is therefore sound for arbitrary
    /// mixed byte sequences drawn from the returned set.
    pub fn precompute_bounded_observation_safe_byte_sets(
        &self,
    ) -> (Box<[U8Set]>, Box<[U8Set]>) {
        const DEAD: u32 = u32::MAX;
        let state_count = self.num_states() as usize;
        if state_count == 0 {
            return (Box::new([]), Box::new([]));
        }

        // Literal self-loops are safe for every horizon and remain the fallback
        // when a finite advancing family shrinks away before H.
        let mut horizon16 = (0..state_count)
            .into_par_iter()
            .map(|state| self.self_loop_bytes(state as u32))
            .collect::<Vec<_>>();
        let mut horizon64 = horizon16.clone();

        // Pick one canonical one-byte continuation family at every raw state:
        // the largest set of bytes that all go to the same target while keeping
        // the complete lexer observation unchanged. The target relation is
        // therefore functional for every byte in the selected family.
        let mut selected_sets = vec![U8Set::empty(); state_count];
        let mut selected_targets = vec![DEAD; state_count];
        let mut compressed = vec![false; state_count];
        for segment in self.compressed_transition_segments.iter() {
            let start = segment.state_offset as usize;
            let end = start + segment.state_count as usize;
            compressed[start..end].fill(true);
        }

        // Ordinary states are relatively few. Group their explicit byte edges
        // by raw target directly.
        for state in 0..state_count {
            if compressed[state] || self.state_has_epsilon_transitions(state as u32) {
                continue;
            }
            let mut groups = SmallVec::<[(u32, U8Set); 8]>::new();
            for (byte, target) in self.transitions_from(state as u32) {
                if let Some((_, bytes)) = groups.iter_mut().find(|(seen, _)| *seen == target) {
                    bytes.insert(byte);
                } else {
                    let mut bytes = U8Set::empty();
                    bytes.insert(byte);
                    groups.push((target, bytes));
                }
            }
            let source_finalizers = self.matched_terminal_bitset(state as u32);
            let source_futures = self.possible_future_terminals(state as u32);
            let mut best = U8Set::empty();
            let mut best_target = DEAD;
            for (target, candidate) in groups {
                if self.state_has_epsilon_transitions(target)
                    || self.matched_terminal_bitset(target) != source_finalizers
                    || self.possible_future_terminals(target) != source_futures
                {
                    continue;
                }
                if candidate.len() > best.len()
                    || (candidate.len() == best.len() && target < best_target)
                {
                    best = candidate;
                    best_target = target;
                }
            }
            selected_sets[state] = best;
            selected_targets[state] = best_target;
        }

        // Large synthesized bounded-repeat products use compressed transition
        // segments. Collapse all tokenizer byte classes sharing a raw target
        // without expanding every byte transition for every million-state row.
        for segment in self.compressed_transition_segments.iter() {
            let segment_states = segment.state_count as usize;
            if segment_states == 0 {
                continue;
            }
            let class_sets = segment
                .class_members
                .iter()
                .map(|members| U8Set::from_bytes(members))
                .collect::<Vec<_>>();

            // Scratch indexed by local target avoids one hash table allocation
            // per row. `target_epoch` lazily clears `target_sets`.
            let mut target_sets = vec![U8Set::empty(); segment_states];
            let mut target_epoch = vec![0u32; segment_states];
            let mut epoch = 0u32;
            let mut touched = SmallVec::<[u32; 16]>::new();

            for local_state in 0..segment_states {
                let source_global = segment.state_offset + local_state as u32;
                if self.state_has_epsilon_transitions(source_global) {
                    continue;
                }
                epoch = epoch.wrapping_add(1);
                if epoch == 0 {
                    target_epoch.fill(0);
                    epoch = 1;
                }
                touched.clear();
                let row_start = segment.row_offsets[local_state] as usize;
                let row_end = segment.row_offsets[local_state + 1] as usize;
                for entry in row_start..row_end {
                    let class = segment.entries.classes[entry] as usize;
                    let target = segment.entries.target(entry) as usize;
                    if target_epoch[target] != epoch {
                        target_epoch[target] = epoch;
                        target_sets[target] = U8Set::empty();
                        touched.push(target as u32);
                    }
                    target_sets[target] |= class_sets[class];
                }

                let source_finalizers = self.matched_terminal_bitset(source_global);
                let source_futures = self.possible_future_terminals(source_global);
                let mut best = U8Set::empty();
                let mut best_target = DEAD;
                for &target_local in &touched {
                    let target_global = segment.state_offset + target_local;
                    if self.state_has_epsilon_transitions(target_global)
                        || self.matched_terminal_bitset(target_global) != source_finalizers
                        || self.possible_future_terminals(target_global) != source_futures
                    {
                        continue;
                    }
                    let candidate = target_sets[target_local as usize];
                    if candidate.len() > best.len()
                        || (candidate.len() == best.len() && target_global < best_target)
                    {
                        best = candidate;
                        best_target = target_global;
                    }
                }
                selected_sets[source_global as usize] = best;
                selected_targets[source_global as usize] = best_target;
            }
        }

        // For a chain S --B0--> T --B1--> U ..., any byte alphabet contained
        // in every Bi follows the same raw-state chain and preserves the same
        // full lexer observation at every prefix. Pointer doubling computes the
        // intersection along 2^k edges, so rounds 4 and 6 give exact conservative
        // 16- and 64-byte alphabets for this canonical chain.
        let mut path_sets = selected_sets;
        let mut jump = selected_targets;
        let mut doubled_sets = vec![U8Set::empty(); state_count];
        let mut doubled_jump = vec![DEAD; state_count];

        for round in 1..=6 {
            doubled_sets
                .par_iter_mut()
                .zip(doubled_jump.par_iter_mut())
                .enumerate()
                .for_each(|(state, (set_slot, jump_slot))| {
                    let middle = jump[state];
                    if middle == DEAD {
                        *set_slot = U8Set::empty();
                        *jump_slot = DEAD;
                        return;
                    }
                    let middle = middle as usize;
                    *set_slot = path_sets[state].intersection(&path_sets[middle]);
                    *jump_slot = jump[middle];
                });
            std::mem::swap(&mut path_sets, &mut doubled_sets);
            std::mem::swap(&mut jump, &mut doubled_jump);

            if round == 4 {
                horizon16
                    .par_iter_mut()
                    .zip(path_sets.par_iter())
                    .for_each(|(current, candidate)| {
                        if candidate.len() > current.len() {
                            *current = *candidate;
                        }
                    });
            } else if round == 6 {
                horizon64
                    .par_iter_mut()
                    .zip(path_sets.par_iter())
                    .for_each(|(current, candidate)| {
                        if candidate.len() > current.len() {
                            *current = *candidate;
                        }
                    });
            }
        }

        (horizon16.into_boxed_slice(), horizon64.into_boxed_slice())
    }

    /// The same exact finite-horizon proof as
    /// [`Self::bounded_observation_safe_horizon_from_state`], additionally
    /// returning states reached while the proof remained valid and the depth
    /// at which each was first observed.
    ///
    /// If the returned source horizon is `H`, a witness `(state, d)` with
    /// `d <= H` is itself proved safe for at least `H - d` bytes: every such
    /// continuation is also a continuation of the source of total length at
    /// most `H`, and all observations through depth `H` equal the source
    /// observation. Runtime masking uses these conservative lower bounds to
    /// amortize one bounded-repeat proof across following counter layers.
    pub fn bounded_observation_safe_horizon_with_witnesses(
        &self,
        source: u32,
        bytes: U8Set,
        active_terminals: &BitSet,
        max_horizon: u8,
    ) -> (u8, Vec<(u32, u8)>) {
        const MAX_FRONTIER_STATES: usize = 4_096;
        // Witnesses are an optional cache accelerator, never part of the
        // correctness proof. Avoid retaining pathological wide-frontier
        // traversals; the source result remains exact if this cap is exceeded.
        const MAX_WITNESSES: usize = 16_384;

        #[inline]
        fn equal_under_mask(left: &BitSet, right: &BitSet, mask: &BitSet) -> bool {
            debug_assert_eq!(left.len(), right.len());
            debug_assert_eq!(left.len(), mask.len());
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        if max_horizon == 0
            || bytes.is_empty()
            || source >= self.num_states()
            || self.state_has_epsilon_transitions(source)
        {
            return (0, Vec::new());
        }

        let source_finalizers = self.matched_terminal_bitset(source);
        let source_futures = self.possible_future_terminals(source);

        // Bounded-repeat products are represented by one compressed segment
        // whose byte equivalence classes are stable across every state in the
        // counter chain. Compute which of those classes intersect B once. A
        // 53-byte ASCII alphabet commonly collapses to one class, replacing 53
        // transition lookups at every depth with one row lookup.
        let source_segment = self.compressed_segment_for_state(source);
        let mut requested_source_classes = SmallVec::<[u8; 16]>::new();
        if let Some(segment) = source_segment {
            for byte in bytes.iter() {
                let class = segment.byte_to_class[byte as usize];
                if !requested_source_classes.contains(&class) {
                    requested_source_classes.push(class);
                }
            }
            requested_source_classes.sort_unstable();
        }

        let mut frontier = vec![source];
        let mut next = Vec::<u32>::with_capacity(16);
        let mut witnesses = Vec::<(u32, u8)>::with_capacity(128);
        witnesses.push((source, 0));
        let mut retain_witnesses = true;
        for depth in 1..=max_horizon {
            next.clear();
            for &state in &frontier {
                if self.state_has_epsilon_transitions(state) {
                    return (depth - 1, witnesses);
                }

                let mut visit_target = |target: u32| -> bool {
                    if target == u32::MAX || self.state_has_epsilon_transitions(target) {
                        return false;
                    }
                    if !equal_under_mask(
                        self.matched_terminal_bitset(target),
                        source_finalizers,
                        active_terminals,
                    ) || !equal_under_mask(
                        self.possible_future_terminals(target),
                        source_futures,
                        active_terminals,
                    ) {
                        return false;
                    }
                    next.push(target);
                    true
                };

                if let Some(segment) = source_segment.filter(|segment| segment.contains_state(state)) {
                    let local_state = state - segment.state_offset;
                    let row_start = segment.row_offsets[local_state as usize] as usize;
                    let row_end = segment.row_offsets[local_state as usize + 1] as usize;
                    let row_classes = segment.entries.class_slice(row_start, row_end);
                    for &class in &requested_source_classes {
                        let Ok(index) = row_classes.binary_search(&class) else {
                            return (depth - 1, witnesses);
                        };
                        let local_target = segment.entries.target(row_start + index);
                        if !visit_target(segment.state_offset + local_target) {
                            return (depth - 1, witnesses);
                        }
                    }
                } else {
                    for byte in bytes.iter() {
                        if !visit_target(self.get_transition(state, byte)) {
                            return (depth - 1, witnesses);
                        }
                    }
                }
            }

            next.sort_unstable();
            next.dedup();
            if next.len() > MAX_FRONTIER_STATES {
                return (depth - 1, witnesses);
            }
            if retain_witnesses {
                if witnesses.len().saturating_add(next.len()) <= MAX_WITNESSES {
                    witnesses.extend(next.iter().copied().map(|state| (state, depth)));
                } else {
                    // Keep the already collected prefix. It remains a valid
                    // lower-bound witness set; simply stop adding more.
                    retain_witnesses = false;
                }
            }
            if next == frontier {
                return (max_horizon, witnesses);
            }
            std::mem::swap(&mut frontier, &mut next);
        }
        (max_horizon, witnesses)
    }

    fn transition_count(&self) -> usize {
        *self.transition_count_cache.get_or_init(|| {
            let packed_segments = self
                .packed_runtime_transition_segments
                .iter()
                .map(|segment| {
                    (0..segment.transitions.state_count() as u32)
                        .map(|state| {
                            segment
                                .transitions
                                .row(state)
                                .map_or(0, |(bytes, _)| bytes.len())
                        })
                        .sum::<usize>()
                })
                .sum::<usize>();
            let packed_compressed = self
                .packed_compressed_transition_segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
            if self.packed_runtime_metadata.is_some() {
                let residual = self
                    .packed_runtime_transitions
                    .as_ref()
                    .map_or_else(|| self.dfa.transition_count(), |transitions| {
                        (0..transitions.state_count() as u32)
                            .map(|state| transitions.row(state).map_or(0, |(bytes, _)| bytes.len()))
                            .sum()
                    });
                return residual + packed_compressed + packed_segments;
            }
            let compressed = self
                .compressed_transition_segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
            let stored_inside_compressed_segments = self
                .compressed_transition_segments
                .iter()
                .map(|segment| {
                    let start = segment.state_offset as usize;
                    let end = start + segment.state_count as usize;
                    self.dfa.states()[start..end]
                        .iter()
                        .map(|state| state.transitions.len())
                        .sum::<usize>()
                })
                .sum::<usize>();
            self.dfa
                .transition_count()
                .saturating_sub(stored_inside_compressed_segments)
                + compressed
                + packed_segments
                + packed_compressed
        })
    }

    /// Detect nullable terminals (those that match the empty string) by
    /// inspecting start-state finalizers, remove them from the DFA, and return
    /// the set.  After this call the tokenizer no longer reports those
    /// terminals as matched at state 0.
    pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        let start = self.start_state();
        let initial_closure = self.dfa.epsilon_closure(&[start]);
        let mut nullable = BTreeSet::new();
        for &state in &initial_closure {
            nullable.extend(
                self.dfa
                    .finalizers(state)
                    .iter()
                    .map(|terminal| terminal as TerminalID),
            );
        }
        if nullable.is_empty() {
            return nullable;
        }
        self.invalidate_derived_caches();

        // The whole initial epsilon closure represents the zero-byte scanner
        // configuration. A component root can also be reached later after a
        // byte transition (for example, a nullable `a*` terminal looping to its
        // root). Clearing its finalizers in place would then remove legitimate
        // non-empty matches. Clone the closure as the post-consumption version,
        // redirect byte entries and external epsilon entries to those clones,
        // and drain finalizers only from the original zero-byte closure.
        let original_state_count = self.dfa.num_states();
        let mut post_byte_state = vec![u32::MAX; original_state_count];
        for &state in &initial_closure {
            let clone = self.dfa.clone_state(state);
            post_byte_state[state as usize] = clone;
        }

        let in_initial_closure = |state: u32| {
            (state as usize) < post_byte_state.len()
                && post_byte_state[state as usize] != u32::MAX
        };

        // Rewrite the cloned closure so all of its internal epsilon structure
        // remains in the post-byte coordinate.
        for &state in &initial_closure {
            let clone = post_byte_state[state as usize];
            let clone_state = &mut self.dfa.states_mut()[clone as usize];
            for (_, target) in clone_state.transitions.iter_mut() {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
            for target in &mut clone_state.epsilon_transitions {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
        }

        // A byte edge always enters the post-byte coordinate. An epsilon edge
        // from outside the initial closure can only be traversed after input has
        // already been consumed, so it does too. Epsilon edges within the
        // original closure remain untouched for the initial zero-byte closure.
        for source in 0..original_state_count {
            let source_in_initial_closure = in_initial_closure(source as u32);
            let state = &mut self.dfa.states_mut()[source];
            for (_, target) in state.transitions.iter_mut() {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
            if !source_in_initial_closure {
                for target in &mut state.epsilon_transitions {
                    if in_initial_closure(*target) {
                        *target = post_byte_state[*target as usize];
                    }
                }
            }
        }

        for state in initial_closure {
            self.dfa.clear_finalizers_for_state(state);
        }
        self.dfa.recompute_possible_futures();
        nullable
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32> {
        if let Some(packed) = &self.packed_runtime_transitions {
            if (state as usize) < packed.state_count() {
                return packed.transition(state, byte);
            }
        }
        if let Some(segment) = self.packed_runtime_transition_segment_for_state(state) {
            return segment.transition(state, byte);
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            return segment.transition(state, byte);
        }
        self.compressed_segment_for_state(state)
            .map_or_else(|| self.dfa.step(state, byte), |segment| segment.transition(state, byte))
    }

    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet {
        let has_any_compressed = !self.compressed_transition_segments.is_empty()
            || !self.packed_compressed_transition_segments.is_empty()
            || !self.packed_runtime_transition_segments.is_empty();
        if !has_any_compressed
            && self.packed_runtime_metadata.is_none()
            && self.packed_runtime_metadata_segments.is_empty()
            && self.packed_runtime_transitions.is_none()
        {
            return self.dfa.step_all(states, byte);
        }
        if states.len() == 1 {
            let state = states[0];
            if !self.state_has_epsilon_transitions(state)
                && let Some(target) = self.step(state, byte)
                && !self.state_has_epsilon_transitions(target)
            {
                return TokenizerStateSet::from_buf([target]);
            }
        }
        let closure = self.epsilon_closure_states(states);
        let mut targets = TokenizerStateSet::new();
        for state in closure {
            if let Some(target) = self.step(state, byte) {
                targets.push(target);
            }
        }
        if targets.is_empty() {
            return targets;
        }
        targets.sort_unstable();
        targets.dedup();
        self.epsilon_closure_states(&targets)
    }

    fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(u32::MAX)
    }

    pub fn run(&self, input: &[u8]) -> TokenizerStateSet {
        self.scan_input(input, self.start_state(), &mut (), |_, _, _, _| {})
    }

    #[doc(hidden)]
    pub fn artifact_metadata_stats(&self) -> (usize, usize, usize, usize) {
        let mut finalizer_bits = 0usize;
        let mut future_bits = 0usize;
        let mut max_finalizers = 0usize;
        let mut max_futures = 0usize;
        for state in 0..self.num_states() {
            let finalizers = self.state_finalizers(state).count_ones();
            let futures = self.state_futures(state).count_ones();
            finalizer_bits += finalizers;
            future_bits += futures;
            max_finalizers = max_finalizers.max(finalizers);
            max_futures = max_futures.max(futures);
        }
        (finalizer_bits, future_bits, max_finalizers, max_futures)
    }

    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.epsilon_closure_states(&[state])
            .into_iter()
            .flat_map(|state| self.matched_terminals_iter(state))
            .collect()
    }

    #[inline]
    pub fn matched_terminals_slice(&self, state: u32) -> &[TerminalID] {
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return metadata
                .finalizer_list(state)
                .expect("packed tokenizer finalizer row must cover every state");
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return segment
                .metadata
                .finalizer_list(segment.local_state(state))
                .expect("packed tokenizer metadata segment must cover every state");
        }
        self.matched_terminals_cache
            .get_or_init(|| {
                let state_count = self.num_states() as usize;
                let mut offsets = Vec::with_capacity(state_count + 1);
                let mut entries = Vec::<TerminalID>::new();
                offsets.push(0);
                for raw_state in 0..self.num_states() {
                    entries.extend(
                        self.state_finalizers(raw_state)
                            .iter()
                            .map(|terminal| terminal as TerminalID),
                    );
                    offsets.push(entries.len());
                }
                Arc::new(MatchedTerminalLists {
                    offsets: offsets.into(),
                    entries: entries.into(),
                })
            })
            .for_state(state)
    }


    pub fn all_singleton_epsilon_closures(&self) -> Arc<SingletonEpsilonClosures> {
        Arc::clone(
            self.singleton_epsilon_closures
                .get_or_init(|| {
                    if self.packed_runtime_metadata.is_none()
                        && self.packed_runtime_metadata_segments.is_empty()
                    {
                        return Arc::new(self.dfa.all_singleton_epsilon_closures());
                    }
                    Arc::new(SingletonEpsilonClosures::Dense(
                        (0..self.num_states())
                            .map(|state| self.epsilon_closure_states(&[state]).into_boxed_slice())
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    ))
                }),
        )

    }


    /// Exact epsilon-closed tokenizer frontiers after one byte from reset.
    /// Entry `b` is epsilon_closure(move(epsilon_closure(reset), b)).
    pub fn initial_byte_frontiers(&self) -> Arc<[TokenizerStateSet]> {
        Arc::clone(self.initial_byte_frontiers.get_or_init(|| {
            let closures = self.all_singleton_epsilon_closures();
            let reset = self.initial_state();
            let reset_closure = closures
                .get(reset as usize)
                .expect("tokenizer reset state must have an epsilon closure");
            let mut rows = Vec::with_capacity(256);
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let mut targets = TokenizerStateSet::new();
                for &state in reset_closure {
                    let Some(target) = self.step(state, byte) else {
                        continue;
                    };
                    let closure = closures
                        .get(target as usize)
                        .expect("tokenizer transition target must have an epsilon closure");
                    targets.extend(closure.iter().copied());
                }
                targets.sort_unstable();
                targets.dedup();
                rows.push(targets);
            }
            Arc::from(rows.into_boxed_slice())
        }))
    }

    pub fn cached_singleton_epsilon_closures(
        &self,
    ) -> Option<&Arc<SingletonEpsilonClosures>> {
        self.singleton_epsilon_closures.get()
    }

    /// Return one exact self-loop byte set per raw tokenizer state.
    ///
    /// This is deliberately separate from `self_loop_bytes(state)`: callers
    /// needing one state keep the local O(out-degree) query, while whole-DFA
    /// compiler passes explicitly opt into one cached O(transitions) build.
    pub fn all_self_loop_bytes(&self) -> Arc<[U8Set]> {
        Arc::clone(self.all_self_loop_bytes_cache.get_or_init(|| {
            Arc::from(
                (0..self.num_states())
                    .map(|state| self.self_loop_bytes(state))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        }))
    }

    pub fn singleton_epsilon_closure(&self, state: u32) -> Box<[u32]> {
        self.epsilon_closure_states(&[state]).into_boxed_slice()
    }

    fn matched_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.state_finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn matched_terminal_bitset(&self, state: u32) -> &BitSet {
        self.state_finalizers(state)
    }

    fn possible_future_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.state_futures(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.state_futures(state)
    }

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32 {
        let mut count = self.dfa.num_states() as u32;
        if let Some(metadata) = self.packed_runtime_metadata.as_deref() {
            count = count.max(metadata.state_count);
        }
        for segment in self.packed_runtime_metadata_segments.iter() {
            count = count.max(segment.state_offset.saturating_add(segment.metadata.state_count));
        }
        count
    }

    fn compute_forced_minimized_state_count(&self) -> usize {
        *self
            .forced_minimized_state_count_cache
            .get_or_init(|| self.dfa.minimize().num_states())
    }

    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult {
        let mut matches = Vec::new();
        let mut end_states = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_all_matches(matches, state, width);
        });
        end_states.retain(|state| !self.is_end(*state));

        TokenizerExecResult {
            end_state: end_states,
            matches,
        }
    }

    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, TokenizerStateSet)>::default();
        let end_states = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state: end_states,
            matches: into_longest_matches(matches),
        }
    }

    /// Exact compact residual scan for compiler analyses that need only each
    /// terminal's longest width and the live states after the complete input.
    /// Unlike `execute_from_state`, this does not retain one matching end-state
    /// set per terminal only to discard it afterward.
    pub fn execute_summary_from_state(
        &self,
        input: &[u8],
        start: u32,
    ) -> (TokenizerStateSet, SmallVec<[(TerminalID, usize); 4]>) {
        let mut matches = SmallVec::<[(TerminalID, usize); 4]>::new();
        let end_states = self.scan_input(
            input,
            start,
            &mut matches,
            |tokenizer, matches, state, width| {
                for terminal in tokenizer.matched_terminals_iter(state) {
                    if let Some((_, longest)) = matches
                        .iter_mut()
                        .find(|(candidate, _)| *candidate == terminal)
                    {
                        *longest = (*longest).max(width);
                    } else {
                        matches.push((terminal, width));
                    }
                }
            },
        );
        matches.sort_unstable_by_key(|(terminal, _)| *terminal);
        (end_states, matches)
    }

    /// Execute the same exact compact residual scan for many starting states,
    /// merging starts as soon as their live lexer states and accumulated
    /// longest matches become identical.
    ///
    /// The returned tuple is `(end_states, longest_matches, starts)`. Expanding
    /// `starts` and comparing each entry with [`Self::execute_summary_from_state`]
    /// yields identical results. This is intended for compiler analyses where
    /// thousands of residual states are tested against the same byte string and
    /// rapidly converge after the first few bytes.
    pub fn execute_summary_groups_from_states(
        &self,
        input: &[u8],
        starts: &[u32],
    ) -> Vec<(
        TokenizerStateSet,
        SmallVec<[(TerminalID, usize); 4]>,
        Vec<u32>,
    )> {
        type ScanKey = (
            TokenizerStateSet,
            SmallVec<[(TerminalID, usize); 4]>,
        );

        let mut active = FxHashMap::<ScanKey, Vec<u32>>::default();
        for &start in starts {
            let states = self.epsilon_closure_states(&[start]);
            active
                .entry((states, SmallVec::new()))
                .or_default()
                .push(start);
        }
        let mut finished = FxHashMap::<ScanKey, Vec<u32>>::default();

        for (index, &byte) in input.iter().enumerate() {
            let width = index + 1;
            let mut next = FxHashMap::<ScanKey, Vec<u32>>::default();
            for ((states, mut matches), support) in active {
                let end_states = self.step_all(&states, byte);
                if end_states.is_empty() {
                    finished
                        .entry((end_states, matches))
                        .or_default()
                        .extend(support);
                    continue;
                }
                for &state in &end_states {
                    for terminal in self.matched_terminals_iter(state) {
                        if let Some((_, longest)) = matches
                            .iter_mut()
                            .find(|(candidate, _)| *candidate == terminal)
                        {
                            *longest = (*longest).max(width);
                        } else {
                            matches.push((terminal, width));
                        }
                    }
                }
                matches.sort_unstable_by_key(|(terminal, _)| *terminal);
                next.entry((end_states, matches))
                    .or_default()
                    .extend(support);
            }
            active = next;
            if active.is_empty() {
                break;
            }
        }
        for (key, support) in active {
            finished.entry(key).or_default().extend(support);
        }

        let mut groups = finished
            .into_iter()
            .map(|((states, matches), mut support)| {
                support.sort_unstable();
                support.dedup();
                (states, matches, support)
            })
            .collect::<Vec<_>>();
        groups.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        groups
    }

    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_states = if exec.end_state.is_empty() {
            SmallVec::from_buf([start])
        } else {
            exec.end_state
        };
        TokenizerResult {
            end_state: end_states,
            matches: group_matches_by_width(exec.matches),
        }
    }

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    /// Scan input bytes and report which terminals of interest matched/finalized.
    ///
    /// Returns a bitset of matched terminals and an optional end state.
    ///
    /// Algorithm:
    /// 1. `remaining = terminals_of_interest`.
    /// 2. `matched = empty`.
    /// 3. For each byte:
    ///    - Check if current state's possible futures overlap `remaining`.
    ///      If not, return `(matched, None)`.
    ///    - Consume byte â†’ next state.
    ///    - If no transition, return `(matched, None)`.
    ///    - Get finalizers at next state, intersect with `remaining`.
    ///    - Add intersection to `matched`, remove from `remaining`.
    /// 4. After all bytes, check futures at end state overlap `remaining`.
    ///    If not, return `(matched, None)`. Otherwise `(matched, Some(end_state))`.
    ///
    /// Important: initial-state finalizers are intentionally ignored.
    /// Only post-byte finalizers count.
    ///
    /// `terminals_of_interest` must have length equal to `self.num_terminals`.
    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, TokenizerStateSet) {
        debug_assert_eq!(terminals_of_interest.len(), self.num_terminals as usize);
        let mut remaining = terminals_of_interest.clone();
        let mut matched = BitSet::new(self.num_terminals as usize);
        let mut states = self.epsilon_closure_states(&[start]);

        for &byte in input {
            let any_future = states
                .iter()
                .any(|&state| !self.possible_future_terminals(state).is_disjoint(&remaining));
            if !any_future {
                return (matched, TokenizerStateSet::new());
            }

            states = self.step_all(&states, byte);
            if states.is_empty() {
                return (matched, states);
            }

            let mut finals = BitSet::new(self.num_terminals as usize);
            for &state in &states {
                finals.union_with(&self.state_finalizers(state).intersection(&remaining));
            }
            matched.union_with(&finals);
            remaining = remaining.difference(&finals);
        }

        states.retain(|state| !self.possible_future_terminals(*state).is_disjoint(&remaining));
        (matched, states)
    }

    fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: u32, width: usize) {
        matches.extend(self.matched_terminals_iter(state).map(|id| TokenizerMatch {
            id,
            width,
            end_state: state,
        }));
    }

    fn record_longest_matches(
        &self,
        matches: &mut FxHashMap<TerminalID, (usize, TokenizerStateSet)>,
        state: u32,
        width: usize,
    ) {
        for terminal in self.matched_terminals_iter(state) {
            let entry = matches
                .entry(terminal)
                .or_insert_with(|| (width, TokenizerStateSet::new()));
            if width > entry.0 {
                entry.0 = width;
                entry.1.clear();
            }
            if width == entry.0 && !entry.1.contains(&state) {
                entry.1.push(state);
            }
        }
    }

    fn scan_input<R>(
        &self,
        input: &[u8],
        start: u32,
        mut matches: &mut R,
        mut record_matches: impl FnMut(&Self, &mut R, u32, usize),
    ) -> TokenizerStateSet {
        // The partitioned runtime tokenizer has a zero-byte dispatcher whose
        // outgoing roots are already deterministic and epsilon-free. Once at
        // least one byte will be consumed, the dispatcher itself cannot remain
        // live, so enter those roots directly instead of materializing its
        // epsilon closure. On very large synthesized tokenizers that closure's
        // generic dense `seen` scratch is otherwise proportional to every raw
        // tokenizer state even though only a handful of roots are live.
        let mut states = if !input.is_empty() && start == self.initial_state_id() {
            self.deterministic_dispatch_roots()
                .map(TokenizerStateSet::from_slice)
                .unwrap_or_else(|| self.epsilon_closure_states(&[start]))
        } else {
            self.epsilon_closure_states(&[start])
        };
        for (index, &byte) in input.iter().enumerate() {
            states = self.step_all(&states, byte);
            if states.is_empty() {
                return states;
            }
            for &state in &states {
                record_matches(self, &mut matches, state, index + 1);
            }
        }
        states
    }


}

impl Lexer for Tokenizer {
    fn start_state(&self) -> u32 { self.start_state() }
    fn num_terminals(&self) -> u32 { self.num_terminals() }
    fn has_epsilon_transitions(&self) -> bool { self.has_epsilon_transitions() }
    fn state_has_epsilon_transitions(&self, state: u32) -> bool { self.state_has_epsilon_transitions(state) }
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ { self.transitions_from(state) }
    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) { self.fill_transition_row(state, row); }
    fn transition_row(&self, state: u32) -> Box<[u32; 256]> { self.transition_row(state) }
    fn self_loop_bytes(&self, state: u32) -> U8Set { self.self_loop_bytes(state) }
    fn transition_count(&self) -> usize { self.transition_count() }
    fn step(&self, state: u32, byte: u8) -> Option<u32> { self.step(state, byte) }
    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet { self.step_all(states, byte) }
    fn get_transition(&self, state: u32, byte: u8) -> u32 { self.get_transition(state, byte) }
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet { self.matched_terminal_bitset(state) }
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.matched_terminals_iter(state) }
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.possible_future_terminals_iter(state) }
    fn possible_future_terminals(&self, state: u32) -> &BitSet { self.possible_future_terminals(state) }
    fn is_end(&self, state: u32) -> bool { self.is_end(state) }
    fn num_states(&self) -> u32 { self.num_states() }
    fn compute_forced_minimized_state_count(&self) -> usize { self.compute_forced_minimized_state_count() }
    fn execute_from_state_all_widths(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state_all_widths(input, start) }
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state(input, start) }
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet { self.execute_from_state_end_only(input, start) }
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult { self.execute_all_matches(input, start) }
    fn initial_state(&self) -> u32 { self.initial_state() }
    fn initial_state_id(&self) -> u32 { self.initial_state_id() }
    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet { self.tokens_accessible_from_state(state) }
    fn scan_terminal_matches_from_state(&self, input: &[u8], start: u32, terminals_of_interest: &BitSet) -> (BitSet, TokenizerStateSet) {
        self.scan_terminal_matches_from_state(input, start, terminals_of_interest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: TokenizerStateSet,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

pub fn arbitrary_epsilon_l1_test_tokenizer() -> Tokenizer {
    let mut dfa = DFA::new(7);
    dfa.ensure_group_capacity(2);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(1, 2);
    dfa.add_epsilon_transition(1, 4);
    dfa.add_transition(2, b'a', 3);
    dfa.add_transition(4, b'a', 5);
    dfa.add_transition(2, b'b', 6);

    let mut terminal_zero = BitSet::new(2);
    terminal_zero.set(0);
    dfa.overwrite_state_metadata(3, terminal_zero.clone(), BitSet::new(2));
    dfa.overwrite_state_metadata(6, terminal_zero, BitSet::new(2));
    let mut terminal_one = BitSet::new(2);
    terminal_one.set(1);
    dfa.overwrite_state_metadata(5, terminal_one, BitSet::new(2));
    dfa.recompute_possible_futures();

    let tokenizer = Tokenizer::from_parts(dfa, 2, None);
    assert!(tokenizer.has_epsilon_transitions());
    assert!(!tokenizer.has_deterministic_dispatch());
    tokenizer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, plus};
    use crate::automata::lexer::compile::build_regex;

    fn tokenizer_from_exprs(exprs: Vec<Expr>) -> Tokenizer {
        let num_terminals = exprs.len() as u32;
        build_regex(&exprs).into_tokenizer(
            num_terminals,
            Some(Arc::from(exprs.into_boxed_slice())),
        )
    }

    fn serialized_roundtrip(tokenizer: &Tokenizer) -> Tokenizer {
        let bytes = bincode::serialize(tokenizer).unwrap();
        let loaded: Tokenizer = bincode::deserialize(&bytes).unwrap();
        assert!(loaded.exprs.is_none(), "Expr sidecars are intentionally not serialized");
        loaded
    }

    #[test]
    fn segment_wire_roundtrips_residual_and_compressed_transitions() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        let mut group = U8Set::empty();
        for byte in [b'a', b'b', b'x'] {
            group.insert(byte);
        }
        dfa.set_group_u8set(0, group);
        dfa.add_transition(0, b'x', 1);
        // Covered rows deliberately remain present in the compile-time DFA;
        // TKS2 must omit them and recover their behavior from the segment.
        dfa.add_transition(1, b'a', 2);
        dfa.add_transition(2, b'b', 3);
        dfa.add_epsilon_transition(0, 1);
        for state in 0..4u32 {
            let mut futures = BitSet::new(1);
            futures.set(0);
            let mut finalizers = BitSet::new(1);
            if state == 3 {
                finalizers.set(0);
            }
            dfa.overwrite_state_metadata(state, finalizers, futures);
        }
        let mut byte_to_class = vec![u8::MAX; 256];
        byte_to_class[b'a' as usize] = 0;
        byte_to_class[b'b' as usize] = 1;
        let segment = CompressedTransitionSegment {
            state_offset: 1,
            state_count: 3,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(
                vec![vec![b'a'].into_boxed_slice(), vec![b'b'].into_boxed_slice()]
                    .into_boxed_slice(),
            ),
            row_offsets: Arc::from(vec![0u32, 1, 2, 2].into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(vec![0, 1], vec![1, 2]),
            expanded_transition_count: 2,
        };
        let original = Tokenizer::from_parts_with_compressed_transitions(
            dfa,
            1,
            None,
            vec![segment],
        );
        let wire = artifact_serde::to_segment_bytes(&original);
        assert!(wire.starts_with(b"TKS2"));
        let loaded = artifact_serde::from_fast_bytes(&wire).unwrap();
        assert_eq!(loaded.compressed_transition_segments.len(), 1);
        assert_eq!(loaded.transition_count(), original.transition_count());
        enumerate_bytes(b"abx", 3, |input| {
            for state in 0..original.num_states() {
                assert_eq!(
                    normalized_exec(&loaded, input, state),
                    normalized_exec(&original, input, state),
                    "TKS2 mismatch from state {state} on {input:?}",
                );
            }
        });

        let huge_wire = artifact_serde::build_huge_bytes(&original).expect("small TKS3 fixture");
        assert!(huge_wire.starts_with(b"TKS3"));
        let huge_loaded = artifact_serde::from_fast_bytes(&huge_wire).unwrap();
        assert!(huge_loaded.has_packed_runtime_metadata());
        assert_eq!(huge_loaded.num_states(), original.num_states());
        assert_eq!(huge_loaded.transition_count(), original.transition_count());
        enumerate_bytes(b"abx", 3, |input| {
            for state in 0..original.num_states() {
                assert_eq!(
                    normalized_exec(&huge_loaded, input, state),
                    normalized_exec(&original, input, state),
                    "TKS3 mismatch from state {state} on {input:?}",
                );
            }
        });
    }

    #[test]
    fn fast_wire_packed_transitions_preserve_multi_state_execution() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_b = BitSet::new(2);
        final_b.set(1);
        dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let wire = artifact_serde::to_fast_bytes(&original);
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.has_packed_runtime_transitions());

        for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
            assert_eq!(
                normalized_exec(&loaded, input, loaded.initial_state()),
                normalized_exec(&original, input, original.initial_state()),
                "fast-wire execution mismatch on {input:?}",
            );
        }
    }

    #[test]
    fn disjoint_union_preserves_loaded_packed_transition_rows_without_materializing() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_b = BitSet::new(2);
        final_b.set(1);
        dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let wire = artifact_serde::to_fast_bytes(&original);
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.packed_runtime_transitions.is_some());
        assert!(loaded.dfa.states()[0].transitions.is_empty());

        let (merged, offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&loaded, 0)]);
        let offset = offsets[0];
        assert!(merged.packed_runtime_transitions.is_none());
        assert_eq!(merged.packed_runtime_transition_segments.len(), 1);
        assert_eq!(merged.step(offset + loaded.initial_state(), b'a'), Some(offset + 1));
        assert_eq!(merged.step(offset + loaded.initial_state(), b'b'), Some(offset + 2));
        for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
            let (source_ends, source_matches) =
                normalized_exec(&loaded, input, loaded.initial_state());
            let expected = (
                source_ends
                    .into_iter()
                    .map(|state| offset + state)
                    .collect::<Vec<_>>(),
                source_matches
                    .into_iter()
                    .map(|(terminal, width, state)| (terminal, width, offset + state))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                normalized_exec(&merged, input, offset + loaded.initial_state()),
                expected,
                "rebased packed execution mismatch on {input:?}",
            );
        }
    }

    #[test]
    fn fast_wire_roundtrip_preserves_nested_packed_transition_segments() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_b = BitSet::new(2);
        final_b.set(1);
        dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let first_wire = artifact_serde::to_fast_bytes(&original);
        let loaded = artifact_serde::from_fast_bytes(&first_wire).expect("first fast tokenizer roundtrip");
        let (nested, offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&loaded, 0)]);
        let offset = offsets[0];
        assert_eq!(nested.packed_runtime_transition_segments.len(), 1);

        let layout = artifact_serde::fast_layout_for_write(&nested)
            .expect("ordinary packed segments should still support TKF2");
        let mut second_wire = vec![0u8; layout.len()];
        artifact_serde::write_fast_bytes_with_layout(&nested, layout, &mut second_wire)
            .expect("nested packed tokenizer should serialize");
        let roundtripped = artifact_serde::from_fast_bytes(&second_wire)
            .expect("nested fast tokenizer roundtrip");

        for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
            let (source_ends, source_matches) = normalized_exec(&loaded, input, loaded.initial_state());
            let expected = (
                source_ends.into_iter().map(|state| offset + state).collect::<Vec<_>>(),
                source_matches
                    .into_iter()
                    .map(|(terminal, width, state)| (terminal, width, offset + state))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                normalized_exec(&roundtripped, input, offset + loaded.initial_state()),
                expected,
                "nested packed fast-wire mismatch on {input:?}",
            );
        }
    }

    #[test]
    fn serialized_terminal_language_equivalence_is_exact_without_exprs() {
        let left = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        let equal = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        let different_same_support =
            serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ba")]));

        assert_eq!(
            left.terminal_language_equivalent_bounded(0, &equal, 0, 64, 10_000),
            Some(true),
        );
        assert_eq!(
            left.terminal_language_equivalent_bounded(
                0,
                &different_same_support,
                0,
                64,
                10_000,
            ),
            Some(false),
        );
    }

    #[test]
    fn serialized_terminal_language_equivalence_budget_fails_closed() {
        let left = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        let right = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        assert_eq!(
            left.terminal_language_equivalent_bounded(0, &right, 0, 1, 10_000),
            None,
            "budget exhaustion must never be interpreted as equivalence",
        );
    }

    fn normalized_exec(
        tokenizer: &Tokenizer,
        input: &[u8],
        start: u32,
    ) -> (Vec<u32>, Vec<(u32, usize, u32)>) {
        let result = tokenizer.execute_from_state_all_widths(input, start);
        let mut end_states = result.end_state.into_vec();
        end_states.sort_unstable();
        let mut matches = result
            .matches
            .into_iter()
            .map(|matched| (matched.id, matched.width, matched.end_state))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        (end_states, matches)
    }

    fn enumerate_bytes(
        alphabet: &[u8],
        max_len: usize,
        mut visit: impl FnMut(&[u8]),
    ) {
        fn rec(
            alphabet: &[u8],
            remaining: usize,
            word: &mut Vec<u8>,
            visit: &mut impl FnMut(&[u8]),
        ) {
            visit(word);
            if remaining == 0 {
                return;
            }
            for &byte in alphabet {
                word.push(byte);
                rec(alphabet, remaining - 1, word, visit);
                word.pop();
            }
        }
        rec(alphabet, max_len, &mut Vec::new(), &mut visit);
    }

    #[test]
    fn grouped_summary_execution_matches_independent_scans() {
        let tokenizer = tokenizer_from_exprs(vec![
            bytes(b"ab"),
            plus(bytes(b"a")),
            bytes(b"ba"),
        ]);
        let starts = (0..tokenizer.num_states()).collect::<Vec<_>>();

        enumerate_bytes(b"abx", 4, |input| {
            let groups = tokenizer.execute_summary_groups_from_states(input, &starts);
            let mut grouped_by_start = std::collections::BTreeMap::new();
            for (end_states, matches, support) in groups {
                for start in support {
                    assert!(
                        grouped_by_start
                            .insert(start, (end_states.clone(), matches.clone()))
                            .is_none(),
                        "start state {start} appeared in more than one summary group for {input:?}",
                    );
                }
            }
            assert_eq!(grouped_by_start.len(), starts.len());
            for &start in &starts {
                assert_eq!(
                    grouped_by_start.get(&start),
                    Some(&tokenizer.execute_summary_from_state(input, start)),
                    "grouped residual scan differs for start state {start}, input {input:?}",
                );
            }
        });
    }

    #[test]
    fn disjoint_union_preserves_every_source_residual_and_unions_resets() {
        let left = tokenizer_from_exprs(vec![bytes(b"ab"), plus(bytes(b"x"))]);
        let right = tokenizer_from_exprs(vec![bytes(b"bc"), plus(bytes(b"y"))]);
        let right_terminal_offset = left.num_terminals();
        let (merged, state_offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[
            (&left, 0),
            (&right, right_terminal_offset),
        ]);
        assert_eq!(state_offsets.len(), 2);

        enumerate_bytes(b"abcxy", 3, |input| {
            for (source, terminal_offset, state_offset) in [
                (&left, 0u32, state_offsets[0]),
                (&right, right_terminal_offset, state_offsets[1]),
            ] {
                for source_state in 0..source.num_states() {
                    let (source_end, source_matches) =
                        normalized_exec(source, input, source_state);
                    let expected_end = source_end
                        .into_iter()
                        .map(|state| state_offset + state)
                        .collect::<Vec<_>>();
                    let expected_matches = source_matches
                        .into_iter()
                        .map(|(terminal, width, state)| {
                            (terminal_offset + terminal, width, state_offset + state)
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        normalized_exec(&merged, input, state_offset + source_state),
                        (expected_end, expected_matches),
                        "residual mismatch from source state {source_state} on {input:?}",
                    );
                }
            }

            let mut expected_end = Vec::new();
            let mut expected_matches = Vec::new();
            for (source, terminal_offset, state_offset) in [
                (&left, 0u32, state_offsets[0]),
                (&right, right_terminal_offset, state_offsets[1]),
            ] {
                let (end, matches) = normalized_exec(source, input, source.start_state());
                expected_end.extend(end.into_iter().map(|state| state_offset + state));
                expected_matches.extend(matches.into_iter().map(|(terminal, width, state)| {
                    (terminal_offset + terminal, width, state_offset + state)
                }));
            }
            if input.is_empty() {
                // The fresh epsilon dispatcher is a real physical state but
                // contributes no finalizer of its own. It remains in the empty
                // execution closure and disappears after the first byte.
                expected_end.push(merged.start_state());
            }
            expected_end.sort_unstable();
            expected_end.dedup();
            expected_matches.sort_unstable();
            expected_matches.dedup();
            assert_eq!(
                normalized_exec(&merged, input, merged.start_state()),
                (expected_end, expected_matches),
                "reset-union mismatch on {input:?}",
            );
        });
    }

    #[test]
    fn full_determinization_accepts_disjoint_union_local_metadata_widths() {
        let left = tokenizer_from_exprs(vec![bytes(b"ab"), bytes(b"ax")]);
        let right = tokenizer_from_exprs(vec![bytes(b"ab")]);
        let (merged, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[
            (&left, 0),
            (&right, left.num_terminals()),
        ]);
        assert!(merged.has_epsilon_transitions());

        let built = merged
            .try_full_determinization(128, 4_096)
            .expect("small disjoint union must determinize across local metadata widths");
        assert!(!built.tokenizer.has_epsilon_transitions());

        enumerate_bytes(b"abx", 3, |input| {
            let source = merged.execute_from_state(input, merged.initial_state());
            let product = built
                .tokenizer
                .execute_from_state(input, built.tokenizer.initial_state());
            let normalize = |matches: Vec<TokenizerMatch>| {
                let mut matches = matches
                    .into_iter()
                    .map(|matched| (matched.id, matched.width))
                    .collect::<Vec<_>>();
                matches.sort_unstable();
                matches.dedup();
                matches
            };
            assert_eq!(normalize(product.matches), normalize(source.matches));
        });
    }

    use crate::automata::lexer::dfa::DFA;

    #[test]
    fn compressed_transition_entries_preserve_pair_sequence_wire_format() {
        let legacy = vec![(0u8, 7u32), (3, 11), (49, 782_231)];
        let entries = CompressedTransitionEntries::from_parts(
            legacy.iter().map(|&(class, _)| class).collect(),
            legacy.iter().map(|&(_, target)| target).collect(),
        );
        assert_eq!(
            bincode::serialize(&entries).unwrap(),
            bincode::serialize(&legacy).unwrap(),
        );
        let decoded: CompressedTransitionEntries =
            bincode::deserialize(&bincode::serialize(&legacy).unwrap()).unwrap();
        assert_eq!(decoded.iter_range(0, decoded.len()).collect::<Vec<_>>(), legacy);
    }

    fn dispatch_prefix_tokenizer(with_appended_residual: bool) -> Tokenizer {
        let mut dfa = DFA::new(if with_appended_residual { 5 } else { 4 });
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 3);
        dfa.add_transition(1, b'a', 2);
        dfa.add_transition(2, b'a', 2);
        dfa.add_transition(3, b'x', 3);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(2, accepting.clone(), BitSet::new(1));
        if with_appended_residual {
            // This state is deliberately not reset-reachable. It models the
            // externally-entered residuals appended by structural synthesis.
            dfa.add_transition(4, b'a', 2);
            dfa.add_transition(4, b'b', 4);
            dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
        }
        dfa.recompute_possible_futures();
        Tokenizer::from_parts(dfa, 1, None)
    }

    #[test]
    fn deterministic_dispatch_execution_enters_roots_without_retaining_dispatcher() {
        let tokenizer = dispatch_prefix_tokenizer(false);
        assert_eq!(tokenizer.deterministic_dispatch_roots(), Some(&[1, 3][..]));
        assert!(tokenizer.has_scalar_deterministic_dispatch());

        let empty = tokenizer.execute_from_state_end_only(b"", tokenizer.initial_state_id());
        assert_eq!(empty.as_slice(), &[0, 1, 3]);

        let a = tokenizer.execute_from_state(b"a", tokenizer.initial_state_id());
        assert!(a.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
        assert_eq!(a.end_state.as_slice(), &[2]);

        let x = tokenizer.execute_from_state_end_only(b"x", tokenizer.initial_state_id());
        assert_eq!(x.as_slice(), &[3]);
    }

    #[test]
    fn matched_terminal_iterator_does_not_materialize_all_state_cache() {
        let tokenizer = dispatch_prefix_tokenizer(false);
        assert!(tokenizer.matched_terminals_cache.get().is_none());

        assert_eq!(tokenizer.matched_terminals_iter(2).collect::<Vec<_>>(), vec![0]);
        assert!(
            tokenizer.matched_terminals_cache.get().is_none(),
            "per-state iteration must not build the all-state matched-terminal cache",
        );

        assert_eq!(tokenizer.matched_terminals_slice(2), &[0]);
        assert!(tokenizer.matched_terminals_cache.get().is_some());
    }

    #[test]
    fn whole_tokenizer_caches_are_reused_and_invalidated_after_mutation() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();

        let loops_before = local.all_self_loop_bytes();
        let loops_before_again = local.all_self_loop_bytes();
        assert!(Arc::ptr_eq(&loops_before, &loops_before_again));
        assert!(loops_before[2].contains(b'a'));
        assert!(loops_before[3].contains(b'x'));
        let closures_before = local.all_singleton_epsilon_closures();
        let transition_count_before = local.transition_count();
        assert!(local.scalar_deterministic_dispatch_cache.get().is_none());
        let scalar_dispatch_before = local.has_scalar_deterministic_dispatch();
        assert_eq!(
            local.scalar_deterministic_dispatch_cache.get(),
            Some(&scalar_dispatch_before),
        );

        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();
        local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .expect("verified append-only component relation");

        assert!(local.scalar_deterministic_dispatch_cache.get().is_none());
        let scalar_dispatch_after = local.has_scalar_deterministic_dispatch();
        assert_eq!(
            local.scalar_deterministic_dispatch_cache.get(),
            Some(&scalar_dispatch_after),
        );
        let loops_after = local.all_self_loop_bytes();
        assert!(!Arc::ptr_eq(&loops_before, &loops_after));
        assert!(loops_after[4].contains(b'b'));
        assert!(!Arc::ptr_eq(
            &closures_before,
            &local.all_singleton_epsilon_closures(),
        ));
        assert!(local.transition_count() > transition_count_before);
    }

    #[test]
    fn precomputed_bounded_observation_sets_respect_16_and_64_horizons() {
        const BAD: u32 = 21;
        let mut dfa = DFA::new((BAD + 1) as usize);
        dfa.ensure_group_capacity(2);

        let stable_finalizers = BitSet::new(2);
        let mut stable_futures = BitSet::new(2);
        stable_futures.set(0);
        let mut bad_futures = BitSet::new(2);
        bad_futures.set(1);

        let string_bytes = (0x20u8..=0x7e)
            .filter(|&byte| !matches!(byte, b'"' | b'\\'))
            .collect::<Vec<_>>();
        for state in 0..BAD {
            dfa.overwrite_state_metadata(
                state,
                stable_finalizers.clone(),
                stable_futures.clone(),
            );
            for &byte in &string_bytes {
                if byte == b'0' {
                    dfa.add_transition(state, byte, state);
                } else {
                    dfa.add_transition(state, byte, state + 1);
                }
            }
        }
        dfa.overwrite_state_metadata(BAD, BitSet::new(2), bad_futures);
        dfa.add_transition(BAD, b'0', BAD);

        let mut tokenizer = Tokenizer::from_parts(dfa, 2, None);
        let advancing_bytes = string_bytes
            .iter()
            .copied()
            .filter(|&byte| byte != b'0')
            .collect::<Vec<_>>();
        // Deliberately split the advancing family across two byte classes.
        // The precompute must recover the larger semantic family by grouping
        // classes with the same destination rather than selecting one class.
        let advancing_a = advancing_bytes
            .iter()
            .copied()
            .filter(|byte| byte & 1 == 0)
            .collect::<Vec<_>>();
        let advancing_b = advancing_bytes
            .iter()
            .copied()
            .filter(|byte| byte & 1 != 0)
            .collect::<Vec<_>>();
        let mut byte_to_class = vec![3u8; 256];
        for &byte in &advancing_a {
            byte_to_class[byte as usize] = 0;
        }
        for &byte in &advancing_b {
            byte_to_class[byte as usize] = 1;
        }
        byte_to_class[b'0' as usize] = 2;
        let class_members = vec![
            advancing_a.into_boxed_slice(),
            advancing_b.into_boxed_slice(),
            vec![b'0'].into_boxed_slice(),
            (0u16..=255)
                .map(|byte| byte as u8)
                .filter(|byte| !advancing_bytes.contains(byte) && *byte != b'0')
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ];
        let mut row_offsets = Vec::<u32>::with_capacity((BAD + 2) as usize);
        let mut classes = Vec::<u8>::new();
        let mut targets = Vec::<u32>::new();
        row_offsets.push(0);
        for state in 0..BAD {
            classes.extend([0, 1, 2]);
            targets.extend([state + 1, state + 1, state]);
            row_offsets.push(classes.len() as u32);
        }
        classes.push(2);
        targets.push(BAD);
        row_offsets.push(classes.len() as u32);
        tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
            state_offset: 0,
            state_count: BAD + 1,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(class_members.into_boxed_slice()),
            row_offsets: Arc::from(row_offsets.into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(classes, targets),
            expanded_transition_count: BAD as usize * 93 + 1,
        }]);
        let (safe16, safe64) = tokenizer.precompute_bounded_observation_safe_byte_sets();

        // The 92 advancing bytes all co-target the same state and are stable
        // for 16 steps from state 0, despite being split across two tokenizer
        // classes. They can reach the observation-changing state before 64
        // steps, so H64 falls back to the infinitely-safe literal self-loop.
        assert!(safe16[0].contains(b'a'));
        assert!(safe16[0].contains(b'Z'));
        assert!(safe16[0].contains(b'_'));
        assert!(!safe16[0].contains(b'0'));
        assert_eq!(safe16[0].len(), 92);
        assert!(!safe64[0].contains(b'a'));
        assert!(safe64[0].contains(b'0'));

        // Cross-check the advertised sets against the generic exact reference
        // proof using the complete terminal domain as the observation mask.
        let mut all_terminals = BitSet::new(2);
        all_terminals.set(0);
        all_terminals.set(1);
        assert_eq!(
            tokenizer.bounded_observation_safe_horizon_from_state(
                0,
                safe16[0],
                &all_terminals,
                16,
            ),
            16,
        );
        assert_eq!(
            tokenizer.bounded_observation_safe_horizon_from_state(
                0,
                safe64[0],
                &all_terminals,
                64,
            ),
            64,
        );
    }

    #[test]
    fn failed_or_noop_mutations_preserve_derived_caches() {
        let mut tokenizer = dispatch_prefix_tokenizer(false);
        let loops = tokenizer.all_self_loop_bytes();
        let closures = tokenizer.all_singleton_epsilon_closures();

        assert!(tokenizer
            .isolate_start_state_and_drain_nullable_terminals()
            .is_empty());
        assert!(Arc::ptr_eq(&loops, &tokenizer.all_self_loop_bytes()));
        assert!(Arc::ptr_eq(
            &closures,
            &tokenizer.all_singleton_epsilon_closures(),
        ));

        let incompatible = Tokenizer::from_parts(DFA::new(1), 2, None);
        assert!(tokenizer
            .augment_from_verified_component_prefixes(&incompatible, &tokenizer.clone(), &[])
            .is_none());
        assert!(Arc::ptr_eq(&loops, &tokenizer.all_self_loop_bytes()));
    }

    #[test]
    fn compressed_bounded_observation_certificate_matches_generic_reference() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        let mut live = BitSet::new(1);
        live.set(0);
        for state in 0..3u32 {
            dfa.overwrite_state_metadata(state, BitSet::new(1), live.clone());
        }
        dfa.overwrite_state_metadata(3, live.clone(), live.clone());

        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
        let byte_to_class = vec![0u8; 256];
        let class_members = vec![(0u16..=255)
            .map(|byte| byte as u8)
            .collect::<Vec<_>>()
            .into_boxed_slice()];
        tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
            state_offset: 0,
            state_count: 4,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(class_members.into_boxed_slice()),
            row_offsets: Arc::from([0u32, 1, 2, 3, 4]),
            entries: CompressedTransitionEntries::from_parts(
                vec![0, 0, 0, 0],
                vec![1, 2, 3, 3],
            ),
            expanded_transition_count: 4 * 256,
        }]);

        let mut bytes = U8Set::empty();
        bytes.insert(b'a');
        bytes.insert(b'b');
        let mut active = BitSet::new(1);
        active.set(0);

        for max_horizon in 1..=8 {
            let optimized = tokenizer.bounded_observation_safe_horizon_from_state(
                0,
                bytes,
                &active,
                max_horizon,
            );
            let reference = <Tokenizer as Lexer>::bounded_observation_safe_horizon_from_state(
                &tokenizer,
                0,
                bytes,
                &active,
                max_horizon,
            );
            assert_eq!(optimized, reference, "max_horizon={max_horizon}");
        }
        assert_eq!(
            tokenizer.bounded_observation_safe_horizon_from_state(0, bytes, &active, 8),
            2,
        );

        let (source_horizon, witnesses) = tokenizer
            .bounded_observation_safe_horizon_with_witnesses(0, bytes, &active, 8);
        assert_eq!(source_horizon, 2);
        for (state, depth) in witnesses {
            if depth > source_horizon {
                continue;
            }
            let inherited = source_horizon - depth;
            let reference = <Tokenizer as Lexer>::bounded_observation_safe_horizon_from_state(
                &tokenizer,
                state,
                bytes,
                &active,
                inherited,
            );
            assert_eq!(
                reference, inherited,
                "witness state={state} depth={depth} did not inherit its conservative bound",
            );
        }
    }

    #[test]
    fn deterministic_reset_dispatch_does_not_certify_a_later_epsilon_frontier() {
        let mut dfa = DFA::new(6);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 4);
        dfa.add_transition(1, b'a', 2);
        dfa.add_epsilon_transition(2, 3);
        dfa.add_transition(3, b'b', 3);
        dfa.add_transition(4, b'x', 5);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(5, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert_eq!(tokenizer.deterministic_dispatch_roots(), Some(&[1, 4][..]));
        assert!(tokenizer.has_deterministic_dispatch());
        assert!(!tokenizer.has_scalar_deterministic_dispatch());

        let result = tokenizer.execute_from_state_end_only(b"ab", tokenizer.initial_state_id());
        assert_eq!(result.as_slice(), &[3]);
    }

    #[test]
    fn structural_prefix_augmentation_rejects_a_non_homomorphic_target_prefix() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();
        local
            .dfa
            .set_transitions_from_sorted_entries(2, vec![(b'a', 3)]);
        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

        assert!(local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .is_none());
    }

    #[test]
    fn structural_prefix_augmentation_clones_only_appended_residuals() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();
        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

        let source_to_local = local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .expect("verified append-only component relation");

        assert_eq!(local.num_states(), source.num_states());
        assert_eq!(source_to_local, vec![0, 1, 2, 3, 4]);
        for input in [b"".as_slice(), b"a", b"b", b"ba", b"bba"] {
            let source_result = source.execute_from_state_all_widths(input, 4);
            let local_result = local.execute_from_state_all_widths(input, source_to_local[4]);
            assert_eq!(source_result.matches, local_result.matches, "input={input:?}");
            assert_eq!(source_result.end_state, local_result.end_state, "input={input:?}");
        }
    }

    #[test]
    fn matched_terminal_cache_is_invalidated_by_alias_canonicalization() {
        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(2);
        let mut alias = BitSet::new(2);
        alias.set(1);
        dfa.overwrite_state_metadata(0, alias, BitSet::new(2));
        dfa.recompute_possible_futures();

        let mut tokenizer = Tokenizer::from_parts(dfa, 2, None);
        assert_eq!(tokenizer.matched_terminals_slice(0), &[1]);
        tokenizer.canonicalize_terminal_aliases(0, &[1]);
        assert_eq!(tokenizer.matched_terminals_slice(0), &[0]);
    }

    #[test]
    fn owned_parent_union_invalidates_preinitialized_runtime_caches() {
        fn one_byte(byte: u8) -> Tokenizer {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.add_transition(0, byte, 1);
            let mut accepting = BitSet::new(1);
            accepting.set(0);
            dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
            dfa.recompute_possible_futures();
            Tokenizer::from_parts(dfa, 1, None)
        }

        let left = one_byte(b'a');
        let right = one_byte(b'b');
        assert_eq!(left.matched_terminals_slice(1), &[0]);
        assert!(left.initial_byte_frontiers()[b'a' as usize].contains(&1));

        let (merged, offsets) =
            Tokenizer::disjoint_union_with_owned_parent(left, 0, &[(&right, 1)]);
        let right_accept = offsets[1] + 1;
        assert_eq!(merged.matched_terminals_slice(right_accept), &[1]);
        assert!(merged.initial_byte_frontiers()[b'b' as usize].contains(&right_accept));
    }

    #[test]
    fn owned_parent_union_preserves_fast_loaded_parent_root_and_new_epsilon_child() {
        fn one_byte(byte: u8) -> Tokenizer {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.add_transition(0, byte, 1);
            let mut accepting = BitSet::new(1);
            accepting.set(0);
            dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
            dfa.recompute_possible_futures();
            Tokenizer::from_parts(dfa, 1, None)
        }

        let parent = one_byte(b'<');
        let left = one_byte(b'a');
        let (half, _) = Tokenizer::disjoint_union_with_owned_parent(parent, 0, &[(&left, 1)]);
        let wire = artifact_serde::to_fast_bytes(&half);
        let mut loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.packed_runtime_transitions.is_some());

        // Tiny tokenizers normally keep metadata in the DFA after TKF2 load,
        // while large TKS3 artifacts use the packed metadata representation
        // that originally exposed this bug. Install the equivalent packed
        // metadata explicitly so this focused regression exercises the same
        // precedence rules without constructing a 100k-state tokenizer.
        let state_count = loaded.dfa.num_states();
        assert!(state_count <= u8::MAX as usize);
        let finalizer_rows = (0..state_count)
            .map(|state| loaded.dfa.finalizers(state as u32).clone())
            .collect::<Vec<_>>();
        let finalizer_lists = finalizer_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let future_rows = (0..state_count)
            .map(|state| loaded.dfa.possible_future_group_ids(state as u32).clone())
            .collect::<Vec<_>>();
        let mut epsilon_states = Vec::new();
        let mut epsilon_offsets = vec![0u32];
        let mut epsilon_targets = Vec::new();
        for (state, row) in loaded.dfa.states().iter().enumerate() {
            if row.epsilon_transitions.is_empty() {
                continue;
            }
            epsilon_states.push(state as u32);
            epsilon_targets.extend_from_slice(&row.epsilon_transitions);
            epsilon_offsets.push(epsilon_targets.len() as u32);
        }
        loaded.packed_runtime_metadata = Some(Arc::new(PackedTokenizerMetadata {
            state_count: state_count as u32,
            finalizer_row_ids: PackedRowIds::U8(Arc::from(
                (0..state_count as u8).collect::<Vec<_>>().into_boxed_slice(),
            )),
            finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids: PackedRowIds::U8(Arc::from(
                (0..state_count as u8).collect::<Vec<_>>().into_boxed_slice(),
            )),
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::from(epsilon_states.into_boxed_slice()),
            epsilon_offsets: Arc::from(epsilon_offsets.into_boxed_slice()),
            epsilon_targets: Arc::from(epsilon_targets.into_boxed_slice()),
        }));

        let right = one_byte(b'b');
        let (nested, offsets) =
            Tokenizer::disjoint_union_with_owned_parent(loaded, 0, &[(&right, 2)]);
        assert_eq!(offsets[0], 0);
        assert_eq!(nested.deterministic_reset_states().as_slice(), &[0]);
        assert!(
            nested.packed_runtime_transitions.is_some(),
            "structural mutation must not expand the loaded parent's packed byte rows",
        );

        for (input, terminal) in [(b"<".as_slice(), 0), (b"a", 1), (b"b", 2)] {
            let result = nested.execute_from_state(input, nested.initial_state_id());
            assert!(
                result
                    .matches
                    .iter()
                    .any(|matched| matched.id == terminal && matched.width == 1),
                "missing terminal {terminal} for {input:?}: {result:?}",
            );
        }
    }

    #[test]
    fn execution_handles_epsilon_edges_before_and_after_a_byte() {
        let mut dfa = DFA::new(6);
        dfa.ensure_group_capacity(2);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(1, 2);
        dfa.add_epsilon_transition(2, 1);
        dfa.add_transition(1, b'a', 3);
        dfa.add_transition(2, b'a', 4);
        dfa.add_epsilon_transition(3, 5);

        let mut terminal_zero = BitSet::new(2);
        terminal_zero.set(0);
        dfa.overwrite_state_metadata(5, terminal_zero, BitSet::new(2));
        let mut terminal_one = BitSet::new(2);
        terminal_one.set(1);
        dfa.overwrite_state_metadata(4, terminal_one, BitSet::new(2));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 2, None);
        let execution = tokenizer.execute_from_state_all_widths(b"a", 0);
        let mut matches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        assert_eq!(matches, vec![(0, 1), (1, 1)]);
        assert!(execution.end_state.is_empty());
        let longest = tokenizer.execute_from_state(b"a", 0);
        assert_eq!(longest.end_state.as_slice(), &[3, 4, 5]);
        assert_eq!(tokenizer.matched_terminals(3), BTreeSet::from([0]));

        let interests = BitSet::all(2);
        let (matched, continuation) =
            tokenizer.scan_terminal_matches_from_state(b"a", 0, &interests);
        assert!(matched.contains(0));
        assert!(matched.contains(1));
        assert!(continuation.is_empty());
    }

    #[test]
    fn draining_nullable_initial_closure_preserves_later_root_matches() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_transition(1, b'a', 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert_eq!(tokenizer.matched_terminals(0), BTreeSet::from([0]));
        assert_eq!(
            tokenizer.isolate_start_state_and_drain_nullable_terminals(),
            BTreeSet::from([0]),
        );
        assert!(tokenizer.matched_terminals(0).is_empty());

        let one = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert!(one.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
        let two = tokenizer.execute_from_state(b"aa", tokenizer.initial_state());
        assert!(two.matches.iter().any(|matched| matched.id == 0 && matched.width == 2));
    }

    #[test]
    fn longest_match_preserves_every_accepting_end_state_for_one_terminal() {
        let mut dfa = DFA::new(5);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 2);
        dfa.add_transition(1, b'a', 3);
        dfa.add_transition(2, b'a', 4);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 1, None);
        let mut end_states = tokenizer
            .execute_from_state(b"a", 0)
            .matches
            .into_iter()
            .filter(|matched| matched.id == 0 && matched.width == 1)
            .map(|matched| matched.end_state)
            .collect::<Vec<_>>();
        end_states.sort_unstable();
        assert_eq!(end_states, vec![3, 4]);
    }

    #[test]
    fn full_determinization_is_exact_subset_construction() {
        let source = arbitrary_epsilon_l1_test_tokenizer();
        let built = source
            .try_full_determinization(128, 4_096)
            .expect("small epsilon tokenizer must fully determinize");
        let deterministic = &built.tokenizer;

        assert!(!deterministic.has_epsilon_transitions());
        assert_eq!(
            built.source_subsets.len(),
            deterministic.num_states() as usize,
        );

        for state in 0..deterministic.num_states() {
            let subset = &built.source_subsets[state as usize];
            let mut finalizers = BitSet::new(source.num_terminals() as usize);
            let mut futures = BitSet::new(source.num_terminals() as usize);
            for &source_state in subset.iter() {
                finalizers.union_with(source.dfa.finalizers(source_state));
                futures.union_with(source.dfa.possible_future_group_ids(source_state));
            }
            assert_eq!(deterministic.dfa.finalizers(state), &finalizers);
            assert_eq!(
                deterministic.dfa.possible_future_group_ids(state),
                &futures,
            );

            for byte in 0u16..=255 {
                let mut expected = SmallVec::<[u32; 8]>::new();
                for &source_state in subset.iter() {
                    if let Some(target) = source.step(source_state, byte as u8) {
                        expected.push(target);
                    }
                }
                expected.sort_unstable();
                expected.dedup();
                let mut expected = source.dfa.epsilon_closure(&expected);
                expected.sort_unstable();
                expected.dedup();

                match deterministic.step(state, byte as u8) {
                    Some(target) => assert_eq!(
                        built.source_subsets[target as usize].as_ref(),
                        expected.as_slice(),
                        "state={state} byte={byte}",
                    ),
                    None => assert!(expected.is_empty(), "state={state} byte={byte}"),
                }
            }
        }

        for input in [
            b"".as_slice(),
            b"a",
            b"b",
            b"aa",
            b"ab",
            b"ba",
            b"aaa",
        ] {
            let source_result = source.execute_from_state(input, source.initial_state_id());
            let deterministic_result = deterministic
                .execute_from_state(input, deterministic.initial_state_id());
            let mut source_matches = source_result
                .matches
                .iter()
                .map(|matched| (matched.id, matched.width))
                .collect::<Vec<_>>();
            source_matches.sort_unstable();
            source_matches.dedup();
            let mut deterministic_matches = deterministic_result
                .matches
                .iter()
                .map(|matched| (matched.id, matched.width))
                .collect::<Vec<_>>();
            deterministic_matches.sort_unstable();
            deterministic_matches.dedup();
            assert_eq!(source_matches, deterministic_matches, "input={input:?}");

            let represented_end_states = deterministic_result
                .end_state
                .iter()
                .flat_map(|&state| built.source_subsets[state as usize].iter().copied())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                represented_end_states,
                source_result.end_state.iter().copied().collect(),
                "input={input:?}",
            );
        }
    }
}
