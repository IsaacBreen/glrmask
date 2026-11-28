use glrmask_artifact::CommitTemplateDfas;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use rayon::prelude::*;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::SmallVec;

use crate::automata::lexer::{
    DFA as LexerDfa, Lexer,
    tokenizer::{TerminalProjectedQuotient, Tokenizer},
};
use crate::automata::lexer::runtime_repeat_product::VirtualBinaryRepeatIntersectionMaskProjection;
use crate::automata::lexer::tokenizer::VirtualResidualMaskProjection;
use crate::automata::lexer::runtime_unit_repeat::VirtualZeroMinUnitRepeatMaskProjection;
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::{DWA, DwaTransitionMap};
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::parser::{
    ParserComponentTableSource, ParserGSS, ScopedSubgrammarLink,
};
use crate::compiler::glr::table::GLRTable;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    VocabPartitionDfa, classify_vocab_char_type,
};
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

use super::mask_cache_payload::DynamicMaskCachePayload;

/// Optional composition-boundary metadata, independent of the masking optimization. `None` adds no metadata; `Tokens` is a conservative parser-state-independent candidate set; `Exact` adds state-sensitive metadata where supported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundaryTriggerDetail {
    None,
    #[default]
    TokenSet,
    Exact,
}

/// Optional reusable hint for dynamic composition boundaries. This is an
/// accelerator only: `None` means the composition runtime must conservatively
/// assume that any model token may cross a component boundary.
#[derive(Debug, Clone, Default)]
pub(crate) enum BoundaryTrigger {
    #[default]
    None,
    /// Conservative original model-token IDs that may contain the first
    /// internal boundary crossing. Parser-state independent.
    TokenSet(Arc<[u32]>),
    /// Exact component-local parser DWA. Its coordinate is deliberately
    /// independent of the ordinary parser-DWA quotient: parser labels are
    /// local LR-state IDs, weight TSIDs are raw local tokenizer-state IDs, and
    /// weight token IDs are original/model token IDs. Proper-prefix boundary
    /// behavior is a stronger observation than the ordinary whole-token mask
    /// language, so reusing the normal TSID/internal-token quotient would need
    /// a separate equivalence proof.
    Exact(Arc<DWA>),
}

impl BoundaryTrigger {
    pub(crate) fn detail(&self) -> BoundaryTriggerDetail {
        match self {
            Self::None => BoundaryTriggerDetail::None,
            Self::TokenSet(_) => BoundaryTriggerDetail::TokenSet,
            Self::Exact(_) => BoundaryTriggerDetail::Exact,
        }
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn token_summary(&self) -> Option<&[u32]> {
        match self {
            Self::TokenSet(tokens) => Some(tokens),
            Self::None | Self::Exact(_) => None,
        }
    }
}

/// Stable identity of the inputs that make a boundary-candidate summary
/// reusable.  The four pieces are intentionally separate so diagnostics can
/// distinguish stale component semantics, interface changes, and vocabulary
/// mismatches instead of treating every miss as an opaque hash failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BoundaryCandidateFingerprint {
    pub(crate) algorithm_version: u16,
    pub(crate) component_semantics: [u8; 32],
    pub(crate) public_interface: [u8; 32],
    pub(crate) vocabulary: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SummaryPrecision {
    Regular,
    ContextRefined,
    BudgetWidened,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SummaryUnavailable {
    Disabled,
    Deferred,
    LegacyArtifact,
    MissingGrammarMetadata,
    UnsupportedFiniteLexer,
    InvalidatedBinding,
    FingerprintMismatch,
    MalformedMetadata,
}

/// Original/model-token ID set used by boundary summaries.  Sparse is the
/// canonical wire form; Dense is useful when a component legitimately retains
/// a large irregular set.  `AllByteTokensAtLeastTwo` is the safe top element
/// for the proper-prefix question: one-byte tokens can never contain a
/// non-empty proper byte prefix.
#[derive(Debug, Clone)]
pub(crate) enum OriginalTokenSet {
    Empty,
    Sparse(Arc<[u32]>),
    Dense(Arc<BitSet>),
    AllByteTokensAtLeastTwo,
}

impl OriginalTokenSet {
    pub(crate) fn from_sorted_unique(ids: Vec<u32>, max_token_id: u32) -> Self {
        if ids.is_empty() {
            return Self::Empty;
        }
        // Dense becomes cheaper only when the set is genuinely dense. Keep a
        // deliberately conservative crossover because Sparse is also the
        // canonical persisted representation.
        let dense_words = (max_token_id as usize + 64) / 64;
        if ids.len() > dense_words.saturating_mul(3) {
            let mut bits = BitSet::new(max_token_id as usize + 1);
            for id in ids {
                bits.set(id as usize);
            }
            Self::Dense(Arc::new(bits))
        } else {
            Self::Sparse(Arc::from(ids.into_boxed_slice()))
        }
    }

    pub(crate) fn contains(&self, token_id: u32, bytes: &[u8]) -> bool {
        match self {
            Self::Empty => false,
            Self::Sparse(ids) => ids.binary_search(&token_id).is_ok(),
            Self::Dense(bits) => {
                (token_id as usize) < bits.len() && bits.get(token_id as usize)
            }
            Self::AllByteTokensAtLeastTwo => bytes.len() >= 2,
        }
    }

    pub(crate) fn canonical_ids<'a>(
        &'a self,
        tokens: impl Iterator<Item = (u32, &'a [u8])>,
    ) -> Vec<u32> {
        match self {
            Self::Empty => Vec::new(),
            Self::Sparse(ids) => ids.to_vec(),
            Self::Dense(bits) => bits.iter_ones().map(|id| id as u32).collect(),
            Self::AllByteTokensAtLeastTwo => tokens
                .filter_map(|(id, bytes)| (bytes.len() >= 2).then_some(id))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BoundaryCandidateSummary {
    Unknown { reason: SummaryUnavailable },
    Known {
        fingerprint: BoundaryCandidateFingerprint,
        tokens: OriginalTokenSet,
        precision: SummaryPrecision,
    },
}

impl Default for BoundaryCandidateSummary {
    fn default() -> Self {
        Self::Unknown {
            reason: SummaryUnavailable::Deferred,
        }
    }
}

impl BoundaryCandidateSummary {
    pub(crate) fn known_tokens_for(
        &self,
        fingerprint: &BoundaryCandidateFingerprint,
    ) -> Option<&OriginalTokenSet> {
        match self {
            Self::Known {
                fingerprint: actual,
                tokens,
                ..
            } if actual == fingerprint => Some(tokens),
            _ => None,
        }
    }

    pub(crate) fn is_known(&self) -> bool {
        matches!(self, Self::Known { .. })
    }
}

/// Small composition-time grammar summary retained with a compiled component.
///
/// For a nonnullable child, substituting the child's language for a parent
/// placeholder needs only:
/// * terminal adjacency (`allowed_follows`),
/// * FIRST/LAST of the component root, and
/// * root nullability.
///
/// Keeping this summary in the outer artifact envelope lets the linker compose
/// grammar legality algebraically instead of rebuilding FIRST/FOLLOW over the
/// fully merged rule graph.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct CompositionGrammarSummary {
    pub(crate) allowed_follows: Vec<BitSet>,
    pub(crate) root_first: BitSet,
    pub(crate) root_last: BitSet,
    pub(crate) root_nullable: bool,
}

/// The same exact parser automaton in either compiler-owned or packed form.
/// Packed constraints never retain a second owned state graph.
#[derive(Debug, Clone)]
pub(crate) enum ParserDwaStorage {
    /// Masks are computed directly or by the linked component accelerators.
    Absent,
    Owned(DWA),
    Packed(Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>),
}

impl Default for ParserDwaStorage {
    fn default() -> Self {
        Self::Absent
    }
}

impl ParserDwaStorage {
    pub(crate) fn owned(&self) -> Option<&DWA> {
        match self {
            Self::Owned(dwa) => Some(dwa),
            Self::Packed(_) | Self::Absent => None,
        }
    }

    pub(crate) fn packed(&self) -> Option<&crate::automata::weighted::dwa::PackedRuntimeDwa> {
        match self {
            Self::Owned(_) | Self::Absent => None,
            Self::Packed(dwa) => Some(dwa),
        }
    }

    pub(crate) fn for_compilation(&self) -> Result<std::borrow::Cow<'_, DWA>, String> {
        match self {
            Self::Absent => Err("constraint has no global parser DWA".into()),
            Self::Owned(dwa) => Ok(std::borrow::Cow::Borrowed(dwa)),
            Self::Packed(dwa) => dwa.to_dwa().map(std::borrow::Cow::Owned),
        }
    }

    pub(crate) fn state_count(&self) -> usize {
        match self {
            Self::Absent => 0,
            Self::Owned(dwa) => dwa.states().len(),
            Self::Packed(dwa) => dwa.state_count(),
        }
    }

    #[cfg(feature = "internal-api")]
    pub(crate) fn transition_count(&self) -> usize {
        match self {
            Self::Absent => 0,
            Self::Owned(dwa) => dwa.num_transitions(),
            Self::Packed(dwa) => dwa.transition_count(),
        }
    }

    pub(crate) fn start_state(&self) -> u32 {
        match self {
            Self::Absent => panic!("constraint has no global parser DWA start state"),
            Self::Owned(dwa) => dwa.start_state(),
            Self::Packed(dwa) => dwa.start_state(),
        }
    }
}

/// Non-DWA weight relations share the final constraint TSID/token coordinate.
/// Runtime storage changes the value representation, never the keys or order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct NonDwaWeights<W> {
    /// Exact depth-one acceptance: encoded parser-state label to the
    /// transition/final-weight intersection after one stack symbol.
    pub(crate) parser_top_accept: BTreeMap<i32, W>,
    /// Direct-regular grammars retain uncombined completion parts, avoiding
    /// construction of one large union weight for each parser state.
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<W>>,
    /// Immediate L1 completion, shared once per grammar terminal.
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, W>,
    /// Exact possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// possible_matches_complete to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: BTreeMap<TerminalID, W>,
}

impl<W> Default for NonDwaWeights<W> {
    fn default() -> Self {
        Self {
            parser_top_accept: BTreeMap::new(),
            parser_top_accept_parts: BTreeMap::new(),
            direct_regular_l1_complete_by_terminal: BTreeMap::new(),
            possible_matches: BTreeMap::new(),
        }
    }
}

impl<W> NonDwaWeights<W> {
    pub(crate) fn final_weights(&self) -> impl Iterator<Item = &W> {
        self.parser_top_accept.values()
            .chain(self.parser_top_accept_parts.values().flatten())
            .chain(self.direct_regular_l1_complete_by_terminal.values())
    }

    pub(crate) fn weights(&self) -> impl Iterator<Item = &W> {
        self.final_weights().chain(self.possible_matches.values())
    }

    pub(crate) fn map_weights<T>(&self, mut map: impl FnMut(&W) -> T) -> NonDwaWeights<T> {
        NonDwaWeights {
            parser_top_accept: self.parser_top_accept.iter()
                .map(|(&key, value)| (key, map(value))).collect(),
            parser_top_accept_parts: self.parser_top_accept_parts.iter()
                .map(|(&key, values)| (key, values.iter().map(&mut map).collect())).collect(),
            direct_regular_l1_complete_by_terminal: self.direct_regular_l1_complete_by_terminal.iter()
                .map(|(&key, value)| (key, map(value))).collect(),
            possible_matches: self.possible_matches.iter()
                .map(|(&key, value)| (key, map(value))).collect(),
        }
    }

    pub(crate) fn top_accept(&self, label: i32) -> Option<&W> {
        self.parser_top_accept.get(&label)
            .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
    }

    pub(crate) fn top_parts(&self, label: i32) -> &[W] {
        self.parser_top_accept_parts.get(&label)
            .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
            .map_or(&[], Vec::as_slice)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PackedNonDwaWeights {
    pub(crate) final_masks: Arc<super::constraint::packed_final_masks::PackedFinalMaskCache>,
    pub(crate) pool: Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    pub(crate) relations: NonDwaWeights<u32>,
}

impl PackedNonDwaWeights {
    pub(crate) fn new(
        pool: Arc<crate::ds::weight::PackedRuntimeWeightPool>,
        relations: NonDwaWeights<u32>,
    ) -> Result<Self, String> {
        if let Some(&id) = relations.weights().find(|&&id| id as usize >= pool.weight_count()) {
            return Err(format!("packed non-DWA Weight id {id} is out of range"));
        }
        Ok(Self { final_masks: Default::default(), pool, relations })
    }

    fn materialize(&self) -> Result<NonDwaWeights<Weight>, String> {
        let weights = crate::ds::weight::unpack_pooled_weights(self.pool.packed_bytes())?;
        if weights.len() != self.pool.weight_count() {
            return Err("packed non-DWA Weight count mismatch".to_owned());
        }
        // IDs were checked against the immutable pool when attached.
        Ok(self.relations.map_weights(|&id| weights[id as usize].clone()))
    }
}

/// Exactly one representation of the labeled non-DWA weights is live.
/// Loaded/compacted constraints do not retain placeholder Weight maps.
#[derive(Debug, Clone)]
pub(crate) enum NonDwaWeightStorage {
    Materialized(NonDwaWeights<Weight>),
    Pooled(Arc<PackedNonDwaWeights>),
}

impl Default for NonDwaWeightStorage {
    fn default() -> Self {
        Self::Materialized(NonDwaWeights::default())
    }
}

impl NonDwaWeightStorage {
    pub(crate) fn materialized(&self) -> Option<&NonDwaWeights<Weight>> {
        match self {
            Self::Materialized(weights) => Some(weights),
            Self::Pooled(_) => None,
        }
    }

    pub(crate) fn materialized_mut(&mut self) -> &mut NonDwaWeights<Weight> {
        match self {
            Self::Materialized(weights) => weights,
            Self::Pooled(_) => panic!("compiler mutation requires materialized non-DWA weights"),
        }
    }

    pub(crate) fn pooled(&self) -> Option<&Arc<PackedNonDwaWeights>> {
        match self {
            Self::Materialized(_) => None,
            Self::Pooled(weights) => Some(weights),
        }
    }

    pub(crate) fn pooled_mut(&mut self) -> Option<&mut Arc<PackedNonDwaWeights>> {
        match self {
            Self::Materialized(_) => None,
            Self::Pooled(weights) => Some(weights),
        }
    }

    pub(crate) fn materialized_for_compilation(
        &self,
    ) -> Result<std::borrow::Cow<'_, NonDwaWeights<Weight>>, String> {
        match self {
            Self::Materialized(weights) => Ok(std::borrow::Cow::Borrowed(weights)),
            Self::Pooled(weights) => weights.materialize().map(std::borrow::Cow::Owned),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularWideFrontierAcceptance {
    /// Pointer identities of immutable replace-target or StackShifts slices in the live table
    /// that all produce this exact frontier. Runtime-only and rebuilt after
    /// deserialization.
    pub(crate) action_origins: Vec<usize>,
    pub(crate) state_count: usize,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) frontier_states: Arc<[u32]>,
    pub(crate) empty_acc_frontier: ParserGSS,
    pub(crate) acceptance_parts: Arc<[Weight]>,
    pub(crate) dense_by_tsid: Arc<DenseAcceptanceRows>,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularDynamicHotFrontier {
    pub(crate) frontier_states: Arc<[u32]>,
    pub(crate) empty_acc_frontier: ParserGSS,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularParserStateAcceptance {
    pub(crate) parser_state: u32,
    pub(crate) acceptance_parts: Arc<[Weight]>,
    pub(crate) dense_by_tsid: Arc<DenseAcceptanceRows>,
}

pub(crate) type DenseWords = Arc<[u64]>;

/// Exact acceptance rows interned independently of tokenizer coordinates.
/// IDs 0 and 1 denote absent and the shared full mask; all others index the
/// slab of actual distinct masks. Constant/range dispatch avoids even a
/// per-state ID when many tokenizer states have the same acceptance.
#[derive(Debug, Clone)]
enum AcceptanceRowDispatch {
    Dense(Arc<[u32]>),
    Runs { len: usize, starts: Arc<[(usize, u32)]> },
}

impl Default for AcceptanceRowDispatch {
    fn default() -> Self { Self::Dense(Arc::from([])) }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DenseAcceptanceRows {
    words_per_row: usize,
    rows: Arc<[u64]>,
    dispatch: AcceptanceRowDispatch,
    full_dense: DenseWords,
}

impl DenseAcceptanceRows {
    pub(crate) fn from_runs(
        words_per_row: usize,
        tsid_count: usize,
        rows: Vec<u64>,
        starts: Vec<(usize, u32)>,
        full_dense: DenseWords,
    ) -> Self {
        let run_bytes = starts.len().saturating_mul(std::mem::size_of::<(usize, u32)>());
        let dense_bytes = tsid_count.saturating_mul(std::mem::size_of::<u32>());
        let dispatch = if run_bytes < dense_bytes {
            AcceptanceRowDispatch::Runs { len: tsid_count, starts: starts.into() }
        } else {
            let mut ids = vec![0u32; tsid_count];
            for (index, &(start, id)) in starts.iter().enumerate() {
                let end = starts.get(index + 1).map_or(tsid_count, |&(end, _)| end);
                ids[start..end].fill(id);
            }
            AcceptanceRowDispatch::Dense(ids.into())
        };
        Self { words_per_row, rows: rows.into(), dispatch, full_dense }
    }

    #[inline]
    pub(crate) fn get(&self, tsid: u32) -> Option<&[u64]> {
        let tsid = tsid as usize;
        let id = match &self.dispatch {
            AcceptanceRowDispatch::Dense(ids) => *ids.get(tsid)?,
            AcceptanceRowDispatch::Runs { len, starts } => {
                if tsid >= *len { return None; }
                starts[starts.partition_point(|&(start, _)| start <= tsid).checked_sub(1)?].1
            }
        };
        match id {
            0 => None,
            1 => Some(self.full_dense.as_ref()),
            _ => {
                let start = (id as usize - 2).checked_mul(self.words_per_row)?;
                self.rows.get(start..start + self.words_per_row)
            }
        }
    }
}

pub(crate) fn empty_dense_words() -> DenseWords {
    Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice())
}

pub(crate) type InternalTokenBufMasks = Vec<(u16, u32)>;
/// Runtime-native fixed-width form of one sparse output-mask entry. The two-byte
/// pad makes the layout exactly eight bytes while keeping the hot fields at
/// their natural offsets; current artifacts can therefore bulk-copy the slab
/// without making commit pay bit shifts on every sparse replay.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PackedInternalTokenBufMask {
    pub(crate) word_idx: u16,
    pub(crate) _pad: u16,
    pub(crate) mask: u32,
}
const _: () = assert!(std::mem::size_of::<PackedInternalTokenBufMask>() == 8);

#[derive(Debug, Clone)]
pub(crate) struct BackedInternalTokenBufMasks {
    backing: Arc<Vec<u8>>,
    entries_start: usize,
    len: usize,
    aligned_base_addr: Option<usize>,
}

impl BackedInternalTokenBufMasks {
    pub(crate) fn new(
        backing: Arc<Vec<u8>>,
        entries_start: usize,
        len: usize,
    ) -> Result<Self, String> {
        let bytes = len
            .checked_mul(std::mem::size_of::<PackedInternalTokenBufMask>())
            .ok_or_else(|| "backed internal-token buffer-mask length overflow".to_owned())?;
        let end = entries_start
            .checked_add(bytes)
            .ok_or_else(|| "backed internal-token buffer-mask range overflow".to_owned())?;
        if end > backing.len() {
            return Err("backed internal-token buffer-mask range is outside artifact".to_owned());
        }
        let ptr = unsafe { backing.as_ptr().add(entries_start) };
        let aligned_base_addr = (cfg!(target_endian = "little")
            && ptr.align_offset(std::mem::align_of::<PackedInternalTokenBufMask>()) == 0)
            .then_some(ptr as usize);
        Ok(Self {
            backing,
            entries_start,
            len,
            aligned_base_addr,
        })
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn append_wire_bytes(&self, out: &mut Vec<u8>) {
        let byte_len = self.len * std::mem::size_of::<PackedInternalTokenBufMask>();
        out.extend_from_slice(&self.backing[self.entries_start..self.entries_start + byte_len]);
    }

    #[inline(always)]
    pub(crate) fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Option<&[PackedInternalTokenBufMask]> {
        if start > end || end > self.len {
            return None;
        }
        let base = self.aligned_base_addr? as *const PackedInternalTokenBufMask;
        // SAFETY: `new` validated the complete backing range and natural
        // alignment, and the retained Arc keeps the allocation alive.
        Some(unsafe { std::slice::from_raw_parts(base.add(start), end - start) })
    }

    #[inline(always)]
    pub(crate) fn for_each_range(
        &self,
        start: usize,
        end: usize,
        mut visit: impl FnMut(u16, u32),
    ) {
        debug_assert!(start <= end && end <= self.len);
        if start > end || end > self.len {
            return;
        }
        if let Some(entries) = self.slice(start, end) {
            for &entry in entries {
                visit(entry.word_idx, entry.mask);
            }
            return;
        }
        let entry_bytes = std::mem::size_of::<PackedInternalTokenBufMask>();
        let base = unsafe { self.backing.as_ptr().add(self.entries_start + start * entry_bytes) };
        for index in 0..(end - start) {
            let entry = unsafe {
                std::ptr::read_unaligned(
                    base.add(index * entry_bytes)
                        .cast::<PackedInternalTokenBufMask>(),
                )
            };
            if cfg!(target_endian = "little") {
                visit(entry.word_idx, entry.mask);
            } else {
                visit(u16::from_le(entry.word_idx), u32::from_le(entry.mask));
            }
        }
    }
}

/// Contiguous dense-mask matrix used by the word-group prefix cache. The old
/// `Vec<Box<[u32]>>` representation allocated one heap object per row; this
/// keeps the same row-slice API while requiring one aligned allocation.
const DENSE_BUF_MASK_ROWS_FLAT_MIN_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum DenseBufMaskRowsStorage {
    Rows(Vec<Box<[u32]>>),
    Flat(Box<[u32]>),
    #[serde(skip)]
    Backed {
        backing: Arc<Vec<u8>>,
        /// Address of the first u32 in `backing`. `from_backed` validates the
        /// complete range and alignment once, so hot row lookup does not need
        /// to redo checked byte-offset arithmetic for every prefix row.
        base_addr: usize,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DenseBufMaskRows {
    storage: DenseBufMaskRowsStorage,
    rows: usize,
    row_len: usize,
}

impl Default for DenseBufMaskRows {
    fn default() -> Self {
        Self {
            storage: DenseBufMaskRowsStorage::Rows(Vec::new()),
            rows: 0,
            row_len: 0,
        }
    }
}

impl DenseBufMaskRows {
    #[inline]
    pub(crate) fn prefer_flat(rows: usize, row_len: usize) -> bool {
        rows.checked_mul(row_len)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u32>()))
            .is_some_and(|bytes| bytes >= DENSE_BUF_MASK_ROWS_FLAT_MIN_BYTES)
    }

    pub(crate) fn from_flat(
        flat: Box<[u32]>,
        rows: usize,
        row_len: usize,
    ) -> Result<Self, String> {
        let expected = rows
            .checked_mul(row_len)
            .ok_or_else(|| "dense mask row dimensions overflow".to_owned())?;
        if flat.len() != expected {
            return Err("dense mask flat length does not match row dimensions".to_owned());
        }
        Ok(Self {
            storage: DenseBufMaskRowsStorage::Flat(flat),
            rows,
            row_len,
        })
    }

    /// Retain a current-format little-endian dense matrix directly in the
    /// artifact allocation. The byte range must be naturally aligned because
    /// callers consume rows as ordinary `&[u32]` slices on the hot path.
    pub(crate) fn from_backed(
        backing: Arc<Vec<u8>>,
        start: usize,
        rows: usize,
        row_len: usize,
    ) -> Result<Self, String> {
        if !cfg!(target_endian = "little") {
            return Err("backed dense mask rows require little-endian host".to_owned());
        }
        let values = rows
            .checked_mul(row_len)
            .ok_or_else(|| "backed dense mask dimensions overflow".to_owned())?;
        let bytes = values
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "backed dense mask byte length overflow".to_owned())?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| "backed dense mask range overflow".to_owned())?;
        if end > backing.len() {
            return Err("backed dense mask range is outside artifact".to_owned());
        }
        let ptr = unsafe { backing.as_ptr().add(start) };
        if ptr.align_offset(std::mem::align_of::<u32>()) != 0 {
            return Err("backed dense mask range is not u32-aligned".to_owned());
        }
        Ok(Self {
            storage: DenseBufMaskRowsStorage::Backed {
                backing,
                base_addr: ptr as usize,
            },
            rows,
            row_len,
        })
    }

    pub(crate) fn from_rows(rows: Vec<Box<[u32]>>) -> Result<Self, String> {
        let row_count = rows.len();
        let row_len = rows.first().map_or(0, |row| row.len());
        if rows.iter().any(|row| row.len() != row_len) {
            return Err("dense mask rows have inconsistent lengths".to_owned());
        }
        Ok(Self {
            storage: DenseBufMaskRowsStorage::Rows(rows),
            rows: row_count,
            row_len,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.rows
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.rows == 0
    }

    #[inline]
    pub(crate) fn row_len(&self) -> usize {
        self.row_len
    }

    /// Return the complete dense matrix as one contiguous runtime slice when
    /// storage is already flat/backed. This is the same memory consumed by hot
    /// row lookups; serialization can therefore copy it without rebuilding
    /// row objects.
    #[inline]
    pub(crate) fn as_contiguous(&self) -> Option<&[u32]> {
        match &self.storage {
            DenseBufMaskRowsStorage::Rows(_) => None,
            DenseBufMaskRowsStorage::Flat(flat) => Some(flat),
            DenseBufMaskRowsStorage::Backed {
                backing: _,
                base_addr,
            } => {
                let values = self.rows.checked_mul(self.row_len)?;
                let ptr = *base_addr as *const u32;
                // SAFETY: `from_backed` validated the complete byte range and
                // alignment, and the retained Arc keeps the allocation alive.
                Some(unsafe { std::slice::from_raw_parts(ptr, values) })
            }
        }
    }

    #[inline]
    pub(crate) fn get(&self, row: usize) -> Option<&[u32]> {
        if row >= self.rows {
            return None;
        }
        match &self.storage {
            DenseBufMaskRowsStorage::Rows(rows) => rows.get(row).map(Box::as_ref),
            DenseBufMaskRowsStorage::Flat(flat) => {
                let start = row * self.row_len;
                flat.get(start..start + self.row_len)
            }
            DenseBufMaskRowsStorage::Backed {
                backing: _,
                base_addr,
            } => {
                // `row < self.rows` and `from_backed` validated the complete
                // rows*row_len slab, so this multiplication and pointer offset
                // are within the retained allocation.
                let value_start = row * self.row_len;
                let ptr = unsafe { (*base_addr as *const u32).add(value_start) };
                // SAFETY: `from_backed` validated the full range and alignment;
                // row boundaries advance by a multiple of four bytes.
                Some(unsafe { std::slice::from_raw_parts(ptr, self.row_len) })
            }
        }
    }

    #[inline]
    pub(crate) fn last(&self) -> Option<&[u32]> {
        self.rows.checked_sub(1).and_then(|row| self.get(row))
    }

    #[inline]
    pub(crate) fn iter(&self) -> DenseBufMaskRowsIter<'_> {
        DenseBufMaskRowsIter {
            rows: self,
            next: 0,
        }
    }
}

impl std::ops::Index<usize> for DenseBufMaskRows {
    type Output = [u32];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("dense mask row index out of bounds")
    }
}

pub(crate) struct DenseBufMaskRowsIter<'a> {
    rows: &'a DenseBufMaskRows,
    next: usize,
}

impl<'a> Iterator for DenseBufMaskRowsIter<'a> {
    type Item = &'a [u32];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let row = self.rows.get(self.next)?;
        self.next += 1;
        Some(row)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rows.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DenseBufMaskRowsIter<'_> {}

impl<'a> IntoIterator for &'a DenseBufMaskRows {
    type Item = &'a [u32];
    type IntoIter = DenseBufMaskRowsIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Materialized token identities have one cache entry, with independent
/// transition bitmap and final-output strategy. The weak allocation guard
/// makes pointer keys stable without retaining the token ranges.
#[derive(Debug, Clone)]
pub(crate) struct WeightTokenMaskEntry {
    _identity: std::sync::Weak<RangeSetBlaze<u32>>,
    dense: Option<DenseWords>,
    pub(crate) output: WeightTokenOutput,
}

#[derive(Debug, Clone)]
pub(crate) enum WeightTokenOutput {
    None,
    DirectSparse,
    SparseBuf(Box<[(u16, u32)]>),
    DenseBuf(Box<[u32]>),
}

impl WeightTokenMaskEntry {
    pub(crate) fn new(
        tokens: &Arc<RangeSetBlaze<u32>>,
        dense: Option<DenseWords>,
        output: WeightTokenOutput,
    ) -> Self {
        Self {
            _identity: Arc::downgrade(tokens),
            dense,
            output,
        }
    }

    #[inline]
    pub(crate) fn dense(&self) -> Option<&DenseWords> {
        self.dense.as_ref()
    }
}

pub(crate) type WeightTokenMaskCache = FxHashMap<usize, WeightTokenMaskEntry>;

/// Dense masks for selected packed-DWA token sets.
///
/// Keep the rows in one contiguous slab rather than one `Arc<[u64]>` per
/// token set. Besides reducing allocator traffic, this makes the cache cheap
/// to persist and restore as two flat arrays while preserving O(1) lookup by
/// packed token-set id.
#[derive(Debug, Clone, Default)]
pub(crate) struct PackedDwaDenseWeightMaskCache {
    words_per_row: usize,
    row_by_token_set: Box<[u32]>,
    token_set_ids: Box<[u32]>,
    rows: DenseWords,
}

impl PackedDwaDenseWeightMaskCache {
    const MISSING_ROW: u32 = u32::MAX;

    pub(crate) fn from_rows(
        token_set_count: usize,
        words_per_row: usize,
        mut rows: Vec<(u32, DenseWords)>,
    ) -> Result<Self, String> {
        rows.sort_unstable_by_key(|(id, _)| *id);
        let mut row_by_token_set = vec![Self::MISSING_ROW; token_set_count];
        let mut token_set_ids = Vec::with_capacity(rows.len());
        let mut flat = Vec::with_capacity(rows.len().saturating_mul(words_per_row));
        for (row_index, (id, words)) in rows.into_iter().enumerate() {
            let slot = row_by_token_set
                .get_mut(id as usize)
                .ok_or_else(|| format!("packed DWA dense-mask token-set id {id} out of bounds"))?;
            if *slot != Self::MISSING_ROW {
                return Err(format!("duplicate packed DWA dense-mask token-set id {id}"));
            }
            if words.len() != words_per_row {
                return Err(format!(
                    "packed DWA dense-mask row has {} words; expected {words_per_row}",
                    words.len(),
                ));
            }
            *slot = u32::try_from(row_index)
                .map_err(|_| "too many packed DWA dense-mask rows".to_owned())?;
            token_set_ids.push(id);
            flat.extend_from_slice(words.as_ref());
        }
        Ok(Self {
            words_per_row,
            row_by_token_set: row_by_token_set.into_boxed_slice(),
            token_set_ids: token_set_ids.into_boxed_slice(),
            rows: Arc::from(flat.into_boxed_slice()),
        })
    }

    pub(crate) fn from_flat(
        token_set_count: usize,
        words_per_row: usize,
        token_set_ids: Vec<u32>,
        rows: Vec<u64>,
    ) -> Result<Self, String> {
        if rows.len() != token_set_ids.len().saturating_mul(words_per_row) {
            return Err(format!(
                "packed DWA dense-mask slab has {} words for {} rows of width {words_per_row}",
                rows.len(),
                token_set_ids.len(),
            ));
        }
        let mut row_by_token_set = vec![Self::MISSING_ROW; token_set_count];
        for (row_index, &id) in token_set_ids.iter().enumerate() {
            let slot = row_by_token_set
                .get_mut(id as usize)
                .ok_or_else(|| format!("packed DWA dense-mask token-set id {id} out of bounds"))?;
            if *slot != Self::MISSING_ROW {
                return Err(format!("duplicate packed DWA dense-mask token-set id {id}"));
            }
            *slot = u32::try_from(row_index)
                .map_err(|_| "too many packed DWA dense-mask rows".to_owned())?;
        }
        Ok(Self {
            words_per_row,
            row_by_token_set: row_by_token_set.into_boxed_slice(),
            token_set_ids: token_set_ids.into_boxed_slice(),
            rows: Arc::from(rows.into_boxed_slice()),
        })
    }

    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<&[u64]> {
        let row = *self.row_by_token_set.get(id as usize)?;
        if row == Self::MISSING_ROW {
            return None;
        }
        let start = (row as usize).checked_mul(self.words_per_row)?;
        self.rows.get(start..start + self.words_per_row)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.token_set_ids.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.token_set_ids.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub(crate) fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    #[inline]
    pub(crate) fn token_set_count(&self) -> usize {
        self.row_by_token_set.len()
    }

    #[inline]
    pub(crate) fn token_set_ids(&self) -> &[u32] {
        &self.token_set_ids
    }

    #[inline]
    pub(crate) fn flat_rows(&self) -> &[u64] {
        self.rows.as_ref()
    }
}
pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
const INLINE_DWA_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitionRow {
    Inline(SmallVec<[(i32, (u32, Weight)); 4]>),
    Hash(FxHashMap<i32, (u32, Weight)>),
    Packed(DwaTransitionMap),
}

impl FastDwaTransitionRow {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (i32, (u32, Weight))>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        if entries.len() <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    pub(crate) fn from_exact_entries(
        len: usize,
        entries: impl IntoIterator<Item = (i32, (u32, Weight))>,
    ) -> Self {
        if len <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries.into_iter().collect())
        } else {
            let mut map = FxHashMap::default();
            map.reserve(len);
            map.extend(entries);
            Self::Hash(map)
        }
    }

    pub(crate) fn from_packed(row: DwaTransitionMap) -> Self {
        debug_assert!(row.is_packed());
        Self::Packed(row)
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Inline(entries) => entries.is_empty(),
            Self::Hash(entries) => entries.is_empty(),
            Self::Packed(row) => row.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<(u32, &Weight)> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, (target, weight))| {
                    (candidate == label).then_some((*target, weight))
                }),
            Self::Hash(entries) => entries.get(label).map(|(target, weight)| (*target, weight)),
            Self::Packed(row) => row.get_entry(label),
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitions {
    Direct(Vec<FastDwaTransitionRow>),
    Shared {
        rows: Vec<FastDwaTransitionRow>,
        state_rows: Vec<u32>,
    },
}

impl Default for FastDwaTransitions {
    fn default() -> Self {
        Self::Direct(Vec::new())
    }
}

impl FastDwaTransitions {
    pub(crate) fn direct(rows: Vec<FastDwaTransitionRow>) -> Self {
        Self::Direct(rows)
    }

    pub(crate) fn shared(rows: Vec<FastDwaTransitionRow>, state_rows: Vec<u32>) -> Self {
        Self::Shared { rows, state_rows }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Direct(rows) => rows.len(),
            Self::Shared { state_rows, .. } => state_rows.len(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, state: usize) -> Option<&FastDwaTransitionRow> {
        match self {
            Self::Direct(rows) => rows.get(state),
            Self::Shared { rows, state_rows } => state_rows
                .get(state)
                .and_then(|&row| rows.get(row as usize)),
        }
    }
}

impl std::ops::Index<usize> for FastDwaTransitions {
    type Output = FastDwaTransitionRow;

    #[inline]
    fn index(&self, state: usize) -> &Self::Output {
        match self {
            Self::Direct(rows) => &rows[state],
            Self::Shared { rows, state_rows } => &rows[state_rows[state] as usize],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseMask {
    Full,
    Dense {
        words: DenseWords,
        start: usize,
        end: usize,
    },
    Empty,
}

#[derive(Debug, Clone)]
pub(crate) struct IndexedDagDenseTransition {
    pub(crate) target: u32,
    pub(crate) masks: IndexedDagDenseTransitionMasks,
}

const INLINE_INDEXED_DAG_TSID_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseTransitionMasks {
    Full,
    Inline(SmallVec<[(u32, IndexedDagDenseMask); 2]>),
    Hash(FxHashMap<u32, IndexedDagDenseMask>),
}

static INDEXED_DAG_FULL_MASK: IndexedDagDenseMask = IndexedDagDenseMask::Full;
static INDEXED_DAG_EMPTY_MASK: IndexedDagDenseMask = IndexedDagDenseMask::Empty;

impl IndexedDagDenseTransitionMasks {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (u32, IndexedDagDenseMask)>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 2]>>();
        if entries.len() <= INLINE_INDEXED_DAG_TSID_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    #[inline]
    pub(crate) fn get(&self, tsid: u32) -> &IndexedDagDenseMask {
        match self {
            Self::Full => &INDEXED_DAG_FULL_MASK,
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, mask)| (*candidate == tsid).then_some(mask))
                .unwrap_or(&INDEXED_DAG_EMPTY_MASK),
            Self::Hash(entries) => entries.get(&tsid).unwrap_or(&INDEXED_DAG_EMPTY_MASK),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseTransitionRow {
    Inline(SmallVec<[(i32, IndexedDagDenseTransition); 4]>),
    Hash(FxHashMap<i32, IndexedDagDenseTransition>),
}

impl IndexedDagDenseTransitionRow {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (i32, IndexedDagDenseTransition)>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        if entries.len() <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<&IndexedDagDenseTransition> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, transition)| (candidate == label).then_some(transition)),
            Self::Hash(entries) => entries.get(label),
        }
    }
}

pub(crate) type IndexedDagDenseTransitions = Vec<IndexedDagDenseTransitionRow>;

#[derive(Debug, Clone)]
pub(crate) enum FastTokenizerTransitions {
    Dense(Vec<[u32; 256]>),
    Classed(crate::automata::lexer::observation::ByteRows),
    /// Compact exact dense DFA rows for tokenizers with fewer than 32,768
    /// states. The high bit marks a target state with at least one finalizer;
    /// u16::MAX is the dead-transition sentinel.
    Flat16 {
        transitions: Arc<[u16]>,
        finalizer_code: Arc<[u32]>,
        single_finalizer_continues: Arc<[u8]>,
    },
    /// Exact dense DFA rows for the strict full-vocabulary walker when the
    /// tokenizer no longer fits the 15-bit Flat16 coordinate. The high bit
    /// marks a target state with at least one finalizer; u32::MAX is dead.
    Flat32 {
        transitions: Arc<[u32]>,
        finalizer_code: Arc<[u32]>,
        single_finalizer_continues: Arc<[u8]>,
    },
    /// Runtime tokenizer already owns an allocation-light exact transition
    /// table; call through instead of rebuilding a second dense table.
    Fallback(usize),
    Hybrid {
        state_to_dense_row: Vec<u32>,
        dense_rows: Vec<[u32; 256]>,
    },
}

impl Default for FastTokenizerTransitions {
    fn default() -> Self {
        Self::Dense(Vec::new())
    }
}

impl FastTokenizerTransitions {
    fn full_walk_finalizer_metadata(tokenizer: &Tokenizer) -> (Arc<[u32]>, Arc<[u8]>) {
        const NONE: u32 = u32::MAX;
        const MULTI: u32 = u32::MAX - 1;
        let num_states = tokenizer.num_states();
        let state_metadata = |state: u32| {
            let finalizers = tokenizer.matched_terminals_slice(state);
            let code = match finalizers {
                [] => NONE,
                [terminal] => *terminal,
                _ => MULTI,
            };
            let continues = match finalizers {
                [terminal]
                    if tokenizer
                        .possible_future_terminals(state)
                        .contains(*terminal as usize) => 1u8,
                _ => 0u8,
            };
            (code, continues)
        };
        let metadata = if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            (0..num_states)
                .into_par_iter()
                .map(state_metadata)
                .collect::<Vec<_>>()
        } else {
            (0..num_states).map(state_metadata).collect::<Vec<_>>()
        };
        let (finalizer_code, single_finalizer_continues): (Vec<_>, Vec<_>) =
            metadata.into_iter().unzip();
        (
            Arc::from(finalizer_code),
            Arc::from(single_finalizer_continues),
        )
    }

    pub(crate) fn flat16_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000 {
            return None;
        }
        // Build finalizer metadata once per state before populating transition
        // cells. The old implementation called `matched_terminals_slice()` for
        // every edge merely to set the high-bit hint, then scanned every state
        // again below to construct these exact metadata arrays. Fresh compiler
        // DFAs make that per-edge lookup materially more expensive than loaded
        // packed tokenizers.
        let (finalizer_code, single_finalizer_continues) =
            Self::full_walk_finalizer_metadata(tokenizer);
        let mut flat = vec![u16::MAX; num_states as usize * 256];
        let fill_row = |(state, row): (usize, &mut [u16])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                let mut encoded = u16::try_from(target)
                    .expect("flat16 tokenizer target exceeds 15-bit state coordinate");
                if finalizer_code[target as usize] != u32::MAX {
                    encoded |= 0x8000;
                }
                row[byte as usize] = encoded;
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat16 {
            transitions: Arc::from(flat),
            finalizer_code,
            single_finalizer_continues,
        })
    }

    /// Dense u16 transition slab for ordinary tokenizer execution. Unlike the
    /// strict full-vocabulary walker, commit/scan only asks this object for the
    /// target state, so constructing per-state finalizer certificates (and
    /// probing finalizers for every transition target) is pure load-time waste.
    pub(crate) fn flat16_transitions_only_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000 {
            return None;
        }
        let mut flat = vec![u16::MAX; num_states as usize * 256];
        let fill_row = |(state, row): (usize, &mut [u16])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                row[byte as usize] = u16::try_from(target)
                    .expect("flat16 tokenizer target exceeds 15-bit state coordinate");
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat16 {
            transitions: Arc::from(flat),
            finalizer_code: Arc::from([]),
            single_finalizer_continues: Arc::from([]),
        })
    }

    pub(crate) fn flat32_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000_0000 {
            return None;
        }
        let (finalizer_code, single_finalizer_continues) =
            Self::full_walk_finalizer_metadata(tokenizer);
        let cell_count = (num_states as usize).checked_mul(256)?;
        let mut flat = vec![u32::MAX; cell_count];
        let fill_row = |(state, row): (usize, &mut [u32])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                debug_assert!(target < 0x8000_0000);
                let mut encoded = target;
                if finalizer_code[target as usize] != u32::MAX {
                    encoded |= 0x8000_0000;
                }
                row[byte as usize] = encoded;
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat32 {
            transitions: Arc::from(flat),
            finalizer_code,
            single_finalizer_continues,
        })
    }

    /// Build the dense transition representation used only by the strict full
    /// vocabulary walker. Flat16 is preferred whenever possible; Flat32 is
    /// available for larger deterministic mask coordinates. `max_bytes`
    /// bounds only transition-cell storage, not the small metadata arrays.
    pub(crate) fn full_walk_dense_for(tokenizer: &Tokenizer, max_bytes: usize) -> Option<Self> {
        let num_states = tokenizer.num_states() as usize;
        let flat16_bytes = num_states
            .checked_mul(256)?
            .checked_mul(std::mem::size_of::<u16>())?;
        if num_states < 0x8000 && flat16_bytes <= max_bytes {
            return Self::flat16_for(tokenizer);
        }
        let flat32_bytes = num_states
            .checked_mul(256)?
            .checked_mul(std::mem::size_of::<u32>())?;
        if num_states < 0x8000_0000 && flat32_bytes <= max_bytes {
            return Self::flat32_for(tokenizer);
        }
        None
    }

    #[inline]
    pub(crate) fn transition(
        &self,
        tokenizer: &Tokenizer,
        state: u32,
        byte: u8,
    ) -> u32 {
        match self {
            Self::Dense(rows) => rows
                .get(state as usize)
                .map_or_else(
                    || tokenizer.get_transition(state, byte),
                    |row| row[byte as usize],
                ),
            Self::Classed(rows) => {
                if (state as usize) < rows.num_states() {
                    rows.trans(state as usize, byte as usize)
                } else {
                    tokenizer.get_transition(state, byte)
                }
            },
            Self::Flat16 { transitions, .. } => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| transitions.get(index))
                .copied()
                .map(|target| {
                    if target == u16::MAX {
                        u32::MAX
                    } else {
                        u32::from(target & 0x7fff)
                    }
                })
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Flat32 { transitions, .. } => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| transitions.get(index))
                .copied()
                .map(|target| {
                    if target == u32::MAX {
                        u32::MAX
                    } else {
                        target & 0x7fff_ffff
                    }
                })
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Fallback(_) => tokenizer.get_transition(state, byte),
            Self::Hybrid {
                state_to_dense_row,
                dense_rows,
            } => {
                let dense = state_to_dense_row
                    .get(state as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if dense == u32::MAX {
                    tokenizer.get_transition(state, byte)
                } else {
                    dense_rows[dense as usize][byte as usize]
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(rows) => rows.len(),
            Self::Classed(rows) => rows.num_states(),
            Self::Flat16 { transitions, .. } => transitions.len() / 256,
            Self::Flat32 { transitions, .. } => transitions.len() / 256,
            Self::Fallback(len) => *len,
            Self::Hybrid {
                state_to_dense_row,
                ..
            } => state_to_dense_row.len(),
        }
    }

    /// Reuse the consumed parent's fast transition rows and append rebased
    /// child rows. Compressed child states remain sparse and fall back to the
    /// merged tokenizer, whose compressed segments have already been rebased.
    pub(crate) fn append_rebased_children(
        self,
        children: &[(&FastTokenizerTransitions, u32)],
    ) -> Option<Self> {
        fn physical_rows(rows: &crate::automata::lexer::observation::ByteRows) -> impl Iterator<Item = [u32; 256]> + '_ {
            (0..rows.num_states()).map(|state| std::array::from_fn(|byte| rows.trans(state, byte)))
        }

        fn rebased_row(row: &[u32; 256], offset: u32) -> [u32; 256] {
            let mut rebased = *row;
            for target in rebased.iter_mut() {
                if *target != u32::MAX {
                    *target = target.checked_add(offset)
                        .expect("composed tokenizer fast-transition target overflow");
                }
            }
            rebased
        }

        let all_dense = children
            .iter()
            .all(|(child, _)| matches!(child, FastTokenizerTransitions::Dense(_)));
        match self {
            Self::Dense(mut rows) if all_dense => {
                for (child, offset) in children {
                    if *offset as usize != rows.len() {
                        return None;
                    }
                    let Self::Dense(child_rows) = child else { unreachable!() };
                    rows.extend(child_rows.iter().map(|row| rebased_row(row, *offset)));
                }
                Some(Self::Dense(rows))
            }
            parent => {
                let (mut state_to_dense_row, mut dense_rows) = match parent {
                    Self::Dense(rows) => {
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Classed(classed) => {
                        let rows = physical_rows(&classed).collect::<Vec<_>>();
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Flat16 { .. } | Self::Flat32 { .. } | Self::Fallback(_) => return None,
                    Self::Hybrid {
                        state_to_dense_row,
                        dense_rows,
                    } => (state_to_dense_row, dense_rows),
                };
                for (child, offset) in children {
                    if *offset as usize != state_to_dense_row.len() {
                        return None;
                    }
                    match child {
                        Self::Flat16 { .. } | Self::Flat32 { .. } | Self::Fallback(_) => return None,
                        Self::Dense(rows) => {
                            for row in rows {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Classed(classed) => {
                            for row in physical_rows(classed) {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(&row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Hybrid {
                            state_to_dense_row: child_mapping,
                            dense_rows: child_rows,
                        } => {
                            for &child_dense in child_mapping {
                                if child_dense == u32::MAX {
                                    state_to_dense_row.push(u32::MAX);
                                } else {
                                    let row = child_rows.get(child_dense as usize)?;
                                    let dense = dense_rows.len() as u32;
                                    dense_rows.push(rebased_row(row, *offset));
                                    state_to_dense_row.push(dense);
                                }
                            }
                        }
                    }
                }
                Some(Self::Hybrid {
                    state_to_dense_row,
                    dense_rows,
                })
            }
        }
    }
}
pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<CommitTemplateDfas>>>;

const INLINE_TEMPLATE_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone, Default)]
pub(crate) enum FastTemplateTransitionRow {
    #[default]
    Empty,
    Inline(SmallVec<[(i32, u32); 4]>),
    Hash(FxHashMap<i32, u32>),
}

impl FastTemplateTransitionRow {
    pub(crate) fn from_entries(entries: impl IntoIterator<Item = (i32, u32)>, count: usize) -> Self {
        if count > INLINE_TEMPLATE_TRANSITION_LIMIT {
            let mut row = FxHashMap::with_capacity_and_hasher(count, Default::default());
            row.extend(entries);
            debug_assert_eq!(row.len(), count);
            return Self::Hash(row);
        }
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        debug_assert_eq!(entries.len(), count);
        match entries.len() {
            0 => Self::Empty,
            len if len <= INLINE_TEMPLATE_TRANSITION_LIMIT => Self::Inline(entries),
            _ => Self::Hash(entries.into_iter().collect()),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: i32) -> Option<u32> {
        match self {
            Self::Empty => None,
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, target)| (*candidate == label).then_some(*target)),
            Self::Hash(entries) => entries.get(&label).copied(),
        }
    }

    #[inline]
    pub(crate) fn for_each(&self, mut f: impl FnMut(i32, u32)) {
        match self {
            Self::Empty => {}
            Self::Inline(entries) => {
                for &(label, target) in entries {
                    f(label, target);
                }
            }
            Self::Hash(entries) => {
                for (&label, &target) in entries {
                    f(label, target);
                }
            }
        }
    }
}

#[cfg(test)]
mod fast_template_row_construction_tests {
    use super::*;

    #[test]
    fn sized_rows_retain_lookup_iteration_and_hash_capacity() {
        for count in [0usize, 1, 4, 8, 9, 16, 65, 129] {
            let pairs = (0..count).map(|i| (i as i32 * 7 - 400, i as u32)).collect::<Vec<_>>();
            let new = FastTemplateTransitionRow::from_entries(pairs.iter().copied(), count);
            let entries = pairs.iter().copied().collect::<SmallVec<[_; 4]>>();
            let old = match count {
                0 => FastTemplateTransitionRow::Empty,
                n if n <= INLINE_TEMPLATE_TRANSITION_LIMIT => FastTemplateTransitionRow::Inline(entries),
                _ => FastTemplateTransitionRow::Hash(entries.into_iter().collect()),
            };
            for key in -420..520 {
                assert_eq!(new.get(key), old.get(key));
            }
            let mut old_order = Vec::new(); let mut new_order = Vec::new();
            old.for_each(|label, target| old_order.push((label, target)));
            new.for_each(|label, target| new_order.push((label, target)));
            assert_eq!(new_order, old_order);
            if let (FastTemplateTransitionRow::Hash(a), FastTemplateTransitionRow::Hash(b)) = (&new, &old) {
                assert_eq!(a.capacity(), b.capacity());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(crate) struct SpecialTokenTerminal {
    pub(crate) terminal_id: TerminalID,
    /// Inclusive lower and upper bounds of one exact-token event.
    pub(crate) token_id: u32,
    pub(crate) token_id_end: u32,
}

// The interval width is new wire data. Reject reversed endpoints while decoding
// before they can reach RangeSet construction, mask allocation or composition.
impl<'de> serde::Deserialize<'de> for SpecialTokenTerminal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire { terminal_id: TerminalID, token_id: u32, token_id_end: u32 }
        let wire = <Wire as serde::Deserialize>::deserialize(deserializer)?;
        if wire.token_id > wire.token_id_end {
            return Err(serde::de::Error::custom("reversed exact-token interval"));
        }
        Ok(Self { terminal_id: wire.terminal_id, token_id: wire.token_id,
            token_id_end: wire.token_id_end })
    }
}

impl SpecialTokenTerminal {
    pub(crate) fn contains(&self, token_id: u32) -> bool {
        self.token_id <= token_id && token_id <= self.token_id_end
    }

    pub(crate) fn token_ids(&self) -> std::ops::RangeInclusive<u32> {
        self.token_id..=self.token_id_end
    }

    /// Union a live interval into the original-token mask in word-sized chunks.
    pub(crate) fn or_mask(&self, mask: &mut [u32]) {
        let first = self.token_id as usize / 32;
        if first >= mask.len() || self.token_id > self.token_id_end { return; }
        let last = (self.token_id_end as usize / 32).min(mask.len() - 1);
        let first_bits = u32::MAX << (self.token_id % 32);
        let last_bits = if last == self.token_id_end as usize / 32 {
            u32::MAX >> (31 - self.token_id_end % 32)
        } else { u32::MAX };
        if first == last {
            mask[first] |= first_bits & last_bits;
        } else {
            mask[first] |= first_bits;
            mask[first + 1..last].fill(u32::MAX);
            mask[last] |= last_bits;
        }
    }
}

/// Compact runtime-only vocabulary trie. It deliberately stores only the
/// information dynamic mask traversal consumes: compressed byte edges, child
/// ranges, and canonical token leaves.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieNode {
    pub(crate) token_id: Option<u32>,
    pub(crate) first_child: u32,
    pub(crate) child_len: u32,
    /// Canonical token ids below this node occupy one contiguous range in
    /// `DynamicMaskTrie::subtree_tokens`.
    pub(crate) subtree_token_start: u32,
    pub(crate) subtree_token_end: u32,
    /// Union of every byte on every edge strictly below this node.
    pub(crate) subtree_bytes: [u64; 4],
    /// Bytes that may occur next on some non-empty token suffix below this
    /// node. Unlike `subtree_bytes`, this records only the first consumed byte;
    /// zero-byte structural layout edges transparently inherit their child's
    /// first-byte set. Dynamic masking uses it to reject whole vocabulary
    /// layout classes before entering the radix walk when the current lexer
    /// configuration cannot consume any of their first bytes.
    pub(crate) subtree_first_bytes: [u64; 4],
    /// Number of vocabulary bytes consumed from the global trie root to reach
    /// this node. Structural partition edges have length zero and therefore do
    /// not affect it. This lets finite-horizon root-state certificates be
    /// reused only for subtrees whose *complete token strings* fit the proof
    /// horizon.
    pub(crate) prefix_byte_len: u32,
    /// Maximum number of token bytes still reachable strictly below this node.
    /// This is a runtime-only certificate aid: dynamic masking can prove that
    /// a lexer configuration stays safely live for this bounded horizon and
    /// accept the whole subtree without walking every token edge.
    pub(crate) subtree_max_byte_len: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct DynamicMaskTrieEdge {
    pub(crate) byte_start: u32,
    pub(crate) byte_len: u32,
    pub(crate) child: u32,
}

/// One radix edge in depth-first preorder. `subtree_end` is the first walk
/// entry after the child subtree, so a failed edge or accepted whole subtree
/// can be skipped with one index assignment.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieWalkEdge {
    pub(crate) byte_start: u32,
    pub(crate) child: u32,
    pub(crate) subtree_end: u32,
    pub(crate) byte_len: u16,
    pub(crate) parent_depth: u16,
}

/// One sequential operation in the strict full-vocabulary byte walk. Every
/// radix edge contributes at least one op; non-empty edges contribute one op
/// per consumed byte. This is purely a flatter view of the vocabulary trie: it
/// does not omit or summarize any edge.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieFullWalkOp {
    meta: u32,
}

const _: () = assert!(std::mem::size_of::<DynamicMaskTrieFullWalkOp>() == 4);

impl DynamicMaskTrieFullWalkOp {
    const CONSUME: u32 = 1 << 24;
    const START: u32 = 1 << 25;
    const END: u32 = 1 << 26;
    const TOKEN: u32 = 1 << 27;

    #[inline(always)]
    pub(crate) fn byte(&self) -> u8 { self.meta as u8 }
    #[inline(always)]
    pub(crate) fn parent_depth(&self) -> u16 { ((self.meta >> 8) & 0xffff) as u16 }
    #[inline(always)]
    pub(crate) fn consumes_byte(&self) -> bool { self.meta & Self::CONSUME != 0 }
    #[inline(always)]
    pub(crate) fn starts_edge(&self) -> bool { self.meta & Self::START != 0 }
    #[inline(always)]
    pub(crate) fn ends_edge(&self) -> bool { self.meta & Self::END != 0 }
    #[inline(always)]
    pub(crate) fn child_is_token(&self) -> bool { self.meta & Self::TOKEN != 0 }
}

/// Rank index over edge-start positions in the strict byte program.
/// One bit per operation plus one prefix rank per 64 operations replaces
/// the previous u32 edge ID stored for every operation.
#[derive(Debug, Clone, Copy, Default)]
struct FullWalkEdgeBlock {
    starts: u64,
    preceding_edges: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskTrie {
    pub(crate) nodes: Vec<DynamicMaskTrieNode>,
    edge_bytes: Vec<u8>,
    subtree_tokens: Vec<u32>,
    walk_edges: Vec<DynamicMaskTrieWalkEdge>,
    full_walk_ops: Vec<DynamicMaskTrieFullWalkOp>,
    full_walk_edge_blocks: Vec<FullWalkEdgeBlock>,
    /// Starting full-walk op index for each DFS-preorder walk edge, plus one
    /// sentinel at the end. Combined with `DynamicMaskTrieWalkEdge::subtree_end`
    /// this gives an O(1) exact jump past a dead child subtree.
    full_walk_edge_op_starts: Vec<u32>,
    /// For an ordinary, unpartitioned radix root, map each possible first byte
    /// directly to the first strict-walk op for its root child. `u32::MAX`
    /// means the vocabulary has no token beginning with that byte. Structural
    /// zero-byte roots deliberately leave this unavailable.
    full_walk_root_byte_op_starts: Option<Box<[u32; 256]>>,
    /// Maximum structural radix-edge parent depth encoded by `full_walk_ops`.
    /// This is deliberately distinct from token byte length: one long
    /// compressed edge consumes many bytes without requiring additional DFS
    /// stack slots in the strict full walker.
    full_walk_max_parent_depth: u16,
    /// Declared regular-language partition id for each zero-byte structural
    /// child of the true root. An empty table disables partition certificates.
    root_layout_classes: Vec<u16>,
    /// True iff every complete token in the corresponding structural root
    /// class is valid UTF-8. Logical-scalar subtree proofs require this exact
    /// vocabulary property before treating non-ASCII bytes as UTF-8 scalars.
    root_layout_all_valid_utf8: Vec<bool>,
}

/// Immediate siblings are reached by skipping entire preorder subtrees.
#[derive(Clone, Copy)]
pub(crate) struct TrieChildren<'a> {
    edges: &'a [DynamicMaskTrieWalkEdge],
    next: usize,
    remaining: usize,
}
impl<'a> Iterator for TrieChildren<'a> {
    type Item = &'a DynamicMaskTrieWalkEdge;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 { return None; }
        let edge = &self.edges[self.next];
        self.next = edge.subtree_end as usize;
        self.remaining -= 1;
        Some(edge)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for TrieChildren<'_> {}
impl TrieChildren<'_> {
    pub(crate) fn is_empty(&self) -> bool { self.remaining == 0 }
    pub(crate) fn iter(&self) -> Self { *self }
}

/// Stable structural class used by the dynamic-mask radix trie.
///
/// This is deliberately *only* the declared regular-language partition id.
/// Older revisions refined it with arbitrary first-byte/character flags; that
/// made one structural root cease to correspond to a first-class language and
/// therefore prevented exact language-containment proofs. Any further runtime
/// acceleration must be represented as an explicit regular language instead.
pub(crate) fn dynamic_mask_vocab_layout_class(base_partition: u8, _bytes: &[u8]) -> u16 {
    u16::from(base_partition)
}

pub(crate) const DYNAMIC_MASK_LLG_MASTER_CACHE_ID: u32 = 0x200;
const DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT: u16 = 1 << 15;
const DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK: u16 = DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT - 1;

/// Structural class for the dynamic-radius LLG vocabulary trie. The low bits
/// are the exact number of Unicode scalar values in a whole token matching the
/// regex-defined safe-string language (`0` means not in that language); the high
/// bit records whole-token whitespace-regex membership. These are language
/// properties only. The trie deliberately does not mark them as compiler
/// partition languages, so generic partition certificates cannot reinterpret
/// the encoding.
pub(crate) fn dynamic_mask_llg_master_layout_class(
    safe_chars: u16,
    whitespace: bool,
) -> u16 {
    debug_assert!(safe_chars <= DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK);
    safe_chars | if whitespace { DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT } else { 0 }
}

#[inline(always)]
pub(crate) fn dynamic_mask_llg_master_safe_chars(class: u16) -> u16 {
    class & DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK
}

#[inline(always)]
pub(crate) fn dynamic_mask_llg_master_is_whitespace(class: u16) -> bool {
    class & DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT != 0
}

impl DynamicMaskTrie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![DynamicMaskTrieNode::default()],
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
            full_walk_ops: Vec::new(),
            full_walk_edge_blocks: Vec::new(),
            full_walk_edge_op_starts: Vec::new(),
            full_walk_root_byte_op_starts: None,
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        }
    }


    #[inline]
    pub(crate) fn root_layout_class(&self, root_slot: usize) -> Option<u16> {
        self.root_layout_classes.get(root_slot).copied()
    }

    #[inline]
    pub(crate) fn root_layout_all_valid_utf8(&self, root_slot: usize) -> bool {
        self.root_layout_all_valid_utf8
            .get(root_slot)
            .copied()
            .unwrap_or(false)
    }
    #[inline]
    pub(crate) fn node(&self, node: u32) -> &DynamicMaskTrieNode {
        &self.nodes[node as usize]
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub(crate) fn children(&self, node: u32) -> TrieChildren<'_> {
        let node = self.node(node);
        TrieChildren {
            edges: &self.walk_edges,
            next: node.first_child as usize,
            remaining: node.child_len as usize,
        }
    }

    #[inline]
    pub(crate) fn edge_bytes(&self, edge: &DynamicMaskTrieWalkEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        &self.edge_bytes[start..start + edge.byte_len as usize]
    }

    #[inline]
    pub(crate) fn walk_edges(&self) -> &[DynamicMaskTrieWalkEdge] {
        &self.walk_edges
    }

    #[inline]
    pub(crate) fn walk_edge_bytes(&self, edge: &DynamicMaskTrieWalkEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        let end = start + edge.byte_len as usize;
        &self.edge_bytes[start..end]
    }

    #[inline]
    pub(crate) fn full_walk_ops(&self) -> &[DynamicMaskTrieFullWalkOp] {
        &self.full_walk_ops
    }

    /// Every full-walk op consumes one byte iff there are no synthetic
    /// empty-edge ops. Non-empty radix edges contribute exactly one op per
    /// byte; empty edges contribute one op and zero bytes.
    #[inline(always)]
    pub(crate) fn full_walk_all_consume(&self) -> bool {
        self.full_walk_ops.len() == self.edge_bytes.len()
    }

    #[inline(always)]
    pub(crate) fn full_walk_dead_subtree(&self, op_index: usize) -> (u32, u32) {
        debug_assert!(op_index < self.full_walk_ops.len());
        let block = unsafe { self.full_walk_edge_blocks.get_unchecked(op_index >> 6) };
        let through = u64::MAX >> (63 - (op_index & 63));
        let edge_index = (block.preceding_edges + (block.starts & through).count_ones() - 1) as usize;
        let edge = unsafe { *self.walk_edges.get_unchecked(edge_index) };
        let subtree_end_edge = edge.subtree_end as usize;
        debug_assert!(subtree_end_edge < self.full_walk_edge_op_starts.len());
        let subtree_end_op =
            unsafe { *self.full_walk_edge_op_starts.get_unchecked(subtree_end_edge) };
        (edge.child, subtree_end_op)
    }

    #[inline]
    pub(crate) fn full_walk_max_parent_depth(&self) -> u16 {
        self.full_walk_max_parent_depth
    }

    #[inline(always)]
    pub(crate) fn has_full_walk_root_byte_index(&self) -> bool {
        self.full_walk_root_byte_op_starts.is_some()
    }

    #[inline(always)]
    pub(crate) fn full_walk_root_byte_range(&self, byte: u8) -> Option<(u32, u32, usize)> {
        let starts = self.full_walk_root_byte_op_starts.as_ref()?;
        let start_op = starts[byte as usize];
        if start_op == u32::MAX {
            return None;
        }
        let (child, end_op) = self.full_walk_dead_subtree(start_op as usize);
        let root_token_offset = usize::from(self.node(0).token_id.is_some());
        let marker_start = self
            .subtree_token_index_range(child)
            .start
            .saturating_sub(root_token_offset);
        Some((start_op, end_op, marker_start))
    }

    #[inline]
    pub(crate) fn subtree_tokens(&self, node: u32) -> &[u32] {
        let node = self.node(node);
        &self.subtree_tokens
            [node.subtree_token_start as usize..node.subtree_token_end as usize]
    }

    #[inline]
    pub(crate) fn subtree_token_index_range(&self, node: u32) -> std::ops::Range<usize> {
        let node = self.node(node);
        node.subtree_token_start as usize..node.subtree_token_end as usize
    }

    #[inline]
    pub(crate) fn all_subtree_tokens(&self) -> &[u32] {
        &self.subtree_tokens
    }

    #[inline]
    pub(crate) fn subtree_bytes(&self, node: u32) -> [u64; 4] {
        self.node(node).subtree_bytes
    }

    #[inline]
    pub(crate) fn subtree_first_bytes(&self, node: u32) -> [u64; 4] {
        self.node(node).subtree_first_bytes
    }

    #[inline]
    pub(crate) fn subtree_max_byte_len(&self, node: u32) -> u32 {
        self.node(node).subtree_max_byte_len
    }

    #[inline]
    pub(crate) fn subtree_max_total_byte_len(&self, node: u32) -> u32 {
        let node = self.node(node);
        node.prefix_byte_len.saturating_add(node.subtree_max_byte_len)
    }

    #[inline]
    pub(crate) fn edge_bytes_len(&self) -> usize { self.edge_bytes.len() }

    /// Build each node and its incoming edge in preorder. The incoming edge
    /// of node n > 0 is walk entry n - 1; there is no second adjacency list.
    fn begin_node(&mut self, parent: Option<u32>, token: Option<u32>, byte_len: usize) -> u32 {
        let start = u32::try_from(self.subtree_tokens.len()).expect("too many trie tokens");
        let mut node = DynamicMaskTrieNode {
            token_id: token, subtree_token_start: start, ..Default::default()
        };
        if let Some(token) = token { self.subtree_tokens.push(token); }
        let Some(parent) = parent else {
            self.nodes[0] = node;
            return 0;
        };
        let id = u32::try_from(self.nodes.len()).expect("too many trie nodes");
        let byte_len = u16::try_from(byte_len).expect("dynamic mask trie radix edge exceeds u16 length");
        let parent_depth = if parent == 0 { 0 } else {
            self.walk_edges[parent as usize - 1].parent_depth.checked_add(1)
                .expect("dynamic mask trie depth exceeds u16")
        };
        assert!(parent_depth < u16::MAX, "dynamic mask trie depth exceeds u16");
        let parent = &mut self.nodes[parent as usize];
        node.prefix_byte_len = parent.prefix_byte_len.checked_add(u32::from(byte_len))
            .expect("dynamic mask trie token byte length exceeds u32");
        if parent.child_len == 0 { parent.first_child = id - 1; }
        parent.child_len += 1;
        self.nodes.push(node);
        self.walk_edges.push(DynamicMaskTrieWalkEdge {
            byte_start: 0, child: id, subtree_end: 0, byte_len, parent_depth,
        });
        id
    }

    /// Finish the subtree after all children. Append incoming bytes in
    /// postorder, matching the legacy radix byte arena layout.
    fn finish_node(&mut self, id: u32, incoming: &[u8]) {
        if id != 0 {
            let edge = &mut self.walk_edges[id as usize - 1];
            edge.byte_start = u32::try_from(self.edge_bytes.len()).expect("trie bytes exceed u32");
            edge.subtree_end = u32::try_from(self.nodes.len() - 1).expect("too many trie edges");
            self.edge_bytes.extend_from_slice(incoming);
        }
        let mut bytes = [0u64; 4];
        let mut first = [0u64; 4];
        let mut max_len = 0u32;
        for edge in self.children(id) {
            let child = self.node(edge.child);
            let segment = self.edge_bytes(edge);
            for &byte in segment {
                bytes[byte as usize >> 6] |= 1 << (byte & 63);
            }
            for (target, value) in bytes.iter_mut().zip(child.subtree_bytes) { *target |= value; }
            if let Some(&byte) = segment.first() {
                first[byte as usize >> 6] |= 1 << (byte & 63);
            } else {
                for (target, value) in first.iter_mut().zip(child.subtree_first_bytes) { *target |= value; }
            }
            max_len = max_len.max(u32::from(edge.byte_len).checked_add(child.subtree_max_byte_len)
                .expect("dynamic mask trie token byte length exceeds u32"));
        }
        let node = &mut self.nodes[id as usize];
        node.subtree_token_end = self.subtree_tokens.len() as u32;
        node.subtree_bytes = bytes;
        node.subtree_first_bytes = first;
        node.subtree_max_byte_len = max_len;
    }

    fn finalize_full_walk(&mut self) {
        self.full_walk_ops.clear();
        self.full_walk_edge_blocks.clear();
        self.full_walk_edge_op_starts.clear();
        self.full_walk_root_byte_op_starts = None;
        self.full_walk_max_parent_depth = self
            .walk_edges
            .iter()
            .map(|edge| edge.parent_depth)
            .max()
            .unwrap_or(0);
        self.full_walk_ops.reserve(self.edge_bytes.len().max(self.walk_edges.len()));
        self.full_walk_edge_op_starts.reserve(self.walk_edges.len() + 1);
        for edge in self.walk_edges.iter().copied() {
            self.full_walk_edge_op_starts
                .push(self.full_walk_ops.len() as u32);
            let start = edge.byte_start as usize;
            let end = start + edge.byte_len as usize;
            let bytes = &self.edge_bytes[start..end];
            let depth = u32::from(edge.parent_depth) << 8;
            let child_is_token = self.nodes[edge.child as usize].token_id.is_some();
            if bytes.is_empty() {
                let mut meta = depth
                    | DynamicMaskTrieFullWalkOp::START
                    | DynamicMaskTrieFullWalkOp::END;
                if child_is_token { meta |= DynamicMaskTrieFullWalkOp::TOKEN; }
                self.full_walk_ops.push(DynamicMaskTrieFullWalkOp { meta });
                continue;
            }
            for (index, &byte) in bytes.iter().enumerate() {
                let mut meta = u32::from(byte) | depth | DynamicMaskTrieFullWalkOp::CONSUME;
                if index == 0 { meta |= DynamicMaskTrieFullWalkOp::START; }
                let last = index + 1 == bytes.len();
                if last {
                    meta |= DynamicMaskTrieFullWalkOp::END;
                    if child_is_token { meta |= DynamicMaskTrieFullWalkOp::TOKEN; }
                }
                self.full_walk_ops.push(DynamicMaskTrieFullWalkOp { meta });
            }
        }
        self.full_walk_edge_op_starts
            .push(self.full_walk_ops.len() as u32);
        self.full_walk_edge_blocks.resize(
            self.full_walk_ops.len().div_ceil(64), FullWalkEdgeBlock::default());
        for &op in &self.full_walk_edge_op_starts[..self.walk_edges.len()] {
            let op = op as usize;
            self.full_walk_edge_blocks[op >> 6].starts |= 1u64 << (op & 63);
        }
        let mut rank = 0u32;
        for block in &mut self.full_walk_edge_blocks {
            block.preceding_edges = rank;
            rank += block.starts.count_ones();
        }
        debug_assert_eq!(
            self.full_walk_edge_op_starts.len(),
            self.walk_edges.len() + 1
        );

        // Ordinary radix tries have one non-empty edge per distinct first byte
        // below the true root. Index those DFS ranges once so sparse lexer
        // roots can enter only byte-live vocabulary subtrees. Partitioned
        // tries have zero-byte structural root edges and intentionally decline
        // this optimization rather than conflating layout with byte language.
        let mut root_byte_starts = Box::new([u32::MAX; 256]);
        let mut root_byte_index_valid = true;
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            if edge.parent_depth != 0 {
                continue;
            }
            let bytes = self.walk_edge_bytes(&edge);
            let Some(&first) = bytes.first() else {
                root_byte_index_valid = false;
                break;
            };
            let slot = &mut root_byte_starts[first as usize];
            if *slot != u32::MAX {
                root_byte_index_valid = false;
                break;
            }
            *slot = self.full_walk_edge_op_starts[edge_index];
        }
        if root_byte_index_valid {
            self.full_walk_root_byte_op_starts = Some(root_byte_starts);
        }
    }

    fn append_prefix_tree(&mut self, root: VocabPrefixTreeNode<'_>, parent: Option<u32>, incoming: &[u8]) {
        enum Task<'a> {
            Enter(VocabPrefixTreeNode<'a>, Option<u32>, &'a [u8]),
            Finish(u32, &'a [u8]),
        }
        let mut work = vec![Task::Enter(root, parent, incoming)];
        while let Some(task) = work.pop() {
            match task {
                Task::Finish(id, bytes) => self.finish_node(id, bytes),
                Task::Enter(node, parent, bytes) => {
                    let id = self.begin_node(parent, node.has_token().then_some(node.token_id() as u32), bytes.len());
                    work.push(Task::Finish(id, bytes));
                    for (edge, child) in node.iter_children().rev() {
                        work.push(Task::Enter(child, Some(id), edge));
                    }
                }
            }
        }
    }

    pub(crate) fn from_vocab_prefix_tree(tree: &VocabPrefixTree) -> Self {
        let mut output = Self::new();
        output.append_prefix_tree(tree.root(), None, &[]);
        output.finalize_full_walk();
        output
    }

    /// Build the same flat runtime radix-trie representation, but place one
    /// zero-byte structural node above each caller-supplied vocabulary layout
    /// class. `entries` must be ordered by `(class, token_bytes)` and token byte
    /// strings must already be canonical/deduplicated.
    ///
    /// The structural edges are a layout device only: they consume no input
    /// and are invisible to lexer semantics. Their purpose is to keep token
    /// families with different byte behaviour from contaminating one another's
    /// subtree metadata, so the generic runtime subtree certificates can skip
    /// large groups without any partition-specific masking logic.
    pub(crate) fn from_partitioned_token_refs(entries: &[(u16, usize, &[u8])]) -> Self {
        Self::from_partitioned_sorted_direct(entries)
    }

    pub(crate) fn from_structurally_partitioned_token_refs(entries: &[(u16, usize, &[u8])]) -> Self {
        let mut trie = Self::from_partitioned_sorted_direct(entries);
        trie.root_layout_classes.clear();
        trie.root_layout_all_valid_utf8.clear();
        trie
    }

    pub(crate) fn from_sorted_token_refs(entries: &[(u16, usize, &[u8])]) -> Self {
        Self::from_sorted_ranges(entries, false)
    }

    #[cfg(test)]
    fn from_partitioned_token_refs_reference(entries: &[(u16, usize, &[u8])]) -> Self {
        let mut output = Self::new();
        if entries.is_empty() {
            return output;
        }

        // Empty-token aliases are canonicalized before this stage, so at most
        // one canonical empty byte string may exist. Keep it on the true root.
        let mut start = 0usize;
        if entries[0].2.is_empty() {
            output.nodes[0].token_id = Some(entries[0].1 as u32);
            start = 1;
        }

        let root_token = output.nodes[0].token_id;
        output.begin_node(None, root_token, 0);
        let mut index = start;
        while index < entries.len() {
            let group_start = index;
            let class = entries[index].0;
            index += 1;
            while index < entries.len() && entries[index].0 == class { index += 1; }
            let refs: Vec<_> = entries[group_start..index].iter()
                .map(|(_, id, bytes)| (*id, *bytes)).collect();
            output.root_layout_classes.push(class);
            output.root_layout_all_valid_utf8.push(refs.iter().all(|(_, bytes)| std::str::from_utf8(bytes).is_ok()));
            let tree = VocabPrefixTree::build_presorted(&refs);
            output.append_prefix_tree(tree.root(), Some(0), &[]);
        }
        output.finish_node(0, &[]);
        output.finalize_full_walk();
        output
    }

    fn from_partitioned_sorted_direct(entries: &[(u16, usize, &[u8])]) -> Self {
        Self::from_sorted_ranges(entries, true)
    }

    fn from_sorted_ranges(entries: &[(u16, usize, &[u8])], partitioned: bool) -> Self {
        enum Task<'a> {
            Enter { lo: usize, hi: usize, depth: usize, edge_from: usize, parent: Option<u32> },
            Finish(u32, &'a [u8]),
        }
        let mut output = Self::new();
        if entries.is_empty() { return output; }
        output.nodes.reserve(entries.len().saturating_mul(2));
        output.walk_edges.reserve(entries.len().saturating_mul(2));
        output.edge_bytes.reserve(entries.iter().map(|e|e.2.len()).sum());
        let mut work = Vec::with_capacity(64);
        if partitioned {
            let empty = entries[0].2.is_empty();
            output.begin_node(None, empty.then_some(entries[0].1 as u32), 0);
            work.push(Task::Finish(0, &[]));
            let mut groups = Vec::new();
            let mut index = usize::from(empty);
            while index < entries.len() {
                let start = index;
                let class = entries[index].0;
                index += 1;
                while index < entries.len() && entries[index].0 == class { index += 1; }
                output.root_layout_classes.push(class);
                output.root_layout_all_valid_utf8.push(entries[start..index].iter().all(|e|std::str::from_utf8(e.2).is_ok()));
                groups.push((start,index));
            }
            for (lo,hi) in groups.into_iter().rev() {
                work.push(Task::Enter {lo,hi,depth:0,edge_from:0,parent:Some(0)});
            }
        } else {
            work.push(Task::Enter {lo:0,hi:entries.len(),depth:0,edge_from:0,parent:None});
        }
        while let Some(task) = work.pop() {
            match task {
                Task::Finish(id, bytes) => output.finish_node(id, bytes),
                Task::Enter {mut lo,hi,depth,edge_from,parent} => {
                    let incoming = &entries[lo].2[edge_from..depth];
                    let token = (entries[lo].2.len() == depth).then_some(entries[lo].1 as u32);
                    let id = output.begin_node(parent, token, incoming.len());
                    work.push(Task::Finish(id, incoming));
                    if token.is_some() { lo += 1; }
                    let mut end = hi;
                    while end > lo {
                        let byte = entries[end-1].2[depth];
                        let mut start = end-1;
                        while start > lo && entries[start-1].2[depth] == byte { start -= 1; }
                        let a=entries[start].2; let b=entries[end-1].2;
                        let mut common=depth+1;
                        while common<a.len().min(b.len()) && a[common]==b[common] { common+=1; }
                        work.push(Task::Enter {lo:start,hi:end,depth:common,edge_from:depth,parent:Some(id)});
                        end=start;
                    }
                }
            }
        }
        output.finalize_full_walk();
        output
    }

    fn all_token_paths_match(&self, mut verify: impl FnMut(u32, &[u8]) -> bool) -> bool {
        if let Some(token) = self.node(0).token_id && !verify(token, &[]) { return false; }
        let mut prefix = Vec::new();
        let mut lengths = vec![0usize];
        for edge in &self.walk_edges {
            let depth = edge.parent_depth as usize;
            prefix.truncate(lengths[depth]);
            lengths.truncate(depth + 1);
            prefix.extend_from_slice(self.edge_bytes(edge));
            if let Some(token) = self.node(edge.child).token_id && !verify(token, &prefix) {
                return false;
            }
            lengths.push(prefix.len());
        }
        true
    }
}

// The existing artifact encoding contains both adjacency and preorder walks.
// Keep the wire schema, but emit the legacy adjacency and token-node arrays
// from the sole retained topology, and rebuild derived indexes on input.
struct LegacyTrieNodes<'a>(&'a DynamicMaskTrie);
struct LegacyTrieEdges<'a>(&'a DynamicMaskTrie);
struct LegacyTrieTokenNodes<'a>(&'a DynamicMaskTrie);
struct LegacyTrieOpEdges<'a>(&'a DynamicMaskTrie);

impl serde::Serialize for LegacyTrieOpEdges<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.full_walk_ops.len()))?;
        for (edge, span) in self.0.full_walk_edge_op_starts.windows(2).enumerate() {
            for _ in span[0]..span[1] {
                seq.serialize_element(&(edge as u32))?;
            }
        }
        seq.end()
    }
}

impl serde::Serialize for LegacyTrieNodes<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.nodes.len()))?;
        let mut next = 0u32;
        for node in &self.0.nodes {
            let mut wire = node.clone();
            wire.first_child = if node.child_len == 0 { 0 } else { next };
            next += node.child_len;
            seq.serialize_element(&wire)?;
        }
        seq.end()
    }
}

impl serde::Serialize for LegacyTrieEdges<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.walk_edges.len()))?;
        for id in 0..self.0.nodes.len() as u32 {
            for edge in self.0.children(id) {
                seq.serialize_element(&DynamicMaskTrieEdge {
                    byte_start: edge.byte_start,
                    byte_len: u32::from(edge.byte_len),
                    child: edge.child,
                })?;
            }
        }
        seq.end()
    }
}

impl serde::Serialize for LegacyTrieTokenNodes<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let root = usize::from(self.0.nodes[0].token_id.is_some());
        let mut seq = serializer.serialize_seq(Some(self.0.subtree_tokens.len() - root))?;
        for edge in &self.0.walk_edges {
            if self.0.nodes[edge.child as usize].token_id.is_some() {
                seq.serialize_element(&edge.child)?;
            }
        }
        seq.end()
    }
}

impl serde::Serialize for DynamicMaskTrie {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut wire = serializer.serialize_struct("DynamicMaskTrie", 13)?;
        wire.serialize_field("nodes", &LegacyTrieNodes(self))?;
        wire.serialize_field("edges", &LegacyTrieEdges(self))?;
        wire.serialize_field("edge_bytes", &self.edge_bytes)?;
        wire.serialize_field("subtree_tokens", &self.subtree_tokens)?;
        wire.serialize_field("walk_edges", &self.walk_edges)?;
        wire.serialize_field("full_walk_ops", &self.full_walk_ops)?;
        wire.serialize_field("full_walk_op_edges", &LegacyTrieOpEdges(self))?;
        wire.serialize_field("full_walk_edge_op_starts", &self.full_walk_edge_op_starts)?;
        wire.serialize_field("full_walk_root_byte_op_starts",
            &self.full_walk_root_byte_op_starts.as_deref().map(|row| row.as_slice()))?;
        wire.serialize_field("full_walk_token_nodes", &LegacyTrieTokenNodes(self))?;
        wire.serialize_field("full_walk_max_parent_depth", &self.full_walk_max_parent_depth)?;
        wire.serialize_field("root_layout_classes", &self.root_layout_classes)?;
        wire.serialize_field("root_layout_all_valid_utf8", &self.root_layout_all_valid_utf8)?;
        wire.end()
    }
}

/// Consume an old derived array without retaining a second runtime graph.
pub(super) struct DiscardedSequence<T>(std::marker::PhantomData<T>);

impl<T> Default for DiscardedSequence<T> {
    fn default() -> Self { Self(std::marker::PhantomData) }
}

impl<T> serde::Serialize for DiscardedSequence<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::ser::SerializeSeq::end(serializer.serialize_seq(Some(0))?)
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for DiscardedSequence<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);
        impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for Visitor<T> {
            type Value = DiscardedSequence<T>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a legacy derived sequence")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                while seq.next_element::<T>()?.is_some() {}
                Ok(DiscardedSequence(std::marker::PhantomData))
            }
        }
        deserializer.deserialize_seq(Visitor(std::marker::PhantomData))
    }
}

#[derive(serde::Deserialize)]
#[serde(rename = "DynamicMaskTrie")]
struct LegacyTrieWire {
    nodes: Vec<DynamicMaskTrieNode>,
    edges: Vec<DynamicMaskTrieEdge>,
    edge_bytes: Vec<u8>,
    #[serde(rename = "subtree_tokens")]
    _subtree_tokens: DiscardedSequence<u32>,
    #[serde(rename = "walk_edges")]
    _walk_edges: DiscardedSequence<DynamicMaskTrieWalkEdge>,
    #[serde(rename = "full_walk_ops")]
    _full_walk_ops: DiscardedSequence<DynamicMaskTrieFullWalkOp>,
    #[serde(rename = "full_walk_op_edges")]
    _full_walk_op_edges: DiscardedSequence<u32>,
    #[serde(rename = "full_walk_edge_op_starts")]
    _full_walk_edge_op_starts: DiscardedSequence<u32>,
    #[serde(rename = "full_walk_root_byte_op_starts")]
    _full_walk_root_byte_op_starts: Option<DiscardedSequence<u32>>,
    #[serde(rename = "full_walk_token_nodes")]
    _full_walk_token_nodes: DiscardedSequence<u32>,
    #[serde(rename = "full_walk_max_parent_depth")]
    _full_walk_max_parent_depth: u16,
    root_layout_classes: Vec<u16>,
    root_layout_all_valid_utf8: Vec<bool>,
}

impl<'de> serde::Deserialize<'de> for DynamicMaskTrie {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = <LegacyTrieWire as serde::Deserialize>::deserialize(deserializer)?;
        let invalid = || serde::de::Error::custom("invalid dynamic mask trie topology");
        if wire.nodes.is_empty() || wire.nodes.len() != wire.edges.len().saturating_add(1)
            || wire.nodes.len() > u32::MAX as usize || wire.edge_bytes.len() > u32::MAX as usize {
            return Err(invalid());
        }
        if (!wire.root_layout_classes.is_empty()
                && wire.root_layout_classes.len() != wire.nodes[0].child_len as usize)
            || (!wire.root_layout_all_valid_utf8.is_empty()
                && wire.root_layout_all_valid_utf8.len() != wire.nodes[0].child_len as usize) {
            return Err(invalid());
        }
        enum Task<'a> { Enter(u32, Option<u32>, &'a [u8]), Finish(u32, &'a [u8]) }
        let mut trie = Self::new();
        trie.nodes.reserve(wire.nodes.len() - 1);
        trie.walk_edges.reserve(wire.edges.len());
        trie.edge_bytes.reserve(wire.edge_bytes.len());
        let mut work = vec![Task::Enter(0, None, &[])];
        let mut total_bytes = 0usize;
        let mut total_ops = 0usize;
        while let Some(task) = work.pop() {
            match task {
                Task::Finish(id, bytes) => trie.finish_node(id, bytes),
                Task::Enter(old, parent, incoming) => {
                    let node = wire.nodes.get(old as usize).ok_or_else(invalid)?;
                    // All historical builders allocate nodes in preorder.
                    // Preserve IDs: persisted slice metadata refers to them.
                    let expected = if parent.is_some() { trie.nodes.len() } else { 0 };
                    if expected != old as usize || incoming.len() > u16::MAX as usize {
                        return Err(invalid());
                    }
                    if let Some(parent) = parent {
                        if parent != 0 && trie.walk_edges[parent as usize - 1].parent_depth >= u16::MAX - 1
                            || trie.nodes[parent as usize].prefix_byte_len
                                .checked_add(incoming.len() as u32).is_none() {
                            return Err(invalid());
                        }
                        total_bytes = total_bytes.checked_add(incoming.len()).ok_or_else(invalid)?;
                        total_ops = total_ops.checked_add(incoming.len().max(1)).ok_or_else(invalid)?;
                        if total_bytes > wire.edge_bytes.len() || total_ops > u32::MAX as usize {
                            return Err(invalid());
                        }
                    }
                    let id = trie.begin_node(parent, node.token_id, incoming.len());
                    work.push(Task::Finish(id, incoming));
                    if node.child_len == 0 { continue; }
                    let first = node.first_child as usize;
                    let end = first.checked_add(node.child_len as usize).ok_or_else(invalid)?;
                    for edge in wire.edges.get(first..end).ok_or_else(invalid)?.iter().rev() {
                        let start = edge.byte_start as usize;
                        let end = start.checked_add(edge.byte_len as usize).ok_or_else(invalid)?;
                        let bytes = wire.edge_bytes.get(start..end).ok_or_else(invalid)?;
                        work.push(Task::Enter(edge.child, Some(id), bytes));
                    }
                }
            }
        }
        if trie.nodes.len() != wire.nodes.len() || total_bytes != wire.edge_bytes.len() {
            return Err(invalid());
        }
        trie.root_layout_classes = wire.root_layout_classes;
        trie.root_layout_all_valid_utf8 = wire.root_layout_all_valid_utf8;
        if trie.walk_edges.len() != 0 || trie.nodes[0].token_id.is_some() {
            trie.finalize_full_walk();
        }
        Ok(trie)
    }
}

impl Default for DynamicMaskTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum PackedDynamicMaskTokenAliases {
    Single(u32),
    Many(Box<[u32]>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum DynamicMaskAliasStore {
    Ordered(Arc<Vec<Vec<u32>>>),
    /// Canonical alias groups already flattened into the runtime hot layout.
    /// Offsets has `canonical_count + 1` entries into `originals`.
    Flat {
        offsets: Arc<Vec<u32>>,
        originals: Arc<Vec<u32>>,
    },
    Packed(Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>),
}

#[derive(Debug)]
struct DynamicMaskCacheEntry {
    hash: u64,
    state: DynamicMaskStateKey,
    // None retains the existing first-use probation state. Equal, already
    // materialized masks can be shared without encoding another payload.
    mask: Option<Arc<DynamicMaskCachePayload>>,
}

#[derive(Debug, Default)]
struct DynamicMaskCache {
    entry_cap: Option<usize>,
    entries: Vec<Option<DynamicMaskCacheEntry>>,
    by_hash: FxHashMap<u64, SmallVec<[usize; 1]>>,
    next_slot: usize,
    values: super::mask_cache_values::MaskValueIndex,
}

#[inline]
pub(crate) fn dynamic_mask_state_key_hash(state: &DynamicMaskStateKey) -> u64 {
    let mut hasher = FxHasher::default();
    state.hash(&mut hasher);
    hasher.finish()
}


/// Interned binary unions of terminal bitsets. Zero denotes the empty set
/// at every height; nonzero nodes have a unique height by construction.
/// A leaf stores its bits, and a branch stores two child IDs in one u64.
const SMALL_SUPPORT_TAG: u32 = 1 << 31;

#[derive(Debug, Clone, Default)]
pub(crate) struct DirectRegularTerminalSupport {
    roots: Vec<u32>,
    completing: Vec<u64>,
    nodes: Vec<u64>,
    small_sets: Vec<[TerminalID; 4]>,
    dense_roots: FxHashMap<u32, Arc<[u64]>>,
    levels: u8,
    num_terminals: usize,
    hot_primary: Option<(u32, u64, u16)>,
}

struct SupportBuilder {
    nodes: Vec<u64>,
    counts: Vec<u16>,
    leaves: FxHashMap<u64, u32>,
    branches: FxHashMap<(u32, u32), u32>,
    unions: FxHashMap<(u32, u32), u32>,
}

impl SupportBuilder {
    fn new() -> Self {
        Self {
            nodes: vec![0],
            counts: vec![0],
            leaves: FxHashMap::default(),
            branches: FxHashMap::default(),
            unions: FxHashMap::default(),
        }
    }

    fn leaf(&mut self, bits: u64) -> u32 {
        if bits == 0 { return 0; }
        if let Some(&id) = self.leaves.get(&bits) { return id; }
        let id = self.nodes.len() as u32;
        self.nodes.push(bits);
        self.counts.push(bits.count_ones() as u16);
        self.leaves.insert(bits, id);
        id
    }

    fn branch(&mut self, left: u32, right: u32) -> u32 {
        if left == 0 && right == 0 { return 0; }
        if let Some(&id) = self.branches.get(&(left, right)) { return id; }
        let id = self.nodes.len() as u32;
        self.nodes.push(pack_support_children(left, right));
        self.counts.push(self.counts[left as usize].saturating_add(self.counts[right as usize]));
        self.branches.insert((left, right), id);
        id
    }

    fn union(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if left == right || right == 0 { return left; }
        if left == 0 { return right; }
        let key = (left.min(right), left.max(right));
        if let Some(&id) = self.unions.get(&key) { return id; }
        let id = if level == 0 {
            self.leaf(self.nodes[left as usize] | self.nodes[right as usize])
        } else {
            let (la, lb) = unpack_support_children(self.nodes[left as usize]);
            let (ra, rb) = unpack_support_children(self.nodes[right as usize]);
            let a = self.union(level - 1, la, ra);
            let b = self.union(level - 1, lb, rb);
            self.branch(a, b)
        };
        self.unions.insert(key, id);
        id
    }

    fn singleton(&mut self, levels: usize, terminal: usize) -> u32 {
        let word = terminal / 64;
        let mut node = self.leaf(1u64 << (terminal % 64));
        for level in 0..levels {
            node = if word & (1 << level) == 0 {
                self.branch(node, 0)
            } else {
                self.branch(0, node)
            };
        }
        node
    }

    fn copy_reachable(&self, level: usize, node: u32, remap: &mut [u32], output: &mut Vec<u64>) -> u32 {
        if node == 0 { return 0; }
        let known = remap[node as usize];
        if known != u32::MAX { return known; }
        let payload = self.nodes[node as usize];
        let payload = if level == 0 {
            payload
        } else {
            let (left, right) = unpack_support_children(payload);
            let left = self.copy_reachable(level - 1, left, remap, output);
            let right = self.copy_reachable(level - 1, right, remap, output);
            pack_support_children(left, right)
        };
        let id = output.len() as u32;
        output.push(payload);
        remap[node as usize] = id;
        id
    }
}

#[inline]
fn pack_support_children(left: u32, right: u32) -> u64 {
    u64::from(left) | (u64::from(right) << 32)
}

#[inline]
fn unpack_support_children(payload: u64) -> (u32, u32) {
    (payload as u32, (payload >> 32) as u32)
}

/// Visit nonzero leaf words only; the shared DAG does not enumerate paths
/// through the parser epsilon graph.
fn visit_support_words(nodes: &[u64], node: u32, level: usize, base: usize, visit: &mut impl FnMut(usize, u64)) {
    if node == 0 { return; }
    let payload = nodes[node as usize];
    if level == 0 {
        visit(base, payload);
    } else {
        let (left, right) = unpack_support_children(payload);
        visit_support_words(nodes, left, level - 1, base, visit);
        visit_support_words(nodes, right, level - 1, base + (1 << (level - 1)), visit);
    }
}

impl DirectRegularTerminalSupport {
    pub(crate) fn build(automaton: &DirectRegularAutomaton, num_terminals: usize) -> Self {
        if automaton.states.len() >= u32::MAX as usize {
            return Self::default();
        }
        let levels = num_terminals.div_ceil(64).max(1).next_power_of_two().trailing_zeros() as usize;
        let mut builder = SupportBuilder::new();
        let singletons = (0..num_terminals)
            .map(|terminal| builder.singleton(levels, terminal))
            .collect::<Vec<_>>();

        let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
        let mut remaining_children = Vec::<usize>::with_capacity(automaton.states.len());
        let mut queue = VecDeque::<u32>::new();
        for (source, state) in automaton.states.iter().enumerate() {
            remaining_children.push(state.epsilons.len());
            if state.epsilons.is_empty() { queue.push_back(source as u32); }
            for &child in &state.epsilons {
                let Some(incoming) = parents.get_mut(child as usize) else { return Self::default(); };
                incoming.push(source as u32);
            }
        }

        // The hot frontier heuristic uses exactly the same epsilon order.
        // Retain only its winner, not a second traversal or per-node work.
        let mut transition_work = (automaton.states.len() >= 16_384)
            .then(|| vec![0u64; automaton.states.len()]);
        let mut hot_primary = None::<(u32, u64, u16)>;
        // Match the parser coordinate: synthetic initial state 0, raw states + 1.
        let mut roots = vec![0; automaton.states.len() + 1];
        let mut completing = BitSet::new(roots.len());
        let mut processed = 0usize;
        while let Some(raw) = queue.pop_front() {
            let state = &automaton.states[raw as usize];
            let mut root = 0;
            for &terminal in state.transitions.keys() {
                if (terminal as usize) < num_terminals {
                    root = builder.union(levels, root, singletons[terminal as usize]);
                }
            }
            for &child in &state.epsilons {
                root = builder.union(levels, root, roots[child as usize + 1]);
            }
            let parser_state = raw as usize + 1;
            roots[parser_state] = root;
            if state.is_accepting || state.epsilons.iter().any(|&child| completing.get(child as usize + 1)) {
                completing.set(parser_state);
            }
            if let Some(work) = &mut transition_work {
                let own = state.transitions.values().map(|targets| targets.len() as u64).sum::<u64>();
                let count = state.epsilons.iter().fold(own, |sum, &child| sum.saturating_add(work[child as usize]));
                work[raw as usize] = count;
                let terminals = builder.counts[root as usize];
                if hot_primary.is_none_or(|(old_raw, old_work, old_terminals)| {
                    (terminals, count, raw) > (old_terminals, old_work, old_raw)
                }) {
                    hot_primary = Some((raw, count, terminals));
                }
            }
            processed += 1;
            for &parent in &parents[raw as usize] {
                let remaining = &mut remaining_children[parent as usize];
                *remaining -= 1;
                if *remaining == 0 { queue.push_back(parent); }
            }
        }
        if processed != automaton.states.len() { return Self::default(); }
        for &raw in &automaton.start_states {
            let parser_state = raw as usize + 1;
            let Some(&root) = roots.get(parser_state) else { return Self::default(); };
            roots[0] = builder.union(levels, roots[0], root);
            if completing.get(parser_state) { completing.set(0); }
        }
        if builder.nodes.len() >= SMALL_SUPPORT_TAG as usize { return Self::default(); }
        // Small roots need no DAG traversal in the hot admission path.
        // Repack only nodes reachable from large roots; discard all builder
        // union temporaries, counts, and singleton-only interior nodes.
        let mut support = Self {
            roots,
            completing: completing.words().to_vec(),
            nodes: vec![0],
            small_sets: Vec::new(),
            dense_roots: FxHashMap::default(),
            levels: levels as u8,
            num_terminals,
            hot_primary,
        };
        let mut remap = vec![u32::MAX; builder.nodes.len()];
        for root in &mut support.roots {
            let old = *root;
            if remap[old as usize] != u32::MAX {
                *root = remap[old as usize];
                continue;
            }
            if builder.counts[old as usize] <= 4 {
                let mut terminals = [TerminalID::MAX; 4];
                let mut len = 0;
                visit_support_words(&builder.nodes, old, levels, 0, &mut |word, mut bits| {
                    while bits != 0 {
                        terminals[len] = (word * 64 + bits.trailing_zeros() as usize) as TerminalID;
                        len += 1;
                        bits &= bits - 1;
                    }
                });
                *root = SMALL_SUPPORT_TAG | support.small_sets.len() as u32;
                support.small_sets.push(terminals);
                remap[old as usize] = *root;
            } else {
                *root = builder.copy_reachable(levels, old, &mut remap, &mut support.nodes);
            }
        }
        for parser_state in std::iter::once(0).chain(automaton.start_states.iter().map(|&raw| raw + 1)) {
            let Some(&root) = support.roots.get(parser_state as usize) else { continue; };
            if root & SMALL_SUPPORT_TAG != 0 || support.dense_roots.contains_key(&root) { continue; }
            let mut words = vec![0u64; num_terminals.div_ceil(64)];
            support.or_state_into(parser_state, &mut words);
            support.dense_roots.insert(root, Arc::from(words));
        }
        support
    }

    pub(crate) fn is_initialized(&self) -> bool {
        !self.roots.is_empty()
    }

    pub(crate) fn is_complete(&self, parser_state: u32) -> bool {
        self.completing.get(parser_state as usize / 64)
            .is_some_and(|word| *word & (1u64 << (parser_state % 64)) != 0)
    }

    pub(crate) fn hot_primary(&self) -> Option<(u32, u64, u16)> {
        self.hot_primary
    }

    #[inline]
    fn small_set(&self, root: u32) -> Option<&[TerminalID]> {
        if root & SMALL_SUPPORT_TAG == 0 { return None; }
        let row = &self.small_sets[(root & !SMALL_SUPPORT_TAG) as usize];
        let len = row.iter().position(|&id| id == TerminalID::MAX).unwrap_or(4);
        Some(&row[..len])
    }

    pub(crate) fn for_each_small_state_terminal(&self, parser_state: u32, mut visit: impl FnMut(TerminalID)) -> bool {
        let Some(terminals) = self.roots.get(parser_state as usize).and_then(|&root| self.small_set(root)) else {
            return false;
        };
        for &terminal in terminals { visit(terminal); }
        true
    }

    #[inline]
    pub(crate) fn contains(&self, parser_state: u32, terminal: TerminalID) -> bool {
        if terminal as usize >= self.num_terminals { return false; }
        let Some(&root) = self.roots.get(parser_state as usize) else { return false; };
        if let Some(terminals) = self.small_set(root) { return terminals.contains(&terminal); }
        let mut node = root;
        let mut level = self.levels as usize;
        let word = terminal as usize / 64;
        while node != 0 && level != 0 {
            let (left, right) = unpack_support_children(self.nodes[node as usize]);
            level -= 1;
            node = if word & (1 << level) == 0 { left } else { right };
        }
        node != 0 && self.nodes[node as usize] & (1u64 << (terminal % 64)) != 0
    }

    pub(crate) fn or_state_into(&self, parser_state: u32, output: &mut [u64]) {
        let Some(&root) = self.roots.get(parser_state as usize) else { return; };
        if let Some(terminals) = self.small_set(root) {
            for &terminal in terminals {
                if let Some(word) = output.get_mut(terminal as usize / 64) {
                    *word |= 1u64 << (terminal % 64);
                }
            }
        } else if let Some(words) = self.dense_roots.get(&root) {
            for (target, source) in output.iter_mut().zip(words.iter()) { *target |= *source; }
        } else {
            visit_support_words(&self.nodes, root, self.levels as usize, 0, &mut |base, bits| {
                if let Some(word) = output.get_mut(base) { *word |= bits; }
            });
        }
    }

    fn intersects_node(&self, node: u32, level: usize, base: usize, terminals: &[u64]) -> bool {
        if node == 0 { return false; }
        let payload = self.nodes[node as usize];
        if level == 0 {
            return terminals.get(base).is_some_and(|word| *word & payload != 0);
        }
        let (left, right) = unpack_support_children(payload);
        self.intersects_node(left, level - 1, base, terminals)
            || self.intersects_node(right, level - 1, base + (1 << (level - 1)), terminals)
    }

    pub(crate) fn intersects(&self, parser_state: u32, terminals: &[u64]) -> bool {
        let Some(&root) = self.roots.get(parser_state as usize) else { return false; };
        if let Some(small) = self.small_set(root) {
            small.iter().any(|&terminal| terminals.get(terminal as usize / 64)
                .is_some_and(|word| *word & (1u64 << (terminal % 64)) != 0))
        } else if let Some(words) = self.dense_roots.get(&root) {
            words.iter().zip(terminals).any(|(left, right)| *left & *right != 0)
        } else {
            self.intersects_node(root, self.levels as usize, 0, terminals)
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularDynamicFrontierCacheEntry {
    /// Retain the source interface so its pointer-derived key cannot be reused
    /// while this cache entry exists.
    pub(crate) source: ParserGSS,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

/// Canonical semantic snapshot of a dynamic-mask residual. Flattening the GSS
/// deliberately removes representation-only Arc identities and accumulator
/// node organization, so equivalent residuals reached after different token
/// commits share one exact cached mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum DynamicMaskLexerStateKey {
    Exact(u32),
    /// Exact scoped tokenizer coordinate used by recursive/provider-native
    /// composition. Keep it distinct from the ordinary tokenizer coordinate:
    /// both are u32 IDs but belong to different state spaces.
    RecursiveExact(u32),
    /// Exact lexer coordinate consumed by dynamic mask execution. Distinct
    /// source/runtime lexer states may map here when their complete one-model-
    /// token continuation languages are identical. Preserve whether the source
    /// state is the true lexer initial state because reset semantics can make
    /// that distinction observable outside the projected coordinate itself.
    MaskProjection { state: u32, initial: bool },
    TerminalObservation { terminal: TerminalID, class: u32, initial: bool },
    /// Sound pre-minimization coordinate of a virtual bounded-code residual's
    /// finite one-model-token projection. This gives the dynamic cache the
    /// useful count collapse of the finite projection without constructing its
    /// potentially large DFA.
    VirtualDenseProjection { runtime: u32, state: u32, initial: bool },
}

/// Paths are top-first and sorted lexicographically with their correlated
/// exclusions. The orientation is private to this in-memory cache; reversing
/// every path is a bijection, so it changes neither key equivalence nor masks.
pub(crate) type DynamicMaskStateKey = Vec<(
    DynamicMaskLexerStateKey,
    Vec<(Vec<u32>, Vec<(u32, Vec<TerminalID>)>)>,
)>;

#[derive(Debug, Clone)]
pub(crate) struct DynamicConfigSubtreeCertificate {
    pub(crate) node: u32,
    /// Lexer NFA configuration in the mask-tokenizer quotient coordinate.
    pub(crate) projected_config: Arc<[u32]>,
    /// Every vocabulary token below `node`, when entered in
    /// `projected_config`, reaches token boundary without another lexer
    /// finalization while retaining at least one of these terminals as a
    /// possible future.  Runtime needs only one terminal to be parser-admissible.
    pub(crate) common_future_terminals: Arc<[TerminalID]>,
}

/// Exact vocabulary-relative continuation row after one lexer terminal has
/// finalized inside a model token and the lexer has reset.  Every token in
/// `tokens` reaches token boundary without a second lexer finalization and is
/// live for at least one terminal in `terminals`.  `terminals` are grouped by
/// exact equality of their fused-token set, so runtime normally tests only a
/// handful of rows even when many grammar terminals share the same lexical
/// continuation language.
#[derive(Debug, Clone)]
pub(crate) struct DynamicFirstMatchPostRow {
    pub(crate) terminals: Arc<[TerminalID]>,
    pub(crate) tokens: Arc<[u32]>,
    /// Prepacked token mask for broad rows.  Sparse rows leave this empty and
    /// are cheaper to apply by setting their handful of token IDs directly.
    pub(crate) dense_mask: Arc<[u32]>,
}

/// Second-finalization continuation from a first-match one-step projection.
/// `terminal` is consumed on the post-first parser stack.  Exact-end tokens
/// become immediately valid after that parser advance; `post_rows` describe
/// residual lexer futures after the second reset for branches that reach token
/// boundary without a third finalization.
#[derive(Debug, Clone)]
pub(crate) struct DynamicFirstMatchSecondRow {
    pub(crate) terminal: TerminalID,
    pub(crate) exact_end_tokens: Arc<[u32]>,
    pub(crate) post_rows: Arc<[DynamicFirstMatchPostRow]>,
    /// Additional terminal finalizations after this terminal resets the lexer.
    /// The row type is recursive so a short vocabulary-relative lexical-effect
    /// program can represent arbitrarily many in-token finalizations without
    /// returning to byte-wise trie traversal.
    pub(crate) next_rows: Arc<[DynamicFirstMatchSecondRow]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicSelfLoopProjection {
    pub(crate) source_state: u32,
    /// Exact possible-future terminal set at `source_state`. Projection token
    /// leaves are certified only when they restore this complete set, making
    /// the projection independent of parser context. Runtime needs only one of
    /// these terminals to be parser-admissible for the continuing witness.
    pub(crate) future_terminals: Arc<[TerminalID]>,
    pub(crate) safe_no_match_mask: Arc<[u32]>,
    pub(crate) safe_subtrees: Arc<[u8]>,
    /// Nodes whose complete suffix language is safe when entered with
    /// `source_state` itself. `safe_subtrees` is relative to the state reached
    /// by consuming the node's root prefix during projection construction; an
    /// intermediate runtime walk may only reuse a projection at nodes where
    /// that reached state has returned to the projection source.
    pub(crate) source_reentry_safe_subtrees: Arc<[u8]>,
    /// For a projection rooted at `source_state`, row `node` is a bitmask over
    /// `future_terminals`: bit i is set iff that terminal remains a live
    /// no-finalization continuation for every vocabulary token below `node`.
    /// This is stronger than the historical exact-future-set projection for
    /// accepting+continuing lexer states: unrelated finalizers/futures may
    /// churn as long as one common continuing terminal witnesses the subtree.
    pub(crate) common_future_masks: Arc<[u64]>,
    /// Sparse trie nodes that are provably useless while following the
    /// no-finalization path from `source_state`: every token below the node
    /// dies before any lexer terminal can match and no token can end with a
    /// live residual lexer state.  This certificate is parser-independent.
    pub(crate) pre_match_dead_words: Arc<[u64]>,
    /// Sparse trie nodes whose incoming radix edge reaches the first lexer
    /// terminal match from `source_state`.  The dead-node certificate above is
    /// no longer applicable below these nodes because parser-dependent reset
    /// branches become possible there.
    pub(crate) pre_match_frontier_words: Arc<[u64]>,
    /// Experimental exact subset for tokens that first finalize the sole
    /// future terminal from one concrete full tokenizer state and whose
    /// post-reset byte suffix is itself an ordinary vocabulary token.
    ///
    /// Runtime validates only the suffix-token candidates after advancing the
    /// parser once on `future_terminals[0]`; an accepted suffix then certifies
    /// the corresponding fused original token.  This is deliberately a
    /// one-sided baseline: tokens not represented here still go through the
    /// ordinary exact dynamic walk.
    pub(crate) first_match_fusion_source_state: u32,
    pub(crate) first_match_fusion_match_state: u32,
    pub(crate) first_match_fusion_candidate_mask: Arc<[u32]>,
    /// One bit per dynamic vocabulary-trie node: set iff the subtree contains
    /// at least one suffix token from `first_match_fusion_candidate_mask`.
    pub(crate) first_match_fusion_candidate_subtrees: Arc<[u64]>,
    /// `(fused_original_token, suffix_original_token)` pairs.
    pub(crate) first_match_fusions: Arc<[(u32, u32)]>,
    /// Experimental exact one-finalization decomposition for a concrete full
    /// tokenizer state.  It is deliberately vocabulary-relative: tokens with
    /// more than one possible first-match width or any second finalization
    /// after reset are listed in `first_match_step_unknown_tokens` and are
    /// validated by the ordinary exact dynamic walker.
    pub(crate) first_match_step_source_state: u32,
    pub(crate) first_match_step_root_live_tokens: Arc<[u32]>,
    pub(crate) first_match_step_exact_end_tokens: Arc<[u32]>,
    pub(crate) first_match_step_post_rows: Arc<[DynamicFirstMatchPostRow]>,
    pub(crate) first_match_step_second_rows: Arc<[DynamicFirstMatchSecondRow]>,
    pub(crate) first_match_step_unknown_tokens: Arc<[u32]>,
    /// One bit per runtime vocabulary-trie node, set iff that subtree contains
    /// at least one token from `first_match_step_unknown_tokens`.
    pub(crate) first_match_step_unknown_subtrees: Arc<[u64]>,
    /// General vocabulary-relative lexical-effect program rooted directly at
    /// a concrete tokenizer state. Unlike `first_match_step_*`, this does not
    /// require a sole first terminal: residual no-finalization futures live in
    /// `root_effect_post_rows`, while `root_effect_rows` encode arbitrary
    /// terminal-finalization/reset sequences. Runtime executes only the parser
    /// effects; unresolved depth-limited tokens fall back to the exact walker.
    pub(crate) root_effect_source_state: u32,
    pub(crate) root_effect_post_rows: Arc<[DynamicFirstMatchPostRow]>,
    pub(crate) root_effect_rows: Arc<[DynamicFirstMatchSecondRow]>,
    pub(crate) root_effect_unknown_tokens: Arc<[u32]>,
    pub(crate) root_effect_unknown_subtrees: Arc<[u64]>,
    /// Sparse post-finalization certificates discovered from repeated reset-NFA
    /// configurations below this projection's first-match frontier.
    pub(crate) config_subtree_certificates: Arc<[DynamicConfigSubtreeCertificate]>,
}

impl DynamicSelfLoopProjection {
    #[inline]
    pub(crate) fn subtree_is_safe(&self, node: u32) -> bool {
        self.safe_subtrees
            .get(node as usize)
            .is_some_and(|&safe| safe != 0)
    }

    #[inline]
    pub(crate) fn subtree_is_safe_from_source(&self, node: u32) -> bool {
        self.source_reentry_safe_subtrees
            .get(node as usize)
            .is_some_and(|&safe| safe != 0)
    }

    #[inline]
    pub(crate) fn subtree_common_future_mask(&self, node: u32) -> u64 {
        self.common_future_masks
            .get(node as usize)
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    pub(crate) fn pre_match_subtree_is_dead(&self, node: u32) -> bool {
        let word = node as usize >> 6;
        let bit = node & 63;
        self.pre_match_dead_words
            .get(word)
            .is_some_and(|bits| bits & (1u64 << bit) != 0)
    }

    #[inline]
    pub(crate) fn pre_match_subtree_is_frontier(&self, node: u32) -> bool {
        let word = node as usize >> 6;
        let bit = node & 63;
        self.pre_match_frontier_words
            .get(word)
            .is_some_and(|bits| bits & (1u64 << bit) != 0)
    }

    #[inline]
    pub(crate) fn has_pre_match_dead_subtrees(&self) -> bool {
        self.pre_match_dead_words.iter().any(|&word| word != 0)
    }

    #[inline]
    pub(crate) fn has_first_match_fusions_from(&self, full_source_state: u32) -> bool {
        self.first_match_fusion_source_state == full_source_state
            && self.first_match_fusion_match_state != u32::MAX
            && !self.first_match_fusions.is_empty()
    }

    #[inline]
    pub(crate) fn has_root_effect_from(&self, full_source_state: u32) -> bool {
        self.root_effect_source_state == full_source_state
    }

    #[inline]
    pub(crate) fn has_first_match_step_from(&self, full_source_state: u32) -> bool {
        self.first_match_step_source_state == full_source_state
            && self.future_terminals.len() == 1
            && (!self.first_match_step_root_live_tokens.is_empty()
                || !self.first_match_step_exact_end_tokens.is_empty()
                || !self.first_match_step_post_rows.is_empty()
                || !self.first_match_step_unknown_tokens.is_empty())
    }

    #[inline]
    pub(crate) fn config_subtree_certificates_for_node(
        &self,
        node: u32,
    ) -> &[DynamicConfigSubtreeCertificate] {
        let certificates = self.config_subtree_certificates.as_ref();
        let start = certificates.partition_point(|certificate| certificate.node < node);
        let end = start
            + certificates[start..]
                .partition_point(|certificate| certificate.node == node);
        &certificates[start..end]
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocabSource {
    pub(crate) trie: Arc<VocabPrefixTree>,
    pub(crate) token_aliases: Arc<Vec<Vec<u32>>>,
}

/// Compact parser-independent master-slice proof sidecar. All arrays are CSR
/// over `(exact tokenizer source, proof slot)` rows. `positive_*` stores true
/// proofs; `coverage_*` stores every terminal for which the build solved the
/// exact residual product, so covered-but-not-positive is an exact false.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedMasterProofArtifact {
    pub(crate) positive_row_ids: Vec<u32>,
    pub(crate) positive_offsets: Vec<u32>,
    pub(crate) positive_terminals: Vec<TerminalID>,
    pub(crate) coverage_row_ids: Vec<u32>,
    pub(crate) coverage_offsets: Vec<u32>,
    pub(crate) coverage_terminals: Vec<TerminalID>,
    /// Safe+ terminals for which the direct residual solver is complete over
    /// every exact source. For these terminals, absence from a source coverage
    /// row means that terminal has no residual coordinate at that source and is
    /// therefore an exact negative rather than an unknown requiring a quotient.
    pub(crate) safe_plus_complete_terminals: Vec<TerminalID>,
    pub(crate) safe_radius_row_ids: Vec<u32>,
    pub(crate) safe_radius_offsets: Vec<u32>,
    pub(crate) safe_radius_entries: Vec<(TerminalID, u16)>,
}

impl PreparedMasterProofArtifact {
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.positive_row_ids.is_empty()
            && self.coverage_row_ids.is_empty()
            && self.safe_plus_complete_terminals.is_empty()
            && self.safe_radius_row_ids.is_empty()
    }
}

#[inline]
fn interned_terminal_row<'a>(
    row_ids: &[u32],
    offsets: &[u32],
    entries: &'a [TerminalID],
    row: usize,
) -> &'a [TerminalID] {
    let row_id = row_ids[row] as usize;
    &entries[offsets[row_id] as usize..offsets[row_id + 1] as usize]
}

#[derive(Debug, Default)]
struct InternedRows<T> {
    row_ids: Vec<u32>,
    offsets: Vec<u32>,
    entries: Vec<T>,
}

impl<T> InternedRows<T> {
    fn row(&self, source: usize) -> Option<&[T]> {
        let id = *self.row_ids.get(source)? as usize;
        Some(&self.entries[self.offsets[id] as usize..self.offsets[id + 1] as usize])
    }
}

fn intern_sorted_rows<T, R>(rows: impl IntoIterator<Item = R>) -> InternedRows<T>
where
    T: Copy + Eq + std::hash::Hash + Default,
    R: AsRef<[T]>,
{
    let mut result = InternedRows::default();
    let mut intern = FxHashMap::<Vec<T>, u32>::default();
    for row in rows {
        let row = row.as_ref();
        let id = if let Some(&id) = intern.get(row) {
            id
        } else {
            if result.offsets.is_empty() {
                result.offsets.push(0);
            }
            let id = intern.len() as u32;
            intern.insert(row.to_vec(), id);
            result.entries.extend_from_slice(row);
            result.offsets.push(result.entries.len() as u32);
            id
        };
        result.row_ids.push(id);
    }
    result
}

fn intern_radius_rows(mut rows: Vec<SmallVec<[(TerminalID, u16); 4]>>) -> InternedRows<(TerminalID, u16)> {
    for row in &mut rows {
        row.sort_unstable_by_key(|&(terminal, _)| terminal);
        row.dedup_by_key(|entry| entry.0);
    }
    intern_sorted_rows(rows)
}

/// One observed terminal belongs to the positive prefix or negative suffix
/// of a canonical proof row. Unlisted terminals are unknown, except for
/// globally complete direct residual solvers.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct MasterProofRow {
    positive: SmallVec<[TerminalID; 4]>,
    negative: SmallVec<[TerminalID; 4]>,
}

impl MasterProofRow {
    fn push(&mut self, terminal: TerminalID, proved: bool) {
        if proved {
            self.positive.push(terminal);
        } else {
            self.negative.push(terminal);
        }
    }

    fn canonicalize(&mut self) {
        self.positive.sort_unstable();
        self.positive.dedup();
        self.negative.sort_unstable();
        self.negative.dedup();
        let positive = &self.positive;
        self.negative.retain(|terminal| positive.binary_search(terminal).is_err());
    }
}

#[derive(Debug)]
enum MasterProofCoverage {
    /// Legacy positive-only artifacts need the associated exact quotient.
    Projected,
    Explicit(Box<[TerminalID]>),
}

#[derive(Debug)]
struct MasterProofRows {
    row_ids: Vec<u32>,
    /// Positive-end and row-end offsets in a single terminal arena.
    ends: Vec<(u32, u32)>,
    entries: Vec<TerminalID>,
    coverage: MasterProofCoverage,
}

impl MasterProofRows {
    fn empty(coverage: MasterProofCoverage) -> Self {
        Self { row_ids: Vec::new(), ends: Vec::new(), entries: Vec::new(), coverage }
    }

    fn append_unique(&mut self, row: &MasterProofRow) -> u32 {
        let id = self.ends.len() as u32;
        self.entries.extend_from_slice(&row.positive);
        let positive_end = self.entries.len() as u32;
        self.entries.extend_from_slice(&row.negative);
        self.ends.push((positive_end, self.entries.len() as u32));
        id
    }

    fn intern(rows: Vec<MasterProofRow>, coverage: MasterProofCoverage) -> Self {
        let mut result = Self::empty(coverage);
        result.row_ids.reserve(rows.len());
        let mut ids = FxHashMap::<MasterProofRow, u32>::default();
        for mut row in rows {
            row.canonicalize();
            let id = if let Some(&id) = ids.get(&row) {
                id
            } else {
                let id = result.append_unique(&row);
                ids.insert(row, id);
                id
            };
            result.row_ids.push(id);
        }
        result
    }

    fn unique_row(&self, id: usize) -> (&[TerminalID], &[TerminalID]) {
        let start = (if id == 0 { 0 } else { self.ends[id - 1].1 }) as usize;
        let (positive_end, end) = self.ends[id];
        (&self.entries[start..positive_end as usize],
         &self.entries[positive_end as usize..end as usize])
    }

    #[inline]
    fn row(&self, source_slot: usize) -> Option<(&[TerminalID], &[TerminalID])> {
        Some(self.unique_row(*self.row_ids.get(source_slot)? as usize))
    }

    fn from_wire(artifact: &PreparedMasterProofArtifact) -> Self {
        let explicit = !artifact.coverage_row_ids.is_empty();
        let mut result = Self::empty(if explicit {
            MasterProofCoverage::Explicit(artifact.safe_plus_complete_terminals.clone().into_boxed_slice())
        } else {
            MasterProofCoverage::Projected
        });
        result.row_ids.reserve(artifact.positive_row_ids.len());
        // The validated wire already interns both tables. Join each pair of
        // row IDs once, rather than expanding every source row.
        let mut joined = FxHashMap::<(u32, u32), u32>::default();
        for (source_slot, &positive_id) in artifact.positive_row_ids.iter().enumerate() {
            let coverage_id = artifact.coverage_row_ids.get(source_slot).copied().unwrap_or(u32::MAX);
            let id = *joined.entry((positive_id, coverage_id)).or_insert_with(|| {
                let positive = interned_terminal_row(
                    &artifact.positive_row_ids, &artifact.positive_offsets,
                    &artifact.positive_terminals, source_slot);
                let mut row = MasterProofRow {
                    positive: SmallVec::from_slice(positive),
                    negative: SmallVec::new(),
                };
                if explicit {
                    let coverage = interned_terminal_row(
                        &artifact.coverage_row_ids, &artifact.coverage_offsets,
                        &artifact.coverage_terminals, source_slot);
                    row.negative.extend(coverage.iter().copied()
                        .filter(|terminal| positive.binary_search(terminal).is_err()));
                }
                result.append_unique(&row)
            });
            result.row_ids.push(id);
        }
        result
    }

    /// Project the combined dictionary into the legacy wire format only
    /// when saving. Re-intern unique rows, not every source-state row.
    fn wire_rows(&self, include_negative: bool) -> InternedRows<TerminalID> {
        let mut result = if include_negative {
            intern_sorted_rows((0..self.ends.len()).map(|id| {
                let (positive, negative) = self.unique_row(id);
                let mut terminals = positive.iter().chain(negative).copied()
                    .collect::<SmallVec<[TerminalID; 8]>>();
                terminals.sort_unstable();
                terminals
            }))
        } else {
            intern_sorted_rows((0..self.ends.len()).map(|id| self.unique_row(id).0))
        };
        result.row_ids = self.row_ids.iter().map(|&id| result.row_ids[id as usize]).collect();
        result
    }
}

#[derive(Debug)]
struct PreparedMasterProofs {
    proofs: MasterProofRows,
    radii: InternedRows<(TerminalID, u16)>,
}

impl PreparedMasterProofs {
    fn from_wire(artifact: PreparedMasterProofArtifact) -> Self {
        let proofs = MasterProofRows::from_wire(&artifact);
        Self {
            proofs,
            radii: InternedRows {
                row_ids: artifact.safe_radius_row_ids,
                offsets: artifact.safe_radius_offsets,
                entries: artifact.safe_radius_entries,
            },
        }
    }

    fn to_wire(&self) -> PreparedMasterProofArtifact {
        let positive = self.proofs.wire_rows(false);
        let (coverage, complete) = match &self.proofs.coverage {
            MasterProofCoverage::Projected => (InternedRows::default(), Vec::new()),
            MasterProofCoverage::Explicit(complete) => (self.proofs.wire_rows(true), complete.to_vec()),
        };
        PreparedMasterProofArtifact {
            positive_row_ids: positive.row_ids,
            positive_offsets: positive.offsets,
            positive_terminals: positive.entries,
            coverage_row_ids: coverage.row_ids,
            coverage_offsets: coverage.offsets,
            coverage_terminals: coverage.entries,
            safe_plus_complete_terminals: complete,
            safe_radius_row_ids: self.radii.row_ids.clone(),
            safe_radius_offsets: self.radii.offsets.clone(),
            safe_radius_entries: self.radii.entries.clone(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicBoundedObservationSets {
    pool: Arc<[U8Set]>,
    horizon16: Arc<[u32]>,
    horizon64: Arc<[u32]>,
}

impl DynamicBoundedObservationSets {
    pub(crate) fn from_raw(horizon16: &[U8Set], horizon64: &[U8Set]) -> Self {
        debug_assert_eq!(horizon16.len(), horizon64.len());
        let mut ids = FxHashMap::<U8Set, u32>::default();
        let mut pool = Vec::<U8Set>::new();
        let mut intern = |set: U8Set| -> u32 {
            if let Some(&id) = ids.get(&set) {
                return id;
            }
            let id = pool.len() as u32;
            pool.push(set);
            ids.insert(set, id);
            id
        };
        let horizon16 = horizon16
            .iter()
            .copied()
            .map(&mut intern)
            .collect::<Vec<_>>();
        let horizon64 = horizon64
            .iter()
            .copied()
            .map(&mut intern)
            .collect::<Vec<_>>();
        Self {
            pool: Arc::from(pool),
            horizon16: Arc::from(horizon16),
            horizon64: Arc::from(horizon64),
        }
    }

    #[inline]
    pub(crate) fn safe_bytes(&self, state: u32, required_horizon: u32) -> Option<U8Set> {
        let ids = if required_horizon <= 16 {
            self.horizon16.as_ref()
        } else if required_horizon <= 64 {
            self.horizon64.as_ref()
        } else {
            return None;
        };
        let id = *ids.get(state as usize)? as usize;
        self.pool.get(id).copied()
    }

    #[inline]
    pub(crate) fn state_count(&self) -> usize {
        self.horizon16.len()
    }

    #[inline]
    pub(crate) fn unique_set_count(&self) -> usize {
        self.pool.len()
    }
}


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskVocabArtifact {
    trie: DynamicMaskTrie,
    token_aliases: DynamicMaskAliasStore,
    llg_slice_leftovers: Vec<Arc<DynamicMaskSliceTrie>>,
    mask_tokenizer: Option<Tokenizer>,
    full_to_mask_state: Vec<u32>,
    grammar_quotiented: bool,
    source_vocab_digest: Option<[u8; 32]>,
}

/// Runtime-only lazily determinized subset-state cache for scalar-dispatch mask execution.
/// Canonical subset states are shared across mask calls within one constraint runtime; this
/// is derived acceleration state, never serialized, and reset by
/// `fresh_runtime_instance`.
#[derive(Debug)]
pub(crate) struct DynamicLazyUnionMetadata {
    pub(crate) finalizer_code: u32,
    pub(crate) single_finalizer_continues: u8,
    pub(crate) matched: BitSet,
    pub(crate) futures: BitSet,
}

#[derive(Debug, Clone)]
pub(crate) enum AdaptiveByteMemo<const UNKNOWN: u32 = { u32::MAX - 1 }> {
    Sparse(SmallVec<[(u8, u32); 8]>),
    Dense(Box<[u32; 256]>),
}

impl<const UNKNOWN: u32> Default for AdaptiveByteMemo<UNKNOWN> {
    fn default() -> Self {
        Self::Sparse(SmallVec::new())
    }
}

impl<const UNKNOWN: u32> AdaptiveByteMemo<UNKNOWN> {

    #[inline]
    pub(crate) fn get(&self, byte: u8) -> Option<u32> {
        match self {
            Self::Sparse(cells) => cells.iter().find_map(|&(b, value)| (b == byte).then_some(value)),
            Self::Dense(row) => {
                let value = row[byte as usize];
                (value != UNKNOWN).then_some(value)
            }
        }
    }

    pub(crate) fn insert(&mut self, byte: u8, value: u32) {
        match self {
            Self::Sparse(cells) => {
                if let Some((_, existing)) = cells.iter_mut().find(|(b, _)| *b == byte) {
                    *existing = value;
                } else if cells.len() < 8 {
                    cells.push((byte, value));
                } else {
                    let mut dense = Box::new([UNKNOWN; 256]);
                    for &(b, v) in cells.iter() {
                        dense[b as usize] = v;
                    }
                    dense[byte as usize] = value;
                    *self = Self::Dense(dense);
                }
            }
            Self::Dense(row) => row[byte as usize] = value,
        }
    }
}

/// One canonical exact physical subset, with its lazily computed observations.
#[derive(Debug)]
pub(crate) struct DynamicLazyUnionState {
    pub(crate) members: SmallVec<[u32; 8]>,
    pub(crate) row: AdaptiveByteMemo,
    pub(crate) metadata: Option<DynamicLazyUnionMetadata>,
}

#[derive(Debug, Default)]
pub(crate) struct DynamicLazyUnionCache {
    /// Exact immutable source identity; retaining the owner prevents address reuse.
    pub(crate) source: Option<Arc<Tokenizer>>,
    /// Allocated only for physical coordinates without a dense transition slab.
    pub(crate) base_rows: Vec<Option<Box<[u32; 256]>>>,
    /// Two-member sets have a lossless packed key. Wider sets are indexed
    /// by hash with exact equality against their sole owned member list.
    pub(crate) state_by_pair: FxHashMap<u64, u32>,
    pub(crate) state_by_hash: FxHashMap<u64, SmallVec<[u32; 1]>>,
    /// Bounded memo of unions of two physical or derived input coordinates.
    pub(crate) state_by_union_pair: FxHashMap<u64, u32>,
    pub(crate) states: Vec<DynamicLazyUnionState>,
    /// Learned output-polarity hint; cleared with the coordinate space.
    pub(crate) dense_output_hints: FxHashMap<u32, u8>,
}
/// One overlapping runtime slice language and the residual vocabulary trie
/// to walk after that language has been proved contained. The slice language
/// itself is not a compiler partition and may overlap/nest with other slices.
#[derive(Debug, Clone)]
enum SliceAliasRows {
    Shared { owner: Arc<CanonicalAliasRows>, preorder_offsets: Arc<Vec<u32>> },
    // Legacy standalone slice wire data may use a foreign canonical
    // coordinate. The flat preorder rows remain an independent authority.
    Flat { offsets: Arc<Vec<u32>>, originals: Arc<Vec<u32>> },
}

impl SliceAliasRows {
    fn offsets(&self) -> &[u32] {
        match self {
            Self::Shared { preorder_offsets, .. } => preorder_offsets,
            Self::Flat { offsets, .. } => offsets,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskSliceTrie {
    cache_id: u32,
    dfa: Arc<VocabPartitionDfa>,
    trie: Arc<DynamicMaskTrie>,
    full_walk_token_markers: Arc<Vec<u64>>,
    aliases: SliceAliasRows,
    slice_original_token_words: Arc<Vec<u32>>,
    /// Conservative byte family containing every byte of every current-vocab
    /// token in this slice. Proving this whole family parser-transparent through
    /// `slice_max_token_byte_len` is sufficient to skip every slice token.
    slice_token_bytes: U8Set,
    slice_max_token_byte_len: u32,
    /// First-byte set of the slice DFA start state. Immutable per slice;
    /// cached because the dynamic pre-collapse gate needs it on every mask.
    first_bytes: OnceLock<U8Set>,
}

impl DynamicMaskSliceTrie {
    #[inline(always)]
    pub(crate) fn cache_id(&self) -> u32 {
        self.cache_id
    }

    #[inline(always)]
    pub(crate) fn dfa(&self) -> &VocabPartitionDfa {
        self.dfa.as_ref()
    }

    /// First-byte set of the slice DFA start state: the exact first-byte
    /// language of this proof slice. Only bytes whose one-byte derivative
    /// can still reach an accepting slice word are relevant here. This is
    /// deliberately computed from the DFA rather than from the bytes that
    /// occur anywhere inside whole slice tokens (UTF-8 continuation bytes,
    /// for example, are not valid first bytes). Immutable per slice;
    /// computed once because the dynamic pre-collapse gate needs it on
    /// every mask.
    pub(crate) fn first_bytes(&self) -> U8Set {
        *self.first_bytes.get_or_init(|| {
            let dfa = self.dfa.as_ref();
            let start = dfa.start_state();
            let mut result = U8Set::empty();
            for byte in 0u16..=255 {
                let byte = byte as u8;
                if dfa.can_reach_accepting(dfa.step(start, byte)) {
                    result.insert(byte);
                }
            }
            result
        })
    }

    #[inline(always)]
    pub(crate) fn trie(&self) -> &DynamicMaskTrie {
        self.trie.as_ref()
    }

    #[inline(always)]
    fn full_walk_token_markers(&self) -> &[u64] {
        self.full_walk_token_markers.as_ref()
    }

    /// Share only if every logical row has exactly the same original IDs.
    /// Foreign canonical coordinates remain independently flat.
    fn try_share_aliases(&mut self, owner: &Arc<CanonicalAliasRows>) {
        let SliceAliasRows::Flat { offsets, originals } = &self.aliases else { return; };
        let matches = self.trie.all_subtree_tokens().iter().enumerate().all(|(ordinal, &canonical)| {
            let index = canonical as usize;
            let Some(bounds) = owner.canonical_original_token_offsets.get(index..index + 2) else {
                return false;
            };
            let start = offsets[ordinal] as usize;
            let end = offsets[ordinal + 1] as usize;
            originals[start..end] == owner.canonical_original_tokens[bounds[0] as usize..bounds[1] as usize]
        });
        if matches {
            let preorder_offsets = DynamicMaskVocab::permuted_alias_offsets(
                &self.trie, &owner.canonical_original_token_offsets,
            ).map_or_else(|| Arc::clone(&owner.canonical_original_token_offsets), Arc::new);
            self.aliases = SliceAliasRows::Shared { owner: Arc::clone(owner), preorder_offsets };
        }
    }
    fn original_aliases_at(&self, ordinal: usize) -> WalkMarkerAliases<'_> {
        match &self.aliases {
            SliceAliasRows::Shared { owner, .. } => {
                let canonical = self.trie.all_subtree_tokens()[ordinal];
                WalkMarkerAliases::Words(owner.word_masks(canonical))
            }
            SliceAliasRows::Flat { offsets, originals } => {
                let start = offsets[ordinal] as usize;
                let end = offsets[ordinal + 1] as usize;
                WalkMarkerAliases::Tokens(&originals[start..end])
            }
        }
    }

    #[inline(always)]
    fn subtree_original_tokens(&self, node: u32) -> SubtreeOriginalTokens<'_> {
        let range = self.trie.subtree_token_index_range(node);
        match &self.aliases {
            SliceAliasRows::Shared { owner, preorder_offsets } => {
                let start = preorder_offsets[range.start] as usize;
                let end = preorder_offsets[range.end] as usize;
                if Arc::ptr_eq(preorder_offsets, &owner.canonical_original_token_offsets) {
                    SubtreeOriginalTokens::Flat(owner.canonical_original_tokens[start..end].iter())
                } else {
                    SubtreeOriginalTokens::Mapped {
                        canonical: self.trie.all_subtree_tokens()[range].iter(),
                        offsets: &owner.canonical_original_token_offsets,
                        originals: &owner.canonical_original_tokens,
                        current: [].iter(),
                        remaining: end - start,
                    }
                }
            }
            SliceAliasRows::Flat { offsets, originals } => SubtreeOriginalTokens::Flat(
                originals[offsets[range.start] as usize..offsets[range.end] as usize].iter(),
            ),
        }
    }

    #[inline(always)]
    pub(crate) fn slice_original_token_words(&self) -> &[u32] {
        self.slice_original_token_words.as_ref()
    }

    #[inline(always)]
    pub(crate) fn slice_token_bytes(&self) -> U8Set {
        self.slice_token_bytes
    }

    #[inline(always)]
    pub(crate) fn slice_max_token_byte_len(&self) -> u32 {
        self.slice_max_token_byte_len
    }
}

struct LegacySliceOriginals<'a>(&'a DynamicMaskSliceTrie);

impl serde::Serialize for LegacySliceOriginals<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let originals = self.0.subtree_original_tokens(0);
        let mut seq = serializer.serialize_seq(Some(originals.len()))?;
        for token in originals { seq.serialize_element(token)?; }
        seq.end()
    }
}

impl serde::Serialize for DynamicMaskSliceTrie {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut wire = serializer.serialize_struct("DynamicMaskSliceTrie", 9)?;
        wire.serialize_field("cache_id", &self.cache_id)?;
        wire.serialize_field("dfa", &self.dfa)?;
        wire.serialize_field("trie", &self.trie)?;
        wire.serialize_field("full_walk_token_markers", &self.full_walk_token_markers)?;
        wire.serialize_field("subtree_original_token_offsets", self.aliases.offsets())?;
        wire.serialize_field("subtree_original_tokens", &LegacySliceOriginals(self))?;
        wire.serialize_field("slice_original_token_words", &self.slice_original_token_words)?;
        wire.serialize_field("slice_token_bytes", &self.slice_token_bytes)?;
        wire.serialize_field("slice_max_token_byte_len", &self.slice_max_token_byte_len)?;
        wire.end()
    }
}

#[derive(serde::Deserialize)]
#[serde(rename = "DynamicMaskSliceTrie")]
struct LegacySliceWire {
    cache_id: u32,
    dfa: Arc<VocabPartitionDfa>,
    trie: Arc<DynamicMaskTrie>,
    #[serde(rename = "full_walk_token_markers")]
    _full_walk_token_markers: DiscardedSequence<u64>,
    subtree_original_token_offsets: Arc<Vec<u32>>,
    subtree_original_tokens: Arc<Vec<u32>>,
    slice_original_token_words: Arc<Vec<u32>>,
    slice_token_bytes: U8Set,
    slice_max_token_byte_len: u32,
}

impl<'de> serde::Deserialize<'de> for DynamicMaskSliceTrie {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let wire = <LegacySliceWire as serde::Deserialize>::deserialize(deserializer)?;
        let offsets = &wire.subtree_original_token_offsets;
        let count = wire.trie.all_subtree_tokens().len();
        if offsets.len() != count + 1 || offsets.first() != Some(&0)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || offsets.last().copied().map(|n| n as usize) != Some(wire.subtree_original_tokens.len())
            || wire.subtree_original_tokens.iter().any(|&id| id > crate::MAX_TOKEN_ID)
        {
            return Err(D::Error::custom("invalid dynamic mask slice alias rows"));
        }
        let markers = wire.trie.all_subtree_tokens().iter().enumerate()
            .skip(usize::from(wire.trie.node(0).token_id.is_some()))
            .map(|(ordinal, &canonical)| {
                let start = offsets[ordinal] as usize;
                let end = offsets[ordinal + 1] as usize;
                DynamicMaskVocab::encode_walk_marker(canonical, &wire.subtree_original_tokens[start..end])
            }).collect();
        Ok(Self {
            cache_id: wire.cache_id, dfa: wire.dfa, trie: wire.trie,
            full_walk_token_markers: Arc::new(markers),
            aliases: SliceAliasRows::Flat {
                offsets: wire.subtree_original_token_offsets,
                originals: wire.subtree_original_tokens,
            },
            slice_original_token_words: wire.slice_original_token_words,
            slice_token_bytes: wire.slice_token_bytes,
            slice_max_token_byte_len: wire.slice_max_token_byte_len,
            first_bytes: OnceLock::new(),
        })
    }
}

type TerminalQuotient = Arc<TerminalProjectedQuotient>;

/// A prepared inventory is authoritative for availability, including an
/// intentionally empty sidecar. Otherwise all successful and declined
/// runtime preparations share one terminal-keyed inventory.
#[derive(Debug, Clone)]
enum ProjectedTerminalInventory {
    Prepared(Arc<[(TerminalID, TerminalQuotient)]>),
    Runtime(Arc<RuntimeTerminalInventory>),
}

#[derive(Debug, Default)]
struct RuntimeTerminalInventory {
    by_terminal: Mutex<FxHashMap<TerminalID, Option<TerminalQuotient>>>,
    /// One eager batch attempt, not a second quotient store.
    batch_complete: OnceLock<()>,
}

impl Default for ProjectedTerminalInventory {
    fn default() -> Self {
        Self::Runtime(Arc::new(RuntimeTerminalInventory::default()))
    }
}

/// Exact mapping into a finite mask language. The variants are mutually
/// exclusive; in particular virtual coordinates are not dense source IDs.
#[derive(Debug, Clone)]
enum MaskCoordinateMap {
    Dense(Arc<[u32]>),
    UnitRepeat(VirtualZeroMinUnitRepeatMaskProjection),
    RepeatIntersections(Arc<[VirtualBinaryRepeatIntersectionMaskProjection]>),
    Residuals(Arc<[VirtualResidualMaskProjection]>),
}

impl MaskCoordinateMap {
    fn project(&self, full_state: u32) -> u32 {
        match self {
            Self::Dense(states) => states.get(full_state as usize).copied().unwrap_or(full_state),
            Self::UnitRepeat(projection) => projection.project(full_state)
                .expect("arithmetic lexer state has no owning finite-mask projection"),
            Self::RepeatIntersections(projections) => projections.iter()
                .find_map(|projection| projection.project(full_state))
                .expect("repeat intersection lexer state has no owning finite-mask projection"),
            Self::Residuals(projections) => projections.iter()
                .find_map(|projection| projection.project(full_state))
                .unwrap_or_else(|| {
                    // Ordinary physical states retain their IDs.
                    assert!(projections.iter().all(|projection| full_state < projection.physical_state_count()),
                        "residual lexer state {full_state} has no owning finite-mask projection");
                    full_state
                }),
        }
    }
}

#[derive(Debug, Clone)]
struct MaskExecutionDerivative {
    tokenizer: Arc<Tokenizer>,
    from_projection: Arc<[u32]>,
}

#[derive(Debug)]
struct MaskSubsetProvenance {
    by_state: Vec<Arc<[u32]>>,
    by_subset: FxHashMap<Arc<[u32]>, u32>,
}

#[derive(Debug, Clone)]
struct MaskProjection {
    tokenizer: Arc<Tokenizer>,
    map: MaskCoordinateMap,
    derivative: Option<MaskExecutionDerivative>,
    /// Subsets in exact-source coordinates for direct determinization, or
    /// in base-projection coordinates when a derivative is present.
    provenance: Option<Arc<MaskSubsetProvenance>>,
}

impl MaskProjection {
    fn runtime_owner(&self) -> &Arc<Tokenizer> {
        self.derivative.as_ref().map_or(&self.tokenizer, |derived| &derived.tokenizer)
    }
}

/// Cumulative safe-string masks and one orthogonal whitespace mask. The
/// complete safe/whitespace slice masks share these same allocations.
#[derive(Debug)]
struct MasterAdmissionWords {
    safe_prefixes: Vec<Arc<Vec<u32>>>,
    whitespace: Arc<Vec<u32>>,
}

impl MasterAdmissionWords {
    fn from_exact(mut exact_safe: Vec<Vec<u32>>, whitespace: Arc<Vec<u32>>) -> Self {
        debug_assert!(!exact_safe.is_empty());
        for radius in 1..exact_safe.len() {
            let (before, after) = exact_safe.split_at_mut(radius);
            for (word, &previous) in after[0].iter_mut().zip(&before[radius - 1]) {
                *word |= previous;
            }
        }
        Self {
            safe_prefixes: exact_safe.into_iter().map(Arc::new).collect(),
            whitespace,
        }
    }

    fn max_radius(&self) -> u16 {
        (self.safe_prefixes.len() - 1) as u16
    }
}

#[derive(Debug, Clone)]
enum VocabAliasState {
    Empty,
    Deferred(DynamicMaskVocabSource),
    Transfer(Arc<Vec<Vec<u32>>>),
    Runtime(Arc<RuntimeVocabIndexes>),
}

/// Exact subtree alias iterator: a contiguous slice for identity order and
/// wire-owned slices, or a concatenation of borrowed canonical alias rows.
pub(crate) enum SubtreeOriginalTokens<'a> {
    Flat(std::slice::Iter<'a, u32>),
    Mapped {
        canonical: std::slice::Iter<'a, u32>,
        offsets: &'a [u32],
        originals: &'a [u32],
        current: std::slice::Iter<'a, u32>,
        remaining: usize,
    },
}

impl<'a> SubtreeOriginalTokens<'a> {
    pub(crate) fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<'a> Iterator for SubtreeOriginalTokens<'a> {
    type Item = &'a u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Flat(iter) => iter.next(),
            Self::Mapped { canonical, offsets, originals, current, remaining } => {
                loop {
                    if let Some(token) = current.next() {
                        *remaining -= 1;
                        return Some(token);
                    }
                    let id = *canonical.next()? as usize;
                    let originals: &'a [u32] = *originals;
                    *current = originals[offsets[id] as usize..offsets[id + 1] as usize].iter();
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            Self::Flat(iter) => iter.len(),
            Self::Mapped { remaining, .. } => *remaining,
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for SubtreeOriginalTokens<'_> {}

/// The encoded marker's direct word is coordinate-free. Its fallback is
/// not: inherited slice tries may use an entirely different canonical ID
/// space. Resolve the owner once before walking, and use preorder ordinals
/// for legacy slice alias rows.
pub(crate) enum WalkMarkerAliases<'a> {
    Words(&'a [(u32, u32)]),
    Tokens(&'a [u32]),
}

#[derive(Clone, Copy)]
enum WalkMarkerOwner<'a> {
    Main(&'a DynamicMaskVocab),
    Slice(&'a DynamicMaskSliceTrie),
}

pub(crate) struct WalkTokenMarkers<'a> {
    pub(crate) values: &'a [u64],
    owner: WalkMarkerOwner<'a>,
}

impl<'a> WalkTokenMarkers<'a> {
    pub(crate) fn fallback(&self, ordinal: usize, canonical: u32) -> WalkMarkerAliases<'a> {
        match self.owner {
            WalkMarkerOwner::Main(vocab) => WalkMarkerAliases::Words(vocab.token_word_masks(canonical)),
            WalkMarkerOwner::Slice(slice) => {
                let index = ordinal + usize::from(slice.trie.node(0).token_id.is_some());
                slice.original_aliases_at(index)
            }
        }
    }

    pub(crate) fn root(&self) -> Option<WalkMarkerAliases<'a>> {
        match self.owner {
            WalkMarkerOwner::Main(vocab) => vocab.trie.node(0).token_id
                .map(|canonical| WalkMarkerAliases::Words(vocab.token_word_masks(canonical))),
            WalkMarkerOwner::Slice(slice) => slice.trie.node(0).token_id
                .map(|_| slice.original_aliases_at(0)),
        }
    }
}

#[derive(Debug)]
struct CanonicalAliasRows {
    canonical_original_token_offsets: Arc<Vec<u32>>,
    canonical_original_tokens: Arc<Vec<u32>>,
    canonical_original_word_offsets: Arc<Vec<u32>>,
    canonical_original_word_masks: Arc<Vec<(u32, u32)>>,
}

impl CanonicalAliasRows {
    #[inline(always)]
    fn word_masks(&self, canonical: u32) -> &[(u32, u32)] {
        let index = canonical as usize;
        let start = self.canonical_original_word_offsets[index] as usize;
        let end = self.canonical_original_word_offsets[index + 1] as usize;
        &self.canonical_original_word_masks[start..end]
    }
}

#[derive(Debug)]
struct RuntimeVocabIndexes {
    aliases: Arc<CanonicalAliasRows>,
    full_walk_token_markers: Arc<Vec<u64>>,
    permuted_subtree_alias_offsets: Option<Vec<u32>>,
    all_original_token_words: Arc<Vec<u32>>,
}

impl std::ops::Deref for RuntimeVocabIndexes {
    type Target = CanonicalAliasRows;
    fn deref(&self) -> &Self::Target { &self.aliases }
}

/// Runtime-only vocabulary data for direct dynamic mask generation.
#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocab {
    pub(crate) trie: Arc<DynamicMaskTrie>,
    alias_state: VocabAliasState,
    llg_slice_leftovers: Arc<Vec<Arc<DynamicMaskSliceTrie>>>,
    /// Shared vocabulary-only acceleration; no duplicated whitespace variant.
    llg_master_admission: Option<Arc<MasterAdmissionWords>>,
    /// Exact remaining strict-walk work for (safe radius, whitespace).
    llg_master_residual_ops: Arc<Vec<[u32; 2]>>,
    /// One immutable inventory of exact parser-independent master proofs.
    /// Slot = exact source TSID * 2 + slice (safe+ or whitespace).
    prepared_master_proofs: Option<Arc<PreparedMasterProofs>>,
    /// True only when trie endpoints represent grammar-proven vocabulary
    /// equivalence classes rather than byte-identical token aliases.
    grammar_quotiented: bool,
    /// Strong identity of the complete original model vocabulary from which a
    /// grammar quotient was built. Ordinary dynamic vocabularies leave this
    /// unset; compiler-created O2 quotients carry it into transfer metadata.
    source_vocab_digest: Option<[u8; 32]>,
    mask_cache: Arc<Mutex<DynamicMaskCache>>,
    lazy_union_cache: Arc<Mutex<DynamicLazyUnionCache>>,
    direct_regular_frontier_cache:
        Arc<Mutex<FxHashMap<usize, DirectRegularDynamicFrontierCacheEntry>>>,
    direct_regular_wide_frontier_index_cache: Arc<Mutex<FxHashMap<usize, (ParserGSS, usize)>>>,
    direct_regular_terminal_support: Arc<DirectRegularTerminalSupport>,
    self_loop_projections: Arc<Vec<DynamicSelfLoopProjection>>,
    projection_by_source: Arc<[u32]>,
    projection_alias_vocab: Arc<[u32]>,
    projection_alias_h64: Arc<[u32]>,
    bounded_observation_sets: Arc<DynamicBoundedObservationSets>,
    terminal_observation_classes: Arc<[(TerminalID, Arc<[u32]>)]>,
    terminal_quotients: ProjectedTerminalInventory,
    /// Parser-independent exact projected-text proof results.  The key is the
    /// terminal residual coordinate plus the vocabulary alphabet being proved.
    /// Sharing this across sequences is safe because parser admission only
    /// decides whether a proof is queried; the proof result itself depends
    /// solely on immutable lexer/vocabulary data.
    projected_terminal_text_cache: Arc<Mutex<FxHashMap<(TerminalID, u32, U8Set, bool), bool>>>,
    /// Exact regular-language containment results for named proof slices.
    /// High key bits distinguish projected, symbolic-residual, and finite-direct
    /// proof namespaces over the same immutable terminal/source coordinates.
    projected_terminal_partition_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32), bool>>>,
    /// Exact bounded repetition radii for regular-language proof slices. The
    /// extra key component is the caller's maximum relevant repetition count;
    /// for model-token masking this is the largest safe-token scalar length.
    projected_terminal_radius_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32, u32), u32>>>,
    /// Conservative bounded certificates over exact physical source coordinates.
    /// Never shared with a different grammar's fresh runtime vocabulary instance.
    raw_terminal_radius_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32, u32, usize), u32>>>,
    /// Exact original-token masks rejected by a token-start maximal-munch
    /// guard. Keys are the canonical sorted `(mask lexer state, terminal)`
    /// memories carried by `InitialPruneGuard`. The result depends only on the
    /// immutable lexer/vocabulary coordinate, so sequences may safely share it.
    pending_guard_blocked_mask_cache:
        Arc<Mutex<FxHashMap<Vec<(u32, TerminalID)>, Arc<Vec<u32>>>>>,
    /// Mask-only finite language and its exact coordinate mapping.
    mask_projection: Option<MaskProjection>,
    mask_tokenizer_fast_transitions: Option<FastTokenizerTransitions>,
}

impl DynamicMaskVocab {
    const FULL_WALK_DENSE_TRANSITION_BYTES: usize = 64 * 1024 * 1024;
    pub(crate) const PREPARED_PROOF_SLOT_COUNT: usize = 2;
    const PREPARED_SAFE_PLUS_SLOT: usize = 0;
    const PREPARED_WHITESPACE_SLOT: usize = 1;

    #[inline]
    pub(crate) fn max_token_byte_len(&self) -> usize {
        self.trie
            .nodes
            .first()
            .map_or(0, |root| root.subtree_max_byte_len as usize)
    }

    fn build_full_walk_fast_transitions(tokenizer: &Tokenizer) -> Option<FastTokenizerTransitions> {
        // Scalar-dispatch projections deliberately execute through the lazy
        // physical-row/subset cache. A complete Flat16 slab is pure derived
        // acceleration and is too expensive to materialize before the first
        // mask; the lazy executor builds only rows actually reached by vocab
        // traversal. Ordinary deterministic tokenizers still use the dense
        // table below.
        if tokenizer.has_any_virtual_runtime() || tokenizer.has_epsilon_transitions() {
            return None;
        }
        FastTokenizerTransitions::full_walk_dense_for(
            tokenizer,
            Self::FULL_WALK_DENSE_TRANSITION_BYTES,
        )
    }

    /// Prepare the dense transition cells used exclusively by the strict full
    /// vocabulary walker. The mask projection, when present, is the exact
    /// coordinate masking advances; otherwise use the source runtime tokenizer.
    /// These tables are derived runtime data and are deliberately not serialized.
    pub(crate) fn prepare_full_walk_fast_transitions(&mut self, source: &Tokenizer) {
        // Every operation that changes the active mask tokenizer either builds
        // this derived table for the new coordinate or explicitly clears it.
        // Dynamic finalization calls this preparation step again after choosing
        // the final coordinate, so rebuilding an already-present table here is
        // duplicate work (and is material for large finite residual projections).
        if self.mask_tokenizer_fast_transitions.is_some() {
            return;
        }
        let tokenizer = self.mask_runtime_tokenizer().unwrap_or(source);
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(tokenizer);
    }

    /// Prepare a deterministic execution representation for the exact finite
    /// lexer coordinate used by mask generation. The source constraint lexer
    /// remains unchanged; this is analogous to choosing Flat16 versus Flat32
    /// for the strict mask walk. If the derived representation does not fit,
    /// the strict walker executes the epsilon-NFA coordinate directly.
    pub(crate) fn prepare_mask_execution(
        &mut self,
        source_tokenizer: &Tokenizer,
    ) -> bool {
        // Bound eager work by the same memory budget that decides whether the
        // result can use Flat32. This avoids introducing an independent
        // state-count determinization policy on top of lexer compilation.
        const CELL_BYTES: usize = std::mem::size_of::<u32>();
        const ALPHABET: usize = 256;
        let state_limit = Self::FULL_WALK_DENSE_TRANSITION_BYTES / (CELL_BYTES * ALPHABET);
        let transition_limit = state_limit.saturating_mul(ALPHABET);
        let source = self.mask_projection_tokenizer().unwrap_or(source_tokenizer);
        if source.has_any_virtual_runtime() {
            return false;
        }
        if !source.has_epsilon_transitions() {
            let fast = Self::build_full_walk_fast_transitions(source);
            if let Some(projection) = self.mask_projection.as_mut() {
                if projection.derivative.take().is_some() {
                    projection.provenance = None;
                }
            }
            self.mask_tokenizer_fast_transitions = fast;
            return self.mask_tokenizer_fast_transitions.is_some();
        }
        let Some((built, source_to_determinized)) = source
            .try_reusing_full_determinization_all_starts(state_limit, transition_limit)
            .or_else(|| source.try_full_determinization_all_starts(state_limit, transition_limit))
        else {
            return false;
        };
        let source_subsets = built.source_subsets;
        let deterministic = built.tokenizer;
        let Some(fast) = Self::build_full_walk_fast_transitions(&deterministic) else {
            return false;
        };
        if let Some(projection) = self.mask_projection.as_mut() {
            projection.derivative = Some(MaskExecutionDerivative {
                tokenizer: Arc::new(deterministic),
                from_projection: Arc::from(source_to_determinized),
            });
            projection.provenance = None;
            self.mask_tokenizer_fast_transitions = Some(fast);
        } else {
            self.publish_mask_projection(
                deterministic,
                MaskCoordinateMap::Dense(Arc::from(source_to_determinized)),
                Some(fast),
            );
        }
        self.set_mask_tokenizer_source_subsets(source_subsets);
        true
    }

    pub(crate) fn from_compiler_artifacts(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_source(DynamicMaskVocabSource { trie, token_aliases })
    }

    pub(crate) fn from_compiler_artifacts_materialized(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        let mut vocab = Self::from_compiler_artifacts(trie, token_aliases);
        let materialized = vocab.materialize_pending_source();
        debug_assert!(materialized);
        vocab
    }

    pub(crate) fn from_materialized_ordered(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_materialized_ordered_with_all_original_token_words(
            trie,
            token_aliases,
            None,
        )
    }

    pub(crate) fn from_materialized_ordered_with_all_original_token_words(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
        prepared_all_original_token_words: Option<Arc<Vec<u32>>>,
    ) -> Self {
        let (offsets, originals) = Self::flatten_owned_ordered_aliases(token_aliases);
        Self::from_ordered_alias_parts(
            trie, offsets, originals, prepared_all_original_token_words,
        )
    }

    fn flatten_owned_ordered_aliases(
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        match Arc::try_unwrap(token_aliases) {
            Ok(groups) => {
                let mut offsets = Vec::with_capacity(groups.len() + 1);
                offsets.push(0);
                for group in &groups {
                    offsets.push(offsets.last().copied().unwrap_or(0) + group.len() as u32);
                }
                let total = offsets.last().copied().unwrap_or(0) as usize;
                let mut iter = groups.into_iter();
                let mut originals = iter.next().unwrap_or_default();
                originals.reserve(total.saturating_sub(originals.len()));
                for group in iter {
                    originals.extend_from_slice(&group);
                }
                (Arc::new(offsets), Arc::new(originals))
            }
            Err(groups) => Self::flatten_canonical_original_tokens(
                &DynamicMaskAliasStore::Ordered(groups),
            ),
        }
    }

    /// Build the same owned runtime vocabulary from preflattened, sorted aliases.
    /// Each offsets interval corresponds to one canonical trie token.
    pub(crate) fn from_materialized_flat_ordered(
        trie: Arc<DynamicMaskTrie>,
        offsets: Arc<Vec<u32>>,
        originals: Arc<Vec<u32>>,
    ) -> Self {
        assert_eq!(offsets.first(), Some(&0));
        assert_eq!(offsets.last().copied().map(|n| n as usize), Some(originals.len()));
        assert!(offsets.windows(2).all(|p| p[0] <= p[1]));
        debug_assert!(offsets.windows(2).all(|p| originals[p[0] as usize..p[1] as usize]
            .windows(2).all(|q| q[0] <= q[1])));
        Self::from_ordered_alias_parts(trie, offsets, originals, None)
    }

    fn from_ordered_alias_parts(
        trie: Arc<DynamicMaskTrie>,
        offsets: Arc<Vec<u32>>,
        originals: Arc<Vec<u32>>,
        prepared_all_original_token_words: Option<Arc<Vec<u32>>>,
    ) -> Self {
        let indexes = Self::build_vocab_indexes(
            trie.as_ref(), offsets, originals, true,
            prepared_all_original_token_words,
        );
        Self {
            trie,
            alias_state: VocabAliasState::Runtime(Arc::new(indexes)),
            ..Self::default()
        }
    }

    fn build_vocab_indexes(
        trie: &DynamicMaskTrie,
        canonical_original_token_offsets: Arc<Vec<u32>>,
        canonical_original_tokens: Arc<Vec<u32>>,
        ordered: bool,
        prepared_all_original_token_words: Option<Arc<Vec<u32>>>,
    ) -> RuntimeVocabIndexes {
        let build_words = || {
            if ordered {
                Self::build_canonical_original_word_masks_sorted(
                    &canonical_original_token_offsets, &canonical_original_tokens,
                )
            } else {
                Self::build_canonical_original_word_masks(
                    &canonical_original_token_offsets, &canonical_original_tokens,
                )
            }
        };
        let build_markers = || Self::build_full_walk_token_markers(
            trie, &canonical_original_token_offsets, &canonical_original_tokens,
        );
        let build_subtree_and_words = || {
            let subtree_tokens = trie.all_subtree_tokens();
            let offsets = Self::permuted_alias_offsets(trie, &canonical_original_token_offsets);
            let identity = offsets.is_none();
            let words = prepared_all_original_token_words.unwrap_or_else(|| {
                if ordered || identity {
                    Self::build_all_original_token_words(canonical_original_tokens.iter().copied())
                } else {
                    Self::build_all_original_token_words(subtree_tokens.iter().flat_map(|&canonical| {
                        let index = canonical as usize;
                        canonical_original_tokens[
                            canonical_original_token_offsets[index] as usize..
                            canonical_original_token_offsets[index + 1] as usize
                        ].iter().copied()
                    }))
                }
            });
            (offsets, words)
        };
        let ((canonical_original_word_offsets, canonical_original_word_masks),
            full_walk_token_markers, (permuted_subtree_alias_offsets, all_original_token_words)) =
            if canonical_original_tokens.len() >= 4_096 && rayon::current_num_threads() > 1 {
                let ((words, markers), subtree) = rayon::join(
                    || rayon::join(build_words, build_markers),
                    build_subtree_and_words,
                );
                (words, markers, subtree)
            } else {
                (build_words(), build_markers(), build_subtree_and_words())
            };
        RuntimeVocabIndexes {
            aliases: Arc::new(CanonicalAliasRows {
                canonical_original_token_offsets,
                canonical_original_tokens,
                canonical_original_word_offsets,
                canonical_original_word_masks,
            }),
            full_walk_token_markers,
            permuted_subtree_alias_offsets,
            all_original_token_words,
        }
    }

    /// A transfer-only vocabulary retains the exact alias relation without
    /// building indexes until its first runtime use.
    pub(crate) fn from_materialized_ordered_for_transfer(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self {
            trie,
            alias_state: VocabAliasState::Transfer(token_aliases),
            ..Self::default()
        }
    }

    /// Share vocabulary-only data, resetting all constraint-specific caches.
    pub(crate) fn fresh_runtime_instance(&self) -> Self {
        debug_assert!(self.is_initialized());
        Self {
            trie: Arc::clone(&self.trie),
            alias_state: self.alias_state.clone(),
            llg_slice_leftovers: Arc::clone(&self.llg_slice_leftovers),
            llg_master_admission: self.llg_master_admission.clone(),
            llg_master_residual_ops: Arc::clone(&self.llg_master_residual_ops),
            grammar_quotiented: self.grammar_quotiented,
            source_vocab_digest: self.source_vocab_digest,
            ..Self::default()
        }
    }

    fn from_source(source: DynamicMaskVocabSource) -> Self {
        Self { alias_state: VocabAliasState::Deferred(source), ..Self::default() }
    }

    pub(crate) fn mark_grammar_quotiented(&mut self) {
        self.grammar_quotiented = true;
    }

    pub(crate) fn set_source_vocab_digest(&mut self, digest: [u8; 32]) {
        self.source_vocab_digest = Some(digest);
    }

    pub(crate) fn source_vocab_digest(&self) -> Option<[u8; 32]> {
        self.source_vocab_digest
    }

    pub(crate) fn is_grammar_quotiented(&self) -> bool {
        self.grammar_quotiented
    }

    pub(crate) fn from_packed(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>,
    ) -> Self {
        Self::from_alias_store(trie, DynamicMaskAliasStore::Packed(token_aliases))
    }

    fn from_alias_store(trie: Arc<DynamicMaskTrie>, token_aliases: DynamicMaskAliasStore) -> Self {
        let (offsets, originals) = Self::flatten_canonical_original_tokens(&token_aliases);
        let indexes = Self::build_vocab_indexes(&trie, offsets, originals, false, None);
        Self {
            trie,
            alias_state: VocabAliasState::Runtime(Arc::new(indexes)),
            ..Self::default()
        }
    }

    /// The trie and exact aliases are available for transfer and bounded queries.
    pub(crate) fn is_initialized(&self) -> bool {
        matches!(self.alias_state, VocabAliasState::Transfer(_) | VocabAliasState::Runtime(_))
    }

    pub(crate) fn has_runtime_indexes(&self) -> bool {
        matches!(self.alias_state, VocabAliasState::Runtime(_))
    }

    #[inline]
    fn indexes(&self) -> &RuntimeVocabIndexes {
        match &self.alias_state {
            VocabAliasState::Runtime(indexes) => indexes,
            _ => panic!("runtime vocabulary indexes have not been materialized"),
        }
    }

    pub(crate) fn materialize_pending_source(&mut self) -> bool {
        let (aliases, ordered) = match std::mem::replace(&mut self.alias_state, VocabAliasState::Empty) {
            VocabAliasState::Deferred(source) => {
                self.trie = Arc::new(DynamicMaskTrie::from_vocab_prefix_tree(source.trie.as_ref()));
                (source.token_aliases, false)
            }
            VocabAliasState::Transfer(aliases) => (aliases, true),
            state => {
                self.alias_state = state;
                return false;
            }
        };
        let (offsets, originals) = Self::flatten_owned_ordered_aliases(aliases);
        self.alias_state = VocabAliasState::Runtime(Arc::new(Self::build_vocab_indexes(
            &self.trie, offsets, originals, ordered, None,
        )));
        true
    }

    fn flatten_canonical_original_tokens(
        token_aliases: &DynamicMaskAliasStore,
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        if let DynamicMaskAliasStore::Flat { offsets, originals } = token_aliases {
            return (Arc::clone(offsets), Arc::clone(originals));
        }
        let alias_slots = match token_aliases {
            DynamicMaskAliasStore::Ordered(aliases) => aliases.len(),
            DynamicMaskAliasStore::Flat { .. } => unreachable!("flat alias store returned above"),
            DynamicMaskAliasStore::Packed(aliases) => aliases.len(),
        };
        let mut offsets = Vec::with_capacity(alias_slots + 1);
        let mut originals = Vec::new();
        offsets.push(0);
        for canonical_token in 0..alias_slots {
            match token_aliases {
                DynamicMaskAliasStore::Ordered(aliases) => {
                    originals.extend_from_slice(&aliases[canonical_token]);
                }
                DynamicMaskAliasStore::Flat { .. } => unreachable!("flat alias store returned above"),
                DynamicMaskAliasStore::Packed(aliases) => {
                    if let Some(alias) = aliases[canonical_token].as_ref() {
                        match alias {
                            PackedDynamicMaskTokenAliases::Single(token_id) => {
                                originals.push(*token_id);
                            }
                            PackedDynamicMaskTokenAliases::Many(token_ids) => {
                                originals.extend_from_slice(token_ids);
                            }
                        }
                    }
                }
            }
            offsets.push(originals.len() as u32);
        }
        (Arc::new(offsets), Arc::new(originals))
    }

    fn build_canonical_original_word_masks(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<(u32, u32)>>) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        let word_len = canonical_original_tokens
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let mut scratch = vec![0u32; word_len];
        let mut touched = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            for &token_id in &canonical_original_tokens[start..end] {
                let word = token_id / 32;
                let slot = unsafe { scratch.get_unchecked_mut(word as usize) };
                if *slot == 0 {
                    touched.push(word);
                }
                *slot |= 1u32 << (token_id % 32);
            }
            for word in touched.drain(..) {
                let bits = unsafe { *scratch.get_unchecked(word as usize) };
                debug_assert_ne!(bits, 0);
                masks.push((word, bits));
                unsafe { *scratch.get_unchecked_mut(word as usize) = 0; }
            }
            offsets.push(masks.len() as u32);
        }
        (Arc::new(offsets), Arc::new(masks))
    }

    /// Faster exact mask construction for the ordered vocabulary coordinate.
    /// Each canonical alias list is sorted by original token ID before this
    /// constructor is called, so equal 32-token words are contiguous.  Build
    /// sparse word masks in one linear pass rather than clearing/probing a
    /// model-vocabulary-sized scratch bitset for every canonical class.
    fn build_canonical_original_word_masks_sorted(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<(u32, u32)>>) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            let aliases = &canonical_original_tokens[start..end];
            debug_assert!(aliases.windows(2).all(|pair| pair[0] <= pair[1]));
            let mut current_word = u32::MAX;
            let mut bits = 0u32;
            for &token_id in aliases {
                let word = token_id / 32;
                if word != current_word {
                    if current_word != u32::MAX {
                        masks.push((current_word, bits));
                    }
                    current_word = word;
                    bits = 0;
                }
                bits |= 1u32 << (token_id % 32);
            }
            if current_word != u32::MAX {
                masks.push((current_word, bits));
            }
            offsets.push(masks.len() as u32);
        }
        (Arc::new(offsets), Arc::new(masks))
    }

    fn build_all_original_token_words(originals: impl Iterator<Item = u32> + Clone) -> Arc<Vec<u32>> {
        let word_len = originals.clone().max()
            .map_or(0, |token| token as usize / 32 + 1);
        let mut words = vec![0u32; word_len];
        for token in originals {
            words[token as usize / 32] |= 1u32 << (token % 32);
        }
        Arc::new(words)
    }

    #[inline]
    pub(crate) fn all_original_token_words(&self) -> &[u32] {
        self.indexes().all_original_token_words.as_ref()
    }

    pub(crate) fn all_original_token_words_arc(&self) -> Arc<Vec<u32>> {
        Arc::clone(&self.indexes().all_original_token_words)
    }

    /// Reuse only vocabulary-global proof languages and admitted-token masks.
    /// The master walk trie itself is deliberately excluded: a grammar quotient
    /// can merge tokens that differ in safe-string length/whitespace class, so
    /// sharing the full-vocabulary master trie would both lose the O2 runtime
    /// reduction and couple the wrong canonical coordinate to this constraint.
    pub(crate) fn inherit_llg_vocab_acceleration_without_master_from(&mut self, source: &Self) {
        self.llg_slice_leftovers = Arc::new(
            source
                .llg_slice_leftovers
                .iter()
                .filter(|slice| slice.cache_id() != DYNAMIC_MASK_LLG_MASTER_CACHE_ID)
                .cloned()
                .collect(),
        );
        self.llg_master_admission = source.llg_master_admission.clone();
        self.llg_master_residual_ops = Arc::new(Vec::new());
    }

    pub(crate) fn set_llg_slice_leftovers(
        &mut self,
        slices: Vec<(
            u32,
            Arc<VocabPartitionDfa>,
            Arc<DynamicMaskTrie>,
            Arc<Vec<u32>>,
            U8Set,
            u32,
        )>,
    ) {
        let mut built = Vec::with_capacity(slices.len());
        for (
            cache_id,
            dfa,
            trie,
            slice_original_token_words,
            slice_token_bytes,
            slice_max_token_byte_len,
        ) in slices
        {
            let full_walk_token_markers = Self::build_full_walk_token_markers(
                trie.as_ref(), &self.indexes().canonical_original_token_offsets,
                &self.indexes().canonical_original_tokens,
            );
            let owner = Arc::clone(&self.indexes().aliases);
            let preorder_offsets = Self::permuted_alias_offsets(trie.as_ref(), &owner.canonical_original_token_offsets)
                .map_or_else(|| Arc::clone(&owner.canonical_original_token_offsets), Arc::new);
            built.push(Arc::new(DynamicMaskSliceTrie {
                cache_id,
                dfa,
                trie,
                full_walk_token_markers,
                aliases: SliceAliasRows::Shared { owner, preorder_offsets },
                slice_original_token_words,
                slice_token_bytes,
                slice_max_token_byte_len,
                first_bytes: OnceLock::new(),
            }));
        }
        self.llg_slice_leftovers = Arc::new(built);
        self.rebuild_llg_master_residual_ops_from_master_trie();
    }

    /// Publish cumulative safe masks in place, sharing the complete masks
    /// with the ordinary safe and whitespace slice acceleration.
    pub(crate) fn set_llg_master_admitted_words(
        &mut self,
        exact_safe_words: Vec<Vec<u32>>,
        whitespace_words: Arc<Vec<u32>>,
    ) -> Arc<Vec<u32>> {
        let admission = MasterAdmissionWords::from_exact(exact_safe_words, whitespace_words);
        let complete_safe = Arc::clone(admission.safe_prefixes.last().expect("radius zero exists"));
        self.llg_master_admission = Some(Arc::new(admission));
        complete_safe
    }

    #[inline(always)]
    pub(crate) fn copy_llg_master_admitted_words(
        &self,
        safe_radius: u16,
        whitespace: bool,
        output: &mut [u32],
    ) -> bool {
        let Some(admission) = &self.llg_master_admission else { return false; };
        let words = &admission.safe_prefixes[safe_radius.min(admission.max_radius()) as usize];
        let copy_len = output.len().min(words.len());
        output[..copy_len].copy_from_slice(&words[..copy_len]);
        output[copy_len..].fill(0);
        if whitespace {
            for (target, &word) in output.iter_mut().zip(admission.whitespace.iter()) {
                *target |= word;
            }
        }
        true
    }

    #[inline(always)]
    pub(crate) fn llg_master_residual_ops(
        &self,
        safe_radius: u16,
        whitespace: bool,
    ) -> Option<usize> {
        let radius = usize::from(safe_radius.min(self.llg_master_max_safe_chars()));
        self.llg_master_residual_ops.get(radius)
            .map(|row| row[usize::from(whitespace)] as usize)
    }

    #[inline(always)]
    pub(crate) fn llg_master_max_safe_chars(&self) -> u16 {
        self.llg_master_admission.as_ref().map_or(0, |words| words.max_radius())
    }

    fn rebuild_llg_master_residual_ops_from_master_trie(&mut self) {
        let Some(master) = self.llg_master_trie() else {
            self.llg_master_residual_ops = Arc::new(Vec::new());
            return;
        };
        let trie = master.trie();
        let max_safe_chars = trie.children(0).iter().enumerate()
            .filter_map(|(slot, _)| trie.root_layout_class(slot))
            .map(dynamic_mask_llg_master_safe_chars)
            .max().unwrap_or(0);
        let mut saved_by_radius = vec![[0usize; 2]; usize::from(max_safe_chars) + 1];
        let mut root_slot = 0;
        let mut total = 0usize;
        let mut whitespace_saved = 0usize;
        let mut complete = true;
        // Visit the root edges directly: their subtree boundaries jump over
        // every interior edge. A skipped class still costs its root op.
        let mut edge_index = 0;
        while edge_index < trie.walk_edges.len() {
            let edge = trie.walk_edges[edge_index];
            let Some(class) = trie.root_layout_class(root_slot) else {
                complete = false;
                break;
            };
            root_slot += 1;
            let Some((&start, &end)) = trie.full_walk_edge_op_starts.get(edge_index)
                .zip(trie.full_walk_edge_op_starts.get(edge.subtree_end as usize))
                else { complete = false; break; };
            if edge.parent_depth != 0 || edge.subtree_end as usize <= edge_index || end <= start {
                complete = false;
                break;
            }
            let work = (end - start) as usize;
            total = total.saturating_add(work);
            let saved = work - 1;
            let safe_chars = dynamic_mask_llg_master_safe_chars(class);
            let whitespace = dynamic_mask_llg_master_is_whitespace(class);
            if whitespace {
                whitespace_saved = whitespace_saved.saturating_add(saved);
            }
            if safe_chars != 0 {
                let entry = &mut saved_by_radius[usize::from(safe_chars)];
                entry[0] = entry[0].saturating_add(saved);
                if !whitespace {
                    entry[1] = entry[1].saturating_add(saved);
                }
            }
            edge_index = edge.subtree_end as usize;
        }
        complete &= !trie.root_layout_classes.is_empty()
            && root_slot == trie.root_layout_classes.len();
        let rows = if complete {
            let mut saved = [0usize; 2];
            saved_by_radius.into_iter().map(|delta| {
                saved[0] = saved[0].saturating_add(delta[0]);
                saved[1] = saved[1].saturating_add(delta[1]);
                [
                    u32::try_from(total.saturating_sub(saved[0])).unwrap_or(u32::MAX),
                    u32::try_from(total.saturating_sub(whitespace_saved).saturating_sub(saved[1]))
                        .unwrap_or(u32::MAX),
                ]
            }).collect()
        } else {
            vec![[u32::try_from(trie.full_walk_ops().len()).unwrap_or(u32::MAX); 2];
                usize::from(max_safe_chars) + 1]
        };
        self.llg_master_residual_ops = Arc::new(rows);
    }

    /// Reconstruct cumulative admitted-token bitsets from the already-transferred
    /// master slice trie. Each root subtree has one exact `(safe_chars, whitespace)`
    /// language class, so this is linear in original-token membership plus the
    /// small number of cumulative rows and avoids serializing those dense rows.
    fn rebuild_llg_master_admitted_words_from_master_trie(&mut self) {
        let Some(master) = self.llg_master_trie() else {
            self.llg_master_admission = None;
            self.llg_master_residual_ops = Arc::new(Vec::new());
            return;
        };
        let root_children = master.trie().children(0);
        let max_safe_chars = root_children
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| master.trie().root_layout_class(slot))
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let word_len = self.indexes().all_original_token_words.len();
        let mut exact_safe_words =
            vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        let mut whitespace_words = vec![0u32; word_len];
        for (slot, edge) in root_children.iter().enumerate() {
            let Some(class) = master.trie().root_layout_class(slot) else {
                continue;
            };
            let safe_chars = dynamic_mask_llg_master_safe_chars(class);
            let whitespace = dynamic_mask_llg_master_is_whitespace(class);
            if safe_chars == 0 && !whitespace {
                continue;
            }
            for &token in master.subtree_original_tokens(edge.child) {
                let word = token as usize / 32;
                if word >= word_len {
                    continue;
                }
                let bit = 1u32 << (token % 32);
                if safe_chars != 0 {
                    exact_safe_words[usize::from(safe_chars)][word] |= bit;
                }
                if whitespace {
                    whitespace_words[word] |= bit;
                }
            }
        }
        self.set_llg_master_admitted_words(exact_safe_words, Arc::new(whitespace_words));
        self.rebuild_llg_master_residual_ops_from_master_trie();
    }

    #[inline(always)]
    pub(crate) fn prepared_master_provers(
        &self,
        source: u32,
        slice_slot: usize,
    ) -> &[TerminalID] {
        if slice_slot >= Self::PREPARED_PROOF_SLOT_COUNT {
            return &[];
        }
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT + slice_slot;
        self.prepared_master_proofs.as_ref()
            .and_then(|prepared| prepared.proofs.row(row))
            .map_or(&[], |(positive, _)| positive)
    }

    pub(crate) fn prepared_master_proof_artifact(&self) -> PreparedMasterProofArtifact {
        self.prepared_master_proofs.as_ref().map_or_else(
            PreparedMasterProofArtifact::default, |prepared| prepared.to_wire())
    }

    pub(crate) fn restore_prepared_master_proof_artifact(
        &mut self,
        artifact: PreparedMasterProofArtifact,
        source_state_count: usize,
    ) -> Result<(), String> {
        let expected_rows = source_state_count
            .checked_mul(Self::PREPARED_PROOF_SLOT_COUNT)
            .ok_or_else(|| "prepared master proof row count overflow".to_owned())?;
        let validate_interned =
            |label: &str, row_ids: &[u32], offsets: &[u32], terminals: &[TerminalID], rows: usize| {
                if row_ids.is_empty() {
                    return if offsets.is_empty() && terminals.is_empty() {
                        Ok(())
                    } else {
                        Err(format!("{label} has table data without row ids"))
                    };
                }
                if row_ids.len() != rows
                    || offsets.is_empty()
                    || offsets.first().copied() != Some(0)
                    || offsets.windows(2).any(|pair| pair[0] > pair[1])
                    || offsets.last().copied().map(|value| value as usize) != Some(terminals.len())
                {
                    return Err(format!("{label} has invalid interned row framing"));
                }
                let unique_rows = offsets.len() - 1;
                if row_ids.iter().any(|&row| row as usize >= unique_rows) {
                    return Err(format!("{label} has an out-of-range row id"));
                }
                for row in 0..unique_rows {
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    if terminals[start..end].windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(format!("{label} unique row {row} is not sorted and unique"));
                    }
                }
                Ok(())
            };
        validate_interned(
            "prepared master positive rows",
            &artifact.positive_row_ids,
            &artifact.positive_offsets,
            &artifact.positive_terminals,
            expected_rows,
        )?;
        validate_interned(
            "prepared master coverage rows",
            &artifact.coverage_row_ids,
            &artifact.coverage_offsets,
            &artifact.coverage_terminals,
            expected_rows,
        )?;
        if !artifact.coverage_row_ids.is_empty() {
            if artifact.positive_row_ids.is_empty() {
                return Err("prepared master coverage exists without positive row framing".to_owned());
            }
            for row in 0..expected_rows {
                let coverage = interned_terminal_row(
                    &artifact.coverage_row_ids,
                    &artifact.coverage_offsets,
                    &artifact.coverage_terminals,
                    row,
                );
                if interned_terminal_row(
                    &artifact.positive_row_ids,
                    &artifact.positive_offsets,
                    &artifact.positive_terminals,
                    row,
                )
                .iter()
                .any(|terminal| coverage.binary_search(terminal).is_err())
                {
                    return Err(format!(
                        "prepared master positive row {row} is not a subset of coverage"
                    ));
                }
            }
        }
        if artifact
            .safe_plus_complete_terminals
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "prepared safe+ complete terminal list is not sorted and unique".to_owned(),
            );
        }
        if !artifact.safe_radius_row_ids.is_empty() {
            if artifact.safe_radius_row_ids.len() != source_state_count
                || artifact.safe_radius_offsets.is_empty()
                || artifact.safe_radius_offsets.first().copied() != Some(0)
                || artifact.safe_radius_offsets.windows(2).any(|pair| pair[0] > pair[1])
                || artifact.safe_radius_offsets.last().copied().map(|value| value as usize)
                    != Some(artifact.safe_radius_entries.len())
            {
                return Err("prepared safe-radius rows have invalid interned framing".to_owned());
            }
            let unique_rows = artifact.safe_radius_offsets.len() - 1;
            if artifact
                .safe_radius_row_ids
                .iter()
                .any(|&row| row as usize >= unique_rows)
            {
                return Err("prepared safe-radius rows have an out-of-range row id".to_owned());
            }
            for row in 0..unique_rows {
                let start = artifact.safe_radius_offsets[row] as usize;
                let end = artifact.safe_radius_offsets[row + 1] as usize;
                if artifact.safe_radius_entries[start..end]
                    .windows(2)
                    .any(|pair| pair[0].0 >= pair[1].0)
                {
                    return Err(format!(
                        "prepared safe-radius unique row {row} is not sorted and unique"
                    ));
                }
            }
        } else if !artifact.safe_radius_offsets.is_empty() || !artifact.safe_radius_entries.is_empty()
        {
            return Err("prepared safe-radius table exists without row ids".to_owned());
        }
        self.prepared_master_proofs = if artifact.is_empty() {
            None
        } else {
            Some(Arc::new(PreparedMasterProofs::from_wire(artifact)))
        };
        Ok(())
    }

    /// Whether build/runtime preparation installed exact master-prover rows for
    /// this tokenizer source. An empty positive-terminal row is still a valid
    /// prepared row, so checking `prepared_master_provers()` itself is not
    /// sufficient.
    #[inline(always)]
    pub(crate) fn has_prepared_master_prover_row(&self, source: u32) -> bool {
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT;
        self.prepared_master_proofs.as_ref().is_some_and(|prepared|
            prepared.proofs.row_ids.get(row + Self::PREPARED_PROOF_SLOT_COUNT - 1).is_some())
    }

    /// Only explicit coverage is self-contained after transfer. A legacy
    /// projected table still needs its associated exact quotient inventory.
    #[inline]
    pub(crate) fn has_complete_prepared_master_prover_rows(
        &self,
        source_state_count: usize,
    ) -> bool {
        source_state_count.checked_mul(Self::PREPARED_PROOF_SLOT_COUNT)
            .is_some_and(|rows| rows != 0 && self.prepared_master_proofs.as_ref().is_some_and(|prepared|
                matches!(&prepared.proofs.coverage, MasterProofCoverage::Explicit(_))
                    && prepared.proofs.row_ids.len() == rows))
    }

    /// A completed proof is positive or negative; missing coverage remains
    /// unknown and must fall back to exact runtime proof.
    #[inline]
    pub(crate) fn prepared_master_proof_result(
        &self,
        source: u32,
        slice_slot: usize,
        terminal: TerminalID,
    ) -> Option<bool> {
        if slice_slot >= Self::PREPARED_PROOF_SLOT_COUNT {
            return None;
        }
        let prepared = self.prepared_master_proofs.as_ref()?;
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT + slice_slot;
        let (positive, negative) = prepared.proofs.row(row)?;
        let proved = positive.binary_search(&terminal).is_ok();
        match &prepared.proofs.coverage {
            MasterProofCoverage::Explicit(complete) => {
                if proved {
                    Some(true)
                } else if negative.binary_search(&terminal).is_ok()
                    || (slice_slot == Self::PREPARED_SAFE_PLUS_SLOT
                        && complete.binary_search(&terminal).is_ok()) {
                    Some(false)
                } else {
                    None
                }
            }
            MasterProofCoverage::Projected => self.projected_terminal_quotient(terminal, source)
                .map(|_| proved),
        }
    }

    #[inline]
    pub(crate) fn prepared_safe_radius(
        &self,
        source: u32,
        terminal: TerminalID,
    ) -> Option<u16> {
        // An empty row can be a solved zero; a missing row cannot.
        let prepared = self.prepared_master_proofs.as_ref()?;
        let row = prepared.radii.row(source as usize)?;
        if let Ok(index) = row.binary_search_by_key(&terminal, |&(candidate, _)| candidate) {
            return Some(row[index].1);
        }
        self.prepared_master_proof_result(source, Self::PREPARED_SAFE_PLUS_SLOT, terminal)
            .map(|_| 0)
    }

    /// Exact all-start-state master-slice proof directly over an independently
    /// compiled terminal DFA retained by the partitioned lexer builder.
    ///
    /// The slice DFA already supplies an exact byte partition. For each
    /// terminal state we group live outgoing targets by slice class and require
    /// complete byte coverage for every class that can remain on an accepting
    /// slice prefix. This avoids materializing a dense global tokenizer-state
    /// quotient while preserving exact containment semantics.
    fn terminal_dfa_partition_all_transparent_states(
        dfa: &LexerDfa,
        group: u32,
        partition: &VocabPartitionDfa,
    ) -> Option<(Vec<bool>, usize, usize)> {
        if dfa.has_epsilon_transitions() {
            return None;
        }
        let q_count = dfa.num_states();
        let p_count = partition.state_count();
        let class_count = partition.class_count();
        if q_count == 0 || p_count == 0 || class_count == 0 {
            return Some((vec![false; q_count], 0, 0));
        }

        let mut class_sizes = vec![0usize; class_count];
        let mut class_representatives = vec![u8::MAX; class_count];
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let class = partition.byte_class(byte) as usize;
            class_sizes[class] += 1;
            if class_representatives[class] == u8::MAX {
                class_representatives[class] = byte;
            }
        }

        let relevant_classes = (0..p_count as u32)
            .map(|p| {
                if !partition.can_reach_accepting(p) {
                    return SmallVec::<[(usize, u32); 8]>::new();
                }
                class_representatives
                    .iter()
                    .enumerate()
                    .filter_map(|(class, &byte)| {
                        let target = partition.step(p, byte);
                        partition
                            .can_reach_accepting(target)
                            .then_some((class, target))
                    })
                    .collect::<SmallVec<[(usize, u32); 8]>>()
            })
            .collect::<Vec<_>>();

        let pair_count = p_count.saturating_mul(q_count);
        let pair_index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut predecessors = vec![SmallVec::<[u32; 8]>::new(); pair_count];
        let mut bad = vec![false; pair_count];
        let mut queue = VecDeque::<u32>::new();
        let mut edge_count = 0usize;

        let state_live = |state: u32| {
            dfa.finalizers(state).contains(group as usize)
                || dfa
                    .possible_future_group_ids(state)
                    .contains(group as usize)
        };

        for q in 0..q_count as u32 {
            if !state_live(q) {
                for p in 0..p_count as u32 {
                    if !partition.can_reach_accepting(p) {
                        continue;
                    }
                    let current = pair_index(p, q);
                    bad[current] = true;
                    queue.push_back(current as u32);
                }
                continue;
            }

            let mut covered = vec![0usize; class_count];
            let mut targets = (0..class_count)
                .map(|_| SmallVec::<[u32; 4]>::new())
                .collect::<Vec<_>>();
            for (byte, target) in dfa.transitions(q) {
                if !state_live(target) {
                    continue;
                }
                let class = partition.byte_class(byte) as usize;
                covered[class] += 1;
                if !targets[class].contains(&target) {
                    targets[class].push(target);
                }
            }

            for p in 0..p_count as u32 {
                if !partition.can_reach_accepting(p) {
                    continue;
                }
                let current = pair_index(p, q);
                let relevant = &relevant_classes[p as usize];
                if relevant
                    .iter()
                    .any(|&(class, _)| covered[class] != class_sizes[class])
                {
                    bad[current] = true;
                    queue.push_back(current as u32);
                    continue;
                }
                for &(class, p_target) in relevant {
                    for &q_target in &targets[class] {
                        let target = pair_index(p_target, q_target);
                        predecessors[target].push(current as u32);
                        edge_count = edge_count.saturating_add(1);
                    }
                }
            }
        }

        while let Some(target) = queue.pop_front() {
            for &pred in &predecessors[target as usize] {
                if !bad[pred as usize] {
                    bad[pred as usize] = true;
                    queue.push_back(pred);
                }
            }
        }

        let start = partition.start_state();
        let mut transparent = vec![false; q_count];
        if partition.can_reach_accepting(start) {
            for q in 0..q_count as u32 {
                transparent[q as usize] = !bad[pair_index(start, q)];
            }
        } else {
            for q in 0..q_count as u32 {
                transparent[q as usize] = state_live(q);
            }
        }
        Some((transparent, pair_count, edge_count))
    }

    /// Exact safe-slice transparency and bounded radius for every residual state
    /// of one retained terminal DFA. Both answers are determined by the same
    /// shortest-counterexample product graph, so compute that graph once instead
    /// of separately solving containment and radius.
    fn terminal_dfa_partition_all_transparency_and_repeat_radii(
        dfa: &LexerDfa,
        group: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
    ) -> Option<(Vec<bool>, Vec<u32>, usize, usize)> {
        if dfa.has_epsilon_transitions() {
            return None;
        }
        let q_count = dfa.num_states();
        let p_count = slice.state_count();
        let class_count = slice.class_count();
        if q_count == 0 || p_count == 0 || class_count == 0 {
            return Some((vec![false; q_count], vec![0; q_count], 0, 0));
        }

        let live_states = (0..q_count as u32)
            .map(|state| {
                dfa.finalizers(state).contains(group as usize)
                    || dfa
                        .possible_future_group_ids(state)
                        .contains(group as usize)
            })
            .collect::<Vec<_>>();
        let state_live = |state: u32| live_states[state as usize];

        let mut class_sizes = vec![0usize; class_count];
        let mut class_representatives = vec![u8::MAX; class_count];
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let class = slice.byte_class(byte) as usize;
            class_sizes[class] += 1;
            if class_representatives[class] == u8::MAX {
                class_representatives[class] = byte;
            }
        }

        let relevant_classes = (0..p_count as u32)
            .map(|p| {
                if !slice.can_reach_accepting(p) {
                    return SmallVec::<[(usize, u32, u8); 8]>::new();
                }
                class_representatives
                    .iter()
                    .enumerate()
                    .filter_map(|(class, &byte)| {
                        let target = slice.step(p, byte);
                        slice.can_reach_accepting(target).then_some((
                            class,
                            target,
                            u8::from(slice.is_accepting(target)),
                        ))
                    })
                    .collect::<SmallVec<[(usize, u32, u8); 8]>>()
            })
            .collect::<Vec<_>>();

        // Minimum completed-atom cost from each slice state to an accepting
        // state. Every byte in one slice class has the same target, so one
        // representative per class is exact here.
        let mut slice_reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); p_count];
        for p in 0..p_count as u32 {
            let mut seen = SmallVec::<[u32; 16]>::new();
            for &byte in &class_representatives {
                let target = slice.step(p, byte);
                if target as usize >= p_count || seen.contains(&target) {
                    continue;
                }
                seen.push(target);
                slice_reverse[target as usize]
                    .push((p, u8::from(slice.is_accepting(target))));
            }
        }
        let mut min_to_accept = vec![u32::MAX; p_count];
        let mut zero_one = VecDeque::<u32>::new();
        for p in 0..p_count as u32 {
            if slice.is_accepting(p) {
                min_to_accept[p as usize] = 0;
                zero_one.push_back(p);
            }
        }
        while let Some(target) = zero_one.pop_front() {
            let base = min_to_accept[target as usize];
            for &(source, cost) in &slice_reverse[target as usize] {
                let candidate = base.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        zero_one.push_front(source);
                    } else {
                        zero_one.push_back(source);
                    }
                }
            }
        }

        let mut covered_by_q = Vec::<Vec<usize>>::with_capacity(q_count);
        let mut targets_by_q = Vec::<Vec<SmallVec<[u32; 4]>>>::with_capacity(q_count);
        for q in 0..q_count as u32 {
            let mut covered = vec![0usize; class_count];
            let mut targets = (0..class_count)
                .map(|_| SmallVec::<[u32; 4]>::new())
                .collect::<Vec<_>>();
            if state_live(q) {
                for (byte, target) in dfa.transitions(q) {
                    if !state_live(target) {
                        continue;
                    }
                    let class = slice.byte_class(byte) as usize;
                    covered[class] += 1;
                    if !targets[class].contains(&target) {
                        targets[class].push(target);
                    }
                }
            }
            covered_by_q.push(covered);
            targets_by_q.push(targets);
        }

        let pair_count = p_count.saturating_mul(q_count);
        let index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); pair_count];
        let mut distance = vec![u32::MAX; pair_count];
        let mut max_seed_distance = 0u32;
        let mut edge_count = 0usize;

        for p in 0..p_count as u32 {
            if !slice.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = index(p, q);
                if !state_live(q) {
                    distance[current] = 0;
                    continue;
                }
                for &(class, p_target, enter_cost) in &relevant_classes[p as usize] {
                    if covered_by_q[q as usize][class] != class_sizes[class] {
                        let completion = min_to_accept[p_target as usize];
                        if completion != u32::MAX {
                            let candidate = u32::from(enter_cost).saturating_add(completion);
                            if candidate < distance[current] {
                                distance[current] = candidate;
                                max_seed_distance = max_seed_distance.max(candidate);
                            }
                        }
                    }
                    for &q_target in &targets_by_q[q as usize][class] {
                        let target = index(p_target, q_target);
                        reverse[target].push((current as u32, enter_cost));
                        edge_count = edge_count.saturating_add(1);
                    }
                }
            }
        }

        // Product edges are 0/1-weighted. When the counterexample seeds are too,
        // 0-1 BFS avoids heap traffic; larger seed distances still use Dijkstra.
        if max_seed_distance <= 1 {
            let mut queue = VecDeque::<(u32, u32)>::new();
            for (state, &dist) in distance.iter().enumerate() {
                match dist {
                    0 => queue.push_front((state as u32, 0)),
                    1 => queue.push_back((state as u32, 1)),
                    _ => {}
                }
            }
            while let Some((target, dist)) = queue.pop_front() {
                if distance[target as usize] != dist {
                    continue;
                }
                for &(pred, cost) in &reverse[target as usize] {
                    let candidate = dist.saturating_add(u32::from(cost));
                    if candidate < distance[pred as usize] {
                        distance[pred as usize] = candidate;
                        if cost == 0 {
                            queue.push_front((pred, candidate));
                        } else {
                            queue.push_back((pred, candidate));
                        }
                    }
                }
            }
        } else {
            let mut heap = BinaryHeap::<(Reverse<u32>, u32)>::new();
            for (state, &dist) in distance.iter().enumerate() {
                if dist != u32::MAX {
                    heap.push((Reverse(dist), state as u32));
                }
            }
            while let Some((Reverse(dist), target)) = heap.pop() {
                if distance[target as usize] != dist {
                    continue;
                }
                for &(pred, cost) in &reverse[target as usize] {
                    let candidate = dist.saturating_add(u32::from(cost));
                    if candidate < distance[pred as usize] {
                        distance[pred as usize] = candidate;
                        heap.push((Reverse(candidate), pred));
                    }
                }
            }
        }

        let start = slice.start_state();
        let start_can_accept = slice.can_reach_accepting(start);
        let transparent = (0..q_count as u32)
            .map(|q| {
                if start_can_accept {
                    state_live(q) && distance[index(start, q)] == u32::MAX
                } else {
                    state_live(q)
                }
            })
            .collect::<Vec<_>>();
        let radii = (0..q_count as u32)
            .map(|q| {
                if !state_live(q) {
                    0
                } else {
                    match distance[index(start, q)] {
                        u32::MAX => max_repetitions,
                        first_counterexample => first_counterexample
                            .saturating_sub(1)
                            .min(max_repetitions),
                    }
                }
            })
            .collect::<Vec<_>>();
        Some((transparent, radii, pair_count, edge_count))
    }

    /// Build exact compact master-slice and bounded-radius certificates from
    /// the partitioned lexer's retained terminal-residual coordinates. The
    /// result contains no terminal quotient transition matrix and is therefore
    /// cheap to transfer and restore.
    pub(crate) fn prepare_master_provers_from_residual_coordinates(
        &mut self,
        tokenizer: &Tokenizer,
        source_state_count: usize,
        safe_plus: &VocabPartitionDfa,
        whitespace: &VocabPartitionDfa,
        safe_slice_token_bytes: U8Set,
        max_safe_chars: u16,
    ) -> Option<(usize, usize, usize)> {
        let coordinates = tokenizer.terminal_residual_coordinates()?;
        if source_state_count == 0 || coordinates.len() == 0 {
            return Some((0, 0, 0));
        }
        let required_bytes = |dfa: &VocabPartitionDfa| {
            let mut bytes = U8Set::empty();
            for raw in 0u16..=255 {
                let byte = raw as u8;
                let used = (0..dfa.state_count() as u32).any(|state| {
                    dfa.can_reach_accepting(state)
                        && dfa.can_reach_accepting(dfa.step(state, byte))
                });
                if used {
                    bytes.insert(byte);
                }
            }
            bytes
        };
        let safe_plus_required = required_bytes(safe_plus);
        let whitespace_required = required_bytes(whitespace);

        struct DirectPreparedPair {
            terminal: TerminalID,
            slice_slot: usize,
            transparent: Vec<bool>,
            product_pairs: usize,
            edges: usize,
        }

        struct DirectSafePreparedPair {
            terminal: TerminalID,
            transparent: Option<Vec<bool>>,
            radii: Option<Vec<u32>>,
            product_pairs: usize,
            edges: usize,
        }

        let candidates = (0..tokenizer.num_terminals())
            .filter(|&terminal| {
                tokenizer
                    .terminal_byte_support(terminal)
                    .is_some_and(|support| safe_slice_token_bytes.is_subset(&support))
                    && coordinates.terminal_dfa_and_group(terminal).is_some()
            })
            .collect::<Vec<_>>();
        let verify_combined =
            crate::env::var_os("GLRMASK_VERIFY_DIRECT_RESIDUAL_MASTER_PROVERS").is_some();
        let build_whitespace = || {
            candidates
                .par_iter()
                .copied()
                .filter(|&terminal| {
                    tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| whitespace_required.is_subset(&support))
                })
                .filter_map(|terminal| {
                    let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
                    let (transparent, product_pairs, edges) =
                        Self::terminal_dfa_partition_all_transparent_states(dfa, group, whitespace)?;
                    Some(DirectPreparedPair {
                        terminal,
                        slice_slot: Self::PREPARED_WHITESPACE_SLOT,
                        transparent,
                        product_pairs,
                        edges,
                    })
                })
                .collect::<Vec<_>>()
        };
        let build_safe = || {
            candidates
                .par_iter()
                .copied()
                .filter_map(|terminal| {
                    let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
                    let safe_supported = tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| safe_plus_required.is_subset(&support));
                    if max_safe_chars == 0 {
                        let (transparent, product_pairs, edges) =
                            Self::terminal_dfa_partition_all_transparent_states(
                                dfa, group, safe_plus,
                            )?;
                        return Some(DirectSafePreparedPair {
                            terminal,
                            transparent: safe_supported.then_some(transparent),
                            radii: None,
                            product_pairs,
                            edges,
                        });
                    }
                    let (transparent, radii, product_pairs, edges) =
                        Self::terminal_dfa_partition_all_transparency_and_repeat_radii(
                            dfa,
                            group,
                            safe_plus,
                            u32::from(max_safe_chars),
                        )?;
                    if verify_combined && safe_supported {
                        let (reference, _, _) = Self::terminal_dfa_partition_all_transparent_states(
                            dfa, group, safe_plus,
                        )?;
                        assert_eq!(
                            transparent, reference,
                            "combined direct residual safe+ solver disagrees for terminal {terminal}"
                        );
                    }
                    Some(DirectSafePreparedPair {
                        terminal,
                        transparent: safe_supported.then_some(transparent),
                        radii: Some(radii),
                        product_pairs,
                        edges,
                    })
                })
                .collect::<Vec<_>>()
        };
        let (proofs, safe_jobs) = rayon::join(build_whitespace, build_safe);

        let terminal_count = tokenizer.num_terminals();
        let mut by_terminal_slot = (0..terminal_count)
            .map(|_| [None::<Vec<bool>>, None::<Vec<bool>>])
            .collect::<Vec<_>>();
        let mut product_pairs = 0usize;
        let mut edges = 0usize;
        for proof in proofs {
            product_pairs = product_pairs.saturating_add(proof.product_pairs);
            edges = edges.saturating_add(proof.edges);
            by_terminal_slot[proof.terminal as usize][proof.slice_slot] = Some(proof.transparent);
        }
        let mut radius_by_terminal =
            (0..terminal_count).map(|_| None::<Vec<u32>>).collect::<Vec<_>>();
        for proof in safe_jobs {
            product_pairs = product_pairs.saturating_add(proof.product_pairs);
            edges = edges.saturating_add(proof.edges);
            if let Some(transparent) = proof.transparent {
                by_terminal_slot[proof.terminal as usize][Self::PREPARED_SAFE_PLUS_SLOT] =
                    Some(transparent);
            }
            if let Some(radii) = proof.radii {
                radius_by_terminal[proof.terminal as usize] = Some(radii);
            }
        }
        let mut safe_plus_complete_terminals = candidates
            .iter()
            .copied()
            .filter(|terminal| {
                by_terminal_slot[*terminal as usize][Self::PREPARED_SAFE_PLUS_SLOT].is_some()
                    && radius_by_terminal[*terminal as usize].is_some()
            })
            .collect::<Vec<_>>();
        safe_plus_complete_terminals.sort_unstable();
        safe_plus_complete_terminals.dedup();

        let row_count = source_state_count * Self::PREPARED_PROOF_SLOT_COUNT;
        let mut proof_rows = vec![MasterProofRow::default(); row_count];
        let mut radius_rows = vec![SmallVec::<[(TerminalID, u16); 4]>::new(); source_state_count];
        let source_limit = source_state_count.min(coordinates.len());
        proof_rows[..source_limit * Self::PREPARED_PROOF_SLOT_COUNT]
            .par_chunks_mut(Self::PREPARED_PROOF_SLOT_COUNT)
            .zip(radius_rows[..source_limit].par_iter_mut())
            .enumerate()
            .for_each(|(source, (proof_rows, radius_row))| {
                let Some(entries) = coordinates.row(source as u32) else {
                    return;
                };
                for &(terminal, residual) in entries {
                    let proof_slots = &by_terminal_slot[terminal as usize];
                    for slice_slot in 0..Self::PREPARED_PROOF_SLOT_COUNT {
                        let Some(transparent) = proof_slots[slice_slot].as_ref() else {
                            continue;
                        };
                        proof_rows[slice_slot].push(terminal,
                            transparent.get(residual as usize).copied().unwrap_or(false));
                    }
                    if let Some(radii) = radius_by_terminal[terminal as usize].as_ref() {
                        let radius = radii
                            .get(residual as usize)
                            .copied()
                            .unwrap_or(0)
                            .min(u32::from(max_safe_chars)) as u16;
                        if radius != 0 {
                            radius_row.push((terminal, radius));
                        }
                    }
                }
            });

        let positive_entry_count = proof_rows.iter().map(|row| row.positive.len()).sum::<usize>();
        let coverage = MasterProofCoverage::Explicit(safe_plus_complete_terminals.into_boxed_slice());
        let (proofs, radii) = if candidates.len() >= 64 && rayon::current_num_threads() > 1 {
            rayon::join(
                || MasterProofRows::intern(proof_rows, coverage),
                || intern_radius_rows(radius_rows),
            )
        } else {
            (MasterProofRows::intern(proof_rows, coverage), intern_radius_rows(radius_rows))
        };
        self.prepared_master_proofs = Some(Arc::new(PreparedMasterProofs { proofs, radii }));
        Some((positive_entry_count, product_pairs, edges))
    }

    /// Build explicit exact parser-independent master-slice certificates for
    /// all represented sources. One product is solved per terminal/slice,
    /// then fanned out through the quotient's source-to-residual mapping.
    /// A skipped candidate remains unknown rather than implicitly negative.
    pub(crate) fn prepare_master_provers_all_sources(
        &mut self,
        tokenizer: &Tokenizer,
        source_state_count: usize,
        include_safe_radii: bool,
    ) -> (usize, usize) {
        let Some(safe_plus) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == 0)
            .cloned()
        else {
            return (0, 0);
        };
        let Some(whitespace) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == 3)
            .cloned()
        else {
            return (0, 0);
        };
        let mut proof_languages = Vec::<(usize, Arc<VocabPartitionDfa>, U8Set)>::new();
        let required_bytes = |dfa: &VocabPartitionDfa| {
            let mut bytes = U8Set::empty();
            for raw in 0u16..=255 {
                let byte = raw as u8;
                let used = (0..dfa.state_count() as u32).any(|state| {
                    dfa.can_reach_accepting(state)
                        && dfa.can_reach_accepting(dfa.step(state, byte))
                });
                if used {
                    bytes.insert(byte);
                }
            }
            bytes
        };
        proof_languages.push((
            Self::PREPARED_SAFE_PLUS_SLOT,
            Arc::clone(&safe_plus.dfa),
            required_bytes(safe_plus.dfa()),
        ));
        proof_languages.push((
            Self::PREPARED_WHITESPACE_SLOT,
            Arc::clone(&whitespace.dfa),
            required_bytes(whitespace.dfa()),
        ));
        let quotients = self.active_projected_terminal_quotients();
        if quotients.is_empty() || source_state_count == 0 {
            self.prepared_master_proofs = None;
            return (0, 0);
        }

        struct PreparedPair {
            terminal: TerminalID,
            slice_slot: usize,
            quotient: Arc<TerminalProjectedQuotient>,
            transparent: Vec<bool>,
            product_pairs: usize,
            elapsed_ns: u64,
            verified: usize,
            mismatches: usize,
            unknown: usize,
        }
        let verify = crate::env::var_os("GLRMASK_VERIFY_PREPARED_MASTER_PROVERS").is_some();
        let jobs = quotients
            .par_iter()
            .flat_map_iter(|(terminal, quotient)| {
                proof_languages
                    .iter()
                    .filter_map(move |(slot, slice, required)| {
                        tokenizer
                            .terminal_byte_support(*terminal)
                            .is_some_and(|support| required.is_subset(&support))
                            .then_some((*terminal, Arc::clone(quotient), *slot, Arc::clone(slice)))
                    })
            })
            .map(|(terminal, quotient, slice_slot, slice)| {
                let started = std::time::Instant::now();
                let (transparent, product_pairs) =
                    Self::terminal_partition_all_transparent_states(quotient.as_ref(), slice.as_ref());
                let mut verified = 0usize;
                let mut mismatches = 0usize;
                let mut unknown = 0usize;
                if verify {
                    for (&source, &projected) in quotient
                        .projected_source_states()
                        .iter()
                        .zip(quotient.projected_states_for_sources())
                    {
                        let batch = transparent.get(projected as usize).copied().unwrap_or(false);
                        match Self::terminal_partition_product_is_transparent(
                            quotient.as_ref(),
                            slice.as_ref(),
                            source,
                            200_000,
                        ) {
                            Some(reference) => {
                                verified += 1;
                                mismatches += usize::from(reference != batch);
                            }
                            None => unknown += 1,
                        }
                    }
                }
                PreparedPair {
                    terminal,
                    slice_slot,
                    quotient,
                    transparent,
                    product_pairs,
                    elapsed_ns: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    verified,
                    mismatches,
                    unknown,
                }
            })
            .collect::<Vec<_>>();

        let mut rows = vec![MasterProofRow::default();
            source_state_count * Self::PREPARED_PROOF_SLOT_COUNT];
        let mut product_pairs = 0usize;
        let mut pair_work_ns = 0u64;
        let mut max_pair_ns = 0u64;
        let mut verified = 0usize;
        let mut mismatches = 0usize;
        let mut unknown = 0usize;
        for job in jobs {
            product_pairs = product_pairs.saturating_add(job.product_pairs);
            pair_work_ns = pair_work_ns.saturating_add(job.elapsed_ns);
            max_pair_ns = max_pair_ns.max(job.elapsed_ns);
            verified += job.verified;
            mismatches += job.mismatches;
            unknown += job.unknown;
            for (&source, &projected) in job.quotient.projected_source_states().iter()
                .zip(job.quotient.projected_states_for_sources()) {
                if source as usize >= source_state_count {
                    continue;
                }
                if let Some(&proved) = job.transparent.get(projected as usize) {
                    rows[source as usize * Self::PREPARED_PROOF_SLOT_COUNT + job.slice_slot]
                        .push(job.terminal, proved);
                }
            }
        }
        if crate::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some() {
            eprintln!(
                "[glrmask/profile][prepared_master_pair_work] jobs={} work_ms={:.3} max_pair_ms={:.3}",
                quotients.len() * 2,
                pair_work_ns as f64 / 1e6,
                max_pair_ns as f64 / 1e6,
            );
        }
        if verify {
            eprintln!(
                "[glrmask/profile][prepared_master_verify] verified={} mismatches={} unknown={}",
                verified, mismatches, unknown,
            );
            assert_eq!(mismatches, 0, "batched master proof disagrees with per-source exact proof");
        }

        let entry_count = rows.iter().map(|row| row.positive.len()).sum::<usize>();
        let proofs = MasterProofRows::intern(rows, MasterProofCoverage::Explicit(Box::new([])));
        let radii = if include_safe_radii {
            let max_radius = self.llg_master_max_safe_chars();
            let radius_jobs = quotients
                .par_iter()
                .map(|(terminal, quotient)| {
                    let radii = Self::terminal_partition_all_repeat_radii(
                        quotient.as_ref(),
                        safe_plus.dfa(),
                        u32::from(max_radius),
                    );
                    (*terminal, Arc::clone(quotient), radii)
                })
                .collect::<Vec<_>>();
            let mut radius_rows =
                vec![SmallVec::<[(TerminalID, u16); 4]>::new(); source_state_count];
            for (terminal, quotient, radii) in radius_jobs {
                for (&source, &projected) in quotient
                    .projected_source_states()
                    .iter()
                    .zip(quotient.projected_states_for_sources())
                {
                    if source as usize >= source_state_count {
                        continue;
                    }
                    let radius = radii
                        .get(projected as usize)
                        .copied()
                        .unwrap_or(0)
                        .min(u32::from(max_radius)) as u16;
                    if radius != 0 {
                        radius_rows[source as usize].push((terminal, radius));
                    }
                }
            }
            intern_radius_rows(radius_rows)
        } else {
            InternedRows::default()
        };
        self.prepared_master_proofs = Some(Arc::new(PreparedMasterProofs { proofs, radii }));
        (entry_count, product_pairs)
    }

    /// Exact all-start-state analogue of `projected_terminal_slice_repeat_radius`.
    /// `distance[p,q]` is the minimum number of additional completed slice
    /// atoms in a completed slice word whose prefix first leaves the live
    /// terminal residual. Solving this reverse shortest-path problem once gives
    /// the bounded radius for every quotient residual state simultaneously.
    fn terminal_partition_all_repeat_radii(
        quotient: &TerminalProjectedQuotient,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
    ) -> Vec<u32> {
        let q_count = quotient.projected_state_count();
        let p_count = slice.state_count();
        if q_count == 0 || p_count == 0 || max_repetitions == 0 {
            return vec![0; q_count];
        }

        let mut representatives = SmallVec::<[(u8, u8, u8); 32]>::new();
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let p_class = slice.byte_class(byte);
            let q_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == p_class && q == q_class)
            {
                representatives.push((p_class, q_class, byte));
            }
        }

        // Minimum completed-atom cost from each slice state to some accepting
        // state. Edge cost is one exactly when the target state completes an
        // atom; UTF-8 continuation transitions therefore cost zero.
        let mut slice_reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); p_count];
        for p in 0..p_count as u32 {
            let mut seen = SmallVec::<[u32; 16]>::new();
            for &(_, _, byte) in &representatives {
                let target = slice.step(p, byte);
                if target as usize >= p_count || seen.contains(&target) {
                    continue;
                }
                seen.push(target);
                slice_reverse[target as usize]
                    .push((p, u8::from(slice.is_accepting(target))));
            }
        }
        let mut min_to_accept = vec![u32::MAX; p_count];
        let mut zero_one = VecDeque::<u32>::new();
        for p in 0..p_count as u32 {
            if slice.is_accepting(p) {
                min_to_accept[p as usize] = 0;
                zero_one.push_back(p);
            }
        }
        while let Some(target) = zero_one.pop_front() {
            let base = min_to_accept[target as usize];
            for &(source, cost) in &slice_reverse[target as usize] {
                let candidate = base.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        zero_one.push_front(source);
                    } else {
                        zero_one.push_back(source);
                    }
                }
            }
        }

        let pair_count = p_count * q_count;
        let index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); pair_count];
        let mut distance = vec![u32::MAX; pair_count];
        let mut heap = BinaryHeap::<(Reverse<u32>, u32)>::new();

        for p in 0..p_count as u32 {
            if !slice.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = index(p, q);
                let q_live = quotient.projected_state_is_accepting(q)
                    || quotient.projected_state_has_future(q);
                if !q_live {
                    distance[current] = 0;
                    heap.push((Reverse(0), current as u32));
                    continue;
                }
                for &(_, q_class, byte) in &representatives {
                    let p_target = slice.step(p, byte);
                    if !slice.can_reach_accepting(p_target) {
                        continue;
                    }
                    let enter_cost = u32::from(slice.is_accepting(p_target));
                    let q_target = quotient.projected_step_class(q, q_class);
                    let target_live = q_target.is_some_and(|target| {
                        quotient.projected_state_is_accepting(target)
                            || quotient.projected_state_has_future(target)
                    });
                    if !target_live {
                        let completion = min_to_accept[p_target as usize];
                        if completion != u32::MAX {
                            let candidate = enter_cost.saturating_add(completion);
                            if candidate < distance[current] {
                                distance[current] = candidate;
                                heap.push((Reverse(candidate), current as u32));
                            }
                        }
                        continue;
                    }
                    let target = index(p_target, q_target.expect("live target exists"));
                    reverse[target].push((current as u32, enter_cost as u8));
                }
            }
        }

        while let Some((Reverse(dist), target)) = heap.pop() {
            if distance[target as usize] != dist {
                continue;
            }
            for &(pred, cost) in &reverse[target as usize] {
                let candidate = dist.saturating_add(u32::from(cost));
                if candidate < distance[pred as usize] {
                    distance[pred as usize] = candidate;
                    heap.push((Reverse(candidate), pred));
                }
            }
        }

        let start = slice.start_state();
        (0..q_count as u32)
            .map(|q| {
                if !quotient.projected_state_is_accepting(q)
                    && !quotient.projected_state_has_future(q)
                {
                    0
                } else {
                    match distance[index(start, q)] {
                        u32::MAX => max_repetitions,
                        first_counterexample => first_counterexample
                            .saturating_sub(1)
                            .min(max_repetitions),
                    }
                }
            })
            .collect()
    }

    /// Exact all-start-state version of `terminal_partition_product_is_transparent`.
    /// A product pair `(p,q)` is bad when some byte that keeps `P` on an
    /// accepting-prefix path either has no live `Q` successor or reaches a bad
    /// product pair. This computes the least backwards closure of those bad
    /// pairs once, yielding the answer for every quotient state simultaneously.
    fn terminal_partition_all_transparent_states(
        quotient: &TerminalProjectedQuotient,
        partition: &VocabPartitionDfa,
    ) -> (Vec<bool>, usize) {
        let q_count = quotient.projected_state_count();
        let p_count = partition.state_count();
        if q_count == 0 || p_count == 0 {
            return (vec![false; q_count], 0);
        }

        // Refine the two exact byte partitions once. One representative is
        // sufficient for each `(P class, Q class)` pair because both automata
        // transition identically for every byte in that pair.
        let mut representatives = SmallVec::<[(u8, u8, u8); 32]>::new();
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let p_class = partition.byte_class(byte);
            let q_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == p_class && q == q_class)
            {
                representatives.push((p_class, q_class, byte));
            }
        }

        let pair_count = p_count.saturating_mul(q_count);
        let pair_index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut predecessors = vec![SmallVec::<[u32; 8]>::new(); pair_count];
        let mut bad = vec![false; pair_count];
        let mut queue = VecDeque::<u32>::new();

        for p in 0..p_count as u32 {
            if !partition.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = pair_index(p, q);
                if !quotient.projected_state_is_accepting(q)
                    && !quotient.projected_state_has_future(q)
                {
                    bad[current] = true;
                    queue.push_back(current as u32);
                    continue;
                }

                let mut immediate_bad = false;
                for &(_, q_class, byte) in &representatives {
                    let p_target = partition.step(p, byte);
                    if !partition.can_reach_accepting(p_target) {
                        continue;
                    }
                    let Some(q_target) = quotient.projected_step_class(q, q_class) else {
                        immediate_bad = true;
                        break;
                    };
                    if !quotient.projected_state_is_accepting(q_target)
                        && !quotient.projected_state_has_future(q_target)
                    {
                        immediate_bad = true;
                        break;
                    }
                    let target = pair_index(p_target, q_target);
                    predecessors[target].push(current as u32);
                }
                if immediate_bad {
                    bad[current] = true;
                    queue.push_back(current as u32);
                }
            }
        }

        while let Some(target) = queue.pop_front() {
            for &pred in &predecessors[target as usize] {
                if !bad[pred as usize] {
                    bad[pred as usize] = true;
                    queue.push_back(pred);
                }
            }
        }

        let start = partition.start_state();
        let mut transparent = vec![false; q_count];
        if partition.can_reach_accepting(start) {
            for q in 0..q_count as u32 {
                transparent[q as usize] = !bad[pair_index(start, q)];
            }
        } else {
            for q in 0..q_count as u32 {
                transparent[q as usize] = quotient.projected_state_is_accepting(q)
                    || quotient.projected_state_has_future(q);
            }
        }
        (transparent, pair_count)
    }

    fn active_projected_terminal_quotients(&self) -> Vec<(TerminalID, TerminalQuotient)> {
        match &self.terminal_quotients {
            ProjectedTerminalInventory::Prepared(quotients) => quotients.to_vec(),
            ProjectedTerminalInventory::Runtime(runtime) => {
                let inventory = runtime.by_terminal.lock().unwrap_or_else(|p| p.into_inner());
                let mut quotients: Vec<_> = inventory.iter().filter_map(|(&terminal, quotient)| {
                    Some((terminal, Arc::clone(quotient.as_ref()?)))
                }).collect();
                quotients.sort_unstable_by_key(|(terminal, _)| *terminal);
                quotients
            }
        }
    }

    #[inline(always)]
    pub(crate) fn llg_master_trie(&self) -> Option<&DynamicMaskSliceTrie> {
        // O2 already walks a grammar quotient that is orders of magnitude
        // smaller than the model vocabulary. Switching to a separately
        // structured master trie loses that quotient and is unnecessary; the
        // exact ordinary quotient walk is both the semantic authority and the
        // cheaper coordinate here.
        if self.grammar_quotiented {
            return None;
        }
        self.llg_slice_by_cache_id(DYNAMIC_MASK_LLG_MASTER_CACHE_ID)
    }

    #[inline(always)]
    pub(crate) fn llg_slice_leftovers(&self) -> &[Arc<DynamicMaskSliceTrie>] {
        self.llg_slice_leftovers.as_ref()
    }

    #[inline(always)]
    pub(crate) fn has_llg_slice_leftovers(&self) -> bool {
        !self.llg_slice_leftovers.is_empty()
    }

    #[inline(always)]
    pub(crate) fn llg_slice_by_cache_id(&self, cache_id: u32) -> Option<&DynamicMaskSliceTrie> {
        self.llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == cache_id)
            .map(Arc::as_ref)
    }

    #[inline(always)]
    pub(crate) fn residual_original_token_words_for(
        &self,
        trie: &DynamicMaskTrie,
    ) -> Option<&[u32]> {
        self.llg_slice_leftovers
            .iter()
            .find(|slice| {
                slice.cache_id() & 0x100 != 0 && std::ptr::eq(trie, slice.trie())
            })
            .map(|slice| slice.slice_original_token_words())
    }

    #[inline(always)]
    pub(crate) fn full_walk_token_markers_for(&self, trie: &DynamicMaskTrie) -> WalkTokenMarkers<'_> {
        if std::ptr::eq(trie, self.trie.as_ref()) {
            return WalkTokenMarkers { values: self.full_walk_token_markers(), owner: WalkMarkerOwner::Main(self) };
        }
        if let Some(slice) = self.llg_slice_leftovers.iter()
            .find(|slice| std::ptr::eq(trie, slice.trie()))
        {
            return WalkTokenMarkers { values: slice.full_walk_token_markers(), owner: WalkMarkerOwner::Slice(slice) };
        }
        debug_assert!(false, "unknown dynamic-mask walk trie");
        WalkTokenMarkers { values: &[], owner: WalkMarkerOwner::Main(self) }
    }

    #[inline(always)]
    pub(crate) fn subtree_original_tokens_for(
        &self,
        trie: &DynamicMaskTrie,
        node: u32,
    ) -> SubtreeOriginalTokens<'_> {
        if std::ptr::eq(trie, self.trie.as_ref()) {
            return self.subtree_original_tokens(node);
        }
        if let Some(slice) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| std::ptr::eq(trie, slice.trie()))
        {
            return slice.subtree_original_tokens(node);
        }
        debug_assert!(false, "unknown dynamic-mask walk trie");
        SubtreeOriginalTokens::Flat([].iter())
    }


    fn permuted_alias_offsets(trie: &DynamicMaskTrie, offsets: &[u32]) -> Option<Vec<u32>> {
        let tokens = trie.all_subtree_tokens();
        if tokens.len() == offsets.len().saturating_sub(1)
            && tokens.iter().copied().enumerate().all(|(index, canonical)| index == canonical as usize)
        {
            return None;
        }
        let mut counts = Vec::with_capacity(tokens.len() + 1);
        counts.push(0u32);
        for &canonical in tokens {
            let index = canonical as usize;
            let count = offsets[index + 1] - offsets[index];
            counts.push(counts.last().copied().unwrap() + count);
        }
        Some(counts)
    }

    fn encode_walk_marker(canonical_token: u32, aliases: &[u32]) -> u64 {
        const FALLBACK_TAG: u64 = 1u64 << 63;
        let fallback = FALLBACK_TAG | (canonical_token as u64 + 1);
        let Some(&first_token) = aliases.first() else { return fallback; };
        let word = first_token / 32;
        let mut bits = 0u32;
        for &token_id in aliases {
            if token_id / 32 != word { return fallback; }
            bits |= 1u32 << (token_id % 32);
        }
        (u64::from(word) << 32) | u64::from(bits)
    }

    fn build_full_walk_token_markers(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> Arc<Vec<u64>> {
        Arc::new(trie.all_subtree_tokens()[usize::from(trie.node(0).token_id.is_some())..]
            .iter().map(|&canonical_token| {
                let index = canonical_token as usize;
                let start = canonical_offsets[index] as usize;
                let end = canonical_offsets[index + 1] as usize;
                Self::encode_walk_marker(canonical_token, &canonical_original_tokens[start..end])
            }).collect())
    }
    #[inline]
    pub(crate) fn subtree_original_tokens(&self, node: u32) -> SubtreeOriginalTokens<'_> {
        let range = self.trie.subtree_token_index_range(node);
        let indexes = self.indexes();
        let offsets = indexes.canonical_original_token_offsets.as_slice();
        let originals = indexes.canonical_original_tokens.as_slice();
        if let Some(counts) = &indexes.permuted_subtree_alias_offsets {
            let remaining = (counts[range.end] - counts[range.start]) as usize;
            SubtreeOriginalTokens::Mapped {
                canonical: self.trie.all_subtree_tokens()[range].iter(),
                offsets,
                originals,
                current: [].iter(),
                remaining,
            }
        } else {
            SubtreeOriginalTokens::Flat(
                originals[offsets[range.start] as usize..offsets[range.end] as usize].iter(),
            )
        }
    }

    #[inline]
    pub(crate) fn canonical_token_count(&self) -> usize {
        match &self.alias_state {
            VocabAliasState::Transfer(aliases) => aliases.len(),
            VocabAliasState::Runtime(indexes) => indexes.canonical_original_token_offsets.len().saturating_sub(1),
            _ => 0,
        }
    }

    pub(crate) fn token_ids(&self, canonical_token_id: u32) -> Option<&[u32]> {
        let index = canonical_token_id as usize;
        match &self.alias_state {
            VocabAliasState::Transfer(aliases) => aliases.get(index)
                .filter(|ids| !ids.is_empty()).map(Vec::as_slice),
            VocabAliasState::Runtime(indexes) => {
                let end_index = index.checked_add(1)?;
                let (&start, &end) = indexes.canonical_original_token_offsets.get(index)
                    .zip(indexes.canonical_original_token_offsets.get(end_index))?;
                (start != end).then(|| &indexes.canonical_original_tokens[start as usize..end as usize])
            }
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn token_word_masks(&self, canonical_token_id: u32) -> &[(u32, u32)] {
        self.indexes().aliases.word_masks(canonical_token_id)
    }

    #[inline(always)]
    pub(crate) fn full_walk_token_markers(&self) -> &[u64] {
        self.indexes().full_walk_token_markers.as_ref()
    }

    pub(crate) fn set_direct_regular_terminal_support(
        &mut self,
        support: DirectRegularTerminalSupport,
    ) {
        self.direct_regular_terminal_support = Arc::new(support);
    }

    pub(crate) fn direct_regular_terminal_support(&self) -> &DirectRegularTerminalSupport {
        self.direct_regular_terminal_support.as_ref()
    }

    pub(crate) fn cached_direct_regular_wide_frontier_index(
        &self,
        key: usize,
    ) -> Option<usize> {
        self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .map(|(_, index)| *index)
    }

    pub(crate) fn cache_direct_regular_wide_frontier_index(
        &self,
        key: usize,
        source: &ParserGSS,
        index: usize,
    ) {
        let mut cache = self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.len() >= 1024 { cache.clear(); }
        // Retain the GSS interface which owns the pointer key.
        cache.insert(key, (source.clone(), index));
    }

    pub(crate) fn set_self_loop_projections(
        &mut self,
        projections: Vec<DynamicSelfLoopProjection>,
    ) {
        let state_count = self.mask_projection_tokenizer().map_or_else(
            || {
                projections
                    .iter()
                    .map(|projection| projection.source_state as usize + 1)
                    .max()
                    .unwrap_or(0)
            },
            |tokenizer| tokenizer.num_states() as usize,
        );
        let mut by_source = vec![u32::MAX; state_count];
        for (index, projection) in projections.iter().enumerate() {
            if let Some(slot) = by_source.get_mut(projection.source_state as usize) {
                *slot = index as u32;
            }
        }
        self.projection_by_source = Arc::from(by_source);
        self.self_loop_projections = Arc::new(projections);
    }

    pub(crate) fn set_projection_alias_vocab(&mut self, aliases: Vec<u32>) {
        self.projection_alias_vocab = Arc::from(aliases);
    }

    pub(crate) fn set_projection_alias_h64(&mut self, aliases: Vec<u32>) {
        self.projection_alias_h64 = Arc::from(aliases);
    }

    fn publish_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        map: MaskCoordinateMap,
        fast: Option<FastTokenizerTransitions>,
    ) {
        self.mask_projection = Some(MaskProjection {
            tokenizer: Arc::new(tokenizer),
            map,
            derivative: None,
            provenance: None,
        });
        self.mask_tokenizer_fast_transitions = fast;
    }

    fn set_mask_projection(&mut self, tokenizer: Tokenizer, map: MaskCoordinateMap) {
        let fast = Self::build_full_walk_fast_transitions(&tokenizer);
        self.publish_mask_projection(tokenizer, map, fast);
    }

    pub(crate) fn set_mask_tokenizer_quotient(
        &mut self,
        tokenizer: Tokenizer,
        full_to_mask_state: Vec<u32>,
    ) {
        debug_assert!(!full_to_mask_state.is_empty());
        debug_assert!(full_to_mask_state.iter().all(|&state| state < tokenizer.num_states()));
        self.set_mask_projection(tokenizer, MaskCoordinateMap::Dense(Arc::from(full_to_mask_state)));
    }

    fn set_mask_tokenizer_source_subsets(&mut self, source_subsets: Vec<Box<[u32]>>) {
        let Some(projection) = self.mask_projection.as_mut() else {
            return;
        };
        if source_subsets.len() != projection.runtime_owner().num_states() as usize
            || source_subsets.is_empty()
        {
            projection.provenance = None;
            return;
        }
        let by_state: Vec<Arc<[u32]>> = source_subsets.into_iter().map(Arc::from).collect();
        let mut by_subset = FxHashMap::default();
        by_subset.reserve(by_state.len());
        for (state, subset) in by_state.iter().enumerate() {
            // Multiple preserved raw IDs may represent the same closure.
            // Any such complete deterministic coordinate is exact.
            by_subset.insert(Arc::clone(subset), state as u32);
        }
        projection.provenance = Some(Arc::new(MaskSubsetProvenance { by_state, by_subset }));
    }

    #[inline]
    pub(crate) fn has_mask_subset_provenance(&self) -> bool {
        self.mask_projection.as_ref().is_some_and(|projection| projection.provenance.is_some())
    }

    /// Inverse lookup of the exact source-state union in the active execution
    /// coordinate, closing in the intermediate projection when appropriate.
    pub(crate) fn mask_runtime_state_for_source_states(
        &self,
        source_tokenizer: &Tokenizer,
        source_states: &[u32],
    ) -> Option<u32> {
        let projection = self.mask_projection.as_ref()?;
        let provenance = projection.provenance.as_ref()?;
        if source_states.is_empty() { return None; }
        let mut subset = SmallVec::<[u32; 32]>::new();
        for &state in source_states {
            let closure = if projection.derivative.is_some() {
                projection.tokenizer.singleton_epsilon_closure(projection.map.project(state))
            } else {
                source_tokenizer.singleton_epsilon_closure(state)
            };
            subset.extend_from_slice(&closure);
        }
        subset.sort_unstable();
        subset.dedup();
        provenance.by_subset.get(subset.as_slice()).copied()
    }

    /// Inverse lookup of an already materialized union of execution states.
    pub(crate) fn mask_projection_state_for_projection_states(
        &self,
        projection_states: &[u32],
    ) -> Option<u32> {
        let provenance = self.mask_projection.as_ref()?.provenance.as_ref()?;
        if projection_states.is_empty() { return None; }
        let mut subset = SmallVec::<[u32; 32]>::new();
        for &state in projection_states {
            subset.extend_from_slice(provenance.by_state.get(state as usize)?);
        }
        subset.sort_unstable();
        subset.dedup();
        provenance.by_subset.get(subset.as_slice()).copied()
    }

    pub(crate) fn set_virtual_unit_repeat_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projection: VirtualZeroMinUnitRepeatMaskProjection,
    ) {
        debug_assert_eq!(tokenizer.num_states(), projection.mask_state_count());
        self.set_mask_projection(tokenizer, MaskCoordinateMap::UnitRepeat(projection));
    }

    pub(crate) fn set_virtual_repeat_intersection_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projection: VirtualBinaryRepeatIntersectionMaskProjection,
    ) {
        self.set_virtual_repeat_intersections_mask_projection(tokenizer, vec![projection]);
    }

    pub(crate) fn set_virtual_repeat_intersections_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projections: Vec<VirtualBinaryRepeatIntersectionMaskProjection>,
    ) {
        debug_assert!(!projections.is_empty());
        self.set_mask_projection(tokenizer, MaskCoordinateMap::RepeatIntersections(Arc::from(projections)));
    }

    pub(crate) fn set_virtual_residuals_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projections: Vec<VirtualResidualMaskProjection>,
    ) {
        debug_assert!(!projections.is_empty());
        self.set_mask_projection(tokenizer, MaskCoordinateMap::Residuals(Arc::from(projections)));
    }

    pub(crate) fn virtual_residual_mask_projection_parts(
        &self,
    ) -> Option<(&Tokenizer, &[VirtualResidualMaskProjection])> {
        let projection = self.mask_projection.as_ref()?;
        let MaskCoordinateMap::Residuals(residuals) = &projection.map else { return None; };
        Some((projection.tokenizer.as_ref(), residuals.as_ref()))
    }

    /// Preserve lexer-derived dynamic-mask metadata when a deferred vocabulary
    /// placeholder is replaced by its fully materialized runtime trie.
    ///
    /// These quotients/projections are constraint/lexer derived while the trie
    /// is vocabulary derived, so deferred dynamic compilation may construct
    /// them at different times. Sharing the immutable Arc-backed metadata
    /// avoids cloning it during that handoff.
    pub(crate) fn inherit_dynamic_lexer_metadata_from(&mut self, source: &Self) {
        self.mask_projection = source.mask_projection.clone();
        self.mask_tokenizer_fast_transitions = source.mask_tokenizer_fast_transitions.clone();
        self.terminal_observation_classes = Arc::clone(&source.terminal_observation_classes);
        self.terminal_quotients = source.terminal_quotients.clone();
        self.prepared_master_proofs = source.prepared_master_proofs.clone();
    }

    pub(crate) fn mask_tokenizer_quotient_for_transfer(&self) -> Option<(Tokenizer, Vec<u32>)> {
        let projection = self.mask_projection.as_ref()?;
        let MaskCoordinateMap::Dense(states) = &projection.map else {
            return None;
        };
        Some((projection.tokenizer.as_ref().clone(), states.to_vec()))
    }

    #[inline]
    pub(crate) fn mask_projection_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_projection.as_ref().map(|projection| projection.tokenizer.as_ref())
    }

    #[inline]
    pub(crate) fn mask_runtime_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_runtime_tokenizer_owner().map(Arc::as_ref)
    }

    #[inline]
    pub(crate) fn mask_runtime_tokenizer_owner(&self) -> Option<&Arc<Tokenizer>> {
        self.mask_projection.as_ref().map(MaskProjection::runtime_owner)
    }

    #[inline]
    pub(crate) fn has_dense_mask_tokenizer_projection(&self) -> bool {
        self.mask_projection.as_ref().is_some_and(|projection| matches!(&projection.map, MaskCoordinateMap::Dense(_)))
    }

    #[inline]
    pub(crate) fn mask_projection_fast_transitions(&self) -> Option<&FastTokenizerTransitions> {
        self.mask_tokenizer_fast_transitions.as_ref()
    }

    pub(crate) fn clear_mask_projection_fast_transitions(&mut self) {
        self.mask_tokenizer_fast_transitions = None;
    }

    #[cfg(test)]
    pub(crate) fn disable_prepared_mask_execution_for_test(&mut self) {
        self.mask_projection = None;
        self.mask_tokenizer_fast_transitions = None;
    }

    #[inline]
    pub(crate) fn mask_projection_state(&self, full_state: u32) -> u32 {
        self.mask_projection.as_ref().map_or(full_state, |projection| projection.map.project(full_state))
    }

    #[inline]
    pub(crate) fn mask_runtime_state(&self, full_state: u32) -> u32 {
        let Some(projection) = self.mask_projection.as_ref() else { return full_state; };
        let projected = projection.map.project(full_state);
        projection.derivative.as_ref().and_then(|derived| {
            derived.from_projection.get(projected as usize).copied()
        }).unwrap_or(projected)
    }

    pub(crate) fn mask_projection_state_multiplicities(&self) -> Option<Vec<usize>> {
        let projection = self.mask_projection.as_ref()?;
        match &projection.map {
            MaskCoordinateMap::Dense(states) => {
                let mut counts = vec![0usize; projection.tokenizer.num_states() as usize];
                for &state in states.iter() {
                    if let Some(count) = counts.get_mut(state as usize) { *count += 1; }
                }
                Some(counts)
            }
            MaskCoordinateMap::UnitRepeat(unit) => Some(unit.multiplicities()),
            // Lazily populated exact state domains have no global finite
            // preimage enumeration.
            _ => None,
        }
    }

    /// Exact unique full-tokenizer preimages, with MAX denoting non-unique
    /// or unreachable mask states.
    pub(crate) fn mask_projection_unique_full_states(&self) -> Option<Vec<u32>> {
        let projection = self.mask_projection.as_ref()?;
        match &projection.map {
            MaskCoordinateMap::Dense(states) => {
                let count = projection.tokenizer.num_states() as usize;
                let mut unique = vec![u32::MAX; count];
                let mut duplicate = vec![false; count];
                for (full, &state) in states.iter().enumerate() {
                    let index = state as usize;
                    if index >= count { continue; }
                    if unique[index] == u32::MAX && !duplicate[index] {
                        unique[index] = full as u32;
                    } else {
                        duplicate[index] = true;
                        unique[index] = u32::MAX;
                    }
                }
                Some(unique)
            }
            MaskCoordinateMap::UnitRepeat(unit) => Some(unit.unique_full_states()),
            _ => None,
        }
    }

    /// Lookup by a state in the active mask-tokenizer coordinate. Callers that
    /// hold an exact committed tokenizer state must project it first.
    pub(crate) fn self_loop_projection(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let index = *self.projection_by_source.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    /// H64 alias lookup in the active mask-tokenizer coordinate.
    pub(crate) fn self_loop_projection_alias_h64(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let index = *self.projection_alias_h64.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    /// Vocabulary alias lookup in the active mask-tokenizer coordinate.
    pub(crate) fn self_loop_projection_alias_vocab(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let index = *self.projection_alias_vocab.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    #[inline]
    pub(crate) fn has_self_loop_projections(&self) -> bool {
        !self.self_loop_projections.is_empty()
    }

    pub(crate) fn set_bounded_observation_sets(
        &mut self,
        sets: DynamicBoundedObservationSets,
    ) {
        self.bounded_observation_sets = Arc::new(sets);
    }

    #[inline]
    pub(crate) fn bounded_observation_safe_bytes(
        &self,
        source: u32,
        required_horizon: u32,
    ) -> Option<U8Set> {
        self.bounded_observation_sets
            .safe_bytes(source, required_horizon)
    }

    #[inline]
    pub(crate) fn bounded_observation_set_counts(&self) -> (usize, usize) {
        (
            self.bounded_observation_sets.state_count(),
            self.bounded_observation_sets.unique_set_count(),
        )
    }

    pub(crate) fn set_terminal_observation_classes(
        &mut self,
        mut classes: Vec<(TerminalID, Arc<[u32]>)>,
    ) {
        classes.sort_unstable_by_key(|(terminal, _)| *terminal);
        classes.dedup_by_key(|(terminal, _)| *terminal);
        self.terminal_observation_classes = Arc::from(classes);
    }

    #[inline]
    pub(crate) fn terminal_observation_class(
        &self,
        terminal: TerminalID,
        state: u32,
    ) -> Option<u32> {
        let index = self
            .terminal_observation_classes
            .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            .ok()?;
        self.terminal_observation_classes[index]
            .1
            .get(state as usize)
            .copied()
            .filter(|&class| class != 0)
    }

    #[inline]
    pub(crate) fn has_terminal_observation_classes(&self) -> bool {
        !self.terminal_observation_classes.is_empty()
    }

    /// Cheap necessary condition for the exact parser-relative observation
    /// certificate: at least one prepared terminal quotient places both lexer
    /// source states in the same nonzero class. This is not sufficient for
    /// parser-relative equivalence, but it cheaply avoids arming the expensive
    /// checkpoint for unrelated source pairs.
    #[inline]
    pub(crate) fn shares_terminal_observation_class(
        &self,
        left_state: u32,
        right_state: u32,
    ) -> bool {
        let left = left_state as usize;
        let right = right_state as usize;
        self.terminal_observation_classes.iter().any(|(_, classes)| {
            let Some(&left_class) = classes.get(left) else {
                return false;
            };
            left_class != 0 && classes.get(right).copied() == Some(left_class)
        })
    }

    /// Prove equality of every parser-admitted terminal observation that is
    /// live at either lexer source, using only the selectively prepared exact
    /// per-terminal quotient rows. Returns the number of relevant live
    /// terminals on success; missing quotient coverage is a conservative
    /// decline.
    ///
    /// Iterate the tiny prepared-class sidecar (normally one or two terminals)
    /// rather than all parser-admitted terminals. The live-count pass is word
    /// based, so the common <=64-terminal grammar costs a handful of integer
    /// operations instead of repeated bitset probes and binary searches.
    #[inline]
    pub(crate) fn terminal_observation_equivalent_for_live_admitted(
        &self,
        left_state: u32,
        right_state: u32,
        admitted: &BitSet,
        left_matched: &BitSet,
        left_future: &BitSet,
        right_matched: &BitSet,
        right_future: &BitSet,
    ) -> Option<usize> {
        if self.terminal_observation_classes.is_empty() {
            return None;
        }

        let mut live_count = 0usize;
        for (word_index, &admitted_word) in admitted.words().iter().enumerate() {
            let word = |set: &BitSet| set.words().get(word_index).copied().unwrap_or(0);
            let live = admitted_word
                & (word(left_matched)
                    | word(left_future)
                    | word(right_matched)
                    | word(right_future));
            live_count += live.count_ones() as usize;
        }
        if live_count == 0 {
            return None;
        }

        let mut covered = 0usize;
        for (terminal, classes) in self.terminal_observation_classes.iter() {
            let terminal = *terminal as usize;
            if !admitted.contains(terminal)
                || !(left_matched.contains(terminal)
                    || left_future.contains(terminal)
                    || right_matched.contains(terminal)
                    || right_future.contains(terminal))
            {
                continue;
            }
            covered += 1;
            let left = classes.get(left_state as usize).copied().unwrap_or(0);
            let right = classes.get(right_state as usize).copied().unwrap_or(0);
            if left == 0 || left != right {
                return None;
            }
        }
        (covered == live_count).then_some(live_count)
    }

    pub(crate) fn terminal_observation_classes_cloned(
        &self,
    ) -> Vec<(TerminalID, Arc<[u32]>)> {
        self.terminal_observation_classes
            .iter()
            .map(|(terminal, classes)| (*terminal, Arc::clone(classes)))
            .collect()
    }

    pub(crate) fn terminal_observation_classes_for_artifact(
        &self,
    ) -> Vec<(TerminalID, Vec<u32>)> {
        self.terminal_observation_classes
            .iter()
            .map(|(terminal, classes)| (*terminal, classes.as_ref().to_vec()))
            .collect()
    }

    pub(crate) fn set_projected_terminal_quotients(
        &mut self,
        mut quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    ) {
        quotients.sort_unstable_by_key(|(terminal, _)| *terminal);
        quotients.dedup_by_key(|(terminal, _)| *terminal);
        self.terminal_quotients = ProjectedTerminalInventory::Prepared(Arc::from(
            quotients
                .into_iter()
                .map(|(terminal, quotient)| (terminal, Arc::new(quotient)))
                .collect::<Vec<_>>(),
        ));
        self.projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub(crate) fn projected_terminal_quotients_for_artifact(
        &self,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        match &self.terminal_quotients {
            ProjectedTerminalInventory::Prepared(quotients) => quotients.iter()
                .map(|(terminal, quotient)| (*terminal, quotient.as_ref().clone()))
                .collect(),
            ProjectedTerminalInventory::Runtime(_) => Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn projected_terminal_quotient(
        &self,
        terminal: TerminalID,
        source: u32,
    ) -> Option<Arc<TerminalProjectedQuotient>> {
        let quotient = match &self.terminal_quotients {
            ProjectedTerminalInventory::Prepared(quotients) => {
                let index = quotients.binary_search_by_key(&terminal, |(id, _)| *id).ok()?;
                Arc::clone(&quotients[index].1)
            }
            ProjectedTerminalInventory::Runtime(runtime) => runtime.by_terminal
                .lock().unwrap_or_else(|p| p.into_inner())
                .get(&terminal).cloned().flatten()?,
        };
        quotient.contains_source(source).then_some(quotient)
    }

    /// Prepare exactly one runtime containment quotient. This is the hot-path
    /// counterpart to `prepare_runtime_projected_terminal_quotients`: callers
    /// that already know the parser/lexer-admitted terminal should not pay to
    /// materialize every terminal whose byte support happens to cover the
    /// generic safe-string alphabet.
    pub(crate) fn prepare_runtime_projected_terminal_quotient(
        &self,
        source: &Tokenizer,
        terminal: TerminalID,
    ) {
        let ProjectedTerminalInventory::Runtime(runtime) = &self.terminal_quotients else {
            return;
        };
        if runtime.batch_complete.get().is_some()
            || runtime.by_terminal.lock().unwrap_or_else(|p| p.into_inner()).contains_key(&terminal)
        {
            return;
        }

        let state_cap = crate::env::var("GLRMASK_EXPERIMENT_DEMAND_PROOF_STATE_CAP")
            .ok().and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2048);
        let built = if crate::env::var_os("GLRMASK_DISABLE_DEMAND_PROOF_CAP").is_none() {
            let edge_cap = crate::env::var("GLRMASK_EXPERIMENT_DEMAND_PROOF_EDGE_CAP")
                .ok().and_then(|value| value.parse::<usize>().ok())
                .unwrap_or_else(|| state_cap.saturating_mul(128));
            let quotient = source.build_terminal_projected_quotient_for_containment_bounded(
                terminal, state_cap, edge_cap,
            );
            if crate::env::var_os("GLRMASK_DIAG_DEMAND_PROOF_CAP").is_some() {
                eprintln!("[demand_proof_cap] terminal={} state_cap={} edge_cap={} built={}",
                    terminal, state_cap, edge_cap, quotient.is_some());
            }
            quotient.map(Arc::new)
        } else {
            source.build_terminal_projected_quotients_for_containment_candidates(&[terminal])
                .into_iter().find_map(|(candidate, quotient)| {
                    (candidate == terminal).then(|| Arc::new(quotient))
                })
        };
        runtime.by_terminal.lock().unwrap_or_else(|p| p.into_inner())
            .entry(terminal).or_insert(built);
    }

    pub(crate) fn prepare_runtime_projected_terminal_quotients(
        &self,
        source: &Tokenizer,
        safe_slice_bytes: &U8Set,
        component_state_cap: Option<usize>,
    ) {
        let ProjectedTerminalInventory::Runtime(runtime) = &self.terminal_quotients else {
            return;
        };
        runtime.batch_complete.get_or_init(|| {
            let candidates = (0..source.num_terminals())
                .filter(|&terminal| {
                    source
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| safe_slice_bytes.is_subset(&support))
                })
                .collect::<Vec<_>>();
            let quotients = match component_state_cap {
                Some(cap) => source.build_terminal_projected_quotients_for_containment_candidates_bounded(
                    &candidates, cap, cap.saturating_mul(256),
                ),
                None => source.build_terminal_projected_quotients_for_containment_candidates(&candidates),
            };
            let mut inventory = runtime.by_terminal.lock().unwrap_or_else(|p| p.into_inner());
            for (terminal, quotient) in quotients {
                // A bounded decline is not an exact negative. A successful
                // batch proof may fill a previously declined entry, but
                // never discard an already usable exact quotient.
                let entry = inventory.entry(terminal).or_default();
                if entry.is_none() {
                    *entry = Some(Arc::new(quotient));
                }
            }
        });
    }

    #[inline]
    pub(crate) fn has_projected_terminal_quotients(&self) -> bool {
        match &self.terminal_quotients {
            ProjectedTerminalInventory::Prepared(quotients) => !quotients.is_empty(),
            ProjectedTerminalInventory::Runtime(runtime) => runtime.by_terminal
                .lock().unwrap_or_else(|p| p.into_inner()).values().any(Option::is_some),
        }
    }

    #[inline]
    pub(crate) fn projected_terminal_quotients_prepared(&self) -> bool {
        matches!(self.terminal_quotients, ProjectedTerminalInventory::Prepared(_))
    }

    pub(crate) fn projected_terminal_text_liveness(
        &self,
        terminal: TerminalID,
        source: u32,
        alphabet: U8Set,
        include_utf8_scalars: bool,
    ) -> Option<bool> {
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let key = (terminal, source, alphabet, include_utf8_scalars);
        if let Some(&cached) = self
            .projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let certified = quotient
            .text_liveness_closed_bounded(
                source,
                alphabet,
                1_024,
                200_000,
                include_utf8_scalars,
            )
            .unwrap_or(false);
        self.projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, certified);
        Some(certified)
    }
    pub(crate) fn projected_terminal_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice_dfa: &VocabPartitionDfa,
    ) -> Option<bool> {
        let key = (terminal, source, 0x8000_0000u32 | slice_cache_id);
        if let Some(&cached) = self
            .projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let certified = Self::terminal_partition_product_is_transparent(
            quotient.as_ref(), slice_dfa, source, 200_000,
        )
        .unwrap_or(false);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, certified);
        Some(certified)
    }

    /// Prove a bounded safe-atom prefix radius directly in the physical lexer.
    /// Unlike the projected proof this explores only the requested horizon,
    /// without first building/minimizing the terminal's entire residual DFA.
    /// The slice must mark each completed atom by entering an accepting state,
    /// as the safe+ UTF-8 slice does. The shortest failing accepted word gives
    /// a universal safe lower bound for all strictly shorter atom layers.
    /// Every traversed prefix must retain the selected terminal. Epsilon or
    /// nonphysical coordinates and exhausted work budgets fail closed.
    pub(crate) fn raw_terminal_slice_repeat_radius(
        &self,
        tokenizer: &Tokenizer,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        if max_repetitions == 0
            || source >= tokenizer.num_states()
            || tokenizer.state_is_virtual_runtime(source)
            || tokenizer.state_has_epsilon_transitions(source)
            || slice.accepting_map().get(slice.start_state() as usize).copied()?
        {
            return None;
        }
        // A dedicated cache cannot alias projected/symbolic certificate keys.
        // Slice IDs refer to immutable languages within this vocabulary instance.
        // Include budget so an earlier cheap failed attempt cannot mask a later
        // larger-budget certificate. Cached zero is always conservative.
        let key = (terminal, source, slice_cache_id, max_repetitions, work_limit);
        if let Some(&cached) = self.raw_terminal_radius_cache.lock()
            .unwrap_or_else(|p| p.into_inner()).get(&key)
        {
            return Some(cached);
        }
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diagnostic = *PROFILE.get_or_init(|| {
            crate::env::var_os("GLRMASK_PROFILE_RAW_TERMINAL_RADIUS").is_some()
        });
        static BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static VERIFY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let batch_rows = *BATCH.get_or_init(|| {
            crate::env::var_os("GLRMASK_DISABLE_RAW_RADIUS_BATCH_ROWS").is_none()
        });
        let verify_rows = *VERIFY.get_or_init(|| {
            crate::env::var_os("GLRMASK_ASSERT_RAW_RADIUS_BATCH_ROWS").is_some()
        });
        let started = diagnostic.then(std::time::Instant::now);
        let mut work = 0usize;
        let mut explored = 0usize;
        let result = (|| {
            let slice_count = slice.accepting_map().len();
            let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_count];
            for state in 0..slice_count as u32 {
                let mut seen = FxHashSet::default();
                for byte in 0u16..=255 {
                    let target = slice.step(state, byte as u8);
                    if (target as usize) < slice_count && seen.insert(target) {
                        reverse[target as usize].push((state, u8::from(slice.accepting_map()[target as usize])));
                    }
                }
            }
            let mut min_to_accept = vec![u32::MAX; slice_count];
            let mut distances = VecDeque::new();
            for (state, &accepting) in slice.accepting_map().iter().enumerate() {
                if accepting { min_to_accept[state] = 0; distances.push_back(state as u32); }
            }
            while let Some(target) = distances.pop_front() {
                for &(state, cost) in &reverse[target as usize] {
                    let next = min_to_accept[target as usize].saturating_add(u32::from(cost));
                    if next < min_to_accept[state as usize] {
                        min_to_accept[state as usize] = next;
                        if cost == 0 { distances.push_front(state); } else { distances.push_back(state); }
                    }
                }
            }
            let mut best = FxHashMap::<(u32, u32), u32>::default();
            let mut queue = VecDeque::from([(slice.start_state(), source, 0u32)]);
            best.insert((slice.start_state(), source), 0);
            let mut first_bad = max_repetitions.saturating_add(1);
            while let Some((slice_state, lexer_state, completed)) = queue.pop_front() {
                if best.get(&(slice_state, lexer_state)).copied() != Some(completed)
                    || completed >= first_bad || completed > max_repetitions
                { continue; }
                if lexer_state >= tokenizer.num_states()
                    || tokenizer.state_is_virtual_runtime(lexer_state)
                    || tokenizer.state_has_epsilon_transitions(lexer_state)
                { return None; }
                explored += 1;
                // The guarded physical, epsilon-free row is exactly step().
                // Materialize once per product node; no row survives the proof.
                let mut row = [None; 256];
                if batch_rows {
                    for (byte, target) in tokenizer.transitions_from(lexer_state) {
                        row[byte as usize] = Some(target);
                    }
                    if verify_rows {
                        for byte in 0u16..=255 {
                            assert_eq!(row[byte as usize], tokenizer.step(lexer_state, byte as u8));
                        }
                    }
                }
                let mut previous_edge = None;
                for byte in 0u16..=255 {
                    let target_slice = slice.step(slice_state, byte as u8);
                    if target_slice as usize >= slice_count || !slice.can_reach_accepting(target_slice) { continue; }
                    let next_completed = completed.saturating_add(u32::from(slice.accepting_map()[target_slice as usize]));
                    let rest = min_to_accept[target_slice as usize];
                    if rest == u32::MAX { continue; }
                    let shortest_word = next_completed.saturating_add(rest);
                    if shortest_word > max_repetitions || shortest_word >= first_bad { continue; }
                    work += 1;
                    if work > work_limit { return None; }
                    let target = if batch_rows {
                        row[byte as usize]
                    } else {
                        tokenizer.step(lexer_state, byte as u8)
                    };
                    // Equal product targets have identical atom costs, liveness
                    // and successors. The original eligible-byte budget has
                    // ALREADY been charged above, including duplicate edges.
                    if batch_rows {
                        let edge = (target_slice, target);
                        if previous_edge == Some(edge) { continue; }
                        previous_edge = Some(edge);
                    }
                    if target.is_some_and(|t| tokenizer.state_is_virtual_runtime(t)) {
                        return None;
                    }
                    let live = target.is_some_and(|t| t < tokenizer.num_states()
                        && (tokenizer.possible_future_terminals(t).contains(terminal as usize)
                            || tokenizer.matched_terminals_slice(t).contains(&terminal)));
                    if !live {
                        first_bad = first_bad.min(shortest_word);
                        continue;
                    }
                    let target = target.expect("live target exists");
                    if tokenizer.state_has_epsilon_transitions(target) { return None; }
                    if next_completed >= first_bad { continue; }
                    let state_key = (target_slice, target);
                    if next_completed < best.get(&state_key).copied().unwrap_or(u32::MAX) {
                        best.insert(state_key, next_completed);
                        if slice.accepting_map()[target_slice as usize] {
                            queue.push_back((target_slice, target, next_completed));
                        } else { queue.push_front((target_slice, target, next_completed)); }
                    }
                }
            }
            Some(first_bad.saturating_sub(1).min(max_repetitions))
        })();
        // Zero is always a sound lower bound. Remember failures so an expensive
        // source cannot repeatedly spend the same bounded proof budget.
        let radius = result.unwrap_or(0);
        self.raw_terminal_radius_cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, radius);
        if diagnostic {
            eprintln!("[raw_terminal_radius] source={} terminal={} radius={} complete={} work={} states={} max={} ms={:.3}",
                source, terminal, radius, result.is_some(), work, explored, max_repetitions,
                started.map_or(0.0, |s| s.elapsed().as_secs_f64()*1000.0));
        }
        Some(radius)
    }


    /// Return the largest completed-atom count `r <= max_repetitions` such
    /// that every word in the regular `slice+` language with at most `r`
    /// completed atoms remains a live prefix of this exact projected terminal.
    ///
    /// The slice DFA must mark completion of one atom by entering an accepting
    /// state (the safe+ UTF-8 DFA has exactly this property). This is the finite
    /// quotient analogue of the symbolic residual repeat-radius proof: find the
    /// shortest slice word that reaches a dead/missing quotient transition, and
    /// admit every strictly shorter completed-atom layer.
    pub(crate) fn projected_terminal_slice_repeat_radius(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        if max_repetitions == 0 || slice.accepting_map().get(slice.start_state() as usize).copied()? {
            return None;
        }
        let key = (terminal, source, slice_cache_id, max_repetitions);
        if let Some(&cached) = self
            .projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let quotient_start = quotient.projected_state_for_source(source)?;
        if !quotient.projected_state_is_accepting(quotient_start)
            && !quotient.projected_state_has_future(quotient_start)
        {
            return Some(0);
        }

        let slice_state_count = slice.accepting_map().len();
        if slice_state_count == 0 {
            return None;
        }

        // Exact representatives for the common refinement of the slice and
        // quotient byte partitions.
        let mut representatives = Vec::<u8>::new();
        let mut seen_classes = FxHashSet::<(u8, u8)>::default();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let pair = (slice.byte_class(byte), quotient.projected_byte_class(byte));
            if seen_classes.insert(pair) {
                representatives.push(byte);
            }
        }

        // Minimum additional completed atoms needed to reach slice acceptance
        // from every slice state. Entering an accepting state completes one
        // atom; UTF-8 continuation states therefore have zero-cost edges.
        let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_state_count];
        for source_state in 0..slice_state_count as u32 {
            let mut seen_targets = FxHashSet::<u32>::default();
            for &byte in &representatives {
                let target = slice.step(source_state, byte);
                if target as usize >= slice_state_count || !seen_targets.insert(target) {
                    continue;
                }
                reverse[target as usize].push((
                    source_state,
                    u8::from(slice.accepting_map()[target as usize]),
                ));
            }
        }
        let mut min_to_accept = vec![u32::MAX; slice_state_count];
        let mut distance_queue = VecDeque::<u32>::new();
        for (state, &accepting) in slice.accepting_map().iter().enumerate() {
            if accepting {
                min_to_accept[state] = 0;
                distance_queue.push_back(state as u32);
            }
        }
        while let Some(target) = distance_queue.pop_front() {
            let target_distance = min_to_accept[target as usize];
            for &(source_state, cost) in &reverse[target as usize] {
                let candidate = target_distance.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source_state as usize] {
                    min_to_accept[source_state as usize] = candidate;
                    if cost == 0 {
                        distance_queue.push_front(source_state);
                    } else {
                        distance_queue.push_back(source_state);
                    }
                }
            }
        }

        let mut best = FxHashMap::<(u32, u32), u32>::default();
        let mut queue = VecDeque::<(u32, u32, u32)>::new();
        let slice_start = slice.start_state();
        best.insert((slice_start, quotient_start), 0);
        queue.push_back((slice_start, quotient_start, 0));
        let mut work = 0usize;
        let mut first_counterexample = max_repetitions.saturating_add(1);

        while let Some((slice_state, quotient_state, completed)) = queue.pop_front() {
            if best.get(&(slice_state, quotient_state)).copied() != Some(completed)
                || completed >= first_counterexample
                || completed > max_repetitions
            {
                continue;
            }
            for &byte in &representatives {
                let slice_target = slice.step(slice_state, byte);
                if slice_target as usize >= slice_state_count
                    || !slice.can_reach_accepting(slice_target)
                {
                    continue;
                }
                let completed_target = completed.saturating_add(u32::from(
                    slice.accepting_map()[slice_target as usize],
                ));
                let completion_cost = min_to_accept[slice_target as usize];
                if completion_cost == u32::MAX {
                    continue;
                }
                let shortest_complete_word = completed_target.saturating_add(completion_cost);
                if shortest_complete_word > max_repetitions {
                    continue;
                }
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let quotient_class = quotient.projected_byte_class(byte);
                let quotient_target = quotient.projected_step_class(quotient_state, quotient_class);
                let target_live = quotient_target.is_some_and(|target| {
                    quotient.projected_state_is_accepting(target)
                        || quotient.projected_state_has_future(target)
                });
                if !target_live {
                    first_counterexample = first_counterexample.min(shortest_complete_word);
                    continue;
                }
                let quotient_target = quotient_target.expect("live quotient target must exist");
                if completed_target >= first_counterexample
                    || completed_target > max_repetitions
                {
                    continue;
                }
                let state_key = (slice_target, quotient_target);
                if completed_target < best.get(&state_key).copied().unwrap_or(u32::MAX) {
                    best.insert(state_key, completed_target);
                    if slice.accepting_map()[slice_target as usize] {
                        queue.push_back((slice_target, quotient_target, completed_target));
                    } else {
                        queue.push_front((slice_target, quotient_target, completed_target));
                    }
                }
            }
        }

        let radius = first_counterexample.saturating_sub(1).min(max_repetitions);
        self.projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, radius);
        Some(radius)
    }

    pub(crate) fn cached_pending_guard_blocked_mask(
        &self,
        memories: &[(u32, TerminalID)],
    ) -> Option<Arc<Vec<u32>>> {
        self.pending_guard_blocked_mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(memories)
            .cloned()
    }

    pub(crate) fn cache_pending_guard_blocked_mask(
        &self,
        memories: &[(u32, TerminalID)],
        mask: Vec<u32>,
    ) -> Arc<Vec<u32>> {
        // Bound retained derived data independently of corpus behavior. A
        // Llama-sized mask is about 16 KiB, so 256 entries cap this cache at
        // roughly 4 MiB plus map/key overhead.
        const MAX_PENDING_GUARD_MASK_CACHE_ENTRIES: usize = 256;
        let mut cache = self
            .pending_guard_blocked_mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache.get(memories) {
            return Arc::clone(existing);
        }
        if cache.len() >= MAX_PENDING_GUARD_MASK_CACHE_ENTRIES {
            cache.clear();
        }
        let mask = Arc::new(mask);
        cache.insert(memories.to_vec(), Arc::clone(&mask));
        mask
    }

    pub(crate) fn cached_residual_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
    ) -> Option<bool> {
        let key = (terminal, source, 0xA000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .copied()
    }

    pub(crate) fn cache_residual_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        contained: bool,
    ) {
        let key = (terminal, source, 0xA000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, contained);
    }

    pub(crate) fn cached_direct_slice_contained(
        &self,
        terminal: TerminalID,
        mask_state: u32,
        slice_cache_id: u32,
    ) -> Option<bool> {
        let key = (terminal, mask_state, 0xC000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .copied()
    }

    pub(crate) fn cache_direct_slice_contained(
        &self,
        terminal: TerminalID,
        mask_state: u32,
        slice_cache_id: u32,
        contained: bool,
    ) {
        let key = (terminal, mask_state, 0xC000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, contained);
    }

    fn terminal_partition_product_is_transparent(
        quotient: &TerminalProjectedQuotient,
        partition: &VocabPartitionDfa,
        source: u32,
        work_limit: usize,
    ) -> Option<bool> {
        let quotient_start = quotient.projected_state_for_source(source)?;
        if !quotient.projected_state_is_accepting(quotient_start)
            && !quotient.projected_state_has_future(quotient_start)
        {
            return Some(false);
        }
        let partition_start = partition.start_state();
        if !partition.can_reach_accepting(partition_start) {
            return Some(true);
        }

        // Refine the two exact global byte partitions. A byte representative is
        // interchangeable only when both automata put it in the same class.
        let mut representatives = Vec::<(u8, u8, u8)>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let partition_class = partition.byte_class(byte);
            let quotient_class = quotient.projected_byte_class(byte);
            if !representatives.iter().any(|&(p, q, _)| {
                p == partition_class && q == quotient_class
            }) {
                representatives.push((partition_class, quotient_class, byte));
            }
        }

        let mut seen = FxHashSet::<(u32, u32)>::default();
        let mut queue = VecDeque::from([(partition_start, quotient_start)]);
        let mut work = 0usize;
        while let Some((partition_state, quotient_state)) = queue.pop_front() {
            if !seen.insert((partition_state, quotient_state)) {
                continue;
            }
            for &(_, quotient_class, byte) in &representatives {
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let partition_target = partition.step(partition_state, byte);
                if !partition.can_reach_accepting(partition_target) {
                    continue;
                }
                let Some(quotient_target) =
                    quotient.projected_step_class(quotient_state, quotient_class)
                else {
                    return Some(false);
                };
                if !quotient.projected_state_is_accepting(quotient_target)
                    && !quotient.projected_state_has_future(quotient_target)
                {
                    return Some(false);
                }
                if !seen.contains(&(partition_target, quotient_target)) {
                    queue.push_back((partition_target, quotient_target));
                }
            }
        }
        Some(true)
    }

    pub(crate) fn cached_direct_regular_frontier(
        &self,
        key: usize,
    ) -> Option<DirectRegularDynamicFrontierCacheEntry> {
        self.direct_regular_frontier_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
    }

    pub(crate) fn cache_direct_regular_frontier(
        &self,
        key: usize,
        entry: DirectRegularDynamicFrontierCacheEntry,
    ) -> DirectRegularDynamicFrontierCacheEntry {
        const MAX_FRONTIER_CACHE_ENTRIES: usize = 1024;
        let mut cache = self
            .direct_regular_frontier_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache.get(&key) {
            return existing.clone();
        }
        if cache.len() >= MAX_FRONTIER_CACHE_ENTRIES {
            // Cache entries retain their source GSS interface, making pointer
            // keys safe. Clearing atomically drops both keys and retained
            // interfaces before any allocator reuse can produce a new key.
            cache.clear();
        }
        cache.insert(key, entry.clone());
        entry
    }

    pub(crate) fn lock_lazy_union_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, DynamicLazyUnionCache> {
        self.lazy_union_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn try_lock_lazy_union_cache(
        &self,
    ) -> Option<std::sync::MutexGuard<'_, DynamicLazyUnionCache>> {
        match self.lazy_union_cache.try_lock() {
            Ok(cache) => Some(cache),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }

    pub(crate) fn copy_cached_mask(
        &self,
        state: &DynamicMaskStateKey,
        hash: u64,
        buf: &mut [u32],
    ) -> bool {
        self.copy_cached_mask_with_predicate(hash, |candidate| candidate == state, buf)
    }

    pub(crate) fn copy_cached_mask_with_predicate<F: Fn(&DynamicMaskStateKey) -> bool>(
        &self,
        hash: u64,
        matches: F,
        buf: &mut [u32],
    ) -> bool {
        let cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(slots) = cache.by_hash.get(&hash) else {
            return false;
        };
        let Some(entry) = slots.iter().rev().find_map(|&slot| {
            cache
                .entries
                .get(slot)
                .and_then(Option::as_ref)
                .filter(|entry| matches(&entry.state))
        }) else {
            return false;
        };
        entry.mask.as_ref().is_some_and(|payload| {
            Self::copy_dynamic_mask_cache_payload(self.all_original_token_words(), payload, buf)
        })
    }

    pub(crate) fn has_cached_mask_with_predicate<F: Fn(&DynamicMaskStateKey) -> bool>(
        &self,
        hash: u64,
        matches: F,
    ) -> bool {
        let cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(slots) = cache.by_hash.get(&hash) else {
            return false;
        };
        slots.iter().rev().any(|&slot| {
            cache
                .entries
                .get(slot)
                .and_then(Option::as_ref)
                .is_some_and(|entry| {
                    matches(&entry.state) && entry.mask.is_some()
                })
        })
    }

    fn copy_dynamic_mask_cache_payload(
        baseline: &[u32],
        payload: &DynamicMaskCachePayload,
        buf: &mut [u32],
    ) -> bool {
        match payload {
            DynamicMaskCachePayload::Probation => return false,
            DynamicMaskCachePayload::Dense(mask) => {
                if mask.len() != buf.len() {
                    return false;
                }
                buf.copy_from_slice(mask);
            }
            DynamicMaskCachePayload::SparseZero(words) => {
                buf.fill(0);
                for &(word, value) in words.iter() {
                    let Some(dst) = buf.get_mut(word as usize) else {
                        return false;
                    };
                    *dst = value;
                }
            }
            DynamicMaskCachePayload::SparseAllOriginal(words) => {
                let copy_len = buf.len().min(baseline.len());
                buf[..copy_len].copy_from_slice(&baseline[..copy_len]);
                if copy_len < buf.len() {
                    buf[copy_len..].fill(0);
                }
                for &(word, value) in words.iter() {
                    let Some(dst) = buf.get_mut(word as usize) else {
                        return false;
                    };
                    *dst = value;
                }
            }
        }
        true
    }

    fn dynamic_mask_cache_payload(&self, mask: &[u32]) -> DynamicMaskCachePayload {
        // G: the entry budget already charges every payload its full dense
        // size. Direct storage therefore stays within that existing payload
        // bound, at the cost of giving up sparse working-set compression.
        // Avoid the two classification scans plus sparse materialization scan.
        // Exact word-for-word copy; key matching, eviction, and admission are
        // unchanged. Experiment only until representative measurements pass.
        static DIRECT_COPY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DIRECT_COPY.get_or_init(|| {
            crate::env::var_os("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_DENSE_ONLY").is_some()
        }) {
            return DynamicMaskCachePayload::Dense(Arc::from(mask));
        }
        super::mask_cache_payload::from_words(mask, self.all_original_token_words())
    }

    /// Limit one immutable vocabulary's result memo independently of the
    /// ordinary full-vocabulary cache budget. Set only before publishing it.
    pub(crate) fn with_mask_cache_entry_cap(self, cap: usize) -> Self {
        {
            let mut cache = self.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
            assert!(cache.entries.is_empty());
            cache.entry_cap = Some(cap.clamp(1, 4096));
        }
        self
    }

    pub(crate) fn cache_mask(
        &self,
        state: DynamicMaskStateKey,
        hash: u64,
        mask: &[u32],
        probation_if_absent: bool,
    ) {
        // Keep enough exact states to cover an ordinary generated sequence.
        // A fixed 64-entry limit caused long source-specialized sequences to
        // evict their expensive early masks during the warmup pass, so every
        // measured pass recomputed them. Bound by bytes instead: Llama-sized
        // masks retain about 2k states in 32 MiB, while tiny vocabularies may
        // retain more without material memory cost.
        // Opt-in bounded capacity experiment; default and key semantics unchanged.
        static CACHE_BUDGET_MIB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let budget_mib = *CACHE_BUDGET_MIB.get_or_init(|| {
            crate::env::var("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_BUDGET_MIB")
                .ok().and_then(|s| s.parse::<usize>().ok())
                .filter(|n| (1..=64).contains(n)).unwrap_or(32)
        });
        const MIN_MASK_CACHE_ENTRIES: usize = 64;
        const MAX_MASK_CACHE_ENTRIES: usize = 4096;
        let mask_bytes = mask.len().saturating_mul(std::mem::size_of::<u32>()).max(1);
        let max_entries = ((budget_mib * 1024 * 1024) / mask_bytes)
            .clamp(MIN_MASK_CACHE_ENTRIES, MAX_MASK_CACHE_ENTRIES);
        let value_hash = super::mask_cache_values::mask_hash(mask);
        let mut cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let max_entries = max_entries.min(cache.entry_cap.unwrap_or(usize::MAX));
        if let Some(slots) = cache.by_hash.get(&hash).cloned() {
            for slot in slots {
                let matches = cache
                    .entries
                    .get(slot)
                    .and_then(Option::as_ref)
                    .is_some_and(|entry| entry.state == state);
                if !matches {
                    continue;
                }
                let needs_upgrade = cache.entries[slot]
                    .as_ref()
                    .is_some_and(|entry| entry.mask.is_none());
                if needs_upgrade {
                    let payload = match cache.values.get(value_hash, mask, self.all_original_token_words()) {
                        Some(existing) => existing,
                        None => {
                            let payload = Arc::new(self.dynamic_mask_cache_payload(mask));
                            cache.values.insert(value_hash, mask.len(), &payload);
                            payload
                        }
                    };
                    cache.entries[slot]
                        .as_mut()
                        .expect("probation cache slot disappeared")
                        .mask = Some(payload);
                }
                return;
            }
        }
        let payload = if let Some(existing) = cache.values.get(value_hash, mask, self.all_original_token_words()) {
            Some(existing)
        } else if probation_if_absent
            && !super::mask_cache_payload::small_first_use_sparse(mask, self.all_original_token_words())
        {
            None
        } else {
            let payload = Arc::new(self.dynamic_mask_cache_payload(mask));
            cache.values.insert(value_hash, mask.len(), &payload);
            Some(payload)
        };
        let entry = DynamicMaskCacheEntry {
            hash,
            state,
            mask: payload,
        };
        let slot = if cache.entries.len() < max_entries {
            let slot = cache.entries.len();
            cache.entries.push(Some(entry));
            slot
        } else {
            let slot = cache.next_slot % max_entries;
            cache.next_slot = (slot + 1) % max_entries;
            if let Some(previous) = cache.entries[slot].take() {
                let mut remove_hash = false;
                if let Some(slots) = cache.by_hash.get_mut(&previous.hash) {
                    if let Some(index) = slots.iter().position(|&candidate| candidate == slot) {
                        slots.swap_remove(index);
                    }
                    remove_hash = slots.is_empty();
                }
                if remove_hash {
                    cache.by_hash.remove(&previous.hash);
                }
            }
            cache.entries[slot] = Some(entry);
            slot
        };
        cache.by_hash.entry(hash).or_default().push(slot);
        if cache.entries.len() == max_entries && cache.next_slot >= max_entries {
            cache.next_slot = 0;
        }
    }
}


impl DynamicMaskVocab {
    fn to_artifact_impl(&self, include_mask_quotient: bool) -> Option<DynamicMaskVocabArtifact> {
        let token_aliases = match &self.alias_state {
            VocabAliasState::Transfer(aliases) => DynamicMaskAliasStore::Ordered(Arc::clone(aliases)),
            VocabAliasState::Runtime(indexes) => DynamicMaskAliasStore::Flat {
                offsets: Arc::clone(&indexes.canonical_original_token_offsets),
                originals: Arc::clone(&indexes.canonical_original_tokens),
            },
            _ => return None,
        };
        let mask_quotient = include_mask_quotient
            .then(|| self.mask_tokenizer_quotient_for_transfer())
            .flatten();
        Some(DynamicMaskVocabArtifact {
            trie: self.trie.as_ref().clone(),
            token_aliases,
            llg_slice_leftovers: self.llg_slice_leftovers.as_ref().clone(),
            mask_tokenizer: mask_quotient.as_ref().map(|(tokenizer, _)| tokenizer.clone()),
            full_to_mask_state: mask_quotient
                .map(|(_, full_to_mask_state)| full_to_mask_state)
                .unwrap_or_default(),
            grammar_quotiented: self.grammar_quotiented,
            source_vocab_digest: self.source_vocab_digest,
        })
    }

    pub(crate) fn to_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        self.to_artifact_impl(true)
    }

    /// Preserve the exact O2 execution coordinate while omitting accelerators
    /// determined solely by the caller's separately supplied model vocabulary.
    pub(crate) fn to_template_external_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        // A full-vocabulary walk has no grammar quotient to persist. Returning
        // None lets the reader reconstruct its pure vocabulary-derived data
        // from the already fingerprint-validated external Vocab. A serialized
        // identity/full cache has no quotient source digest and must not be
        // misidentified as a grammar quotient by the transfer reader.
        if !self.grammar_quotiented { return None; }
        let mut artifact = self.to_artifact_impl(true)?;
        if artifact.grammar_quotiented {
            artifact.llg_slice_leftovers.clear();
        }
        Some(artifact)
    }

    /// Serialize only vocabulary-derived runtime data. Constraint-specific
    /// mask-tokenizer quotients and projections are reconstructed after load.
    pub(crate) fn to_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        self.to_artifact_impl(false)
    }

    /// External-vocabulary transfer can omit model-vocabulary-only slice/master
    /// acceleration for grammar quotients. `load_with_vocab` reattaches the
    /// already-prepared full-vocabulary structure in the parent process.
    pub(crate) fn to_external_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        let mut artifact = self.to_artifact_impl(false)?;
        if artifact.grammar_quotiented {
            artifact.llg_slice_leftovers.clear();
        }
        Some(artifact)
    }

    pub(crate) fn from_artifact(artifact: DynamicMaskVocabArtifact) -> Result<Self, String> {
        Self::from_artifact_impl(artifact, None)
    }

    pub(crate) fn from_external_vocab_artifact(
        artifact: DynamicMaskVocabArtifact,
        full_vocab_template: &DynamicMaskVocab,
    ) -> Result<Self, String> {
        Self::from_artifact_impl(artifact, Some(full_vocab_template))
    }

    fn from_artifact_impl(
        artifact: DynamicMaskVocabArtifact,
        full_vocab_template: Option<&DynamicMaskVocab>,
    ) -> Result<Self, String> {
        if artifact.trie.nodes.is_empty() {
            return Err("dynamic-mask vocabulary artifact has no trie root".to_owned());
        }
        let (offsets, originals) = Self::flatten_canonical_original_tokens(&artifact.token_aliases);
        if offsets.first() != Some(&0)
            || offsets.last().copied().map(|n| n as usize) != Some(originals.len())
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || originals.iter().any(|&id| id > crate::MAX_TOKEN_ID)
        {
            return Err("dynamic-mask vocabulary artifact has invalid alias rows".into());
        }
        let mut total = 0u32;
        for &canonical in artifact.trie.all_subtree_tokens() {
            let index = canonical as usize;
            let Some(bounds) = offsets.get(index..index + 2) else {
                return Err("dynamic-mask vocabulary artifact references an invalid alias row".into());
            };
            total = total.checked_add(bounds[1] - bounds[0])
                .ok_or("dynamic-mask vocabulary artifact alias length overflow")?;
        }
        let mut llg_slice_leftovers = artifact.llg_slice_leftovers;
        let external_acceleration = artifact.grammar_quotiented
            && llg_slice_leftovers.is_empty()
            && full_vocab_template.is_some();
        let mut result = Self::from_alias_store(
            Arc::new(artifact.trie),
            DynamicMaskAliasStore::Flat { offsets, originals },
        );
        if external_acceleration {
            result.inherit_llg_vocab_acceleration_without_master_from(
                full_vocab_template.expect("external acceleration template checked above"),
            );
        } else {
            let owner = Arc::clone(&result.indexes().aliases);
            for slice in &mut llg_slice_leftovers {
                Arc::make_mut(slice).try_share_aliases(&owner);
            }
            result.llg_slice_leftovers = Arc::new(llg_slice_leftovers);
            result.rebuild_llg_master_admitted_words_from_master_trie();
        }
        result.grammar_quotiented = artifact.grammar_quotiented;
        result.source_vocab_digest = artifact.source_vocab_digest;
        match artifact.mask_tokenizer {
            Some(tokenizer) => {
                if artifact.full_to_mask_state.is_empty()
                    || artifact
                        .full_to_mask_state
                        .iter()
                        .any(|&state| state >= tokenizer.num_states())
                {
                    return Err(
                        "dynamic-mask vocabulary artifact has an invalid mask-tokenizer quotient"
                            .to_owned(),
                    );
                }
                result.set_mask_tokenizer_quotient(tokenizer, artifact.full_to_mask_state);
            }
            None if !artifact.full_to_mask_state.is_empty() => {
                return Err(
                    "dynamic-mask vocabulary artifact has a quotient map without a tokenizer"
                        .to_owned(),
                );
            }
            None => {}
        }
        Ok(result)
    }
}

impl DynamicMaskVocab {
    pub(crate) fn restore_root_layout_metadata_from_token_bytes(
        &mut self,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) {
        let mut canonical_meta =
            Vec::<Option<(u16, bool)>>::with_capacity(self.canonical_token_count());
        for canonical in 0..self.canonical_token_count() as u32 {
            let Some(originals) = self.token_ids(canonical) else {
                return;
            };
            let Some(first_original) = originals.first() else {
                return;
            };
            let Some(bytes) = token_bytes.get(first_original).map(Vec::as_slice) else {
                return;
            };
            canonical_meta.push((!bytes.is_empty()).then(|| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    std::str::from_utf8(bytes).is_ok(),
                )
            }));
        }

        let mut root_layout_classes = Vec::with_capacity(self.trie.children(0).len());
        let mut root_layout_all_valid_utf8 = Vec::with_capacity(self.trie.children(0).len());
        for edge in self.trie.children(0) {
            let mut class = None::<u16>;
            let mut all_valid_utf8 = true;
            let mut saw_token = false;
            for &canonical in self.trie.subtree_tokens(edge.child) {
                let Some(Some((token_class, valid_utf8))) = canonical_meta.get(canonical as usize)
                else {
                    return;
                };
                if class.is_some_and(|existing| existing != *token_class) {
                    return;
                }
                class = Some(*token_class);
                all_valid_utf8 &= *valid_utf8;
                saw_token = true;
            }
            let Some(class) = class.filter(|_| saw_token) else {
                return;
            };
            root_layout_classes.push(class);
            root_layout_all_valid_utf8.push(all_valid_utf8);
        }

        let trie = Arc::make_mut(&mut self.trie);
        trie.root_layout_classes = root_layout_classes;
        trie.root_layout_all_valid_utf8 = root_layout_all_valid_utf8;
    }

    /// Verify that this vocabulary-only runtime index represents exactly the
    /// supplied original token-id -> byte mapping. This is used when loading a
    /// self-contained dynamic artifact: the persisted trie is an accelerator,
    /// never an independent source of vocabulary semantics.
    pub(crate) fn matches_token_bytes_exact(&self, token_bytes: &BTreeMap<u32, Vec<u8>>) -> bool {
        if !self.has_runtime_indexes() { return false; }
        if self.grammar_quotiented {
            return self.matches_grammar_quotiented_token_bytes(token_bytes);
        }
        if self.indexes().canonical_original_tokens.len() != token_bytes.len() {
            return false;
        }

        // Ordinary dynamic vocabularies canonicalize only byte-identical model
        // tokens. Keep the historical strong validation for those artifacts.
        let mut sorted_tokens = token_bytes
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
            .collect::<Vec<_>>();
        let sort_tokens = |left: &(u32, &[u8]), right: &(u32, &[u8])| {
            left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0))
        };
        if rayon::current_num_threads() == 1 {
            sorted_tokens.sort_unstable_by(sort_tokens);
        } else {
            sorted_tokens.par_sort_unstable_by(sort_tokens);
        }
        let mut canonical_bytes = Vec::<&[u8]>::with_capacity(self.canonical_token_count());
        let mut start = 0usize;
        while start < sorted_tokens.len() {
            let bytes = sorted_tokens[start].1;
            let mut end = start + 1;
            while end < sorted_tokens.len() && sorted_tokens[end].1 == bytes {
                end += 1;
            }
            let canonical = canonical_bytes.len() as u32;
            let Some(originals) = self.token_ids(canonical) else {
                return false;
            };
            if originals.len() != end - start
                || !originals
                    .iter()
                    .copied()
                    .eq(sorted_tokens[start..end].iter().map(|(token_id, _)| *token_id))
            {
                return false;
            }
            canonical_bytes.push(bytes);
            start = end;
        }
        if canonical_bytes.len() != self.canonical_token_count() {
            return false;
        }

        let mut canonical_seen = vec![false; canonical_bytes.len()];
        if !self.trie.all_token_paths_match(|canonical, prefix| {
            let canonical = canonical as usize;
            if canonical >= canonical_seen.len() || canonical_seen[canonical]
                || canonical_bytes[canonical] != prefix {
                return false;
            }
            canonical_seen[canonical] = true;
            true
        }) { return false; }

        canonical_seen.into_iter().all(|seen| seen)
    }

    fn matches_grammar_quotiented_token_bytes(
        &self,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) -> bool {
        if self.indexes().canonical_original_tokens.len() != token_bytes.len() {
            return false;
        }
        let mut covered = FxHashSet::<u32>::default();
        for &token_id in self.indexes().canonical_original_tokens.iter() {
            if !token_bytes.contains_key(&token_id) || !covered.insert(token_id) {
                return false;
            }
        }
        if covered.len() != token_bytes.len() {
            return false;
        }

        let mut canonical_seen = vec![false; self.canonical_token_count()];
        if !self.trie.all_token_paths_match(|canonical, prefix| {
            let index = canonical as usize;
            if index >= canonical_seen.len() || canonical_seen[index] { return false; }
            let Some(representative) = self.token_ids(canonical).and_then(|ids| ids.first()) else {
                return false;
            };
            if token_bytes.get(representative).is_none_or(|bytes| bytes.as_slice() != prefix) {
                return false;
            }
            canonical_seen[index] = true;
            true
        }) { return false; }

        canonical_seen.into_iter().all(|seen| seen)
    }
}

impl Default for DynamicMaskVocab {
    fn default() -> Self {
        static EMPTY_TRIE: std::sync::OnceLock<Arc<DynamicMaskTrie>> = std::sync::OnceLock::new();
        Self {
            trie: Arc::clone(EMPTY_TRIE.get_or_init(|| Arc::new(DynamicMaskTrie::new()))),
            alias_state: VocabAliasState::Empty,
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admission: None,
            llg_master_residual_ops: Arc::new(Vec::new()),
            prepared_master_proofs: None,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            terminal_quotients: ProjectedTerminalInventory::default(),
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_projection: None,
            mask_tokenizer_fast_transitions: None,
        }
    }
}

/// One authoritative bound vocabulary. Deferred prepared bodies have no
/// vocabulary until their external artifact is reloaded with a supplied
/// model vocabulary. Bound empty vocabularies remain distinguishable.
#[derive(Debug, Clone, Default)]
pub(crate) enum TokenByteDomain {
    #[default]
    Unbound,
    Deferred([u8; 32]),
    Bound(crate::Vocab),
}

impl TokenByteDomain {
    pub(crate) fn from_vocab(vocab: &crate::Vocab) -> Self {
        Self::Bound(vocab.clone())
    }

    pub(crate) fn vocab(&self) -> Option<&crate::Vocab> {
        match self {
            Self::Bound(vocab) => Some(vocab),
            _ => None,
        }
    }

    pub(crate) fn entries(&self) -> &crate::vocab::VocabEntries {
        static EMPTY: OnceLock<crate::vocab::VocabEntries> = OnceLock::new();
        match self.vocab() {
            Some(vocab) => vocab.entries_map(),
            None => EMPTY.get_or_init(Default::default),
        }
    }

    pub(crate) fn matches_vocab(&self, vocab: &crate::Vocab) -> bool {
        self.vocab().is_some_and(|bound| bound.same_model_vocab(vocab))
    }

    pub(crate) fn bind_exact(&mut self, vocab: &crate::Vocab) -> bool {
        if !self.matches_vocab(vocab) { return false; }
        *self = Self::from_vocab(vocab);
        true
    }

    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<&[u8]> {
        self.vocab()?.get(id)
    }

    pub(crate) fn len(&self) -> usize { self.vocab().map_or(0, crate::Vocab::len) }
    pub(crate) fn is_empty(&self) -> bool { self.len() == 0 }

    pub(crate) fn max_token_id(&self) -> Option<u32> {
        self.vocab()?.max_event_token_id()
    }

    pub(crate) fn max_byte_len(&self) -> usize {
        self.vocab().map_or(0, crate::Vocab::max_token_byte_len)
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (u32, &[u8])> + '_ {
        self.entries().iter().map(|(&id, bytes)| (id, bytes))
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub(crate) enum ConstraintRuntimeBackend {
    #[default]
    Static,
    Dynamic,
}

pub(crate) type SegmentedParserLink = ScopedSubgrammarLink;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveParserLeafLayout {
    pub(crate) state_offset: u32,
    pub(crate) state_count: u32,
    /// Immediate wrapper of the composition whose layout was requested. Inner
    /// leaf changes within one nested component deliberately keep this owner.
    pub(crate) top_component: u32,
    /// Immediate-component path from the requested composition root to this
    /// intact LR table.
    pub(crate) component_path: Vec<u32>,
}

const RECURSIVE_VIRTUAL_TOKENIZER_STATE_LIMIT: u32 = 1 << 31;

#[derive(Debug, Default)]
struct RecursiveVirtualTokenizerStateStore {
    next_state: u32,
    scoped_by_local: FxHashMap<(u32, u32), u32>,
    local_by_scoped: FxHashMap<u32, (u32, u32)>,
}

#[derive(Debug, Default)]
pub(crate) struct RecursiveVirtualTokenizerStates {
    store: Mutex<RecursiveVirtualTokenizerStateStore>,
}

impl RecursiveVirtualTokenizerStates {
    pub(crate) fn scoped_state(
        &self,
        physical_state_count: u32,
        leaf_index: usize,
        local_state: u32,
    ) -> Option<u32> {
        let leaf_index = u32::try_from(leaf_index).ok()?;
        let mut store = self.store.lock().ok()?;
        if let Some(&scoped) = store.scoped_by_local.get(&(leaf_index, local_state)) {
            return Some(scoped);
        }
        if store.next_state < physical_state_count {
            store.next_state = physical_state_count;
        }
        if store.next_state >= RECURSIVE_VIRTUAL_TOKENIZER_STATE_LIMIT {
            return None;
        }
        let scoped = store.next_state;
        store.next_state = store.next_state.checked_add(1)?;
        store.scoped_by_local.insert((leaf_index, local_state), scoped);
        store.local_by_scoped.insert(scoped, (leaf_index, local_state));
        Some(scoped)
    }

    pub(crate) fn local_state(&self, scoped_state: u32) -> Option<(usize, u32)> {
        let store = self.store.lock().ok()?;
        let &(leaf_index, local_state) = store.local_by_scoped.get(&scoped_state)?;
        Some((leaf_index as usize, local_state))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveParserLayout {
    pub(crate) component_offsets: Vec<u32>,
    pub(crate) leaves: Vec<RecursiveParserLeafLayout>,
    /// Derived once so ordinary recursive commits do not scan every leaf
    /// merely to discover that no exact residual-liveness query is needed.
    pub(crate) has_virtual_residuals: bool,
    pub(crate) leaf_state_offsets: Vec<u32>,
    /// Disjoint-union tokenizer-state coordinate over the same intact leaves.
    /// Parser and tokenizer leaves have identical ordering/component paths, but
    /// independent offsets because their local state counts differ.
    pub(crate) leaf_tokenizer_state_offsets: Vec<u32>,
    pub(crate) total_tokenizer_states: u32,
    /// Disjoint-union terminal coordinate over the same intact leaves. Runtime
    /// byte terminals are encoded after the outer/materialized terminal range,
    /// so the `u32` is self-describing while byte scanning no longer needs a
    /// leaf-local -> outer -> leaf-local round trip.
    pub(crate) leaf_terminal_offsets: Vec<u32>,
    pub(crate) total_leaf_terminals: u32,
    /// Size of the transitional outer/global terminal prefix at layout-build
    /// time. Live recursive terminal IDs use this immutable tag boundary and
    /// therefore do not need to consult the materialized outer table again.
    pub(crate) outer_terminal_count: u32,
    /// Lazily mapped future-terminal support in the live runtime terminal
    /// coordinate for each scoped leaf tokenizer state.
    pub(crate) tokenizer_future_scoped: Vec<OnceLock<BitSet>>,
    /// Linker controls rewritten only into the leaf-component coordinate.
    pub(crate) links: Vec<SegmentedParserLink>,
    /// Outer/global terminal -> intact leaf-local terminals. Ordinary byte
    /// commit no longer consumes this relation; it remains for compiler-side
    /// static-B materialization, compatibility/reference evaluation, and outer
    /// special-token routing.
    pub(crate) terminal_targets: Vec<SmallVec<[(u32, TerminalID); 4]>>,
    pub(crate) total_states: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct StaticDynamicOverlayMetadata {
    /// Exact segmented parser backend. Each retained source constraint keeps
    /// its own parser/token coordinate and is projected from the composed
    /// tokenizer/parser coordinates at mask time.
    #[serde(skip, default)]
    pub(crate) segmented_parser_components: Vec<SegmentedParserComponent>,
    /// Explicit local-coordinate calls between the wrappers above. This is the
    /// semantic relation consumed by the compact parser-action provider; it is
    /// intentionally scoped to this composition level only.
    #[serde(skip, default)]
    pub(crate) segmented_parser_links: Vec<SegmentedParserLink>,
    /// Runtime-derived endpoint parser view. The semantic component tree stays
    /// literal; this cache contains only the flat leaf coordinate required by
    /// the existing `LeveledGSS<u32, ...>` action provider.
    #[serde(skip, default)]
    pub(crate) recursive_parser_layout: OnceLock<Arc<RecursiveParserLayout>>,
    /// Exact endpoint lexical relation: recursive leaf tokenizer state -> this
    /// constraint's internal TSID set. Runtime-product components can make the
    /// image genuinely set-valued, so this must not be collapsed to one TSID.
    #[serde(skip, default)]
    pub(crate) recursive_tokenizer_internal_tsids: OnceLock<Arc<Vec<Vec<u32>>>>,
    /// Exact finite observation coordinate for virtual component lexers. Kept
    /// separate from component A's TSID quotient and from exact runtime IDs.
    #[serde(skip, default)]
    pub(crate) recursive_static_observation: Option<Arc<super::static_observation::RecursiveStaticObservation>>,
    /// Lazily allocated outer scoped IDs for exact virtual tokenizer states in
    /// retained recursive leaves. Physical states keep their contiguous layout
    /// IDs; only actually reached virtual states enter this runtime-only map.
    #[serde(skip, default)]
    pub(crate) recursive_virtual_tokenizer_states: Arc<RecursiveVirtualTokenizerStates>,
}

impl StaticDynamicOverlayMetadata {
    pub(crate) fn boundary_shards(&self) -> impl Iterator<Item = &SegmentedBoundaryShard> {
        self.segmented_parser_components.iter().filter_map(|component| component.boundary.as_ref())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedBoundaryShard {
    /// Optional conservative model-token summary for the first internal
    /// component crossing. `None` means no trigger information is available;
    /// it must never be interpreted as "no crossings".
    pub(crate) candidate_tokens: Option<Arc<[u32]>>,
    /// Runtime-only immutable vocabulary scoped to this binding. Never key a
    /// global cache by model IDs alone: IDs can spell different bytes and LR
    /// coordinates can differ in another binding. Rebuilt on artifact load.
    pub(crate) mask_vocabulary: Arc<OnceLock<Result<super::dynamic_mask::PreparedMaskVocabulary, String>>>,
    pub(crate) backend: SegmentedBoundaryShardBackend,
}

#[derive(Debug, Clone)]
pub(crate) enum SegmentedBoundaryShardBackend {
    StaticParser(Arc<SegmentedBoundaryParser>),
    /// Exact dynamic crossing with no required composition-specific B artifact.
    /// Uses the scoped shared-prefix vocabulary walker.
    DynamicDirect,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedParserComponent {
    pub(crate) constraint: Arc<Constraint>,
    /// Composition-specific crossing backend for model tokens that start in
    /// this component. This lives on the wrapper, not on the reusable
    /// underlying constraint.
    pub(crate) boundary: Option<SegmentedBoundaryShard>,
    pub(crate) tokenizer_state_offset: u32,
    pub(crate) terminal_offset: u32,
    /// Outer composed terminals which are canonical aliases for a local
    /// terminal of this component. Direct interval mapping remains implicit;
    /// this stores only exceptional many-scope aliases (currently globally
    /// equivalent ignore terminals).
    pub(crate) global_terminal_aliases: Vec<(u32, u32)>,
    /// Component-local Static TSID -> composed Static TSID relation retained
    /// for lazy exact tokenizer states that are allocated after link time.
    pub(crate) local_tsid_to_global_tsids: Vec<Vec<u32>>,
}


#[derive(Clone, Copy)]
pub(crate) struct SegmentedParserComponentTables<'a> {
    components: &'a [SegmentedParserComponent],
}

impl<'a> SegmentedParserComponentTables<'a> {
    #[inline]
    pub(crate) fn new(components: &'a [SegmentedParserComponent]) -> Self {
        Self { components }
    }
}

impl ParserComponentTableSource for SegmentedParserComponentTables<'_> {
    #[inline]
    fn component_count(&self) -> usize {
        self.components.len()
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.components
            .get(component as usize)
            .map(|component| &*component.constraint.table)
    }

    #[inline]
    fn component_ignore_terminal(&self, component: u32) -> Option<u32> {
        let component = self.components.get(component as usize)?;
        // A composed component table already owns its scoped ignores in its
        // rows (`skip_terminals` provenance); provider-level Identity would
        // shadow those rows. Only raw component tables need it here.
        let ignore = component.constraint.ignore_terminal?;
        (!component.constraint.table.skip_terminals.contains(&ignore)).then_some(ignore)
    }
}

/// Scanner coordinate used by a boundary parser's weights.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum BoundaryTsidCoordinate {
    Composed,
    /// Leaf-packed tokenizer states, without the artificial merged reset state.
    ShardLocal(Vec<u32>),
}

impl BoundaryTsidCoordinate {
    pub(crate) fn is_composed(&self) -> bool {
        matches!(self, Self::Composed)
    }

    pub(crate) fn private_map(&self) -> &[u32] {
        match self {
            Self::Composed => &[],
            Self::ShardLocal(map) => map,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryParser {
    /// Exact deterministic boundary parser in the recursive leaf parser-state coordinate.
    pub(crate) parser_dwa: DWA,
    pub(crate) coordinate: BoundaryTsidCoordinate,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub(crate) enum DeferredTerminalExprBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    /// Dynamic worker-transfer artifacts keep the terminal-expression section
    /// compressed until a later composition actually asks for source
    /// expressions. Ordinary mask/commit execution never needs these trees.
    CompressedOwned(Arc<[u8]>),
    /// Same compressed representation, retained directly inside an owned
    /// sectioned transfer backing allocation.
    CompressedBacked {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl DeferredTerminalExprBytes {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
            Self::CompressedOwned(bytes) => bytes,
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }

    pub(crate) fn decode_exprs(&self) -> Result<Vec<Expr>, String> {
        let raw;
        let bytes = match self {
            Self::Owned(bytes) => bytes.as_ref(),
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
            Self::CompressedOwned(bytes) => {
                raw = zstd::stream::decode_all(bytes.as_ref()).map_err(|err| err.to_string())?;
                raw.as_slice()
            }
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => {
                raw = zstd::stream::decode_all(&backing[*start..*start + *len])
                    .map_err(|err| err.to_string())?;
                raw.as_slice()
            }
        };
        bincode::deserialize(bytes).map_err(|err| err.to_string())
    }

    /// Append the canonical uncompressed bincode expression section used by
    /// the self-contained Constraint artifact format. A transfer-loaded
    /// compressed blob therefore remains lazy until either composition or a
    /// later explicit save requests it.
    pub(crate) fn append_raw_serialized(&self, out: &mut Vec<u8>) -> Result<(), String> {
        match self {
            Self::Owned(bytes) => out.extend_from_slice(bytes),
            Self::Backed {
                backing,
                start,
                len,
            } => out.extend_from_slice(&backing[*start..*start + *len]),
            Self::CompressedOwned(bytes) => {
                zstd::stream::copy_decode(bytes.as_ref(), out).map_err(|err| err.to_string())?;
            }
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => {
                zstd::stream::copy_decode(&backing[*start..*start + *len], out)
                    .map_err(|err| err.to_string())?;
            }
        }
        Ok(())
    }
}

/// Opaque current-format composition metadata retained without eagerly
/// rebuilding the large parser-template/characterization graphs. Ordinary
/// runtime masking never needs these bytes; constraint composition materializes
/// them on demand.
#[derive(Debug, Clone)]
pub(crate) enum DeferredCompositionMetadataBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl DeferredCompositionMetadataBytes {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }
}

/// Compact persisted metadata for an unresolved external-grammar slot.
///
/// The terminal ID is an internal linker coordinate, never a public/model
/// token ID. Keeping this record beside the compiled parser artifact lets a
/// loaded parent be linked without parsing or recompiling its source grammar.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LateGrammarSlot {
    pub(crate) name: String,
    pub(crate) terminal_id: TerminalID,
}

/// Internal runtime handle.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ConstraintInner {
    pub(crate) data: Arc<ConstraintData>,
    pub(crate) cached_mask_bit_len: OnceLock<usize>,
}

pub(crate) type Constraint = ConstraintInner;

/// Semantic language data shared by all newly compiled runtime forms.
pub(crate) struct RuntimeLanguage {
    pub table: super::parser_backend::ParserExecutable,
    pub tokenizer: Arc<Tokenizer>,
    pub terminal_display_names: Vec<String>,
    pub direct_regular_automaton: Option<DirectRegularAutomaton>,
    pub ignore_terminal: Option<TerminalID>,
    pub ignore_expr: Option<Expr>,
    pub special_token_terminals: Vec<SpecialTokenTerminal>,
}

/// The mask program and the coordinates in which its weights are expressed.
/// A linked coordinator has no global weighted program; its leaves own theirs.
pub(crate) enum CompiledMask {
    Weighted {
        parser_dwa: DWA,
        non_dwa_weights: NonDwaWeights<Weight>,
        possible_matches_complete: bool,
        tokenizer_tsids: super::TokenizerTsidMap,
        token_classes: super::TokenClasses,
        composition_grammar_summary: Option<CompositionGrammarSummary>,
    },
    Dynamic,
    Linked,
}

impl Constraint {
    pub(crate) fn from_compiler_parts(
        language: RuntimeLanguage,
        vocab: &crate::Vocab,
        mask: CompiledMask,
        dynamic_mask_vocab: DynamicMaskVocab,
        tokenizer_fast_transitions: FastTokenizerTransitions,
    ) -> Self {
        use super::parser_backend::ParserExecutable;
        let (runtime_backend, parser_dwa, non_dwa_weights, possible_matches_complete,
            tokenizer_tsids, token_classes, composition_grammar_summary,
            tokenizer_has_epsilon_transitions) = match mask {
            CompiledMask::Weighted {
                parser_dwa, non_dwa_weights, possible_matches_complete,
                tokenizer_tsids, token_classes, composition_grammar_summary,
            } => {
                assert!(matches!(&language.table, ParserExecutable::Native(_)));
                (ConstraintRuntimeBackend::Static, ParserDwaStorage::Owned(parser_dwa),
                    NonDwaWeightStorage::Materialized(non_dwa_weights), possible_matches_complete,
                    tokenizer_tsids, token_classes, composition_grammar_summary,
                    false)
            }
            CompiledMask::Dynamic => {
                assert!(matches!(&language.table, ParserExecutable::Native(_) | ParserExecutable::ExplicitLr(_)));
                (ConstraintRuntimeBackend::Dynamic, ParserDwaStorage::Absent, Default::default(),
                    false, Default::default(), Default::default(), None,
                    false)
            }
            CompiledMask::Linked => {
                assert!(matches!(&language.table, ParserExecutable::LinkedLr(_)));
                (ConstraintRuntimeBackend::Static, ParserDwaStorage::Absent, Default::default(),
                    true, super::TokenizerTsidMap::partition(vec![0], vec![vec![0]]),
                    super::TokenClasses::mapped(Vec::new(), Vec::new()), None,
                    language.tokenizer.has_epsilon_transitions())
            }
        };
        Self::from_data(ConstraintData {
            end_tokens: Arc::from([]),
            marker_queries: None,
            runtime_backend,
            static_dynamic_overlay: None,
            boundary_trigger: BoundaryTrigger::None,
            boundary_candidate_summary: OnceLock::new(),
            late_grammar_slots: Vec::new(),
            parser_dwa,
            non_dwa_weights,
            direct_regular_wide_frontier_acceptance: Vec::new(),
            direct_regular_dynamic_hot_frontiers: Vec::new(),
            direct_regular_parser_state_acceptance: Vec::new(),
            direct_regular_automaton: language.direct_regular_automaton,
            table: language.table,
            terminal_display_names: language.terminal_display_names,
            tokenizer: language.tokenizer,
            tokenizer_has_epsilon_transitions,
            ignore_terminal: language.ignore_terminal,
            special_token_terminals: language.special_token_terminals,
            dynamic_mask_vocab,
            lazy_dynamic_mask_vocab: OnceLock::new(),
            possible_matches_complete,
            tokenizer_tsids,
            composition_grammar_summary,
            token_classes,
            token_bytes: TokenByteDomain::from_vocab(vocab),
            token_projection: Default::default(),
            internal_token_dense_words: 0,
            weight_token_masks: Default::default(),
            packed_dwa_token_dense_masks: Default::default(),
            seed_terminal_dense: Default::default(),
            seed_terminal_dense_fallback: Default::default(),
            seed_universe_dense: Arc::from([]),
            dwa_fast_transitions: Default::default(),
            parser_runtime_caches_prebuilt: false,
            indexed_dag_dense_transitions: Vec::new(),
            indexed_dag_dense_finals: Vec::new(),
            tokenizer_fast_transitions,
            parser_state_domain_labels: Vec::new(),
            ignore_expr: language.ignore_expr,
            serialized_artifact_cache: None,
            deferred_terminal_exprs_blob: None,
            deferred_terminal_exprs: Default::default(),
            deferred_composition_metadata_blob: None,
            deferred_table_rules_blob: None,
            deferred_table_rules: Default::default(),
        })
    }
}

impl Constraint {
    pub(crate) fn from_data(data: ConstraintData) -> Self {
        Self { data: Arc::new(data), cached_mask_bit_len: OnceLock::new() }
    }

    pub(crate) fn as_data(&self) -> &ConstraintData {
        &self.data
    }
}

impl std::ops::Deref for Constraint {
    type Target = ConstraintData;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for Constraint {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Internal compilation, linking and loading may change the token domain.
        // Invalidate this handle only; other handles retain their immutable data.
        self.cached_mask_bit_len.take();
        Arc::make_mut(&mut self.data)
    }
}

// Declare the runtime layout and private remote Serde schema together.
// Only the remote definition receives serialization attributes: Constraint
// and its public Deref target retain their existing non-Serde API.
macro_rules! constraint_data_schema {
    ($(
        $(#[$doc:meta])*
        $(@serde($($wire:tt)*))?
        $vis:vis $name:ident: $ty:ty,
    )*) => {
        /// Inner compiled data shared by handles. Hidden from the documented API;
        /// its visibility allows the public Deref target to be named by rustc.
        #[doc(hidden)]
        #[derive(Debug, Clone)]
        pub struct ConstraintData {
            $(
                $(#[$doc])*
                $vis $name: $ty,
            )*
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(remote = "ConstraintData")]
        pub(crate) struct ConstraintSerde {
            $(
                $(#[serde($($wire)*)])?
                $vis $name: $ty,
            )*
        }
    };
}

constraint_data_schema! {
    /// Final generation policy, not part of the embeddable compiled body.
    /// The outer root artifact stores these separately from compiler data.
    @serde(skip, default)
    pub(crate) end_tokens: Arc<[u32]>,
    @serde(skip, default)
    pub(crate) marker_queries: Option<Arc<crate::runtime::markers::MarkerQueries>>,
    @serde(default)
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    @serde(skip, default)
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    /// Reusable component-local trigger metadata for dynamic composition.
    /// Ordinary static/dynamic compilation leaves this at `None` so there is
    /// no trigger construction cost unless explicitly requested in the future.
    @serde(skip, default)
    pub(crate) boundary_trigger: BoundaryTrigger,
    /// Grammar-aware proper-prefix model-token summary used only by static
    /// composition boundary preparation.  It is intentionally independent of
    /// `boundary_trigger`: the latter is a dynamic-runtime accelerator with a
    /// different observation contract.  The lock permits lazy preparation by
    /// composition without mutating the immutable constraint API.
    @serde(skip, default)
    pub(crate) boundary_candidate_summary: OnceLock<BoundaryCandidateSummary>,
    /// Named, compiler-generated linker terminals for unresolved
    /// `extern grammar` declarations. The token IDs backing these terminals
    /// are deliberately private and outside the model vocabulary; callers
    /// address slots only by `name` through the late-binding API.
    @serde(skip, default)
    pub(crate) late_grammar_slots: Vec<LateGrammarSlot>,
    /// Exact parser DWA; serialized independently in the native DWA section.
    @serde(skip, default)
    pub(crate) parser_dwa: ParserDwaStorage,
    /// Labeled non-DWA weights. The current core carries the same relation
    /// schema using validated pool IDs, not this runtime representation.
    @serde(skip, default)
    pub(crate) non_dwa_weights: NonDwaWeightStorage,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    @serde(skip, default)
    pub(crate) direct_regular_wide_frontier_acceptance:
        Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    @serde(skip, default)
    pub(crate) direct_regular_dynamic_hot_frontiers:
        Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    @serde(skip, default)
    pub(crate) direct_regular_parser_state_acceptance:
        Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    @serde(default)
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    @serde(with = "crate::runtime::parser_backend::table_serde")
    pub(crate) table: crate::runtime::parser_backend::ParserExecutable,
    @serde(default)
    pub(crate) terminal_display_names: Vec<String>,
    @serde(with = "crate::runtime::artifact::immutable_tokenizer_serde")
    pub(crate) tokenizer: Arc<Tokenizer>,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    @serde(skip, default)
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    @serde(default)
    pub(crate) ignore_terminal: Option<TerminalID>,
    @serde(default)
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    @serde(skip, default)
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    @serde(skip, default)
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    @serde(default)
    pub(crate) possible_matches_complete: bool,
    pub(crate) tokenizer_tsids: crate::runtime::TokenizerTsidMap,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    @serde(skip, default)
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    @serde(skip, default)
    pub(crate) token_classes: crate::runtime::TokenClasses,
    @serde(skip, default)
    pub(crate) token_bytes: TokenByteDomain,
    /// Canonical original-token output projection.
    @serde(skip, default)
    pub(crate) token_projection: super::constraint::TokenProjection,
    @serde(skip)
    pub(crate) internal_token_dense_words: usize,
    @serde(skip)
    pub(crate) weight_token_masks: WeightTokenMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    @serde(skip, default)
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    @serde(skip)
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    @serde(skip, default)
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    @serde(skip, default = "empty_dense_words")
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    @serde(skip)
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    @serde(skip, default)
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    @serde(skip, default)
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    @serde(skip, default)
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    @serde(skip)
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    @serde(skip, default)
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    @serde(skip, default)
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    @serde(skip, default)
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    @serde(skip, default)
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    @serde(skip, default)
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Cold composition metadata: grammar adjacency, trigger and boundary
    /// candidate proof. Materialized only on an owned link input.
    @serde(skip, default)
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    @serde(skip, default)
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    @serde(skip, default)
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}


#[cfg(test)]
mod dynamic_mask_vocab_cache_boundary_tests {
    use super::*;

    #[test]
    fn full_walk_flat32_covers_tokenizers_beyond_flat16_state_space() {
        const TARGET: u32 = 32_768;
        let tokenizer = crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();

        assert!(FastTokenizerTransitions::flat16_for(&tokenizer).is_none());
        let flat32 = FastTokenizerTransitions::flat32_for(&tokenizer)
            .expect("32769-state tokenizer must fit the Flat32 full-walk coordinate");
        assert_eq!(flat32.len(), TARGET as usize + 1);
        assert_eq!(flat32.transition(&tokenizer, 0, b'a'), TARGET);
        assert_eq!(flat32.transition(&tokenizer, 0, b'b'), u32::MAX);
        let FastTokenizerTransitions::Flat32 { transitions, .. } = flat32 else {
            panic!("wide tokenizer unexpectedly used a non-Flat32 representation");
        };
        let encoded = transitions[b'a' as usize];
        assert_ne!(encoded & 0x8000_0000, 0, "finalizer bit was not encoded");
        assert_eq!(encoded & 0x7fff_ffff, TARGET);
    }

    #[test]
    fn packed_dwa_dense_mask_cache_rejects_malformed_flat_layouts() {
        assert!(PackedDwaDenseWeightMaskCache::from_flat(4, 2, vec![1], vec![7]).is_err());
        assert!(
            PackedDwaDenseWeightMaskCache::from_flat(4, 2, vec![1, 1], vec![1, 2, 3, 4])
                .is_err()
        );
        assert!(
            PackedDwaDenseWeightMaskCache::from_flat(2, 1, vec![2], vec![7]).is_err()
        );
    }

    #[test]
    fn vocab_only_artifact_rejects_missing_trie_root() {
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let mut artifact = vocab
            .to_vocab_artifact()
            .expect("initialized vocabulary should serialize");
        assert!(artifact.mask_tokenizer.is_none());
        assert!(artifact.full_to_mask_state.is_empty());
        artifact.trie.nodes.clear();
        let error = DynamicMaskVocab::from_artifact(artifact).unwrap_err();
        assert!(error.contains("no trie root"));
    }

    #[test]
    fn dense_mask_projection_union_lookup_matches_exact_source_subset_union() {
        let source =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let (built, full_to_mask_state) = source
            .try_full_determinization_all_starts(256, 16_384)
            .expect("small epsilon tokenizer should determinize from every raw start");
        let source_subsets = built.source_subsets.clone();
        let state_count = built.tokenizer.num_states();

        let mut vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        vocab.set_mask_tokenizer_quotient(built.tokenizer, full_to_mask_state);
        vocab.set_mask_tokenizer_source_subsets(source_subsets.clone());

        for left in 0..state_count {
            for right in 0..state_count {
                let mut union = source_subsets[left as usize].to_vec();
                union.extend_from_slice(&source_subsets[right as usize]);
                union.sort_unstable();
                union.dedup();
                let expected = source_subsets
                    .iter()
                    .position(|subset| subset.as_ref() == union.as_slice())
                    .map(|state| state as u32);
                assert_eq!(
                    vocab.mask_projection_state_for_projection_states(&[left, right]),
                    expected,
                    "projection-state union mismatch for ({left}, {right})",
                );
            }
        }
    }

    #[test]
    fn full_artifact_round_trips_dense_mask_tokenizer_quotient() {
        let source =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let (built, full_to_mask_state) = source
            .try_full_determinization_all_starts(256, 16_384)
            .expect("small epsilon tokenizer should determinize from every raw start");
        let expected_states = built.tokenizer.num_states();

        let mut vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        vocab.set_mask_tokenizer_quotient(built.tokenizer, full_to_mask_state.clone());

        let artifact = vocab.to_artifact().expect("full runtime artifact should serialize");
        assert_eq!(artifact.full_to_mask_state, full_to_mask_state);
        assert_eq!(
            artifact.mask_tokenizer.as_ref().map(Tokenizer::num_states),
            Some(expected_states),
        );

        let loaded = DynamicMaskVocab::from_artifact(artifact).unwrap();
        assert_eq!(loaded.mask_tokenizer_quotient_for_transfer().unwrap().1, full_to_mask_state);
        assert_eq!(
            loaded.mask_projection_tokenizer().map(Tokenizer::num_states),
            Some(expected_states),
        );
        assert!(matches!(
            loaded.mask_projection_fast_transitions(),
            Some(FastTokenizerTransitions::Flat16 { .. }),
        ));
    }

    #[test]
    fn vocab_artifact_restores_root_layout_metadata_from_token_bytes() {
        let token_bytes = BTreeMap::from([
            (0u32, b"abc".to_vec()),
            (1u32, "é".as_bytes().to_vec()),
        ]);
        let mut entries = token_bytes
            .iter()
            .enumerate()
            .map(|(canonical, (_, bytes))| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    canonical,
                    bytes.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
        let expected_classes = trie.root_layout_classes.clone();
        let expected_utf8 = trie.root_layout_all_valid_utf8.clone();
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(trie),
            Arc::new(vec![vec![0], vec![1]]),
        );
        let artifact = vocab.to_vocab_artifact().unwrap();
        let loaded = DynamicMaskVocab::from_artifact(artifact).unwrap();
        assert_eq!(loaded.trie.root_layout_classes, expected_classes);
        assert_eq!(loaded.trie.root_layout_all_valid_utf8, expected_utf8);
    }

    #[test]
    fn full_walk_all_consume_exactly_rejects_empty_edges() {
        fn one_edge(bytes: &[u8]) -> DynamicMaskTrie {
            let mut trie = DynamicMaskTrie::new();
            let child = trie.begin_node(Some(0), Some(0), bytes.len());
            trie.finish_node(child, bytes);
            trie.finish_node(0, &[]);
            trie.finalize_full_walk();
            trie
        }

        let nonempty = one_edge(b"abc");
        assert_eq!(nonempty.full_walk_ops().len(), 3);
        assert!(nonempty.full_walk_all_consume());

        let empty = one_edge(b"");
        assert_eq!(empty.full_walk_ops().len(), 1);
        assert!(!empty.full_walk_all_consume());
    }

    #[test]
    fn vocab_runtime_exactly_checks_original_token_bytes() {
        let trie = DynamicMaskTrie::from_sorted_token_refs(&[(0, 0, b"a")]);
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(trie),
            Arc::new(vec![vec![7]]),
        );
        assert!(vocab.matches_token_bytes_exact(&BTreeMap::from([(7, b"a".to_vec())])));
        assert!(!vocab.matches_token_bytes_exact(&BTreeMap::from([(7, b"b".to_vec())])));
    }

    #[test]
    fn lazy_union_cache_try_lock_is_nonblocking_when_in_use() {
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let _owner = vocab.lock_lazy_union_cache();
        assert!(vocab.try_lock_lazy_union_cache().is_none());
    }

    #[test]
    fn fresh_runtime_instance_shares_only_vocab_derived_data() {
        let template = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let fresh = template.fresh_runtime_instance();

        assert!(Arc::ptr_eq(&template.trie, &fresh.trie));
        let (VocabAliasState::Runtime(left), VocabAliasState::Runtime(right)) =
            (&template.alias_state, &fresh.alias_state) else {
                panic!("expected materialized vocabulary indexes")
            };
        assert!(Arc::ptr_eq(left, right));

        assert!(!Arc::ptr_eq(&template.mask_cache, &fresh.mask_cache));
        assert!(!Arc::ptr_eq(
            &template.direct_regular_frontier_cache,
            &fresh.direct_regular_frontier_cache,
        ));
        assert!(!Arc::ptr_eq(
            &template.direct_regular_wide_frontier_index_cache,
            &fresh.direct_regular_wide_frontier_index_cache,
        ));
        assert!(!Arc::ptr_eq(
            &template.direct_regular_terminal_support,
            &fresh.direct_regular_terminal_support,
        ));
        assert!(!Arc::ptr_eq(
            &template.self_loop_projections,
            &fresh.self_loop_projections,
        ));
    }
}

/// Arc ownership is a runtime/lifecycle detail; keep the existing tokenizer
/// artifact bytes unchanged. A retained observer can pin this same allocation,
/// and all subsequent mutation must use copy-on-write on the Constraint side.
pub(crate) mod immutable_tokenizer_serde {
    use super::{Arc, Tokenizer};
    use serde::{Deserializer, Serializer};
    pub(crate) fn serialize<S: Serializer>(tokenizer: &Arc<Tokenizer>, serializer: S) -> Result<S::Ok,S::Error> {
        crate::automata::lexer::tokenizer::artifact_serde::serialize(tokenizer.as_ref(), serializer)
    }
    pub(crate) fn deserialize<'de,D: Deserializer<'de>>(deserializer: D) -> Result<Arc<Tokenizer>,D::Error> {
        crate::automata::lexer::tokenizer::artifact_serde::deserialize(deserializer).map(Arc::new)
    }
}

#[cfg(test)]
mod immutable_tokenizer_tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice, plus};
    use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};
    use serde::{Deserialize, Serialize};
    #[derive(Serialize)]
    struct OwnedRef<'a>(#[serde(with = "crate::automata::lexer::tokenizer::artifact_serde")] &'a Tokenizer);
    #[derive(Serialize, Deserialize)]
    struct Shared(#[serde(with = "super::immutable_tokenizer_serde")] Arc<Tokenizer>);
    #[test]
    fn shared_tokenizer_retains_exact_owned_wire_and_byte_behavior() {
        let expressions=vec![choice(vec![bytes(b"abc"),bytes(b"abd")]),plus(bytes(b" "))];
        for tokenizer in [build_regex_monolithic(&expressions).into_tokenizer(2,None),build_regex_partitioned(&expressions,&[0,1]).into_tokenizer(2,None)] {
            let owned_wire=bincode::serialize(&OwnedRef(&tokenizer)).unwrap();let shared=Shared(Arc::new(tokenizer));
            let wire=bincode::serialize(&shared).unwrap();assert_eq!(owned_wire,wire);
            let loaded:Shared=bincode::deserialize(&wire).unwrap();assert_eq!(bincode::serialize(&loaded).unwrap(),owned_wire);
            for q in 0..shared.0.num_states(){for b in 0..=255u8{assert_eq!(shared.0.step_all(&[q],b),loaded.0.step_all(&[q],b));}}
        }
    }
    #[test]
    fn pinned_tokenizer_is_unchanged_after_copy_on_write_mutation() {
        let expressions=vec![Expr::Epsilon,bytes(b"a")];
        let mut owner=Arc::new(build_regex_monolithic(&expressions).into_tokenizer(2,None));let pinned=Arc::clone(&owner);
        let original=bincode::serialize(&OwnedRef(pinned.as_ref())).unwrap();assert!(Arc::ptr_eq(&owner,&pinned));
        let drained=Arc::make_mut(&mut owner).isolate_start_state_and_drain_nullable_terminals();
        assert!(drained.contains(&0),"fixture must exercise a real nullable-state mutation");assert!(!Arc::ptr_eq(&owner,&pinned));
        assert_eq!(bincode::serialize(&OwnedRef(pinned.as_ref())).unwrap(),original);
        assert_ne!(bincode::serialize(&OwnedRef(owner.as_ref())).unwrap(),original);
        assert!(!Arc::make_mut(&mut owner).isolate_start_state_and_drain_nullable_terminals().contains(&0));
    }
    #[test]
    fn constraint_clone_shares_the_same_immutable_lexer() {
        let vocab=crate::Vocab::from_entries_unchecked(vec![(1,b"a".to_vec()),(2,b"b".to_vec())]);
        let source=Constraint::from_glrm_grammar(r#"start x; nt x = "a" | "b";"#,&vocab).unwrap();
        let copy=source.clone();assert!(Arc::ptr_eq(&source.tokenizer,&copy.tokenizer));assert_eq!(source.save(),copy.save());
    }
}

#[cfg(test)]
mod raw_terminal_radius_tests {
    use super::*;
    use crate::{runtime::Constraint, Grammar, Vocab};

    /// Exhaustively enumerate the finite test alphabet, independent of the
    /// product/shortest-distance algorithm being checked. No quotient proofs.
    fn reference_radius(
        tokenizer: &Tokenizer, terminal: TerminalID, source: u32,
        atoms: &[&[u8]], horizon: u32,
    ) -> u32 {
        let mut frontier = vec![source];
        for count in 1..=horizon {
            let mut next = Vec::new();
            for state in frontier {
                for atom in atoms {
                    let mut end = state;
                    for &byte in *atom {
                        let Some(target) = tokenizer.step(end, byte) else { return count - 1; };
                        if target >= tokenizer.num_states()
                            || (!tokenizer.possible_future_terminals(target).contains(terminal as usize)
                                && !tokenizer.matched_terminals_slice(target).contains(&terminal))
                        {
                            return count - 1;
                        }
                        end = target;
                    }
                    next.push(end);
                }
            }
            frontier = next;
        }
        horizon
    }

    #[test]
    fn raw_terminal_radius_matches_exhaustive_ascii_and_multibyte_words() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec()),
                                    (2, "é".as_bytes().to_vec())]);
        let fixtures: [(&str, &str, Vec<&[u8]>); 3] = [
            (r"[ab]{1,4}", r"[ab]+", vec![b"a", b"b"]),
            (r"a{1,4}", r"[ab]+", vec![b"a", b"b"]),
            (r"(?:a|é){1,4}", r"(?:a|\xC3\xA9)+", vec![b"a", "é".as_bytes()]),
        ];
        let mut positive = 0usize;
        let mut checked = 0usize;
        for (pattern, slice_pattern, atoms) in fixtures {
            let grammar = format!("glrm 1; start s; t WORD = /{pattern}/; nt s = WORD;");
            let constraint = Constraint::compile(Grammar::glrm(&grammar), &vocab).unwrap();
            let terminal = constraint.terminal_display_names.iter()
                .position(|name| name == "WORD").unwrap() as TerminalID;
            let slice = VocabPartitionDfa::compile_byte_regex("test-atoms", slice_pattern).unwrap();
            for source in 0..constraint.tokenizer.num_states() {
                if constraint.tokenizer.state_has_epsilon_transitions(source)
                    || constraint.tokenizer.state_is_virtual_runtime(source) { continue; }
                let expected = reference_radius(&constraint.tokenizer, terminal, source, &atoms, 6);
                let actual = constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                    &constraint.tokenizer, terminal, source, 901, &slice, 6, 65_536,
                );
                assert_eq!(actual, Some(expected), "pattern={pattern} source={source}");
                checked += 1;
                positive += usize::from(expected > 0);
                // Budget exhaustion never creates a certificate, and its key
                // cannot poison a subsequent larger-budget query.
                assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                    &constraint.tokenizer, terminal, source, 901, &slice, 6, 0,
                ), Some(0));
                assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                    &constraint.tokenizer, terminal, source, 901, &slice, 6, 65_536,
                ), actual);
            }
            let empty = VocabPartitionDfa::compile_byte_regex("nullable", r"a*").unwrap();
            assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                &constraint.tokenizer, terminal, 0, 902, &empty, 6, 65_536,
            ), None);
            assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                &constraint.tokenizer, terminal, u32::MAX, 901, &slice, 6, 65_536,
            ), None);
        }
        assert!(checked > 4 && positive > 0, "must exercise real positive and negative certificates");
    }
}

#[cfg(test)]
#[test]
fn direct_flat_mask_trie_preserves_all_reference_runtime_fields() {
    fn check(entries: &[(u16, usize, &[u8])]) {
        let reference = DynamicMaskTrie::from_partitioned_token_refs_reference(entries);
        let direct = DynamicMaskTrie::from_partitioned_sorted_direct(entries);
        assert_eq!(bincode::serialize(&reference).unwrap(), bincode::serialize(&direct).unwrap(),
            "all nodes, edges, byte layout, metadata, walk ops and language classes must agree");
    }
    check(&[]);
    let mut random = 7u64;
    for case in 0..512 {
        let mut next = || {random ^= random << 13; random ^= random >> 7; random ^= random << 17; random};
        let mut words = vec![(0u16, Vec::<u8>::new())];
        for _ in 0..case % 150 {
            let class = (next() % 5) as u16;
            let len = (next() % 32) as usize;
            words.push((class, (0..len).map(|_|(next() % 256) as u8).collect()));
        }
        words.sort(); words.dedup();
        let entries = words.iter().enumerate().map(|(i,(class,word))|
            (*class,i*17,word.as_slice())).collect::<Vec<_>>();
        check(&entries);
    }
    for depth in [1,16,64,256,512] {
        let words = (0..depth).map(|n|vec![b'a';n]).collect::<Vec<_>>();
        let entries = words.iter().enumerate().map(|(i,word)|(0u16,i,word.as_slice())).collect::<Vec<_>>();
        check(&entries);
    }
}
