use glrmask_artifact::CommitTemplateDfas;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::automata::regex::Expr;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::{DWA, DwaTransitionMap};
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

use super::mask_mapping::FinalMaskMapping;

pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;

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

#[derive(Debug)]
pub(crate) struct PackedNonDwaWeights {
    pub(crate) pool: Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    pub(crate) parser_top_accept: BTreeMap<i32, u32>,
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<u32>>,
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, u32>,
    pub(crate) possible_matches: BTreeMap<TerminalID, u32>,
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

/// Exact dense acceptance indexed directly by internal tokenizer-state ID.
///
/// `row_kinds` uses 0 for empty, 1 for an ordinary row in `rows`, and 2 for the
/// shared all-token row. Keeping all ordinary rows in one flat allocation avoids
/// tens of thousands of per-state `Arc` allocations during finalization and
/// makes hot-path lookup a bounds check plus one slice operation.
#[derive(Debug, Clone, Default)]
pub(crate) struct DenseAcceptanceRows {
    words_per_row: usize,
    rows: Arc<[u64]>,
    row_kinds: Arc<[u8]>,
    full_dense: DenseWords,
}

impl DenseAcceptanceRows {
    pub(crate) fn new(
        words_per_row: usize,
        rows: Vec<u64>,
        row_kinds: Vec<u8>,
        full_dense: DenseWords,
    ) -> Self {
        debug_assert_eq!(rows.len(), words_per_row.saturating_mul(row_kinds.len()));
        Self {
            words_per_row,
            rows: rows.into(),
            row_kinds: row_kinds.into(),
            full_dense,
        }
    }

    #[inline]
    pub(crate) fn get(&self, tsid: u32) -> Option<&[u64]> {
        let tsid = tsid as usize;
        match self.row_kinds.get(tsid).copied()? {
            0 => None,
            2 => Some(self.full_dense.as_ref()),
            _ => {
                let start = tsid.checked_mul(self.words_per_row)?;
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

pub(crate) type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;

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
pub(crate) type DenseWeightBufMaskCache = FxHashMap<usize, Box<[u32]>>;
pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u16, u32)]>>;
pub(crate) type DirectSparseWeightTokenSetCache = FxHashSet<usize>;
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
    Dense(Vec<Box<[u32; 256]>>),
    Flat(Arc<[u32]>),
    /// Runtime tokenizer already owns an allocation-light exact transition
    /// table; call through instead of rebuilding a second dense table.
    Fallback(usize),
    Hybrid {
        state_to_dense_row: Vec<u32>,
        dense_rows: Vec<Box<[u32; 256]>>,
    },
}

impl Default for FastTokenizerTransitions {
    fn default() -> Self {
        Self::Dense(Vec::new())
    }
}

impl FastTokenizerTransitions {
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
                .map_or(u32::MAX, |row| row[byte as usize]),
            Self::Flat(flat) => flat
                .get(state as usize * 256 + byte as usize)
                .copied()
                .unwrap_or(u32::MAX),
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
            Self::Flat(flat) => flat.len() / 256,
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
        fn flat_rows(flat: &[u32]) -> Option<Vec<Box<[u32; 256]>>> {
            let chunks = flat.chunks_exact(256);
            if !chunks.remainder().is_empty() {
                return None;
            }
            chunks
                .map(|chunk| {
                    let row: &[u32; 256] = chunk.try_into().ok()?;
                    Some(Box::new(*row))
                })
                .collect()
        }

        fn rebased_row(row: &[u32; 256], offset: u32) -> Box<[u32; 256]> {
            let mut rebased = Box::new(*row);
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
                    Self::Flat(flat) => {
                        let rows = flat_rows(&flat)?;
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Fallback(_) => return None,
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
                        Self::Fallback(_) => return None,
                        Self::Dense(rows) => {
                            for row in rows {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Flat(flat) => {
                            let rows = flat_rows(flat)?;
                            for row in rows {
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
pub(crate) type FastTemplateDfasByTerminal = Vec<Option<Arc<FastCommitTemplateDfas>>>;

const INLINE_TEMPLATE_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone, Default)]
pub(crate) enum FastTemplateTransitionRow {
    #[default]
    Empty,
    Inline(SmallVec<[(i32, u32); 4]>),
    Hash(FxHashMap<i32, u32>),
}

impl FastTemplateTransitionRow {
    fn from_entries(entries: impl IntoIterator<Item = (i32, u32)>) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
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

#[derive(Debug, Clone, Default)]
pub(crate) struct FastTemplateDfaState {
    pub(crate) is_accepting: bool,
    pub(crate) default_target: Option<u32>,
    pub(crate) transitions: FastTemplateTransitionRow,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastTemplateDfa {
    pub(crate) states: Vec<FastTemplateDfaState>,
    pub(crate) start_state: u32,
}

impl FastTemplateDfa {
    fn from_dfa(dfa: &UnweightedDfa) -> Self {
        Self {
            states: dfa
                .states
                .iter()
                .map(|state| FastTemplateDfaState {
                    is_accepting: state.is_accepting,
                    default_target: state.transitions.get(&DEFAULT_LABEL).copied(),
                    transitions: FastTemplateTransitionRow::from_entries(
                        state
                            .transitions
                            .iter()
                            .filter(|(label, _)| **label != DEFAULT_LABEL)
                            .map(|(&label, &target)| (label, target)),
                    ),
                })
                .collect(),
            start_state: dfa.start_state,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastCommitTemplateDfas {
    pub(crate) pop: FastTemplateDfa,
    pub(crate) read: FastTemplateDfa,
    pub(crate) push: FastTemplateDfa,
    pub(crate) pop_to_read: Vec<Option<u32>>,
    pub(crate) pop_to_push: Vec<Option<u32>>,
    pub(crate) read_to_push: Vec<Option<u32>>,
}

impl FastCommitTemplateDfas {
    pub(crate) fn from_template(template: &CommitTemplateDfas) -> Self {
        Self {
            pop: FastTemplateDfa::from_dfa(&template.pop),
            read: FastTemplateDfa::from_dfa(&template.read),
            push: FastTemplateDfa::from_dfa(&template.push),
            pop_to_read: template.pop_to_read.clone(),
            pop_to_push: template.pop_to_push.clone(),
            read_to_push: template.read_to_push.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct SpecialTokenTerminal {
    pub(crate) terminal_id: TerminalID,
    pub(crate) token_id: u32,
}

/// Compact runtime-only vocabulary trie. It deliberately stores only the
/// information dynamic mask traversal consumes: compressed byte edges, child
/// ranges, and canonical token leaves.
#[derive(Debug, Clone, Default)]
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

#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicMaskTrieEdge {
    pub(crate) byte_start: u32,
    pub(crate) byte_len: u32,
    pub(crate) child: u32,
}

/// One radix edge in depth-first preorder. `subtree_end` is the first walk
/// entry after the child subtree, so a failed edge or accepted whole subtree
/// can be skipped with one index assignment.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DynamicMaskTrieWalkEdge {
    pub(crate) byte_start: u32,
    pub(crate) child: u32,
    pub(crate) subtree_end: u32,
    pub(crate) byte_len: u16,
    pub(crate) parent_depth: u16,
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskTrie {
    pub(crate) nodes: Vec<DynamicMaskTrieNode>,
    pub(crate) edges: Vec<DynamicMaskTrieEdge>,
    edge_bytes: Vec<u8>,
    subtree_tokens: Vec<u32>,
    walk_edges: Vec<DynamicMaskTrieWalkEdge>,
}

/// Vocab-only layout refinement used by the dynamic-mask radix trie.
///
/// `base_partition` is the existing p0/p1/... character-type class.  The
/// extra bits deliberately describe only coarse byte shape, not grammar
/// semantics: their job is to stop a few lexer-sensitive bytes from
/// contaminating otherwise uniform large subtrees.  The runtime never
/// interprets this value after construction.
pub(crate) fn dynamic_mask_vocab_layout_class(base_partition: u8, bytes: &[u8]) -> u16 {
    #[inline]
    fn first_kind(byte: Option<u8>) -> u16 {
        match byte {
            None => 0,
            Some(b' ') => 1,
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'_') => 2,
            Some(b'0'..=b'9') => 3,
            Some(b'"' | b'\\' | b'\'') => 4,
            Some(0..=31 | 127) => 5,
            Some(128..=255) => 6,
            Some(_) => 7,
        }
    }

    let mut flags = 0u16;
    for &byte in bytes {
        flags |= match byte {
            b'"' => 1 << 0,
            b'\\' => 1 << 1,
            b'\'' => 1 << 2,
            0..=31 | 127 => 1 << 3,
            b' ' => 1 << 4,
            b'{' | b'}' | b'[' | b']' | b'(' | b')' | b',' | b':' | b';' => 1 << 5,
            b'.' | b'+' | b'-' | b'*' | b'/' | b'%' | b'=' | b'<' | b'>' | b'!'
            | b'?' | b'&' | b'|' | b'^' | b'~' | b'@' | b'#' | b'$' | b'`' => 1 << 6,
            128..=255 => 1 << 7,
            _ => 0,
        };
    }

    (u16::from(base_partition) << 11) | (first_kind(bytes.first().copied()) << 8) | flags
}

impl DynamicMaskTrie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![DynamicMaskTrieNode::default()],
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
        }
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
    pub(crate) fn children(&self, node: u32) -> &[DynamicMaskTrieEdge] {
        let node = self.node(node);
        let start = node.first_child as usize;
        let end = start + node.child_len as usize;
        &self.edges[start..end]
    }

    #[inline]
    pub(crate) fn edge_bytes(&self, edge: &DynamicMaskTrieEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        let end = start + edge.byte_len as usize;
        &self.edge_bytes[start..end]
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

    pub(crate) fn push_edge_bytes(&mut self, bytes: &[u8]) -> (u32, u32) {
        let start = self.edge_bytes.len() as u32;
        self.edge_bytes.extend_from_slice(bytes);
        (start, bytes.len() as u32)
    }

    #[inline]
    pub(crate) fn edge_bytes_len(&self) -> usize {
        self.edge_bytes.len()
    }

    fn collect_subtree_metadata(
        &mut self,
        node_id: u32,
        prefix_byte_len: u32,
    ) -> ([u64; 4], [u64; 4], u32) {
        self.nodes[node_id as usize].prefix_byte_len = prefix_byte_len;
        let start = self.subtree_tokens.len() as u32;
        if let Some(token_id) = self.nodes[node_id as usize].token_id {
            self.subtree_tokens.push(token_id);
        }

        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        let mut subtree_bytes = [0u64; 4];
        let mut subtree_first_bytes = [0u64; 4];
        let mut subtree_max_byte_len = 0u32;
        for edge_index in first_child..first_child + child_len {
            // Copy the compact edge fields before recursing so no borrow of
            // `self.edges` remains live across the mutable recursive call.
            let edge = self.edges[edge_index].clone();
            let byte_start = edge.byte_start as usize;
            let byte_end = byte_start + edge.byte_len as usize;
            for &byte in &self.edge_bytes[byte_start..byte_end] {
                subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
            }
            let child_prefix_byte_len = prefix_byte_len
                .checked_add(edge.byte_len)
                .expect("dynamic mask trie token byte length exceeds u32");
            let (child_bytes, child_first_bytes, child_max_byte_len) =
                self.collect_subtree_metadata(edge.child, child_prefix_byte_len);
            for (target, child) in subtree_bytes.iter_mut().zip(child_bytes) {
                *target |= child;
            }
            if edge.byte_len == 0 {
                for (target, child) in subtree_first_bytes.iter_mut().zip(child_first_bytes) {
                    *target |= child;
                }
            } else {
                let first = self.edge_bytes[byte_start];
                subtree_first_bytes[first as usize >> 6] |= 1u64 << (first & 63);
            }
            subtree_max_byte_len = subtree_max_byte_len.max(
                edge.byte_len
                    .checked_add(child_max_byte_len)
                    .expect("dynamic mask trie token byte length exceeds u32"),
            );
        }

        let end = self.subtree_tokens.len() as u32;
        let node = &mut self.nodes[node_id as usize];
        node.subtree_token_start = start;
        node.subtree_token_end = end;
        node.subtree_bytes = subtree_bytes;
        node.subtree_first_bytes = subtree_first_bytes;
        node.subtree_max_byte_len = subtree_max_byte_len;
        (subtree_bytes, subtree_first_bytes, subtree_max_byte_len)
    }

    pub(crate) fn finalize_subtree_metadata(&mut self) {
        self.subtree_tokens.clear();
        self.subtree_tokens.reserve(self.nodes.len());
        if !self.nodes.is_empty() {
            self.collect_subtree_metadata(0, 0);
        }
        self.finalize_walk_edges();
    }

    fn append_walk_edges(&mut self, node_id: u32, parent_depth: u16) {
        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        for edge_index in first_child..first_child + child_len {
            let edge = self.edges[edge_index].clone();
            let byte_len = u16::try_from(edge.byte_len)
                .expect("dynamic mask trie radix edge exceeds u16 length");
            let entry_index = self.walk_edges.len();
            self.walk_edges.push(DynamicMaskTrieWalkEdge {
                byte_start: edge.byte_start,
                child: edge.child,
                subtree_end: 0,
                byte_len,
                parent_depth,
            });
            self.append_walk_edges(
                edge.child,
                parent_depth
                    .checked_add(1)
                    .expect("dynamic mask trie depth exceeds u16"),
            );
            self.walk_edges[entry_index].subtree_end = self.walk_edges.len() as u32;
        }
    }

    fn finalize_walk_edges(&mut self) {
        self.walk_edges.clear();
        self.walk_edges.reserve(self.edges.len());
        if !self.nodes.is_empty() {
            self.append_walk_edges(0, 0);
        }
        debug_assert_eq!(self.walk_edges.len(), self.edges.len());
    }

    fn flatten_vocab_node(node: &VocabPrefixTreeNode, output: &mut Self) -> u32 {
        let node_id = output.nodes.len() as u32;
        output.nodes.push(DynamicMaskTrieNode {
            token_id: node.has_token().then_some(node.token_id() as u32),
            first_child: 0,
            child_len: 0,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });

        let children = node.children();
        if children.is_empty() {
            return node_id;
        }

        let first_child = output.edges.len() as u32;
        output
            .edges
            .resize_with(output.edges.len() + children.len(), DynamicMaskTrieEdge::default);
        output.nodes[node_id as usize].first_child = first_child;
        output.nodes[node_id as usize].child_len = children.len() as u32;

        for (offset, (segment, child)) in node.iter_children().enumerate() {
            let child_id = Self::flatten_vocab_node(child, output);
            let (byte_start, byte_len) = output.push_edge_bytes(segment);
            output.edges[first_child as usize + offset] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: child_id,
            };
        }

        node_id
    }

    fn from_vocab_prefix_tree_node(node: &VocabPrefixTreeNode) -> Self {
        let mut output = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
        };
        let root = Self::flatten_vocab_node(node, &mut output);
        debug_assert_eq!(root, 0);
        output.finalize_subtree_metadata();
        output
    }

    pub(crate) fn from_vocab_prefix_tree(tree: &VocabPrefixTree) -> Self {
        // Root children are disjoint lexical subtrees. Flattening them in
        // parallel is safe, then the compact fragments are stitched with fixed
        // index offsets. This keeps the runtime representation lean without
        // making finalization wait on a single 140k-node recursive walk.
        let root = &tree.root;
        let root_children = root.children();
        if rayon::current_num_threads() == 1 || root_children.len() < 8 {
            return Self::from_vocab_prefix_tree_node(root);
        }

        let root_prefix_len = root.prefix().len();
        let mut fragments: Vec<(Box<[u8]>, Self)> = root_children
            .par_iter()
            .map(|child| {
                let edge = child.prefix()[root_prefix_len..].to_vec().into_boxed_slice();
                (edge, Self::from_vocab_prefix_tree_node(child))
            })
            .collect();
        let node_capacity = 1 + fragments.iter().map(|(_, fragment)| fragment.nodes.len()).sum::<usize>();
        let edge_capacity = root_children.len()
            + fragments.iter().map(|(_, fragment)| fragment.edges.len()).sum::<usize>();
        let byte_capacity = fragments
            .iter()
            .map(|(edge, fragment)| edge.len() + fragment.edge_bytes.len())
            .sum::<usize>();
        let mut output = Self {
            nodes: Vec::with_capacity(node_capacity),
            edges: Vec::with_capacity(edge_capacity),
            edge_bytes: Vec::with_capacity(byte_capacity),
            subtree_tokens: Vec::with_capacity(node_capacity),
            walk_edges: Vec::with_capacity(edge_capacity),
        };
        output.nodes.push(DynamicMaskTrieNode {
            token_id: root.has_token().then_some(root.token_id() as u32),
            first_child: 0,
            child_len: root_children.len() as u32,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });
        output
            .edges
            .resize_with(root_children.len(), DynamicMaskTrieEdge::default);

        for (root_slot, (root_edge, mut fragment)) in fragments.drain(..).enumerate() {
            let node_base = output.nodes.len() as u32;
            let edge_base = output.edges.len() as u32;
            let byte_base = output.edge_bytes.len() as u32;
            output.edge_bytes.extend_from_slice(&fragment.edge_bytes);
            for node in &mut fragment.nodes {
                if node.child_len != 0 {
                    node.first_child += edge_base;
                }
            }
            for edge in &mut fragment.edges {
                edge.byte_start += byte_base;
                edge.child += node_base;
            }
            output.nodes.append(&mut fragment.nodes);
            output.edges.append(&mut fragment.edges);
            let (byte_start, byte_len) = output.push_edge_bytes(&root_edge);
            output.edges[root_slot] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: node_base,
            };
        }

        output.finalize_subtree_metadata();
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

        let mut groups = Vec::<Self>::new();
        let mut index = start;
        while index < entries.len() {
            let class = entries[index].0;
            let group_start = index;
            index += 1;
            while index < entries.len() && entries[index].0 == class {
                index += 1;
            }
            let refs = entries[group_start..index]
                .iter()
                .map(|(_, token_id, bytes)| (*token_id, *bytes))
                .collect::<Vec<_>>();
            debug_assert!(refs.windows(2).all(|pair| pair[0].1 <= pair[1].1));
            let tree = VocabPrefixTree::build_presorted(&refs);
            groups.push(Self::from_vocab_prefix_tree_node(&tree.root));
        }

        let root_child_count = groups.len();
        output.edges.resize_with(root_child_count, DynamicMaskTrieEdge::default);
        output.nodes[0].first_child = 0;
        output.nodes[0].child_len = root_child_count as u32;

        for (root_slot, mut fragment) in groups.into_iter().enumerate() {
            let node_base = output.nodes.len() as u32;
            let edge_base = output.edges.len() as u32;
            let byte_base = output.edge_bytes.len() as u32;

            output.edge_bytes.extend_from_slice(&fragment.edge_bytes);
            for node in &mut fragment.nodes {
                if node.child_len != 0 {
                    node.first_child += edge_base;
                }
            }
            for edge in &mut fragment.edges {
                edge.byte_start += byte_base;
                edge.child += node_base;
            }
            output.nodes.append(&mut fragment.nodes);
            output.edges.append(&mut fragment.edges);

            // Structural class edge: no lexer byte is consumed here.
            let (byte_start, byte_len) = output.push_edge_bytes(&[]);
            output.edges[root_slot] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: node_base,
            };
        }

        output.finalize_subtree_metadata();
        output
    }
}

impl Default for DynamicMaskTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum PackedDynamicMaskTokenAliases {
    Single(u32),
    Many(Box<[u32]>),
}

#[derive(Debug, Clone)]
pub(crate) enum DynamicMaskAliasStore {
    Ordered(Arc<Vec<Vec<u32>>>),
    Packed(Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>),
}

#[derive(Debug)]
struct DynamicMaskCacheEntry {
    state: DynamicMaskStateKey,
    mask: Arc<[u32]>,
}


#[derive(Debug, Clone, Copy)]
enum DirectRegularSupportNode {
    Leaf(u64),
    Branch(u32, u32),
}

#[derive(Debug, Clone, Copy)]
struct DirectRegularSmallSupport {
    len: u8,
    terminals: [u16; 4],
}

impl DirectRegularSmallSupport {
    const UNAVAILABLE: u8 = u8::MAX;

    fn unavailable() -> Self {
        Self {
            len: Self::UNAVAILABLE,
            terminals: [0; 4],
        }
    }

    fn from_leaf(mut value: u64) -> Self {
        if value.count_ones() > 4 {
            return Self::unavailable();
        }
        let mut result = Self {
            len: 0,
            terminals: [0; 4],
        };
        while value != 0 {
            result.terminals[result.len as usize] = value.trailing_zeros() as u16;
            result.len += 1;
            value &= value - 1;
        }
        result
    }

    fn combine(left: Self, right: Self, right_offset: usize) -> Self {
        if left.len == Self::UNAVAILABLE
            || right.len == Self::UNAVAILABLE
            || usize::from(left.len) + usize::from(right.len) > 4
            || right_offset > u16::MAX as usize
        {
            return Self::unavailable();
        }
        let mut result = Self {
            len: left.len + right.len,
            terminals: [0; 4],
        };
        result.terminals[..left.len as usize]
            .copy_from_slice(&left.terminals[..left.len as usize]);
        for (index, &terminal) in right.terminals[..right.len as usize].iter().enumerate() {
            let Some(terminal) = usize::from(terminal).checked_add(right_offset) else {
                return Self::unavailable();
            };
            let Ok(terminal) = u16::try_from(terminal) else {
                return Self::unavailable();
            };
            result.terminals[left.len as usize + index] = terminal;
        }
        result
    }

    fn terminals(&self) -> Option<&[u16]> {
        (self.len != Self::UNAVAILABLE).then(|| &self.terminals[..self.len as usize])
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DirectRegularTerminalSupport {
    roots: Vec<u32>,
    nodes: Vec<DirectRegularSupportNode>,
    node_counts: Vec<u16>,
    node_small_support: Vec<DirectRegularSmallSupport>,
    dense_state_rows: FxHashMap<u32, Arc<[u64]>>,
    zero: Vec<u32>,
    levels: u8,
    num_terminals: usize,
}

struct DirectRegularTerminalSupportBuilder {
    nodes: Vec<DirectRegularSupportNode>,
    node_counts: Vec<u16>,
    node_small_support: Vec<DirectRegularSmallSupport>,
    leaf_intern: FxHashMap<u64, u32>,
    branch_intern: Vec<FxHashMap<(u32, u32), u32>>,
    union_memo: Vec<FxHashMap<(u32, u32), u32>>,
    zero: Vec<u32>,
}

impl DirectRegularTerminalSupportBuilder {
    fn new(levels: usize) -> Self {
        let mut builder = Self {
            nodes: Vec::new(),
            node_counts: Vec::new(),
            node_small_support: Vec::new(),
            leaf_intern: FxHashMap::default(),
            branch_intern: (0..=levels).map(|_| FxHashMap::default()).collect(),
            union_memo: (0..=levels).map(|_| FxHashMap::default()).collect(),
            zero: Vec::with_capacity(levels + 1),
        };
        let leaf = builder.intern_leaf(0);
        builder.zero.push(leaf);
        for level in 1..=levels {
            let child = builder.zero[level - 1];
            let root = builder.intern_branch(level, child, child);
            builder.zero.push(root);
        }
        builder
    }

    fn intern_leaf(&mut self, value: u64) -> u32 {
        if let Some(&id) = self.leaf_intern.get(&value) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(DirectRegularSupportNode::Leaf(value));
        self.node_counts.push(value.count_ones() as u16);
        self.node_small_support
            .push(DirectRegularSmallSupport::from_leaf(value));
        self.leaf_intern.insert(value, id);
        id
    }

    fn intern_branch(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if let Some(&id) = self.branch_intern[level].get(&(left, right)) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes
            .push(DirectRegularSupportNode::Branch(left, right));
        self.node_counts.push(
            self.node_counts[left as usize].saturating_add(self.node_counts[right as usize]),
        );
        let right_offset = 64usize << (level - 1);
        self.node_small_support.push(DirectRegularSmallSupport::combine(
            self.node_small_support[left as usize],
            self.node_small_support[right as usize],
            right_offset,
        ));
        self.branch_intern[level].insert((left, right), id);
        id
    }

    fn union(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if left == right {
            return left;
        }
        if left == self.zero[level] {
            return right;
        }
        if right == self.zero[level] {
            return left;
        }
        let key = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        if let Some(&id) = self.union_memo[level].get(&key) {
            return id;
        }
        let result = if level == 0 {
            let DirectRegularSupportNode::Leaf(left_value) = self.nodes[left as usize] else {
                unreachable!()
            };
            let DirectRegularSupportNode::Leaf(right_value) = self.nodes[right as usize] else {
                unreachable!()
            };
            self.intern_leaf(left_value | right_value)
        } else {
            let DirectRegularSupportNode::Branch(left_a, left_b) = self.nodes[left as usize] else {
                unreachable!()
            };
            let DirectRegularSupportNode::Branch(right_a, right_b) = self.nodes[right as usize]
            else {
                unreachable!()
            };
            let a = self.union(level - 1, left_a, right_a);
            let b = self.union(level - 1, left_b, right_b);
            self.intern_branch(level, a, b)
        };
        self.union_memo[level].insert(key, result);
        result
    }

    fn singleton(&mut self, levels: usize, terminal: usize) -> u32 {
        let word = terminal / 64;
        let mut node = self.intern_leaf(1u64 << (terminal % 64));
        for level in 1..=levels {
            let zero = self.zero[level - 1];
            node = if ((word >> (level - 1)) & 1) == 0 {
                self.intern_branch(level, node, zero)
            } else {
                self.intern_branch(level, zero, node)
            };
        }
        node
    }
}

impl DirectRegularTerminalSupport {
    pub(crate) fn build(automaton: &DirectRegularAutomaton, num_terminals: usize) -> Self {
        if automaton.states.is_empty() || num_terminals == 0 {
            return Self::default();
        }
        let word_count = num_terminals.div_ceil(64).next_power_of_two();
        let levels = word_count.trailing_zeros() as usize;
        let mut builder = DirectRegularTerminalSupportBuilder::new(levels);
        let singletons = (0..num_terminals)
            .map(|terminal| builder.singleton(levels, terminal))
            .collect::<Vec<_>>();

        let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
        let mut remaining_children = Vec::<u32>::with_capacity(automaton.states.len());
        let mut queue = VecDeque::<u32>::new();
        for (source, state) in automaton.states.iter().enumerate() {
            remaining_children.push(state.epsilons.len() as u32);
            if state.epsilons.is_empty() {
                queue.push_back(source as u32);
            }
            for &child in &state.epsilons {
                parents[child as usize].push(source as u32);
            }
        }

        let mut roots = vec![builder.zero[levels]; automaton.states.len()];
        let mut processed = 0usize;
        while let Some(raw) = queue.pop_front() {
            let state = &automaton.states[raw as usize];
            let mut root = builder.zero[levels];
            for &terminal in state.transitions.keys() {
                if (terminal as usize) < num_terminals {
                    root = builder.union(levels, root, singletons[terminal as usize]);
                }
            }
            for &child in &state.epsilons {
                root = builder.union(levels, root, roots[child as usize]);
            }
            roots[raw as usize] = root;
            processed += 1;
            for &parent in &parents[raw as usize] {
                let remaining = &mut remaining_children[parent as usize];
                *remaining -= 1;
                if *remaining == 0 {
                    queue.push_back(parent);
                }
            }
        }
        if processed != automaton.states.len() {
            return Self::default();
        }
        let mut support = Self {
            roots,
            nodes: builder.nodes,
            node_counts: builder.node_counts,
            node_small_support: builder.node_small_support,
            dense_state_rows: FxHashMap::default(),
            zero: builder.zero,
            levels: levels as u8,
            num_terminals,
        };
        let dense_word_count = num_terminals.div_ceil(64);
        for &raw_state in &automaton.start_states {
            let mut words = vec![0u64; dense_word_count];
            support.or_state_into(raw_state, &mut words);
            support
                .dense_state_rows
                .insert(raw_state, Arc::from(words));
        }
        support
    }

    pub(crate) fn is_initialized(&self) -> bool {
        !self.roots.is_empty()
    }

    pub(crate) fn for_each_small_state_terminal(
        &self,
        raw_state: u32,
        mut visit: impl FnMut(TerminalID),
    ) -> bool {
        let Some(root) = self.root_id(raw_state) else {
            return false;
        };
        let Some(terminals) = self.node_small_support[root as usize].terminals() else {
            return false;
        };
        for &terminal in terminals {
            let terminal = TerminalID::from(terminal);
            if (terminal as usize) < self.num_terminals {
                visit(terminal);
            }
        }
        true
    }

    #[inline]
    pub(crate) fn contains(&self, raw_state: u32, terminal: TerminalID) -> bool {
        let terminal = terminal as usize;
        if terminal >= self.num_terminals {
            return false;
        }
        let Some(&mut_node) = self.roots.get(raw_state as usize) else {
            return false;
        };
        let mut node = mut_node;
        let mut level = self.levels as usize;
        let word = terminal / 64;
        while level != 0 {
            let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
                return false;
            };
            node = if ((word >> (level - 1)) & 1) == 0 {
                left
            } else {
                right
            };
            level -= 1;
        }
        let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
            return false;
        };
        value & (1u64 << (terminal % 64)) != 0
    }

    fn or_node(&self, node: u32, level: usize, word_base: usize, output: &mut [u64]) {
        if node == self.zero[level] {
            return;
        }
        if level == 0 {
            let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
                return;
            };
            if let Some(word) = output.get_mut(word_base) {
                *word |= value;
            }
            return;
        }
        let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
            return;
        };
        let half = 1usize << (level - 1);
        self.or_node(left, level - 1, word_base, output);
        self.or_node(right, level - 1, word_base + half, output);
    }

    pub(crate) fn or_state_into(&self, raw_state: u32, output: &mut [u64]) {
        if let Some(words) = self.dense_state_rows.get(&raw_state) {
            for (target, source) in output.iter_mut().zip(words.iter()) {
                *target |= *source;
            }
            return;
        }
        if let Some(&root) = self.roots.get(raw_state as usize) {
            self.or_node(root, self.levels as usize, 0, output);
        }
    }

    #[inline]
    pub(crate) fn root_id(&self, raw_state: u32) -> Option<u32> {
        self.roots.get(raw_state as usize).copied()
    }

    #[inline]
    pub(crate) fn state_terminal_count(&self, raw_state: u32) -> Option<u16> {
        let root = *self.roots.get(raw_state as usize)?;
        self.node_counts.get(root as usize).copied()
    }

    pub(crate) fn singleton_terminal(&self, raw_state: u32) -> Option<TerminalID> {
        let root = self.root_id(raw_state)?;
        let terminals = self.node_small_support[root as usize].terminals()?;
        let [terminal] = terminals else {
            return None;
        };
        Some(TerminalID::from(*terminal))
    }

    fn intersects_node(
        &self,
        node: u32,
        level: usize,
        word_base: usize,
        terminals: &[u64],
    ) -> bool {
        if node == self.zero[level] {
            return false;
        }
        if level == 0 {
            let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
                return false;
            };
            return terminals
                .get(word_base)
                .is_some_and(|word| (*word & value) != 0);
        }
        let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
            return false;
        };
        let half = 1usize << (level - 1);
        self.intersects_node(left, level - 1, word_base, terminals)
            || self.intersects_node(right, level - 1, word_base + half, terminals)
    }

    pub(crate) fn intersects(&self, raw_state: u32, terminals: &[u64]) -> bool {
        if let Some(words) = self.dense_state_rows.get(&raw_state) {
            return words
                .iter()
                .zip(terminals)
                .any(|(left, right)| (*left & *right) != 0);
        }
        self.roots.get(raw_state as usize).is_some_and(|&root| {
            self.intersects_node(root, self.levels as usize, 0, terminals)
        })
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
pub(crate) type DynamicMaskStateKey =
    Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<TerminalID>)>)>)>;

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

#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicBoundedObservationSets {
    pool: Arc<[U8Set]>,
    horizon16: Arc<[u32]>,
    horizon64: Arc<[u32]>,
}

impl DynamicBoundedObservationSets {
    pub(crate) fn from_raw(horizon16: Box<[U8Set]>, horizon64: Box<[U8Set]>) -> Self {
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

/// Runtime-only vocabulary data for direct dynamic mask generation.
#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocab {
    pub(crate) trie: Arc<DynamicMaskTrie>,
    token_aliases: DynamicMaskAliasStore,
    canonical_original_token_offsets: Arc<Vec<u32>>,
    canonical_original_tokens: Arc<Vec<u32>>,
    node_token_markers: Arc<Vec<u64>>,
    subtree_original_token_offsets: Arc<Vec<u32>>,
    subtree_original_tokens: Arc<Vec<u32>>,
    pending_source: Option<DynamicMaskVocabSource>,
    initialized: bool,
    mask_cache: Arc<Mutex<Vec<DynamicMaskCacheEntry>>>,
    direct_regular_frontier_cache:
        Arc<Mutex<FxHashMap<usize, DirectRegularDynamicFrontierCacheEntry>>>,
    direct_regular_wide_frontier_index_cache: Arc<Mutex<FxHashMap<usize, usize>>>,
    direct_regular_terminal_support: Arc<DirectRegularTerminalSupport>,
    self_loop_projections: Arc<Vec<DynamicSelfLoopProjection>>,
    projection_by_source: Arc<[u32]>,
    projection_alias_vocab: Arc<[u32]>,
    projection_alias_h64: Arc<[u32]>,
    bounded_observation_sets: Arc<DynamicBoundedObservationSets>,
    /// Optional mask-only finite-token quotient. Commit continues to use the
    /// exact tokenizer stored on `Constraint`; dynamic mask projections may be
    /// built in this smaller coordinate and indexed from exact runtime states
    /// through `full_to_mask_state`.
    mask_tokenizer: Option<Arc<Tokenizer>>,
    full_to_mask_state: Arc<[u32]>,
}

impl DynamicMaskVocab {
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
        let token_aliases = DynamicMaskAliasStore::Ordered(token_aliases);
        let (canonical_original_token_offsets, canonical_original_tokens) =
            Self::flatten_canonical_original_tokens(&token_aliases);
        let node_token_markers = Self::build_node_token_markers(
            trie.as_ref(),
            &canonical_original_token_offsets,
            &canonical_original_tokens,
        );
        let (subtree_original_token_offsets, subtree_original_tokens) =
            Self::flatten_subtree_original_tokens(
                trie.as_ref(),
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
        Self {
            trie,
            token_aliases,
            canonical_original_token_offsets,
            canonical_original_tokens,
            node_token_markers,
            subtree_original_token_offsets,
            subtree_original_tokens,
            pending_source: None,
            initialized: true,
            mask_cache: Arc::new(Mutex::new(Vec::new())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            mask_tokenizer: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
        }
    }

    /// Create a constraint-local runtime value from a fully initialized,
    /// vocabulary-only template.
    ///
    /// The immutable trie and token indexes are shared. Every cache or
    /// accelerator whose contents can depend on parser, lexer, or constraint
    /// state is recreated empty, so repeated schema builds cannot inherit
    /// schema-derived runtime state.
    pub(crate) fn fresh_runtime_instance(&self) -> Self {
        debug_assert!(self.initialized);
        debug_assert!(self.pending_source.is_none());
        Self {
            trie: Arc::clone(&self.trie),
            token_aliases: self.token_aliases.clone(),
            canonical_original_token_offsets: Arc::clone(
                &self.canonical_original_token_offsets,
            ),
            canonical_original_tokens: Arc::clone(&self.canonical_original_tokens),
            node_token_markers: Arc::clone(&self.node_token_markers),
            subtree_original_token_offsets: Arc::clone(
                &self.subtree_original_token_offsets,
            ),
            subtree_original_tokens: Arc::clone(&self.subtree_original_tokens),
            pending_source: None,
            initialized: true,
            mask_cache: Arc::new(Mutex::new(Vec::new())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(
                FxHashMap::default(),
            )),
            direct_regular_terminal_support: Arc::new(
                DirectRegularTerminalSupport::default(),
            ),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            mask_tokenizer: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
        }
    }

    fn from_source(source: DynamicMaskVocabSource) -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            canonical_original_token_offsets: Arc::new(vec![0]),
            canonical_original_tokens: Arc::new(Vec::new()),
            node_token_markers: Arc::new(vec![0]),
            subtree_original_token_offsets: Arc::new(vec![0]),
            subtree_original_tokens: Arc::new(Vec::new()),
            pending_source: Some(source),
            initialized: false,
            mask_cache: Arc::new(Mutex::new(Vec::new())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            mask_tokenizer: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
        }
    }

    pub(crate) fn from_packed(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>,
    ) -> Self {
        let token_aliases = DynamicMaskAliasStore::Packed(token_aliases);
        let (canonical_original_token_offsets, canonical_original_tokens) =
            Self::flatten_canonical_original_tokens(&token_aliases);
        let node_token_markers = Self::build_node_token_markers(
            trie.as_ref(),
            &canonical_original_token_offsets,
            &canonical_original_tokens,
        );
        let (subtree_original_token_offsets, subtree_original_tokens) =
            Self::flatten_subtree_original_tokens(
                trie.as_ref(),
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
        Self {
            trie,
            token_aliases,
            canonical_original_token_offsets,
            canonical_original_tokens,
            node_token_markers,
            subtree_original_token_offsets,
            subtree_original_tokens,
            pending_source: None,
            initialized: true,
            mask_cache: Arc::new(Mutex::new(Vec::new())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            mask_tokenizer: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
        }
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub(crate) fn materialize_pending_source(&mut self) -> bool {
        let Some(source) = self.pending_source.take() else {
            return false;
        };
        self.trie = Arc::new(DynamicMaskTrie::from_vocab_prefix_tree(source.trie.as_ref()));
        self.token_aliases = DynamicMaskAliasStore::Ordered(source.token_aliases);
        (self.canonical_original_token_offsets, self.canonical_original_tokens) =
            Self::flatten_canonical_original_tokens(&self.token_aliases);
        self.node_token_markers = Self::build_node_token_markers(
            self.trie.as_ref(),
            &self.canonical_original_token_offsets,
            &self.canonical_original_tokens,
        );
        (self.subtree_original_token_offsets, self.subtree_original_tokens) =
            Self::flatten_subtree_original_tokens(
                self.trie.as_ref(),
                &self.canonical_original_token_offsets,
                &self.canonical_original_tokens,
            );
        self.initialized = true;
        true
    }

    fn flatten_canonical_original_tokens(
        token_aliases: &DynamicMaskAliasStore,
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        let alias_slots = match token_aliases {
            DynamicMaskAliasStore::Ordered(aliases) => aliases.len(),
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

    fn flatten_subtree_original_tokens(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        let subtree_canonical_tokens = trie.all_subtree_tokens();
        let mut offsets = Vec::with_capacity(subtree_canonical_tokens.len() + 1);
        let mut originals = Vec::new();
        offsets.push(0);
        for &canonical_token in subtree_canonical_tokens {
            let index = canonical_token as usize;
            let start = canonical_offsets[index] as usize;
            let end = canonical_offsets[index + 1] as usize;
            originals.extend_from_slice(&canonical_original_tokens[start..end]);
            offsets.push(originals.len() as u32);
        }
        (Arc::new(offsets), Arc::new(originals))
    }

    fn build_node_token_markers(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> Arc<Vec<u64>> {
        const FALLBACK_TAG: u64 = 1u64 << 63;
        let mut markers = Vec::with_capacity(trie.nodes.len());
        for node in &trie.nodes {
            let Some(canonical_token) = node.token_id else {
                markers.push(0);
                continue;
            };
            let index = canonical_token as usize;
            let start = canonical_offsets[index] as usize;
            let end = canonical_offsets[index + 1] as usize;
            let aliases = &canonical_original_tokens[start..end];
            let Some(&first_token) = aliases.first() else {
                markers.push(FALLBACK_TAG | (canonical_token as u64 + 1));
                continue;
            };
            let word = first_token / 32;
            let mut bits = 0u32;
            let mut one_word = true;
            for &token_id in aliases {
                if token_id / 32 != word {
                    one_word = false;
                    break;
                }
                bits |= 1u32 << (token_id % 32);
            }
            if one_word {
                debug_assert_ne!(bits, 0);
                debug_assert!(word < (1u32 << 31));
                markers.push((u64::from(word) << 32) | u64::from(bits));
            } else {
                markers.push(FALLBACK_TAG | (canonical_token as u64 + 1));
            }
        }
        Arc::new(markers)
    }

    #[inline]
    pub(crate) fn subtree_original_tokens(&self, node: u32) -> &[u32] {
        let canonical_range = self.trie.subtree_token_index_range(node);
        let start = self.subtree_original_token_offsets[canonical_range.start] as usize;
        let end = self.subtree_original_token_offsets[canonical_range.end] as usize;
        &self.subtree_original_tokens[start..end]
    }

    #[inline]
    pub(crate) fn canonical_token_count(&self) -> usize {
        self.canonical_original_token_offsets.len().saturating_sub(1)
    }

    pub(crate) fn token_ids(&self, canonical_token_id: u32) -> Option<&[u32]> {
        let index = canonical_token_id as usize;
        let end_index = index.checked_add(1)?;
        let (&start, &end) = self
            .canonical_original_token_offsets
            .get(index)
            .zip(self.canonical_original_token_offsets.get(end_index))?;
        (start != end).then(|| {
            &self.canonical_original_tokens[start as usize..end as usize]
        })
    }

    #[inline(always)]
    pub(crate) fn node_token_marker(&self, node: u32) -> u64 {
        debug_assert!((node as usize) < self.node_token_markers.len());
        unsafe { *self.node_token_markers.get_unchecked(node as usize) }
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
            .copied()
    }

    pub(crate) fn cache_direct_regular_wide_frontier_index(
        &self,
        key: usize,
        index: usize,
    ) {
        self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, index);
    }

    pub(crate) fn set_self_loop_projections(
        &mut self,
        projections: Vec<DynamicSelfLoopProjection>,
    ) {
        let state_count = self.mask_tokenizer.as_ref().map_or_else(
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

    pub(crate) fn set_mask_tokenizer_quotient(
        &mut self,
        tokenizer: Tokenizer,
        full_to_mask_state: Vec<u32>,
    ) {
        debug_assert!(!full_to_mask_state.is_empty());
        debug_assert!(full_to_mask_state
            .iter()
            .all(|&state| state < tokenizer.num_states()));
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.full_to_mask_state = Arc::from(full_to_mask_state);
    }

    /// Preserve mask-only quotient metadata when a deferred dynamic-vocabulary
    /// placeholder is replaced by its fully materialized runtime trie.
    ///
    /// The quotient is constraint/lexer derived while the trie is vocabulary
    /// derived, so deferred dynamic compilation may construct them at different
    /// times. Sharing the immutable Arc-backed metadata avoids cloning the
    /// quotient tokenizer during that handoff.
    pub(crate) fn inherit_mask_tokenizer_quotient_from(&mut self, source: &Self) {
        self.mask_tokenizer = source.mask_tokenizer.clone();
        self.full_to_mask_state = Arc::clone(&source.full_to_mask_state);
    }

    pub(crate) fn mask_tokenizer_quotient_for_transfer(&self) -> Option<(Tokenizer, Vec<u32>)> {
        self.mask_tokenizer.as_ref().map(|tokenizer| {
            ((**tokenizer).clone(), self.full_to_mask_state.as_ref().to_vec())
        })
    }

    #[inline]
    pub(crate) fn mask_projection_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_tokenizer.as_deref()
    }

    #[inline]
    pub(crate) fn mask_projection_state(&self, full_state: u32) -> u32 {
        self.full_to_mask_state
            .get(full_state as usize)
            .copied()
            .unwrap_or(full_state)
    }

    pub(crate) fn mask_projection_state_multiplicities(&self) -> Option<Vec<usize>> {
        let tokenizer = self.mask_tokenizer.as_ref()?;
        let mut counts = vec![0usize; tokenizer.num_states() as usize];
        for &state in self.full_to_mask_state.iter() {
            if let Some(count) = counts.get_mut(state as usize) {
                *count += 1;
            }
        }
        Some(counts)
    }

    /// Exact full-tokenizer preimage for quotient states that have exactly one
    /// runtime source.  Non-unique and unreachable quotient states are
    /// represented by `u32::MAX`.
    pub(crate) fn mask_projection_unique_full_states(&self) -> Option<Vec<u32>> {
        let tokenizer = self.mask_tokenizer.as_ref()?;
        let mut unique = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut duplicate = vec![false; tokenizer.num_states() as usize];
        for (full_state, &mask_state) in self.full_to_mask_state.iter().enumerate() {
            let index = mask_state as usize;
            if index >= unique.len() {
                continue;
            }
            if unique[index] == u32::MAX && !duplicate[index] {
                unique[index] = full_state as u32;
            } else {
                duplicate[index] = true;
                unique[index] = u32::MAX;
            }
        }
        Some(unique)
    }

    pub(crate) fn self_loop_projection(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let source_state = self.mask_projection_state(source_state);
        let index = *self.projection_by_source.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    pub(crate) fn self_loop_projection_alias_h64(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let source_state = self.mask_projection_state(source_state);
        let index = *self.projection_alias_h64.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    pub(crate) fn self_loop_projection_alias_vocab(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let source_state = self.mask_projection_state(source_state);
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

    pub(crate) fn copy_cached_mask(
        &self,
        state: &DynamicMaskStateKey,
        buf: &mut [u32],
    ) -> bool {
        let cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(entry) = cache.iter().rev().find(|entry| entry.state == *state) else {
            return false;
        };
        if entry.mask.len() != buf.len() {
            return false;
        }
        buf.copy_from_slice(&entry.mask);
        true
    }

    pub(crate) fn cache_mask(&self, state: DynamicMaskStateKey, mask: &[u32]) {
        // Keep enough exact states to cover an ordinary generated sequence.
        // A fixed 64-entry limit caused long source-specialized sequences to
        // evict their expensive early masks during the warmup pass, so every
        // measured pass recomputed them. Bound by bytes instead: Llama-sized
        // masks retain about 512 states in 8 MiB, while tiny vocabularies may
        // retain more without material memory cost.
        const MASK_CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;
        const MIN_MASK_CACHE_ENTRIES: usize = 64;
        const MAX_MASK_CACHE_ENTRIES: usize = 4096;
        let mask_bytes = mask.len().saturating_mul(std::mem::size_of::<u32>()).max(1);
        let max_entries = (MASK_CACHE_BUDGET_BYTES / mask_bytes)
            .clamp(MIN_MASK_CACHE_ENTRIES, MAX_MASK_CACHE_ENTRIES);
        let mut cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.iter().any(|entry| entry.state == state) {
            return;
        }
        if cache.len() >= max_entries {
            cache.remove(0);
        }
        cache.push(DynamicMaskCacheEntry {
            state,
            mask: Arc::from(mask),
        });
    }
}

impl Default for DynamicMaskVocab {
    fn default() -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            canonical_original_token_offsets: Arc::new(vec![0]),
            canonical_original_tokens: Arc::new(Vec::new()),
            node_token_markers: Arc::new(vec![0]),
            subtree_original_token_offsets: Arc::new(vec![0]),
            subtree_original_tokens: Arc::new(Vec::new()),
            pending_source: None,
            initialized: false,
            mask_cache: Arc::new(Mutex::new(Vec::new())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            mask_tokenizer: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
        }
    }
}

/// Version-scoped serde for the inverse token-id map. Current sectioned
/// artifacts carry only `original_token_to_internal`; the inverse is exactly
/// derivable from it and is rebuilt after the core section is decoded. Older
/// artifact versions leave this mode disabled and retain their historical wire
/// shape.
pub(crate) mod internal_token_inverse_artifact_serde {
    use std::cell::Cell;

    use serde::{Deserialize, Serialize};

    thread_local! {
        static OMIT_INVERSE: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_omit(enabled: bool) -> bool {
        OMIT_INVERSE.with(|mode| mode.replace(enabled))
    }

    pub fn serialize<S>(value: &[Vec<u32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if OMIT_INVERSE.with(Cell::get) {
            return ().serialize(serializer);
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if OMIT_INVERSE.with(Cell::get) {
            <()>::deserialize(deserializer)?;
            return Ok(Vec::new());
        }
        Vec::<Vec<u32>>::deserialize(deserializer)
    }
}

/// Current-core serialization can omit the tokenizer-state inverse when it is
/// exactly derivable from the scalar state -> internal-TSID map. The default
/// mode preserves the historical `Vec<Vec<u32>>` bincode wire, so legacy
/// artifact decoding is unchanged.
pub(crate) mod internal_tsid_inverse_artifact_serde {
    use std::cell::Cell;

    use serde::{Deserialize, Serialize};

    thread_local! {
        static OMIT_INVERSE: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_omit(enabled: bool) -> bool {
        OMIT_INVERSE.with(|mode| mode.replace(enabled))
    }

    pub fn serialize<S>(value: &[Vec<u32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if OMIT_INVERSE.with(Cell::get) {
            return ().serialize(serializer);
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if OMIT_INVERSE.with(Cell::get) {
            <()>::deserialize(deserializer)?;
            return Ok(Vec::new());
        }
        Vec::<Vec<u32>>::deserialize(deserializer)
    }
}

/// Compact v14+ wire form for the dense original-token -> internal-token map.
/// Internal IDs are normally only a few thousand wide even for 128k-token
/// vocabularies, so fixed-width u32 storage wastes roughly half this field.
/// Zero encodes the historical `u32::MAX` sentinel; ordinary IDs are stored as
/// `id + 1` varints. Older artifact versions leave this mode disabled.
pub(crate) mod original_token_map_artifact_serde {
    use std::cell::Cell;
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    const VARINT_MAGIC: &[u8; 4] = b"OTM1";
    const FIXED_MAGIC: &[u8; 4] = b"OTM2";
    const FIXED_HEADER_LEN: usize = FIXED_MAGIC.len() + 1 + 4;

    #[derive(Debug)]
    pub(crate) struct PackedOriginalTokenMap {
        backing: Arc<Vec<u8>>,
        payload_start: usize,
        count: usize,
        width: usize,
    }

    impl PackedOriginalTokenMap {
        pub(crate) fn parse_backed(
            backing: Arc<Vec<u8>>,
            start: usize,
            len: usize,
        ) -> Result<Self, String> {
            let end = start
                .checked_add(len)
                .ok_or_else(|| "fixed original-token map range overflows".to_owned())?;
            let input = backing
                .get(start..end)
                .ok_or_else(|| "fixed original-token map is outside artifact backing".to_owned())?;
            if input.len() < FIXED_HEADER_LEN || !input.starts_with(FIXED_MAGIC) {
                return Err("invalid fixed original-token map header".to_owned());
            }
            let width = input[FIXED_MAGIC.len()] as usize;
            if !matches!(width, 1 | 2 | 4) {
                return Err("invalid fixed original-token map width".to_owned());
            }
            let count_start = FIXED_MAGIC.len() + 1;
            let count = u32::from_le_bytes(
                input[count_start..count_start + 4]
                    .try_into()
                    .expect("fixed original-token count has fixed width"),
            ) as usize;
            let payload_len = count
                .checked_mul(width)
                .ok_or_else(|| "fixed original-token map payload overflows".to_owned())?;
            if FIXED_HEADER_LEN
                .checked_add(payload_len)
                .is_none_or(|expected| expected != input.len())
            {
                return Err("invalid fixed original-token map length".to_owned());
            }
            Ok(Self {
                backing,
                payload_start: start + FIXED_HEADER_LEN,
                count,
                width,
            })
        }

        #[inline]
        pub(crate) fn len(&self) -> usize {
            self.count
        }

        #[inline]
        pub(crate) fn is_empty(&self) -> bool {
            self.count == 0
        }

        #[inline]
        pub(crate) fn get(&self, index: usize) -> Option<u32> {
            if index >= self.count {
                return None;
            }
            let start = self.payload_start + index * self.width;
            let bytes = self.backing.get(start..start + self.width)?;
            Some(match self.width {
                1 => {
                    let value = bytes[0];
                    if value == u8::MAX { u32::MAX } else { value as u32 }
                }
                2 => {
                    let value = u16::from_le_bytes([bytes[0], bytes[1]]);
                    if value == u16::MAX { u32::MAX } else { value as u32 }
                }
                4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _ => unreachable!(),
            })
        }

        pub(crate) fn materialize(&self) -> Vec<u32> {
            (0..self.count)
                .map(|index| self.get(index).expect("validated packed original-token map index"))
                .collect()
        }
    }

    thread_local! {
        static PACKED: Cell<bool> = const { Cell::new(false) };
        static EXTERNAL: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_packed(enabled: bool) -> bool {
        PACKED.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn set_external(enabled: bool) -> bool {
        EXTERNAL.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn to_fast_bytes(value: &[u32]) -> Vec<u8> {
        pack_fixed(value)
    }

    pub(crate) fn from_fast_bytes(input: &[u8]) -> Result<Vec<u32>, String> {
        unpack(input)
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
    fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
        let mut value = 0u32;
        let mut shift = 0u32;
        for _ in 0..5 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed original-token map".to_owned())?;
            *pos += 1;
            if shift == 28 && byte > 0x0f {
                return Err("overflowing packed original-token map".to_owned());
            }
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err("overflowing packed original-token map".to_owned())
    }

    fn pack_varint(value: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(value.len().saturating_mul(2));
        out.extend_from_slice(VARINT_MAGIC);
        put_var_u32(
            &mut out,
            u32::try_from(value.len()).expect("token vocabulary should fit u32"),
        );
        for &internal in value {
            let encoded = if internal == u32::MAX {
                0
            } else {
                internal
                    .checked_add(1)
                    .expect("u32::MAX is reserved as the unmapped-token sentinel")
            };
            put_var_u32(&mut out, encoded);
        }
        out
    }

    fn unpack_varint(input: &[u8]) -> Result<Vec<u32>, String> {
        if !input.starts_with(VARINT_MAGIC) {
            return Err("invalid varint original-token map header".to_owned());
        }
        let mut pos = VARINT_MAGIC.len();
        let count = take_var_u32(input, &mut pos)? as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let encoded = take_var_u32(input, &mut pos)?;
            out.push(if encoded == 0 { u32::MAX } else { encoded - 1 });
        }
        if pos != input.len() {
            return Err("trailing bytes in packed original-token map".to_owned());
        }
        Ok(out)
    }

    fn pack_fixed(value: &[u32]) -> Vec<u8> {
        let max_internal = value
            .iter()
            .copied()
            .filter(|&internal| internal != u32::MAX)
            .max()
            .unwrap_or(0);
        let width = if max_internal < u8::MAX as u32 {
            1u8
        } else if max_internal < u16::MAX as u32 {
            2u8
        } else {
            4u8
        };
        let payload_len = value
            .len()
            .checked_mul(width as usize)
            .expect("original-token map payload should fit usize");
        let mut out = Vec::with_capacity(FIXED_HEADER_LEN + payload_len);
        out.extend_from_slice(FIXED_MAGIC);
        out.push(width);
        out.extend_from_slice(
            &u32::try_from(value.len())
                .expect("token vocabulary should fit u32")
                .to_le_bytes(),
        );
        match width {
            1 => {
                for &internal in value {
                    out.push(if internal == u32::MAX {
                        u8::MAX
                    } else {
                        internal as u8
                    });
                }
            }
            2 => {
                for &internal in value {
                    let encoded = if internal == u32::MAX {
                        u16::MAX
                    } else {
                        internal as u16
                    };
                    out.extend_from_slice(&encoded.to_le_bytes());
                }
            }
            4 => {
                for &internal in value {
                    out.extend_from_slice(&internal.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        out
    }

    fn unpack_fixed(input: &[u8]) -> Result<Vec<u32>, String> {
        if input.len() < FIXED_HEADER_LEN || !input.starts_with(FIXED_MAGIC) {
            return Err("invalid fixed original-token map header".to_owned());
        }
        let width = input[FIXED_MAGIC.len()] as usize;
        if !matches!(width, 1 | 2 | 4) {
            return Err("invalid fixed original-token map width".to_owned());
        }
        let count_start = FIXED_MAGIC.len() + 1;
        let count = u32::from_le_bytes(
            input[count_start..count_start + 4]
                .try_into()
                .expect("fixed original-token count has fixed width"),
        ) as usize;
        let payload_len = count
            .checked_mul(width)
            .ok_or_else(|| "fixed original-token map payload overflows".to_owned())?;
        if FIXED_HEADER_LEN
            .checked_add(payload_len)
            .is_none_or(|expected| expected != input.len())
        {
            return Err("invalid fixed original-token map length".to_owned());
        }
        let payload = &input[FIXED_HEADER_LEN..];
        let mut out = Vec::with_capacity(count);
        match width {
            1 => out.extend(payload.iter().map(|&encoded| {
                if encoded == u8::MAX {
                    u32::MAX
                } else {
                    encoded as u32
                }
            })),
            2 => out.extend(payload.chunks_exact(2).map(|bytes| {
                let encoded = u16::from_le_bytes([bytes[0], bytes[1]]);
                if encoded == u16::MAX {
                    u32::MAX
                } else {
                    encoded as u32
                }
            })),
            4 => out.extend(payload.chunks_exact(4).map(|bytes| {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            })),
            _ => unreachable!(),
        }
        Ok(out)
    }

    fn unpack(input: &[u8]) -> Result<Vec<u32>, String> {
        if input.starts_with(FIXED_MAGIC) {
            unpack_fixed(input)
        } else if input.starts_with(VARINT_MAGIC) {
            unpack_varint(input)
        } else {
            Err("invalid packed original-token map header".to_owned())
        }
    }

    pub fn serialize<S>(value: &[u32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if EXTERNAL.with(Cell::get) {
            return 0u8.serialize(serializer);
        }
        if !PACKED.with(Cell::get) {
            return value.serialize(serializer);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let pack_started = profile.then(std::time::Instant::now);
        let packed = pack_fixed(value);
        let pack_ms = pack_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        let wire_bytes = packed.len();
        let result = packed.serialize(serializer);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][original_token_map_encode] entries={} wire_bytes={} pack_ms={:.3} total_ms={:.3}",
                value.len(),
                wire_bytes,
                pack_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if EXTERNAL.with(Cell::get) {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom(
                    "invalid external original-token map placeholder",
                ));
            }
            return Ok(Vec::new());
        }
        if !PACKED.with(Cell::get) {
            return Vec::<u32>::deserialize(deserializer);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let packed = Vec::<u8>::deserialize(deserializer)?;
        let packed_len = packed.len();
        let result = unpack(&packed).map_err(serde::de::Error::custom);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][original_token_map_decode] wire_bytes={} ms={:.3}",
                packed_len,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fixed_original_token_map_roundtrips_all_widths_and_legacy() {
            for values in [
                vec![0, 12, u32::MAX, 254],
                vec![0, 255, 4096, u32::MAX, 65534],
                vec![0, 65535, 1_000_000, u32::MAX],
            ] {
                let packed = pack_fixed(&values);
                assert_eq!(unpack(&packed).unwrap(), values);

                let legacy = pack_varint(&values);
                assert_eq!(unpack(&legacy).unwrap(), values);
            }
        }
    }
}

/// Compact v14+ wire encoding for the immutable model-token byte vocabulary.
/// Ordinary LLM vocabs are dense in token id, so the historical
/// `BTreeMap<u32, Vec<u8>>` representation spends more bytes on map keys and
/// per-Vec lengths than on useful token data. The packed form stores a dense
/// sequence of varint lengths followed by token bytes, with a sparse fallback
/// for unusual vocabularies. Deserialization reconstructs the exact historical
/// in-memory BTreeMap, so compiler/composition/runtime APIs do not change.
pub(crate) mod token_bytes_artifact_serde {
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    const LEGACY_MAGIC: &[u8; 4] = b"TBP1";
    const INDEXED_MAGIC: &[u8; 4] = b"TBP2";
    const INDEXED_HEADER_LEN: usize = INDEXED_MAGIC.len() + 1 + 4;

    thread_local! {
        static PACKED: Cell<bool> = const { Cell::new(false) };
        static DEFER_UNPACK: Cell<bool> = const { Cell::new(false) };
        static EXTERNAL: Cell<bool> = const { Cell::new(false) };
        static DEFERRED: RefCell<Option<Arc<PackedTokenBytes>>> = const { RefCell::new(None) };
    }

    pub(crate) fn set_packed(enabled: bool) -> bool {
        PACKED.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn set_defer_unpack(enabled: bool) -> bool {
        DEFER_UNPACK.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn set_external(enabled: bool) -> bool {
        EXTERNAL.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn take_deferred() -> Option<Arc<PackedTokenBytes>> {
        DEFERRED.with(|slot| slot.borrow_mut().take())
    }

    #[derive(Debug)]
    pub(crate) struct PackedTokenBytes {
        wire: Arc<Vec<u8>>,
        wire_start: usize,
        wire_len: usize,
        indexed: Option<PackedTokenBytesIndexed>,
        spans: Box<[(u32, u32)]>,
        sparse_ids: Option<Box<[u32]>>,
    }

    #[derive(Debug, Clone, Copy)]
    struct PackedTokenBytesIndexed {
        count: usize,
        sparse_ids_start: Option<usize>,
        offsets_start: usize,
        data_start: usize,
    }

    impl PackedTokenBytes {
        pub(crate) fn from_runtime_entries(value: &BTreeMap<u32, Vec<u8>>) -> Result<Self, String> {
            // The indexed representation is useful runtime state in its own
            // right: token lookup/iteration reads it directly. Build it once
            // when a compiler-created Constraint is finalized rather than
            // rebuilding the same index inside every save().
            Self::parse(pack(value))
        }

        fn parse(wire: Vec<u8>) -> Result<Self, String> {
            let wire_len = wire.len();
            Self::parse_backed(Arc::new(wire), 0, wire_len)
        }

        pub(crate) fn parse_backed(
            wire: Arc<Vec<u8>>,
            wire_start: usize,
            wire_len: usize,
        ) -> Result<Self, String> {
            let wire_end = wire_start
                .checked_add(wire_len)
                .ok_or_else(|| "overflowing packed token-byte backing range".to_owned())?;
            let input = wire
                .get(wire_start..wire_end)
                .ok_or_else(|| "packed token-byte backing range is out of bounds".to_owned())?;
            if input.starts_with(INDEXED_MAGIC) {
                return Self::parse_indexed_backed(wire, wire_start, wire_len);
            }
            if !input.starts_with(LEGACY_MAGIC) {
                return Err("invalid packed token-byte header".to_owned());
            }
            let mut pos = LEGACY_MAGIC.len();
            let sparse = match input.get(pos).copied() {
                Some(0) => false,
                Some(1) => true,
                _ => return Err("invalid packed token-byte mode".to_owned()),
            };
            pos += 1;
            let count = take_var_u32(input, &mut pos)? as usize;
            let mut spans = Vec::with_capacity(count);
            let mut sparse_ids = sparse.then(|| Vec::with_capacity(count));
            let mut previous_end = 0u64;
            for dense_id in 0..count {
                let id = if sparse {
                    let gap = take_var_u32(input, &mut pos)? as u64;
                    let id = previous_end
                        .checked_add(gap)
                        .ok_or_else(|| "overflowing packed token id".to_owned())?;
                    let id = u32::try_from(id)
                        .map_err(|_| "overflowing packed token id".to_owned())?;
                    previous_end = id as u64 + 1;
                    sparse_ids.as_mut().expect("sparse ids enabled").push(id);
                    id
                } else {
                    u32::try_from(dense_id)
                        .map_err(|_| "dense packed token id exceeds u32".to_owned())?
                };
                let _ = id;
                let len = take_var_u32(input, &mut pos)? as usize;
                let start = pos;
                let end = start
                    .checked_add(len)
                    .ok_or_else(|| "overflowing packed token-byte length".to_owned())?;
                if end > input.len() {
                    return Err("truncated packed token bytes".to_owned());
                }
                spans.push((
                    u32::try_from(start)
                        .map_err(|_| "packed token byte offset exceeds u32".to_owned())?,
                    u32::try_from(len)
                        .map_err(|_| "packed token byte length exceeds u32".to_owned())?,
                ));
                pos = end;
            }
            if pos != input.len() {
                return Err("trailing bytes in packed token-byte vocabulary".to_owned());
            }
            Ok(Self {
                wire,
                wire_start,
                wire_len,
                indexed: None,
                spans: spans.into_boxed_slice(),
                sparse_ids: sparse_ids.map(Vec::into_boxed_slice),
            })
        }

        fn parse_indexed_backed(
            wire: Arc<Vec<u8>>,
            wire_start: usize,
            wire_len: usize,
        ) -> Result<Self, String> {
            let wire_end = wire_start
                .checked_add(wire_len)
                .ok_or_else(|| "overflowing indexed token-byte backing range".to_owned())?;
            let input = wire
                .get(wire_start..wire_end)
                .ok_or_else(|| "indexed token-byte backing range is out of bounds".to_owned())?;
            if input.len() < INDEXED_HEADER_LEN || !input.starts_with(INDEXED_MAGIC) {
                return Err("invalid indexed token-byte header".to_owned());
            }
            let sparse = match input[INDEXED_MAGIC.len()] {
                0 => false,
                1 => true,
                _ => return Err("invalid indexed token-byte mode".to_owned()),
            };
            let count_start = INDEXED_MAGIC.len() + 1;
            let count = u32::from_le_bytes(
                input[count_start..count_start + 4]
                    .try_into()
                    .expect("indexed token count has fixed width"),
            ) as usize;
            let sparse_ids_start = sparse.then_some(INDEXED_HEADER_LEN);
            let ids_bytes = if sparse {
                count
                    .checked_mul(4)
                    .ok_or_else(|| "indexed token-id table overflows".to_owned())?
            } else {
                0
            };
            let offsets_start = INDEXED_HEADER_LEN
                .checked_add(ids_bytes)
                .ok_or_else(|| "indexed token-byte offsets start overflows".to_owned())?;
            let offsets_bytes = count
                .checked_add(1)
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| "indexed token-byte offset table overflows".to_owned())?;
            let data_start = offsets_start
                .checked_add(offsets_bytes)
                .ok_or_else(|| "indexed token-byte data start overflows".to_owned())?;
            if data_start > input.len() {
                return Err("truncated indexed token-byte tables".to_owned());
            }
            let first_offset = read_u32_at(input, offsets_start)? as usize;
            let final_offset = read_u32_at(input, offsets_start + count * 4)? as usize;
            if first_offset != 0 || final_offset != input.len() - data_start {
                return Err("invalid indexed token-byte offset bounds".to_owned());
            }
            Ok(Self {
                wire,
                wire_start,
                wire_len,
                indexed: Some(PackedTokenBytesIndexed {
                    count,
                    sparse_ids_start,
                    offsets_start,
                    data_start,
                }),
                spans: Box::new([]),
                sparse_ids: None,
            })
        }

        #[inline]
        pub(crate) fn wire(&self) -> &[u8] {
            &self.wire[self.wire_start..self.wire_start + self.wire_len]
        }

        pub(crate) fn whole_wire_arc(&self) -> Option<Arc<Vec<u8>>> {
            (self.wire_start == 0 && self.wire_len == self.wire.len())
                .then(|| Arc::clone(&self.wire))
        }

        #[inline]
        pub(crate) fn len(&self) -> usize {
            self.indexed.map_or(self.spans.len(), |indexed| indexed.count)
        }

        #[inline]
        pub(crate) fn get(&self, token_id: u32) -> Option<&[u8]> {
            if let Some(indexed) = self.indexed {
                let index = match indexed.sparse_ids_start {
                    None => usize::try_from(token_id)
                        .ok()
                        .filter(|&index| index < indexed.count)?,
                    Some(_) => self.indexed_sparse_token_index(indexed, token_id)?,
                };
                return self.indexed_bytes_at(indexed, index);
            }
            let index = match &self.sparse_ids {
                None => usize::try_from(token_id).ok().filter(|&index| index < self.spans.len())?,
                Some(ids) => ids.binary_search(&token_id).ok()?,
            };
            let (start, len) = self.spans[index];
            let start = start as usize;
            self.wire().get(start..start + len as usize)
        }

        pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> + '_ {
            (0..self.len()).map(|index| {
                let token_id = self
                    .token_id_at(index)
                    .expect("validated packed token index should have an id");
                let bytes = self
                    .bytes_at_index(index)
                    .expect("validated packed token index should have bytes");
                (token_id, bytes)
            })
        }

        pub(crate) fn max_token_id(&self) -> Option<u32> {
            if self.indexed.is_some() {
                return self.len().checked_sub(1).and_then(|index| self.token_id_at(index));
            }
            match &self.sparse_ids {
                Some(ids) => ids.last().copied(),
                None => self.spans.len().checked_sub(1).map(|id| id as u32),
            }
        }

        pub(crate) fn materialize(&self) -> Arc<BTreeMap<u32, Vec<u8>>> {
            Arc::new(
                self.iter()
                    .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                    .collect(),
            )
        }

        fn token_id_at(&self, index: usize) -> Option<u32> {
            if let Some(indexed) = self.indexed {
                if index >= indexed.count {
                    return None;
                }
                return indexed.sparse_ids_start.map_or_else(
                    || u32::try_from(index).ok(),
                    |start| read_u32_at(self.wire(), start + index * 4).ok(),
                );
            }
            self.sparse_ids
                .as_ref()
                .map_or_else(|| u32::try_from(index).ok(), |ids| ids.get(index).copied())
        }

        fn bytes_at_index(&self, index: usize) -> Option<&[u8]> {
            if let Some(indexed) = self.indexed {
                return self.indexed_bytes_at(indexed, index);
            }
            let &(start, len) = self.spans.get(index)?;
            let start = start as usize;
            self.wire().get(start..start + len as usize)
        }

        fn indexed_bytes_at(
            &self,
            indexed: PackedTokenBytesIndexed,
            index: usize,
        ) -> Option<&[u8]> {
            if index >= indexed.count {
                return None;
            }
            let wire = self.wire();
            let start = read_u32_at(wire, indexed.offsets_start + index * 4).ok()? as usize;
            let end = read_u32_at(wire, indexed.offsets_start + (index + 1) * 4).ok()? as usize;
            if start > end {
                return None;
            }
            wire.get(indexed.data_start + start..indexed.data_start + end)
        }

        fn indexed_sparse_token_index(
            &self,
            indexed: PackedTokenBytesIndexed,
            token_id: u32,
        ) -> Option<usize> {
            let start = indexed.sparse_ids_start?;
            let wire = self.wire();
            let mut lo = 0usize;
            let mut hi = indexed.count;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let candidate = read_u32_at(wire, start + mid * 4).ok()?;
                match candidate.cmp(&token_id) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                    std::cmp::Ordering::Equal => return Some(mid),
                }
            }
            None
        }
    }

    #[inline]
    fn read_u32_at(input: &[u8], pos: usize) -> Result<u32, String> {
        let end = pos
            .checked_add(4)
            .ok_or_else(|| "packed token-byte u32 offset overflows".to_owned())?;
        let bytes = input
            .get(pos..end)
            .ok_or_else(|| "truncated packed token-byte u32".to_owned())?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("packed token u32 has fixed width"),
        ))
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
    fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
        let mut value = 0u32;
        let mut shift = 0u32;
        for _ in 0..5 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed token-byte varint".to_owned())?;
            *pos += 1;
            if shift == 28 && byte > 0x0f {
                return Err("overflowing packed token-byte varint".to_owned());
            }
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err("overflowing packed token-byte varint".to_owned())
    }

    fn pack_legacy(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
        let dense = value
            .keys()
            .copied()
            .enumerate()
            .all(|(expected, actual)| actual as usize == expected);
        let mut out = Vec::new();
        out.extend_from_slice(LEGACY_MAGIC);
        out.push(u8::from(!dense));
        put_var_u32(
            &mut out,
            u32::try_from(value.len()).expect("token vocabulary should fit u32"),
        );
        let mut previous_end = 0u64;
        for (&id, bytes) in value {
            if !dense {
                let gap = (id as u64)
                    .checked_sub(previous_end)
                    .expect("token ids are sorted");
                put_var_u32(
                    &mut out,
                    u32::try_from(gap).expect("token-id gap should fit u32"),
                );
                previous_end = id as u64 + 1;
            }
            put_var_u32(
                &mut out,
                u32::try_from(bytes.len()).expect("token byte length should fit u32"),
            );
            out.extend_from_slice(bytes);
        }
        out
    }

    fn pack(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
        let dense = value
            .keys()
            .copied()
            .enumerate()
            .all(|(expected, actual)| actual as usize == expected);
        let count = value.len();
        let data_len = value.values().try_fold(0usize, |total, bytes| {
            total.checked_add(bytes.len())
        });
        let Some(data_len) = data_len.filter(|&len| u32::try_from(len).is_ok()) else {
            return pack_legacy(value);
        };
        let ids_len = if dense { 0 } else { count.saturating_mul(4) };
        let Some(offsets_len) = count.checked_add(1).and_then(|count| count.checked_mul(4)) else {
            return pack_legacy(value);
        };
        let capacity = INDEXED_HEADER_LEN
            .checked_add(ids_len)
            .and_then(|len| len.checked_add(offsets_len))
            .and_then(|len| len.checked_add(data_len));
        let Some(capacity) = capacity else {
            return pack_legacy(value);
        };
        // Reserve the complete indexed representation up front and fill the
        // id/offset/data regions in one BTreeMap traversal.  The previous
        // implementation walked the pointer-heavy map once for offsets and a
        // second time for bytes; on 100k+ token vocabularies that dominated a
        // genuinely fresh Constraint::save().
        let mut out = vec![0u8; capacity];
        out[..INDEXED_MAGIC.len()].copy_from_slice(INDEXED_MAGIC);
        out[INDEXED_MAGIC.len()] = u8::from(!dense);
        out[INDEXED_MAGIC.len() + 1..INDEXED_HEADER_LEN].copy_from_slice(
            &u32::try_from(count)
                .expect("token vocabulary should fit u32")
                .to_le_bytes(),
        );

        let ids_start = INDEXED_HEADER_LEN;
        let offsets_start = ids_start + ids_len;
        let data_start = offsets_start + offsets_len;
        out[offsets_start..offsets_start + 4].copy_from_slice(&0u32.to_le_bytes());

        let mut offset = 0u32;
        let mut data_pos = data_start;
        for (index, (&token_id, bytes)) in value.iter().enumerate() {
            if !dense {
                let id_pos = ids_start + index * 4;
                out[id_pos..id_pos + 4].copy_from_slice(&token_id.to_le_bytes());
            }
            let next_offset = offset
                .checked_add(bytes.len() as u32)
                .expect("indexed token-byte data length was prevalidated");
            let offset_pos = offsets_start + (index + 1) * 4;
            out[offset_pos..offset_pos + 4].copy_from_slice(&next_offset.to_le_bytes());
            let data_end = data_pos + bytes.len();
            out[data_pos..data_end].copy_from_slice(bytes);
            data_pos = data_end;
            offset = next_offset;
        }
        debug_assert_eq!(data_pos, capacity);
        out
    }

    pub(crate) fn pack_external(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
        pack(value)
    }

    fn unpack(input: &[u8]) -> Result<Arc<BTreeMap<u32, Vec<u8>>>, String> {
        if input.starts_with(INDEXED_MAGIC) {
            return PackedTokenBytes::parse(input.to_vec()).map(|packed| packed.materialize());
        }
        if !input.starts_with(LEGACY_MAGIC) {
            return Err("invalid packed token-byte header".to_owned());
        }
        let mut pos = LEGACY_MAGIC.len();
        let sparse = match input.get(pos).copied() {
            Some(0) => false,
            Some(1) => true,
            _ => return Err("invalid packed token-byte mode".to_owned()),
        };
        pos += 1;
        let count = take_var_u32(input, &mut pos)? as usize;
        let mut map = BTreeMap::new();
        let mut previous_end = 0u64;
        for dense_id in 0..count {
            let id = if sparse {
                let gap = take_var_u32(input, &mut pos)? as u64;
                let id = previous_end
                    .checked_add(gap)
                    .ok_or_else(|| "overflowing packed token id".to_owned())?;
                let id = u32::try_from(id)
                    .map_err(|_| "overflowing packed token id".to_owned())?;
                previous_end = id as u64 + 1;
                id
            } else {
                u32::try_from(dense_id)
                    .map_err(|_| "dense packed token id exceeds u32".to_owned())?
            };
            let len = take_var_u32(input, &mut pos)? as usize;
            let end = pos
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-byte length".to_owned())?;
            let bytes = input
                .get(pos..end)
                .ok_or_else(|| "truncated packed token bytes".to_owned())?
                .to_vec();
            pos = end;
            if map.insert(id, bytes).is_some() {
                return Err("duplicate packed token id".to_owned());
            }
        }
        if pos != input.len() {
            return Err("trailing bytes in packed token-byte vocabulary".to_owned());
        }
        Ok(Arc::new(map))
    }

    pub fn serialize<S>(
        value: &Arc<BTreeMap<u32, Vec<u8>>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if !PACKED.with(Cell::get) {
            return value.serialize(serializer);
        }
        if EXTERNAL.with(Cell::get) {
            return Vec::<u8>::new().serialize(serializer);
        }
        pack(value).serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Arc<BTreeMap<u32, Vec<u8>>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if !PACKED.with(Cell::get) {
            return Arc::<BTreeMap<u32, Vec<u8>>>::deserialize(deserializer);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total = profile.then(std::time::Instant::now);
        let packed_started = profile.then(std::time::Instant::now);
        let packed = Vec::<u8>::deserialize(deserializer)?;
        let packed_len = packed.len();
        let packed_ms = packed_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        if EXTERNAL.with(Cell::get) {
            if !packed.is_empty() {
                return Err(serde::de::Error::custom(
                    "external packed token-byte placeholder must be empty",
                ));
            }
            return Ok(Arc::new(BTreeMap::new()));
        }
        let unpack_started = profile.then(std::time::Instant::now);
        let result = if DEFER_UNPACK.with(Cell::get) {
            let deferred = PackedTokenBytes::parse(packed)
                .map(Arc::new)
                .map_err(serde::de::Error::custom)?;
            DEFERRED.with(|slot| *slot.borrow_mut() = Some(deferred));
            Ok(Arc::new(BTreeMap::new()))
        } else {
            unpack(&packed).map_err(serde::de::Error::custom)
        };
        if let Some(total) = total {
            eprintln!(
                "[glrmask/profile][token_bytes_decode] wire_bytes={} vec_ms={packed_ms:.3} unpack_ms={:.3} total_ms={:.3}",
                packed_len,
                unpack_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0),
                total.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn check_roundtrip(value: BTreeMap<u32, Vec<u8>>) {
            let packed = pack(&value);
            assert!(packed.starts_with(INDEXED_MAGIC));
            let view = PackedTokenBytes::parse(packed).unwrap();
            assert_eq!(view.len(), value.len());
            assert_eq!(
                view.iter()
                    .map(|(id, bytes)| (id, bytes.to_vec()))
                    .collect::<BTreeMap<_, _>>(),
                value
            );
            for (&id, bytes) in &value {
                assert_eq!(view.get(id), Some(bytes.as_slice()));
            }
            assert_eq!(view.max_token_id(), value.keys().next_back().copied());

            let legacy = pack_legacy(&value);
            let legacy_view = PackedTokenBytes::parse(legacy).unwrap();
            assert_eq!(
                legacy_view
                    .iter()
                    .map(|(id, bytes)| (id, bytes.to_vec()))
                    .collect::<BTreeMap<_, _>>(),
                value
            );
        }

        #[test]
        fn indexed_token_bytes_roundtrip_dense_and_sparse() {
            check_roundtrip(BTreeMap::from([
                (0, b"a".to_vec()),
                (1, b"bc".to_vec()),
                (2, Vec::new()),
            ]));
            check_roundtrip(BTreeMap::from([
                (2, b"a".to_vec()),
                (9, b"bc".to_vec()),
                (1000, b"xyz".to_vec()),
            ]));
        }
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

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct StaticDynamicOverlayMetadata {
    /// Global terminal-id offsets for the transported composition components.
    pub(crate) terminal_offsets: Vec<u32>,
    /// Global raw-tokenizer-state offsets for those same components. State zero
    /// is the merged reset dispatcher and deliberately belongs to no component.
    pub(crate) tokenizer_state_offsets: Vec<u32>,
    /// Terminals whose composed parser template has behavior absent from the
    /// transported component parser artifacts (including scoped-ignore repair
    /// and conservative unsafe terminals).
    pub(crate) repair_terminals: Vec<bool>,
    /// Composed LR states which belong to one or more child components but not
    /// to the parent component. Runtime lookahead-return factoring is useful
    /// only while the concrete top state is still inside such a child-owned
    /// region; ordinary parent reductions must not pay for that machinery.
    #[serde(default)]
    pub(crate) non_parent_only_parser_states: Vec<bool>,
    /// Experimental exact segmented parser backend. Each source constraint
    /// retains its own parser-DWA/token coordinate and is projected from the
    /// composed tokenizer/LR coordinates at mask time. This deliberately stays
    /// runtime-only until the representation is validated and compacted; the
    /// ordinary flattened parser artifact remains the serialization fallback.
    #[serde(skip, default)]
    pub(crate) segmented_parser_components: Vec<SegmentedParserComponent>,
    /// Compressed deterministic union root for `segmented_parser_components`.
    /// Entry `g` is the unique component selected by composed LR state `g`, or
    /// `u32::MAX` when no component has a root transition on that state. When
    /// non-empty, the component collection is one deterministic parser DWA in
    /// segmented storage: a synthetic root followed by one cached component
    /// body. No runtime parser-NWA branching is involved.
    #[serde(skip, default)]
    pub(crate) segmented_component_union_root_dispatch: Vec<u32>,
    #[serde(skip, default)]
    pub(crate) segmented_boundary_parser: Option<Box<SegmentedBoundaryParser>>,
    #[serde(skip, default)]
    pub(crate) segmented_boundary_terminal_trie: Option<Box<SegmentedBoundaryTerminalTrie>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedParserComponent {
    pub(crate) constraint: Arc<Constraint>,
    pub(crate) tokenizer_state_offset: u32,
    pub(crate) terminal_offset: u32,
    /// Terminal to suppress only on the synthetic union-root empty-stack
    /// projection. Shared component artifacts retain their standalone start
    /// final weight; this root-only disallow is exactly the old cloned-artifact
    /// start-final subtraction without mutating the shared parser DWA.
    pub(crate) root_disallowed_terminal: Option<u32>,
    pub(crate) global_to_local_parser_state: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalTrieNode {
    pub(crate) children: Vec<(u32, u32)>,
    /// Bit i means private boundary token class i is accepted at this node.
    pub(crate) outputs: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryTerminalTrie {
    pub(crate) nodes: Vec<BoundaryTerminalTrieNode>,
    pub(crate) root_by_tsid: Vec<u32>,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryParser {
    /// Generic wire/reference representation. Compact in-memory boundary
    /// parsers leave this as the empty one-state DWA and use
    /// `compact_parser_dwa`; V17 serialization materializes the compact machine
    /// back into this exact generic wire shape.
    pub(crate) parser_dwa: DWA,
    #[serde(skip, default)]
    pub(crate) compact_parser_dwa:
        Option<crate::compiler::stages::parser_dwa::SmallBoundaryDwa>,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
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
        }
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

/// Fully compiled, immutable grammar constraint.
///
/// A `Constraint` is intended to be reused across generated sequences. Call
/// [`Constraint::start`] to create a mutable per-sequence state.
#[derive(Debug, Clone)]
pub struct Constraint {
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    /// Runtime-derived exact original-token sets for `Skip` terminals in a
    /// composed grammar. Each token is wholly in `L(skip)+`: it can be
    /// consumed as one or more complete instances of that scoped-ignore
    /// terminal with a lexer reset between instances. This is deliberately
    /// not serialized; it is cheap to rebuild from the retained terminal
    /// expression and vocabulary and therefore does not change artifact wire
    /// compatibility.
    pub(crate) scoped_ignore_only_tokens: Vec<(TerminalID, Box<[u32]>)>,
    /// Exact byte-token fusions `(fused, suffix)` grouped by scoped Skip. The
    /// fused token begins with one or more complete instances of the Skip
    /// language and the remaining bytes equal `suffix` exactly. If `suffix`
    /// is admitted by the ordinary static mask, `fused` is therefore admitted
    /// as well. Runtime-only for the same wire-compatibility reason as above.
    pub(crate) scoped_ignore_prefix_fusions: Vec<(TerminalID, Box<[(u32, u32)]>)>,
    pub(crate) parser_dwa: DWA,
    /// Current-format loaded constraints retain the immutable parser DWA in
    /// its compact canonical pools instead of reconstructing RangeSet/Weight
    /// objects. Compiler-created and legacy-loaded constraints leave this
    /// empty and use `parser_dwa` directly.
    pub(crate) packed_parser_dwa:
        Option<Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>>,
    /// Runtime-only override for the parser-DWA start final. Composition uses
    /// this to suppress a component's standalone globally-erased ignore at the
    /// union root without materializing or mutating the full parser DWA.
    pub(crate) parser_start_final_override: Option<Weight>,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    /// Uncombined exact depth-one acceptance parts. Direct-regular grammars
    /// retain terminal completion weights separately to avoid constructing one
    /// large union weight per parser state at compile time.
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<Weight>>,
    /// Immediate-completion L1 terminal weights for direct-regular parsers.
    /// Kept once per grammar terminal rather than duplicated across every
    /// epsilon-closed parser row.
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
    pub(crate) packed_non_dwa_weights: Option<Arc<PackedNonDwaWeights>>,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    pub(crate) direct_regular_wide_frontier_acceptance:
        Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    pub(crate) direct_regular_dynamic_hot_frontiers:
        Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    pub(crate) direct_regular_parser_state_acceptance:
        Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    pub(crate) table: GLRTable,
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Tokenizer,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    pub(crate) ignore_terminal: Option<TerminalID>,
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// `possible_matches_complete` to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Ordinary tokenizers have one internal TSID per physical state, making
    /// `internal_tsid_to_states` the exact bucket inverse of
    /// `state_to_internal_tsid`. Current artifacts can omit that redundant
    /// allocation and reconstruct it only for composition/debug paths.
    pub(crate) deferred_internal_tsid_to_states: OnceLock<Vec<Vec<u32>>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    /// Named unresolved `extern grammar` slots retained by a compiled parent.
    /// Values are parent-local hidden placeholder terminal IDs. Stored in the
    /// outer composition metadata so cached parents can be rebound after load.
    pub(crate) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    /// Composition-time parser stack-effect templates retained from the
    /// original compile. These are the unspecialized per-terminal DFAs used to
    /// build parser DWAs, so a later linker can transport unchanged component
    /// behavior instead of re-characterizing the component LR table.
    /// Stored in the outer versioned artifact envelope for compatibility with
    /// older inner `Constraint` bincode layouts.
    pub(crate) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    /// Composition-time symbolic parser characterizations retained from the
    /// original compile. A later linker can append only the boundary-induced
    /// reductions/rereductions and recompile affected terminal templates,
    /// rather than re-solving the component's reduction closure from scratch.
    pub(crate) composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    /// Runtime-only inverse lexer-metadata index used by compiled-constraint
    /// composition. Row `t` lists exactly the raw tokenizer states whose
    /// epsilon closure has terminal `t` matched or still reachable.
    pub(crate) terminal_live_states: Vec<Vec<u32>>,
    /// Runtime-only CSR view of the exact state -> internal-TSID relation.
    /// Ordinary tokenizers have one entry per state. A fully determinized
    /// runtime lexer may represent several old lexer states and therefore
    /// several independent TSID lanes in one physical state.
    pub(crate) state_internal_tsid_offsets: Vec<u32>,
    pub(crate) state_internal_tsids: Vec<u32>,
    /// Final-runtime subset states followed by an exact copy of the source
    /// tokenizer. `runtime_source_state_offset` is the boundary between the
    /// two coordinates. Empty metadata means no runtime-only determinization.
    pub(crate) runtime_source_state_offset: Option<u32>,
    /// CSR offsets for product-state -> exact source-state subset. There is one
    /// row per product state and therefore `product_state_count + 1` offsets.
    pub(crate) runtime_product_source_offsets: Vec<u32>,
    pub(crate) runtime_product_source_states: Vec<u32>,
    /// Scalar source representative for product states that are exactly one
    /// source state's epsilon closure; `u32::MAX` otherwise.
    pub(crate) runtime_product_exact_source_states: Vec<u32>,
    /// Runtime-only inverse used to re-coalesce a uniform source frontier.
    pub(crate) runtime_product_state_by_source_subset: FxHashMap<Box<[u32]>, u32>,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Runtime-only compact transition view for commit template products.
    pub(crate) fast_template_dfas_by_terminal: FastTemplateDfasByTerminal,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Current-format loads retain the fixed-width original-token map inside
    /// the owned artifact instead of expanding all model-token entries to
    /// `u32`. Ordinary static mask/commit performs direct packed lookups; only
    /// composition/debug-style bulk access materializes the vector lazily.
    pub(crate) packed_original_token_to_internal:
        Option<Arc<original_token_map_artifact_serde::PackedOriginalTokenMap>>,
    pub(crate) deferred_original_token_to_internal: OnceLock<Vec<u32>>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    /// Current-format loads can defer reconstructing the explicit inverse of
    /// `original_token_to_internal`. Static mask/commit only needs the internal
    /// token count and the already-serialized mask fragments; composition and
    /// token-space expansion materialize this inverse on first use.
    pub(crate) deferred_internal_token_to_tokens: OnceLock<Vec<Vec<u32>>>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Indexed immutable vocabulary used directly by runtime token lookup and
    /// iteration. Loaded constraints can point into artifact backing; compiled
    /// constraints own the same indexed representation alongside the source
    /// map used by compiler/composition code.
    pub(crate) packed_token_bytes: Option<Arc<token_bytes_artifact_serde::PackedTokenBytes>>,
    // Compiler-side scratch/result metadata only. No runtime or composition
    // path reads this field; composition rebuilds the map for its result when
    // needed. Persisting it duplicated token bytes inside every constraint.
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    pub(crate) pair_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    pub(crate) quad_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    pub(crate) super_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    pub(crate) mega_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    pub(crate) giga_word_group_buf_masks: DenseBufMaskRows,
    /// Sparse OR-union for each 64-token internal word group.
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    pub(crate) word_group_prefix_buf_masks: DenseBufMaskRows,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for quad groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    pub(crate) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for byte groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    pub(crate) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) word_group_sparse_total_entries: usize,
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    pub(crate) internal_token_dense_words: usize,
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_dense_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Final-weight token sets eligible for the direct sparse-intersection
    /// path. Their full output masks are intentionally not materialized: the
    /// runtime intersects them with the current dense state on every use.
    pub(crate) direct_sparse_weight_token_sets: DirectSparseWeightTokenSetCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    pub(crate) internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    /// Current IBM2 loads can retain the runtime-native flat sparse-mask slab
    /// directly inside the owned artifact instead of copying ~0.5-1 MiB.
    pub(crate) backed_internal_token_buf_flat: Option<BackedInternalTokenBufMasks>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy Ã— buf_len).
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    pub(crate) final_mask_mapping: FinalMaskMapping,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Serialized composition-only metadata (reset-token rows, parser template
    /// cache, symbolic characterizations, and grammar summary). Current-format
    /// loads keep this cold section backed by the artifact and materialize it
    /// only if the constraint is later used as a composition component.
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}

// Private Serde definition used only by the versioned artifact encoder/decoder.
// Keeping this remote definition separate prevents `Constraint` itself from
// implementing Serde, so `Constraint::save`/`Constraint::load` remain the only
// public persistence contract.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "Constraint")]
pub(crate) struct ConstraintSerde {
    #[serde(default)]
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    #[serde(skip, default)]
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    /// Runtime-derived exact original-token sets for `Skip` terminals in a
    /// composed grammar. Each token is wholly in `L(skip)+`: it can be
    /// consumed as one or more complete instances of that scoped-ignore
    /// terminal with a lexer reset between instances. This is deliberately
    /// not serialized; it is cheap to rebuild from the retained terminal
    /// expression and vocabulary and therefore does not change artifact wire
    /// compatibility.
    #[serde(skip, default)]
    pub(crate) scoped_ignore_only_tokens: Vec<(TerminalID, Box<[u32]>)>,
    /// Exact byte-token fusions `(fused, suffix)` grouped by scoped Skip. The
    /// fused token begins with one or more complete instances of the Skip
    /// language and the remaining bytes equal `suffix` exactly. If `suffix`
    /// is admitted by the ordinary static mask, `fused` is therefore admitted
    /// as well. Runtime-only for the same wire-compatibility reason as above.
    #[serde(skip, default)]
    pub(crate) scoped_ignore_prefix_fusions: Vec<(TerminalID, Box<[(u32, u32)]>)>,
    pub(crate) parser_dwa: DWA,
    /// Current-format loaded constraints retain the immutable parser DWA in
    /// its compact canonical pools instead of reconstructing RangeSet/Weight
    /// objects. Compiler-created and legacy-loaded constraints leave this
    /// empty and use `parser_dwa` directly.
    #[serde(skip, default)]
    pub(crate) packed_parser_dwa:
        Option<Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>>,
    #[serde(skip, default)]
    pub(crate) parser_start_final_override: Option<Weight>,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    #[serde(default)]
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    /// Uncombined exact depth-one acceptance parts. Direct-regular grammars
    /// retain terminal completion weights separately to avoid constructing one
    /// large union weight per parser state at compile time.
    #[serde(default)]
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<Weight>>,
    /// Immediate-completion L1 terminal weights for direct-regular parsers.
    /// Kept once per grammar terminal rather than duplicated across every
    /// epsilon-closed parser row.
    #[serde(default)]
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
    #[serde(skip, default)]
    pub(crate) packed_non_dwa_weights: Option<Arc<PackedNonDwaWeights>>,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    #[serde(skip, default)]
    pub(crate) direct_regular_wide_frontier_acceptance:
        Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    #[serde(skip, default)]
    pub(crate) direct_regular_dynamic_hot_frontiers:
        Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    #[serde(skip, default)]
    pub(crate) direct_regular_parser_state_acceptance:
        Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    #[serde(default)]
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    #[serde(with = "crate::compiler::glr::table::artifact_serde")]
    pub(crate) table: GLRTable,
    #[serde(default)]
    pub(crate) terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::artifact_serde")]
    pub(crate) tokenizer: Tokenizer,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    #[serde(skip, default)]
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    #[serde(skip, default)]
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    #[serde(skip, default)]
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// `possible_matches_complete` to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    #[serde(default)]
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    #[serde(default, with = "internal_tsid_inverse_artifact_serde")]
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Ordinary tokenizers have one internal TSID per physical state, making
    /// `internal_tsid_to_states` the exact bucket inverse of
    /// `state_to_internal_tsid`. Current artifacts can omit that redundant
    /// allocation and reconstruct it only for composition/debug paths.
    #[serde(skip, default)]
    pub(crate) deferred_internal_tsid_to_states: OnceLock<Vec<Vec<u32>>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    /// Named unresolved `extern grammar` slots retained by a compiled parent.
    /// Values are parent-local hidden placeholder terminal IDs. Stored in the
    /// outer composition metadata so cached parents can be rebound after load.
    #[serde(skip, default)]
    pub(crate) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    /// Composition-time parser stack-effect templates retained from the
    /// original compile. These are the unspecialized per-terminal DFAs used to
    /// build parser DWAs, so a later linker can transport unchanged component
    /// behavior instead of re-characterizing the component LR table.
    /// Stored in the outer versioned artifact envelope for compatibility with
    /// older inner `Constraint` bincode layouts.
    #[serde(skip, default)]
    pub(crate) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    /// Composition-time symbolic parser characterizations retained from the
    /// original compile. A later linker can append only the boundary-induced
    /// reductions/rereductions and recompile affected terminal templates,
    /// rather than re-solving the component's reduction closure from scratch.
    #[serde(skip, default)]
    pub(crate) composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    /// Runtime-only inverse lexer-metadata index used by compiled-constraint
    /// composition. Row `t` lists exactly the raw tokenizer states whose
    /// epsilon closure has terminal `t` matched or still reachable.
    #[serde(skip, default)]
    pub(crate) terminal_live_states: Vec<Vec<u32>>,
    /// Runtime-only CSR view of the exact state -> internal-TSID relation.
    /// Ordinary tokenizers have one entry per state. A fully determinized
    /// runtime lexer may represent several old lexer states and therefore
    /// several independent TSID lanes in one physical state.
    #[serde(skip, default)]
    pub(crate) state_internal_tsid_offsets: Vec<u32>,
    #[serde(skip, default)]
    pub(crate) state_internal_tsids: Vec<u32>,
    /// Final-runtime subset states followed by an exact copy of the source
    /// tokenizer. `runtime_source_state_offset` is the boundary between the
    /// two coordinates. Empty metadata means no runtime-only determinization.
    #[serde(default)]
    pub(crate) runtime_source_state_offset: Option<u32>,
    /// CSR offsets for product-state -> exact source-state subset. There is one
    /// row per product state and therefore `product_state_count + 1` offsets.
    #[serde(default)]
    pub(crate) runtime_product_source_offsets: Vec<u32>,
    #[serde(default)]
    pub(crate) runtime_product_source_states: Vec<u32>,
    /// Scalar source representative for product states that are exactly one
    /// source state's epsilon closure; `u32::MAX` otherwise.
    #[serde(default)]
    pub(crate) runtime_product_exact_source_states: Vec<u32>,
    /// Runtime-only inverse used to re-coalesce a uniform source frontier.
    #[serde(skip, default)]
    pub(crate) runtime_product_state_by_source_subset: FxHashMap<Box<[u32]>, u32>,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Runtime-only compact transition view for commit template products.
    #[serde(skip, default)]
    pub(crate) fast_template_dfas_by_terminal: FastTemplateDfasByTerminal,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    #[serde(default, with = "original_token_map_artifact_serde")]
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Current-format loads retain the fixed-width original-token map inside
    /// the owned artifact instead of expanding all model-token entries to
    /// `u32`. Ordinary static mask/commit performs direct packed lookups; only
    /// composition/debug-style bulk access materializes the vector lazily.
    #[serde(skip, default)]
    pub(crate) packed_original_token_to_internal:
        Option<Arc<original_token_map_artifact_serde::PackedOriginalTokenMap>>,
    #[serde(skip, default)]
    pub(crate) deferred_original_token_to_internal: OnceLock<Vec<u32>>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    #[serde(default, with = "internal_token_inverse_artifact_serde")]
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    /// Current-format loads can defer reconstructing the explicit inverse of
    /// `original_token_to_internal`. Static mask/commit only needs the internal
    /// token count and the already-serialized mask fragments; composition and
    /// token-space expansion materialize this inverse on first use.
    #[serde(skip, default)]
    pub(crate) deferred_internal_token_to_tokens: OnceLock<Vec<Vec<u32>>>,
    #[serde(with = "token_bytes_artifact_serde")]
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Indexed immutable vocabulary used directly by runtime token lookup and
    /// iteration. Loaded constraints can point into artifact backing; compiled
    /// constraints own the same indexed representation alongside the source
    /// map used by compiler/composition code.
    #[serde(skip, default)]
    pub(crate) packed_token_bytes: Option<Arc<token_bytes_artifact_serde::PackedTokenBytes>>,
    // Compiler-side scratch/result metadata only. No runtime or composition
    // path reads this field; composition rebuilds the map for its result when
    // needed. Persisting it duplicated token bytes inside every constraint.
    #[serde(skip, default)]
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    #[serde(skip)]
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    #[serde(skip)]
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    #[serde(skip)]
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    #[serde(skip)]
    pub(crate) pair_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    #[serde(skip)]
    pub(crate) quad_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    #[serde(skip)]
    pub(crate) super_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    #[serde(skip)]
    pub(crate) mega_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    #[serde(skip)]
    pub(crate) giga_word_group_buf_masks: DenseBufMaskRows,
    /// Sparse OR-union for each 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    #[serde(skip)]
    pub(crate) word_group_prefix_buf_masks: DenseBufMaskRows,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    #[serde(skip)]
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    #[serde(skip)]
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for quad groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    #[serde(skip)]
    pub(crate) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    #[serde(skip)]
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for byte groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    #[serde(skip)]
    pub(crate) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) word_group_sparse_total_entries: usize,
    #[serde(skip)]
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    #[serde(skip)]
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    #[serde(skip)]
    pub(crate) internal_token_dense_words: usize,
    #[serde(skip)]
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_dense_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    #[serde(skip, default)]
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Final-weight token sets eligible for the direct sparse-intersection
    /// path. Their full output masks are intentionally not materialized: the
    /// runtime intersects them with the current dense state on every use.
    #[serde(skip)]
    pub(crate) direct_sparse_weight_token_sets: DirectSparseWeightTokenSetCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    #[serde(skip)]
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    #[serde(skip, default)]
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    #[serde(skip, default = "empty_dense_words")]
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    #[serde(skip, default)]
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    #[serde(skip, default)]
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    #[serde(skip, default)]
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    #[serde(skip)]
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    #[serde(skip)]
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    #[serde(skip)]
    pub(crate) internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    /// Current IBM2 loads can retain the runtime-native flat sparse-mask slab
    /// directly inside the owned artifact instead of copying ~0.5-1 MiB.
    #[serde(skip, default)]
    pub(crate) backed_internal_token_buf_flat: Option<BackedInternalTokenBufMasks>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    #[serde(skip)]
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    #[serde(skip)]
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    #[serde(skip)]
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy Ã— buf_len).
    #[serde(skip)]
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    #[serde(skip)]
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    #[serde(skip)]
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    #[serde(skip)]
    pub(crate) final_mask_mapping: FinalMaskMapping,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    #[serde(skip, default)]
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    #[serde(skip, default)]
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    #[serde(skip, default)]
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    #[serde(skip, default)]
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    #[serde(skip, default)]
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Serialized composition-only metadata (reset-token rows, parser template
    /// cache, symbolic characterizations, and grammar summary). Current-format
    /// loads keep this cold section backed by the artifact and materialize it
    /// only if the constraint is later used as a composition component.
    #[serde(skip, default)]
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    #[serde(skip, default)]
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    #[serde(skip, default)]
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}


#[cfg(test)]
mod dynamic_mask_vocab_cache_boundary_tests {
    use super::*;

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
    fn fresh_runtime_instance_shares_only_vocab_derived_data() {
        let template = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let fresh = template.fresh_runtime_instance();

        assert!(Arc::ptr_eq(&template.trie, &fresh.trie));
        match (&template.token_aliases, &fresh.token_aliases) {
            (DynamicMaskAliasStore::Ordered(left), DynamicMaskAliasStore::Ordered(right)) => {
                assert!(Arc::ptr_eq(left, right));
            }
            _ => panic!("materialized ordered vocabulary changed alias representation"),
        }
        assert!(Arc::ptr_eq(
            &template.canonical_original_token_offsets,
            &fresh.canonical_original_token_offsets,
        ));
        assert!(Arc::ptr_eq(
            &template.canonical_original_tokens,
            &fresh.canonical_original_tokens,
        ));
        assert!(Arc::ptr_eq(
            &template.node_token_markers,
            &fresh.node_token_markers,
        ));
        assert!(Arc::ptr_eq(
            &template.subtree_original_token_offsets,
            &fresh.subtree_original_token_offsets,
        ));
        assert!(Arc::ptr_eq(
            &template.subtree_original_tokens,
            &fresh.subtree_original_tokens,
        ));

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
