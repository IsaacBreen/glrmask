use glrmask_artifact::CommitTemplateDfas;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::automata::regex::Expr;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted_u32::nwa::NWA;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use crate::ds::bitset::BitSet;

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
pub(crate) type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;
pub(crate) type DenseWeightBufMaskCache = FxHashMap<usize, Box<[u32]>>;
pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u16, u32)]>>;
pub(crate) type DirectSparseWeightTokenSetCache = FxHashSet<usize>;
pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
const INLINE_DWA_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitionRow {
    Inline(SmallVec<[(i32, (u32, Weight)); 4]>),
    Hash(FxHashMap<i32, (u32, Weight)>),
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

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Inline(entries) => entries.is_empty(),
            Self::Hash(entries) => entries.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<&(u32, Weight)> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, transition)| (candidate == label).then_some(transition)),
            Self::Hash(entries) => entries.get(label),
        }
    }
}

pub(crate) type FastDwaTransitions = Vec<FastDwaTransitionRow>;

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

    pub(crate) fn push_edge_bytes(&mut self, bytes: &[u8]) -> (u32, u32) {
        let start = self.edge_bytes.len() as u32;
        self.edge_bytes.extend_from_slice(bytes);
        (start, bytes.len() as u32)
    }

    #[inline]
    pub(crate) fn edge_bytes_len(&self) -> usize {
        self.edge_bytes.len()
    }

    fn collect_subtree_metadata(&mut self, node_id: u32) -> [u64; 4] {
        let start = self.subtree_tokens.len() as u32;
        if let Some(token_id) = self.nodes[node_id as usize].token_id {
            self.subtree_tokens.push(token_id);
        }

        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        let mut subtree_bytes = [0u64; 4];
        for edge_index in first_child..first_child + child_len {
            // Copy the compact edge fields before recursing so no borrow of
            // `self.edges` remains live across the mutable recursive call.
            let edge = self.edges[edge_index].clone();
            let byte_start = edge.byte_start as usize;
            let byte_end = byte_start + edge.byte_len as usize;
            for &byte in &self.edge_bytes[byte_start..byte_end] {
                subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
            }
            let child_bytes = self.collect_subtree_metadata(edge.child);
            for (target, child) in subtree_bytes.iter_mut().zip(child_bytes) {
                *target |= child;
            }
        }

        let end = self.subtree_tokens.len() as u32;
        let node = &mut self.nodes[node_id as usize];
        node.subtree_token_start = start;
        node.subtree_token_end = end;
        node.subtree_bytes = subtree_bytes;
        subtree_bytes
    }

    pub(crate) fn finalize_subtree_metadata(&mut self) {
        self.subtree_tokens.clear();
        self.subtree_tokens.reserve(self.nodes.len());
        if !self.nodes.is_empty() {
            self.collect_subtree_metadata(0);
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
pub(crate) struct DynamicSelfLoopProjection {
    pub(crate) source_state: u32,
    pub(crate) required_terminal: TerminalID,
    pub(crate) safe_no_match_mask: Arc<[u32]>,
    pub(crate) safe_subtrees: Arc<[u8]>,
}

impl DynamicSelfLoopProjection {
    #[inline]
    pub(crate) fn subtree_is_safe(&self, node: u32) -> bool {
        self.safe_subtrees
            .get(node as usize)
            .is_some_and(|&safe| safe != 0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocabSource {
    pub(crate) trie: Arc<VocabPrefixTree>,
    pub(crate) token_aliases: Arc<Vec<Vec<u32>>>,
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
        self.self_loop_projections = Arc::new(projections);
    }

    pub(crate) fn self_loop_projection(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        self.self_loop_projections
            .iter()
            .find(|projection| projection.source_state == source_state)
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
    #[serde(skip, default)]
    pub(crate) segmented_boundary_parser: Option<Box<SegmentedBoundaryParser>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedParserComponent {
    pub(crate) constraint: Box<Constraint>,
    pub(crate) tokenizer_state_offset: u32,
    pub(crate) terminal_offset: u32,
    pub(crate) global_to_local_parser_state: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedBoundaryParser {
    pub(crate) parser_nwa: NWA,
    /// When true, parser-NWA labels retain the template stack-effect alphabet:
    /// positive labels pop/match the current stack and negative labels push a
    /// concrete LR state. The segmented mask evaluator interprets those
    /// effects directly instead of requiring compile-time negative resolution.
    pub(crate) signed_stack_effects: bool,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
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
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
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
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
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
    pub(crate) pair_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    pub(crate) quad_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    pub(crate) super_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    pub(crate) mega_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    pub(crate) giga_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Sparse OR-union for each 64-token internal word group.
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    pub(crate) word_group_prefix_buf_masks: Vec<Box<[u32]>>,
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
    pub(crate) internal_token_buf_flat: Box<[(u16, u32)]>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy × buf_len).
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
    #[serde(default)]
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
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
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
    #[serde(default)]
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    #[serde(default)]
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(default)]
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
    pub(crate) pair_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    #[serde(skip)]
    pub(crate) quad_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    #[serde(skip)]
    pub(crate) super_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    #[serde(skip)]
    pub(crate) mega_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    #[serde(skip)]
    pub(crate) giga_word_group_buf_masks: Vec<Box<[u32]>>,
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
    pub(crate) word_group_prefix_buf_masks: Vec<Box<[u32]>>,
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
    pub(crate) internal_token_buf_flat: Box<[(u16, u32)]>,
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
    /// Total cost of all heavy tokens combined (n_heavy × buf_len).
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
}


#[cfg(test)]
mod dynamic_mask_vocab_cache_boundary_tests {
    use super::*;

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
