use glrmask_artifact::CommitTemplateDfas;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::GLRTable;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};

use super::mask_mapping::FinalMaskMapping;

pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;

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
            Self::Hybrid {
                state_to_dense_row,
                ..
            } => state_to_dense_row.len(),
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
    fn all_subtree_tokens(&self) -> &[u32] {
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

/// Canonical semantic snapshot of a dynamic-mask residual. Flattening the GSS
/// deliberately removes representation-only Arc identities and accumulator
/// node organization, so equivalent residuals reached after different token
/// commits share one exact cached mask.
pub(crate) type DynamicMaskStateKey =
    Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<TerminalID>)>)>)>;

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
        const MAX_DYNAMIC_MASK_CACHE_ENTRIES: usize = 64;
        let mut cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.iter().any(|entry| entry.state == state) {
            return;
        }
        if cache.len() == MAX_DYNAMIC_MASK_CACHE_ENTRIES {
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

/// Fully compiled, immutable grammar constraint.
///
/// A `Constraint` is intended to be reused across generated sequences. Call
/// [`Constraint::start`] to create a mutable per-sequence state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    #[serde(default)]
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
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
    /// An empty table may represent deferred possible-match construction. Static
    /// masking then uses the exact dynamic-mask fallback whenever token-start
    /// terminal exclusions make possible matches necessary.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. When true, an absent
    /// state/terminal row is a known-empty token set rather than a signal to
    /// invoke the exact dynamic fallback.
    #[serde(default)]
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
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
}
