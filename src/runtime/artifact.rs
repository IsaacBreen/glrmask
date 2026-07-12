use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::u8set::U8Set;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

use super::mask_mapping::FinalMaskMapping;

pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
pub(crate) type DenseWords = Arc<[u64]>;

pub(crate) fn empty_dense_words() -> DenseWords {
    Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice())
}

pub(crate) type InternalTokenBufMasks = Vec<(u16, u32)>;
pub(crate) type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;
pub(crate) type DenseWeightBufMaskCache = FxHashMap<usize, Box<[u32]>>;
pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u16, u32)]>>;
pub(crate) type DirectSparseWeightTokenSetCache = FxHashSet<usize>;
pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
pub(crate) type FastDwaTransitions = Vec<FxHashMap<i32, (u32, Weight)>>;
pub(crate) type FastTokenizerTransitions = Vec<Box<[u32; 256]>>;
pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<CommitTemplateDfas>>>;

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

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskTrie {
    pub(crate) nodes: Vec<DynamicMaskTrieNode>,
    pub(crate) edges: Vec<DynamicMaskTrieEdge>,
    edge_bytes: Vec<u8>,
    subtree_tokens: Vec<u32>,
}

impl DynamicMaskTrie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![DynamicMaskTrieNode::default()],
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
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
    pub(crate) fn subtree_tokens(&self, node: u32) -> &[u32] {
        let node = self.node(node);
        &self.subtree_tokens
            [node.subtree_token_start as usize..node.subtree_token_end as usize]
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

/// Whole-token no-finalization outcomes for one tokenizer residual. Tokens are
/// grouped by the exact set of live tokenizer states reachable after consuming
/// the token without committing a terminal. At runtime an admissible group can
/// be ORed into the mask in one dense operation; only inadmissible groups need
/// the full terminal-commit traversal.
#[derive(Debug)]
pub(crate) struct DynamicContinuationGroup {
    pub(crate) end_states: Box<[u32]>,
    pub(crate) mask: Box<[u32]>,
    pub(crate) token_count: usize,
}

#[derive(Debug)]
pub(crate) struct DynamicContinuationPartition {
    pub(crate) groups: Box<[DynamicContinuationGroup]>,
    token_groups: Box<[u16]>,
    subtree_groups: Box<[u64]>,
}

const CONTINUATION_NFA_CONFIG_UNKNOWN: u32 = u32::MAX;
const CONTINUATION_NFA_CONFIG_DEAD: u32 = u32::MAX - 1;

struct ContinuationNfaScanCache<'tok> {
    tokenizer: &'tok Tokenizer,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    raw_start_config: Vec<u32>,
}

impl<'tok> ContinuationNfaScanCache<'tok> {
    fn new(tokenizer: &'tok Tokenizer) -> Self {
        Self {
            tokenizer,
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            transitions: Vec::new(),
            raw_start_config: vec![
                CONTINUATION_NFA_CONFIG_UNKNOWN;
                tokenizer.num_states() as usize
            ],
        }
    }

    fn intern_config(&mut self, mut states: Vec<u32>) -> u32 {
        states.sort_unstable();
        states.dedup();
        if let Some(&id) = self.config_ids.get(states.as_slice()) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push(None);
        id
    }

    fn config_for_raw_start(&mut self, state: u32) -> u32 {
        let slot = state as usize;
        let cached = self.raw_start_config[slot];
        if cached != CONTINUATION_NFA_CONFIG_UNKNOWN {
            return cached;
        }
        let closure = self
            .tokenizer
            .execute_from_state_end_only(&[], state)
            .into_vec();
        let config = self.intern_config(closure);
        self.raw_start_config[slot] = config;
        config
    }

    fn step_config(&mut self, config: u32, byte: u8) -> Option<u32> {
        let config_index = config as usize;
        if let Some(row) = self.transitions[config_index].as_ref() {
            let cached = row[byte as usize];
            if cached != CONTINUATION_NFA_CONFIG_UNKNOWN {
                return (cached != CONTINUATION_NFA_CONFIG_DEAD).then_some(cached);
            }
        }

        let direct_targets = {
            let states = &self.configs[config_index];
            let mut targets = Vec::<u32>::new();
            for &state in states.iter() {
                if let Some(target) = self.tokenizer.step(state, byte) {
                    targets.push(target);
                }
            }
            targets.sort_unstable();
            targets.dedup();
            targets
        };

        let target = if direct_targets.is_empty() {
            CONTINUATION_NFA_CONFIG_DEAD
        } else {
            let mut closed_targets = Vec::<u32>::new();
            for target in direct_targets {
                let target_config = self.config_for_raw_start(target);
                closed_targets.extend_from_slice(&self.configs[target_config as usize]);
            }
            self.intern_config(closed_targets)
        };

        let row = self.transitions[config_index]
            .get_or_insert_with(|| Box::new([CONTINUATION_NFA_CONFIG_UNKNOWN; 256]));
        row[byte as usize] = target;
        (target != CONTINUATION_NFA_CONFIG_DEAD).then_some(target)
    }

    fn non_end_states(&self, config: u32) -> Vec<u32> {
        self.configs[config as usize]
            .iter()
            .copied()
            .filter(|&state| !self.tokenizer.is_end(state))
            .collect()
    }
}

impl DynamicContinuationPartition {
    #[inline]
    pub(crate) fn token_group(&self, token_id: u32) -> Option<usize> {
        self.token_groups
            .get(token_id as usize)
            .copied()
            .filter(|&group| group != u16::MAX)
            .map(usize::from)
    }

    #[inline]
    pub(crate) fn subtree_groups(&self, node: u32) -> u64 {
        self.subtree_groups[node as usize]
    }
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
    pending_source: Option<DynamicMaskVocabSource>,
    initialized: bool,
    continuation_partitions:
        Arc<Mutex<FxHashMap<u32, Arc<DynamicContinuationPartition>>>>,
    mask_cache: Arc<Mutex<Vec<DynamicMaskCacheEntry>>>,
}

impl DynamicMaskVocab {
    pub(crate) fn from_compiler_artifacts(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_source(DynamicMaskVocabSource { trie, token_aliases })
    }

    fn from_source(source: DynamicMaskVocabSource) -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            pending_source: Some(source),
            initialized: false,
            continuation_partitions: Arc::new(Mutex::new(FxHashMap::default())),
            mask_cache: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn from_packed(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>,
    ) -> Self {
        Self {
            trie,
            token_aliases: DynamicMaskAliasStore::Packed(token_aliases),
            pending_source: None,
            initialized: true,
            continuation_partitions: Arc::new(Mutex::new(FxHashMap::default())),
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
        self.initialized = true;
        true
    }

    #[inline]
    pub(crate) fn token_ids(&self, canonical_token_id: u32) -> Option<&[u32]> {
        match &self.token_aliases {
            DynamicMaskAliasStore::Ordered(token_aliases) => token_aliases
                .get(canonical_token_id as usize)
                .map(Vec::as_slice),
            DynamicMaskAliasStore::Packed(token_aliases) => match token_aliases
                .get(canonical_token_id as usize)
                .and_then(Option::as_ref)
            {
                Some(PackedDynamicMaskTokenAliases::Single(token_id)) => {
                    Some(std::slice::from_ref(token_id))
                }
                Some(PackedDynamicMaskTokenAliases::Many(token_ids)) => Some(token_ids),
                None => None,
            },
        }
    }

    #[inline]
    fn set_alias_bits(&self, canonical_token_id: u32, mask: &mut [u32]) -> usize {
        let Some(token_ids) = self.token_ids(canonical_token_id) else {
            return 0;
        };
        for &token_id in token_ids {
            let word = token_id as usize / 32;
            let bit = token_id % 32;
            if let Some(slot) = mask.get_mut(word) {
                *slot |= 1u32 << bit;
            }
        }
        token_ids.len()
    }

    fn collect_canonical_token_entries(
        &self,
        node_id: u32,
        prefix: &mut Vec<u8>,
        entries: &mut Vec<(u32, Box<[u8]>)>,
    ) {
        let node = self.trie.node(node_id);
        if let Some(token_id) = node.token_id {
            entries.push((token_id, prefix.clone().into_boxed_slice()));
        }
        for edge in self.trie.children(node_id) {
            let old_len = prefix.len();
            prefix.extend_from_slice(self.trie.edge_bytes(edge));
            self.collect_canonical_token_entries(edge.child, prefix, entries);
            prefix.truncate(old_len);
        }
    }

    fn canonical_token_entries(&self) -> Vec<(u32, Box<[u8]>)> {
        let mut entries = Vec::with_capacity(self.trie.subtree_tokens(0).len());
        self.collect_canonical_token_entries(0, &mut Vec::new(), &mut entries);
        entries
    }

    fn fill_continuation_subtree_groups(
        &self,
        node_id: u32,
        token_groups: &[u16],
        subtree_groups: &mut [u64],
    ) -> u64 {
        let node = self.trie.node(node_id);
        let mut groups = node
            .token_id
            .and_then(|token_id| token_groups.get(token_id as usize).copied())
            .filter(|&group| group != u16::MAX)
            .map_or(0, |group| 1u64 << group);
        for edge in self.trie.children(node_id) {
            groups |= self.fill_continuation_subtree_groups(
                edge.child,
                token_groups,
                subtree_groups,
            );
        }
        subtree_groups[node_id as usize] = groups;
        groups
    }

    fn collect_dfa_continuation_groups(
        &self,
        tokenizer: &Tokenizer,
        node_id: u32,
        state: u32,
        by_end_states: &mut BTreeMap<Vec<u32>, Vec<u32>>,
    ) {
        let node = self.trie.node(node_id);
        if let Some(token_id) = node.token_id {
            let end_states = if tokenizer.is_end(state) {
                Vec::new()
            } else {
                vec![state]
            };
            by_end_states.entry(end_states).or_default().push(token_id);
        }

        for edge in self.trie.children(node_id) {
            let mut next_state = state;
            let mut blocked = false;
            for &byte in self.trie.edge_bytes(edge) {
                next_state = tokenizer.get_transition(next_state, byte);
                if next_state == u32::MAX {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                by_end_states
                    .entry(Vec::new())
                    .or_default()
                    .extend_from_slice(self.trie.subtree_tokens(edge.child));
            } else {
                self.collect_dfa_continuation_groups(
                    tokenizer,
                    edge.child,
                    next_state,
                    by_end_states,
                );
            }
        }
    }

    fn collect_nfa_continuation_groups(
        &self,
        scan_cache: &mut ContinuationNfaScanCache<'_>,
        node_id: u32,
        config: u32,
        by_end_states: &mut BTreeMap<Vec<u32>, Vec<u32>>,
    ) -> bool {
        let node = self.trie.node(node_id);
        if let Some(token_id) = node.token_id {
            let end_states = scan_cache.non_end_states(config);
            by_end_states.entry(end_states).or_default().push(token_id);
            if by_end_states.len() > 64 {
                return false;
            }
        }

        for edge in self.trie.children(node_id) {
            let mut next_config = config;
            let mut blocked = false;
            for &byte in self.trie.edge_bytes(edge) {
                let Some(next) = scan_cache.step_config(next_config, byte) else {
                    blocked = true;
                    break;
                };
                next_config = next;
            }
            if blocked {
                by_end_states
                    .entry(Vec::new())
                    .or_default()
                    .extend_from_slice(self.trie.subtree_tokens(edge.child));
                if by_end_states.len() > 64 {
                    return false;
                }
            } else {
                if !self.collect_nfa_continuation_groups(
                    scan_cache,
                    edge.child,
                    next_config,
                    by_end_states,
                ) {
                    return false;
                }
            }
        }
        true
    }

    fn build_continuation_partition(
        &self,
        tokenizer: &Tokenizer,
        source_state: u32,
        mask_words: usize,
        entries: &[(u32, Box<[u8]>)],
    ) -> Option<DynamicContinuationPartition> {
        let mut by_end_states = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        if tokenizer.has_epsilon_transitions() {
            let mut scan_cache = ContinuationNfaScanCache::new(tokenizer);
            let start_config = scan_cache.config_for_raw_start(source_state);
            let completed = self.collect_nfa_continuation_groups(
                &mut scan_cache,
                0,
                start_config,
                &mut by_end_states,
            );

            if std::env::var_os("GLRMASK_DYNAMIC_CONTINUATION_NFA_STRICT_REFERENCE").is_some() {
                let mut reference = BTreeMap::<Vec<u32>, Vec<u32>>::new();
                for (canonical_token_id, bytes) in entries {
                    let mut end_states = tokenizer
                        .execute_from_state_end_only(bytes, source_state)
                        .into_iter()
                        .filter(|&state| !tokenizer.is_end(state))
                        .collect::<Vec<_>>();
                    end_states.sort_unstable();
                    end_states.dedup();
                    reference
                        .entry(end_states)
                        .or_default()
                        .push(*canonical_token_id);
                }
                if completed {
                    assert_eq!(
                        by_end_states, reference,
                        "NFA continuation trie traversal differed from scalar token replay"
                    );
                } else {
                    assert!(
                        reference.len() > 64,
                        "NFA continuation trie traversal aborted at the group cap but scalar replay did not"
                    );
                }
            }
            if !completed {
                return None;
            }
        } else {
            self.collect_dfa_continuation_groups(
                tokenizer,
                0,
                source_state,
                &mut by_end_states,
            );
        }

        if by_end_states.len() > 64 {
            return None;
        }

        let max_token_id = self
            .trie
            .subtree_tokens(0)
            .iter()
            .map(|token_id| *token_id as usize)
            .max()
            .unwrap_or(0);
        let mut token_groups = vec![u16::MAX; max_token_id.saturating_add(1)];
        let mut groups = Vec::with_capacity(by_end_states.len());
        for (group_id, (end_states, canonical_token_ids)) in by_end_states.into_iter().enumerate() {
            let mut mask = vec![0u32; mask_words];
            let mut token_count = 0usize;
            for canonical_token_id in canonical_token_ids {
                token_count += self.set_alias_bits(canonical_token_id, &mut mask);
                token_groups[canonical_token_id as usize] = group_id as u16;
            }
            groups.push(DynamicContinuationGroup {
                end_states: end_states.into_boxed_slice(),
                mask: mask.into_boxed_slice(),
                token_count,
            });
        }
        let mut subtree_groups = vec![0u64; self.trie.nodes.len()];
        if !self.trie.nodes.is_empty() {
            self.fill_continuation_subtree_groups(0, &token_groups, &mut subtree_groups);
        }
        Some(DynamicContinuationPartition {
            groups: groups.into_boxed_slice(),
            token_groups: token_groups.into_boxed_slice(),
            subtree_groups: subtree_groups.into_boxed_slice(),
        })
    }

    pub(crate) fn cached_continuation_partition(
        &self,
        source_state: u32,
    ) -> Option<Arc<DynamicContinuationPartition>> {
        self.continuation_partitions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&source_state)
            .cloned()
    }

    pub(crate) fn prebuild_continuation_partitions(
        &self,
        tokenizer: &Tokenizer,
        mask_words: usize,
    ) {
        let source_discovery_started_at = std::time::Instant::now();
        let mut broad_loop_states = (0..tokenizer.num_states())
            .map(|state| (state, tokenizer.self_loop_bytes(state).len()))
            .filter(|&(_, width)| width >= 32)
            .collect::<Vec<_>>();
        let Some(max_loop_width) = broad_loop_states.iter().map(|&(_, width)| width).max() else {
            return;
        };
        broad_loop_states.retain(|&(_, width)| width == max_loop_width);
        broad_loop_states.sort_unstable_by_key(|&(state, _)| state);

        let broad_targets = broad_loop_states
            .iter()
            .map(|&(state, _)| state)
            .collect::<FxHashSet<_>>();
        let epsilon_closures = tokenizer.has_epsilon_transitions().then(|| {
            (0..tokenizer.num_states())
                .map(|state| {
                    tokenizer
                        .execute_from_state_end_only(&[], state)
                        .into_vec()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>()
        });
        let target_closure_reaches_broad = epsilon_closures.as_ref().map(|closures| {
            closures
                .iter()
                .map(|closure| closure.iter().any(|state| broad_targets.contains(state)))
                .collect::<Vec<_>>()
        });
        let mut entry_sources = Vec::<(u32, usize)>::new();
        let deterministic = !tokenizer.has_epsilon_transitions();
        for source_state in 0..tokenizer.num_states() {
            if broad_targets.contains(&source_state) {
                continue;
            }
            let covered_len = if deterministic {
                tokenizer
                    .transitions_from(source_state)
                    .filter(|(_, target)| broad_targets.contains(target))
                    .count()
            } else {
                let mut covered = U8Set::empty();
                let closures = epsilon_closures.as_ref().unwrap();
                let target_reaches_broad = target_closure_reaches_broad.as_ref().unwrap();
                for &closure_state in closures[source_state as usize].iter() {
                    for (byte, target) in tokenizer.transitions_from(closure_state) {
                        if target_reaches_broad[target as usize] {
                            covered.insert(byte);
                        }
                    }
                }

                if std::env::var_os("GLRMASK_DYNAMIC_CONTINUATION_NFA_STRICT_REFERENCE")
                    .is_some()
                {
                    let mut reference = U8Set::empty();
                    for byte in 0..=u8::MAX {
                        let execution =
                            tokenizer.execute_from_state_end_only(&[byte], source_state);
                        if execution
                            .iter()
                            .any(|end_state| broad_targets.contains(end_state))
                        {
                            reference.insert(byte);
                        }
                    }
                    assert_eq!(
                        covered, reference,
                        "NFA continuation source discovery differed from scalar reference"
                    );
                }
                covered.len()
            };
            if covered_len >= 32 {
                entry_sources.push((source_state, covered_len));
            }
        }
        entry_sources.sort_unstable_by_key(|&(state, width)| (state, std::cmp::Reverse(width)));
        entry_sources.truncate(1);

        let mut source_states = broad_loop_states
            .into_iter()
            .map(|(state, _)| state)
            .collect::<Vec<_>>();
        source_states.extend(entry_sources.iter().map(|&(state, _)| state));
        source_states.sort_unstable();
        source_states.dedup();
        source_states.truncate(4);

        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_continuation_prebuild] sources={:?} source_discovery_ms={:.3}",
                source_states,
                source_discovery_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let entries = Arc::new(self.canonical_token_entries());
        let build = |source_state: &u32| {
            let started = profile.then(std::time::Instant::now);
            let partition = self
                .build_continuation_partition(tokenizer, *source_state, mask_words, &entries)
                .map(Arc::new);
            if let Some(started) = started {
                if let Some(partition) = &partition {
                    eprintln!(
                        "[glrmask/profile][dynamic_continuation_partition] source={} groups={} tokens={} build_ms={:.3}",
                        source_state,
                        partition.groups.len(),
                        partition.groups.iter().map(|group| group.token_count).sum::<usize>(),
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                } else {
                    eprintln!(
                        "[glrmask/profile][dynamic_continuation_partition] source={} declined=true build_ms={:.3}",
                        source_state,
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }
            partition.map(|partition| (*source_state, partition))
        };
        let built = if rayon::current_num_threads() == 1 || source_states.len() < 2 {
            source_states.iter().filter_map(build).collect::<Vec<_>>()
        } else {
            source_states.par_iter().filter_map(build).collect::<Vec<_>>()
        };
        let mut partitions = self
            .continuation_partitions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        partitions.extend(built);
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
            pending_source: None,
            initialized: false,
            continuation_partitions: Arc::new(Mutex::new(FxHashMap::default())),
            mask_cache: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CommitTemplateDfas {
    pub(crate) pop: UnweightedDfa,
    pub(crate) read: UnweightedDfa,
    pub(crate) push: UnweightedDfa,
    pub(crate) pop_to_read: Vec<Option<u32>>,
    pub(crate) pop_to_push: Vec<Option<u32>>,
    pub(crate) read_to_push: Vec<Option<u32>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    #[serde(default)]
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    pub(crate) table: GLRTable,
    #[serde(default)]
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Tokenizer,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,
    /// Whether the grammar start language admits the empty string.
    ///
    /// The normalized GLR grammar contains no epsilon productions, so runtime
    /// completion needs this semantic bit to recognize the untouched initial
    /// parser stack as accepting.
    #[serde(default)]
    pub(crate) start_accepts_empty: bool,

    /// Runtime-only vocabulary data for direct dynamic masking.
    #[serde(skip, default)]
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    #[serde(skip)]
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
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
    pub(crate) eos_token_id: Option<u32>,
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
    #[serde(skip)]
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
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
    /// Dense bitmap of the full internal token universe.
    #[serde(skip, default = "empty_dense_words")]
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
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
