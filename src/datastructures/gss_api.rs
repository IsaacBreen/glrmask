use crate::datastructures::arc_wrapper::ArcPtrWrapper;
use crate::datastructures::leveled_gss::LeveledGSS;
use crate::datastructures::trie::{EdgeInserter, God, GodWrapper, Trie, Trie2Index};
use bimap::BiBTreeMap;
use deterministic_hash::DeterministicHasher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::grammar::Terminal;
use crate::glr::parser::{GLRParserState, ParseStateEdgeContent};
use crate::glr::table::StateID;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
use profiler_macro::{time_it, timeit};

use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;

// --- Types moved from constraint.rs to break circular dependency ---

pub type StateIDBV = HybridBitset;
pub(crate) type LLMTokenBV = HybridBitset;
pub(crate) type TerminalBV = HybridBitset;
/// A 2D bitset where L1 is tokenizer state and L2 is terminal ID.
pub type TerminalInfo = HybridL2Bitset;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    pub(crate) end: bool,
    pub(crate) live_tokens: LLMTokenBV,
}

impl PrecomputedNodeContents {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self { end: false, live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1) }
    }

    pub(crate) fn internal() -> Self {
        Self { end: false, live_tokens: LLMTokenBV::zeros() }
    }

    pub(crate) fn leaf() -> Self {
        Self { end: true, live_tokens: LLMTokenBV::zeros() }
    }
}

impl JSONConvertible for PrecomputedNodeContents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clean_end".to_string(), self.end.to_json());
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let end = obj.remove("clean_end").ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                                   .and_then(bool::from_json)?;
                let live_tokens = obj.remove("live_tokens").ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents".to_string())
                                       .and_then(LLMTokenBV::from_json)?;
                Ok(PrecomputedNodeContents { end, live_tokens })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents0 {
    pub(crate) live_tokens: LLMTokenBV,
    pub(crate) final_tokenizer_state: Option<TokenizerStateID>,
}

impl PrecomputedNodeContents0 {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self {
            live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1),
            final_tokenizer_state: None,
        }
    }

    pub(crate) fn internal() -> Self {
        Self {
            live_tokens: LLMTokenBV::zeros(),
            final_tokenizer_state: None,
        }
    }

    pub(crate) fn leaf(final_sid: TokenizerStateID) -> Self {
        Self { live_tokens: LLMTokenBV::zeros(), final_tokenizer_state: Some(final_sid) }
    }
}

impl JSONConvertible for PrecomputedNodeContents0 {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clean_end".to_string(), self.final_tokenizer_state.is_some().to_json());
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        obj.insert("final_tokenizer_state".to_string(), self.final_tokenizer_state.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let live_tokens = obj.remove("live_tokens").ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents0".to_string())
                                       .and_then(LLMTokenBV::from_json)?;
                let final_tokenizer_state = obj.remove("final_tokenizer_state").ok_or_else(|| "Missing field final_tokenizer_state for PrecomputedNodeContents0".to_string())
                                               .and_then(|n| Option::<TokenizerStateID>::from_json(n))?;
                Ok(PrecomputedNodeContents0 { live_tokens, final_tokenizer_state })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents0".to_string()),
        }
    }
}

impl Into<PrecomputedNodeContents> for PrecomputedNodeContents0 {
    fn into(self) -> PrecomputedNodeContents {
        PrecomputedNodeContents { end: self.final_tokenizer_state.is_some(), live_tokens: self.live_tokens }
    }
}


pub type PrecomputeNode0 = Trie<Option<(TerminalID, Option<TokenizerStateID>)>, LLMTokenBV, PrecomputedNodeContents0>;
pub type PrecomputeNode1 = Trie<Option<TerminalID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode3 = Trie<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;

pub type PrecomputeNode0Index = Trie2Index;
pub type PrecomputeNode1Index = Trie2Index;
pub type PrecomputeNode2Index = Trie2Index;
pub type PrecomputeNode3Index = Trie2Index;

pub type Precomputed0 = BTreeMap<TokenizerStateID, PrecomputeNode0Index>;
pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode1Index>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, PrecomputeNode2Index>;
pub type Precomputed3 = BTreeMap<TokenizerStateID, PrecomputeNode3Index>;

pub type Trie0GodWrapper = GodWrapper<Option<(TerminalID, Option<TokenizerStateID>)>, HybridBitset, PrecomputedNodeContents0>;
pub type Trie0God = God<Option<(TerminalID, Option<TokenizerStateID>)>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1GodWrapper = GodWrapper<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1God = God<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie2GodWrapper = GodWrapper<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie2God = God<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie3GodWrapper = GodWrapper<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
pub type Trie3God = God<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;


// --- GSS API Bridge ---

pub(crate)type StoredPrecomputeNodeIndex = PrecomputeNode3Index;
pub(crate)type StoredPrecomputeNode = PrecomputeNode3;
pub(crate)type StoredTrieGod = Trie3God;
pub(crate) type StoredTrieGodWrapper = Trie3GodWrapper;


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    stored_trie_nodes: BTreeSet<StoredPrecomputeNodeIndex>,
}

impl Acc {
    pub(crate) fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    pub(crate) fn new_dead() -> Self {
        Self {
            llm_tokens_union: HybridBitset::zeros(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    pub(crate) fn is_merge_neutral(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.terminals_union == HybridL2Bitset::all()
            && self.stored_trie_nodes.is_empty()
    }

    pub(crate) fn is_default(&self) -> bool {
        self.is_merge_neutral()
    }

    #[allow(dead_code)] pub(crate) fn new_with_local_constraints(llm_tokens: HybridBitset, terminals: HybridL2Bitset) -> Self {
        Self {
            llm_tokens_union: llm_tokens,
            terminals_union: terminals,
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    pub(crate) fn narrow(from: &Self, to: &Self) -> Self {
        Acc {
            llm_tokens_union: &from.llm_tokens_union & &to.llm_tokens_union,
            terminals_union: &from.terminals_union & &to.terminals_union,
            stored_trie_nodes: &from.stored_trie_nodes | &to.stored_trie_nodes,
        }
    }

    pub(crate) fn merge(lhs: &Self, rhs: &Self) -> Self {
        Acc {
            llm_tokens_union: &lhs.llm_tokens_union | &rhs.llm_tokens_union,
            terminals_union: &lhs.terminals_union | &rhs.terminals_union,
            stored_trie_nodes: &lhs.stored_trie_nodes | &rhs.stored_trie_nodes,
        }
    }

    pub(crate) fn union_llm_tokens(&self) -> HybridBitset { self.llm_tokens_union.clone() }
    pub(crate) fn stored_trie_nodes(&self) -> &BTreeSet<StoredPrecomputeNodeIndex> { &self.stored_trie_nodes }
    pub(crate) fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<StoredPrecomputeNodeIndex> { &mut self.stored_trie_nodes }
}


pub type GSSNode = LeveledGSS<Acc, ParseStateEdgeContent>;

pub fn new_fresh() -> Arc<GSSNode> {
    Arc::new(GSSNode::new_leaf(Acc::new_fresh()))
}

pub fn new_dead() -> Arc<GSSNode> {
    Arc::new(GSSNode::new_leaf(Acc::new_dead()))
}

pub fn new(acc: Acc) -> Arc<GSSNode> {
    Arc::new(GSSNode::new_leaf(acc))
}


#[derive(Debug, Clone, Default)]
pub struct GSSPopper {
    pub paths: BTreeMap<Arc<GSSNode>, Arc<Acc>>,
    pub below_bottom: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>>,
}

#[derive(Clone, Copy)]
pub(crate) struct GSSPopperItem<'a> {
    node: &'a Arc<GSSNode>,
    path_acc: &'a Arc<Acc>,
}

#[derive(Clone, Copy)]
pub struct GSSPopperItemPeek<'a> {
    path_acc: &'a Arc<Acc>,
    parent_arc: &'a Arc<GSSNode>,
    edge_value: &'a ParseStateEdgeContent,
    predecessor_node: &'a Arc<GSSNode>,
}

impl GSSPopper {
    pub fn new_from_node(node: Arc<GSSNode>, acc: Arc<Acc>) -> Self {
        let mut popper = Self {
            paths: BTreeMap::new(),
            below_bottom: BTreeMap::new(),
        };
        if node.is_leaf() {
            let mut by_edge = BTreeMap::new();
            let narrowed_acc = Arc::new(Acc::narrow(&acc, node.leaf_data().unwrap()));
            // A leaf has no edge, but we need a placeholder.
            // The old GSS had a complex logic for this. Here we simplify.
            // We'll say it's at depth 0.
            popper.below_bottom.entry(0).or_default();
        } else {
            popper.paths.insert(node, acc);
        }
        popper
    }

    pub fn iter(&self) -> impl Iterator<Item = GSSPopperItem<'_>> {
        self.paths.iter().map(|(node, acc)| GSSPopperItem {
            node,
            path_acc: acc,
        })
    }

    pub fn num_predecessors(&self) -> usize {
        self.paths.len()
    }

    pub fn popn(&mut self, n: usize) {
        for _ in 0..n {
            let mut new_below: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>> = BTreeMap::new();
            for (k, by_edge) in std::mem::take(&mut self.below_bottom) {
                new_below.insert(k + 1, by_edge);
            }

            let mut new_paths: BTreeMap<Arc<GSSNode>, Arc<Acc>> = BTreeMap::new();
            for (parent, path_acc) in std::mem::take(&mut self.paths) {
                for (edge_val, child) in parent.pop() {
                    let next_path = path_acc.clone(); // In LeveledGSS, Acc is only at leaves.

                    if child.is_leaf() {
                        let final_acc = Arc::new(Acc::narrow(&next_path, child.leaf_data().unwrap()));
                        let by_edge = new_below.entry(1).or_default();
                        if let Some(existing) = by_edge.get_mut(edge_val) {
                            let merged = Arc::new(Acc::merge(existing, &final_acc));
                            *existing = merged;
                        } else {
                            by_edge.insert(edge_val.clone(), final_acc);
                        }
                    } else {
                        if let Some(existing_acc) = new_paths.get_mut(child) {
                            *existing_acc = Arc::new(Acc::merge(existing_acc, &next_path));
                        } else {
                            new_paths.insert(child.clone(), next_path.clone());
                        }
                    }
                }
            }
            self.paths = new_paths;
            self.below_bottom = new_below;
        }
    }
}

impl<'a> GSSPopperItem<'a> {
    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPopperItemPeek<'_>> {
        self.node.pop().map(move |(edge_val, pred_arc)| {
            GSSPopperItemPeek {
                path_acc: &self.path_acc,
                parent_arc: self.node,
                edge_value: edge_val,
                predecessor_node: pred_arc,
            }
        })
    }
}

impl<'a> GSSPopperItemPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        // LeveledGSS nodes are already isolated paths by construction.
        self.parent_arc.clone()
    }
}


pub trait GSSNodeExt {
    fn push(&self, edge_value: ParseStateEdgeContent) -> Arc<Self>;
    fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Arc<Self>;
    fn popn(&self, n: usize) -> GSSPopper;
    fn merge_with_depth(&mut self, other: &Self, merge_depth: usize);
    fn allowed_llm_tokens(&self) -> LLMTokenBV;
    fn disallowed_terminals(&self) -> TerminalInfo;
    fn is_alive(&self) -> bool;
    fn is_ok(&self) -> bool;
    fn is_root(&self) -> bool;
    fn is_empty(&self) -> bool;
    fn num_predecessors(&self) -> usize;
    fn max_depth(&self) -> usize;
    fn peek_iter(parent_arc: &Arc<Self>) -> impl Iterator<Item = GSSPeek<'_>>;
    fn merge_many_with_depth(merge_depth: usize, nodes: impl IntoIterator<Item = Arc<Self>>) -> Arc<Self>;
}

impl GSSNodeExt for GSSNode {
    fn push(&self, edge_value: ParseStateEdgeContent) -> Arc<Self> {
        Arc::new(self.push(edge_value))
    }

    fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Arc<Self> {
        let mut current = self.clone();
        for edge in edge_values {
            current = current.push(edge);
        }
        Arc::new(current)
    }

    fn popn(&self, n: usize) -> GSSPopper {
        let mut popper = GSSPopper::new_from_node(Arc::new(self.clone()), Arc::new(Acc::new_fresh()));
        popper.popn(n);
        popper
    }

    fn merge_with_depth(&mut self, other: &Self, _merge_depth: usize) {
        self.merge(other);
    }

    fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.fold_leaves(LLMTokenBV::zeros(), |mut acc, leaf_data| {
            acc |= &leaf_data.llm_tokens_union;
            acc
        })
    }

    fn disallowed_terminals(&self) -> TerminalInfo {
        self.fold_leaves(HybridL2Bitset::empty(), |mut acc, leaf_data| {
            acc |= &leaf_data.terminals_union;
            acc
        }).complement()
    }

    fn is_alive(&self) -> bool {
        !self.allowed_llm_tokens().is_empty()
    }

    fn is_ok(&self) -> bool {
        self.is_alive()
    }

    fn is_root(&self) -> bool {
        self.is_leaf()
    }

    fn is_empty(&self) -> bool {
        self.is_leaf()
    }

    fn num_predecessors(&self) -> usize {
        self.pop().count()
    }

    fn max_depth(&self) -> usize {
        self.depth()
    }

    fn peek_iter(parent_arc: &Arc<Self>) -> impl Iterator<Item = GSSPeek<'_>> {
        parent_arc.pop().map(move |(edge, pred)| GSSPeek {
            parent_arc: parent_arc,
            edge_value: edge,
            predecessor_node: pred,
        })
    }

    fn merge_many_with_depth(merge_depth: usize, nodes: impl IntoIterator<Item = Arc<Self>>) -> Arc<Self> {
        let mut iter = nodes.into_iter();
        if let Some(first) = iter.next() {
            let mut merged = Arc::try_unwrap(first).unwrap_or_else(|arc| (*arc).clone());
            for other in iter {
                merged.merge_with_depth(&other, merge_depth);
            }
            Arc::new(merged)
        } else {
            new_fresh()
        }
    }
}


#[derive(Clone, Copy)]
pub(crate) struct GSSPeek<'a> {
    parent_arc: &'a Arc<GSSNode>,
    edge_value: &'a ParseStateEdgeContent,
    predecessor_node: &'a Arc<GSSNode>,
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }
    pub fn predecessor_node(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    pub fn resolved_llm_tokens_union(&self) -> LLMTokenBV {
        self.predecessor_node.allowed_llm_tokens()
    }

    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.parent_arc.push(edge_value)
    }

    pub fn popn(&self, len: usize) -> GSSPopper {
        self.parent_arc.popn(len)
    }

    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        self.parent_arc.clone()
    }
}


pub(crate) type PruneAndTransformRecursiveMemo = HashMap<*const GSSNode, Option<Arc<GSSNode>>>;

fn transform_recursive(
    node_arc: &Arc<GSSNode>,
    closure: &mut impl FnMut(&mut Acc) -> bool, // returns true to prune
    memo: &mut PruneAndTransformRecursiveMemo,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    let result = if node_arc.is_leaf() {
        let mut new_acc = node_arc.leaf_data().unwrap().clone();
        if closure(&mut new_acc) {
            None // Prune
        } else {
            Some(Arc::new(GSSNode::new_leaf(new_acc)))
        }
    } else {
        let mut new_node = GSSNode::new_empty_internal();
        let mut changed = false;
        for (edge, pred) in node_arc.pop() {
            if let Some(new_pred) = transform_recursive(pred, closure, memo) {
                new_node.add_predecessor(edge.clone(), new_pred);
            } else {
                changed = true;
            }
        }

        if new_node.num_predecessors() == 0 {
            None
        } else if !changed && new_node.num_predecessors() == node_arc.num_predecessors() {
            Some(node_arc.clone())
        } else {
            Some(Arc::new(new_node))
        }
    };

    memo.insert(node_ptr, result.clone());
    result
}


pub fn allow_only_llm_tokens_and_prune(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
) {
    let mut memo = HashMap::new();
    allow_only_llm_tokens_and_prune_arc(root_arc, allowed_tokens, &mut memo);
}

pub(crate) fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        acc.llm_tokens_union &= allowed_tokens;
        acc.llm_tokens_union.is_empty()
    };
    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub(crate) fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let allowed_mask = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed_mask, memo);
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        acc.llm_tokens_union = HybridBitset::max_ones();
        false
    };
    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub(crate) fn reset_terminals(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        acc.terminals_union = HybridL2Bitset::all();
        false
    };
    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub(crate) fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        acc.terminals_union -= disallowed_terminals;
        false // This function doesn't prune, just updates.
    };
    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        if acc.terminals_union == HybridL2Bitset::all() {
            return false;
        }

        let mut forbidden_llm_tokens = LLMTokenBV::zeros();
        let disallowed_terminals_l2 = acc.terminals_union.complement();

        for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
            if disallowed_terminals_for_range.is_empty() {
                continue;
            }

            let relevant_possible_matches = possible_matches.range(TokenizerStateID(*tokenizer_state_range.start())..=TokenizerStateID(*tokenizer_state_range.end()));

            for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                    if disallowed_terminals_for_range.contains(terminal_id.0) {
                        forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                    }
                }
            }
        }

        if forbidden_llm_tokens.is_empty() {
            return false;
        }

        acc.llm_tokens_union -= &forbidden_llm_tokens;

        acc.llm_tokens_union.is_empty()
    };

    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                return true; // Prune
            }
        }
        false
    };
    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        let mut new_terminals_btreemap = BTreeMap::new();

        for (old_state_id, new_state_id) in map {
            let bv_source = acc.terminals_union.get_l2_bitset(old_state_id.0).unwrap();
            new_terminals_btreemap.entry(*new_state_id)
                .and_modify(|bv| *bv |= bv_source)
                .or_insert_with(|| bv_source.clone());
        }

        let mut new_terminals_l2_bitset = HybridL2Bitset::all();
        for (state_id, bv) in new_terminals_btreemap {
            new_terminals_l2_bitset.insert_l2_bitset(state_id.0, bv);
        }

        acc.terminals_union = new_terminals_l2_bitset;
        false
    };
    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_dead();
    }
}

pub(crate) fn merge_stored_trie_nodes(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    let mut new_destinations = BTreeMap::new();

    let mut closure = |acc: &mut Acc| {
        if !acc.stored_trie_nodes.iter().any(
            |n| n.as_arc().read(stored_trie_god).expect("poison").value.live_tokens != acc.llm_tokens_union
        ) {
            return false;
        }

        let new_destination = new_destinations.entry((acc.stored_trie_nodes.clone(), acc.llm_tokens_union.clone()))
            .or_insert_with(|| StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal()))))
            .clone();
        let edge_key = (0, acc.llm_tokens_union.clone());
        let edge_value = StateIDBV::max_ones();
        let tokens_for_edge = acc.llm_tokens_union.clone();

        for source_wrapper in &acc.stored_trie_nodes {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = EdgeInserter::new(
                &stored_trie_god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= &tokens_for_edge,
                |_, _| {}, // Unconditional insertion
            );
            inserter.try_destination(new_destination.clone()).expect("Cycle detected when merging stored_trie nodes; this should be impossible.");
        }

        new_destination.write(stored_trie_god).expect("poison").value.live_tokens |= &tokens_for_edge;

        acc.stored_trie_nodes = BTreeSet::from([new_destination]);
        false
    };

    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        unreachable!();
    }
}

pub fn fuse_predecessors_recursive(
    node_arc: &Arc<GSSNode>,
    levels: usize,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
) -> Arc<GSSNode> {
    if levels == 0 {
        return node_arc.clone();
    }
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(fused_arc) = memo.get(&node_ptr) {
        return fused_arc.clone();
    }

    let mut recursively_fused_predecessors = BTreeMap::new();
    for (edge_val, pred_arc) in node_arc.pop() {
        let fused_pred_arc = fuse_predecessors_recursive(pred_arc, levels - 1, memo);
        recursively_fused_predecessors.entry(edge_val.clone()).or_insert_with(Vec::new).push(fused_pred_arc);
    }

    let mut new_node = GSSNode::new_empty_internal();
    for (edge_val, pred_arcs) in recursively_fused_predecessors {
        let merged_pred = GSSNode::merge_many_with_depth(1, pred_arcs);
        new_node.add_predecessor(edge_val, merged_pred);
    }

    let result_arc = Arc::new(new_node);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}

#[time_it]
pub(crate) fn is_simple_gss(
    node: &Arc<GSSNode>,
    hallucinated_state_id: StateID,
) -> Option<(StateID, Arc<Acc>)> {
    // This optimization is complex and tied to the old GSS structure.
    // With LeveledGSS, the structure is different and this specific pattern may not occur
    // or may not be as beneficial to optimize. Returning None to disable it for now.
    None
}

pub(crate) fn simplify_roots_in_place(roots: &mut [Arc<GSSNode>]) {
    // LeveledGSS already performs significant structural sharing.
    // This function can be a no-op for now.
}

pub fn simplify(states: &mut BTreeMap<TokenizerStateID, Arc<GSSNode>>) {
    // LeveledGSS already performs significant structural sharing.
}

pub fn popn_collect_isolated_parents(
    node_arc: &Arc<GSSNode>,
    n: usize,
) -> Vec<(crate::glr::table::StateID, Arc<GSSNode>)> {
    let popper = node_arc.popn(n);
    let mut out = Vec::new();
    for item in popper.iter() {
        for peek in item.peek_iter() {
            out.push((peek.edge_value().state_id, peek.isolated_parent()));
        }
    }
    out
}

pub(crate) fn get_roots<'a>(nodes: impl IntoIterator<Item = &'a GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    let mut results: BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> = BTreeMap::new();
    let mut q: VecDeque<(&'a GSSNode, Arc<Acc>)> = VecDeque::new();

    for node in nodes {
        if node.is_leaf() {
            // This is a root of a path, but has no incoming edge.
        } else {
            q.push_back((node, Arc::new(Acc::new_fresh())));
        }
    }

    let mut visited = HashSet::new();

    while let Some((node, path_acc)) = q.pop_front() {
        if !visited.insert(node as *const _) {
            continue;
        }

        for (edge, pred) in node.pop() {
            if pred.is_leaf() {
                let final_acc = Arc::new(Acc::narrow(&path_acc, pred.leaf_data().unwrap()));
                results.entry(edge.clone()).or_default().insert(final_acc);
            } else {
                q.push_back((pred, path_acc.clone()));
            }
        }
    }

    results
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GSSStats {
    pub(crate) num_roots: usize,
    pub(crate) num_root_predecessors: usize,
    pub(crate) num_unique_root_predecessor_keys: usize,
    pub(crate) total_edges: usize,
    pub(crate) unique_nodes: usize,
    pub(crate) num_leaves: usize,
    pub(crate) structurally_unique_nodes: usize,
    pub(crate) structural_redundancy: f64,
    pub(crate) num_redundant_nodes: usize,
    pub(crate) max_depth: usize,
    pub(crate) average_depth: f64,
    pub(crate) merge_points: usize,
    pub(crate) max_predecessors_with_values: usize,
    pub(crate) average_predecessors_with_values: f64,
}

#[time_it]
pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    for root_node in roots {
        queue.push_back(*root_node);
    }

    while let Some(node) = queue.pop_front() {
        let ptr = node as *const _;
        if !visited.insert(ptr) {
            continue;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(node.depth());

        if node.is_leaf() {
            stats.num_leaves += 1;
        } else {
            for (_, pred) in node.pop() {
                stats.total_edges += 1;
                queue.push_back(pred);
            }
        }
    }

    stats
}

pub(crate) fn find_longest_path(root_node: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    if root_node.is_leaf() {
        return None;
    }

    fn find_longest_recursive(
        node_arc: &Arc<GSSNode>,
        memo: &mut HashMap<*const GSSNode, Vec<(ParseStateEdgeContent, Arc<GSSNode>)>>,
    ) -> Vec<(ParseStateEdgeContent, Arc<GSSNode>)> {
        let node_ptr = Arc::as_ptr(node_arc);
        if let Some(cached) = memo.get(&node_ptr) {
            return cached.clone();
        }

        if node_arc.is_leaf() {
            return Vec::new();
        }

        let mut longest_path = Vec::new();
        for (edge_val, pred_arc) in node_arc.pop() {
            let mut path_from_pred = find_longest_recursive(pred_arc, memo);
            path_from_pred.insert(0, (edge_val.clone(), pred_arc.clone()));
            if path_from_pred.len() > longest_path.len() {
                longest_path = path_from_pred;
            }
        }

        memo.insert(node_ptr, longest_path.clone());
        longest_path
    }

    let mut memo = HashMap::new();
    let path = find_longest_recursive(root_node, &mut memo);
    if path.is_empty() { None } else { Some(path) }
}

#[allow(dead_code)] pub(crate) fn sample_path(roots: &[&GSSNode], seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    if roots.is_empty() {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let root_index = rng.gen_range(0..roots.len());
    let mut current_node = roots[root_index];

    let mut path = Vec::new();

    loop {
        if current_node.is_leaf() {
            break;
        }

        let predecessors: Vec<_> = current_node.pop().collect();
        if predecessors.is_empty() {
            break;
        }

        let chosen_index = rng.gen_range(0..predecessors.len());
        let (edge, pred) = predecessors[chosen_index];

        path.push(edge.clone());
        current_node = pred;
    }
    path.reverse();
    Some(path)
}


#[derive(Default)]
pub struct GSSPrintConfig<'a> {
    pub(crate) labels: Option<&'a [String]>,
    pub(crate) max_edges: usize,
    pub(crate) original_internal_bimap: Option<&'a BTreeMap<usize, usize>>,
    pub(crate) llm_token_map: Option<&'a BiBTreeMap<Vec<u8>, LLMTokenID>>,
    pub(crate) verbose: bool,
}

pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    config: &GSSPrintConfig,
) -> (String, Vec<StateID>) {
    fn print_recursive(
        node_arc: &Arc<GSSNode>,
        node_ids: &mut HashMap<*const GSSNode, usize>,
        visited_nodes: &mut HashSet<*const GSSNode>,
        prefix: &str,
        node_count: &mut usize,
        output: &mut String,
        terminal_map: &BiBTreeMap<Terminal, TerminalID>,
        state_ids_in_order: &mut Vec<StateID>,
        seen_state_ids: &mut HashSet<StateID>,
        config: &GSSPrintConfig,
    ) -> Result<(), std::fmt::Error> {
        let node_ptr = Arc::as_ptr(node_arc);
        if visited_nodes.contains(&node_ptr) {
            return Ok(());
        }
        visited_nodes.insert(node_ptr);

        let predecessors: Vec<_> = node_arc.pop().collect();

        for (i, (edge_val, pred_arc)) in predecessors.iter().enumerate() {
            if *node_count >= config.max_edges {
                writeln!(output, "{}... (Truncated)", prefix)?;
                return Ok(());
            }

            let is_last = i == predecessors.len() - 1;
            let connector = if is_last { "└──" } else { "├──" };
            let new_prefix = if is_last {
                format!("{}  ", prefix)
            } else {
                format!("{}│ ", prefix)
            };

            let pred_ptr = Arc::as_ptr(pred_arc);
            let node_ids_len = node_ids.len();
            let pred_id = *node_ids.entry(pred_ptr).or_insert(node_ids_len);

            if seen_state_ids.insert(edge_val.state_id) {
                state_ids_in_order.push(edge_val.state_id);
            }

            let acc_child = if pred_arc.is_leaf() {
                format_acc(pred_arc.leaf_data().unwrap(), terminal_map, config)
            } else {
                String::new()
            };

            writeln!(
                output,
                "{}{} edge {} -> Node {} {}",
                prefix, connector, edge_val.state_id.0, pred_id, acc_child,
            )?;
            *node_count += 1;

            print_recursive(
                pred_arc, node_ids, visited_nodes, &new_prefix, node_count,
                output, terminal_map, state_ids_in_order, seen_state_ids, config,
            )?;
        }
        Ok(())
    }

    let mut node_ids = HashMap::new();
    let mut visited_nodes = HashSet::new();
    let mut count = 0;
    let mut out_str = String::new();
    let mut state_ids_in_order = Vec::new();
    let mut seen_state_ids = HashSet::new();

    if roots.is_empty() { return ("GSS Forest: (No roots)".to_string(), state_ids_in_order); }
    writeln!(&mut out_str, "GSS Forest (Max Edges: {}):", config.max_edges).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        if count >= config.max_edges {
            writeln!(&mut out_str, "... (Truncated)").unwrap();
            break;
        }

        let root_ptr = Arc::as_ptr(root_arc);
        let node_ids_len = node_ids.len();
        let root_id = *node_ids.entry(root_ptr).or_insert(node_ids_len);

        let acc_str = if root_arc.is_leaf() {
            format_acc(root_arc.leaf_data().unwrap(), terminal_map, config)
        } else {
            String::new()
        };
        let root_label = config.labels.map_or_else(|| format!("Root {}", i), |l| l[i].clone());

        writeln!(&mut out_str, "{}: Node {} {}", root_label, root_id, acc_str).unwrap();
        count += 1;

        let _ = print_recursive(
            root_arc, &mut node_ids, &mut visited_nodes, "  ", &mut count,
            &mut out_str, terminal_map, &mut state_ids_in_order, &mut seen_state_ids, config,
        );
    }

    (out_str, state_ids_in_order)
}

pub(crate) fn format_acc(
    acc: &Acc,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    config: &GSSPrintConfig,
) -> String {
    let summarize_llm = |bv: &HybridBitset, label: &str| -> Option<String> {
        if *bv == HybridBitset::max_ones() {
            return None;
        }
        if bv.is_empty() {
            return Some(format!("{}=∅", label));
        }
        Some(format!("{}({})", label, bv.len()))
    };

    let summarize_disallowed_terminals = |allowed_terminals: &HybridL2Bitset, label: &str| -> Option<String> {
        if allowed_terminals.is_all() {
            return None;
        }
        Some(format!("Disallowed {}(...)", label))
    };

    let union_llm_opt = summarize_llm(&acc.llm_tokens_union, "LLM(U)");
    let union_terminals_opt = summarize_disallowed_terminals(&acc.terminals_union, "Term(U)");

    let stored_trie_nodes_str = {
        let n = acc.stored_trie_nodes.len();
        if n == 0 {
            None
        } else {
            Some(format!("Trie(n={})", n))
        }
    };

    let mut parts: Vec<String> = Vec::new();
    if let Some(s) = union_llm_opt { parts.push(s); }
    if let Some(s) = union_terminals_opt { parts.push(s); }
    if let Some(s) = stored_trie_nodes_str { parts.push(s); }

    if parts.is_empty() {
        String::new()
    } else {
        format!("[{}]", parts.join(", "))
    }
}

#[time_it]
pub(crate) fn deep_add_precompute_trie_edges(
    root_arc: &mut Arc<GSSNode>,
    god: &StoredTrieGodWrapper,
    edge_key: &(usize, LLMTokenBV),
    edge_value: &StateIDBV,
    tokens_for_update: &LLMTokenBV,
    destination_provider: &mut impl FnMut() -> PrecomputeNode3Index,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut closure = |acc: &mut Acc| {
        if !acc.stored_trie_nodes().is_empty() {
            let destination = destination_provider();

            for source_wrapper in acc.stored_trie_nodes() {
                let source_arc = source_wrapper.as_arc().clone();

                let inserter = EdgeInserter::new(
                    god,
                    source_arc,
                    edge_key.clone(),
                    edge_value.clone(),
                    |e, n| *e |= n,
                    |node_value, _edge_value| node_value.live_tokens |= tokens_for_update,
                    |_, _| {}, // Unconditional insertion
                );
                inserter.try_destination(destination.clone()).expect("Cycle detected when adding precompute trie edges");
            }

            destination.write(god).expect("poison").value.live_tokens |= tokens_for_update;

            *acc.stored_trie_nodes_mut() = BTreeSet::from([destination]);
        }
        false
    };

    if let Some(new_root) = transform_recursive(root_arc, &mut closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = new_fresh();
    }
}
