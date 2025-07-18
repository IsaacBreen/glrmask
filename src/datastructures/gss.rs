use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use bimap::BiBTreeMap;
use deterministic_hash::DeterministicHasher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::glr::parser::ParseStateEdgeContent;
use crate::constraint::{LLMTokenBV, LLMVocab, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::glr::grammar::Terminal;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
use std::ops::{BitOr, BitOrAssign};
use profiler_macro::{time_it, timeit};
// --- Type Aliases ---

pub type MaxDepth = usize;
/// Maps a node's depth to its predecessors at that depth.
type NodeMap = BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, Arc<GSSNode>>>;
/// A cache for structurally unique nodes, mapping a predecessor structure to a canonical node.
type NodeCache = HashMap<NodeMap, Arc<GSSNode>>;
/// A temporary set of predecessors used during node construction and simplification.
type NodeSet = BTreeSet<(Arc<GSSNode>, ParseStateEdgeContent)>;

/// Represents the set of disallowed LLM tokens for a path. `None` means no tokens are disallowed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenInfo {
    llm_tokens: Option<LLMTokenBV>,
    llm_vocab: Option<Arc<LLMVocab>>,
}
impl LLMTokenInfo {
    pub fn none(llm_vocab: Option<Arc<LLMVocab>>) -> Self {
        Self { llm_tokens: None, llm_vocab }
    }
    pub fn all(llm_vocab: Option<Arc<LLMVocab>>) -> Self {
        let mut this = Self::none(llm_vocab.clone());
        this.llm_tokens = Some(LLMTokenBV::ones(this.max_num_llm_tokens()));
        this
    }
    pub fn disallowed(&self) -> LLMTokenBV {
        self.llm_tokens.clone().unwrap_or_else(LLMTokenBV::zeros)
    }
    pub fn allowed(&self) -> LLMTokenBV {
        let all_tokens = LLMTokenBV::ones(self.max_num_llm_tokens());
        all_tokens - self.disallowed()
    }
    pub fn is_empty(&self) -> bool {
        self.llm_tokens.is_none() || self.llm_tokens.as_ref().unwrap().is_empty()
    }
    pub fn is_all(&self) -> bool {
        self.disallowed() == LLMTokenBV::ones(self.max_num_llm_tokens())
    }
    pub fn llm_vocab(&self) -> &Option<Arc<LLMVocab>> {
        &self.llm_vocab
    }
    pub fn max_num_llm_tokens(&self) -> usize {
        self.llm_vocab.as_ref().map_or(usize::MAX, |vocab| vocab.internal_max_llm_token.saturating_add(1))
    }
}

impl BitOr<&LLMTokenBV> for LLMTokenInfo {
    type Output = Self;
    fn bitor(mut self, rhs: &LLMTokenBV) -> Self::Output {
        if self.llm_tokens.is_none() {
            if !rhs.is_empty() {
                self.llm_tokens = Some(rhs.clone());
            }
        } else {
            self.llm_tokens.as_mut().unwrap().bitor_assign(rhs);
        }
        self
    }
}

impl BitOrAssign<&LLMTokenBV> for LLMTokenInfo {
    fn bitor_assign(&mut self, rhs: &LLMTokenBV) {
        if self.llm_tokens.is_none() {
            if !rhs.is_empty() {
                self.llm_tokens = Some(rhs.clone());
            }
        } else {
            self.llm_tokens.as_mut().unwrap().bitor_assign(rhs);
        }
    }
}

/// For a given tokenizer state, holds the bitvector of disallowed terminals.
pub type TerminalInfo = BTreeMap<TokenizerStateID, TerminalBV>;


// --- Accumulator (Acc) & AccManager ---

/// Represents a set of constraints (disallowed tokens/terminals) for a GSS path or node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    llm_token_info: LLMTokenInfo,
    disallowed_terminals: TerminalInfo,
}

impl Acc {
    pub fn new(llm_token_info: LLMTokenInfo, disallowed_terminals: TerminalInfo) -> Self {
        Self { llm_token_info, disallowed_terminals }
    }

    /// Creates a fresh, unconstrained accumulator.
    pub fn new_fresh(llm_vocab: Option<Arc<LLMVocab>>) -> Self {
        Self {
            llm_token_info: LLMTokenInfo::none(llm_vocab),
            disallowed_terminals: BTreeMap::new(),
        }
    }

    pub fn llm_tokens(&self) -> &LLMTokenInfo { &self.llm_token_info }
    pub fn llm_tokens_mut(&mut self) -> &mut LLMTokenInfo { &mut self.llm_token_info }
    pub fn disallowed_terminals(&self) -> &TerminalInfo { &self.disallowed_terminals }
    pub fn disallowed_terminals_mut(&mut self) -> &mut TerminalInfo { &mut self.disallowed_terminals }

    /// Checks if the accumulator is in its default, unconstrained state.
    pub fn is_empty(&self) -> bool {
        self.llm_token_info.is_empty() && self.disallowed_terminals.is_empty()
    }

    /// Checks if the path is dead (e.g., allows no LLM tokens).
    pub fn is_dead(&self) -> bool {
        self.llm_token_info.is_all()
    }

    pub fn is_alive(&self) -> bool { !self.is_dead() }

    /// Accumulates constraints sequentially (e.g., adding a new constraint to a path).
    /// This is a union of constraints.
    // #[time_it]
    pub fn accumulate_seq(&self, other: &Self) -> Self {
        // LLM tokens: union of disallowed sets
        let mut new_llm_tokens = self.llm_token_info.disallowed();
        new_llm_tokens |= &other.llm_token_info.disallowed();
        let new_llm_info = LLMTokenInfo {
            llm_tokens: if new_llm_tokens.is_empty() { None } else { Some(new_llm_tokens) },
            llm_vocab: self.llm_token_info.llm_vocab().clone().or_else(|| other.llm_token_info.llm_vocab().clone()),
        };

        // Terminals: union of disallowed sets
        let mut new_disallowed_terminals = self.disallowed_terminals.clone();
        for (state_id, other_bv) in &other.disallowed_terminals {
            *new_disallowed_terminals.entry(*state_id).or_insert_with(HybridBitset::zeros) |= other_bv;
        }
        new_disallowed_terminals.retain(|_, v| !v.is_empty());

        Acc {
            llm_token_info: new_llm_info,
            disallowed_terminals: new_disallowed_terminals,
        }
    }

    /// Merges constraints from parallel paths (union of paths).
    // #[time_it]
    pub fn merge_parallel<'a>(accs: impl IntoIterator<Item = &'a Acc>, llm_vocab: Option<Arc<LLMVocab>>) -> Self {
        let accs_vec: Vec<&'a Acc> = accs.into_iter().collect();
        if accs_vec.is_empty() {
            return Acc::new_fresh(llm_vocab);
        }

        // LLM tokens: intersection of disallowed sets.
        // If path A disallows DA and path B disallows DB, the merged path allows (!DA | !DB),
        // which means it disallows (DA & DB).
        let mut merged_llm_bv = LLMTokenBV::zeros();
        for acc in &accs_vec {
            merged_llm_bv |= &acc.llm_token_info.disallowed();
        }
        let merged_llm_info = LLMTokenInfo {
            llm_tokens: if merged_llm_bv.is_empty() { None } else { Some(merged_llm_bv) },
            llm_vocab,
        };

        // Terminals: union of disallowed sets.
        let mut merged_terminals = BTreeMap::new();
        for acc in &accs_vec {
            for (state_id, bv) in &acc.disallowed_terminals {
                *merged_terminals.entry(*state_id).or_insert_with(HybridBitset::zeros) |= bv;
            }
        }
        merged_terminals.retain(|_, v| !v.is_empty());

        Acc {
            llm_token_info: merged_llm_info,
            disallowed_terminals: merged_terminals,
        }
    }

    /// Intersects constraints from parallel paths.
    // #[time_it]
    pub fn intersect_parallel<'a>(accs: impl IntoIterator<Item = &'a Acc>, llm_vocab: Option<Arc<LLMVocab>>) -> Self {
        let accs_vec: Vec<&'a Acc> = accs.into_iter().collect();
        if accs_vec.is_empty() {
            return Acc::new_fresh(llm_vocab);
        }

        // LLM tokens: union of disallowed sets.
        let mut intersected_llm_bv = LLMTokenBV::zeros();
        for acc in &accs_vec {
            intersected_llm_bv &= &acc.llm_token_info.disallowed();
        }
        let intersected_llm_info = LLMTokenInfo {
            llm_tokens: if intersected_llm_bv.is_empty() { None } else { Some(intersected_llm_bv) },
            llm_vocab,
        };

        // Terminals: intersection of disallowed sets.
        let mut acc_iter = accs_vec.into_iter();
        let mut intersected_terminals = acc_iter.next().unwrap().disallowed_terminals.clone();
        for acc in acc_iter {
            intersected_terminals.retain(|state_id, bv| {
                if let Some(other_bv) = acc.disallowed_terminals.get(state_id) {
                    // Keep only those terminals that are disallowed in both accumulators.
                    *bv &= other_bv;
                    !bv.is_empty() // Retain only non-empty bitsets
                } else {
                    false // If the state_id is not present in the other accumulator, remove it
                }
            });
        }

        Acc {
            llm_token_info: intersected_llm_info,
            disallowed_terminals: intersected_terminals,
        }
    }
}

/// Manages the local and aggregated path accumulators for a GSS node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AccManager {
    /// Constraints applied locally at this node/edge.
    pub local: Arc<Acc>,
    /// The union of constraints over all paths from a root to this node (excluding local).
    pub union: Arc<Acc>,
    /// The intersection of constraints over all paths from a root to this node (excluding local).
    pub intersection: Arc<Acc>,
}


// --- GSS Node & Core Implementation ---

/// A node in the Graph-Structured Stack (GSS).
#[derive(Debug, Clone)]
pub struct GSSNode {
    acc_manager: AccManager,
    predecessors: NodeMap,
    hash_key_cache: u64,
    max_depth: MaxDepth,
}

/// Represents the result of a `pop` operation on a `GSSNode` or another `GSSPop`.
#[derive(Debug, Clone)]
pub struct GSSPop<'a> {
    pub parent_node: &'a GSSNode,
    pub node_map: NodeMap,
}

/// A read-only view into a single path segment of the GSS, from a parent to a predecessor.
#[derive(Clone, Copy)]
pub struct GSSPeek<'a> {
    pub(crate) parent_node: &'a GSSNode,
    edge_value: &'a ParseStateEdgeContent,
    pub(crate) predecessor_node: &'a Arc<GSSNode>,
}

// Helper functions for GSSNode construction
fn compute_max_depth(predecessors: &NodeMap) -> MaxDepth {
    predecessors.keys().next_back().map_or(0, |max_pred_depth| max_pred_depth + 1)
}

fn compute_hash_key(predecessors: &NodeMap, acc_manager: &AccManager) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    // acc_manager.hash(&mut hasher);
    for (depth, preds_for_depth) in predecessors {
        depth.hash(&mut hasher);
        for (edge_val, pred_arc) in preds_for_depth {
            edge_val.hash(&mut hasher);
            pred_arc.hash_key_cache.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Processes a set of incoming predecessors, grouping them by depth and edge,
/// and merging nodes that share the same edge to create a canonical `NodeMap`.
// #[time_it]
fn process_predecessors(incoming: &NodeSet) -> NodeMap {
    let mut grouped_by_depth: BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, Vec<Arc<GSSNode>>>> = BTreeMap::new();

    for (pred_arc, edge_val) in incoming {
        grouped_by_depth
            .entry(pred_arc.max_depth)
            .or_default()
            .entry(edge_val.clone())
            .or_default()
            .push(pred_arc.clone());
    }

    let mut result: NodeMap = BTreeMap::new();
    for (depth, grouped_by_edge) in grouped_by_depth {
        let mut result_for_depth = BTreeMap::new();
        for (edge_val, pred_arcs) in grouped_by_edge {
            if pred_arcs.is_empty() { continue; }

            let mut iter = pred_arcs.into_iter();
            let first = iter.next().unwrap();

            if iter.len() == 0 {
                result_for_depth.insert(edge_val, first);
            } else {
                let mut merged_node = (*first).clone();
                for other_arc in iter {
                    merged_node.merge(&other_arc);
                }
                result_for_depth.insert(edge_val, Arc::new(merged_node));
            }
        }
        if !result_for_depth.is_empty() {
            result.insert(depth, result_for_depth);
        }
    }
    result
}

/// Merges the `source` NodeMap into the `target` NodeMap.
// #[time_it]
fn merge_node_maps(target: &mut NodeMap, source: NodeMap) {
    for (depth, source_preds_for_depth) in source {
        let target_preds_for_depth = target.entry(depth).or_default();
        for (edge_val, source_pred_arc) in source_preds_for_depth {
            match target_preds_for_depth.entry(edge_val.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(source_pred_arc);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    Arc::make_mut(entry.get_mut()).merge(&source_pred_arc);
                }
            }
        }
    }
}

// Basic node creation and manipulation
impl GSSNode {
    /// Creates a new GSS root node with no predecessors.
    pub fn new(local_acc: Acc) -> Self {
        let llm_vocab = local_acc.llm_tokens().llm_vocab().clone();
        let acc_manager = AccManager {
            local: Arc::new(local_acc),
            union: Arc::new(Acc::new_fresh(llm_vocab.clone())),
            intersection: Arc::new(Acc::new_fresh(llm_vocab)),
        };
        let predecessors = NodeMap::new();
        let hash_key_cache = compute_hash_key(&predecessors, &acc_manager);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc_manager, predecessors, hash_key_cache, max_depth }
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    // #[time_it]
    fn new_with_map(local_acc: Arc<Acc>, predecessors: NodeMap) -> Self {
        let llm_vocab = local_acc.llm_tokens().llm_vocab().clone();

        let pred_full_unions: Vec<_> = predecessors.values().flat_map(|m| m.values()).map(|p| p.full_union_acc()).collect();
        let pred_full_intersections: Vec<_> = predecessors.values().flat_map(|m| m.values()).map(|p| p.full_intersection_acc()).collect();

        let final_union = Arc::new(Acc::merge_parallel(pred_full_unions.iter(), llm_vocab.clone()));
        let final_intersection = Arc::new(Acc::intersect_parallel(pred_full_intersections.iter(), llm_vocab));

        let acc_manager = AccManager {
            local: local_acc,
            union: final_union,
            intersection: final_intersection,
        };

        let hash_key_cache = compute_hash_key(&predecessors, &acc_manager);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc_manager, predecessors, hash_key_cache, max_depth }
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    // #[time_it]
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, local_acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        let mut inner_map = BTreeMap::new();
        inner_map.insert(edge_value, predecessor_arc.clone());
        predecessors_map.insert(predecessor_arc.max_depth, inner_map);
        Self::new_with_map(Arc::new(local_acc), predecessors_map)
    }

    pub fn predecessors(&self) -> &NodeMap { &self.predecessors }
    pub fn num_predecessors(&self) -> usize { self.predecessors.values().map(|inner_map| inner_map.len()).sum() }
    pub fn is_empty(&self) -> bool { self.predecessors.is_empty() }
    pub fn acc_manager(&self) -> &AccManager { &self.acc_manager }

    /// Returns the full union of constraints for any path ending at this node.
    // #[time_it]
    pub fn full_union_acc(&self) -> Acc {
        self.acc_manager.union.accumulate_seq(&self.acc_manager.local)
    }

    /// Returns the full intersection of constraints for all paths ending at this node.
    // #[time_it]
    pub fn full_intersection_acc(&self) -> Acc {
        self.acc_manager.intersection.accumulate_seq(&self.acc_manager.local)
    }

    pub fn llm_tokens(&self) -> LLMTokenInfo {
        self.full_union_acc().llm_token_info
    }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    // #[time_it]
    pub fn push(&self, edge_value: ParseStateEdgeContent, local_acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, local_acc_for_new_node)
    }

    /// Pops the top state from the stack(s), returning a `GSSPop` structure.
    /// The accumulators of predecessors are adjusted to include this node's local constraints.
    // #[time_it("GSSNode::pop")]
    pub fn pop(&self) -> GSSPop {
        let mut new_node_map = NodeMap::new();
        let parent_local_acc = &self.acc_manager.local;

        for (depth, preds_for_depth) in &self.predecessors {
            let mut new_preds_for_depth = BTreeMap::new();
            for (edge_val, pred_arc) in preds_for_depth {
                let mut new_pred_node = (**pred_arc).clone();

                // Create a new local accumulator for the popped node by accumulating the parent's local constraints.
                let new_local_acc = Arc::new(pred_arc.acc_manager.local.accumulate_seq(parent_local_acc));
                
                // Check for dead paths *after* accumulation.
                let new_full_union = new_pred_node.acc_manager.union.accumulate_seq(&new_local_acc);
                if new_full_union.is_dead() {
                    crate::debug!(6, "Dead path after accumulating\n{:?}\nwith local\n{:?}\nresulting in\n{:?}", new_pred_node.acc_manager.union, new_local_acc, new_full_union);
                    continue;
                }

                new_pred_node.acc_manager.local = new_local_acc;
                new_pred_node.hash_key_cache = compute_hash_key(&new_pred_node.predecessors, &new_pred_node.acc_manager);

                let new_pred_arc = Arc::new(new_pred_node);
                new_preds_for_depth.insert(edge_val.clone(), new_pred_arc);
            }
            if !new_preds_for_depth.is_empty() {
                new_node_map.insert(*depth, new_preds_for_depth);
            }
        }

        GSSPop { parent_node: self, node_map: new_node_map }
    }

    /// Pops `n` levels from the GSS.
    // #[time_it("GSSNode::popn")]
    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        self.pop().popn(n).to_node()
    }

    /// Merges another `GSSNode` into this one.
    // #[time_it]
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        if other.predecessors.is_empty() && other.acc_manager.local.is_empty() { return; }
        if self.predecessors.is_empty() && self.acc_manager.local.is_empty() {
            *self = other.clone();
            return;
        }

        // Merge local accumulators
        let llm_vocab = self.acc_manager.local.llm_tokens().llm_vocab().clone().or_else(|| other.acc_manager.local.llm_tokens().llm_vocab().clone());
        let merged_local = Acc::merge_parallel([self.acc_manager.local.as_ref(), other.acc_manager.local.as_ref()], llm_vocab);
        
        // Merge predecessor maps
        let mut new_predecessors = self.predecessors.clone();
        merge_node_maps(&mut new_predecessors, other.predecessors.clone());
        
        // Create a new node with the merged properties.
        let new_node = GSSNode::new_with_map(Arc::new(merged_local), new_predecessors);
        *self = new_node;
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    // #[time_it]
    pub fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let acc = (*self.acc_manager.local).clone();
        self.push(edge_value, acc)
    }
    
    /// Returns an iterator over all direct predecessor paths (`GSSPeek`s).
    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPeek<'_>> {
        self.predecessors.values().flat_map(|m| m.iter()).map(|(edge_val, pred_arc)| {
            GSSPeek {
                parent_node: self,
                edge_value: edge_val,
                predecessor_node: pred_arc,
            }
        })
    }
}

impl GSSPop<'_> {
    fn _pop(node_map: &NodeMap) -> NodeMap {
        let mut combined_node_map = NodeMap::new();
        for node_arc in node_map.values().flat_map(|m| m.values()) {
            let popped = node_arc.pop();
            merge_node_maps(&mut combined_node_map, popped.node_map);
        }
        combined_node_map
    }

    // #[time_it]
    pub fn pop(&self) -> GSSPop {
        let node_map = Self::_pop(&self.node_map);
        GSSPop { parent_node: self.parent_node, node_map }
    }

    // #[time_it("GSSPop::popn")]
    pub fn popn(&self, n: usize) -> GSSPop {
        if n == 0 {
            return self.clone();
        }
        let mut current = self.node_map.clone();
        for _ in 0..n {
            current = Self::_pop(&current);
        }
        GSSPop { parent_node: self.parent_node, node_map: current }
    }

    /// Converts the `GSSPop` into a single `GSSNode`.
    // #[time_it("GSSPop::to_node")]
    pub fn to_node(&self) -> GSSNode {
        let llm_vocab = self.parent_node.llm_tokens().llm_vocab().clone();
        // The new node is an aggregation point, so it has no local constraints of its own.
        let local_acc = Arc::new(Acc::new_fresh(llm_vocab));
        GSSNode::new_with_map(local_acc, self.node_map.clone())
    }
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }
    pub fn predecessor(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    // #[time_it]
    pub fn to_node(&self) -> GSSNode {
        let local_acc = self.parent_node.acc_manager.local.accumulate_seq(&self.predecessor_node.full_union_acc());
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            local_acc,
        )
    }

    pub fn to_arc_node(&self) -> Arc<GSSNode> {
        Arc::new(self.to_node())
    }

    // #[time_it("GSSPeek::popn")]
    pub fn popn(&self, n: usize) -> Arc<GSSNode> {
        Arc::new(self.to_arc_node().popn(n))
    }
}

// Trait implementations for GSSNode
impl Hash for GSSNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
    }
}

impl PartialEq for GSSNode {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (
            self.hash_key_cache == other.hash_key_cache &&
            self.acc_manager == other.acc_manager &&
            self.predecessors == other.predecessors
        )
    }
}

impl Eq for GSSNode {}

impl PartialOrd for GSSNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GSSNode {
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; }
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.acc_manager.cmp(&other.acc_manager))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}


// --- Pruning and Transformation ---

fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    closure: &impl Fn(&GSSNode) -> Option<(Acc, bool)>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(node_arc.as_ref()) {
        None => { // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_local_acc, continue_recursion)) => {
            let new_predecessors_set = if continue_recursion {
                node_arc.predecessors.values().flat_map(|m| m.iter())
                    .filter_map(|(edge_val, pred_arc)| {
                        prune_and_transform_recursive(pred_arc, closure, memo)
                            .map(|new_pred_arc| (new_pred_arc, edge_val.clone()))
                    })
                    .collect::<NodeSet>()
            } else { // Don't recurse, keep existing predecessors.
                node_arc.predecessors.values().flat_map(|m| m.iter())
                    .map(|(edge_val, pred_arc)| (pred_arc.clone(), edge_val.clone()))
                    .collect::<NodeSet>()
            };

            if new_predecessors_set.is_empty() && !node_arc.predecessors.is_empty() {
                memo.insert(node_ptr, None);
                return None;
            }

            let new_node_predecessors_map = process_predecessors(&new_predecessors_set);
            let transformed_node = GSSNode::new_with_map(Arc::new(new_local_acc), new_node_predecessors_map);

            let result_arc = Arc::new(transformed_node);
            memo.insert(node_ptr, Some(result_arc.clone()));
            Some(result_arc)
        }
    }
}

// #[time_it]
pub fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let newly_disallowed = LLMTokenBV::ones(root_arc.llm_tokens().max_num_llm_tokens()) - allowed_tokens.clone();
    disallow_llm_tokens_and_prune_arc(
        root_arc,
        &newly_disallowed,
        memo,
    );
}

// #[time_it]
pub fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_local_acc = (*node.acc_manager.local).clone();
        new_local_acc.llm_tokens_mut().bitor_assign(tokens_to_disallow);

        let temp_full_acc = node.full_union_acc().accumulate_seq(&new_local_acc);
        if temp_full_acc.is_alive() {
            Some((new_local_acc, false))
        } else {
            None
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh(root_arc.llm_tokens().llm_vocab().clone())));
    }
}

// #[time_it]
pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_local_acc = (*node.acc_manager.local).clone();
        let continue_recursion = !new_local_acc.llm_tokens().is_empty();
        new_local_acc.llm_token_info = LLMTokenInfo::none(new_local_acc.llm_tokens().llm_vocab().clone());
        Some((new_local_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh(root_arc.llm_tokens().llm_vocab().clone())));
    }
}

// #[time_it]
pub fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_local_acc = (*node.acc_manager.local).clone();
        for (state_id, bv) in disallowed_terminals {
            *new_local_acc.disallowed_terminals_mut().entry(*state_id).or_insert_with(HybridBitset::zeros) |= bv;
        }
        Some((new_local_acc, false))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh(root_arc.llm_tokens().llm_vocab().clone())));
    }
}

// #[time_it]
pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        for (state_id, disallowed_by_gss) in node.full_intersection_acc().disallowed_terminals() {
            if let Some(matched) = matched_terminals.get(state_id) {
                if !disallowed_by_gss.is_disjoint(matched) {
                    return None; // All paths to this node disallow a terminal that was just matched. Prune.
                }
            }
        }
        // If we can't prune the whole node, we might need to recurse to prune sub-paths.
        // This is determined by checking the union accumulator.
        let mut needs_recursion = false;
        for (state_id, disallowed_by_gss_union) in node.full_union_acc().disallowed_terminals() {
             if let Some(matched) = matched_terminals.get(state_id) {
                if !disallowed_by_gss_union.is_disjoint(matched) {
                    needs_recursion = true;
                    break;
                }
            }
        }
        Some(((*node.acc_manager.local).clone(), true))
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh(root_arc.llm_tokens().llm_vocab().clone())));
    }
}

// #[time_it]
pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_local_acc = (*node.acc_manager.local).clone();
        let mut new_disallowed = BTreeMap::new();
        let mut changed = false;

        for (old_id, bv) in new_local_acc.disallowed_terminals() {
            if let Some(&new_id) = map.get(old_id) {
                *new_disallowed.entry(new_id).or_insert_with(HybridBitset::zeros) |= bv;
                if old_id != &new_id {
                    changed = true;
                }
            } else {
                changed = true; // A state was removed.
            }
        }
        
        if new_disallowed.len() != new_local_acc.disallowed_terminals().len() {
            changed = true;
        }

        new_local_acc.disallowed_terminals = new_disallowed;
        Some((new_local_acc, true))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh(root_arc.llm_tokens().llm_vocab().clone())));
    }
}


// --- Simplification ---

fn simplify_node_recursive(
    node_arc: &Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
    cache: &mut NodeCache,
) -> Arc<GSSNode> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(simplified_arc) = memo.get(&node_ptr) {
        return simplified_arc.clone();
    }

    let simplified_predecessors_set: NodeSet = node_arc.predecessors.values().flat_map(|m| m.iter())
        .map(|(edge_val, pred_arc)| {
            let simplified_pred_arc = simplify_node_recursive(pred_arc, memo, cache);
            (simplified_pred_arc, edge_val.clone())
        })
        .collect();

    let simplified_predecessors_map = process_predecessors(&simplified_predecessors_set);

    let cached_structural_node = cache.entry(simplified_predecessors_map.clone())
        .or_insert_with(|| {
            let llm_vocab = node_arc.llm_tokens().llm_vocab().clone();
            let canonical_local_acc = Arc::new(Acc::new_fresh(llm_vocab));
            Arc::new(GSSNode::new_with_map(canonical_local_acc, simplified_predecessors_map))
        });

    // Create the final simplified node. It has the canonical structure, but with the original local accumulator.
    // The union/intersection accumulators are re-calculated based on the simplified predecessors.
    let final_node = GSSNode::new_with_map(
        node_arc.acc_manager.local.clone(),
        cached_structural_node.predecessors.clone(),
    );

    let result_arc = Arc::new(final_node);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}

impl GSSNode {
    pub fn simplify(&mut self) {
        let temp_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new();
        let simplified_arc = simplify_node_recursive(&temp_arc, &mut memo, &mut cache);
        *self = Arc::try_unwrap(simplified_arc).unwrap_or_else(|arc| (*arc).clone());
    }

    pub fn simplify_together(nodes: &mut [&mut Arc<Self>]) {
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new();
        for node_arc_ref_mut in nodes {
            let current_arc = (*node_arc_ref_mut).clone();
            let simplified_arc = simplify_node_recursive(&current_arc, &mut memo, &mut cache);
            **node_arc_ref_mut = simplified_arc;
        }
    }

    /// Fuses predecessor nodes that share the same edge value, even if they are at different depths.
    /// This can simplify the GSS by reducing path diversity. The fusion process is applied
    /// recursively for `levels` number of levels down from this node.
    ///
    /// For example, if a node has predecessors `(A, edgeX)` at depth 5 and `(B, edgeX)` at depth 3,
    /// they will be merged into a single predecessor `(merged(A,B), edgeX)`. This process
    /// continues recursively down the graph structure.
    ///
    /// The process is post-order: children are fused before their parents. This means that
    /// deeper parts of the graph are simplified first.
    // #[time_it]
    pub fn fuse_predecessors(&mut self, levels: usize) {
        if levels == 0 {
            return;
        }
        let temp_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let fused_arc = fuse_predecessors_recursive(&temp_arc, levels, &mut memo);
        *self = Arc::try_unwrap(fused_arc).unwrap_or_else(|arc| (*arc).clone());
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

    // 1. Recursively fuse the predecessors first (post-order traversal).
    let mut recursively_fused_predecessors = Vec::new();
    for (_, preds_for_depth) in &node_arc.predecessors {
        for (edge_val, pred_arc) in preds_for_depth {
            let fused_pred_arc = fuse_predecessors_recursive(pred_arc, levels - 1, memo);
            recursively_fused_predecessors.push((edge_val.clone(), fused_pred_arc));
        }
    }

    // 2. Group the now-fused predecessors by their edge value.
    let mut grouped_by_edge = BTreeMap::<ParseStateEdgeContent, Vec<Arc<GSSNode>>>::new();
    for (edge_val, pred_arc) in recursively_fused_predecessors {
        grouped_by_edge.entry(edge_val).or_default().push(pred_arc);
    }

    // 3. For each edge value, merge all predecessors associated with it into a single node.
    let mut new_predecessors_set = NodeSet::new();
    for (edge_val, pred_arcs_to_merge) in grouped_by_edge {
        if pred_arcs_to_merge.is_empty() { continue; }

        let mut iter = pred_arcs_to_merge.into_iter();
        let first = iter.next().unwrap();

        let final_pred_arc = if iter.len() == 0 {
            first
        } else {
            let mut merged_node = (*first).clone();
            for other_arc in iter {
                merged_node.merge(&other_arc);
            }
            Arc::new(merged_node)
        };
        new_predecessors_set.insert((final_pred_arc, edge_val));
    }

    // 4. Rebuild the current node with the new, fused set of predecessors.
    let new_predecessors_map = process_predecessors(&new_predecessors_set);
    let fused_node = GSSNode::new_with_map(node_arc.acc_manager.local.clone(), new_predecessors_map);

    let result_arc = Arc::new(fused_node);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}

// --- Analysis and Debugging ---

impl GSSNode {
    pub fn reset_llm_tokens(&mut self) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        reset_llm_tokens(&mut node_arc, &mut memo);
        *self = Arc::try_unwrap(node_arc).unwrap_or_else(|arc| (*arc).clone());
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GSSStats {
    pub num_roots: usize,
    pub unique_nodes: usize,
    pub structurally_unique_nodes: usize,
    pub structural_redundancy: f64,
    pub max_depth: usize,
    pub average_depth: f64,
    pub merge_points: usize,
    pub max_predecessors_with_values: usize,
    pub average_predecessors_with_values: f64,
}

/// Gathers statistics about the structure and complexity of a GSS forest.
// #[time_it]
pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut total_depth = 0u64;
    let mut total_preds = 0u64;

    for root_node in roots {
        let node_ptr = *root_node as *const GSSNode;
        if visited.insert(node_ptr) {
            queue.push_back((*root_node, 0));
        }
    }

    // Reset visited for the main traversal to correctly process all nodes.
    visited.clear();

    while let Some((node, depth)) = queue.pop_front() {
        let node_ptr = node as *const GSSNode;
        if !visited.insert(node_ptr) {
            continue;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        total_depth += depth as u64;

        let num_preds = node.num_predecessors();
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds);
        total_preds += num_preds as u64;

        let unique_pred_arcs: HashSet<_> = node.predecessors.values().flat_map(|m| m.values())
            .map(Arc::as_ptr)
            .collect();
        if unique_pred_arcs.len() > 1 {
            stats.merge_points += 1;
        }

        for pred_arc in node.predecessors.values().flat_map(|m| m.values()) {
            queue.push_back((pred_arc.as_ref(), depth + 1));
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }

    // Calculate structural uniqueness
    let mut structural_memo = HashMap::new();
    let mut structural_cache = BTreeMap::new();
    for root_node in roots {
        get_structural_id(root_node, &mut structural_memo, &mut structural_cache);
    }
    stats.structurally_unique_nodes = structural_cache.len();
    if stats.unique_nodes > 0 {
        stats.structural_redundancy = 1.0 - (stats.structurally_unique_nodes as f64 / stats.unique_nodes as f64);
    }
    stats
}

/// Helper for `gather_gss_stats` to compute a unique ID for a node's structure.
fn get_structural_id(
    node: &GSSNode,
    memo: &mut HashMap<*const GSSNode, usize>,
    structural_cache: &mut BTreeMap<BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, usize>>, usize>,
) -> usize {
    let node_ptr = node as *const GSSNode;
    if let Some(id) = memo.get(&node_ptr) {
        return *id;
    }

    let mut pred_structural_ids = BTreeMap::new();
    for (depth, preds_for_depth) in &node.predecessors {
        let mut inner_map = BTreeMap::new();
        for (edge_val, pred_arc) in preds_for_depth {
            let pred_id = get_structural_id(pred_arc.as_ref(), memo, structural_cache);
            inner_map.insert(edge_val.clone(), pred_id);
        }
        pred_structural_ids.insert(*depth, inner_map);
    }

    let next_id = structural_cache.len();
    let id = *structural_cache.entry(pred_structural_ids).or_insert(next_id);

    memo.insert(node_ptr, id);
    id
}

/// Finds the longest path from any leaf to the given root node.
/// Returns `None` if the node has no predecessors.
pub fn find_longest_path(root_node: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    if root_node.predecessors.is_empty() {
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

        if node_arc.predecessors.is_empty() {
            return Vec::new();
        }

        let mut longest_path = Vec::new();
        for (edge_val, pred_arc) in node_arc.predecessors.values().flat_map(|m| m.iter()) {
            let mut path_from_pred = find_longest_recursive(pred_arc, memo);
            // The path ends with the edge leading to `node_arc` and `node_arc` itself.
            path_from_pred.push((edge_val.clone(), node_arc.clone()));
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

/// Randomly samples a single path from a GSS forest.
///
/// A path is defined as a sequence of `ParseStateEdgeContent` from a root to a leaf.
/// The sampling process starts by picking a random root from the provided slice.
/// Then, it traverses down to a leaf by randomly selecting a predecessor at each step.
///
/// # Arguments
/// * `roots` - A slice of `GSSNode` references representing the roots of the forest.
/// * `seed` - A seed for the random number generator to ensure deterministic sampling.
///
/// # Returns
/// * `Some(Vec<ParseStateEdgeContent>)` containing the sampled path if roots are provided.
///   The path is ordered from the root-most edge to the leaf-most edge.
///   Returns an empty vector if a root is also a leaf.
/// * `None` if the `roots` slice is empty.
pub fn sample_path(roots: &[&GSSNode], seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    if roots.is_empty() {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let root_index = rng.gen_range(0..roots.len());
    let mut current_node = roots[root_index];

    let mut path = Vec::new();
    let mut temp_arc_storage: Arc<GSSNode>;

    loop {
        if current_node.is_empty() {
            break;
        }

        let predecessors: Vec<_> = current_node.peek_iter().collect();
        if predecessors.is_empty() {
            break;
        }

        let chosen_index = rng.gen_range(0..predecessors.len());
        let chosen_peek = &predecessors[chosen_index];

        path.push(chosen_peek.edge_value().clone());

        temp_arc_storage = chosen_peek.predecessor().clone();
        current_node = &temp_arc_storage;
    }

    Some(path)
}

/// Pretty-prints a GSS forest for debugging.
// #[time_it]
pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    labels: Option<&[String]>,
    max_nodes: usize,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) -> String {
    // Recursive helper to print predecessors.
    fn print_predecessors_recursive(
        node_arc: &Arc<GSSNode>,
        node_ids: &mut HashMap<*const GSSNode, usize>,
        visited_nodes: &mut HashSet<*const GSSNode>,
        prefix: &str,
        node_count: &mut usize,
        max_nodes: usize,
        output: &mut String,
        terminal_map: &BiBTreeMap<Terminal, TerminalID>,
        original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
        llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    ) -> Result<(), std::fmt::Error> {
        let node_ptr = Arc::as_ptr(node_arc);
        if visited_nodes.contains(&node_ptr) {
            return Ok(()); // Avoid re-printing children of shared nodes.
        }
        visited_nodes.insert(node_ptr);

        let predecessors: Vec<_> = node_arc.predecessors()
            .values()
            .flat_map(|m| m.iter())
            .collect();

        for (i, (edge_val, pred_arc)) in predecessors.iter().enumerate() {
            if *node_count >= max_nodes {
                writeln!(output, "{}... (Truncated)", prefix)?;
                return Ok(());
            }

            let is_last = i == predecessors.len() - 1;
            let connector = if is_last { "└──" } else { "├──" };
            let new_prefix = format!("{}  {}", prefix, if is_last { "  " } else { "│ " });

            let pred_ptr = Arc::as_ptr(pred_arc);
            let node_ids_len = node_ids.len();
            let pred_id = *node_ids.entry(pred_ptr).or_insert(node_ids_len);

            let acc_child = format_acc(pred_arc.as_ref(), terminal_map, original_internal_bimap, llm_token_map);
            writeln!(
                output,
                "{}{} Edge {:?} -> Node {} (depth {}) {}",
                prefix, connector, edge_val.state_id, pred_id, pred_arc.max_depth, acc_child,
            )?;
            *node_count += 1;

            print_predecessors_recursive(
                pred_arc, node_ids, visited_nodes, &new_prefix, node_count, max_nodes,
                output, terminal_map, original_internal_bimap, llm_token_map,
            )?;
        }
        Ok(())
    }

    let mut node_ids = HashMap::new();
    let mut visited_nodes = HashSet::new();
    let mut count = 0;
    let mut out_str = String::new();

    if roots.is_empty() { return "GSS Forest: (No roots)".to_string(); }
    writeln!(&mut out_str, "GSS Forest (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        if count >= max_nodes {
            writeln!(&mut out_str, "... (Truncated)").unwrap();
            break;
        }

        let root_ptr = Arc::as_ptr(root_arc);
        let node_ids_len = node_ids.len();
        let root_id = *node_ids.entry(root_ptr).or_insert(node_ids_len);

        let acc_str = format_acc(root_arc.as_ref(), terminal_map, original_internal_bimap, llm_token_map);
        let root_label = labels.map_or_else(|| format!("Root {}", i), |l| l[i].clone());

        writeln!(&mut out_str, "{}: Node {} (depth {}) {}", root_label, root_id, root_arc.max_depth, acc_str).unwrap();
        count += 1;

        let _ = print_predecessors_recursive(
            root_arc, &mut node_ids, &mut visited_nodes, "  ", &mut count, max_nodes,
            &mut out_str, terminal_map, original_internal_bimap, llm_token_map,
        );
    }

    out_str
}

/// Formats an accumulator for concise display in the GSS printout.
fn format_acc(
    node: &GSSNode,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) -> String {
    let format_single_acc = |acc: &Acc, label: &str| -> String {
        let disallowed_llm_info = acc.llm_tokens();
        let llm_info = if disallowed_llm_info.is_empty() {
            "LLM(None)".to_string()
        } else {
            let bv = disallowed_llm_info.disallowed();
            if let (Some(bimap), Some(token_map)) = (original_internal_bimap, llm_token_map) {
                const MAX_SAMPLES: usize = 3;
                let token_samples: Vec<_> = bv.iter().take(MAX_SAMPLES)
                    .filter_map(|internal_id| bimap.get_by_right(&internal_id))
                    .filter_map(|original_id| token_map.get_by_right(&LLMTokenID(*original_id)))
                    .map(|token_bytes| format!("{:?}", String::from_utf8_lossy(token_bytes)))
                    .collect();

                let samples_str = token_samples.join(", ");
                let total_tokens = bv.len();
                if total_tokens > MAX_SAMPLES {
                    format!("LLM({} tokens: [{}, ...])", total_tokens, samples_str)
                } else {
                    format!("LLM({} tokens: [{}])", total_tokens, samples_str)
                }
            } else {
                format!("LLM({} tokens)", bv.len())
            }
        };

        if acc.disallowed_terminals().is_empty() {
            return format!("{}:({})", label, llm_info);
        }

        let disallowed_info = acc.disallowed_terminals().iter()
            .map(|(state_id, bv)| {
                let names: Vec<_> = bv.iter()
                    .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                        .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.to_string()))
                    .collect();
                format!("State {}:[{}]", state_id.0, names.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; ");

        format!("{}:({}, Terminals({}))", label, llm_info, disallowed_info)
    };

    let local_str = format_single_acc(&node.acc_manager.local, "Local");
    let union_str = format_single_acc(&node.full_union_acc(), "FullUnion");

    format!("[{}, {}]", local_str, union_str)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::table::StateID;

    fn mock_acc(val: usize) -> Acc {
        let mut bv = LLMTokenBV::zeros();
        bv.insert(val);
        let disallowed_info = LLMTokenInfo { llm_tokens: Some(bv), llm_vocab: None };
        Acc::new(disallowed_info, Default::default())
    }

    fn empty_acc() -> Acc {
        Acc::new_fresh(None)
    }

    fn mock_edge(id: usize) -> ParseStateEdgeContent {
        ParseStateEdgeContent { state_id: StateID(id) }
    }

    #[test]
    fn test_gss_new_node() {
        let acc = mock_acc(1);
        let node = GSSNode::new(acc.clone());
        assert_eq!(*node.acc_manager.local, acc);
        assert!(node.acc_manager.union.is_empty());
        assert!(node.acc_manager.intersection.is_empty());
        assert!(node.predecessors.is_empty());
        assert_eq!(node.max_depth, 0);
    }

    #[test]
    fn test_gss_push() {
        let root = Arc::new(GSSNode::new(mock_acc(1)));
        let pushed = root.push(mock_edge(10), mock_acc(2));

        assert_eq!(pushed.max_depth, 1);
        assert_eq!(*pushed.acc_manager.local, mock_acc(2));
        
        // The union acc of the new node should be the full union of its predecessor.
        let expected_union = root.full_union_acc();
        assert_eq!(*pushed.acc_manager.union, expected_union);

        // The full union of the new node is its union + its local.
        let expected_full_union = expected_union.accumulate_seq(&mock_acc(2));
        assert_eq!(pushed.full_union_acc(), expected_full_union);
        assert_eq!(pushed.full_union_acc().llm_tokens().disallowed().iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn test_gss_pop() {
        let root = Arc::new(GSSNode::new(mock_acc(1)));
        let pushed = Arc::new(root.push(mock_edge(10), mock_acc(2)));
        
        let pop_result = pushed.pop();
        assert_eq!(pop_result.node_map.len(), 1);
        
        let popped_node_arc = pop_result.node_map.values().next().unwrap().values().next().unwrap();
        
        // Popping from `pushed` should yield `root`, but with its local acc updated
        // by `pushed`'s local acc.
        let expected_local = root.acc_manager.local.accumulate_seq(&pushed.acc_manager.local);
        assert_eq!(*popped_node_arc.acc_manager.local, expected_local);
        assert_eq!(popped_node_arc.acc_manager.local.llm_tokens().disallowed().iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn test_gss_merge() {
        // Path 1: 0 -> 10(acc1)
        let n0 = Arc::new(GSSNode::new(empty_acc()));
        let n1 = Arc::new(n0.push(mock_edge(0), mock_acc(1)));

        // Path 2: 0 -> 20(acc2)
        let n2 = Arc::new(n0.push(mock_edge(0), mock_acc(2)));

        // Merge n1 and n2
        let mut merged = (*n1).clone();
        merged.merge(&n2);

        // Merged local acc should be the parallel merge (intersection of disallowed)
        let expected_local = Acc::merge_parallel([n1.acc_manager.local.as_ref(), n2.acc_manager.local.as_ref()], None);
        assert!(expected_local.llm_tokens().is_empty()); // intersection of {1} and {2} is empty
        assert_eq!(*merged.acc_manager.local, expected_local);

        // Merged node should have one predecessor (n0) from two paths
        assert_eq!(merged.num_predecessors(), 1);

        // The full union of the merged node should be the parallel merge of the predecessors' full unions,
        // plus the merged local acc.
        let pred_full_union = n0.full_union_acc(); // empty
        let expected_union = Acc::merge_parallel([&pred_full_union, &pred_full_union], None); // still empty
        assert_eq!(*merged.acc_manager.union, expected_union);
        assert!(merged.full_union_acc().is_empty());
    }

    #[test]
    fn test_gss_simplification_basic() {
        // This test mimics the structure of the old test but uses the new AccManager logic.
        let acc_base = mock_acc(0); // Disallows {0}
        let acc_other = mock_acc(1); // Disallows {1}

        // Leaf nodes
        let n4_v1 = Arc::new(GSSNode::new(acc_base.clone()));
        let n4_v2 = Arc::new(GSSNode::new(acc_other.clone()));

        // Path 1
        let d1_orig = Arc::new(n4_v1.push(mock_edge(40), empty_acc()));
        let c1_orig = Arc::new(d1_orig.push(mock_edge(30), empty_acc()));
        let b1_orig = Arc::new(c1_orig.push(mock_edge(20), empty_acc()));

        // Path 2
        let d2_orig = Arc::new(n4_v2.push(mock_edge(40), empty_acc()));

        // Root node A1, merging Path 1 and Path 2
        let mut a1_preds_set = NodeSet::new();
        a1_preds_set.insert((b1_orig.clone(), mock_edge(10)));
        a1_preds_set.insert((d2_orig.clone(), mock_edge(10)));
        let a1_preds_map = process_predecessors(&a1_preds_set);
        let mut a1_orig = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), a1_preds_map));

        // --- Verification before simplification ---
        // Full union of b1 should be acc_base
        assert_eq!(b1_orig.full_union_acc(), acc_base);
        // Full union of d2 should be acc_other
        assert_eq!(d2_orig.full_union_acc(), acc_other);
        // Full union of a1 is the parallel merge of its predecessors' full unions.
        // merge_parallel of Disallowed{0} and Disallowed{1} is Disallowed{}.
        assert!(a1_orig.full_union_acc().is_empty());

        // --- Simplify ---
        let mut roots_to_simplify = vec![&mut a1_orig];
        GSSNode::simplify_together(&mut roots_to_simplify);
        let s_a1 = roots_to_simplify[0].clone();

        // --- Verification after simplification ---
        assert!(s_a1.full_union_acc().is_empty(), "A1 full union should be empty");
        assert_eq!(s_a1.predecessors.len(), 2, "A1 should have 2 predecessor maps for different depths");

        // Check path from B1
        let preds_at_depth_3 = s_a1.predecessors.get(&3).expect("No predecessors at depth 3");
        let s_b1 = preds_at_depth_3.get(&mock_edge(10)).expect("Edge 10 not found for depth 3 pred");
        assert_eq!(s_b1.full_union_acc(), acc_base, "Simplified B1 full union mismatch");

        // Check path from D2
        let preds_at_depth_1 = s_a1.predecessors.get(&1).expect("No predecessors at depth 1");
        let s_d2 = preds_at_depth_1.get(&mock_edge(10)).expect("Edge 10 not found for depth 1 pred");
        assert_eq!(s_d2.full_union_acc(), acc_other, "Simplified D2 full union mismatch");

        // Check leaf nodes
        let s_c1 = s_b1.predecessors.get(&2).unwrap().get(&mock_edge(20)).unwrap();
        let s_d1 = s_c1.predecessors.get(&1).unwrap().get(&mock_edge(30)).unwrap();
        let s_n4_from_d1 = s_d1.predecessors.get(&0).unwrap().get(&mock_edge(40)).unwrap();
        assert_eq!(s_n4_from_d1.full_union_acc(), acc_base);

        let s_n4_from_d2 = s_d2.predecessors.get(&0).unwrap().get(&mock_edge(40)).unwrap();
        assert_eq!(s_n4_from_d2.full_union_acc(), acc_other);

        // The two N4 leaf nodes should be different because their accumulators are different.
        assert_ne!(s_n4_from_d1, s_n4_from_d2);
        assert!(!Arc::ptr_eq(s_n4_from_d1, s_n4_from_d2));
        
        // The structures leading to the leaves should be distinct, not shared.
        assert!(!Arc::ptr_eq(s_d1, s_d2));
    }

    #[test]
    fn test_gss_fuse_predecessors() {
        // Structure:
        // root -> (B, edge 100) at depth 1
        // root -> (C, edge 100) at depth 3
        let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
        let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));
        let b = Arc::new(leaf1.push(mock_edge(1), empty_acc()));
        let c_tmp = Arc::new(leaf2.push(mock_edge(2), empty_acc()));
        let c_tmp2 = Arc::new(c_tmp.push(mock_edge(3), empty_acc()));
        let c = Arc::new(c_tmp2.push(mock_edge(4), empty_acc()));

        assert_eq!(b.max_depth, 1);
        assert_eq!(c.max_depth, 3);

        let mut preds_map = BTreeMap::new();
        let mut b_map = BTreeMap::new();
        b_map.insert(mock_edge(100), b.clone());
        preds_map.insert(b.max_depth, b_map);

        let mut c_map = BTreeMap::new();
        c_map.insert(mock_edge(100), c.clone());
        preds_map.insert(c.max_depth, c_map);

        let mut root = GSSNode::new_with_map(Arc::new(empty_acc()), preds_map);
        assert_eq!(root.num_predecessors(), 2);

        // Fuse predecessors of root (levels=1)
        root.fuse_predecessors(1);

        assert_eq!(root.num_predecessors(), 1);
        let fused_pred_arc = root.predecessors().values().next().unwrap().values().next().unwrap();

        let mut disallowed_vec: Vec<_> = fused_pred_arc.full_union_acc().llm_tokens().disallowed().iter().collect();
        disallowed_vec.sort();
        assert_eq!(disallowed_vec, vec![1, 2]);
        assert_eq!(fused_pred_arc.num_predecessors(), 2);
    }

    #[test]
    fn test_sample_path() {
        // Structure:
        // root -> (B, edge 10)
        // B -> (C, edge 20)
        // C -> (D, edge 30)
        // C -> (E, edge 40)
        // D, E are leaves
        let d = Arc::new(GSSNode::new(empty_acc()));
        let e = Arc::new(GSSNode::new(empty_acc()));

        let mut c_preds = NodeSet::new();
        c_preds.insert((d, mock_edge(30)));
        c_preds.insert((e, mock_edge(40)));
        let c_preds_map = process_predecessors(&c_preds);
        let c = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), c_preds_map));

        let b = Arc::new(c.push(mock_edge(20), empty_acc()));
        let root = b.push(mock_edge(10), empty_acc());

        // With seed 0, it should pick one path.
        let path1 = sample_path(&[&root], 0).unwrap();
        // With seed 1, it might pick the other path.
        let path2 = sample_path(&[&root], 1).unwrap();

        // Path is root -> B -> C -> (D or E). Edges are 10, 20, (30 or 40)
        assert_eq!(path1.len(), 3);
        assert_eq!(path1[0], mock_edge(10));
        assert_eq!(path1[1], mock_edge(20));
        assert!(path1[2] == mock_edge(30) || path1[2] == mock_edge(40));

        // Test for determinism.
        let path1_again = sample_path(&[&root], 0).unwrap();
        assert_eq!(path1, path1_again);
    }
}
