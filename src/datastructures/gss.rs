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
use crate::constraint::{LLMTokenBV, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::grammar::Terminal;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
use std::ops::{BitAnd, BitOr};
use profiler_macro::{time_it, timeit};

// --- Type Aliases ---

pub type MaxDepth = usize;
pub type DestKey = MaxDepth;
/// Maps a node's depth to its predecessors at that depth.
type NodeMap = BTreeMap<(ParseStateEdgeContent, DestKey), Arc<GSSNode>>;
/// A cache for structurally unique nodes, mapping a predecessor structure to a canonical node.
type NodeCache = HashMap<NodeMap, Arc<GSSNode>>;
/// A temporary set of predecessors used during node construction and simplification.
type NodeSet = BTreeSet<(Arc<GSSNode>, ParseStateEdgeContent)>;
/// A 2D bitset where L1 is tokenizer state and L2 is terminal ID.
pub type TerminalInfo = HybridL2Bitset;


// --- Accumulator (Acc) ---

/// Represents the full set of allowed tokens and terminals for a GSS node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    // Allowed LLM tokens
    local_union_llm_tokens: HybridBitset,
    local_intersection_llm_tokens: HybridBitset,
    path_union_llm_tokens: HybridBitset,
    path_intersection_llm_tokens: HybridBitset,

    // Allowed Terminals
    local_union_terminals: HybridL2Bitset,
    local_intersection_terminals: HybridL2Bitset,
    path_union_terminals: HybridL2Bitset,
    path_intersection_terminals: HybridL2Bitset,
}

impl Acc {
    /// Creates a fresh, unconstrained accumulator (all tokens/terminals allowed).
    pub fn new_fresh() -> Self {
        Self {
            local_union_llm_tokens: HybridBitset::max_ones(),
            local_intersection_llm_tokens: HybridBitset::max_ones(),
            path_union_llm_tokens: HybridBitset::zeros(),
            path_intersection_llm_tokens: HybridBitset::max_ones(),
            local_union_terminals: HybridL2Bitset::all(),
            local_intersection_terminals: HybridL2Bitset::all(),
            path_union_terminals: HybridL2Bitset::new(),
            path_intersection_terminals: HybridL2Bitset::all(),
        }
    }

    /// Creates an accumulator with specific local constraints for a root node.
    pub fn new_with_local_constraints(local_llm: HybridBitset, local_terminals: HybridL2Bitset) -> Self {
        Self {
            local_union_llm_tokens: local_llm.clone(),
            local_intersection_llm_tokens: local_llm,
            path_union_llm_tokens: HybridBitset::zeros(),
            path_intersection_llm_tokens: HybridBitset::max_ones(),
            local_union_terminals: local_terminals.clone(),
            local_intersection_terminals: local_terminals,
            path_union_terminals: HybridL2Bitset::new(),
            path_intersection_terminals: HybridL2Bitset::all(),
        }
    }

    /// Creates an accumulator for a new node from its local constraints and predecessors.
    fn from_preds<'a>(local: &Acc, pred_accs: impl IntoIterator<Item = &'a Arc<Acc>>) -> Self {
        let mut path_union_llm_tokens = HybridBitset::zeros();
        let mut path_intersection_llm_tokens = HybridBitset::max_ones();
        let mut path_union_terminals = HybridL2Bitset::new();
        let mut path_intersection_terminals = HybridL2Bitset::all();
        let mut has_preds = false;

        for p_acc in pred_accs {
            has_preds = true;
            let p_union_llm = p_acc.union_llm_tokens();
            path_union_llm_tokens |= &p_union_llm;
            let p_intersection_llm = p_acc.intersection_llm_tokens();
            path_intersection_llm_tokens &= &p_intersection_llm;

            let p_union_terminals = p_acc.union_terminals();
            path_union_terminals |= &p_union_terminals;
            let p_intersection_terminals = p_acc.intersection_terminals();
            path_intersection_terminals &= &p_intersection_terminals;
        }

        if !has_preds {
            return local.clone();
        }

        Self {
            local_union_llm_tokens: local.local_union_llm_tokens.clone(),
            local_intersection_llm_tokens: local.local_intersection_llm_tokens.clone(),
            path_union_llm_tokens,
            path_intersection_llm_tokens,
            local_union_terminals: local.local_union_terminals.clone(),
            local_intersection_terminals: local.local_intersection_terminals.clone(),
            path_union_terminals,
            path_intersection_terminals,
        }
    }

    // --- Accessors for final computed sets ---
    pub fn union_llm_tokens(&self) -> HybridBitset { &self.path_union_llm_tokens | &self.local_union_llm_tokens }
    pub fn intersection_llm_tokens(&self) -> HybridBitset { &self.path_intersection_llm_tokens & &self.local_intersection_llm_tokens }
    pub fn union_terminals(&self) -> HybridL2Bitset { &self.path_union_terminals | &self.local_union_terminals }
    pub fn intersection_terminals(&self) -> HybridL2Bitset { &self.path_intersection_terminals & &self.local_intersection_terminals }
}


// --- GSS Node & Core Implementation ---

/// A node in the Graph-Structured Stack (GSS).
#[derive(Debug, Clone)]
pub struct GSSNode {
    acc: Arc<Acc>,
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
    predecessors.keys().map(|(_, dest_key)| *dest_key).max().map_or(0, |max_pred_depth| max_pred_depth + 1)
}

fn compute_hash_key(predecessors: &NodeMap, acc: &Acc) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    acc.local_union_llm_tokens.hash(&mut hasher);
    acc.local_intersection_llm_tokens.hash(&mut hasher);
    acc.local_union_terminals.hash(&mut hasher);
    acc.local_intersection_terminals.hash(&mut hasher);
    for ((edge_val, dest_key), pred_arc) in predecessors {
        edge_val.hash(&mut hasher);
        dest_key.hash(&mut hasher);
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}

/// Processes a set of incoming predecessors, grouping them by depth and edge,
/// and merging nodes that share the same edge to create a canonical `NodeMap`.
// #[time_it]
fn process_predecessors(incoming: &NodeSet) -> NodeMap {
    let mut grouped: BTreeMap<(ParseStateEdgeContent, DestKey), Vec<Arc<GSSNode>>> = BTreeMap::new();

    for (pred_arc, edge_val) in incoming {
        grouped
            .entry((edge_val.clone(), pred_arc.dest_key()))
            .or_default()
            .push(pred_arc.clone());
    }

    let mut result: NodeMap = BTreeMap::new();
    for (key, pred_arcs) in grouped {
        if pred_arcs.is_empty() { continue; }

        let mut iter = pred_arcs.into_iter();
        let first = iter.next().unwrap();

        if iter.len() == 0 {
            result.insert(key, first);
        } else {
            let mut merged_node = (*first).clone();
            for other_arc in iter {
                merged_node.merge(&other_arc);
            }
            result.insert(key, Arc::new(merged_node));
        }
    }
    result
}

/// Merges the `source` NodeMap into the `target` NodeMap.
// #[time_it]
fn merge_node_maps(target: &mut NodeMap, source: NodeMap) {
    for (key, source_pred_arc) in source {
        match target.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(source_pred_arc);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                Arc::make_mut(entry.get_mut()).merge(&source_pred_arc);
            }
        }
    }
}

// Basic node creation and manipulation
impl GSSNode {
    /// Creates a new GSS root node with the given local constraints.
    pub fn new(local_acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let acc = Arc::new(local_acc);
        let hash_key_cache = compute_hash_key(&predecessors, &acc);
        Self { acc, predecessors, hash_key_cache, max_depth: 0 }
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    fn new_with_map(local_acc: Arc<Acc>, predecessors: NodeMap) -> Self {
        let pred_accs = predecessors.values().map(|p| &p.acc);
        let acc = Arc::new(Acc::from_preds(local_acc.as_ref(), pred_accs));
        let hash_key_cache = compute_hash_key(&predecessors, &acc);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, local_acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        predecessors_map.insert((edge_value, predecessor_arc.max_depth()), predecessor_arc.clone());
        Self::new_with_map(Arc::new(local_acc), predecessors_map)
    }

    pub fn new_fresh() -> Self {
        Self::new(Acc::new_fresh())
    }

    fn predecessors(&self) -> &NodeMap { &self.predecessors }

    pub fn num_predecessors(&self) -> usize { self.predecessors.len() }
    fn max_depth(&self) -> MaxDepth { self.max_depth }
    fn dest_key(&self) -> DestKey { self as *const GSSNode as usize }
    
    /// Returns the set of LLM tokens allowed by *any* path ending at this node.
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV { self.acc.union_llm_tokens() }
    
    /// Returns a map of disallowed terminals for each tokenizer state.
    /// A terminal is disallowed if it's disallowed on *every* path to this node.
    pub fn disallowed_terminals(&self) -> TerminalInfo {
        self.acc.intersection_terminals().complement()
    }

    pub fn is_empty(&self) -> bool { self.predecessors.is_empty() }
    
    /// A path is alive if it allows at least one LLM token.
    pub fn is_alive(&self) -> bool { !self.allowed_llm_tokens().is_empty() }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    pub fn push(&self, edge_value: ParseStateEdgeContent, local_acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, local_acc_for_new_node)
    }

    /// Pops the top state from the stack(s), returning a `GSSPop` structure.
    /// The constraints of this node are applied to its predecessors.
    pub fn pop(&self) -> GSSPop {
        let mut new_node_map = NodeMap::new();
        let parent_acc = &self.acc;

        for ((edge_val, dest_key), pred_arc) in &self.predecessors {
            let mut new_pred_node = (**pred_arc).clone();

            let new_local_union_llm = &pred_arc.acc.local_union_llm_tokens & &parent_acc.local_union_llm_tokens;
            let new_local_intersection_llm = &pred_arc.acc.local_intersection_llm_tokens & &parent_acc.local_intersection_llm_tokens;

            let new_local_union_terminals = &pred_arc.acc.local_union_terminals & &parent_acc.local_union_terminals;
            let new_local_intersection_terminals = &pred_arc.acc.local_intersection_terminals & &parent_acc.local_intersection_terminals;

            let new_acc = Arc::new(Acc {
                local_union_llm_tokens: new_local_union_llm,
                local_intersection_llm_tokens: new_local_intersection_llm,
                path_union_llm_tokens: pred_arc.acc.path_union_llm_tokens.clone(),
                path_intersection_llm_tokens: pred_arc.acc.path_intersection_llm_tokens.clone(),
                local_union_terminals: new_local_union_terminals,
                local_intersection_terminals: new_local_intersection_terminals,
                path_union_terminals: pred_arc.acc.path_union_terminals.clone(),
                path_intersection_terminals: pred_arc.acc.path_intersection_terminals.clone(),
            });

            if new_acc.union_llm_tokens().is_empty() {
                crate::debug!(6, "Dead path after pop");
                continue;
            }

            new_pred_node.acc = new_acc;
            new_pred_node.hash_key_cache = compute_hash_key(&new_pred_node.predecessors, &new_pred_node.acc);

            let new_pred_arc = Arc::new(new_pred_node);
            new_node_map.insert((edge_val.clone(), *dest_key), new_pred_arc);
        }

        GSSPop { parent_node: self, node_map: new_node_map }
    }

    /// Pops `n` levels from the GSS.
    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        self.pop().popn(n - 1).to_node()
    }

    /// Merges another `GSSNode` into this one. This is a union of possibilities.
    #[time_it]
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        if other.predecessors.is_empty() && other.acc.local_union_llm_tokens == HybridBitset::max_ones() { return; }
        if self.predecessors.is_empty() && self.acc.local_union_llm_tokens == HybridBitset::max_ones() {
            *self = other.clone();
            return;
        }

        let merged_local_union_llm = &self.acc.local_union_llm_tokens | &other.acc.local_union_llm_tokens;
        let merged_local_intersection_llm = &self.acc.local_intersection_llm_tokens & &other.acc.local_intersection_llm_tokens;
        let merged_local_union_terminals = &self.acc.local_union_terminals | &other.acc.local_union_terminals;
        let merged_local_intersection_terminals = &self.acc.local_intersection_terminals & &other.acc.local_intersection_terminals;
        let merged_local_acc = Arc::new(Acc {
            local_union_llm_tokens: merged_local_union_llm,
            local_intersection_llm_tokens: merged_local_intersection_llm,
            local_union_terminals: merged_local_union_terminals,
            local_intersection_terminals: merged_local_intersection_terminals,
            ..Acc::new_fresh()
        });
        
        let mut new_predecessors = self.predecessors.clone();
        merge_node_maps(&mut new_predecessors, other.predecessors.clone());
        
        *self = GSSNode::new_with_map(merged_local_acc, new_predecessors);
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let acc = (*self.acc).clone();
        self.push(edge_value, acc)
    }
    
    /// Returns an iterator over all direct predecessor paths (`GSSPeek`s).
    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPeek<'_>> {
        self.predecessors.iter().map(|((edge_val, _dest_key), pred_arc)| {
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
        for node_arc in node_map.values() {
            let popped = node_arc.pop();
            merge_node_maps(&mut combined_node_map, popped.node_map);
        }
        combined_node_map
    }

    pub fn pop(&self) -> GSSPop {
        let node_map = Self::_pop(&self.node_map);
        GSSPop { parent_node: self.parent_node, node_map }
    }

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
    pub fn to_node(&self) -> GSSNode {
        let local_acc = Arc::new(Acc::new_fresh());
        GSSNode::new_with_map(local_acc, self.node_map.clone())
    }
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }
    pub fn predecessor(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    pub fn to_node(&self) -> GSSNode {
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            (*self.parent_node.acc).clone(),
        )
    }

    pub fn to_arc_node(&self) -> Arc<GSSNode> {
        Arc::new(self.to_node())
    }

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
            self.acc == other.acc &&
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
            .then_with(|| self.acc.cmp(&other.acc))
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
            let new_node_predecessors_map = if continue_recursion {
                let new_predecessors_set = node_arc.predecessors.iter()
                    .filter_map(|((edge_val, _), pred_arc)| {
                        prune_and_transform_recursive(pred_arc, closure, memo)
                            .map(|new_pred_arc| (new_pred_arc, edge_val.clone()))
                    })
                    .collect::<NodeSet>();
                if new_predecessors_set.is_empty() && !node_arc.predecessors.is_empty() {
                    memo.insert(node_ptr, None);
                    return None;
                }
                process_predecessors(&new_predecessors_set)
            } else { // Don't recurse, keep existing predecessors.
                node_arc.predecessors.clone()
            };

            let transformed_node = GSSNode::new_with_map(Arc::new(new_local_acc), new_node_predecessors_map);

            let result_arc = Arc::new(transformed_node);
            memo.insert(node_ptr, Some(result_arc.clone()));
            Some(result_arc)
        }
    }
}

pub fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        new_acc.local_union_llm_tokens &= allowed_tokens;
        new_acc.local_intersection_llm_tokens &= allowed_tokens;

        if new_acc.union_llm_tokens().is_empty() {
            None
        } else {
            Some((new_acc, true))
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

pub fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let allowed_mask = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed_mask, memo);
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        let continue_recursion = new_acc.local_union_llm_tokens != HybridBitset::max_ones() || new_acc.local_intersection_llm_tokens != HybridBitset::max_ones();
        new_acc.local_union_llm_tokens = HybridBitset::max_ones();
        new_acc.local_intersection_llm_tokens = HybridBitset::max_ones();
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

pub fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        new_acc.local_union_terminals = &new_acc.local_union_terminals - disallowed_terminals;
        new_acc.local_intersection_terminals = &new_acc.local_intersection_terminals - disallowed_terminals;
        Some((new_acc, true))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let allowed_by_all_paths = node.acc.intersection_terminals();
        for (state_id, matched_bv) in matched_terminals {
            if let Some(allowed_l2) = allowed_by_all_paths.get_l2_bitset(state_id.0) {
                if !matched_bv.is_subset(allowed_l2) {
                    return None;
                }
            } else {
                if !matched_bv.is_empty() {
                    return None;
                }
            }
        }
        Some(((*node.acc).clone(), true))
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();

        let mut map_one = |terminals: &HybridL2Bitset| -> (HybridL2Bitset, bool) {
            let mut new_terminals = HybridL2Bitset::new();
            let mut changed = false;

            for (old_id_val, bv) in terminals.iter_l1_bitsets() {
                let old_id = TokenizerStateID(old_id_val);
                if let Some(&new_id) = map.get(&old_id) {
                    new_terminals.insert_l2_bitset(new_id.0, bv.clone());
                    if old_id != new_id {
                        changed = true;
                    }
                } else {
                    changed = true; // a mapping was removed
                }
            }
            (new_terminals, changed)
        };

        let (new_local_union, changed_union) = map_one(&new_acc.local_union_terminals);
        let (new_local_intersection, changed_intersection) = map_one(&new_acc.local_intersection_terminals);

        new_acc.local_union_terminals = new_local_union;
        new_acc.local_intersection_terminals = new_local_intersection;

        Some((new_acc, changed_union || changed_intersection))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

impl GSSNode {
    /// Fuses predecessor nodes that share the same edge value, even if they are at different depths.
    #[time_it]
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
    for ((edge_val, _), pred_arc) in &node_arc.predecessors {
        let fused_pred_arc = fuse_predecessors_recursive(pred_arc, levels - 1, memo);
        recursively_fused_predecessors.push((edge_val.clone(), fused_pred_arc));
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
    let fused_node = GSSNode::new_with_map(node_arc.acc.clone(), new_predecessors_map);

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

        let unique_pred_arcs: HashSet<_> = node.predecessors.values()
            .map(Arc::as_ptr)
            .collect();
        if unique_pred_arcs.len() > 1 {
            stats.merge_points += 1;
        }

        for pred_arc in node.predecessors.values() {
            queue.push_back((pred_arc.as_ref(), depth + 1));
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }

    // Calculate structural uniqueness
    let mut structural_memo = HashMap::new();
    let mut structural_cache: BTreeMap<BTreeMap<(ParseStateEdgeContent, DestKey), usize>, usize> = BTreeMap::new();
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
    structural_cache: &mut BTreeMap<BTreeMap<(ParseStateEdgeContent, DestKey), usize>, usize>,
) -> usize {
    let node_ptr = node as *const GSSNode;
    if let Some(id) = memo.get(&node_ptr) {
        return *id;
    }

    let mut pred_structural_ids = BTreeMap::new();
    for ((edge_val, dest_key), pred_arc) in &node.predecessors {
        let pred_id = get_structural_id(pred_arc.as_ref(), memo, structural_cache);
        pred_structural_ids.insert((edge_val.clone(), *dest_key), pred_id);
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
        for ((edge_val, _), pred_arc) in node_arc.predecessors.iter() {
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
            .iter()
            .map(|((edge_val, _), pred_arc)| (edge_val, pred_arc))
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
    let format_allowed_llm = |bv: &HybridBitset, label: &str| -> String {
        if let (Some(bimap), Some(token_map)) = (original_internal_bimap, llm_token_map) {
            const MAX_TO_SHOW: usize = 5;
            let total_tokens = bv.len();
            let token_samples: Vec<_> = bv.iter().take(MAX_TO_SHOW)
                .filter_map(|internal_id| bimap.get_by_right(&internal_id))
                .filter_map(|original_id| token_map.get_by_right(&LLMTokenID(*original_id)))
                .map(|token_bytes| format!("{:?}", String::from_utf8_lossy(token_bytes)))
                .collect();

            let samples_str = token_samples.join(", ");
            if total_tokens > MAX_TO_SHOW {
                format!("{}({} tokens: [{}, ...])", label, total_tokens, samples_str)
            } else if total_tokens > 0 {
                format!("{}([{}])", label, samples_str)
            } else {
                format!("{}(0 tokens)", label)
            }
        } else {
            format!("{}({} tokens)", label, bv.len())
        }
    };

    let format_disallowed_terminals = |allowed_terminals: &HybridL2Bitset| -> String {
        if allowed_terminals.is_empty() {
            return "Terminals(All Disallowed)".to_string();
        }
        let mut parts = Vec::new();
        const MAX_RANGES_TO_SHOW: usize = 5;
        for (range, allowed_bv) in allowed_terminals.range_values() {
            if parts.len() >= MAX_RANGES_TO_SHOW {
                parts.push("...".to_string());
                break;
            }
            let disallowed_bv = HybridBitset::max_ones() - allowed_bv;
            if !disallowed_bv.is_empty() {
                const MAX_NAMES_TO_SHOW: usize = 5;
                let num_disallowed = disallowed_bv.len();
                let names: Vec<_> = disallowed_bv.iter().take(MAX_NAMES_TO_SHOW)
                    .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                        .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.to_string()))
                    .collect();
                let names_str = names.join(", ");

                let range_str = if range.start() == range.end() {
                    format!("{}", range.start())
                } else {
                    format!("{}..={}", range.start(), range.end())
                };
                if num_disallowed > MAX_NAMES_TO_SHOW {
                    parts.push(format!("State(s) {} ({} disallowed): [{}, ...]", range_str, num_disallowed, names_str));
                } else {
                    parts.push(format!("State(s) {}: [{}]", range_str, names_str));
                }
            }
        }
        if parts.is_empty() {
            "Terminals(None Disallowed)".to_string()
        } else {
            format!("Disallowed Terminals({})", parts.join("; "))
        }
    };

    let local_union_llm_str = format_allowed_llm(&node.acc.local_union_llm_tokens, "LocalUnionLLM");
    let local_intersection_llm_str = format_allowed_llm(&node.acc.local_intersection_llm_tokens, "LocalIntersectionLLM");
    let union_llm_str = format_allowed_llm(&node.acc.union_llm_tokens(), "UnionLLM");
    let disallowed_terminals_str = format_disallowed_terminals(&node.acc.intersection_terminals());

    format!("[{}, {}, {}, {}]", local_union_llm_str, local_intersection_llm_str, union_llm_str, disallowed_terminals_str)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::table::StateID;

    // Helper to create a local Acc that disallows a single token.
    fn mock_acc(val: usize) -> Acc {
        let mut disallowed_bv = LLMTokenBV::zeros();
        disallowed_bv.insert(val);
        let allowed_bv = HybridBitset::max_ones() - disallowed_bv;
        Acc::new_with_local_constraints(allowed_bv, HybridL2Bitset::all())
    }

    fn empty_acc() -> Acc {
        Acc::new_fresh()
    }

    fn mock_edge(id: usize) -> ParseStateEdgeContent {
        ParseStateEdgeContent { state_id: StateID(id) }
    }

    #[test]
    fn test_gss_new_node() {
        let acc = mock_acc(1);
        let node = GSSNode::new(acc.clone());
        assert_eq!(node.acc.local_union_llm_tokens, acc.local_union_llm_tokens);
        assert_eq!(node.acc.local_intersection_llm_tokens, acc.local_intersection_llm_tokens);
        assert!(node.acc.path_union_llm_tokens.is_empty());
        assert_eq!(node.acc.path_intersection_llm_tokens, HybridBitset::max_ones());
        assert!(node.predecessors.is_empty());
        assert_eq!(node.max_depth, 0);
    }

    #[test]
    fn test_gss_push() {
        let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
        let pushed = root.push(mock_edge(10), mock_acc(2)); // Allows all but 2

        assert_eq!(pushed.max_depth, 1);
        assert_eq!(pushed.acc.local_union_llm_tokens, mock_acc(2).local_union_llm_tokens);
        assert_eq!(pushed.acc.local_intersection_llm_tokens, mock_acc(2).local_intersection_llm_tokens);
        
        let expected_path_union = root.acc.union_llm_tokens();
        assert_eq!(pushed.acc.path_union_llm_tokens, expected_path_union);

        let full_union = pushed.acc.union_llm_tokens();
        assert_eq!(full_union, HybridBitset::max_ones());

        let full_intersection = pushed.acc.intersection_llm_tokens();
        let mut expected_disallowed = HybridBitset::zeros();
        expected_disallowed.insert(1);
        expected_disallowed.insert(2);
        assert_eq!(full_intersection, HybridBitset::max_ones() - expected_disallowed);
    }

    #[test]
    fn test_gss_pop() {
        let root = Arc::new(GSSNode::new(mock_acc(1)));
        let pushed = Arc::new(root.push(mock_edge(10), mock_acc(2)));
        
        let pop_result = pushed.pop();
        assert_eq!(pop_result.node_map.len(), 1);
        
        let popped_node_arc = pop_result.node_map.values().next().unwrap();
        
        let expected_local_union = &root.acc.local_union_llm_tokens & &pushed.acc.local_union_llm_tokens;
        assert_eq!(popped_node_arc.acc.local_union_llm_tokens, expected_local_union);
        let expected_local_intersection = &root.acc.local_intersection_llm_tokens & &pushed.acc.local_intersection_llm_tokens;
        assert_eq!(popped_node_arc.acc.local_intersection_llm_tokens, expected_local_intersection);
        
        let mut disallowed = HybridBitset::zeros();
        disallowed.insert(1);
        disallowed.insert(2);
        let expected_allowed = HybridBitset::max_ones() - disallowed;
        assert_eq!(popped_node_arc.acc.local_union_llm_tokens, expected_allowed);
    }

    #[test]
    fn test_gss_merge() {
        let n0 = Arc::new(GSSNode::new(empty_acc()));
        let n1 = Arc::new(n0.push(mock_edge(0), mock_acc(1)));
        let n2 = Arc::new(n0.push(mock_edge(0), mock_acc(2)));

        let mut merged = (*n1).clone();
        merged.merge(&n2);

        assert_eq!(merged.acc.local_union_llm_tokens, HybridBitset::max_ones());
        assert_ne!(merged.acc.local_intersection_llm_tokens, HybridBitset::max_ones());

        assert_eq!(merged.num_predecessors(), 1);

        let pred_full_union = n0.acc.union_llm_tokens();
        let expected_path_union = &pred_full_union | &pred_full_union;
        assert_eq!(merged.acc.path_union_llm_tokens, expected_path_union);
        assert_eq!(merged.acc.union_llm_tokens(), HybridBitset::max_ones());
    }

    #[test]
    fn test_gss_fuse_predecessors() {
        let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
        let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));
        let b = Arc::new(leaf1.push(mock_edge(1), empty_acc()));
        let c_tmp = Arc::new(leaf2.push(mock_edge(2), empty_acc()));
        let c_tmp2 = Arc::new(c_tmp.push(mock_edge(3), empty_acc()));
        let c = Arc::new(c_tmp2.push(mock_edge(4), empty_acc()));

        assert_eq!(b.max_depth, 1);
        assert_eq!(c.max_depth, 3);

        let mut preds_map = NodeMap::new();
        preds_map.insert((mock_edge(100), b.max_depth()), b.clone());
        preds_map.insert((mock_edge(100), c.max_depth()), c.clone());

        let mut root = GSSNode::new_with_map(Arc::new(empty_acc()), preds_map);
        assert_eq!(root.num_predecessors(), 2);

        root.fuse_predecessors(1);

        assert_eq!(root.num_predecessors(), 1);
        let fused_pred_arc = root.predecessors().values().next().unwrap();

        assert_eq!(fused_pred_arc.acc.local_union_llm_tokens, HybridBitset::max_ones());
        assert_ne!(fused_pred_arc.acc.local_intersection_llm_tokens, HybridBitset::max_ones());
        assert_eq!(fused_pred_arc.num_predecessors(), 2);
    }

    #[test]
    fn test_sample_path() {
        let d = Arc::new(GSSNode::new(empty_acc()));
        let e = Arc::new(GSSNode::new(empty_acc()));

        let mut c_preds = NodeSet::new();
        c_preds.insert((d, mock_edge(30)));
        c_preds.insert((e, mock_edge(40)));
        let c_preds_map = process_predecessors(&c_preds);
        let c = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), c_preds_map));

        let b = Arc::new(c.push(mock_edge(20), empty_acc()));
        let root = b.push(mock_edge(10), empty_acc());

        let path1 = sample_path(&[&root], 0).unwrap();
        let path2 = sample_path(&[&root], 1).unwrap();

        assert_eq!(path1.len(), 3);
        assert_eq!(path1[0], mock_edge(10));
        assert_eq!(path1[1], mock_edge(20));
        assert!(path1[2] == mock_edge(30) || path1[2] == mock_edge(40));

        let path1_again = sample_path(&[&root], 0).unwrap();
        assert_eq!(path1, path1_again);
    }
}
