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
    pub llm_tokens_union: HybridBitset,
    pub llm_tokens_intersection: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub terminals_intersection: HybridL2Bitset,
}

impl Acc {
    /// Creates a fresh, unconstrained accumulator (all tokens/terminals allowed).
    pub fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            llm_tokens_intersection: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            terminals_intersection: HybridL2Bitset::all(),
        }
    }

    /// Creates an accumulator with specific local constraints for a root node.
    pub fn new_with_local_constraints(llm_tokens: HybridBitset, terminals: HybridL2Bitset) -> Self {
        Self {
            llm_tokens_union: llm_tokens.clone(),
            llm_tokens_intersection: llm_tokens,
            terminals_union: terminals.clone(),
            terminals_intersection: terminals,
        }
    }

    pub fn narrow(from: &Self, to: &Self) -> Self {
        Acc {
            llm_tokens_union: &from.llm_tokens_union & &to.llm_tokens_union,
            llm_tokens_intersection: &from.llm_tokens_union & &to.llm_tokens_intersection,
            terminals_union: &from.terminals_union & &to.terminals_union,
            terminals_intersection: &from.terminals_union & &to.terminals_intersection,
        }
    }

    pub fn merge(lhs: &Self, rhs: &Self) -> Self {
        Acc {
            llm_tokens_union: &lhs.llm_tokens_union | &rhs.llm_tokens_union,
            llm_tokens_intersection: &lhs.llm_tokens_intersection & &rhs.llm_tokens_intersection,
            terminals_union: &lhs.terminals_union | &rhs.terminals_union,
            terminals_intersection: &lhs.terminals_intersection & &rhs.terminals_intersection,
        }
    }

    // --- Accessors for final computed sets ---
    pub fn union_llm_tokens(&self) -> HybridBitset { self.llm_tokens_union.clone() }
    pub fn intersection_terminals(&self) -> HybridL2Bitset { self.terminals_intersection.clone() }
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

/// A read-only view into a single path segment of the GSS, from a parent to a predecessor.
#[derive(Clone, Copy)]
pub struct GSSPeek<'a> {
    parent_acc: &'a Arc<Acc>,
    edge_value: &'a ParseStateEdgeContent,
    predecessor_node: &'a Arc<GSSNode>,
}

/// Represents the result of a `pop` operation, containing a map of resulting nodes
/// and the accumulated constraints (`Acc`) for each path leading to them.
#[derive(Debug, Clone, Default)]
pub struct GSSPopper {
    /// A map where the key is a node, and the value is the accumulated `Acc` for all paths leading to it.
    /// and the value is the accumulated `Acc` for that path.
    paths: BTreeMap<Arc<GSSNode>, Arc<Acc>>,
}

/// An item yielded by iterating over a `GSSPopper`, representing a single resulting path.
#[derive(Clone, Copy)]
pub struct GSSPopperItem<'a> {
    node: &'a Arc<GSSNode>,
    path_acc: &'a Arc<Acc>,
}

#[derive(Clone, Copy)]
pub struct GSSPopperItemPeek<'a> {
    path_acc: &'a Arc<Acc>,
    parent_acc: &'a Arc<Acc>,
    edge_value: &'a ParseStateEdgeContent,
    predecessor_node: &'a Arc<GSSNode>,
}

impl GSSPopper {
    pub fn new_from_node(node: Arc<GSSNode>, acc: Arc<Acc>) -> Self {
        let mut popper = Self::default();
        popper.paths.insert(node, acc);
        popper
    }

    /// Returns an iterator over the items in the popper.
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
            let mut new_paths: BTreeMap<Arc<GSSNode>, Arc<Acc>> = BTreeMap::new();
            for (parent, path_acc) in std::mem::take(&mut self.paths) {
                let new_path_acc = Arc::new(Acc::narrow(&path_acc, &parent.acc));
                for child in parent.predecessors.values() {
                    if let Some(existing_acc) = new_paths.get_mut(child) {
                        *existing_acc = Arc::new(Acc::merge(existing_acc, &new_path_acc));
                    } else {
                        new_paths.insert(child.clone(), new_path_acc.clone());
                    }
                }
            }
            self.paths = new_paths;
        }
    }
}

impl<'a> GSSPopperItem<'a> {
    /// Returns the combined `Acc` of the path and the destination node.
    pub fn resolved_acc(&self) -> Acc {
        Acc::narrow(&self.path_acc, &self.node.acc)
    }

    /// Returns a new `GSSNode` representing the destination node, but with its `Acc`
    /// resolved against the path's `Acc`.
    pub fn resolved_node(&self) -> GSSNode {
        GSSNode::new_with_map(Arc::new(self.resolved_acc()), self.node.predecessors.clone())
    }

    /// Pushes a new state onto the resolved node from this popper item.
    pub fn push(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.resolved_node().push(edge_value)
    }

    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPopperItemPeek<'_>> {
        self.node.predecessors.iter().map(|((edge_val, _dest_key), pred_arc)| {
            GSSPopperItemPeek {
                path_acc: &self.path_acc,
                parent_acc: &self.node.acc,
                edge_value: edge_val,
                predecessor_node: pred_arc,
            }
        })
    }
}

impl<'a> GSSPopperItemPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    /// Returns the combined `Acc` of the path and the predecessor node.
    pub fn resolved_acc(&self) -> Acc {
        Acc::narrow(&Acc::narrow(self.path_acc, self.parent_acc), &self.predecessor_node.acc)
    }

    /// Returns a new `GSSNode` representing the predecessor, but with its `Acc`
    /// resolved against the path's `Acc`.
    pub fn resolved_predecessor_node(&self) -> GSSNode {
        GSSNode::new_with_map(Arc::new(self.resolved_acc()), self.predecessor_node.predecessors.clone())
    }

    /// Pushes a new state onto the resolved predecessor.
    pub fn push_on_predecessor(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let resolved_acc = self.resolved_acc();
        GSSNode::new_with_single_predecessor(self.predecessor_node.clone(), edge_value, resolved_acc)
    }

    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.isolated_parent().push(edge_value)
    }
    pub fn popn(&self, len: usize) -> GSSPopper {
        let isolated_parent = Arc::new(self.isolated_parent());
        let mut popper = GSSPopper::new_from_node(isolated_parent, Arc::new(Acc::new_fresh()));
        popper.popn(len);
        popper
    }

    pub fn isolated_parent(&self) -> GSSNode {
        let acc = Acc::narrow(&Acc::narrow(self.path_acc, self.parent_acc), &self.predecessor_node.acc);
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            acc.clone()
        )
    }
}

// Helper functions for GSSNode construction
fn compute_max_depth(predecessors: &NodeMap) -> MaxDepth {
    predecessors.keys().map(|(_, dest_key)| *dest_key).max().map_or(0, |max_pred_depth| max_pred_depth + 1)
}

fn compute_hash_key(predecessors: &NodeMap, acc: &Acc) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    acc.llm_tokens_union.hash(&mut hasher);
    acc.llm_tokens_intersection.hash(&mut hasher);
    acc.terminals_union.hash(&mut hasher);
    acc.terminals_intersection.hash(&mut hasher);
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
            .entry((edge_val.clone(), pred_arc.max_depth()))
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
    pub fn new(acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let arc_acc = Arc::new(acc);
        let hash_key_cache = compute_hash_key(&predecessors, &arc_acc);
        Self { acc: arc_acc, predecessors, hash_key_cache, max_depth: 0 }
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    fn new_with_map(acc: Arc<Acc>, predecessors: NodeMap) -> Self {
        let pred_accs = predecessors.values().map(|p| &p.acc);
        let hash_key_cache = compute_hash_key(&predecessors, &acc);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        predecessors_map.insert((edge_value, predecessor_arc.max_depth()), predecessor_arc.clone());
        Self::new_with_map(Arc::new(acc), predecessors_map)
    }

    pub fn new_fresh() -> Self {
        Self::new(Acc::new_fresh())
    }

    pub fn predecessors(&self) -> &NodeMap { &self.predecessors }

    pub fn num_predecessors(&self) -> usize { self.predecessors.len() }
    fn max_depth(&self) -> MaxDepth { self.max_depth }
    // fn dest_key(&self) -> DestKey { self as *const GSSNode as usize }
    fn dest_key(&self) -> DestKey { self.max_depth() }

    /// Returns the set of LLM tokens allowed by *any* path ending at this node.
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV { self.acc.llm_tokens_union.clone() }
    
    /// Returns a map of disallowed terminals for each tokenizer state.
    /// A terminal is disallowed if it's disallowed on *every* path to this node.
    pub fn disallowed_terminals(&self) -> TerminalInfo {
        self.acc.terminals_union.complement()
    }

    pub fn is_empty(&self) -> bool { self.predecessors.is_empty() }
    
    /// A path is alive if it allows at least one LLM token.
    pub fn is_alive(&self) -> bool { !self.allowed_llm_tokens().is_empty() }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    pub fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        let acc = (*self.acc).clone();
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, acc)
    }

    /// Performs a multi-level pop operation on this node.
    pub fn popn(&self, n: usize) -> GSSPopper {
        let mut popper = GSSPopper::new_from_node(Arc::new(self.clone()), Arc::new(Acc::new_fresh()));
        popper.popn(n);
        popper
    }

    /// Merges another `GSSNode` into this one. This is a union of possibilities.
    #[time_it]
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        if other.predecessors.is_empty() && other.acc.llm_tokens_union == HybridBitset::max_ones() { return; }
        if self.predecessors.is_empty() && self.acc.llm_tokens_union == HybridBitset::max_ones() {
            *self = other.clone();
            return;
        }

        let merged_acc = Arc::new(Acc::merge(&self.acc, &other.acc));
        
        let mut new_predecessors = self.predecessors.clone();
        merge_node_maps(&mut new_predecessors, other.predecessors.clone());
        
        *self = GSSNode::new_with_map(merged_acc, new_predecessors);
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let acc = (*self.acc).clone();
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, acc)
    }
    
    /// Returns an iterator over all direct predecessor paths (`GSSPeek`s).
    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPeek<'_>> {
        self.predecessors.iter().map(|((edge_val, _dest_key), pred_arc)| {
            GSSPeek {
                parent_acc: &self.acc,
                edge_value: edge_val,
                predecessor_node: pred_arc,
            }
        })
    }
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    /// Returns the combined `Acc` of the parent and the predecessor.
    pub fn resolved_acc(&self) -> Acc {
        Acc::narrow(self.parent_acc, &self.predecessor_node.acc)
    }

    /// Returns a new `GSSNode` representing the predecessor, but with its `Acc`
    /// resolved against the parent's `Acc`.
    pub fn resolved_predecessor_node(&self) -> GSSNode {
        let resolved_acc = self.resolved_acc();
        GSSNode::new_with_map(Arc::new(resolved_acc), self.predecessor_node.predecessors.clone())
    }

    /// Pushes a new state onto the resolved predecessor.
    pub fn push_on_predecessor(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let resolved_acc = self.resolved_acc();
        GSSNode::new_with_single_predecessor(self.predecessor_node.clone(), edge_value, resolved_acc)
    }

    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.isolated_parent().push(edge_value)
    }
    pub fn popn(&self, len: usize) -> GSSPopper {
        let isolated_parent = Arc::new(self.isolated_parent());
        let mut popper = GSSPopper::new_from_node(isolated_parent, Arc::new(Acc::new_fresh()));
        popper.popn(len);
        popper
    }

    /// Creates a new `GSSNode` that represents only the path segment of this peek.
    /// The new node has the parent's `Acc` and a single predecessor (the one from this peek).
    pub fn isolated_parent(&self) -> GSSNode {
        let acc = Acc::narrow(self.parent_acc, &self.predecessor_node.acc);
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            acc.clone()
        )
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
        new_acc.llm_tokens_union &= allowed_tokens;
        new_acc.llm_tokens_intersection &= allowed_tokens;

        // Prune if the union of possibilities is empty.
        if new_acc.llm_tokens_union.is_empty() {
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
        let continue_recursion = new_acc.llm_tokens_intersection != HybridBitset::max_ones();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        new_acc.llm_tokens_intersection = HybridBitset::max_ones();
        Some((new_acc, true))
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
        new_acc.terminals_union -= disallowed_terminals;
        new_acc.terminals_intersection -= disallowed_terminals;
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
        // If any of the matched terminals is disallowed, that's a problem.
        // If any of the matched terminals is missing from the union of allowed terminals, then this entire node is pruned.
        // (In other words, since the union is the union of all sub-nodes, if it's missing from the union then it's missing from all the sub-nodes.)
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = node.acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                // If the matched terminal is not a subset of the allowed terminals, we prune this node.
                return None;
            }
        }
        // If any of the matched terminals is missing from the intersection of allowed terminals, we continue recursion,
        // because this means it's missing from one of the subnode's union of allowed terminals.
        // (In other words, since the intersection is the intersection of all sub-nodes, if it's missing from the intersection then it's missing from at least one sub-node.)
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_intersection = node.acc.terminals_intersection.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_intersection) {
                // If the matched terminal is not a subset of the allowed terminals, we continue recursion.
                return Some(((*node.acc).clone(), true));
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
    println!("Mapping allowed terminals to tokenizer states with map: {:?}", map);
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();

        let map_one = |terminals: &HybridL2Bitset| -> (HybridL2Bitset, bool) {
            let mut new_terminals = HybridL2Bitset::all();

            for (old_state_id, new_state_id) in map {
                let bv_source = terminals.get_l2_bitset(old_state_id.0).unwrap();
                let bv_dst_existing = new_terminals.get_l2_bitset(new_state_id.0).unwrap();
                let bv_dst_combined = bv_source | bv_dst_existing;
                new_terminals.insert_l2_bitset(new_state_id.0, bv_dst_combined);
            }
            let changed = new_terminals != *terminals;
            (new_terminals, changed)
        };

        let (new_terminals_union, changed_union) = map_one(&new_acc.terminals_union);
        let (new_terminals_intersection, changed_intersection) = map_one(&new_acc.terminals_intersection);

        new_acc.terminals_union = new_terminals_union;
        new_acc.terminals_intersection = new_terminals_intersection;

        println!("Accs before: {:?}", node.acc);
        println!("Accs after: {:?}", new_acc);

        Some((new_acc, true))
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

        temp_arc_storage = chosen_peek.predecessor_node.clone();
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
    // return "".to_string();
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
        if *bv == HybridBitset::max_ones() {
            return format!("{}(All)", label);
        }
        if let (Some(bimap), Some(token_map)) = (original_internal_bimap, llm_token_map) {
            const MAX_TO_SHOW: usize = 5;
            let total_tokens = bv.len();
            let token_samples: Vec<_> = bv.iter().take(MAX_TO_SHOW)
                .map(|internal_id| {
                    bimap.get_by_right(&internal_id)
                        .and_then(|original_id| token_map.get_by_right(&LLMTokenID(*original_id)))
                        .map(|token_bytes| format!("{:?}", String::from_utf8_lossy(token_bytes)))
                        .unwrap_or_else(|| format!("<internal_id:{}>", internal_id))
                })
                .collect();

            let samples_str = token_samples.join(", ");
            if total_tokens > MAX_TO_SHOW {
                format!("{}({} tokens: [{}, ...])", label, total_tokens, samples_str)
            } else {
                format!("{}([{}])", label, samples_str)
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
                let range_str = if range.start() == range.end() {
                    format!("{}", range.start())
                } else {
                    format!("{}..={}", range.start(), range.end())
                };

                if disallowed_bv == HybridBitset::max_ones() {
                    parts.push(format!("State(s) {}: All disallowed", range_str));
                    continue;
                }

                const MAX_NAMES_TO_SHOW: usize = 5;
                let num_disallowed = disallowed_bv.len();
                let names: Vec<_> = disallowed_bv.iter().take(MAX_NAMES_TO_SHOW)
                    .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                        .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.to_string()))
                    .collect();
                let names_str = names.join(", ");

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

    let union_llm_str = format_allowed_llm(&node.acc.llm_tokens_union, "LLM");
    let disallowed_terminals_str = format_disallowed_terminals(&node.acc.terminals_intersection);

    format!("[{}, {}]", union_llm_str, disallowed_terminals_str)
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
        assert_eq!(node.acc.llm_tokens_union, acc.llm_tokens_union);
        assert!(node.predecessors.is_empty());
        assert_eq!(node.max_depth, 0);
    }

    #[test]
    fn test_gss_push() {
        let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
        let pushed = root.push(mock_edge(10));

        assert_eq!(pushed.max_depth, 1);

        // The new logic for `push` is to inherit the predecessor's acc, as the local acc is fresh.
        assert_eq!(*pushed.acc, *root.acc);
    }

    #[test]
    fn test_gss_pop() {
        let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
        let pushed = Arc::new(root.push(mock_edge(10))); // Now inherits root's acc.

        // Pop 1 level from `pushed`. The initial_acc is "fresh" (all allowed), so it doesn't constrain the path.
        let pop_result = pushed.popn(1);
        assert_eq!(pop_result.paths.len(), 1);

        // The result of the pop is one path, ending at `root`.
        let (popped_node_arc, path_acc) = pop_result.paths.iter().next().unwrap();

        // The node we landed on is the original root.
        assert!(Arc::ptr_eq(popped_node_arc, &root));
        
        // The `path_acc` is the `acc` from the node we popped from (`pushed`).
        assert_eq!(*path_acc, pushed.acc);

        // The `resolved_acc` for this path is the intersection of the path's constraints
        // and the destination node's own constraints.
        let popper_item = pop_result.iter().next().unwrap();
        let resolved_acc = popper_item.resolved_acc();

        // `pushed.acc` (same as `root.acc`) allows all but 1.
        // The intersection should allow all but 1.
        let mut disallowed = HybridBitset::zeros();
        disallowed.insert(1);
        let expected_allowed = HybridBitset::max_ones() - disallowed;
        assert_eq!(resolved_acc.llm_tokens_union, expected_allowed);
    }

    #[test]
    fn test_gss_merge() {
        let n0 = Arc::new(GSSNode::new(empty_acc()));
        let n1 = Arc::new(n0.push(mock_edge(0)));
        let n2 = Arc::new(n0.push(mock_edge(0)));

        let mut merged = (*n1).clone();
        merged.merge(&n2);

        assert_eq!(merged.acc.llm_tokens_union, HybridBitset::max_ones());

        assert_eq!(merged.num_predecessors(), 1);
    }

    #[test]
    fn test_gss_fuse_predecessors() {
        let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
        let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));
        let b = Arc::new(leaf1.push(mock_edge(1)));
        let c_tmp = Arc::new(leaf2.push(mock_edge(2)));
        let c_tmp2 = Arc::new(c_tmp.push(mock_edge(3)));
        let c = Arc::new(c_tmp2.push(mock_edge(4)));

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

        assert_eq!(fused_pred_arc.acc.llm_tokens_union, HybridBitset::max_ones());
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

        let b = Arc::new(c.push(mock_edge(20)));
        let root = b.push(mock_edge(10));

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
