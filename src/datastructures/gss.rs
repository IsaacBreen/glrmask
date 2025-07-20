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
/// For a given tokenizer state, holds the bitvector of disallowed terminals.
pub type TerminalInfo = BTreeMap<TokenizerStateID, TerminalBV>;


// --- Constraint Set Abstraction ---

/// A trait for set-like objects used in constraint tracking.
trait BitSetLike:
    for<'a> BitAnd<&'a Self, Output = Self> +
    for<'a> BitOr<&'a Self, Output = Self> +
    Clone + Debug + PartialEq + Eq + PartialOrd + Ord + Hash + Send + Sync
{
    /// A set containing all possible elements.
    fn all() -> Self;
    /// A set containing no elements.
    fn empty() -> Self;
    /// Checks if the set is empty.
    fn is_empty(&self) -> bool;
}

impl BitSetLike for HybridBitset {
    fn all() -> Self { HybridBitset::max_ones() }
    fn empty() -> Self { HybridBitset::zeros() }
    fn is_empty(&self) -> bool { self.is_empty() }
}

impl BitSetLike for HybridL2Bitset {
    fn all() -> Self { HybridL2Bitset::all() }
    fn empty() -> Self { HybridL2Bitset::new() }
    fn is_empty(&self) -> bool { self.is_empty() }
}

/// Tracks the allowed set of items (e.g., LLM tokens or terminals) for a GSS node.
/// It stores the local constraints and aggregates path constraints.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ConstraintSet<T: BitSetLike> {
    /// The set of items allowed by the local constraint at this node/edge.
    local: T,
    /// The union of allowed items from all paths leading to this node (pre-local).
    path_union: T,
    /// The intersection of allowed items over all paths leading to this node (pre-local).
    path_intersection: T,
}

impl<T: BitSetLike> ConstraintSet<T> {
    /// Creates a constraint set for a root node.
    fn new_root(local: T) -> Self {
        Self {
            local,
            path_union: T::empty(),
            path_intersection: T::all(),
        }
    }

    /// Creates a constraint set for a new node based on its predecessors.
    fn from_preds<'a>(local: T, preds: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut path_union = T::empty();
        let mut path_intersection = T::all();
        let mut has_preds = false;

        for p in preds {
            has_preds = true;
            path_union = &path_union | &p.union();
            path_intersection = &path_intersection & &p.intersection();
        }

        if !has_preds {
            return Self::new_root(local);
        }

        Self { local, path_union, path_intersection }
    }

    /// Returns the union of allowed items for any path ending at this node.
    /// This is the set of items allowed by *any* path, intersected with local constraints.
    fn union(&self) -> T { &self.path_union | &self.local }

    /// Returns the intersection of allowed items over all paths ending at this node.
    /// This is the set of items allowed by *all* paths, intersected with local constraints.
    fn intersection(&self) -> T { &self.path_intersection & &self.local }
}


// --- Accumulator (Acc) ---

/// Represents the full set of allowed tokens and terminals for a GSS node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    llm_tokens: ConstraintSet<HybridBitset>,
    terminals: ConstraintSet<HybridL2Bitset>,
}

impl Acc {
    /// Creates a fresh, unconstrained accumulator (all tokens/terminals allowed).
    pub fn new_fresh() -> Self {
        Self {
            llm_tokens: ConstraintSet::new_root(HybridBitset::all()),
            terminals: ConstraintSet::new_root(HybridL2Bitset::all()),
        }
    }

    /// Creates an accumulator with specific local constraints for a root node.
    pub fn new_with_local_constraints(local_llm: HybridBitset, local_terminals: HybridL2Bitset) -> Self {
        Self {
            llm_tokens: ConstraintSet::new_root(local_llm),
            terminals: ConstraintSet::new_root(local_terminals),
        }
    }

    /// Creates an accumulator for a new node from its local constraints and predecessors.
    fn from_preds<'a>(local: &Acc, pred_accs: impl IntoIterator<Item = &'a Arc<Acc>> + Clone) -> Self {
        Self {
            llm_tokens: ConstraintSet::from_preds(
                local.llm_tokens.local.clone(),
                pred_accs.clone().into_iter().map(|a| &a.llm_tokens)
            ),
            terminals: ConstraintSet::from_preds(
                local.terminals.local.clone(),
                pred_accs.into_iter().map(|a| &a.terminals)
            ),
        }
    }

    // --- Compatibility Wrappers ---
    pub fn new_fresh_without_vocab() -> Self { Self::new_fresh() }
    pub fn new_fresh_from_existing(_acc: &Acc) -> Self { Self::new_fresh() }
    pub fn new_fresh_from_existing_stack(_stack: &GSSNode) -> Self { Self::new_fresh() }
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
    acc.llm_tokens.local.hash(&mut hasher);
    // Hashing L2 bitset can be slow, consider if it's necessary for structural identity.
    // For now, we keep it to ensure correctness.
    acc.terminals.local.hash(&mut hasher);
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
        let acc = Arc::new(Acc {
            llm_tokens: ConstraintSet::new_root(local_acc.llm_tokens.local),
            terminals: ConstraintSet::new_root(local_acc.terminals.local),
        });
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

    pub fn fresh_from_existing(_node: &GSSNode) -> Self {
        Self::new(Acc::new_fresh())
    }

    fn predecessors(&self) -> &NodeMap { &self.predecessors }

    pub fn num_predecessors(&self) -> usize { self.predecessors.len() }
    pub fn max_depth(&self) -> MaxDepth { self.max_depth }
    pub fn dest_key(&self) -> DestKey { self as *const GSSNode as usize }
    
    /// Returns the set of LLM tokens allowed by *any* path ending at this node.
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV { self.acc.llm_tokens.union() }
    
    /// Returns a map of disallowed terminals for each tokenizer state.
    /// A terminal is disallowed if it's disallowed on *every* path to this node.
    pub fn disallowed_terminals(&self) -> TerminalInfo {
        let allowed_terminals = self.acc.terminals.intersection();
        if allowed_terminals.is_empty() {
            // If nothing is allowed, everything is disallowed.
            // This is a simplification; we'd need a universe of tokenizer states.
            // Returning an empty map is safer and often correct in context.
            return BTreeMap::new();
        }
        // We can only report on tokenizer states present in the allowed set.
        let mut disallowed = BTreeMap::new();
        for (l1_index, allowed_l2) in allowed_terminals.iter() {
            let state_id = TokenizerStateID(l1_index);
            let disallowed_l2 = HybridBitset::max_ones() - allowed_l2;
            if !disallowed_l2.is_empty() {
                disallowed.insert(state_id, disallowed_l2);
            }
        }
        disallowed
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

            // The popped node's new local constraints are the intersection of its
            // original local constraints and the parent's local constraints.
            let new_local_llm = &pred_arc.acc.llm_tokens.local & &parent_acc.llm_tokens.local;
            let new_local_terminals = &pred_arc.acc.terminals.local & &parent_acc.terminals.local;

            // Create a new Acc for the popped node. Path constraints are the same, local is updated.
            let new_acc = Arc::new(Acc {
                llm_tokens: ConstraintSet {
                    local: new_local_llm,
                    path_union: pred_arc.acc.llm_tokens.path_union.clone(),
                    path_intersection: pred_arc.acc.llm_tokens.path_intersection.clone(),
                },
                terminals: ConstraintSet {
                    local: new_local_terminals,
                    path_union: pred_arc.acc.terminals.path_union.clone(),
                    path_intersection: pred_arc.acc.terminals.path_intersection.clone(),
                },
            });

            // Check for dead paths after applying parent constraints.
            if new_acc.llm_tokens.union().is_empty() {
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
        self.pop().popn(n).to_node()
    }

    /// Merges another `GSSNode` into this one. This is a union of possibilities.
    #[time_it]
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        if other.predecessors.is_empty() && other.acc.llm_tokens.local == HybridBitset::all() { return; }
        if self.predecessors.is_empty() && self.acc.llm_tokens.local == HybridBitset::all() {
            *self = other.clone();
            return;
        }

        // Merge local constraints by taking the union of allowed sets.
        let merged_local_llm = &self.acc.llm_tokens.local | &other.acc.llm_tokens.local;
        let merged_local_terminals = &self.acc.terminals.local | &other.acc.terminals.local;
        let merged_local_acc = Arc::new(Acc::new_with_local_constraints(merged_local_llm, merged_local_terminals));
        
        // Merge predecessor maps
        let mut new_predecessors = self.predecessors.clone();
        merge_node_maps(&mut new_predecessors, other.predecessors.clone());
        
        // Create a new node with the merged properties.
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
        // The new node is an aggregation point, so it has no local constraints of its own.
        let local_acc = Arc::new(Acc::new_fresh());
        GSSNode::new_with_map(local_acc, self.node_map.clone())
    }
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }
    pub fn predecessor(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    pub fn to_node(&self) -> GSSNode {
        // The new node's local constraints are the parent's.
        // It has a single predecessor.
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
        new_acc.llm_tokens.local = &new_acc.llm_tokens.local & allowed_tokens;

        if new_acc.llm_tokens.union().is_empty() {
            None
        } else {
            Some((new_acc, true))
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
    }
}

pub fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let allowed_mask = HybridBitset::max_ones() - tokens_to_disallow;
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed_mask, memo);
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        let continue_recursion = new_acc.llm_tokens.local != HybridBitset::all();
        new_acc.llm_tokens.local = HybridBitset::all();
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
    }
}

pub fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        let mut allowed_mask = new_acc.terminals.local.clone();
        
        for (state_id, disallowed_bv) in disallowed_terminals {
            if let Some(mut current_allowed) = allowed_mask.remove(state_id.0) {
                current_allowed = current_allowed - disallowed_bv;
                if !current_allowed.is_empty() {
                    allowed_mask.insert(state_id.0, current_allowed);
                }
            }
        }
        new_acc.terminals.local = allowed_mask;
        Some((new_acc, true))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        // Prune if all paths to this node disallow a matched terminal.
        let allowed_by_all_paths = node.acc.terminals.union();
        for (state_id, matched_bv) in matched_terminals {
            if let Some(allowed_l2) = allowed_by_all_paths.get_l2_bitset(state_id.0) {
                if !matched_bv.is_subset(allowed_l2) {
                    return None; // A matched terminal is not in the allowed set. Prune.
                }
            } else {
                // No terminals are allowed for this state_id, but we matched some. Prune.
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
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        let mut new_local_terminals = HybridL2Bitset::new();
        let mut changed = false;

        for (old_id_val, bv) in new_acc.terminals.local.iter() {
            let old_id = TokenizerStateID(old_id_val);
            if let Some(&new_id) = map.get(&old_id) {
                // This is inefficient. A better way would be to rebuild the RangeMapBlaze.
                // For now, this is a simple approximation.
                new_local_terminals.insert(new_id.0, bv.clone());
                if old_id != new_id {
                    changed = true;
                }
            } else {
                changed = true; // A state was removed.
            }
        }
        
        new_acc.terminals.local = new_local_terminals;
        Some((new_acc, changed))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
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
            const MAX_SAMPLES: usize = 3;
            let token_samples: Vec<_> = bv.iter().take(MAX_SAMPLES)
                .filter_map(|internal_id| bimap.get_by_right(&internal_id))
                .filter_map(|original_id| token_map.get_by_right(&LLMTokenID(*original_id)))
                .map(|token_bytes| format!("{:?}", String::from_utf8_lossy(token_bytes)))
                .collect();

            let samples_str = token_samples.join(", ");
            let total_tokens = bv.len();
            if total_tokens > MAX_SAMPLES {
                format!("{}({} tokens: [{}, ...])", label, total_tokens, samples_str)
            } else {
                format!("{}({} tokens: [{}])", label, total_tokens, samples_str)
            }
        } else {
            format!("{}({} tokens)", label, bv.len())
        }
    };

    let format_disallowed_terminals = |allowed_terminals: &HybridL2Bitset| -> String {
        if allowed_terminals.is_empty() {
            return "Terminals(All Disallowed)".to_string();
        }
        // This is complex to display concisely. We show disallowed terminals per state.
        // We can only know about states present in the allowed set.
        let mut parts = Vec::new();
        for (state_val, allowed_bv) in allowed_terminals.iter() {
             let disallowed_bv = HybridBitset::max_ones() - allowed_bv;
             if !disallowed_bv.is_empty() {
                let names: Vec<_> = disallowed_bv.iter()
                    .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                        .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.to_string()))
                    .collect();
                parts.push(format!("State {}:[{}]", state_val, names.join(", ")));
             }
        }
        if parts.is_empty() {
            "Terminals(None Disallowed)".to_string()
        } else {
            format!("Terminals({})", parts.join("; "))
        }
    };

    let local_llm_str = format_allowed_llm(&node.acc.llm_tokens.local, "LocalLLM");
    let union_llm_str = format_allowed_llm(&node.acc.llm_tokens.union(), "UnionLLM");
    let disallowed_terminals_str = format_disallowed_terminals(&node.acc.terminals.intersection());

    format!("[{}, {}, {}]", local_llm_str, union_llm_str, disallowed_terminals_str)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::table::StateID;

    // Helper to create a local Acc that disallows a single token.
    fn mock_acc(val: usize) -> Acc {
        let mut disallowed_bv = LLMTokenBV::zeros();
        disallowed_bv.insert(val);
        let allowed_bv = HybridBitset::max_ones() - &disallowed_bv;
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
        assert_eq!(node.acc.llm_tokens.local, acc.llm_tokens.local);
        assert!(node.acc.llm_tokens.path_union.is_empty());
        assert_eq!(node.acc.llm_tokens.path_intersection, HybridBitset::all());
        assert!(node.predecessors.is_empty());
        assert_eq!(node.max_depth, 0);
    }

    #[test]
    fn test_gss_push() {
        let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
        let pushed = root.push(mock_edge(10), mock_acc(2)); // Allows all but 2

        assert_eq!(pushed.max_depth, 1);
        assert_eq!(pushed.acc.llm_tokens.local, mock_acc(2).llm_tokens.local);
        
        // The path_union of the new node should be the full union of its predecessor.
        let expected_path_union = root.acc.llm_tokens.union();
        assert_eq!(pushed.acc.llm_tokens.path_union, expected_path_union);

        // The full union of the new node is its path_union | its local.
        // (All but 1) | (All but 2) = All
        let full_union = pushed.acc.llm_tokens.union();
        assert_eq!(full_union, HybridBitset::all());

        // The full intersection is path_intersection & local.
        // (All but 1) & (All but 2) = All but {1, 2}
        let full_intersection = pushed.acc.llm_tokens.intersection();
        let mut expected_disallowed = HybridBitset::zeros();
        expected_disallowed.insert(1);
        expected_disallowed.insert(2);
        assert_eq!(full_intersection, HybridBitset::max_ones() - &expected_disallowed);
    }

    #[test]
    fn test_gss_pop() {
        let root = Arc::new(GSSNode::new(mock_acc(1)));
        let pushed = Arc::new(root.push(mock_edge(10), mock_acc(2)));
        
        let pop_result = pushed.pop();
        assert_eq!(pop_result.node_map.len(), 1);
        
        let popped_node_arc = pop_result.node_map.values().next().unwrap();
        
        // Popping from `pushed` should yield `root`, but with its local acc updated
        // by `pushed`'s local acc. The new local is an intersection of allowed sets.
        let expected_local = &root.acc.llm_tokens.local & &pushed.acc.llm_tokens.local;
        assert_eq!(popped_node_arc.acc.llm_tokens.local, expected_local);
        
        let mut disallowed = HybridBitset::zeros();
        disallowed.insert(1);
        disallowed.insert(2);
        let expected_allowed = HybridBitset::max_ones() - &disallowed;
        assert_eq!(popped_node_arc.acc.llm_tokens.local, expected_allowed);
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

        // Merged local acc should be the union of allowed sets.
        // (All but 1) | (All but 2) = All
        assert_eq!(merged.acc.llm_tokens.local, HybridBitset::all());

        // Merged node should have one predecessor (n0) from two paths
        assert_eq!(merged.num_predecessors(), 1);

        // The full union of the merged node should be the union of its predecessors' unions,
        // ORed with the merged local.
        let pred_full_union = n0.acc.llm_tokens.union(); // All
        let expected_path_union = &pred_full_union | &pred_full_union; // Still All
        assert_eq!(merged.acc.llm_tokens.path_union, expected_path_union);
        assert_eq!(merged.acc.llm_tokens.union(), HybridBitset::all());
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

        let mut preds_map = NodeMap::new();
        preds_map.insert((mock_edge(100), b.max_depth()), b.clone());
        preds_map.insert((mock_edge(100), c.max_depth()), c.clone());

        let mut root = GSSNode::new_with_map(Arc::new(empty_acc()), preds_map);
        assert_eq!(root.num_predecessors(), 2);

        // Fuse predecessors of root (levels=1)
        root.fuse_predecessors(1);

        assert_eq!(root.num_predecessors(), 1);
        let fused_pred_arc = root.predecessors().values().next().unwrap();

        // The fused predecessor's local acc should be the union of the original locals.
        // (All but 1) | (All but 2) = All
        assert_eq!(fused_pred_arc.acc.llm_tokens.local, HybridBitset::all());
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
