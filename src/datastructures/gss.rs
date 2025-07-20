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
use crate::glr::grammar::Terminal;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
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
/// For a given tokenizer state, holds the bitvector of allowed terminals.
pub type AllowedTerminals = BTreeMap<TokenizerStateID, TerminalBV>;


// --- Accumulator (Acc) ---

/// Represents a set of constraints (allowed tokens/terminals) for a GSS path or node.
/// This struct holds the complete constraint information for a set of paths (e.g., all paths to a node).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    allowed_llm_tokens: LLMTokenBV,
    allowed_terminals: AllowedTerminals,
}

impl Default for Acc {
    fn default() -> Self {
        Self {
            allowed_llm_tokens: LLMTokenBV::max_ones(),
            allowed_terminals: BTreeMap::new(),
        }
    }
}

/// Combines multiple terminal maps using a provided bitwise operation.
fn combine_terminal_maps<'a>(
    maps: impl IntoIterator<Item = &'a AllowedTerminals>,
    op: impl Fn(&TerminalBV, &TerminalBV) -> TerminalBV,
    identity: TerminalBV,
) -> AllowedTerminals {
    let maps_vec: Vec<_> = maps.into_iter().collect();
    if maps_vec.is_empty() {
        return BTreeMap::new();
    }

    let mut all_keys = BTreeSet::new();
    for map in &maps_vec {
        all_keys.extend(map.keys());
    }

    let mut result_map = BTreeMap::new();
    let max_ones = TerminalBV::max_ones();
    for key in all_keys {
        let mut combined_bv = identity.clone();
        let mut first = true;
        for map in &maps_vec {
            // If a key is missing, it implies all terminals are allowed for that state.
            let bv = map.get(key).unwrap_or(&max_ones);
            if first {
                combined_bv = bv.clone();
                first = false;
            } else {
                combined_bv = op(&combined_bv, bv);
            }
        }
        if !combined_bv.is_empty() {
            result_map.insert(*key, combined_bv);
        }
    }
    result_map
}

impl Acc {
    /// Creates a fresh, unconstrained accumulator (all tokens/terminals allowed).
    pub fn new_fresh() -> Self {
        Self {
            allowed_llm_tokens: LLMTokenBV::max_ones(),
            allowed_terminals: BTreeMap::new(),
        }
    }

    pub fn new_fresh_without_vocab() -> Self {
        Self::new_fresh()
    }

    pub fn new_fresh_from_existing(_acc: &Acc) -> Self {
        Self::new_fresh()
    }

    pub fn new_fresh_from_existing_stack(_stack: &GSSNode) -> Self {
        Self::new_fresh()
    }

    /// Combines this accumulator with another sequentially, as if chaining constraints.
    /// This results in the **intersection** of allowed sets.
    pub fn split(&self, other: &Self) -> Self {
        let new_llm_tokens = &self.allowed_llm_tokens & &other.allowed_llm_tokens;
        let new_terminals = combine_terminal_maps(
            [&self.allowed_terminals, &other.allowed_terminals],
            |a, b| a & b,
            TerminalBV::max_ones(),
        );
        Self {
            allowed_llm_tokens: new_llm_tokens,
            allowed_terminals: new_terminals,
        }
    }

    /// Merges a collection of accumulators from parallel paths.
    /// This results in the **union** of allowed sets.
    pub fn merge_many<'a>(accs: impl IntoIterator<Item = &'a Self>) -> Self {
        let accs_vec: Vec<_> = accs.into_iter().collect();
        if accs_vec.is_empty() {
            return Self::new_fresh();
        }
        let new_llm_tokens = accs_vec.iter().fold(LLMTokenBV::zeros(), |acc, item| &acc | &item.allowed_llm_tokens);
        let new_terminals = combine_terminal_maps(
            accs_vec.iter().map(|acc| &acc.allowed_terminals),
            |a, b| a | b,
            TerminalBV::zeros(),
        );
        Self {
            allowed_llm_tokens: new_llm_tokens,
            allowed_terminals: new_terminals,
        }
    }

    /// Intersects a collection of accumulators from parallel paths.
    /// This results in the **intersection** of allowed sets.
    pub fn intersect_many<'a>(accs: impl IntoIterator<Item = &'a Self>) -> Self {
        let accs_vec: Vec<_> = accs.into_iter().collect();
        if accs_vec.is_empty() {
            return Self::new_fresh();
        }
        let new_llm_tokens = accs_vec.iter().fold(LLMTokenBV::max_ones(), |acc, item| &acc & &item.allowed_llm_tokens);
        let new_terminals = combine_terminal_maps(
            accs_vec.iter().map(|acc| &acc.allowed_terminals),
            |a, b| a & b,
            TerminalBV::max_ones(),
        );
        Self {
            allowed_llm_tokens: new_llm_tokens,
            allowed_terminals: new_terminals,
        }
    }

    pub fn is_alive(&self) -> bool {
        !self.allowed_llm_tokens.is_empty()
    }
}


// --- GSS Node & Core Implementation ---

/// A node in the Graph-Structured Stack (GSS).
#[derive(Debug, Clone)]
pub struct GSSNode {
    /// Local constraints applied at this specific node/edge.
    local_acc: Arc<Acc>,
    /// The union of constraints over all paths from a root to this node's predecessors.
    pred_union_acc: Arc<Acc>,
    /// The intersection of constraints over all paths from a root to this node's predecessors.
    pred_intersection_acc: Arc<Acc>,

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

fn compute_hash_key(predecessors: &NodeMap, local_acc: &Acc) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    local_acc.hash(&mut hasher);
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
    /// Creates a new GSS root node with no predecessors.
    pub fn new(local_acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let local_acc = Arc::new(local_acc);
        let pred_union_acc = Arc::new(Acc { allowed_llm_tokens: LLMTokenBV::zeros(), ..Default::default() });
        let pred_intersection_acc = Arc::new(Acc { allowed_llm_tokens: LLMTokenBV::max_ones(), ..Default::default() });

        let hash_key_cache = compute_hash_key(&predecessors, &local_acc);
        let max_depth = compute_max_depth(&predecessors);
        Self { local_acc, pred_union_acc, pred_intersection_acc, predecessors, hash_key_cache, max_depth }
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    fn new_with_map(local_acc: Arc<Acc>, predecessors: NodeMap) -> Self {
        let pred_full_unions: Vec<_> = predecessors.values().map(|p| p.full_union_acc()).collect();
        let pred_full_intersections: Vec<_> = predecessors.values().map(|p| p.full_intersection_acc()).collect();

        let pred_union_acc = Arc::new(Acc::merge_many(pred_full_unions.iter()));
        let pred_intersection_acc = Arc::new(Acc::intersect_many(pred_full_intersections.iter()));

        let hash_key_cache = compute_hash_key(&predecessors, &local_acc);
        let max_depth = compute_max_depth(&predecessors);
        Self { local_acc, pred_union_acc, pred_intersection_acc, predecessors, hash_key_cache, max_depth }
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, local_acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        predecessors_map.insert((edge_value, predecessor_arc.dest_key()), predecessor_arc.clone());
        Self::new_with_map(Arc::new(local_acc), predecessors_map)
    }

    pub fn fresh_from_existing(node: &GSSNode) -> Self {
        Self::new(Acc::new_fresh_from_existing_stack(&node))
    }

    fn predecessors(&self) -> &NodeMap { &self.predecessors }

    /// Returns the full union of constraints for any path ending at this node.
    pub fn full_union_acc(&self) -> Acc {
        self.pred_union_acc.split(&self.local_acc)
    }

    /// Returns the full intersection of constraints for all paths ending at this node.
    pub fn full_intersection_acc(&self) -> Acc {
        self.pred_intersection_acc.split(&self.local_acc)
    }

    pub fn num_predecessors(&self) -> usize { self.predecessors.len() }
    pub fn max_depth(&self) -> MaxDepth { self.max_depth }
    pub fn dest_key(&self) -> DestKey { self as *const GSSNode as usize }
    
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.full_union_acc().allowed_llm_tokens
    }
    
    pub fn disallowed_terminals(&self) -> BTreeMap<TokenizerStateID, TerminalBV> {
        let allowed = &self.full_union_acc().allowed_terminals;
        let mut disallowed = BTreeMap::new();
        for (state_id, bv) in allowed {
            let inverted_bv = bv.inverted();
            if !inverted_bv.is_empty() {
                disallowed.insert(*state_id, inverted_bv);
            }
        }
        disallowed
    }

    pub fn is_empty(&self) -> bool { self.predecessors.is_empty() }
    pub fn is_alive(&self) -> bool { self.full_union_acc().is_alive() }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    pub fn push(&self, edge_value: ParseStateEdgeContent, local_acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, local_acc_for_new_node)
    }

    /// Pops the top state from the stack(s), returning a `GSSPop` structure.
    /// The accumulators of predecessors are adjusted to include this node's local constraints.
    #[time_it("GSSNode::pop")]
    pub fn pop(&self) -> GSSPop {
        let mut new_node_map = NodeMap::new();
        let parent_local_acc = &self.local_acc;

        for ((edge_val, dest_key), pred_arc) in &self.predecessors {
            let new_local_acc = Arc::new(pred_arc.local_acc.split(parent_local_acc));
            
            let new_full_union = pred_arc.pred_union_acc.split(&new_local_acc);
            if !new_full_union.is_alive() {
                crate::debug!(6, "Dead path after splitting\n{:?}\nwith local\n{:?}\nresulting in\n{:?}", pred_arc.pred_union_acc, new_local_acc, new_full_union);
                continue;
            }

            let new_pred_node = GSSNode {
                local_acc: new_local_acc,
                pred_union_acc: pred_arc.pred_union_acc.clone(),
                pred_intersection_acc: pred_arc.pred_intersection_acc.clone(),
                predecessors: pred_arc.predecessors.clone(),
                hash_key_cache: compute_hash_key(&pred_arc.predecessors, &pred_arc.local_acc),
                max_depth: pred_arc.max_depth,
            };

            let new_pred_arc = Arc::new(new_pred_node);
            new_node_map.insert((edge_val.clone(), *dest_key), new_pred_arc);
        }

        GSSPop { parent_node: self, node_map: new_node_map }
    }

    /// Pops `n` levels from the GSS.
    #[time_it("GSSNode::popn")]
    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        self.pop().popn(n).to_node()
    }

    /// Merges another `GSSNode` into this one.
    #[time_it]
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        let new_local = Arc::new(Acc::merge_many([self.local_acc.as_ref(), other.local_acc.as_ref()]));
        
        let mut new_predecessors = self.predecessors.clone();
        merge_node_maps(&mut new_predecessors, other.predecessors.clone());
        
        let new_node = GSSNode::new_with_map(new_local, new_predecessors);
        *self = new_node;
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let acc = (*self.local_acc).clone();
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

    #[time_it("GSSPop::popn")]
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
    #[time_it("GSSPop::to_node")]
    pub fn to_node(&self) -> GSSNode {
        let local_acc = Arc::new(Acc::new_fresh());
        GSSNode::new_with_map(local_acc, self.node_map.clone())
    }
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }
    pub fn predecessor(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    pub fn to_node(&self) -> GSSNode {
        let local_acc = self.parent_node.local_acc.split(&self.predecessor_node.full_union_acc());
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            local_acc,
        )
    }

    pub fn to_arc_node(&self) -> Arc<GSSNode> {
        Arc::new(self.to_node())
    }

    #[time_it("GSSPeek::popn")]
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
            self.local_acc == other.local_acc &&
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
            .then_with(|| self.local_acc.cmp(&other.local_acc))
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
        let mut new_local_acc = (*node.local_acc).clone();
        new_local_acc.allowed_llm_tokens &= allowed_tokens;

        let temp_full_acc = node.pred_union_acc.split(&new_local_acc);
        if temp_full_acc.is_alive() {
            Some((new_local_acc, true))
        } else {
            None
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
    }
}

#[time_it]
pub fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let allowed_tokens = &LLMTokenBV::max_ones() - tokens_to_disallow;
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed_tokens, memo);
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_local_acc = (*node.local_acc).clone();
        let continue_recursion = new_local_acc.allowed_llm_tokens != LLMTokenBV::max_ones();
        new_local_acc.allowed_llm_tokens = LLMTokenBV::max_ones();
        Some((new_local_acc, continue_recursion))
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
        let mut new_local_acc = (*node.local_acc).clone();
        for (state_id, bv_to_disallow) in disallowed_terminals {
            let entry = new_local_acc.allowed_terminals.entry(*state_id).or_insert_with(TerminalBV::max_ones);
            *entry -= bv_to_disallow;
        }
        Some((new_local_acc, true))
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
    let max_ones = TerminalBV::max_ones();
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let intersection_acc = node.full_intersection_acc();
        for (state_id, matched) in matched_terminals {
            let allowed_by_gss = intersection_acc.allowed_terminals.get(state_id).unwrap_or(&max_ones);
            if allowed_by_gss.is_disjoint(matched) {
                return None; // All paths to this node disallow a terminal that was just matched. Prune.
            }
        }
        
        let union_acc = node.full_union_acc();
        let mut needs_recursion = false;
        for (state_id, matched) in matched_terminals {
            let allowed_by_gss = union_acc.allowed_terminals.get(state_id).unwrap_or(&max_ones);
            if allowed_by_gss.is_disjoint(matched) {
                needs_recursion = true;
                break;
            }
        }
        Some(((*node.local_acc).clone(), needs_recursion))
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
        let mut new_local_acc = (*node.local_acc).clone();
        let mut new_allowed = BTreeMap::new();
        let mut changed = false;

        for (old_id, bv) in &new_local_acc.allowed_terminals {
            if let Some(&new_id) = map.get(old_id) {
                *new_allowed.entry(new_id).or_insert_with(TerminalBV::max_ones) &= bv;
                if old_id != &new_id {
                    changed = true;
                }
            } else {
                changed = true; // A state was removed.
            }
        }
        
        if new_allowed.len() != new_local_acc.allowed_terminals.len() {
            changed = true;
        }

        new_local_acc.allowed_terminals = new_allowed;
        Some((new_local_acc, changed))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(Acc::new_fresh()));
    }
}

impl GSSNode {
    /// Fuses predecessor nodes that share the same edge value, even if they are at different depths.
    /// This can simplify the GSS by reducing path diversity. The fusion process is applied
    /// recursively for `levels` number of levels down from this node.
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
    let fused_node = GSSNode::new_with_map(node_arc.local_acc.clone(), new_predecessors_map);

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
    let format_single_acc = |acc: &Acc, label: &str| -> String {
        let bv = &acc.allowed_llm_tokens;
        let llm_info = if let (Some(bimap), Some(token_map)) = (original_internal_bimap, llm_token_map) {
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
        };

        if acc.allowed_terminals.is_empty() {
            return format!("{}:({})", label, llm_info);
        }

        let allowed_info = acc.allowed_terminals.iter()
            .map(|(state_id, bv)| {
                let names: Vec<_> = bv.iter()
                    .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                        .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.to_string()))
                    .collect();
                format!("State {}:[{}]", state_id.0, names.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; ");

        format!("{}:({}, Terminals({}))", label, llm_info, allowed_info)
    };

    let local_str = format_single_acc(&node.local_acc, "Local");
    let union_str = format_single_acc(&node.full_union_acc(), "FullUnion");

    format!("[{}, {}]", local_str, union_str)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::table::StateID;

    fn mock_llm_acc(vals: &[usize]) -> Acc {
        Acc {
            allowed_llm_tokens: LLMTokenBV::from_iter(vals.iter().cloned()),
            allowed_terminals: BTreeMap::new(),
        }
    }

    fn empty_acc() -> Acc {
        Acc {
            allowed_llm_tokens: LLMTokenBV::zeros(),
            allowed_terminals: BTreeMap::new(),
        }
    }
    
    fn fresh_acc() -> Acc {
        Acc::new_fresh()
    }

    fn mock_edge(id: usize) -> ParseStateEdgeContent {
        ParseStateEdgeContent { state_id: StateID(id) }
    }

    #[test]
    fn test_gss_new_node() {
        let acc = mock_llm_acc(&[1]);
        let node = GSSNode::new(acc.clone());
        assert_eq!(*node.local_acc, acc);
        assert!(node.pred_union_acc.allowed_llm_tokens.is_empty());
        assert_eq!(node.pred_intersection_acc.allowed_llm_tokens, LLMTokenBV::max_ones());
        assert!(node.predecessors.is_empty());
        assert_eq!(node.max_depth, 0);
    }

    #[test]
    fn test_gss_push() {
        let root = Arc::new(GSSNode::new(mock_llm_acc(&[1, 2])));
        let pushed = root.push(mock_edge(10), mock_llm_acc(&[2, 3]));

        assert_eq!(pushed.max_depth, 1);
        assert_eq!(pushed.local_acc.allowed_llm_tokens, LLMTokenBV::from_iter(vec![2, 3]));
        
        // The pred_union_acc of the new node should be the full_union_acc of its predecessor.
        let expected_pred_union = root.full_union_acc();
        assert_eq!(*pushed.pred_union_acc, expected_pred_union);

        // The full_union_acc of the new node is its pred_union_acc split with its local_acc (intersection).
        let expected_full_union = expected_pred_union.split(&pushed.local_acc);
        assert_eq!(pushed.full_union_acc(), expected_full_union);
        assert_eq!(pushed.full_union_acc().allowed_llm_tokens.iter().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn test_gss_pop() {
        let root = Arc::new(GSSNode::new(mock_llm_acc(&[1, 2])));
        let pushed = Arc::new(root.push(mock_edge(10), mock_llm_acc(&[2, 3])));
        
        let pop_result = pushed.pop();
        assert_eq!(pop_result.node_map.len(), 1);
        
        let popped_node_arc = pop_result.node_map.values().next().unwrap();
        
        // Popping from `pushed` should yield `root`, but with its local acc updated
        // by `pushed`'s local acc.
        let expected_local = root.local_acc.split(&pushed.local_acc);
        assert_eq!(*popped_node_arc.local_acc, expected_local);
        assert_eq!(popped_node_arc.local_acc.allowed_llm_tokens.iter().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn test_gss_merge() {
        // Path 1: 0 -> 10(acc1)
        let n0 = Arc::new(GSSNode::new(fresh_acc()));
        let n1 = Arc::new(n0.push(mock_edge(0), mock_llm_acc(&[1])));

        // Path 2: 0 -> 20(acc2)
        let n2 = Arc::new(n0.push(mock_edge(0), mock_llm_acc(&[2])));

        // Merge n1 and n2
        let mut merged = (*n1).clone();
        merged.merge(&n2);

        // Merged local acc should be the merge (union) of the two local accs.
        let expected_local = Acc::merge_many([n1.local_acc.as_ref(), n2.local_acc.as_ref()]);
        assert_eq!(expected_local.allowed_llm_tokens.iter().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(*merged.local_acc, expected_local);

        // Merged node should have one predecessor (n0) from two paths
        assert_eq!(merged.num_predecessors(), 1);

        // The full union of the merged node is its pred_union split with its local.
        let pred_full_union = n0.full_union_acc();
        let expected_pred_union = Acc::merge_many([&pred_full_union, &pred_full_union]);
        assert_eq!(*merged.pred_union_acc, expected_pred_union);
        
        let full_union = merged.full_union_acc();
        assert_eq!(full_union.allowed_llm_tokens.iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn test_gss_fuse_predecessors() {
        // Structure:
        // root -> (B, edge 100)
        // root -> (C, edge 100)
        let leaf1 = Arc::new(GSSNode::new(mock_llm_acc(&[1])));
        let leaf2 = Arc::new(GSSNode::new(mock_llm_acc(&[2])));
        let b = Arc::new(leaf1.push(mock_edge(1), fresh_acc()));
        let c_tmp = Arc::new(leaf2.push(mock_edge(2), fresh_acc()));
        let c = Arc::new(c_tmp.push(mock_edge(3), fresh_acc()));

        let mut preds_map = NodeMap::new();
        preds_map.insert((mock_edge(100), b.dest_key()), b.clone());
        preds_map.insert((mock_edge(100), c.dest_key()), c.clone());

        let mut root = GSSNode::new_with_map(Arc::new(fresh_acc()), preds_map);
        assert_eq!(root.num_predecessors(), 2);

        // Fuse predecessors of root (levels=1)
        root.fuse_predecessors(1);

        assert_eq!(root.num_predecessors(), 1);
        let fused_pred_arc = root.predecessors().values().next().unwrap();

        let mut allowed_vec: Vec<_> = fused_pred_arc.full_union_acc().allowed_llm_tokens.iter().collect();
        allowed_vec.sort();
        assert_eq!(allowed_vec, vec![1, 2]);
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
