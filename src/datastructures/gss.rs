use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};
use bimap::BiBTreeMap;
use deterministic_hash::DeterministicHasher;

use crate::glr::parser::ParseStateEdgeContent;
use crate::constraint::{LLMTokenBV, TerminalBV};
use crate::datastructures::gss::acc_mod::Acc;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::glr::grammar::Terminal;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;

// --- Type Aliases ---

pub type MaxDepth = usize;
/// Maps a node's depth to its predecessors at that depth.
type NodeMap = BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, Arc<GSSNode>>>;
/// A cache for structurally unique nodes, mapping a predecessor structure to a canonical node.
type NodeCache = HashMap<NodeMap, Arc<GSSNode>>;
/// A temporary set of predecessors used during node construction and simplification.
type NodeSet = BTreeSet<(Arc<GSSNode>, ParseStateEdgeContent)>;
/// Represents the set of allowed LLM tokens for a path. `None` means all tokens are allowed.
pub type LLMTokenInfo = Option<LLMTokenBV>;
/// For a given tokenizer state, holds the union and intersection of disallowed terminals from different paths.
pub type TerminalInfo = BTreeMap<TokenizerStateID, TerminalInfoValue>;


// --- TerminalInfo & Path Accumulation ---

/// Stores disallowed terminals for a single tokenizer state, aggregated across multiple paths.
/// `union` tracks all terminals disallowed by *any* path.
/// `intersection` tracks terminals disallowed by *all* paths.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalInfoValue {
    pub union: TerminalBV,
    pub intersection: TerminalBV,
}

impl TerminalInfoValue {
    pub fn new(union: TerminalBV, intersection: TerminalBV) -> Self {
        Self { union, intersection }
    }

    /// Creates an empty value, allowing all terminals.
    fn zeros() -> Self {
        Self {
            union: TerminalBV::zeros(),
            intersection: TerminalBV::zeros(),
        }
    }

    /// The identity element for path merging operations.
    /// An empty union and a full intersection mean "no constraints yet".
    pub fn identity_for_union_or_intersection() -> Self {
        Self {
            union: TerminalBV::zeros(),
            intersection: TerminalBV::max_ones(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.union.is_empty()
    }

    pub fn contains(&self, terminal: usize) -> bool {
        self.union.contains(terminal)
    }
}

// Bitwise operations for combining TerminalInfoValue with other bitsets.
// These are used for applying external constraints.

impl BitAnd<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitand(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union & rhs,
            intersection: &self.intersection & rhs,
        }
    }
}

impl BitAndAssign<&TerminalBV> for TerminalInfoValue {
    fn bitand_assign(&mut self, rhs: &TerminalBV) {
        self.union &= rhs;
        self.intersection &= rhs;
    }
}

impl BitOr<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitor(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | rhs,
            intersection: &self.intersection & rhs, // Note: intersection is intentional
        }
    }
}

impl BitOrAssign<&TerminalBV> for TerminalInfoValue {
    fn bitor_assign(&mut self, rhs: &TerminalBV) {
        self.union |= rhs;
        self.intersection &= rhs; // Note: intersection is intentional
    }
}

impl BitXor<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitxor(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | rhs,
            intersection: &self.intersection | rhs,
        }
    }
}

impl BitXorAssign<&TerminalBV> for TerminalInfoValue {
    fn bitxor_assign(&mut self, rhs: &TerminalBV) {
        self.union |= rhs;
        self.intersection |= rhs;
    }
}

impl Sub<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn sub(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union - rhs,
            intersection: &self.intersection - rhs,
        }
    }
}

impl SubAssign<&TerminalBV> for TerminalInfoValue {
    fn sub_assign(&mut self, rhs: &TerminalBV) {
        self.union -= rhs;
        self.intersection -= rhs;
    }
}

// Bitwise operations for combining two TerminalInfoValue instances.
// These are used for merging paths within the GSS.

impl BitAnd<&TerminalInfoValue> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitand(self, rhs: &TerminalInfoValue) -> Self::Output {
        TerminalInfoValue {
            union: &self.union & &rhs.union,
            intersection: &self.intersection & &rhs.intersection,
        }
    }
}

impl BitAndAssign<&TerminalInfoValue> for TerminalInfoValue {
    fn bitand_assign(&mut self, rhs: &TerminalInfoValue) {
        self.union &= &rhs.union;
        self.intersection &= &rhs.intersection;
    }
}

impl BitOr<&TerminalInfoValue> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    /// Merges two paths. The new union is the union of both.
    /// The new intersection is the intersection of both.
    fn bitor(self, rhs: &TerminalInfoValue) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | &rhs.union,
            intersection: &self.intersection & &rhs.intersection,
        }
    }
}

impl BitOrAssign<&TerminalInfoValue> for TerminalInfoValue {
    fn bitor_assign(&mut self, rhs: &TerminalInfoValue) {
        self.union |= &rhs.union;
        self.intersection &= &rhs.intersection;
    }
}

impl BitXor<&TerminalInfoValue> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitxor(self, rhs: &TerminalInfoValue) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | &rhs.union,
            intersection: &self.intersection | &rhs.intersection,
        }
    }
}

impl BitXorAssign<&TerminalInfoValue> for TerminalInfoValue {
    fn bitxor_assign(&mut self, rhs: &TerminalInfoValue) {
        self.union |= &rhs.union;
        self.intersection |= &rhs.intersection;
    }
}

/// A trait for data that can be accumulated along paths in the GSS.
pub trait PathAccumulator<Other=Self>: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash {
    /// Merges another path into this one (e.g., for a `merge` operation).
    fn union_assign(&mut self, other: Other);
    /// Intersects this path with a successor path (e.g., for a `pop` operation).
    fn intersect_assign(&mut self, right: Other);

    fn union(mut self, other: Other) -> Self {
        self.union_assign(other);
        self
    }
    fn intersect(mut self, right: Other) -> Self {
        self.intersect_assign(right);
        self
    }
    fn intersect_has_effect(&self, right: &Other) -> bool;
}

impl PathAccumulator for () {
    fn union_assign(&mut self, _other: Self) { }
    fn intersect_assign(&mut self, _right: Self) { }
    fn intersect_has_effect(&self, _right: &Self) -> bool { false }
}

impl PathAccumulator for TerminalInfoValue {
    fn union_assign(&mut self, other: Self) {
        self.union |= &other.union;
        self.intersection &= &other.intersection;
    }

    fn intersect_assign(&mut self, right: Self) {
        self.union &= &right.union;
        self.union |= &right.intersection;
        self.intersection |= &right.intersection;
    }

    fn intersect_has_effect(&self, right: &Self) -> bool {
        // A full comparison is needed as the logic is complex.
        self.clone().intersect(right.clone()) != *self
    }
}

impl PathAccumulator for Option<LLMTokenBV> {
    fn union_assign(&mut self, other: Self) {
        match (self.as_mut(), other) {
            (Some(self_bv), Some(other_bv)) => *self_bv |= other_bv,
            (None, Some(other_bv)) => *self = Some(other_bv),
            (Some(_), None) => { /* self remains Some, representing the union */ },
            (None, None) => { /* self remains None */ },
        }
    }

    fn intersect_assign(&mut self, right: Self) {
        match (self.as_mut(), right) {
            (Some(self_bv), Some(right_bv)) => *self_bv &= right_bv,
            (None, Some(right_bv)) => *self = Some(right_bv),
            (Some(_), None) => { /* self remains Some, representing the intersection */ },
            (None, None) => { /* self remains None */ },
        }
    }

    fn intersect_has_effect(&self, right: &Self) -> bool {
        match (self, right) {
            (Some(self_bv), Some(right_bv)) => !self_bv.is_subset(right_bv),
            (None, Some(_)) => true, // `None` (all) intersecting with `Some` will change.
            _ => false, // Intersecting with `None` (all) has no effect.
        }
    }
}

/// Helper functions for manipulating `TerminalInfo` maps.
mod terminal_info_helpers {
    use super::*;

    pub fn disallowed_terminals_intersect_assign(left: &mut TerminalInfo, right: TerminalInfo) {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(left.keys());
        all_keys.extend(right.keys());
        for tokenizer_state_id in all_keys {
            // An absent key implies "no terminals disallowed", which is the identity for intersection.
            let left_value = left.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
            let right_value = right.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
            let intersection = &left_value & &right_value;
            if !intersection.is_empty() {
                left.insert(tokenizer_state_id, intersection);
            } else {
                left.remove(&tokenizer_state_id);
            }
        }
    }

    pub fn disallowed_terminals_union_assign(left: &mut TerminalInfo, right: TerminalInfo) {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(left.keys());
        all_keys.extend(right.keys());
        for tokenizer_state_id in all_keys {
            // An absent key implies "no terminals disallowed", which is the identity for union.
            let left_value = left.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
            let right_value = right.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
            let union = &left_value | &right_value;
            if !union.is_empty() {
                left.insert(tokenizer_state_id, union);
            } else {
                left.remove(&tokenizer_state_id);
            }
        }
    }

    /// Adds a new set of disallowed terminals to an existing `TerminalInfo`.
    /// This is not a standard union; it uses a `BitXor` logic to update both union and intersection.
    pub fn add_disallowed_terminals(left: &mut TerminalInfo, right: BTreeMap<TokenizerStateID, TerminalBV>) {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(left.keys());
        all_keys.extend(right.keys());
        for tokenizer_state_id in all_keys {
            let mut left_value = left.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::zeros);
            if let Some(right_value) = right.get(&tokenizer_state_id) {
                left_value ^= right_value;
                left.insert(tokenizer_state_id, left_value);
            }
        }
    }
}


// --- Accumulator (Acc) ---

pub mod acc_mod {
    use super::*;
    use terminal_info_helpers::*;

    /// The accumulator for a GSS node, containing all path-dependent information.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Acc {
        /// The set of valid LLM tokens for this path. `None` means unconstrained.
        llm_token_info: LLMTokenInfo,
        /// A map from tokenizer state to terminals that are disallowed on this path.
        disallowed_terminals: TerminalInfo,
    }

    impl Acc {
        pub fn new(llm_token_info: LLMTokenInfo, disallowed_terminals: TerminalInfo) -> Self {
            Self { llm_token_info, disallowed_terminals }
        }

        /// Creates a fresh, unconstrained accumulator.
        pub fn new_fresh() -> Self {
            Self { llm_token_info: None, disallowed_terminals: BTreeMap::new() }
        }

        /// Creates a fresh, unconstrained accumulator. Alias for `new_fresh`.
        pub fn new_for_merging() -> Self {
            Self::new_fresh()
        }

        pub fn llm_token_info(&self) -> &LLMTokenInfo { &self.llm_token_info }
        pub fn llm_token_info_mut(&mut self) -> &mut LLMTokenInfo { &mut self.llm_token_info }
        pub fn disallowed_terminals(&self) -> &TerminalInfo { &self.disallowed_terminals }
        pub fn disallowed_terminals_mut(&mut self) -> &mut TerminalInfo { &mut self.disallowed_terminals }

        /// Checks if the accumulator is in its default, unconstrained state.
        pub fn is_default(&self) -> bool {
            self.llm_token_info.is_none() && self.disallowed_terminals.is_empty()
        }

        /// Checks if the path is dead (e.g., allows no LLM tokens).
        pub fn is_dead(&self) -> bool {
            if let Some(acc) = &self.llm_token_info {
                if acc.is_empty() {
                    return true;
                }
            }
            false
        }

        pub fn is_alive(&self) -> bool {
            !self.is_dead()
        }
    }

    impl PathAccumulator for Acc {
        fn union_assign(&mut self, other: Self) {
            self.llm_token_info.union_assign(other.llm_token_info);
            disallowed_terminals_union_assign(&mut self.disallowed_terminals, other.disallowed_terminals);
        }

        fn intersect_assign(&mut self, right: Self) {
            self.llm_token_info.intersect_assign(right.llm_token_info);
            disallowed_terminals_intersect_assign(&mut self.disallowed_terminals, right.disallowed_terminals);
        }

        fn intersect_has_effect(&self, right: &Self) -> bool {
            self.llm_token_info.intersect_has_effect(&right.llm_token_info) ||
            self.clone().intersect(right.clone()).disallowed_terminals != self.disallowed_terminals
        }
    }
}


// --- GSS Node & Core Implementation ---

/// A node in the Graph-Structured Stack (GSS).
///
/// Each `GSSNode` represents a set of parser stacks that share the same top state.
/// It is defined by its set of predecessors, where each predecessor is another `GSSNode`
/// connected by an edge representing a parser state transition.
///
/// The `GSSNode` is immutable and uses `Arc` for sharing, enabling efficient representation
/// of the potentially exponential number of parse paths.
///
/// The `acc` field holds a `PathAccumulator` that aggregates information (like allowed
/// tokens) along the paths leading to this node.
#[derive(Debug, Clone)]
pub struct GSSNode {
    acc: Acc,
    predecessors: NodeMap,
    hash_key_cache: u64,
    max_depth: MaxDepth,
}

/// A read-only view into a single path segment of the GSS, from a parent to a predecessor.
#[derive(Clone, Copy)]
pub struct GSSPeek<'a> {
    pub(crate) parent_node: &'a GSSNode,
    edge_value: &'a ParseStateEdgeContent,
    pub(crate) predecessor_node: &'a Arc<GSSNode>,
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }
    pub fn predecessor(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    /// Returns a GSS node representing the stack for this specific peeked path.
    /// This is equivalent to popping 0 elements.
    pub fn to_node(&self) -> GSSNode {
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            self.parent_node.acc.clone(),
        )
    }

    pub fn to_arc_node(&self) -> Arc<GSSNode> {
        Arc::new(self.to_node())
    }

    /// Pops `n` elements from the stack represented by this peek.
    /// The accumulator of the returned node is correctly adjusted for the path.
    pub fn popn(&self, n: usize) -> Arc<GSSNode> {
        if n == 0 {
            return self.to_arc_node();
        }

        // For n >= 1, the result is based on the predecessor.
        // First, calculate the accumulator for the path to the predecessor.
        let path_acc = self.parent_node.acc.clone().intersect(self.predecessor_node.acc.clone());
        let pred_with_path_acc = Arc::new(self.predecessor_node.as_ref().clone().with_acc(path_acc));

        if n == 1 {
            pred_with_path_acc
        } else { // n > 1
            Arc::new(pred_with_path_acc.popn(n - 1))
        }
    }
}

// Helper functions for GSSNode construction
fn compute_max_depth(predecessors: &NodeMap) -> MaxDepth {
    predecessors.keys().next_back().map_or(0, |max_pred_depth| max_pred_depth + 1)
}

fn compute_hash_key(predecessors: &NodeMap) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
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

// Basic node creation and manipulation
impl GSSNode {
    /// Creates a new GSS root node with no predecessors.
    pub fn new(acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let hash_key_cache = compute_hash_key(&predecessors);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    fn new_with_map(acc: Acc, predecessors: NodeMap) -> Self {
        let hash_key_cache = compute_hash_key(&predecessors);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        let mut inner_map = BTreeMap::new();
        inner_map.insert(edge_value, predecessor_arc.clone());
        predecessors_map.insert(predecessor_arc.max_depth, inner_map);
        Self::new_with_map(acc, predecessors_map)
    }

    pub fn predecessors(&self) -> &NodeMap { &self.predecessors }
    pub fn num_predecessors(&self) -> usize { self.predecessors.values().map(|inner_map| inner_map.len()).sum() }
    pub fn is_empty(&self) -> bool { self.predecessors.is_empty() }
    pub fn acc(&self) -> &Acc { &self.acc }
    pub fn acc_mut(&mut self) -> &mut Acc { &mut self.acc }
    pub fn llm_token_info(&self) -> &LLMTokenInfo { self.acc.llm_token_info() }
    pub fn llm_token_info_mut(&mut self) -> &mut LLMTokenInfo { self.acc.llm_token_info_mut() }

    /// Helper to clone the node and set a new accumulator.
    fn with_acc(mut self, acc: Acc) -> Self {
        self.acc = acc;
        // The hash key depends on predecessors, not the accumulator, so no recalculation is needed.
        self
    }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    /// Returns a new `GSSNode` with `self` as its single predecessor.
    pub fn push(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, acc_for_new_node)
    }

    /// Consumes the node to push a new state, useful for chaining.
    pub fn push_with_acc(self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, acc_for_new_node)
    }

    /// Pops the top state from the stack(s), returning a new node representing the merged predecessors.
    /// The accumulator of the new node is the union of the accumulators of all valid predecessor paths.
    pub fn pop(&self) -> Self {
        let mut result_accs = Vec::new();
        let mut result_predecessors = NodeMap::new();

        for pred_arc in self.predecessors.values().flat_map(|m| m.values()) {
            // The acc of the path *through* self to pred_arc is self.acc intersected with pred_arc.acc
            let path_acc = self.acc.clone().intersect(pred_arc.acc.clone());
            if path_acc.is_dead() {
                continue;
            }
            result_accs.push(path_acc.clone());

            // Merge predecessors of pred_arc into result_predecessors.
            // Each merged predecessor needs its acc updated based on the path_acc.
            for (inner_depth, inner_preds_for_depth) in &pred_arc.predecessors {
                let result_preds_for_depth = result_predecessors.entry(*inner_depth).or_default();
                for (inner_edge, inner_pred_arc) in inner_preds_for_depth {
                    let mut new_inner_pred_node_data = (**inner_pred_arc).clone();
                    new_inner_pred_node_data.acc = path_acc.clone().intersect(inner_pred_arc.acc.clone());
                    if new_inner_pred_node_data.acc.is_dead() {
                        continue;
                    }

                    match result_preds_for_depth.entry(inner_edge.clone()) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(Arc::new(new_inner_pred_node_data));
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            // If an entry already exists, merge the new one into it.
                            Arc::make_mut(entry.get_mut()).merge(&Arc::new(new_inner_pred_node_data));
                        }
                    }
                }
            }
        }

        // The final accumulator is the union of all valid path accumulators.
        let result_acc = result_accs.into_iter().reduce(|mut acc, next| {
            acc.union_assign(next);
            acc
        }).unwrap_or_else(Acc::new_fresh);

        Self::new_with_map(result_acc, result_predecessors)
    }

    /// Pops `n` elements from the stack.
    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            self.clone()
        } else {
            self.pop().popn(n - 1)
        }
    }

    /// Merges another `GSSNode` into this one.
    /// This is the core operation for handling ambiguity in parsing.
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        if other.predecessors.is_empty() { return; }
        if self.predecessors.is_empty() {
            *self = other.clone();
            return;
        }

        self.acc.union_assign(other.acc.clone());

        for (other_depth, other_preds_for_depth) in &other.predecessors {
            let self_preds_for_depth = self.predecessors.entry(*other_depth).or_default();
            for (edge_val, other_pred_arc) in other_preds_for_depth {
                match self_preds_for_depth.entry(edge_val.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(other_pred_arc.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        // This is a merge point. Ensure the node is mutable and merge the other into it.
                        Arc::make_mut(entry.get_mut()).merge(other_pred_arc);
                    }
                }
            }
        }
        self.hash_key_cache = compute_hash_key(&self.predecessors);
        self.max_depth = compute_max_depth(&self.predecessors);
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.push(edge_value, self.acc().clone())
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

// Trait implementations for GSSNode
impl Hash for GSSNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
        self.acc.hash(state);
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

/// Recursively traverses the GSS, applying a transformation closure to each node's accumulator.
///
/// # Arguments
/// * `node_arc`: The starting node for the transformation.
/// * `closure`: A function that takes the current node's `Acc` and returns:
///   - `None`: to prune the node and its entire subgraph.
///   - `Some((new_acc, continue_recursion))`: to keep the node.
///     - `new_acc`: The transformed accumulator for the node.
///     - `continue_recursion`: If `true`, the transformation recurses to predecessors. If `false`,
///       predecessors are kept as-is, providing a "shallow" update for performance.
/// * `memo`: A memoization table to avoid re-processing nodes in the DAG.
///
/// # Returns
/// `Some(new_node_arc)` if the node is kept, or `None` if it's pruned.
fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    closure: &impl Fn(&Acc) -> Option<(Acc, bool)>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(node_arc.acc()) {
        None => { // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_acc, continue_recursion)) => {
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
                // All predecessors were pruned, so this node must be pruned too.
                memo.insert(node_ptr, None);
                return None;
            }

            let new_node_predecessors_map = process_predecessors(&new_predecessors_set);
            let transformed_node = GSSNode::new_with_map(new_acc, new_node_predecessors_map);

            let result_arc = Arc::new(transformed_node);
            memo.insert(node_ptr, Some(result_arc.clone()));
            Some(result_arc)
        }
    }
}

/// Intersects the `LLMTokenBV` of all nodes in the GSS with a given set of tokens and prunes dead paths.
pub fn intersect_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_intersect: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        if let Some(bv) = new_acc.llm_token_info_mut() {
            *bv &= tokens_to_intersect;
        } else {
            // If unconstrained, it becomes constrained to the intersection set.
            *new_acc.llm_token_info_mut() = Some(tokens_to_intersect.clone());
        }

        if new_acc.is_alive() {
            // Perform a shallow update for performance. Changes are propagated later via pop/merge.
            let continue_recursion = false;
            Some((new_acc, continue_recursion))
        } else {
            None // Prune this node
        }
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned.
        *root_arc = Arc::new(GSSNode::new(root_arc.acc().clone()));
    }
}

/// Subtracts a set of LLM tokens from all nodes in the GSS and prunes dead paths.
pub fn subtract_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    llm_tokens: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        if let Some(bv) = new_acc.llm_token_info_mut() {
            *bv -= llm_tokens;
        } else {
            // If unconstrained (all tokens), it becomes all tokens minus the given set.
            *new_acc.llm_token_info_mut() = Some(LLMTokenBV::max_ones() - llm_tokens.clone());
        }
        if new_acc.is_alive() {
            let continue_recursion = false;
            Some((new_acc, continue_recursion))
        } else {
            None // Prune this node
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc().clone()));
    }
}

/// Resets the LLM token constraints on all nodes, making them unconstrained (`None`).
pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let continue_recursion = !current_acc.is_default();
        let new_acc = Acc::new(None, current_acc.disallowed_terminals().clone());
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc().clone()));
    }
}

/// Adds a set of disallowed terminals to all nodes in the GSS.
pub fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        terminal_info_helpers::add_disallowed_terminals(new_acc.disallowed_terminals_mut(), disallowed_terminals.clone());
        if new_acc.is_alive() {
            let continue_recursion = false;
            Some((new_acc, continue_recursion))
        } else {
            None
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc().clone()));
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    terminals_map: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    // terminals_map: For each TokenizerStateID, a TerminalBV of terminals that are disallowed.
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut continue_recursion = false;
        let mut new_acc = current_acc.clone();
        for (gss_state_id, gss_disallowed_bv) in new_acc.disallowed_terminals_mut().iter_mut() {
            if let Some(actual_bv_for_state) = terminals_map.get(gss_state_id) {
                // If any terminal disallowed by GSS is also matched by current segment, prune.
                // This means (gss_disallowed_bv AND actual_bv_for_state) must be empty.
                if !gss_disallowed_bv.intersection.is_disjoint(actual_bv_for_state) {
                    return None;
                }
                if !gss_disallowed_bv.union.is_disjoint(actual_bv_for_state) {
                    continue_recursion = true;
                    *gss_disallowed_bv -= actual_bv_for_state;
                }
            }
        }
        Some((new_acc, continue_recursion))
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc().clone()));
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_disallowed_terminals = BTreeMap::new();
        let mut changed = false;

        for (old_id, bv) in current_acc.disallowed_terminals() {
            if let Some(&new_id) = map.get(old_id) {
                *new_disallowed_terminals.entry(new_id).or_insert_with(TerminalInfoValue::zeros) ^= bv;
                if new_disallowed_terminals.get(&new_id) != Some(bv) || old_id != &new_id {
                    changed = true;
                }
            } else {
                changed = true; // A state was removed, which is a change.
            }
        }
        new_disallowed_terminals.retain(|_, bv| !bv.is_empty());

        let new_acc = Acc::new(current_acc.llm_token_info().clone(), new_disallowed_terminals);
        let continue_recursion = changed || !current_acc.disallowed_terminals().is_empty();
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc().clone()));
    }
}


// --- Simplification ---

/// Recursively simplifies a GSS node by maximizing structural sharing.
///
/// It uses a `cache` to store canonical versions of nodes based on their predecessor
/// structure. Nodes with identical structures but different accumulators will share
/// the same underlying predecessor graph, reducing memory usage.
fn simplify_node_recursive(
    node_arc: &Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
    cache: &mut NodeCache,
) -> Arc<GSSNode> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(simplified_arc) = memo.get(&node_ptr) {
        return simplified_arc.clone();
    }

    // Recursively simplify predecessors.
    let simplified_predecessors_set: NodeSet = node_arc.predecessors.values().flat_map(|m| m.iter())
        .map(|(edge_val, pred_arc)| {
            let simplified_pred_arc = simplify_node_recursive(pred_arc, memo, cache);
            (simplified_pred_arc, edge_val.clone())
        })
        .collect();

    let simplified_predecessors_map = process_predecessors(&simplified_predecessors_set);

    // Get a structurally canonical Arc from the cache, or create and insert it.
    let cached_structural_node = cache.entry(simplified_predecessors_map.clone())
        .or_insert_with(|| {
            // The accumulator for a canonical structural node is the union of its predecessors' accumulators.
            let unioned_acc = simplified_predecessors_map.values().flat_map(|m| m.values())
                .map(|p_arc| p_arc.acc().clone())
                .reduce(|mut acc, next| {
                    acc.union_assign(next);
                    acc
                }).unwrap_or_else(Acc::new_fresh);

            Arc::new(GSSNode::new_with_map(unioned_acc, simplified_predecessors_map))
        });

    // The final simplified node has the structure of the cached node,
    // but its accumulator is the one from the original node.
    let mut final_node_data = (**cached_structural_node).clone();
    final_node_data.acc = node_arc.acc.clone();
    final_node_data.hash_key_cache = compute_hash_key(&final_node_data.predecessors);

    let result_arc = Arc::new(final_node_data);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}

impl GSSNode {
    /// Simplifies the GSS rooted at this node in-place by maximizing structural sharing.
    pub fn simplify(&mut self) {
        let temp_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new();
        let simplified_arc = simplify_node_recursive(&temp_arc, &mut memo, &mut cache);

        // Replace self's content with the simplified content.
        *self = Arc::try_unwrap(simplified_arc).unwrap_or_else(|arc| (*arc).clone());
    }

    /// Simplifies a set of GSS root nodes together, maximizing sharing across all of them.
    /// The `nodes` slice is updated in-place with the simplified `Arc`s.
    pub fn simplify_together(nodes: &mut [&mut Arc<Self>]) {
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new();
        for node_arc_ref_mut in nodes {
            let current_arc = (*node_arc_ref_mut).clone();
            let simplified_arc = simplify_node_recursive(&current_arc, &mut memo, &mut cache);
            **node_arc_ref_mut = simplified_arc;
        }
    }
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

#[derive(Debug, Clone, Default)]
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

            let acc_child = format_acc(pred_arc.acc(), terminal_map, original_internal_bimap, llm_token_map);
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

        let acc_str = format_acc(root_arc.acc(), terminal_map, original_internal_bimap, llm_token_map);
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
    acc: &Acc,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) -> String {
    let llm_info = match acc.llm_token_info() {
        None => "LLM(Any)".to_string(),
        Some(bv) if bv.is_empty() => "LLM(None)".to_string(),
        Some(bv) => {
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
        }
    };

    if acc.disallowed_terminals().is_empty() {
        return format!("({})", llm_info);
    }

    let disallowed_info = acc.disallowed_terminals().iter()
        .map(|(state_id, tiv)| {
            let format_names = |bv: &TerminalBV| -> String {
                let names: Vec<_> = bv.iter()
                    .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                        .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.0.clone()))
                    .collect();
                format!("[{}]", names.join(", "))
            };
            let u_str = format!("U:{}", format_names(&tiv.union));
            let i_str = format!("I:{}", format_names(&tiv.intersection));
            format!("State {}: ({}, {})", state_id.0, u_str, i_str)
        })
        .collect::<Vec<_>>()
        .join("; ");

    format!("({}, Disallowed({}))", llm_info, disallowed_info)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::table::StateID;

    fn mock_acc(val: usize) -> Acc {
        let mut bv = LLMTokenBV::zeros();
        bv.insert(val);
        Acc::new(Some(bv), Default::default())
    }

    fn mock_edge(id: usize) -> ParseStateEdgeContent {
        ParseStateEdgeContent { state_id: StateID(id) }
    }

    #[test]
    fn test_gss_simplification_basic() {
        let acc_base = mock_acc(0);
        let acc_other = mock_acc(1);

        // Node N4 (leaf)
        let n4_v1 = Arc::new(GSSNode::new(acc_base.clone()));
        let n4_v2 = Arc::new(GSSNode::new(acc_other.clone()));

        // D1: ... -> 40 -> N4(acc_base)
        let d1_orig = Arc::new(n4_v1.push(mock_edge(40), acc_base.clone()));

        // D2: ... -> 40 -> N4(acc_other)
        let d2_orig = Arc::new(n4_v2.push(mock_edge(40), acc_other.clone()));

        // C1: ... -> 30 -> D1
        let c1_orig = Arc::new(d1_orig.as_ref().push(mock_edge(30), acc_base.clone()));

        // B1: ... -> 20 -> C1
        let b1_orig = Arc::new(c1_orig.as_ref().push(mock_edge(20), acc_base.clone()));

        // A1: (root) with two predecessors at different depths for the same edge value.
        let mut a1_preds_set = NodeSet::new();
        a1_preds_set.insert((b1_orig.clone(), mock_edge(10)));
        a1_preds_set.insert((d2_orig.clone(), mock_edge(10)));

        let acc_a1 = acc_base.clone().union(acc_other.clone());
        let a1_preds_map = process_predecessors(&a1_preds_set);
        let mut a1_orig = Arc::new(GSSNode::new_with_map(acc_a1.clone(), a1_preds_map));

        // Simplify the structure
        let mut roots_to_simplify = vec![&mut a1_orig];
        GSSNode::simplify_together(&mut roots_to_simplify);

        let s_a1 = roots_to_simplify[0].clone();

        // --- Verification ---
        // A1 should have two predecessor maps because its predecessors have different depths.
        assert_eq!(s_a1.predecessors.len(), 2, "A1 should have 2 predecessor maps for different depths");
        assert_eq!(s_a1.acc(), &acc_a1, "A1 accumulator mismatch");

        // Check predecessor from D2 (depth 1)
        let preds_at_depth_1 = s_a1.predecessors.get(&1).expect("No predecessors at depth 1");
        let s_d2 = preds_at_depth_1.get(&mock_edge(10)).expect("Edge 10 not found for depth 1 pred");
        assert_eq!(s_d2.acc(), &acc_other, "Simplified D2 accumulator mismatch");
        assert_eq!(s_d2.max_depth, 1, "Simplified D2 depth mismatch");

        // Check predecessor from B1 (depth 3)
        let preds_at_depth_3 = s_a1.predecessors.get(&3).expect("No predecessors at depth 3");
        let s_b1 = preds_at_depth_3.get(&mock_edge(10)).expect("Edge 10 not found for depth 3 pred");
        assert_eq!(s_b1.acc(), &acc_base, "Simplified B1 accumulator mismatch");
        assert_eq!(s_b1.max_depth, 3, "Simplified B1 depth mismatch");

        // Verify the structure of the unmerged paths
        let s_c1 = s_b1.predecessors.get(&2).unwrap().get(&mock_edge(20)).unwrap();
        assert_eq!(s_c1.acc(), &acc_base);
        let s_d1 = s_c1.predecessors.get(&1).unwrap().get(&mock_edge(30)).unwrap();
        assert_eq!(s_d1.acc(), &acc_base);
        let s_n4_from_d1 = s_d1.predecessors.get(&0).unwrap().get(&mock_edge(40)).unwrap();
        assert_eq!(s_n4_from_d1.acc(), &acc_base);
        assert!(s_n4_from_d1.predecessors.is_empty());

        // Path from s_d2
        let s_n4_from_d2 = s_d2.predecessors.get(&0).unwrap().get(&mock_edge(40)).unwrap();
        assert_eq!(s_n4_from_d2.acc(), &acc_other);
        assert!(s_n4_from_d2.predecessors.is_empty());

        // The two N4 leaf nodes should be different because their accumulators are different.
        assert_ne!(s_n4_from_d1, s_n4_from_d2);
        assert!(!Arc::ptr_eq(s_n4_from_d1, s_n4_from_d2));

        // Count total unique nodes in the simplified graph starting from s_a1
        let mut all_nodes = HashSet::new();
        fn collect_all_nodes(node: &Arc<GSSNode>, set: &mut HashSet<*const GSSNode>) {
            if set.insert(Arc::as_ptr(node)) {
                for pred_map in node.predecessors.values() {
                    for pred_arc in pred_map.values() {
                        collect_all_nodes(pred_arc, set);
                    }
                }
            }
        }
        collect_all_nodes(&s_a1, &mut all_nodes);
        // Expected nodes: A1, B1, C1, D1, N4_v1, D2, N4_v2 -> Total = 7 nodes
        assert_eq!(all_nodes.len(), 7, "Incorrect number of unique nodes in simplified graph.");
    }
}