use std::sync::RwLock;
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
use crate::datastructures::arc_wrapper::ArcPtrWrapper;
use crate::datastructures::trie::{EdgeInserter, Trie};

use crate::glr::parser::ParseStateEdgeContent;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::grammar::Terminal;
use crate::glr::table::StateID;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
use profiler_macro::{time_it, timeit};

pub(crate) type LLMTokenBV = HybridBitset;
pub(crate) type TerminalBV = HybridBitset;

// --- Type Aliases ---

pub(crate) type MaxDepth = usize;
pub(crate) type DestKey = MaxDepth;
/// Maps an edge value to a map of destination keys (depths) to a list of predecessor nodes.
type NodeMap = BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<Arc<GSSNode>>>>;
/// A temporary set of predecessors used during node construction and simplification.
type NodeSet = OrderedHashSet<(Arc<GSSNode>, ParseStateEdgeContent)>;
/// A 2D bitset where L1 is tokenizer state and L2 is terminal ID.
pub(crate) type TerminalInfo = HybridL2Bitset;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PrecomputedNodeContents {
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

use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use ordered_hash_map::OrderedHashSet;

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

pub(crate) type PrecomputeNode2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;


// --- Accumulator (Acc) ---

/// Represents the full set of allowed tokens and terminals for a GSS node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Acc {
    pub(crate) llm_tokens_union: HybridBitset,
    pub(crate) llm_tokens_intersection: HybridBitset,
    pub(crate) terminals_union: HybridL2Bitset,
    pub(crate) terminals_intersection: HybridL2Bitset,
    pub(crate) needs_push_down: bool,
    pub(crate) trie2_nodes: BTreeSet<ArcPtrWrapper<RwLock<PrecomputeNode2>>>,
}

impl Acc {
    /// Creates a fresh, unconstrained accumulator (all tokens/terminals allowed).
    pub(crate) fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            llm_tokens_intersection: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            terminals_intersection: HybridL2Bitset::all(),
            needs_push_down: false,
            trie2_nodes: BTreeSet::new(),
        }
    }

    /// Returns true if this Acc acts as a neutral element for merging.
    /// That is, it contributes no constraints and carries no trie2 nodes.
    /// This is used to detect safe early-return cases in GSS merges.
    pub(crate) fn is_merge_neutral(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.llm_tokens_intersection == HybridBitset::max_ones()
            && self.terminals_union == HybridL2Bitset::all()
            && self.terminals_intersection == HybridL2Bitset::all()
            && self.trie2_nodes.is_empty()
    }

    /// Creates an accumulator with specific local constraints for a root node.
    #[allow(dead_code)] pub(crate) fn new_with_local_constraints(llm_tokens: HybridBitset, terminals: HybridL2Bitset) -> Self {
        Self {
            llm_tokens_union: llm_tokens.clone(),
            llm_tokens_intersection: llm_tokens,
            terminals_union: terminals.clone(),
            terminals_intersection: terminals,
            needs_push_down: false,
            trie2_nodes: BTreeSet::new(),
        }
    }

    pub(crate) fn narrow(from: &Self, to: &Self) -> Self {
        Acc {
            llm_tokens_union: &from.llm_tokens_union & &to.llm_tokens_union, // Correct
            llm_tokens_intersection: &from.llm_tokens_intersection & &to.llm_tokens_intersection,
            terminals_union: &from.terminals_union & &to.terminals_union, // Correct
            terminals_intersection: &from.terminals_intersection & &to.terminals_intersection,
            needs_push_down: false,
            // Keep trie2 nodes conservative: intersect sets to avoid leaking edges upward.
            trie2_nodes: to.trie2_nodes.clone(),
        }
        // Acc {
        //     llm_tokens_union: timeit!(&from.llm_tokens_union & &to.llm_tokens_union),
        //     llm_tokens_intersection: timeit!(&from.llm_tokens_union & &to.llm_tokens_intersection),
        //     terminals_union: timeit!(&from.terminals_union & &to.terminals_union),
        //     terminals_intersection: timeit!(&from.terminals_union & &to.terminals_intersection),
        // }
    }

    pub(crate) fn merge(lhs: &Self, rhs: &Self) -> Self {
        Acc {
            llm_tokens_union: &lhs.llm_tokens_union | &rhs.llm_tokens_union,
            llm_tokens_intersection: &lhs.llm_tokens_intersection & &rhs.llm_tokens_intersection,
            terminals_union: &lhs.terminals_union | &rhs.terminals_union,
            terminals_intersection: &lhs.terminals_intersection & &rhs.terminals_intersection,
            needs_push_down: false,
            trie2_nodes: &lhs.trie2_nodes | &rhs.trie2_nodes,
        }
    }

    // --- Accessors for final computed sets ---
    pub(crate) fn union_llm_tokens(&self) -> HybridBitset { self.llm_tokens_union.clone() }
    #[allow(dead_code)] pub(crate) fn intersection_terminals(&self) -> HybridL2Bitset { self.terminals_intersection.clone() }
}


// --- GSS Node & Core Implementation ---

/// A node in the Graph-Structured Stack (GSS).
#[derive(Debug, Clone)]
pub(crate) struct GSSNode {
    acc: Arc<Acc>,
    predecessors: NodeMap,
    hash_key_cache: u64,
    max_depth: MaxDepth,
}

/// A read-only view into a single path segment of the GSS, from a parent to a predecessor.
#[derive(Clone, Copy)]
pub(crate) struct GSSPeek<'a> {
    parent_arc: &'a Arc<GSSNode>,
    edge_value: &'a ParseStateEdgeContent,
    predecessor_node: &'a Arc<GSSNode>,
}

/// Represents the result of a `pop` operation, containing a map of resulting nodes
/// and the accumulated constraints (`Acc`) for each path leading to them.
#[derive(Debug, Clone, Default)]
pub(crate) struct GSSPopper {
    /// A map where the key is a node, and the value is the accumulated `Acc` for all paths leading to it.
    /// and the value is the accumulated `Acc` for that path.
    paths: BTreeMap<Arc<GSSNode>, Arc<Acc>>,
    /// Tracks how far below the bottom of the stack we've popped.
    /// Key is the number of extra pops beyond reaching the bottom (0 means exactly at bottom),
    /// and the value is the combined Acc for all paths that resulted in that depth.
    /// Multiple contributions to the same depth are merged via Acc::merge.
    below_bottom: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>>,
}

/// An item yielded by iterating over a `GSSPopper`, representing a single resulting path.
#[derive(Clone, Copy)]
pub(crate) struct GSSPopperItem<'a> {
    node: &'a Arc<GSSNode>,
    path_acc: &'a Arc<Acc>,
}

#[derive(Clone, Copy)]
pub(crate) struct GSSPopperItemPeek<'a> {
    path_acc: &'a Arc<Acc>,
    parent_arc: &'a Arc<GSSNode>,
    edge_value: &'a ParseStateEdgeContent,
    predecessor_node: &'a Arc<GSSNode>,
}

impl GSSPopper {
    pub(crate) fn new_from_node(node: Arc<GSSNode>, acc: Arc<Acc>) -> Self {
        let mut popper = Self {
            paths: BTreeMap::new(),
            below_bottom: BTreeMap::new(),
        };
        if node.is_root() {
            // At bottom with no last-edge yet: record an empty edge map at depth 0.
            // We intentionally do not associate an Acc because there is no “last edge” to key it under.
            popper.below_bottom.entry(0).or_insert_with(BTreeMap::new);
        } else {
            popper.paths.insert(node, acc);
        }
        popper
    }

    /// Returns an iterator over the items in the popper.
    pub(crate) fn iter(&self) -> impl Iterator<Item = GSSPopperItem<'_>> {
        self.paths.iter().map(|(node, acc)| GSSPopperItem {
            node,
            path_acc: acc,
        })
    }

    pub(crate) fn below_bottom(&self) -> &BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>> {
        &self.below_bottom
    }

    pub(crate) fn num_predecessors(&self) -> usize {
        self.paths.len()
    }

    pub(crate) fn popn(&mut self, n: usize) {
        for _ in 0..n {
            // Shift existing "below bottom" entries down by 1, since we're popping one more time.
            let mut new_below: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>> = BTreeMap::new();
            for (k, by_edge) in std::mem::take(&mut self.below_bottom) {
                new_below.insert(k + 1, by_edge);
            }

            let mut new_paths: BTreeMap<Arc<GSSNode>, Arc<Acc>> = BTreeMap::new();
            for (parent, path_acc) in std::mem::take(&mut self.paths) {
                let new_path_acc = Arc::new(Acc::narrow(&path_acc, &parent.acc));
                for (edge_val, preds_by_depth) in parent.predecessors.iter() {
                    for pred_vec in preds_by_depth.values() {
                        for child in pred_vec {
                            if child.is_root() {
                                // Reached the bottom on this pop. Do not keep root in paths.
                                // Register as "at bottom" under the last edge, and merge accs per edge.
                                let combined = Arc::new(Acc::narrow(&new_path_acc, &child.acc));
                                let by_edge = new_below.entry(1).or_insert_with(BTreeMap::new);
                                if let Some(existing) = by_edge.get_mut(&edge_val.clone()) {
                                    let merged = Arc::new(Acc::merge(existing, &combined));
                                    *existing = merged;
                                } else {
                                    by_edge.insert(edge_val.clone(), combined);
                                }
                            } else {
                                if let Some(existing_acc) = new_paths.get_mut(child) {
                                    *existing_acc = Arc::new(Acc::merge(existing_acc, &new_path_acc));
                                } else {
                                    new_paths.insert(child.clone(), new_path_acc.clone());
                                }
                            }
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
    /// Returns the combined `Acc` of the path and the destination node.
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Acc {
        Acc::narrow(&self.path_acc, &self.node.acc)
    }

    /// Returns a new `GSSNode` representing the destination node, but with its `Acc`
    /// resolved against the path's `Acc`.
    #[allow(dead_code)] pub(crate) fn resolved_node(&self) -> Arc<GSSNode> {
        let resolved_acc = self.resolved_acc();
        if *self.node.acc == resolved_acc {
            return self.node.clone();
        }
        Arc::new(GSSNode::new_with_map(Arc::new(resolved_acc), self.node.predecessors.clone()))
    }

    /// Pushes a new state onto the resolved node from this popper item.
    #[allow(dead_code)] pub(crate) fn push(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.resolved_node().as_ref().push(edge_value)
    }

    pub(crate) fn peek_iter(&self) -> impl Iterator<Item = GSSPopperItemPeek<'_>> {
        self.node.predecessors.iter().flat_map(move |(edge_val, preds_by_depth)| {
            preds_by_depth.values().flat_map(move |pred_vec| {
                pred_vec.iter().map(move |pred_arc| {
                    GSSPopperItemPeek {
                        path_acc: &self.path_acc,
                        parent_arc: self.node,
                        edge_value: edge_val,
                        predecessor_node: pred_arc,
                    }
                })
            })
        })
    }
}

impl<'a> GSSPopperItemPeek<'a> {
    pub(crate) fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    /// Returns the combined `Acc` of the path and the predecessor node.
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Acc {
        Acc::narrow(&Acc::narrow(self.path_acc, &self.parent_arc.acc), &self.predecessor_node.acc)
    }

    /// Returns a new `GSSNode` representing the predecessor, but with its `Acc`
    /// resolved against the path's `Acc`.
    #[allow(dead_code)] pub(crate) fn resolved_predecessor_node(&self) -> Arc<GSSNode> {
        let resolved_acc = self.resolved_acc();
        if *self.predecessor_node.acc == resolved_acc {
            return self.predecessor_node.clone();
        }
        Arc::new(GSSNode::new_with_map(Arc::new(resolved_acc), self.predecessor_node.predecessors.clone()))
    }

    /// Pushes a new state onto the resolved predecessor.
    #[allow(dead_code)] pub(crate) fn push_on_predecessor(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let mut resolved_acc = self.resolved_acc();
        resolved_acc.trie2_nodes.clear();
        GSSNode::new_with_single_predecessor(self.predecessor_node.clone(), edge_value, resolved_acc)
    }

    pub(crate) fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.isolated_parent().as_ref().push(edge_value)
    }
    #[allow(dead_code)] pub(crate) fn popn(&self, len: usize) -> GSSPopper {
        let isolated_parent = self.isolated_parent();
        let mut popper = GSSPopper::new_from_node(isolated_parent, Arc::new(Acc::new_fresh()));
        popper.popn(len);
        popper
    }

    pub(crate) fn isolated_parent(&self) -> Arc<GSSNode> {
        let new_acc = Acc::narrow(&Acc::narrow(self.path_acc, &self.parent_arc.acc), &self.predecessor_node.acc);

        if self.parent_arc.num_predecessors() == 1 && *self.parent_arc.acc == new_acc {
            return self.parent_arc.clone();
        }

        Arc::new(GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            new_acc
        ))
    }
}

// Helper functions for GSSNode construction
fn compute_max_depth(predecessors: &NodeMap) -> MaxDepth {
    predecessors
        .values()
        .flat_map(|preds_by_depth| preds_by_depth.values())
        .flat_map(|pred_vec| pred_vec.iter())
        .map(|pred_arc| pred_arc.max_depth() + 1)
        .max()
        .unwrap_or(0)
}

fn compute_hash_key(predecessors: &NodeMap, acc: &Acc) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    acc.llm_tokens_union.hash(&mut hasher);
    acc.llm_tokens_intersection.hash(&mut hasher);
    acc.terminals_union.hash(&mut hasher);
    acc.terminals_intersection.hash(&mut hasher);
    for trie2_node in &acc.trie2_nodes {
        trie2_node.hash(&mut hasher);
    }
    for (edge_val, preds_by_depth) in predecessors {
        edge_val.hash(&mut hasher);
        for (dest_key, pred_vec) in preds_by_depth {
            dest_key.hash(&mut hasher);
            for pred_arc in pred_vec {
                pred_arc.hash_key_cache.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Processes a set of incoming predecessors, grouping them by depth and edge,
/// and merging nodes that share the same edge to create a canonical `NodeMap`.
// #[time_it]
fn process_predecessors(incoming: &NodeSet) -> NodeMap {
    let mut grouped: BTreeMap<(ParseStateEdgeContent, DestKey), Vec<Arc<GSSNode>>> =
        BTreeMap::new();

    for (pred_arc, edge_val) in incoming {
        grouped
            .entry((edge_val.clone(), pred_arc.dest_key()))
            .or_default()
            .push(pred_arc.clone());
    }

    let mut result: NodeMap = BTreeMap::new();
    for ((edge_val, dest_key), pred_arcs) in grouped {
        if pred_arcs.is_empty() {
            continue;
        }

        let mut iter = pred_arcs.into_iter();
        let first = iter.next().unwrap();

        let final_node = if iter.len() == 0 {
            first
        } else {
            let mut merged_node = (*first).clone();
            for other_arc in iter {
                merged_node.merge_with_depth(1, &other_arc);
            }
            Arc::new(merged_node)
        };
        result
            .entry(edge_val)
            .or_default()
            .insert(dest_key, vec![final_node]);
    }
    result
}

/// Merges the `source` NodeMap into the `target` NodeMap.
// #[time_it]
fn merge_node_maps(target: &mut NodeMap, source: NodeMap, merge_depth: usize) {
    for (edge_val, source_preds_by_depth) in source {
        let target_preds_by_depth = target.entry(edge_val.clone()).or_default();
        for (dest_key, source_preds_vec) in source_preds_by_depth {
            let target_preds_vec = target_preds_by_depth.entry(dest_key).or_default();

            // TODO: ...I mean come on
            //  clean this up
            if merge_depth == 0 {
                if *target_preds_vec == source_preds_vec {
                    continue;
                } else if target_preds_vec.len() == 1 && source_preds_vec.len() > 1 {
                    if source_preds_vec.contains(&target_preds_vec[0]) {
                        *target_preds_vec = source_preds_vec;
                        continue;
                    } else {
                        target_preds_vec.extend(source_preds_vec);
                        continue;
                    }
                } else if target_preds_vec.len() > 1 && source_preds_vec.len() == 1 {
                    if target_preds_vec.contains(&source_preds_vec[0]) {
                        continue;
                    } else {
                        target_preds_vec.extend(source_preds_vec);
                        continue;
                    }
                } else {
                    target_preds_vec.extend(source_preds_vec);
                    continue;
                }
            }

            let mut nodes_to_merge = source_preds_vec;
            if !target_preds_vec.is_empty() {
                nodes_to_merge.extend(target_preds_vec.drain(..));
            }

            if nodes_to_merge.len() <= 1 {
                *target_preds_vec = nodes_to_merge;
            } else {
                let mut iter = nodes_to_merge.into_iter();
                let first = iter.next().unwrap();
                let mut merged = first.as_ref().clone();
                for other in iter {
                    merged._merge(&other, merge_depth - 1);
                }
                let mut merged = Arc::new(merged);
                if merged == first {
                    merged = first;
                }
                *target_preds_vec = vec![merged];
            }
        }
    }
}

// Basic node creation and manipulation
impl GSSNode {
    /// Creates a new GSS root node with the given local constraints.
    pub(crate) fn new(acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let arc_acc = Arc::new(acc);
        let hash_key_cache = compute_hash_key(&predecessors, &arc_acc);
        Self { acc: arc_acc, predecessors, hash_key_cache, max_depth: 0 }
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    fn new_with_map(mut acc: Arc<Acc>, predecessors: NodeMap) -> Self {
        if !predecessors.is_empty() && !acc.trie2_nodes.is_empty() {
            Arc::make_mut(&mut acc).trie2_nodes.clear();
        }
        let hash_key_cache = compute_hash_key(&predecessors, &acc);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, mut acc: Acc) -> Self {
        acc.trie2_nodes.clear();
        let mut predecessors_map = NodeMap::new();
        predecessors_map
            .entry(edge_value)
            .or_default()
            .insert(predecessor_arc.dest_key(), vec![predecessor_arc.clone()]);
        Self::new_with_map(Arc::new(acc), predecessors_map)
    }

    /// Helper to create a GSSNode with multiple predecessors, used by `push_many`.
    fn new_with_many_predecessors(predecessor_arc: Arc<GSSNode>, edge_values: Vec<ParseStateEdgeContent>, mut acc: Acc) -> Self {
        acc.trie2_nodes.clear();
        let mut predecessors_map = NodeMap::new();
        for edge_value in edge_values {
            predecessors_map
                .entry(edge_value)
                .or_default()
                .entry(predecessor_arc.dest_key())
                .or_default()
                .push(predecessor_arc.clone());
        }
        Self::new_with_map(Arc::new(acc), predecessors_map)
    }

    pub(crate) fn new_fresh() -> Self {
        Self::new(Acc::new_fresh())
    }

    pub(crate) fn acc(&self) -> &Arc<Acc> { &self.acc }

    pub(crate) fn predecessors(&self) -> &NodeMap { &self.predecessors }

    pub(crate) fn num_predecessors(&self) -> usize {
        self.predecessors
            .values()
            .map(|preds_by_depth| preds_by_depth.values().map(|v| v.len()).sum::<usize>())
            .sum()
    }
    pub(crate) fn max_depth(&self) -> MaxDepth { self.max_depth }
    // fn dest_key(&self) -> DestKey { self as *const GSSNode as usize }
    fn dest_key(&self) -> DestKey { self.max_depth() }

    /// Returns the set of LLM tokens allowed by *any* path ending at this node.
    pub(crate) fn allowed_llm_tokens(&self) -> LLMTokenBV { self.acc.llm_tokens_union.clone() }

    /// Returns a map of disallowed terminals for each tokenizer state.
    /// A terminal is disallowed if it's disallowed on *every* path to this node.
    pub(crate) fn disallowed_terminals(&self) -> TerminalInfo {
        self.acc.terminals_union.complement()
    }

    pub(crate) fn is_empty(&self) -> bool { self.predecessors.is_empty() }

    /// A path is alive if it allows at least one LLM token.
    pub(crate) fn is_alive(&self) -> bool { !self.allowed_llm_tokens().is_empty() }

    pub(crate) fn is_root(&self) -> bool {
        self.predecessors.is_empty()
    }

    pub(crate) fn merge_many_with_depth(merge_depth: usize, nodes: impl IntoIterator<Item = Arc<GSSNode>>) -> Arc<GSSNode> {
        timeit!(format!("GSSNode::merge_many_with_depth({})", merge_depth), {
        let mut iter = nodes.into_iter();
        if let Some(first) = iter.next() {
            let mut merged = first.as_ref().clone();
            for other in iter {
                merged.merge_with_depth(merge_depth, &other);
            }
            Arc::new(merged)
        } else {
            Arc::new(GSSNode::new_fresh())
        }
        })
    }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    pub(crate) fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        let mut acc = (*self.acc).clone();
        acc.trie2_nodes.clear();
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, acc)
    }

    pub(crate) fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Self {
        let mut acc = (*self.acc).clone();
        acc.trie2_nodes.clear();
        Self::new_with_many_predecessors(Arc::new(self.clone()), edge_values, acc)
    }

    /// Performs a multi-level pop operation on this node.
    pub(crate) fn popn(&self, n: usize) -> GSSPopper {
        let mut popper = GSSPopper::new_from_node(Arc::new(self.clone()), Arc::new(Acc::new_fresh()));
        popper.popn(n);
        popper
    }

    /// Merges another `GSSNode` into this one. This is a union of possibilities.
    // #[time_it]
    #[allow(dead_code)] pub(crate) fn merge(&mut self, other: &Self) {
        self._merge(other, 1);
    }

    // #[time_it]
    pub(crate) fn merge_with_depth(&mut self, merge_depth: usize, other: &Self) {
        self._merge(other, merge_depth);
    }

    // #[time_it]
    fn _merge(&mut self, other: &Self, merge_depth: usize) {
        if self == other { return; }

        // Only treat a leaf as ignorable if its Acc is fully neutral (no constraints, no trie2 nodes).
        if other.predecessors.is_empty() && other.acc.is_merge_neutral() {
            return;
        }
        if self.predecessors.is_empty() && self.acc.is_merge_neutral() {
            *self = other.clone();
            return;
        }

        let mut self_predecessors = self.get_pushed_down_predecessors();
        let other_predecessors = other.get_pushed_down_predecessors();

        merge_node_maps(&mut self_predecessors, other_predecessors, merge_depth);

        let final_predecessors = if merge_depth > 0 {
            // After merging, unify structurally identical predecessors to increase sharing.
            // This is important for preventing the GSS from bloating with redundant nodes
            // when merging branches that have common substructures.
            let mut canonical_map: BTreeMap<GSSNode, Arc<GSSNode>> = BTreeMap::new();
            let mut unified_predecessors = BTreeMap::new();

            for (edge_val, preds_by_depth) in self_predecessors {
                let mut unified_preds_by_depth = BTreeMap::new();
                for (depth, pred_vec) in preds_by_depth {
                    let mut unified_pred_vec = Vec::new();
                    for pred_arc in pred_vec {
                        // Find or create a canonical Arc for the node's value.
                        let canonical_arc = canonical_map.entry((*pred_arc).clone()).or_insert_with(|| pred_arc.clone()).clone();
                        unified_pred_vec.push(canonical_arc);
                    }
                    // Remove duplicate Arcs.
                    unified_pred_vec.sort_by_key(|a| Arc::as_ptr(a) as usize);
                    unified_pred_vec.dedup_by_key(|a| Arc::as_ptr(a));
                    unified_preds_by_depth.insert(depth, unified_pred_vec);
                }
                unified_predecessors.insert(edge_val, unified_preds_by_depth);
            }
            unified_predecessors
        } else {
            self_predecessors
        };

        let mut merged_acc_val = Acc::merge(&self.acc, &other.acc);
        merged_acc_val.needs_push_down = false;
        let merged_acc = Arc::new(merged_acc_val);

        *self = GSSNode::new_with_map(merged_acc, final_predecessors);
    }

    fn get_pushed_down_predecessors(&self) -> NodeMap {
        if !self.acc.needs_push_down {
            return self.predecessors.clone();
        }

        let mut predecessors = self.predecessors.clone();
        for preds_by_depth in predecessors.values_mut() {
            for pred_vec in preds_by_depth.values_mut() {
                for pred_arc in pred_vec.iter_mut() {
                    Self::narrow_with_acc(pred_arc, &self.acc);
                }
            }
        }
        predecessors
    }

    fn narrow_with_acc(node: &mut Arc<GSSNode>, acc: &Acc) {
        let mut new_acc = Acc::narrow(acc, &node.acc);
        // let acc_changed = new_acc.llm_tokens_union != node.acc.llm_tokens_union ||
        //                   new_acc.llm_tokens_intersection != node.acc.llm_tokens_intersection ||
        //                   new_acc.terminals_union != node.acc.terminals_union ||
        //                   new_acc.terminals_intersection != node.acc.terminals_intersection ||
        //                   new_acc.trie2_nodes != node.acc.trie2_nodes;
        let acc_changed = *node.acc != new_acc;

        if !acc_changed {
            return;
        }
        // The acc of this node has changed, so it needs to be pushed down to its own children later.
        new_acc.needs_push_down = true;
        let new_node = GSSNode::new_with_map(Arc::new(new_acc), node.predecessors.clone());
        *node = Arc::new(new_node);
    }

    #[allow(dead_code)] pub(crate) fn merged(mut self, other: Self, merge_depth: usize) -> Self {
        self.merge_with_depth(merge_depth, &other);
        self
    }

    #[allow(dead_code)] pub(crate) fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let mut acc = (*self.acc).clone();
        acc.trie2_nodes.clear();
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, acc)
    }

    /// Returns an iterator over all direct predecessor paths (`GSSPeek`s).
    pub(crate) fn peek_iter(parent_arc: &Arc<GSSNode>) -> impl Iterator<Item = GSSPeek<'_>> {
        parent_arc.predecessors.iter().flat_map(move |(edge_val, preds_by_depth)| {
            preds_by_depth.values().flat_map(move |pred_vec| {
                pred_vec.iter().map(move |pred_arc| GSSPeek {
                    parent_arc,
                    edge_value: edge_val,
                    predecessor_node: pred_arc,
                })
            })
        })
    }
}


impl<'a> GSSPeek<'a> {
    pub(crate) fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    #[allow(dead_code)] pub(crate) fn predecessor_node(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    /// Returns the combined `Acc` of the parent and the predecessor.
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Acc {
        Acc::narrow(&self.parent_arc.acc, &self.predecessor_node.acc)
    }

    /// Returns the resolved union of LLM tokens, without computing other parts of `Acc`.
    pub(crate) fn resolved_llm_tokens_union(&self) -> LLMTokenBV {
        &self.parent_arc.acc.llm_tokens_union & &self.predecessor_node.acc.llm_tokens_union
    }

    /// Returns a new `GSSNode` representing the predecessor, but with its `Acc`
    /// resolved against the parent's `Acc`.
    #[allow(dead_code)] pub(crate) fn resolved_predecessor_node(&self) -> Arc<GSSNode> {
        let resolved_acc = self.resolved_acc();
        if *self.predecessor_node.acc == resolved_acc {
            return self.predecessor_node.clone();
        }
        Arc::new(GSSNode::new_with_map(Arc::new(resolved_acc), self.predecessor_node.predecessors.clone()))
    }

    /// Pushes a new state onto the resolved predecessor.
    #[allow(dead_code)] pub(crate) fn push_on_predecessor(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let mut resolved_acc = self.resolved_acc();
        resolved_acc.trie2_nodes.clear();
        GSSNode::new_with_single_predecessor(self.predecessor_node.clone(), edge_value, resolved_acc)
    }

    pub(crate) fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.isolated_parent().as_ref().push(edge_value)
    }
    pub(crate) fn popn(&self, len: usize) -> GSSPopper {
        let isolated_parent = self.isolated_parent();
        let mut popper = GSSPopper::new_from_node(isolated_parent, Arc::new(Acc::new_fresh()));
        popper.popn(len);
        popper
    }

    /// Creates a new `GSSNode` that represents only the path segment of this peek.
    /// The new node has the parent's `Acc` and a single predecessor (the one from this peek).
    pub(crate) fn isolated_parent(&self) -> Arc<GSSNode> {
        let new_acc = Acc::narrow(&self.parent_arc.acc, &self.predecessor_node.acc);

        if self.parent_arc.num_predecessors() == 1 && *self.parent_arc.acc == new_acc {
            return self.parent_arc.clone();
        }

        Arc::new(GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            new_acc
        ))
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

pub(crate) type PruneAndTransformRecursiveMemo = HashMap<*const GSSNode, Option<Arc<GSSNode>>>;

fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    closure: &impl Fn(&GSSNode) -> Option<(Acc, bool)>,
    memo: &mut PruneAndTransformRecursiveMemo,
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
        Some((mut new_local_acc, continue_recursion)) => {
            // Case 1: Do not recurse; only possibly adjust local acc.
            if !continue_recursion {
                let acc_changed = *node_arc.acc != new_local_acc;
                return if acc_changed {
                    // Mark for push down so children get narrowed later.
                    new_local_acc.needs_push_down = true;
                    let transformed_node =
                        GSSNode::new_with_map(Arc::new(new_local_acc), node_arc.predecessors.clone());
                    let result_arc = Arc::new(transformed_node);
                    memo.insert(node_ptr, Some(result_arc.clone()));
                    Some(result_arc)
                } else {
                    // No changes at all.
                    memo.insert(node_ptr, Some(node_arc.clone()));
                    Some(node_arc.clone())
                }
            }

            // Case 2: Recurse into children. Preserve the original predecessor structure.
            // Important: Do not canonicalize/merge here; a no-op transform should be a no-op.
            let mut any_child_changed = false;
            let mut had_any_pred = false;

            // Build a new NodeMap mirroring the existing shape.
            let mut new_predecessors_map: NodeMap = BTreeMap::new();

            for (edge_val, preds_by_depth) in &node_arc.predecessors {
                let mut new_preds_by_depth: BTreeMap<DestKey, Vec<Arc<GSSNode>>> = BTreeMap::new();
                for (dest_key, pred_vec) in preds_by_depth {
                    let mut new_vec: Vec<Arc<GSSNode>> = Vec::new();
                    for pred_arc in pred_vec {
                        had_any_pred = true;
                        match prune_and_transform_recursive(pred_arc, closure, memo) {
                            Some(new_pred_arc) => {
                                if !Arc::ptr_eq(&new_pred_arc, pred_arc) {
                                    any_child_changed = true;
                                }
                                new_vec.push(new_pred_arc);
                            }
                            None => {
                                // Child was pruned.
                                any_child_changed = true;
                            }
                        }
                    }
                    if !new_vec.is_empty() {
                        new_preds_by_depth.insert(*dest_key, new_vec);
                    }
                }
                if !new_preds_by_depth.is_empty() {
                    new_predecessors_map.insert(edge_val.clone(), new_preds_by_depth);
                }
            }

            // If all predecessors were pruned away, and this node originally had any predecessors,
            // prune this node too (consistent with previous behavior).
            let new_has_any_pred = new_predecessors_map
                .values()
                .any(|by_depth| by_depth.values().any(|v| !v.is_empty()));
            if !new_has_any_pred && had_any_pred {
                memo.insert(node_ptr, None);
                return None;
            }

            // Decide whether anything changed at this node.
            let acc_changed = *node_arc.acc != new_local_acc;
            if !acc_changed && !any_child_changed {
                // No changes at all: return the original arc to preserve identity and structure.
                memo.insert(node_ptr, Some(node_arc.clone()));
                return Some(node_arc.clone());
            }

            // Some change happened: rebuild the node with the same predecessor shape (no merging).
            // Note: When recursing, we don't set needs_push_down even if acc changed; that flag is
            // used for the non-recursing path to defer narrowing into children.
            let transformed_node =
                GSSNode::new_with_map(Arc::new(new_local_acc), new_predecessors_map);
            let result_arc = Arc::new(transformed_node);
            memo.insert(node_ptr, Some(result_arc.clone()));
            Some(result_arc)
        }
    }
}

pub(crate) fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        new_acc.llm_tokens_union &= allowed_tokens;
        new_acc.llm_tokens_intersection &= allowed_tokens;

        // Prune if the union of possibilities is empty.
        if new_acc.llm_tokens_union.is_empty() {
            None
        } else {
            Some((new_acc, false))
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
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

pub(crate) fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        let continue_recursion = new_acc.llm_tokens_intersection != HybridBitset::max_ones();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        new_acc.llm_tokens_intersection = HybridBitset::max_ones();
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        unreachable!();
    }
}

pub(crate) fn reset_terminals(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        let continue_recursion = new_acc.terminals_intersection != HybridL2Bitset::all();
        new_acc.terminals_union = HybridL2Bitset::all();
        new_acc.terminals_intersection = HybridL2Bitset::all();
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        unreachable!();
    }
}

pub(crate) fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    memo: &mut PruneAndTransformRecursiveMemo,
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
        unreachable!();
    }
}

pub(crate) fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut PruneAndTransformRecursiveMemo,
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
        Some(((*node.acc).clone(), false))
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

pub(crate) fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();

        let map_one = |terminals: &HybridL2Bitset| -> (HybridL2Bitset, bool) {
            let mut new_terminals_btreemap = BTreeMap::new();

            for (old_state_id, new_state_id) in map {
                let bv_source = terminals.get_l2_bitset(old_state_id.0).unwrap();
                new_terminals_btreemap.entry(*new_state_id)
                    .and_modify(|bv| *bv |= bv_source)
                    .or_insert_with(|| bv_source.clone());
            }

            let mut new_terminals_l2_bitset = HybridL2Bitset::all();
            for (state_id, bv) in new_terminals_btreemap {
                new_terminals_l2_bitset.insert_l2_bitset(state_id.0, bv);
            }

            let changed = new_terminals_l2_bitset != *terminals;
            (new_terminals_l2_bitset, changed)
        };

        let (new_terminals_union, changed_union) = map_one(&new_acc.terminals_union);
        let (new_terminals_intersection, changed_intersection) = map_one(&new_acc.terminals_intersection);

        new_acc.terminals_union = new_terminals_union;
        new_acc.terminals_intersection = new_terminals_intersection;

        Some((new_acc, changed_union || changed_intersection))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

pub(crate) fn merge_trie2_nodes_if_needed(
    root_arc: &mut Arc<GSSNode>,
    merge_threshold: usize,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let closure = |node: &GSSNode| -> Option<(Acc, bool)> {
        let mut new_acc = (*node.acc).clone();
        if new_acc.trie2_nodes.len() > merge_threshold {
            let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> = BTreeMap::new();
            let edge_key = (0, None);

            // Shared fallback destination (for sources without any eligible existing child).
            let fallback_dest = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));

            for existing_trie2_node in &new_acc.trie2_nodes {
                let source_arc = existing_trie2_node.as_arc().clone();
                let tokens_to_push = {
                    let g = source_arc.read().expect("poison");
                    g.value.live_tokens.clone()
                };
                if tokens_to_push.is_empty() {
                    continue;
                }

                // Build an iterator of all eligible strong children under edge_key
                let eligible_iter_builder = || {
                    let g = source_arc.read().expect("poison");
                    let mut v = Vec::new();
                    if let Some(dest_map) = g.children().get(&edge_key) {
                        for (node_ptr, _ev) in dest_map.iter() {
                            if !node_ptr.is_strong() { continue; }
                            if let Some(dest_arc) = node_ptr.upgrade() {
                                let dl = dest_arc.read().expect("poison").value.live_tokens.clone();
                                if (&dl & &tokens_to_push).is_empty() && !dest_arc.read().unwrap().value.end {
                                    v.push(dest_arc.clone());
                                }
                            }
                        }
                    }
                    v.into_iter()
                };

                let mut inserter = EdgeInserter::new(
                    source_arc.clone(),
                    edge_key,
                    tokens_to_push.clone(),
                    |e, n| *e |= n,
                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                    |ev, t| *ev &= &t.live_tokens,
                ).try_destinations_iter_with(eligible_iter_builder);

                inserter = inserter.try_destination_auto(fallback_dest.clone());

                let final_dest_arc = inserter.clone_into_option().expect("merge_trie2_nodes_if_needed: insert failed");
                let final_dest_wr = ArcPtrWrapper::new(final_dest_arc.clone());

                dest_agg.entry(final_dest_wr.clone()).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
            }

            for (dst_wr, added) in &dest_agg {
                let mut dg = dst_wr.as_arc().write().expect("poison");
                dg.value.live_tokens |= added.clone();
            }

            new_acc.trie2_nodes = dest_agg.keys().cloned().collect();
        }
        if new_acc == *node.acc {
            new_acc = (*node.acc).clone();
        }
        Some((new_acc, true))
    };
    // let (s, _) = print_gss_forest(&[root_arc.clone()], &BiBTreeMap::new(), &GSSPrintConfig::default());
    // println!("Before merge_trie2_nodes_if_needed:\n{s}");
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
        // let (s, _) = print_gss_forest(&[root_arc.clone()], &BiBTreeMap::new(), &GSSPrintConfig::default());
        // println!("After merge_trie2_nodes_if_needed:\n{}", s);
    } else {
        // This shouldn't happen as we never return None from closure
        unreachable!();
    }
}

impl GSSNode {
    /// Fuses predecessor nodes that share the same edge value, even if they are at different depths.
    #[time_it]
    pub(crate) fn fuse_predecessors(&mut self, levels: usize) {
        if levels == 0 {
            return;
        }
        let temp_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let fused_arc = fuse_predecessors_recursive(&temp_arc, levels, &mut memo);
        *self = Arc::try_unwrap(fused_arc).unwrap_or_else(|arc| (*arc).clone());
    }
}

pub(crate) fn fuse_predecessors_recursive(
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
    for (edge_val, preds_by_depth) in &node_arc.predecessors {
        for pred_vec in preds_by_depth.values() {
            for pred_arc in pred_vec {
                let fused_pred_arc = fuse_predecessors_recursive(pred_arc, levels - 1, memo);
                recursively_fused_predecessors.push((edge_val.clone(), fused_pred_arc));
            }
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
                merged_node.merge_with_depth(1, &other_arc);
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

pub(crate) fn deep_clone_gss_with_trie2_map(
    root: &Arc<GSSNode>,
    trie2_map: &HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
) -> Arc<GSSNode> {
    fn clone_one(
        node: &Arc<GSSNode>,
        trie2_map: &HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
        memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
    ) -> Arc<GSSNode> {
        let ptr = Arc::as_ptr(node);
        if let Some(cached) = memo.get(&ptr) {
            return cached.clone();
        }

        // 1) Clone predecessors recursively
        let mut new_preds: BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<Arc<GSSNode>>>> = BTreeMap::new();
        for (edge_val, preds_by_depth) in &node.predecessors {
            let mut new_by_depth = BTreeMap::new();
            for (dest_key, pred_vec) in preds_by_depth {
                let mut new_vec = Vec::with_capacity(pred_vec.len());
                for pred in pred_vec {
                    new_vec.push(clone_one(pred, trie2_map, memo));
                }
                new_by_depth.insert(*dest_key, new_vec);
            }
            new_preds.insert(edge_val.clone(), new_by_depth);
        }

        // 2) Clone Acc, remapping trie2_nodes via trie2_map
        let mut new_acc = (*node.acc).clone();
        if !new_acc.trie2_nodes.is_empty() {
            let mut new_set = BTreeSet::new();
            for old_wr in &new_acc.trie2_nodes {
                let old_arc = old_wr.as_arc().clone();
                let old_ptr = Arc::as_ptr(&old_arc);
                if let Some(new_arc) = trie2_map.get(&old_ptr) {
                    new_set.insert(ArcPtrWrapper::new(new_arc.clone()));
                } else {
                    // Should not happen if the cloned Trie2 was complete
                    // Fallback to original (not ideal, but prevents panic)
                    new_set.insert(ArcPtrWrapper::new(old_arc));
                }
            }
            new_acc.trie2_nodes = new_set;
        }

        // 3) Build a new node; new_with_map recomputes hash and depth
        let new_node = GSSNode::new_with_map(Arc::new(new_acc), new_preds);
        let out = Arc::new(new_node);
        memo.insert(ptr, out.clone());
        out
    }

    let mut memo: HashMap<*const GSSNode, Arc<GSSNode>> = HashMap::new();
    clone_one(root, trie2_map, &mut memo)
}

// --- Analysis and Debugging ---
#[derive(Debug, Clone, Eq, Hash)]
#[allow(dead_code)] pub(crate) struct RootItem<'a> {
    node: &'a GSSNode,
    path_acc: Arc<Acc>,
}

impl<'a> PartialEq for RootItem<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.path_acc == other.path_acc
    }
}

impl<'a> PartialOrd for RootItem<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for RootItem<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.cmp(other.node)
            .then_with(|| self.path_acc.cmp(&other.path_acc))
    }
}

impl<'a> RootItem<'a> {
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Arc<Acc> {
        Arc::new(Acc::narrow(&self.path_acc, &self.node.acc))
    }
}

/// Traverses the GSS graph from the given nodes and returns all unique root nodes (nodes with no predecessors).
pub(crate) fn get_roots<'a>(nodes: impl IntoIterator<Item = &'a GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    // We carry the "last edge" used to reach the next node; when we finally hit a root,
    // that last edge is the key used for the result map.
    let mut queue: BTreeMap<
        MaxDepth,
        BTreeMap<(*const GSSNode, Option<ParseStateEdgeContent>), Arc<Acc>>
    > = BTreeMap::new();

    let mut results: BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> = BTreeMap::new();

    for node in nodes {
        let node_ptr = node as *const GSSNode;
        let depth = node.max_depth();
        queue
            .entry(depth)
            .or_default()
            .entry((node_ptr, None))
            .or_insert_with(|| Arc::new(Acc::new_fresh()));
    }

    while let Some((_depth, nodes_at_depth)) = queue.pop_last() {
        for ((node_ptr, last_edge_opt), path_acc) in nodes_at_depth {
            let current_node = unsafe { &*node_ptr };

            if current_node.is_root() {
                if let Some(edge) = last_edge_opt {
                    let final_acc = Arc::new(Acc::narrow(&path_acc, &current_node.acc));
                    results
                        .entry(edge)
                        .or_default()
                        .insert(final_acc);
                }
            } else {
                let new_path_acc_base = Arc::new(Acc::narrow(&path_acc, &current_node.acc));
                for (edge_val, preds_by_depth) in current_node.predecessors.iter() {
                    for pred_arc in preds_by_depth.values().flatten() {
                        let pred_ptr = pred_arc.as_ref() as *const GSSNode;
                        let pred_depth = pred_arc.max_depth();
                        queue
                            .entry(pred_depth)
                            .or_default()
                            .entry((pred_ptr, Some(edge_val.clone())))
                            .and_modify(|e| *e = Arc::new(Acc::merge(e, &new_path_acc_base)))
                            .or_insert_with(|| new_path_acc_base.clone());
                    }
                }
            }
        }
    }

    results
}

impl GSSNode {
    #[allow(dead_code)] pub(crate) fn reset_llm_tokens(&mut self) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        reset_llm_tokens(&mut node_arc, &mut memo);
        *self = Arc::try_unwrap(node_arc).unwrap_or_else(|arc| (*arc).clone());
    }

    #[allow(dead_code)] pub(crate) fn get_roots(&self) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
        get_roots(std::iter::once(self))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GSSStats {
    pub(crate) num_roots: usize,
    pub(crate) num_root_predecessors: usize,
    pub(crate) num_unique_root_predecessor_keys: usize,
    pub(crate) total_edges: usize,
    pub(crate) unique_nodes: usize,
    pub(crate) structurally_unique_nodes: usize,
    pub(crate) structural_redundancy: f64,
    pub(crate) num_redundant_nodes: usize,
    pub(crate) max_depth: usize,
    pub(crate) average_depth: f64,
    pub(crate) merge_points: usize,
    pub(crate) max_predecessors_with_values: usize,
    pub(crate) average_predecessors_with_values: f64,
}

/// Gathers statistics about the structure and complexity of a GSS forest.
#[time_it]
pub(crate) fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut total_depth = 0u64;
    let mut total_preds = 0u64;

    let mut root_predecessor_dest_keys = HashSet::new();
    for root_node in roots {
        stats.num_root_predecessors += root_node.num_predecessors();
        for edge_value in root_node.predecessors().keys() {
            root_predecessor_dest_keys.insert(edge_value.clone());
        }

        let node_ptr = *root_node as *const GSSNode;
        if visited.insert(node_ptr) {
            queue.push_back((*root_node, 0));
        }
    }
    stats.num_unique_root_predecessor_keys = root_predecessor_dest_keys.len();

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

        let unique_pred_arcs: HashSet<_> = node
            .predecessors
            .values()
            .flat_map(|v| v.values())
            .flat_map(|v| v.iter())
            .map(Arc::as_ptr)
            .collect();
        if unique_pred_arcs.len() > 1 {
            stats.merge_points += 1;
        }

        for pred_arc in node
            .predecessors
            .values()
            .flat_map(|v| v.values())
            .flat_map(|v| v.iter()) {
            queue.push_back((pred_arc.as_ref(), depth + 1));
        }
    }

    stats.total_edges = total_preds as usize;

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }

    // Calculate structural uniqueness
    let mut structural_memo = HashMap::new();
    let mut structural_cache: BTreeMap<BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<usize>>>, usize> = BTreeMap::new();
    for root_node in roots {
        get_structural_id(root_node, &mut structural_memo, &mut structural_cache);
    }
    stats.structurally_unique_nodes = structural_cache.len();
    if stats.unique_nodes > 0 {
        stats.structural_redundancy = 1.0 - (stats.structurally_unique_nodes as f64 / stats.unique_nodes as f64);
    }
    stats.num_redundant_nodes = stats.unique_nodes - stats.structurally_unique_nodes;
    stats
}

/// Helper for `gather_gss_stats` to compute a unique ID for a node's structure.
fn get_structural_id(
    node: &GSSNode,
    memo: &mut HashMap<*const GSSNode, usize>,
    structural_cache: &mut BTreeMap<BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<usize>>>, usize>,
) -> usize {
    let node_ptr = node as *const GSSNode;
    if let Some(id) = memo.get(&node_ptr) {
        return *id;
    }

    let mut pred_structural_ids = BTreeMap::new();
    for (edge_val, preds_by_depth) in &node.predecessors {
        let mut ids_by_depth = BTreeMap::new();
        for (dest_key, pred_vec) in preds_by_depth {
            let mut ids_vec = Vec::new();
            for pred_arc in pred_vec {
                let pred_id = get_structural_id(pred_arc.as_ref(), memo, structural_cache);
                ids_vec.push(pred_id);
            }
            ids_vec.sort(); // For canonical representation
            ids_by_depth.insert(*dest_key, ids_vec);
        }
        pred_structural_ids.insert(edge_val.clone(), ids_by_depth);
    }

    let next_id = structural_cache.len();
    let id = *structural_cache.entry(pred_structural_ids).or_insert(next_id);
    memo.insert(node_ptr, id);
    id
}

/// Finds the longest path from any leaf to the given root node.
/// Returns `None` if the node has no predecessors.
pub(crate) fn find_longest_path(root_node: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
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
        for (edge_val, preds_by_depth) in node_arc.predecessors.iter() {
            for pred_vec in preds_by_depth.values() {
                for pred_arc in pred_vec {
                    let mut path_from_pred = find_longest_recursive(pred_arc, memo);
                    // The path ends with the edge leading to `node_arc` and `node_arc` itself.
                    path_from_pred.push((edge_val.clone(), node_arc.clone()));
                    if path_from_pred.len() > longest_path.len() {
                        longest_path = path_from_pred;
                    }
                }
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
#[allow(dead_code)] pub(crate) fn sample_path(roots: &[&GSSNode], seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    if roots.is_empty() {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let root_index = rng.random_range(0..roots.len());
    let mut current_node_arc = Arc::new(roots[root_index].clone());

    let mut path = Vec::new();

    loop {
        if current_node_arc.is_empty() {
            break;
        }

        let predecessors: Vec<_> = GSSNode::peek_iter(&current_node_arc).collect();
        if predecessors.is_empty() {
            break;
        }

        let chosen_index = rng.random_range(0..predecessors.len());
        let chosen_peek = &predecessors[chosen_index];

        path.push(chosen_peek.edge_value().clone());

        current_node_arc = chosen_peek.predecessor_node.clone();
    }

    Some(path)
}

pub(crate) struct GSSPrintConfig<'a> {
    pub(crate) labels: Option<&'a [String]>,
    pub(crate) max_edges: usize,
    pub(crate) original_internal_bimap: Option<&'a BiBTreeMap<usize, usize>>,
    pub(crate) llm_token_map: Option<&'a BiBTreeMap<Vec<u8>, LLMTokenID>>,
    pub(crate) verbose: bool,
}

impl<'a> Default for GSSPrintConfig<'a> {
    fn default() -> Self {
        Self {
            labels: None,
            max_edges: usize::MAX,
            original_internal_bimap: None,
            llm_token_map: None,
            verbose: false,
        }
    }
}

/// Pretty-prints a GSS forest for debugging.
pub(crate) fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    config: &GSSPrintConfig,
) -> (String, Vec<StateID>) {
    // if !GSS_LOGGING_ENABLED {
    //     return "".to_string();
    // }
    // Recursive helper to print predecessors.
    fn print_predecessors_recursive(
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
            return Ok(()); // Avoid re-printing children of shared nodes.
        }
        visited_nodes.insert(node_ptr);

        let predecessors: Vec<_> = node_arc.predecessors()
            .iter()
            .flat_map(|(edge_val, preds_by_depth)| {
                preds_by_depth.values().flat_map(move |pred_vec| {
                    pred_vec.iter().map(move |pred_arc| (edge_val, pred_arc))
                })
            })
            .collect();

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

            // Collect state ID for explanation
            if seen_state_ids.insert(edge_val.state_id) {
                state_ids_in_order.push(edge_val.state_id);
            }

            let acc_child = format_acc(
                pred_arc.as_ref(),
                terminal_map,
                config.original_internal_bimap,
                config.llm_token_map,
            );
            // Print cleaner edge line:
            // - No "StateID(...)" wrapper; just the numeric/state value
            // - No depth
            if config.verbose {
                if acc_child.is_empty() {
                    writeln!(
                        output,
                        "{}{} edge {} -> Node {} (ptr: {:p}, hash: {:x})",
                        prefix, connector, edge_val.state_id.0, pred_id, pred_ptr, pred_arc.hash_key_cache,
                    )?;
                } else {
                    writeln!(
                        output,
                        "{}{} edge {} -> Node {} (ptr: {:p}, hash: {:x}) {}",
                        prefix, connector, edge_val.state_id.0, pred_id, pred_ptr, pred_arc.hash_key_cache, acc_child,
                    )?;
                }
            } else if acc_child.is_empty() {
                writeln!(
                    output,
                    "{}{} edge {} -> Node {}",
                    prefix, connector, edge_val.state_id.0, pred_id,
                )?;
            } else {
                writeln!(
                    output,
                    "{}{} edge {} -> Node {} {}",
                    prefix, connector, edge_val.state_id.0, pred_id, acc_child,
                )?;
            }
            *node_count += 1;

            print_predecessors_recursive(
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

        let acc_str = format_acc(
            root_arc.as_ref(),
            terminal_map,
            config.original_internal_bimap,
            config.llm_token_map,
        );
        let root_label = config.labels.map_or_else(|| format!("Root {}", i), |l| l[i].clone());

        // Cleaner root line:
        // - No depth
        if config.verbose {
            if acc_str.is_empty() {
                writeln!(
                    &mut out_str,
                    "{}: Node {} (ptr: {:p}, hash: {:x})",
                    root_label, root_id, root_ptr, root_arc.hash_key_cache
                ).unwrap();
            } else {
                writeln!(
                    &mut out_str,
                    "{}: Node {} (ptr: {:p}, hash: {:x}) {}",
                    root_label, root_id, root_ptr, root_arc.hash_key_cache, acc_str
                ).unwrap();
            }
        } else if acc_str.is_empty() {
            writeln!(&mut out_str, "{}: Node {}", root_label, root_id).unwrap();
        } else {
            writeln!(&mut out_str, "{}: Node {} {}", root_label, root_id, acc_str).unwrap();
        }
        count += 1;

        let _ = print_predecessors_recursive(
            root_arc, &mut node_ids, &mut visited_nodes, "  ", &mut count,
            &mut out_str, terminal_map, &mut state_ids_in_order, &mut seen_state_ids, config,
        );
    }

    (out_str, state_ids_in_order)
}

/// Formats an accumulator for concise display in the GSS printout.
pub(crate) fn format_acc(
    node: &GSSNode,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) -> String {
    // Avoid unused-parameter warnings; hooks left for future improvements.
    let _ = (original_internal_bimap, llm_token_map);

    // Summarize a bitset with a small sample; omit entirely if it's "all".
    let summarize_llm = |bv: &HybridBitset, label: &str| -> Option<String> {
        if *bv == HybridBitset::max_ones() {
            return None; // Omit when unconstrained
        }
        if bv.is_empty() {
            return Some(format!("{}=∅", label));
        }
        let total = bv.len();
        const MAX_SHOW: usize = 8;
        let sample: Vec<String> = bv.iter().take(MAX_SHOW).map(|id| id.to_string()).collect();
        if total > MAX_SHOW {
            Some(format!("{}({}): [{} …]", label, total, sample.join(", ")))
        } else {
            Some(format!("{}({}): [{}]", label, total, sample.join(", ")))
        }
    };

    // Summarize disallowed terminals (complement of allowed). Omit if none are disallowed.
    let summarize_disallowed_terminals = |allowed_terminals: &HybridL2Bitset, label: &str| -> Option<String> {
        let mut any_disallowed = false;
        let mut parts = Vec::new();
        const MAX_RANGES_TO_SHOW: usize = 3;
        for (range, allowed_bv) in allowed_terminals.range_values() {
            let disallowed_bv = HybridBitset::max_ones() - allowed_bv;
            if disallowed_bv.is_empty() {
                continue;
            }
            any_disallowed = true;
            if parts.len() >= MAX_RANGES_TO_SHOW {
                break;
            }
            let range_str = if range.start() == range.end() {
                format!("{}", range.start())
            } else {
                format!("{}..={}", range.start(), range.end())
            };

            if disallowed_bv == HybridBitset::max_ones() {
                parts.push(format!("state(s) {}: all", range_str));
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
                parts.push(format!("state(s) {} ({}): [{}, …]", range_str, num_disallowed, names_str));
            } else {
                parts.push(format!("state(s) {}: [{}]", range_str, names_str));
            }
        }
        if !any_disallowed {
            None
        } else if parts.is_empty() {
            Some(format!("Disallowed {}(…)", label))
        } else {
            Some(format!("Disallowed {}({})", label, parts.join("; ")))
        }
    };

    // LLM summaries (omit when "all")
    let union_llm_opt = summarize_llm(&node.acc.llm_tokens_union, "LLM(U)");
    let intersection_llm_opt = summarize_llm(&node.acc.llm_tokens_intersection, "LLM(I)");

    // Terminal summaries: show only when something is actually disallowed
    let union_terminals_opt = summarize_disallowed_terminals(&node.acc.terminals_union, "Term(U)");
    let intersection_terminals_opt = summarize_disallowed_terminals(&node.acc.terminals_intersection, "Term(I)");

    // Trie2 nodes: omit when empty; otherwise show a compact summary
    let trie2_nodes_str = {
        const MAX_PTRS_TO_SHOW: usize = 5;
        let n = node.acc.trie2_nodes.len();
        if n == 0 {
            None
        } else if n <= MAX_PTRS_TO_SHOW {
            let ptrs: Vec<String> = node
                .acc
                .trie2_nodes
                .iter()
                .map(|wrapper| format!("{:p}", { let ptr = Arc::as_ptr(wrapper.as_arc()) as *const PrecomputeNode2; ptr}))
                .collect();
            Some(format!("Trie2(n={}, [{}])", n, ptrs.join(", ")))
        } else {
            let ptrs_sample: Vec<String> = node
                .acc
                .trie2_nodes
                .iter()
                .take(MAX_PTRS_TO_SHOW)
                .map(|wrapper| format!("{:p}", Arc::as_ptr(wrapper.as_arc())))
                .collect();
            let remaining = n - MAX_PTRS_TO_SHOW;
            Some(format!("Trie2(n={}, first {}: {}, …; +{} more)", n, MAX_PTRS_TO_SHOW, ptrs_sample.join(", "), remaining))
        }
    };

    // Collect only the non-empty components.
    let mut parts: Vec<String> = Vec::new();
    if let Some(s) = union_llm_opt { parts.push(s); }
    if let Some(s) = intersection_llm_opt { parts.push(s); }
    if let Some(s) = union_terminals_opt { parts.push(s); }
    if let Some(s) = intersection_terminals_opt { parts.push(s); }
    if let Some(s) = trie2_nodes_str { parts.push(s); }

    if parts.is_empty() {
        String::new()
    } else {
        format!("[{}]", parts.join(", "))
    }
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
        assert_eq!(node.acc().llm_tokens_union, acc.llm_tokens_union);
        assert!(node.predecessors().is_empty());
        assert_eq!(node.max_depth(), 0);
    }

    #[test]
    fn test_gss_push() {
        let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
        let pushed = root.push(mock_edge(10));

        assert_eq!(pushed.max_depth(), 1);

        // The new logic for `push` is to inherit the predecessor's acc, as the local acc is fresh.
        assert_eq!(*pushed.acc(), *root.acc());
    }

    #[test]
    fn test_gss_pop() {
        let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
        let pushed = Arc::new(root.push(mock_edge(10))); // Now inherits root's acc.

        // Pop 1 level from `pushed`. The initial_acc is "fresh" (all allowed), so it doesn't constrain the path.
        let pop_result = pushed.popn(1);
        // We should not keep root nodes in paths.
        assert_eq!(pop_result.paths.len(), 0);
        assert_eq!(pop_result.below_bottom.len(), 1);
        // We reached the bottom exactly (depth 0).
        let combined_acc_map = pop_result.below_bottom.get(&1).unwrap(); // Depth 1 entry holds last-edge grouped map
        // The map should contain the edge 10 leading to the root
        let combined_acc = combined_acc_map.get(&mock_edge(10)).unwrap();

        // `pushed.acc` (same as `root.acc`) allows all but 1.
        // The intersection should allow all but 1.
        let mut disallowed = HybridBitset::zeros();
        disallowed.insert(1);
        let expected_allowed = HybridBitset::max_ones() - disallowed;
        assert_eq!(combined_acc.llm_tokens_union, expected_allowed);
    }

    #[test]
    fn test_gss_merge() {
        let n0 = Arc::new(GSSNode::new(empty_acc()));
        let n1 = Arc::new(n0.push(mock_edge(0)));
        let n2 = Arc::new(n0.push(mock_edge(0)));

        let mut merged = (*n1).clone();
        merged.merge_with_depth(1, &n2);

        assert_eq!(merged.acc().llm_tokens_union, HybridBitset::max_ones());

        assert_eq!(merged.num_predecessors(), 1);
    }

    #[test]
    fn test_popper_new_from_root_and_shift() {
        let root = Arc::new(GSSNode::new(mock_acc(1)));
        let mut popper = GSSPopper::new_from_node(root.clone(), Arc::new(Acc::new_fresh()));
        // Should not store roots in paths.
        assert!(popper.paths.is_empty());
        // Now below_bottom has an empty map at depth 0
        assert_eq!(popper.below_bottom.len(), 1);
        assert!(popper.below_bottom.get(&0).unwrap().is_empty());
        // Pop once; it shifts down since no edges are present
        popper.popn(1);
        assert!(popper.below_bottom.get(&0).is_none());
        assert!(popper.below_bottom.get(&1).unwrap().is_empty());
        // Pop two more steps; now it should be at 3 (still empty maps)
        popper.popn(2);
        assert!(popper.below_bottom.get(&1).is_none());
        assert!(popper.below_bottom.get(&3).unwrap().is_empty());
    }

    #[test]
    fn test_popper_below_bottom_shifts_from_non_root() {
        let root = Arc::new(GSSNode::new(mock_acc(1)));
        let pushed = Arc::new(root.push(mock_edge(10)));
        let mut popper = pushed.popn(1); // Reaches bottom via edge 10

        assert!(popper.paths.is_empty());
        let by_edge_1 = popper.below_bottom.get(&1).expect("depth 1 entry missing");
        assert_eq!(by_edge_1.len(), 1);
        let acc0 = by_edge_1.get(&mock_edge(10)).expect("edge 10 missing at depth 1").clone();
        // Shift down by 2 more pops.
        popper.popn(2);
        assert!(popper.below_bottom.get(&1).is_none());
        let by_edge_3 = popper.below_bottom.get(&3).expect("depth 3 entry missing");
        let acc2 = by_edge_3.get(&mock_edge(10)).expect("edge 10 missing at depth 3").clone();
        assert_eq!(*acc0, *acc2);
    }

    #[test]
    fn test_popper_merges_below_bottom_accs() {
        // Build a node that has two root predecessors with different disallowed tokens.
        let root1 = Arc::new(GSSNode::new(mock_acc(1))); // disallow token 1
        let root2 = Arc::new(GSSNode::new(mock_acc(2))); // disallow token 2
        let mut preds = NodeSet::new();
        preds.insert((root1.clone(), mock_edge(100)));
        preds.insert((root2.clone(), mock_edge(200)));
        let preds_map = process_predecessors(&preds);
        let parent = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), preds_map));

        let (s, _) = print_gss_forest(
            &[parent.clone()],
            &BiBTreeMap::new(),
            &GSSPrintConfig::default(),
        );
        println!("GSS Forest:\n{}", s);

        let popper = parent.popn(1);
        assert!(popper.paths.is_empty());
        let by_edge = popper.below_bottom.get(&1).expect("depth 1 entry missing");
        assert_eq!(by_edge.len(), 2);

        // Edge 100 (root1)
        {
            let acc_below_100 = by_edge.get(&mock_edge(100)).expect("edge 100 missing at depth 1");
            // Union should disallow token 1.
            let mut disallowed = HybridBitset::zeros();
            disallowed.insert(1);
            let expected_intersection = HybridBitset::max_ones() - disallowed;
            assert_eq!(acc_below_100.llm_tokens_union, expected_intersection);
        }

        // Edge 200 (root2)
        {
            let acc_below_200 = by_edge.get(&mock_edge(200)).expect("edge 200 missing at depth 1");
            let mut disallowed = HybridBitset::zeros();
            disallowed.insert(2);
            let expected_intersection = HybridBitset::max_ones() - disallowed;
            assert_eq!(acc_below_200.llm_tokens_union, expected_intersection);
        }
    }

    #[test]
    fn test_gss_fuse_predecessors() {
        let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
        let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));
        let b = Arc::new(leaf1.push(mock_edge(1)));
        let c_tmp = Arc::new(leaf2.push(mock_edge(2)));
        let c_tmp2 = Arc::new(c_tmp.push(mock_edge(3)));
        let c = Arc::new(c_tmp2.push(mock_edge(4)));

        assert_eq!(b.max_depth(), 1);
        assert_eq!(c.max_depth(), 3);

        let mut preds_map = NodeMap::new();
        preds_map.entry(mock_edge(100)).or_default().insert(b.dest_key(), vec![b.clone()]);
        preds_map.entry(mock_edge(100)).or_default().insert(c.dest_key(), vec![c.clone()]);

        let mut root = GSSNode::new_with_map(Arc::new(empty_acc()), preds_map);
        assert_eq!(root.num_predecessors(), 2);

        root.fuse_predecessors(1);

        assert_eq!(root.num_predecessors(), 1);
        let fused_pred_arc = root
            .predecessors()
            .values()
            .next()
            .unwrap()
            .values()
            .next()
            .unwrap()[0]
            .clone();

        assert_eq!(fused_pred_arc.acc().llm_tokens_union, HybridBitset::max_ones());
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
        // let path2 = sample_path(&[&root], 1).unwrap();

        assert_eq!(path1.len(), 3);
        assert_eq!(path1[0], mock_edge(10));
        assert_eq!(path1[1], mock_edge(20));
        assert!(path1[2] == mock_edge(30) || path1[2] == mock_edge(40));

        let path1_again = sample_path(&[&root], 0).unwrap();
        assert_eq!(path1, path1_again);
    }

    #[test]
    fn test_merge_maintains_structural_sharing() {
        // This test reproduces a scenario where merging two GSSs with shared
        // sub-structure leads to duplicated nodes instead of sharing them.

        // 1. Create a common leaf node.
        let leaf = Arc::new(GSSNode::new(empty_acc()));

        // 2. Create two intermediate nodes. They are structurally identical
        // (same acc, same single predecessor 'leaf' with the same edge),
        // but they are different objects in memory.
        let intermediate1 = Arc::new(GSSNode::new_with_single_predecessor(
            leaf.clone(),
            mock_edge(960),
            empty_acc(),
        ));
        let intermediate2 = Arc::new(GSSNode::new_with_single_predecessor(
            leaf.clone(),
            mock_edge(960),
            empty_acc(),
        ));

        // Sanity check: they are equal in value, but different pointers.
        assert_eq!(*intermediate1, *intermediate2);
        assert_ne!(Arc::as_ptr(&intermediate1), Arc::as_ptr(&intermediate2));

        // 3. Create two GSS root nodes, each with one of the intermediate nodes as a predecessor.
        // The edges leading to the intermediate nodes are different.
        let mut gss1 = GSSNode::new_with_single_predecessor(
            intermediate1,
            mock_edge(161),
            empty_acc(),
        );
        let gss2 = GSSNode::new_with_single_predecessor(
            intermediate2,
            mock_edge(0),
            empty_acc(),
        );

        // 4. Merge gss2 into gss1.
        gss1.merge_with_depth(1, &gss2);

        // 5. Analyze the merged GSS.
        let stats = gather_gss_stats(&[&gss1]);

        // After the merge, the root `gss1` has two predecessors. Because `intermediate1` and
        // `intermediate2` are structurally identical, a correct merge operation should unify
        // them into a single shared node. This means the number of unique nodes should equal
        // the number of structurally unique nodes.
        assert_eq!(
            stats.unique_nodes, stats.structurally_unique_nodes,
            "Merge created redundant structures. Stats: {:?}",
            stats
        );
        assert_eq!(stats.unique_nodes, 3, "Expected 3 unique nodes after merge, but found {}. Stats: {:?}", stats.unique_nodes, stats);
    }

#[test]
    fn test_get_roots() {
        let acc1 = Arc::new(mock_acc(1));
        let leaf1 = Arc::new(GSSNode::new((*acc1).clone()));
        let acc2 = Arc::new(mock_acc(2));
        let leaf2 = Arc::new(GSSNode::new((*acc2).clone()));

        let acc_b = Arc::new(mock_acc(3));
        let b = Arc::new(GSSNode::new_with_single_predecessor(leaf1.clone(), mock_edge(1), (*acc_b).clone()));

        let acc_c = Arc::new(mock_acc(4));
        let c = Arc::new(GSSNode::new_with_single_predecessor(leaf2.clone(), mock_edge(2), (*acc_c).clone()));

        let mut preds_map = NodeMap::new();
        preds_map.entry(mock_edge(10)).or_default().insert(b.dest_key(), vec![b.clone()]);
        preds_map.entry(mock_edge(20)).or_default().insert(c.dest_key(), vec![c.clone()]);
        let acc_root = Arc::new(mock_acc(5));
        let root = GSSNode::new_with_map(acc_root.clone(), preds_map);

        // Test from root
        let roots_map = get_roots(std::iter::once(&root));
        let mut expected = BTreeMap::new();
        let path_acc1 = Arc::new(Acc::narrow(&Acc::narrow(&acc_root, &acc_b), &acc1));
        expected.insert(mock_edge(1), BTreeSet::from([path_acc1.clone()]));
        let path_acc2 = Arc::new(Acc::narrow(&Acc::narrow(&acc_root, &acc_c), &acc2));
        expected.insert(mock_edge(2), BTreeSet::from([path_acc2.clone()]));
        assert_eq!(roots_map, expected);

        // Test from multiple sources. The path from `b` as a root will "win" for leaf1's path_acc
        // because its initial `fresh` acc has a wider union of possibilities.
        let roots_multi = get_roots(vec![&root, b.as_ref()]);
        let from_root_edge1 = Arc::new(Acc::narrow(&Acc::narrow(&acc_root, &acc_b), &acc1));
        let from_b_edge1 = Arc::new(Acc::narrow(&acc_b, &acc1));
        let expected_merged_edge1 = Arc::new(Acc::merge(&from_root_edge1, &from_b_edge1));
        assert_eq!(*roots_multi.get(&mock_edge(1)).expect("edge 1 missing"), BTreeSet::from([expected_merged_edge1]));
        assert_eq!(*roots_multi.get(&mock_edge(2)).expect("edge 2 missing"), BTreeSet::from([path_acc2.clone()]));

        // Test from leaves -> no last edge, so contributes nothing
        let roots_leaves = get_roots(vec![leaf1.as_ref(), leaf2.as_ref()]);
        assert!(roots_leaves.is_empty());

        // Test empty
        assert!(get_roots(Vec::<&GSSNode>::new()).is_empty());
    }

    #[test]
    fn test_prune_and_transform_noop_does_not_merge_distinct_predecessors() {
        // This test checks for a bug where prune_and_transform_recursive with a no-op
        // closure would still modify the GSS by merging structurally distinct predecessor
        // nodes that happen to share the same edge value and depth.

        // 1. Create two distinct leaf nodes.
        let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
        let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));

        // 2. Create two intermediate nodes that are structurally different because they
        // have different predecessors.
        let intermediate1 = Arc::new(leaf1.push(mock_edge(10)));
        let intermediate2 = Arc::new(leaf2.push(mock_edge(10)));
        assert_ne!(*intermediate1, *intermediate2, "Intermediates should be structurally different");
        assert_eq!(intermediate1.max_depth(), 1);
        assert_eq!(intermediate2.max_depth(), 1);

        // 3. Manually construct a root node that has both intermediates as predecessors
        // under the same edge value and at the same depth. This structure is key to
        // reproducing the bug. The `Vec` in the NodeMap contains multiple distinct nodes.
        let mut root_preds = NodeMap::new();
        root_preds
            .entry(mock_edge(100))
            .or_default()
            .insert(1, vec![intermediate1.clone(), intermediate2.clone()]);

        let root = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), root_preds));
        assert_eq!(root.num_predecessors(), 2);

        // 4. Run prune_and_transform_recursive with a no-op closure.
        // This should not change the structure of the GSS at all.
        let mut memo = HashMap::new();
        let new_root_opt = prune_and_transform_recursive(
            &root,
            &|node| Some((node.acc().as_ref().clone(), true)), // No-op: keep node, recurse
            &mut memo,
        );

        // 5. Assert that the structure is unchanged.
        let new_root = new_root_opt.expect("Root should not be pruned");

        // Check full equality for good measure. This is the most important check.
        // With the bug, this fails because the new_root will have its predecessors merged.
        assert_eq!(*root, *new_root, "The GSS structure should be identical after a no-op transform");
        assert_eq!(new_root.num_predecessors(), 2, "Should still have 2 predecessors");
    }

    #[test]
    fn test_merge_preserves_trie2_nodes() {
        // This test reproduces a bug where merging GSS nodes would cause
        // trie2_nodes from leaf predecessors to be lost due to incorrect
        // constraint propagation (narrowing).

        // --- GSS 1 Setup ---
        let trie2_node1 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let trie2_node2 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let trie2_node3 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));

        let mut acc_l1 = empty_acc();
        acc_l1.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node1.clone()));
        let l1 = Arc::new(GSSNode::new(acc_l1));

        let mut acc_l2 = empty_acc();
        acc_l2.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node2.clone()));
        let l2 = Arc::new(GSSNode::new(acc_l2));

        let mut acc_l3 = empty_acc();
        acc_l3.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node3.clone()));
        let l3 = Arc::new(GSSNode::new(acc_l3));

        let mut gss1_preds = NodeMap::new();
        gss1_preds.entry(mock_edge(0)).or_default().insert(l1.dest_key(), vec![l1.clone()]);
        gss1_preds.entry(mock_edge(1)).or_default().insert(l2.dest_key(), vec![l2.clone()]);
        gss1_preds.entry(mock_edge(2)).or_default().insert(l3.dest_key(), vec![l3.clone()]);

        let mut gss1 = GSSNode::new_with_map(Arc::new(mock_acc(0)), gss1_preds); // mock_acc(0) restricts token 0
        // Arc::make_mut(&mut gss1.acc()).needs_push_down = true; // Simulate state that triggers narrowing

        // --- GSS 2 Setup ---
        let mut acc_l4 = empty_acc();
        acc_l4.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node1.clone())); // Shared trie2_node
        let l4 = Arc::new(GSSNode::new(acc_l4));
        let i1 = Arc::new(l4.push(mock_edge(0)));
        let gss2 = i1.push(mock_edge(1));

        // --- Merge ---
        gss1.merge_with_depth(1, &gss2);

        // --- Assertions ---
        // Traverse the merged GSS and collect all trie2_nodes from all leaf nodes.
        let mut q = VecDeque::new();
        q.push_back(Arc::new(gss1));
        let mut visited = HashSet::new();
        let mut final_leaf_trie2_nodes = BTreeSet::new();

        while let Some(node) = q.pop_front() {
            if !visited.insert(Arc::as_ptr(&node)) { continue; }
            if node.is_root() { final_leaf_trie2_nodes.extend(node.acc().trie2_nodes.clone()); }
            for p in node.predecessors().values().flat_map(|m| m.values()).flatten() { q.push_back(p.clone()); }
        }

        assert!(final_leaf_trie2_nodes.contains(&ArcPtrWrapper::new(trie2_node1)), "trie2_node1 missing");
        assert!(final_leaf_trie2_nodes.contains(&ArcPtrWrapper::new(trie2_node2)), "trie2_node2 missing");
        assert!(final_leaf_trie2_nodes.contains(&ArcPtrWrapper::new(trie2_node3)), "trie2_node3 missing");
        assert_eq!(final_leaf_trie2_nodes.len(), 3, "Should have 3 unique trie2 nodes in the leaves");
    }

    #[test]
    fn test_merge_does_not_incorrectly_collapse_branches() {
        // This test reproduces a bug where merging two GSSs with a common edge value
        // but different sub-structures would incorrectly collapse the distinct sub-structures.

        // --- Shared Nodes ---
        let trie2_node1 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let mut acc1 = empty_acc();
        acc1.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node1.clone()));
        let leaf1 = Arc::new(GSSNode::new(acc1)); // This is "Node 2" with trie ...6f0

        let trie2_node2 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let mut acc2 = empty_acc();
        acc2.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node2.clone()));
        let leaf2 = Arc::new(GSSNode::new(acc2)); // This is "Node 2" with trie ...560

        // --- GSS A ---
        // Root -> (edge 1) -> leaf1
        let gss_a = GSSNode::new_with_single_predecessor(
            leaf1.clone(),
            mock_edge(1),
            empty_acc(),
        );

        // --- GSS B ---
        // intermediate -> (edge 0) -> leaf2
        let intermediate_b = Arc::new(GSSNode::new_with_single_predecessor(
            leaf2.clone(),
            mock_edge(0),
            empty_acc(),
        ));
        // Root -> (edge 1) -> leaf1
        //      -> (edge 1) -> intermediate
        let mut gss_b_preds = NodeMap::new();
        gss_b_preds.entry(mock_edge(1)).or_default().insert(leaf1.dest_key(), vec![leaf1.clone()]);
        gss_b_preds.entry(mock_edge(1)).or_default().insert(intermediate_b.dest_key(), vec![intermediate_b.clone()]);
        let gss_b = GSSNode::new_with_map(Arc::new(empty_acc()), gss_b_preds);

        // --- Merge ---
        let mut merged_gss = gss_a.clone();
        merged_gss.merge_with_depth(usize::MAX, &gss_b);

        // --- Assertions ---
        // The merged GSS should have two distinct predecessors under edge 1, because
        // they have different depths and structures. The incorrect behavior collapses them into one.
        assert_eq!(merged_gss.num_predecessors(), 2, "Merged GSS should have two predecessors");

        let preds_for_edge1 = merged_gss.predecessors().get(&mock_edge(1)).expect("Edge 1 should exist");
        assert_eq!(preds_for_edge1.len(), 2, "Edge 1 should have predecessors at two different depths");
    }

    #[test]
    fn test_merge_with_different_depth_predecessors() {
        // This test reproduces a bug where merging two GSSs with a common edge value
        // but different sub-structures would incorrectly collapse the distinct sub-structures.
        // GSS A: Root -> (edge 1) -> leaf_a
        // GSS B: Root -> (edge 1) -> intermediate_b -> (edge 0) -> leaf_b
        // Merged should have two predecessors from root via edge 1, at different depths.

        // --- GSS A setup ---
        let trie2_node_a = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let mut acc_a = empty_acc();
        acc_a.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node_a.clone()));
        let leaf_a = Arc::new(GSSNode::new(acc_a));

        let gss_a = GSSNode::new_with_single_predecessor(
            leaf_a.clone(),
            mock_edge(1),
            empty_acc(),
        );

        // --- GSS B setup ---
        let trie2_node_b = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let mut acc_b = empty_acc();
        acc_b.trie2_nodes.insert(ArcPtrWrapper::new(trie2_node_b.clone()));
        let leaf_b = Arc::new(GSSNode::new(acc_b));

        let intermediate_b = Arc::new(GSSNode::new_with_single_predecessor(
            leaf_b.clone(),
            mock_edge(0),
            empty_acc(),
        ));
        let gss_b = GSSNode::new_with_single_predecessor(intermediate_b.clone(), mock_edge(1), empty_acc());

        // --- Merge ---
        let mut merged_gss = gss_a.clone();
        merged_gss.merge_with_depth(usize::MAX, &gss_b);

        // --- Assertions ---
        // The merged GSS should have two distinct predecessors under edge 1, because
        // they have different depths and structures. The incorrect behavior collapses them into one.
        assert_eq!(merged_gss.num_predecessors(), 2, "Merged GSS should have two predecessors");

        let preds_for_edge1 = merged_gss.predecessors().get(&mock_edge(1)).expect("Edge 1 should exist");
        assert_eq!(preds_for_edge1.len(), 2, "Edge 1 should have predecessors at two different depths");
    }

    #[test]
    fn test_merge_unions_trie2_nodes_across_identical_towers() {
        // This test reproduces a bug where merging multiple identical towers (same edges and structure)
        // but with different trie2_nodes at the leaf results in the leaf keeping only one
        // of the trie2_nodes instead of the union of all of them.
        //
        // Structure for each tower:
        // Root -> (edge 2) -> ... -> Leaf [Trie2={unique}]
        //
        // After merging two such towers, the single leaf should contain the union of the two distinct
        // trie2 nodes.

        // --- Build two distinct trie2 nodes ---
        let t1 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
        let t2 = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));

        // Helper to build one tower given a leaf with a unique trie2 node.
        let build_tower_from_leaf = |leaf: Arc<GSSNode>| -> GSSNode {
            let n5 = Arc::new(GSSNode::new_with_single_predecessor(leaf, mock_edge(5), empty_acc()));
            let n1 = Arc::new(n5.push(mock_edge(1)));
            n1.push(mock_edge(2))
        };

        // --- Leaf 1 with trie2_node t1 ---
        let mut acc1 = empty_acc();
        acc1.trie2_nodes.insert(ArcPtrWrapper::new(t1.clone()));
        let leaf1 = Arc::new(GSSNode::new(acc1));
        let tower1 = build_tower_from_leaf(leaf1);

        // --- Leaf 2 with trie2_node t2 ---
        let mut acc2 = empty_acc();
        acc2.trie2_nodes.insert(ArcPtrWrapper::new(t2.clone()));
        let leaf2 = Arc::new(GSSNode::new(acc2));
        let tower2 = build_tower_from_leaf(leaf2);

        // --- Merge the two identical towers ---
        let mut merged = tower1.clone();
        merged.merge_with_depth(usize::MAX, &tower2);

        // --- Traverse to collect leaves and inspect trie2_nodes at the bottom ---
        let mut q = VecDeque::new();
        q.push_back(Arc::new(merged));
        let mut visited = HashSet::new();
        let mut leaves = Vec::new();
        while let Some(node) = q.pop_front() {
            if !visited.insert(Arc::as_ptr(&node)) { continue; }
            if node.is_root() { leaves.push(node.clone()); }
            for p in node.predecessors().values().flat_map(|m| m.values()).flatten() {
                q.push_back(p.clone());
            }
        }

        assert_eq!(leaves.len(), 1, "Merging identical towers should result in a single unified leaf node");
        let leaf = &leaves[0];
        let trie2_nodes = &leaf.acc().trie2_nodes;
        assert_eq!(trie2_nodes.len(), 2, "Unified leaf should contain the union of all trie2 nodes from merged towers");
        assert!(trie2_nodes.contains(&ArcPtrWrapper::new(t1)), "Unified leaf missing trie2 node 1");
        assert!(trie2_nodes.contains(&ArcPtrWrapper::new(t2)), "Unified leaf missing trie2 node 2");
    }
}
