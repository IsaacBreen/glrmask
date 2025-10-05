use crate::datastructures::arc_wrapper::ArcPtrWrapper;
use deterministic_hash::DeterministicHasher;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};

use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::parser::{GLRParserState, ParseStateEdgeContent};
use profiler_macro::{time_it, timeit};

pub use crate::datastructures::gss_analysis::*;
pub use crate::datastructures::gss_pruning::*;
pub use crate::datastructures::gss_simplification::*;
pub use crate::datastructures::gss_trie_utils::*;

use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use crate::constraint::{LLMTokenBV, PrecomputeNode3, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, TerminalBV, TerminalInfo, Trie3God, Trie3GodWrapper};
use crate::datastructures::{gss_analysis, gss_simplification};
use crate::datastructures::trie::{EdgeInserter, God};
use crate::tokenizer::TokenizerStateID;
use crate::types::TerminalID;
// --- Type Aliases ---

pub(crate) type MaxDepth = usize;
pub(crate) type DestKey = MaxDepth;
/// Maps an edge value to a map of destination keys (depths) to a list of predecessor nodes.
pub(crate) type NodeMap = BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<Arc<GSSNode>>>>;
/// A temporary set of predecessors used during node construction and simplification.
pub(crate) type NodeSet = ordered_hash_map::OrderedHashSet<(Arc<GSSNode>, ParseStateEdgeContent)>;

pub(crate)type StoredPrecomputeNodeIndex = PrecomputeNode3Index;
pub(crate)type StoredPrecomputeNode = PrecomputeNode3;
pub(crate)type StoredTrieGod = Trie3God;
pub(crate) type StoredTrieGodWrapper = Trie3GodWrapper;

/// Recursively traverses a GSS, and for each node (both internal and root) that has
/// `stored_trie_nodes`, it:
/// 1. Gets a new destination trie node from `destination_provider`.
/// In the simplified design, only root nodes carry Acc values. Internal nodes' Acc
/// is computed on demand by aggregating the Accs of all reachable roots.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub stored_trie_nodes: BTreeSet<StoredPrecomputeNodeIndex>,
}

impl Acc {
    /// Creates a fresh, unconstrained accumulator (all tokens/terminals allowed).
    pub(crate) fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    /// Creates a "dead" accumulator that allows no tokens.
    pub(crate) fn new_dead() -> Self {
        Self {
            llm_tokens_union: HybridBitset::zeros(),
            terminals_union: HybridL2Bitset::all(), // Doesn't matter, token check will fail first
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    /// Returns true if this Acc acts as a neutral element for merging.
    /// That is, it contributes no constraints and carries no stored_trie nodes.
    /// This is used to detect safe early-return cases in GSS merges.
    pub(crate) fn is_merge_neutral(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.terminals_union == HybridL2Bitset::all()
            && self.stored_trie_nodes.is_empty()
    }

    /// Returns true if this Acc is the default (unconstrained) value.
    pub(crate) fn is_default(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.terminals_union == HybridL2Bitset::all()
            && self.stored_trie_nodes.is_empty()
    }

    /// Creates an accumulator with specific local constraints for a root node.
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
            // For the simplified design, we do not propagate stored_trie changes through internal nodes.
            stored_trie_nodes: to.stored_trie_nodes.clone(),
        }
    }

    pub(crate) fn merge(lhs: &Self, rhs: &Self) -> Self {
        Acc {
            llm_tokens_union: &lhs.llm_tokens_union | &rhs.llm_tokens_union,
            terminals_union: &lhs.terminals_union | &rhs.terminals_union,
            stored_trie_nodes: &lhs.stored_trie_nodes | &rhs.stored_trie_nodes,
        }
    }

    // --- Accessors for final computed sets ---
    pub(crate) fn union_llm_tokens(&self) -> HybridBitset { self.llm_tokens_union.clone() }
    pub(crate) fn stored_trie_nodes(&self) -> &BTreeSet<StoredPrecomputeNodeIndex> { &self.stored_trie_nodes }
    pub(crate) fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<StoredPrecomputeNodeIndex> { &mut self.stored_trie_nodes }
}


// --- GSS Node & Core Implementation ---

/// A node in the Graph-Structured Stack (GSS).
/// Simplified design: only root nodes carry Acc; internal nodes carry only predecessors.
#[derive(Debug, Clone)]
pub enum GSSNode {
    Root(GSSRoot),
    Internal(GSSInternal),
}

#[derive(Debug, Clone)]
pub(crate) struct GSSRoot {
    acc: Arc<Acc>,
    hash_key_cache: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct GSSInternal {
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
pub struct GSSPopper {
    /// A map where the key is a node, and the value is the accumulated `Acc` for all paths leading to it.
    paths: BTreeMap<Arc<GSSNode>, Arc<Acc>>,
    /// Tracks how far below the bottom of the stack we've popped.
    /// Key is the number of extra pops beyond reaching the bottom (0 means exactly at bottom),
    /// and the value is the combined Acc for all paths that resulted in that depth.
    /// Multiple contributions to the same depth are merged via Acc::merge.
    pub below_bottom: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>>,
}

/// An item yielded by iterating over a `GSSPopper`, representing a single resulting path.
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
        if node.is_root() {
            // At bottom with no last-edge yet: record an empty edge map at depth 0.
            popper.below_bottom.entry(0).or_insert_with(BTreeMap::new);
        } else {
            // Initialize path_acc by applying the node's local Acc.
            let init = Arc::new(Acc::narrow(&acc, &node.local_acc()));
            popper.paths.insert(node, init);
        }
        popper
    }

    /// Returns an iterator over the items in the popper.
    pub fn iter(&self) -> impl Iterator<Item = GSSPopperItem<'_>> {
        self.paths.iter().map(|(node, acc)| GSSPopperItem {
            node,
            path_acc: acc,
        })
    }

    pub fn below_bottom(&self) -> &BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>> {
        &self.below_bottom
    }

    pub fn num_predecessors(&self) -> usize {
        self.paths.len()
    }

    pub fn popn(&mut self, n: usize) {
        for _ in 0..n {
            // Shift existing "below bottom" entries down by 1, since we're popping one more time.
            let mut new_below: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>> = BTreeMap::new();
            for (k, by_edge) in std::mem::take(&mut self.below_bottom) {
                new_below.insert(k + 1, by_edge);
            }

            let mut new_paths: BTreeMap<Arc<GSSNode>, Arc<Acc>> = BTreeMap::new();
            for (parent, path_acc) in std::mem::take(&mut self.paths) {
                for (edge_val, preds_by_depth) in parent.predecessors().iter() {
                    for pred_vec in preds_by_depth.values() {
                        for child in pred_vec {
                            // Accumulate child's local Acc on the path.
                            let next_path = Arc::new(Acc::narrow(&path_acc, &child.local_acc()));
                            match child.as_ref() {
                                GSSNode::Root(_) => {
                                    // Reached the bottom on this pop (via `edge_val`).
                                    let by_edge = new_below.entry(1).or_insert_with(BTreeMap::new);
                                    if let Some(existing) = by_edge.get_mut(&edge_val.clone()) {
                                        let merged = Arc::new(Acc::merge(existing, &next_path));
                                        *existing = merged;
                                    } else {
                                        by_edge.insert(edge_val.clone(), next_path);
                                    }
                                }
                                GSSNode::Internal(_) => {
                                    if let Some(existing_acc) = new_paths.get_mut(child) {
                                        *existing_acc = Arc::new(Acc::merge(existing_acc, &next_path));
                                    } else {
                                        new_paths.insert(child.clone(), next_path.clone());
                                    }
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
        // path_acc already includes narrowing through `node`'s local Acc when this item was produced.
        self.path_acc.as_ref().clone()
    }

    /// Returns a new `GSSNode` representing the destination node.
    /// In the simplified design, `Acc` is only at roots; this just returns the node.
    #[allow(dead_code)] pub(crate) fn resolved_node(&self) -> Arc<GSSNode> {
        self.node.clone()
    }

    /// Pushes a new state onto the resolved node from this popper item.
    #[allow(dead_code)] pub(crate) fn push(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.resolved_node().as_ref().push(edge_value)
    }

    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPopperItemPeek<'_>> {
        self.node.predecessors().iter().flat_map(move |(edge_val, preds_by_depth)| {
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
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    /// Returns the combined `Acc` of the path and the predecessor node.
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Acc {
        Acc::narrow(self.path_acc, &self.predecessor_node.local_acc())
    }

    /// Returns a new `GSSNode` representing the predecessor; in the simplified design, just clones.
    #[allow(dead_code)] pub(crate) fn resolved_predecessor_node(&self) -> Arc<GSSNode> {
        self.predecessor_node.clone()
    }

    /// Pushes a new state onto the resolved predecessor.
    #[allow(dead_code)] pub(crate) fn push_on_predecessor(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let resolved_acc = self.resolved_acc(); // becomes the local Acc on the new node
        GSSNode::new_with_single_predecessor(self.predecessor_node.clone(), edge_value, resolved_acc)
    }

    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.isolated_parent().as_ref().push(edge_value)
    }
    #[allow(dead_code)] pub(crate) fn popn(&self, len: usize) -> GSSPopper {
        let isolated_parent = self.isolated_parent();
        let mut popper = GSSPopper::new_from_node(isolated_parent, Arc::new(Acc::new_fresh()));
        popper.popn(len);
        popper
    }

    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        let narrowed_acc = Arc::new(Acc::narrow(self.path_acc, &self.parent_arc.local_acc()));

        // Optimization: if the parent already represents this single path and the path_acc
        // didn't add any constraints, we can return it directly.
        if self.parent_arc.num_predecessors() == 1 && narrowed_acc == self.parent_arc.local_acc() {
            return self.parent_arc.clone();
        }

        // Otherwise, create a new parent node representing only this path.
        let mut predecessors = NodeMap::new();
        predecessors
            .entry(self.edge_value.clone())
            .or_default()
            .insert(self.predecessor_node.dest_key(), vec![self.predecessor_node.clone()]);

        Arc::new(GSSNode::new_with_map(narrowed_acc, predecessors))
    }
}

// Static empty NodeMap for roots
fn empty_nodemap() -> &'static NodeMap {
    static EMPTY: OnceLock<NodeMap> = OnceLock::new();
    EMPTY.get_or_init(|| NodeMap::new())
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

fn compute_hash_key_internal_with_acc(acc: &Acc, predecessors: &NodeMap) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    // Include the local acc
    acc.llm_tokens_union.hash(&mut hasher);
    acc.terminals_union.hash(&mut hasher);
    for t2 in &acc.stored_trie_nodes { t2.hash(&mut hasher); }
    for (edge_val, preds_by_depth) in predecessors {
        edge_val.hash(&mut hasher);
        for (dest_key, pred_vec) in preds_by_depth {
            dest_key.hash(&mut hasher);
            for pred_arc in pred_vec {
                pred_arc.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn compute_hash_key_root(acc: &Acc) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    acc.llm_tokens_union.hash(&mut hasher);
    acc.terminals_union.hash(&mut hasher);
    for stored_trie_node in &acc.stored_trie_nodes {
        stored_trie_node.hash(&mut hasher);
    }
    hasher.finish()
}

/// Processes a set of incoming predecessors, grouping them by depth and edge,
/// and merging nodes that share the same edge to create a canonical `NodeMap`.
pub(crate) fn process_predecessors(incoming: &NodeSet) -> NodeMap {
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
fn merge_node_maps(target: &mut NodeMap, source: NodeMap, merge_depth: usize) {
    for (edge_val, source_preds_by_depth) in source {
        let target_preds_by_depth = target.entry(edge_val.clone()).or_default();
        for (dest_key, source_preds_vec) in source_preds_by_depth {
            let target_preds_vec = target_preds_by_depth.entry(dest_key).or_default();

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

impl GSSInternal {
    pub(crate) fn predecessors(&self) -> &NodeMap { &self.predecessors }
    pub(crate) fn max_depth(&self) -> MaxDepth { self.max_depth }
    pub(crate) fn acc(&self) -> &Arc<Acc> { &self.acc }
}

impl GSSRoot {
    pub(crate) fn acc(&self) -> &Arc<Acc> { &self.acc }
}

impl GSSNode {
    /// Creates a new GSS root node with the given local constraints.
    pub(crate) fn new(acc: Acc) -> Self {
        let arc_acc = Arc::new(acc);
        let hash_key_cache = compute_hash_key_root(&arc_acc);
        GSSNode::Root(GSSRoot { acc: arc_acc, hash_key_cache })
    }

    /// Private constructor for internal methods that build a node from a pre-computed map.
    /// Now: internal nodes also carry a local `Acc` (fresh by default).
    pub(crate) fn new_with_map(acc: Arc<Acc>, predecessors: NodeMap) -> Self {
        // An internal node must have predecessors. If the map is effectively empty, create a root node instead.
        // The provided `acc` becomes the local accumulator for this new root.
        if predecessors.values().all(|by_depth| by_depth.values().all(Vec::is_empty)) {
            return GSSNode::new((*acc).clone());
        }

        // Internal nodes must not carry stored_trie_nodes. They belong only to leaves.
        let final_acc = if acc.stored_trie_nodes.is_empty() {
            acc
        } else {
            let mut sanitized_acc = (*acc).clone();
            sanitized_acc.stored_trie_nodes.clear();
            Arc::new(sanitized_acc)
        };
        debug_assert!(final_acc.stored_trie_nodes.is_empty(), "Internal nodes must not carry stored_trie_nodes");

        let hash_key_cache = compute_hash_key_internal_with_acc(&final_acc, &predecessors);
        let max_depth = compute_max_depth(&predecessors);
        GSSNode::Internal(GSSInternal { acc: final_acc, predecessors, hash_key_cache, max_depth })
    }

    /// Helper to create a GSSNode with a single predecessor, used by `push`.
    /// In the new design, `acc` becomes the local Acc on this new internal node.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, acc: Acc) -> Self {
        // --- SUCK UP LOGIC ---
        let pred_local_acc = predecessor_arc.local_acc();

        let has_constraints_to_suck_up = pred_local_acc.llm_tokens_union != HybridBitset::max_ones()
            || pred_local_acc.terminals_union != HybridL2Bitset::all();

        let (final_parent_acc, final_predecessor_arc) = if has_constraints_to_suck_up {
            // Suck up constraints.
            let mut parent_acc = acc; // Start with the acc passed to the function.
            parent_acc.llm_tokens_union &= &pred_local_acc.llm_tokens_union;
            parent_acc.terminals_union &= &pred_local_acc.terminals_union;

            // Create new predecessor with stripped constraints but original payload (stored_trie_nodes).
            let mut new_pred_acc = Acc::new_fresh();
            new_pred_acc.stored_trie_nodes = pred_local_acc.stored_trie_nodes.clone();

            let new_predecessor = match predecessor_arc.as_ref() {
                GSSNode::Root(_) => GSSNode::new(new_pred_acc),
                GSSNode::Internal(i) => GSSNode::new_with_map(Arc::new(new_pred_acc), i.predecessors.clone()),
            };

            (parent_acc, Arc::new(new_predecessor))
        } else {
            // No constraints to suck up.
            (acc, predecessor_arc)
        };

        let pred_tx = final_predecessor_arc;
        let mut predecessors_map = NodeMap::new();
        predecessors_map
            .entry(edge_value)
            .or_default()
            .insert(pred_tx.dest_key(), vec![pred_tx]);
        Self::new_with_map(Arc::new(final_parent_acc), predecessors_map)
    }

    /// Helper to create a GSSNode with multiple predecessors, used by `push_many`.
    /// In the new design, `acc` becomes the local Acc on this new internal node.
    fn new_with_many_predecessors(predecessor_arc: Arc<GSSNode>, edge_values: Vec<ParseStateEdgeContent>, acc: Acc) -> Self {
        // --- SUCK UP LOGIC (identical to single predecessor) ---
        let pred_local_acc = predecessor_arc.local_acc();

        let has_constraints_to_suck_up = pred_local_acc.llm_tokens_union != HybridBitset::max_ones()
            || pred_local_acc.terminals_union != HybridL2Bitset::all();

        let (final_parent_acc, final_predecessor_arc) = if has_constraints_to_suck_up {
            // Suck up constraints.
            let mut parent_acc = acc; // Start with the acc passed to the function.
            parent_acc.llm_tokens_union &= &pred_local_acc.llm_tokens_union;
            parent_acc.terminals_union &= &pred_local_acc.terminals_union;

            // Create new predecessor with stripped constraints but original payload (stored_trie_nodes).
            let mut new_pred_acc = Acc::new_fresh();
            new_pred_acc.stored_trie_nodes = pred_local_acc.stored_trie_nodes.clone();

            let new_predecessor = match predecessor_arc.as_ref() {
                GSSNode::Root(_) => GSSNode::new(new_pred_acc),
                GSSNode::Internal(i) => GSSNode::new_with_map(Arc::new(new_pred_acc), i.predecessors.clone()),
            };

            (parent_acc, Arc::new(new_predecessor))
        } else {
            // No constraints to suck up.
            (acc, predecessor_arc)
        };

        let pred_tx = final_predecessor_arc;
        let mut predecessors_map = NodeMap::new();
        for edge_value in edge_values {
            predecessors_map
                .entry(edge_value)
                .or_default()
                .entry(pred_tx.dest_key())
                .or_default()
                .push(pred_tx.clone());
        }
        Self::new_with_map(Arc::new(final_parent_acc), predecessors_map)
    }

    pub fn new_dead() -> Self {
        Self::new(Acc::new_dead())
    }

    pub fn new_fresh() -> Self {
        Self::new(Acc::new_fresh())
    }

    /// Returns the aggregate Acc for this node.
    /// - If root: returns the node's Acc.
    /// - If internal: merges its childrens' Accs and narrows through its own local Acc.
    pub(crate) fn acc(&self) -> Arc<Acc> {
        match self {
            GSSNode::Root(r) => r.acc.clone(),
            GSSNode::Internal(i) => {
                // This is a recursive method that can be slow. It computes the aggregated
                // Acc for the entire subgraph rooted at this node. For performance-critical
                // checks like `is_alive`, a custom traversal is used instead.

                // 1. Merge the aggregated Accs of all children (predecessors).
                let mut merged_children_acc: Option<Acc> = None;

                for predecessor_arc in i.predecessors.values().flat_map(|m| m.values()).flatten() {
                    let child_acc = predecessor_arc.acc();
                    if let Some(merged) = merged_children_acc.as_mut() {
                        *merged = Acc::merge(merged, &child_acc);
                    } else {
                        merged_children_acc = Some((*child_acc).clone());
                    }
                }

                // 2. Narrow the merged children's Acc with this node's local Acc.
                if let Some(acc) = merged_children_acc {
                    Arc::new(Acc::narrow(&i.acc, &acc))
                } else {
                    // An internal node must have predecessors. If it somehow doesn't,
                    // its aggregated Acc is just its local Acc.
                    i.acc.clone()
                }
            }
        }
    }

    /// Returns the local Acc stored on this node (both internal and root).
    pub fn local_acc(&self) -> Arc<Acc> {
        match self {
            GSSNode::Root(r) => r.acc.clone(),
            GSSNode::Internal(i) => i.acc.clone(),
        }
    }

    pub fn predecessors(&self) -> &NodeMap {
        match self {
            GSSNode::Root(_) => empty_nodemap(),
            GSSNode::Internal(i) => &i.predecessors,
        }
    }

    pub(crate) fn num_predecessors(&self) -> usize {
        self.predecessors()
            .values()
            .map(|preds_by_depth| preds_by_depth.values().map(|v| v.len()).sum::<usize>())
            .sum()
    }

    pub fn max_depth(&self) -> MaxDepth {
        match self {
            GSSNode::Root(_) => 0,
            GSSNode::Internal(i) => i.max_depth,
        }
    }

    fn dest_key(&self) -> DestKey { self.max_depth() }

    /// Returns the set of LLM tokens allowed by any root reachable from this node.
    /// Applies the node's local Acc as a final intersection (blanket restriction).
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.acc().llm_tokens_union.clone()
    }

    /// Returns a map of disallowed terminals for each tokenizer state.
    /// A terminal is disallowed if it's disallowed on every root reachable from this node.
    pub fn disallowed_terminals(&self) -> TerminalInfo {
        self.acc().terminals_union.clone().complement()
    }

    pub fn is_empty(&self) -> bool { self.predecessors().is_empty() }

    pub fn is_alive(&self) -> bool {
        // This is an optimized version of `!self.allowed_llm_tokens().is_empty()`.
        // The original `allowed_llm_tokens` is slow because it calls `self.acc()`, which
        // traverses the entire subgraph to compute the union of all reachable root tokens.
        //
        // The check `!(&local & &aggregated).is_empty()` is equivalent to checking if there
        // exists *any* reachable root `r` such that `(&local & &r.acc.tokens).is_empty()` is false.
        // This allows us to traverse the graph and exit early as soon as we find such a root.

        let local_acc = self.local_acc();
        if local_acc.llm_tokens_union.is_empty() {
            return false;
        }

        match self {
            GSSNode::Root(_) => {
                // For a root, `allowed_llm_tokens()` is just its own `llm_tokens_union`.
                // Since we've already checked that it's not empty, it must be alive.
                true
            }
            GSSNode::Internal(_) => {
                // For an internal node, traverse to find at least one reachable root
                // whose tokens have a non-empty intersection with this node's local tokens.
                let mut visited_nodes: HashSet<*const GSSNode> = HashSet::new();
                let mut queue: VecDeque<Arc<GSSNode>> = VecDeque::new();

                // Start traversal from self's direct predecessors.
                for preds_by_depth in self.predecessors().values() {
                    for pred_vec in preds_by_depth.values() {
                        for pred in pred_vec {
                            queue.push_back(pred.clone());
                        }
                    }
                }

                while let Some(node) = queue.pop_front() {
                    let ptr = Arc::as_ptr(&node);
                    if !visited_nodes.insert(ptr) {
                        continue;
                    }

                    match node.as_ref() {
                        GSSNode::Root(r) => {
                            // Found a root. Check for non-empty intersection.
                            if !(&local_acc.llm_tokens_union & &r.acc.llm_tokens_union).is_empty() {
                                return true; // Early exit: found a valid path.
                            }
                        }
                        GSSNode::Internal(ii) => {
                            // Not a root, continue traversal.
                            for preds_by_depth in ii.predecessors.values() {
                                for pred_vec in preds_by_depth.values() {
                                    for pred in pred_vec {
                                        queue.push_back(pred.clone());
                                    }
                                }
                            }
                        }
                    }
                }

                // Traversed all paths and found no live ones.
                false
            }
        }
    }

    pub fn is_ok(&self) -> bool { self.is_alive() }

    pub fn is_root(&self) -> bool {
        matches!(self, GSSNode::Root(_))
    }

    pub fn merge_many_with_depth(merge_depth: usize, nodes: impl IntoIterator<Item = Arc<GSSNode>>) -> Arc<GSSNode> {
        timeit!(format!("GSSNode::merge_many_with_depth({})", merge_depth), {
        let mut iter = nodes.into_iter();
        if let Some(first) = iter.next() {
            let mut merged = (*first).clone();
            for other in iter {
                merged.merge_with_depth(merge_depth, &other);
            }
            Arc::new(merged)
        } else {
            Arc::new(GSSNode::new_fresh())
        }
        })
    }

    pub fn flatten(&self) -> Vec<(Vec<ParseStateEdgeContent>, Acc)> {
        let mut memo: HashMap<*const GSSNode, Vec<(Vec<ParseStateEdgeContent>, Acc)>> = HashMap::new();
        let mut paths = self._flatten_recursive(&mut memo);
        paths
    }

    fn _flatten_recursive(
        &self,
        memo: &mut HashMap<*const GSSNode, Vec<(Vec<ParseStateEdgeContent>, Acc)>>,
    ) -> Vec<(Vec<ParseStateEdgeContent>, Acc)> {
        let ptr = self as *const GSSNode;
        if let Some(cached_paths) = memo.get(&ptr) {
            return cached_paths.clone();
        }

        if self.is_root() {
            // A root node represents the end of a path (or the start of a reversed one).
            // The path from here is empty, and the acc is its own.
            let result = vec![(vec![], (*self.local_acc()).clone())];
            memo.insert(ptr, result.clone());
            return result;
        }

        let mut all_paths = Vec::new();
        if let GSSNode::Internal(internal) = self {
            // The local acc of this internal node needs to be applied to all paths passing through it.
            let local_acc = self.local_acc();

            for (edge_val, preds_by_depth) in &internal.predecessors {
                for pred_vec in preds_by_depth.values() {
                    for pred_arc in pred_vec {
                        let sub_paths = pred_arc._flatten_recursive(memo);
                        for (mut sub_path, sub_acc) in sub_paths {
                            // The path from the predecessor is extended with the current edge.
                            sub_path.push(edge_val.clone());
                            // The acc from the predecessor's path is narrowed by this node's local acc.
                            let new_acc = Acc::narrow(&local_acc, &sub_acc);
                            all_paths.push((sub_path, new_acc));
                        }
                    }
                }
            }
        }

        memo.insert(ptr, all_paths.clone());
        all_paths
    }
}

// Core GSS operations
impl GSSNode {
    /// Pushes a new state onto the stack(s) represented by this node.
    pub(crate) fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        GSSNode::new_with_single_predecessor(Arc::new(self.clone()), edge_value, Acc::new_fresh())
    }

    pub(crate) fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Self {
        GSSNode::new_with_many_predecessors(Arc::new(self.clone()), edge_values, Acc::new_fresh())
    }

    /// Performs a multi-level pop operation on this node.
    pub fn popn(&self, n: usize) -> GSSPopper {
        let mut popper = GSSPopper::new_from_node(Arc::new(self.clone()), Arc::new(Acc::new_fresh()));
        popper.popn(n);
        popper
    }

    #[allow(dead_code)] pub(crate) fn merge(&mut self, other: &Self) {
        self._merge(other, 1);
    }

    pub(crate) fn merge_with_depth(&mut self, merge_depth: usize, other: &Self) {
        timeit!(format!("GSSNode::merge_with_depth({}, ...)", merge_depth), {
            self._merge( other, merge_depth);
        });
    }

    fn _merge(&mut self, other: &Self, merge_depth: usize) {
        if self == other { return; }

        // Merge two roots by merging their Accs (unifies stored_trie_nodes, etc.)
        if let (GSSNode::Root(lhs), GSSNode::Root(rhs)) = (&self, other) {
            let merged = Acc::merge(&lhs.acc, &rhs.acc);
            let acc_arc = Arc::new(merged);
            *self = GSSNode::Root(GSSRoot {
                acc: acc_arc.clone(),
                hash_key_cache: compute_hash_key_root(&acc_arc),
            });
            return;
        }

        // If the other node is a root with a neutral Acc, it contributes nothing structurally.
        if other.is_root() {
            if other.acc().is_merge_neutral() {
                return;
            }
            // If self is a neutral root, replace with other.
            if self.is_root() && self.acc().is_merge_neutral() {
                *self = other.clone();
            }
            // Otherwise, ignore merging a non-neutral root into an internal node (no structural effect).
            return;
        }

        // If self is a neutral root, adopt other's structure.
        if self.is_root() && self.acc().is_merge_neutral() {
            *self = other.clone();
            return;
        }

        // Both sides are internal (or self is internal, other internal); merge NodeMaps.
        let mut self_predecessors = self.predecessors().clone();
        let mut other_predecessors = other.predecessors().clone();

        let self_acc = self.local_acc();
        if !self_acc.is_default() {
            for preds_by_depth in self_predecessors.values_mut() {
                for pred_vec in preds_by_depth.values_mut() {
                    for pred_arc in pred_vec.iter_mut() {
                        let pred_local_acc = pred_arc.local_acc();
                        let new_local_acc = Arc::new(Acc::narrow(&self_acc, &pred_local_acc));
                        if new_local_acc != pred_local_acc {
                            let new_node = match &**pred_arc {
                                GSSNode::Root(_) => GSSNode::new((*new_local_acc).clone()),
                                GSSNode::Internal(i) => {
                                    GSSNode::new_with_map(new_local_acc, i.predecessors.clone())
                                }
                            };
                            *pred_arc = Arc::new(new_node);
                        }
                    }
                }
            }
        }

        let other_acc = other.local_acc();
        if !other_acc.is_default() {
            for preds_by_depth in other_predecessors.values_mut() {
                for pred_vec in preds_by_depth.values_mut() {
                    for pred_arc in pred_vec.iter_mut() {
                        let pred_local_acc = pred_arc.local_acc();
                        let new_local_acc = Arc::new(Acc::narrow(&other_acc, &pred_local_acc));
                        if new_local_acc != pred_local_acc {
                            let new_node = match &**pred_arc {
                                GSSNode::Root(_) => GSSNode::new((*new_local_acc).clone()),
                                GSSNode::Internal(i) => {
                                    GSSNode::new_with_map(new_local_acc, i.predecessors.clone())
                                }
                            };
                            *pred_arc = Arc::new(new_node);
                        }
                    }
                }
            }
        }

        let mut merged_map = self_predecessors;
        merge_node_maps(&mut merged_map, other_predecessors, merge_depth);

        // --- NEW SUCK UP LOGIC ---
        // Collect all unique predecessor accs from the merged map.
        let mut child_accs: BTreeSet<Arc<Acc>> = BTreeSet::new();
        let mut has_preds = false;
        for preds_by_depth in merged_map.values() {
            for pred_vec in preds_by_depth.values() {
                if !pred_vec.is_empty() {
                    has_preds = true;
                    for pred_arc in pred_vec {
                        child_accs.insert(pred_arc.local_acc());
                    }
                }
            }
        }

        let mut final_predecessors_map = merged_map;
        let final_parent_acc;

        if has_preds && child_accs.len() == 1 {
            // All children have the same acc.
            let common_acc = child_accs.into_iter().next().unwrap();

            // Separate constraints from payload (stored_trie_nodes).
            let mut parent_sucked_up_acc = Acc::new_fresh();
            parent_sucked_up_acc.llm_tokens_union = common_acc.llm_tokens_union.clone();
            parent_sucked_up_acc.terminals_union = common_acc.terminals_union.clone();

            // Check if there are any constraints to suck up.
            let has_constraints_to_suck = parent_sucked_up_acc.llm_tokens_union != HybridBitset::max_ones()
                || parent_sucked_up_acc.terminals_union != HybridL2Bitset::all();

            if has_constraints_to_suck {
                final_parent_acc = Arc::new(parent_sucked_up_acc);

                // Build separate Acc arcs for root children and internal children.
                // Root children keep the payload (stored_trie_nodes).
                let mut child_acc_root = Acc::new_fresh();
                child_acc_root.stored_trie_nodes = common_acc.stored_trie_nodes.clone();
                let new_child_acc_root = Arc::new(child_acc_root);

                // Internal children must not carry any acc (neither constraints nor payload).
                let new_child_acc_internal = Arc::new(Acc::new_fresh());

                // Rebuild children with this new acc.
                let mut new_preds_map = BTreeMap::new();
                for (edge, preds_by_depth) in final_predecessors_map {
                    let mut new_preds_by_depth = BTreeMap::new();
                    for (depth, pred_vec) in preds_by_depth {
                        let mut new_pred_vec = Vec::with_capacity(pred_vec.len());
                        for pred_arc in pred_vec {
                            // Choose the correct stripped acc based on whether the child is a root or internal node.
                            let new_pred = match &*pred_arc {
                                GSSNode::Root(_) => GSSNode::new((*new_child_acc_root).clone()),
                                GSSNode::Internal(i) => GSSNode::new_with_map(new_child_acc_internal.clone(), i.predecessors.clone()),
                            };
                            new_pred_vec.push(Arc::new(new_pred));
                        }
                        new_preds_by_depth.insert(depth, new_pred_vec);
                    }
                    new_preds_map.insert(edge, new_preds_by_depth);
                }
                final_predecessors_map = new_preds_map;
            } else {
                // No constraints to suck up. Parent gets a fresh acc, children are not modified.
                final_parent_acc = Arc::new(Acc::new_fresh());
            }
        } else {
            // Children have different accs, or there are no children.
            // Parent gets a fresh acc.
            final_parent_acc = Arc::new(Acc::new_fresh());
        }

        let final_predecessors = if merge_depth > 0 {
            // After merging, unify structurally identical predecessors to increase sharing.
            let mut canonical_map: BTreeMap<GSSNode, Arc<GSSNode>> = BTreeMap::new();
            let mut unified_predecessors = BTreeMap::new();

            for (edge_val, preds_by_depth) in final_predecessors_map {
                let mut unified_preds_by_depth = BTreeMap::new();
                for (depth, pred_vec) in preds_by_depth {
                    let mut unified_pred_vec = Vec::new();
                    for pred_arc in pred_vec {
                        let canonical_arc = canonical_map.entry((*pred_arc).clone()).or_insert_with(|| pred_arc.clone()).clone();
                        unified_pred_vec.push(canonical_arc);
                    }
                    unified_pred_vec.sort_by_key(|a| Arc::as_ptr(a) as usize);
                    unified_pred_vec.dedup_by_key(|a| Arc::as_ptr(a));
                    unified_preds_by_depth.insert(depth, unified_pred_vec);
                }
                unified_predecessors.insert(edge_val, unified_preds_by_depth);
            }
            unified_predecessors
        } else {
            final_predecessors_map
        };

        *self = GSSNode::new_with_map(final_parent_acc, final_predecessors);
    }

    #[allow(dead_code)] pub(crate) fn merged(mut self, other: Self, merge_depth: usize) -> Self {
        self.merge_with_depth(merge_depth, &other);
        self
    }

    #[allow(dead_code)] pub(crate) fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        // In the simplified design, Acc is only at roots; this behaves like push.
        Self::new_with_single_predecessor(Arc::new(self.clone()), edge_value, Acc::new_fresh())
    }

    /// Returns an iterator over all direct predecessor paths (`GSSPeek`s).
    pub(crate) fn peek_iter(parent_arc: &Arc<GSSNode>) -> impl Iterator<Item = GSSPeek<'_>> {
        parent_arc.predecessors().iter().flat_map(move |(edge_val, preds_by_depth)| {
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
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent { self.edge_value }

    #[allow(dead_code)] pub(crate) fn predecessor_node(&self) -> &'a Arc<GSSNode> { self.predecessor_node }

    /// Returns the combined `Acc` of the parent and the predecessor.
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Acc {
        Acc::narrow(&self.parent_arc.local_acc(), &self.predecessor_node.local_acc())
    }

    /// Returns the resolved union of LLM tokens, without computing other parts of `Acc`.
    pub fn resolved_llm_tokens_union(&self) -> LLMTokenBV {
        let parent_local = &self.parent_arc.local_acc().llm_tokens_union;
        let pred = self.predecessor_node.allowed_llm_tokens();
        parent_local & &pred
    }

    /// Returns a new `GSSNode` representing the predecessor.
    #[allow(dead_code)] pub(crate) fn resolved_predecessor_node(&self) -> Arc<GSSNode> {
        self.predecessor_node.clone()
    }

    /// Pushes a new state onto the resolved predecessor.
    #[allow(dead_code)] pub(crate) fn push_on_predecessor(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        GSSNode::new_with_single_predecessor(self.predecessor_node.clone(), edge_value, Acc::new_fresh())
    }

    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        self.isolated_parent().as_ref().push(edge_value)
    }
    pub fn popn(&self, len: usize) -> GSSPopper {
        let isolated_parent = self.isolated_parent();
        let mut popper = GSSPopper::new_from_node(isolated_parent, Arc::new(Acc::new_fresh()));
        popper.popn(len);
        popper
    }

    /// Creates a new `GSSNode` that represents only the path segment of this peek.
    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        // Optimization: if the parent already represents this single path, return it.
        if self.parent_arc.num_predecessors() == 1 {
            return self.parent_arc.clone();
        }

        // Create a new parent node representing only this path segment.
        let mut predecessors = NodeMap::new();
        predecessors
            .entry(self.edge_value.clone())
            .or_default()
            .insert(self.predecessor_node.dest_key(), vec![self.predecessor_node.clone()]);

        // The new node keeps the original parent's local acc.
        Arc::new(GSSNode::new_with_map(self.parent_arc.local_acc(), predecessors))
    }
}

// Trait implementations for GSSNode
impl Hash for GSSNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            GSSNode::Root(r) => r.hash_key_cache.hash(state),
            GSSNode::Internal(i) => i.hash_key_cache.hash(state),
        }
    }
}

impl PartialEq for GSSNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GSSNode::Root(a), GSSNode::Root(b)) => {
                a.hash_key_cache == b.hash_key_cache && a.acc == b.acc
            }
            (GSSNode::Internal(a), GSSNode::Internal(b)) => {
                a.hash_key_cache == b.hash_key_cache && a.acc == b.acc && a.predecessors == b.predecessors
            }
            _ => false,
        }
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
        use std::mem::discriminant;
        let da = discriminant(self);
        let db = discriminant(other);
        if da != db {
            // Order by variant first
            return if matches!(self, GSSNode::Root(_)) { Ordering::Less } else { Ordering::Greater };
        }

        match (self, other) {
            (GSSNode::Root(a), GSSNode::Root(b)) => {
                a.hash_key_cache.cmp(&b.hash_key_cache)
                    .then_with(|| a.acc.cmp(&b.acc))
            }
            (GSSNode::Internal(a), GSSNode::Internal(b)) => {
                a.hash_key_cache.cmp(&b.hash_key_cache)
                    .then_with(|| a.acc.cmp(&b.acc))
                    .then_with(|| a.predecessors.cmp(&b.predecessors))
            }
            _ => Ordering::Equal,
        }
    }
}


// --- Pruning and Transformation ---

pub(crate) type PruneAndTransformRecursiveMemo = HashMap<*const GSSNode, Option<Arc<GSSNode>>>;

/// Prunes and/or transforms a GSS by:
/// - Invoking `internal_closure` on internal nodes to decide if they should be pruned entirely.
/// - Invoking `root_closure` on root nodes to determine the replacement Acc or prune the root.
///
/// Note:
/// - There is no early-continue/stop: recursion always traverses into children of internal nodes
///   unless `internal_closure` prunes that node.
/// - Internal nodes never hold Acc; only roots do.
fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    internal_closure: &mut impl FnMut(&GSSInternal) -> Option<(Arc<Acc>, bool)>,
    root_closure: &mut impl FnMut(&GSSRoot) -> Option<Arc<Acc>>,
    memo: &mut PruneAndTransformRecursiveMemo,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }
    let result = match node_arc.as_ref() {
        GSSNode::Root(root) => {
            match root_closure(root) {
                None => None, // Prune
                Some(new_acc_arc) => {
                    if Arc::ptr_eq(&new_acc_arc, &root.acc) {
                        // No change
                        Some(node_arc.clone())
                    } else {
                        // Acc changed, create new root node
                        let new_node = GSSNode::new((*new_acc_arc).clone());
                        Some(Arc::new(new_node))
                    }
                }
            }
        }
        GSSNode::Internal(internal) => {
            match internal_closure(internal) {
                None => { // Prune
                    memo.insert(node_ptr, None);
                    return None;
                }
                Some((new_acc_arc, recurse)) => {
                    let acc_changed = !Arc::ptr_eq(&new_acc_arc, &internal.acc);

                    if !recurse {
                        let result = if acc_changed {
                            // Acc changed, but no recursion. Rebuild node with old predecessors.
                            let new_node = GSSNode::new_with_map(new_acc_arc, internal.predecessors.clone());
                            Some(Arc::new(new_node))
                        } else {
                            // No change at all.
                            Some(node_arc.clone())
                        };
                        memo.insert(node_ptr, result.clone());
                        return result;
                    }

                    // Recurse into children.
                    let mut any_child_changed = false;
                    let mut new_predecessors_map: NodeMap = BTreeMap::new();

                    for (edge_val, preds_by_depth) in &internal.predecessors {
                        let mut new_preds_by_depth: BTreeMap<DestKey, Vec<Arc<GSSNode>>> = BTreeMap::new();
                        for (dest_key, pred_vec) in preds_by_depth {
                            let mut new_vec: Vec<Arc<GSSNode>> = Vec::new();
                            for pred_arc in pred_vec {
                                match prune_and_transform_recursive(pred_arc, internal_closure, root_closure, memo) {
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

                    if new_predecessors_map.is_empty() {
                        // All children pruned, so prune this node.
                        None
                    } else if !any_child_changed && !acc_changed {
                        // No change in children or acc, so no change in this node.
                        Some(node_arc.clone())
                    } else {
                        // Children or acc changed, create new internal node.
                        let transformed_node = GSSNode::new_with_map(new_acc_arc, new_predecessors_map);
                        Some(Arc::new(transformed_node))
                    }
                }
            }
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
    let node_ptr = Arc::as_ptr(root_arc);
    if let Some(cached) = memo.get(&node_ptr) {
        *root_arc = cached.clone().unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
        return;
    }
    let new_arc_opt = match root_arc.as_ref() {
        GSSNode::Root(root) => {
            let mut new_acc = (*root.acc).clone();
            new_acc.llm_tokens_union &= allowed_tokens;
            if new_acc.llm_tokens_union.is_empty() {
                None
            } else {
                Some(Arc::new(GSSNode::new(new_acc)))
            }
        }
        GSSNode::Internal(internal) => {
            let mut new_acc = (*internal.acc).clone();
            new_acc.llm_tokens_union &= allowed_tokens;
            if new_acc.llm_tokens_union.is_empty() {
                None
            } else {
                Some(Arc::new(GSSNode::new_with_map(
                    Arc::new(new_acc),
                    internal.predecessors.clone(),
                )))
            }
        }
    };
    memo.insert(node_ptr, new_arc_opt.clone());
    *root_arc = new_arc_opt.unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
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
    let mut internal_closure = |internal: &GSSInternal| {
        let mut new_acc = (*internal.acc).clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        Some((Arc::new(new_acc), true))
    };
    let mut root_closure = |root: &GSSRoot| {
        let mut new_acc = (*root.acc).clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        Some(Arc::new(new_acc))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub(crate) fn reset_terminals(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut internal_closure = |internal: &GSSInternal| {
        let mut new_acc = (*internal.acc).clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        Some((Arc::new(new_acc), true))
    };
    let mut root_closure = |root: &GSSRoot| {
        let mut new_acc = (*root.acc).clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        Some(Arc::new(new_acc))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub(crate) fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let node_ptr = Arc::as_ptr(root_arc);
    if let Some(cached) = memo.get(&node_ptr) {
        *root_arc = cached.clone().unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
        return;
    }

    let new_node = match root_arc.as_ref() {
        GSSNode::Root(root) => {
            let mut new_acc = (*root.acc).clone();
            new_acc.terminals_union -= disallowed_terminals;
            GSSNode::new(new_acc)
        }
        GSSNode::Internal(internal) => {
            let mut new_acc = (*internal.acc).clone();
            new_acc.terminals_union -= disallowed_terminals;
            GSSNode::new_with_map(Arc::new(new_acc), internal.predecessors.clone())
        }
    };
    let new_arc = Arc::new(new_node);
    memo.insert(node_ptr, Some(new_arc.clone()));
    *root_arc = new_arc;
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let transform_acc = |acc: &Arc<Acc>| -> Option<Arc<Acc>> {
        if acc.terminals_union == HybridL2Bitset::all() {
            return Some(acc.clone());
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
            return Some(acc.clone());
        }

        let mut new_acc = (**acc).clone();
        new_acc.llm_tokens_union -= &forbidden_llm_tokens;

        if new_acc.llm_tokens_union.is_empty() {
            None // Prune this path
        } else {
            Some(Arc::new(new_acc))
        }
    };

    let mut internal_closure = |internal: &GSSInternal| transform_acc(&internal.acc).map(|new_acc| (new_acc, true));
    let mut root_closure = |root: &GSSRoot| transform_acc(&root.acc);

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let check_and_prune = |node_acc: &Arc<Acc>| -> bool {
        // Returns true if the node should be pruned.
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = node_acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                return true;
            }
        }
        false
    };

    let mut internal_closure = |internal: &GSSInternal| {
        if check_and_prune(&internal.acc) {
            None
        } else {
            Some((internal.acc.clone(), true))
        }
    };
    let mut root_closure = |root: &GSSRoot| -> Option<Arc<Acc>> {
        if check_and_prune(&root.acc) { None } else { Some(root.acc.clone()) }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let map_one = |terminals: &HybridL2Bitset| -> HybridL2Bitset {
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

        new_terminals_l2_bitset
    };

    let transform_acc = |acc: &Arc<Acc>| -> Option<Arc<Acc>> {
        let mut new_acc = (**acc).clone();
        let new_terminals_union = map_one(&acc.terminals_union);
        new_acc.terminals_union = new_terminals_union;
        Some(Arc::new(new_acc))
    };

    let mut internal_closure = |internal: &GSSInternal| transform_acc(&internal.acc).map(|acc| (acc, true));
    let mut root_closure = |root: &GSSRoot| transform_acc(&root.acc);

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub(crate) fn merge_stored_trie_nodes(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    let mut new_destinations = BTreeMap::new();

    let mut internal_closure = |internal: &GSSInternal| Some((internal.acc.clone(), true));
    let mut root_closure = |root: &GSSRoot| -> Option<Arc<Acc>> {
        if !root.acc.stored_trie_nodes.iter().any(
            // TODO: can this condition be relaxed to a subset or something?
            |n| n.as_arc().read(stored_trie_god).expect("poison").value.live_tokens != root.acc.llm_tokens_union
        ) {
            return Some(root.acc.clone());
        }
        let mut new_acc = (*root.acc).clone();
        // Create a single new destination for this merge operation.
        let new_destination = new_destinations.entry((new_acc.stored_trie_nodes.clone(), root.acc.llm_tokens_union.clone()))
            .or_insert_with(|| StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal()))))
            .clone();
        let edge_key = (0, new_acc.llm_tokens_union.clone());
        let edge_value = StateIDBV::max_ones();
        let tokens_for_edge = new_acc.llm_tokens_union.clone();

        for source_wrapper in &new_acc.stored_trie_nodes {
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
            // Insert a strong edge to the new shared destination.
            inserter.try_destination(new_destination.clone()).expect("Cycle detected when merging stored_trie nodes; this should be impossible.");
        }

        // Update the live tokens on the new destination node.
        new_destination.write(stored_trie_god).expect("poison").value.live_tokens |= &tokens_for_edge;

        // The acc now points only to this new merged destination.
        new_acc.stored_trie_nodes = BTreeSet::from([new_destination]);
        Some(Arc::new(new_acc))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
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
        let fused_arc = gss_simplification::fuse_predecessors_recursive(&temp_arc, levels, &mut memo);
        *self = Arc::try_unwrap(fused_arc).unwrap_or_else(|arc| (*arc).clone());
    }
}

pub(crate) fn deep_clone_gss_with_stored_trie_map(
    root: &Arc<GSSNode>,
    stored_trie_map: &HashMap<StoredPrecomputeNodeIndex, StoredPrecomputeNodeIndex>,
) -> Arc<GSSNode> {
    fn clone_one(
        node: &Arc<GSSNode>,
        stored_trie_map: &HashMap<StoredPrecomputeNodeIndex, StoredPrecomputeNodeIndex>,
        memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
    ) -> Arc<GSSNode> {
        let ptr = Arc::as_ptr(node);
        if let Some(cached) = memo.get(&ptr) {
            return cached.clone();
        }

        let out = match node.as_ref() {
            GSSNode::Root(root_node) => {
                // Remap stored_trie_nodes for the root Acc
                let mut new_acc = (*root_node.acc).clone();
                if !new_acc.stored_trie_nodes.is_empty() {
                    let mut new_set = BTreeSet::new();
                    for old_wr in &new_acc.stored_trie_nodes {
                        let old_arc = old_wr.as_arc().clone();
                        let old_ptr = old_arc;
                        if let Some(new_arc) = stored_trie_map.get(&old_ptr) {
                            new_set.insert(new_arc.clone());
                        } else {
                            new_set.insert(old_arc);
                        }
                    }
                    new_acc.stored_trie_nodes = new_set;
                }
                Arc::new(GSSNode::new(new_acc))
            }
            GSSNode::Internal(internal) => {
                // Clone predecessors recursively
                let mut new_preds: BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<Arc<GSSNode>>>> = BTreeMap::new();
                for (edge_val, preds_by_depth) in &internal.predecessors {
                    let mut new_by_depth = BTreeMap::new();
                    for (dest_key, pred_vec) in preds_by_depth {
                        let mut new_vec = Vec::with_capacity(pred_vec.len());
                        for pred in pred_vec {
                            new_vec.push(clone_one(pred, stored_trie_map, memo));
                        }
                        new_by_depth.insert(*dest_key, new_vec);
                    }
                    new_preds.insert(edge_val.clone(), new_by_depth);
                }
                Arc::new(GSSNode::new_with_map(internal.acc.clone(), new_preds))
            }
        };

        memo.insert(ptr, out.clone());
        out
    }

    let mut memo: HashMap<*const GSSNode, Arc<GSSNode>> = HashMap::new();
    clone_one(root, stored_trie_map, &mut memo)
}

impl GSSNode {
    #[allow(dead_code)] pub(crate) fn reset_llm_tokens(&mut self) {
        let mut node_arc = Arc::new(self.clone());
        reset_llm_tokens(&mut node_arc, &mut HashMap::new());
        *self = Arc::try_unwrap(node_arc).unwrap_or_else(|arc| (*arc).clone());
    }

    #[allow(dead_code)] pub(crate) fn get_roots(&self) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
        get_roots(std::iter::once(self))
    }
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

impl GSSNode {
    pub(crate) fn hash_code(&self) -> u64 {
        match self {
            GSSNode::Root(r) => r.hash_key_cache,
            GSSNode::Internal(i) => i.hash_key_cache,
        }
    }

    pub fn print(&self) -> String {
        let mut config = GSSPrintConfig::default();
        config.verbose = true;
        let terminal_map = bimap::BiBTreeMap::new();
        let (s, _) = gss_analysis::print_gss_forest(&[Arc::new(self.clone())], &terminal_map, &config);
        s
    }
}

#[cfg(test)]
mod tests {
    use bimap::BiBTreeMap;
    use crate::constraint::PrecomputedNodeContents;
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
        // The narrowed union should allow all but 1.
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
            &[parent.clone()], //
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
        let mut memo = PruneAndTransformRecursiveMemo::new();
        let new_root_opt = super::prune_and_transform_recursive(
            &root,
            &mut |internal: &GSSInternal| Some((internal.acc.clone(), true)), // don't prune internal nodes
            &mut |root_node: &GSSRoot| Some(root_node.acc.clone()), // No-op: keep root
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
    fn test_merge_preserves_stored_trie_nodes() {
        // This test reproduces a bug where merging GSS nodes would cause
        // stored_trie_nodes from leaf predecessors to be lost due to incorrect
        // constraint propagation (narrowing).

        // --- GSS 1 Setup ---
        let stored_trie_god = StoredTrieGodWrapper::new();
        let stored_trie_node1 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let stored_trie_node2 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let stored_trie_node3 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));

        let mut acc_l1 = empty_acc();
        acc_l1.stored_trie_nodes_mut().insert(stored_trie_node1.clone());
        let l1 = Arc::new(GSSNode::new(acc_l1));

        let mut acc_l2 = empty_acc();
        acc_l2.stored_trie_nodes_mut().insert(stored_trie_node2.clone());
        let l2 = Arc::new(GSSNode::new(acc_l2));

        let mut acc_l3 = empty_acc();
        acc_l3.stored_trie_nodes_mut().insert(stored_trie_node3.clone());
        let l3 = Arc::new(GSSNode::new(acc_l3));

        let mut gss1_preds = NodeMap::new();
        gss1_preds.entry(mock_edge(0)).or_default().insert(l1.dest_key(), vec![l1.clone()]);
        gss1_preds.entry(mock_edge(1)).or_default().insert(l2.dest_key(), vec![l2.clone()]);
        gss1_preds.entry(mock_edge(2)).or_default().insert(l3.dest_key(), vec![l3.clone()]);

        let mut gss1 = GSSNode::new_with_map(Arc::new(mock_acc(0)), gss1_preds); // mock_acc(0) restricts token 0

        // --- GSS 2 Setup ---
        let mut acc_l4 = empty_acc();
        acc_l4.stored_trie_nodes_mut().insert(stored_trie_node1.clone()); // Shared stored_trie_node
        let l4 = Arc::new(GSSNode::new(acc_l4));
        let i1 = Arc::new(l4.push(mock_edge(0)));
        let gss2 = i1.push(mock_edge(1));

        // --- Merge ---
        gss1.merge_with_depth(1, &gss2);

        // --- Assertions ---
        // Traverse the merged GSS and collect all stored_trie_nodes from all leaf nodes.
        let mut q = VecDeque::new();
        q.push_back(Arc::new(gss1));
        let mut visited = HashSet::new();
        let mut final_leaf_stored_trie_nodes = BTreeSet::new();

        while let Some(node) = q.pop_front() {
            if !visited.insert(Arc::as_ptr(&node)) { continue; }
            if node.is_root() { final_leaf_stored_trie_nodes.extend(node.acc().stored_trie_nodes().clone()); }
            for p in node.predecessors().values().flat_map(|m| m.values()).flatten() { q.push_back(p.clone()); }
        }

        assert!(final_leaf_stored_trie_nodes.contains(&stored_trie_node1), "stored_trie_node1 missing");
        assert!(final_leaf_stored_trie_nodes.contains(&stored_trie_node2), "stored_trie_node2 missing");
        assert!(final_leaf_stored_trie_nodes.contains(&stored_trie_node3), "stored_trie_node3 missing");
        assert_eq!(final_leaf_stored_trie_nodes.len(), 3, "Should have 3 unique stored_trie nodes in the leaves");
    }

    #[test]
    fn test_merge_does_not_incorrectly_collapse_branches() {
        // This test reproduces a bug where merging two GSSs with a common edge value
        // but different sub-structures would incorrectly collapse the distinct sub-structures.

        // --- Shared Nodes ---
        let stored_trie_god = StoredTrieGodWrapper::new();
        let stored_trie_node1 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let mut acc1 = empty_acc();
        acc1.stored_trie_nodes_mut().insert(stored_trie_node1.clone());
        let leaf1 = Arc::new(GSSNode::new(acc1)); // This is "Node 2" with trie ...6f0

        let stored_trie_node2 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let mut acc2 = empty_acc();
        acc2.stored_trie_nodes_mut().insert(stored_trie_node2.clone());
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
        let stored_trie_god = StoredTrieGodWrapper::new();
        let stored_trie_node_a = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let mut acc_a = empty_acc();
        acc_a.stored_trie_nodes_mut().insert(stored_trie_node_a.clone());
        let leaf_a = Arc::new(GSSNode::new(acc_a));

        let gss_a = GSSNode::new_with_single_predecessor(
            leaf_a.clone(),
            mock_edge(1),
            empty_acc(),
        );

        // --- GSS B setup ---
        let stored_trie_node_b = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let mut acc_b = empty_acc();
        acc_b.stored_trie_nodes_mut().insert(stored_trie_node_b.clone());
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
    fn test_merge_unions_stored_trie_nodes_across_identical_towers() {
        // This test reproduces a bug where merging multiple identical towers (same edges and structure)
        // but with different stored_trie_nodes at the leaf results in the leaf keeping only one
        // of the stored_trie_nodes instead of the union of all of them.
        //
        // Structure for each tower:
        // Root -> (edge 2) -> ... -> Leaf [Trie={unique}]
        //
        // After merging two such towers, the single leaf should contain the union of the two distinct
        // of the stored_trie_nodes instead of the union of all of them.

        // --- Build two distinct stored_trie nodes ---
        let stored_trie_god = StoredTrieGodWrapper::new();
        let t1 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
        let t2 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));

        // Helper to build one tower given a leaf with a unique stored_trie node.
        let build_tower_from_leaf = |leaf: Arc<GSSNode>| -> GSSNode {
            let n5 = Arc::new(GSSNode::new_with_single_predecessor(leaf, mock_edge(5), empty_acc()));
            let n1 = Arc::new(n5.push(mock_edge(1)));
            n1.push(mock_edge(2))
        };

        // --- Leaf 1 with stored_trie_node t1 ---
        let mut acc1 = empty_acc();
        acc1.stored_trie_nodes_mut().insert(t1.clone());
        let leaf1 = Arc::new(GSSNode::new(acc1.clone()));
        let tower1 = build_tower_from_leaf(leaf1);

        // --- Leaf 2 with stored_trie_node t2 ---
        let mut acc2 = empty_acc();
        acc2.stored_trie_nodes_mut().insert(t2.clone());
        let leaf2 = Arc::new(GSSNode::new(acc2.clone()));
        let tower2 = build_tower_from_leaf(leaf2);

        // --- Merge the two identical towers ---
        let mut merged = tower1.clone();
        merged.merge_with_depth(usize::MAX, &tower2);

        // --- Assertions ---
        // With the new hoisting logic, the merged acc from the leaves should be hoisted
        // all the way to the top-level node of the merged tower.
        let final_acc = Acc::merge(&acc1, &acc2);
        let stored_trie_nodes = final_acc.stored_trie_nodes();

        assert_eq!(stored_trie_nodes.len(), 2, "Merged tower root should contain the union of stored_trie nodes from the leaves");
        assert!(stored_trie_nodes.contains(&t1), "Merged acc missing stored_trie node 1");
        assert!(stored_trie_nodes.contains(&t2), "Merged acc missing stored_trie node 2");

        // --- New assertions ---
        // 1. Check get_roots
        let roots_map = get_roots(std::iter::once(&merged));
        assert_eq!(roots_map.len(), 1, "get_roots should find one root path");
        let (last_edge, acc_set) = roots_map.iter().next().unwrap();
        assert_eq!(*last_edge, mock_edge(5));
        assert_eq!(acc_set.len(), 1, "There should be one unique path acc");
        let path_acc = acc_set.iter().next().unwrap();
        assert_eq!(**path_acc, final_acc, "Path acc from get_roots should match the hoisted acc");

        // 2. Check popping
        let tower_depth = 3;
        let popper = merged.popn(tower_depth + 1); // Pop one level past the bottom
        assert!(popper.paths.is_empty(), "Popper paths should be empty after popping past bottom");
        assert_eq!(popper.below_bottom.len(), 1, "Should have one entry in below_bottom");
        let (depth, by_edge) = popper.below_bottom.iter().next().unwrap();
        assert_eq!(*depth, 2, "Popping 1 level past bottom should result in depth key 2");
        assert_eq!(by_edge.len(), 1, "Should be one edge leading to bottom");
        let (edge, acc) = by_edge.iter().next().unwrap();
        assert_eq!(*edge, mock_edge(5));
        assert_eq!(**acc, final_acc, "Acc from popping past bottom should match hoisted acc");
    }

    #[test]
    fn test_allow_only_llm_tokens_and_prune_arc_simple_tower() {
        // This test is based on a real-world bug where filtering did not seem to apply.
        // Structure: Root -> (edge 2) -> Node 1 -> (edge 0) -> Node 2 (leaf)

        // 1. Build the GSS tower.
        let leaf = Arc::new(GSSNode::new(empty_acc())); // Node 2
        let intermediate = Arc::new(leaf.push(mock_edge(0))); // Node 1
        let mut root_arc = Arc::new(intermediate.push(mock_edge(2))); // Root 0

        // 2. Check initial state.
        assert_eq!(
            root_arc.allowed_llm_tokens(),
            HybridBitset::max_ones(),
            "Initial allowed tokens should be everything"
        );

        // 3. Filter to allow only token 0.
        let mut allowed_tokens = LLMTokenBV::zeros();
        allowed_tokens.insert(0);
        allow_only_llm_tokens_and_prune_arc(&mut root_arc, &allowed_tokens, &mut HashMap::new());

        // 4. Assert that the allowed tokens for the whole GSS have been updated.
        assert_eq!(
            root_arc.allowed_llm_tokens(),
            allowed_tokens,
            "Allowed tokens should be restricted to only token 0 after filtering"
        );
    }

    #[test]
    fn test_popn_collect_isolated_parents_preserves_acc() {
        // Setup: Root(acc42) -> Intermediate(empty) -> Leaf(empty)
        let leaf = Arc::new(GSSNode::new(empty_acc()));
        let intermediate = Arc::new(leaf.push(mock_edge(10)));
        assert!(intermediate.local_acc().is_default());

        let root_acc = mock_acc(42);
        let root = Arc::new(GSSNode::new_with_single_predecessor(
            intermediate.clone(),
            mock_edge(20),
            root_acc.clone(),
        ));
        assert_eq!(*root.local_acc(), root_acc);

        // Action: pop 1 level. We expect to get back a node representing the `intermediate`
        // node, but with the path constraint from the root applied.
        let result = popn_collect_isolated_parents(&root, 1);

        assert_eq!(result.len(), 1);
        let (_state_id, isolated_parent) = &result[0];

        // Validation: The `isolated_parent` is a reconstruction of the `intermediate` node's
        // path to its predecessor (`leaf`). The overall `acc()` of this new structure
        // should reflect the constraint from the popped `root`.
        let final_acc = isolated_parent.acc();

        // The final acc should be the intersection of the root's acc and the rest of the path.
        // Since the rest of the path is empty, it should just be the root's acc.
        assert_eq!(*final_acc, root_acc);
        assert!(!final_acc.llm_tokens_union.contains(42));
    }
}
