use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper
use crate::glr::table::StateID; // Import StateID

/// Represents an edge in the GSS, linking a predecessor node and carrying a value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PredecessorLink<T: Ord + Clone + Debug + Hash> {
    pub node: ArcPtrWrapper<GSSNode<T>>, // Arc to the predecessor GSSNode
    pub edge_value: T,                   // Value on the edge FROM `node` TO the GSSNode containing this PredecessorLink
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSNode<T: Ord + Clone + Debug + Hash> { // T is the edge data type
    pub state_id: StateID, // The "value" of the node itself (parser state ID)
    predecessors: BTreeSet<PredecessorLink<T>>,
}

impl<T: Ord + Clone + Debug + Hash> GSSNode<T> {
    /// Creates a new GSS node with a state ID and no predecessors.
    pub fn new(state_id: StateID) -> Self {
        Self {
            state_id,
            predecessors: BTreeSet::new(),
        }
    }

    /// Creates a new GSS node with a state ID and a list of predecessor links.
    pub fn new_with_predecessors(state_id: StateID, predecessors: Vec<PredecessorLink<T>>) -> Self {
        Self {
            state_id,
            predecessors: predecessors.into_iter().collect(),
        }
    }

    /// Creates a successor GSS node given a predecessor node, the state ID for the new node,
    /// and the value on the edge leading from the predecessor to the new node.
    pub fn make_successor_node(predecessor_arc: Arc<GSSNode<T>>, successor_state_id: StateID, edge_val: T) -> GSSNode<T>
    where T: Ord + Clone + Debug + Hash
    {
        GSSNode {
            state_id: successor_state_id,
            predecessors: {
                let mut preds = BTreeSet::new();
                preds.insert(PredecessorLink {
                    node: ArcPtrWrapper::new(predecessor_arc),
                    edge_value: edge_val,
                });
                preds
            }
        }
    }

    /// Creates a GSS path from an iterator of (StateID, T) tuples.
    /// The first StateID is for the root node, and the first T is the edge value
    /// conceptually leading to the root (often Default::default() or a starting value).
    /// Subsequent tuples define successor nodes and the edge values leading to them.
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (StateID, T)>,
        T: Ord + Clone + Debug + Hash + Default, // Added Default bound for initial edge value
    {
        let mut iter = iter.into_iter();
        // Get the first state ID for the root node. The first edge value is ignored or used conceptually.
        let (first_state_id, _first_edge_val) = iter.next().expect("from_iter requires at least one element");
        // The root node has no predecessors in this simple linear construction.
        let mut current_node_arc = Arc::new(Self::new(first_state_id));

        for (state_id, edge_val) in iter {
            let new_node = Self::make_successor_node(current_node_arc.clone(), state_id, edge_val);
            current_node_arc = Arc::new(new_node);
        }
        // from_iter returns the last node created.
        Arc::try_unwrap(current_node_arc).unwrap_or_else(|arc| (*arc).clone())
    }

    // The original `push` method is removed as its signature and consumption of `self` is problematic
    // with the new structure. Use `make_successor_node` instead.

    /// Returns a vector of `PredecessorLink`s representing the incoming edges to this node.
    pub fn pop(&self) -> Vec<PredecessorLink<T>> {
        self.predecessors.iter().cloned().collect()
    }

    /// Traverses up the GSS `n` steps from the current node and returns a vector of tuples,
    /// where each tuple contains an ancestor node at that distance and the value of the edge
    /// that directly connects that ancestor to one of the nodes reached at distance `n-1`.
    /// This implementation is a sketch and may need refinement for efficiency/correctness
    /// for complex graphs and large `n`.
    pub fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, T)>
    where
        T: Ord + Clone + Debug + Hash,
    {
        if n == 0 {
            // Returning ancestors at distance 0 is ill-defined with (Arc<Self>, T) return type.
            // The T is the edge value leading *to* the node, there's no edge leading to self from self at distance 0.
            panic!("popn with n=0 is not supported for return type Vec<(Arc<GSSNode<T>>, T)>");
        }
        if n == 1 {
            // Distance 1 means immediate predecessors.
            return self.predecessors.iter().map(|link| (link.node.as_arc().clone(), link.edge_value.clone())).collect();
        }

        let mut current_level_nodes: Vec<(Arc<GSSNode<T>>, T)> = Vec::new();

        // Start with the predecessors at distance 1
        for link in &self.predecessors {
            current_level_nodes.push((link.node.as_arc().clone(), link.edge_value.clone()));
        }

        // Traverse up n-1 more steps
        for _depth in 1..n {
            let mut next_level_nodes_map: BTreeMap<(usize, T), (Arc<GSSNode<T>>, T)> = BTreeMap::new(); // Map to deduplicate (node_ptr, edge_value) pairs

            for (node_arc, _edge_val_to_node_arc) in current_level_nodes {
                // node_arc is a node at the current depth. We want its predecessors (at depth + 1 relative to start).
                for pred_link in &node_arc.predecessors {
                    // pred_link.node.as_arc() is the node at the next depth up.
                    // pred_link.edge_value is the edge value connecting pred_link.node to node_arc.
                    // The item we want for the result is (pred_link.node.as_arc(), pred_link.edge_value).
                    let key = (Arc::as_ptr(pred_link.node.as_arc()) as usize, pred_link.edge_value.clone());
                    next_level_nodes_map.entry(key).or_insert_with(|| {
                        (pred_link.node.as_arc().clone(), pred_link.edge_value.clone())
                    });
                }
            }
            current_level_nodes = next_level_nodes_map.into_values().collect();
            if current_level_nodes.is_empty() {
                break; // No more ancestors found
            }
        }
        current_level_nodes // This contains nodes at distance n, paired with the edge value that led to them from distance n-1
    }

    // The original `peek` method is removed as node value is now state_id.
    /// Returns the state ID of this GSS node.
    pub fn get_state_id(&self) -> StateID {
        self.state_id
    }

    // The original `value_mut` method is removed as node value is StateID (Copy).

    /// Flattens the GSS structure starting from this node into a list of paths.
    /// Each path is represented as a vector of tuples `(StateID, Option<T>)`,
    /// where the `Option<T>` is the edge value leading to that node in the path
    /// (None for the first node in a path).
    pub fn flatten(&self) -> Vec<Vec<(StateID, Option<T>)>>
    where
        T: Ord + Clone + Debug + Hash,
    {
        let mut result = Vec::new();
        // Stack stores (GSSNode Arc, Option<T> for edge leading to it in THIS path, current_path_accumulator)
        // The accumulator stores the path from the END BACKWARDS.
        let mut work_stack: Vec<(Arc<GSSNode<T>>, Option<T>, Vec<(StateID, Option<T>)>)> = Vec::new();

        // Initial call for self: no incoming edge T for the very first node in a path.
        work_stack.push((Arc::new(self.clone()), None, Vec::new()));

        while let Some((node_arc, edge_val_to_node, mut current_path_segment)) = work_stack.pop() {
            // Add the current node and the edge value that led to it (in this specific path)
            current_path_segment.push((node_arc.state_id, edge_val_to_node.clone()));

            if node_arc.predecessors.is_empty() {
                // Reached a root of this path, reverse to get correct order
                current_path_segment.reverse();
                result.push(current_path_segment);
            } else {
                for pred_link in &node_arc.predecessors {
                    work_stack.push((
                        pred_link.node.as_arc().clone(), // The predecessor node is the next item in the stack
                        Some(pred_link.edge_value.clone()), // The edge value to the *current* node becomes the edge value to the *next* item from this path's perspective
                        current_path_segment.clone(), // Clone the path segment
                    ));
                }
            }
        }
        result
    }

    /// Flattens a slice of GSS nodes into a vector of paths.
    pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<(StateID, Option<T>)>>
    where
        T: Ord + Clone + Debug + Hash,
    {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    /// Merges the predecessors of `other` into `self`. Assumes `self.state_id == other.state_id`.
    pub fn merge(&mut self, mut other: Self)
    where
        T: Ord + Clone + Debug + Hash, // T needs bounds
    {
        assert!(self.state_id == other.state_id); // Assert state IDs match
        self.predecessors.extend(std::mem::take(&mut other.predecessors)); // Extend predecessor links
    }

    /// Merges the predecessors of `other` into `self` without checking state IDs.
    pub fn merge_unchecked(&mut self, mut other: Self)
    where
        T: Ord + Clone + Debug + Hash, // T needs bounds
    {
        self.predecessors.extend(std::mem::take(&mut other.predecessors)); // Extend predecessor links
    }

    // The original `map` method is removed as it's complex to adapt and not used.
}

impl<T: Ord + Clone + Debug + Hash> Drop for GSSNode<T> {
    // Custom drop to iteratively drop predecessors and break potential cycles.
    fn drop(&mut self) {
        // Take the predecessors to drop them outside of holding the mutex
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        // Worklist contains Arc<GSSNode<T>> directly
        let mut worklist: Vec<Arc<GSSNode<T>>> = predecessors_to_process_further.into_iter().map(|link| link.node.into_arc()).collect(); // Use into_arc on the node ArcPtrWrapper

        while let Some(node_arc) = worklist.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                // Successfully got unique ownership, take predecessors and add their nodes to worklist
                worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter().map(|link| link.node.into_arc())); // Use into_arc
            }
            // Else: Arc is still shared, it will be dropped when the last ArcPtrWrapper wrapper is dropped.
        }
    }
}

// GSSTrait and its implementations are removed as the structure has changed significantly.
// pub trait GSSTrait<T> { ... }

// BulkMerge trait generic parameter should be the type of items in the Vec.
// For Vec<(Arc<GSSNode<T>>, T)>, the item type is (Arc<GSSNode<T>>, T).
pub trait BulkMerge<Item> {
    fn bulk_merge(&mut self);
}

impl<T_edge: Ord + Clone + Debug + Hash> BulkMerge<(Arc<GSSNode<T_edge>>, T_edge)> for Vec<(Arc<GSSNode<T_edge>>, T_edge)> {
    fn bulk_merge(&mut self) {
        // Groups by (StateID of node, edge_value leading to node)
        let mut groups: BTreeMap<(StateID, T_edge), Vec<Arc<GSSNode<T_edge>>>> = BTreeMap::new();

        for (node_arc, edge_val) in self.drain(..) {
            groups.entry((node_arc.state_id, edge_val)).or_default().push(node_arc);
        }

        let mut new_self = Vec::new();
        for ((state_id_key, edge_val_key), node_arcs_in_group) in groups {
            if node_arcs_in_group.is_empty() {
                continue;
            }
            if node_arcs_in_group.len() == 1 {
                new_self.push((node_arcs_in_group.into_iter().next().unwrap(), edge_val_key));
            } else {
                // Merge GSSNodes that share the same state_id_key and came via same edge_val_key
                // All nodes in node_arcs_in_group have state_id == state_id_key.
                // We need to merge their predecessor lists.
                let mut arcs_iter = node_arcs_in_group.into_iter();
                let mut first_arc = arcs_iter.next().unwrap(); // This is the one we'll modify (if shared, it clones)

                let mut merged_predecessors = BTreeSet::new();
                // Add predecessors of the first_arc (if it's cloned by Arc::make_mut, these are copies)
                // If Arc::make_mut is used, it handles cloning if necessary.
                merged_predecessors.extend(Arc::make_mut(&mut first_arc).predecessors.iter().cloned());

                for sibling_arc in arcs_iter { // These are other GSSNodes with same state_id and via same edge_val_key
                    for pred_link in &sibling_arc.predecessors {
                        merged_predecessors.insert(pred_link.clone());
                    }
                }
                // Update the predecessors of the (potentially cloned) first_arc
                Arc::make_mut(&mut first_arc).predecessors = merged_predecessors;
                new_self.push((first_arc, edge_val_key));
            }
        }
        *self = new_self;
    }
}


// Helper function for prune_and_transform_roots
/// Recursive helper for prune_and_transform.
/// Traverses the GSS structure from `node_arc`, applying `closure`.
/// `edge_to_node_value` is the value on the edge leading *to* `node_arc` from its parent in the current traversal path.
/// Returns `Option<(Arc<GSSNode<T>>, T)>` which is the potentially transformed node and the NEW edge value leading to it.
pub fn prune_and_transform_recursive<T: Ord + Clone + Debug + Hash>(
    node_arc: &Arc<GSSNode<T>>,
    edge_to_node_value: &T, // The value on the edge leading *to* this node in this path
    closure: &impl Fn(&(StateID, &T)) -> Option<((StateID, T), bool)>, // Closure takes (NodeStateID, &EdgeValueToNode) and returns Option<((NewNodeStateID, NewEdgeValueToNode), ContinueRecursion)>
    // Memoization key is the node pointer, value is the transformed (node_arc, edge_value_to_node) pair if kept.
    // Note: Memoization keyed only by node pointer is not fully correct if the outcome depends *only* on the edge value,
    // as the same node might be reached via different edge values with different closure results.
    // A more robust memo key would be HashMap<(*const GSSNode<T>, T), ...> but T might not be Hashable or too large.
    // Using node pointer only assumes the closure's decision for a node is consistent regardless of the incoming edge value T.
    memo: &mut HashMap<*const GSSNode<T>, Option<(Arc<GSSNode<T>>, T)>>,
) -> Option<(Arc<GSSNode<T>>, T)> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    // Apply the closure to determine if this node/edge combination should be kept and transformed
    match closure(&(node_arc.state_id, edge_to_node_value)) {
        None => {
            // Prune this path at this node/edge
            memo.insert(node_ptr, None); // Mark this node as pruned *via this path*. Memo key limitation makes this approximate.
            None
        }
        Some(((new_state_id, new_edge_val_for_this_link), continue_recursion)) => {
            let new_node_arc = if !continue_recursion {
                // Stop recursion here. Create a new node with the potentially changed state ID
                // and the predecessors from the original node, but only those that were
                // successfully pruned_and_transformed up to this point (if recursion stopped lower)
                // or just keep the original predecessors if stopping here means keeping the structure below.
                // With value on edges, stopping recursion means predecessors are kept, but their edge values
                // to this node are determined by the closure's returned new_edge_val_for_this_link.

                // Reconstruct predecessor links with the original predecessor nodes and the *new* edge value
                let transformed_predecessors: Vec<PredecessorLink<T>> = node_arc.predecessors.iter()
                    .map(|pred_link| PredecessorLink {
                         node: pred_link.node.clone(), // Keep original predecessor node pointer
                         edge_value: new_edge_val_for_this_link.clone(), // Use the new edge value returned by closure
                    })
                    .collect();

                Arc::new(GSSNode::new_with_predecessors(new_state_id, transformed_predecessors))

            } else {
                // Continue recursion for predecessors.
                let mut new_predecessor_links = Vec::new();
                for pred_link in &node_arc.predecessors {
                     // Recursively prune and transform the predecessor node and the edge leading to it
                    if let Some((new_pred_node_arc, new_pred_edge_val)) = prune_and_transform_recursive(
                        pred_link.node.as_arc(), // The predecessor node
                        &pred_link.edge_value,  // The edge value leading to the predecessor
                        closure,
                        memo,
                    ) {
                         // The new predecessor link points to the transformed predecessor node
                         // and uses the NEW edge value that was returned by the recursive call for the predecessor.
                        new_predecessor_links.push(PredecessorLink {
                            node: ArcPtrWrapper::new(new_pred_node_arc),
                            edge_value: new_pred_edge_val,
                        });
                    }
                }

                // If all predecessors were pruned, this path is pruned.
                if new_predecessor_links.is_empty() && !node_arc.predecessors.is_empty() {
                     memo.insert(node_ptr, None); // Mark as pruned
                     return None; // Return None, pruning this node
                } else {
                     // Create a new node with the new state ID and the transformed predecessor links
                     Arc::new(GSSNode::new_with_predecessors(new_state_id, new_predecessor_links))
                }
            };

            // Store the transformed node and the new edge value leading TO it in the memo
            // The edge value stored is the one returned by the closure for *this* node.
            let result_pair = (new_node_arc.clone(), new_edge_val_for_this_link);
            memo.insert(node_ptr, Some(result_pair.clone()));
            Some(result_pair)
        }
    }
}

/// Traverses the GSS forest defined by `roots`, applying `closure` to each node's value.
/// Handles shared nodes using memoization. Prunes branches where `closure` returns `None`.
/// Stops recursion down a path if `closure` returns `(_, false)`.
///
/// Note: With edge values, the "root" nodes of the input forest conceptually have an implicit
/// incoming edge. The closure needs to be applied starting from these roots. The edge value
/// for these initial calls should represent the state *before* processing the first item.
///
/// This function assumes the `roots` provided are the starting points of independent GSS paths.
/// The closure is applied to `(root.state_id, &initial_edge_value_for_root)`.
///
/// Returns a Vec of `Option<(Arc<GSSNode<T>>, T)>` corresponding to the input `roots`.
/// The returned `T` in the tuple is the new edge value leading to the transformed root node.
pub fn prune_and_transform_roots<T: Ord + Clone + Debug + Hash + Default>(
    roots: &[Arc<GSSNode<T>>],
    initial_edge_value_for_roots: T, // The conceptual edge value leading to the root nodes
    closure: &impl Fn(&(StateID, &T)) -> Option<((StateID, T), bool)>, // Closure takes (NodeStateID, &EdgeValueToNode)
) -> Vec<Option<(Arc<GSSNode<T>>, T)>> {
    let mut memo: HashMap<*const GSSNode<T>, Option<(Arc<GSSNode<T>>, T)>> = HashMap::new(); // Memoization keyed by node pointer

    roots
        .iter()
        .map(|root| {
            // Start the recursive process for each root, providing the initial edge value.
            // The result is Option<(transformed_root_arc, new_edge_val_to_transformed_root)>.
            prune_and_transform_recursive(root, &initial_edge_value_for_roots, closure, &mut memo)
        })
        .collect()
}


// --- Longest Path ---

// Recursive helper for find_longest_path.
// Returns the longest path *ending* at node_arc, discovered so far.
fn find_longest_path_recursive<T: Ord + Clone + Debug + Hash>(
    node_arc: &Arc<GSSNode<T>>,
    memo: &mut HashMap<*const GSSNode<T>, Vec<Arc<GSSNode<T>>>>, // Stores longest path ending at the key node
    visited_recursion: &mut HashSet<*const GSSNode<T>>, // Detects cycles during the current DFS traversal
) -> Vec<Arc<GSSNode<T>>> {
    let node_ptr = Arc::as_ptr(node_arc);

    // Check memo first
    if let Some(cached_path) = memo.get(&node_ptr) {
        return cached_path.clone();
    }

    // Cycle detection for the current traversal path
    if !visited_recursion.insert(node_ptr) {
        // Cycle detected, return an empty path to avoid infinite recursion
        // and signal that this path shouldn't be considered the longest.
        return Vec::new();
    }

    let mut longest_pred_path: Vec<Arc<GSSNode<T>>> = Vec::new();

    // Explore predecessors recursively
    if !node_arc.predecessors.is_empty() {
        for pred_link in &node_arc.predecessors { // pred_link is &PredecessorLink<T>
            let pred_arc = pred_link.node.as_arc(); // pred_arc is &Arc<GSSNode<T>>
            let pred_path = find_longest_path_recursive(pred_arc, memo, visited_recursion);
            // Only update if the predecessor path is valid (non-empty, meaning no cycle encountered below)
            if !pred_path.is_empty() && pred_path.len() > longest_pred_path.len() {
                longest_pred_path = pred_path;
            }
        }
    }
    // else: This node has no predecessors, it's a starting point for paths ending here.

    // Construct the path ending at the current node
    let mut current_path = longest_pred_path; // Starts with the longest path ending at a predecessor
    current_path.push(node_arc.clone()); // Appends the current node

    // Store in memo and backtrack from recursion stack
    memo.insert(node_ptr, current_path.clone());
    visited_recursion.remove(&node_ptr);

    current_path
}

/// Finds one of the longest paths in the GSS forest defined by the given roots.
/// Handles cycles by ignoring paths that contain them.
/// Returns the path as a Vec of nodes from a root to a leaf (or the longest path found).
/// Returns `None` if there are no roots or no valid paths (e.g., only cycles).
pub fn find_longest_path<T: Ord + Clone + Debug + Hash>(roots: &[Arc<GSSNode<T>>]) -> Option<Vec<Arc<GSSNode<T>>>> {
    let mut memo: HashMap<*const GSSNode<T>, Vec<Arc<GSSNode<T>>>> = HashMap::new();

    // Populate the memo by traversing from all roots
    for root_arc in roots {
        let mut visited_recursion = HashSet::new(); // Reset cycle detection for each root
        find_longest_path_recursive(root_arc, &mut memo, &mut visited_recursion);
    }

    // Find the longest path among all paths stored in the memo values
    memo.into_values().max_by_key(|path| path.len())
}

/// Statistics about the structure of a GSS forest.
#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    /// Number of root nodes provided.
    pub num_roots: usize,
    /// Total number of unique nodes reachable from the roots.
    pub unique_nodes: usize,
    /// Maximum depth encountered (distance from a root node).
    pub max_depth: usize,
    /// Average depth of nodes (distance from a root node).
    pub average_depth: f64,
    /// Number of nodes with more than one predecessor (merge points).
    pub merge_points: usize,
    /// Maximum number of predecessors for any single node.
    pub max_predecessors: usize,
    /// Average number of predecessors per node.
    pub average_predecessors: f64,
}

/// Gathers statistics about the GSS forest defined by the given roots.
/// Traverses the graph using BFS to calculate depths from roots.
pub fn gather_gss_stats<T: Ord + Clone + Debug + Hash>(roots: &[Arc<GSSNode<T>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited: HashSet<*const GSSNode<T>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<T>>, usize)> = VecDeque::new(); // (node, depth)

    let mut total_depth_sum: u64 = 0;
    let mut total_predecessors_sum: u64 = 0;

    for root_arc in roots {
        let root_ptr = Arc::as_ptr(root_arc);
        if visited.insert(root_ptr) {
            queue.push_back((root_arc.clone(), 0));
        }
    }

    while let Some((current_node_arc, current_depth)) = queue.pop_front() {
        let current_node = current_node_arc.as_ref(); // Borrow the content
        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(current_depth);
        total_depth_sum += current_depth as u64;

        let num_predecessors = current_node.predecessors.len();
        stats.max_predecessors = stats.max_predecessors.max(num_predecessors);
        total_predecessors_sum += num_predecessors as u64;
        if num_predecessors > 1 {
            stats.merge_points += 1;
        }

        for pred_link in &current_node.predecessors { // pred_link is &PredecessorLink<T>
            let pred_arc = pred_link.node.as_arc(); // pred_arc is &Arc<GSSNode<T>>
            let pred_raw_ptr = Arc::as_ptr(pred_arc);
            if visited.insert(pred_raw_ptr) {
                queue.push_back((pred_arc.clone(), current_depth + 1)); // Queue the Arc
            }
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth_sum as f64 / stats.unique_nodes as f64;
        stats.average_predecessors = total_predecessors_sum as f64 / stats.unique_nodes as f64;
    }

    stats
}

/// Recursive helper to build the string representation of the GSS structure.
fn print_gss_node_recursive<T: Debug + Ord + Clone + Hash>( // T needs bounds for PredecessorLink
    node_arc: &Arc<GSSNode<T>>,
    visited: &mut HashSet<*const GSSNode<T>>,
    indent: usize,
    node_count: &mut usize,
    max_nodes: usize,
    output: &mut String,
) -> Result<(), std::fmt::Error> {
    if *node_count >= max_nodes {
        return Ok(()); // Stop recursion if max_nodes limit is reached
    }

    let node_ptr = Arc::as_ptr(node_arc);
    let prefix = format!("{:indent$}", "", indent = indent * 2);

    if visited.contains(&node_ptr) {
        writeln!(output, "{}- Node {:p} (State: {:?}, Visited)", prefix, node_ptr, node_arc.state_id)?;
        return Ok(());
    }

    visited.insert(node_ptr);
    *node_count += 1;

    // Print current node info (StateID)
    writeln!(output, "{}- Node {:p}: State {:?}", prefix, node_ptr, node_arc.state_id)?;

    // Print predecessors (links with edge values)
    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessors:", prefix)?;
        for pred_link in &node_arc.predecessors { // pred_link is &PredecessorLink<T>
            let pred_arc = pred_link.node.as_arc(); // pred_arc is &Arc<GSSNode<T>>
            // Print info about the edge and the predecessor node
            writeln!(output, "{}    Edge Value: {:?}", prefix, pred_link.edge_value)?;
            // Recursively print predecessors
            print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes, output)?; // Corrected indent
            if *node_count >= max_nodes {
                 // Check again after recursive call in case it hit the limit
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Generates a string representation of the GSS forest structure starting from the given roots.
///
/// Traverses the graph, handling cycles and shared nodes. Stops printing if the number
/// of unique nodes encountered exceeds `max_nodes`.
///
/// # Arguments
/// * `roots` - A slice of `Arc<GSSNode<T>>` representing the roots of the forest.
/// * `max_nodes` - The maximum number of unique nodes to include in the output string.
///
/// # Returns
/// A `String` containing the formatted GSS structure, potentially truncated.
pub fn print_gss_forest<T: Debug + Ord + Clone + Hash>(roots: &[Arc<GSSNode<T>>], max_nodes: usize) -> String {
    let mut visited = HashSet::new();
    let mut node_count = 0;
    let mut output = String::new();

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }

    writeln!(&mut output, "GSS Forest Roots (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        writeln!(&mut output, "Root {}:", i).unwrap();
        // Need a conceptual edge value for the root to start the recursive call if the closure
        // needs it. However, the current recursive print function doesn't take the edge value.
        // It prints the node, then its predecessors (and their incoming edge values).
        // So, start the recursion from the root node itself. The edge values are printed
        // when recursing to its predecessors.
        match print_gss_node_recursive(root_arc, &mut visited, 1, &mut node_count, max_nodes, &mut output) {
            Ok(_) => {
                if node_count >= max_nodes {
                    writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
                    break; // Stop processing more roots if limit reached
                }
            }
            Err(e) => {
                // Should not happen with String::write_fmt
                eprintln!("Error writing GSS structure to string: {}", e);
                return format!("Error generating GSS string: {}", e);
            }
        }
    }

    if node_count < max_nodes && node_count > visited.len() {
         // This condition indicates some nodes were visited but not printed due to the limit being hit mid-recursion
         writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
    } else if node_count == max_nodes && roots.len() > 0 && visited.len() > max_nodes {
         // If we processed all roots but hit the node limit, indicate truncation.
         // Visited might be > max_nodes if a node was visited before the count was checked after a recursive call returned.
         writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
    }


    output
}
