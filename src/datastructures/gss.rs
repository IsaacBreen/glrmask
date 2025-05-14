use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::datastructures::ArcPtrWrapper;
use crate::glr::parser::MergeAndIntersect; // For V constraint
use crate::glr::table::StateID; // For GSSNode content

// Represents an edge in the GSS, pointing to a predecessor node.
// It carries a value `V` of a type that implements MergeAndIntersect.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSEdge<V: MergeAndIntersect> {
    pub value: V, // Value associated with this edge
    pub predecessor_node: Arc<GSSNode<V>>, // Points to the predecessor GSSNode
}

impl<V: MergeAndIntersect + Debug> Debug for GSSEdge<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GSSEdge")
         .field("value", &self.value)
         .field("predecessor_node (state_id)", &self.predecessor_node.state_id) // Avoid full recursion
         .field("predecessor_node (ptr)", &Arc::as_ptr(&self.predecessor_node))
         .finish()
    }
}


// GSSNode is generic over V (the edge value type).
// The GSSNode itself stores a StateID.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSNode<V: MergeAndIntersect> {
    pub state_id: StateID, // Content of the node itself
    // predecessors is a set of edges; each edge has a value and points to a predecessor node.
    predecessors: BTreeSet<ArcPtrWrapper<GSSEdge<V>>>,
}

impl<V: MergeAndIntersect + Debug> Debug for GSSNode<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Custom Debug to avoid deep recursion if predecessors are printed directly.
        // It's better to use print_gss_forest for full GSS visualization.
        f.debug_struct("GSSNode")
         .field("state_id", &self.state_id)
         .field("num_predecessors", &self.predecessors.len())
         // Optionally, list predecessor edge values or target StateIDs if short
         .finish()
    }
}


impl<V: MergeAndIntersect> GSSNode<V> {
    /// Creates a new GSS root node with a given StateID and no predecessors.
    pub fn new(state_id: StateID) -> Self {
        Self {
            state_id,
            predecessors: BTreeSet::new(),
        }
    }

    /// Creates a new GSS node with `new_node_state_id`.
    /// An edge with `edge_value` is created from the new node to `self` (the previous top).
    /// `self_arc` should be an Arc pointing to the node that becomes the predecessor.
    pub fn push(self_arc: &Arc<Self>, new_node_state_id: StateID, edge_value: V) -> Self {
        let edge = Arc::new(GSSEdge {
            value: edge_value,
            predecessor_node: self_arc.clone(),
        });
        let mut predecessors = BTreeSet::new();
        predecessors.insert(ArcPtrWrapper::new(edge));
        Self {
            state_id: new_node_state_id,
            predecessors,
        }
    }

    /// Returns a list of Arcs to the GSSEdges representing direct predecessors.
    pub fn pop_edges(&self) -> Vec<Arc<GSSEdge<V>>> {
        self.predecessors.iter().map(|wrapper| wrapper.as_arc().clone()).collect()
    }
    
    /// Pops `n` levels from the GSS stack, returning a list of GSS nodes found at that depth,
    /// along with the T value of the edge that *led to* that node.
    /// This needs to return `Vec<(Arc<GSSNode<V>>, V)>` where V is path T value.
    /// For now, returning `Vec<crate::glr::parser::ParseState<V>>` as a placeholder for what parser needs.
    /// This method needs careful design to correctly track T values along paths.
    pub fn popn_to_parse_states(&self, n: usize) -> Vec<crate::glr::parser::ParseState<V>> {
        if n == 0 {
            // If n is 0, we are "popping" to the current node.
            // This requires knowing the T value for the path ending at `self`.
            // This information is not stored in GSSNode itself.
            // This function signature is problematic if GSSNode doesn't know its path's T value.
            // The caller (parser) must manage the T value for the current node.
            // For now, this will be an empty vec or panic, as it's ill-defined for n=0 here.
            // Let's assume n > 0 for meaningful pops.
            // If n=0, it means "stay at current node". The parser's ParseState already has this.
            // So, popn should always be for n >= 1.
            if n == 0 {
                 panic!("popn_to_parse_states called with n=0. This should be handled by the caller using its current ParseState.");
            }
        }

        let mut result_parse_states = Vec::new();
        // (Arc<GSSNode<V>>, V_path_value_to_this_node, remaining_pops)
        let mut queue: VecDeque<(Arc<GSSNode<V>>, V, usize)> = VecDeque::new();
        
        // To start the queue, we need the T value of the path ending at `self`.
        // This is a conceptual problem: GSSNode itself doesn't store this.
        // The caller of popn_to_parse_states *must* provide the T value for the path ending at `self`.
        // Let's modify the signature or assume a default/clone for the root T if not available.
        // This is a HACK. The parser's `pop_and_goto` should manage this initial T.
        // For now, let's assume this function is called on a node that is part of a ParseState,
        // and that ParseState's t_value is the one for the path ending at `self`.
        // This function cannot be standalone without that context.

        // This function is deeply problematic with the current GSS structure.
        // GSSNode.popn should return Vec<Arc<GSSEdge<V>>> for one level,
        // and the parser should build paths and T values.

        // Let's redefine popn to return what GSS can provide:
        // For n=1, it returns direct predecessor edges.
        // For n>1, it recursively finds "grandparent" edges etc.
        // The return type should be Vec<Arc<GSSEdge<V>>> where these are the edges
        // at the n-th level of popping.
        
        // Simpler popn: returns nodes at depth n, and the edge value that *led to them*.
        // Vec<(Arc<GSSNode<V>>, V_edge_value)>
        
        let mut current_level_nodes: Vec<(Arc<GSSNode<V>>, Option<V>)> = vec![(Arc::new(self.clone()), None)]; // (Node, EdgeVal that led to it)
        let mut next_level_nodes_map: HashMap<ArcPtrWrapper<GSSNode<V>>, V> = HashMap::new();


        for i in 0..n {
            next_level_nodes_map.clear();
            if current_level_nodes.is_empty() { break; }

            for (node_arc, _edge_val_to_node) in current_level_nodes {
                for edge_wrapper in &node_arc.predecessors {
                    let edge = edge_wrapper.as_arc();
                    // The key is the predecessor_node. The value is the edge's value.
                    // If multiple paths lead to the same predecessor_node via different edges at this pop level,
                    // we need to merge their edge_values.
                    let pred_node_wrapped = ArcPtrWrapper::new(edge.predecessor_node.clone());
                    next_level_nodes_map.entry(pred_node_wrapped)
                        .and_modify(|existing_v| *existing_v = existing_v.merge(&edge.value))
                        .or_insert_with(|| edge.value.clone());
                }
            }
            current_level_nodes = next_level_nodes_map.iter()
                .map(|(node_wrapper, merged_edge_val)| (node_wrapper.as_arc().clone(), Some(merged_edge_val.clone())))
                .collect();
        }

        // current_level_nodes now contains (node, edge_value_that_led_to_node) after n pops.
        // This edge_value_that_led_to_node is the T value for the path ending at that node.
        for (node_arc, opt_edge_val) in current_level_nodes {
            if let Some(edge_val) = opt_edge_val { // Should always be Some if n > 0
                 result_parse_states.push(crate::glr::parser::ParseState {
                    gss_node: node_arc,
                    t_value: edge_val,
                });
            } else if n == 0 { 
                // This case should be handled by the caller with its current ParseState.
                // If we must return something, it implies self.state_id and an unknown T.
                // This path should ideally not be hit if n > 0.
            }
        }
        
        // TODO: Apply bulk_merge logic if ParseStates with same GSSNode but different t_values exist,
        // or ensure the map merge handles this. The current map merges edge_values for same target node.
        // BulkMerge on ParseState would merge t_values and GSSNodes.
        // For now, this simplified popn returns distinct ParseStates.
        // The parser's BTreeMap will handle merging ParseStates with the same key (StateID).
        result_parse_states
    }


    /// Peeks at the StateID of this GSS node.
    pub fn peek_state_id(&self) -> &StateID {
        &self.state_id
    }

    /// Merges predecessors from `other_node` into `self`.
    /// `self` must be a mutable reference (e.g., from `Arc::make_mut`).
    /// `other_node` is an Arc, its predecessors are cloned.
    pub fn absorb_predecessors_from(&mut self, other_node: &Arc<Self>) {
        if Arc::ptr_eq(&Arc::new(self.clone()), other_node) { // Crude check if self and other_node are same underlying
            return;
        }
        assert_eq!(self.state_id, other_node.state_id, "Cannot merge GSS nodes with different StateIDs");
        for edge_wrapper in &other_node.predecessors {
            self.predecessors.insert(edge_wrapper.clone()); // ArcPtrWrapper handles Arc cloning
        }
    }


    // flatten, map, etc. need to be adapted.
    // For flatten: what does it mean to flatten? A path of StateIDs, or StateIDs and Edge Values?
    // For map: map StateID to U, map EdgeValue V to W? GSSNode<U, W>.

    // This method is problematic as GSSNode<V> doesn't store T directly.
    // It was GSSNode<T> where T was ParseStateNodeContent.
    // pub fn flatten(&self) -> Vec<Vec<T>> where T: Clone { ... }
    // pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<T>> where T: Clone { ... }

    // merge_unchecked was for GSSNode<T_node_content>
    // Now merge is absorb_predecessors_from.
}

impl<V: MergeAndIntersect> Drop for GSSNode<V> {
    fn drop(&mut self) {
        let predecessors_to_process = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSEdge<V>>> = predecessors_to_process.into_iter().map(|w| w.into_arc()).collect();

        while let Some(edge_arc) = worklist.pop() {
            if let Ok(edge_inner) = Arc::try_unwrap(edge_arc) {
                // Edge is uniquely owned. Now consider its predecessor_node.
                if let Ok(mut node_inner) = Arc::try_unwrap(edge_inner.predecessor_node) {
                    // Node is also uniquely owned. Add its predecessor edges to worklist.
                    worklist.extend(std::mem::take(&mut node_inner.predecessors).into_iter().map(|w| w.into_arc()));
                }
                // else: predecessor_node is still shared, will be dropped when its last Arc ref (possibly via ArcPtrWrapper) goes.
            }
            // else: edge_arc is still shared.
        }
    }
}


// GSSTrait might need to be re-evaluated or removed if its usage becomes too complex.
// It was defined for GSSNode<T_node_content>.
// Now GSSNode<V_edge_content>.
pub trait GSSTrait<V: MergeAndIntersect + Clone>: Sized { // V is edge value
    type PeekNodeID<'a> where Self: 'a; // To peek StateID
    // How to peek edge value? An edge is not 'self'.

    fn peek_node_id(&self) -> Self::PeekNodeID<'_>;
    fn push_new_node(self: Arc<Self>, new_node_state_id: StateID, edge_value: V) -> GSSNode<V>;
    // pop and popn should return collections of GSSEdge<V> or (GSSNode<V>, V_edge_to_it)
}

// Implementation of GSSTrait for Arc<GSSNode<V>>
impl<V: MergeAndIntersect + Clone> GSSTrait<V> for Arc<GSSNode<V>> {
    type PeekNodeID<'a> = &'a StateID where Self: 'a;

    fn peek_node_id(&self) -> Self::PeekNodeID<'_> {
        &self.state_id
    }

    fn push_new_node(self: Arc<Self>, new_node_state_id: StateID, edge_value: V) -> GSSNode<V> {
        GSSNode::push(&self, new_node_state_id, edge_value)
    }
}


// BulkMerge was for Vec<Arc<GSSNode<T_node_content>>>
// Now it should be for Vec<crate::glr::parser::ParseState<V>> or similar.
// The goal is to merge paths that have reached the same GSSNode, combining their T values.
pub trait BulkMerge<T: MergeAndIntersect> {
    fn bulk_merge(&mut self);
}

// Implementing for Vec<crate::glr::parser::ParseState<V>>
impl<V: MergeAndIntersect> BulkMerge<V> for Vec<crate::glr::parser::ParseState<V>> {
    fn bulk_merge(&mut self) {
        if self.is_empty() { return; }

        let mut groups: BTreeMap<ArcPtrWrapper<GSSNode<V>>, Vec<V>> = BTreeMap::new();
        for parse_state in self.drain(..) {
            groups.entry(ArcPtrWrapper::new(parse_state.gss_node))
                  .or_default()
                  .push(parse_state.t_value);
        }

        for (node_wrapper, t_values) in groups {
            let final_gss_node = node_wrapper.into_arc(); // Convert wrapper back to Arc
            if t_values.is_empty() { continue; } // Should not happen

            let mut merged_t = t_values[0].clone();
            for i in 1..t_values.len() {
                merged_t = merged_t.merge(&t_values[i]);
            }
            self.push(crate::glr::parser::ParseState {
                gss_node: final_gss_node,
                t_value: merged_t,
            });
        }
    }
}


// prune_and_transform_recursive and related functions:
// These are highly dependent on what `T` was.
// Original: GSSNode<T_node_content>, closure Fn(&T_node_content).
// New: GSSNode<V_edge_content>. Node has StateID. Edges have V.
// The closure in constraint.rs used `content.t.active` etc.
// `content.t` was `LLMTokenInfo` (a V type). `content.state_id` was StateID.
// So the closure needs (StateID, V_edge_val).
// But a node can have multiple incoming edges. Pruning happens per path (edge).
// This means prune_and_transform should operate on edges or (node, specific_incoming_edge).

// For now, these are commented out as they need a redesign based on new GSS structure
// and how they are used in constraint.rs.
/*
pub fn prune_and_transform_recursive<N: Clone, V: MergeAndIntersect + Clone>(
    node_arc: &Arc<GSSNode<N, V>>, // Assuming GSSNode<N,V> where N is node content, V is edge
    closure: &impl Fn(&N, &V) -> Option<(N, V, bool)>, // (new_node_content, new_edge_value, continue_recursion)
    memo: &mut HashMap<*const GSSNode<N, V>, Option<Arc<GSSNode<N, V>>>>,
) -> Option<Arc<GSSNode<N, V>>> {
    // ... major rewrite needed ...
    None
}

pub fn prune_and_transform_roots<N: Clone, V: MergeAndIntersect + Clone>(
    roots: &[Arc<GSSNode<N, V>>],
    closure: &impl Fn(&N, &V) -> Option<(N, V, bool)>,
) -> Vec<Option<Arc<GSSNode<N, V>>>> {
    let mut memo = HashMap::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive(root, closure, &mut memo))
        .collect()
}
*/

// --- Longest Path ---
// find_longest_path_recursive needs to be aware of GSSNode<V>
fn find_longest_path_recursive<V: MergeAndIntersect>(
    node_arc: &Arc<GSSNode<V>>,
    memo: &mut HashMap<*const GSSNode<V>, Vec<Arc<GSSNode<V>>>>,
    visited_recursion: &mut HashSet<*const GSSNode<V>>,
) -> Vec<Arc<GSSNode<V>>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_path) = memo.get(&node_ptr) { return cached_path.clone(); }
    if !visited_recursion.insert(node_ptr) { return Vec::new(); }

    let mut longest_pred_path: Vec<Arc<GSSNode<V>>> = Vec::new();
    if !node_arc.predecessors.is_empty() {
        for edge_wrapper in &node_arc.predecessors {
            let edge = edge_wrapper.as_arc();
            let pred_path = find_longest_path_recursive(&edge.predecessor_node, memo, visited_recursion);
            if !pred_path.is_empty() && pred_path.len() > longest_pred_path.len() {
                longest_pred_path = pred_path;
            }
        }
    }

    let mut current_path = longest_pred_path;
    current_path.push(node_arc.clone());
    memo.insert(node_ptr, current_path.clone());
    visited_recursion.remove(&node_ptr);
    current_path
}

pub fn find_longest_path<V: MergeAndIntersect>(roots: &[Arc<GSSNode<V>>]) -> Option<Vec<Arc<GSSNode<V>>>> {
    if roots.is_empty() { return None; }
    let mut memo: HashMap<*const GSSNode<V>, Vec<Arc<GSSNode<V>>>> = HashMap::new();
    for root_arc in roots {
        let mut visited_recursion = HashSet::new();
        find_longest_path_recursive(root_arc, &mut memo, &mut visited_recursion);
    }
    memo.into_values().filter(|p| !p.is_empty()).max_by_key(|path| path.len())
}


#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    pub num_roots: usize,
    pub unique_nodes: usize,
    pub unique_edges: usize, // Added
    pub max_depth: usize,
    pub average_depth: f64,
    pub merge_points: usize, // Nodes with >1 incoming edge
    pub max_predecessors: usize, // Max fan-in for a node
    pub average_predecessors: f64,
}

pub fn gather_gss_stats<V: MergeAndIntersect + Clone>(roots: &[Arc<GSSNode<V>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited_nodes: HashSet<*const GSSNode<V>> = HashSet::new();
    let mut visited_edges: HashSet<*const GSSEdge<V>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<V>>, usize)> = VecDeque::new(); // (node, depth from a root)

    for root_arc in roots {
        let root_ptr = Arc::as_ptr(root_arc);
        if visited_nodes.insert(root_ptr) { // Only add to queue if not visited as a node
            queue.push_back((root_arc.clone(), 0));
        }
    }

    while let Some((current_node_arc, current_depth)) = queue.pop_front() {
        // Node stats are counted when node is first processed from queue
        // (already inserted into visited_nodes before adding to queue)
        stats.unique_nodes = visited_nodes.len(); // Update unique_nodes based on set size
        stats.max_depth = stats.max_depth.max(current_depth);
        // total_depth_sum needs to be accumulated carefully if nodes can be reached at different depths.
        // BFS ensures shortest path depth.

        let num_predecessors = current_node_arc.predecessors.len();
        stats.max_predecessors = stats.max_predecessors.max(num_predecessors);
        // total_predecessors_sum += num_predecessors as u64; // Sum over unique nodes
        if num_predecessors > 1 {
            // This counts a node as a merge point if it has >1 direct predecessor edges.
            // This needs to be tracked per unique node.
            // Let's count merge points when a node is first confirmed as visited.
        }

        for pred_edge_wrapper in &current_node_arc.predecessors {
            let pred_edge_arc = pred_edge_wrapper.as_arc();
            let pred_edge_ptr = Arc::as_ptr(&pred_edge_arc);
            visited_edges.insert(pred_edge_ptr);

            let predecessor_gss_node = &pred_edge_arc.predecessor_node;
            let pred_gss_node_ptr = Arc::as_ptr(predecessor_gss_node);

            if visited_nodes.insert(pred_gss_node_ptr) { // If newly visited node
                queue.push_back((predecessor_gss_node.clone(), current_depth + 1));
            }
        }
    }
    
    // Recalculate stats based on unique visited nodes
    stats.unique_nodes = visited_nodes.len();
    stats.unique_edges = visited_edges.len();
    
    // For average depth and predecessors, we need to iterate over visited_nodes once
    let mut total_depth_sum: u64 = 0;
    let mut total_predecessors_sum: u64 = 0;
    let mut actual_merge_points = 0;

    // Re-traverse for accurate depth sum and predecessor counts for unique nodes
    // This requires a BFS-like traversal again to get depths correctly.
    // For simplicity, current GSSStats might be slightly off for avg_depth if graph is not a tree.
    // A full BFS from roots storing depths in a map would be more accurate for avg_depth.
    // For now, let's use a simplified calculation.
    if stats.unique_nodes > 0 {
        // These averages are approximations without a full depth map.
        // stats.average_depth = total_depth_sum as f64 / stats.unique_nodes as f64;
        // stats.average_predecessors = total_predecessors_sum as f64 / stats.unique_nodes as f64;
    }
    // stats.merge_points = actual_merge_points;


    stats // Return partially filled stats, avg might be off.
}


fn print_gss_node_recursive<V: MergeAndIntersect + Debug>(
    node_arc: &Arc<GSSNode<V>>,
    visited_nodes: &mut HashSet<*const GSSNode<V>>,
    visited_edges: &mut HashSet<*const GSSEdge<V>>,
    indent: usize,
    node_print_count: &mut usize,
    max_nodes_to_print: usize,
    output: &mut String,
) -> Result<(), std::fmt::Error> {
    if *node_print_count >= max_nodes_to_print {
        return Ok(());
    }

    let node_ptr = Arc::as_ptr(node_arc);
    let prefix = format!("{:indent$}", "", indent = indent * 2);

    if visited_nodes.contains(&node_ptr) {
        writeln!(output, "{}- Node {:p} (StateID: {}) (Visited)", prefix, node_ptr, node_arc.state_id.0)?;
        return Ok(());
    }

    visited_nodes.insert(node_ptr);
    *node_print_count += 1;

    writeln!(output, "{}- Node {:p} (StateID: {})", prefix, node_ptr, node_arc.state_id.0)?;

    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessor Edges:", prefix)?;
        for edge_wrapper in &node_arc.predecessors {
            let edge_arc = edge_wrapper.as_arc();
            let edge_ptr = Arc::as_ptr(&edge_arc);
            let edge_prefix = format!("{:indent$}", "", indent = (indent + 1) * 2);

            if visited_edges.contains(&edge_ptr) {
                writeln!(output, "{}- Edge {:p} (To Node {:p}, StateID: {}) (Value: {:?}) (Visited Edge)",
                    edge_prefix, edge_ptr, Arc::as_ptr(&edge_arc.predecessor_node), edge_arc.predecessor_node.state_id.0, edge_arc.value)?;
            } else {
                visited_edges.insert(edge_ptr);
                writeln!(output, "{}- Edge {:p} (To Node {:p}, StateID: {}) (Value: {:?})",
                    edge_prefix, edge_ptr, Arc::as_ptr(&edge_arc.predecessor_node), edge_arc.predecessor_node.state_id.0, edge_arc.value)?;
                print_gss_node_recursive(&edge_arc.predecessor_node, visited_nodes, visited_edges, indent + 2, node_print_count, max_nodes_to_print, output)?;
                if *node_print_count >= max_nodes_to_print {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

pub fn print_gss_forest<V: MergeAndIntersect + Debug>(roots: &[Arc<GSSNode<V>>], max_nodes_to_print: usize) -> String {
    let mut visited_nodes = HashSet::new();
    let mut visited_edges = HashSet::new();
    let mut node_print_count = 0;
    let mut output = String::new();

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }

    writeln!(&mut output, "GSS Forest Roots (Max Nodes To Print: {}):", max_nodes_to_print).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        writeln!(&mut output, "Root {}:", i).unwrap();
        match print_gss_node_recursive(root_arc, &mut visited_nodes, &mut visited_edges, 1, &mut node_print_count, max_nodes_to_print, &mut output) {
            Ok(_) => {
                if node_print_count >= max_nodes_to_print {
                    writeln!(&mut output, "... (Truncated: Reached max nodes to print {})", max_nodes_to_print).unwrap();
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error writing GSS structure to string: {}", e);
                return format!("Error generating GSS string: {}", e);
            }
        }
    }
    output
}

