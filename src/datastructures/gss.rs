use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper
use crate::glr::parser::MergeAndIntersect; // Import MergeAndIntersect

// GSSNode now takes two type parameters: N for node content, E for edge value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSNode<N, E> {
    pub node_content: N, // Content stored directly in the node
    // Predecessors map from the predecessor node (ArcPtrWrapper) to the edge value (E)
    predecessors: BTreeMap<ArcPtrWrapper<GSSNode<N, E>>, E>,
}

impl<N, E> GSSNode<N, E> {
    /// Creates a new GSS node with given content and no predecessors.
    pub fn new(node_content: N) -> Self {
        Self {
            node_content,
            predecessors: BTreeMap::new(),
        }
    }

    /// Creates a new GSS node with given content and a list of predecessors with associated edge values.
    pub fn new_with_predecessors(node_content: N, predecessors_list: Vec<(Arc<GSSNode<N, E>>, E)>) -> Self {
        Self {
            node_content,
            predecessors: predecessors_list.into_iter().map(|(node_arc, edge_val)| (ArcPtrWrapper::new(node_arc), edge_val)).collect(),
        }
    }

    // from_iter adaptation (assuming (N, E) pairs after the first node)
    pub fn from_iter_nodes_and_edges(first_node_content: N, iter_edges_and_then_nodes: impl IntoIterator<Item = (E, N)>) -> Self
    where
        N: Clone,
        E: Clone,
    {
        let mut root = Self::new(first_node_content);
        for (edge_val, next_node_content) in iter_edges_and_then_nodes {
            root = root.push(next_node_content, edge_val);
        }
        root
    }

    /// Creates a new GSS node with `node_content` and adds `self` as a predecessor with `edge_value`.
    pub fn push(self, node_content: N, edge_value: E) -> Self
    where
        N: Clone,
        E: Clone,
    {
        let mut new_node = Self::new(node_content);
        new_node.predecessors.insert(ArcPtrWrapper::new(Arc::new(self)), edge_value);
        new_node
    }

    /// Returns a vector of tuples, where each tuple contains the edge value leading to this node
    /// from a predecessor, and an Arc to the predecessor node.
    pub fn pop(&self) -> Vec<(E, Arc<Self>)>
    where
        E: Clone,
    {
        self.predecessors.iter().map(|(wrapper, edge_val)| (edge_val.clone(), wrapper.as_arc().clone())).collect()
    }

    /// Returns a vector of tuples, where each tuple contains an Arc to a node `n` steps back
    /// in the GSS forest reachable from `self`, and the accumulated edge value along the path
    /// from that node to `self`. The `edge_value_for_self` is the semantic value on the edge
    /// leading *to* `self`.
    pub fn popn(&self, n: usize, edge_value_for_self: E) -> Vec<(Arc<Self>, E)>
    where
        N: Clone,
        E: Clone + MergeAndIntersect,
    {
        if n == 0 {
            // Return the current node and the edge value leading to it.
            // We need an Arc to the current node, but self is &Self.
            // This function is intended to be called on Arc<GSSNode<N,E>> via the trait.
            // Let's assume the trait implementation for Arc handles getting the Arc.
            // If called directly on &Self, this needs adjustment (e.g., return Cow<Arc<Self>>).
            // Sticking to the trait's expected behavior: this is called on Arc<Self>.
             panic!("popn(0) called on &GSSNode. This should be called on Arc<GSSNode<N,E>> via the trait.");
        }

        let mut result: Vec<(Arc<Self>, E)> = Vec::new();
        let mut seen_paths: HashSet<(*const GSSNode<N, E>, E)> = HashSet::new(); // Track visited (node, accumulated_E) pairs

        let mut worklist: VecDeque<(Arc<Self>, E, usize)> = VecDeque::new(); // (current_node_arc, accumulated_t_on_path, steps_taken)
        // Start with the predecessors of self. The edge value from self to a predecessor is not meaningful here.
        // Instead, the path accumulation starts *from* the predecessors.
        // The edge value `edge_value_for_self` is the value on the edge *to* the node `self` is called on.
        // When we traverse to a predecessor, the edge value `edge_val_to_self` is the value *from* that predecessor.

        // Let's reconsider the goal of popn. It should find nodes N steps back and the accumulated semantic value *at* those nodes.
        // The `current_t` in `ParseState` is the accumulated value. This value lives *outside* the GSS edge values E.
        // The original popn returned `Vec<Arc<Self>>`. This new popn needs to return the ancestor node along with the accumulated value.
        // This suggests `popn` on GSSNode might not be the right place for `current_t` accumulation.
        // The accumulation should happen during the traversal in `pop_and_goto`.

        // Let's revert `popn` to just return the ancestor nodes (Arc<Self>) and the edge value leading to them (E),
        // similar to the original structure but including E.

        if n == 0 {
            // This path should not be taken if called from `Arc<GSSNode>::popn(0, ...)`.
             return vec![]; // Indicate no predecessors at distance 0.
        }

        let mut result_tuples = Vec::new();
        let mut seen_nodes: HashSet<*const GSSNode<N, E>> = HashSet::new();

        if n == 1 {
            // Directly return predecessors and their incoming edge values
            for (pred_wrapper, edge_val) in &self.predecessors {
                let pred_arc = pred_wrapper.as_arc().clone();
                let pred_ptr = Arc::as_ptr(&pred_arc);
                 if seen_nodes.insert(pred_ptr) {
                     result_tuples.push((pred_arc, edge_val.clone())); // Return predecessor and edge value TO self
                 }
            }
            return result_tuples;
        }

        // n > 1: Recurse on predecessors
        for (pred_wrapper, edge_val_to_pred) in &self.predecessors {
             let pred_arc = pred_wrapper.as_arc();
             // Recursive call: we need the edge value leading *to* the predecessor.
             // This suggests `popn` needs the edge value leading *to the node it is called on*.
             // The current `edge_value_for_self` is the value leading *to* `self`. This is passed correctly.
             // The edge value leading *to* a predecessor `pred_arc` is the one found in *its* predecessors map.
             // This implies we need to carry the edge value from the calling context.

             // Let's pass the edge value *from* the predecessor *to* the current node in the recursive call context?
             // No, the recursive call needs the edge value *into* the node it's being called on.

             // The structure `Vec<(Arc<Self>, E)>` returned by popn needs clarification on what the `E` represents.
             // Let's assume it's the accumulated edge value *along the path* of length `n`.

             let mut visited_recursion: HashSet<*const GSSNode<N, E>> = HashSet::new(); // Local cycle detection for paths

             // Let's try again with recursive helper
             let mut memo: HashMap<(*const GSSNode<N, E>, usize), Vec<(Arc<Self>, E)>> = HashMap::new(); // (node_ptr, steps), results

             // Initial call on self's predecessors
             for (pred_wrapper, edge_val_to_self) in &self.predecessors {
                 let pred_arc = pred_wrapper.as_arc().clone();
                 let mut visited_rec = HashSet::new();
                 // Recurse asking for n-1 steps back from the predecessor, with edge_val_to_self as the value *to* the predecessor?
                 // No, edge_val_to_self is from predecessor to self.

                 // The `popn` should return the node `n` steps back and the *accumulated* edge value.
                 // The accumulation should start from the predecessor.
                 let results_from_pred = GSSNode::popn_recursive(pred_arc.clone(), n - 1, edge_val_to_self.clone(), &mut memo, &mut visited_rec);

                 // Add results from this predecessor's paths
                 for res_tuple in results_from_pred {
                     let (ancestor_node_arc, accumulated_e) = res_tuple;
                     // The path from ancestor_node_arc goes through pred_arc to self.
                     // The accumulated_e is the value from ancestor to pred_arc.
                     // The edge from pred_arc to self has value `edge_val_to_self`.
                     // We need to combine `accumulated_e` and `edge_val_to_self`.
                     let total_accumulated_e = accumulated_e.merge(&edge_val_to_self); // Merge edge values

                     let tuple_to_add = (ancestor_node_arc, total_accumulated_e);

                      // Check for duplicate (node_ptr, accumulated_e) before adding
                     let add_key = (Arc::as_ptr(&tuple_to_add.0), tuple_to_add.1.clone());
                      if seen_paths.insert(add_key) {
                         result_tuples.push(tuple_to_add);
                     }
                 }
             }
             result_tuples.bulk_merge(); // Merge results based on ancestor node and accumulated E
             result_tuples // Return the collected and potentially merged results
        }
    }

    // Recursive helper for popn. Returns Vec<(AncestorNode, AccumulatedEdgeValue)>.
    fn popn_recursive(
        node_arc: Arc<Self>,
        steps_remaining: usize,
        edge_value_to_node: E, // Edge value leading *to* node_arc
        memo: &mut HashMap<(*const GSSNode<N, E>, usize), Vec<(Arc<Self>, E)>>,
        visited_recursion: &mut HashSet<*const GSSNode<N, E>>,
    ) -> Vec<(Arc<Self>, E)>
    where
        N: Clone,
        E: Clone + MergeAndIntersect,
    {
        let node_ptr = Arc::as_ptr(&node_arc);
        let memo_key = (node_ptr, steps_remaining);

        // Check memo
        if let Some(cached_result) = memo.get(&memo_key) {
            return cached_result.clone();
        }

        // Cycle detection for the current path
        if !visited_recursion.insert(node_ptr) {
            // Cycle detected, return empty path to break recursion
            return Vec::new();
        }

        let mut results: Vec<(Arc<Self>, E)> = Vec::new();

        if steps_remaining == 0 {
            // Reached the desired number of steps back. This node is an ancestor.
            // The accumulated value is the `edge_value_to_node` that led *to this node*.
            results.push((node_arc.clone(), edge_value_to_node));
        } else {
            // Need more steps back, recurse on predecessors
            for (pred_wrapper, edge_val_pred_to_node) in &node_arc.predecessors {
                let pred_arc = pred_wrapper.as_arc().clone();
                // The edge value leading *to* the predecessor is `edge_val_pred_to_node`.
                let results_from_pred = GSSNode::popn_recursive(
                    pred_arc,
                    steps_remaining - 1,
                    edge_val_pred_to_node.clone(), // Pass the edge value from pred to current node
                    memo,
                    visited_recursion,
                );
                results.extend(results_from_pred);
            }
        }

        // Backtrack from recursion stack
        visited_recursion.remove(&node_ptr);

        // Store result in memo (and bulk merge if necessary before storing?)
        // Let's bulk merge results before storing to avoid redundant entries if different paths lead to the same ancestor node with mergeable edge values.
        results.bulk_merge();
        memo.insert(memo_key, results.clone());

        results
    }


    /// Returns a reference to the node's content.
    pub fn peek_node_content(&self) -> &N {
        &self.node_content
    }

    /// Returns a mutable reference to the node's content.
    pub fn node_content_mut(&mut self) -> &mut N {
        &mut self.node_content
    }

    /// Flattens the GSS forest from this node down into a list of paths.
    /// Each path is a list of (NodeContent, EdgeValue) tuples, starting from a root
    /// and ending at this node. The `edge_value_for_self` is the semantic value
    /// on the edge leading to this node.
    pub fn flatten(&self, edge_value_for_self: E) -> Vec<Vec<(N, E)>>
    where
        N: Clone,
        E: Clone,
    {
        let mut result = Vec::new();
        // Stack stores (current_node, path_so_far, edge_value_leading_to_current)
        let mut stack: Vec<(&Self, Vec<(N, E)>, E)> = Vec::new();
        stack.push((self, Vec::new(), edge_value_for_self));

        let mut visited_paths: HashSet<(*const Self, Vec<(N, E)>)> = HashSet::new(); // To prevent infinite loops with cycles

        while let Some((node, mut path, edge_val_to_node)) = stack.pop() {
            let path_key = (node as *const Self, path.clone()); // Use node pointer and current path as key

             // If path already visited leading to this node, skip
             if visited_paths.contains(&path_key) {
                 continue;
             }
            visited_paths.insert(path_key);


            // Add the current node and the edge value leading to it to the path
            path.push((node.node_content.clone(), edge_val_to_node.clone()));

            if node.predecessors.is_empty() {
                // Reached a root, add the complete path
                path.reverse(); // Reverse to get root-to-leaf order
                result.push(path);
            } else {
                // Explore predecessors
                for (predecessor_wrapper, edge_val_to_node) in &node.predecessors { // edge_val_to_node is value from predecessor to current node
                    let predecessor_node = predecessor_wrapper.as_ref(); // &GSSNode<N, E>
                    stack.push((predecessor_node, path.clone(), edge_val_to_node.clone())); // Pass the edge value from pred to current
                }
            }
        }
         result.into_iter().unique().collect() // Remove duplicate paths if cycles or merging created them
    }

    // This needs adjustment since T is now (N, E) tuples in the paths.
    // pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<(N, E)>>
    // where
    //     N: Clone,
    //     E: Clone,
    // {
    //     nodes.iter().flat_map(|node| node.flatten()).collect() // Needs edge value for initial calls
    // }

    /// Merges `other` into `self`. Assumes `self.node_content == other.node_content`.
    /// Merges the predecessor maps, combining edge values using `E::merge`.
    pub fn merge(&mut self, mut other: Self)
    where
        N: PartialEq,
        E: MergeAndIntersect + Clone, // Merge requires E: MergeAndIntersect + Clone
    {
        assert!(self.node_content == other.node_content);
        for (key_wrapper, other_edge_val) in std::mem::take(&mut other.predecessors) {
            self.predecessors.entry(key_wrapper)
                .and_modify(|self_edge_val| *self_edge_val = self_edge_val.merge(&other_edge_val))
                .or_insert(other_edge_val);
        }
    }

    /// Merges predecessors from `other` into `self` without checking if `node_content` is equal.
    /// Combines edge values using `E::merge`.
    pub fn merge_unchecked(&mut self, mut other: Self)
    where
        E: MergeAndIntersect + Clone, // Merge requires E: MergeAndIntersect + Clone
    {
        for (key_wrapper, other_edge_val) in std::mem::take(&mut other.predecessors) {
            self.predecessors.entry(key_wrapper)
                .and_modify(|self_edge_val| *self_edge_val = self_edge_val.merge(&other_edge_val))
                .or_insert(other_edge_val);
        }
    }

    /// Maps the node content and edge values to new types `NewN` and `NewE`.
    pub fn map<NewN, NewE, FMapNode, FMapEdge>(&self, map_node: FMapNode, map_edge: FMapEdge) -> GSSNode<NewN, NewE>
    where
        FMapNode: Copy + Fn(&N) -> NewN,
        FMapEdge: Copy + Fn(&E) -> NewE + Clone, // Map edge needs Clone for the new edge value
        N: Clone, // Clone might be needed for recursive calls if mapping changes structure
        E: Clone, // Clone is needed for iterating predecessor edges
    {
        GSSNode {
            node_content: map_node(&self.node_content),
            predecessors: self.predecessors.iter()
                .map(|(wrapper, edge_val)| {
                    // Recursively map the predecessor node and its incoming edge value
                    let mapped_pred_node = wrapper.as_ref().map(map_node, map_edge);
                     // The edge value leading to the new node is the mapped old edge value
                    let mapped_edge_val = map_edge(edge_val);
                    (ArcPtrWrapper::new(Arc::new(mapped_pred_node)), mapped_edge_val)
                })
                .collect(),
        }
    }
}

impl<N, E> Drop for GSSNode<N, E> {
    // Custom drop to iteratively drop predecessors and break potential cycles.
    fn drop(&mut self) {
        // Take the predecessors to drop them outside of holding the mutex
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        // Worklist stores Arc to predecessor nodes
        let mut worklist: Vec<Arc<GSSNode<N, E>>> = predecessors_to_process_further.into_iter().map(|(wrapper, _edge_val)| wrapper.into_arc()).collect(); // Use into_arc, ignore edge_val

        while let Some(node_arc) = worklist.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                // Successfully got unique ownership, take predecessors and add to worklist
                worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter().map(|(wrapper, _edge_val)| wrapper.into_arc())); // Use into_arc, ignore edge_val
            }
            // Else: Arc is still shared, it will be dropped when the last ArcPtrWrapper wrapper is dropped.
        }
    }
}

// GSSTrait needs to be adapted for N and E.
// The `T` in the original trait was the node content + semantic value.
// Now `N` is node content, `E` is edge value.
// The trait should reflect operations on the GSS structure itself.
pub trait GSSTrait<N: Clone, E: Clone> {
    type Peek<'a> where N: 'a;
    fn peek_node_content(&self) -> Self::Peek<'_>; // Peek node content

    // push takes node content and edge value to the new node
    fn push(&self, node_content: N, edge_value: E) -> GSSNode<N, E>;

    // pop returns predecessors and the edge value *to* self from that predecessor
    fn pop(&self) -> Vec<(E, Arc<GSSNode<N, E>>)>;

    // popn returns nodes n steps back and the accumulated edge value along the path.
    // This needs the accumulated edge value *to* self as a starting point for accumulation.
    // Let's define popn to return (AncestorNode, AccumulatedEdgeValue from Ancestor to Self)
    fn popn(&self, n: usize, edge_value_for_self: E) -> Vec<(Arc<GSSNode<N, E>>, E)>;
}

// Implement GSSTrait for GSSNode
impl<N: Clone, E: Clone + MergeAndIntersect> GSSTrait<N, E> for GSSNode<N, E> {
    type Peek<'a> = &'a N where N: 'a;

    fn peek_node_content(&self) -> Self::Peek<'_> {
        &self.node_content
    }

    fn push(&self, node_content: N, edge_value: E) -> GSSNode<N, E> {
        // Clone self to create the predecessor Arc
        let mut new_node = GSSNode::new(node_content);
        new_node.predecessors.insert(ArcPtrWrapper::new(Arc::new(self.clone())), edge_value);
        new_node
    }

    fn pop(&self) -> Vec<(E, Arc<GSSNode<N, E>>)> {
        GSSNode::pop(self)
    }

    fn popn(&self, n: usize, edge_value_for_self: E) -> Vec<(Arc<GSSNode<N, E>>, E)> {
        // Delegate to the recursive implementation, starting from predecessors
        let mut visited_rec = HashSet::new(); // New visited set for each top-level call
        let mut memo = HashMap::new(); // New memo for each top-level call
         // The recursive popn starts from the predecessors of `self`.
         // For each predecessor, the edge value leading *to* that predecessor is the one from `self.predecessors`.
         // The edge value `edge_value_for_self` is the value *to* the node `self` is called on.

        let mut results: Vec<(Arc<GSSNode<N, E>>, E)> = Vec::new();
         for (pred_wrapper, edge_val_pred_to_self) in &self.predecessors {
             let pred_arc = pred_wrapper.as_arc().clone();
             let results_from_pred = GSSNode::popn_recursive(
                 pred_arc,
                 n - 1, // We are looking for nodes n-1 steps back from the predecessor
                 edge_val_pred_to_self.clone(), // The edge value leading *to* the predecessor is the one from pred to self? No.
                                               // The recursive call needs the edge value *into* the node it's called on.
                                               // This means the edge value *to* `pred_arc`.
                                               // This implies the edge value passed to the first recursive call should be the edge value *into* the predecessor.
                                               // But `popn_recursive` is defined to accumulate.

             // Let's redefine popn_recursive slightly:
             // popn_recursive(node_arc, steps_remaining, accumulated_t_to_node, memo, visited)
             // Base case steps_remaining == 0: return (node_arc, accumulated_t_to_node)
             // Recursive step: for each pred, call popn_recursive(pred_arc, steps-1, accumulated_t_to_node.merge(edge_val_pred_to_node), ...)

             // Initial call from `self`: The value accumulated *to* `self` is `edge_value_for_self`.
             // When we go to a predecessor, the edge value from that predecessor to `self` is `edge_val_pred_to_self`.
             // The accumulated value at the predecessor is not directly available here.
             // The recursive call needs to know the accumulated value *up to that point*.

             // Let's try the recursive call starting from the predecessors, seeking `n-1` steps, and passing the edge value *from* the predecessor *to* `self`.
             // The accumulation should happen in the recursive step.

             let results_from_pred = GSSNode::popn_recursive_accumulate(
                 pred_arc, // Start from the predecessor node
                 n - 1, // Look for nodes n-1 steps back from the predecessor
                 edge_val_pred_to_self.clone(), // Pass the edge value from pred to self for accumulation
                 &mut memo,
                 &mut visited_rec,
             );
             results.extend(results_from_pred);
         }
        results.bulk_merge(); // Merge results from different predecessor branches
        results
    }
}

// Recursive helper for popn, accumulating edge values.
// Returns Vec<(AncestorNode, AccumulatedEdgeValue from Ancestor to current node)>
fn popn_recursive_accumulate(
    node_arc: Arc<GSSNode<N, E>>,
    steps_remaining: usize,
    accumulated_edge_value_to_node: E, // Accumulated value from some ancestor *to* this node
    memo: &mut HashMap<(*const GSSNode<N, E>, E, usize), Vec<(Arc<GSSNode<N, E>>, E)>>, // (node_ptr, accumulated_E, steps_remaining)
    visited_recursion: &mut HashSet<*const GSSNode<N, E>>,
) -> Vec<(Arc<GSSNode<N, E>>, E)>
where
    N: Clone,
    E: Clone + MergeAndIntersect + Hash + Eq, // Need Hash and Eq for the memo key on E
{
    let node_ptr = Arc::as_ptr(&node_arc);
    let memo_key = (node_ptr, accumulated_edge_value_to_node.clone(), steps_remaining);

    // Check memo
    if let Some(cached_result) = memo.get(&memo_key) {
        return cached_result.clone();
    }

    // Cycle detection for the current path (based on node pointer only)
    if !visited_recursion.insert(node_ptr) {
        // Cycle detected, return empty path
        return Vec::new();
    }

    let mut results: Vec<(Arc<GSSNode<N, E>>, E)> = Vec::new();

    if steps_remaining == 0 {
        // Reached the desired number of steps back. This node is an ancestor.
        // The accumulated value is the value from some root *to* this node.
        results.push((node_arc.clone(), accumulated_edge_value_to_node));
    } else {
        // Need more steps back, recurse on predecessors
        for (pred_wrapper, edge_val_pred_to_node) in &node_arc.predecessors {
            let pred_arc = pred_wrapper.as_arc().clone();
            // Accumulate the edge value from the predecessor to the current node.
            // This is the value on the edge `pred_arc -> node_arc`.
            let new_accumulated_value = accumulated_edge_value_to_node.merge(edge_val_pred_to_node); // Merge order? Accumulate from ancestor -> ... -> pred -> node.

            // Recurse from the predecessor, with the new accumulated value.
            let results_from_pred = popn_recursive_accumulate(
                pred_arc,
                steps_remaining - 1,
                new_accumulated_value,
                memo,
                visited_recursion,
            );
            results.extend(results_from_pred);
        }
    }

    // Backtrack from recursion stack
    visited_recursion.remove(&node_ptr);

    // Store result in memo (and bulk merge before storing if needed)
    results.bulk_merge(); // Merge results reaching here from different sub-paths at this step_remaining
    memo.insert(memo_key, results.clone());

    results
}


// Implement GSSTrait for Arc<GSSNode>
impl<N: Clone, E: Clone + MergeAndIntersect + Hash + Eq> GSSTrait<N, E> for Arc<GSSNode<N, E>> {
    type Peek<'a> = &'a N where N: 'a;

    fn peek_node_content(&self) -> Self::Peek<'_> {
        &self.node_content
    }

    fn push(&self, node_content: N, edge_value: E) -> GSSNode<N, E> {
        let mut new_node = GSSNode::new(node_content);
        // The predecessor is self (the Arc)
        new_node.predecessors.insert(ArcPtrWrapper::new(self.clone()), edge_value);
        new_node
    }

    fn pop(&self) -> Vec<(E, Arc<GSSNode<N, E>>)> {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize, edge_value_for_self: E) -> Vec<(Arc<GSSNode<N, E>>, E)> {
        if n == 0 {
            // Return the current node (self) and the edge value leading to it.
            return vec![(self.clone(), edge_value_for_self)];
        }
        // Delegate to the recursive implementation, starting from self's predecessors.
        // The initial accumulated value is `edge_value_for_self`.
         let mut visited_rec = HashSet::new();
         let mut memo = HashMap::new();
         self.as_ref().popn_recursive_accumulate(
             self.clone(), // Start recursive call from self
             n, // Look for nodes N steps back from self
             edge_value_for_self, // Accumulated value up to self
             &mut memo,
             &mut visited_rec,
         )
    }
}

// Implement GSSTrait for Option<Arc<GSSNode>> (Requires N: Clone, E: Clone)
impl<N: Clone, E: Clone + MergeAndIntersect + Hash + Eq> GSSTrait<N, E> for Option<Arc<GSSNode<N, E>>> {
    type Peek<'a> = Option<&'a N> where N: 'a;

    fn peek_node_content(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek_node_content())
    }

    fn push(&self, node_content: N, edge_value: E) -> GSSNode<N, E> {
        self.clone().map(|node| node.push(node_content.clone(), edge_value.clone())).unwrap_or_else(|| GSSNode::new(node_content)) // No predecessor if Option is None
    }

    fn pop(&self) -> Vec<(E, Arc<GSSNode<N, E>>)> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize, edge_value_for_self: E) -> Vec<(Arc<GSSNode<N, E>>, E)> {
        self.as_ref().map(|node| node.popn(n, edge_value_for_self)).unwrap_or_default()
    }
}

// Implement GSSTrait for Option<GSSNode> (Requires N: Clone, E: Clone)
impl<N: Clone, E: Clone + MergeAndIntersect + Hash + Eq> GSSTrait<N, E> for Option<GSSNode<N, E>> {
    type Peek<'a> = Option<&'a N> where N: 'a;

    fn peek_node_content(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek_node_content())
    }

    fn push(&self, node_content: N, edge_value: E) -> GSSNode<N, E> {
        self.clone().map(|node| node.push(node_content.clone(), edge_value.clone())).unwrap_or_else(|| GSSNode::new(node_content)) // No predecessor if Option is None
    }

    fn pop(&self) -> Vec<(E, Arc<GSSNode<N, E>>)> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize, edge_value_for_self: E) -> Vec<(Arc<GSSNode<N, E>>, E)> {
         // Need to get an Arc from &GSSNode first to call Arc::popn
         // This is cumbersome. The primary use case should be with Arc<GSSNode>.
         // Let's make this panic or return empty for now.
         panic!("popn on &GSSNode is not properly implemented for edge values.");
    }
}

// BulkMerge now operates on Vec<Arc<GSSNode<N, E>>>
pub trait BulkMerge<N, E> {
    fn bulk_merge(&mut self);
}

impl<N: Clone + Ord, E: Clone + MergeAndIntersect> BulkMerge<N, E> for Vec<Arc<GSSNode<N, E>>> {
    fn bulk_merge(&mut self) {
        // Group nodes by their node content (N)
        let mut groups: BTreeMap<N, HashMap<*const GSSNode<N, E>, Arc<GSSNode<N, E>>>> = BTreeMap::new();
        for node_arc in self.drain(..) {
            groups.entry(node_arc.node_content.clone()).or_default().entry(Arc::as_ptr(&node_arc)).or_insert(node_arc);
        }

        self.clear(); // Clear the original vector

        for mut group in groups.into_values() {
            if group.is_empty() { continue; }

            // Convert HashMap values back to a Vec<Arc<...>> for merging
            let mut nodes_in_group: Vec<Arc<GSSNode<N, E>>> = group.into_values().collect();

            if nodes_in_group.len() <= 1 {
                // No merging needed for this group
                self.extend(nodes_in_group);
            } else {
                // Take the first node as the merge target
                let mut first_arc = nodes_in_group.pop().unwrap();
                let mut first_mut_ref = Arc::make_mut(&mut first_arc);

                // Merge predecessors from all other nodes in the group into the first
                for sibling_arc in nodes_in_group {
                    // Use merge_unchecked as we already grouped by node_content
                     first_mut_ref.merge_unchecked(Arc::unwrap_or_clone(sibling_arc));
                }
                self.push(first_arc); // Add the merged node back
            }
        }
         // Need to potentially re-sort if order matters after merge
         // This depends on how the merged nodes are used later.
         // For GLRParserState's BTreeMap keys (StateID), the order doesn't matter here.
    }
}

// Helper function for prune_and_transform_roots
// Prunes/transforms based on NodeContent (N) and EdgeValue (E) leading to the node.
// Returns Option<(TransformedNodeArc, TransformedEdgeValueToNode, ContinueRecursion)>.
// The transformed edge value is the new E value for the edge leading *to* the transformed node.
fn prune_and_transform_recursive<N: Clone, E: Clone + MergeAndIntersect>(
    node_arc: &Arc<GSSNode<N, E>>,
    edge_val_to_node: &E, // Edge value leading *to* node_arc
    closure: &impl Fn(&N, &E) -> Option<(N, E, bool)>, // Returns Option<(NewNodeContent, NewEdgeValueToNode, ContinueRecursion)>
    memo: &mut HashMap<(*const GSSNode<N, E>, E), Option<(Arc<GSSNode<N, E>>, E)>>, // Memo key includes node ptr and incoming edge value
) -> Option<(Arc<GSSNode<N, E>>, E)> { // Returns (transformed_node_arc, transformed_edge_value_to_node)
    let node_ptr = Arc::as_ptr(node_arc);
    let memo_key = (node_ptr, edge_val_to_node.clone());

    if let Some(cached_result) = memo.get(&memo_key) {
        return cached_result.clone();
    }

    match closure(&node_arc.node_content, edge_val_to_node) {
        None => {
            // Prune this path at this node
            memo.insert(memo_key, None);
            None
        }
        Some((new_node_content, new_edge_val_to_node, continue_recursion)) => {
            let mut new_predecessors: Vec<(Arc<GSSNode<N, E>>, E)> = Vec::new();

            if continue_recursion {
                // Continue recursion for predecessors
                for (pred_wrapper, edge_val_pred_to_node) in &node_arc.predecessors {
                    let pred_arc = pred_wrapper.as_arc();
                    // Recurse on predecessor, passing the edge value from pred to node
                    if let Some((new_pred_arc, new_edge_val_to_pred)) =
                        prune_and_transform_recursive(pred_arc, edge_val_pred_to_node, closure, memo)
                    {
                         // The edge from new_pred_arc to the new node will have the transformed value
                         // This seems backwards. The closure returns new_edge_val_to_node which is the value *to* the current node.
                         // The edge value stored in new_predecessors should be the one *from* the new_pred_arc *to* the new node.

                         // Let's reconsider the closure return: Option<(NewNodeContent, NewEdgeValueFromPredToNode, ContinueRecursion)>
                         // This seems more logical. The closure decides the new value for the edge it is traversing to reach the current node.

                         // Redefined Closure: Fn(&N, &E) -> Option<(N, E, bool)>
                         // Input: (&NodeContent of current node, &EdgeValue leading to current node)
                         // Return: Option<(NewNodeContent for current node, NewEdgeValue FOR THIS INCOMING EDGE, ContinueRecursion)>

                         // Let's stick to the previous definition for now and clarify.
                         // Closure: Fn(&N, &E) -> Option<(NewNodeContent, NewEdgeValue FOR THIS INCOMING EDGE, ContinueRecursion)>
                         // Input: (&NodeContent of node_arc, &edge_val_to_node)
                         // Return: Option<(NewNodeContent for node_arc, New edge_val_to_node, ContinueRecursion)>

                         // When we recurse on a predecessor `pred_arc`, the edge value leading to `pred_arc` is `edge_val_pred_to_node`.
                         // The recursive call `prune_and_transform_recursive(pred_arc, edge_val_pred_to_node, ...)` will return:
                         // Option<(NewNodeContent for pred_arc, New edge_val_pred_to_node, ContinueRecursion for pred_arc)>
                         // If it returns `Some((new_pred_content, new_edge_val_to_pred, _))`, we get the transformed predecessor Arc.
                         // The edge value from `new_pred_arc` to the new node will be `new_edge_val_to_node` from the current call's closure result.

                         // This is confusing. Let's simplify the closure's return value and focus on the edge transformation.
                         // Closure: Fn(&N, &E) -> Option<(N, E, bool)>
                         // Input: (&NodeContent, &EdgeValue leading to this node)
                         // Output: Option<(TransformedNodeContent, TransformedEdgeValue leading to this node, ContinueRecursion)>

                         // Recursive step:
                         // for (pred_wrapper, edge_val_pred_to_node) in &node_arc.predecessors {
                         //     let pred_arc = pred_wrapper.as_arc();
                         //     if let Some((transformed_pred_arc, transformed_edge_val_to_pred)) = prune_and_transform_recursive(pred_arc, edge_val_pred_to_node, closure, memo) {
                         //         // `transformed_pred_arc` is the new Arc for the predecessor node.
                         //         // `transformed_edge_val_to_pred` is the new edge value leading *to* `transformed_pred_arc`.
                         //         // We need to add an edge from `transformed_pred_arc` to the new current node.
                         //         // The value of this edge should be the `new_edge_val_to_node` returned by the closure for the current node.
                         //         new_predecessors.push((transformed_pred_arc.clone(), new_edge_val_to_node.clone()));
                         //     }
                         // }

                         // This seems more consistent. The closure decides the new value for the edge it just 'traversed' to reach the current node.

                         // Let's use this interpretation: Closure: Fn(&N, &E) -> Option<(N, E, bool)>
                         // Input: (&NodeContent of node_arc, &edge_val_to_node)
                         // Return: Option<(NewNodeContent for node_arc, New edge_val_to_node, ContinueRecursion)>

                         // Recursive step inside `continue_recursion`:
                         for (pred_wrapper, edge_val_pred_to_node) in &node_arc.predecessors {
                             let pred_arc = pred_wrapper.as_arc();
                             // Recurse on predecessor, passing the edge value from pred to node (`edge_val_pred_to_node`)
                             if let Some((transformed_pred_arc, _transformed_edge_val_to_pred)) =
                                 prune_and_transform_recursive(pred_arc, edge_val_pred_to_node, closure, memo)
                             {
                                 // We only need the transformed predecessor node Arc.
                                 // The edge value from this transformed predecessor to the new current node
                                 // is the `new_edge_val_to_node` determined by the closure for the current node.
                                 new_predecessors.push((transformed_pred_arc.clone(), new_edge_val_to_node.clone()));
                             }
                         }
                     } else {
                         // Stop recursion at this node's predecessors.
                         // Create a new node with the transformed value, but use the original predecessors
                         // and apply the new edge value (`new_edge_val_to_node`) to those edges.
                         // This doesn't seem right for pruning. If we stop recursion, it means this path is truncated.
                         // If `continue_recursion` is false, we should create a node with the new value, but it has NO predecessors. This effectively ends the path here.
                         // Let's adjust the logic for `continue_recursion = false`.

                          // If continue_recursion is false, we stop exploring predecessors *down this path*.
                          // The resulting node *should not* have predecessors from this path branch.
                          // It should have no predecessors added from this recursive call.
                     }

                     // Create the new node with the transformed content and the (potentially filtered/transformed) predecessors.
                     let transformed_node_arc = Arc::new(GSSNode::new_with_predecessors(new_node_content.clone(), new_predecessors));

                     // Store result in memo
                     memo.insert(memo_key.clone(), Some((transformed_node_arc.clone(), new_edge_val_to_node.clone())));
                     Some((transformed_node_arc, new_edge_val_to_node))
                }
        }
    }

/// Traverses the GSS forest defined by `roots` (each with an associated edge value), applying `closure` to each node and its incoming edge value.
/// Handles shared nodes using memoization. Prunes branches where `closure` returns `None`.
/// Stops recursion down a path from a node if `closure` returns `(_, _, false)`.
/// Returns a Vec of `Option<(Arc<GSSNode<N, E>>, E)>` corresponding to the input `roots`,
/// where each tuple contains the transformed root node and the transformed edge value leading to it.
pub fn prune_and_transform_roots<N: Clone, E: Clone + MergeAndIntersect>(
    roots: &[(Arc<GSSNode<N, E>>, E)], // Roots with their incoming edge values
    closure: &impl Fn(&N, &E) -> Option<(N, E, bool)>, // Returns Option<(NewNodeContent, NewEdgeValueToNode, ContinueRecursion)>
) -> Vec<Option<(Arc<GSSNode<N, E>>, E)>> {
    // We need a processing order that ensures children are processed before parents
    // if we want the early-stop optimization (`continue_recursion = false`) to work reliably
    // with shared nodes. A simple recursive approach might process shared children multiple times
    // or incorrectly reuse non-transformed predecessors.
    // For now, let's proceed with the simple recursive approach + memoization, acknowledging the
    // potential issue with the early-stop logic accuracy for shared nodes below the stop point.
    // A full topological sort or iterative approach might be needed for perfect early-stop.

    let mut memo: HashMap<(*const GSSNode<N, E>, E), Option<(Arc<GSSNode<N, E>>, E)>> = HashMap::new();
    roots
        .iter()
        .map(|(root_arc, edge_val_to_root)| prune_and_transform_recursive(root_arc, edge_val_to_root, closure, &mut memo))
        .collect()
}


// --- Longest Path ---

// Recursive helper for find_longest_path.
// Returns the longest path (Vec<Arc<GSSNode<N, E>>>) ending at node_arc, discovered so far.
// Edge values are not used in path length calculation, only node count.
fn find_longest_path_recursive<N, E>(
    node_arc: &Arc<GSSNode<N, E>>,
    memo: &mut HashMap<*const GSSNode<N, E>, Vec<Arc<GSSNode<N, E>>>>, // Stores longest path ending at the key node
    visited_recursion: &mut HashSet<*const GSSNode<N, E>>, // Detects cycles during the current DFS traversal
) -> Vec<Arc<GSSNode<N, E>>> {
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

    let mut longest_pred_path: Vec<Arc<GSSNode<N, E>>> = Vec::new();

    // Explore predecessors recursively
    if !node_arc.predecessors.is_empty() {
        for (pred_wrapper, _edge_val) in &node_arc.predecessors { // Ignore edge value for path finding
            let pred_arc = pred_wrapper.as_arc();
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
pub fn find_longest_path<N, E>(roots: &[Arc<GSSNode<N, E>>]) -> Option<Vec<Arc<GSSNode<N, E>>>> {
    let mut memo: HashMap<*const GSSNode<N, E>, Vec<Arc<GSSNode<N, E>>>> = HashMap::new();

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
pub fn gather_gss_stats<N: Clone, E: Clone>(roots: &[Arc<GSSNode<N, E>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited: HashSet<*const GSSNode<N, E>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<N, E>>, usize)> = VecDeque::new(); // (node_arc, depth)

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

        for (pred_wrapper, _edge_val) in &current_node.predecessors { // Ignore edge value for stats traversal
            let pred_arc = pred_wrapper.as_arc();
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
fn print_gss_node_recursive<N: Debug, E: Debug>(
    node_arc: &Arc<GSSNode<N, E>>,
    visited: &mut HashSet<*const GSSNode<N, E>>,
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
        writeln!(output, "{}- Node {:p} (Visited)", prefix, node_ptr)?;
        return Ok(());
    }

    visited.insert(node_ptr);
    *node_count += 1;

    // Print current node info
    writeln!(output, "{}- Node {:p}: NodeContent={:?}", prefix, node_ptr, node_arc.node_content)?;

    // Print predecessors
    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessors:", prefix)?;
        for (pred_wrapper, edge_val) in &node_arc.predecessors {
            let pred_arc = pred_wrapper.as_arc();
            writeln!(output, "{}- EdgeValue: {:?}", prefix, edge_val)?; // Print edge value
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
/// * `roots` - A slice of `Arc<GSSNode<N, E>>` representing the roots of the forest.
/// * `max_nodes` - The maximum number of unique nodes to include in the output string.
///
/// # Returns
/// A `String` containing the formatted GSS structure, potentially truncated.
pub fn print_gss_forest<N: Debug, E: Debug>(roots: &[Arc<GSSNode<N, E>>], max_nodes: usize) -> String {
    let mut visited = HashSet::new();
    let mut node_count = 0;
    let mut output = String::new();

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }

    writeln!(&mut output, "GSS Forest Roots (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        writeln!(&mut output, "Root {}:", i).unwrap();
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
    }


    output
}
