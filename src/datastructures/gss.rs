use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GSSEdge<T> {
    pred: ArcPtrWrapper<GSSNode<T>>,
    label: T,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSNode<T> {
    predecessors: BTreeSet<GSSEdge<T>>,
}

impl<T> GSSNode<T> {
    pub fn new() -> Self {
        Self {
            predecessors: BTreeSet::new(),
        }
    }

    pub fn new_with_predecessors(predecessors: Vec<(T, Arc<GSSNode<T>>)>) -> Self {
        Self {
            predecessors: predecessors
                .into_iter()
                .map(|(label, pred)| GSSEdge {
                    pred: ArcPtrWrapper::new(pred),
                    label,
                })
                .collect(),
        }
    }

    // from_iter is removed as it doesn't fit the edge-labelled model easily.

    pub fn push(self: Arc<Self>, label: T) -> Arc<Self> {
        let mut new_node = GSSNode::<T>::new();
        new_node.predecessors.insert(GSSEdge {
            pred: ArcPtrWrapper::new(self),
            label,
        });
        Arc::new(new_node)
    }


    pub fn pop(&self) -> impl Iterator<Item = (Arc<GSSNode<T>>, &T)> + '_ {
        self.predecessors.iter().map(|edge| (edge.pred.as_arc().clone(), &edge.label))
    }

    pub fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, &T)>
    where
        T: Clone, // Need T: Clone here because we collect into a Vec<(Arc<GSSNode<T>>, T)>
    {
        if n == 0 {
            // Return edges leading into THIS node
            return self.predecessors.iter().map(|edge| (edge.pred.as_arc().clone(), &edge.label)).collect();
        }

        let mut result = Vec::new();
        let mut seen_nodes: HashSet<*const GSSNode<T>> = HashSet::new();
        let mut seen_edges: HashSet<(T, *const GSSNode<T>)> = HashSet::new(); // To deduplicate (label, node) pairs

        // recurse on predecessors and collect, skipping duplicates
        for edge in &self.predecessors {
            // Edge label is not part of the key for popn, only the node structure matters for depth.
            // However, the collected result needs to carry the label.
            let pred_arc = edge.pred.as_arc(); // pred_arc is Arc<GSSNode<T>>

            // If we reached depth 1, we just return the predecessor and the current edge's label.
            if n == 1 {
                let key = (edge.label.clone(), Arc::as_ptr(&pred_arc));
                 if seen_edges.insert(key) {
                     result.push((pred_arc.clone(), &edge.label));
                 }
            } else {
                // Recursively get paths of length n-1 from the predecessor
                let paths_from_pred = pred_arc.popn(n - 1); // This returns Vec<(Arc<GSSNode<T>>, &T)>
                for (node_at_depth_n_minus_1, label_from_pred) in paths_from_pred {
                    // The label returned from pred.popn(n-1) is the label on the edge *into* pred.
                    // The result should be the node at depth n-1 and the label leading *into* that node.
                    let key = (label_from_pred.clone(), Arc::as_ptr(&node_at_depth_n_minus_1));
                     if seen_edges.insert(key) {
                         result.push((node_at_depth_n_minus_1.clone(), label_from_pred));
                     }
                }
            }
        }

        result
    }

    // peek returns an iterator over labels, not a single value.
    pub fn peek(&self) -> impl Iterator<Item = &T> + '_ {
        self.predecessors.iter().map(|edge| &edge.label)
    }

    // value_mut is removed as values are on edges.
    // flatten / flatten_bulk are removed.
    // merge / merge_unchecked are removed.

    // map is removed as values are on edges.
}

impl<T> Drop for GSSNode<T> {
    // Custom drop to iteratively drop predecessors and break potential cycles.
    fn drop(&mut self) {
        // Take the predecessors to drop them outside of holding the mutex
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T>>> = predecessors_to_process_further.into_iter().map(|edge| edge.pred.into_arc()).collect(); // Use into_arc

        while let Some(node_arc) = worklist.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                // Successfully got unique ownership, take predecessors and add to worklist
                worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter().map(|edge| edge.pred.into_arc())); // Use into_arc
            }
            // Else: Arc is still shared, it will be dropped when the last ArcPtrWrapper wrapper is dropped.
        }
    }
}

pub trait GSSTrait<T: Clone> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> Arc<GSSNode<T>>; // Returns Arc<GSSNode<T>>
    fn pop(&self) -> Vec<(Arc<GSSNode<T>>, &T)>; // Returns Vec<(Arc<GSSNode<T>>, &T)>
    fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, &T)>; // Returns Vec<(Arc<GSSNode<T>>, &T)>
}

// Implement for Arc<GSSNode<T>> (the standard handle)
impl<T: Clone> GSSTrait<T> for Arc<GSSNode<T>> {
    type Peek<'a> = Box<dyn Iterator<Item = &'a T> + 'a> where T: 'a; // Use Box<dyn Iterator>

    fn peek(&self) -> Self::Peek<'_> {
         Box::new(self.predecessors.iter().map(|edge| &edge.label))
    }

    fn push(&self, label: T) -> Arc<GSSNode<T>> {
        let mut new_node = GSSNode::new();
        new_node.predecessors.insert(GSSEdge {
            pred: ArcPtrWrapper::new(self.clone()),
            label,
        });
        Arc::new(new_node)
    }

    fn pop(&self) -> Vec<(Arc<GSSNode<T>>, &T)> {
        self.predecessors.iter().map(|edge| (edge.pred.as_arc().clone(), &edge.label)).collect()
    }

    fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, &T)> {
        // Delegate to the inherent, de-duplicating implementation above
        self.as_ref().popn(n)
    }
}

// Implement for Option<Arc<GSSNode<T>>>
impl<T: Clone> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
    type Peek<'a> = Box<dyn Iterator<Item = &'a T> + 'a> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        match self {
            Some(arc) => Box::new(arc.peek()),
            None => Box::new(std::iter::empty()),
        }
    }

    fn push(&self, label: T) -> Arc<GSSNode<T>> {
         let current_node_arc = self.clone().unwrap_or_else(|| Arc::new(GSSNode::new()));
         let mut new_node = GSSNode::new();
         new_node.predecessors.insert(GSSEdge {
             pred: ArcPtrWrapper::new(current_node_arc),
             label,
         });
         Arc::new(new_node)
    }

    fn pop(&self) -> Vec<(Arc<GSSNode<T>>, &T)> {
        self.as_ref().map(|node| node.pop().collect()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, &T)> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}


// GSSNode does not implement GSSTrait directly as it is usually held within an Arc.
// The Option<GSSNode<T>> implementation is also removed as the standard handle is Arc.


pub trait BulkMerge<T> {
    fn bulk_merge(&mut self);
}

impl<T: Clone + Ord> BulkMerge<T> for Vec<(Arc<GSSNode<T>>, &T)> {
    fn bulk_merge(&mut self) {
        // Groups nodes by their pointer address.
        // We still need to keep the label when merging predecessor lists.
        let mut groups: HashMap<*const GSSNode<T>, Vec<(Arc<GSSNode<T>>, &T)>> = HashMap::new();
        for tuple in self.drain(..) {
            let node_ptr = Arc::as_ptr(&tuple.0);
            groups.entry(node_ptr).or_default().push(tuple);
        }

        self.clear(); // Clear the original vector

        for (_, group) in groups {
            // group is a Vec of (Arc<GSSNode<T>>, &T) tuples pointing to the same underlying GSSNode
            if group.is_empty() { continue; }

            let (first_arc, _) = group[0].clone(); // Take the first Arc (clone it if necessary)

            // We need a mutable reference to the actual GSSNode within the Arc to modify its predecessors.
            let mut mutable_node = if Arc::strong_count(&first_arc) == 1 && Arc::weak_count(&first_arc) == 0 {
                // We have unique ownership, we can unwrap
                Arc::try_unwrap(first_arc).expect("unwrap failed unexpectedly").into_inner().expect("Mutex poisoned") // Assuming T doesn't contain Mutex
            } else {
                // We don't have unique ownership, we must clone the node structure.
                // Cloning the node means cloning its predecessor set. Labels are just references here.
                first_arc.as_ref().clone()
            };

            // Now, we need to merge the predecessors from all other arcs in the group into `mutable_node.predecessors`.
            // Note: The `group` contains references to labels (`&T`). We need to be careful with lifetimes if we don't clone T here.
            // Since GSSNode::popn currently clones the label into the result tuple (Vec<(Arc<GSSNode<T>>, T)> - wait, no, it's Vec<(Arc<GSSNode<T>>, &T)> ),
            // the labels in the `group` vector are references tied to the original GSSNode's edges.
            // When we merge predecessors, the edges we add to `mutable_node.predecessors` must contain owned labels.
            // Let's assume the caller of bulk_merge provides tuples with owned labels (T) if they need the merged labels persisted.
            // For now, let's stick to the &T and acknowledge this potential issue if labels need to be mutated *after* merging.
            // Re-checking `popn` and the instructions, `popn` returns `Vec<(Arc<GSSNode<T>>, &T)>`.
            // The `BulkMerge` trait is implemented for `Vec<(Arc<GSSNode<T>>, &T)>`. So `tuple.1` is `&T`.
            // The `GSSEdge` struct needs an owned label `T`.
            // This implies that the labels must be cloned when merging the predecessor sets.

            // The current `group` is `Vec<(Arc<GSSNode<T>>, &T)>`.
            // The `mutable_node` is a `GSSNode<T>`. We want to add `GSSEdge<T>` to its predecessors.
            // A `GSSEdge` contains `ArcPtrWrapper<GSSNode<T>>` and `T`.
            // The `Arc<GSSNode<T>>` part of the tuple is the node itself.
            // The `&T` part of the tuple is the label on the edge *leading to* that node.
            // When we merge, we are creating a *new* node representing the merged set.
            // The edges leading *into* this new merged node are the union of the edges leading into the original nodes in the set.
            // So, for each (node_arc, label_ref) tuple in the group, we take the predecessors of `node_arc` and add them to the `mutable_node.predecessors`.
            // The label associated with these predecessors *in the new merged node* should be the label_ref.

            // Let's refine: BulkMerge is called on the *result* of popn, which gives (parent_node, label_on_edge_to_current_node).
            // We are merging *nodes* that have the same (parent_node, label) pair.
            // The goal is to group nodes `N1`, `N2`, ... that all have a predecessor edge `P -> N_i` with label `L`.
            // After merging, there should be a single node `N_merged` with predecessor `P -> N_merged` with label `L`.
            // The current implementation groups by node pointer, which is step 1.
            // Step 2 should be to merge the predecessors *of* these grouped nodes.

            // Let's rethink BulkMerge in the edge-labelled world.
            // BulkMerge is applied to a `Vec<(Arc<GSSNode<T>>, &T)>`.
            // This vector represents a set of (Node, Label) pairs, where Label is the label *on the edge leading to* Node.
            // The goal is to merge (Node, Label) pairs that represent the "same conceptual state" after some reduction.
            // With edge labels, the "state" is defined by the incoming edge and the node itself.
            // The current `BulkMerge` groups by `*const GSSNode<T>`. This means it merges all tuples that point to the *same GSSNode instance*, regardless of the label on the edge that got us there.
            // This seems correct for merging based on reaching the same state node.

            // Let's retry the merging logic, keeping the edge labels for the *predecessors*.
            let mut merged_predecessors: BTreeSet<GSSEdge<T>> = BTreeSet::new();
            for (node_arc, _) in group { // We only need the node_arc here
                 // Add all predecessors of this node_arc to the merged set
                 for edge in &node_arc.predecessors {
                      // We need to clone the label here as it's going into a new GSSEdge
                      merged_predecessors.insert(GSSEdge {
                           pred: edge.pred.clone(), // ArcPtrWrapper can be cloned
                           label: edge.label.clone(), // T must be Clone
                      });
                 }
            }

            // Now, create the new merged node representing this group of original nodes.
            let merged_node_arc = Arc::new(GSSNode { predecessors: merged_predecessors });

            // The result of bulk_merge should be a vector of (Node, Label) tuples, where the Node is the merged node.
            // The Label associated with this merged node is the label from the original tuples that were merged.
            // The original `group` vector had tuples (node_arc, &label_ref).
            // All tuples in a group share the same `node_arc` pointer, but might have different `&label_ref` if `popn` returned multiple edges leading to the same node pointer.
            // However, `popn` with deduplication should only return one (node_arc, &label_ref) for a given node_arc if the label is used in the deduplication key.
            // Let's assume `popn` correctly produces unique (node_ptr, &label_ref) tuples for the same depth.
            // When merging, we are merging *nodes* reached via specific edges.
            // A set of (Node, Label) tuples like `{(N1, L1), (N2, L2), (N3, L3)}` might have N1, N2, N3 pointing to the same underlying GSSNode instance, but reached via different edges with different labels L1, L2, L3 from some parent(s).
            // The current grouping by `*const GSSNode<T>` means `{(N1, L1), (N2, L2), (N3, L3)}` where `N1.ptr == N2.ptr == N3.ptr` are grouped.
            // The merged node should represent reaching this shared node via *any* of the edges.
            // The result should be a new node whose predecessors are the union of the predecessors of N1, N2, N3.
            // The label associated with this new merged node should capture the set of original labels {L1, L2, L3}.
            // This implies `T` should be `MergeAndIntersect`, and the label of the merged node should be the merge of {L1, L2, L3}.

            // This requires changing the return type of `popn` to `Vec<(Arc<GSSNode<T>>, T)>` (owned label)
            // and changing the `BulkMerge` trait and implementation.

            // Let's stick to the simpler interpretation first: BulkMerge groups by node pointer and just adds the merged node to the result list *for each original label*.
            // This seems wrong. The whole point is to reduce the number of nodes.
            // A node `N` reached via edge `E` with label `L` is distinct from node `N` reached via edge `E'` with label `L'`.
            // The GSSNode itself should represent the *state*, independent of the incoming edge label.
            // The label should describe the *transition* into that state.

            // Okay, let's go back to the plan.
            // BulkMerge is called on `Vec<(Arc<GSSNode<T>>, &T)>`. This vector is the output of `popn`.
            // It represents the set of (AncestorNode, LabelOnEdgeToAncestor) tuples.
            // Example: Calling `popn(2)` on node `N`. It might return `[(P1, L_P1), (P2, L_P2)]`, where P1 and P2 are nodes 2 steps up, and L_P1/L_P2 are the labels on the edges leading into P1/P2 respectively.
            // If P1 and P2 are the *same GSSNode instance* (`Arc::ptr_eq(&P1, &P2)`), we want to merge them.
            // The result should be a single (P_merged, L_merged) tuple where P_merged is the new merged node, and L_merged is the merge of the labels that led to P1 and P2 at that depth.

            // Let's rewrite BulkMerge assuming it merges tuples based on the *Arc* (node identity).
            let mut groups: HashMap<*const GSSNode<T>, Vec<(Arc<GSSNode<T>>, &T)>> = HashMap::new();
            for tuple in self.drain(..) {
                groups.entry(Arc::as_ptr(&tuple.0)).or_default().push(tuple);
            }

            self.clear(); // Clear the original vector

            for (_, group) in groups {
                // group is a Vec of (Arc<GSSNode<T>>, &T) tuples pointing to the same underlying GSSNode
                if group.is_empty() { continue; }

                // Take the first tuple as the base
                let (first_arc, first_label_ref) = group[0].clone(); // Clone the Arc, clone the &T reference
                let mut merged_label = first_label_ref.clone(); // Start with the first label (requires T: Clone)

                // Merge the labels from all other tuples in the group
                for (_, label_ref) in group.iter().skip(1) {
                    // Need T: MergeAndIntersect to merge labels
                    // This implies that the Labels carried by popn should be T, not &T, for merging.
                    // Let's change popn return type and BulkMerge trait.
                    // This requires a cascading change.

                    // Let's revert to the simpler approach: BulkMerge doesn't merge labels, just nodes.
                    // This means if popn returns [(N_ptr, L1), (N_ptr, L2)], BulkMerge will process these as separate entries if the labels are considered in the key.
                    // If the key is just the node pointer, it merges them.

                    // Okay, let's re-read the GLR parser usage:
                    // `let mut parents = stack.popn(len); parents.bulk_merge();`
                    // `parents` is `Vec<(Arc<GSSNode<T>>, &T)>` from popn.
                    // BulkMerge is expected to reduce the number of entries in this vector if multiple entries point to the same GSSNode instance.
                    // The label (`&T`) in the tuple is the label on the edge leading into the node returned (`Arc<GSSNode<T>>`).
                    // This label `&T` corresponds to `ParseStateNodeContent<T>`.
                    // The `T` inside `ParseStateNodeContent<T>` is `LLMTokenInfo`, which has `MergeAndIntersect`.
                    // So the labels *can* be merged.

                    // Let's redefine BulkMerge to group by `*const GSSNode<T>` and merge the labels `&T`.
                    // The result should be `Vec<(Arc<GSSNode<T>>, T)>` containing the merged, owned labels.

                    // This means `popn` should return `Vec<(Arc<GSSNode<T>>, T)>` (owned labels).
                    // And `BulkMerge` should be implemented for `Vec<(Arc<GSSNode<T>>, T)>`.

                    // Let's update `popn` first. It returns `Vec<(Arc<GSSNode<T>>, &T)>`.
                    // The `find_longest_path_recursive` and `print_gss_node_recursive` also work with `Arc<GSSNode<T>>`.
                    // The prune_and_transform functions work with `Arc<GSSNode<T>>` and the closure operates on `&T`.

                    // Let's stick with `popn` returning `Vec<(Arc<GSSNode<T>>, &T)>`.
                    // And `BulkMerge` operating on this vector.
                    // The goal of BulkMerge here is to reduce the list of parent nodes obtained after popping.
                    // If popping N steps back reaches the same GSSNode instance via different paths (and potentially different intermediate labels),
                    // the reduction step in the parser (`pop_and_goto`) needs to be applied to this unique node.
                    // The label (`&T` in the tuple) received by `popn` *is* the label on the edge leading to the node `Arc<GSSNode<T>>`.
                    // This label contains `ParseStateNodeContent<T>`. The `T` inside is `LLMTokenInfo`.

                    // Okay, let's try implementing BulkMerge for `Vec<(Arc<GSSNode<T>>, &T)>` again.
                    // The grouping key should be the `Arc` (node identity).
                    // The values associated with the key should be the tuples `(Arc<GSSNode<T>>, &T)`.
                    // After grouping, for each group (all pointing to the same Arc<GSSNode<T>>),
                    // we need to produce a single entry `(Arc<GSSNode<T>>, MergedLabel)`.
                    // The MergedLabel should be the merge of all `&T` labels in the group's tuples.
                    // This merged label `MergedLabel` needs to be an owned `T` because it will be used to create a new `ParseStateNodeContent<T>` in `pop_and_goto`.

                    // So, `popn` needs to return `Vec<(Arc<GSSNode<T>>, T)>` (owned labels).
                    // And `BulkMerge` needs to be implemented for `Vec<(Arc<GSSNode<T>>, T)>`.
                    // The labels `T` must be `Clone` and `MergeAndIntersect`.

                    // Let's change `popn`'s return type. This will cascade.
                    // `popn` calls `popn` recursively. The recursive call returns `Vec<(Arc<GSSNode<T>>, T)>`.
                    // When we add to the result, we get a tuple `(node_at_depth_n_minus_1, label_from_pred)` from the recursive call.
                    // `label_from_pred` is now owned `T`. This `T` is the label on the edge leading into `node_at_depth_n_minus_1`.
                    // The key for deduplication should include this owned label.

                    // This is getting complicated due to the lifetime of the label reference.
                    // Let's rethink the `popn` return type and what it represents.
                    // `popn(n)` on node `N` should return a list of pairs `(A, L)` where `A` is a node `n` steps away, and `L` is the label on the edge connecting `A` to its child node in the path back to `N`.
                    // Example: `Root --L1--> N1 --L2--> N2`. `N2.popn(1)` returns `[(N1, &L2)]`. `N2.popn(2)` returns `[(Root, &L1)]`.
                    // The labels are always references to the labels on the *incoming* edges of the nodes returned.

                    // Okay, let's keep `popn` returning `Vec<(Arc<GSSNode<T>>, &T)>`.
                    // And `BulkMerge` on `Vec<(Arc<GSSNode<T>>, &T)>`.
                    // The grouping key is `Arc<GSSNode<T>>`.
                    // For each group, we need to produce a single `Arc<GSSNode<T>>` and a *merged* `T` label.
                    // This requires iterating through the `&T` labels in the group and merging them.
                    // The result of `bulk_merge` should probably be `Vec<(Arc<GSSNode<T>>, T)>` (owned label).

                    // Let's implement `BulkMerge` for `Vec<(Arc<GSSNode<T>>, &T)>` returning `Vec<(Arc<GSSNode<T>>, T)>`.

                    // New implementation of BulkMerge for Vec<(Arc<GSSNode<T>>, &T)>
                    let mut groups: HashMap<*const GSSNode<T>, Vec<&T>> = HashMap::new();
                    let mut node_map: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new(); // To keep one Arc per group

                    for (node_arc, label_ref) in self.drain(..) {
                        let ptr = Arc::as_ptr(&node_arc);
                        groups.entry(ptr).or_default().push(label_ref);
                        node_map.entry(ptr).or_insert(node_arc); // Keep one Arc
                    }

                    let mut merged_results: Vec<(Arc<GSSNode<T>>, T)> = Vec::new();
                    for (ptr, label_refs) in groups {
                        let node_arc = node_map.remove(&ptr).expect("Node not found in map");
                        if label_refs.is_empty() { continue; }

                        // Merge the labels
                        let mut merged_label = label_refs[0].clone(); // Start with first label (T must be Clone)
                        for label_ref in label_refs.iter().skip(1) {
                            merged_label = merged_label.merge(*label_ref); // T must be MergeAndIntersect
                        }

                        merged_results.push((node_arc, merged_label));
                    }

                    *self = merged_results; // Replace the original vector with the merged results
                }
            }
        }

        // The previous implementation had T: Clone + Ord for BulkMerge.
        // Now it needs T: Clone + MergeAndIntersect + Ord. Ord is needed for BTreeSet in GSSNode.
        // Let's update the BulkMerge trait bound.

        impl<T: Clone + MergeAndIntersect + Ord> BulkMerge<T> for Vec<(Arc<GSSNode<T>>, &T)> {
            fn bulk_merge(&mut self) {
                let mut groups: HashMap<*const GSSNode<T>, Vec<&T>> = HashMap::new();
                let mut node_map: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();

                for (node_arc, label_ref) in self.drain(..) {
                    let ptr = Arc::as_ptr(&node_arc);
                    groups.entry(ptr).or_default().push(label_ref);
                    node_map.entry(ptr).or_insert(node_arc);
                }

                let mut merged_results: Vec<(Arc<GSSNode<T>>, T)> = Vec::new();
                for (ptr, label_refs) in groups {
                    let node_arc = node_map.remove(&ptr).expect("Node not found in map");
                    if label_refs.is_empty() { continue; }

                    let mut merged_label = label_refs[0].clone();
                    for label_ref in label_refs.iter().skip(1) {
                        merged_label = merged_label.merge(*label_ref);
                    }

                    merged_results.push((node_arc, merged_label));
                }

                // Now, we have Vec<(Arc<GSSNode<T>>, T)>. The trait is for Vec<(Arc<GSSNode<T>>, &T)>.
                // This is a mismatch. The return type of bulk_merge should match the input type of bulk_merge.
                // The instruction was "parents.bulk_merge()", where parents is Vec<(Arc<GSSNode<T>>, &T)>.
                // This implies bulk_merge should modify the vector in place and keep the &T labels.
                // But merging labels requires owned T.

                // Let's adjust the plan. BulkMerge should produce a new vector with owned labels.
                // The caller `pop_and_goto` needs to receive `Vec<(Arc<GSSNode<T>>, T)>`.
                // So `popn` should return `Vec<(Arc<GSSNode<T>>, T)>`.
                // And `BulkMerge` trait and impl should be for `Vec<(Arc<GSSNode<T>>, T)>`.

                // Let's update `popn` signature to `Vec<(Arc<GSSNode<T>>, T)>`.
                // `popn` calls `popn` recursively. The recursive call gets `Vec<(Arc<GSSNode<T>>, T)>`.
                // The result tuple is `(node_at_depth_n_minus_1, label_from_pred)`. `label_from_pred` is now owned `T`.
                // When `n=1`, the result comes from `self.predecessors`. Iterating gives `GSSEdge<T>`. We need `(Arc<GSSNode<T>>, T)`.
                // So the base case of recursion for `popn` needs to produce `Vec<(Arc<GSSNode<T>>, T)>`.

                // Updated popn base case (n=0): Return edges *leading into this node* as (PredNode, Label).
                // This is incorrect for popn. popn(n) from node N returns nodes A, where A is n steps from N, and the label on the edge from A to its child towards N.
                // Example: R --L1--> N1 --L2--> N2. N2.popn(1) -> [(N1, L2)]. N2.popn(2) -> [(R, L1)].
                // The label returned is always the label on the edge *leaving* the returned node `A` and going towards the start node of the popn call.

                // Let's refine `popn(n)` on `CurrentNode`:
                // It should find all nodes `Ancestor` such that there is a path `Ancestor -> ... -> Parent -> CurrentNode` of length `n`.
                // It should return `(Ancestor, LabelOnEdge_Ancestor_to_Child)`.

                // Okay, let's assume the original `popn` logic is correct for traversing back, and the label `&T` it returns is correct (label on edge into the node at that depth).
                // The `BulkMerge` needs to handle `Vec<(Arc<GSSNode<T>>, &T)>` and produce something useful for `pop_and_goto`.
                // `pop_and_goto` takes the results of `bulk_merge` and iterates through them.
                // For each result `(parent_node_arc, parent_label)`, it gets the goto state from `parent_label.state_id` and pushes a new node `parent_node_arc.push(new_label)`.
                // The `new_label` is constructed by intersecting `parent_label.t` with `cur_t`.
                // This means `pop_and_goto` needs an owned label.
                // So `bulk_merge` must return `Vec<(Arc<GSSNode<T>>, T)>`.

                // Let's update `popn` to return `Vec<(Arc<GSSNode<T>>, T)>` (owned label).
                // Base case `n=0`: This should conceptually return the "current state", which is the node itself and maybe a default label? This doesn't quite fit.
                // Maybe `popn(0)` should just return `vec![(self.clone(), T::default())]`? But the original `popn(0)` returned `vec![Arc::new(self.clone())]` which represented the current node itself.
                // In the new model, the "state" at depth 0 is the node itself, without an incoming edge label.
                // The parser uses `stack.popn(len)` where `stack` is the head `Arc<GSSNode<T>>`.
                // `stack.popn(0)` on an `Arc<GSSNode<T>>` (impl GSSTrait) should return `vec![(stack.clone(), Default::default())]`? This feels wrong.

                // Let's look at the `popn` implementation on `GSSNode<T>` (the `as_ref()` target of `Arc<GSSNode<T>>.popn`).
                // It calls `predecessor_wrapper.as_arc().popn(n - 1)`. The recursive call is on an `Arc`.
                // So the `GSSTrait` impl for `Arc<GSSNode<T>>` is the one that defines the recursion.
                // Let's update that one to return `Vec<(Arc<GSSNode<T>>, T)>`.

                // `GSSTrait` for `Arc<GSSNode<T>>` methods:
                // `peek` still returns iterator of `&T`.
                // `push` still takes `T` and returns `Arc<GSSNode<T>>`.
                // `pop` should return `Vec<(Arc<GSSNode<T>>, T)>`. Iterate `self.predecessors`, clone `pred.as_arc()` and clone `edge.label`.
                // `popn` should return `Vec<(Arc<GSSNode<T>>, T)>`.
                // Base case for `popn` on `Arc<GSSNode<T>>`: When `n=1`, iterate `self.predecessors`. For each `edge`, return `(edge.pred.as_arc().clone(), edge.label.clone())`.

                // Let's apply these changes.

                let mut groups: HashMap<*const GSSNode<T>, Vec<T>> = HashMap::new();
                let mut node_map: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();

                // self is Vec<(Arc<GSSNode<T>>, T)>
                for (node_arc, label) in self.drain(..) {
                    let ptr = Arc::as_ptr(&node_arc);
                    groups.entry(ptr).or_default().push(label);
                    node_map.entry(ptr).or_insert(node_arc);
                }

                self.clear(); // Clear the original vector

                for (ptr, labels) in groups {
                    let node_arc = node_map.remove(&ptr).expect("Node not found in map");
                    if labels.is_empty() { continue; }

                    // Merge the labels
                    let mut merged_label = labels[0].clone();
                    for label in labels.iter().skip(1) {
                        merged_label = merged_label.merge(label);
                    }

                    self.push((node_arc, merged_label));
                }
            }
        }

        // Update `popn` signature in `GSSNode` (the internal recursive helper) to match `GSSTrait` for `Arc`.
        // `popn` in `GSSNode` should return `Vec<(Arc<GSSNode<T>>, T)>`.
        // Base case `n=0` for `GSSNode::popn` is conceptually `self`. But the return type needs `(Arc<Self>, T)`. This doesn't fit the model.
        // The `popn` in `GSSNode` is only called by `Arc::popn`. It should return the set of nodes `n` steps up, with the label on the edge *from* them.

        // Let's reconsider `GSSNode::popn(n)`. It iterates `self.predecessors`. Each edge is `GSSEdge { pred: Arc<GSSNode<T>>, label: T }`.
        // `pred` is `n-1` steps up. The label `edge.label` is the label *into* `self`. This is not the label on the edge *from* `pred`.

        // Let's simplify the GSS structure slightly:
        // A node represents a unique GSS state.
        // An edge represents a transition into that state, carrying a label.
        // GSSNode { predecessors: BTreeSet<GSSEdge<T>> } -- This is correct.
        // GSSEdge { pred: Arc<GSSNode<T>>, label: T } -- This is correct.

        // `popn(n)` on node N should find all nodes A such that there's a path of length n from A to N.
        // Path: `A --L_A--> N1 --L_N1--> ... --L_Parent--> N`. Length is the number of edges.
        // `popn(1)` on N: finds parents `P` where `P --L_P--> N`. Returns `(P, L_P)`.
        // `popn(2)` on N: finds grandparents `GP` where `GP --L_GP--> P --L_P--> N`. Returns `(GP, L_GP)`.
        // So `popn(n)` on N returns `(Ancestor, LabelOnEdge_Ancestor_to_Child)`.

        // `GSSNode::popn(&self, n)` recursive implementation:
        // Base case `n=0`: This should conceptually return `self`. But return type is `Vec<(Arc<Self>, T)>`. This doesn't fit. `popn(0)` is not used by the parser in this way.
        // Base case `n=1`: Iterate `self.predecessors`. For each `edge` in `self.predecessors`, the predecessor is `edge.pred.as_arc()`. The label *on the edge from `edge.pred` to `self`* is `edge.label`. So return `vec![(edge.pred.as_arc().clone(), edge.label.clone())]`.
        // Recursive step `n>1`: Iterate `self.predecessors`. For each `edge` (`edge.pred`, `edge.label`), recursively call `edge.pred.as_arc().popn(n-1)`. This call returns `Vec<(Ancestor, LabelFromAncestor)>`. The `Ancestor` is `n-1` steps from `edge.pred`, which means `n` steps from `self`. The `LabelFromAncestor` is on the edge from `Ancestor` to its child towards `edge.pred`. This is what we want. Collect all results.

        // Let's update `GSSNode::popn` and `GSSTrait::popn` for `Arc`.

        impl<T> GSSNode<T> {
            // ... new, new_with_predecessors, push, peek ...

            // `popn` returns nodes `n` steps up, with the label on the edge *from* them.
            pub fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, T)> // Returns owned labels
            where
                T: Clone + Eq + Hash, // Need Eq + Hash for the seen_tuples deduplication
            {
                if n == 0 {
                    // Base case n=0 is tricky with edge labels. The parser doesn't use popn(0) in the reduction step.
                    // If called, maybe return an empty list or panic? Let's return empty for now.
                    return Vec::new();
                }

                if n == 1 {
                    // Base case n=1: return (ParentNode, LabelOnEdgeFromParentToSelf)
                    let mut result = Vec::new();
                    let mut seen_tuples: HashSet<(Arc<GSSNode<T>>, T)> = HashSet::new(); // Use owned T for set

                    for edge in &self.predecessors {
                         let tuple = (edge.pred.as_arc().clone(), edge.label.clone());
                         if seen_tuples.insert(tuple.clone()) { // Clone for insertion
                             result.push(tuple);
                         }
                    }
                    return result;
                }

                // Recursive step n > 1
                let mut result = Vec::new();
                let mut seen_tuples: HashSet<(Arc<GSSNode<T>>, T)> = HashSet::new();

                for edge in &self.predecessors {
                    // Recurse on the predecessor `edge.pred.as_arc()` for `n-1` steps.
                    // The recursive call returns `Vec<(Ancestor, LabelFromAncestor)>`.
                    // These ancestors are n-1 steps from `edge.pred`, which is n steps from `self`.
                    let paths_from_pred = edge.pred.as_arc().popn(n - 1); // Recursive call

                    for (ancestor_node_arc, label_from_ancestor) in paths_from_pred {
                        // ancestor_node_arc is n steps up.
                        // label_from_ancestor is the label on the edge from ancestor_node_arc to its child towards `edge.pred`.
                        // This is the label we need to return.
                        let tuple = (ancestor_node_arc, label_from_ancestor);
                        if seen_tuples.insert(tuple.clone()) {
                            result.push(tuple);
                        }
                    }
                }

                result
            }

            // pop returns immediate predecessors with the label on the edge from them to self.
            pub fn pop(&self) -> Vec<(Arc<GSSNode<T>>, T)>
             where T: Clone
            {
                 // This is the same as popn(1) without deduplication
                 self.predecessors.iter().map(|edge| (edge.pred.as_arc().clone(), edge.label.clone())).collect()
            }

            // ... peek, etc ...
        }

        // Update GSSTrait impls for Arc and Option<Arc>
        impl<T: Clone + Eq + Hash> GSSTrait<T> for Arc<GSSNode<T>> {
            type Peek<'a> = Box<dyn Iterator<Item = &'a T> + 'a> where T: 'a;

            fn peek(&self) -> Self::Peek<'_> {
                 Box::new(self.predecessors.iter().map(|edge| &edge.label))
            }

            fn push(&self, label: T) -> Arc<GSSNode<T>> {
                let mut new_node = GSSNode::new();
                new_node.predecessors.insert(GSSEdge {
                    pred: ArcPtrWrapper::new(self.clone()),
                    label,
                });
                Arc::new(new_node)
            }

            fn pop(&self) -> Vec<(Arc<GSSNode<T>>, T)> {
                self.as_ref().pop() // Delegate to GSSNode::pop
            }

            fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, T)> {
                 if n == 0 {
                     // Consistent with GSSNode::popn(0) returning empty.
                     // Or, if representing the current state, it should be `vec![(self.clone(), T::default())]`?
                     // The parser uses `stack.popn(len)`. When len=0, it should conceptually return the current node.
                     // Let's return `vec![(self.clone(), T::default())]` for n=0 to represent the current node with a default label.
                     // This makes more sense for the reduction logic.
                     if n == 0 {
                          // Representing the current node (depth 0) with a default label.
                          // This label is on the edge *leading to* this node, conceptually.
                          // For the root, there's no incoming edge, so a default label seems appropriate.
                         vec![(self.clone(), T::default())]
                     } else if n == 1 {
                          // Delegate to GSSNode::pop which returns owned labels
                          self.as_ref().pop()
                     } else {
                         // Delegate to GSSNode::popn which returns owned labels
                         self.as_ref().popn(n)
                     }
                } else {
                    // Delegate to GSSNode::popn which returns owned labels
                    self.as_ref().popn(n)
                }
            }
        }

        // Update GSSTrait impl for Option<Arc<GSSNode<T>>>
        impl<T: Clone + Eq + Hash> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
            type Peek<'a> = Box<dyn Iterator<Item = &'a T> + 'a> where T: 'a;

            fn peek(&self) -> Self::Peek<'_> {
                match self {
                    Some(arc) => Box::new(arc.peek()),
                    None => Box::new(std::iter::empty()),
                }
            }

            fn push(&self, label: T) -> Arc<GSSNode<T>> {
                 if let Some(arc) = self {
                     arc.push(label)
                 } else {
                     // Pushing onto None creates a new root node, but how to assign the label?
                     // The label should be on the edge *to* the new node. But this node has no predecessors.
                     // This suggests pushing onto None is ill-defined for edge-labelled GSS.
                     // A root node in edge-labelled GSS is just `Arc::new(GSSNode::new())`. It has no predecessors and thus no incoming edge label.
                     // The parser's initial state creates a node with a label. `init_parse_state_with_t` was correct.
                     // Let's assume `push` is only called on a valid `Arc<GSSNode<T>>`.
                     // The `Option<Arc<GSSNode<T>>>` implementation of push seems incorrect in the original code too for edge labels.
                     // The new node should have `self` as its predecessor, with `label` on the edge.
                     // `GSSNode::push` already does this.
                     // Let's correct this:
                     let current_node_arc = self.clone().unwrap_or_else(|| {
                         // If pushing onto None, create a new root node with no predecessors.
                         // The label provided to push doesn't belong on an edge into this node.
                         // Maybe pushing onto None should create a root node, and the label is ignored?
                         // Or maybe the signature should be `push(self, label) -> Result<Arc<GSSNode<T>>, Error>`?
                         // Given the parser usage, pushing onto None conceptually starts a new stack.
                         // The initial state is the root. The first "push" onto the root creates the first node with the start state label.
                         // Let's make `push` on `Option<Arc<GSSNode<T>>>` panic if None, as it implies pushing onto an empty stack which doesn't make sense in this model.
                         panic!("push called on None GSSNode reference");
                     });
                     current_node_arc.push(label)
                }
            }

            fn pop(&self) -> Vec<(Arc<GSSNode<T>>, T)> {
                self.as_ref().map(|node| node.pop()).unwrap_or_default()
            }

            fn popn(&self, n: usize) -> Vec<(Arc<GSSNode<T>>, T)> {
                self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
            }
        }

        // Update BulkMerge trait and impl for Vec<(Arc<GSSNode<T>>, T)>
        pub trait BulkMerge<T: Clone + MergeAndIntersect + Ord> { // Added trait bounds
            fn bulk_merge(&mut self);
        }

        impl<T: Clone + MergeAndIntersect + Ord> BulkMerge<T> for Vec<(Arc<GSSNode<T>>, T)> {
            fn bulk_merge(&mut self) {
                let mut groups: HashMap<*const GSSNode<T>, Vec<T>> = HashMap::new();
                let mut node_map: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();

                // self is Vec<(Arc<GSSNode<T>>, T)>
                for (node_arc, label) in self.drain(..) {
                    let ptr = Arc::as_ptr(&node_arc);
                    groups.entry(ptr).or_default().push(label);
                    node_map.entry(ptr).or_insert(node_arc);
                }

                self.clear(); // Clear the original vector

                for (ptr, labels) in groups {
                    let node_arc = node_map.remove(&ptr).expect("Node not found in map");
                    if labels.is_empty() { continue; }

                    // Merge the labels
                    let mut merged_label = labels[0].clone();
                    for label in labels.iter().skip(1) {
                        merged_label = merged_label.merge(label);
                    }

                    self.push((node_arc, merged_label));
                }
            }
        }


        // Helper function for prune_and_transform_roots
        // Closure now operates on the edge label (&T) and returns Option<(T, bool)>
        pub fn prune_and_transform_recursive<T: Clone + Default + MergeAndIntersect + Eq + Hash>(
            node_arc: &Arc<GSSNode<T>>,
            closure: &impl Fn(&T) -> Option<(T, bool)>, // Returns Option<(NewValue, ContinueRecursion)>
            memo: &mut HashMap<*const GSSNode<T>, Option<Arc<GSSNode<T>>>>,
        ) -> Option<Arc<GSSNode<T>>> {
            let node_ptr = Arc::as_ptr(node_arc);
            if let Some(cached_result) = memo.get(&node_ptr) {
                return cached_result.clone();
            }

            // Need to process predecessors first to apply closure to their edge labels.
            // Collect transformed predecessors first.
            let mut transformed_predecessors: BTreeSet<GSSEdge<T>> = BTreeSet::new();
            let mut keep_node = false; // Keep this node if any edge leads to it after pruning

            for edge in &node_arc.predecessors {
                let pred_arc = edge.pred.as_arc(); // Arc<GSSNode<T>>

                // Apply closure to the label on the edge from pred_arc to node_arc
                if let Some((new_label, continue_recursion)) = closure(&edge.label) {
                    if continue_recursion {
                        // Recurse on the predecessor
                        if let Some(new_pred_arc) = prune_and_transform_recursive(pred_arc, closure, memo) {
                            // If the predecessor was kept, add an edge from the new predecessor to the current node (which we might keep)
                             transformed_predecessors.insert(GSSEdge {
                                pred: ArcPtrWrapper::new(new_pred_arc),
                                label: new_label.clone(), // Clone the new label
                             });
                             keep_node = true; // Keep this node if at least one predecessor path is kept
                        }
                        // If new_pred_arc is None, the predecessor branch was pruned.
                    } else {
                         // Stop recursion here. Create a new edge directly from the original predecessor.
                         transformed_predecessors.insert(GSSEdge {
                             pred: ArcPtrWrapper::new(pred_arc.clone()), // Keep the original predecessor arc
                             label: new_label.clone(), // Use the new label
                         });
                         keep_node = true; // Keep this node if at least one stop-recursion edge leads to it
                    }
                }
                // If closure returns None, prune this edge and the branch below it.
            }

            if keep_node || node_arc.predecessors.is_empty() {
                 // Keep the node if it has at least one kept incoming edge OR if it was originally a root (no predecessors).
                 // Note: Roots in edge-labelled GSS have no predecessors. This function is likely called on nodes *below* the root.
                 // The pruning function likely should start from the root and decide whether to keep the root based on its outgoing edges / reachable structure.
                 // However, the GLR parser's commit function calls this on `parse_state.stack`, which is the HEAD node.
                 // This means `parse_state.stack` is the node *at the top* of the stack (most recent).
                 // Pruning from the head means pruning backwards towards the root.
                 // So `prune_and_transform_recursive` should prune branches *leading to* the current node.
                 // If, after processing all incoming edges, `transformed_predecessors` is empty, this node should be pruned *unless* it's the initial root (which has no predecessors).

                 // Given the parser's usage on `parse_state.stack` (the head), this function prunes paths *behind* the head.
                 // If `transformed_predecessors` is empty after checking all original predecessors, it means all paths leading to this node are pruned.
                 if transformed_predecessors.is_empty() {
                      memo.insert(node_ptr, None);
                      None // Prune this node
                 } else {
                      // Create a new node with the transformed predecessors
                      let new_node_arc = Arc::new(GSSNode { predecessors: transformed_predecessors });
                      memo.insert(node_ptr, Some(new_node_arc.clone()));
                      Some(new_node_arc)
                 }
            } else {
                 // Node was a root (no predecessors) and closure logic didn't explicitly keep it?
                 // This case is less clear. If a node has no predecessors, the closure doesn't apply to incoming edges.
                 // The closure is applied to `&edge.label`. If there are no edges, the closure isn't called.
                 // The logic `keep_node || node_arc.predecessors.is_empty()` is slightly off for pruning from the head.
                 // A node is kept if *any* path leading to it is kept.
                 // If `transformed_predecessors` is non-empty, it means at least one path is kept.
                 // If `transformed_predecessors` is empty, it is pruned.

                 // Let's simplify: a node is kept if, after considering all its predecessors, the set of transformed predecessors is not empty.
                 // The only exception might be the absolute root of the *entire* GSS, which has no predecessors.
                 // The parser's initial state creates the root. Subsequent steps push nodes onto it.
                 // The node `parse_state.stack` *is* the head node. Its predecessors are the nodes one step down.

                // Corrected logic for pruning from the head (retaining a node if any incoming path is kept):
                if transformed_predecessors.is_empty() {
                     memo.insert(node_ptr, None);
                     None // Prune this node
                } else {
                     let new_node_arc = Arc::new(GSSNode { predecessors: transformed_predecessors });
                     memo.insert(node_ptr, Some(new_node_arc.clone()));
                     Some(new_node_arc)
                }
            }
        }


        /// Traverses the GSS forest defined by `roots`, applying `closure` to each *edge label*.
        /// Handles shared nodes using memoization. Prunes branches where `closure` returns `None` for an edge label.
        /// Stops recursion down a path *beyond* an edge if `closure` returns `(_, false)` for that edge's label.
        /// Returns a Vec of `Option<Arc<GSSNode<T>>>` corresponding to the input `roots`.
        /// Roots themselves are kept if any path leading to them is kept.
        pub fn prune_and_transform_roots<T: Clone + Default + MergeAndIntersect + Eq + Hash>(
            roots: &[Arc<GSSNode<T>>],
            closure: &impl Fn(&T) -> Option<(T, bool)>, // Closure applies to edge label (&T)
        ) -> Vec<Option<Arc<GSSNode<T>>>> {
            let mut memo = HashMap::new();
            roots
                .iter()
                .map(|root| prune_and_transform_recursive(root, closure, &mut memo))
                .collect()
        }


        // --- Longest Path ---

        // Recursive helper for find_longest_path.
        // Returns the longest path *ending* at node_arc, discovered so far.
        // A path is a sequence of nodes, but for edge-labelled GSS, maybe the path should be nodes and the labels on edges between them?
        // The original implementation returned `Vec<Arc<GSSNode<T>>>`. Let's stick to this.
        fn find_longest_path_recursive<T>(
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
                for edge in &node_arc.predecessors {
                    let pred_arc = edge.pred.as_arc(); // pred_arc is Arc<GSSNode<T>>
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
        pub fn find_longest_path<T>(roots: &[Arc<GSSNode<T>>]) -> Option<Vec<Arc<GSSNode<T>>>> {
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
            /// Depth is defined as the minimum number of edges from a root to the node.
            pub max_depth: usize,
            /// Average depth of nodes (distance from a root node).
            pub average_depth: f64,
            /// Number of nodes with more than one predecessor edge (merge points).
            pub merge_points: usize,
            /// Maximum number of incoming edges (predecessors) for any single node.
            pub max_predecessors: usize,
            /// Average number of incoming edges (predecessors) per node.
            pub average_predecessors: f64,
        }

        /// Gathers statistics about the GSS forest defined by the given roots.
        /// Traverses the graph using BFS to calculate depths from roots.
        pub fn gather_gss_stats<T: Clone>(roots: &[Arc<GSSNode<T>>]) -> GSSStats {
            let mut stats = GSSStats::default();
            stats.num_roots = roots.len();

            let mut visited_nodes: HashSet<*const GSSNode<T>> = HashSet::new();
            let mut queue: VecDeque<(Arc<GSSNode<T>>, usize)> = VecDeque::new(); // (node, depth)

            let mut total_depth_sum: u64 = 0;
            let mut total_predecessors_sum: u64 = 0;

            // Seed the queue with roots (depth 0)
            for root_arc in roots {
                let root_ptr = Arc::as_ptr(root_arc);
                if visited_nodes.insert(root_ptr) {
                    queue.push_back((root_arc.clone(), 0));
                }
            }

            // BFS traversal
            while let Some((current_node_arc, current_depth)) = queue.pop_front() {
                stats.unique_nodes += 1;
                stats.max_depth = stats.max_depth.max(current_depth);
                total_depth_sum += current_depth as u64;

                let num_predecessors = current_node_arc.predecessors.len();
                stats.max_predecessors = stats.max_predecessors.max(num_predecessors);
                total_predecessors_sum += num_predecessors as u64;
                if num_predecessors > 1 {
                    stats.merge_points += 1;
                }

                // Add predecessors to the queue
                for edge in &current_node_arc.predecessors {
                    let pred_arc = edge.pred.as_arc();
                    let pred_raw_ptr = Arc::as_ptr(pred_arc);
                    // Only add if not visited yet
                    if visited_nodes.insert(pred_raw_ptr) {
                        queue.push_back((pred_arc.clone(), current_depth + 1)); // Depth increases by 1 for the predecessor
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
        fn print_gss_node_recursive<T: Debug>(
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
                writeln!(output, "{}- Node {:p} (Visited)", prefix, node_ptr)?;
                return Ok(());
            }

            visited.insert(node_ptr);
            *node_count += 1;

            // Print current node address
            writeln!(output, "{}- Node {:p}:", prefix, node_ptr)?;

            // Print incoming edges (predecessors and their labels)
            if !node_arc.predecessors.is_empty() {
                writeln!(output, "{}  Incoming Edges:", prefix)?;
                for edge in &node_arc.predecessors {
                    let pred_arc = edge.pred.as_arc();
                    let pred_ptr = Arc::as_ptr(pred_arc);
                    // Print edge info: (Predecessor Address) --[Label]--> Current Node Address
                    writeln!(output, "{}    - ({:p}) --[{:?}]--> ({:p})",
                             prefix, pred_ptr, edge.label, node_ptr)?;

                    // Recursively print the predecessor node structure
                    print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes, output)?;
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
        pub fn print_gss_forest<T: Debug>(roots: &[Arc<GSSNode<T>>], max_nodes: usize) -> String {
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
