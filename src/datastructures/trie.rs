use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};

use deterministic_hash::DeterministicHasher;
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use profiler_macro::time_it;

use crate::datastructures::arena::{Arena, NodeId};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::PROGRESS_BAR_ENABLED;
use kdam::tqdm;

/// Error type indicating that a cycle was detected during an operation
/// that updates graph structure or properties like max_depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

impl fmt::Display for CycleDetectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle detected in Trie structure")
    }
}

impl Error for CycleDetectedError {}

/// Represents a node within a `Trie`.
#[derive(Debug, Clone)]
pub struct TrieNode<EK: Ord, EV, T> {
    pub value: T,
    /// Stores a map from EdgeKey to a map of destination node IDs and edge values.
    pub children: BTreeMap<EK, OrderedHashMap<NodeId, EV>>,
    /// The "longest distance" from some source node.
    pub max_depth: usize,
}

/// A Trie-like data structure that allows shared subtrees (DAGs).
/// It owns all nodes in an arena and provides methods to manipulate the graph.
#[derive(Debug, Clone)]
pub struct Trie<EK: Ord, EV, T> {
    pub arena: Arena<TrieNode<EK, EV, T>>,
    pub root_id: NodeId,
}

impl<EK, EV, T> JSONConvertible for TrieNode<EK, EV, T>
where
    EK: Ord + Clone + JSONConvertible,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        JSONNode::Object(BTreeMap::from([
            ("value".to_string(), self.value.to_json()),
            ("children".to_string(), self.children.to_json()),
            ("max_depth".to_string(), self.max_depth.to_json()),
        ]))
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let value = obj
                    .remove("value")
                    .ok_or("Missing 'value' field")
                    .and_then(T::from_json)?;
                let children = obj
                    .remove("children")
                    .ok_or("Missing 'children' field")
                    .and_then(|n| BTreeMap::<EK, OrderedHashMap<NodeId, EV>>::from_json(n))?;
                let max_depth = obj
                    .remove("max_depth")
                    .ok_or("Missing 'max_depth' field")
                    .and_then(usize::from_json)?;
                Ok(TrieNode {
                    value,
                    children,
                    max_depth,
                })
            }
            _ => Err("Expected JSONNode::Object for TrieNode".to_string()),
        }
    }
}

impl<EK, EV, T> JSONConvertible for Trie<EK, EV, T>
where
    EK: Ord + Clone + JSONConvertible,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        JSONNode::Object(BTreeMap::from([
            ("arena".to_string(), self.arena.to_json()),
            ("root_id".to_string(), self.root_id.to_json()),
        ]))
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let arena = obj
                    .remove("arena")
                    .ok_or("Missing 'arena' field")
                    .and_then(|n| Arena::<TrieNode<EK, EV, T>>::from_json(n))?;
                let root_id = obj
                    .remove("root_id")
                    .ok_or("Missing 'root_id' field")
                    .and_then(NodeId::from_json)?;
                Ok(Trie { arena, root_id })
            }
            _ => Err("Expected JSONNode::Object for Trie".to_string()),
        }
    }
}

impl<EK: Ord + Clone, EV, T> Trie<EK, EV, T> {
    /// Creates a new Trie with a single root node.
    pub fn new(value: T) -> Self {
        let mut arena = Arena::new();
        let root_node = TrieNode {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        };
        let root_id = arena.alloc(root_node);
        Trie { arena, root_id }
    }

    /// Returns the `NodeId` of the root node.
    pub fn root_id(&self) -> NodeId {
        self.root_id
    }

    /// Returns a reference to a `TrieNode` in the arena.
    pub fn get_node(&self, id: NodeId) -> &TrieNode<EK, EV, T> {
        self.arena.get(id)
    }

    /// Returns a mutable reference to a `TrieNode` in the arena.
    fn get_node_mut(&mut self, id: NodeId) -> &mut TrieNode<EK, EV, T> {
        self.arena.get_mut(id)
    }

    /// Creates a new node in the trie and returns its `NodeId`.
    pub fn create_node(&mut self, value: T) -> NodeId {
        self.arena.alloc(TrieNode {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        })
    }

    /// Inserts an edge without checking for cycles or updating depths. Use with caution.
    pub fn force_insert(&mut self, src_id: NodeId, edge_key: EK, edge_value: EV, dst_id: NodeId) {
        self.get_node_mut(src_id)
            .children
            .entry(edge_key)
            .or_default()
            .insert(dst_id, edge_value);
    }

    /// Attempts to insert an edge, checking for cycles and propagating depth updates.
    #[time_it]
    pub fn try_insert(
        &mut self,
        src_id: NodeId,
        edge_key: EK,
        edge_value: EV,
        dst_id: NodeId,
    ) -> Result<(), CycleDetectedError> {
        if self.detect_cycle(src_id, dst_id) {
            return Err(CycleDetectedError);
        }

        let src_depth = self.get_node(src_id).max_depth;
        let candidate_depth = src_depth.saturating_add(1);

        let needs_update = {
            let dst_node = self.get_node(dst_id);
            candidate_depth > dst_node.max_depth
        };

        if needs_update {
            self.get_node_mut(dst_id).max_depth = candidate_depth;
            self.propagate_max_depth(dst_id)?;
        }

        self.get_node_mut(src_id)
            .children
            .entry(edge_key)
            .or_default()
            .insert(dst_id, edge_value);

        Ok(())
    }

    /// Returns `true` if `target_id` is reachable from `start_id`.
    #[time_it]
    pub fn detect_cycle(&self, target_id: NodeId, start_id: NodeId) -> bool {
        if target_id == start_id {
            return true;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        if visited.insert(start_id) {
            queue.push_back(start_id);
        }

        while let Some(node_id) = queue.pop_front() {
            if node_id == target_id {
                return true;
            }
            let node = self.get_node(node_id);
            for dest_map in node.children.values() {
                for &child_id in dest_map.keys() {
                    if visited.insert(child_id) {
                        queue.push_back(child_id);
                    }
                }
            }
        }
        false
    }

    /// Propagates a max_depth update to all descendant nodes.
    fn propagate_max_depth(&mut self, start_id: NodeId) -> Result<(), CycleDetectedError> {
        let mut rec_stack: HashSet<NodeId> = HashSet::new();
        self._propagate_max_depth(start_id, &mut rec_stack)
    }

    fn _propagate_max_depth(
        &mut self,
        node_id: NodeId,
        rec_stack: &mut HashSet<NodeId>,
    ) -> Result<(), CycleDetectedError> {
        if !rec_stack.insert(node_id) {
            return Err(CycleDetectedError);
        }

        let current_depth = self.get_node(node_id).max_depth;
        let candidate_child_depth = current_depth.saturating_add(1);

        let children_to_update: Vec<NodeId> = {
            let node = self.get_node(node_id);
            node.children
                .values()
                .flat_map(|dest_map| dest_map.keys())
                .filter(|&&child_id| self.get_node(child_id).max_depth < candidate_child_depth)
                .cloned()
                .collect()
        };

        for child_id in children_to_update {
            self.get_node_mut(child_id).max_depth = candidate_child_depth;
            self._propagate_max_depth(child_id, rec_stack)?;
        }

        rec_stack.remove(&node_id);
        Ok(())
    }

    /// Collects all *unique* nodes reachable from the given roots (BFS).
    pub fn all_nodes(&self, roots: &[NodeId]) -> Vec<NodeId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        for &root_id in roots {
            if visited.insert(root_id) {
                queue.push_back(root_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);
            let node = self.get_node(node_id);
            for children_map in node.children.values() {
                for &child_id in children_map.keys() {
                    if visited.insert(child_id) {
                        queue.push_back(child_id);
                    }
                }
            }
        }
        result
    }

    /// Checks if there are any cycles reachable from the given `root_id`.
    pub fn has_any_cycle(&self, root_id: NodeId) -> bool {
        let mut global_visited: HashSet<NodeId> = HashSet::new();
        let mut recursion_stack: HashSet<NodeId> = HashSet::new();
        self._has_any_cycle_recursive(root_id, &mut global_visited, &mut recursion_stack)
    }

    fn _has_any_cycle_recursive(
        &self,
        node_id: NodeId,
        global_visited: &mut HashSet<NodeId>,
        recursion_stack: &mut HashSet<NodeId>,
    ) -> bool {
        if recursion_stack.contains(&node_id) {
            return true;
        }
        if global_visited.contains(&node_id) {
            return false;
        }

        recursion_stack.insert(node_id);
        global_visited.insert(node_id);

        let children_ids: Vec<NodeId> = self
            .get_node(node_id)
            .children
            .values()
            .flat_map(|dest_map| dest_map.keys())
            .cloned()
            .collect();

        for child_id in children_ids {
            if self._has_any_cycle_recursive(child_id, global_visited, recursion_stack) {
                return true;
            }
        }

        recursion_stack.remove(&node_id);
        false
    }

    /// Recomputes `max_depth` for all nodes reachable from the given roots.
    pub fn recompute_all_max_depths(&mut self, roots: &[NodeId]) {
        let all_node_ids = self.all_nodes(roots);
        if all_node_ids.is_empty() {
            return;
        }

        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &node_id in &all_node_ids {
            in_degree.entry(node_id).or_insert(0);
            adj.entry(node_id).or_default();
            let node = self.get_node(node_id);
            for child_id in node.children.values().flat_map(|m| m.keys()) {
                adj.entry(node_id).or_default().push(*child_id);
                *in_degree.entry(*child_id).or_default() += 1;
            }
        }

        let mut queue = VecDeque::new();
        for &node_id in &all_node_ids {
            if in_degree.get(&node_id).cloned().unwrap_or(0) == 0 {
                queue.push_back(node_id);
                self.get_node_mut(node_id).max_depth = 0;
            } else {
                self.get_node_mut(node_id).max_depth = 0;
            }
        }

        while let Some(u_id) = queue.pop_front() {
            let u_depth = self.get_node(u_id).max_depth;
            if let Some(children_ids) = adj.get(&u_id) {
                for &v_id in children_ids {
                    let v_node = self.get_node_mut(v_id);
                    v_node.max_depth = v_node.max_depth.max(u_depth + 1);

                    let v_in_degree = in_degree.get_mut(&v_id).unwrap();
                    *v_in_degree -= 1;
                    if *v_in_degree == 0 {
                        queue.push_back(v_id);
                    }
                }
            }
        }
    }
}

impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie<EK, EV, T> {
    /// Performs a specialized breadth-first traversal.
    #[time_it]
    pub fn special_map<V: Clone>(
        &self,
        initial_nodes_and_values: Vec<(NodeId, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &TrieNode<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
    ) {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped_nodes: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        let total_edges: usize = self.all_nodes(&initial_nodes_and_values.iter().map(|(id, _)| *id).collect::<Vec<_>>())
            .iter()
            .map(|&id| self.get_node(id).children.values().map(|dests| dests.len()).sum::<usize>())
            .sum();

        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        for (node_id, v0) in initial_nodes_and_values {
            values
                .entry(node_id)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = self.get_node(node_id).max_depth;
            todo.entry(depth).or_default().insert(node_id);
        }

        while let Some((_depth, node_ids)) = todo.pop_first() {
            for &node_id in &node_ids {
                if stopped_nodes.contains(&node_id) {
                    continue;
                }

                let mut agg_v = match values.remove(&node_id) {
                    Some(v) => v,
                    None => continue,
                };

                let node = self.get_node(node_id);
                let proceed = process(node, &mut agg_v);

                if !proceed {
                    stopped_nodes.insert(node_id);
                    continue;
                }

                let edges: Vec<(EK, EV, NodeId)> = node
                    .children
                    .iter()
                    .flat_map(|(ek, dst_map)| {
                        dst_map
                            .iter()
                            .map(move |(&child_id, ev)| (ek.clone(), ev.clone(), child_id))
                    })
                    .collect();

                for (ek, ev, child_id) in edges {
                    let _ = pb.update(1);
                    if stopped_nodes.contains(&child_id) {
                        continue;
                    }

                    let child_node = self.get_node(child_id);
                    if let Some(new_v) = step(&agg_v, &ek, &ev, child_node) {
                        values
                            .entry(child_id)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        let child_depth = self.get_node(child_id).max_depth;
                        todo.entry(child_depth).or_default().insert(child_id);
                    }
                }
            }
        }
    }

    /// Performs a specialized breadth-first traversal, grouping children by edge key.
    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        &self,
        initial_nodes_and_values: Vec<(NodeId, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
    ) where
        V: Clone,
        S: FnMut(&V, &EK, &OrderedHashMap<NodeId, EV>) -> I,
        I: IntoIterator<Item = (NodeId, V)>,
    {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped_nodes: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        let total_edges: usize = self.all_nodes(&initial_nodes_and_values.iter().map(|(id, _)| *id).collect::<Vec<_>>())
            .iter()
            .map(|&id| self.get_node(id).children.values().map(|dests| dests.len()).sum::<usize>())
            .sum();
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        for (node_id, v0) in initial_nodes_and_values {
            values
                .entry(node_id)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = self.get_node(node_id).max_depth;
            todo.entry(depth).or_default().insert(node_id);
        }

        while let Some((_depth, node_ids)) = todo.pop_first() {
            for &node_id in &node_ids {
                if stopped_nodes.contains(&node_id) {
                    continue;
                }

                let mut agg_v = match values.remove(&node_id) {
                    Some(v) => v,
                    None => continue,
                };

                let node = self.get_node(node_id);
                let proceed = process(node, &mut agg_v);

                if !proceed {
                    stopped_nodes.insert(node_id);
                    continue;
                }

                let children_by_ek: Vec<(EK, OrderedHashMap<NodeId, EV>)> = node
                    .children
                    .iter()
                    .map(|(ek, dst_map)| (ek.clone(), dst_map.clone()))
                    .collect();

                for (ek, dest_map) in children_by_ek {
                    let _ = pb.update(dest_map.len());
                    let new_values_for_children = step(&agg_v, &ek, &dest_map);

                    for (child_id, new_v) in new_values_for_children {
                        if stopped_nodes.contains(&child_id) {
                            continue;
                        }

                        values
                            .entry(child_id)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        let child_depth = self.get_node(child_id).max_depth;
                        todo.entry(child_depth).or_default().insert(child_id);
                    }
                }
            }
        }
    }
}

impl<EK, EV, T> PartialEq for Trie<EK, EV, T>
where
    EK: Ord,
    EV: PartialEq,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        let mut cache = HashMap::new();
        self.compare_nodes_recursive(self.root_id, other, other.root_id, &mut cache)
    }
}

impl<EK, EV, T> Eq for Trie<EK, EV, T>
where
    EK: Ord,
    EV: Eq,
    T: Eq,
{
}

impl<EK, EV, T> Hash for Trie<EK, EV, T>
where
    EK: Ord + Hash,
    EV: Hash,
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut cache = HashMap::new();
        self.hash_node_recursive(self.root_id, state, &mut cache);
    }
}

impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord,
    EV: PartialEq,
    T: PartialEq,
{
    fn compare_nodes_recursive(
        &self,
        self_id: NodeId,
        other: &Self,
        other_id: NodeId,
        cache: &mut HashMap<(NodeId, NodeId), bool>,
    ) -> bool {
        let (id1, id2) = if self_id < other_id {
            (self_id, other_id)
        } else {
            (other_id, self_id)
        };
        if let Some(&result) = cache.get(&(id1, id2)) {
            return result;
        }
        cache.insert((id1, id2), true); // Assume true for cycles

        let self_node = self.get_node(self_id);
        let other_node = other.get_node(other_id);

        if self_node.value != other_node.value
            || self_node.max_depth != other_node.max_depth
            || self_node.children.len() != other_node.children.len()
        {
            cache.insert((id1, id2), false);
            return false;
        }

        for (self_ek, self_dest_map) in &self_node.children {
            if let Some(other_dest_map) = other_node.children.get(self_ek) {
                if self_dest_map.len() != other_dest_map.len() {
                    cache.insert((id1, id2), false);
                    return false;
                }
                let mut other_pairs: Vec<_> = other_dest_map.iter().collect();
                for (&self_child_id, self_ev) in self_dest_map {
                    let mut found_match = false;
                    for i in 0..other_pairs.len() {
                        let (&other_child_id, other_ev) = other_pairs[i];
                        if self_ev == *other_ev {
                            if self.compare_nodes_recursive(
                                self_child_id,
                                other,
                                other_child_id,
                                cache,
                            ) {
                                other_pairs.remove(i);
                                found_match = true;
                                break;
                            }
                        }
                    }
                    if !found_match {
                        cache.insert((id1, id2), false);
                        return false;
                    }
                }
            } else {
                cache.insert((id1, id2), false);
                return false;
            }
        }
        true
    }
}

impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Hash,
    EV: Hash,
    T: Hash,
{
    fn hash_node_recursive<H: Hasher>(
        &self,
        id: NodeId,
        state: &mut H,
        cache: &mut HashMap<NodeId, u64>,
    ) {
        if cache.contains_key(&id) {
            id.hash(state);
            return;
        }
        cache.insert(id, 0); // Placeholder for cycles

        let node = self.get_node(id);
        node.value.hash(state);
        node.max_depth.hash(state);

        let mut edge_hashes = Vec::new();
        for (ek, dest_map) in &node.children {
            for (&child_id, ev) in dest_map {
                let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                ek.hash(&mut pair_hasher);
                ev.hash(&mut pair_hasher);
                self.hash_node_recursive(child_id, &mut pair_hasher, cache);
                edge_hashes.push(pair_hasher.finish());
            }
        }
        edge_hashes.sort_unstable();
        for h in edge_hashes {
            h.hash(state);
        }
    }
}

// -----------------------------------------------------------------------------
// TESTS
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datastructures::hybrid_bitset::HybridBitset;

    type TestTrie = Trie<&'static str, &'static str, i32>;

    #[test]
    fn test_insertion_and_retrieval() {
        let mut trie = TestTrie::new(0);
        let child1_id = trie.create_node(1);
        let child2_id = trie.create_node(2);
        let child3_id = trie.create_node(3);
        let root_id = trie.root_id();

        trie.try_insert(root_id, "a", "edge_a1", child1_id).unwrap();
        trie.try_insert(root_id, "b", "edge_b", child2_id).unwrap();
        trie.try_insert(root_id, "a", "edge_a3", child3_id).unwrap();

        let root_node = trie.get_node(root_id);
        let children_a = root_node.children.get("a").unwrap();
        assert_eq!(children_a.len(), 2);
        assert_eq!(children_a.get(&child1_id), Some(&"edge_a1"));
        assert_eq!(children_a.get(&child3_id), Some(&"edge_a3"));

        let children_b = root_node.children.get("b").unwrap();
        assert_eq!(children_b.len(), 1);
        assert_eq!(children_b.get(&child2_id), Some(&"edge_b"));

        assert!(trie.get_node(child1_id).children.is_empty());
        assert!(trie.is_leaf(child1_id));
        assert!(!trie.is_leaf(root_id));
    }

    #[test]
    fn test_cycle_detection_on_try_insert() {
        let mut trie = TestTrie::new(0);
        let root_id = trie.root_id();
        let child_id = trie.create_node(1);

        trie.try_insert(root_id, "r->c", "e1", child_id).unwrap();
        assert_eq!(trie.get_node(child_id).max_depth, 1);

        let result = trie.try_insert(child_id, "c->r", "e2", root_id);
        assert_eq!(result, Err(CycleDetectedError));

        let child_node = trie.get_node(child_id);
        assert!(!child_node.children.contains_key(&"c->r"));
    }

    #[test]
    fn test_all_nodes_diamond() {
        let mut trie = TestTrie::new(0);
        let root_id = trie.root_id();
        let child1_id = trie.create_node(1);
        let child2_id = trie.create_node(2);
        let grandchild_id = trie.create_node(3);

        trie.force_insert(root_id, "r1", "e1", child1_id);
        trie.force_insert(root_id, "r2", "e2", child2_id);
        trie.force_insert(child1_id, "c1", "e3", grandchild_id);
        trie.force_insert(child2_id, "c2", "e4", grandchild_id);

        let all_nodes = trie.all_nodes(&[root_id]);
        assert_eq!(all_nodes.len(), 4);
        let node_ids: HashSet<_> = all_nodes.into_iter().collect();
        assert!(node_ids.contains(&root_id));
        assert!(node_ids.contains(&child1_id));
        assert!(node_ids.contains(&child2_id));
        assert!(node_ids.contains(&grandchild_id));
    }

    #[test]
    fn test_has_any_cycle() {
        let mut trie = TestTrie::new(0);
        let root_id = trie.root_id();
        let child_id = trie.create_node(1);
        trie.force_insert(root_id, "r->c", "e1", child_id);
        assert!(!trie.has_any_cycle(root_id));
        trie.force_insert(child_id, "c->r", "e2", root_id);
        assert!(trie.has_any_cycle(root_id));
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        let mut trie = TestTrie::new(0);
        let root_id = trie.root_id();
        let child1_id = trie.create_node(1);
        let child2_id = trie.create_node(2);
        let grandchild_id = trie.create_node(3);

        trie.try_insert(root_id, "r->c1", "edge1", child1_id).unwrap();
        trie.try_insert(root_id, "r->c2", "edge2", child2_id).unwrap();
        trie.try_insert(child1_id, "c1->gc", "edge3", grandchild_id).unwrap();
        trie.try_insert(child2_id, "c2->gc", "edge4", grandchild_id).unwrap();

        let mut processed_nodes = HashMap::new();
        trie.special_map(
            vec![(root_id, 100)],
            |p_val, _, _, _| Some(p_val + 1),
            |current_v, new_v| *current_v = (*current_v).max(new_v),
            |node, final_v| {
                processed_nodes.insert(node.value, *final_v);
                true
            },
        );

        assert_eq!(processed_nodes.len(), 4);
        assert_eq!(processed_nodes.get(&0), Some(&100));
        assert_eq!(processed_nodes.get(&1), Some(&101));
        assert_eq!(processed_nodes.get(&2), Some(&101));
        assert_eq!(processed_nodes.get(&3), Some(&102));
    }

    #[test]
    fn test_equality_and_hash() {
        let mut trie1 = TestTrie::new(0);
        let c1_1 = trie1.create_node(1);
        trie1.try_insert(trie1.root_id(), "a", "e1", c1_1).unwrap();

        let mut trie2 = TestTrie::new(0);
        let c1_2 = trie2.create_node(1);
        trie2.try_insert(trie2.root_id(), "a", "e1", c1_2).unwrap();

        assert_eq!(trie1, trie2);

        let mut hasher1 = DefaultHasher::new();
        trie1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        trie2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);

        // Change trie2
        let c2_2 = trie2.create_node(2);
        trie2.try_insert(trie2.root_id(), "b", "e2", c2_2).unwrap();
        assert_ne!(trie1, trie2);
    }

    #[test]
    fn test_json_serialization() {
        let mut trie = TestTrie::new(0);
        let c1 = trie.create_node(1);
        let c2 = trie.create_node(2);
        trie.try_insert(trie.root_id(), "a", "e1", c1).unwrap();
        trie.try_insert(c1, "b", "e2", c2).unwrap();

        let json = trie.to_json();
        let trie_deserialized = TestTrie::from_json(json).unwrap();

        assert_eq!(trie, trie_deserialized);
    }

    #[test]
    fn test_new_api_for_merging_edge() {
        type TestTrieEI = Trie<&'static str, HybridBitset, String>;
        let mut trie = TestTrieEI::new("source".to_string());
        let root_id = trie.root_id();
        let dest_id = trie.create_node("dest".to_string());

        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        trie.try_insert(root_id, "key", initial_edge_val, dest_id).unwrap();

        // Now, merge a new value.
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        let root_node = trie.get_node_mut(root_id);
        let dest_map = root_node.children.entry("key").or_default();
        if let Some(existing_ev) = dest_map.get_mut(&dest_id) {
            *existing_ev |= new_edge_val;
        } else {
            // This branch won't be taken in this test
        }

        let final_ev = trie.get_node(root_id).children.get("key").unwrap().get(&dest_id).unwrap();
        assert_eq!(*final_ev, merged_edge_val);
    }

    #[test]
    fn test_new_api_for_create_fallback() {
        type TestTrieEI = Trie<&'static str, HybridBitset, String>;
        let mut trie = TestTrieEI::new("source".to_string());
        let root_id = trie.root_id();
        let dest1_id = trie.create_node("dest1".to_string());

        // Setup: dest1 -> source creates a cycle
        trie.force_insert(dest1_id, "d1->s", HybridBitset::zeros(), root_id);

        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Try to insert to dest1, which should fail
        let result = trie.try_insert(root_id, "key", new_edge_val.clone(), dest1_id);
        assert!(result.is_err());

        // Fallback: create a new node
        let fallback_id = trie.create_node("fallback".to_string());
        trie.try_insert(root_id, "key", new_edge_val.clone(), fallback_id).unwrap();

        let root_node = trie.get_node(root_id);
        let children = root_node.children.get("key").unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children.get(&fallback_id), Some(&new_edge_val));
        assert_eq!(trie.get_node(fallback_id).value, "fallback");
    }
}