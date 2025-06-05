use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use deterministic_hash::DeterministicHasher;
use std::any::{Any, TypeId};

use crate::glr::parser::ParseStateEdgeContent;
use crate::constraint::{LLMTokenBV};
use crate::datastructures::gss::acc_mod::Acc;

// Type aliases for cleaner signatures, now concrete
type NodeCache = HashMap<NodeMap, Arc<GSSNode>>;
type NodeMap = BTreeMap<ParseStateEdgeContent, Arc<GSSNode>>;
type NodeSet = BTreeSet<(Arc<GSSNode>, ParseStateEdgeContent)>;

pub type LLMTokenInfo = Option<LLMTokenBV>;

pub trait PathAccumulator: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash {
    fn union_assign(&mut self, other: Self);
    fn intersect_assign(&mut self, right: Self); // Renamed from pop_assign
    fn union(mut self, other: Self) -> Self {
        self.union_assign(other);
        self
    }
    fn intersect(mut self, right: Self) -> Self { // Renamed from pop
        self.intersect_assign(right);
        self
    }
    fn intersect_has_effect(&self, right: &Self) -> bool;
}

impl PathAccumulator for () {
    fn union_assign(&mut self, _other: Self) { }
    fn intersect_assign(&mut self, _right: Self) { } // Renamed from pop_assign
    fn intersect_has_effect(&self, _right: &Self) -> bool { false }
}

impl PathAccumulator for Option<LLMTokenBV> {
    fn union_assign(&mut self, other: Self) {
        match (self.as_mut(), other) {
            (Some(self_bv), Some(other_bv)) => {
                *self_bv |= &other_bv;
                // An empty bitset resulting from a union is still Some(empty_bv), not None.
            }
            (None, Some(other_bv)) => {
                *self = Some(LLMTokenBV::max_ones());
            }
            (Some(_), None) => {
                *self = Some(LLMTokenBV::max_ones());
            }
            (None, None) => {
                // self remains None
            }
        }
    }

    fn intersect_assign(&mut self, right: Self) {
        match (self.as_mut(), right) {
            (Some(self_bv), Some(right_bv)) => {
                *self_bv &= right_bv;
            }
            (None, Some(right_bv)) => {
                *self = Some(right_bv);
            }
            (Some(_), None) => {}
            (None, None) => {}
        }
    }

    fn intersect_has_effect(&self, right: &Self) -> bool {
        // self.clone().intersect(right.clone()) != *self
        match (self, right) {
            (Some(self_bv), Some(right_bv)) => {
                self_bv.is_subset(right_bv)
            }
            (None, Some(right_bv)) => {
                true
            }
            (Some(_), None) => {
                false
            }
            (None, None) => {
                false
            }
        }
    }
}

fn compute_hash_key(predecessors: &NodeMap) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    for (edge_val, pred_arc) in predecessors {
        edge_val.hash(&mut hasher);
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}

pub mod acc_mod {
    use std::collections::{BTreeMap, BTreeSet};
    use crate::constraint::LLMTokenBV;
    use crate::datastructures::gss::{LLMTokenInfo, PathAccumulator};
    use crate::tokenizer::TokenizerStateID;
    use crate::types::TerminalID;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Acc {
        acc: LLMTokenInfo,
        forbidden_terminals: BTreeMap<TokenizerStateID, BTreeSet<TerminalID>>,
    }

    impl Acc {
        pub fn new(acc: LLMTokenInfo, forbidden_terminals: BTreeMap<TokenizerStateID, BTreeSet<TerminalID>>) -> Self {
            Self { acc, forbidden_terminals }
        }

        pub fn new_for_merging() -> Self {
            Self { acc: Some(LLMTokenBV::new()), forbidden_terminals: BTreeMap::new() }
        }

        pub fn acc(&self) -> &LLMTokenInfo {
            &self.acc
        }

        pub fn acc_mut(&mut self) -> &mut LLMTokenInfo {
            &mut self.acc
        }

        pub fn forbidden_terminals(&self) -> &BTreeMap<TokenizerStateID, BTreeSet<TerminalID>> {
            &self.forbidden_terminals
        }

        pub fn is_default(&self) -> bool {
            self.acc.is_none() && self.forbidden_terminals.is_empty()
        }

        pub fn is_dead(&self) -> bool {
            self.acc.clone().is_none_or(|bv| bv.is_empty())
        }

        pub fn is_alive(&self) -> bool {
            !self.is_dead()
        }
    }

    impl PathAccumulator for Acc {
        fn union_assign(&mut self, other: Self) {
            self.acc.union_assign(other.acc);
            for (tokenizer_state_id, other_terminals) in other.forbidden_terminals {
                todo!()
            }
        }
        fn intersect_assign(&mut self, right: Self) {
            self.acc.intersect_assign(right.acc);
            for (tokenizer_state_id, other_terminals) in right.forbidden_terminals {
                todo!()
            }
        }
        fn intersect_has_effect(&self, right: &Self) -> bool {
            self.acc.intersect_has_effect(&right.acc)
        }
    }

    impl Default for Acc {
        fn default() -> Self {
            Self::new(None, BTreeMap::new())
        }
    }
}

#[derive(Debug, Clone)]
pub struct GSSNode {
    acc: acc_mod::Acc,
    predecessors: NodeMap,
    hash_key_cache: u64,
}

#[derive(Clone)]
pub struct PathsIter<'a> { // No longer generic
    queue: VecDeque<(&'a GSSNode, Vec<ParseStateEdgeContent>)>,
}

impl<'a> Iterator for PathsIter<'a> { // No longer generic
    type Item = Vec<ParseStateEdgeContent>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((current_node, mut path_suffix)) = self.queue.pop_front() {
            if current_node.predecessors.is_empty() {
                path_suffix.reverse();
                return Some(path_suffix);
            }

            for (edge_val, pred_arc) in &current_node.predecessors {
                let mut new_path = path_suffix.clone();
                new_path.push(edge_val.clone());
                self.queue.push_back((pred_arc.as_ref(), new_path));
            }
        }
        None
    }
}

fn process_predecessors(
    incoming: &NodeSet
) -> NodeMap {
    let mut grouped: BTreeMap<ParseStateEdgeContent, Vec<Arc<GSSNode>>> = BTreeMap::new();
    for (pred_arc, edge_val) in incoming {
        grouped.entry(edge_val.clone()).or_default().push(pred_arc.clone());
    }

    let mut result = NodeMap::new();
    for (edge_val, pred_arcs) in grouped {
        if pred_arcs.is_empty() { continue; }

        let mut iter = pred_arcs.into_iter();
        let first = iter.next().unwrap(); // Safe due to is_empty check

        if iter.len() == 0 { // Only one predecessor for this edge value
            result.insert(edge_val, first);
        } else { // Multiple predecessors for this edge value, merge them
            let mut merged_node_data = (*first).clone(); // Clone the GSSNode data
            for other_arc in iter {
                merged_node_data.merge(&other_arc); // Merge other GSSNode data into it
            }
            result.insert(edge_val, Arc::new(merged_node_data));
        }
    }
    result
}

// Basic node creation and manipulation
impl GSSNode {
    pub fn new(acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let hash_key_cache = compute_hash_key(&predecessors);
        Self { acc, predecessors, hash_key_cache }
    }
    
    // Private constructor used by simplification and other internal methods
    fn new_with_map(acc: Acc, predecessors: NodeMap) -> Self {
        let hash_key_cache = compute_hash_key(&predecessors);
        Self { acc, predecessors, hash_key_cache }
    }

    // Helper to create a GSSNode with a single predecessor, used by push.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        predecessors_map.insert(edge_value, predecessor_arc);
        Self::new_with_map(acc, predecessors_map)
    }

    fn predecessors_with_values(&self) -> impl IntoIterator<Item = (&Arc<Self>, &ParseStateEdgeContent)> {
        self.predecessors.iter().map(|(edge_val, pred_arc)| (pred_arc, edge_val))
    }

    fn predecessors(&self) -> &NodeMap {
        &self.predecessors
    }

    pub fn num_predecessors(&self) -> usize {
        self.predecessors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predecessors.is_empty()
    }

    pub fn acc_acc(&self) -> &LLMTokenInfo {
        &self.acc.acc()
    }

    pub fn acc_acc_mut(&mut self) -> &mut LLMTokenInfo {
        self.acc.acc_mut()
    }

    pub fn acc2(&self) -> &Acc {
        &self.acc
    }

    pub fn acc_mut2(&mut self) -> &mut Acc {
        &mut self.acc
    }

    // Helper to clone the node and set a new accumulator. Used internally.
    fn with_acc(mut self, acc: Acc) -> Self {
        self.acc = acc;
        self.hash_key_cache = compute_hash_key(&self.predecessors); // Recalculate hash if acc changes meaning
        self
    }
}


// Core manipulation methods
impl GSSNode {
    // Push now takes the acc for the new node
    pub fn push(self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self), edge_value, acc_for_new_node)
    }
    
    // pop_into is complex with private acc_mut, might need rethink or careful internal use
    // For now, assume pop() and popn() are the main public interfaces for this.
    // If pop_into is essential, it would need to return a new Self or take &mut Self and manage acc carefully.

    pub fn pop(&self) -> Self {
        let mut result_acc = Acc::new_for_merging();
        let mut result_predecessors = NodeMap::new();

        for (pred_arc, _edge_val) in self.predecessors_with_values() {
            // The acc of the path *through* self to pred_arc is self.acc intersected with pred_arc.acc
            let path_acc = self.acc.clone().intersect(pred_arc.acc.clone());
            result_acc.union_assign(path_acc.clone()); // Union accs of all popped paths

            // Merge predecessors of pred_arc into result_predecessors
            // Each merged predecessor needs its acc updated based on path_acc
            for (inner_edge, inner_pred_arc) in &pred_arc.predecessors {
                let mut new_inner_pred_node_data = (**inner_pred_arc).clone();
                new_inner_pred_node_data.acc = path_acc.clone().intersect(inner_pred_arc.acc.clone());

                match result_predecessors.entry(inner_edge.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(Arc::new(new_inner_pred_node_data));
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        Arc::make_mut(entry.get_mut()).merge(&Arc::new(new_inner_pred_node_data));
                    }
                }
            }
        }
        Self::new_with_map(result_acc, result_predecessors)
    }


    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            self.clone()
        } else {
            self.pop().popn(n - 1)
        }
    }

    pub fn pop_iter(&self) -> Vec<(Arc<Self>, ParseStateEdgeContent)> {
        self.predecessors.iter().map(|(edge_val, pred_arc)| {
            let mut pred_arc = pred_arc.clone();
            // The acc for the path ending at pred_arc (after popping self)
            // is self.acc intersected with pred_arc's original acc.
            if self.acc.intersect_has_effect(&pred_arc.acc) {
                let path_acc = self.acc.clone().intersect(pred_arc.acc.clone());
                pred_arc = Arc::new(pred_arc.as_ref().clone().with_acc(path_acc));
            }
            (pred_arc, edge_val.clone())
        }).collect()
    }

    // Internal helper, needs careful handling due to private acc_mut
    fn push_down_acc(&mut self) {
        for pred_arc_val in self.predecessors.values_mut() { // Renamed pred_arc
            let mut_pred_node = Arc::make_mut(pred_arc_val);
            mut_pred_node.acc.intersect_assign(self.acc.clone());
            // After modifying acc, hash_key_cache might need update if acc is part of it.
            // Current compute_hash_key does not include self.acc, only predecessors.
            // However, if the acc of a predecessor changes, its own hash_key_cache changes.
            mut_pred_node.hash_key_cache = compute_hash_key(&mut_pred_node.predecessors);
        }
        // self.hash_key_cache = compute_hash_key(&self.predecessors); // Recompute for self too
    }

    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        self.acc.union_assign(other.acc.clone());

        for (edge_val, other_pred_arc) in &other.predecessors {
            match self.predecessors.entry(edge_val.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(other_pred_arc.clone());
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    Arc::make_mut(entry.get_mut()).merge(other_pred_arc);
                }
            }
        }
        self.hash_key_cache = compute_hash_key(&self.predecessors);
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn iter_paths(&self) -> PathsIter<'_> {
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new()));
        PathsIter { queue }
    }

    pub fn flatten(&self) -> Vec<Vec<(ParseStateEdgeContent, LLMTokenInfo)>> {
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new()));

        while let Some((node, mut path)) = queue.pop_front() {
            if node.predecessors.is_empty() {
                path.reverse();
                results.push(path);
            } else {
                for (edge_val, pred_arc) in &node.predecessors {
                    let mut new_path = path.clone();
                    new_path.push((edge_val.clone(), node.acc.acc().clone()));
                    queue.push_back((pred_arc.as_ref(), new_path));
                }
            }
        }
        results
    }

    pub fn flatten_bulk(nodes: &[Arc<Self>]) -> Vec<Vec<(ParseStateEdgeContent, LLMTokenInfo)>> {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    // map method is complex with non-generic GSSNode. If needed, it would be specific.
    // For now, let's assume it's not immediately required for this refactoring.
}

// Trait implementations
impl Hash for GSSNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
        self.acc.hash(state); // Accumulator should be part of the hash for equality
    }
}

impl PartialEq for GSSNode {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (
            self.hash_key_cache == other.hash_key_cache && // Structural hash
            self.acc == other.acc && // Accumulator equality
            self.predecessors == other.predecessors // Deep predecessor equality
        )
    }
}

impl Eq for GSSNode {}

impl PartialOrd for GSSNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); }
        // Order by hash_key_cache, then acc, then predecessors
        self.hash_key_cache.partial_cmp(&other.hash_key_cache)
            .and_then(|ord| if ord == Ordering::Equal { self.acc.partial_cmp(&other.acc) } else { Some(ord) })
            .and_then(|ord| if ord == Ordering::Equal { self.predecessors.partial_cmp(&other.predecessors) } else { Some(ord) })
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

impl Drop for GSSNode {
    fn drop(&mut self) {
        // Custom drop logic to break cycles if Arcs are used internally in a complex way.
        // Standard Arc drop should handle most cases unless there are self-referential Arcs
        // not managed by the main GSS structure (which shouldn't be the case here).
        // The current predecessor map uses Arc, so standard drop is likely sufficient.
        // The previous custom drop logic was to manually traverse and break cycles
        // if Arc::try_unwrap could be used. This is complex and error-prone.
        // Relying on Arc's standard drop is safer unless specific cycle issues are proven.
    }
}

// Simplified trait for GSS operations
pub trait GSSTrait { // No longer generic
    fn push(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> GSSNode;
    // push_to is removed as it's complex with private acc_mut and less idiomatic with Arc.
    fn pop(&self) -> GSSNode;
    fn popn(&self, n: usize) -> GSSNode;
}

impl GSSTrait for GSSNode {
    fn push(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> GSSNode {
        self.clone().push(edge_value, acc_for_new_node)
    }

    fn pop(&self) -> GSSNode {
        GSSNode::pop(self)
    }

    fn popn(&self, n: usize) -> GSSNode {
        GSSNode::popn(self, n)
    }
}

impl GSSTrait for Arc<GSSNode> {
    fn push(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> GSSNode {
        GSSNode::new_with_single_predecessor(self.clone(), edge_value, acc_for_new_node)
    }

    fn pop(&self) -> GSSNode {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize) -> GSSNode {
        self.as_ref().popn(n)
    }
}

// Removed GSSTrait for Option<Arc<GSSNode>> and Option<GSSNode> for brevity,
// can be added back if specific use cases require them.

// Pruning and Transformation
fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    closure: &impl Fn(&Acc) -> Option<(Acc, bool)>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.acc2()) {
        None => { // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_acc, continue_recursion)) => {
            let new_predecessors_set = if continue_recursion {
                node_arc.predecessors.iter()
                    .filter_map(|(edge_val, pred_arc_val)| { // Renamed pred_arc
                        prune_and_transform_recursive(pred_arc_val, closure, memo)
                            .map(|new_pred_arc| (new_pred_arc, edge_val.clone())) // Renamed new_pred
                    })
                    .collect::<NodeSet>() // Explicit type for collect
            } else { // Don't recurse, keep existing predecessors but point to original Arcs
                node_arc.predecessors_with_values().into_iter()
                    .map(|(pred_arc_val, edge_val)| (pred_arc_val.clone(), edge_val.clone())) // Renamed pred_arc
                    .collect::<NodeSet>() // Explicit type for collect
            };

            // Create a new node with the transformed accumulator and new predecessors
            // GSSNode::new_with_predecessors computes its own acc by union. We want new_acc.
            let new_node_predecessors_map = process_predecessors(&new_predecessors_set);
            let transformed_node = GSSNode::new_with_map(new_acc, new_node_predecessors_map);
            
            let result_arc = Arc::new(transformed_node);
            memo.insert(node_ptr, Some(result_arc.clone()));
            Some(result_arc)
        }
    }
}


pub fn intersect_tokens_and_prune_arc(root_arc: &mut Arc<GSSNode>, tokens_to_intersect: &LLMTokenBV) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        if let Some(bv) = new_acc.acc_mut() {
            *bv &= tokens_to_intersect;
        } else {
            new_acc = Acc::new(Some(tokens_to_intersect.clone()), current_acc.forbidden_terminals().clone());
        }
        if new_acc.is_alive() {
            Some((new_acc, false))
        } else {
            None // Prune this node
        }
    };

    let mut memo = HashMap::new();
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, &mut memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn subtract_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    llm_tokens: &LLMTokenBV,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        if let Some(bv) = new_acc.acc_mut() {
            *bv -= llm_tokens;
        } else {
            new_acc = Acc::new(Some(LLMTokenBV::max_ones() - llm_tokens.clone()), current_acc.forbidden_terminals().clone());
        }
        if new_acc.acc().clone().is_none_or(|bv| !bv.is_empty()) {
            Some((new_acc, false))
        } else {
            None // Prune this node
        }
    };
    let mut memo = HashMap::new();
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, &mut memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn reset_tokens(root_arc: &mut Arc<GSSNode>) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let continue_recursion = current_acc.acc().is_some();
        Some((Acc::new(None, current_acc.forbidden_terminals().clone()), continue_recursion)) // Keep node, continue recursion
    };
    let mut memo = HashMap::new();
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, &mut memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn find_longest_path(
    root_node: &GSSNode
) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    if root_node.predecessors.is_empty() {
        return None;
    }

    fn find_longest_recursive(
        node_arc: &Arc<GSSNode>,
        memo: &mut HashMap<*const GSSNode, Vec<(ParseStateEdgeContent, Arc<GSSNode>)>>,
        visited: &mut HashSet<*const GSSNode>,
    ) -> Vec<(ParseStateEdgeContent, Arc<GSSNode>)> {
        let node_ptr = Arc::as_ptr(node_arc);

        if let Some(cached) = memo.get(&node_ptr) {
            return cached.clone();
        }
        if !visited.insert(node_ptr) { // Cycle detected
            return Vec::new();
        }

        if node_arc.predecessors.is_empty() { // Base case: leaf node in recursion
            visited.remove(&node_ptr);
            memo.insert(node_ptr, Vec::new());
            return Vec::new();
        }

        let mut longest = Vec::new();
        for (edge_val, pred_arc_val) in &node_arc.predecessors { // Renamed pred_arc
            let mut path = find_longest_recursive(pred_arc_val, memo, visited);
            path.push((edge_val.clone(), node_arc.clone())); // Path stores (edge, child_node_it_points_to)
            if path.len() > longest.len() {
                longest = path;
            }
        }

        memo.insert(node_ptr, longest.clone());
        visited.remove(&node_ptr);
        longest
    }

    let mut memo = HashMap::new();
    let mut longest_overall_path = Vec::new(); // Initialize with an empty path

    // The root_node itself is the start of paths, its predecessors are the first step.
    // The path should be from a leaf up to the direct children of root_node.
    for (edge_val, pred_arc) in root_node.predecessors() {
        let mut visited_for_this_branch = HashSet::new();
         // Path from a leaf up to pred_arc
        let mut path_to_pred = find_longest_recursive(pred_arc, &mut memo, &mut visited_for_this_branch);
        path_to_pred.push((edge_val.clone(), Arc::new(root_node.clone()))); // Add the step from pred_arc to root_node

        if path_to_pred.len() > longest_overall_path.len() {
            longest_overall_path = path_to_pred;
        }
    }
    if longest_overall_path.is_empty() { None } else { Some(longest_overall_path) }
}

impl GSSNode {
    pub fn prune_and_transform_recursive(
        &mut self,
        closure: &impl Fn(&Acc) -> Option<(Acc, bool)>,
        memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
    ) {
        let node_arc = Arc::new(self.clone());
        if let Some(new_node_arc) = prune_and_transform_recursive(&node_arc, closure, memo) {
            *self = new_node_arc.as_ref().clone();
        } else {
            *self = GSSNode::new(self.acc2().clone());
        }
    }

    pub fn intersect_tokens_and_prune_arc(
        &mut self,
        llm_tokens: &LLMTokenBV,
    ) {
        let mut node_arc = Arc::new(self.clone());
        intersect_tokens_and_prune_arc(&mut node_arc, &llm_tokens);
        *self = node_arc.as_ref().clone();
    }

    pub fn subtract_tokens_and_prune_arc(
        &mut self,
        llm_tokens: &LLMTokenBV,
    ) {
        let mut node_arc = Arc::new(self.clone());
        subtract_tokens_and_prune_arc(&mut node_arc, &llm_tokens);
        *self = node_arc.as_ref().clone();
    }

    pub fn reset_tokens(&mut self) {
        let mut node_arc = Arc::new(self.clone());
        reset_tokens(&mut node_arc);
        *self = node_arc.as_ref().clone();
    }

    pub fn find_longest_path(&self) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
        find_longest_path(&self)
    }
}

#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    pub num_roots: usize,
    pub unique_nodes: usize,
    pub max_depth: usize,
    pub average_depth: f64,
    pub merge_points: usize,
    pub max_predecessors_with_values: usize,
    pub average_predecessors_with_values: f64,
}

pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats { // Takes slice of references
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited_pointers = HashSet::new(); // To track unique nodes by pointer
    let mut processed_pointers = HashSet::new(); // For BFS traversal
    let mut queue = VecDeque::new();
    let mut total_depth = 0u64;
    let mut total_preds = 0u64;

    for root_node_ref in roots { // Renamed root to root_node_ref
        let node_ptr = *root_node_ref as *const GSSNode;
        if visited_pointers.insert(node_ptr) { // Check against visited_pointers for uniqueness
            queue.push_back((*root_node_ref, 0)); // Push the reference and depth
        }
    }
    stats.unique_nodes = visited_pointers.len(); // Initial unique nodes are the unique roots

    // Reset visited_pointers for BFS traversal if we want to count all reachable nodes
    // Or, ensure the queue only gets truly unique items.
    // The current logic for unique_nodes might be off if roots share children.
    // Let's refine:
    visited_pointers.clear(); // Clear for BFS count
    stats.unique_nodes = 0; // Reset unique_nodes for BFS count

    let mut bfs_queue = VecDeque::new();
    for root_node_ref in roots {
        let node_ptr = *root_node_ref as *const GSSNode;
        if !processed_pointers.contains(&node_ptr) { // Ensure each root starts BFS once
             bfs_queue.push_back((*root_node_ref, 0));
             processed_pointers.insert(node_ptr); // Mark as added to queue
        }
    }
    processed_pointers.clear(); // Clear for actual processing check

    while let Some((node, depth)) = bfs_queue.pop_front() {
        let node_ptr = node as *const GSSNode;
        if !visited_pointers.insert(node_ptr) { // If already visited and processed by BFS
            continue;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        total_depth += depth as u64;

        let num_preds = node.predecessors.len();
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds);
        total_preds += num_preds as u64;

        let unique_pred_arcs: HashSet<_> = node.predecessors.values()
            .map(|arc_val| Arc::as_ptr(arc_val)) // Renamed arc
            .collect();
        if unique_pred_arcs.len() > 1 && num_preds > 1 { // A merge point has multiple distinct predecessor nodes
            stats.merge_points += 1;
        }

        for (_, pred_arc_val) in &node.predecessors { // Renamed pred_arc
            let pred_ptr = pred_arc_val.as_ref() as *const GSSNode;
             // Add to queue if not yet added for BFS processing from any path
            if !processed_pointers.contains(&pred_ptr) {
                bfs_queue.push_back((pred_arc_val.as_ref(), depth + 1));
                processed_pointers.insert(pred_ptr);
            }
        }
    }


    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }
    stats
}


pub fn print_gss_forest(
    roots: &[Arc<GSSNode>], 
    max_nodes: usize
) -> String {
    fn print_node_recursive( // Renamed print_node to print_node_recursive
        node_arc: &Arc<GSSNode>,
        visited: &mut HashSet<*const GSSNode>,
        indent: usize,
        node_count: &mut usize,
        max_nodes: usize,
        output: &mut String,
    ) -> Result<(), std::fmt::Error> {
        if *node_count >= max_nodes {
            return Ok(());
        }

        let node_ptr = Arc::as_ptr(node_arc);
        let prefix = format!("{:indent$}", "", indent = indent * 2);

        if visited.contains(&node_ptr) {
            writeln!(output, "{}- Node {:p} (Visited)", prefix, node_ptr)?;
            return Ok(());
        }

        visited.insert(node_ptr);
        *node_count += 1;

        writeln!(output, "{}- Node {:p}: (acc_mod::Acc: {:?})", prefix, node_ptr, node_arc.acc.acc())?;

        if !node_arc.predecessors.is_empty() {
            writeln!(output, "{}  Predecessors:", prefix)?;
            for (edge_val, pred_arc_val) in &node_arc.predecessors { // Renamed pred_arc
                writeln!(output, "{}    - Edge: {:?} -> {:p}", prefix, edge_val, Arc::as_ptr(pred_arc_val))?;
                if *node_count < max_nodes {
                    print_node_recursive(pred_arc_val, visited, indent + 2, node_count, max_nodes, output)?;
                }
                if *node_count >= max_nodes {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    let mut visited_nodes = HashSet::new(); // Renamed visited
    let mut count = 0; // Renamed node_count
    let mut out_str = String::new(); // Renamed output

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }

    writeln!(&mut out_str, "GSS Forest (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root_arc_val) in roots.iter().enumerate() { // Renamed root
        writeln!(&mut out_str, "Root {}: {:p}", i, Arc::as_ptr(root_arc_val)).unwrap();
        if print_node_recursive(root_arc_val, &mut visited_nodes, 1, &mut count, max_nodes, &mut out_str).is_err() {
            return format!("Error writing GSS structure");
        }
        if count >= max_nodes && i < roots.len() - 1 {
            writeln!(&mut out_str, "... (Truncated)").unwrap();
            break;
        }
    }

    out_str
}

// Simplification methods
// This is the main simplification routine. It uses a cache for structural sharing.
fn simplify_node_recursive(
    node_arc: &Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>, // Memoizes input Arc raw pointer to simplified Arc
    cache: &mut NodeCache, // Cache for structural sharing: NodeMap -> Arc<GSSNode>
) -> Arc<GSSNode> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(simplified_arc) = memo.get(&node_ptr) { // Renamed simplified
        return simplified_arc.clone();
    }

    // Recursively simplify predecessors
    let simplified_predecessors_set: NodeSet = node_arc.predecessors.iter()
        .map(|(edge_val, pred_arc_val)| { // Renamed pred_arc
            let simplified_pred_arc = simplify_node_recursive(pred_arc_val, memo, cache); // Renamed simplified_pred
            (simplified_pred_arc, edge_val.clone())
        })
        .collect();
    
    let simplified_predecessors_map = process_predecessors(&simplified_predecessors_set);

    // Get a structurally canonical Arc from the cache, or create and insert it.
    // The acc of this cached_structural_node is the union of its predecessors' accs.
    let cached_structural_node = cache.entry(simplified_predecessors_map.clone())
        .or_insert_with(|| {
            let unioned_acc = if simplified_predecessors_map.is_empty() {
                Acc::new_for_merging()
            } else {
                let mut iter = simplified_predecessors_map.values();
                let mut acc = iter.next().unwrap().acc2().clone();
                for p_arc in iter { // Renamed p
                    acc.union_assign(p_arc.acc2().clone());
                }
                acc
            };
            Arc::new(GSSNode::new_with_map(unioned_acc, simplified_predecessors_map))
        });

    // The final simplified node has the structure of cached_structural_node,
    // but its accumulator is the one from the original node_arc.
    let mut final_node_data = (**cached_structural_node).clone(); // Clone GSSNode data
    *final_node_data.acc.acc_mut() = node_arc.acc.acc().clone(); // Set the specific acc from original node
    // Recompute hash key for final_node_data as its acc might differ from cached_structural_node's acc
    final_node_data.hash_key_cache = compute_hash_key(&final_node_data.predecessors);


    let result_arc = Arc::new(final_node_data);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}


impl GSSNode {
    pub fn simplify(&mut self) {
        // Create a temporary Arc to self to use with simplify_node_recursive
        // This requires `self` to be cloneable and then update `self` with the result.
        let temp_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new(); // Cache for structural sharing
        let simplified_arc = simplify_node_recursive(&temp_arc, &mut memo, &mut cache);
        
        // Update self with the simplified version's data
        // This is safe because simplify_node_recursive returns a potentially new Arc.
        // We take ownership of the data from the simplified Arc.
        if Arc::ptr_eq(&temp_arc, &simplified_arc) {
            // No change, or already canonical.
            // However, predecessors might have changed, so self might need update.
            // The most robust way is to replace self's content.
        }
        // Replace self's content with the (potentially) new simplified content
        let new_data = Arc::try_unwrap(simplified_arc).unwrap_or_else(|arc| (*arc).clone());
        *self = new_data;

    }

    // simplify_recursive is effectively what simplify_node_recursive does.
    // pub fn simplify_recursive(
    //     this_arc: &mut Arc<Self>,
    //     memo: &mut HashMap<*const Self, Arc<Self>>,
    //     cache: &mut NodeCache,
    // ) {
    //     *this_arc = simplify_node_recursive(this_arc, memo, cache);
    // }

    pub fn simplify_together(nodes: &mut [&mut Arc<Self>]) {
        let mut memo = HashMap::new(); // Memoization for input node pointers
        let mut cache = NodeCache::new(); // Cache for structural sharing of predecessor maps
        for node_arc_ref_mut in nodes { // Renamed node_arc
            // We need to pass a reference to the Arc to simplify_node_recursive
            // and then update the Arc in the slice.
            let current_arc = (*node_arc_ref_mut).clone(); // Clone the Arc to pass by value/ref
            let simplified_arc = simplify_node_recursive(&current_arc, &mut memo, &mut cache);
            **node_arc_ref_mut = simplified_arc; // Update the Arc in the slice
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint::LLMTokenBV;
    use super::*;
    use crate::glr::parser::ParseStateEdgeContent;
    use crate::glr::table::StateID;

    // MockPathAccumulator is now LLMTokenInfo, use that directly or a simplified version if needed for tests.
    // For simplicity, let's use LLMTokenInfo with basic active/intersection sets.

    type TestGSSNode = GSSNode; // GSSNode is now concrete

    fn mock_llm_token_info(active_val: usize, intersection_val: usize) -> Acc {
        let mut active = LLMTokenBV::new();
        active.insert(active_val);
        Acc::new(Some(active), BTreeMap::new())
    }
    
    fn mock_edge(id: usize) -> ParseStateEdgeContent {
        ParseStateEdgeContent { state_id: StateID(id), user_data: Arc::new(()) }
    }


    #[test]
    fn test_gss_simplification_basic() {
        let acc_base = mock_llm_token_info(0,0);
        let acc_other = mock_llm_token_info(1,1);
        let acc_shared_pred_structure = acc_base.clone().union(acc_other.clone());


        // Node N4 (leaf) - will be shared by D1 and D2 after simplification of D1/D2's predecessors
        // D1 -> 40 -> N4(acc_base)
        // D2 -> 40 -> N4(acc_other)
        // After simplification of D1's predecessors, N4 will be canonical.
        // When D2's predecessors are simplified, it should reuse the canonical N4 structure.

        let n4_v1 = Arc::new(TestGSSNode::new(acc_base.clone()));
        let n4_v2 = Arc::new(TestGSSNode::new(acc_other.clone()));


        // D1: C1 -> 30 -> D1(acc_base_pred_d1) -> 40 -> N4(acc_base)
        // acc_base_pred_d1 is acc_base
        let d1_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            n4_v1.clone(), mock_edge(40), acc_base.clone()
        ));

        // D2: (no C layer) -> 10 -> D2(acc_other_pred_d2) -> 40 -> N4(acc_other)
        // acc_other_pred_d2 is acc_other
         let d2_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            n4_v2.clone(), mock_edge(40), acc_other.clone()
        ));

        // C1: B1 -> 20 -> C1(acc_base_pred_c1) -> 30 -> D1
        // acc_base_pred_c1 is acc_base
        let c1_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            d1_orig.clone(), mock_edge(30), acc_base.clone()
        ));

        // B1: A1 -> 10 -> B1(acc_base_pred_b1) -> 20 -> C1
        // acc_base_pred_b1 is acc_base
        let b1_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            c1_orig.clone(), mock_edge(20), acc_base.clone()
        ));
        
        // A1: (root)
        // preds: B1 (via edge 10), D2 (via edge 10)
        // acc of A1 should be union of B1.acc and D2.acc if they were direct children.
        // Here, A1 is the root, its acc is what it is.
        // The structure is A1 --10--> B1 ... and A1 --10--> D2 ...
        // This means edge 10 from A1 points to two different conceptual children.
        // Simplification should merge these if B1 and D2 become structurally equivalent *after their own acc*.
        
        let mut a1_preds_set = NodeSet::new();
        a1_preds_set.insert((b1_orig.clone(), mock_edge(10)));
        a1_preds_set.insert((d2_orig.clone(), mock_edge(10)));
        
        // acc_mod::Acc for A1 is the union of paths leading to it.
        // Let's assume A1's acc is a union of acc_base and acc_other for this test.
        let acc_a1 = acc_base.clone().union(acc_other.clone());
        let a1_orig = Arc::new(TestGSSNode::new_with_map(acc_a1.clone(), process_predecessors(&a1_preds_set)));


        let mut roots_to_simplify = vec![a1_orig.clone()];
        let mut refs_to_simplify: Vec<&mut Arc<TestGSSNode>> = roots_to_simplify.iter_mut().collect();
        TestGSSNode::simplify_together(&mut refs_to_simplify);
        
        let s_a1 = refs_to_simplify[0].clone();

        // --- Verification ---
        // A1 should have one predecessor for edge 10, which is a merged node.
        assert_eq!(s_a1.predecessors.len(), 1, "A1 should have 1 predecessor map entry after merge");
        let (edge10, merged_b1_d2_node) = s_a1.predecessors.iter().next().unwrap();
        assert_eq!(edge10.state_id.0, 10, "Edge from A1 should be 10");

        // Accumulator of A1 should remain as it was.
        assert_eq!(s_a1.acc2(), &acc_a1, "A1 accumulator mismatch");

        // The merged_b1_d2_node is the result of B1 and D2 paths.
        // Its acc should be the union of B1's original acc and D2's original acc.
        // B1's original acc was acc_base. D2's original acc was acc_other.
        let expected_merged_acc = acc_base.clone().union(acc_other.clone());
        assert_eq!(merged_b1_d2_node.acc2(), &expected_merged_acc, "Merged B1/D2 node accumulator mismatch");

        // Structure of merged_b1_d2_node:
        // It should have two distinct predecessor edges:
        // - Edge 20 (from original B1 path) leading to a simplified C1.
        // - Edge 40 (from original D2 path) leading to a simplified N4 (shared).
        assert_eq!(merged_b1_d2_node.predecessors.len(), 2, "Merged B1/D2 node should have 2 predecessor map entries");

        let s_c1_via_b1 = merged_b1_d2_node.predecessors.get(&mock_edge(20)).expect("Edge 20 not found");
        let s_n4_via_d2 = merged_b1_d2_node.predecessors.get(&mock_edge(40)).expect("Edge 40 not found");

        // Check acc of s_c1_via_b1 (this is the simplified C1)
        // Original C1's acc was acc_base.
        assert_eq!(s_c1_via_b1.acc2(), &acc_base, "Simplified C1 accumulator mismatch");
        // Structure of s_c1_via_b1: edge 30 to simplified D1
        assert_eq!(s_c1_via_b1.predecessors.len(), 1);
        let (_edge30, s_d1_via_c1) = s_c1_via_b1.predecessors.iter().next().unwrap();
        assert_eq!(s_d1_via_c1.acc2(), &acc_base, "Simplified D1 accumulator mismatch");
        // Structure of s_d1_via_c1: edge 40 to simplified N4 (v1)
        assert_eq!(s_d1_via_c1.predecessors.len(), 1);
        let (_edge40_d1, s_n4_v1_via_d1) = s_d1_via_c1.predecessors.iter().next().unwrap();
        assert_eq!(s_n4_v1_via_d1.acc2(), &acc_base, "Simplified N4_v1 accumulator mismatch");
        assert!(s_n4_v1_via_d1.predecessors.is_empty(), "Simplified N4_v1 should be a leaf");


        // Check acc of s_n4_via_d2 (this is the simplified N4 from D2's path)
        // Original N4 from D2's path (n4_v2) had acc_other.
        assert_eq!(s_n4_via_d2.acc2(), &acc_other, "Simplified N4_v2 accumulator mismatch");
        assert!(s_n4_via_d2.predecessors.is_empty(), "Simplified N4_v2 should be a leaf");
        
        // Crucially, s_n4_v1_via_d1 and s_n4_via_d2 should point to different Arc<GSSNode> instances
        // if their acc differs, even if their predecessor structure (empty) is the same.
        // The *structural* node from cache might be shared, but then cloned and acc set.
        // Here, their accs (acc_base vs acc_other) are different, so they must be different Arcs.
        assert!(!Arc::ptr_eq(s_n4_v1_via_d1, s_n4_via_d2), "N4 nodes from different paths with different accs should not be the same Arc instance");

        // Count total unique nodes in the simplified graph starting from s_a1
        let mut all_nodes = HashSet::new();
        fn collect_all_nodes(node: &Arc<TestGSSNode>, set: &mut HashSet<*const TestGSSNode>) {
            if set.insert(Arc::as_ptr(node)) {
                for pred_arc in node.predecessors.values() {
                    collect_all_nodes(pred_arc, set);
                }
            }
        }
        collect_all_nodes(&s_a1, &mut all_nodes);
        // Expected nodes: A1, merged_B1_D2, C1_from_B1, D1_from_C1, N4_from_D1(acc_base), N4_from_D2(acc_other)
        // Total = 6 nodes
        assert_eq!(all_nodes.len(), 6, "Incorrect number of unique nodes in simplified graph. Actual: {:?}", all_nodes.len());
    }
}

