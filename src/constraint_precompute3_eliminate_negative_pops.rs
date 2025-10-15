// src/constraint_precompute3_eliminate_negative_pops.rs
//
// Simplified design for negative-pop elimination. This version uses a simple,
// custom graph structure instead of the more complex Trie, removing the dependency
// on `trie.rs`.
//
// This file provides:
// - A simple graph representation (`SimpleGraph`, `Edge`).
// - Graph-level transformation functions that operate on this structure by
//   extracting all paths, processing them with stack-based logic, and
//   rebuilding an unrolled graph from the results.
// - Fully implemented reference, stack-only algorithms:
//    * stack_eliminate_internal_negative_pops: cancels internal negative/positive
//      run pairs in a single stack.
//    * stack_eliminate_trailing_negative_pops: removes trailing negative pops.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Represents an edge in the SimpleGraph.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Edge<T> {
    pub pop: isize,
    pub value: T,
}

/// A simple directed graph implementation using an adjacency list.
/// Nodes are identified by their `usize` index.
#[derive(Debug, Clone)]
pub struct SimpleGraph<T> {
    adj: Vec<Vec<(usize, Edge<T>)>>,
    next_node_id: usize,
}

impl<T: Clone> SimpleGraph<T> {
    pub fn new() -> Self {
        Self {
            adj: Vec::new(),
            next_node_id: 0,
        }
    }

    pub fn add_node(&mut self) -> usize {
        let id = self.next_node_id;
        self.next_node_id += 1;
        if id >= self.adj.len() {
            self.adj.resize_with(id + 1, Vec::new);
        }
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize, edge: Edge<T>) {
        self.adj[from].push((to, edge));
    }
}

/// Extracts all unique paths starting from a given root node.
fn get_paths_from_root<T>(graph: &SimpleGraph<T>, root: usize) -> BTreeSet<Vec<Edge<T>>>
where
    T: Ord + Clone,
{
    let mut paths = BTreeSet::new();
    let mut visiting = BTreeSet::new();
    get_paths_recursive(graph, root, vec![], &mut paths, &mut visiting);
    paths
}

fn get_paths_recursive<T>(
    graph: &SimpleGraph<T>,
    node_id: usize,
    current_path: Vec<Edge<T>>,
    all_paths: &mut BTreeSet<Vec<Edge<T>>>,
    visiting: &mut BTreeSet<usize>,
) where
    T: Ord + Clone,
{
    if !visiting.insert(node_id) {
        return; // Cycle detected
    }

    if graph.adj.get(node_id).map_or(true, |edges| edges.is_empty()) {
        all_paths.insert(current_path);
    } else {
        for (child_id, edge) in &graph.adj[node_id] {
            let mut new_path = current_path.clone();
            new_path.push(edge.clone());
            get_paths_recursive(graph, *child_id, new_path, all_paths, visiting);
        }
    }

    visiting.remove(&node_id);
}

/// Perform negative-pop elimination on the graph.
/// This is an orchestrator that calls the internal and trailing elimination stages.
pub fn eliminate_negative_pops<T, FReplace, FIntersect, FCanRemove>(
    graph: &mut SimpleGraph<T>,
    roots: &[usize],
    mut replace_pop: FReplace,
    mut intersect_checks: FIntersect,
    mut can_remove: FCanRemove,
) where
    T: Ord + Clone,
    FReplace: FnMut(&Edge<T>, isize) -> Edge<T>,
    FIntersect: FnMut(&Edge<T>, &Edge<T>) -> bool,
    FCanRemove: FnMut(&Edge<T>) -> bool,
{
    eliminate_internal_negative_pops_on_graph(
        graph,
        roots,
        &mut replace_pop,
        &mut intersect_checks,
        &mut can_remove,
    );
    eliminate_trailing_negative_pops_on_graph(graph, roots, &mut can_remove);
}

/// Graph-level transform: eliminate internal negative pops by pairwise cancellation.
/// This function extracts all paths, processes them, and rebuilds the graph.
pub fn eliminate_internal_negative_pops_on_graph<T, FReplace, FIntersect, FCanRemove>(
    graph: &mut SimpleGraph<T>,
    roots: &[usize],
    replace_pop: &mut FReplace,
    intersect_checks: &mut FIntersect,
    can_remove: &mut FCanRemove,
) where
    T: Ord + Clone,
    FReplace: FnMut(&Edge<T>, isize) -> Edge<T>,
    FIntersect: FnMut(&Edge<T>, &Edge<T>) -> bool,
    FCanRemove: FnMut(&Edge<T>) -> bool,
{
    let mut processed_paths_by_root = BTreeMap::new();
    for &root in roots {
        let paths = get_paths_from_root(graph, root);
        let mut processed_for_root = BTreeSet::new();
        for path in paths {
            let get_pop = |edge: &Edge<T>| edge.pop;
            if let Some(new_path) = stack_eliminate_internal_negative_pops(
                path,
                get_pop,
                |e, p| replace_pop(e, p),
                |e1, e2| intersect_checks(e1, e2),
                |e| can_remove(e),
            ) {
                processed_for_root.insert(new_path);
            }
        }
        processed_paths_by_root.insert(root, processed_for_root);
    }

    // Rebuild the graph from the processed paths, preserving root IDs.
    let old_roots = roots.to_vec();
    *graph = SimpleGraph::new();

    for &root_id in &old_roots {
        while graph.next_node_id <= root_id {
            graph.add_node();
        }
    }

    for (old_root_id, paths) in processed_paths_by_root {
        for path in paths {
            let mut current_node_id = old_root_id;
            for edge in path {
                let new_node_id = graph.add_node();
                graph.add_edge(current_node_id, new_node_id, edge);
                current_node_id = new_node_id;
            }
        }
    }
}

/// Graph-level transform: remove trailing negative pops at the ends of paths.
pub fn eliminate_trailing_negative_pops_on_graph<T, FCanRemove>(
    graph: &mut SimpleGraph<T>,
    roots: &[usize],
    can_remove: &mut FCanRemove,
) where
    T: Ord + Clone,
    FCanRemove: FnMut(&Edge<T>) -> bool,
{
    let mut processed_paths_by_root = BTreeMap::new();
    for &root in roots {
        let paths = get_paths_from_root(graph, root);
        let mut processed_for_root = BTreeSet::new();
        for path in paths {
            let get_pop = |edge: &Edge<T>| edge.pop;
            let new_path =
                stack_eliminate_trailing_negative_pops(path, get_pop, |e| can_remove(e));
            processed_for_root.insert(new_path);
        }
        processed_paths_by_root.insert(root, processed_for_root);
    }

    // Rebuild the graph from the processed paths, preserving root IDs.
    let old_roots = roots.to_vec();
    *graph = SimpleGraph::new();

    for &root_id in &old_roots {
        while graph.next_node_id <= root_id {
            graph.add_node();
        }
    }

    for (old_root_id, paths) in processed_paths_by_root {
        for path in paths {
            let mut current_node_id = old_root_id;
            for edge in path {
                let new_node_id = graph.add_node();
                graph.add_edge(current_node_id, new_node_id, edge);
                current_node_id = new_node_id;
            }
        }
    }
}

/// Reference stack function: eliminate internal negative pops by canceling adjacent
/// negative/positive run pairs.
/// Returns:
/// - Some(new_stack) if no mismatches were found.
/// - None if any run pair exhibited a mismatch (entire stack eliminated).
pub fn stack_eliminate_internal_negative_pops<EK, FGet, FReplace, FIntersect, FCanRemove>(
    stack: Vec<EK>,
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut intersect_checks: FIntersect,
    mut can_remove: FCanRemove,
) -> Option<Vec<EK>>
where
    EK: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FIntersect: FnMut(&EK, &EK) -> bool,
    FCanRemove: FnMut(&EK) -> bool,
{
    let mut out: Vec<EK> = Vec::with_capacity(stack.len());
    let mut neg_buf: Vec<EK> = Vec::new();
    let mut pos_buf: Vec<EK> = Vec::new();
    let mut in_pos = false;

    fn process_pair<EK, FGet, FReplace, FIntersect, FCanRemove>(
        neg_buf: Vec<EK>,
        pos_buf: Vec<EK>,
        get_pop: &mut FGet,
        replace_pop: &mut FReplace,
        intersect_checks: &mut FIntersect,
        can_remove: &mut FCanRemove,
    ) -> Option<(Vec<EK>, Vec<EK>)>
    where
        EK: Clone,
        FGet: FnMut(&EK) -> isize,
        FReplace: FnMut(&EK, isize) -> EK,
        FIntersect: FnMut(&EK, &EK) -> bool,
        FCanRemove: FnMut(&EK) -> bool,
    {
        let mut neg_rev: Vec<EK> = Vec::with_capacity(neg_buf.len());
        for ek in neg_buf.iter().rev() {
            let p = get_pop(ek);
            debug_assert!(p <= 0);
            neg_rev.push(replace_pop(ek, -p));
        }

        let mut neg_map: BTreeMap<usize, Vec<&EK>> = BTreeMap::new();
        let mut pos_map: BTreeMap<usize, Vec<&EK>> = BTreeMap::new();

        let mut cum = 0usize;
        for ek in &neg_rev {
            let p = get_pop(ek);
            debug_assert!(p >= 0);
            cum += p as usize;
            neg_map.entry(cum).or_default().push(ek);
        }
        cum = 0;
        for ek in &pos_buf {
            let p = get_pop(ek);
            debug_assert!(p > 0);
            cum += p as usize;
            pos_map.entry(cum).or_default().push(ek);
        }

        let mut neg_it = neg_map.iter().peekable();
        let mut pos_it = pos_map.iter().peekable();
        while let (Some((npos, neks)), Some((ppos, peks))) = (neg_it.peek(), pos_it.peek()) {
            if *npos == *ppos {
                for nek in neks.iter() {
                    for pek in peks.iter() {
                        if !intersect_checks(nek, pek) {
                            return None;
                        }
                    }
                }
                neg_it.next();
                pos_it.next();
            } else if *npos < *ppos {
                neg_it.next();
            } else {
                pos_it.next();
            }
        }

        let sum_neg: usize = neg_rev.iter().map(|ek| get_pop(ek).max(0) as usize).sum();
        let sum_pos: usize = pos_buf.iter().map(|ek| get_pop(ek).max(0) as usize).sum();
        let cancel_amt = sum_neg.min(sum_pos);

        fn subtract_from_front<EK, FGet, FReplace, FCanRemove>(
            seq: Vec<EK>,
            mut amt: usize,
            get_pop: &mut FGet,
            replace_pop: &mut FReplace,
            can_remove: &mut FCanRemove,
        ) -> Vec<EK>
        where
            EK: Clone,
            FGet: FnMut(&EK) -> isize,
            FReplace: FnMut(&EK, isize) -> EK,
            FCanRemove: FnMut(&EK) -> bool,
        {
            if amt == 0 {
                return seq;
            }
            let mut out: Vec<EK> = Vec::with_capacity(seq.len());
            for ek in seq.into_iter() {
                if amt == 0 {
                    out.push(ek);
                    continue;
                }
                let p = get_pop(&ek);
                debug_assert!(p > 0);
                let pu = p as usize;
                if pu > amt {
                    out.push(replace_pop(&ek, (pu - amt) as isize));
                    amt = 0;
                } else {
                    if !can_remove(&ek) {
                        out.push(replace_pop(&ek, 0));
                    }
                    amt -= pu;
                }
            }
            out
        }

        let neg_rev_left =
            subtract_from_front(neg_rev, cancel_amt, get_pop, replace_pop, can_remove);
        let pos_left =
            subtract_from_front(pos_buf.clone(), cancel_amt, get_pop, replace_pop, can_remove);

        let mut leftover_neg: Vec<EK> = Vec::with_capacity(neg_rev_left.len());
        for ek in neg_rev_left.into_iter().rev() {
            let p = get_pop(&ek);
            debug_assert!(p >= 0);
            leftover_neg.push(replace_pop(&ek, -p));
        }

        Some((leftover_neg, pos_left))
    }

    for ek in stack.into_iter() {
        let p = get_pop(&ek);
        if p < 0 {
            if !in_pos {
                neg_buf.push(ek);
            } else {
                let pair = process_pair(
                    neg_buf,
                    pos_buf,
                    &mut get_pop,
                    &mut replace_pop,
                    &mut intersect_checks,
                    &mut can_remove,
                )?;
                let (leftover_neg, leftover_pos) = pair;
                out.extend(leftover_pos);
                neg_buf = leftover_neg;
                pos_buf = Vec::new();
                in_pos = false;
                neg_buf.push(ek);
            }
        } else if p > 0 {
            if neg_buf.is_empty() && !in_pos {
                out.push(ek);
            } else {
                in_pos = true;
                pos_buf.push(ek);
            }
        } else {
            if in_pos {
                let pair = process_pair(
                    neg_buf,
                    pos_buf,
                    &mut get_pop,
                    &mut replace_pop,
                    &mut intersect_checks,
                    &mut can_remove,
                )?;
                let (leftover_neg, leftover_pos) = pair;
                out.extend(leftover_pos);
                neg_buf = leftover_neg;
                pos_buf = Vec::new();
                in_pos = false;
            }
            out.extend(neg_buf.into_iter());
            neg_buf = Vec::new();
            out.push(ek);
        }
    }

    if in_pos {
        let pair = process_pair(
            neg_buf,
            pos_buf,
            &mut get_pop,
            &mut replace_pop,
            &mut intersect_checks,
            &mut can_remove,
        )?;
        let (leftover_neg, leftover_pos) = pair;
        out.extend(leftover_pos);
        neg_buf = leftover_neg;
    }

    out.extend(neg_buf.into_iter());

    out.retain(|ek| get_pop(ek) != 0 || !can_remove(ek));

    Some(out)
}

/// Reference stack function: remove trailing negative pops and zero-pop items.
pub fn stack_eliminate_trailing_negative_pops<EK, FGet, FCanRemove>(
    stack: Vec<EK>,
    mut get_pop: FGet,
    mut can_remove: FCanRemove,
) -> Vec<EK>
where
    EK: Clone,
    FGet: FnMut(&EK) -> isize,
    FCanRemove: FnMut(&EK) -> bool,
{
    let last_good_idx = stack.iter().rposition(|ek| get_pop(ek) >= 0);

    let mut result = match last_good_idx {
        Some(idx) => stack[..=idx].to_vec(),
        None => Vec::new(),
    };

    result.retain(|ek| get_pop(ek) != 0 || !can_remove(ek));

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // Test harness types
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct TestEdgeValue {
        check: Option<BTreeSet<usize>>,
        ev: Option<usize>,
    }
    type TestEdge = Edge<TestEdgeValue>;
    type TestGraph = SimpleGraph<TestEdgeValue>;

    // Helpers for closures
    fn get_pop(edge: &TestEdge) -> isize {
        edge.pop
    }

    fn replace_pop(edge: &TestEdge, new_pop: isize) -> TestEdge {
        TestEdge {
            pop: new_pop,
            value: edge.value.clone(),
        }
    }

    fn checks_intersect(a: &TestEdge, b: &TestEdge) -> bool {
        match (&a.value.check, &b.value.check) {
            (None, _) | (_, None) => true,
            (Some(s1), Some(s2)) => s1.iter().any(|x| s2.contains(x)),
        }
    }

    fn can_remove(edge: &TestEdge) -> bool {
        edge.value.ev.is_none()
    }

    fn edge(pop: isize, ids: Option<&[usize]>, ev: Option<usize>) -> TestEdge {
        TestEdge {
            pop,
            value: TestEdgeValue {
                check: ids.map(|s| s.iter().cloned().collect()),
                ev,
            },
        }
    }

    // -- Unit tests for stack elimination behavior --

    #[test]
    fn zero_pop_and_cancellation_with_ev() {
        let input1 = vec![edge(1, None, None), edge(0, None, None), edge(2, None, None)];
        let got1 = stack_eliminate_internal_negative_pops(
            input1,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got1, vec![edge(1, None, None), edge(2, None, None)]);

        let input2 = vec![
            edge(-1, None, None),
            edge(0, None, Some(123)),
            edge(1, None, None),
        ];
        let got2 = stack_eliminate_internal_negative_pops(
            input2.clone(),
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got2, input2, "0-pop with EV should prevent cancellation");

        let input3 = vec![edge(-1, None, Some(1)), edge(1, None, None)];
        let got3 = stack_eliminate_internal_negative_pops(
            input3,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got3, vec![edge(0, None, Some(1))]);

        let input4 = vec![edge(-1, None, Some(1)), edge(1, None, Some(2))];
        let got4 = stack_eliminate_internal_negative_pops(
            input4,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got4, vec![edge(0, None, Some(2)), edge(0, None, Some(1))]);

        let input5 = vec![edge(-2, None, Some(1)), edge(1, None, None)];
        let got5 = stack_eliminate_internal_negative_pops(
            input5,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got5, vec![edge(-1, None, Some(1))]);
    }

    #[test]
    fn run_pair_full_cancel_with_remainder_positive() {
        let input = vec![
            edge(-1, Some(&[1]), None),
            edge(-1, Some(&[2]), None),
            edge(1, Some(&[2]), None),
            edge(1, Some(&[1]), None),
            edge(1, Some(&[0]), None),
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(got, vec![edge(1, Some(&[0]), None)]);
    }

    #[test]
    fn run_pair_mismatch_eliminates_stack() {
        let input = vec![edge(-1, Some(&[1]), None), edge(1, Some(&[2]), None)];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        );
        assert!(got.is_none());
    }

    #[test]
    fn trailing_negative_pops_are_removed() {
        let input = vec![
            edge(1, Some(&[0]), None),
            edge(-1, Some(&[1]), None),
            edge(0, None, None),
        ];
        let trimmed = stack_eliminate_trailing_negative_pops(input, get_pop, can_remove);
        assert_eq!(trimmed, vec![edge(1, Some(&[0]), None)]);
    }

    // --- Graph-level scenario tests ---

    fn run_graph_vs_stack_comparison_test(mut graph: TestGraph, roots: &[usize]) {
        // 1. Calculate EXPECTED stacks from the original graph.
        let mut expected_stacks_by_root = BTreeMap::new();
        for &root in roots {
            let initial_stacks = get_paths_from_root(&graph, root);
            let mut expected_stacks = BTreeSet::new();
            for s in initial_stacks {
                if let Some(mid) = stack_eliminate_internal_negative_pops(
                    s,
                    get_pop,
                    replace_pop,
                    checks_intersect,
                    can_remove,
                ) {
                    let fin = stack_eliminate_trailing_negative_pops(mid, get_pop, can_remove);
                    expected_stacks.insert(fin);
                }
            }
            expected_stacks_by_root.insert(root, expected_stacks);
        }

        // 2. Calculate ACTUAL stacks by running the graph-level transform.
        eliminate_negative_pops(
            &mut graph,
            roots,
            replace_pop,
            checks_intersect,
            can_remove,
        );

        let mut actual_stacks_by_root = BTreeMap::new();
        for &root in roots {
            actual_stacks_by_root.insert(root, get_paths_from_root(&graph, root));
        }

        // 3. Compare the results.
        assert_eq!(expected_stacks_by_root, actual_stacks_by_root);
    }

    #[test]
    fn test_simple_cancel_on_graph() {
        let mut graph = TestGraph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();
        graph.add_edge(a, b, edge(1, Some(&[0]), None));
        graph.add_edge(b, c, edge(-1, Some(&[0]), None));
        graph.add_edge(a, d, edge(1, None, None));
        run_graph_vs_stack_comparison_test(graph, &[a]);
    }

    #[test]
    fn test_mismatch_eliminates_one_path_on_graph() {
        let mut graph = TestGraph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();
        graph.add_edge(a, b, edge(1, Some(&[0]), None));
        graph.add_edge(b, c, edge(-1, Some(&[1]), None));
        graph.add_edge(a, d, edge(2, Some(&[2]), None));
        run_graph_vs_stack_comparison_test(graph, &[a]);
    }

    #[test]
    fn test_trailing_negative_is_removed_on_graph() {
        let mut graph = TestGraph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();
        graph.add_edge(a, b, edge(2, None, None));
        graph.add_edge(b, c, edge(-1, None, None));
        graph.add_edge(a, d, edge(5, None, None));
        run_graph_vs_stack_comparison_test(graph, &[a]);
    }

    #[test]
    fn test_shared_node_with_divergent_outcomes_on_graph() {
        let mut graph = TestGraph::new();
        let a = graph.add_node();
        let d = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();

        graph.add_edge(a, b, edge(1, Some(&[0]), None));
        graph.add_edge(d, b, edge(-1, Some(&[1]), None));
        graph.add_edge(b, c, edge(1, Some(&[0]), None));

        run_graph_vs_stack_comparison_test(graph, &[a, d]);
    }

    #[test]
    fn test_all_paths_eliminated_on_graph() {
        let mut graph = TestGraph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();
        let e = graph.add_node();

        graph.add_edge(a, b, edge(1, Some(&[0]), None));
        graph.add_edge(b, c, edge(-1, Some(&[1]), None));
        graph.add_edge(a, d, edge(-2, Some(&[2]), None));
        graph.add_edge(d, e, edge(2, Some(&[2]), None));

        run_graph_vs_stack_comparison_test(graph, &[a]);
    }
}
