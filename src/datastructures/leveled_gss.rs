//! Pure Rust, Python-free core implementation of the Leveled GSS.
//! This module contains the generic data structures and algorithms, parameterized
//! over stack item type `T` and accumulator type `A`.
//!
//! - T must implement Clone + Eq + Hash
//! - A must implement Clone + Eq + Hash + Merge
//!
//! The representation mirrors the Python implementation's semantics and supports:
//! - from_stacks / to_stacks
//! - push, pop, popn
//! - is_empty
//! - isolate, isolate_many
//! - apply, prune, apply_and_prune
//! - merge, peek, reduce_acc
//!
//! All logic here is pure Rust and contains no Python bindings.

use im::{HashMap as IHashMap, OrdMap};
use std::collections::{BTreeMap, HashMap as StdHashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use profiler_macro::time_it;

/// Trait for accumulator types that can be merged.
pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;
}

type Children<T, N> = IHashMap<T, OrdMap<isize, Arc<N>>>;

#[derive(Clone)]
struct Lower<T: Clone + Eq + Hash> {
    children: Children<T, Lower<T>>,
    empty: bool,
    max_depth: isize,
}

#[derive(Clone)]
struct Interface<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Lower<T>>,
    acc: A,
}

#[derive(Clone)]
struct UpperBranch<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
    max_depth: isize,
}

#[derive(Clone)]
enum Upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    Branch(Arc<UpperBranch<T, A>>),
    Interface(Arc<Interface<T, A>>),
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Upper<T, A> {
    fn max_depth(&self) -> isize {
        match self {
            Upper::Branch(b) => b.max_depth,
            Upper::Interface(i) => {
                if i.inner.children.is_empty() {
                    0
                } else {
                    i.inner.max_depth + 1
                }
            }
        }
    }

    fn children_keys(&self) -> Vec<T> {
        match self {
            Upper::Branch(b) => b.children.keys().cloned().collect(),
            Upper::Interface(i) => i.inner.children.keys().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // A simple accumulator that just collects integers.
    #[derive(Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
    struct IntAcc(im::HashSet<i32>);

    impl Merge for IntAcc {
        fn merge(&self, other: &Self) -> Self {
            IntAcc(self.0.clone().union(other.0.clone()))
        }
    }

    impl IntAcc {
        fn new(vals: &[i32]) -> Self {
            let mut set = im::HashSet::new();
            for &v in vals {
                set.insert(v);
            }
            IntAcc(set)
        }
    }

    type TestGSS = LeveledGSS<String, IntAcc>;

    fn gss_from_str_stacks(stacks: &[(&[&str], &[i32])]) -> TestGSS {
        let stacks_vec: Vec<(Vec<String>, IntAcc)> = stacks
            .iter()
            .map(|(s, a)| {
                (
                    s.iter().map(|&v| v.to_string()).collect(),
                    IntAcc::new(a),
                )
            })
            .collect();
        TestGSS::from_stacks(&stacks_vec)
    }

    #[test]
    fn test_pop_sharing_and_determinism() {
        let gss0 = gss_from_str_stacks(&[(
            &["A", "B", "C"],
            &[1],
        )]);

        let gss1 = gss0.pop();
        let gss2 = gss0.pop();

        // Popping is deterministic and should produce structurally identical GSS
        // with maximum sharing.
        assert!(gss1.inner_ptrs_eq(&gss2));
        assert!(!gss1.ptr_eq(&gss2)); // The top-level Arc will be different.
    }

    #[test]
    fn test_push_pop_identity() {
        let gss0 = gss_from_str_stacks(&[(
            &["A", "B"],
            &[1],
        )]);

        let gss1 = gss0.push("X".to_string()).pop();

        // push followed by pop should return the exact same GSS structure.
        assert!(gss0.inner_ptrs_eq(&gss1));
        assert!(!gss0.ptr_eq(&gss1)); // The top-level Arc will be different.
    }

    #[test]
    fn test_push_pop_identity_from_empty() {
        let gss0 = TestGSS::empty();
        let gss1 = gss0.push("X".to_string()).pop();

        // On an empty GSS, push is a no-op, so pop is also a no-op on the result.
        assert!(gss0.ptr_eq(&gss1));
    }

    #[test]
    fn test_pop_preserves_child_node_sharing() {
        let gss_abc = gss_from_str_stacks(&[(
            &["C", "B", "A"],
            &[1],
        )]);

        let gss_bc_from_pop = gss_abc.pop();

        // Manually find the node for "B"->"C" in the original GSS
        let preds = gss_abc.predecessors();
        let children_of_a = preds.get(&"A".to_string()).unwrap();
        let gss_bc_from_preds = children_of_a.values().next().unwrap().first().unwrap();

        // The GSS resulting from pop should be structurally identical to the
        // predecessor GSS found inside the original.
        assert!(gss_bc_from_pop.inner_ptrs_eq(gss_bc_from_preds));

        // And more strongly, the children of their roots should be the *same pointers*.
        let inner_pop = &gss_bc_from_pop.inner;
        let inner_preds = &gss_bc_from_preds.inner;

        match (&**inner_pop, &**inner_preds) {
            (Upper::Interface(i_pop), Upper::Interface(i_preds)) => {
                let children_pop = i_pop.inner.children.get(&"B".to_string()).unwrap();
                let children_preds = i_preds.inner.children.get(&"B".to_string()).unwrap();
                let child_c_pop = children_pop.values().next().unwrap();
                let child_c_preds = children_preds.values().next().unwrap();
                assert!(Arc::ptr_eq(child_c_pop, child_c_preds));
            }
            _ => panic!("Expected Interface nodes"),
        }
    }

    #[test]
    fn test_parallel_push_identity_empty() {
        let gss0 = gss_from_str_stacks(&[(
            &[],
            &[1],
        )]);

        let gss1 = gss0.push("X".to_string());
        let gss2 = gss0.push("X".to_string());

        assert!(gss1.inner_ptrs_eq(&gss2));
        assert!(!gss1.ptr_eq(&gss2)); // The top-level Arc will be different.
    }

    #[test]
    fn test_parallel_push_identity_one() {
        let gss0 = gss_from_str_stacks(&[(
            &["A"],
            &[1],
        )]);

        let gss1 = gss0.push("X".to_string());
        let gss2 = gss0.push("X".to_string());

        assert!(gss1.inner_ptrs_eq(&gss2));
        assert!(!gss1.ptr_eq(&gss2)); // The top-level Arc will be different.
    }

    #[test]
    fn test_parallel_push_identity_two() {
        let gss0 = gss_from_str_stacks(&[(
            &["A", "B"],
            &[1],
        )]);

        let gss1 = gss0.push("X".to_string());
        let gss2 = gss0.push("X".to_string());

        assert!(gss1.inner_ptrs_eq(&gss2));
        assert!(!gss1.ptr_eq(&gss2)); // The top-level Arc will be different.
    }

    #[test]
    fn test_isolate_preserves_ptr_on_noop() {
        // Case 1: Isolate single child, no empty. Should be a no-op.
        let gss0 = gss_from_str_stacks(&[(&["A"], &[1])]);
        let gss1 = gss0.isolate(Some("A".to_string()));
        assert!(gss0.ptr_eq(&gss1));

        // Case 2: Isolate None on a GSS that is only an empty path. Should be a no-op.
        let gss2 = gss_from_str_stacks(&[(&[], &[1])]);
        let gss3 = gss2.isolate(None);
        assert!(gss2.ptr_eq(&gss3));

        // Case 3: Isolate single child from multiple. Should NOT be a no-op.
        let gss4 = gss_from_str_stacks(&[(&["A"], &[1]), (&["B"], &[2])]);
        let gss5 = gss4.isolate(Some("A".to_string()));
        assert!(!gss4.ptr_eq(&gss5));

        // Case 4: Isolate None from a GSS with children. Should NOT be a no-op.
        let gss6 = gss_from_str_stacks(&[(&["A"], &[1]), (&[], &[2])]);
        let gss7 = gss6.isolate(None);
        assert!(!gss6.ptr_eq(&gss7));
    }

    #[test]
    fn test_isolate_many_preserves_ptr_on_noop() {
        let gss0 = gss_from_str_stacks(&[(&["A"], &[1]), (&["B"], &[2]), (&[], &[3])]);

        // Case 1: Select all children and empty. Should be a no-op.
        let gss1 = gss0.isolate_many(vec![Some("A".to_string()), Some("B".to_string()), None]);
        assert!(gss0.ptr_eq(&gss1));

        // Case 2: Select a superset of children. Should be a no-op.
        let gss2 = gss0.isolate_many(vec![Some("A".to_string()), Some("B".to_string()), Some("C".to_string()), None]);
        assert!(gss0.ptr_eq(&gss2));

        // Case 3: Select a subset of children. Should NOT be a no-op.
        let gss3 = gss0.isolate_many(vec![Some("A".to_string()), None]);
        assert!(!gss0.ptr_eq(&gss3));

        // Case 4: Select all children but forget empty. Should NOT be a no-op.
        let gss4 = gss0.isolate_many(vec![Some("A".to_string()), Some("B".to_string())]);
        assert!(!gss0.ptr_eq(&gss4));

        // Case 5: GSS with no empty, select all children and None. Should NOT be a no-op.
        let gss5 = gss_from_str_stacks(&[(&["A"], &[1]), (&["B"], &[2])]);
        let gss6 = gss5.isolate_many(vec![Some("A".to_string()), Some("B".to_string()), None]);
        assert!(!gss5.ptr_eq(&gss6));
    }

    #[test]
    fn test_filter_by_length_preserves_ptr_on_noop() {
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
            (&["X", "Y"], &[2]),
            (&["Z"], &[3]),
            (&[], &[4]),
        ]);

        // Case 1: No filter applied. Should be a no-op.
        let gss1 = gss0.filter_by_length(None, None);
        assert!(gss0.ptr_eq(&gss1));

        // Case 2: Filter range includes all paths. Should be a no-op.
        // Paths have lengths 0, 1, 2, 3.
        let gss2 = gss0.filter_by_length(Some(0), Some(3));
        assert!(gss0.ptr_eq(&gss2));
        let gss3 = gss0.filter_by_length(Some(-1), Some(10));
        assert!(gss0.ptr_eq(&gss3));

        // Case 3: Filter prunes some paths. Should NOT be a no-op.
        let gss4 = gss0.filter_by_length(Some(1), Some(2));
        assert!(!gss0.ptr_eq(&gss4));
        assert_eq!(gss4.to_stacks().len(), 2);

        // Case 4: Filter on empty GSS.
        let gss_empty = TestGSS::empty();
        let gss_empty_filtered = gss_empty.filter_by_length(Some(1), Some(2));
        assert!(gss_empty.ptr_eq(&gss_empty_filtered));
    }

    #[test]
    fn test_prune_preserves_ptr_on_noop() {
        // GSS with two different accumulators.
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B"], &[1, 2]), // acc IntAcc({1, 2})
            (&["X"], &[3]),       // acc IntAcc({3})
            (&[], &[1, 3]),       // acc IntAcc({1, 3})
        ]);

        // Case 1: Predicate keeps everything. Should be a no-op.
        let gss1 = gss0.prune(|_acc| true);
        assert!(gss0.ptr_eq(&gss1));

        // Case 2: Predicate prunes something. Should NOT be a no-op.
        let gss2 = gss0.prune(|acc| acc.0.contains(&1));
        assert!(!gss0.ptr_eq(&gss2));
        // The path ["X"] with acc {3} should be pruned.
        assert_eq!(gss2.to_stacks().len(), 2);

        // Case 3: Predicate prunes everything. Result should be empty.
        let gss3 = gss0.prune(|_acc| false);
        assert!(gss3.is_empty());
        assert!(!gss0.ptr_eq(&gss3));

        // Case 4: Prune on empty GSS.
        let gss_empty = TestGSS::empty();
        let gss_empty_pruned = gss_empty.prune(|_acc| true);
        assert!(gss_empty.ptr_eq(&gss_empty_pruned));
    }

    #[test]
    fn test_normalization_worsens_sharing_factor() {
        // This test reproduces a scenario where normalization was found to decrease
        // the structural sharing factor, which is counter-intuitive.
        // The scenario involves a GSS with a multi-depth child map, which `fuse`
        // resolves by merging. The resulting structure, after canonicalization,
        // had a worse sharing factor (structurally_unique / total_unique).

        // GSS 1 represents a structure with max_depth=3.
        let gss1 = gss_from_str_stacks(&[
            (&["A", "B", "T"], &[1]),
        ]);
        assert_eq!(gss1.max_depth(), 3);

        // GSS 2 represents a structure with max_depth=4 and some branching.
        // It shares the terminal "T" with GSS 1.
        let gss2 = gss_from_str_stacks(&[
            (&["A", "C", "D1", "T"], &[2]),
            (&["A", "C", "D2", "T"], &[3]),
        ]);
        assert_eq!(gss2.max_depth(), 4);

        // Create a GSS with a multi-depth child map.
        // We push gss1 and gss2 under a new root "X", then merge.
        // Because gss1 and gss2 have different max_depths, the merge will
        // create a child map for "X" with two entries: one for depth 3
        // and one for depth 4.
        let parent1 = gss1.push("X".to_string());
        let parent2 = gss2.push("X".to_string());
        let gss_to_test = parent1.merge(&parent2);

        // Verify the multi-depth structure exists before normalization.
        if let Upper::Branch(b) = &*gss_to_test.inner {
            if let Some(kids) = b.children.get(&"X".to_string()) {
                assert_eq!(kids.len(), 2, "Test setup failed: multi-depth map not created.");
            } else {
                panic!("Test setup failed: child 'X' not found.");
            }
        } else {
            panic!("Test setup failed: root is not an UpperBranch.");
        }

        let stats_before = gss_to_test.stats();
        let normalized_gss = gss_to_test.normalize();
        let stats_after = normalized_gss.stats();

        // 1. Normalization should not increase the total number of unique nodes.
        assert!(
            stats_after.total_unique_nodes <= stats_before.total_unique_nodes,
            "Total unique nodes increased after normalization: {} -> {}",
            stats_before.total_unique_nodes, stats_after.total_unique_nodes
        );

        // 2. The structural sharing factor should not get worse. This is the assertion
        // that is expected to fail, reproducing the reported issue.
        assert!(
            stats_after.structural_sharing_factor >= stats_before.structural_sharing_factor,
            "Structural sharing factor decreased after normalization: {:.3} -> {:.3}",
            stats_before.structural_sharing_factor, stats_after.structural_sharing_factor
        );
    }
}

// --------------------
// Small, reusable helpers
// --------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct GSSPathsInfo {
    pub num_paths: usize,
    pub total_depth: usize,
}

impl std::ops::Add for GSSPathsInfo {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            num_paths: self.num_paths + rhs.num_paths,
            total_depth: self.total_depth + rhs.total_depth,
        }
    }
}

impl std::ops::AddAssign for GSSPathsInfo {
    fn add_assign(&mut self, rhs: Self) {
        self.num_paths += rhs.num_paths;
        self.total_depth += rhs.total_depth;
    }
}

fn merge_optional_acc<A: Merge + Clone>(a: &Option<A>, b: &Option<A>) -> Option<A> {
    match (a, b) {
        (None, Some(bv)) => Some(bv.clone()),
        (Some(av), None) => Some(av.clone()),
        (Some(av), Some(bv)) => Some(av.merge(bv)),
        (None, None) => None,
    }
}

fn max_depth_from_children<T, N, F>(children: &Children<T, N>, depth_of: F) -> isize
where
    T: Clone + Eq + Hash,
    F: Fn(&Arc<N>) -> isize,
{
    children
        .values()
        .flat_map(|kids| kids.values())
        .map(|c| depth_of(c))
        .max()
        .map_or(0, |d| d + 1)
}

fn merge_children<T, N, F>(c1: &Children<T, N>, c2: &Children<T, N>, merge_fn: F) -> Children<T, N>
where
    T: Clone + Eq + Hash,
    F: Fn(&Arc<N>, &Arc<N>) -> Arc<N>,
{
    if c1.ptr_eq(c2) {
        return c1.clone();
    }
    let mut merged = c1.clone();
    for (k, v2_map) in c2.iter() {
        if let Some(v1_map) = merged.get(k) {
            let mut new_map = v1_map.clone();
            for (depth, child2) in v2_map.iter() {
                if let Some(child1) = new_map.get(depth) {
                    let merged_child = merge_fn(child1, child2);
                    new_map.insert(*depth, merged_child);
                } else {
                    new_map.insert(*depth, child2.clone());
                }
            }
            merged.insert(k.clone(), new_map);
        } else {
            merged.insert(k.clone(), v2_map.clone());
        }
    }
    merged
}

fn new_lower<T: Clone + Eq + Hash>(children: Children<T, Lower<T>>, empty: bool) -> Arc<Lower<T>> {
    let max_depth = max_depth_from_children(&children, |n: &Arc<Lower<T>>| n.max_depth);
    Arc::new(Lower {
        children,
        empty,
        max_depth,
    })
}

fn new_interface<T, A>(inner: Arc<Lower<T>>, acc: A) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    Arc::new(Upper::Interface(Arc::new(Interface { inner, acc })))
}

fn new_branch<T, A>(
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let max_depth = max_depth_from_children(&children, |n: &Arc<Upper<T, A>>| n.max_depth());
    Arc::new(Upper::Branch(Arc::new(UpperBranch {
        children,
        empty,
        max_depth,
    })))
}

fn empty_upper_inner<T, A>() -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    new_branch(IHashMap::new(), None)
}

// --------------------
// Filtering
// --------------------

fn filter_lower<T: Clone + Eq + Hash>(
    node: &Arc<Lower<T>>,
    current_depth: isize,
    min_len: Option<isize>,
    max_len: Option<isize>,
) -> Option<Arc<Lower<T>>> {
    let min_d = min_len.unwrap_or(0);
    let max_d = max_len.unwrap_or(isize::MAX);

    if current_depth > max_d {
        return None;
    }

    let keep_empty = node.empty && current_depth >= min_d;

    let mut new_children: Children<T, Lower<T>> = IHashMap::new();
    let mut children_identical = true;

    if current_depth < max_d {
        for (v, kids) in node.children.iter() {
            let mut new_kids: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
            let mut same_kids = true;
            let mut count = 0usize;
            for (orig_depth, child) in kids.iter() {
                if let Some(new_child) =
                    filter_lower(child, current_depth + 1, min_len, max_len)
                {
                    // Preserve identicality if pointer and depth key match
                    if !Arc::ptr_eq(&new_child, child) || new_child.max_depth != *orig_depth {
                        same_kids = false;
                    }
                    new_kids.insert(new_child.max_depth, new_child);
                    count += 1;
                } else {
                    same_kids = false;
                }
            }
            if count > 0 {
                new_children.insert(v.clone(), new_kids);
            } else {
                children_identical = false;
            }
            children_identical &= same_kids;
        }
    } else {
        // We cannot descend; identical only if there were no children to begin with
        children_identical = node.children.is_empty();
    }

    // If nothing changed at this node, return the original pointer for maximal sharing.
    if keep_empty == node.empty && children_identical {
        return Some(node.clone());
    }

    if !keep_empty && new_children.is_empty() {
        None
    } else {
        Some(new_lower(new_children, keep_empty))
    }
}

fn filter_upper<T, A>(
    node: &Arc<Upper<T, A>>,
    current_depth: isize,
    min_len: Option<isize>,
    max_len: Option<isize>,
) -> Option<Arc<Upper<T, A>>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let min_d = min_len.unwrap_or(0);
    let max_d = max_len.unwrap_or(isize::MAX);

    if current_depth > max_d {
        return None;
    }

    match &**node {
        Upper::Branch(b) => {
            let keep_empty = b.empty.is_some() && current_depth >= min_d;
            let new_empty = if keep_empty { b.empty.clone() } else { None };

            let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
            let mut children_identical = true;
            if current_depth < max_d {
                for (v, kids) in b.children.iter() {
                    let mut new_kids: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
                    let mut same_kids = true;
                    let mut count = 0usize;
                    for (orig_depth, child) in kids.iter() {
                        if let Some(new_child) =
                            filter_upper(child, current_depth + 1, min_len, max_len)
                        {
                            if !Arc::ptr_eq(&new_child, child)
                                || new_child.max_depth() != *orig_depth
                            {
                                same_kids = false;
                            }
                            new_kids.insert(new_child.max_depth(), new_child);
                            count += 1;
                        } else {
                            same_kids = false;
                        }
                    }
                    if count > 0 {
                        new_children.insert(v.clone(), new_kids);
                    } else {
                        children_identical = false;
                    }
                    children_identical &= same_kids;
                }
            } else {
                // We cannot descend; identical only if there were no children to begin with
                children_identical = b.children.is_empty();
            }

            // If nothing changed at this node, return original pointer for maximal sharing.
            if new_empty == b.empty && children_identical {
                return Some(node.clone());
            }

            if new_children.is_empty() && new_empty.is_none() {
                None
            } else {
                let new_b = new_branch(new_children, new_empty);
                Some(try_promote(&new_b))
            }
        }
        Upper::Interface(i) => {
            let keep_empty = i.inner.empty && current_depth >= min_d;

            let mut new_children: Children<T, Lower<T>> = IHashMap::new();
            let mut children_identical = true;
            if current_depth < max_d {
                for (v, kids) in i.inner.children.iter() {
                    let mut new_kids: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
                    let mut same_kids = true;
                    let mut count = 0usize;
                    for (orig_depth, child) in kids.iter() {
                        if let Some(new_child) =
                            filter_lower(child, current_depth + 1, min_len, max_len)
                        {
                            if !Arc::ptr_eq(&new_child, child)
                                || new_child.max_depth != *orig_depth
                            {
                                same_kids = false;
                            }
                            new_kids.insert(new_child.max_depth, new_child);
                            count += 1;
                        } else {
                            same_kids = false;
                        }
                    }
                    if count > 0 {
                        new_children.insert(v.clone(), new_kids);
                    } else {
                        children_identical = false;
                    }
                    children_identical &= same_kids;
                }
            } else {
                children_identical = i.inner.children.is_empty();
            }

            if !keep_empty && new_children.is_empty() {
                return None;
            }

            if keep_empty == i.inner.empty && children_identical {
                return Some(node.clone());
            }

            let new_inner = new_lower(new_children, keep_empty);
            Some(new_interface(new_inner, i.acc.clone()))
        }
    }
}

// --------------------
// Conversions and merges
// --------------------

fn merge_lower<T: Clone + Eq + Hash>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {
    if Arc::ptr_eq(l1, l2) {
        return l1.clone();
    }
    let new_empty = l1.empty || l2.empty;
    let merged_children = merge_children(&l1.children, &l2.children, |a, b| merge_lower(a, b));
    new_lower(merged_children, new_empty)
}

fn interface_to_upperbranch<T, A>(it: &Arc<Interface<T, A>>) -> Arc<UpperBranch<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let mut children: Children<T, Upper<T, A>> = IHashMap::new();
    for (v, kids) in it.inner.children.iter() {
        let mut v_map: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
        for lchild in kids.values() {
            let ci = new_interface(lchild.clone(), it.acc.clone());
            v_map.insert(ci.max_depth(), ci);
        }
        if !v_map.is_empty() {
            children.insert(v.clone(), v_map);
        }
    }

    let new_empty = if it.inner.empty {
        Some(it.acc.clone())
    } else {
        None
    };

    let max_depth = max_depth_from_children(&children, |n: &Arc<Upper<T, A>>| n.max_depth());
    Arc::new(UpperBranch {
        children,
        empty: new_empty,
        max_depth,
    })
}

fn merge_upperbranches<T, A>(
    a: &Arc<UpperBranch<T, A>>,
    b: &Arc<UpperBranch<T, A>>,
) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    if Arc::ptr_eq(a, b) {
        return Arc::new(Upper::Branch(a.clone()));
    }
    let new_empty = merge_optional_acc(&a.empty, &b.empty);
    let merged_children = merge_children(&a.children, &b.children, |x, y| merge_upper(x, y));
    let new_b = new_branch(merged_children, new_empty);
    try_promote(&new_b)
}

fn merge_interfaces<T, A>(a: &Arc<Interface<T, A>>, b: &Arc<Interface<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    // Keep as Interface if:
    // - both represent exactly the same set of stacks (lower identity: inner pointer-eq), or
    // - both already have the same accumulator.
    if a.acc == b.acc || Arc::ptr_eq(&a.inner, &b.inner) {
        let merged_lower = merge_lower(&a.inner, &b.inner);
        let new_acc = a.acc.merge(&b.acc);
        new_interface(merged_lower, new_acc)
    } else {
        let ub1 = interface_to_upperbranch(a);
        let ub2 = interface_to_upperbranch(b);
        merge_upperbranches(&ub1, &ub2)
    }
}

fn merge_upper<T, A>(u1: &Arc<Upper<T, A>>, u2: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    if Arc::ptr_eq(u1, u2) {
        return u1.clone();
    }
    match (&**u1, &**u2) {
        (Upper::Interface(i1), Upper::Interface(i2)) => merge_interfaces(i1, i2),
        (Upper::Branch(b1), Upper::Branch(b2)) => merge_upperbranches(b1, b2),
        (Upper::Branch(b), Upper::Interface(i)) | (Upper::Interface(i), Upper::Branch(b)) => {
            let ub = interface_to_upperbranch(i);
            merge_upperbranches(b, &ub)
        }
    }
}

fn try_promote<T, A>(node: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    if let Upper::Branch(b) = &**node {
        let all_children: Vec<_> = b
            .children
            .values()
            .flat_map(|kids| kids.values())
            .collect();

        // Leaf-branch with explicit empty: represent as Interface with lower.empty=True and no children
        if all_children.is_empty() {
            if let Some(empty) = &b.empty {
                let lower_root = new_lower(IHashMap::new(), true);
                return new_interface(lower_root, empty.clone());
            }
            return node.clone();
        }

        // Must have all Interface children to be promotable
        if !all_children
            .iter()
            .all(|c| matches!(&***c, Upper::Interface(_)))
        {
            return node.clone();
        }

        // Collect all accumulators present across U.empty and children's acc
        let mut accs: HashSet<A> = HashSet::new();
        if let Some(empty) = &b.empty {
            accs.insert(empty.clone());
        }
        for c in all_children {
            if let Upper::Interface(ic) = &**c {
                accs.insert(ic.acc.clone());
            }
        }

        if accs.len() <= 1 {
            if let Some(the_acc) = accs.into_iter().next() {
                // Build lower layer by collapsing interface children to lowers
                let mut l_children: Children<T, Lower<T>> = IHashMap::new();
                for (v, kids) in b.children.iter() {
                    let mut v_map: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
                    for child in kids.values() {
                        if let Upper::Interface(ci) = &**child {
                            let lower = ci.inner.clone();
                            v_map.insert(lower.max_depth, lower);
                        }
                    }
                    if !v_map.is_empty() {
                        l_children.insert(v.clone(), v_map);
                    }
                }
                let lower_root = new_lower(l_children, b.empty.is_some());
                return new_interface(lower_root, the_acc);
            } else {
                return empty_upper_inner();
            }
        }
    }
    node.clone()
}

fn empty_upper<T, A>() -> LeveledGSS<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    LeveledGSS {
        inner: empty_upper_inner(),
    }
}

#[derive(Clone, PartialEq)]
pub struct LeveledGSSStats<T: Clone + Eq + Hash, A: Clone + Eq + Hash> {
    pub top_values: HashSet<T>,
    pub num_upperbranch_nodes: usize,
    pub num_interface_nodes: usize,
    pub num_lower_nodes: usize,
    pub total_unique_nodes: usize,
    pub num_structurally_unique_nodes: usize,
    pub upper_edges: usize,
    pub interface_to_lower_edges: usize,
    pub lower_edges: usize,
    pub total_edges: usize,
    pub max_upper_depth: isize,
    pub max_lower_depth: isize,
    pub distinct_values_count: usize,
    pub distinct_values: HashSet<T>,
    pub unique_accumulators_count: usize,
    pub unique_accumulators: HashSet<A>,
    pub total_accumulator_instances: usize,
    pub num_upper_with_empty: usize,
    pub num_interfaces_with_empty: usize,
    pub num_lower_terminal_nodes: usize,
    pub num_interface_implicit_terminals: usize,
    pub num_multi_depth_slots_upper: usize,
    pub num_multi_depth_slots_lower: usize,
    pub max_multiplicity_per_value_upper: usize,
    pub max_multiplicity_per_value_lower: usize,
    pub average_in_degree: f64,
    pub max_in_degree: usize,
    pub structural_sharing_factor: f64,
    pub promotable_upper_nodes: usize,
}

impl<T: Clone + Eq + Hash + std::fmt::Debug, A: Clone + Eq + Hash + std::fmt::Debug> std::fmt::Debug
    for LeveledGSSStats<T, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeveledGSSStats")
            // .field("top_values", &self.top_values)
            .field("num_upperbranch_nodes", &self.num_upperbranch_nodes)
            .field("num_interface_nodes", &self.num_interface_nodes)
            .field("num_lower_nodes", &self.num_lower_nodes)
            .field("total_unique_nodes", &self.total_unique_nodes)
            .field("num_structurally_unique_nodes", &self.num_structurally_unique_nodes)
            .field("upper_edges", &self.upper_edges)
            .field("interface_to_lower_edges", &self.interface_to_lower_edges)
            .field("lower_edges", &self.lower_edges)
            .field("total_edges", &self.total_edges)
            .field("max_upper_depth", &self.max_upper_depth)
            .field("max_lower_depth", &self.max_lower_depth)
            .field("distinct_values_count", &self.distinct_values_count)
            // .field("distinct_values", &self.distinct_values)
            .field("unique_accumulators_count", &self.unique_accumulators_count)
            // .field("unique_accumulators", &self.unique_accumulators)
            .field("total_accumulator_instances", &self.total_accumulator_instances)
            .field("num_upper_with_empty", &self.num_upper_with_empty)
            .field("num_interfaces_with_empty", &self.num_interfaces_with_empty)
            .field("num_lower_terminal_nodes", &self.num_lower_terminal_nodes)
            .field("num_interface_implicit_terminals", &self.num_interface_implicit_terminals)
            .field("num_multi_depth_slots_upper", &self.num_multi_depth_slots_upper)
            .field("num_multi_depth_slots_lower", &self.num_multi_depth_slots_lower)
            .field("max_multiplicity_per_value_upper", &self.max_multiplicity_per_value_upper)
            .field("max_multiplicity_per_value_lower", &self.max_multiplicity_per_value_lower)
            .field("average_in_degree", &self.average_in_degree)
            .field("max_in_degree", &self.max_in_degree)
            .field("structural_sharing_factor", &self.structural_sharing_factor)
            .field("promotable_upper_nodes", &self.promotable_upper_nodes)
            .finish()
    }
}

fn is_promotable<T, A>(node: &Arc<UpperBranch<T, A>>) -> bool
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let all_children: Vec<_> = node
        .children
        .values()
        .flat_map(|kids| kids.values())
        .collect();
    if all_children.is_empty() {
        // Under new rule, leaf UpperBranch with an empty acc is promotable.
        return node.empty.is_some();
    }
    if !all_children
        .iter()
        .all(|c| matches!(&***c, Upper::Interface(_)))
    {
        return false;
    }
    let mut accs: HashSet<A> = HashSet::new();
    if let Some(empty) = &node.empty {
        accs.insert(empty.clone());
    }
    for c in all_children {
        if let Upper::Interface(ic) = &**c {
            accs.insert(ic.acc.clone());
        }
    }
    accs.len() <= 1
}

// --------------------
// Normalization (hash-cons + depth fusion)
// --------------------

#[derive(Clone, PartialEq, Eq)]
struct LowerSig<T: Clone + Eq + Hash> {
    empty: bool,
    // Order-independent: label -> sorted list of canonical child ids
    edges: StdHashMap<T, Vec<usize>>,
}

#[derive(Clone, PartialEq, Eq)]
enum UpperSig<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    Branch {
        empty: Option<A>,
        // Order-independent: label -> sorted list of canonical child ids (Upper)
        edges: StdHashMap<T, Vec<usize>>,
    },
    Interface {
        acc: A,
        // Order-independent: label -> sorted list of canonical child ids (Lower)
        edges: StdHashMap<T, Vec<usize>>,
        // Note: Interface no longer stores explicit empty accumulator.
    },
}

struct NormalizationLowerInterner<T: Clone + Eq + Hash> {
    // hash -> bucket of (signature, canonical id, canonical node)
    map: StdHashMap<u64, Vec<(LowerSig<T>, usize, Arc<Lower<T>>)>>,
    next_id: usize,
}

impl<T: Clone + Eq + Hash> Default for NormalizationLowerInterner<T> {
    fn default() -> Self {
        Self {
            map: StdHashMap::new(),
            next_id: 0,
        }
    }
}

struct NormalizationUpperInterner<
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
> {
    // hash -> bucket of (signature, canonical id, canonical node)
    map: StdHashMap<u64, Vec<(UpperSig<T, A>, usize, Arc<Upper<T, A>>)>>,
    next_id: usize,
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Default
    for NormalizationUpperInterner<T, A>
{
    fn default() -> Self {
        Self {
            map: StdHashMap::new(),
            next_id: 0,
        }
    }
}

fn lower_sig_hash<T: Clone + Eq + Hash>(sig: &LowerSig<T>) -> u64 {
    use std::hash::{Hash as _, Hasher};
    let mut seed: u64 = if sig.empty {
        0x9e37_79b9_7f4a_7c15
    } else {
        0x243f_6a88_85a3_08d3
    };
    seed ^= (sig.edges.len() as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    let mut xor_acc: u64 = 0;
    let mut sum_acc: u64 = 0;
    let mut prod_acc: u64 = 0x517c_c1b7_2722_0a95 ^ (sig.edges.len() as u64);
    for (k, ids) in &sig.edges {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        ids.hash(&mut h); // ids are sorted
        let e = h.finish();
        xor_acc ^= e;
        sum_acc = sum_acc.wrapping_add(e);
        prod_acc = prod_acc.wrapping_mul(e.wrapping_add(0x9e37_79b9_7f4a_7c15));
    }
    seed ^ xor_acc ^ sum_acc ^ prod_acc
}

fn upper_sig_hash<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    sig: &UpperSig<T, A>,
) -> u64 {
    use std::hash::{Hash as _, Hasher};
    let mut seed: u64 = match sig {
        UpperSig::Branch { empty, edges } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            if let Some(e) = empty {
                e.hash(&mut h);
            }
            0x6a09_e667_f3bc_c908u64 ^ h.finish() ^ (edges.len() as u64)
        }
        UpperSig::Interface { acc, edges } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            acc.hash(&mut h);
            0xbb67_ae85_84ca_a73bu64 ^ h.finish() ^ (edges.len() as u64)
        }
    };
    let edges = match sig {
        UpperSig::Branch { edges, .. } => edges,
        UpperSig::Interface { edges, .. } => edges,
    };
    let mut xor_acc: u64 = 0;
    let mut sum_acc: u64 = 0;
    let mut prod_acc: u64 = 0x3c6e_f372_fe94_f82b ^ (edges.len() as u64);
    for (k, ids) in edges {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        ids.hash(&mut h); // ids are sorted
        let e = h.finish();
        xor_acc ^= e;
        sum_acc = sum_acc.wrapping_add(e);
        prod_acc = prod_acc.wrapping_mul(e.wrapping_add(0xa54f_f53a_5f1d_36f1));
    }
    seed ^ xor_acc ^ sum_acc ^ prod_acc
}

fn normalize_canonicalize_lower<T>(
    node: &Arc<Lower<T>>,
    memo_lower: &mut StdHashMap<usize, (usize, Arc<Lower<T>>)>,
    interner_lower: &mut NormalizationLowerInterner<T>,
) -> (usize, Arc<Lower<T>>)
where
    T: Clone + Eq + Hash,
{
    let ptr = Arc::as_ptr(node) as usize;
    if let Some((id, arc)) = memo_lower.get(&ptr) {
        return (*id, arc.clone());
    }

    // Recurse: canonicalize children first.
    let mut edges_raw: StdHashMap<T, Vec<(usize, Arc<Lower<T>>)>> = StdHashMap::new();
    for (v, kids) in node.children.iter() {
        let entry = edges_raw.entry(v.clone()).or_default();
        for child in kids.values() {
            let (cid, carc) = normalize_canonicalize_lower(child, memo_lower, interner_lower);
            entry.push((cid, carc));
        }
    }

    // Build order-independent signature: label -> sorted child ids
    let mut sig_edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
    for (v, items) in &edges_raw {
        let mut ids: Vec<usize> = items.iter().map(|(cid, _)| *cid).collect();
        ids.sort_unstable();
        sig_edges.insert(v.clone(), ids);
    }
    let sig = LowerSig {
        empty: node.empty,
        edges: sig_edges,
    };
    let h = lower_sig_hash(&sig);

    if let Some(bucket) = interner_lower.map.get_mut(&h) {
        for (existing_sig, id, arc) in bucket.iter() {
            if existing_sig == &sig {
                memo_lower.insert(ptr, (*id, arc.clone()));
                return (*id, arc.clone());
            }
        }
    }

    // Rebuild canonical children: deduplicate by depth key, merging if necessary.
    let mut new_children: Children<T, Lower<T>> = IHashMap::new();
    for (v, items) in edges_raw {
        let mut ord: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
        for (_, carc) in items {
            let d = carc.max_depth;
            if let Some(prev) = ord.get(&d) {
                let merged = merge_lower(prev, &carc);
                ord.insert(d, merged);
            } else {
                ord.insert(d, carc);
            }
        }
        if !ord.is_empty() {
            new_children.insert(v, ord);
        }
    }
    let new_node = new_lower(new_children, node.empty);

    let id = interner_lower.next_id;
    interner_lower.next_id += 1;
    interner_lower
        .map
        .entry(h)
        .or_default()
        .push((sig, id, new_node.clone()));
    memo_lower.insert(ptr, (id, new_node.clone()));
    (id, new_node)
}

fn normalize_canonicalize_upper<T, A>(
    node: &Arc<Upper<T, A>>,
    memo_upper: &mut StdHashMap<usize, (usize, Arc<Upper<T, A>>)>,
    memo_lower: &mut StdHashMap<usize, (usize, Arc<Lower<T>>)>,
    interner_upper: &mut NormalizationUpperInterner<T, A>,
    interner_lower: &mut NormalizationLowerInterner<T>,
) -> (usize, Arc<Upper<T, A>>)
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let ptr = Arc::as_ptr(node) as usize;
    if let Some((id, arc)) = memo_upper.get(&ptr) {
        return (*id, arc.clone());
    }

    match &**node {
        Upper::Branch(b) => {
            // Recurse into children.
            let mut edges_raw: StdHashMap<T, Vec<(usize, Arc<Upper<T, A>>)>> = StdHashMap::new();
            for (v, kids) in b.children.iter() {
                let entry = edges_raw.entry(v.clone()).or_default();
                for child in kids.values() {
                    let (cid, carc) = normalize_canonicalize_upper(
                        child,
                        memo_upper,
                        memo_lower,
                        interner_upper,
                        interner_lower,
                    );
                    entry.push((cid, carc));
                }
            }

            // Compute signature (branch)
            let mut sig_edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
            for (v, items) in &edges_raw {
                let mut ids: Vec<usize> = items.iter().map(|(cid, _)| *cid).collect();
                ids.sort_unstable();
                sig_edges.insert(v.clone(), ids);
            }
            let mut sig = UpperSig::Branch {
                empty: b.empty.clone(),
                edges: sig_edges,
            };
            let mut h = upper_sig_hash(&sig);

            if let Some(bucket) = interner_upper.map.get_mut(&h) {
                for (existing_sig, id, arc) in bucket.iter() {
                    if existing_sig == &sig {
                        memo_upper.insert(ptr, (*id, arc.clone()));
                        return (*id, arc.clone());
                    }
                }
            }

            // Build new branch with canonical children; dedup per depth.
            let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
            for (v, items) in &edges_raw {
                let mut ord: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
                for (_, carc) in items {
                    let d = carc.max_depth();
                    if let Some(prev) = ord.get(&d) {
                        let merged = merge_upper(prev, carc);
                        ord.insert(d, merged);
                    } else {
                        ord.insert(d, carc.clone());
                    }
                }
                if !ord.is_empty() {
                    new_children.insert(v.clone(), ord);
                }
            }
            let new_b = new_branch(new_children, b.empty.clone());
            let mut new_node = try_promote(&new_b); // Might become Interface

            // If promoted, ensure lower children are canonical and rebuild the interface for interning.
            match &*new_node {
                Upper::Interface(i2) => {
                    // Canonicalize lower children
                    let mut edges_raw_lower: StdHashMap<T, Vec<(usize, Arc<Lower<T>>)>> =
                        StdHashMap::new();
                    for (v, kids) in i2.inner.children.iter() {
                        let entry = edges_raw_lower.entry(v.clone()).or_default();
                        for child_l in kids.values() {
                            let (lid, larc) =
                                normalize_canonicalize_lower(child_l, memo_lower, interner_lower);
                            entry.push((lid, larc));
                        }
                    }
                    // Rebuild interface with canonical lower children; dedup per depth
                    let mut final_children: Children<T, Lower<T>> = IHashMap::new();
                    for (v, items) in &edges_raw_lower {
                        let mut ord: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
                        for (_, larc) in items {
                            let d = larc.max_depth;
                            if let Some(prev) = ord.get(&d) {
                                let merged = merge_lower(prev, larc);
                                ord.insert(d, merged);
                            } else {
                                ord.insert(d, larc.clone());
                            }
                        }
                        if !ord.is_empty() {
                            final_children.insert(v.clone(), ord);
                        }
                    }
                    let rebuilt_iface =
                        new_interface(new_lower(final_children, i2.inner.empty), i2.acc.clone());
                    new_node = rebuilt_iface;

                    // Compute interface signature
                    let mut sig_edges2: StdHashMap<T, Vec<usize>> = StdHashMap::new();
                    if let Upper::Interface(i3) = &*new_node {
                        for (v, kids) in i3.inner.children.iter() {
                            let mut ids: Vec<usize> = kids
                                .values()
                                .map(|child_l| {
                                    let (lid, _) = normalize_canonicalize_lower(
                                        child_l,
                                        memo_lower,
                                        interner_lower,
                                    );
                                    lid
                                })
                                .collect();
                            ids.sort_unstable();
                            sig_edges2.insert(v.clone(), ids);
                        }
                        sig = UpperSig::Interface {
                            acc: i3.acc.clone(),
                            edges: sig_edges2,
                        };
                        h = upper_sig_hash(&sig);
                    }
                }
                Upper::Branch(_) => {
                    // keep branch signature as computed
                }
            }

            // Intern (possibly with updated signature)
            if let Some(bucket) = interner_upper.map.get_mut(&h) {
                for (existing_sig, id, arc) in bucket.iter() {
                    if existing_sig == &sig {
                        memo_upper.insert(ptr, (*id, arc.clone()));
                        return (*id, arc.clone());
                    }
                }
            }

            let id = interner_upper.next_id;
            interner_upper.next_id += 1;
            interner_upper
                .map
                .entry(h)
                .or_default()
                .push((sig, id, new_node.clone()));
            memo_upper.insert(ptr, (id, new_node.clone()));
            (id, new_node)
        }
        Upper::Interface(i) => {
            // Canonicalize lower children
            let mut edges_raw_lower: StdHashMap<T, Vec<(usize, Arc<Lower<T>>)>> =
                StdHashMap::new();
            for (v, kids) in i.inner.children.iter() {
                let entry = edges_raw_lower.entry(v.clone()).or_default();
                for child_l in kids.values() {
                    let (lid, larc) =
                        normalize_canonicalize_lower(child_l, memo_lower, interner_lower);
                    entry.push((lid, larc));
                }
            }
            // Build signature
            let mut sig_edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
            for (v, items) in &edges_raw_lower {
                let mut ids: Vec<usize> = items.iter().map(|(lid, _)| *lid).collect();
                ids.sort_unstable();
                sig_edges.insert(v.clone(), ids);
            }
            let sig = UpperSig::Interface {
                acc: i.acc.clone(),
                edges: sig_edges,
            };
            let h = upper_sig_hash(&sig);

            if let Some(bucket) = interner_upper.map.get_mut(&h) {
                for (existing_sig, id, arc) in bucket.iter() {
                    if existing_sig == &sig {
                        memo_upper.insert(ptr, (*id, arc.clone()));
                        return (*id, arc.clone());
                    }
                }
            }

            // Rebuild canonical interface children; dedup per depth
            let mut final_children: Children<T, Lower<T>> = IHashMap::new();
            for (v, items) in edges_raw_lower {
                let mut ord: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
                for (_, larc) in items {
                    let d = larc.max_depth;
                    if let Some(prev) = ord.get(&d) {
                        let merged = merge_lower(prev, &larc);
                        ord.insert(d, merged);
                    } else {
                        ord.insert(d, larc);
                    }
                }
                if !ord.is_empty() {
                    final_children.insert(v, ord);
                }
            }
            let new_node = new_interface(new_lower(final_children, i.inner.empty), i.acc.clone());

            let id = interner_upper.next_id;
            interner_upper.next_id += 1;
            interner_upper
                .map
                .entry(h)
                .or_default()
                .push((sig, id, new_node.clone()));
            memo_upper.insert(ptr, (id, new_node.clone()));
            (id, new_node)
        }
    }
}

// --------------------
// Public GSS type
// --------------------

#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Upper<T, A>>,
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn inner_ptrs_eq(&self, other: &Self) -> bool {
        match (&*self.inner, &*other.inner) {
            (Upper::Branch(b1), Upper::Branch(b2)) => {
                if b1.empty != b2.empty || b1.children.len() != b2.children.len() || b1.max_depth != b2.max_depth {
                    return false;
                }
                let keys1: HashSet<_> = b1.children.keys().collect();
                let keys2: HashSet<_> = b2.children.keys().collect();
                if keys1 != keys2 {
                    return false;
                }
                for (v, kids1) in b1.children.iter() {
                    let kids2 = b2.children.get(v).unwrap();
                    if kids1.len() != kids2.len() || !kids1.keys().eq(kids2.keys()) {
                        return false;
                    }
                    for (d, c1) in kids1.iter() {
                        let c2 = kids2.get(d).unwrap();
                        if !Arc::ptr_eq(c1, c2) {
                            return false;
                        }
                    }
                }
                true
            }
            (Upper::Interface(i1), Upper::Interface(i2)) => {
                if i1.acc != i2.acc || i1.inner.children.len() != i2.inner.children.len() || i1.inner.max_depth != i2.inner.max_depth {
                    return false;
                }
                let keys1: HashSet<_> = i1.inner.children.keys().collect();
                let keys2: HashSet<_> = i2.inner.children.keys().collect();
                if keys1 != keys2 {
                    return false;
                }
                for (v, kids1) in i1.inner.children.iter() {
                    let kids2 = i2.inner.children.get(v).unwrap();
                    if kids1.len() != kids2.len() || !kids1.keys().eq(kids2.keys()) {
                        return false;
                    }
                    for (d, c1) in kids1.iter() {
                        let c2 = kids2.get(d).unwrap();
                        if !Arc::ptr_eq(c1, c2) {
                            return false;
                        }
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn empty() -> Self {
        empty_upper()
    }

    pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {
        // Canonicalize: merge accumulators for identical stacks
        let mut canon: StdHashMap<Vec<T>, A> = StdHashMap::new();
        for (vals, acc) in stacks {
            if let Some(existing) = canon.get_mut(vals) {
                let merged = existing.merge(acc);
                *existing = merged;
            } else {
                canon.insert(vals.clone(), acc.clone());
            }
        }

        // Build a trie: map value -> { end: Option<A>, sub: Trie }
        struct Entry<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
            end: Option<A>,
            sub: StdHashMap<T, Entry<T, A>>,
        }

        impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Default for Entry<T, A> {
            fn default() -> Self {
                Self {
                    end: None,
                    sub: StdHashMap::new(),
                }
            }
        }

        let mut trie: StdHashMap<T, Entry<T, A>> = StdHashMap::new();
        let mut empty_acc: Option<A> = None;

        for (mut vals, acc) in canon.into_iter() {
            if vals.is_empty() {
                empty_acc = match empty_acc.take() {
                    None => Some(acc),
                    Some(prev) => Some(prev.merge(&acc)),
                };
                continue;
            }

            vals.reverse();
            if let Some(last_val) = vals.pop() {
                let mut node = &mut trie;
                for v in vals {
                    node = &mut node.entry(v).or_default().sub;
                }
                let final_entry = node.entry(last_val).or_default();
                final_entry.end = Some(acc);
            }
        }

        fn build_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            d: &StdHashMap<T, Entry<T, A>>,
        ) -> Arc<Lower<T>> {
            let mut l_children: Children<T, Lower<T>> = IHashMap::new();
            for (v, e) in d.iter() {
                let sub_children = if e.sub.is_empty() {
                    IHashMap::new()
                } else {
                    build_lower(&e.sub).children.clone()
                };
                let node_for_v = new_lower(sub_children, e.end.is_some());
                l_children.insert(v.clone(), OrdMap::unit(node_for_v.max_depth, node_for_v));
            }
            new_lower(l_children, false)
        }

        fn build_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            d: &StdHashMap<T, Entry<T, A>>,
            root_empty: Option<A>,
        ) -> Arc<Upper<T, A>> {
            let mut children: Children<T, Upper<T, A>> = IHashMap::new();
            let mut all_child_nodes: Vec<Arc<Upper<T, A>>> = Vec::new();

            for (v, e) in d.iter() {
                let mut nodes_for_v: Vec<Arc<Upper<T, A>>> = Vec::new();
                if let Some(end_acc) = &e.end {
                    let leaf = new_branch(IHashMap::new(), Some(end_acc.clone()));
                    nodes_for_v.push(try_promote(&leaf));
                }
                if !e.sub.is_empty() {
                    nodes_for_v.push(build_upper(&e.sub, None));
                }
                if !nodes_for_v.is_empty() {
                    let mut kids_map: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
                    for n in nodes_for_v.iter() {
                        kids_map.insert(n.max_depth(), n.clone());
                    }
                    children.insert(v.clone(), kids_map);
                    all_child_nodes.extend(nodes_for_v);
                }
            }

            // If all children are Interfaces and share a single accumulator (including root_empty),
            // we can represent this level as an Interface with a Lower tree.
            let all_interfaces = all_child_nodes
                .iter()
                .all(|c| matches!(&**c, Upper::Interface(_)));

            if all_interfaces {
                let mut accs: HashSet<A> = HashSet::new();
                for node in &all_child_nodes {
                    if let Upper::Interface(i) = &**node {
                        accs.insert(i.acc.clone());
                    }
                }
                if let Some(e) = &root_empty {
                    accs.insert(e.clone());
                }

                if accs.len() <= 1 {
                    if let Some(the_acc) = accs.into_iter().next() {
                        let lower_tree = build_lower(d);
                        let lower_root = new_lower(lower_tree.children.clone(), root_empty.is_some());
                        return new_interface(lower_root, the_acc);
                    } else {
                        return empty_upper_inner();
                    }
                }
            }

            new_branch(children, root_empty)
        }

        LeveledGSS {
            inner: build_upper(&trie, empty_acc),
        }
    }

    pub fn to_stacks(&self) -> Vec<(Vec<T>, A)> {
        let mut res: Vec<(Vec<T>, A)> = Vec::new();

        fn dfs_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            l: &Lower<T>,
            pref: &mut Vec<T>,
            acc: &A,
            out: &mut Vec<(Vec<T>, A)>,
        ) {
            if l.empty {
                let mut stack = pref.clone();
                stack.reverse();
                out.push((stack, acc.clone()));
            }
            for (v, kids) in l.children.iter() {
                for child in kids.values() {
                    pref.push(v.clone());
                    dfs_lower(child, pref, acc, out);
                    pref.pop();
                }
            }
        }

        fn dfs_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            u: &Upper<T, A>,
            pref: &mut Vec<T>,
            out: &mut Vec<(Vec<T>, A)>,
        ) {
            match u {
                Upper::Branch(b) => {
                    if let Some(e) = &b.empty {
                        let mut stack = pref.clone();
                        stack.reverse();
                        out.push((stack, e.clone()));
                    }
                    for (v, kids) in b.children.iter() {
                        for child in kids.values() {
                            pref.push(v.clone());
                            dfs_upper(child, pref, out);
                            pref.pop();
                        }
                    }
                }
                Upper::Interface(i) => {
                    if i.inner.empty {
                        let mut stack = pref.clone();
                        stack.reverse();
                        out.push((stack, i.acc.clone()));
                    }
                    for (v, kids) in i.inner.children.iter() {
                        for child in kids.values() {
                            pref.push(v.clone());
                            dfs_lower(child, pref, &i.acc, out);
                            pref.pop();
                        }
                    }
                }
            }
        }

        dfs_upper(&self.inner, &mut vec![], &mut res);
        res
    }

    pub fn push(&self, value: T) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let new_inner = match &*self.inner {
            Upper::Interface(i) => {
                let mut new_children: Children<T, Lower<T>> = IHashMap::new();
                new_children.insert(value, OrdMap::unit(i.inner.max_depth, i.inner.clone()));
                let new_lower_root = new_lower(new_children, false);
                new_interface(new_lower_root, i.acc.clone())
            }
            Upper::Branch(_) => {
                let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
                new_children.insert(
                    value,
                    OrdMap::unit(self.inner.max_depth(), self.inner.clone()),
                );
                new_branch(new_children, None)
            }
        };
        LeveledGSS { inner: new_inner }
    }

    pub fn popn(&self, n: isize) -> Self {
        if n <= 0 {
            return self.clone();
        }
        if self.is_empty() {
            return self.clone();
        }

        let mut memo_upper: StdHashMap<(usize, isize), Arc<Upper<T, A>>> = StdHashMap::new();
        let mut memo_lower: StdHashMap<(usize, isize), Arc<Lower<T>>> = StdHashMap::new();

        fn popn_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Lower<T>>,
            k: isize,
            memo_lower: &mut StdHashMap<(usize, isize), Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if k == 0 {
                return node.clone();
            }
            let key = (Arc::as_ptr(node) as usize, k);
            if let Some(cached) = memo_lower.get(&key) {
                return cached.clone();
            }

            let all_children: Vec<_> = node
                .children
                .values()
                .flat_map(|kids| kids.values())
                .cloned()
                .collect();
            if all_children.is_empty() {
                let res = new_lower(IHashMap::new(), false);
                memo_lower.insert(key, res.clone());
                return res;
            }

            let popped_children: Vec<_> = all_children
                .into_iter()
                .map(|child| popn_lower::<T, A>(&child, k - 1, memo_lower))
                .collect();

            let mut it = popped_children.into_iter();
            let first = it.next().unwrap();
            let res = it.fold(first, |acc, next| merge_lower(&acc, &next));
            memo_lower.insert(key, res.clone());
            res
        }

        fn popn_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            k: isize,
            memo_upper: &mut StdHashMap<(usize, isize), Arc<Upper<T, A>>>,
            memo_lower: &mut StdHashMap<(usize, isize), Arc<Lower<T>>>,
        ) -> Arc<Upper<T, A>> {
            if k == 0 {
                return node.clone();
            }
            let key = (Arc::as_ptr(node) as usize, k);
            if let Some(cached) = memo_upper.get(&key) {
                return cached.clone();
            }

            let res = match &**node {
                Upper::Branch(b) => {
                    let all_children: Vec<_> = b
                        .children
                        .values()
                        .flat_map(|kids| kids.values())
                        .cloned()
                        .collect();
                    if all_children.is_empty() {
                        return empty_upper_inner();
                    }
                    let popped_children: Vec<_> = all_children
                        .into_iter()
                        .map(|child| popn_upper::<T, A>(&child, k - 1, memo_upper, memo_lower))
                        .collect();
                    let mut it = popped_children.into_iter();
                    let first = it.next().unwrap();
                    let merged = it.fold(first, |acc, next| merge_upper(&acc, &next));
                    try_promote(&merged)
                }
                Upper::Interface(i) => {
                    let all_children: Vec<_> = i
                        .inner
                        .children
                        .values()
                        .flat_map(|kids| kids.values())
                        .cloned()
                        .collect();
                    if all_children.is_empty() {
                        return empty_upper_inner();
                    }
                    let popped_lower = popn_lower::<T, A>(&i.inner, k, memo_lower);
                    if popped_lower.children.is_empty() && !popped_lower.empty {
                        empty_upper_inner()
                    } else {
                        new_interface(popped_lower, i.acc.clone())
                    }
                }
            };

            memo_upper.insert(key, res.clone());
            res
        }

        let new_inner = popn_upper::<T, A>(&self.inner, n, &mut memo_upper, &mut memo_lower);
        LeveledGSS { inner: new_inner }
    }

    pub fn pop(&self) -> Self {
        self.popn(1)
    }

    pub fn is_empty(&self) -> bool {
        match &*self.inner {
            Upper::Branch(b) => b.children.is_empty() && b.empty.is_none(),
            Upper::Interface(_) => false,
        }
    }

    pub fn max_depth(&self) -> isize {
        self.inner.max_depth()
    }

    pub fn isolate(&self, value: Option<T>) -> Self {
        // Fast-path: if the isolation would yield an identical structure, return self.
        if let Some(ref v) = value {
            match &*self.inner {
                Upper::Branch(b) => {
                    if b.empty.is_none() && b.children.len() == 1 && b.children.contains_key(v) {
                        return self.clone();
                    }
                }
                Upper::Interface(i) => {
                    if !i.inner.empty && i.inner.children.len() == 1 && i.inner.children.contains_key(v) {
                        return self.clone();
                    }
                }
            }
        } else {
            match &*self.inner {
                Upper::Branch(b) => {
                    if b.children.is_empty() {
                        return self.clone();
                    }
                }
                Upper::Interface(i) => {
                    if i.inner.children.is_empty() && i.inner.empty {
                        return self.clone();
                    }
                }
            }
        }

        let new_inner = if let Some(val) = value {
            match &*self.inner {
                Upper::Branch(b) => {
                    let filtered_children = b
                        .children
                        .get(&val)
                        .map(|kids| IHashMap::unit(val.clone(), kids.clone()))
                        .unwrap_or_else(IHashMap::new);
                    let new_b = new_branch(filtered_children, None);
                    try_promote(&new_b)
                }
                Upper::Interface(i) => {
                    if let Some(kids) = i.inner.children.get(&val) {
                        let filtered_children = IHashMap::unit(val.clone(), kids.clone());
                        let new_lower_root = new_lower(filtered_children, false);
                        new_interface(new_lower_root, i.acc.clone())
                    } else {
                        empty_upper_inner()
                    }
                }
            }
        } else {
            let empty_acc = match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => {
                    if i.inner.empty {
                        Some(i.acc.clone())
                    } else {
                        None
                    }
                }
            };
            let new_b = new_branch(IHashMap::new(), empty_acc);
            try_promote(&new_b)
        };
        LeveledGSS { inner: new_inner }
    }

    pub fn isolate_many<I: IntoIterator<Item = Option<T>>>(&self, values: I) -> Self {
        let values_set: HashSet<Option<T>> = values.into_iter().collect();

        // Fast-path: if the selection keeps everything exactly as-is, return self.
        match &*self.inner {
            Upper::Branch(b) => {
                let all_children_kept = b
                    .children
                    .keys()
                    .all(|k| values_set.contains(&Some(k.clone())));
                let empty_kept_ok = b.empty.is_some() == values_set.contains(&None);
                if all_children_kept && empty_kept_ok {
                    return self.clone();
                }
            }
            Upper::Interface(i) => {
                let all_children_kept = i
                    .inner
                    .children
                    .keys()
                    .all(|k| values_set.contains(&Some(k.clone())));
                let empty_kept_ok = i.inner.empty == values_set.contains(&None);
                if all_children_kept && empty_kept_ok {
                    return self.clone();
                }
            }
        }

        let new_inner = match &*self.inner {
            Upper::Branch(b) => {
                let new_empty: Option<A> = if values_set.contains(&None) {
                    b.empty.clone()
                } else {
                    None
                };
                let mut filtered_children: Children<T, Upper<T, A>> = IHashMap::new();
                for (v, kids) in b.children.iter() {
                    if values_set.contains(&Some(v.clone())) {
                        filtered_children.insert(v.clone(), kids.clone());
                    }
                }
                let new_b = new_branch(filtered_children, new_empty);
                try_promote(&new_b)
            }
            Upper::Interface(i) => {
                let keep_empty = values_set.contains(&None) && i.inner.empty;
                let mut filtered_children: Children<T, Lower<T>> = IHashMap::new();
                for (v, kids) in i.inner.children.iter() {
                    if values_set.contains(&Some(v.clone())) {
                        filtered_children.insert(v.clone(), kids.clone());
                    }
                }
                if !filtered_children.is_empty() || keep_empty {
                    let new_lower_root = new_lower(filtered_children, keep_empty);
                    new_interface(new_lower_root, i.acc.clone())
                } else {
                    new_branch(IHashMap::new(), None)
                }
            }
        };

        LeveledGSS { inner: new_inner }
    }

    pub fn filter_by_length(&self, min_len: Option<isize>, max_len: Option<isize>) -> Self {
        if self.is_empty() {
            return self.clone();
        }

        let new_inner_opt = filter_upper::<T, A>(&self.inner, 0, min_len, max_len);
        new_inner_opt.map_or_else(Self::empty, |inner| LeveledGSS { inner })
    }

    pub fn apply<B, F>(&self, mut func: F) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        F: FnMut(&A) -> B,
    {
        // Memoize per-accumulator transformation so the closure is not invoked more than once per unique A.
        let mut acc_memo: StdHashMap<A, B> = StdHashMap::new();

        fn map_acc<A, B, F>(a: &A, memo: &mut StdHashMap<A, B>, f: &mut F) -> B
        where
            A: Clone + Eq + Hash,
            B: Clone,
            F: FnMut(&A) -> B,
        {
            if let Some(v) = memo.get(a) {
                return v.clone();
            }
            let r = f(a);
            memo.insert(a.clone(), r.clone());
            r
        }

        fn transform<T, A, B, F>(
            node: &Arc<Upper<T, A>>,
            memo_acc: &mut StdHashMap<A, B>,
            f: &mut F,
        ) -> Arc<Upper<T, B>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            B: Merge + Clone + Eq + Hash,
            F: FnMut(&A) -> B,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc = map_acc(&i.acc, memo_acc, f);
                    let res = new_interface(i.inner.clone(), new_acc);
                    try_promote(&res)
                }
                Upper::Branch(b) => {
                    let new_empty = b.empty.as_ref().map(|e| map_acc(e, memo_acc, f));
                    let mut new_children: Children<T, Upper<T, B>> = IHashMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: OrdMap<isize, Arc<Upper<T, B>>> = OrdMap::new();
                        for child in kids.values() {
                            let new_child = transform::<T, A, B, F>(child, memo_acc, f);
                            new_kids.insert(new_child.max_depth(), new_child);
                        }
                        new_children.insert(v.clone(), new_kids);
                    }
                    let res = new_branch(new_children, new_empty);
                    try_promote(&res)
                }
            }
        }

        LeveledGSS {
            inner: transform::<T, A, B, F>(&self.inner, &mut acc_memo, &mut func),
        }
    }

    pub fn prune<P>(&self, mut predicate: P) -> Self
    where
        P: FnMut(&A) -> bool,
    {
        // Memoize per-accumulator predicate
        let mut acc_memo: StdHashMap<A, bool> = StdHashMap::new();

        fn test_acc<A, P>(a: &A, memo: &mut StdHashMap<A, bool>, p: &mut P) -> bool
        where
            A: Clone + Eq + Hash,
            P: FnMut(&A) -> bool,
        {
            if let Some(v) = memo.get(a) {
                return *v;
            }
            let r = p(a);
            memo.insert(a.clone(), r);
            r
        }

        fn transform<T, A, P>(
            node: &Arc<Upper<T, A>>,
            acc_memo: &mut StdHashMap<A, bool>,
            p: &mut P,
        ) -> Option<Arc<Upper<T, A>>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            P: FnMut(&A) -> bool,
        {
            match &**node {
                Upper::Interface(i) => {
                    let keep_acc = test_acc(&i.acc, acc_memo, p);
                    if !keep_acc {
                        None
                    } else {
                        Some(node.clone())
                    }
                }
                Upper::Branch(b) => {
                    let new_empty = b
                        .empty
                        .as_ref()
                        .and_then(|e| if test_acc(e, acc_memo, p) { Some(e.clone()) } else { None });

                    let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
                    let mut children_identical = true;
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
                        let mut same_kids = true;
                        let mut count = 0usize;
                        for (orig_depth, child) in kids.iter() {
                            if let Some(nc) = transform::<T, A, P>(child, acc_memo, p) {
                                if !Arc::ptr_eq(&nc, child) || nc.max_depth() != *orig_depth {
                                    same_kids = false;
                                }
                                new_kids.insert(nc.max_depth(), nc);
                                count += 1;
                            } else {
                                same_kids = false;
                            }
                        }
                        if count > 0 {
                            new_children.insert(v.clone(), new_kids);
                        } else {
                            children_identical = false;
                        }
                        children_identical &= same_kids;
                    }

                    // If nothing changed, preserve pointer.
                    if new_empty == b.empty && children_identical {
                        return Some(node.clone());
                    }

                    if new_children.is_empty() && new_empty.is_none() {
                        None
                    } else {
                        let new_b = new_branch(new_children, new_empty);
                        Some(try_promote(&new_b))
                    }
                }
            }
        }

        let res_inner_opt = transform::<T, A, P>(&self.inner, &mut acc_memo, &mut predicate);
        res_inner_opt.map_or_else(Self::empty, |inner| LeveledGSS { inner })
    }

    pub fn apply_and_prune<B, M>(&self, mut mutator: M) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        M: FnMut(&A) -> Option<B>,
    {
        // Memoize per-accumulator mutate/prune
        let mut acc_memo: StdHashMap<A, Option<B>> = StdHashMap::new();

        fn mutate_acc<A, B, M>(
            a: &A,
            memo: &mut StdHashMap<A, Option<B>>,
            m: &mut M,
        ) -> Option<B>
        where
            A: Clone + Eq + Hash,
            B: Clone,
            M: FnMut(&A) -> Option<B>,
        {
            if let Some(v) = memo.get(a) {
                return v.clone();
            }
            let r = m(a);
            memo.insert(a.clone(), r.clone());
            r
        }

        fn transform<T, A, B, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut StdHashMap<A, Option<B>>,
            m: &mut M,
        ) -> Option<Arc<Upper<T, B>>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            B: Merge + Clone + Eq + Hash,
            M: FnMut(&A) -> Option<B>,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc_opt = mutate_acc(&i.acc, memo, m);
                    if let Some(new_acc) = new_acc_opt {
                        let new_i = new_interface(i.inner.clone(), new_acc);
                        Some(try_promote(&new_i))
                    } else {
                        None
                    }
                }
                Upper::Branch(b) => {
                    let new_empty_opt = b.empty.as_ref().and_then(|e| mutate_acc(e, memo, m));
                    let mut new_children: Children<T, Upper<T, B>> = IHashMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: OrdMap<isize, Arc<Upper<T, B>>> = OrdMap::new();
                        for child in kids.values() {
                            if let Some(nc) = transform::<T, A, B, M>(child, memo, m) {
                                new_kids.insert(nc.max_depth(), nc);
                            }
                        }
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }

                    if new_children.is_empty() && new_empty_opt.is_none() {
                        None
                    } else {
                        let new_b = new_branch(new_children, new_empty_opt);
                        Some(try_promote(&new_b))
                    }
                }
            }
        }

        let res_inner_opt = transform::<T, A, B, M>(&self.inner, &mut acc_memo, &mut mutator);
        res_inner_opt.map_or_else(LeveledGSS::<T, B>::empty, |inner| LeveledGSS::<T, B> { inner })
    }

    pub fn merge(&self, other: &Self) -> Self {
        let merged_inner = merge_upper(&self.inner, &other.inner);
        LeveledGSS {
            inner: merged_inner,
        }
    }

    pub fn fuse(&self, levels: Option<isize>) -> Self {
        if let Some(l) = levels {
            if l <= 0 {
                return self.clone();
            }
        }

        let mut memo_upper: StdHashMap<(usize, Option<isize>), Arc<Upper<T, A>>> = StdHashMap::new();
        let mut memo_lower: StdHashMap<(usize, Option<isize>), Arc<Lower<T>>> = StdHashMap::new();

        fn fuse_lower<T, A>(
            node: &Arc<Lower<T>>,
            remain: Option<isize>,
            memo: &mut StdHashMap<(usize, Option<isize>), Arc<Lower<T>>>,
        ) -> Arc<Lower<T>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            if let Some(r) = remain {
                if r == 0 {
                    return node.clone();
                }
            }
            let key = (Arc::as_ptr(node) as usize, remain);
            if let Some(cached) = memo.get(&key) {
                return cached.clone();
            }

            let next_remain = remain.map(|r| r - 1);

            let has_multi_depth_slots = node.children.values().any(|kids| kids.len() > 1);

            let mut new_children_by_value: StdHashMap<T, Vec<Arc<Lower<T>>>> = StdHashMap::new();
            let mut children_changed = false;

            for (v, kids) in node.children.iter() {
                for child in kids.values() {
                    let fused_child = fuse_lower::<T, A>(child, next_remain, memo);
                    if !Arc::ptr_eq(&fused_child, child) {
                        children_changed = true;
                    }
                    new_children_by_value
                        .entry(v.clone())
                        .or_default()
                        .push(fused_child);
                }
            }

            if !has_multi_depth_slots && !children_changed {
                memo.insert(key, node.clone());
                return node.clone();
            }

            let mut final_children: Children<T, Lower<T>> = IHashMap::new();
            for (v, fused_kids) in new_children_by_value {
                if fused_kids.is_empty() {
                    continue;
                }
                let mut it = fused_kids.into_iter();
                let first = it.next().unwrap();
                let merged_child = it.fold(first, |acc, next| merge_lower(&acc, &next));
                final_children.insert(v, OrdMap::unit(merged_child.max_depth, merged_child));
            }

            let res = new_lower(final_children, node.empty);
            memo.insert(key, res.clone());
            res
        }

        fn fuse_upper<T, A>(
            node: &Arc<Upper<T, A>>,
            remain: Option<isize>,
            memo_upper: &mut StdHashMap<(usize, Option<isize>), Arc<Upper<T, A>>>,
            memo_lower: &mut StdHashMap<(usize, Option<isize>), Arc<Lower<T>>>,
        ) -> Arc<Upper<T, A>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            if let Some(r) = remain {
                if r == 0 {
                    return node.clone();
                }
            }
            let key = (Arc::as_ptr(node) as usize, remain);
            if let Some(cached) = memo_upper.get(&key) {
                return cached.clone();
            }

            let next_remain = remain.map(|r| r - 1);

            let res = match &**node {
                Upper::Interface(i) => {
                    let has_multi_depth_slots = i.inner.children.values().any(|kids| kids.len() > 1);
                    let fused_lower = fuse_lower::<T, A>(&i.inner, next_remain, memo_lower);
                    if !has_multi_depth_slots && Arc::ptr_eq(&fused_lower, &i.inner) {
                        memo_upper.insert(key, node.clone());
                        return node.clone();
                    }
                    new_interface(fused_lower, i.acc.clone())
                }
                Upper::Branch(b) => {
                    let has_multi_depth_slots = b.children.values().any(|kids| kids.len() > 1);
                    let mut new_children_by_value: StdHashMap<T, Vec<Arc<Upper<T, A>>>> =
                        StdHashMap::new();
                    let mut children_changed = false;

                    for (v, kids) in b.children.iter() {
                        for child in kids.values() {
                            let fused_child =
                                fuse_upper(child, next_remain, memo_upper, memo_lower);
                            if !Arc::ptr_eq(&fused_child, child) {
                                children_changed = true;
                            }
                            new_children_by_value
                                .entry(v.clone())
                                .or_default()
                                .push(fused_child);
                        }
                    }

                    if !has_multi_depth_slots && !children_changed {
                        memo_upper.insert(key, node.clone());
                        return node.clone();
                    }

                    let mut final_children: Children<T, Upper<T, A>> = IHashMap::new();
                    for (v, fused_kids) in new_children_by_value {
                        if fused_kids.is_empty() {
                            continue;
                        }
                        let mut it = fused_kids.into_iter();
                        let first = it.next().unwrap();
                        let merged_child = it.fold(first, |acc, next| merge_upper(&acc, &next));
                        final_children
                            .insert(v, OrdMap::unit(merged_child.max_depth(), merged_child));
                    }
                    let new_b = new_branch(final_children, b.empty.clone());
                    try_promote(&new_b)
                }
            };

            memo_upper.insert(key, res.clone());
            res
        }

        let new_inner = fuse_upper::<T, A>(&self.inner, levels, &mut memo_upper, &mut memo_lower);
        if Arc::ptr_eq(&new_inner, &self.inner) {
            self.clone()
        } else {
            LeveledGSS { inner: new_inner }
        }
    }

    #[time_it]
    pub fn normalize(&self) -> Self {
        // 1) Fuse all levels to collapse per-value multiplicity across depths.
        //    This dramatically reduces branching in most real-world cases.
        let fused = self.fuse(None);

        // 2) Hash-cons the resulting DAG bottom-up to maximally share all equal subgraphs.
        //    We memoize by pointer to avoid repeated work and use interners keyed by
        //    order-independent structural signatures.
        let mut memo_upper: StdHashMap<usize, (usize, Arc<Upper<T, A>>)> = StdHashMap::new();
        let mut memo_lower: StdHashMap<usize, (usize, Arc<Lower<T>>)> = StdHashMap::new();
        let mut interner_upper: NormalizationUpperInterner<T, A> = Default::default();
        let mut interner_lower: NormalizationLowerInterner<T> = Default::default();

        let (_id, inner) = normalize_canonicalize_upper::<T, A>(
            &fused.inner,
            &mut memo_upper,
            &mut memo_lower,
            &mut interner_upper,
            &mut interner_lower,
        );

        // Typically the inner pointer changes due to canonicalization. We just return the
        // canonicalized GSS regardless.
        LeveledGSS { inner }
    }

    fn expand_lower_recursive(
        node: &Arc<Lower<T>>,
        acc: &A,
        memo_lower: &mut StdHashMap<(usize, A), Arc<Upper<T, A>>>,
    ) -> Arc<Upper<T, A>> {
        let key = (Arc::as_ptr(node) as usize, acc.clone());
        if let Some(cached) = memo_lower.get(&key) {
            return cached.clone();
        }

        let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
        for (v, kids) in node.children.iter() {
            let mut new_kids = OrdMap::new();
            for child in kids.values() {
                let new_child = Self::expand_lower_recursive(child, acc, memo_lower);
                new_kids.insert(new_child.max_depth(), new_child);
            }
            new_children.insert(v.clone(), new_kids);
        }

        let empty = if node.empty { Some(acc.clone()) } else { None };
        let new_node = new_branch(new_children, empty);

        memo_lower.insert(key, new_node.clone());
        new_node
    }

    pub fn peek(&self) -> HashSet<T> {
        self.inner.children_keys().into_iter().collect()
    }

    /// Visit each unique accumulator present anywhere in the structure exactly once.
    ///
    /// This traverses the DAG of `Upper` nodes, deduplicating both shared subgraphs
    /// (by pointer) and accumulators (by value). The visitor is invoked at most once
    /// for each distinct accumulator value `A` that appears as:
    /// - `Interface.acc`
    /// - `Upper::Branch.empty` (when present)
    ///
    /// The visit order is not specified.
    pub fn visit_accs<F>(&self, mut f: F)
    where
        F: FnMut(&A),
    {
        // Deduplicate by accumulator value so the visitor sees each A once.
        let mut seen: HashSet<A> = HashSet::new();
        // Deduplicate by node pointer to avoid revisiting shared subgraphs.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<Arc<Upper<T, A>>> = VecDeque::new();

        queue.push_back(self.inner.clone());
        while let Some(node) = queue.pop_front() {
            let ptr = Arc::as_ptr(&node) as usize;
            if !visited.insert(ptr) {
                continue;
            }
            match &*node {
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        if seen.insert(acc.clone()) {
                            f(acc);
                        }
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            queue.push_back(child.clone());
                        }
                    }
                }
                Upper::Interface(i) => {
                    if seen.insert(i.acc.clone()) {
                        f(&i.acc);
                    }
                }
            }
        }
    }

    pub fn reduce_acc(&self) -> Option<A> {
        // Collect unique accumulators, then merge them all
        let mut unique: HashSet<A> = HashSet::new();
        let mut queue: VecDeque<Arc<Upper<T, A>>> = VecDeque::new();
        let mut visited: HashSet<usize> = HashSet::new();

        queue.push_back(self.inner.clone());
        while let Some(node) = queue.pop_front() {
            let ptr = Arc::as_ptr(&node) as usize;
            if !visited.insert(ptr) {
                continue;
            }
            match &*node {
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        unique.insert(acc.clone());
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            queue.push_back(child.clone());
                        }
                    }
                }
                Upper::Interface(i) => {
                    unique.insert(i.acc.clone());
                }
            }
        }

        let mut it = unique.into_iter();
        let first = it.next()?;
        let reduced = it.fold(first, |acc, next| acc.merge(&next));
        Some(reduced)
    }


    pub fn stats(&self) -> LeveledGSSStats<T, A> {
        let top_values: HashSet<T> = self.inner.children_keys().into_iter().collect();

        let mut visited_upperbranch: HashSet<usize> = HashSet::new();
        let mut visited_interface: HashSet<usize> = HashSet::new();
        let mut visited_lower: HashSet<usize> = HashSet::new();

        let mut num_upperbranch_nodes = 0;
        let mut num_interface_nodes = 0;
        let mut num_lower_nodes = 0;

        let mut upper_edges = 0;
        let mut interface_to_lower_edges = 0;
        let mut lower_edges = 0;

        let mut distinct_values: HashSet<T> = HashSet::new();
        let mut unique_accumulators: HashSet<A> = HashSet::new();
        let mut total_accumulator_instances = 0;

        let mut num_upper_with_empty = 0;
        let mut num_interfaces_with_empty = 0;
        let mut num_lower_terminal_nodes = 0;
        let mut num_interface_implicit_terminals = 0;

        let mut num_multi_depth_slots_upper = 0;
        let mut num_multi_depth_slots_lower = 0;
        let mut max_multiplicity_per_value_upper = 1;
        let mut max_multiplicity_per_value_lower = 1;

        let mut max_lower_depth = 0;

        let mut incoming_edges: StdHashMap<usize, usize> = StdHashMap::new();

        let mut promotable_upper_nodes = 0;

        let mut upper_queue: VecDeque<Arc<Upper<T, A>>> = VecDeque::new();
        upper_queue.push_back(self.inner.clone());
        let mut lower_queue: VecDeque<Arc<Lower<T>>> = VecDeque::new();

        while let Some(node) = upper_queue.pop_front() {
            match &*node {
                Upper::Branch(b) => {
                    let nid = Arc::as_ptr(b) as usize;
                    if visited_upperbranch.insert(nid) {
                        num_upperbranch_nodes += 1;
                        if b.empty.is_some() {
                            num_upper_with_empty += 1;
                            unique_accumulators.insert(b.empty.as_ref().unwrap().clone());
                            total_accumulator_instances += 1;
                        }
                        for (v, kids) in b.children.iter() {
                            distinct_values.insert(v.clone());
                            if kids.len() > 1 {
                                num_multi_depth_slots_upper += 1;
                                max_multiplicity_per_value_upper =
                                    std::cmp::max(max_multiplicity_per_value_upper, kids.len());
                            }
                            for child in kids.values() {
                                upper_edges += 1;
                                *incoming_edges.entry(Arc::as_ptr(child) as usize).or_insert(0) += 1;
                                upper_queue.push_back(child.clone());
                            }
                        }
                        if is_promotable(b) {
                            promotable_upper_nodes += 1;
                        }
                    }
                }
                Upper::Interface(i) => {
                    let nid = Arc::as_ptr(i) as usize;
                    if visited_interface.insert(nid) {
                        num_interface_nodes += 1;
                        unique_accumulators.insert(i.acc.clone());
                        total_accumulator_instances += 1;
                        if i.inner.empty {
                            num_interfaces_with_empty += 1;
                        }
                        for (v, kids) in i.inner.children.iter() {
                            distinct_values.insert(v.clone());
                            if kids.len() > 1 {
                                num_multi_depth_slots_lower += 1;
                                max_multiplicity_per_value_lower =
                                    std::cmp::max(max_multiplicity_per_value_lower, kids.len());
                            }
                            for child in kids.values() {
                                interface_to_lower_edges += 1;
                                *incoming_edges.entry(Arc::as_ptr(child) as usize).or_insert(0) += 1;
                                lower_queue.push_back(child.clone());
                            }
                        }
                    }
                }
            }
        }

        while let Some(node) = lower_queue.pop_front() {
            let nid = Arc::as_ptr(&node) as usize;
            if visited_lower.insert(nid) {
                num_lower_nodes += 1;
                if node.empty {
                    num_lower_terminal_nodes += 1;
                }
                max_lower_depth = std::cmp::max(max_lower_depth, node.max_depth);
                for (v, kids) in node.children.iter() {
                    distinct_values.insert(v.clone());
                    if kids.len() > 1 {
                        num_multi_depth_slots_lower += 1;
                        max_multiplicity_per_value_lower =
                            std::cmp::max(max_multiplicity_per_value_lower, kids.len());
                    }
                    for child in kids.values() {
                        lower_edges += 1;
                        *incoming_edges.entry(Arc::as_ptr(child) as usize).or_insert(0) += 1;
                        lower_queue.push_back(child.clone());
                    }
                }
            }
        }

        // Compute the number of structurally unique nodes in the maximally compressed
        // nondeterministic DAG obtained by merging isomorphic subgraphs of the current GSS,
        // completely ignoring accumulators. This baseline is always <= total_unique_nodes.
        #[derive(Clone, PartialEq, Eq)]
        struct StatsSig<T: Clone + Eq + Hash> {
            terminal: bool,
            // label -> sorted, deduplicated list of canonical child ids (set per label)
            edges: StdHashMap<T, Vec<usize>>,
        }

        fn stats_sig_hash<T: Clone + Eq + Hash>(sig: &StatsSig<T>) -> u64 {
            use std::hash::{Hash as _, Hasher};
            let mut seed: u64 = if sig.terminal {
                0x9e37_79b9_7f4a_7c15
            } else {
                0x243f_6a88_85a3_08d3
            };
            seed ^= (sig.edges.len() as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
            let mut xor_acc: u64 = 0;
            let mut sum_acc: u64 = 0;
            let mut prod_acc: u64 = 0x517c_c1b7_2722_0a95 ^ (sig.edges.len() as u64);
            for (k, ids) in &sig.edges {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                k.hash(&mut h);
                ids.hash(&mut h); // ids sorted and deduplicated
                let e = h.finish();
                xor_acc ^= e;
                sum_acc = sum_acc.wrapping_add(e);
                prod_acc = prod_acc.wrapping_mul(e.wrapping_add(0x9e37_79b9_7f4a_7c15));
            }
            seed ^ xor_acc ^ sum_acc ^ prod_acc
        }

        struct StatsInterner<T: Clone + Eq + Hash> {
            map: StdHashMap<u64, Vec<(StatsSig<T>, usize)>>,
            next_id: usize,
        }

        impl<T: Clone + Eq + Hash> Default for StatsInterner<T> {
            fn default() -> Self {
                Self {
                    map: StdHashMap::new(),
                    next_id: 0,
                }
            }
        }

        fn intern_stats_sig<T: Clone + Eq + Hash>(
            sig: StatsSig<T>,
            interner: &mut StatsInterner<T>,
        ) -> usize {
            let h = stats_sig_hash(&sig);
            let bucket = interner.map.entry(h).or_default();
            for (existing, id) in bucket.iter() {
                if existing == &sig {
                    return *id;
                }
            }
            let id = interner.next_id;
            interner.next_id += 1;
            bucket.push((sig, id));
            id
        }

        fn canon_lower_for_stats<T, A>(
            node: &Arc<Lower<T>>,
            memo_lower: &mut StdHashMap<usize, usize>,
            interner: &mut StatsInterner<T>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            let ptr = Arc::as_ptr(node) as usize;
            if let Some(id) = memo_lower.get(&ptr) {
                return *id;
            }
            let mut edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
            for (v, kids) in node.children.iter() {
                let e = edges.entry(v.clone()).or_default();
                for child in kids.values() {
                    let cid = canon_lower_for_stats::<T, A>(child, memo_lower, interner);
                    e.push(cid);
                }
                e.sort_unstable();
                e.dedup();
            }
            let sig = StatsSig {
                terminal: node.empty,
                edges,
            };
            let id = intern_stats_sig(sig, interner);
            memo_lower.insert(ptr, id);
            id
        }

        fn canon_upper_for_stats<T, A>(
            node: &Arc<Upper<T, A>>,
            memo_upper: &mut StdHashMap<usize, usize>,
            memo_lower: &mut StdHashMap<usize, usize>,
            interner: &mut StatsInterner<T>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            let ptr = Arc::as_ptr(node) as usize;
            if let Some(id) = memo_upper.get(&ptr) {
                return *id;
            }
            let (terminal, edges) = match &**node {
                Upper::Branch(b) => {
                    let mut edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
                    for (v, kids) in b.children.iter() {
                        let e = edges.entry(v.clone()).or_default();
                        for child in kids.values() {
                            let cid = canon_upper_for_stats::<T, A>(child, memo_upper, memo_lower, interner);
                            e.push(cid);
                        }
                        e.sort_unstable();
                        e.dedup();
                    }
                    (b.empty.is_some(), edges)
                }
                Upper::Interface(i) => {
                    let mut edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
                    for (v, kids) in i.inner.children.iter() {
                        let e = edges.entry(v.clone()).or_default();
                        for child in kids.values() {
                            let cid = canon_lower_for_stats::<T, A>(child, memo_lower, interner);
                            e.push(cid);
                        }
                        e.sort_unstable();
                        e.dedup();
                    }
                    let terminal = i.inner.empty;
                    (terminal, edges)
                }
            };
            let sig = StatsSig { terminal, edges };
            let id = intern_stats_sig(sig, interner);
            memo_upper.insert(ptr, id);
            id
        }

        fn count_structural_unique_nodes_ignoring_accs<T, A>(root: &Arc<Upper<T, A>>) -> usize
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            let mut interner: StatsInterner<T> = Default::default();
            let mut memo_upper: StdHashMap<usize, usize> = StdHashMap::new();
            let mut memo_lower: StdHashMap<usize, usize> = StdHashMap::new();
            let _root_id = canon_upper_for_stats::<T, A>(root, &mut memo_upper, &mut memo_lower, &mut interner);
            interner.next_id
        }

        let num_structurally_unique_nodes =
            count_structural_unique_nodes_ignoring_accs::<T, A>(&self.inner);

        let total_unique_nodes = num_upperbranch_nodes + num_interface_nodes + num_lower_nodes;
        let total_edges = upper_edges + interface_to_lower_edges + lower_edges;
        let max_upper_depth = self.inner.max_depth();
        let distinct_values_count = distinct_values.len();
        let unique_accumulators_count = unique_accumulators.len();

        let (max_in_degree, average_in_degree) = if !incoming_edges.is_empty() {
            let max_val = *incoming_edges.values().max().unwrap_or(&0);
            let sum: usize = incoming_edges.values().sum();
            let avg = sum as f64 / incoming_edges.len() as f64;
            (max_val, avg)
        } else {
            (0, 0.0)
        };

        let structural_sharing_factor = if total_unique_nodes > 0 {
            num_structurally_unique_nodes as f64 / total_unique_nodes as f64
        } else {
            1.0
        };

        LeveledGSSStats {
            top_values,
            num_upperbranch_nodes,
            num_interface_nodes,
            num_lower_nodes,
            total_unique_nodes,
            num_structurally_unique_nodes,
            upper_edges,
            interface_to_lower_edges,
            lower_edges,
            total_edges,
            max_upper_depth,
            max_lower_depth,
            distinct_values_count,
            distinct_values,
            unique_accumulators_count,
            unique_accumulators,
            total_accumulator_instances,
            num_upper_with_empty,
            num_interfaces_with_empty,
            num_lower_terminal_nodes,
            num_interface_implicit_terminals,
            num_multi_depth_slots_upper,
            num_multi_depth_slots_lower,
            max_multiplicity_per_value_upper,
            max_multiplicity_per_value_lower,
            average_in_degree,
            max_in_degree,
            structural_sharing_factor,
            promotable_upper_nodes,
        }
    }

    pub fn to_graph_string(&self, upper_only: bool) -> String
    where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        let mut memo = HashSet::new();
        self.to_graph_string_with_memo(&mut memo, upper_only)
    }

    pub fn to_graph_string_with_memo(&self, memo: &mut HashSet<usize>, upper_only: bool) -> String
    where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        let mut output_lines = Vec::new();
        let root = &self.inner;
        let root_id = Arc::as_ptr(root) as usize;

        if memo.contains(&root_id) {
            output_lines.push(format!("--- Root -> Ref to Node @ {:#x} ---", root_id));
        } else {
            output_lines.push(format!("--- Root {} ---", Self::get_node_info_upper(root)));
            Self::format_recursive_upper(root, "", memo, &mut output_lines, upper_only);
        }

        output_lines.join("\n")
    }

    fn get_node_info_lower(node: &Arc<Lower<T>>) -> String
    where
        T: std::fmt::Debug,
    {
        let mut info = format!(
            "Lower @ {:#x} (MaxDepth: {})",
            Arc::as_ptr(node) as usize,
            node.max_depth
        );
        if node.empty {
            info.push_str(" [TERMINAL]");
        }
        info
    }

    fn get_node_info_upper(node: &Arc<Upper<T, A>>) -> String
    where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        match &**node {
            Upper::Branch(b) => {
                let mut info = format!(
                    "UpperBranch @ {:#x} (MaxDepth: {})",
                    Arc::as_ptr(b) as usize,
                    b.max_depth
                );
                if let Some(e) = &b.empty {
                    info.push_str(&format!(" [TERMINAL empty: {:?}]", e));
                }
                info
            }
            Upper::Interface(i) => {
                let mut info = format!(
                    "Interface @ {:#x} (MaxDepth: {}) | acc: {:?}",
                    Arc::as_ptr(i) as usize,
                    node.max_depth(),
                    i.acc
                );
                if i.inner.empty {
                    info.push_str(&format!(" [TERMINAL via acc: {:?}]", i.acc));
                }
                info
            }
        }
    }

    fn format_recursive_lower(
        node: &Arc<Lower<T>>,
        current_prefix: &str,
        printed_nodes: &mut HashSet<usize>,
        output_lines: &mut Vec<String>,
    ) where
        T: std::fmt::Debug,
    {
        printed_nodes.insert(Arc::as_ptr(node) as usize);

        let mut children_to_print = Vec::new();
        let mut sorted_values: Vec<_> = node.children.keys().collect();
        sorted_values.sort_by_key(|v| format!("{:?}", v));

        for v in sorted_values {
            if let Some(kids_at_depths) = node.children.get(v) {
                for (depth, child) in kids_at_depths.iter() {
                    let label = format!("Edge {:?} (d={})", v, depth);
                    children_to_print.push((label, child.clone()));
                }
            }
        }

        let num_children = children_to_print.len();
        for (i, (label, child)) in children_to_print.into_iter().enumerate() {
            let is_last = i == num_children - 1;
            let prefix_char = if is_last { "└── " } else { "├── " };
            let child_id = Arc::as_ptr(&child) as usize;

            if printed_nodes.contains(&child_id) {
                let line = format!(
                    "{}{}{} -> Ref to Node @ {:#x}",
                    current_prefix, prefix_char, label, child_id
                );
                output_lines.push(line);
            } else {
                let line = format!(
                    "{}{}{} -> {}",
                    current_prefix,
                    prefix_char,
                    label,
                    Self::get_node_info_lower(&child)
                );
                output_lines.push(line);

                let child_prefix = format!("{}{}", current_prefix, if is_last { "    " } else { "│   " });
                Self::format_recursive_lower(&child, &child_prefix, printed_nodes, output_lines);
            }
        }
    }

    fn format_recursive_upper(
        node: &Arc<Upper<T, A>>,
        current_prefix: &str,
        printed_nodes: &mut HashSet<usize>,
        output_lines: &mut Vec<String>,
        upper_only: bool,
    ) where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        printed_nodes.insert(Arc::as_ptr(node) as usize);

        match &**node {
            Upper::Branch(b) => {
                let mut children_to_print = Vec::new();
                let mut sorted_values: Vec<_> = b.children.keys().collect();
                sorted_values.sort_by_key(|v| format!("{:?}", v));

                for v in sorted_values {
                    if let Some(kids_at_depths) = b.children.get(v) {
                        for (depth, child) in kids_at_depths.iter() {
                            let label = format!("Edge {:?} (d={})", v, depth);
                            children_to_print.push((label, child.clone()));
                        }
                    }
                }

                let num_children = children_to_print.len();
                for (i, (label, child)) in children_to_print.into_iter().enumerate() {
                    let is_last = i == num_children - 1;
                    let prefix_char = if is_last { "└── " } else { "├── " };
                    let child_id = Arc::as_ptr(&child) as usize;

                    if printed_nodes.contains(&child_id) {
                        let line = format!(
                            "{}{}{} -> Ref to Node @ {:#x}",
                            current_prefix, prefix_char, label, child_id
                        );
                        output_lines.push(line);
                    } else {
                        let line = format!(
                            "{}{}{} -> {}",
                            current_prefix,
                            prefix_char,
                            label,
                            Self::get_node_info_upper(&child)
                        );
                        output_lines.push(line);

                        let child_prefix =
                            format!("{}{}", current_prefix, if is_last { "    " } else { "│   " });
                        Self::format_recursive_upper(
                            &child,
                            &child_prefix,
                            printed_nodes,
                            output_lines,
                            upper_only,
                        );
                    }
                }
            }
            Upper::Interface(i) => {
                if upper_only && !i.inner.children.is_empty() {
                    let prefix_char = "└── ";
                    let num_lower_edges: usize = i.inner.children.values().map(|kids| kids.len()).sum();
                    let line = format!("{}[{} lower edges omitted]", prefix_char, num_lower_edges);
                    output_lines.push(format!("{}{}", current_prefix, line));
                    return;
                }

                let mut children_to_print = Vec::new();
                let mut sorted_values: Vec<_> = i.inner.children.keys().collect();
                sorted_values.sort_by_key(|v| format!("{:?}", v));

                for v in sorted_values {
                    if let Some(kids_at_depths) = i.inner.children.get(v) {
                        for (depth, child) in kids_at_depths.iter() {
                            let label = format!("Edge {:?} (d={})", v, depth);
                            children_to_print.push((label, child.clone()));
                        }
                    }
                }

                let num_children = children_to_print.len();
                for (i_idx, (label, child)) in children_to_print.into_iter().enumerate() {
                    let is_last = i_idx == num_children - 1;
                    let prefix_char = if is_last { "└── " } else { "├── " };
                    let child_id = Arc::as_ptr(&child) as usize;

                    if printed_nodes.contains(&child_id) {
                        let line = format!(
                            "{}{}{} -> Ref to Node @ {:#x}",
                            current_prefix, prefix_char, label, child_id
                        );
                        output_lines.push(line);
                    } else {
                        let line = format!(
                            "{}{}{} -> {}",
                            current_prefix,
                            prefix_char,
                            label,
                            Self::get_node_info_lower(&child)
                        );
                        output_lines.push(line);

                        let child_prefix =
                            format!("{}{}", current_prefix, if is_last { "    " } else { "│   " });
                        Self::format_recursive_lower(
                            &child,
                            &child_prefix,
                            printed_nodes,
                            output_lines,
                        );
                    }
                }
            }
        }
    }

    pub fn predecessors(&self) -> BTreeMap<T, BTreeMap<isize, Vec<Self>>>
    where
        T: Clone + Eq + Hash + Ord,
        A: Merge + Clone + Eq + Hash + Ord,
    {
        let mut result = BTreeMap::new();
        match &*self.inner {
            Upper::Branch(b) => {
                for (edge_val, children_by_depth) in &b.children {
                    let mut preds_by_depth: BTreeMap<isize, Vec<Self>> = BTreeMap::new();
                    for (depth, child_upper_arc) in children_by_depth {
                        let gss = LeveledGSS {
                            inner: child_upper_arc.clone(),
                        };
                        preds_by_depth.entry(*depth).or_default().push(gss);
                    }
                    result.insert(edge_val.clone(), preds_by_depth);
                }
            }
            Upper::Interface(i) => {
                for (edge_val, children_by_depth) in &i.inner.children {
                    let mut preds_by_depth: BTreeMap<isize, Vec<Self>> = BTreeMap::new();
                    for (depth, child_lower_arc) in children_by_depth {
                        let new_interface_upper = new_interface(
                            child_lower_arc.clone(),
                            i.acc.clone(),
                        );
                        let gss = LeveledGSS {
                            inner: new_interface_upper,
                        };
                        preds_by_depth.entry(*depth).or_default().push(gss);
                    }
                    result.insert(edge_val.clone(), preds_by_depth);
                }
            }
        }
        result
    }

    pub fn num_paths(&self) -> usize {
        self.paths_info().num_paths
    }

    pub fn paths_info(&self) -> GSSPathsInfo {
        let mut memo_upper = StdHashMap::new();
        let mut memo_lower = StdHashMap::new();
        Self::paths_info_upper(&self.inner, &mut memo_upper, &mut memo_lower)
    }

    fn paths_info_lower(
        node: &Arc<Lower<T>>,
        memo: &mut StdHashMap<usize, GSSPathsInfo>,
    ) -> GSSPathsInfo {
        let ptr = Arc::as_ptr(node) as usize;
        if let Some(cached) = memo.get(&ptr) {
            return *cached;
        }

        let mut info = if node.empty {
            GSSPathsInfo {
                num_paths: 1,
                total_depth: 0,
            }
        } else {
            GSSPathsInfo::default()
        };

        for children in node.children.values() {
            for child in children.values() {
                let child_info = Self::paths_info_lower(child, memo);
                info.num_paths += child_info.num_paths;
                info.total_depth += child_info.total_depth + child_info.num_paths;
            }
        }

        memo.insert(ptr, info);
        info
    }

    fn paths_info_upper(
        node: &Arc<Upper<T, A>>,
        memo_upper: &mut StdHashMap<usize, GSSPathsInfo>,
        memo_lower: &mut StdHashMap<usize, GSSPathsInfo>,
    ) -> GSSPathsInfo {
        let ptr = Arc::as_ptr(node) as usize;
        if let Some(cached) = memo_upper.get(&ptr) {
            return *cached;
        }

        let info = match &**node {
            Upper::Branch(b) => {
                let mut info = if b.empty.is_some() {
                    GSSPathsInfo {
                        num_paths: 1,
                        total_depth: 0,
                    }
                } else {
                    GSSPathsInfo::default()
                };
                for children in b.children.values() {
                    for child in children.values() {
                        let child_info = Self::paths_info_upper(child, memo_upper, memo_lower);
                        info.num_paths += child_info.num_paths;
                        info.total_depth += child_info.total_depth + child_info.num_paths;
                    }
                }
                info
            }
            Upper::Interface(i) => {
                let mut info = if i.inner.empty {
                    GSSPathsInfo {
                        num_paths: 1,
                        total_depth: 0,
                    }
                } else {
                    GSSPathsInfo::default()
                };

                for children in i.inner.children.values() {
                    for child in children.values() {
                        let child_info = Self::paths_info_lower(child, memo_lower);
                        info.num_paths += child_info.num_paths;
                        info.total_depth += child_info.total_depth + child_info.num_paths;
                    }
                }
                info
            }
        };

        memo_upper.insert(ptr, info);
        info
    }

    pub fn get_first_path(&self) -> Option<(Vec<T>, A)> {
        let mut path = Vec::new();
        Self::get_first_path_upper(&self.inner, &mut path)
    }

    fn get_first_path_lower(
        node: &Arc<Lower<T>>,
        path: &mut Vec<T>,
        acc: &A,
    ) -> Option<(Vec<T>, A)> {
        if node.empty {
            let mut p = path.clone();
            p.reverse();
            return Some((p, acc.clone()));
        }
        for (v, children) in &node.children {
            for child in children.values() {
                path.push(v.clone());
                if let Some(res) = Self::get_first_path_lower(child, path, acc) {
                    return Some(res);
                }
                path.pop();
            }
        }
        None
    }

    fn get_first_path_upper(
        node: &Arc<Upper<T, A>>,
        path: &mut Vec<T>,
    ) -> Option<(Vec<T>, A)> {
        match &**node {
            Upper::Branch(b) => {
                if let Some(acc) = &b.empty {
                    let mut p = path.clone();
                    p.reverse();
                    return Some((p, acc.clone()));
                }
                for (v, children) in &b.children {
                    for child in children.values() {
                        path.push(v.clone());
                        if let Some(res) = Self::get_first_path_upper(child, path) {
                            return Some(res);
                        }
                        path.pop();
                    }
                }
            }
            Upper::Interface(i) => {
                if i.inner.empty {
                    let mut p = path.clone();
                    p.reverse();
                    return Some((p, i.acc.clone()));
                }
                for (v, children) in &i.inner.children {
                    for child in children.values() {
                        path.push(v.clone());
                        if let Some(res) = Self::get_first_path_lower(child, path, &i.acc) {
                            return Some(res);
                        }
                        path.pop();
                    }
                }
            }
        }
        None
    }

    pub fn get_longest_path(&self) -> Option<(Vec<T>, A)> {
        let mut longest: Option<(Vec<T>, A)> = None;
        let mut path = Vec::new();
        Self::get_longest_path_upper(&self.inner, &mut path, &mut longest);
        longest
    }

    fn get_longest_path_lower(
        node: &Arc<Lower<T>>,
        path: &mut Vec<T>,
        acc: &A,
        longest: &mut Option<(Vec<T>, A)>,
    ) {
        if node.empty {
            if longest.as_ref().map_or(true, |(p, _)| path.len() > p.len()) {
                let mut p = path.clone();
                p.reverse();
                *longest = Some((p, acc.clone()));
            }
        }
        for (v, children) in &node.children {
            for child in children.values() {
                path.push(v.clone());
                Self::get_longest_path_lower(child, path, acc, longest);
                path.pop();
            }
        }
    }

    fn get_longest_path_upper(
        node: &Arc<Upper<T, A>>,
        path: &mut Vec<T>,
        longest: &mut Option<(Vec<T>, A)>,
    ) {
        match &**node {
            Upper::Branch(b) => {
                if let Some(acc) = &b.empty {
                    if longest.as_ref().map_or(true, |(p, _)| path.len() > p.len()) {
                        let mut p = path.clone();
                        p.reverse();
                        *longest = Some((p, acc.clone()));
                    }
                }
                for (v, children) in &b.children {
                    for child in children.values() {
                        path.push(v.clone());
                        Self::get_longest_path_upper(child, path, longest);
                        path.pop();
                    }
                }
            }
            Upper::Interface(i) => {
                if i.inner.empty {
                    if longest.as_ref().map_or(true, |(p, _)| path.len() > p.len()) {
                        let mut p = path.clone();
                        p.reverse();
                        *longest = Some((p, i.acc.clone()));
                    }
                }
                for (v, children) in &i.inner.children {
                    for child in children.values() {
                        path.push(v.clone());
                        Self::get_longest_path_lower(child, path, &i.acc, longest);
                        path.pop();
                    }
                }
            }
        }
    }

    pub fn as_single_path(&self) -> Option<(Vec<T>, A)> {
        let mut path = Vec::new();
        if let Some(acc) = Self::as_single_path_upper(&self.inner, &mut path) {
            path.reverse();
            Some((path, acc))
        } else {
            None
        }
    }

    fn as_single_path_lower(node: &Arc<Lower<T>>, path: &mut Vec<T>) -> bool {
        if node.children.is_empty() {
            return node.empty;
        }
        if node.empty || node.children.len() > 1 || node.children.values().next().unwrap().len() > 1 {
            return false;
        }
        let (v, children) = node.children.iter().next().unwrap();
        let child = children.values().next().unwrap();
        path.push(v.clone());
        Self::as_single_path_lower(child, path)
    }

    fn as_single_path_upper(node: &Arc<Upper<T, A>>, path: &mut Vec<T>) -> Option<A> {
        match &**node {
            Upper::Branch(b) => {
                if b.children.is_empty() {
                    return b.empty.clone();
                }
                if b.empty.is_some()
                    || b.children.len() > 1
                    || b.children.values().next().unwrap().len() > 1
                {
                    return None;
                }
                let (v, children) = b.children.iter().next().unwrap();
                let child = children.values().next().unwrap();
                path.push(v.clone());
                Self::as_single_path_upper(child, path)
            }
            Upper::Interface(i) => {
                if i.inner.children.is_empty() {
                    return if i.inner.empty {
                        Some(i.acc.clone())
                    } else {
                        None
                    };
                }
                if i.inner.empty
                    || i.inner.children.len() > 1
                    || i.inner.children.values().next().unwrap().len() > 1
                {
                    return None;
                }
                let (v, children) = i.inner.children.iter().next().unwrap();
                let child = children.values().next().unwrap();
                path.push(v.clone());
                if Self::as_single_path_lower(child, path) {
                    Some(i.acc.clone())
                } else {
                    None
                }
            }
        }
    }
}
