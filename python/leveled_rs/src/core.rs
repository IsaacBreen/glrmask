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
use std::collections::{HashMap as StdHashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

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
    children: Children<T, Lower<T>>,
    acc: A,
    empty: Option<A>,
    max_depth: isize,
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
            Upper::Interface(i) => i.max_depth,
        }
    }

    fn children_keys(&self) -> Vec<T> {
        match self {
            Upper::Branch(b) => b.children.keys().cloned().collect(),
            Upper::Interface(i) => i.children.keys().cloned().collect(),
        }
    }
}

fn get_max_depth_lower<T: Clone + Eq + Hash>(children: &Children<T, Lower<T>>) -> isize {
    children
        .values()
        .flat_map(|kids| kids.values())
        .map(|c| c.max_depth)
        .max()
        .map_or(0, |d| d + 1)
}

fn get_max_depth_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    children: &Children<T, Upper<T, A>>,
) -> isize {
    children
        .values()
        .flat_map(|kids| kids.values())
        .map(|c| c.max_depth())
        .max()
        .map_or(0, |d| d + 1)
}

fn merge_optional_acc<A: Merge + Clone>(a: &Option<A>, b: &Option<A>) -> Option<A> {
    match (a, b) {
        (None, Some(bv)) => Some(bv.clone()),
        (Some(av), None) => Some(av.clone()),
        (Some(av), Some(bv)) => Some(av.merge(bv)),
        (None, None) => None,
    }
}

fn merge_children_lower<T: Clone + Eq + Hash>(
    c1: &Children<T, Lower<T>>,
    c2: &Children<T, Lower<T>>,
) -> Children<T, Lower<T>> {
    if c1.ptr_eq(c2) {
        return c1.clone();
    }
    let mut merged = c1.clone();
    for (k, v2_map) in c2.iter() {
        if let Some(v1_map) = merged.get(k) {
            let mut new_map = v1_map.clone();
            for (depth, child2) in v2_map.iter() {
                if let Some(child1) = new_map.get(depth) {
                    let merged_child = merge_lower(child1, child2);
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

fn merge_children_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    c1: &Children<T, Upper<T, A>>,
    c2: &Children<T, Upper<T, A>>,
) -> Children<T, Upper<T, A>> {
    if c1.ptr_eq(c2) {
        return c1.clone();
    }
    let mut merged = c1.clone();
    for (k, v2_map) in c2.iter() {
        if let Some(v1_map) = merged.get(k) {
            let mut new_map = v1_map.clone();
            for (depth, child2) in v2_map.iter() {
                if let Some(child1) = new_map.get(depth) {
                    let merged_child = merge_upper(child1, child2);
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

fn merge_lower<T: Clone + Eq + Hash>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {
    if Arc::ptr_eq(l1, l2) {
        return l1.clone();
    }
    let new_empty = l1.empty || l2.empty;
    let merged_children = merge_children_lower(&l1.children, &l2.children);
    let max_depth = get_max_depth_lower(&merged_children);
    Arc::new(Lower {
        children: merged_children,
        empty: new_empty,
        max_depth,
    })
}

fn interface_to_upperbranch<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    it: &Arc<Interface<T, A>>,
) -> Arc<UpperBranch<T, A>> {
    let mut children: Children<T, Upper<T, A>> = IHashMap::new();
    for (v, kids) in it.children.iter() {
        let mut v_map: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
        for lchild in kids.values() {
            let empty = if lchild.empty { Some(it.acc.clone()) } else { None };
            let max_depth = get_max_depth_lower(&lchild.children);
            let ci = Arc::new(Upper::Interface(Arc::new(Interface {
                children: lchild.children.clone(),
                acc: it.acc.clone(),
                empty,
                max_depth,
            })));
            v_map.insert(ci.max_depth(), ci);
        }
        if !v_map.is_empty() {
            children.insert(v.clone(), v_map);
        }
    }
    let mut new_empty = it.empty.clone();
    if it.children.is_empty() && new_empty.is_none() {
        new_empty = Some(it.acc.clone());
    }
    let max_depth = get_max_depth_upper(&children);
    Arc::new(UpperBranch {
        children,
        empty: new_empty,
        max_depth,
    })
}

fn merge_upperbranches<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    a: &Arc<UpperBranch<T, A>>,
    b: &Arc<UpperBranch<T, A>>,
) -> Arc<Upper<T, A>> {
    if Arc::ptr_eq(a, b) {
        return Arc::new(Upper::Branch(a.clone()));
    }
    let new_empty = merge_optional_acc(&a.empty, &b.empty);
    let merged_children = merge_children_upper(&a.children, &b.children);
    let max_depth = get_max_depth_upper(&merged_children);
    let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
        children: merged_children,
        empty: new_empty,
        max_depth,
    })));
    try_promote(&new_b)
}

fn merge_interfaces<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    a: &Arc<Interface<T, A>>,
    b: &Arc<Interface<T, A>>,
) -> Arc<Upper<T, A>> {
    if a.acc == b.acc || a.children.ptr_eq(&b.children) {
        let merged_children = merge_children_lower(&a.children, &b.children);
        let new_acc = a.acc.merge(&b.acc);
        let new_empty = merge_optional_acc(&a.empty, &b.empty);
        let max_depth = get_max_depth_lower(&merged_children);
        Arc::new(Upper::Interface(Arc::new(Interface {
            children: merged_children,
            acc: new_acc,
            empty: new_empty,
            max_depth,
        })))
    } else {
        let ub1 = interface_to_upperbranch(a);
        let ub2 = interface_to_upperbranch(b);
        merge_upperbranches(&ub1, &ub2)
    }
}

fn merge_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    u1: &Arc<Upper<T, A>>,
    u2: &Arc<Upper<T, A>>,
) -> Arc<Upper<T, A>> {
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

fn try_promote<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    node: &Arc<Upper<T, A>>,
) -> Arc<Upper<T, A>> {
    if let Upper::Branch(b) = &**node {
        let all_children: Vec<_> = b.children.values().flat_map(|kids| kids.values()).collect();
        if all_children.is_empty() {
            if let Some(empty) = &b.empty {
                return Arc::new(Upper::Interface(Arc::new(Interface {
                    children: IHashMap::new(),
                    acc: empty.clone(),
                    empty: Some(empty.clone()),
                    max_depth: 0,
                })));
            }
            return node.clone();
        }

        if !all_children
            .iter()
            .all(|c| matches!(&***c, Upper::Interface(_)))
        {
            return node.clone();
        }

        let mut accs: HashSet<A> = HashSet::new();
        if let Some(empty) = &b.empty {
            accs.insert(empty.clone());
        }
        for c in all_children {
            if let Upper::Interface(ic) = &**c {
                accs.insert(ic.acc.clone());
                if let Some(empty) = &ic.empty {
                    accs.insert(empty.clone());
                }
            }
        }

        if accs.len() <= 1 {
            if let Some(the_acc) = accs.into_iter().next() {
                let mut l_children: Children<T, Lower<T>> = IHashMap::new();
                for (v, kids) in b.children.iter() {
                    let mut v_map: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
                    for child in kids.values() {
                        if let Upper::Interface(ci) = &**child {
                            let empty = ci.empty.is_some();
                            let max_depth = get_max_depth_lower(&ci.children);
                            let lower = Arc::new(Lower {
                                children: ci.children.clone(),
                                empty,
                                max_depth,
                            });
                            v_map.insert(lower.max_depth, lower);
                        }
                    }
                    if !v_map.is_empty() {
                        l_children.insert(v.clone(), v_map);
                    }
                }
                let max_depth = get_max_depth_lower(&l_children);
                return Arc::new(Upper::Interface(Arc::new(Interface {
                    children: l_children,
                    acc: the_acc,
                    empty: b.empty.clone(),
                    max_depth,
                })));
            } else {
                return empty_upper().inner;
            }
        }
    }
    node.clone()
}

fn empty_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>() -> LeveledGSS<T, A> {
    let inner = Arc::new(Upper::Branch(Arc::new(UpperBranch {
        children: IHashMap::new(),
        empty: None,
        max_depth: 0,
    })));
    LeveledGSS { inner }
}

#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Upper<T, A>>,
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
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
        #[derive(Default)]
        struct Entry<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
            end: Option<A>,
            sub: StdHashMap<T, Entry<T, A>>,
        }

        let mut trie: StdHashMap<T, Entry<T, A>> = StdHashMap::new();
        let mut empty_acc: Option<A> = None;

        for (vals, acc) in canon.into_iter() {
            if vals.is_empty() {
                empty_acc = match empty_acc.take() {
                    None => Some(acc),
                    Some(prev) => Some(prev.merge(&acc)),
                };
                continue;
            }

            let mut node = &mut trie;
            let rev_vals: Vec<T> = vals.iter().cloned().rev().collect();
            for (i, v) in rev_vals.into_iter().enumerate() {
                let entry = node.entry(v.clone()).or_default();
                if i == vals.len() - 1 {
                    entry.end = Some(acc.clone());
                } else {
                    node = &mut entry.sub;
                }
            }
        }

        fn build_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            d: &StdHashMap<T, Entry<T, A>>,
        ) -> Arc<Lower<T>> {
            let mut l_children: Children<T, Lower<T>> = IHashMap::new();
            for (v, e) in d.iter() {
                let sub_lower = if e.sub.is_empty() {
                    Arc::new(Lower {
                        children: IHashMap::new(),
                        empty: false,
                        max_depth: 0,
                    })
                } else {
                    build_lower(&e.sub)
                };
                let node_for_v = Arc::new(Lower {
                    children: sub_lower.children.clone(),
                    empty: e.end.is_some(),
                    max_depth: get_max_depth_lower(&sub_lower.children),
                });
                l_children.insert(v.clone(), OrdMap::unit(node_for_v.max_depth, node_for_v));
            }
            let max_depth = get_max_depth_lower(&l_children);
            Arc::new(Lower {
                children: l_children,
                empty: false,
                max_depth,
            })
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
                    let leaf = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: IHashMap::new(),
                        empty: Some(end_acc.clone()),
                        max_depth: 0,
                    })));
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

            let all_interfaces = all_child_nodes
                .iter()
                .all(|c| matches!(&**c, Upper::Interface(_)));

            if all_interfaces {
                let mut accs: HashSet<A> = HashSet::new();
                for node in &all_child_nodes {
                    if let Upper::Interface(i) = &**node {
                        accs.insert(i.acc.clone());
                        if let Some(e) = &i.empty {
                            accs.insert(e.clone());
                        }
                    }
                }
                if let Some(e) = &root_empty {
                    accs.insert(e.clone());
                }

                if accs.len() <= 1 {
                    if let Some(the_acc) = accs.into_iter().next() {
                        let lower_tree = build_lower(d);
                        let max_depth = get_max_depth_lower(&lower_tree.children);
                        return Arc::new(Upper::Interface(Arc::new(Interface {
                            children: lower_tree.children.clone(),
                            acc: the_acc,
                            empty: root_empty,
                            max_depth,
                        })));
                    } else {
                        return empty_upper().inner;
                    }
                }
            }

            let max_depth = get_max_depth_upper(&children);
            Arc::new(Upper::Branch(Arc::new(UpperBranch {
                children,
                empty: root_empty,
                max_depth,
            })))
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
                    if let Some(e) = &i.empty {
                        let mut stack = pref.clone();
                        stack.reverse();
                        out.push((stack, e.clone()));
                    }
                    if i.children.is_empty() && i.empty.is_none() {
                        let mut stack = pref.clone();
                        stack.reverse();
                        out.push((stack, i.acc.clone()));
                    } else {
                        for (v, kids) in i.children.iter() {
                            for child in kids.values() {
                                pref.push(v.clone());
                                dfs_lower(child, pref, &i.acc, out);
                                pref.pop();
                            }
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
                let lower_node = Arc::new(Lower {
                    children: i.children.clone(),
                    empty: i.empty.is_some(),
                    max_depth: get_max_depth_lower(&i.children),
                });
                let mut new_children: Children<T, Lower<T>> = IHashMap::new();
                new_children.insert(value, OrdMap::unit(lower_node.max_depth, lower_node));
                let max_depth = get_max_depth_lower(&new_children);
                Arc::new(Upper::Interface(Arc::new(Interface {
                    children: new_children,
                    acc: i.acc.clone(),
                    empty: None,
                    max_depth,
                })))
            }
            Upper::Branch(_) => {
                let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
                new_children.insert(
                    value,
                    OrdMap::unit(self.inner.max_depth(), self.inner.clone()),
                );
                let max_depth = get_max_depth_upper(&new_children);
                Arc::new(Upper::Branch(Arc::new(UpperBranch {
                    children: new_children,
                    empty: None,
                    max_depth,
                })))
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
                let res = Arc::new(Lower {
                    children: IHashMap::new(),
                    empty: false,
                    max_depth: 0,
                });
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
                        return empty_upper().inner;
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
                        .children
                        .values()
                        .flat_map(|kids| kids.values())
                        .cloned()
                        .collect();
                    if all_children.is_empty() {
                        return empty_upper().inner;
                    }
                    let popped_children: Vec<_> = all_children
                        .into_iter()
                        .map(|child| popn_lower::<T, A>(&child, k - 1, memo_lower))
                        .collect();
                    let mut it = popped_children.into_iter();
                    let first = it.next().unwrap();
                    let merged = it.fold(first, |acc, next| merge_lower(&acc, &next));

                    let new_empty = if merged.empty { Some(i.acc.clone()) } else { None };
                    if merged.children.is_empty() && new_empty.is_none() {
                        empty_upper().inner
                    } else {
                        let max_depth = get_max_depth_lower(&merged.children);
                        Arc::new(Upper::Interface(Arc::new(Interface {
                            children: merged.children.clone(),
                            acc: i.acc.clone(),
                            empty: new_empty,
                            max_depth,
                        })))
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

    pub fn isolate(&self, value: Option<T>) -> Self {
        let new_inner = if let Some(val) = value {
            match &*self.inner {
                Upper::Branch(b) => {
                    let filtered_children = b
                        .children
                        .get(&val)
                        .map(|kids| IHashMap::unit(val.clone(), kids.clone()))
                        .unwrap_or_else(IHashMap::new);
                    let max_depth = get_max_depth_upper(&filtered_children);
                    let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: filtered_children,
                        empty: None,
                        max_depth,
                    })));
                    try_promote(&new_b)
                }
                Upper::Interface(i) => {
                    if let Some(kids) = i.children.get(&val) {
                        let filtered_children = IHashMap::unit(val.clone(), kids.clone());
                        let max_depth = get_max_depth_lower(&filtered_children);
                        Arc::new(Upper::Interface(Arc::new(Interface {
                            children: filtered_children,
                            acc: i.acc.clone(),
                            empty: None,
                            max_depth,
                        })))
                    } else {
                        empty_upper().inner
                    }
                }
            }
        } else {
            let empty_acc = match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => i.empty.clone(),
            };
            let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                children: IHashMap::new(),
                empty: empty_acc,
                max_depth: 0,
            })));
            try_promote(&new_b)
        };
        LeveledGSS { inner: new_inner }
    }

    pub fn isolate_many<I: IntoIterator<Item = Option<T>>>(&self, values: I) -> Self {
        let values_set: HashSet<Option<T>> = values.into_iter().collect();

        let new_empty: Option<A> = if values_set.contains(&None) {
            match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => i.empty.clone(),
            }
        } else {
            None
        };

        let new_inner = match &*self.inner {
            Upper::Branch(b) => {
                let mut filtered_children: Children<T, Upper<T, A>> = IHashMap::new();
                for (v, kids) in b.children.iter() {
                    if values_set.contains(&Some(v.clone())) {
                        filtered_children.insert(v.clone(), kids.clone());
                    }
                }
                let max_depth = get_max_depth_upper(&filtered_children);
                let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                    children: filtered_children,
                    empty: new_empty,
                    max_depth,
                })));
                try_promote(&new_b)
            }
            Upper::Interface(i) => {
                let mut filtered_children: Children<T, Lower<T>> = IHashMap::new();
                for (v, kids) in i.children.iter() {
                    if values_set.contains(&Some(v.clone())) {
                        filtered_children.insert(v.clone(), kids.clone());
                    }
                }
                if !filtered_children.is_empty() {
                    let max_depth = get_max_depth_lower(&filtered_children);
                    Arc::new(Upper::Interface(Arc::new(Interface {
                        children: filtered_children,
                        acc: i.acc.clone(),
                        empty: new_empty,
                        max_depth,
                    })))
                } else {
                    Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: IHashMap::new(),
                        empty: new_empty,
                        max_depth: 0,
                    })))
                }
            }
        };

        LeveledGSS { inner: new_inner }
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
                    let new_empty = i.empty.as_ref().map(|e| map_acc(e, memo_acc, f));
                    let res = Arc::new(Upper::Interface(Arc::new(Interface {
                        children: i.children.clone(),
                        acc: new_acc,
                        empty: new_empty,
                        max_depth: i.max_depth,
                    })));
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
                    let max_depth = get_max_depth_upper(&new_children);
                    let res = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: new_children,
                        empty: new_empty,
                        max_depth,
                    })));
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
                    let keep_empty = i.empty.as_ref().map_or(false, |e| test_acc(e, acc_memo, p));
                    if !keep_acc && !keep_empty {
                        None
                    } else if !keep_acc && keep_empty {
                        let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: IHashMap::new(),
                            empty: i.empty.clone(),
                            max_depth: 0,
                        })));
                        Some(try_promote(&new_b))
                    } else {
                        let new_empty = if keep_empty { i.empty.clone() } else { None };
                        let new_i = Arc::new(Upper::Interface(Arc::new(Interface {
                            children: i.children.clone(),
                            acc: i.acc.clone(),
                            empty: new_empty,
                            max_depth: i.max_depth,
                        })));
                        Some(try_promote(&new_i))
                    }
                }
                Upper::Branch(b) => {
                    let new_empty = b
                        .empty
                        .as_ref()
                        .and_then(|e| if test_acc(e, acc_memo, p) { Some(e.clone()) } else { None });

                    let mut new_children: Children<T, Upper<T, A>> = IHashMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
                        for child in kids.values() {
                            if let Some(nc) = transform::<T, A, P>(child, acc_memo, p) {
                                new_kids.insert(nc.max_depth(), nc);
                            }
                        }
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }

                    if new_children.is_empty() && new_empty.is_none() {
                        None
                    } else {
                        let max_depth = get_max_depth_upper(&new_children);
                        let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: new_children,
                            empty: new_empty,
                            max_depth,
                        })));
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
                    let new_empty_opt = i.empty.as_ref().and_then(|e| mutate_acc(e, memo, m));
                    let keep_acc = new_acc_opt.is_some();
                    let keep_empty = new_empty_opt.is_some();
                    if !keep_acc && !keep_empty {
                        None
                    } else if !keep_acc && keep_empty {
                        let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: IHashMap::new(),
                            empty: new_empty_opt,
                            max_depth: 0,
                        })));
                        Some(try_promote(&new_b))
                    } else {
                        let new_i = Arc::new(Upper::Interface(Arc::new(Interface {
                            children: i.children.clone(),
                            acc: new_acc_opt.unwrap(),
                            empty: new_empty_opt,
                            max_depth: i.max_depth,
                        })));
                        Some(try_promote(&new_i))
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
                        let max_depth = get_max_depth_upper(&new_children);
                        let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: new_children,
                            empty: new_empty_opt,
                            max_depth,
                        })));
                        Some(try_promote(&new_b))
                    }
                }
            }
        }

        let res_inner_opt = transform::<T, A, B, M>(&self.inner, &mut acc_memo, &mut mutator);
        res_inner_opt.map_or_else(Self::empty, |inner| LeveledGSS { inner })
    }

    pub fn merge(&self, other: &Self) -> Self {
        let merged_inner = merge_upper(&self.inner, &other.inner);
        LeveledGSS {
            inner: merged_inner,
        }
    }

    pub fn peek(&self) -> HashSet<T> {
        self.inner.children_keys().into_iter().collect()
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
                    if let Some(acc) = &i.empty {
                        unique.insert(acc.clone());
                    }
                }
            }
        }

        let mut it = unique.into_iter();
        let first = it.next()?;
        let reduced = it.fold(first, |acc, next| acc.merge(&next));
        Some(reduced)
    }
}
