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
use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::Arc;

/// Trait for accumulator types that can be merged.
pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;
}

type Children<T, N> = IHashMap<T, OrdMap<isize, Arc<N>>>;

#[derive(Clone)]
struct Lower<T: Clone + Eq + Hash + Ord> {
    children: Children<T, Lower<T>>,
    empty: bool,
    max_depth: isize,
}

impl<T: Clone + Eq + Hash + Ord> PartialEq for Lower<T> {
    fn eq(&self, other: &Self) -> bool {
        self.empty == other.empty && self.children == other.children
    }
}
impl<T: Clone + Eq + Hash + Ord> Eq for Lower<T> {}

impl<T: Clone + Eq + Hash + Ord> PartialOrd for Lower<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Clone + Eq + Hash + Ord> Ord for Lower<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.empty.cmp(&other.empty)
            .then_with(|| self.children.cmp(&other.children))
    }
}

impl<T: Clone + Eq + Hash + Ord> Hash for Lower<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.empty.hash(state);
        self.children.hash(state);
    }
}

#[derive(Clone)]
struct Interface<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> {
    children: Children<T, Lower<T>>,
    acc: A,
    empty: Option<A>,
    max_depth: isize,
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialEq for Interface<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.acc == other.acc && self.empty == other.empty && self.children == other.children
    }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Eq for Interface<T, A> {}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialOrd for Interface<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Ord for Interface<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.acc.cmp(&other.acc)
            .then_with(|| self.empty.cmp(&other.empty))
            .then_with(|| self.children.cmp(&other.children))
    }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Hash for Interface<T, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.acc.hash(state);
        self.empty.hash(state);
        self.children.hash(state);
    }
}

#[derive(Clone)]
struct UpperBranch<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> {
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
    max_depth: isize,
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialEq for UpperBranch<T, A> {
    fn eq(&self, other: &Self) -> bool { self.empty == other.empty && self.children == other.children }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Eq for UpperBranch<T, A> {}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialOrd for UpperBranch<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Ord for UpperBranch<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.empty.cmp(&other.empty).then_with(|| self.children.cmp(&other.children))
    }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Hash for UpperBranch<T, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.empty.hash(state); self.children.hash(state); }
}

#[derive(Clone)]
enum Upper<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> {
    Branch(Arc<UpperBranch<T, A>>),
    Interface(Arc<Interface<T, A>>),
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Upper<T, A> {
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

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> LeveledGSS<T, A> {
    pub fn for_each_stack<F>(&self, mut f: F)
    where
        F: FnMut(Vec<T>, A),
    {
        fn dfs_lower<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord, F: FnMut(Vec<T>, A)>(
            l: &Lower<T>,
            pref: &mut Vec<T>,
            acc: &A,
            f: &mut F,
        ) {
            if l.empty {
                let mut stack = pref.clone();
                stack.reverse();
                f(stack, acc.clone());
            }
            for (v, kids) in l.children.iter() {
                for child in kids.values() {
                    pref.push(v.clone());
                    dfs_lower(child, pref, acc, f);
                    pref.pop();
                }
            }
        }

        fn dfs_upper<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord, F: FnMut(Vec<T>, A)>(
            u: &Upper<T, A>,
            pref: &mut Vec<T>,
            f: &mut F,
        ) {
            match u {
                Upper::Branch(b) => {
                    if let Some(e) = &b.empty {
                        let mut stack = pref.clone();
                        stack.reverse();
                        f(stack, e.clone());
                    }
                    for (v, kids) in b.children.iter() {
                        for child in kids.values() {
                            pref.push(v.clone());
                            dfs_upper(child, pref, f);
                            pref.pop();
                        }
                    }
                }
                Upper::Interface(i) => {
                    if let Some(e) = &i.empty {
                        let mut stack = pref.clone();
                        stack.reverse();
                        f(stack, e.clone());
                    }
                    if i.children.is_empty() && i.empty.is_none() {
                        let mut stack = pref.clone();
                        stack.reverse();
                        f(stack, i.acc.clone());
                    } else {
                        for (v, kids) in i.children.iter() {
                            for child in kids.values() {
                                pref.push(v.clone());
                                dfs_lower(child, pref, &i.acc, f);
                                pref.pop();
                            }
                        }
                    }
                }
            }
        }

        dfs_upper(&self.inner, &mut vec![], &mut f);
    }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialEq for Upper<T, A> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Upper::Branch(a), Upper::Branch(b)) => a == b,
            (Upper::Interface(a), Upper::Interface(b)) => a == b,
            _ => false,
        }
    }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Eq for Upper<T, A> {}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialOrd for Upper<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Ord for Upper<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Upper::Branch(a), Upper::Branch(b)) => a.cmp(b),
            (Upper::Interface(a), Upper::Interface(b)) => a.cmp(b),
            (Upper::Branch(_), Upper::Interface(_)) => Ordering::Less,
            (Upper::Interface(_), Upper::Branch(_)) => Ordering::Greater,
        }
    }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Hash for Upper<T, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Upper::Branch(b) => b.hash(state),
            Upper::Interface(i) => i.hash(state),
        }
    }
}// --------------------
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
    T: Clone + Eq + Hash + Ord,
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
    T: Clone + Eq + Hash + Ord,
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

fn new_lower<T: Clone + Eq + Hash + Ord>(children: Children<T, Lower<T>>, empty: bool) -> Arc<Lower<T>> {
    let max_depth = max_depth_from_children(&children, |n: &Arc<Lower<T>>| n.max_depth);
    Arc::new(Lower {
        children,
        empty,
        max_depth,
    })
}

fn new_interface<T, A>(
    children: Children<T, Lower<T>>,
    acc: A,
    empty: Option<A>,
) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
{
    let max_depth = max_depth_from_children(&children, |n: &Arc<Lower<T>>| n.max_depth);
    Arc::new(Upper::Interface(Arc::new(Interface {
        children,
        acc,
        empty,
        max_depth,
    })))
}

fn new_branch<T, A>(
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
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
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
{
    new_branch(IHashMap::new(), None)
}

// --------------------
// Filtering
// --------------------

fn filter_lower<T: Clone + Eq + Hash + Ord>(
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
    if current_depth < max_d {
        for (v, kids) in node.children.iter() {
            let mut new_kids: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
            for child in kids.values() {
                if let Some(new_child) =
                    filter_lower(child, current_depth + 1, min_len, max_len)
                {
                    new_kids.insert(new_child.max_depth, new_child);
                }
            }
            if !new_kids.is_empty() {
                new_children.insert(v.clone(), new_kids);
            }
        }
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
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
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
            if current_depth < max_d {
                for (v, kids) in b.children.iter() {
                    let mut new_kids: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
                    for child in kids.values() {
                        if let Some(new_child) =
                            filter_upper(child, current_depth + 1, min_len, max_len)
                        {
                            new_kids.insert(new_child.max_depth(), new_child);
                        }
                    }
                    if !new_kids.is_empty() {
                        new_children.insert(v.clone(), new_kids);
                    }
                }
            }

            if new_children.is_empty() && new_empty.is_none() {
                None
            } else {
                let new_b = new_branch(new_children, new_empty);
                Some(try_promote(&new_b))
            }
        }
        Upper::Interface(i) => {
            let keep_empty = i.empty.is_some() && current_depth >= min_d;
            let new_empty = if keep_empty { i.empty.clone() } else { None };

            let mut new_l_children: Children<T, Lower<T>> = IHashMap::new();
            if current_depth < max_d {
                for (v, kids) in i.children.iter() {
                    let mut new_kids: OrdMap<isize, Arc<Lower<T>>> = OrdMap::new();
                    for child in kids.values() {
                        if let Some(new_child) =
                            filter_lower(child, current_depth + 1, min_len, max_len)
                        {
                            new_kids.insert(new_child.max_depth, new_child);
                        }
                    }
                    if !new_kids.is_empty() {
                        new_l_children.insert(v.clone(), new_kids);
                    }
                }
            }

            let is_leaf_interface = i.children.is_empty() && i.empty.is_none();
            let keep_leaf_interface = is_leaf_interface && current_depth >= min_d;

            if new_l_children.is_empty() && new_empty.is_none() && !keep_leaf_interface {
                None
            } else {
                let new_i = new_interface(new_l_children, i.acc.clone(), new_empty);
                Some(try_promote(&new_i))
            }
        }
    }
}

// --------------------
// Conversions and merges
// --------------------

fn merge_lower<T: Clone + Eq + Hash + Ord>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {
    if Arc::ptr_eq(l1, l2) {
        return l1.clone();
    }
    let new_empty = l1.empty || l2.empty;
    let merged_children = merge_children(&l1.children, &l2.children, |a, b| merge_lower(a, b));
    new_lower(merged_children, new_empty)
}

fn interface_to_upperbranch<T, A>(it: &Arc<Interface<T, A>>) -> Arc<UpperBranch<T, A>>
where
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
{
    let mut children: Children<T, Upper<T, A>> = IHashMap::new();
    for (v, kids) in it.children.iter() {
        let mut v_map: OrdMap<isize, Arc<Upper<T, A>>> = OrdMap::new();
        for lchild in kids.values() {
            let empty = if lchild.empty {
                Some(it.acc.clone())
            } else {
                None
            };
            let ci = new_interface(lchild.children.clone(), it.acc.clone(), empty);
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
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
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
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
{
    if a.acc == b.acc || a.children.ptr_eq(&b.children) {
        let merged_children =
            merge_children(&a.children, &b.children, |x, y| merge_lower(x, y));
        let new_acc = a.acc.merge(&b.acc);
        let new_empty = merge_optional_acc(&a.empty, &b.empty);
        new_interface(merged_children, new_acc, new_empty)
    } else {
        let ub1 = interface_to_upperbranch(a);
        let ub2 = interface_to_upperbranch(b);
        merge_upperbranches(&ub1, &ub2)
    }
}

fn merge_upper<T, A>(u1: &Arc<Upper<T, A>>, u2: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
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
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
{
    if let Upper::Branch(b) = &**node {
        let all_children: Vec<_> = b
            .children
            .values()
            .flat_map(|kids| kids.values())
            .collect();

        // Leaf-branch with explicit empty: represent as Interface with no children
        if all_children.is_empty() {
            if let Some(empty) = &b.empty {
                return new_interface(IHashMap::new(), empty.clone(), Some(empty.clone()));
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

        // Collect all accumulators present across children and root empty
        let mut accs: HashSet<A> = HashSet::new();
        if let Some(empty) = &b.empty {
            accs.insert(empty.clone());
        }
        for c in all_children {
            if let Upper::Interface(ic) = &**c {
                accs.insert(ic.acc.clone());
                if let Some(e) = &ic.empty {
                    accs.insert(e.clone());
                }
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
                            let lower = new_lower(ci.children.clone(), ci.empty.is_some());
                            v_map.insert(lower.max_depth, lower);
                        }
                    }
                    if !v_map.is_empty() {
                        l_children.insert(v.clone(), v_map);
                    }
                }
                return new_interface(l_children, the_acc, b.empty.clone());
            } else {
                return empty_upper_inner();
            }
        }
    }
    node.clone()
}

fn empty_upper<T, A>() -> LeveledGSS<T, A>
where
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash + Ord,
{
    LeveledGSS {
        inner: empty_upper_inner(),
    }
}

// --------------------
// Public GSS type
// --------------------

#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> {
    inner: Arc<Upper<T, A>>,
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialEq for LeveledGSS<T, A> {
    fn eq(&self, other: &Self) -> bool { self.inner == other.inner }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Eq for LeveledGSS<T, A> {}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> PartialOrd for LeveledGSS<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Ord for LeveledGSS<T, A> {
    fn cmp(&self, other: &Self) -> Ordering { self.inner.cmp(&other.inner) }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Hash for LeveledGSS<T, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.inner.hash(state); }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> LeveledGSS<T, A> {
    pub fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn empty() -> Self {
        empty_upper()
    }

    pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {
        // Canonicalize: merge accumulators for identical stacks
        let mut canon: StdHashMap<Vec<T>, A> = StdHashMap::new(); // T must be Eq+Hash
        for (vals, acc) in stacks {
            if let Some(existing) = canon.get_mut(vals) {
                let merged = existing.merge(acc);
                *existing = merged;
            } else {
                canon.insert(vals.clone(), acc.clone());
            }
        }

        // Build a trie: map value -> { end: Option<A>, sub: Trie }
        struct Entry<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> {
            end: Option<A>,
            sub: StdHashMap<T, Entry<T, A>>,
        }

        impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord> Default for Entry<T, A> {
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

        fn build_lower<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
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

        fn build_upper<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
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
                        return new_interface(lower_tree.children.clone(), the_acc, root_empty);
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

        fn dfs_lower<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
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

        fn dfs_upper<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
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
                let lower_node = new_lower(i.children.clone(), i.empty.is_some());
                let mut new_children: Children<T, Lower<T>> = IHashMap::new();
                new_children.insert(value, OrdMap::unit(lower_node.max_depth, lower_node));
                new_interface(new_children, i.acc.clone(), None)
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

        fn popn_lower<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
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

        fn popn_upper<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
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
                        .map(|child| popn_lower::<T, A>(&child, k - 1, memo_lower))
                        .collect();
                    let mut it = popped_children.into_iter();
                    let first = it.next().unwrap();
                    let merged = it.fold(first, |acc, next| merge_lower(&acc, &next));

                    let new_empty = if merged.empty {
                        Some(i.acc.clone())
                    } else {
                        None
                    };
                    if merged.children.is_empty() && new_empty.is_none() {
                        empty_upper_inner()
                    } else {
                        new_interface(merged.children.clone(), i.acc.clone(), new_empty)
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
                    if let Some(kids) = i.children.get(&val) {
                        let filtered_children = IHashMap::unit(val.clone(), kids.clone());
                        new_interface(filtered_children, i.acc.clone(), None)
                    } else {
                        empty_upper_inner()
                    }
                }
            }
        } else {
            let empty_acc = match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => i.empty.clone(),
            };
            let new_b = new_branch(IHashMap::new(), empty_acc);
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
                let new_b = new_branch(filtered_children, new_empty);
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
                    new_interface(filtered_children, i.acc.clone(), new_empty)
                } else {
                    new_branch(IHashMap::new(), new_empty)
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
        B: Merge + Clone + Eq + Hash + Ord,
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
            T: Clone + Eq + Hash + Ord,
            A: Merge + Clone + Eq + Hash + Ord,
            B: Merge + Clone + Eq + Hash + Ord,
            F: FnMut(&A) -> B,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc = map_acc(&i.acc, memo_acc, f);
                    let new_empty = i.empty.as_ref().map(|e| map_acc(e, memo_acc, f));
                    let res = new_interface(i.children.clone(), new_acc, new_empty);
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
            T: Clone + Eq + Hash + Ord,
            A: Merge + Clone + Eq + Hash + Ord,
            P: FnMut(&A) -> bool,
        {
            match &**node {
                Upper::Interface(i) => {
                    let keep_acc = test_acc(&i.acc, acc_memo, p);
                    let keep_empty = i.empty.as_ref().map_or(false, |e| test_acc(e, acc_memo, p));
                    if !keep_acc && !keep_empty {
                        None
                    } else if !keep_acc && keep_empty {
                        let new_b = new_branch(IHashMap::new(), i.empty.clone());
                        Some(try_promote(&new_b))
                    } else {
                        let new_empty = if keep_empty { i.empty.clone() } else { None };
                        let new_i = new_interface(i.children.clone(), i.acc.clone(), new_empty);
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
        B: Merge + Clone + Eq + Hash + Ord,
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
            T: Clone + Eq + Hash + Ord,
            A: Merge + Clone + Eq + Hash + Ord,
            B: Merge + Clone + Eq + Hash + Ord,
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
                        let new_b = new_branch(IHashMap::new(), new_empty_opt);
                        Some(try_promote(&new_b))
                    } else {
                        let new_i = new_interface(i.children.clone(), new_acc_opt.unwrap(), new_empty_opt);
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
            T: Clone + Eq + Hash + std::cmp::Ord,
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
            T: Clone + Eq + Hash + std::cmp::Ord,
            A: Merge + Clone + Eq + Hash + std::cmp::Ord,
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
                    let has_multi_depth_slots = i.children.values().any(|kids| kids.len() > 1);
                    let mut new_children_by_value: StdHashMap<T, Vec<Arc<Lower<T>>>> =
                        StdHashMap::new();
                    let mut children_changed = false;

                    for (v, kids) in i.children.iter() {
                        for child in kids.values() {
                            let fused_child = fuse_lower::<T, A>(child, next_remain, memo_lower);
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
                    new_interface(final_children, i.acc.clone(), i.empty.clone())
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
                for (edge_val, children_by_depth) in &i.children {
                    let mut preds_by_depth: BTreeMap<isize, Vec<Self>> = BTreeMap::new();
                    for (depth, child_lower_arc) in children_by_depth {
                        let empty = if child_lower_arc.empty { Some(i.acc.clone()) } else { None };
                        let new_interface_upper = new_interface(
                            child_lower_arc.children.clone(),
                            i.acc.clone(),
                            empty,
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

    pub fn get_accs_by_root_edge(&self) -> BTreeMap<T, BTreeSet<A>> {
        let mut out = BTreeMap::new();

        fn collect_accs<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash + Ord>(
            node: &Arc<Upper<T, A>>,
            accs: &mut BTreeSet<A>,
            visited: &mut HashSet<usize>,
        ) {
            let ptr = Arc::as_ptr(node) as usize;
            if !visited.insert(ptr) {
                return;
            }

            match &**node {
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        accs.insert(acc.clone());
                    }
                    for children in b.children.values() {
                        for child in children.values() {
                            collect_accs(child, accs, visited);
                        }
                    }
                }
                Upper::Interface(i) => {
                    if let Some(acc) = &i.empty {
                        accs.insert(acc.clone());
                    }

                    let is_leaf_interface = i.children.is_empty() && i.empty.is_none();
                    if is_leaf_interface {
                        accs.insert(i.acc.clone());
                    } else if i
                        .children
                        .values()
                        .any(|kids| kids.values().any(|l| l.empty || !l.children.is_empty()))
                    {
                        accs.insert(i.acc.clone());
                    }
                }
            }
        }

        match &*self.inner {
            Upper::Branch(b) => {
                for (edge, children) in &b.children {
                    let accs = out.entry(edge.clone()).or_default();
                    for child in children.values() {
                        collect_accs(child, accs, &mut HashSet::new());
                    }
                }
            }
            Upper::Interface(i) => {
                for (edge, children) in &i.children {
                    if children
                        .values()
                        .any(|child_map| child_map.values().any(|l| l.empty || !l.children.is_empty()))
                    {
                        out.entry(edge.clone()).or_default().insert(i.acc.clone());
                    }
                }
            }
        }

        out
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
                let mut info = if i.empty.is_some() {
                    GSSPathsInfo {
                        num_paths: 1,
                        total_depth: 0,
                    }
                } else {
                    GSSPathsInfo::default()
                };

                if i.children.is_empty() && i.empty.is_none() {
                    info.num_paths += 1;
                } else {
                    for children in i.children.values() {
                        for child in children.values() {
                            let child_info = Self::paths_info_lower(child, memo_lower);
                            info.num_paths += child_info.num_paths;
                            info.total_depth += child_info.total_depth + child_info.num_paths;
                        }
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
                if let Some(acc) = &i.empty {
                    let mut p = path.clone();
                    p.reverse();
                    return Some((p, acc.clone()));
                }
                if i.children.is_empty() && i.empty.is_none() {
                    let mut p = path.clone();
                    p.reverse();
                    return Some((p, i.acc.clone()));
                }
                for (v, children) in &i.children {
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
                if let Some(acc) = &i.empty {
                    if longest.as_ref().map_or(true, |(p, _)| path.len() > p.len()) {
                        let mut p = path.clone();
                        p.reverse();
                        *longest = Some((p, acc.clone()));
                    }
                }
                if i.children.is_empty() && i.empty.is_none() {
                    if longest.as_ref().map_or(true, |(p, _)| path.len() > p.len()) {
                        let mut p = path.clone();
                        p.reverse();
                        *longest = Some((p, i.acc.clone()));
                    }
                } else {
                    for (v, children) in &i.children {
                        for child in children.values() {
                            path.push(v.clone());
                            Self::get_longest_path_lower(child, path, &i.acc, longest);
                            path.pop();
                        }
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
                if i.children.is_empty() {
                    return if i.empty.is_some() {
                        i.empty.clone()
                    } else {
                        Some(i.acc.clone())
                    };
                }
                if i.empty.is_some()
                    || i.children.len() > 1
                    || i.children.values().next().unwrap().len() > 1
                {
                    return None;
                }
                let (v, children) = i.children.iter().next().unwrap();
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
