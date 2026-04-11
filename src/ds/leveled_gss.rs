use im::{HashMap as IHashMap, OrdMap};
use smallvec::{SmallVec, smallvec};
use arrayvec::ArrayVec;
use std::collections::{HashMap as StdHashMap, HashSet, VecDeque};
#[cfg(test)]
use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::{Arc, OnceLock};

pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;
}

/// A map optimized for small sizes (≤4 entries). Uses inline SmallVec storage
/// for small maps and falls back to im::HashMap for larger ones.
/// Drop-in replacement for im::HashMap in GSS children maps.
#[derive(Clone, PartialEq, Eq)]
enum CompactMap<K: Clone + Eq + Hash, V: Clone> {
    Inline(SmallVec<[(K, V); 4]>),
    Large(IHashMap<K, V>),
}

impl<K: Clone + Eq + Hash, V: Clone> CompactMap<K, V> {
    #[inline(always)]
    fn new() -> Self {
        CompactMap::Inline(SmallVec::new())
    }

    #[inline(always)]
    fn unit(key: K, value: V) -> Self {
        let mut sv = SmallVec::new();
        sv.push((key, value));
        CompactMap::Inline(sv)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            CompactMap::Inline(sv) => sv.len(),
            CompactMap::Large(m) => m.len(),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self {
            CompactMap::Inline(sv) => sv.is_empty(),
            CompactMap::Large(m) => m.is_empty(),
        }
    }

    #[inline(always)]
    fn get(&self, key: &K) -> Option<&V> {
        match self {
            CompactMap::Inline(sv) => sv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            CompactMap::Large(m) => m.get(key),
        }
    }

    #[inline(always)]
    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self {
            CompactMap::Inline(sv) => sv.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            CompactMap::Large(m) => m.get_mut(key),
        }
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self {
            CompactMap::Inline(sv) => {
                for entry in sv.iter_mut() {
                    if entry.0 == key {
                        let old = std::mem::replace(&mut entry.1, value);
                        return Some(old);
                    }
                }
                if sv.len() < 4 {
                    sv.push((key, value));
                    None
                } else {
                    // Promote to Large
                    let mut m = IHashMap::new();
                    for (k, v) in sv.drain(..) {
                        m.insert(k, v);
                    }
                    let result = m.insert(key, value);
                    *self = CompactMap::Large(m);
                    result
                }
            }
            CompactMap::Large(m) => m.insert(key, value),
        }
    }

    #[inline(always)]
    fn contains_key(&self, key: &K) -> bool {
        match self {
            CompactMap::Inline(sv) => sv.iter().any(|(k, _)| k == key),
            CompactMap::Large(m) => m.contains_key(key),
        }
    }

    fn keys(&self) -> CompactMapKeys<'_, K, V> {
        match self {
            CompactMap::Inline(sv) => CompactMapKeys::Inline(sv.iter()),
            CompactMap::Large(m) => CompactMapKeys::Large(m.keys()),
        }
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CompactMap::Large(a), CompactMap::Large(b)) => a.ptr_eq(b),
            _ => false,
        }
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            CompactMap::Inline(sv) => {
                if let Some(pos) = sv.iter().position(|(k, _)| k == key) {
                    Some(sv.swap_remove(pos).1)
                } else {
                    None
                }
            }
            CompactMap::Large(m) => m.remove(key),
        }
    }

    fn values(&self) -> CompactMapValues<'_, K, V> {
        match self {
            CompactMap::Inline(sv) => CompactMapValues::Inline(sv.iter()),
            CompactMap::Large(m) => CompactMapValues::Large(m.values()),
        }
    }

    fn iter(&self) -> CompactMapIter<'_, K, V> {
        match self {
            CompactMap::Inline(sv) => CompactMapIter::Inline(sv.iter()),
            CompactMap::Large(m) => CompactMapIter::Large(m.iter()),
        }
    }
}

enum CompactMapKeys<'a, K, V> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(im::hashmap::Keys<'a, K, V>),
}

impl<'a, K: Clone, V> Iterator for CompactMapKeys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactMapKeys::Inline(it) => it.next().map(|(k, _)| k),
            CompactMapKeys::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactMapKeys::Inline(it) => it.size_hint(),
            CompactMapKeys::Large(it) => it.size_hint(),
        }
    }
}

enum CompactMapValues<'a, K, V> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(im::hashmap::Values<'a, K, V>),
}

impl<'a, K, V> Iterator for CompactMapValues<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactMapValues::Inline(it) => it.next().map(|(_, v)| v),
            CompactMapValues::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactMapValues::Inline(it) => it.size_hint(),
            CompactMapValues::Large(it) => it.size_hint(),
        }
    }
}

enum CompactMapIter<'a, K, V> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(im::hashmap::Iter<'a, K, V>),
}

impl<'a, K: Clone, V: Clone> Iterator for CompactMapIter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactMapIter::Inline(it) => it.next().map(|(k, v)| (k, v)),
            CompactMapIter::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactMapIter::Inline(it) => it.size_hint(),
            CompactMapIter::Large(it) => it.size_hint(),
        }
    }
}

impl<'a, K: Clone + Eq + Hash, V: Clone> IntoIterator for &'a CompactMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = CompactMapIter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A map optimized for small sizes (≤2 entries) keyed by u32 (depth).
/// Replaces `OrdMap<u32, Arc<N>>` in the GSS children maps.
/// Typical GSS paths have 1 entry; this avoids B-tree overhead.
#[derive(Clone, PartialEq, Eq)]
enum CompactOrdMap<V: Clone> {
    Inline(SmallVec<[(u32, V); 2]>),
    Large(OrdMap<u32, V>),
}

impl<V: Clone> CompactOrdMap<V> {
    #[inline(always)]
    fn new() -> Self {
        CompactOrdMap::Inline(SmallVec::new())
    }

    #[inline(always)]
    fn unit(key: u32, value: V) -> Self {
        let mut sv = SmallVec::new();
        sv.push((key, value));
        CompactOrdMap::Inline(sv)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            CompactOrdMap::Inline(sv) => sv.len(),
            CompactOrdMap::Large(m) => m.len(),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self {
            CompactOrdMap::Inline(sv) => sv.is_empty(),
            CompactOrdMap::Large(m) => m.is_empty(),
        }
    }

    #[inline(always)]
    fn get(&self, key: &u32) -> Option<&V> {
        match self {
            CompactOrdMap::Inline(sv) => sv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            CompactOrdMap::Large(m) => m.get(key),
        }
    }

    fn insert(&mut self, key: u32, value: V) -> Option<V> {
        match self {
            CompactOrdMap::Inline(sv) => {
                for entry in sv.iter_mut() {
                    if entry.0 == key {
                        let old = std::mem::replace(&mut entry.1, value);
                        return Some(old);
                    }
                }
                if sv.len() < 2 {
                    sv.push((key, value));
                    None
                } else {
                    // Promote to Large
                    let mut m = OrdMap::new();
                    for (k, v) in sv.drain(..) {
                        m.insert(k, v);
                    }
                    let result = m.insert(key, value);
                    *self = CompactOrdMap::Large(m);
                    result
                }
            }
            CompactOrdMap::Large(m) => m.insert(key, value),
        }
    }

    fn keys(&self) -> CompactOrdMapKeys<'_, V> {
        match self {
            CompactOrdMap::Inline(sv) => CompactOrdMapKeys::Inline(sv.iter()),
            CompactOrdMap::Large(m) => CompactOrdMapKeys::Large(m.keys()),
        }
    }

    fn get_max(&self) -> Option<(&u32, &V)> {
        match self {
            CompactOrdMap::Inline(sv) => {
                sv.iter().max_by_key(|(k, _)| *k).map(|(k, v)| (k, v))
            }
            CompactOrdMap::Large(m) => m.get_max().map(|(k, v)| (k, v)),
        }
    }

    fn iter(&self) -> CompactOrdMapIter<'_, V> {
        match self {
            CompactOrdMap::Inline(sv) => CompactOrdMapIter::Inline(sv.iter()),
            CompactOrdMap::Large(m) => CompactOrdMapIter::Large(m.iter()),
        }
    }

    fn values(&self) -> CompactOrdMapValues<'_, V> {
        match self {
            CompactOrdMap::Inline(sv) => CompactOrdMapValues::Inline(sv.iter()),
            CompactOrdMap::Large(m) => CompactOrdMapValues::Large(m.values()),
        }
    }
}

enum CompactOrdMapIter<'a, V> {
    Inline(std::slice::Iter<'a, (u32, V)>),
    Large(im::ordmap::Iter<'a, u32, V>),
}

impl<'a, V: Clone> Iterator for CompactOrdMapIter<'a, V> {
    type Item = (&'a u32, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactOrdMapIter::Inline(it) => it.next().map(|(k, v)| (k, v)),
            CompactOrdMapIter::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactOrdMapIter::Inline(it) => it.size_hint(),
            CompactOrdMapIter::Large(it) => it.size_hint(),
        }
    }
}

enum CompactOrdMapValues<'a, V> {
    Inline(std::slice::Iter<'a, (u32, V)>),
    Large(im::ordmap::Values<'a, u32, V>),
}

impl<'a, V: Clone> Iterator for CompactOrdMapValues<'a, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactOrdMapValues::Inline(it) => it.next().map(|(_, v)| v),
            CompactOrdMapValues::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactOrdMapValues::Inline(it) => it.size_hint(),
            CompactOrdMapValues::Large(it) => it.size_hint(),
        }
    }
}

impl<V: Clone> std::iter::FromIterator<(u32, V)> for CompactOrdMap<V> {
    fn from_iter<I: IntoIterator<Item = (u32, V)>>(iter: I) -> Self {
        let mut map = CompactOrdMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

enum CompactOrdMapKeys<'a, V> {
    Inline(std::slice::Iter<'a, (u32, V)>),
    Large(im::ordmap::Keys<'a, u32, V>),
}

impl<'a, V> Iterator for CompactOrdMapKeys<'a, V> {
    type Item = &'a u32;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactOrdMapKeys::Inline(it) => it.next().map(|(k, _)| k),
            CompactOrdMapKeys::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactOrdMapKeys::Inline(it) => it.size_hint(),
            CompactOrdMapKeys::Large(it) => it.size_hint(),
        }
    }
}

impl<'a, V: Clone> IntoIterator for &'a CompactOrdMap<V> {
    type Item = (&'a u32, &'a V);
    type IntoIter = CompactOrdMapIter<'a, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

type Children<T, N> = CompactMap<T, CompactOrdMap<Arc<N>>>;

/// Maximum number of values packed into a single Segment node.
const SEGMENT_CAP: usize = 32;

/// Linear segment of the stack: multiple values packed into one node.
/// `values[0]` is the deepest value (closest to `next`),
/// `values[last]` is the shallowest (top of stack).
/// Intermediate levels (all except the top) are guaranteed to have empty=false.
/// Values are stack-allocated up to SEGMENT_CAP.
/// Segments are always non-accepting (empty is implicitly false).
struct Segment<T: Clone + Eq + Hash> {
    values: ArrayVec<T, SEGMENT_CAP>,
    next: Arc<Lower<T>>,
    max_depth: u32,
    segments_len: usize,
    rest: OnceLock<Arc<Lower<T>>>,
}

impl<T: Clone + Eq + Hash> Clone for Segment<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            next: self.next.clone(),
            max_depth: self.max_depth,
            segments_len: self.segments_len,
            rest: OnceLock::new(),
        }
    }
}

impl<T: Clone + Eq + Hash> PartialEq for Segment<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
            && self.next == other.next
            && self.max_depth == other.max_depth
            && self.segments_len == other.segments_len
    }
}

impl<T: Clone + Eq + Hash> Eq for Segment<T> {}

#[derive(Clone, PartialEq, Eq)]
enum Lower<T: Clone + Eq + Hash> {
    General {
        children: Children<T, Lower<T>>,
        empty: bool,
        max_depth: u32,
    },
    Segment(Arc<Segment<T>>),
}

/// Get a stable identity for a Lower node wrapped in Arc.
/// For Segment nodes, uses the inner Arc<Segment> pointer (since the outer Arc<Lower>
/// may be ephemeral when constructed from segment_rest_arc or children()).
/// For General nodes, uses the outer Arc<Lower> pointer directly.
#[inline]
fn lower_node_id<T: Clone + Eq + Hash>(node: &Arc<Lower<T>>) -> usize {
    match &**node {
        Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
        _ => Arc::as_ptr(node) as usize,
    }
}

impl<T: Clone + Eq + Hash> Lower<T> {
    #[inline(always)]
    fn empty(&self) -> bool {
        match self {
            Lower::General { empty, .. } => *empty,
            Lower::Segment(_) => false,
        }
    }

    #[inline(always)]
    fn max_depth(&self) -> u32 {
        match self {
            Lower::General { max_depth, .. } => *max_depth,
            Lower::Segment(seg) => seg.max_depth,
        }
    }

    #[inline(always)]
    fn segments_len(&self) -> usize {
        match self {
            Lower::Segment(seg) => seg.segments_len,
            Lower::General { .. } => 0,
        }
    }

    /// If this node is a deterministic chain link (Segment or single-child
    /// General with one edge), return the next node below it plus the number
    /// of stack values represented at this node.
    #[inline]
    fn chain_step(&self) -> Option<(&Arc<Lower<T>>, usize)> {
        match self {
            Lower::Segment(seg) => Some((&seg.next, seg.values.len())),
            Lower::General { children, .. } if children.len() == 1 => {
                let ordmap = children.values().next().unwrap();
                if ordmap.len() == 1 {
                    Some((ordmap.iter().next().unwrap().1, 1))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Append the values represented by this deterministic chain node,
    /// top-first, into `out`.
    #[inline]
    fn append_chain_values_top_first(&self, out: &mut SmallVec<[T; 16]>) {
        match self {
            Lower::Segment(seg) => {
                for value in seg.values.iter().rev() {
                    out.push(value.clone());
                }
            }
            Lower::General { children, .. } => {
                out.push(children.keys().next().unwrap().clone());
            }
        }
    }

    /// Get children as a general Children map.
    /// For Segment, constructs a map with the top value → rest-of-segment.
    fn children(&self) -> Children<T, Lower<T>> {
        match self {
            Lower::General { children, .. } => children.clone(),
            Lower::Segment(seg) => {
                let top_value = seg.values.last().unwrap().clone();
                let child = self.segment_rest_arc();
                CompactMap::unit(top_value, CompactOrdMap::unit(child.max_depth(), child))
            }
        }
    }

    /// Consume self and return (children, empty, max_depth).
    fn into_parts(self) -> (Children<T, Lower<T>>, bool, u32) {
        match self {
            Lower::General { children, empty, max_depth } => (children, empty, max_depth),
            Lower::Segment(seg) => {
                let top_value = seg.values.last().unwrap().clone();
                let seg = Arc::try_unwrap(seg).unwrap_or_else(|arc| (*arc).clone());
                let max_depth = seg.max_depth;
                let child = if seg.values.len() == 1 {
                    seg.next
                } else {
                    let mut rest_values = seg.values;
                    rest_values.pop();
                    new_segment(rest_values, seg.next)
                };
                let children = CompactMap::unit(top_value, CompactOrdMap::unit(child.max_depth(), child));
                (children, false, max_depth)
            }
        }
    }

    /// Number of distinct child keys (at the top level).
    #[inline(always)]
    fn children_len(&self) -> usize {
        match self {
            Lower::General { children, .. } => children.len(),
            Lower::Segment(_) => 1,
        }
    }

    /// Whether there are no children.
    #[inline(always)]
    fn children_is_empty(&self) -> bool {
        match self {
            Lower::General { children, .. } => children.is_empty(),
            Lower::Segment(_) => false,
        }
    }

    /// Check if the top-level children contains a key.
    fn children_contains_key(&self, key: &T) -> bool {
        match self {
            Lower::General { children, .. } => children.contains_key(key),
            Lower::Segment(seg) => seg.values.last().unwrap() == key,
        }
    }

    /// Ensure this Lower is in General form, converting from Segment if necessary.
    fn ensure_general(&mut self) {
        if let Lower::Segment(_) = self {
            let old = std::mem::replace(self, Lower::General {
                children: CompactMap::new(),
                empty: false,
                max_depth: 0,
            });
            let (children, empty, max_depth) = old.into_parts();
            *self = Lower::General { children, empty, max_depth };
        }
    }

    /// Returns true if this is a Segment variant.
    #[inline(always)]
    fn is_segment(&self) -> bool {
        matches!(self, Lower::Segment(_))
    }

    /// For Segment variant, get the shallowest (top) value by reference.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_top_value(&self) -> &T {
        match self {
            Lower::Segment(seg) => seg.values.last().unwrap(),
            Lower::General { .. } => panic!("segment_top_value called on General"),
        }
    }

    /// For Segment variant, get the deep-end next pointer.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_next(&self) -> &Arc<Lower<T>> {
        match self {
            Lower::Segment(seg) => &seg.next,
            Lower::General { .. } => panic!("segment_next called on General"),
        }
    }

    /// For Segment variant, get the values slice.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_values(&self) -> &[T] {
        match self {
            Lower::Segment(seg) => &seg.values,
            Lower::General { .. } => panic!("segment_values called on General"),
        }
    }

    /// For Segment variant, return an Arc to the "rest" (everything below the top value).
    /// If len==1, wraps next in Arc. Otherwise creates a new shorter Segment.
    fn segment_rest_arc(&self) -> Arc<Lower<T>> {
        match self {
            Lower::Segment(seg) if seg.values.len() == 1 => seg.next.clone(),
            Lower::Segment(seg) => seg.rest.get_or_init(|| {
                let rest_values: ArrayVec<T, SEGMENT_CAP> =
                    seg.values[..seg.values.len() - 1].iter().cloned().collect();
                new_segment(rest_values, seg.next.clone())
            }).clone(),
            Lower::General { .. } => panic!("segment_rest_arc called on General"),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Interface<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Lower<T>>,
    acc: A,
}

#[derive(Clone, PartialEq, Eq)]
struct UpperBranch<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
    max_depth: u32,
}

#[derive(Clone, PartialEq, Eq)]
enum Upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    Branch(Arc<UpperBranch<T, A>>),
    Interface(Arc<Interface<T, A>>),
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Upper<T, A> {
    fn max_depth(&self) -> u32 {
        match self {
            Upper::Branch(branch) => branch.max_depth,
            Upper::Interface(interface) => interface.inner.max_depth(),
        }
    }

    fn children_keys(&self) -> SmallVec<[T; 8]> {
        match self {
            Upper::Branch(branch) => branch.children.keys().cloned().collect(),
            Upper::Interface(interface) => match &*interface.inner {
                Lower::Segment(seg) => smallvec![seg.values.last().unwrap().clone()],
                Lower::General { children, .. } => children.keys().cloned().collect(),
            },
        }
    }

    fn single_child_key(&self) -> Option<T> {
        match self {
            Upper::Branch(branch) => {
                if branch.children.len() == 1 {
                    branch.children.keys().next().cloned()
                } else {
                    None
                }
            }
            Upper::Interface(interface) => {
                if interface.inner.children_len() == 1 {
                    match &*interface.inner {
                        Lower::Segment(seg) => Some(seg.values.last().unwrap().clone()),
                        Lower::General { children, .. } => children.keys().next().cloned(),
                    }
                } else {
                    None
                }
            }
        }
    }

    fn single_child_key_without_empty(&self) -> Option<T> {
        match self {
            Upper::Branch(branch) => {
                if branch.empty.is_none() && branch.children.len() == 1 {
                    branch.children.keys().next().cloned()
                } else {
                    None
                }
            }
            Upper::Interface(interface) => {
                if !interface.inner.empty() && interface.inner.children_len() == 1 {
                    match &*interface.inner {
                        Lower::Segment(seg) => Some(seg.values.last().unwrap().clone()),
                        Lower::General { children, .. } => children.keys().next().cloned(),
                    }
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LeveledGSSSummary {
    pub top_values_count: usize,
    pub upperbranch_nodes: usize,
    pub interface_nodes: usize,
    pub lower_nodes: usize,
    pub total_unique_nodes: usize,
    pub total_edges: usize,
    pub accumulator_instances: usize,
    pub max_depth: u32,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GSSPathsInfo {
    pub num_paths: usize,
    pub total_depth: usize,
}

#[cfg(test)]
impl std::ops::Add for GSSPathsInfo {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            num_paths: self.num_paths + rhs.num_paths,
            total_depth: self.total_depth + rhs.total_depth,
        }
    }
}

#[cfg(test)]
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

fn max_depth_from_children<T, N, F>(children: &Children<T, N>, depth_of: F) -> u32
where
    T: Clone + Eq + Hash,
    F: Fn(&Arc<N>) -> u32,
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
    // Use Segment variant when there's exactly one key with one depth entry.
    // NOTE: We do NOT pack into existing child Segments here. Packing only happens
    // in batch-construction paths (into_gss) so that incremental push/pop
    // preserves Arc sharing (the child Arc is reused on pop).
    // Only compress to Segment when empty is false — Segments are non-accepting.
    if !empty && children.len() == 1 {
        let (key, ord_map) = children.iter().next().unwrap();
        if ord_map.len() == 1 {
            let (_, next) = ord_map.iter().next().unwrap();
            let mut values = ArrayVec::<T, SEGMENT_CAP>::new();
            values.push(key.clone());
            return new_segment(values, next.clone());
        }
    }
    let max_depth = max_depth_from_children(&children, |n: &Arc<Lower<T>>| n.max_depth());
    Arc::new(Lower::General {
        children,
        empty,
        max_depth,
    })
}

fn new_segment<T: Clone + Eq + Hash>(values: ArrayVec<T, SEGMENT_CAP>, next: Arc<Lower<T>>) -> Arc<Lower<T>> {
    let max_depth = next.max_depth() + values.len() as u32;
    let segments_len = values.len() + next.segments_len();
    Arc::new(Lower::Segment(Arc::new(Segment {
        values,
        next,
        max_depth,
        segments_len,
        rest: OnceLock::new(),
    })))
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
    new_branch(CompactMap::new(), None)
}

#[cfg(test)]
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

    let keep_empty = node.empty() && current_depth >= min_d;

    let mut new_children: Children<T, Lower<T>> = CompactMap::new();
    let mut children_identical = true;

    if current_depth < max_d {
        match &**node {
            Lower::Segment(seg) => {
                let value = seg.values.last().unwrap();
                let rest = node.segment_rest_arc();
                if let Some(new_child) = filter_lower(&rest, current_depth + 1, min_len, max_len) {
                    if !Arc::ptr_eq(&new_child, &rest) || new_child.max_depth() != rest.max_depth() {
                        children_identical = false;
                    }
                    new_children.insert(value.clone(), CompactOrdMap::unit(new_child.max_depth(), new_child));
                } else {
                    children_identical = false;
                }
            }
            Lower::General { children: node_children, .. } => {
                for (v, kids) in node_children.iter() {
                    let mut new_kids: CompactOrdMap<Arc<Lower<T>>> = CompactOrdMap::new();
                    let mut same_kids = true;
                    let mut count = 0usize;
                    for (orig_depth, child) in kids.iter() {
                        if let Some(new_child) =
                            filter_lower(child, current_depth + 1, min_len, max_len)
                        {
                            if !Arc::ptr_eq(&new_child, child) || new_child.max_depth() != *orig_depth {
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
            }
        }
    } else {
        
        children_identical = node.children_is_empty();
    }

    if keep_empty == node.empty() && children_identical {
        return Some(node.clone());
    }

    if !keep_empty && new_children.is_empty() {
        None
    } else {
        Some(new_lower(new_children, keep_empty))
    }
}

#[cfg(test)]
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

            let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
            let mut children_identical = true;
            if current_depth < max_d {
                for (v, kids) in b.children.iter() {
                    let mut new_kids: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
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
                
                children_identical = b.children.is_empty();
            }

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
            let keep_empty = i.inner.empty() && current_depth >= min_d;

            let mut new_children: Children<T, Lower<T>> = CompactMap::new();
            let mut children_identical = true;
            if current_depth < max_d {
                match &*i.inner {
                    Lower::Segment(seg) => {
                        let value = seg.values.last().unwrap();
                        let rest = i.inner.segment_rest_arc();
                        if let Some(new_child) = filter_lower(&rest, current_depth + 1, min_len, max_len) {
                            if !Arc::ptr_eq(&new_child, &rest) || new_child.max_depth() != rest.max_depth() {
                                children_identical = false;
                            }
                            new_children.insert(value.clone(), CompactOrdMap::unit(new_child.max_depth(), new_child));
                        } else {
                            children_identical = false;
                        }
                    }
                    Lower::General { children: inner_children, .. } => {
                        for (v, kids) in inner_children.iter() {
                            let mut new_kids: CompactOrdMap<Arc<Lower<T>>> = CompactOrdMap::new();
                            let mut same_kids = true;
                            let mut count = 0usize;
                            for (orig_depth, child) in kids.iter() {
                                if let Some(new_child) =
                                    filter_lower(child, current_depth + 1, min_len, max_len)
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
                    }
                }
            } else {
                children_identical = i.inner.children_is_empty();
            }

            if !keep_empty && new_children.is_empty() {
                return None;
            }

            if keep_empty == i.inner.empty() && children_identical {
                return Some(node.clone());
            }

            let new_inner = new_lower(new_children, keep_empty);
            Some(new_interface(new_inner, i.acc.clone()))
        }
    }
}

fn truncate_lower<T: Clone + Eq + Hash>(
    node: &Arc<Lower<T>>,
    current_depth: isize,
    max_len: isize,
    memo: &mut StdHashMap<usize, Option<Arc<Lower<T>>>>,
) -> Option<Arc<Lower<T>>> {
    let ptr = lower_node_id(node);
    if let Some(cached) = memo.get(&ptr) {
        return cached.clone();
    }

    if current_depth == max_len {
        let res = if node.empty() || !node.children_is_empty() {
            Some(new_lower(CompactMap::new(), true))
        } else {
            None
        };
        memo.insert(ptr, res.clone());
        return res;
    }

    let mut new_children: Children<T, Lower<T>> = CompactMap::new();
    let mut children_identical = true;

    match &**node {
        Lower::Segment(seg) => {
            let value = seg.values.last().unwrap();
            let rest = node.segment_rest_arc();
            if let Some(new_child) = truncate_lower(&rest, current_depth + 1, max_len, memo) {
                if !Arc::ptr_eq(&rest, &new_child) || rest.max_depth() != new_child.max_depth() {
                    children_identical = false;
                }
                new_children.insert(value.clone(), CompactOrdMap::unit(new_child.max_depth(), new_child));
            } else {
                children_identical = false;
            }
        }
        Lower::General { children: node_children, .. } => {
            for (v, kids) in node_children.iter() {
                let mut new_kids_map = CompactOrdMap::new();
                let mut kids_identical = true;
                for (depth, child) in kids.iter() {
                    if let Some(new_child) = truncate_lower(child, current_depth + 1, max_len, memo) {
                        if !Arc::ptr_eq(child, &new_child) || *depth != new_child.max_depth() {
                            kids_identical = false;
                        }
                        new_kids_map.insert(new_child.max_depth(), new_child);
                    } else {
                        kids_identical = false;
                    }
                }
                if !new_kids_map.is_empty() {
                    new_children.insert(v.clone(), new_kids_map);
                } else {
                    children_identical = false;
                }
                children_identical &= kids_identical;
            }
        }
    }

    if node.empty() && children_identical {
        memo.insert(ptr, Some(node.clone()));
        return Some(node.clone());
    }

    let res = if !node.empty() && new_children.is_empty() {
        None
    } else {
        Some(new_lower(new_children, node.empty()))
    };
    memo.insert(ptr, res.clone());
    res
}

fn truncate_upper<T, A>(
    node: &Arc<Upper<T, A>>,
    current_depth: isize,
    max_len: isize,
    memo_upper: &mut StdHashMap<usize, Option<Arc<Upper<T, A>>>>,
    memo_lower: &mut StdHashMap<usize, Option<Arc<Lower<T>>>>,
) -> Option<Arc<Upper<T, A>>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let ptr = Arc::as_ptr(node) as usize;
    if let Some(cached) = memo_upper.get(&ptr) {
        return cached.clone();
    }

    if current_depth == max_len {
        let sub_gss = LeveledGSS {
            inner: node.clone(),
        };
        let res = if let Some(acc) = sub_gss.reduce_acc() {
            let terminal_lower = new_lower(CompactMap::new(), true);
            Some(new_interface(terminal_lower, acc))
        } else {
            None
        };
        memo_upper.insert(ptr, res.clone());
        return res;
    }

    let res = match &**node {
        Upper::Branch(b) => {
            let new_empty = b.empty.clone();
            let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
            let mut children_identical = true;

            for (v, kids) in b.children.iter() {
                let mut new_kids_map = CompactOrdMap::new();
                let mut kids_identical = true;
                for (depth, child) in kids.iter() {
                    if let Some(new_child) =
                        truncate_upper(child, current_depth + 1, max_len, memo_upper, memo_lower)
                    {
                        if !Arc::ptr_eq(child, &new_child) || *depth != new_child.max_depth() {
                            kids_identical = false;
                        }
                        new_kids_map.insert(new_child.max_depth(), new_child);
                    } else {
                        kids_identical = false;
                    }
                }
                if !new_kids_map.is_empty() {
                    new_children.insert(v.clone(), new_kids_map);
                } else {
                    children_identical = false;
                }
                children_identical &= kids_identical;
            }

            if new_empty == b.empty && children_identical {
                memo_upper.insert(ptr, Some(node.clone()));
                return Some(node.clone());
            }

            if new_children.is_empty() && new_empty.is_none() {
                None
            } else {
                Some(try_promote(&new_branch(new_children, new_empty)))
            }
        }
        Upper::Interface(i) => {
            if let Some(new_inner) = truncate_lower(&i.inner, current_depth, max_len, memo_lower) {
                if Arc::ptr_eq(&i.inner, &new_inner) {
                    Some(node.clone())
                } else {
                    Some(new_interface(new_inner, i.acc.clone()))
                }
            } else {
                None
            }
        }
    };

    memo_upper.insert(ptr, res.clone());
    res
}

#[cfg(test)]
fn accs_by_depth_lower<T, A>(
    node: &Arc<Lower<T>>,
    current_depth: isize,
    acc: &A,
    accs: &mut BTreeMap<isize, A>,
    memo_lower: &mut HashSet<usize>,
) where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let ptr = lower_node_id(node);
    if !memo_lower.insert(ptr) {
        return;
    }

    if node.empty() {
        if let Some(existing) = accs.get_mut(&current_depth) {
            *existing = existing.merge(acc);
        } else {
            accs.insert(current_depth, acc.clone());
        }
    }

    for kids in node.children().values() {
        for child in kids.values() {
            accs_by_depth_lower(child, current_depth + 1, acc, accs, memo_lower);
        }
    }
}

#[cfg(test)]
fn accs_by_depth_upper<T, A>(
    node: &Arc<Upper<T, A>>,
    current_depth: isize,
    accs: &mut BTreeMap<isize, A>,
    memo_upper: &mut HashSet<usize>,
) where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let ptr = Arc::as_ptr(node) as usize;
    if !memo_upper.insert(ptr) {
        return;
    }

    match &**node {
        Upper::Branch(b) => {
            if let Some(acc) = &b.empty {
                if let Some(existing) = accs.get_mut(&current_depth) {
                    *existing = existing.merge(acc);
                } else {
                    accs.insert(current_depth, acc.clone());
                }
            }
            for kids in b.children.values() {
                for child in kids.values() {
                    accs_by_depth_upper(child, current_depth + 1, accs, memo_upper);
                }
            }
        }
        Upper::Interface(i) => {
            let mut memo_lower = HashSet::new();
            accs_by_depth_lower(&i.inner, current_depth, &i.acc, accs, &mut memo_lower);
        }
    }
}

fn merge_lower<T: Clone + Eq + Hash>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {
    if Arc::ptr_eq(l1, l2) {
        return l1.clone();
    }
    let new_empty = l1.empty() || l2.empty();
    // Fast path: merge two Segment nodes or Segment + General without building full Children
    let merged_children = match (&**l1, &**l2) {
        (Lower::Segment(_), Lower::Segment(_)) => {
            let v1 = l1.segment_top_value();
            let v2 = l2.segment_top_value();
            let r1 = l1.segment_rest_arc();
            let r2 = l2.segment_rest_arc();
            if v1 == v2 {
                // Same key: merge the rests
                let merged_next = merge_lower(&r1, &r2);
                CompactMap::unit(v1.clone(), CompactOrdMap::unit(merged_next.max_depth(), merged_next))
            } else {
                // Different keys: two entries
                let mut c = CompactMap::unit(v1.clone(), CompactOrdMap::unit(r1.max_depth(), r1));
                c.insert(v2.clone(), CompactOrdMap::unit(r2.max_depth(), r2));
                c
            }
        }
        (Lower::Segment(_), Lower::General { children, .. }) |
        (Lower::General { children, .. }, Lower::Segment(_)) => {
            let seg = if l1.is_segment() { l1 } else { l2 };
            let value = seg.segment_top_value();
            let rest = seg.segment_rest_arc();
            let mut merged = children.clone();
            let seg_kids = CompactOrdMap::unit(rest.max_depth(), rest.clone());
            if let Some(existing) = merged.get(value) {
                let mut new_map = existing.clone();
                let depth = rest.max_depth();
                if let Some(existing_child) = new_map.get(&depth) {
                    new_map.insert(depth, merge_lower(existing_child, &rest));
                } else {
                    new_map.insert(depth, rest);
                }
                merged.insert(value.clone(), new_map);
            } else {
                merged.insert(value.clone(), seg_kids);
            }
            merged
        }
        (Lower::General { children: c1, .. }, Lower::General { children: c2, .. }) => {
            merge_children(c1, c2, |a, b| merge_lower(a, b))
        }
    };
    new_lower(merged_children, new_empty)
}

fn interface_to_upperbranch<T, A>(it: &Arc<Interface<T, A>>) -> Arc<UpperBranch<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let mut children: Children<T, Upper<T, A>> = CompactMap::new();
    match &*it.inner {
        Lower::Segment(seg) => {
            let value = seg.values.last().unwrap();
            let rest = it.inner.segment_rest_arc();
            let ci = new_interface(rest, it.acc.clone());
            let v_map = CompactOrdMap::unit(ci.max_depth(), ci);
            children.insert(value.clone(), v_map);
        }
        Lower::General { children: inner_children, .. } => {
            for (v, kids) in inner_children.iter() {
                let mut v_map: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
                for lchild in kids.values() {
                    let ci = new_interface(lchild.clone(), it.acc.clone());
                    v_map.insert(ci.max_depth(), ci);
                }
                if !v_map.is_empty() {
                    children.insert(v.clone(), v_map);
                }
            }
        }
    }

    let new_empty = if it.inner.empty() {
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
    let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
        children: merged_children,
        empty: new_empty,
        max_depth: a.max_depth.max(b.max_depth),
    })));
    try_promote(&new_b)
}

fn merge_interfaces<T, A>(a: &Arc<Interface<T, A>>, b: &Arc<Interface<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
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
        // Check all children are Interface (early exit without allocation).
        let mut has_children = false;
        for c in b.children.values().flat_map(|kids| kids.values()) {
            has_children = true;
            if !matches!(&**c, Upper::Interface(_)) {
                return node.clone();
            }
        }

        if !has_children {
            if let Some(empty) = &b.empty {
                let lower_root = new_lower(CompactMap::new(), true);
                return new_interface(lower_root, empty.clone());
            }
            return node.clone();
        }

        // All children are Interface. Collect accumulators (re-iterate).
        let mut accs: HashSet<A> = HashSet::new();
        if let Some(empty) = &b.empty {
            accs.insert(empty.clone());
        }
        for c in b.children.values().flat_map(|kids| kids.values()) {
            if let Upper::Interface(ic) = &**c {
                accs.insert(ic.acc.clone());
            }
        }

        if accs.len() <= 1 {
            if let Some(the_acc) = accs.into_iter().next() {
                
                let mut l_children: Children<T, Lower<T>> = CompactMap::new();
                for (v, kids) in b.children.iter() {
                    let mut v_map: CompactOrdMap<Arc<Lower<T>>> = CompactOrdMap::new();
                    for child in kids.values() {
                        if let Upper::Interface(ci) = &**child {
                            let lower = ci.inner.clone();
                            v_map.insert(lower.max_depth(), lower);
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

#[cfg(test)]
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
    pub max_upper_depth: u32,
    pub max_lower_depth: u32,
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

#[cfg(test)]
impl<T: Clone + Eq + Hash + std::fmt::Debug, A: Clone + Eq + Hash + std::fmt::Debug> std::fmt::Debug
    for LeveledGSSStats<T, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeveledGSSStats")
            
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
            
            .field("unique_accumulators_count", &self.unique_accumulators_count)
            
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

#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, PartialEq, Eq)]
struct LowerSig<T: Clone + Eq + Hash> {
    empty: bool,
    edges: StdHashMap<T, Vec<usize>>,
}

#[cfg(test)]
#[derive(Clone, PartialEq, Eq)]
enum UpperSig<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    Branch {
        empty: Option<A>,
        edges: StdHashMap<T, Vec<usize>>,
    },
    Interface {
        acc: A,
        inner_id: usize,
    },
}

#[cfg(test)]
struct NormalizationLowerInterner<T: Clone + Eq + Hash> {
    map: StdHashMap<u64, Vec<(LowerSig<T>, usize, Arc<Lower<T>>)>>,
    next_id: usize,
}

#[cfg(test)]
impl<T: Clone + Eq + Hash> Default for NormalizationLowerInterner<T> {
    fn default() -> Self {
        Self {
            map: StdHashMap::new(),
            next_id: 0,
        }
    }
}

#[cfg(test)]
struct NormalizationUpperInterner<
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
> {
    map: StdHashMap<u64, Vec<(UpperSig<T, A>, usize, Arc<Upper<T, A>>)>>,
    next_id: usize,
}

#[cfg(test)]
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

#[cfg(test)]
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
        ids.hash(&mut h); 
        let e = h.finish();
        xor_acc ^= e;
        sum_acc = sum_acc.wrapping_add(e);
        prod_acc = prod_acc.wrapping_mul(e.wrapping_add(0x9e37_79b9_7f4a_7c15));
    }
    seed ^ xor_acc ^ sum_acc ^ prod_acc
}

#[cfg(test)]
fn upper_sig_hash<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
    sig: &UpperSig<T, A>,
) -> u64 {
    use std::hash::{Hash as _, Hasher};
    match sig {
        UpperSig::Branch { empty, edges } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            if let Some(e) = empty {
                e.hash(&mut h);
            }
            let seed = 0x6a09_e667_f3bc_c908u64 ^ h.finish() ^ (edges.len() as u64);

            let mut xor_acc: u64 = 0;
            let mut sum_acc: u64 = 0;
            let mut prod_acc: u64 = 0x3c6e_f372_fe94_f82b ^ (edges.len() as u64);
            for (k, ids) in edges {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                k.hash(&mut h);
                ids.hash(&mut h); 
                let e = h.finish();
                xor_acc ^= e;
                sum_acc = sum_acc.wrapping_add(e);
                prod_acc = prod_acc.wrapping_mul(e.wrapping_add(0xa54f_f53a_5f1d_36f1));
            }
            seed ^ xor_acc ^ sum_acc ^ prod_acc
        }
        UpperSig::Interface { acc, inner_id } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            acc.hash(&mut h);
            inner_id.hash(&mut h);
            0xbb67_ae85_84ca_a73bu64 ^ h.finish()
        }
    }
}

#[cfg(test)]
fn normalize_canonicalize_lower<T>(
    node: &Arc<Lower<T>>,
    memo_lower: &mut StdHashMap<usize, (usize, Arc<Lower<T>>)>,
    interner_lower: &mut NormalizationLowerInterner<T>,
) -> (usize, Arc<Lower<T>>)
where
    T: Clone + Eq + Hash,
{
    let ptr = lower_node_id(node);
    if let Some((id, arc)) = memo_lower.get(&ptr) {
        return (*id, arc.clone());
    }

    let mut edges_raw: StdHashMap<T, Vec<(usize, Arc<Lower<T>>)>> = StdHashMap::new();
    let node_children = node.children();
    for (v, kids) in node_children.iter() {
        let entry = edges_raw.entry(v.clone()).or_default();
        for child in kids.values() {
            let (cid, carc) = normalize_canonicalize_lower(child, memo_lower, interner_lower);
            entry.push((cid, carc));
        }
    }

    let mut sig_edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
    for (v, items) in &edges_raw {
        let mut ids: Vec<usize> = items.iter().map(|(cid, _)| *cid).collect();
        ids.sort_unstable();
        sig_edges.insert(v.clone(), ids);
    }
    let sig = LowerSig {
        empty: node.empty(),
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

    let mut new_children: Children<T, Lower<T>> = CompactMap::new();
    for (v, items) in edges_raw {
        let mut ord: CompactOrdMap<Arc<Lower<T>>> = CompactOrdMap::new();
        for (_, carc) in items {
            let d = carc.max_depth();
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
    let new_node = new_lower(new_children, node.empty());

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

#[cfg(test)]
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

            let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
            for (v, items) in &edges_raw {
                let mut ord: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
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
            let mut new_node = try_promote(&new_b); 

            match &*new_node {
                Upper::Interface(i2) => {
                    let i2_owned = i2.clone(); 
                    
                    let (inner_id, canonical_inner) = normalize_canonicalize_lower(
                        &i2_owned.inner,
                        memo_lower,
                        interner_lower,
                    );
                    
                    new_node = new_interface(canonical_inner, i2_owned.acc.clone());

                    sig = UpperSig::Interface {
                        acc: i2_owned.acc.clone(),
                        inner_id,
                    };
                    h = upper_sig_hash(&sig);
                }
                Upper::Branch(_) => {
                }
            }

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
            let (inner_id, canonical_inner) =
                normalize_canonicalize_lower(&i.inner, memo_lower, interner_lower);

            let sig = UpperSig::Interface {
                acc: i.acc.clone(),
                inner_id,
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

            let new_node = new_interface(canonical_inner, i.acc.clone());

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

#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Upper<T, A>>,
}

/// Opaque handle to the tail of a chain in the GSS.
/// Used by the chain optimization to reconstruct the GSS after walking a chain.
pub struct ChainTail<T: Clone + Eq + Hash> {
    inner: Arc<Lower<T>>,
}

/// A mutable view of the top portion of a GSS as a flat stack of values.
/// Works when the top of the GSS is a deterministic chain (single-child Segments
/// and single-child Generals).
///
/// Instead of extracting all states upfront, this keeps a reference to the
/// original chain and only tracks pushed states (from goto operations).
/// Pops walk through the original chain via Arc references.
/// On commit, only the pushed portion needs new Segment nodes.
///
/// The stack has a "floor" — the Lower node below the deterministic chain.
/// When a pop would cross the floor, the caller falls back to the general path.
pub struct VirtualStack<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    values: ArrayVec<T, SEGMENT_CAP>,
    next: Arc<Lower<T>>,
    acc: A,
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> VirtualStack<T, A> {
    /// The current top-of-stack value, or None if the stack is empty.
    #[inline]
    pub fn top(&self) -> Option<&T> {
        self.values.last()
    }

    /// Pop `n` values from the top.
    /// Returns the number of values that could not be popped because the
    /// segment chain ended at a non-Segment lower node.
    #[inline]
    pub fn pop(&mut self, mut remaining: usize) -> usize {
        while remaining > 0 {
            let take = remaining.min(self.values.len());
            let keep = self.values.len() - take;
            self.values.truncate(keep);
            remaining -= take;
            if remaining == 0 {
                break;
            }
            match &*self.next {
                Lower::Segment(seg) => {
                    self.values = seg.values.clone();
                    self.next = seg.next.clone();
                }
                _ => break,
            }
        }
        // If values was exactly drained, advance to next Segment so top() works.
        if self.values.is_empty() {
            if let Lower::Segment(seg) = &*self.next {
                self.values = seg.values.clone();
                self.next = seg.next.clone();
            }
        }
        remaining
    }

    /// Push a value onto the top of the stack.
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.values.len() == SEGMENT_CAP {
            let spilled = std::mem::take(&mut self.values);
            let old_next = std::mem::replace(&mut self.next, new_lower(CompactMap::new(), false));
            let max_depth = old_next.max_depth() + spilled.len() as u32;
            let segments_len = spilled.len() + old_next.segments_len();
            self.next = Arc::new(Lower::Segment(Arc::new(Segment {
                values: spilled, next: old_next, max_depth, segments_len, rest: OnceLock::new(),
            })));
        }
        self.values.push(value);
    }

    /// The total number of values available across the current segment chain.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len() + self.next.segments_len()
    }

    /// Materialize the virtual stack back into a GSS.
    pub fn into_gss(self) -> LeveledGSS<T, A> {
        LeveledGSS {
            inner: new_interface(new_segment(self.values, self.next), self.acc),
        }
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> PartialEq for LeveledGSS<T, A> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Eq for LeveledGSS<T, A> {}

impl<T: Clone + Eq + Hash + std::fmt::Debug, A: Merge + Clone + Eq + Hash + std::fmt::Debug> std::fmt::Debug for LeveledGSS<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stacks = self.to_stacks();
        f.debug_struct("LeveledGSS")
            .field("num_stacks", &stacks.len())
            .finish()
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn ptr_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    #[cfg(test)]
    fn lower_ptrs_eq(a: &Arc<Lower<T>>, b: &Arc<Lower<T>>) -> bool {
        if a.empty() != b.empty() || a.children_len() != b.children_len() || a.max_depth() != b.max_depth() {
            return false;
        }

        let c1 = a.children();
        let c2 = b.children();
        let keys1: HashSet<_> = c1.keys().collect();
        let keys2: HashSet<_> = c2.keys().collect();
        if keys1 != keys2 {
            return false;
        }
        for (v, kids1) in c1.iter() {
            let kids2 = c2.get(v).unwrap();
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

    #[cfg(test)]
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
                if i1.acc != i2.acc {
                    return false;
                }
                Self::lower_ptrs_eq(&i1.inner, &i2.inner)
            }
            _ => false,
        }
    }

    pub fn empty() -> Self {
        empty_upper()
    }

    /// Create a GSS from a tail Lower node and an accumulator.
    /// Used by the chain optimization to reconstruct the GSS at the bottom of a chain.
    pub fn from_chain_tail_and_acc(tail: ChainTail<T>, acc: A) -> Self {
        LeveledGSS {
            inner: new_interface(tail.inner, acc),
        }
    }

    pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {
        
        let mut canon: StdHashMap<Vec<T>, A> = StdHashMap::new();
        for (vals, acc) in stacks {
            if let Some(existing) = canon.get_mut(vals) {
                let merged = existing.merge(acc);
                *existing = merged;
            } else {
                canon.insert(vals.clone(), acc.clone());
            }
        }

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
            let mut l_children: Children<T, Lower<T>> = CompactMap::new();
            for (v, e) in d.iter() {
                let sub_children = if e.sub.is_empty() {
                    CompactMap::new()
                } else {
                    build_lower(&e.sub).children()
                };
                let node_for_v = new_lower(sub_children, e.end.is_some());
                l_children.insert(v.clone(), CompactOrdMap::unit(node_for_v.max_depth(), node_for_v));
            }
            new_lower(l_children, false)
        }

        fn build_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            d: &StdHashMap<T, Entry<T, A>>,
            root_empty: Option<A>,
        ) -> Arc<Upper<T, A>> {
            let mut children: Children<T, Upper<T, A>> = CompactMap::new();
            let mut all_child_nodes: Vec<Arc<Upper<T, A>>> = Vec::new();

            for (v, e) in d.iter() {
                let mut nodes_for_v: Vec<Arc<Upper<T, A>>> = Vec::new();
                if let Some(end_acc) = &e.end {
                    let leaf = new_branch(CompactMap::new(), Some(end_acc.clone()));
                    nodes_for_v.push(try_promote(&leaf));
                }
                if !e.sub.is_empty() {
                    nodes_for_v.push(build_upper(&e.sub, None));
                }
                if !nodes_for_v.is_empty() {
                    let mut kids_map: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
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
                    }
                }
                if let Some(e) = &root_empty {
                    accs.insert(e.clone());
                }

                if accs.len() <= 1 {
                    if let Some(the_acc) = accs.into_iter().next() {
                        let lower_tree = build_lower(d);
                        let lower_root = new_lower(lower_tree.children(), root_empty.is_some());
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
            if l.empty() {
                let mut stack = pref.clone();
                stack.reverse();
                out.push((stack, acc.clone()));
            }
            match l {
                Lower::Segment(seg) => {
                    // Push values top-to-bottom (reverse order since values[0]=deepest)
                    for v in seg.values.iter().rev() {
                        pref.push(v.clone());
                    }
                    dfs_lower(&seg.next, pref, acc, out);
                    for _ in 0..seg.values.len() {
                        pref.pop();
                    }
                }
                Lower::General { children, .. } => {
                    for (v, kids) in children.iter() {
                        for child in kids.values() {
                            pref.push(v.clone());
                            dfs_lower(child, pref, acc, out);
                            pref.pop();
                        }
                    }
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
                    if i.inner.empty() {
                        let mut stack = pref.clone();
                        stack.reverse();
                        out.push((stack, i.acc.clone()));
                    }
                    match &*i.inner {
                        Lower::Segment(seg) => {
                            for v in seg.values.iter().rev() {
                                pref.push(v.clone());
                            }
                            dfs_lower(&seg.next, pref, &i.acc, out);
                            for _ in 0..seg.values.len() {
                                pref.pop();
                            }
                        }
                        Lower::General { children, .. } => {
                            for (v, kids) in children.iter() {
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
                let new_lower_root = new_segment(
                    { let mut v = ArrayVec::new(); v.push(value); v },
                    i.inner.clone(),
                );
                new_interface(new_lower_root, i.acc.clone())
            }
            Upper::Branch(_) => {
                let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
                new_children.insert(
                    value,
                    CompactOrdMap::unit(self.inner.max_depth(), self.inner.clone()),
                );
                new_branch(new_children, None)
            }
        };
        LeveledGSS { inner: new_inner }
    }

    /// Equivalent to merging `self.isolate(Some(from)).push(to)` for each
    /// `(from, to)` pair, but avoids repeated isolate/push/merge churn by
    /// rebuilding the shifted top layer in one pass.
    pub fn shift_top_values<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let pairs: SmallVec<[(T, T); 8]> = shifts.into_iter().collect();
        if pairs.is_empty() {
            return Self::empty();
        }
        if pairs.len() == 1 {
            let (ref from, ref to) = pairs[0];
            if self.single_exclusive_top_value().as_ref() == Some(from) {
                return self.push(to.clone());
            }
        }

        match &*self.inner {
            Upper::Interface(i) => {
                // Use SmallVec instead of HashMap for grouping by target.
                // Linear scan is faster than HashMap for typical ≤8 shift pairs.
                let mut children_by_target: SmallVec<[(T, Children<T, Lower<T>>); 4]> = SmallVec::new();

                // Build a Segment-aware lookup closure
                let inner_children;
                let seg_entry: Option<(&T, CompactOrdMap<Arc<Lower<T>>>)>;
                match &*i.inner {
                    Lower::Segment(seg) => {
                        let top_val = seg.values.last().unwrap();
                        let rest = i.inner.segment_rest_arc();
                        inner_children = None;
                        seg_entry = Some((top_val, CompactOrdMap::unit(rest.max_depth(), rest)));
                    }
                    Lower::General { children, .. } => {
                        inner_children = Some(children);
                        seg_entry = None;
                    }
                }

                for (from, to) in pairs.iter().cloned() {
                    let kids_opt = if let Some((cv, ref ck)) = seg_entry {
                        if *cv == from { Some(ck) } else { None }
                    } else {
                        inner_children.unwrap().get(&from)
                    };
                    let Some(kids) = kids_opt else {
                        continue;
                    };
                    // Find or create the target entry
                    let target_children = if let Some(pos) = children_by_target.iter().position(|(t, _)| *t == to) {
                        &mut children_by_target[pos].1
                    } else {
                        children_by_target.push((to, CompactMap::new()));
                        &mut children_by_target.last_mut().unwrap().1
                    };
                    match target_children.get(&from) {
                        Some(existing_kids) => {
                            let mut merged_kids = existing_kids.clone();
                            for (depth, child) in kids.iter() {
                                if let Some(existing_child) = merged_kids.get(depth) {
                                    merged_kids.insert(*depth, merge_lower(existing_child, child));
                                } else {
                                    merged_kids.insert(*depth, child.clone());
                                }
                            }
                            target_children.insert(from, merged_kids);
                        }
                        None => {
                            target_children.insert(from, kids.clone());
                        }
                    }
                }

                if children_by_target.is_empty() {
                    return Self::empty();
                }

                let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
                for (to, lower_children) in children_by_target {
                    let lower = new_lower(lower_children, false);
                    shifted_children.insert(to, CompactOrdMap::unit(lower.max_depth(), lower));
                }

                let shifted_root = new_lower(shifted_children, false);
                LeveledGSS {
                    inner: new_interface(shifted_root, i.acc.clone()),
                }
            }
            Upper::Branch(_) => {
                let shifted = pairs
                    .into_iter()
                    .map(|(from, to)| self.isolate(Some(from)).push(to));
                Self::merge_many(shifted)
            }
        }
    }

    /// Like `shift_top_values` but takes ownership, allowing extraction of
    /// children by move instead of clone when the Arcs are uniquely owned.
    pub fn shift_top_values_owned<I>(self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let pairs: SmallVec<[(T, T); 8]> = shifts.into_iter().collect();
        if pairs.is_empty() {
            return Self::empty();
        }
        if pairs.len() == 1 {
            let (ref from, ref to) = pairs[0];
            if self.single_exclusive_top_value().as_ref() == Some(from) {
                return self.push(to.clone());
            }
        }

        // Try to extract children by move if we have unique ownership
        let (acc, mut children) = match Arc::try_unwrap(self.inner) {
            Ok(Upper::Interface(iface_arc)) => {
                match Arc::try_unwrap(iface_arc) {
                    Ok(Interface { inner: lower_arc, acc }) => {
                        match Arc::try_unwrap(lower_arc) {
                            Ok(lower) => {
                                let (c, _empty, _md) = lower.into_parts();
                                (acc, c)
                            }
                            Err(lower_arc) => {
                                // Can't unwrap lower, fall back to clone path
                                let i = Interface { inner: lower_arc, acc: acc.clone() };
                                let gss = LeveledGSS { inner: Arc::new(Upper::Interface(Arc::new(i))) };
                                return gss.shift_top_values(pairs);
                            }
                        }
                    }
                    Err(iface_arc) => {
                        let gss = LeveledGSS { inner: Arc::new(Upper::Interface(iface_arc)) };
                        return gss.shift_top_values(pairs);
                    }
                }
            }
            Ok(upper @ Upper::Branch(_)) => {
                let gss = LeveledGSS { inner: Arc::new(upper) };
                return gss.shift_top_values(pairs);
            }
            Err(arc) => {
                let gss = LeveledGSS { inner: arc };
                return gss.shift_top_values(pairs);
            }
        };

        // We own `children` by value — extract entries without cloning
        let mut children_by_target: SmallVec<[(T, Children<T, Lower<T>>); 4]> = SmallVec::new();

        for (from, to) in pairs {
            let Some(kids) = children.remove(&from) else {
                continue;
            };
            let target_children = if let Some(pos) = children_by_target.iter().position(|(t, _)| *t == to) {
                &mut children_by_target[pos].1
            } else {
                children_by_target.push((to, CompactMap::new()));
                &mut children_by_target.last_mut().unwrap().1
            };
            match target_children.get(&from) {
                Some(existing_kids) => {
                    let mut merged_kids = existing_kids.clone();
                    for (depth, child) in kids.iter() {
                        if let Some(existing_child) = merged_kids.get(depth) {
                            merged_kids.insert(*depth, merge_lower(existing_child, child));
                        } else {
                            merged_kids.insert(*depth, child.clone());
                        }
                    }
                    target_children.insert(from, merged_kids);
                }
                None => {
                    // Move kids directly — no clone needed
                    target_children.insert(from, kids);
                }
            }
        }

        if children_by_target.is_empty() {
            return Self::empty();
        }

        let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
        for (to, lower_children) in children_by_target {
            let lower = new_lower(lower_children, false);
            shifted_children.insert(to, CompactOrdMap::unit(lower.max_depth(), lower));
        }

        let shifted_root = new_lower(shifted_children, false);
        LeveledGSS {
            inner: new_interface(shifted_root, acc),
        }
    }

    /// Equivalent to `self.merge(&base.push(value))` but avoids intermediate
    /// allocations by directly inserting into self's structure via Arc::make_mut.
    /// Falls back to the standard path for non-Interface cases.
    pub fn absorb_push(self, value: T, base: &Self) -> Self {
        if base.is_empty() {
            return self;
        }
        if self.is_empty() {
            return base.push(value);
        }
        // Fast path: both are Interface with equal acc
        if let (Upper::Interface(base_iface), Upper::Interface(self_iface)) =
            (&*base.inner, &*self.inner)
        {
            if self_iface.acc == base_iface.acc {
                return self.absorb_push_interface_inplace(value, base_iface);
            }
        }
        // Fallback
        self.merge(&base.push(value))
    }

    /// Like `absorb_push` but assumes the caller has already verified that
    /// both `self` and `base` are Interface variants with identical `acc`
    /// values. This avoids an expensive O(n) annotation equality check.
    ///
    /// # Safety contract
    /// Caller must guarantee that both `self` and `base` have Interface
    /// inner variants with equal `acc` annotations.
    pub fn absorb_push_same_acc(self, value: T, base: &Self) -> Self {
        if base.is_empty() {
            return self;
        }
        if self.is_empty() {
            return base.push(value);
        }
        if let Upper::Interface(base_iface) = &*base.inner {
            return self.absorb_push_interface_inplace(value, base_iface);
        }
        // Fallback (shouldn't happen if caller's guarantee holds)
        self.merge(&base.push(value))
    }

    fn absorb_push_interface_inplace(
        mut self,
        value: T,
        base_iface: &Arc<Interface<T, A>>,
    ) -> Self {
        let child_depth = base_iface.inner.max_depth();
        let child_node = base_iface.inner.clone();

        let inner_mut = Arc::make_mut(&mut self.inner);
        if let Upper::Interface(self_iface_arc) = inner_mut {
            let iface_mut = Arc::make_mut(self_iface_arc);
            let lower_mut = Arc::make_mut(&mut iface_mut.inner);
            // Always convert to General for in-place mutation
            lower_mut.ensure_general();
            match lower_mut {
                Lower::General { children, max_depth, .. } => {
                    if let Some(existing_ordmap) = children.get_mut(&value) {
                        match existing_ordmap.get(&child_depth).cloned() {
                            Some(existing_child) => {
                                existing_ordmap.insert(child_depth, merge_lower(&existing_child, &child_node));
                            }
                            None => {
                                existing_ordmap.insert(child_depth, child_node);
                            }
                        }
                    } else {
                        children.insert(value, CompactOrdMap::unit(child_depth, child_node));
                    }

                    if child_depth + 1 > *max_depth {
                        *max_depth = child_depth + 1;
                    }
                }
                Lower::Segment(_) => unreachable!(),
            }

            return self;
        }
        // Fallback
        self.merge(&LeveledGSS { inner: Arc::new(Upper::Interface(base_iface.clone())) })
    }

    /// Combined `isolate(Some(value)).popn(n)` that avoids creating the
    /// intermediate isolated GSS. For Interface variants with a single
    /// interface path at `value`, walks directly down `n` levels without
    /// any intermediate allocations.
    pub fn isolate_popn(&self, value: T, n: isize) -> Self {
        if n <= 0 {
            return self.isolate(Some(value));
        }
        if self.is_empty() {
            return Self::empty();
        }
        // Fast path for Interface variant
        if let Upper::Interface(interface) = &*self.inner {
            // Get children for the isolated value — handle Segment directly
            let child_opt = if interface.inner.is_segment() {
                if interface.inner.segment_top_value() == &value {
                    Some(interface.inner.segment_rest_arc())
                } else {
                    None
                }
            } else if let Lower::General { children, .. } = &*interface.inner {
                children.get(&value).and_then(|kids| {
                    if kids.len() == 1 {
                        Some(kids.values().next().unwrap().clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            };
            if let Some(child) = child_opt {
                // Now walk down n-1 more levels (we already descended 1 level by isolating)
                let remaining = n - 1;
                if remaining <= 0 {
                    // We need to return the child wrapped as a GSS
                    // Handle empty flag on the original lower level
                    let mut result = child.clone();
                    if interface.inner.empty() {
                        // The isolated level had an empty marker — at depth 1 pop
                        // this means we include the empty terminal
                        if remaining == 0 {
                            let terminal = new_lower(CompactMap::new(), true);
                            result = merge_lower(&result, &terminal);
                        }
                    }
                    if result.children_is_empty() && !result.empty() {
                        return Self::empty();
                    }
                    return Self {
                        inner: new_interface(result, interface.acc.clone()),
                    };
                }
                // Walk down remaining levels using the child directly
                let temp = Self {
                    inner: new_interface(child.clone(), interface.acc.clone()),
                };
                if let Some(fast) = temp.popn_single_interface_path(remaining) {
                    return fast;
                }
            } else {
                return Self::empty();
            }
        }
        // Fallback to separate isolate + popn
        self.isolate(Some(value)).popn(n)
    }

    /// Combined isolate_popn + enumerate top values and their isolated bases.
    /// Returns (top_value, base_GSS) pairs where base_GSS = popped.isolate(Some(top_value)).
    /// This avoids creating the intermediate "popped" GSS for the common case.
    pub fn isolate_popn_bases(&self, value: T, n: isize) -> SmallVec<[(T, Self); 4]> {
        if self.is_empty() {
            return SmallVec::new();
        }
        // Fast path: walk down through single-child Interface levels
        // without creating intermediate GSSes.
        if let Upper::Interface(interface) = &*self.inner {
            // Get first child for this value — handle Segment directly
            let rest_storage;
            let first_child_opt: Option<&Arc<Lower<T>>> = if interface.inner.is_segment() {
                if interface.inner.segment_top_value() == &value {
                    rest_storage = interface.inner.segment_rest_arc();
                    Some(&rest_storage)
                } else {
                    None
                }
            } else if let Lower::General { children, .. } = &*interface.inner {
                children.get(&value).and_then(|kids| {
                    if kids.len() == 1 {
                        Some(kids.values().next().unwrap())
                    } else {
                        None // Multiple depth entries — fall through
                    }
                })
            } else {
                None
            };

            if let Some(first_child) = first_child_opt {

                    // Walk down (n-1) more levels through single-child chains
                    // without creating intermediate GSSes.
                    let mut current: &Lower<T> = &**first_child;
                    let mut remaining = n - 1;
                    while remaining > 0 {
                        if current.empty() {
                            break;
                        }
                        match current {
                            Lower::Segment(seg) => {
                                let seg_len = seg.values.len() as isize;
                                if remaining >= seg_len {
                                    current = &seg.next;
                                    remaining -= seg_len;
                                } else {
                                    break; // Would land inside segment — fall to slow path
                                }
                            }
                            Lower::General { children, .. } => {
                                if children.len() == 1 {
                                    let ordmap = children.values().next().unwrap();
                                    if ordmap.len() != 1 {
                                        break;
                                    }
                                    let (_, child) = ordmap.iter().next().unwrap();
                                    current = &**child;
                                    remaining -= 1;
                                } else {
                                    break;
                                }
                            }
                        }
                    }

                    if remaining == 0 {
                        // Successfully walked all n levels. `current` is the
                        // base's inner Lower node.
                        if current.children_len() == 0 {
                            return SmallVec::new();
                        }
                        let acc = &interface.acc;
                        if current.children_len() == 1 {
                            let goto_from = match current {
                                Lower::Segment(seg) => seg.values.last().unwrap().clone(),
                                Lower::General { children, .. } => {
                                    children.keys().next().unwrap().clone()
                                }
                            };
                            let base = Self {
                                inner: new_interface(Arc::new(current.clone()), acc.clone()),
                            };
                            return smallvec![(goto_from, base)];
                        }
                        // Multiple children at the base — split (must be General)
                        let mut result = SmallVec::new();
                        if let Lower::General { children, .. } = current {
                            for (k, ordmap) in children.iter() {
                                let filtered = CompactMap::unit(k.clone(), ordmap.clone());
                                let new_lower = new_lower(filtered, false);
                                let base = Self {
                                    inner: new_interface(new_lower, acc.clone()),
                                };
                                result.push((k.clone(), base));
                            }
                        }
                        return result;
                    }
                    // Chain walk couldn't reach the target depth (branching
                    // encountered). Fall through to general path.
            } else if !interface.inner.children_contains_key(&value) {
                return SmallVec::new();
            }
        }
        // General fallback: use isolate_popn then iterate
        let popped = self.isolate_popn(value, n);
        if popped.is_empty() {
            return SmallVec::new();
        }
        if let Some(v) = popped.single_top_value() {
            // Single top value: isolate is identity, base = popped
            return smallvec![(v, popped)];
        }
        let top_vals = popped.peek_values();
        let mut result = SmallVec::new();
        for v in top_vals {
            let base = popped.isolate(Some(v.clone()));
            result.push((v, base));
        }
        result
    }

    pub fn popn(&self, n: isize) -> Self {
        if n <= 0 {
            return self.clone();
        }
        if self.is_empty() {
            return self.clone();
        }
        if let Some(fast) = self.popn_single_interface_path(n) {
            return fast;
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
            let node_id = match &**node {
                Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
                _ => Arc::as_ptr(node) as usize,
            };
            let key = (node_id, k);
            if let Some(cached) = memo_lower.get(&key) {
                return cached.clone();
            }

            // Segment fast path: pop through the whole segment at once
            let merged: Option<Arc<Lower<T>>> = if node.is_segment() {
                let values = node.segment_values();
                let seg_len = values.len() as isize;
                if k >= seg_len {
                    // Pop past entire segment
                    let next_arc = node.segment_next().clone();
                    let popped = popn_lower::<T, A>(&next_arc, k - seg_len, memo_lower);
                    Some(popped)
                } else {
                    // Pop within segment: create shorter segment with remaining values
                    let keep = (seg_len - k) as usize;
                    let new_values: ArrayVec<T, SEGMENT_CAP> = values[..keep].iter().cloned().collect();
                    let next = node.segment_next();
                    Some(new_segment(new_values, next.clone()))
                }
            } else {
                let mut m: Option<Arc<Lower<T>>> = None;
                if let Lower::General { children, .. } = &**node {
                    for child in children.values().flat_map(|kids| kids.values()) {
                        let popped_child = popn_lower::<T, A>(child, k - 1, memo_lower);
                        m = Some(match m {
                            Some(acc) => merge_lower(&acc, &popped_child),
                            None => popped_child,
                        });
                    }
                }
                m
            };

            let mut res = merged.unwrap_or_else(|| new_lower(CompactMap::new(), false));

            if node.empty() && k == 1 {
                let terminal_node = new_lower(CompactMap::new(), true);
                res = merge_lower(&res, &terminal_node);
            }

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
                    let mut merged: Option<Arc<Upper<T, A>>> = None;
                    for kids in b.children.values() {
                        for child in kids.values() {
                            let popped_child = popn_upper(child, k - 1, memo_upper, memo_lower);
                            merged = Some(match merged {
                                Some(acc) => merge_upper(&acc, &popped_child),
                                None => popped_child,
                            });
                        }
                    }

                    if let Some(acc) = &b.empty {
                        if k == 1 {
                            let terminal_lower = new_lower(CompactMap::new(), true);
                            let terminal_upper = new_interface(terminal_lower, acc.clone());
                            merged = Some(match merged {
                                Some(current) => merge_upper(&current, &terminal_upper),
                                None => terminal_upper,
                            });
                        }
                    }

                    if let Some(merged) = merged {
                        try_promote(&merged)
                    } else {
                        empty_upper_inner()
                    }
                }
                Upper::Interface(i) => {
                    let popped_lower = popn_lower::<T, A>(&i.inner, k, memo_lower);
                    if popped_lower.children_is_empty() && !popped_lower.empty() {
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

    fn popn_single_interface_path(&self, n: isize) -> Option<Self> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };

        let mut current = interface.inner.clone();
        let mut remaining = n;
        while remaining > 0 {
            let next_child: Option<Arc<Lower<T>>> = if current.is_segment() {
                let values = current.segment_values();
                if values.len() as isize <= remaining {
                    // Decrement by (len-1) here; the loop bottom adds the final -1
                    remaining -= values.len() as isize - 1;
                    Some(current.segment_next().clone())
                } else {
                    // Would land inside segment — can't use this fast path
                    return None;
                }
            } else {
                match current.children_len() {
                    0 => None,
                    1 => {
                        if let Lower::General { children, .. } = &*current {
                            let kids = children.values().next().expect("single child entry");
                            if kids.len() != 1 {
                                return None;
                            }
                            Some(kids.values().next().expect("single child node").clone())
                        } else {
                            None
                        }
                    }
                    _ => return None,
                }
            };

            if remaining == 1 {
                let mut result = next_child
                    .unwrap_or_else(|| new_lower(CompactMap::new(), false));
                if current.empty() {
                    let terminal = new_lower(CompactMap::new(), true);
                    result = merge_lower(&result, &terminal);
                }

                if result.children_is_empty() && !result.empty() {
                    return Some(Self::empty());
                }

                return Some(Self {
                    inner: new_interface(result, interface.acc.clone()),
                });
            }

            let Some(child) = next_child else {
                return Some(Self::empty());
            };
            current = child;
            remaining -= 1;
        }

        if current.children_is_empty() && !current.empty() {
            Some(Self::empty())
        } else {
            Some(Self {
                inner: new_interface(current, interface.acc.clone()),
            })
        }
    }

    #[cfg(test)]
    pub fn popn_with_underflow(&self, n: isize) -> (Self, StdHashMap<usize, A>) {
        if n <= 0 {
            return (self.clone(), StdHashMap::new());
        }
        if self.is_empty() {
            return (self.clone(), StdHashMap::new());
        }

        let mut underflows: StdHashMap<usize, A> = StdHashMap::new();

        fn merge_underflow<A: Merge + Clone>(map: &mut StdHashMap<usize, A>, shortfall: usize, acc: A) {
            map.entry(shortfall)
                .and_modify(|existing| *existing = existing.merge(&acc))
                .or_insert(acc);
        }

        fn popn_lower_uf<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Lower<T>>,
            k: isize,
            underflows: &mut StdHashMap<usize, A>,
            acc: &A,
            memo_lower: &mut StdHashMap<(usize, isize), Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if k == 0 {
                return node.clone();
            }
            let node_id = match &**node {
                Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
                _ => Arc::as_ptr(node) as usize,
            };
            let key = (node_id, k);
            if let Some(cached) = memo_lower.get(&key) {
                return cached.clone();
            }

            // Segment fast path: walk through the segment values directly
            let res = if let Lower::Segment(seg) = &**node {
                let seg_len = seg.values.len() as isize;
                if k >= seg_len {
                    // Pop past entire segment
                    let next_arc = seg.next.clone();
                    popn_lower_uf::<T, A>(&next_arc, k - seg_len, underflows, acc, memo_lower)
                } else {
                    // Pop within segment: create shorter segment with remaining values
                    let keep = (seg_len - k) as usize;
                    let new_values: ArrayVec<T, SEGMENT_CAP> = seg.values[..keep].iter().cloned().collect();
                    new_segment(new_values, seg.next.clone())
                }
            } else {
                let all_children: Vec<_> = node
                    .children()
                    .values()
                    .flat_map(|kids| kids.values())
                    .cloned()
                    .collect();

                if all_children.is_empty() {
                    new_lower(CompactMap::new(), false)
                } else {
                    let popped_children: Vec<_> = all_children
                        .into_iter()
                        .map(|child| popn_lower_uf::<T, A>(&child, k - 1, underflows, acc, memo_lower))
                        .collect();

                    let mut it = popped_children.into_iter();
                    let first = it.next().unwrap();
                    it.fold(first, |acc, next| merge_lower(&acc, &next))
                }
            };

            if node.empty() && k >= 1 {
                
                merge_underflow(underflows, k as usize, acc.clone());
            }

            memo_lower.insert(key, res.clone());
            res
        }

        fn popn_upper_uf<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            k: isize,
            underflows: &mut StdHashMap<usize, A>,
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
                    let mut popped = Vec::new();
                    for kids in b.children.values() {
                        for child in kids.values() {
                            popped.push(popn_upper_uf(child, k - 1, underflows, memo_upper, memo_lower));
                        }
                    }

                    if let Some(acc) = &b.empty {
                        
                        merge_underflow(underflows, k as usize, acc.clone());
                    }

                    if popped.is_empty() {
                        empty_upper_inner()
                    } else {
                        let mut it = popped.into_iter();
                        let first = it.next().unwrap();
                        let merged = it.fold(first, |acc, next| merge_upper(&acc, &next));
                        try_promote(&merged)
                    }
                }
                Upper::Interface(i) => {
                    let popped_lower = popn_lower_uf::<T, A>(&i.inner, k, underflows, &i.acc, memo_lower);
                    if popped_lower.children_is_empty() && !popped_lower.empty() {
                        empty_upper_inner()
                    } else {
                        new_interface(popped_lower, i.acc.clone())
                    }
                }
            };

            memo_upper.insert(key, res.clone());
            res
        }

        let mut memo_upper: StdHashMap<(usize, isize), Arc<Upper<T, A>>> = StdHashMap::new();
        let mut memo_lower: StdHashMap<(usize, isize), Arc<Lower<T>>> = StdHashMap::new();
        let new_inner = popn_upper_uf::<T, A>(&self.inner, n, &mut underflows, &mut memo_upper, &mut memo_lower);
        (LeveledGSS { inner: new_inner }, underflows)
    }

    pub fn pop(&self) -> Self {
        self.popn(1)
    }

    /// Decompose the top level and pop one level in a single pass.
    /// Returns `(value, popped_gss)` for each top-level child value.
    /// Equivalent to calling `self.isolate(Some(v)).pop()` for each v in `peek_values()`,
    /// but avoids repeated HashMap lookups.
    pub fn decompose_and_pop(&self) -> Vec<(T, Self)> {
        match &*self.inner {
            Upper::Branch(b) => {
                let mut result = Vec::with_capacity(b.children.len());
                for (val, kids) in b.children.iter() {
                    // Fast path: single child needs no merge.
                    let m = if kids.len() == 1 {
                        kids.values().next().unwrap().clone()
                    } else {
                        let mut it = kids.values();
                        let mut acc = it.next().unwrap().clone();
                        for child in it {
                            acc = merge_upper(&acc, child);
                        }
                        acc
                    };
                    // Skip try_promote if already Interface (try_promote is a no-op for Interface).
                    let inner = if matches!(&*m, Upper::Interface(_)) {
                        m
                    } else {
                        try_promote(&m)
                    };
                    let is_empty = matches!(&*inner,
                        Upper::Branch(b) if b.children.is_empty() && b.empty.is_none());
                    if !is_empty {
                        result.push((val.clone(), LeveledGSS { inner }));
                    }
                }
                result
            }
            Upper::Interface(i) => {
                // Segment fast path: single child, no iteration needed
                if i.inner.is_segment() {
                    let val = i.inner.segment_top_value().clone();
                    let lower = i.inner.segment_rest_arc();
                    if !lower.children_is_empty() || lower.empty() {
                        let upper = new_interface(lower, i.acc.clone());
                        return vec![(val, LeveledGSS { inner: upper })];
                    }
                    return vec![];
                }
                let mut result = Vec::with_capacity(i.inner.children_len());
                if let Lower::General { children, .. } = &*i.inner {
                    for (val, kids) in children.iter() {
                        // Fast path: single child needs no merge.
                        let lower = if kids.len() == 1 {
                            kids.values().next().unwrap().clone()
                        } else {
                            let mut it = kids.values();
                            let mut acc = it.next().unwrap().clone();
                            for child in it {
                                acc = merge_lower(&acc, child);
                            }
                            acc
                        };
                        if !lower.children_is_empty() || lower.empty() {
                            let upper = new_interface(lower, i.acc.clone());
                            result.push((val.clone(), LeveledGSS { inner: upper }));
                        }
                    }
                }
                result
            }
        }
    }

    /// Like `decompose_and_pop` but invokes a callback for each (value, popped_gss) pair
    /// instead of allocating a Vec. Avoids heap allocation for the common single-element case.
    pub fn for_each_decomposed(&self, mut f: impl FnMut(T, Self)) {
        match &*self.inner {
            Upper::Branch(b) => {
                for (val, kids) in b.children.iter() {
                    let m = if kids.len() == 1 {
                        kids.values().next().unwrap().clone()
                    } else {
                        let mut it = kids.values();
                        let mut acc = it.next().unwrap().clone();
                        for child in it {
                            acc = merge_upper(&acc, child);
                        }
                        acc
                    };
                    let inner = if matches!(&*m, Upper::Interface(_)) {
                        m
                    } else {
                        try_promote(&m)
                    };
                    let is_empty = matches!(&*inner,
                        Upper::Branch(b) if b.children.is_empty() && b.empty.is_none());
                    if !is_empty {
                        f(val.clone(), LeveledGSS { inner });
                    }
                }
            }
            Upper::Interface(i) => {
                if i.inner.is_segment() {
                    let val = i.inner.segment_top_value().clone();
                    let lower = i.inner.segment_rest_arc();
                    if !lower.children_is_empty() || lower.empty() {
                        let upper = new_interface(lower, i.acc.clone());
                        f(val, LeveledGSS { inner: upper });
                    }
                    return;
                }
                if let Lower::General { children, .. } = &*i.inner {
                    for (val, kids) in children.iter() {
                        let lower = if kids.len() == 1 {
                            kids.values().next().unwrap().clone()
                        } else {
                            let mut it = kids.values();
                            let mut acc = it.next().unwrap().clone();
                            for child in it {
                                acc = merge_lower(&acc, child);
                            }
                            acc
                        };
                        if !lower.children_is_empty() || lower.empty() {
                            let upper = new_interface(lower, i.acc.clone());
                            f(val.clone(), LeveledGSS { inner: upper });
                        }
                    }
                }
            }
        }
    }

    /// Extract the top chain of states plus the accumulator and tail.
    ///
    /// Returns `(chain_states_top_first, acc, tail_lower)` where:
    /// - `chain_states_top_first`: parser states from the top of the chain downward
    /// - `acc`: reference to the accumulator at the Interface
    /// - `tail_lower`: the Lower node at the bottom of the chain
    ///
    /// Returns `None` if the GSS is not an Interface or has no chain.
    /// The chain must have at least 2 states for this to be worthwhile.
    pub fn extract_chain_and_tail(&self) -> Option<(SmallVec<[T; 16]>, &A, ChainTail<T>)> {
        let interface = match &*self.inner {
            Upper::Interface(iface) => iface,
            _ => return None,
        };

        let mut states = SmallVec::<[T; 16]>::new();
        let mut current: &Lower<T> = &*interface.inner;

        while let Some((next, _)) = current.chain_step() {
            current.append_chain_values_top_first(&mut states);
            current = next;
        }

        if states.len() < 2 {
            return None;
        }

        Some((states, &interface.acc, ChainTail { inner: Arc::new(current.clone()) }))
    }

    /// Try to view the top of this GSS as a flat virtual stack.
    /// Succeeds when the GSS is an Interface whose top is a chain of Segment nodes.
    /// The chain is extracted until a non-Segment node is hit — that node becomes
    /// the "floor". The floor can be a General with splits, an empty terminal, etc.
    ///
    /// Returns `None` if the GSS is not an Interface whose top node is a Segment.
    pub fn try_virtual_stack(&self) -> Option<VirtualStack<T, A>> {
        let interface = match &*self.inner {
            Upper::Interface(iface) => iface,
            _ => return None,
        };
        let (values, next) = match &*interface.inner {
            Lower::Segment(seg) => (seg.values.clone(), seg.next.clone()),
            _ => return None,
        };
        Some(VirtualStack { values, next, acc: interface.acc.clone() })
    }

    pub fn is_empty(&self) -> bool {
        match &*self.inner {
            Upper::Branch(b) => b.children.is_empty() && b.empty.is_none(),
            Upper::Interface(_) => false,
        }
    }

    pub fn max_depth(&self) -> u32 {
        self.inner.max_depth()
    }

    pub fn summary(&self) -> LeveledGSSSummary {
        let mut visited_upperbranch: HashSet<usize> = HashSet::new();
        let mut visited_interface: HashSet<usize> = HashSet::new();
        let mut visited_lower: HashSet<usize> = HashSet::new();

        let mut upperbranch_nodes = 0usize;
        let mut interface_nodes = 0usize;
        let mut lower_nodes = 0usize;
        let mut total_edges = 0usize;
        let mut accumulator_instances = 0usize;

        let mut upper_queue: VecDeque<Arc<Upper<T, A>>> = VecDeque::new();
        upper_queue.push_back(self.inner.clone());
        let mut lower_queue: VecDeque<Arc<Lower<T>>> = VecDeque::new();

        while let Some(node) = upper_queue.pop_front() {
            match &*node {
                Upper::Branch(branch) => {
                    let node_id = Arc::as_ptr(branch) as usize;
                    if !visited_upperbranch.insert(node_id) {
                        continue;
                    }
                    upperbranch_nodes += 1;
                    if branch.empty.is_some() {
                        accumulator_instances += 1;
                    }
                    for children in branch.children.values() {
                        total_edges += children.len();
                        for child in children.values() {
                            upper_queue.push_back(child.clone());
                        }
                    }
                }
                Upper::Interface(interface) => {
                    let node_id = Arc::as_ptr(interface) as usize;
                    if !visited_interface.insert(node_id) {
                        continue;
                    }
                    interface_nodes += 1;
                    accumulator_instances += 1;
                    total_edges += 1;
                    lower_queue.push_back(interface.inner.clone());
                }
            }
        }

        while let Some(node) = lower_queue.pop_front() {
            let node_id = lower_node_id(&node);
            if !visited_lower.insert(node_id) {
                continue;
            }
            lower_nodes += 1;
            // Walk through this node and any owned segment chain below it.
            let mut current: &Lower<T> = &*node;
            loop {
                match current {
                    Lower::Segment(seg) => {
                        total_edges += seg.values.len();
                        match &*seg.next {
                            Lower::Segment(inner_seg) => {
                                let inner_id = Arc::as_ptr(inner_seg) as usize;
                                if !visited_lower.insert(inner_id) { break; }
                                lower_nodes += 1;
                                current = &*seg.next;
                            }
                            Lower::General { children, .. } => {
                                lower_nodes += 1;
                                for kids in children.values() {
                                    total_edges += kids.len();
                                    for child in kids.values() {
                                        lower_queue.push_back(child.clone());
                                    }
                                }
                                break;
                            }
                        }
                    }
                    Lower::General { children, .. } => {
                        for kids in children.values() {
                            total_edges += kids.len();
                            for child in kids.values() {
                                lower_queue.push_back(child.clone());
                            }
                        }
                        break;
                    }
                }
            }
        }

        LeveledGSSSummary {
            top_values_count: self.inner.children_keys().len(),
            upperbranch_nodes,
            interface_nodes,
            lower_nodes,
            total_unique_nodes: upperbranch_nodes + interface_nodes + lower_nodes,
            total_edges,
            accumulator_instances,
            max_depth: self.max_depth(),
        }
    }

    pub fn isolate(&self, value: Option<T>) -> Self {
        
        if let Some(ref v) = value {
            match &*self.inner {
                Upper::Branch(b) => {
                    if b.empty.is_none() && b.children.len() == 1 && b.children.contains_key(v) {
                        return self.clone();
                    }
                }
                Upper::Interface(i) => {
                    if !i.inner.empty() && i.inner.children_len() == 1 && i.inner.children_contains_key(v) {
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
                    if i.inner.children_is_empty() && i.inner.empty() {
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
                        .map(|kids| CompactMap::unit(val.clone(), kids.clone()))
                        .unwrap_or_else(CompactMap::new);
                    let max_depth = b
                        .children
                        .get(&val)
                        .and_then(|kids| kids.get_max().map(|(depth, _)| *depth + 1))
                        .unwrap_or(0);
                    let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: filtered_children,
                        empty: None,
                        max_depth,
                    })));
                    try_promote(&new_b)
                }
                Upper::Interface(i) => {
                    // Fast path for Segment: if isolating the top value, reconstruct
                    if i.inner.is_segment() {
                        if i.inner.segment_top_value() == &val {
                            let rest = i.inner.segment_rest_arc();
                            let new_lower_root = new_lower(
                                CompactMap::unit(val.clone(), CompactOrdMap::unit(rest.max_depth(), rest)),
                                false,
                            );
                            new_interface(new_lower_root, i.acc.clone())
                        } else {
                            empty_upper_inner()
                        }
                    } else if let Lower::General { children, .. } = &*i.inner {
                        if let Some(kids) = children.get(&val) {
                            let filtered_children = CompactMap::unit(val.clone(), kids.clone());
                            let new_lower_root = new_lower(filtered_children, false);
                            new_interface(new_lower_root, i.acc.clone())
                        } else {
                            empty_upper_inner()
                        }
                    } else {
                        empty_upper_inner()
                    }
                }
            }
        } else {
            let empty_acc = match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => {
                    if i.inner.empty() {
                        Some(i.acc.clone())
                    } else {
                        None
                    }
                }
            };
            let new_b = new_branch(CompactMap::new(), empty_acc);
            try_promote(&new_b)
        };
        LeveledGSS { inner: new_inner }
    }

    #[cfg(test)]
    pub fn isolate_many<I: IntoIterator<Item = Option<T>>>(&self, values: I) -> Self {
        let values_set: HashSet<Option<T>> = values.into_iter().collect();

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
                    .children()
                    .keys()
                    .all(|k| values_set.contains(&Some(k.clone())));
                let empty_kept_ok = i.inner.empty() == values_set.contains(&None);
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
                let mut filtered_children: Children<T, Upper<T, A>> = CompactMap::new();
                for (v, kids) in b.children.iter() {
                    if values_set.contains(&Some(v.clone())) {
                        filtered_children.insert(v.clone(), kids.clone());
                    }
                }
                let new_b = new_branch(filtered_children, new_empty);
                try_promote(&new_b)
            }
            Upper::Interface(i) => {
                let keep_empty = values_set.contains(&None) && i.inner.empty();
                let mut filtered_children: Children<T, Lower<T>> = CompactMap::new();
                for (v, kids) in i.inner.children().iter() {
                    if values_set.contains(&Some(v.clone())) {
                        filtered_children.insert(v.clone(), kids.clone());
                    }
                }
                if !filtered_children.is_empty() || keep_empty {
                    let new_lower_root = new_lower(filtered_children, keep_empty);
                    new_interface(new_lower_root, i.acc.clone())
                } else {
                    new_branch(CompactMap::new(), None)
                }
            }
        };

        LeveledGSS { inner: new_inner }
    }

    #[cfg(test)]
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
                    let mut new_children: Children<T, Upper<T, B>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: CompactOrdMap<Arc<Upper<T, B>>> = CompactOrdMap::new();
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

    #[cfg(test)]
    pub fn prune<P>(&self, mut predicate: P) -> Self
    where
        P: FnMut(&A) -> bool,
    {
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

                    let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
                    let mut children_identical = true;
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
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
        // Fast path: single Interface at root — no memo or tree traversal needed.
        if let Upper::Interface(i) = &*self.inner {
            return match mutator(&i.acc) {
                Some(new_acc) => LeveledGSS { inner: new_interface(i.inner.clone(), new_acc) },
                None => LeveledGSS::empty(),
            };
        }

        // Use a flat Vec for memo instead of HashMap — avoids hashing cost
        // for the typical case of 2-4 unique accumulators.
        let mut acc_memo: Vec<(A, Option<B>)> = Vec::with_capacity(4);

        fn mutate_acc<A, B, M>(
            a: &A,
            memo: &mut Vec<(A, Option<B>)>,
            m: &mut M,
        ) -> Option<B>
        where
            A: Clone + Eq,
            B: Clone,
            M: FnMut(&A) -> Option<B>,
        {
            for (k, v) in memo.iter() {
                if k == a {
                    return v.clone();
                }
            }
            let r = m(a);
            memo.push((a.clone(), r.clone()));
            r
        }

        fn transform<T, A, B, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut Vec<(A, Option<B>)>,
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
                    let mut new_children: Children<T, Upper<T, B>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: CompactOrdMap<Arc<Upper<T, B>>> = CompactOrdMap::new();
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

    /// Like a cross-type no-promote transform followed by decompose_and_pop, but avoids
    /// building the root-level Branch node. Returns (value, sub_gss) pairs directly,
    /// plus a Vec of "root accumulators" (transformed empty values at the root Branch)
    /// to be checked for final_weight separately.
    pub fn apply_transform_and_decompose<B, M>(
        &self,
        mut mutator: M,
    ) -> (Vec<(T, LeveledGSS<T, B>)>, Vec<B>)
    where
        B: Merge + Clone + Eq + Hash,
        M: FnMut(&A) -> Option<B>,
    {
        let mut acc_memo: Vec<(A, Option<B>)> = Vec::with_capacity(4);

        fn mutate_acc_td<A, B, M>(
            a: &A,
            memo: &mut Vec<(A, Option<B>)>,
            m: &mut M,
        ) -> Option<B>
        where
            A: Clone + Eq,
            B: Clone,
            M: FnMut(&A) -> Option<B>,
        {
            for (k, v) in memo.iter() {
                if k == a {
                    return v.clone();
                }
            }
            let r = m(a);
            memo.push((a.clone(), r.clone()));
            r
        }

        fn transform_td<T, A, B, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut Vec<(A, Option<B>)>,
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
                    mutate_acc_td(&i.acc, memo, m)
                        .map(|new_acc| new_interface(i.inner.clone(), new_acc))
                }
                Upper::Branch(b) => {
                    let new_empty_opt = b.empty.as_ref().and_then(|e| mutate_acc_td(e, memo, m));
                    // Fast path: single child entry with single child.
                    if b.children.len() == 1 && new_empty_opt.is_none() {
                        let (v, kids) = b.children.iter().next().unwrap();
                        if kids.len() == 1 {
                            let child = kids.values().next().unwrap();
                            if let Some(nc) = transform_td::<T, A, B, M>(child, memo, m) {
                                let new_kids = CompactOrdMap::unit(nc.max_depth(), nc);
                                let new_children = CompactMap::unit(v.clone(), new_kids);
                                return Some(new_branch(new_children, None));
                            } else {
                                return None;
                            }
                        }
                    }
                    let mut new_children: Children<T, Upper<T, B>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let new_kids: CompactOrdMap<Arc<Upper<T, B>>> = kids.values()
                            .filter_map(|child| transform_td::<T, A, B, M>(child, memo, m))
                            .map(|nc| (nc.max_depth(), nc))
                            .collect();
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }
                    if new_children.is_empty() && new_empty_opt.is_none() {
                        None
                    } else {
                        Some(new_branch(new_children, new_empty_opt))
                    }
                }
            }
        }

        match &*self.inner {
            Upper::Interface(i) => {
                // Interface root: transform acc, then decompose inner Lower's children.
                let new_acc = match mutate_acc_td(&i.acc, &mut acc_memo, &mut mutator) {
                    Some(a) => a,
                    None => return (Vec::new(), Vec::new()),
                };
                let mut result = Vec::with_capacity(i.inner.children_len());
                match &*i.inner {
                    Lower::Segment(seg) => {
                        let value = seg.values.last().unwrap();
                        let rest = i.inner.segment_rest_arc();
                        if !rest.children_is_empty() || rest.empty() {
                            let upper = new_interface(rest, new_acc.clone());
                            result.push((value.clone(), LeveledGSS { inner: upper }));
                        }
                    }
                    Lower::General { children, .. } => {
                        for (val, kids) in children.iter() {
                            let lower = if kids.len() == 1 {
                                kids.values().next().unwrap().clone()
                            } else {
                                let mut it = kids.values();
                                let mut acc = it.next().unwrap().clone();
                                for child in it {
                                    acc = merge_lower(&acc, child);
                                }
                                acc
                            };
                            if !lower.children_is_empty() || lower.empty() {
                                let upper = new_interface(lower, new_acc.clone());
                                result.push((val.clone(), LeveledGSS { inner: upper }));
                            }
                        }
                    }
                }
                (result, Vec::new())
            }
            Upper::Branch(b) => {
                // Branch root: transform each child subtree, decompose into (value, sub_gss) pairs.
                let root_accs: Vec<B> = b.empty.iter()
                    .filter_map(|e| mutate_acc_td(e, &mut acc_memo, &mut mutator))
                    .collect();
                let mut result = Vec::with_capacity(b.children.len());
                for (val, kids) in b.children.iter() {
                    // Transform each child, collect into new_kids.
                    let mut new_kids: Vec<Arc<Upper<T, B>>> = Vec::new();
                    for child in kids.values() {
                        if let Some(nc) = transform_td::<T, A, B, M>(child, &mut acc_memo, &mut mutator) {
                            new_kids.push(nc);
                        }
                    }
                    if new_kids.is_empty() {
                        continue;
                    }
                    // Merge children (like decompose_and_pop does).
                    let merged = if new_kids.len() == 1 {
                        new_kids.into_iter().next().unwrap()
                    } else {
                        let mut it = new_kids.into_iter();
                        let mut acc = it.next().unwrap();
                        for child in it {
                            acc = merge_upper(&acc, &child);
                        }
                        acc
                    };
                    let is_empty = matches!(&*merged,
                        Upper::Branch(b) if b.children.is_empty() && b.empty.is_none());
                    if !is_empty {
                        result.push((val.clone(), LeveledGSS { inner: merged }));
                    }
                }
                (result, root_accs)
            }
        }
    }

    /// Like apply_and_prune but skips try_promote. Use when the tree is already
    /// canonical and the transformation preserves structure (e.g., DenseMaskAcc → DenseMaskAcc).
    pub fn apply_and_prune_no_promote(&self, mut mutator: impl FnMut(&A) -> Option<A>) -> Self {
        // Fast path: single Interface at root.
        if let Upper::Interface(i) = &*self.inner {
            return match mutator(&i.acc) {
                Some(new_acc) => LeveledGSS { inner: new_interface(i.inner.clone(), new_acc) },
                None => LeveledGSS::empty(),
            };
        }

        let mut acc_memo: Vec<(A, Option<A>)> = Vec::with_capacity(4);

        fn mutate_acc_np<A, M>(
            a: &A,
            memo: &mut Vec<(A, Option<A>)>,
            m: &mut M,
        ) -> Option<A>
        where
            A: Clone + Eq,
            M: FnMut(&A) -> Option<A>,
        {
            for (k, v) in memo.iter() {
                if k == a {
                    return v.clone();
                }
            }
            let r = m(a);
            memo.push((a.clone(), r.clone()));
            r
        }

        fn transform_np<T, A, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut Vec<(A, Option<A>)>,
            m: &mut M,
        ) -> Option<Arc<Upper<T, A>>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            M: FnMut(&A) -> Option<A>,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc_opt = mutate_acc_np(&i.acc, memo, m);
                    new_acc_opt.map(|new_acc| new_interface(i.inner.clone(), new_acc))
                }
                Upper::Branch(b) => {
                    let new_empty_opt = b.empty.as_ref().and_then(|e| mutate_acc_np(e, memo, m));
                    // Fast path: single child entry with single child.
                    if b.children.len() == 1 && new_empty_opt.is_none() {
                        let (v, kids) = b.children.iter().next().unwrap();
                        if kids.len() == 1 {
                            let child = kids.values().next().unwrap();
                            if let Some(nc) = transform_np::<T, A, M>(child, memo, m) {
                                let new_kids = CompactOrdMap::unit(nc.max_depth(), nc);
                                let new_children = CompactMap::unit(v.clone(), new_kids);
                                return Some(new_branch(new_children, None));
                            } else {
                                return None;
                            }
                        }
                    }
                    let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let new_kids: CompactOrdMap<Arc<Upper<T, A>>> = kids.values()
                            .filter_map(|child| transform_np::<T, A, M>(child, memo, m))
                            .map(|nc| (nc.max_depth(), nc))
                            .collect();
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }
                    if new_children.is_empty() && new_empty_opt.is_none() {
                        None
                    } else {
                        Some(new_branch(new_children, new_empty_opt))
                    }
                }
            }
        }

        let res_inner_opt = transform_np::<T, A, _>(&self.inner, &mut acc_memo, &mut mutator);
        res_inner_opt.map_or_else(Self::empty, |inner| Self { inner })
    }

    pub fn merge(&self, other: &Self) -> Self {
        let merged_inner = merge_upper(&self.inner, &other.inner);
        LeveledGSS {
            inner: merged_inner,
        }
    }

    pub fn merge_many(gsses: impl IntoIterator<Item = Self>) -> Self {
        let mut items: Vec<Self> = gsses.into_iter().collect();
        if items.is_empty() {
            return LeveledGSS::empty();
        }
        if items.len() == 1 {
            return items.into_iter().next().unwrap();
        }
        while items.len() > 1 {
            let mut next = Vec::with_capacity((items.len() + 1) / 2);
            let mut iter = items.into_iter();
            while let Some(a) = iter.next() {
                if let Some(b) = iter.next() {
                    next.push(a.merge(&b));
                } else {
                    next.push(a);
                }
            }
            items = next;
        }
        items.into_iter().next().unwrap()
    }

    pub fn fuse(&self, levels: Option<isize>) -> Self {
        if let Some(l) = levels {
            if l <= 0 {
                return self.clone();
            }
        }

        // Fast path for fuse(Some(1)): children see remain=0 → identity.
        // So fuse is a no-op iff the top node has no multi-depth slots.
        if levels == Some(1) {
            let no_multi_depth = match &*self.inner {
                Upper::Interface(i) => {
                    match &*i.inner {
                        Lower::Segment(_) => true,
                        Lower::General { children, .. } => !children.values().any(|kids| kids.len() > 1),
                    }
                }
                Upper::Branch(b) => {
                    !b.children.values().any(|kids| kids.len() > 1)
                }
            };
            if no_multi_depth {
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
            let key = (lower_node_id(node), remain);
            if let Some(cached) = memo.get(&key) {
                return cached.clone();
            }

            let next_remain = remain.map(|r| r - 1);

            // Fast path for Segment: no multi-depth slots, recurse on next only
            if let Lower::Segment(seg) = &**node {
                let next_remain_seg = remain.map(|r| r - seg.values.len() as isize);
                let next_arc = seg.next.clone();
                let fused_next = fuse_lower::<T, A>(&next_arc, next_remain_seg, memo);
                if Arc::ptr_eq(&fused_next, &next_arc) {
                    memo.insert(key, node.clone());
                    return node.clone();
                }
                let res = new_segment(seg.values.clone(), fused_next.clone());
                memo.insert(key, res.clone());
                return res;
            }

            // General path
            let Lower::General { children, .. } = &**node else { unreachable!() };
            let has_multi_depth_slots = children.values().any(|kids| kids.len() > 1);

            let mut new_children_by_value: StdHashMap<T, Vec<Arc<Lower<T>>>> = StdHashMap::new();
            let mut children_changed = false;

            for (v, kids) in children.iter() {
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

            let mut final_children: Children<T, Lower<T>> = CompactMap::new();
            for (v, fused_kids) in new_children_by_value {
                if fused_kids.is_empty() {
                    continue;
                }
                let mut it = fused_kids.into_iter();
                let first = it.next().unwrap();
                let merged_child = it.fold(first, |acc, next| merge_lower(&acc, &next));
                final_children.insert(v, CompactOrdMap::unit(merged_child.max_depth(), merged_child));
            }

            let res = new_lower(final_children, node.empty());
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
                    let has_multi_depth_slots = match &*i.inner {
                        Lower::Segment(_) => false,
                        Lower::General { children, .. } => children.values().any(|kids| kids.len() > 1),
                    };
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

                    let mut final_children: Children<T, Upper<T, A>> = CompactMap::new();
                    for (v, fused_kids) in new_children_by_value {
                        if fused_kids.is_empty() {
                            continue;
                        }
                        let mut it = fused_kids.into_iter();
                        let first = it.next().unwrap();
                        let merged_child = it.fold(first, |acc, next| merge_upper(&acc, &next));
                        final_children
                            .insert(v, CompactOrdMap::unit(merged_child.max_depth(), merged_child));
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

    #[cfg(test)]
    pub fn normalize(&self) -> Self {
        
        let fused = self.fuse(None);

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

        LeveledGSS { inner }
    }

    pub fn peek(&self) -> HashSet<T> {
        self.inner.children_keys().into_iter().collect()
    }

    pub fn peek_values(&self) -> SmallVec<[T; 8]> {
        self.inner.children_keys()
    }

    /// Iterate over top values without allocating a Vec. 
    /// Calls `f` for each top-level value in the GSS.
    pub fn for_each_top_value<F: FnMut(T)>(&self, mut f: F) {
        match &*self.inner {
            Upper::Branch(branch) => {
                for k in branch.children.keys() {
                    f(k.clone());
                }
            }
            Upper::Interface(interface) => {
                match &*interface.inner {
                    Lower::Segment(seg) => f(seg.values.last().unwrap().clone()),
                    Lower::General { children, .. } => {
                        for k in children.keys() {
                            f(k.clone());
                        }
                    }
                }
            }
        }
    }

    pub fn single_top_value(&self) -> Option<T> {
        self.inner.single_child_key()
    }

    pub fn single_exclusive_top_value(&self) -> Option<T> {
        self.inner.single_child_key_without_empty()
    }

    pub fn path_count_at_most(&self, limit: usize) -> usize {
        if limit == 0 || self.is_empty() {
            return 0;
        }

        fn capped_add(acc: usize, value: usize, limit: usize) -> usize {
            acc.saturating_add(value).min(limit)
        }

        fn count_lower<T>(
            node: &Arc<Lower<T>>,
            limit: usize,
            memo: &mut StdHashMap<usize, usize>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
        {
            let ptr = lower_node_id(node);
            if let Some(&cached) = memo.get(&ptr) {
                return cached;
            }
            let count = count_lower_inner(&**node, limit, memo);
            memo.insert(ptr, count);
            count
        }

        fn count_lower_inner<T>(
            node: &Lower<T>,
            limit: usize,
            memo: &mut StdHashMap<usize, usize>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
        {
            let mut count = usize::from(node.empty());
            match node {
                Lower::Segment(seg) => {
                    count = capped_add(count, count_lower_inner(&seg.next, limit, memo), limit);
                }
                Lower::General { children, .. } => {
                    for kids in children.values() {
                        for child in kids.values() {
                            count = capped_add(count, count_lower(child, limit, memo), limit);
                            if count == limit {
                                return count;
                            }
                        }
                    }
                }
            }

            count
        }

        fn count_upper<T, A>(
            node: &Arc<Upper<T, A>>,
            limit: usize,
            memo_upper: &mut StdHashMap<usize, usize>,
            memo_lower: &mut StdHashMap<usize, usize>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            let ptr = Arc::as_ptr(node) as usize;
            if let Some(&cached) = memo_upper.get(&ptr) {
                return cached;
            }

            let mut count = match &**node {
                Upper::Branch(b) => usize::from(b.empty.is_some()),
                Upper::Interface(i) => usize::from(i.inner.empty()),
            };

            match &**node {
                Upper::Branch(b) => {
                    for children in b.children.values() {
                        for child in children.values() {
                            count = capped_add(
                                count,
                                count_upper(child, limit, memo_upper, memo_lower),
                                limit,
                            );
                            if count == limit {
                                memo_upper.insert(ptr, count);
                                return count;
                            }
                        }
                    }
                }
                Upper::Interface(i) => {
                    match &*i.inner {
                        Lower::Segment(seg) => {
                            count = capped_add(count, count_lower_inner(&seg.next, limit, memo_lower), limit);
                        }
                        Lower::General { children, .. } => {
                            for kids in children.values() {
                                for child in kids.values() {
                                    count = capped_add(count, count_lower(child, limit, memo_lower), limit);
                                    if count == limit {
                                        memo_upper.insert(ptr, count);
                                        return count;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            memo_upper.insert(ptr, count);
            count
        }

        let mut memo_upper = StdHashMap::new();
        let mut memo_lower = StdHashMap::new();
        count_upper(&self.inner, limit, &mut memo_upper, &mut memo_lower)
    }

    pub fn is_single_path(&self) -> bool {
        self.path_count_at_most(2) <= 1
    }

    pub fn reduce_acc(&self) -> Option<A> {
        
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

    /// Visit each accumulator in the GSS without collecting or merging.
    /// Uses pointer-based visited set to avoid hashing accumulators.
    pub fn for_each_acc(&self, mut f: impl FnMut(&A)) {
        const INLINE_VISITED_PTRS: usize = 16;

        enum VisitedPtrs {
            Small(SmallVec<[usize; INLINE_VISITED_PTRS]>),
            Large(HashSet<usize>),
        }

        impl VisitedPtrs {
            fn new() -> Self {
                Self::Small(SmallVec::new())
            }

            fn insert(&mut self, ptr: usize) -> bool {
                match self {
                    Self::Small(seen) => {
                        if seen.contains(&ptr) {
                            return false;
                        }
                        if seen.len() < INLINE_VISITED_PTRS {
                            seen.push(ptr);
                            return true;
                        }
                        let mut upgraded = HashSet::with_capacity(seen.len() * 2);
                        for &existing in seen.iter() {
                            upgraded.insert(existing);
                        }
                        let inserted = upgraded.insert(ptr);
                        *self = Self::Large(upgraded);
                        inserted
                    }
                    Self::Large(seen) => seen.insert(ptr),
                }
            }
        }

        fn walk<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            visited: &mut VisitedPtrs,
            f: &mut impl FnMut(&A),
        ) {
            let ptr = Arc::as_ptr(node) as usize;
            if !visited.insert(ptr) {
                return;
            }
            match &**node {
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        f(acc);
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            walk(child, visited, f);
                        }
                    }
                }
                Upper::Interface(i) => {
                    f(&i.acc);
                }
            }
        }
        let mut visited = VisitedPtrs::new();
        walk(&self.inner, &mut visited, &mut f);
    }

    /// Returns true if all accumulators in the upper tree satisfy the predicate.
    /// Short-circuits on the first accumulator that doesn't.
    /// For a single Interface node (common case), this is O(1).
    pub fn all_accs_satisfy(&self, pred: impl Fn(&A) -> bool) -> bool {
        fn check<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            pred: &impl Fn(&A) -> bool,
        ) -> bool {
            match &**node {
                Upper::Interface(iface) => pred(&iface.acc),
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        if !pred(acc) {
                            return false;
                        }
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            if !check(child, pred) {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
        }
        check(&self.inner, &pred)
    }

    pub fn truncate(&self, max_len: isize) -> Self {
        if max_len < 0 {
            return Self::empty();
        }

        let mut memo_upper = StdHashMap::new();
        let mut memo_lower = StdHashMap::new();

        let new_inner = truncate_upper(
            &self.inner,
            0,
            max_len,
            &mut memo_upper,
            &mut memo_lower,
        );

        new_inner.map_or_else(Self::empty, |inner| Self { inner })
    }

    #[cfg(test)]
    pub fn split_at_depth(&self, depth: isize) -> (Self, Self) {
        if depth < 0 {
            return (self.clone(), Self::empty());
        }
        let below = self.popn(depth);
        let above = self.truncate(depth);
        (below, above)
    }

    #[cfg(test)]
    pub fn accs_by_depth(&self) -> BTreeMap<isize, A>
    where
        A: Ord,
    {
        let mut accs = BTreeMap::new();
        let mut memo_upper = HashSet::new();
        accs_by_depth_upper(&self.inner, 0, &mut accs, &mut memo_upper);
        accs
    }

    #[cfg(test)]
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
        let num_interface_implicit_terminals = 0;

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
                        if i.inner.empty() {
                            num_interfaces_with_empty += 1;
                        }

                        lower_queue.push_back(i.inner.clone());

                        interface_to_lower_edges += 1;
                        *incoming_edges.entry(Arc::as_ptr(&i.inner) as usize).or_insert(0) += 1;
                    }
                }
            }
        }

        while let Some(node) = lower_queue.pop_front() {
            let nid = match &*node {
                Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
                _ => Arc::as_ptr(&node) as usize,
            };
            if visited_lower.insert(nid) {
                num_lower_nodes += 1;
                if node.empty() {
                    num_lower_terminal_nodes += 1;
                }
                max_lower_depth = std::cmp::max(max_lower_depth, node.max_depth());
                for (v, kids) in node.children().iter() {
                    distinct_values.insert(v.clone());
                    if kids.len() > 1 {
                        num_multi_depth_slots_lower += 1;
                        max_multiplicity_per_value_lower =
                            std::cmp::max(max_multiplicity_per_value_lower, kids.len());
                    }
                    for child in kids.values() {
                        lower_edges += 1;
                        let child_id = match &**child {
                            Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
                            _ => Arc::as_ptr(child) as usize,
                        };
                        *incoming_edges.entry(child_id).or_insert(0) += 1;
                        lower_queue.push_back(child.clone());
                    }
                }
            }
        }

        #[derive(Clone, PartialEq, Eq)]
        struct StatsSig<T: Clone + Eq + Hash> {
            terminal: bool,
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
                ids.hash(&mut h); 
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
            let ptr = match &**node {
                Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
                _ => Arc::as_ptr(node) as usize,
            };
            if let Some(id) = memo_lower.get(&ptr) {
                return *id;
            }
            let mut edges: StdHashMap<T, Vec<usize>> = StdHashMap::new();
            for (v, kids) in node.children().iter() {
                let e = edges.entry(v.clone()).or_default();
                for child in kids.values() {
                    let cid = canon_lower_for_stats::<T, A>(child, memo_lower, interner);
                    e.push(cid);
                }
                e.sort_unstable();
                e.dedup();
            }
            let sig = StatsSig {
                terminal: node.empty(),
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
                    for (v, kids) in i.inner.children().iter() {
                        let e = edges.entry(v.clone()).or_default();
                        for child in kids.values() {
                            let cid = canon_lower_for_stats::<T, A>(child, memo_lower, interner);
                            e.push(cid);
                        }
                        e.sort_unstable();
                        e.dedup();
                    }
                    let terminal = i.inner.empty();
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

    #[cfg(test)]
    pub fn to_graph_string(&self, upper_only: bool) -> String
    where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        let mut memo = HashSet::new();
        self.to_graph_string_with_memo(&mut memo, upper_only)
    }

    #[cfg(test)]
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
            node.max_depth()
        );
        if node.empty() {
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
                    "Interface @ {:#x} -> Lower @ {:#x} (MaxDepth: {}) | acc: {:?}",
                    Arc::as_ptr(i) as usize,
                    Arc::as_ptr(&i.inner) as usize,
                    node.max_depth(),
                    i.acc
                );
                if i.inner.empty() {
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
        let node_children = node.children();
        let mut sorted_values: Vec<_> = node_children.keys().collect();
        sorted_values.sort_by_key(|v| format!("{:?}", v));

        for v in sorted_values {
            if let Some(kids_at_depths) = node_children.get(v) {
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
                if upper_only && !i.inner.children().is_empty() {
                    let prefix_char = "└── ";
                    let num_lower_edges: usize = i.inner.children().values().map(|kids| kids.len()).sum();
                    let line = format!("{}[{} lower edges omitted]", prefix_char, num_lower_edges);
                    output_lines.push(format!("{}{}", current_prefix, line));
                    return;
                }

                let mut children_to_print = Vec::new();
                let iface_children = i.inner.children();
                let mut sorted_values: Vec<_> = iface_children.keys().collect();
                sorted_values.sort_by_key(|v| format!("{:?}", v));

                for v in sorted_values {
                    if let Some(kids_at_depths) = iface_children.get(v) {
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

    #[cfg(test)]
    pub fn predecessors(&self) -> BTreeMap<T, BTreeMap<u32, Vec<Self>>>
    where
        T: Clone + Eq + Hash + Ord,
        A: Merge + Clone + Eq + Hash + Ord,
    {
        let mut result = BTreeMap::new();
        match &*self.inner {
            Upper::Branch(b) => {
                for (edge_val, children_by_depth) in &b.children {
                    let mut preds_by_depth: BTreeMap<u32, Vec<Self>> = BTreeMap::new();
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
                for (edge_val, children_by_depth) in &i.inner.children() {
                    let mut preds_by_depth: BTreeMap<u32, Vec<Self>> = BTreeMap::new();
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

    #[cfg(test)]
    pub fn num_paths(&self) -> usize {
        self.paths_info().num_paths
    }

    #[cfg(test)]
    pub fn paths_info(&self) -> GSSPathsInfo {
        let mut memo_upper = StdHashMap::new();
        let mut memo_lower = StdHashMap::new();
        Self::paths_info_upper(&self.inner, &mut memo_upper, &mut memo_lower)
    }

    #[cfg(test)]
    fn paths_info_lower(
        node: &Arc<Lower<T>>,
        memo: &mut StdHashMap<usize, GSSPathsInfo>,
    ) -> GSSPathsInfo {
        let ptr = Arc::as_ptr(node) as usize;
        if let Some(cached) = memo.get(&ptr) {
            return *cached;
        }

        let mut info = if node.empty() {
            GSSPathsInfo {
                num_paths: 1,
                total_depth: 0,
            }
        } else {
            GSSPathsInfo::default()
        };

        for children in node.children().values() {
            for child in children.values() {
                let child_info = Self::paths_info_lower(child, memo);
                info.num_paths += child_info.num_paths;
                info.total_depth += child_info.total_depth + child_info.num_paths;
            }
        }

        memo.insert(ptr, info);
        info
    }

    #[cfg(test)]
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
                let mut info = if i.inner.empty() {
                    GSSPathsInfo {
                        num_paths: 1,
                        total_depth: 0,
                    }
                } else {
                    GSSPathsInfo::default()
                };

                for children in i.inner.children().values() {
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

}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use super::*;
    use std::sync::Arc;

    #[derive(Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
    struct IntAcc(BTreeSet<i32>);

    impl Merge for IntAcc {
        fn merge(&self, other: &Self) -> Self {
            IntAcc(&self.0 | &other.0)
        }
    }

    impl IntAcc {
        fn new(vals: &[i32]) -> Self {
            let mut set = BTreeSet::new();
            for &v in vals {
                set.insert(v);
            }
            IntAcc(set)
        }
    }

    type TestGSS = LeveledGSS<String, IntAcc>;
    type TestGSSInt = LeveledGSS<i32, IntAcc>;

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

        assert!(gss1.inner_ptrs_eq(&gss2));
        assert!(!gss1.ptr_eq(&gss2)); 
    }

    #[test]
    fn test_push_pop_identity() {
        let gss0 = gss_from_str_stacks(&[(
            &["A", "B"],
            &[1],
        )]);

        let gss1 = gss0.push("X".to_string()).pop();

        assert!(gss0.inner_ptrs_eq(&gss1));
        assert!(!gss0.ptr_eq(&gss1)); 
    }

    #[test]
    fn test_push_pop_identity_from_empty() {
        let gss0 = TestGSS::empty();
        let gss1 = gss0.push("X".to_string()).pop();

        assert!(gss0.ptr_eq(&gss1));
    }

    #[test]
    fn test_pop_preserves_child_node_sharing() {
        let gss_abc = gss_from_str_stacks(&[(
            &["C", "B", "A"],
            &[1],
        )]);

        let gss_bc_from_pop = gss_abc.pop();

        let preds = gss_abc.predecessors();
        let children_of_a = preds.get(&"A".to_string()).unwrap();
        let gss_bc_from_preds = children_of_a.values().next().unwrap().first().unwrap();

        assert!(gss_bc_from_pop.inner_ptrs_eq(gss_bc_from_preds));

        let inner_pop = &gss_bc_from_pop.inner;
        let inner_preds = &gss_bc_from_preds.inner;

        match (&**inner_pop, &**inner_preds) {
            (Upper::Interface(i_pop), Upper::Interface(i_preds)) => {
                let pop_children = i_pop.inner.children();
                let preds_children = i_preds.inner.children();
                let children_pop = pop_children.get(&"B".to_string()).unwrap();
                let children_preds = preds_children.get(&"B".to_string()).unwrap();
                let child_c_pop = children_pop.values().next().unwrap();
                let child_c_preds = children_preds.values().next().unwrap();
                assert!(Arc::ptr_eq(child_c_pop, child_c_preds));
            }
            _ => panic!("Expected Interface nodes"),
        }
    }

    #[test]
    fn test_path_count_at_most_distinguishes_single_vs_branched() {
        let single = gss_from_str_stacks(&[(&["A", "B", "C"], &[1])]);
        assert_eq!(single.path_count_at_most(2), 1);
        assert!(single.is_single_path());

        let shared_prefix = gss_from_str_stacks(&[
            (&["A", "B"], &[1]),
            (&["A", "C"], &[2]),
        ]);
        assert_eq!(shared_prefix.path_count_at_most(2), 2);
        assert!(!shared_prefix.is_single_path());

        let disjoint = gss_from_str_stacks(&[
            (&["X"], &[3]),
            (&["Y"], &[4]),
        ]);
        assert_eq!(disjoint.path_count_at_most(2), 2);
        assert!(!disjoint.is_single_path());
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
        assert!(!gss1.ptr_eq(&gss2)); 
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
        assert!(!gss1.ptr_eq(&gss2)); 
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
        assert!(!gss1.ptr_eq(&gss2)); 
    }

    #[test]
    fn test_isolate_preserves_ptr_on_noop() {
        
        let gss0 = gss_from_str_stacks(&[(&["A"], &[1])]);
        let gss1 = gss0.isolate(Some("A".to_string()));
        assert!(gss0.ptr_eq(&gss1));

        let gss2 = gss_from_str_stacks(&[(&[], &[1])]);
        let gss3 = gss2.isolate(None);
        assert!(gss2.ptr_eq(&gss3));

        let gss4 = gss_from_str_stacks(&[(&["A"], &[1]), (&["B"], &[2])]);
        let gss5 = gss4.isolate(Some("A".to_string()));
        assert!(!gss4.ptr_eq(&gss5));

        let gss6 = gss_from_str_stacks(&[(&["A"], &[1]), (&[], &[2])]);
        let gss7 = gss6.isolate(None);
        assert!(!gss6.ptr_eq(&gss7));
    }

    #[test]
    fn test_isolate_many_preserves_ptr_on_noop() {
        let gss0 = gss_from_str_stacks(&[(&["A"], &[1]), (&["B"], &[2]), (&[], &[3])]);

        let gss1 = gss0.isolate_many(vec![Some("A".to_string()), Some("B".to_string()), None]);
        assert!(gss0.ptr_eq(&gss1));

        let gss2 = gss0.isolate_many(vec![Some("A".to_string()), Some("B".to_string()), Some("C".to_string()), None]);
        assert!(gss0.ptr_eq(&gss2));

        let gss3 = gss0.isolate_many(vec![Some("A".to_string()), None]);
        assert!(!gss0.ptr_eq(&gss3));

        let gss4 = gss0.isolate_many(vec![Some("A".to_string()), Some("B".to_string())]);
        assert!(!gss0.ptr_eq(&gss4));

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

        let gss1 = gss0.filter_by_length(None, None);
        assert!(gss0.ptr_eq(&gss1));

        let gss2 = gss0.filter_by_length(Some(0), Some(3));
        assert!(gss0.ptr_eq(&gss2));
        let gss3 = gss0.filter_by_length(Some(-1), Some(10));
        assert!(gss0.ptr_eq(&gss3));

        let gss4 = gss0.filter_by_length(Some(1), Some(2));
        assert!(!gss0.ptr_eq(&gss4));
        assert_eq!(gss4.to_stacks().len(), 2);

        let gss_empty = TestGSS::empty();
        let gss_empty_filtered = gss_empty.filter_by_length(Some(1), Some(2));
        assert!(gss_empty.ptr_eq(&gss_empty_filtered));
    }

    #[test]
    fn test_prune_preserves_ptr_on_noop() {
        
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B"], &[1, 2]), 
            (&["X"], &[3]),       
            (&[], &[1, 3]),       
        ]);

        let gss1 = gss0.prune(|_acc| true);
        assert!(gss0.ptr_eq(&gss1));

        let gss2 = gss0.prune(|acc| acc.0.contains(&1));
        assert!(!gss0.ptr_eq(&gss2));
        
        assert_eq!(gss2.to_stacks().len(), 2);

        let gss3 = gss0.prune(|_acc| false);
        assert!(gss3.is_empty());
        assert!(!gss0.ptr_eq(&gss3));

        let gss_empty = TestGSS::empty();
        let gss_empty_pruned = gss_empty.prune(|_acc| true);
        assert!(gss_empty.ptr_eq(&gss_empty_pruned));
    }

    #[test]
    fn test_normalization_sharing_factor_regression() {
        type TestGSSInt = LeveledGSS<i32, IntAcc>;

        let l_terminal = new_lower::<i32>(CompactMap::new(), true);

        let l_500_parent = new_lower(
            CompactMap::unit(500, CompactOrdMap::unit(l_terminal.max_depth(), l_terminal.clone())),
            false,
        );

        let i_54_inner = new_lower(
            CompactMap::unit(569, CompactOrdMap::unit(l_500_parent.max_depth(), l_500_parent.clone())),
            false,
        );
        let i_54 = new_interface(i_54_inner, IntAcc::new(&[54]));

        let acc_vals = vec![38, 39, 40, 44, 45, 46, 47, 48, 49, 50];
        let interfaces_10: Vec<_> = acc_vals
            .iter()
            .map(|&acc| new_interface(l_500_parent.clone(), IntAcc::new(&[acc])))
            .collect();

        let edge_vals = vec![419, 437, 66, 477, 531, 541, 556, 558, 560, 562];
        let mut children_101 = CompactMap::new();
        for (edge, interface) in edge_vals.iter().zip(interfaces_10.iter()) {
            children_101.insert(*edge, CompactOrdMap::unit(interface.max_depth(), interface.clone()));
        }
        let ub_101 = new_branch(children_101, None);

        let ub_295_d4 = new_branch(
            CompactMap::unit(101, CompactOrdMap::unit(ub_101.max_depth(), ub_101)),
            None,
        );

        let mut children_295 = CompactOrdMap::new();
        children_295.insert(i_54.max_depth(), i_54);
        children_295.insert(ub_295_d4.max_depth(), ub_295_d4);
        let root_children = CompactMap::unit(295, children_295);
        let root_inner = new_branch(root_children, None);

        let gss = TestGSSInt { inner: root_inner };

        let stats_before = gss.stats();

        let gss_after = gss.normalize();
        let stats_after = gss_after.stats();

        assert_eq!(gss.to_stacks().len(), 11);
        assert_eq!(gss_after.to_stacks().len(), 11);

        assert!(
            stats_after.total_unique_nodes <= stats_before.total_unique_nodes,
            "total unique nodes increased: {} vs {}",
            stats_before.total_unique_nodes,
            stats_after.total_unique_nodes
        );

        assert_eq!(stats_before.num_lower_nodes, 3);
        assert_eq!(stats_after.num_lower_nodes, 2);

        assert_eq!(stats_before.num_multi_depth_slots_upper, 1);
        assert_eq!(stats_after.num_multi_depth_slots_upper, 0);

        assert!(
            stats_after.num_structurally_unique_nodes < stats_before.num_structurally_unique_nodes,
            "num structurally unique nodes did not decrease: {} vs {}",
            stats_before.num_structurally_unique_nodes,
            stats_after.num_structurally_unique_nodes
        );
        assert_eq!(stats_before.num_structurally_unique_nodes, 6);
        assert_eq!(stats_after.num_structurally_unique_nodes, 5);
    }

    #[test]
    fn test_split_at_depth() {
        let gss = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
            (&["A", "D"], &[2]),
            (&["E"], &[3]),
        ]);

        let (below1, above1) = gss.split_at_depth(1);

        let expected_below1 = gss_from_str_stacks(&[
            (&["A", "B"], &[1]),
            (&["A"], &[2]),
            (&[], &[3]),
        ]);
        let expected_above1 = gss_from_str_stacks(&[
            (&["C"], &[1]),
            (&["D"], &[2]),
            (&["E"], &[3]),
        ]);

        assert_eq!(below1.to_stacks().into_iter().collect::<HashSet<_>>(), expected_below1.to_stacks().into_iter().collect::<HashSet<_>>());
        assert_eq!(above1.to_stacks().into_iter().collect::<HashSet<_>>(), expected_above1.to_stacks().into_iter().collect::<HashSet<_>>());

        let (below2, above2) = gss.split_at_depth(2);
        let expected_below2 = gss_from_str_stacks(&[
            (&["A"], &[1]),
            (&[], &[2, 3]),
        ]);
        let expected_above2 = gss_from_str_stacks(&[
            (&["B", "C"], &[1]),
            (&["A", "D"], &[2]),
            (&["E"], &[3]),
        ]);
        assert_eq!(below2.to_stacks().into_iter().collect::<HashSet<_>>(), expected_below2.to_stacks().into_iter().collect::<HashSet<_>>());
        assert_eq!(above2.to_stacks().into_iter().collect::<HashSet<_>>(), expected_above2.to_stacks().into_iter().collect::<HashSet<_>>());
    }

    #[test]
    fn test_accs_by_depth() {
        let gss = gss_from_str_stacks(&[
            (&["A", "B"], &[1]), 
            (&["C", "D"], &[2]), 
            (&["E"], &[3]),       
            (&[], &[4]),         
        ]);

        let accs = gss.accs_by_depth();

        let mut expected = BTreeMap::new();
        expected.insert(0, IntAcc::new(&[4]));
        expected.insert(1, IntAcc::new(&[3]));
        expected.insert(2, IntAcc::new(&[1, 2]));

        assert_eq!(accs.len(), 3);
        assert_eq!(accs.get(&0), expected.get(&0));
        assert_eq!(accs.get(&1), expected.get(&1));
        assert_eq!(accs.get(&2), expected.get(&2));
    }

    #[test]
    fn test_popn_with_underflow_basic() {
        
        let gss = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),  
            (&["X", "Y"], &[2]),       
            (&["Z"], &[3]),            
            (&[], &[4]),               
        ]);

        let (result, underflows) = gss.popn_with_underflow(3);

        let result_stacks = result.to_stacks();
        assert_eq!(result_stacks.len(), 1);
        
        assert!(result_stacks.iter().any(|(stack, acc)| stack.is_empty() && acc.0.contains(&1)));

        assert_eq!(underflows.len(), 3);
        
        assert!(underflows.get(&1).map(|a| a.0.contains(&2)).unwrap_or(false), 
            "Expected shortfall 1 to contain acc 2, got {:?}", underflows);
        
        assert!(underflows.get(&2).map(|a| a.0.contains(&3)).unwrap_or(false),
            "Expected shortfall 2 to contain acc 3, got {:?}", underflows);
        
        assert!(underflows.get(&3).map(|a| a.0.contains(&4)).unwrap_or(false),
            "Expected shortfall 3 to contain acc 4, got {:?}", underflows);
    }

    #[test]
    fn test_popn_with_underflow_no_underflow() {
        
        let gss = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
            (&["X", "Y", "Z"], &[2]),
        ]);

        let (result, underflows) = gss.popn_with_underflow(2);

        let result_stacks = result.to_stacks();
        assert_eq!(result_stacks.len(), 2);
        
        assert!(underflows.is_empty(), "Expected no underflows, got {:?}", underflows);
    }

    #[test]
    fn test_popn_with_underflow_all_underflow() {
        
        let gss = gss_from_str_stacks(&[
            (&["A"], &[1]),
            (&["B"], &[2]),
            (&[], &[3]),
        ]);

        let (result, underflows) = gss.popn_with_underflow(5);

        assert!(result.is_empty(), "Expected empty GSS, got {:?}", result.to_stacks());

        assert!(underflows.contains_key(&4), "Expected shortfall 4");
        assert!(underflows.contains_key(&5), "Expected shortfall 5");
        
        let shortfall_4 = underflows.get(&4).unwrap();
        assert!(shortfall_4.0.contains(&1) && shortfall_4.0.contains(&2));
    }

    #[test]
    fn test_popn_with_underflow_n_zero() {
        let gss = gss_from_str_stacks(&[
            (&["A", "B"], &[1]),
            (&[], &[2]),
        ]);

        let (result, underflows) = gss.popn_with_underflow(0);

        assert_eq!(result.to_stacks().len(), gss.to_stacks().len());
        assert!(underflows.is_empty());
    }

    #[test]
    fn test_chain_deep_push_pop_roundtrip() {
        // Build a deep stack: ["A", "B", "C", "D", "E"] with acc [1]
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C", "D", "E"], &[1]),
        ]);
        let stacks0 = gss0.to_stacks();
        assert_eq!(stacks0.len(), 1);
        assert_eq!(stacks0[0].0, vec!["A", "B", "C", "D", "E"]);

        // Pop 1 at a time and verify
        let gss1 = gss0.popn(1);
        let s1 = gss1.to_stacks();
        assert_eq!(s1.len(), 1, "After pop 1: {:?}", s1);
        assert_eq!(s1[0].0, vec!["A", "B", "C", "D"]);

        let gss2 = gss0.popn(2);
        let s2 = gss2.to_stacks();
        assert_eq!(s2.len(), 1, "After pop 2: {:?}", s2);
        assert_eq!(s2[0].0, vec!["A", "B", "C"]);

        let gss3 = gss0.popn(3);
        let s3 = gss3.to_stacks();
        assert_eq!(s3.len(), 1, "After pop 3: {:?}", s3);
        assert_eq!(s3[0].0, vec!["A", "B"]);

        let gss4 = gss0.popn(4);
        let s4 = gss4.to_stacks();
        assert_eq!(s4.len(), 1, "After pop 4: {:?}", s4);
        assert_eq!(s4[0].0, vec!["A"]);

        let gss5 = gss0.popn(5);
        let s5 = gss5.to_stacks();
        assert_eq!(s5.len(), 1, "After pop 5: {:?}", s5);
        assert_eq!(s5[0].0, Vec::<String>::new());
    }

    #[test]
    fn test_chain_deep_isolate_and_push() {
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
        ]);

        // Isolate top value
        let isolated = gss0.isolate(Some("C".to_string()));
        let s_iso = isolated.to_stacks();
        assert_eq!(s_iso.len(), 1, "After isolate C: {:?}", s_iso);
        assert_eq!(s_iso[0].0, vec!["A", "B", "C"]);

        // Pop then push
        let popped = gss0.popn(1);
        let pushed = popped.push("X".to_string());
        let s_pushed = pushed.to_stacks();
        assert_eq!(s_pushed.len(), 1, "After pop1+pushX: {:?}", s_pushed);
        assert_eq!(s_pushed[0].0, vec!["A", "B", "X"]);
    }

    #[test]
    fn test_chain_deep_merge_and_fuse() {
        let gss1 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
        ]);
        let gss2 = gss_from_str_stacks(&[
            (&["A", "B", "D"], &[1]),
        ]);
        let merged = gss1.merge(&gss2);
        let s_merged = merged.to_stacks();
        assert_eq!(s_merged.len(), 2, "After merge: {:?}", s_merged);

        let fused = merged.fuse(None);
        let s_fused = fused.to_stacks();
        assert_eq!(s_fused.len(), 2, "After fuse: {:?}", s_fused);
    }

    #[test]
    fn test_chain_shift_top_values() {
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
        ]);

        let shifted = gss0.shift_top_values(vec![
            ("C".to_string(), "X".to_string()),
        ]);
        let s = shifted.to_stacks();
        assert_eq!(s.len(), 1, "After shift: {:?}", s);
        assert_eq!(s[0].0, vec!["A", "B", "C", "X"]);
    }

    #[test]
    fn test_chain_absorb_push() {
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
        ]);
        let base = gss_from_str_stacks(&[
            (&["A", "B"], &[1]),
        ]);

        let result = gss0.absorb_push("X".to_string(), &base);
        let s = result.to_stacks();
        assert_eq!(s.len(), 2, "After absorb_push: {:?}", s);
    }

    #[test]
    fn test_chain_apply_and_prune() {
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
        ]);
        let pruned = gss0.apply_and_prune_no_promote(|_acc| Some(IntAcc::new(&[1])));
        let s = pruned.to_stacks();
        assert_eq!(s.len(), 1, "After prune: {:?}", s);
        assert_eq!(s[0].0, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_chain_decompose() {
        let gss0 = gss_from_str_stacks(&[
            (&["A", "B", "C"], &[1]),
        ]);
        let (decomposed, _root_accs) = gss0.apply_transform_and_decompose(|acc| Some(acc.clone()));
        // Should decompose into one entry for top value "C"
        assert_eq!(decomposed.len(), 1, "decomposed: {:?}", decomposed.len());
        let (val, sub_gss) = &decomposed[0];
        assert_eq!(val, "C");
        let sub_stacks = sub_gss.to_stacks();
        assert_eq!(sub_stacks.len(), 1, "sub stacks: {:?}", sub_stacks);
        assert_eq!(sub_stacks[0].0, vec!["A", "B"]);
    }

    #[test]
    fn test_chain_parser_like_sequence() {
        // Simulate a parser-like sequence of operations with integer GSS
        type IGSS = LeveledGSS<i32, IntAcc>;

        // Start: state 0 at bottom
        let gss = IGSS::from_stacks(&[(vec![0], IntAcc::new(&[0]))]);
        assert_eq!(gss.to_stacks().len(), 1);
        
        // Push state 5
        let gss = gss.push(5);
        assert_eq!(gss.to_stacks()[0].0, vec![0, 5]);
        
        // Push state 10
        let gss = gss.push(10);
        assert_eq!(gss.to_stacks()[0].0, vec![0, 5, 10]);
        
        // Isolate state 10
        let sub = gss.isolate(Some(10));
        assert!(!sub.is_empty(), "isolate 10 should not be empty");
        
        // Pop 2 (simulate a reduce of RHS len 2)
        let popped = sub.popn(2);
        assert!(!popped.is_empty(), "popn 2 should not be empty");
        let popped_stacks = popped.to_stacks();
        assert_eq!(popped_stacks.len(), 1);
        assert_eq!(popped_stacks[0].0, vec![0]);
        
        // Isolate goto_from state 0
        let base = popped.isolate(Some(0));
        assert!(!base.is_empty(), "isolate 0 should not be empty");
        
        // Push goto target state 7
        let result = base.push(7);
        let result_stacks = result.to_stacks();
        assert_eq!(result_stacks.len(), 1);
        assert_eq!(result_stacks[0].0, vec![0, 7]);
        
        // Merge with original and shift
        let merged = gss.merge(&result);
        let m_stacks = merged.to_stacks();
        assert_eq!(m_stacks.len(), 2, "merged stacks: {:?}", m_stacks);
    }

    #[test]
    fn test_chain_multi_level_push_isolate_pop() {
        type IGSS = LeveledGSS<i32, IntAcc>;

        // Build a 5-level deep stack
        let gss = IGSS::from_stacks(&[(vec![0, 1, 2, 3, 4], IntAcc::new(&[0]))]);

        // Isolate top (4) and pop 3
        let sub = gss.isolate(Some(4));
        let popped = sub.popn(3);
        let s = popped.to_stacks();
        assert_eq!(s.len(), 1, "popped stacks: {:?}", s);
        assert_eq!(s[0].0, vec![0, 1]);

        // Isolate again, pop again
        let sub2 = popped.isolate(Some(1));
        let popped2 = sub2.popn(1);
        let s2 = popped2.to_stacks();
        assert_eq!(s2.len(), 1, "popped2 stacks: {:?}", s2);
        assert_eq!(s2[0].0, vec![0]);
    }
}
