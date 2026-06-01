use im::{HashMap as IHashMap, OrdMap};
use smallvec::{SmallVec, smallvec};
use super::stack_vecs::dispatch::DynStackVec;
use std::collections::{HashMap as StdHashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, OnceLock};

/// Type alias for segment values. Set `STACKVEC` before process startup:
///   normal builds: `vec` (default), `arc`
type SV<T> = DynStackVec<T>;

pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;

    fn subsumes(&self, _other: &Self) -> bool {
        false
    }
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

/// Linear segment of the stack: multiple values packed into one node.
/// `values[0]` is the deepest value (closest to `next`),
/// `values[last]` is the shallowest (top of stack).
/// Intermediate levels (all except the top) are guaranteed to have empty=false.
/// Values are stored in SV<T> (type-aliased segment vector).
/// Segments are always non-accepting (empty is implicitly false).
struct Segment<T: Clone + Eq + Hash> {
    values: SV<T>,
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
                    // Pop the top value by taking all-but-last. O(1) for view-based types.
                    // Don't call new_segment() — no need to merge, just shrink the segment.
                    let rest_values = seg.values.take(seg.values.len() - 1);
                    let child_max_depth = seg.max_depth - 1;
                    let segments_len = seg.segments_len - 1;
                    Arc::new(Lower::Segment(Arc::new(Segment {
                        values: rest_values,
                        next: seg.next,
                        max_depth: child_max_depth,
                        segments_len,
                        rest: OnceLock::new(),
                    })))
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

    /// For Segment variant, get the values vector.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_values(&self) -> &SV<T> {
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
                let rest_values = seg.values.take(seg.values.len() - 1);
                // Don't merge with child; just create a shorter segment.
                let max_depth = seg.max_depth - 1;
                let segments_len = seg.segments_len - 1;
                Arc::new(Lower::Segment(Arc::new(Segment {
                    values: rest_values,
                    next: seg.next.clone(),
                    max_depth,
                    segments_len,
                    rest: OnceLock::new(),
                })))
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
    pub lower_general_nodes: usize,
    pub lower_segment_nodes: usize,
    pub total_unique_nodes: usize,
    pub total_edges: usize,
    pub accumulator_instances: usize,
    pub max_depth: u32,
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
            let values = SV::unit(key.clone());
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

fn new_segment<T: Clone + Eq + Hash>(values: SV<T>, next: Arc<Lower<T>>) -> Arc<Lower<T>> {
    // Merge with child segment if possible and fits.
    if let Lower::Segment(child_seg) = &*next {
        if let Some(merged) = child_seg.values.try_append(&values) {
            let max_depth = child_seg.next.max_depth() + merged.len() as u32;
            let segments_len = merged.len() + child_seg.next.segments_len();
            return Arc::new(Lower::Segment(Arc::new(Segment {
                values: merged,
                next: child_seg.next.clone(),
                max_depth,
                segments_len,
                rest: OnceLock::new(),
            })));
        }
    }
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




fn merge_lower<T: Clone + Eq + Hash>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {
    if Arc::ptr_eq(l1, l2) {
        return l1.clone();
    }

    let new_empty = l1.empty() || l2.empty();
    let merged_children = match (&**l1, &**l2) {
        (Lower::Segment(s1), Lower::Segment(s2)) => {
            // Fast path: if both segments share the same tail and have identical
            // values, they are structurally identical — skip deep recursion.
            if Arc::ptr_eq(&s1.next, &s2.next) && s1.values == s2.values {
                return l1.clone();
            }

            let v1 = l1.segment_top_value();
            let v2 = l2.segment_top_value();
            let r1 = l1.segment_rest_arc();
            let r2 = l2.segment_rest_arc();
            if v1 == v2 {
                let merged_next = merge_lower(&r1, &r2);
                CompactMap::unit(v1.clone(), CompactOrdMap::unit(merged_next.max_depth(), merged_next))
            } else {
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

fn nonempty_deterministic_top_step<T>(lower: &Arc<Lower<T>>) -> Option<(T, Arc<Lower<T>>)> 
where
    T: Clone + Eq + Hash,
{
    match &**lower {
        Lower::Segment(seg) => {
            let value = seg.values.last()?.clone();
            Some((value, lower.segment_rest_arc()))
        }
        Lower::General { children, empty, .. } if !*empty && children.len() == 1 => {
            let (value, kids) = children.iter().next()?;
            if kids.len() != 1 {
                return None;
            }
            let child = kids.values().next()?.clone();
            Some((value.clone(), child))
        }
        _ => None,
    }
}

fn shared_nonempty_deterministic_prefix<T>(
    left: &Arc<Lower<T>>,
    right: &Arc<Lower<T>>,
) -> (SmallVec<[T; 8]>, Arc<Lower<T>>, Arc<Lower<T>>)
where
    T: Clone + Eq + Hash,
{
    let mut prefix = SmallVec::<[T; 8]>::new();
    let mut left_rest = left.clone();
    let mut right_rest = right.clone();

    loop {
        let Some((left_value, next_left)) = nonempty_deterministic_top_step(&left_rest) else {
            break;
        };
        let Some((right_value, next_right)) = nonempty_deterministic_top_step(&right_rest) else {
            break;
        };

        if left_value != right_value {
            break;
        }

        prefix.push(left_value);
        left_rest = next_left;
        right_rest = next_right;
    }

    (prefix, left_rest, right_rest)
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
    let acc_equal = a.acc == b.acc;
    let inner_ptr_eq = Arc::ptr_eq(&a.inner, &b.inner);
    if inner_ptr_eq {
        if b.acc.subsumes(&a.acc) {
            return Arc::new(Upper::Interface(b.clone()));
        }
        if a.acc.subsumes(&b.acc) {
            return Arc::new(Upper::Interface(a.clone()));
        }
        return new_interface(a.inner.clone(), a.acc.merge(&b.acc));
    }
    if acc_equal {
        let merged_lower = merge_lower(&a.inner, &b.inner);
        let new_acc = a.acc.merge(&b.acc);
        new_interface(merged_lower, new_acc)
    } else {
        let (shared_prefix, left_rest, right_rest) =
            shared_nonempty_deterministic_prefix(&a.inner, &b.inner);
        if !shared_prefix.is_empty() {
            let left_remainder = Arc::new(Interface {
                inner: left_rest,
                acc: a.acc.clone(),
            });
            let right_remainder = Arc::new(Interface {
                inner: right_rest,
                acc: b.acc.clone(),
            });

            let mut merged = merge_interfaces(&left_remainder, &right_remainder);
            for value in shared_prefix.into_iter().rev() {
                let children = CompactMap::unit(value, CompactOrdMap::unit(merged.max_depth(), merged));
                merged = try_promote(&new_branch(children, None));
            }
            merged
        } else {
            let ub1 = interface_to_upperbranch(a);
            let ub2 = interface_to_upperbranch(b);
            merge_upperbranches(&ub1, &ub2)
        }
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














#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Upper<T, A>>,
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
/// When a pop would cross the floor, the caller can materialize the current
/// chain and continue with GSS-level operations for the leftover pop depth.
///
/// `pending_top` is a lazy optimization: pushes set `pending_top` instead of
/// immediately modifying the backing values. If a pop immediately follows,
/// we consume `pending_top` first, avoiding touching the segment chain at all.
/// This is a common pattern during deterministic reduce chains:
///   pop(n) → push(goto) → pop(m) → push(goto2) → ...
#[derive(Clone)]
pub struct VirtualStack<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    values: SV<T>,
    next: Arc<Lower<T>>,
    acc: A,
    pending_top: Option<T>,
}

/// Controls how eagerly VirtualStack creates new segments vs reusing existing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PushMode {
    /// Use try_push only. On failure, create a new segment. (default)
    Lazy,
    /// Use try_harder_push first (clone shared data if needed).
    /// Only create a new segment if that also fails.
    Eager,
}

static PUSH_MODE: OnceLock<PushMode> = OnceLock::new();

fn push_mode() -> PushMode {
    *PUSH_MODE.get_or_init(|| {
        match std::env::var("PUSH_MODE").as_deref() {
            Ok("eager") | Ok("harder") => PushMode::Eager,
            _ => PushMode::Lazy,
        }
    })
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> VirtualStack<T, A> {
    /// The current top-of-stack value, or None if the stack is empty.
    #[inline]
    pub fn top(&self) -> Option<&T> {
        self.pending_top.as_ref().or_else(|| self.values.last())
    }

    /// Return the top value that would be visible after popping `remaining`
    /// values, without mutating or cloning the virtual stack.
    #[inline]
    pub fn top_after_popping(&self, mut remaining: usize) -> Option<&T> {
        if let Some(top) = self.pending_top.as_ref() {
            if remaining == 0 {
                return Some(top);
            }
            remaining -= 1;
        }

        let mut values = &self.values;
        let mut next = &self.next;
        loop {
            let len = values.len();
            if remaining < len {
                return values.iter().rev().nth(remaining);
            }
            remaining -= len;
            match &**next {
                Lower::Segment(seg) => {
                    values = &seg.values;
                    next = &seg.next;
                }
                _ => return None,
            }
        }
    }

    /// Flush pending_top into the backing values.
    #[inline]
    fn flush_pending(&mut self) {
        if let Some(val) = self.pending_top.take() {
            self.realize_push(val);
        }
    }

    /// Actually push a value into the backing storage.
    #[inline]
    fn realize_push(&mut self, value: T) {
        let pushed = match push_mode() {
            PushMode::Lazy => self.values.try_push(value.clone()),
            PushMode::Eager => {
                if self.values.try_push(value.clone()) {
                    true
                } else {
                    self.values.try_harder_push(value.clone())
                }
            }
        };
        if !pushed {
            let seg = new_segment(self.values.clone(), self.next.clone());
            self.next = seg;
            self.values = SV::unit(value);
        }
    }

    /// Pop `n` values from the top.
    /// Returns the number of values that could not be popped because the
    /// segment chain ended at a non-Segment lower node.
    #[inline]
    pub fn pop(&mut self, mut remaining: usize) -> usize {
        // Consume pending_top first (free).
        if remaining > 0 && self.pending_top.is_some() {
            self.pending_top = None;
            remaining -= 1;
        }
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
        if self.values.is_empty() && self.pending_top.is_none() {
            if let Lower::Segment(seg) = &*self.next {
                self.values = seg.values.clone();
                self.next = seg.next.clone();
            }
        }
        remaining
    }

    /// Push a value onto the top of the stack.
    /// Defers the actual push — stores in pending_top. If there's already a
    /// pending value, flushes it to the backing storage first.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.flush_pending();
        self.pending_top = Some(value);
    }

    /// Return the value immediately below the current top, if any.
    #[inline]
    pub fn parent_of_top(&self) -> Option<T> {
        self.top_after_popping(1).cloned()
    }

    /// Replace the current top-of-stack value in place.
    #[inline]
    pub fn replace_top(&mut self, value: T) -> bool {
        if self.top().is_none() {
            return false;
        }
        if self.pending_top.is_some() {
            self.pending_top = Some(value);
            return true;
        }

        let len = self.values.len();
        if len > 0 {
            self.values.truncate(len - 1);
            self.pending_top = Some(value);
            return true;
        }

        true
    }

    /// The total number of values available across the current segment chain.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len() + self.next.segments_len() + if self.pending_top.is_some() { 1 } else { 0 }
    }

    /// Materialize the virtual stack back into a GSS.
    pub fn into_gss(mut self) -> LeveledGSS<T, A> {
        self.flush_pending();
        if self.values.is_empty() {
            return LeveledGSS {
                inner: new_interface(self.next, self.acc),
            };
        }
        LeveledGSS {
            inner: new_interface(new_segment(self.values, self.next), self.acc),
        }
    }

    pub fn into_gss_after_popping(mut self, n: usize) -> LeveledGSS<T, A> {
        self.flush_pending();
        let remaining = self.pop(n);
        let gss = self.into_gss();
        if remaining == 0 {
            gss
        } else {
            gss.popn(remaining as isize)
        }
    }

    pub fn into_gss_after_popping_and_pushing_branches<'a, I>(
        mut self,
        n: usize,
        pushes: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        self.flush_pending();
        if self.pop(n) != 0 {
            return None;
        }

        let base = if self.values.is_empty() {
            self.next
        } else {
            new_segment(self.values, self.next)
        };

        let mut children: Children<T, Lower<T>> = CompactMap::new();
        for pushes in pushes {
            let (top, prefix) = pushes.split_last()?;
            let mut child = base.clone();
            for value in prefix {
                child = new_segment(SV::unit(value.clone()), child);
            }

            let depth = child.max_depth();
            if let Some(existing) = children.get_mut(top) {
                if let Some(existing_child) = existing.get(&depth).cloned() {
                    existing.insert(depth, merge_lower(&existing_child, &child));
                } else {
                    existing.insert(depth, child);
                }
            } else {
                children.insert(top.clone(), CompactOrdMap::unit(depth, child));
            }
        }

        if children.is_empty() {
            return Some(LeveledGSS {
                inner: new_interface(base, self.acc),
            });
        }

        Some(LeveledGSS {
            inner: new_interface(new_lower(children, false), self.acc),
        })
    }

    pub fn into_gss_after_popping_and_pushing_single_branches<'a, I>(
        mut self,
        n: usize,
        targets: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        self.flush_pending();
        if self.pop(n) != 0 {
            return None;
        }

        let base = if self.values.is_empty() {
            self.next
        } else {
            new_segment(self.values, self.next)
        };
        let base_depth = base.max_depth();

        let mut children: Children<T, Lower<T>> = CompactMap::new();
        for target in targets {
            if children.get(target).is_none() {
                children.insert(target.clone(), CompactOrdMap::unit(base_depth, base.clone()));
            }
        }

        if children.is_empty() {
            return Some(LeveledGSS {
                inner: new_interface(base, self.acc),
            });
        }

        Some(LeveledGSS {
            inner: new_interface(new_lower(children, false), self.acc),
        })
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

    pub(crate) fn single_interface_lower_id(&self) -> Option<usize> {
        match &*self.inner {
            Upper::Interface(i) => Some(lower_node_id(&i.inner)),
            Upper::Branch(_) => None,
        }
    }



    pub fn empty() -> Self {
        empty_upper()
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

    pub fn from_single_stack(values: Vec<T>, acc: A) -> Self {
        let floor = new_lower(CompactMap::new(), true);
        let inner = if values.is_empty() {
            new_interface(floor, acc)
        } else {
            new_interface(new_segment(SV::from_vec(values), floor), acc)
        };
        LeveledGSS { inner }
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

    /// Apply a set of stack effects by materializing the single concrete stack.
    ///
    /// This is only a win for already-deterministic parser states that have a
    /// large `StackShifts` action. In that shape, the generic GSS branch builder
    /// can spend most of its time constructing and merging branches that collapse
    /// back to one or two concrete stacks.
    pub fn apply_stack_effects_to_single_concrete_path<'a, I>(
        &self,
        effects: I,
        max_materialized_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        if let Some(stack) = self.try_virtual_stack() {
            let mut out: Option<Self> = None;
            for (pop, pushes) in effects {
                let mut branch = stack.clone();
                if branch.pop(pop) != 0 {
                    continue;
                }
                for value in pushes {
                    branch.push(value.clone());
                }
                let branch = branch.into_gss();
                out = Some(match out {
                    Some(existing) => existing.merge(&branch),
                    None => branch,
                });
            }
            if let Some(out) = out {
                return Some(out);
            }
            let empty: Vec<(Vec<T>, A)> = Vec::new();
            return Some(Self::from_stacks(&empty));
        }

        if self.max_depth() as usize > max_materialized_depth {
            return None;
        }

        let mut stacks = self.to_stacks();
        if stacks.len() != 1 {
            return None;
        }
        let (stack, acc) = stacks.pop().unwrap();

        let mut out: Vec<(Vec<T>, A)> = Vec::new();
        for (pop, pushes) in effects {
            if pop > stack.len() {
                continue;
            }

            let keep = stack.len() - pop;
            let mut next = Vec::with_capacity(keep + pushes.len());
            next.extend_from_slice(&stack[..keep]);
            next.extend_from_slice(pushes);

            if let Some((_, existing_acc)) =
                out.iter_mut().find(|(existing_stack, _)| *existing_stack == next)
            {
                *existing_acc = existing_acc.merge(&acc);
            } else {
                out.push((next, acc.clone()));
            }
        }

        match out.len() {
            1 => {
                let (stack, acc) = out.pop().unwrap();
                Some(Self::from_single_stack(stack, acc))
            }
            _ => Some(Self::from_stacks(&out)),
        }
    }

    pub fn apply_shared_pop_push_branches<'a, I>(
        &self,
        pop: usize,
        pushes: I,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        self.try_virtual_stack()?
            .into_gss_after_popping_and_pushing_branches(pop, pushes)
    }

    pub fn apply_shared_pop_push_single_branches<'a, I>(
        &self,
        pop: usize,
        targets: I,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        self.try_virtual_stack()?
            .into_gss_after_popping_and_pushing_single_branches(pop, targets)
    }

    pub fn apply_guarded_stack_effects_to_single_concrete_path<'a, I, G>(
        &self,
        effects: I,
        max_materialized_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (G, usize, &'a [T])>,
        G: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        if self.max_depth() as usize > max_materialized_depth {
            return None;
        }

        let mut stacks = self.to_stacks();
        if stacks.len() != 1 {
            return None;
        }
        let (stack, acc) = stacks.pop().unwrap();

        let mut out: Vec<(Vec<T>, A)> = Vec::new();
        for (guards, pop, pushes) in effects {
            if pop > stack.len() {
                continue;
            }

            let mut allowed = true;
            for (guard_pop, guard_states) in guards {
                if guard_pop >= stack.len() {
                    allowed = false;
                    break;
                }
                let state = &stack[stack.len() - 1 - guard_pop];
                if !guard_states.iter().any(|candidate| candidate == state) {
                    allowed = false;
                    break;
                }
            }
            if !allowed {
                continue;
            }

            let keep = stack.len() - pop;
            let mut next = Vec::with_capacity(keep + pushes.len());
            next.extend_from_slice(&stack[..keep]);
            next.extend_from_slice(pushes);

            if let Some((_, existing_acc)) =
                out.iter_mut().find(|(existing_stack, _)| *existing_stack == next)
            {
                *existing_acc = existing_acc.merge(&acc);
            } else {
                out.push((next, acc.clone()));
            }
        }

        match out.len() {
            1 => {
                let (stack, acc) = out.pop().unwrap();
                Some(Self::from_single_stack(stack, acc))
            }
            _ => Some(Self::from_stacks(&out)),
        }
    }

    pub fn push(&self, value: T) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let new_inner = match &*self.inner {
            Upper::Interface(i) => {
                let new_lower_root = new_segment(
                    SV::unit(value),
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
    /// Apply a bulk remapping to the current frontier values.
    pub fn remap_top_values<I>(&self, shifts: I) -> Self
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

    /// Like `remap_top_values` but takes ownership, allowing extraction of
    /// children by move instead of clone when the Arcs are uniquely owned.
    pub fn remap_top_values_owned<I>(self, shifts: I) -> Self
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
                                return gss.remap_top_values(pairs);
                            }
                        }
                    }
                    Err(iface_arc) => {
                        let gss = LeveledGSS { inner: Arc::new(Upper::Interface(iface_arc)) };
                        return gss.remap_top_values(pairs);
                    }
                }
            }
            Ok(upper @ Upper::Branch(_)) => {
                let gss = LeveledGSS { inner: Arc::new(upper) };
                return gss.remap_top_values(pairs);
            }
            Err(arc) => {
                let gss = LeveledGSS { inner: arc };
                return gss.remap_top_values(pairs);
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

    /// Apply pure top-frontier shifts in one pass. Each tuple is
    /// `(current_top, target_top, replace_top)`.
    pub fn try_apply_selective_top_pure_shifts<I>(&self, shifts: I) -> Option<Self>
    where
        I: IntoIterator<Item = (T, T, bool)>,
    {
        let shifts: SmallVec<[(T, T, bool); 2]> = shifts.into_iter().collect();
        let [(from, to, replace_top)] = shifts.as_slice() else {
            return None;
        };

        let Upper::Interface(i) = &*self.inner else {
            return None;
        };
        let Lower::General { children, .. } = &*i.inner else {
            return None;
        };
        let kids = children.get(from)?;

        fn lower_with_top<T: Clone + Eq + Hash>(
            top: T,
            kids: &CompactOrdMap<Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if kids.len() == 1 {
                let (_, child) = kids.iter().next().unwrap();
                new_segment(SV::unit(top), child.clone())
            } else {
                new_lower(CompactMap::unit(top, kids.clone()), false)
            }
        }

        let shifted_root = if *replace_top {
            lower_with_top(to.clone(), kids)
        } else {
            new_segment(SV::unit(to.clone()), lower_with_top(from.clone(), kids))
        };

        Some(LeveledGSS {
            inner: new_interface(shifted_root, i.acc.clone()),
        })
    }

    pub fn apply_top_pure_shifts<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T, bool)>,
    {
        let shifts: SmallVec<[(T, T, bool); 8]> = shifts.into_iter().collect();
        if shifts.is_empty() {
            return Self::empty();
        }

        fn insert_lower_child<T: Clone + Eq + Hash>(
            children: &mut Children<T, Lower<T>>,
            key: T,
            depth: u32,
            child: Arc<Lower<T>>,
        ) {
            if let Some(ord_map) = children.get_mut(&key) {
                if let Some(existing) = ord_map.get(&depth).cloned() {
                    ord_map.insert(depth, merge_lower(&existing, &child));
                } else {
                    ord_map.insert(depth, child);
                }
            } else {
                children.insert(key, CompactOrdMap::unit(depth, child));
            }
        }

        match &*self.inner {
            Upper::Interface(i) => {
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

                let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
                for (from, to, replace_top) in shifts {
                    let kids_opt = if let Some((seg_top, ref seg_kids)) = seg_entry {
                        if *seg_top == from { Some(seg_kids) } else { None }
                    } else {
                        inner_children.unwrap().get(&from)
                    };
                    let Some(kids) = kids_opt else {
                        continue;
                    };

                    if replace_top {
                        for (depth, child) in kids.iter() {
                            insert_lower_child(&mut shifted_children, to.clone(), *depth, child.clone());
                        }
                    } else {
                        let mut pushed_children: Children<T, Lower<T>> = CompactMap::new();
                        pushed_children.insert(from, kids.clone());
                        let pushed_child = new_lower(pushed_children, false);
                        insert_lower_child(
                            &mut shifted_children,
                            to,
                            pushed_child.max_depth(),
                            pushed_child,
                        );
                    }
                }

                if shifted_children.is_empty() {
                    return Self::empty();
                }
                let shifted_root = new_lower(shifted_children, false);
                LeveledGSS {
                    inner: new_interface(shifted_root, i.acc.clone()),
                }
            }
            Upper::Branch(_) => {
                let shifted = shifts.into_iter().map(|(from, to, replace_top)| {
                    let base = self.isolate(Some(from));
                    if replace_top {
                        base.popn(1).push(to)
                    } else {
                        base.push(to)
                    }
                });
                Self::merge_many(shifted)
            }
        }
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

    pub fn absorb_vstack_same_acc(mut self, stack: &VirtualStack<T, A>) -> Self {
        let mut stack = stack.clone();
        stack.flush_pending();

        if stack.values.is_empty() {
            return self;
        }
        if self.is_empty() {
            return stack.into_gss();
        }

        let top = stack.values.last().unwrap().clone();
        let child_node = if stack.values.len() == 1 {
            stack.next.clone()
        } else {
            new_segment(stack.values.take(stack.values.len() - 1), stack.next.clone())
        };
        let child_depth = child_node.max_depth();

        let inner_mut = Arc::make_mut(&mut self.inner);
        if let Upper::Interface(self_iface_arc) = inner_mut {
            let iface_mut = Arc::make_mut(self_iface_arc);
            if iface_mut.acc == stack.acc {
                let lower_mut = Arc::make_mut(&mut iface_mut.inner);
                lower_mut.ensure_general();
                match lower_mut {
                    Lower::General { children, max_depth, .. } => {
                        if let Some(existing_ordmap) = children.get_mut(&top) {
                            match existing_ordmap.get(&child_depth).cloned() {
                                Some(existing_child) => {
                                    existing_ordmap.insert(child_depth, merge_lower(&existing_child, &child_node));
                                }
                                None => {
                                    existing_ordmap.insert(child_depth, child_node);
                                }
                            }
                        } else {
                            children.insert(top, CompactOrdMap::unit(child_depth, child_node));
                        }

                        if child_depth + 1 > *max_depth {
                            *max_depth = child_depth + 1;
                        }
                    }
                    Lower::Segment(_) => unreachable!(),
                }
                return self;
            }
        }

        self.merge(&stack.into_gss())
    }

    pub fn absorb_vstack_same_acc_owned(mut self, mut stack: VirtualStack<T, A>) -> Self {
        stack.flush_pending();

        if stack.values.is_empty() {
            return self;
        }
        if self.is_empty() {
            return stack.into_gss();
        }

        let top = stack.values.last().unwrap().clone();
        let child_node = if stack.values.len() == 1 {
            stack.next.clone()
        } else {
            new_segment(stack.values.take(stack.values.len() - 1), stack.next.clone())
        };
        let child_depth = child_node.max_depth();

        let inner_mut = Arc::make_mut(&mut self.inner);
        if let Upper::Interface(self_iface_arc) = inner_mut {
            let iface_mut = Arc::make_mut(self_iface_arc);
            if iface_mut.acc == stack.acc {
                let lower_mut = Arc::make_mut(&mut iface_mut.inner);
                lower_mut.ensure_general();
                match lower_mut {
                    Lower::General { children, max_depth, .. } => {
                        if let Some(existing_ordmap) = children.get_mut(&top) {
                            match existing_ordmap.get(&child_depth).cloned() {
                                Some(existing_child) => {
                                    existing_ordmap.insert(child_depth, merge_lower(&existing_child, &child_node));
                                }
                                None => {
                                    existing_ordmap.insert(child_depth, child_node);
                                }
                            }
                        } else {
                            children.insert(top, CompactOrdMap::unit(child_depth, child_node));
                        }

                        if child_depth + 1 > *max_depth {
                            *max_depth = child_depth + 1;
                        }
                    }
                    Lower::Segment(_) => unreachable!(),
                }
                return self;
            }
        }

        self.merge(&stack.into_gss())
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
                    let new_values = values.take(keep);
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


    pub fn pop(&self) -> Self {
        self.popn(1)
    }

    /// Fast path for a top-level interface whose alternatives all share the
    /// same base after popping one value.
    pub fn pop1_common_interface_base(&self) -> Option<Self> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };
        let Lower::General {
            children,
            empty: false,
            ..
        } = &*interface.inner
        else {
            return None;
        };
        if children.len() < 2 {
            return None;
        }

        let mut common_child: Option<Arc<Lower<T>>> = None;
        let mut common_child_id: Option<usize> = None;
        for kids in children.values() {
            if kids.len() != 1 {
                return None;
            }
            let child = kids.values().next().expect("single child");
            let child_id = lower_node_id(child);
            match common_child_id {
                None => {
                    common_child = Some(child.clone());
                    common_child_id = Some(child_id);
                }
                Some(id) if id == child_id => {}
                Some(_) => return None,
            }
        }

        Some(Self {
            inner: new_interface(common_child?, interface.acc.clone()),
        })
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
        Some(VirtualStack { values, next, acc: interface.acc.clone(), pending_top: None })
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
        let mut lower_general_nodes = 0usize;
        let mut lower_segment_nodes = 0usize;
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
            match &*node {
                Lower::Segment(_) => lower_segment_nodes += 1,
                Lower::General { .. } => lower_general_nodes += 1,
            }
            // Walk through this node and any owned segment chain below it.
            let mut current: &Lower<T> = &*node;
            loop {
                match current {
                    Lower::Segment(seg) => {
                        total_edges += 1; // One edge from this Segment to its next
                        match &*seg.next {
                            Lower::Segment(inner_seg) => {
                                let inner_id = Arc::as_ptr(inner_seg) as usize;
                                if !visited_lower.insert(inner_id) { break; }
                                lower_nodes += 1;
                                lower_segment_nodes += 1;
                                current = &*seg.next;
                            }
                            Lower::General { children, .. } => {
                                lower_nodes += 1;
                                lower_general_nodes += 1;
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
            lower_general_nodes,
            lower_segment_nodes,
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
        if self.is_empty() { return other.clone(); }
        if other.is_empty() { return self.clone(); }
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

    pub fn single_path_top_first_and_acc(&self, out: &mut SmallVec<[T; 16]>) -> Option<A> {
        fn push_lower_path<T>(node: &Arc<Lower<T>>, out: &mut SmallVec<[T; 16]>) -> bool
        where
            T: Clone + Eq + Hash,
        {
            match &**node {
                Lower::Segment(seg) => {
                    for value in seg.values.iter().rev() {
                        out.push(value.clone());
                    }
                    push_lower_path(&seg.next, out)
                }
                Lower::General { children, empty, .. } => {
                    if *empty {
                        return children.is_empty();
                    }
                    if children.len() != 1 {
                        return false;
                    }
                    let (value, kids) = children.iter().next().unwrap();
                    if kids.len() != 1 {
                        return false;
                    }
                    out.push(value.clone());
                    push_lower_path(kids.values().next().unwrap(), out)
                }
            }
        }

        fn push_upper_path<T, A>(
            node: &Arc<Upper<T, A>>,
            out: &mut SmallVec<[T; 16]>,
        ) -> Option<A>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            match &**node {
                Upper::Interface(interface) => {
                    if push_lower_path(&interface.inner, out) {
                        Some(interface.acc.clone())
                    } else {
                        None
                    }
                }
                Upper::Branch(branch) => {
                    if branch.empty.is_some() || branch.children.len() != 1 {
                        return None;
                    }
                    let (value, kids) = branch.children.iter().next().unwrap();
                    if kids.len() != 1 {
                        return None;
                    }
                    out.push(value.clone());
                    push_upper_path(kids.values().next().unwrap(), out)
                }
            }
        }

        out.clear();
        let start_len = out.len();
        match push_upper_path(&self.inner, out) {
            Some(acc) => Some(acc),
            None => {
                out.truncate(start_len);
                None
            }
        }
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

}

#[cfg(test)]
mod tests {
    use super::LeveledGSS;
    use super::Merge;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestAcc(u32);

    impl Merge for TestAcc {
        fn merge(&self, other: &Self) -> Self {
            Self(self.0.max(other.0))
        }
    }

    #[test]
    fn apply_shared_pop_push_branches_matches_virtual_stack_branch_builder() {
        let gss = LeveledGSS::from_single_stack(vec![10_u32, 20, 30, 40], TestAcc(1));
        let pushes = [vec![50_u32, 60], vec![70_u32, 80], vec![90_u32, 60]];

        let expected = gss
            .try_virtual_stack()
            .unwrap()
            .into_gss_after_popping_and_pushing_branches(2, pushes.iter().map(|push| push.as_slice()))
            .unwrap();
        let actual = gss
            .apply_shared_pop_push_branches(2, pushes.iter().map(|push| push.as_slice()))
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual.to_stacks(), expected.to_stacks());
    }

    #[test]
    fn apply_shared_pop_push_single_branches_deduplicates_targets() {
        let gss = LeveledGSS::from_single_stack(vec![10_u32, 20, 30, 40], TestAcc(1));
        let targets = [60_u32, 70, 60];

        let expected = LeveledGSS::from_stacks(&[
            (vec![10_u32, 20, 60], TestAcc(1)),
            (vec![10_u32, 20, 70], TestAcc(1)),
        ]);
        let actual = gss
            .apply_shared_pop_push_single_branches(2, targets.iter())
            .unwrap();

        let actual_stacks = actual.to_stacks();
        let expected_stacks = expected.to_stacks();
        assert_eq!(actual_stacks.len(), expected_stacks.len());
        for expected_stack in expected_stacks {
            assert!(actual_stacks.contains(&expected_stack));
        }
    }

    #[test]
    fn selective_top_pure_shift_extracts_one_shared_prefix_path() {
        let acc = TestAcc(7);
        let gss = LeveledGSS::from_stacks(&[
            (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 133], acc.clone()),
        ]);

        let shifted = gss
            .try_apply_selective_top_pure_shifts([(131_u32, 96_u32, false)])
            .unwrap();

        assert_eq!(
            shifted.to_stacks(),
            vec![(vec![0_u32, 1, 17, 47, 74, 131, 96], acc)]
        );
    }

    #[test]
    fn generic_top_pure_shift_matches_selective_shared_prefix_shape() {
        let acc = TestAcc(7);
        let gss = LeveledGSS::from_stacks(&[
            (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 133], acc.clone()),
        ]);

        let shifted = gss.apply_top_pure_shifts([(131_u32, 96_u32, false)]);

        assert_eq!(
            shifted.to_stacks(),
            vec![(vec![0_u32, 1, 17, 47, 74, 131, 96], acc)]
        );
    }

    #[test]
    #[ignore]
    fn bench_generic_top_pure_shift_shared_prefix_shape() {
        let acc = TestAcc(7);
        let gss = LeveledGSS::from_stacks(&[
            (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 133], acc),
        ]);

        let iterations = 100_000u32;
        let start = std::time::Instant::now();
        let mut shifted = None;
        for _ in 0..iterations {
            shifted = Some(std::hint::black_box(&gss).apply_top_pure_shifts(std::hint::black_box([
                (131_u32, 96_u32, false),
            ])));
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / u128::from(iterations);
        let shifted = shifted.unwrap();

        println!(
            "generic_top_pure_shift_shared_prefix_shape: avg={}ns iterations={}",
            avg_ns, iterations
        );
        assert_eq!(
            shifted.to_stacks(),
            vec![(vec![0_u32, 1, 17, 47, 74, 131, 96], TestAcc(7))]
        );
    }
}
