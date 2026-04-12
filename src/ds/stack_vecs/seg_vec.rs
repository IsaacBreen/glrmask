use std::hash::{Hash, Hasher};
use std::sync::Arc;
use super::stack_vec::StackVec;

/// A persistent vector backed by an `Arc<[T]>` with a length view.
///
/// - **Clone**: O(1) — Arc::clone + copy len
/// - **last()**: O(1) — flat array index
/// - **take(k)**: O(1) — same backing data, truncated view
/// - **iter()**: O(1) to start — iterates over contiguous memory
/// - **push()**: O(n) — COW: always creates a new allocation
///
/// Designed for GSS segments where values are written once and then
/// shared/cloned many times during nondeterministic operations.
#[derive(Clone)]
pub struct SegVec<T> {
    data: Arc<[T]>,
    len: usize,
}

impl<T> SegVec<T> {
    /// Create an empty SegVec.
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Arc::from([]),
            len: 0,
        }
    }

    /// Number of elements in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a reference to the last element (top of stack).
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            Some(&self.data[self.len - 1])
        }
    }

    /// Get the elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    /// Iterate over the elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data[..self.len].iter()
    }

    /// Create a view of the first `n` elements. O(1) — shares backing data.
    #[inline]
    pub fn take(&self, n: usize) -> Self {
        Self {
            data: self.data.clone(),
            len: n.min(self.len),
        }
    }

    /// Truncate to `new_len` elements. O(1).
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.len = new_len.min(self.len);
    }
}

impl<T: Clone> SegVec<T> {
    /// Create from a single element.
    #[inline]
    pub fn unit(val: T) -> Self {
        Self {
            data: Arc::from([val]),
            len: 1,
        }
    }

    /// Create from a Vec.
    #[inline]
    pub fn from_vec(v: Vec<T>) -> Self {
        let len = v.len();
        Self {
            data: Arc::from(v.into_boxed_slice()),
            len,
        }
    }

    /// Append another SegVec's elements, producing a new SegVec.
    /// O(self.len + other.len).
    pub fn append(&self, other: &Self) -> Self {
        if other.len == 0 {
            return self.clone();
        }
        if self.len == 0 {
            return other.clone();
        }
        let mut v = Vec::with_capacity(self.len + other.len);
        v.extend_from_slice(&self.data[..self.len]);
        v.extend_from_slice(&other.data[..other.len]);
        Self::from_vec(v)
    }

    /// Push a value, producing a new SegVec. O(self.len + 1).
    pub fn push(&self, val: T) -> Self {
        let mut v = Vec::with_capacity(self.len + 1);
        v.extend_from_slice(&self.data[..self.len]);
        v.push(val);
        Self::from_vec(v)
    }

    /// Pop the last element, returning the value and a truncated view.
    /// The truncated view is O(1) — shares backing data.
    #[inline]
    pub fn pop(&self) -> Option<(T, Self)> {
        if self.len == 0 {
            None
        } else {
            let val = self.data[self.len - 1].clone();
            Some((val, self.take(self.len - 1)))
        }
    }

    /// Convert to a Vec (copies data). O(n).
    pub fn to_vec(&self) -> Vec<T> {
        self.data[..self.len].to_vec()
    }
}

impl<T: PartialEq> PartialEq for SegVec<T> {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: same backing + same view
        if Arc::ptr_eq(&self.data, &other.data) && self.len == other.len {
            return true;
        }
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for SegVec<T> {}

impl<T: Hash> Hash for SegVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SegVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Clone> FromIterator<T> for SegVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<T: Clone> Default for SegVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq + Hash> StackVec<T> for SegVec<T>
{
    #[inline]
    fn unit(val: T) -> Self {
        SegVec::unit(val)
    }

    #[inline]
    fn from_vec(v: Vec<T>) -> Self {
        SegVec::from_vec(v)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn last(&self) -> Option<&T> {
        SegVec::last(self)
    }

    #[inline]
    fn take(&self, n: usize) -> Self {
        SegVec::take(self, n)
    }

    #[inline]
    fn truncate(&mut self, new_len: usize) {
        SegVec::truncate(self, new_len);
    }

    /// SegVec cannot push in-place (Arc<[T]> is immutable). Always returns false.
    #[inline]
    fn try_push(&mut self, _val: T) -> bool {
        false
    }

    fn try_harder_push(&mut self, val: T) -> bool {
        // Materialize into a new Vec, push, and rebuild as Arc<[T]>.
        let mut v = self.to_vec();
        v.push(val);
        *self = SegVec::from_vec(v);
        true
    }

    fn append(&self, other: &Self) -> Self {
        SegVec::append(self, other)
    }

    fn to_vec(&self) -> Vec<T> {
        SegVec::to_vec(self)
    }
}
