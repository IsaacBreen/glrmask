use std::hash::{Hash, Hasher};
use std::sync::Arc;
use super::stack_vec::StackVec;

/// Stack vector backed by `Arc<Vec<T>>` with a window length.
///
/// - **Clone**: O(1) — Arc::clone + copy len
/// - **last()**: O(1) — flat array index
/// - **take(k)**: O(1) — same backing data, truncated window
/// - **try_push**: O(1) if sole owner and at data boundary, else fails
/// - **pop/truncate**: O(1) — just decrements window length
///
/// When `try_push` fails (shared data or window < data length),
/// the caller should create a new segment.
#[derive(Clone, Debug)]
pub struct ArcArrayVec<T> {
    data: Arc<Vec<T>>,
    nw: usize, // window length: only data[0..nw] is visible
}

impl<T> ArcArrayVec<T> {
    /// Create an empty ArcArrayVec.
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Arc::new(Vec::new()),
            nw: 0,
        }
    }

    /// Get elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.nw]
    }

    /// Iterate from bottom to top.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data[..self.nw].iter()
    }

    /// Number of elements in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.nw
    }

    /// Whether the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nw == 0
    }

    /// Last element (top of stack).
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.nw == 0 {
            None
        } else {
            Some(&self.data[self.nw - 1])
        }
    }

    /// O(1) — shares backing data, just adjusts window.
    #[inline]
    pub fn take(&self, n: usize) -> Self {
        Self {
            data: self.data.clone(),
            nw: n.min(self.nw),
        }
    }

    /// O(1) — just shrinks window.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.nw = new_len.min(self.nw);
    }
}

impl<T: Clone> ArcArrayVec<T> {
    /// Create from a single value.
    #[inline]
    pub fn unit(val: T) -> Self {
        Self {
            data: Arc::new(vec![val]),
            nw: 1,
        }
    }

    /// Create from a Vec.
    #[inline]
    pub fn from_vec(v: Vec<T>) -> Self {
        let nw = v.len();
        Self {
            data: Arc::new(v),
            nw,
        }
    }

    /// Append other's elements on top, producing a new instance.
    pub fn append(&self, other: &Self) -> Self {
        if other.nw == 0 {
            return self.clone();
        }
        if self.nw == 0 {
            return other.clone();
        }
        let mut v = Vec::with_capacity(self.nw + other.nw);
        v.extend_from_slice(&self.data[..self.nw]);
        v.extend_from_slice(&other.data[..other.nw]);
        Self::from_vec(v)
    }

    /// Convert to a Vec. O(n).
    pub fn to_vec(&self) -> Vec<T> {
        self.data[..self.nw].to_vec()
    }

    /// Try to push. Succeeds only if sole owner of backing data.
    #[inline]
    pub fn try_push(&mut self, val: T) -> bool {
        if Arc::strong_count(&self.data) != 1 {
            return false;
        }
        let data = Arc::make_mut(&mut self.data);
        if self.nw < data.len() {
            data.truncate(self.nw);
        }
        data.push(val);
        self.nw = data.len();
        true
    }
}

impl<T: PartialEq> PartialEq for ArcArrayVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.data, &other.data) && self.nw == other.nw {
            return true;
        }
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for ArcArrayVec<T> {}

impl<T: Hash> Hash for ArcArrayVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T> Default for ArcArrayVec<T> {
    fn default() -> Self {
        Self {
            data: Arc::new(Vec::new()),
            nw: 0,
        }
    }
}

impl<T: Clone + Eq + Hash> StackVec<T> for ArcArrayVec<T>
{
    #[inline]
    fn unit(val: T) -> Self { ArcArrayVec::unit(val) }
    #[inline]
    fn from_vec(v: Vec<T>) -> Self { ArcArrayVec::from_vec(v) }
    #[inline]
    fn len(&self) -> usize { self.nw }
    #[inline]
    fn last(&self) -> Option<&T> { ArcArrayVec::last(self) }
    #[inline]
    fn take(&self, n: usize) -> Self { ArcArrayVec::take(self, n) }
    #[inline]
    fn truncate(&mut self, new_len: usize) { ArcArrayVec::truncate(self, new_len) }
    #[inline]
    fn try_push(&mut self, val: T) -> bool { ArcArrayVec::try_push(self, val) }
    fn try_harder_push(&mut self, val: T) -> bool {
        // If shared, clone the backing data to make it uniquely owned, then push.
        if Arc::strong_count(&self.data) != 1 {
            let mut new_data = self.as_slice().to_vec();
            new_data.push(val);
            self.data = Arc::new(new_data);
            self.nw = self.data.len();
            return true;
        }
        ArcArrayVec::try_push(self, val)
    }
    fn append(&self, other: &Self) -> Self { ArcArrayVec::append(self, other) }
    fn to_vec(&self) -> Vec<T> { ArcArrayVec::to_vec(self) }
}
