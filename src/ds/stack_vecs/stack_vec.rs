use std::fmt::Debug;
use std::hash::Hash;

/// Common interface for stack-like vectors used in GSS segments.
///
/// Implementations provide different trade-offs:
/// - `ArrayStackVec`: Fixed-capacity inline array. O(n) clone, O(1) access.
/// - `ImStackVec`: im::Vector tree. O(1) clone, O(log n) access.
/// - `SegVec`: Arc<[T]> view. O(1) clone, O(1) access, O(n) push.
/// - `ArcArrayVec`: Arc<Vec<T>> with window. O(1) clone, O(1) access, COW push.
/// - `VecStackVec`: Plain Vec. O(n) clone, O(1) access.
/// - `SmallStackVec`: SmallVec inline/heap hybrid. O(n) clone, O(1) access.
/// - `RpdsStackVec`: rpds::Stack persistent list. O(1) clone/push, O(n) access.
pub trait StackVec<T>: Clone + PartialEq + Eq + Hash + Debug + Default
where
    T: Clone + Eq + Hash,
{
    /// Create from a single value.
    fn unit(val: T) -> Self;

    /// Create from a Vec (consumes it).
    fn from_vec(v: Vec<T>) -> Self;

    /// Number of elements.
    fn len(&self) -> usize;

    /// Whether empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Last element (top of stack).
    fn last(&self) -> Option<&T>;

    /// Create a prefix view of the first `n` elements.
    /// Complexity varies by implementation: O(1) for view-based, O(n) for copy-based.
    fn take(&self, n: usize) -> Self;

    /// Truncate in place to `new_len` elements.
    fn truncate(&mut self, new_len: usize);

    /// Try to push a value onto the top.
    /// Returns `true` if successful (push was performed in-place).
    /// Returns `false` if the push cannot be done (e.g., shared backing data, at capacity).
    /// When `false`, the caller should create a new segment and push there.
    fn try_push(&mut self, val: T) -> bool;

    /// Append `other`'s elements on top of `self`, producing a new instance.
    fn append(&self, other: &Self) -> Self;

    /// Convert to a Vec. O(n).
    fn to_vec(&self) -> Vec<T>;
}
