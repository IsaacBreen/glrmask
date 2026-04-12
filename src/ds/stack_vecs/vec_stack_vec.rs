use std::hash::{Hash, Hasher};
use super::stack_vec::StackVec;

/// Stack vector backed by plain `Vec<T>`.
/// O(n) clone, O(1) access, O(1) amortized push.
/// `try_push` always succeeds.
#[derive(Clone, Debug)]
pub struct VecStackVec<T>(Vec<T>);

impl<T: Clone + Eq + Hash> VecStackVec<T> {
    /// Iterate from bottom to top.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    /// Get elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T: PartialEq> PartialEq for VecStackVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for VecStackVec<T> {}

impl<T: Hash> Hash for VecStackVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> Default for VecStackVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T: Clone + Eq + Hash + std::fmt::Debug> StackVec<T> for VecStackVec<T>
{
    #[inline]
    fn unit(val: T) -> Self {
        Self(vec![val])
    }

    #[inline]
    fn from_vec(v: Vec<T>) -> Self {
        Self(v)
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn last(&self) -> Option<&T> {
        self.0.last()
    }

    fn take(&self, n: usize) -> Self {
        let n = n.min(self.0.len());
        Self(self.0[..n].to_vec())
    }

    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.0.truncate(new_len);
    }

    #[inline]
    fn try_push(&mut self, val: T) -> bool {
        self.0.push(val);
        true
    }

    fn append(&self, other: &Self) -> Self {
        let mut v = Vec::with_capacity(self.0.len() + other.0.len());
        v.extend_from_slice(&self.0);
        v.extend_from_slice(&other.0);
        Self(v)
    }

    #[inline]
    fn to_vec(&self) -> Vec<T> {
        self.0.clone()
    }
}
