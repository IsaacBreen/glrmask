use std::hash::{Hash, Hasher};
use im::Vector as ImVector;
use super::stack_vec::StackVec;

/// Stack vector backed by `im::Vector<T>`.
/// O(1) clone, O(log n) access, O(log n) push/pop.
/// `try_push` always succeeds.
#[derive(Clone, Debug)]
pub struct ImStackVec<T: Clone>(ImVector<T>);

impl<T: Clone + Eq + Hash> ImStackVec<T> {
    /// Iterate from bottom to top.
    #[inline]
    pub fn iter(&self) -> im::vector::Iter<'_, T> {
        self.0.iter()
    }
}

impl<T: Clone + PartialEq> PartialEq for ImStackVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Clone + Eq> Eq for ImStackVec<T> {}

impl<T: Clone + Hash> Hash for ImStackVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for item in self.0.iter() {
            item.hash(state);
        }
    }
}

impl<T: Clone> Default for ImStackVec<T> {
    fn default() -> Self {
        Self(ImVector::new())
    }
}

impl<T: Clone + Eq + Hash> StackVec<T> for ImStackVec<T>
{
    #[inline]
    fn unit(val: T) -> Self {
        let mut v = ImVector::new();
        v.push_back(val);
        Self(v)
    }

    fn from_vec(v: Vec<T>) -> Self {
        Self(v.into_iter().collect())
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
        Self(self.0.take(n))
    }

    fn truncate(&mut self, new_len: usize) {
        let new_len = new_len.min(self.0.len());
        self.0.truncate(new_len);
    }

    #[inline]
    fn try_push(&mut self, val: T) -> bool {
        self.0.push_back(val);
        true
    }

    fn append(&self, other: &Self) -> Self {
        let mut merged = self.0.clone();
        merged.append(other.0.clone());
        Self(merged)
    }

    fn to_vec(&self) -> Vec<T> {
        self.0.iter().cloned().collect()
    }
}
