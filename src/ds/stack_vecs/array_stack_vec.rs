use std::hash::{Hash, Hasher};
use arrayvec::ArrayVec;
use super::stack_vec::StackVec;

/// Stack vector backed by `ArrayVec<T, CAP>`.
/// O(n) clone (memcpy), O(1) access, fixed capacity.
/// `try_push` returns false when at capacity.
#[derive(Clone, Debug)]
pub struct ArrayStackVec<T, const CAP: usize>(ArrayVec<T, CAP>);

impl<T: Clone + Eq + Hash, const CAP: usize> ArrayStackVec<T, CAP> {
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

impl<T: PartialEq, const CAP: usize> PartialEq for ArrayStackVec<T, CAP> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_slice() == other.0.as_slice()
    }
}

impl<T: Eq, const CAP: usize> Eq for ArrayStackVec<T, CAP> {}

impl<T: Hash, const CAP: usize> Hash for ArrayStackVec<T, CAP> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_slice().hash(state);
    }
}

impl<T, const CAP: usize> Default for ArrayStackVec<T, CAP> {
    fn default() -> Self {
        Self(ArrayVec::new())
    }
}

impl<T: Clone + Eq + Hash, const CAP: usize> StackVec<T> for ArrayStackVec<T, CAP>
{
    #[inline]
    fn unit(val: T) -> Self {
        let mut av = ArrayVec::new();
        av.push(val);
        Self(av)
    }

    fn from_vec(v: Vec<T>) -> Self {
        let mut av = ArrayVec::new();
        for item in v {
            av.push(item);
        }
        Self(av)
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
        let mut av = ArrayVec::new();
        for item in &self.0[..n] {
            av.push(item.clone());
        }
        Self(av)
    }

    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.0.truncate(new_len);
    }

    #[inline]
    fn try_push(&mut self, val: T) -> bool {
        if self.0.remaining_capacity() > 0 {
            self.0.push(val);
            true
        } else {
            false
        }
    }

    fn append(&self, other: &Self) -> Self {
        let mut av = ArrayVec::new();
        for item in self.0.iter() {
            av.push(item.clone());
        }
        for item in other.0.iter() {
            av.push(item.clone());
        }
        Self(av)
    }

    fn to_vec(&self) -> Vec<T> {
        self.0.to_vec()
    }
}

// Type aliases for common sizes
pub type ArrayStackVec4<T> = ArrayStackVec<T, 4>;
pub type ArrayStackVec8<T> = ArrayStackVec<T, 8>;
pub type ArrayStackVec16<T> = ArrayStackVec<T, 16>;
pub type ArrayStackVec32<T> = ArrayStackVec<T, 32>;
pub type ArrayStackVec64<T> = ArrayStackVec<T, 64>;
pub type ArrayStackVec128<T> = ArrayStackVec<T, 128>;
pub type ArrayStackVec256<T> = ArrayStackVec<T, 256>;
