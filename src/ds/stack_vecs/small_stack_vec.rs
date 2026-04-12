use std::hash::{Hash, Hasher};
use smallvec::SmallVec;
use super::stack_vec::StackVec;

macro_rules! define_small_stack_vec {
    ($name:ident, $n:literal) => {
        /// Stack vector backed by `SmallVec` with inline storage.
        /// O(n) clone, O(1) access, O(1) amortized push.
        /// `try_push` always succeeds.
        #[derive(Clone, Debug)]
        pub struct $name<T>(SmallVec<[T; $n]>);

        impl<T> $name<T> {
            #[inline]
            pub fn iter(&self) -> std::slice::Iter<'_, T> { self.0.iter() }
            #[inline]
            pub fn as_slice(&self) -> &[T] { self.0.as_slice() }
        }

        impl<T: PartialEq> PartialEq for $name<T> {
            fn eq(&self, other: &Self) -> bool { self.0.as_slice() == other.0.as_slice() }
        }
        impl<T: Eq> Eq for $name<T> {}
        impl<T: Hash> Hash for $name<T> {
            fn hash<H: Hasher>(&self, state: &mut H) { self.0.as_slice().hash(state); }
        }
        impl<T> Default for $name<T> {
            fn default() -> Self { Self(SmallVec::new()) }
        }

        impl<T: Clone + Eq + Hash> StackVec<T> for $name<T> {
            #[inline]
            fn unit(val: T) -> Self {
                let mut sv = SmallVec::new();
                sv.push(val);
                Self(sv)
            }
            fn from_vec(v: Vec<T>) -> Self { Self(SmallVec::from_vec(v)) }
            #[inline]
            fn len(&self) -> usize { self.0.len() }
            #[inline]
            fn last(&self) -> Option<&T> { self.0.last() }
            fn take(&self, n: usize) -> Self {
                let n = n.min(self.0.len());
                Self(self.0[..n].iter().cloned().collect())
            }
            #[inline]
            fn truncate(&mut self, new_len: usize) { self.0.truncate(new_len); }
            #[inline]
            fn try_push(&mut self, val: T) -> bool { self.0.push(val); true }
            fn append(&self, other: &Self) -> Self {
                let mut sv = SmallVec::with_capacity(self.0.len() + other.0.len());
                sv.extend(self.0.iter().cloned());
                sv.extend(other.0.iter().cloned());
                Self(sv)
            }
            fn to_vec(&self) -> Vec<T> { self.0.to_vec() }
        }
    };
}

define_small_stack_vec!(SmallStackVec32, 32);
define_small_stack_vec!(SmallStackVec64, 64);
define_small_stack_vec!(SmallStackVec128, 128);
