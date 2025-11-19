use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeSet;

/// A frozen set implementation in Rust, similar to Python's `frozenset`,
/// backed by a sorted, deduplicated boxed slice. This is much more compact
/// than a full `BTreeSet` while preserving set semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrozenSet<T: Eq + Ord> {
    inner: Box<[T]>,
}

impl<T: Eq + Ord + JSONConvertible + Clone> JSONConvertible for FrozenSet<T> {
    fn to_json(&self) -> JSONNode {
        // Preserve the previous JSON format by round-tripping through BTreeSet.
        let tmp: BTreeSet<T> = self.inner.iter().cloned().collect();
        tmp.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        BTreeSet::<T>::from_json(node).map(FrozenSet::from)
    }
}

impl<T: Eq + Ord> FrozenSet<T> {
    /// Creates a new empty `FrozenSet`.
    pub fn new() -> Self {
        FrozenSet {
            inner: Box::new([]),
        }
    }

    /// Constructs a `FrozenSet` from an iterator.
    ///
    /// The resulting set is sorted and deduplicated.
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec: Vec<T> = iter.into_iter().collect();
        vec.sort();
        vec.dedup();
        FrozenSet {
            inner: vec.into_boxed_slice(),
        }
    }

    /// Checks if the set contains a value.
    pub fn contains(&self, value: &T) -> bool {
        self.inner.binary_search(value).is_ok()
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns an iterator over the elements of the set.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.inner.iter(),
        }
    }

    /// Returns an owning iterator over the elements of the set.
    pub fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            inner: self.inner.into_vec().into_iter(),
        }
    }
}

impl<T> Default for FrozenSet<T>
where
    T: Eq + Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Ord> FromIterator<T> for FrozenSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        FrozenSet::from_iter(iter)
    }
}

impl<T: Eq + Ord> From<BTreeSet<T>> for FrozenSet<T> {
    fn from(inner: BTreeSet<T>) -> Self {
        // BTreeSet already stores elements in sorted, unique order, so we can
        // skip the extra sort/dedup work performed by `from_iter`. This is
        // especially important in hot paths like NFA -> DFA subset construction
        // where we repeatedly build FrozenSets from BTreeSets of state IDs.
        let vec: Vec<T> = inner.into_iter().collect();
        FrozenSet {
            inner: vec.into_boxed_slice(),
        }
    }
}

/// Extension trait for `BTreeSet` to allow conversion into a `FrozenSet`.
pub trait FreezeBTreeSet<T: Eq + Ord> {
    fn freeze(self) -> FrozenSet<T>;
}

impl<T: Eq + Ord> FreezeBTreeSet<T> for BTreeSet<T> {
    fn freeze(self) -> FrozenSet<T> {
        FrozenSet::from(self)
    }
}

/// Unfreeze a `FrozenSet` into a `BTreeSet`.
pub trait UnfreezeBTreeSet<T: Eq + Ord> {
    fn unfreeze(self) -> BTreeSet<T>;
}

impl<T: Eq + Ord> UnfreezeBTreeSet<T> for FrozenSet<T> {
    fn unfreeze(self) -> BTreeSet<T> {
        self.inner.into_vec().into_iter().collect()
    }
}

/// An iterator over the elements of a `FrozenSet`.
pub struct Iter<'a, T: 'a> {
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T: Eq + Ord> IntoIterator for &'a FrozenSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An owning iterator over the elements of a `FrozenSet`.
pub struct IntoIter<T> {
    inner: std::vec::IntoIter<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T: Eq + Ord> IntoIterator for FrozenSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_serialization::JSONConvertible;

    #[test]
    fn test_empty_set() {
        let fs: FrozenSet<i32> = FrozenSet::new();
        assert!(fs.is_empty());
    }

    #[test]
    fn test_from_iter() {
        let data = [1, 2, 3, 4, 5];
        let fs = FrozenSet::from_iter(data.iter().cloned());
        assert_eq!(fs.len(), 5);
        for i in 1..=5 {
            assert!(fs.contains(&i));
        }
    }

    #[test]
    fn test_contains() {
        let fs = FrozenSet::from_iter(vec![1, 2, 3]);
        assert!(fs.contains(&1));
        assert!(!fs.contains(&4));
    }

    #[test]
    fn test_freeze_BTreeSet() {
        let hs: BTreeSet<i32> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let fs = hs.freeze();
        assert_eq!(fs.len(), 5);
        for i in 1..=5 {
            assert!(fs.contains(&i));
        }
    }

    #[test]
    fn test_freeze_btreeset() {
        let bs: BTreeSet<i32> = [5, 4, 3, 2, 1].iter().cloned().collect();
        let fs = bs.freeze();
        assert_eq!(fs.len(), 5);
        for i in 1..=5 {
            assert!(fs.contains(&i));
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let fs = FrozenSet::from_iter(vec![3, 1, 2]);
        let json = fs.to_json();
        let fs2 = FrozenSet::<i32>::from_json(json).unwrap();
        assert_eq!(fs, fs2);
    }
}
