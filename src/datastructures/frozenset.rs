use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

/// A frozen set implementation in Rust, similar to Python's `frozenset`,
/// backed by a sorted, deduplicated boxed slice. This is much more compact
/// than a full `BTreeSet` while preserving set semantics.
#[derive(Debug, Clone, Eq, PartialOrd, Ord)]
pub struct FrozenSet<T: Eq + Ord + Hash> {
    inner: Box<[T]>,
    hash: u64,
}

impl<T: Eq + Ord + Hash> PartialEq for FrozenSet<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        self.inner == other.inner
    }
}

impl<T: Eq + Ord + Hash> Hash for FrozenSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl<T: Eq + Ord + JSONConvertible + Clone + Hash> JSONConvertible for FrozenSet<T> {
    fn to_json(&self) -> JSONNode {
        // Preserve the previous JSON format by round-tripping through BTreeSet.
        let tmp: BTreeSet<T> = self.inner.iter().cloned().collect();
        tmp.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        BTreeSet::<T>::from_json(node).map(FrozenSet::from)
    }
}

impl<T: Eq + Ord + Hash> FrozenSet<T> {
    fn calculate_hash(slice: &[T]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        slice.hash(&mut hasher);
        hasher.finish()
    }

    /// Creates a new empty `FrozenSet`.
    pub fn new() -> Self {
        let inner: Box<[T]> = Box::new([]);
        let hash = Self::calculate_hash(&inner);
        FrozenSet {
            inner,
            hash,
        }
    }

    /// Constructs a `FrozenSet` from an iterator.
    ///
    /// The resulting set is sorted and deduplicated.
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec: Vec<T> = iter.into_iter().collect();
        vec.sort();
        vec.dedup();
        let inner = vec.into_boxed_slice();
        let hash = Self::calculate_hash(&inner);
        FrozenSet {
            inner,
            hash,
        }
    }

    /// Creates a `FrozenSet` from a vector that is already sorted and deduplicated.
    pub fn new_unchecked(vec: Vec<T>) -> Self {
        let inner = vec.into_boxed_slice();
        let hash = Self::calculate_hash(&inner);
        FrozenSet { inner, hash }
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
    T: Eq + Ord + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Ord + Hash> FromIterator<T> for FrozenSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        FrozenSet::from_iter(iter)
    }
}

impl<T: Eq + Ord + Hash> From<BTreeSet<T>> for FrozenSet<T> {
    fn from(inner: BTreeSet<T>) -> Self {
        // BTreeSet already stores elements in sorted, unique order, so we can
        // skip the extra sort/dedup work performed by `from_iter`. This is
        // especially important in hot paths like NFA -> DFA subset construction
        // where we repeatedly build FrozenSets from BTreeSets of state IDs.
        let vec: Vec<T> = inner.into_iter().collect();
        let inner = vec.into_boxed_slice();
        let hash = Self::calculate_hash(&inner);
        FrozenSet {
            inner,
            hash,
        }
    }
}

/// Extension trait for `BTreeSet` to allow conversion into a `FrozenSet`.
pub trait FreezeBTreeSet<T: Eq + Ord + Hash> {
    fn freeze(self) -> FrozenSet<T>;
}

impl<T: Eq + Ord + Hash> FreezeBTreeSet<T> for BTreeSet<T> {
    fn freeze(self) -> FrozenSet<T> {
        FrozenSet::from(self)
    }
}

/// Unfreeze a `FrozenSet` into a `BTreeSet`.
pub trait UnfreezeBTreeSet<T: Eq + Ord + Hash> {
    fn unfreeze(self) -> BTreeSet<T>;
}

impl<T: Eq + Ord + Hash> UnfreezeBTreeSet<T> for FrozenSet<T> {
    fn unfreeze(self) -> BTreeSet<T> {
        self.inner.into_vec().into_iter().collect()
    }
}

/// An iterator over the elements of a `FrozenSet`.
pub struct Iter<'a, T: 'a + Hash + Eq + Ord> {
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T: Hash + Eq + Ord> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T: Eq + Ord + Hash> IntoIterator for &'a FrozenSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An owning iterator over the elements of a `FrozenSet`.
pub struct IntoIter<T: Hash + Eq + Ord> {
    inner: std::vec::IntoIter<T>,
}

impl<T: Hash + Eq + Ord> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T: Eq + Ord + Hash> IntoIterator for FrozenSet<T> {
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
