use crate::json_serialization::{JSONConvertible, JSONNode};
// Added
use std::collections::{BTreeSet, HashSet};
use std::hash::{Hash, Hasher};
// Added for T bound in JSONConvertible for HashSet

/// A frozen set implementation in Rust, similar to Python's frozenset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenSet<T: Eq + Ord + Hash> {
    inner: HashSet<T>,
}

impl<T: Eq + Ord + Hash> Hash for FrozenSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut members: Vec<_> = self.inner.iter().collect();
        members.sort_unstable(); // Requires T: Ord
        for member in members {
            member.hash(state);
        }
    }
}

impl<T: Eq + Ord + Hash + JSONConvertible> JSONConvertible for FrozenSet<T> {
    fn to_json(&self) -> JSONNode {
        let mut original_members: Vec<_> = self.inner.iter().collect();
        original_members.sort_unstable();
        JSONNode::Array(
            original_members
                .iter()
                .map(|item| (*item).to_json())
                .collect(),
        )
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.into_iter().map(T::from_json).collect::<Result<HashSet<T>, String>>()
                                      .map(|inner| FrozenSet { inner }),
            _ => Err("Expected JSONNode::Array for FrozenSet".to_string()),
        }
    }
}


impl<T: Eq + Ord + Hash> FrozenSet<T> {
    /// Creates a new empty FrozenSet.
    pub fn new() -> Self {
        let inner = HashSet::new();
        FrozenSet { inner }
    }

    /// Constructs a FrozenSet from an iterator.
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let inner = HashSet::from_iter(iter);
        FrozenSet { inner }
    }

    /// Checks if the set contains a value.
    pub fn contains(&self, value: &T) -> bool {
        self.inner.contains(value)
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
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
        Self::from_iter(iter) // Calls the inherent method
    }
}

impl<T: Eq + Ord + Hash> From<BTreeSet<T>> for FrozenSet<T> {
    fn from(inner: BTreeSet<T>) -> Self {
        FrozenSet { inner: inner.into_iter().collect() }
    }
}

/// Extension trait for BTreeSet to allow conversion into a FrozenSet.
pub trait Freeze<T: Eq + Ord + Hash> {
    fn freeze(self) -> FrozenSet<T>;
}

impl<T: Eq + Ord + Hash> Freeze<T> for BTreeSet<T> {
    fn freeze(self) -> FrozenSet<T> {
        FrozenSet::from_iter(self)
    }
}

/// Unfreeze a FrozenSet into a BTreeSet.
pub trait Unfreeze<T: Eq + Ord + Hash> {
    fn unfreeze_to_btree_set(self) -> BTreeSet<T>;
}

impl<T: Eq + Ord + Hash> Unfreeze<T> for FrozenSet<T> {
    fn unfreeze_to_btree_set(self) -> BTreeSet<T> {
        self.inner.into_iter().collect()
    }
}

/// An iterator over the elements of a `FrozenSet`.
pub struct Iter<'a, T: 'a> {
    inner: std::collections::hash_set::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T: Eq + Ord + Hash> IntoIterator for &'a FrozenSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            inner: self.inner.iter(),
        }
    }
}

/// An iterator over the elements of a `FrozenSet`.
pub struct IntoIter<T> {
    inner: std::collections::hash_set::IntoIter<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T: Eq + Ord + Hash> IntoIterator for FrozenSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<T: Eq + Ord + Hash> FrozenSet<T> {
    pub fn iter(&self) -> Iter<T> {
        Iter {
            inner: self.inner.iter(),
        }
    }

    pub fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_freeze_btreeset() {
        let hs: BTreeSet<i32> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let fs = hs.freeze();
        assert_eq!(fs.len(), 5);
        for i in 1..=5 {
            assert!(fs.contains(&i));
        }
    }
}



