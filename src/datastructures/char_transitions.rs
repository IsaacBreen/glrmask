use crate::datastructures::u8set::U8Set;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use std::fmt;
use std::ops::{Index, IndexMut};

/// A compact map from `u8` keys to values, optimized for the case where
/// each state has only a small number of outgoing transitions.
///
/// Internally this is just a `Vec<(u8, T)>`. All operations are linear in the
/// number of entries, but in DFA states this number is typically tiny.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct CharTransitions<T> {
    entries: Vec<(u8, T)>,
}

impl<T> CharTransitions<T> {
    /// Construct an empty map.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Construct from a vector of entries that is already sorted by key.
    pub fn from_sorted_entries(entries: Vec<(u8, T)>) -> Self {
        Self { entries }
    }

    /// Number of key-value pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the map empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Insert a value for `key`, returning the old value if any.
    pub fn insert(&mut self, key: u8, value: T) -> Option<T> {
        match self.entries.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(index) => Some(std::mem::replace(&mut self.entries[index].1, value)),
            Err(index) => {
                self.entries.insert(index, (key, value));
                None
            }
        }
    }

    /// Get a shared reference to the value for `key`, if present.
    pub fn get(&self, key: u8) -> Option<&T> {
        self.entries
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(|index| &self.entries[index].1)
    }

    /// Get a mutable reference to the value for `key`, if present.
    pub fn get_mut(&mut self, key: u8) -> Option<&mut T> {
        self.entries
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(move |index| &mut self.entries[index].1)
    }

    /// Test if the map contains a given key.
    pub fn contains_key(&self, key: u8) -> bool {
        self.get(key).is_some()
    }

    /// Iterator over `(key, &value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u8, &T)> {
        self.entries.iter().map(|(k, v)| (*k, v))
    }

    /// Mutable iterator over `(key, &mut value)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u8, &mut T)> {
        self.entries.iter_mut().map(|(k, v)| (*k, v))
    }

    /// Iterator over values.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().map(|(_, v)| v)
    }

    /// Return a `U8Set` containing all keys.
    pub fn keys_as_u8set(&self) -> U8Set {
        let mut set = U8Set::none();
        for (k, _) in &self.entries {
            set.insert(*k);
        }
        set
    }
}

impl<T> Index<u8> for CharTransitions<T> {
    type Output = T;

    fn index(&self, key: u8) -> &Self::Output {
        self.get(key).expect("Key not found")
    }
}

impl<T> IndexMut<u8> for CharTransitions<T> {
    fn index_mut(&mut self, key: u8) -> &mut Self::Output {
        self.get_mut(key).expect("Key not found")
    }
}

/// Borrowed iterator.
pub struct CharTransitionsIter<'a, T> {
    inner: std::slice::Iter<'a, (u8, T)>,
}

impl<'a, T> Iterator for CharTransitionsIter<'a, T> {
    type Item = (u8, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }
}

/// Mutable borrowed iterator.
pub struct CharTransitionsIterMut<'a, T> {
    inner: std::slice::IterMut<'a, (u8, T)>,
}

impl<'a, T> Iterator for CharTransitionsIterMut<'a, T> {
    type Item = (u8, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }
}

impl<'a, T> IntoIterator for &'a CharTransitions<T> {
    type Item = (u8, &'a T);
    type IntoIter = CharTransitionsIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        CharTransitionsIter {
            inner: self.entries.iter(),
        }
    }
}

impl<'a, T> IntoIterator for &'a mut CharTransitions<T> {
    type Item = (u8, &'a mut T);
    type IntoIter = CharTransitionsIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        CharTransitionsIterMut {
            inner: self.entries.iter_mut(),
        }
    }
}

impl<T> Extend<(u8, T)> for CharTransitions<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (u8, T)>,
    {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<T> FromIterator<(u8, T)> for CharTransitions<T> {
    fn from_iter<I: IntoIterator<Item = (u8, T)>>(iter: I) -> Self {
        let mut map = CharTransitions::new();
        map.extend(iter);
        map
    }
}

impl<T: fmt::Debug> fmt::Debug for CharTransitions<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dm = f.debug_map();
        for (k, v) in &self.entries {
            dm.entry(k, v);
        }
        dm.finish()
    }
}

impl<T: JSONConvertible> JSONConvertible for CharTransitions<T> {
    fn to_json(&self) -> JSONNode {
        // Sparse array format: [[byte_val, target], ...]
        let pairs = self
            .entries
            .iter()
            .map(|(k, v)| JSONNode::Array(vec![JSONNode::UInt(*k as u128), v.to_json()]))
            .collect();
        JSONNode::Array(pairs)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut entries = Vec::with_capacity(arr.len());
                for (i, pair_node) in arr.into_iter().enumerate() {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let val_node = pair_vec.pop().unwrap();
                            let key_node = pair_vec.pop().unwrap();
                            let key = u8::from_json(key_node).map_err(|e| {
                                format!(
                                    "While deserializing CharTransitions at $[{}][0] (key): {}",
                                    i, e
                                )
                            })?;
                            let value = T::from_json(val_node).map_err(|e| {
                                format!(
                                    "While deserializing CharTransitions at $[{}][1] (value): {}",
                                    i, e
                                )
                            })?;
                            entries.push((key, value));
                        }
                        other => {
                            return Err(format!(
                                "Expected 2-element array for CharTransitions entry at $[{}], got {}",
                                i,
                                other.short_preview()
                            ))
                        }
                    }
                }
                // Entries should already be sorted from serialization, but verify/sort just in case
                entries.sort_by_key(|(k, _)| *k);
                Ok(CharTransitions::from_sorted_entries(entries))
            }
            _ => Err("Expected JSONNode::Array for CharTransitions".to_string()),
        }
    }
}

