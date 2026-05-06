use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Index;

use rustc_hash::FxHashMap;
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::action::Action;
use crate::grammar::flat::{NonterminalID, TerminalID};

const INLINE_ROW_CAPACITY: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum SparseRow<K: Copy + Eq + Hash, V: Clone> {
    Inline(SmallVec<[(K, V); INLINE_ROW_CAPACITY]>),
    Large(FxHashMap<K, V>),
}

impl<K: Copy + Eq + Hash, V: Clone> Default for SparseRow<K, V> {
    fn default() -> Self {
        Self::Inline(SmallVec::new())
    }
}

impl<K: Copy + Eq + Hash, V: Clone> SparseRow<K, V> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Inline(entries) => entries.len(),
            Self::Large(entries) => entries.len(),
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find(|(entry_key, _)| entry_key == key)
                .map(|(_, value)| value),
            Self::Large(entries) => entries.get(key),
        }
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self {
            Self::Inline(entries) => entries
                .iter_mut()
                .find(|(entry_key, _)| entry_key == key)
                .map(|(_, value)| value),
            Self::Large(entries) => entries.get_mut(key),
        }
    }

    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self {
            Self::Inline(entries) => {
                for (entry_key, entry_value) in entries.iter_mut() {
                    if *entry_key == key {
                        return Some(std::mem::replace(entry_value, value));
                    }
                }
                if entries.len() < INLINE_ROW_CAPACITY {
                    entries.push((key, value));
                    None
                } else {
                    let mut large = FxHashMap::default();
                    for (entry_key, entry_value) in entries.drain(..) {
                        large.insert(entry_key, entry_value);
                    }
                    let previous = large.insert(key, value);
                    *self = Self::Large(large);
                    previous
                }
            }
            Self::Large(entries) => entries.insert(key, value),
        }
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            Self::Inline(entries) => {
                let position = entries.iter().position(|(entry_key, _)| entry_key == key)?;
                Some(entries.swap_remove(position).1)
            }
            Self::Large(entries) => entries.remove(key),
        }
    }

    #[inline]
    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    #[inline]
    pub(crate) fn iter(&self) -> SparseRowIter<'_, K, V> {
        match self {
            Self::Inline(entries) => SparseRowIter::Inline(entries.iter()),
            Self::Large(entries) => SparseRowIter::Large(entries.iter()),
        }
    }

    #[inline]
    pub(crate) fn keys(&self) -> SparseRowKeys<'_, K, V> {
        match self {
            Self::Inline(entries) => SparseRowKeys::Inline(entries.iter()),
            Self::Large(entries) => SparseRowKeys::Large(entries.keys()),
        }
    }

    #[inline]
    pub(crate) fn values(&self) -> SparseRowValues<'_, K, V> {
        match self {
            Self::Inline(entries) => SparseRowValues::Inline(entries.iter()),
            Self::Large(entries) => SparseRowValues::Large(entries.values()),
        }
    }
}

impl<K: Copy + Eq + Hash, V: Clone + PartialEq> PartialEq for SparseRow<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<K: Copy + Eq + Hash, V: Clone + Eq> Eq for SparseRow<K, V> {}

impl<K, V> Serialize for SparseRow<K, V>
where
    K: Copy + Eq + Hash + Serialize,
    V: Clone + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, value) in self.iter() {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de, K, V> Deserialize<'de> for SparseRow<K, V>
where
    K: Copy + Eq + Hash + Deserialize<'de>,
    V: Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SparseRowVisitor<K, V>(PhantomData<(K, V)>);

        impl<'de, K, V> Visitor<'de> for SparseRowVisitor<K, V>
        where
            K: Copy + Eq + Hash + Deserialize<'de>,
            V: Clone + Deserialize<'de>,
        {
            type Value = SparseRow<K, V>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a sparse row map")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut row = SparseRow::default();
                while let Some((key, value)) = map.next_entry()? {
                    row.insert(key, value);
                }
                Ok(row)
            }
        }

        deserializer.deserialize_map(SparseRowVisitor::<K, V>(PhantomData))
    }
}

impl<'a, K: Copy + Eq + Hash, V: Clone> IntoIterator for &'a SparseRow<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = SparseRowIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Copy + Eq + Hash, V: Clone> Index<&K> for SparseRow<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).expect("sparse row index missing key")
    }
}

impl<K: Copy + Eq + Hash, V: Clone> FromIterator<(K, V)> for SparseRow<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut row = Self::default();
        for (key, value) in iter {
            row.insert(key, value);
        }
        row
    }
}

pub(crate) enum SparseRowIter<'a, K: Copy + Eq + Hash, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Iter<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(key, value)| (key, value)),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub(crate) enum SparseRowKeys<'a, K: Copy + Eq + Hash, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Keys<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowKeys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(key, _)| key),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub(crate) enum SparseRowValues<'a, K: Copy + Eq + Hash, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Values<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowValues<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(_, value)| value),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub(crate) type ActionRow = SparseRow<TerminalID, Action>;
pub(crate) type GotoRow = SparseRow<NonterminalID, (u32, bool)>;