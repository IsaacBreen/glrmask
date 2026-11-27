//! Sparse nonempty weighted relations, with inline storage for singleton rows.
use std::collections::hash_map::{Entry, IntoIter};
use std::hash::Hash;
use rustc_hash::FxHashMap;

#[derive(Clone)]
pub(crate) enum SparseWeightRow<K, V> {
    Empty,
    One(K, V),
    Many(FxHashMap<K, V>),
}

impl<K, V> Default for SparseWeightRow<K, V> {
    fn default() -> Self {
        Self::Empty
    }
}

impl<K: Copy + Eq + Hash, V> SparseWeightRow<K, V> {
    pub(crate) fn merge_with(
        &mut self,
        key: K,
        add: V,
        is_empty: impl FnOnce(&V) -> bool,
        merge: impl FnOnce(&mut V, V) -> bool,
    ) -> bool {
        if is_empty(&add) {
            return false;
        }
        match self {
            Self::Empty => {
                *self = Self::One(key, add);
                true
            }
            Self::One(existing, weight) if *existing == key => merge(weight, add),
            Self::One(..) => {
                let Self::One(previous, weight) = std::mem::take(self) else {
                    unreachable!("singleton row")
                };
                let mut entries = FxHashMap::with_capacity_and_hasher(4, Default::default());
                entries.insert(previous, weight);
                entries.insert(key, add);
                *self = Self::Many(entries);
                true
            }
            Self::Many(entries) => match entries.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(add);
                    true
                }
                Entry::Occupied(mut entry) => merge(entry.get_mut(), add),
            },
        }
    }

    #[inline]
    pub(crate) fn get_ref(&self, key: &K) -> Option<&V> {
        match self {
            Self::Empty => None,
            Self::One(existing, weight) => (existing == key).then_some(weight),
            Self::Many(entries) => entries.get(key),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(..) => 1,
            Self::Many(entries) => entries.len(),
        }
    }

    pub(crate) fn visit(&self, mut f: impl FnMut(K, &V)) {
        match self {
            Self::Empty => {}
            Self::One(key, weight) => f(*key, weight),
            Self::Many(entries) => {
                for (&key, weight) in entries {
                    f(key, weight);
                }
            }
        }
    }

    pub(crate) fn into_entries_iter(self) -> SparseWeightRowIter<K, V> {
        match self {
            Self::Empty => SparseWeightRowIter::Empty,
            Self::One(key, weight) => SparseWeightRowIter::One(Some((key, weight))),
            Self::Many(entries) => SparseWeightRowIter::Many(entries.into_iter()),
        }
    }
}

pub(crate) enum SparseWeightRowIter<K, V> {
    Empty,
    One(Option<(K, V)>),
    Many(IntoIter<K, V>),
}

impl<K, V> Iterator for SparseWeightRowIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::One(entry) => entry.take(),
            Self::Many(entries) => entries.next(),
        }
    }
}