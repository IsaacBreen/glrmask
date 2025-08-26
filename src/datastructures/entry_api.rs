use ordered_hash_map::OrderedHashMap;
use std::hash::Hash;

/// A lightweight, trait-based "entry" API for OrderedHashMap.
///
/// This is an extension trait that provides a familiar entry-style workflow:
/// map.entry(key).and_modify(|v| { .. }).or_insert(value)
///
/// Notes:
/// - This implementation is intentionally minimal and only exposes the most commonly
///   used methods: `or_insert`, `or_insert_with`, and `and_modify`.
/// - To keep the implementation generic and safe without depending on OrderedHashMap
///   internals, we require K: Clone so that we can re-query after insertion.
///   This allows returning an &'a mut V using only the standard get_mut/insert APIs.
///
/// If your keys are not Clone, consider introducing a Clone wrapper (e.g., Arc, newtype
/// around indices, etc.) or extend OrderedHashMap with a more direct insert-and-get_mut
/// operation.
pub trait EntryApi<K, V> {
    type Entry<'a>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    /// Returns an "entry" object for the given key that supports:
    /// - and_modify(|v| { ... })
    /// - or_insert(default_value)
    /// - or_insert_with(|| default_value)
    fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a>;
}

/// The entry wrapper returned by EntryApi::entry.
/// It allows calling and_modify(...).or_insert(...) style chains.
pub struct OrderedMapEntry<'a, K, V> {
    map: &'a mut OrderedHashMap<K, V>,
    key: K,
}

impl<'a, K, V> OrderedMapEntry<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    /// If the key is present, apply `f` to the existing value, then return self for chaining.
    /// If the key is absent, do nothing and return self for chaining.
    ///
    /// Example:
    ///   map.entry(k).and_modify(|v| *v += 1).or_insert(0);
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Some(v) = self.map.get_mut(&self.key) {
            f(v);
        }
        self
    }

    /// Inserts `default` if the key is absent, then returns a mutable reference
    /// to the value for the key (existing or newly inserted).
    pub fn or_insert(self, default: V) -> &'a mut V {
        if let Some(index) = self.map.get_index_of(&self.key) {
            return self.map.get_index_mut(index).unwrap().1;
        }
        let (index, _) = self.map.insert_full(self.key, default);
        self.map.get_index_mut(index).unwrap().1
    }

    /// Inserts the result of `f` if the key is absent, then returns a mutable reference
    /// to the value for the key (existing or newly inserted).
    pub fn or_insert_with<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        if let Some(index) = self.map.get_index_of(&self.key) {
            return self.map.get_index_mut(index).unwrap().1;
        }
        let (index, _) = self.map.insert_full(self.key, f());
        self.map.get_index_mut(index).unwrap().1
    }
}

impl<K, V> EntryApi<K, V> for OrderedHashMap<K, V>
where
    K: Eq + Hash + Clone,
{
    type Entry<'a> = OrderedMapEntry<'a, K, V> where Self: 'a, K: 'a, V: 'a;

    fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a> {
        OrderedMapEntry { map: self, key }
    }
}
