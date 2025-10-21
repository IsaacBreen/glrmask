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
    /// - or_default()
    fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a>;
}

/// A view into a single entry in an `OrderedHashMap`, which may either be vacant or occupied.
pub enum OrderedMapEntry<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, V>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
}

/// A view into an occupied entry in an `OrderedHashMap`.
pub struct OccupiedEntry<'a, V> {
    value: &'a mut V,
}

impl<'a, V> OccupiedEntry<'a, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        self.value
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        self.value
    }

    /// Converts the entry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself.
    pub fn into_mut(self) -> &'a mut V {
        self.value
    }
}

/// A view into a vacant entry in an `OrderedHashMap`.
pub struct VacantEntry<'a, K, V> {
    map: &'a mut OrderedHashMap<K, V>,
    key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    /// Inserts the value into the entry and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        self.map.insert(self.key.clone(), value);
        self.map.get_mut(&self.key).unwrap()
    }
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
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let OrderedMapEntry::Occupied(mut entry) = self {
            f(entry.value);
            OrderedMapEntry::Occupied(entry)
        } else {
            self
        }
    }

    /// Inserts `default` if the key is absent, then returns a mutable reference
    /// to the value for the key (existing or newly inserted).
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            OrderedMapEntry::Occupied(entry) => entry.value,
            OrderedMapEntry::Vacant(entry) => entry.or_insert(default),
        }
    }

    /// Inserts the result of `f` if the key is absent, then returns a mutable reference
    /// to the value for the key (existing or newly inserted).
    pub fn or_insert_with<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            OrderedMapEntry::Occupied(entry) => entry.value,
            OrderedMapEntry::Vacant(entry) => entry.or_insert_with(f),
        }
    }

    /// Inserts the default value for the value type if the key is absent,
    /// then returns a mutable reference to the value for the key (existing or newly inserted).
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        match self {
            OrderedMapEntry::Occupied(entry) => entry.value,
            OrderedMapEntry::Vacant(entry) => entry.or_default(),
        }
    }
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    /// Inserts `default` if the key is absent, then returns a mutable reference
    /// to the value for the key (newly inserted).
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.map.insert(self.key.clone(), default);
        self.map.get_mut(&self.key).unwrap()
    }

    /// Inserts the result of `f` if the key is absent, then returns a mutable reference
    /// to the value for the key (newly inserted).
    pub fn or_insert_with<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        self.map.insert(self.key.clone(), f());
        self.map.get_mut(&self.key).unwrap()
    }

    /// Inserts the default value for the value type if the key is absent,
    /// then returns a mutable reference to the value for the key (newly inserted).
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
}

impl<K, V> EntryApi<K, V> for OrderedHashMap<K, V>
where
    K: Eq + Hash + Clone,
{
    type Entry<'a> = OrderedMapEntry<'a, K, V> where Self: 'a, K: 'a, V: 'a;

    fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a> {
        if self.get(&key).is_some() {
            let value = self.get_mut(&key).unwrap();
            OrderedMapEntry::Occupied(OccupiedEntry { value })
        } else {
            OrderedMapEntry::Vacant(VacantEntry { map: self, key })
        }
    }
}
