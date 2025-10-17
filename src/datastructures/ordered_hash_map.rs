use std::hash::Hash;
use ordered_hash_map::OrderedHashMap;

pub trait Retain<K, V> {
    fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, f: F);
}

impl<K: Clone + Eq + Hash, V: Clone> Retain<K, V> for ordered_hash_map::OrderedHashMap<K, V> {
    fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        let mut retained: OrderedHashMap<K, V> = ordered_hash_map::OrderedHashMap::new();
        for (k, v) in self.into_iter() {
            if f(k, v) {
                retained.insert(k.clone(), v.clone());
            }
        }
        *self = retained;
    }
}

pub trait Pop<K, V> {
    /// Removes the last key-value pair from the map and returns it, or `None` if the map is empty.
    fn pop(&mut self) -> Option<(K, V)>;
}

impl<K: Eq + Hash, V> Pop<K, V> for ordered_hash_map::OrderedHashMap<K, V> {
    fn pop(&mut self) -> Option<(K, V)> {
        // We use the fully qualified path to call the inherent `pop` method
        // on the `OrderedHashMap` type, avoiding the recursive call to the
        // trait method we are implementing.
        ordered_hash_map::OrderedHashMap::pop(self)
    }
}

