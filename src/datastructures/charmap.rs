use crate::datastructures::u8set::U8Set;
use std::ops::{Index, IndexMut};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap;
use std::fmt::{self, Debug, Formatter};
// Added for derive macro pattern


const CHARMAP_SIZE: usize = 256;

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct TrieMap<T> {
    data: Vec<Option<Box<T>>>,
    // TODO: what's the point of `children`? Is it for nondeterminism? If so, let's remove it.
    children: Vec<Vec<usize>>, // This field is problematic for general JSON serialization if its meaning is tied to specific graph structures.
                               // For now, we'll serialize it as is, but it might need context-specific handling.
    u8set: U8Set,
}

// Helper function to format u8 keys.
// It displays printable ASCII characters as char literals (e.g., 'A'),
// and other byte values as numbers.
fn format_u8_key(val: u8, f: &mut Formatter<'_>) -> fmt::Result {
    if val >= 0x20 && val <= 0x7E { // Standard printable ASCII range
        // Use char's Debug impl, which handles escapes like '\'' correctly
        Debug::fmt(&(val as char), f)
    } else {
        // Fall back to u8's Debug impl (numeric) for non-printable characters
        Debug::fmt(&val, f)
    }
}

// Helper enum to represent single keys or ranges for Debug output.
// This allows formatting like 'A' or 'A'..'Z'.
enum KeyDisplay {
    Single(u8),
    Range(u8, u8),
}

impl Debug for KeyDisplay {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            KeyDisplay::Single(k) => format_u8_key(k, f),
            KeyDisplay::Range(start, end) => {
                format_u8_key(start, f)?;
                f.write_str("..")?;
                format_u8_key(end, f)
            }
        }
    }
}

impl<T: Debug> Debug for TrieMap<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut ds = f.debug_struct("TrieMap");

        // Helper struct for formatting entries with range compression.
        // Generic V is used to avoid conflict with the outer T.
        struct Entries<'a, V: Debug>(&'a TrieMap<V>);
        impl<'a, V: Debug> Debug for Entries<'a, V> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut map_formatter = f.debug_map();
                let trie_map = self.0; // This is &'a TrieMap<V>

                let mut i = 0;
                while i < CHARMAP_SIZE {
                    // Check data[i] for a value
                    if let Some(current_val_box) = &trie_map.data[i] {
                        let start_key = i as u8;
                        let current_val_ref = current_val_box.as_ref(); // This is &V

                        // Scan forward to find the end of the run with the same value.
                        // "Same value" is determined by pointer equality (std::ptr::eq).
                        let mut end_key = start_key;
                        let mut j = i + 1;
                        while j < CHARMAP_SIZE {
                            if let Some(next_val_box) = &trie_map.data[j] {
                                let next_val_ref = next_val_box.as_ref(); // This is &V
                                if std::ptr::eq(current_val_ref, next_val_ref) {
                                    end_key = j as u8;
                                } else {
                                    // Value changed (or different Box for same logical value)
                                    break; 
                                }
                            } else {
                                // Gap (None), run ends
                                break; 
                            }
                            j += 1;
                        }

                        // Format the key as a single key or a range
                        let key_display = if start_key == end_key {
                            KeyDisplay::Single(start_key)
                        } else {
                            KeyDisplay::Range(start_key, end_key)
                        };
                        map_formatter.entry(&key_display, current_val_ref);

                        // Advance i to the position after the processed range
                        i = (end_key as usize) + 1; 
                    } else {
                        // No value at data[i], advance to the next key
                        i += 1; 
                    }
                }
                map_formatter.finish()
            }
        }

        // Helper for formatting transitions (keys also use KeyDisplay for consistency).
        struct Transitions<'a>(&'a [Vec<usize>]);
        impl<'a> Debug for Transitions<'a> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut map_formatter = f.debug_map();
                for (idx, list) in self.0.iter().enumerate() {
                    if !list.is_empty() {
                        // idx is uaranteed to be 0..CHARMAP_SIZE-1
                        map_formatter.entry(&KeyDisplay::Single(idx as u8), list);
                    }
                }
                map_formatter.finish()
            }
        }

        ds.field("len", &self.len())
          .field("entries", &Entries(self));

        // Only show transitions if there actually are some.
        if self.children.iter().any(|v| !v.is_empty()) {
            ds.field("transitions", &Transitions(&self.children));
        }

        ds.finish()
    }
}

impl<T: JSONConvertible> JSONConvertible for TrieMap<T> {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();

        // Serialize non-None data entries as a map from u8 to T's JSON
        let mut data_map_json = StdMap::new();
        for (i, opt_boxed_t) in self.data.iter().enumerate() {
            if let Some(boxed_t) = opt_boxed_t {
                data_map_json.insert((i as u8).to_string(), boxed_t.as_ref().to_json());
            }
        }
        obj.insert("data".to_string(), JSONNode::Object(data_map_json));

        // Serialize children as is (array of arrays of numbers)
        // This might not be universally useful without context.
        obj.insert("children_vectors".to_string(), self.children.to_json());
        obj.insert("u8set".to_string(), self.u8set.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let data_node = obj.remove("data").ok_or_else(|| "Missing field data for TrieMap".to_string())?;
                let mut data_vec: Vec<Option<Box<T>>> = std::iter::repeat_with(|| None).take(CHARMAP_SIZE).collect();
                match data_node {
                    JSONNode::Object(json_obj) => {
                        for (key_str, val_node) in json_obj {
                            let index = key_str.parse::<u8>().map_err(|e| format!("Invalid u8 key in TrieMap data: {}, err: {}", key_str, e))? as usize;
                            if index < CHARMAP_SIZE {
                                data_vec[index] = Some(Box::new(T::from_json(val_node)?));
                            } else {
                                return Err(format!("Index {} out of bounds for TrieMap data", index));
                            }
                        }
                    }
                    _ => return Err("Expected JSONNode::Object for TrieMap data field".to_string()),
                }

                let children_vectors = obj.remove("children_vectors").ok_or_else(|| "Missing field children_vectors for TrieMap".to_string())
                                          .and_then(Vec::<Vec<usize>>::from_json)?;
                if children_vectors.len() != CHARMAP_SIZE && !children_vectors.is_empty() { // Allow empty if it was default
                     // If it was default constructed, it might be empty. If serialized, it should be CHARMAP_SIZE.
                     // This logic might need adjustment based on how default/empty TrieMaps are handled.
                     // For now, if it's present and not empty, enforce size.
                    // return Err(format!("TrieMap children_vectors field has incorrect length: {} (expected {})", children_vectors.len(), CHARMAP_SIZE));
                }


                let u8set = obj.remove("u8set").ok_or_else(|| "Missing field u8set for TrieMap".to_string())
                                 .and_then(U8Set::from_json)?;

                Ok(TrieMap {
                    data: data_vec,
                    children: children_vectors,
                    u8set,
                })
            }
            _ => Err("Expected JSONNode::Object for TrieMap".to_string()),
        }
    }
}


impl<T> TrieMap<T> {
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(CHARMAP_SIZE);
        for _ in 0..CHARMAP_SIZE {
            data.push(None);
        }

        Self {
            data,
            children: vec![Vec::new(); CHARMAP_SIZE],
            u8set: U8Set::none(),
        }
    }

    pub fn insert(&mut self, key: u8, value: T) -> Option<T> {
        let index = key as usize;
        let old_value = self.data[index].take();
        self.data[index] = Some(Box::new(value));
        self.u8set.insert(key);
        old_value.map(|v| *v)
    }

    pub fn get(&self, key: u8) -> Option<&T> {
        let index = key as usize;
        self.data[index].as_ref().map(|v| v.as_ref())
    }

    pub fn get_mut(&mut self, key: u8) -> Option<&mut T> {
        let index = key as usize;
        self.data[index].as_mut().map(|v| v.as_mut())
    }

    pub fn remove(&mut self, key: u8) -> Option<T> {
        let index = key as usize;
        let old_value = self.data[index].take();
        if old_value.is_some() {
            self.u8set.remove(key);
        }
        old_value.map(|v| *v)
    }

    pub fn contains_key(&self, key: u8) -> bool {
        let index = key as usize;
        self.data[index].is_some()
    }

    pub fn clear(&mut self) {
        for element in self.data.iter_mut() {
            *element = None;
        }
        self.u8set.clear();
    }

    pub fn drain(&mut self) -> impl Iterator<Item = (u8, T)> + '_ {
        self.u8set.clear();
        self.data.iter_mut().enumerate().filter_map(|(i, option)| {
            option.take().map(|boxed| (i as u8, *boxed))
        })
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(u8, &mut T) -> bool,
    {
        for (i, value) in self.data.iter_mut().enumerate() {
            if let Some(v) = value {
                let key = i as u8;
                if !f(key, v.as_mut()) {
                    *value = None;
                    self.u8set.remove(key);
                }
            }
        }
    }

    pub fn capacity(&self) -> usize {
        CHARMAP_SIZE
    }

    pub fn len(&self) -> usize {
        self.u8set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.u8set.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (u8, &T)> {
        self.data.iter().enumerate().filter_map(|(i, option)| {
            option.as_ref().map(|boxed| (i as u8, boxed.as_ref()))
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u8, &mut T)> {
        self.data.iter_mut().enumerate().filter_map(|(i, option)| {
            option.as_mut().map(|boxed| (i as u8, boxed.as_mut()))
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = u8> + '_ {
        self.iter().map(|(key, _)| key)
    }

    pub fn keys_as_u8set(&self) -> U8Set {
        debug_assert_eq!(self.data.len(), CHARMAP_SIZE);
        debug_assert_eq!(self.children.len(), CHARMAP_SIZE);
        self.u8set.clone()
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.iter().map(|(_, value)| value)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.iter_mut().map(|(_, value)| value)
    }

    pub fn entry(&mut self, key: u8) -> Entry<'_, T> {
        let index = key as usize;
        if self.data[index].is_some() {
            Entry::Occupied(OccupiedEntry { map: self, index })
        } else {
            Entry::Vacant(VacantEntry { map: self, index })
        }
    }

    pub fn transition(&self, key: u8) -> Option<&Vec<usize>> {
        let index = key as usize;
        if self.u8set.contains(key) {
            Some(&self.children[index])
        } else {
            None
        }
    }

    pub fn transition_mut(&mut self, key: u8) -> Option<&mut Vec<usize>> {
        let index = key as usize;
        if self.u8set.contains(key) {
            Some(&mut self.children[index])
        } else {
            None
        }
    }

    pub fn add_transition(&mut self, from: u8, to: usize) {
        let index = from as usize;
        self.children[index].push(to);
    }
}

impl<T> Default for TrieMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<u8> for TrieMap<T> {
    type Output = T;

    fn index(&self, key: u8) -> &Self::Output {
        self.get(key).expect("Key not found")
    }
}

impl<T> IndexMut<u8> for TrieMap<T> {
    fn index_mut(&mut self, key: u8) -> &mut Self::Output {
        self.get_mut(key).expect("Key not found")
    }
}

pub enum Entry<'a, T> {
    Occupied(OccupiedEntry<'a, T>),
    Vacant(VacantEntry<'a, T>),
}

pub struct OccupiedEntry<'a, T> {
    map: &'a mut TrieMap<T>,
    index: usize,
}

pub struct VacantEntry<'a, T> {
    map: &'a mut TrieMap<T>,
    index: usize,
}

impl<'a, T> Entry<'a, T> {
    pub fn key(&self) -> u8 {
        match self {
            Entry::Occupied(occupied) => occupied.index as u8,
            Entry::Vacant(vacant) => vacant.index as u8,
        }
    }

    pub fn or_insert(self, default: T) -> &'a mut T {
        match self {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => vacant.insert(default),
        }
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut T
    where
        F: FnOnce() -> T,
    {
        match self {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => vacant.insert(default()),
        }
    }
}

impl<'a, T> OccupiedEntry<'a, T> {
    pub fn get(&self) -> &T {
        self.map.data[self.index].as_ref().unwrap().as_ref()
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.map.data[self.index].as_mut().unwrap().as_mut()
    }

    pub fn into_mut(self) -> &'a mut T {
        self.map.data[self.index].as_mut().unwrap().as_mut()
    }

    pub fn insert(&mut self, value: T) -> T {
        *std::mem::replace(&mut self.map.data[self.index], Some(Box::new(value))).unwrap()
    }

    pub fn remove(self) -> T {
        self.map.u8set.remove(self.index as u8); // Keep u8set consistent
        *self.map.data[self.index].take().unwrap()
    }
}

impl<'a, T> VacantEntry<'a, T> {
    pub fn insert(self, value: T) -> &'a mut T {
        self.map.data[self.index] = Some(Box::new(value));
        self.map.u8set.insert(self.index as u8); // Keep u8set consistent
        self.map.data[self.index].as_mut().unwrap().as_mut()
    }
}

impl<T> IntoIterator for TrieMap<T> {
    type Item = (u8, T);
    type IntoIter = std::vec::IntoIter<Self::Item>; // Corrected

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().enumerate().filter_map(|(i, option)| {
            option.map(|boxed| (i as u8, *boxed))
        }).collect::<Vec<_>>().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a TrieMap<T> {
    type Item = (u8, &'a T);
    type IntoIter = std::vec::IntoIter<Self::Item>; // Corrected

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a mut TrieMap<T> {
    type Item = (u8, &'a mut T);
    type IntoIter = std::vec::IntoIter<Self::Item>; // Corrected

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut().collect::<Vec<_>>().into_iter()
    }
}

impl<T> Extend<(u8, T)> for TrieMap<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (u8, T)>,
    {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<V, const N: usize> From<[(u8, V); N]> for TrieMap<V> {
    fn from(array: [(u8, V); N]) -> Self {
        let mut map = Self::new();
        map.extend(array);
        map
    }
}

impl<V> FromIterator<(u8, V)> for TrieMap<V> {
    fn from_iter<I: IntoIterator<Item = (u8, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}