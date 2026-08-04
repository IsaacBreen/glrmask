use std::fmt;
use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct CharTransitions<T> {
    entries: Vec<(u8, T)>,
}

impl<T> CharTransitions<T> {
    #[inline]
    fn entry_index(&self, key: u8) -> Result<usize, usize> {
        self.entries.binary_search_by_key(&key, |(existing_key, _)| *existing_key)
    }

    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn from_sorted_entries(entries: Vec<(u8, T)>) -> Self {
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, key: u8, value: T) -> Option<T> {
        match self.entry_index(key) {
            Ok(index) => Some(std::mem::replace(&mut self.entries[index].1, value)),
            Err(index) => {
                self.entries.insert(index, (key, value));
                None
            }
        }
    }

    pub fn get(&self, key: u8) -> Option<&T> {
        self.entry_index(key)
            .ok()
            .map(|index| &self.entries[index].1)
    }

    pub fn get_mut(&mut self, key: u8) -> Option<&mut T> {
        self.entry_index(key)
            .ok()
            .map(move |index| &mut self.entries[index].1)
    }

    pub fn contains_key(&self, key: u8) -> bool {
        self.get(key).is_some()
    }

    pub fn iter(&self) -> CharTransitionsIter<'_, T> {
        CharTransitionsIter {
            inner: self.entries.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> CharTransitionsIterMut<'_, T> {
        CharTransitionsIterMut {
            inner: self.entries.iter_mut(),
        }
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().map(|(_, v)| v)
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

pub struct CharTransitionsIter<'a, T> {
    inner: std::slice::Iter<'a, (u8, T)>,
}

impl<'a, T> Iterator for CharTransitionsIter<'a, T> {
    type Item = (u8, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }
}

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
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut CharTransitions<T> {
    type Item = (u8, &'a mut T);
    type IntoIter = CharTransitionsIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Extend<(u8, T)> for CharTransitions<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (u8, T)>,
    {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<T> FromIterator<(u8, T)> for CharTransitions<T> {
    fn from_iter<I: IntoIterator<Item = (u8, T)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<T: fmt::Debug> fmt::Debug for CharTransitions<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (key, value) in &self.entries {
            map.entry(key, value);
        }
        map.finish()
    }
}
