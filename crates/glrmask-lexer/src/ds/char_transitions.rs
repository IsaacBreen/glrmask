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
        debug_assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
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
        // DFA builders commonly visit the byte alphabet in ascending order.
        // Appending that next entry needs neither binary search nor a shift.
        // Equal/out-of-order keys retain the exact replacement/insertion path.
        if self.entries.last().is_none_or(|(last, _)| *last < key) {
            self.entries.push((key, value));
            return None;
        }
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

/// A view of the sorted entries. Preserve the underlying slice's exact
/// remaining length so collectors can reserve once, without scanning values.
pub struct CharTransitionsIter<'a, T> {
    inner: std::slice::Iter<'a, (u8, T)>,
}

impl<'a, T> Iterator for CharTransitionsIter<'a, T> {
    type Item = (u8, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
}

impl<T> ExactSizeIterator for CharTransitionsIter<'_, T> {
    #[inline]
    fn len(&self) -> usize { self.inner.len() }
}

impl<T> std::iter::FusedIterator for CharTransitionsIter<'_, T> {}

pub struct CharTransitionsIterMut<'a, T> {
    inner: std::slice::IterMut<'a, (u8, T)>,
}

impl<'a, T> Iterator for CharTransitionsIterMut<'a, T> {
    type Item = (u8, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
}

impl<T> ExactSizeIterator for CharTransitionsIterMut<'_, T> {
    #[inline]
    fn len(&self) -> usize { self.inner.len() }
}

impl<T> std::iter::FusedIterator for CharTransitionsIterMut<'_, T> {}

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
        let iter = iter.into_iter();
        // There can be at most 256 distinct byte keys. Use a known input
        // length without reserving unbounded space for repeated replacements.
        self.entries.reserve(iter.size_hint().0.min(256usize.saturating_sub(self.entries.len())));
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

#[cfg(test)]
mod iterator_contract_tests {
    use super::CharTransitions;

    #[test]
    fn exact_lengths_survive_partial_consumption() {
        for count in [0usize, 1, 3, 31, 100, 256] {
            let transitions: CharTransitions<u32> = (0..count)
                .map(|byte| (byte as u8, byte as u32 + 1000))
                .collect();
            for consumed in 0..=count {
                let mut iter = transitions.iter();
                for byte in 0..consumed {
                    assert_eq!(iter.next(), Some((byte as u8, &(byte as u32 + 1000))));
                }
                assert_eq!(iter.len(), count - consumed);
                assert_eq!(iter.size_hint(), (count - consumed, Some(count - consumed)));
                assert_eq!(iter.count(), count - consumed);
            }
            let mut iter = transitions.iter();
            while iter.next().is_some() {}
            assert_eq!(iter.next(), None);
            assert_eq!(iter.size_hint(), (0, Some(0)));
        }
    }

    #[test]
    fn mutable_exact_lengths_preserve_updates_and_key_order() {
        let mut transitions: CharTransitions<u32> = [(9, 90), (1, 10), (4, 40)].into_iter().collect();
        let mut iter = transitions.iter_mut();
        assert_eq!(iter.len(), 3);
        let (key, value) = iter.next().unwrap();
        assert_eq!(key, 1);
        *value = 11;
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.count(), 2);
        assert_eq!(transitions.iter().map(|(key, value)| (key, *value)).collect::<Vec<_>>(),
                   vec![(1, 11), (4, 40), (9, 90)]);
        let mut iter = transitions.iter_mut();
        while iter.next().is_some() {}
        assert!(iter.next().is_none());
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn sorted_append_and_arbitrary_replacements_match_ordered_map() {
        use std::collections::BTreeMap;
        for stride in [1u32, 3, 17, 255] {
            let mut transitions = CharTransitions::new();
            let mut reference = BTreeMap::new();
            for value in 0..1024u32 {
                let key = value.wrapping_mul(stride) as u8;
                assert_eq!(transitions.insert(key, value), reference.insert(key, value));
            }
            assert_eq!(transitions.iter().map(|(key, value)| (key, *value)).collect::<Vec<_>>(),
                       reference.into_iter().collect::<Vec<_>>());
            assert_eq!(transitions.len(), 256);
        }
    }
}
