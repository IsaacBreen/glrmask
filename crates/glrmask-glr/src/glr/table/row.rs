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
pub enum SparseRow<K: Copy + Eq + Hash + Ord, V: Clone> {
    Inline(SmallVec<[(K, V); INLINE_ROW_CAPACITY]>),
    Sorted(Vec<(K, V)>),
    Large(FxHashMap<K, V>),
}

impl<K: Copy + Eq + Hash + Ord, V: Clone> Default for SparseRow<K, V> {
    fn default() -> Self {
        Self::Inline(SmallVec::new())
    }
}

impl<K: Copy + Eq + Hash + Ord, V: Clone> SparseRow<K, V> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Inline(entries) => entries.len(),
            Self::Sorted(entries) => entries.len(),
            Self::Large(entries) => entries.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find(|(entry_key, _)| entry_key == key)
                .map(|(_, value)| value),
            Self::Sorted(entries) => entries
                .binary_search_by_key(key, |(entry_key, _)| *entry_key)
                .ok()
                .map(|index| &entries[index].1),
            Self::Large(entries) => entries.get(key),
        }
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self {
            Self::Inline(entries) => entries
                .iter_mut()
                .find(|(entry_key, _)| entry_key == key)
                .map(|(_, value)| value),
            Self::Sorted(entries) => entries
                .binary_search_by_key(key, |(entry_key, _)| *entry_key)
                .ok()
                .map(|index| &mut entries[index].1),
            Self::Large(entries) => entries.get_mut(key),
        }
    }

    /// Construct directly from a sorted, duplicate-free entry vector.
    ///
    /// Builders commonly already own such a vector. Re-inserting it through
    /// `insert` would needlessly perform O(n²) inline duplicate scans and an
    /// avoidable inline-to-hash-map promotion.
    pub fn from_sorted_unique(entries: Vec<(K, V)>) -> Self {
        debug_assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
        if entries.len() <= INLINE_ROW_CAPACITY {
            Self::Inline(SmallVec::from_vec(entries))
        } else {
            Self::Sorted(entries)
        }
    }

    /// Consume a hash map without reinserting every entry. Small rows retain
    /// the compact inline representation; large rows preserve the map.
    pub fn from_hash_map(entries: FxHashMap<K, V>) -> Self {
        if entries.len() <= INLINE_ROW_CAPACITY {
            Self::Inline(entries.into_iter().collect())
        } else {
            Self::Large(entries)
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
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
            Self::Sorted(entries) => match entries.binary_search_by_key(&key, |(entry_key, _)| *entry_key) {
                Ok(index) => Some(std::mem::replace(&mut entries[index].1, value)),
                Err(index) => {
                    entries.insert(index, (key, value));
                    None
                }
            },
            Self::Large(entries) => entries.insert(key, value),
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            Self::Inline(entries) => {
                let position = entries.iter().position(|(entry_key, _)| entry_key == key)?;
                Some(entries.swap_remove(position).1)
            }
            Self::Sorted(entries) => {
                let position = entries
                    .binary_search_by_key(key, |(entry_key, _)| *entry_key)
                    .ok()?;
                Some(entries.remove(position).1)
            }
            Self::Large(entries) => entries.remove(key),
        }
    }

    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    #[inline]
    pub fn iter(&self) -> SparseRowIter<'_, K, V> {
        match self {
            Self::Inline(entries) => SparseRowIter::Inline(entries.iter()),
            Self::Sorted(entries) => SparseRowIter::Sorted(entries.iter()),
            Self::Large(entries) => SparseRowIter::Large(entries.iter()),
        }
    }

    #[inline]
    pub fn keys(&self) -> SparseRowKeys<'_, K, V> {
        match self {
            Self::Inline(entries) => SparseRowKeys::Inline(entries.iter()),
            Self::Sorted(entries) => SparseRowKeys::Sorted(entries.iter()),
            Self::Large(entries) => SparseRowKeys::Large(entries.keys()),
        }
    }

    #[inline]
    pub fn values(&self) -> SparseRowValues<'_, K, V> {
        match self {
            Self::Inline(entries) => SparseRowValues::Inline(entries.iter()),
            Self::Sorted(entries) => SparseRowValues::Sorted(entries.iter()),
            Self::Large(entries) => SparseRowValues::Large(entries.values()),
        }
    }

    #[inline]
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut V)) {
        match self {
            Self::Inline(entries) => {
                for (_, value) in entries {
                    f(value);
                }
            }
            Self::Sorted(entries) => {
                for (_, value) in entries {
                    f(value);
                }
            }
            Self::Large(entries) => {
                for value in entries.values_mut() {
                    f(value);
                }
            }
        }
    }
}

impl<K: Copy + Eq + Hash + Ord, V: Clone + PartialEq> PartialEq for SparseRow<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<K: Copy + Eq + Hash + Ord, V: Clone + Eq> Eq for SparseRow<K, V> {}

impl<K, V> Serialize for SparseRow<K, V>
where
    K: Copy + Eq + Hash + Ord + Serialize,
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
    K: Copy + Eq + Hash + Ord + Deserialize<'de>,
    V: Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SparseRowVisitor<K, V>(PhantomData<(K, V)>);

        impl<'de, K, V> Visitor<'de> for SparseRowVisitor<K, V>
        where
            K: Copy + Eq + Hash + Ord + Deserialize<'de>,
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

impl<'a, K: Copy + Eq + Hash + Ord, V: Clone> IntoIterator for &'a SparseRow<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = SparseRowIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Copy + Eq + Hash + Ord, V: Clone> Index<&K> for SparseRow<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).expect("sparse row index missing key")
    }
}

impl<K: Copy + Eq + Hash + Ord, V: Clone> FromIterator<(K, V)> for SparseRow<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut row = Self::default();
        for (key, value) in iter {
            row.insert(key, value);
        }
        row
    }
}

pub enum SparseRowIter<'a, K: Copy + Eq + Hash + Ord, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Sorted(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Iter<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash + Ord, V: Clone> Iterator for SparseRowIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(key, value)| (key, value)),
            Self::Sorted(entries) => entries.next().map(|(key, value)| (key, value)),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub enum SparseRowKeys<'a, K: Copy + Eq + Hash + Ord, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Sorted(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Keys<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash + Ord, V: Clone> Iterator for SparseRowKeys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(key, _)| key),
            Self::Sorted(entries) => entries.next().map(|(key, _)| key),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub enum SparseRowValues<'a, K: Copy + Eq + Hash + Ord, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Sorted(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Values<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash + Ord, V: Clone> Iterator for SparseRowValues<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(_, value)| value),
            Self::Sorted(entries) => entries.next().map(|(_, value)| value),
            Self::Large(entries) => entries.next(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionRow {
    Sparse(SparseRow<TerminalID, Action>),
    Default {
        default: Action,
        exceptions: SparseRow<TerminalID, Option<Action>>,
        num_terminals: TerminalID,
    },
}

impl Default for ActionRow {
    fn default() -> Self {
        Self::Sparse(SparseRow::default())
    }
}

impl ActionRow {
    #[inline]
    pub fn is_default_compressed(&self) -> bool {
        matches!(self, Self::Default { .. })
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Sparse(row) => row.len(),
            Self::Default {
                exceptions,
                num_terminals,
                ..
            } => {
                let null_exceptions = exceptions.values().filter(|value| value.is_none()).count();
                *num_terminals as usize - null_exceptions
            }
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, key: &TerminalID) -> Option<&Action> {
        match self {
            Self::Sparse(row) => row.get(key),
            Self::Default {
                default,
                exceptions,
                num_terminals,
            } => {
                if *key >= *num_terminals {
                    return None;
                }
                match exceptions.get(key) {
                    Some(Some(action)) => Some(action),
                    Some(None) => None,
                    None => Some(default),
                }
            }
        }
    }

    pub fn get_mut(&mut self, key: &TerminalID) -> Option<&mut Action> {
        if matches!(self, Self::Default { .. }) {
            self.expand_default_to_sparse();
        }
        match self {
            Self::Sparse(row) => row.get_mut(key),
            Self::Default { .. } => unreachable!("default rows should have been expanded"),
        }
    }

    pub fn insert(&mut self, key: TerminalID, value: Action) -> Option<Action> {
        match self {
            Self::Sparse(row) => row.insert(key, value),
            Self::Default {
                default,
                exceptions,
                num_terminals,
            } => {
                if key >= *num_terminals {
                    self.expand_default_to_sparse();
                    return self.insert(key, value);
                }

                let previous = match exceptions.get(&key) {
                    Some(Some(action)) => Some(action.clone()),
                    Some(None) => None,
                    None => Some(default.clone()),
                };

                if value == *default {
                    exceptions.remove(&key);
                } else {
                    exceptions.insert(key, Some(value));
                }
                previous
            }
        }
    }

    pub fn remove(&mut self, key: &TerminalID) -> Option<Action> {
        match self {
            Self::Sparse(row) => row.remove(key),
            Self::Default {
                default,
                exceptions,
                num_terminals,
            } => {
                if *key >= *num_terminals {
                    return exceptions.remove(key).flatten();
                }

                match exceptions.get(key) {
                    Some(None) => None,
                    Some(Some(action)) => {
                        let removed = action.clone();
                        exceptions.insert(*key, None);
                        Some(removed)
                    }
                    None => {
                        exceptions.insert(*key, None);
                        Some(default.clone())
                    }
                }
            }
        }
    }

    #[inline]
    pub fn contains_key(&self, key: &TerminalID) -> bool {
        self.get(key).is_some()
    }

    #[inline]
    pub fn iter(&self) -> ActionRowIter<'_> {
        match self {
            Self::Sparse(row) => ActionRowIter::Sparse(row.iter()),
            Self::Default {
                default,
                exceptions,
                num_terminals,
            } => ActionRowIter::Default(DefaultActionRowIter {
                next_terminal: 0,
                default,
                exceptions,
                num_terminals: *num_terminals,
            }),
        }
    }

    #[inline]
    pub fn keys(&self) -> ActionRowKeys<'_> {
        ActionRowKeys { iter: self.iter() }
    }

    #[inline]
    pub fn values(&self) -> ActionRowValues<'_> {
        ActionRowValues { iter: self.iter() }
    }

    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Action)) {
        self.expand_default_to_sparse();
        match self {
            Self::Sparse(row) => row.for_each_value_mut(|action| f(action)),
            Self::Default { .. } => unreachable!("default row should have been expanded"),
        }
    }

    pub fn compress_default(&mut self, num_terminals: TerminalID) {
        let Self::Sparse(row) = self else {
            return;
        };
        if row.is_empty() || num_terminals == 0 {
            return;
        }

        let mut counts: FxHashMap<Action, usize> = FxHashMap::default();
        for (terminal, action) in row.iter() {
            if *terminal >= num_terminals {
                return;
            }
            *counts.entry(action.clone()).or_insert(0) += 1;
        }

        let sparse_cost = row.len();
        let present_count = sparse_cost;
        if present_count > num_terminals as usize {
            return;
        }

        let Some((default_action, default_count)) = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
        else {
            return;
        };

        let absent_count = num_terminals as usize - present_count;
        let non_default_present = present_count - default_count;
        let exception_count = absent_count + non_default_present;
        let default_cost = 1 + exception_count;
        if default_cost >= sparse_cost {
            return;
        }

        let mut exceptions = SparseRow::default();
        for terminal in 0..num_terminals {
            match row.get(&terminal) {
                Some(action) if *action == default_action => {}
                Some(action) => {
                    exceptions.insert(terminal, Some(action.clone()));
                }
                None => {
                    exceptions.insert(terminal, None);
                }
            }
        }

        *self = Self::Default {
            default: default_action,
            exceptions,
            num_terminals,
        };
    }

    fn expand_default_to_sparse(&mut self) {
        let Self::Default {
            default,
            exceptions,
            num_terminals,
        } = self
        else {
            return;
        };

        let default = default.clone();
        let exceptions = exceptions.clone();
        let num_terminals = *num_terminals;
        let mut row = SparseRow::default();
        for terminal in 0..num_terminals {
            match exceptions.get(&terminal) {
                Some(Some(action)) => {
                    row.insert(terminal, action.clone());
                }
                Some(None) => {}
                None => {
                    row.insert(terminal, default.clone());
                }
            }
        }
        *self = Self::Sparse(row);
    }
}

impl PartialEq for ActionRow {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(terminal, action)| other.get(&terminal) == Some(action))
    }
}

impl Eq for ActionRow {}

impl<'a> IntoIterator for &'a ActionRow {
    type Item = (TerminalID, &'a Action);
    type IntoIter = ActionRowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Index<&TerminalID> for ActionRow {
    type Output = Action;

    fn index(&self, index: &TerminalID) -> &Self::Output {
        self.get(index).expect("action row index missing key")
    }
}

impl FromIterator<(TerminalID, Action)> for ActionRow {
    fn from_iter<T: IntoIterator<Item = (TerminalID, Action)>>(iter: T) -> Self {
        Self::Sparse(iter.into_iter().collect())
    }
}

pub enum ActionRowIter<'a> {
    Sparse(SparseRowIter<'a, TerminalID, Action>),
    Default(DefaultActionRowIter<'a>),
}

impl<'a> Iterator for ActionRowIter<'a> {
    type Item = (TerminalID, &'a Action);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Sparse(iter) => iter.next().map(|(terminal, action)| (*terminal, action)),
            Self::Default(iter) => iter.next(),
        }
    }
}

pub struct DefaultActionRowIter<'a> {
    next_terminal: TerminalID,
    default: &'a Action,
    exceptions: &'a SparseRow<TerminalID, Option<Action>>,
    num_terminals: TerminalID,
}

impl<'a> Iterator for DefaultActionRowIter<'a> {
    type Item = (TerminalID, &'a Action);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_terminal < self.num_terminals {
            let terminal = self.next_terminal;
            self.next_terminal += 1;
            match self.exceptions.get(&terminal) {
                Some(Some(action)) => return Some((terminal, action)),
                Some(None) => continue,
                None => return Some((terminal, self.default)),
            }
        }
        None
    }
}

pub struct ActionRowKeys<'a> {
    iter: ActionRowIter<'a>,
}

impl<'a> Iterator for ActionRowKeys<'a> {
    type Item = TerminalID;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(terminal, _)| terminal)
    }
}

pub struct ActionRowValues<'a> {
    iter: ActionRowIter<'a>,
}

impl<'a> Iterator for ActionRowValues<'a> {
    type Item = &'a Action;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, action)| action)
    }
}

pub type GotoRow = SparseRow<NonterminalID, (u32, bool)>;

#[cfg(test)]
mod tests {
    use super::{ActionRow, SparseRow};
    use crate::compiler::glr::table::{Action, AdmissionPolicy, GlrTableConstruction, GLRTable};

    fn shift(target: u32) -> Action {
        Action::Shift(target, false)
    }

    #[test]
    fn default_row_lookup_and_iter_handle_null_and_override_exceptions() {
        let row = ActionRow::Default {
            default: Action::Accept,
            exceptions: SparseRow::from_iter([
                (1, None),
                (3, Some(shift(9))),
            ]),
            num_terminals: 5,
        };

        assert_eq!(row.get(&0), Some(&Action::Accept));
        assert_eq!(row.get(&1), None);
        assert_eq!(row.get(&3), Some(&shift(9)));
        assert_eq!(row.len(), 4);

        let entries: Vec<_> = row.iter().collect();
        assert_eq!(entries, vec![
            (0, &Action::Accept),
            (2, &Action::Accept),
            (3, &shift(9)),
            (4, &Action::Accept),
        ]);
    }

    #[test]
    fn default_row_insert_and_remove_track_null_exceptions() {
        let mut row = ActionRow::Default {
            default: Action::Accept,
            exceptions: SparseRow::from_iter([(1, None)]),
            num_terminals: 4,
        };

        assert_eq!(row.insert(1, shift(7)), None);
        assert_eq!(row.get(&1), Some(&shift(7)));
        assert_eq!(row.insert(2, Action::Accept), Some(Action::Accept));
        assert_eq!(row.remove(&0), Some(Action::Accept));
        assert_eq!(row.get(&0), None);
        assert_eq!(row.remove(&1), Some(shift(7)));
        assert_eq!(row.get(&1), None);
    }

    #[test]
    fn default_row_keys_iterate_effective_present_terminals() {
        let row = ActionRow::Default {
            default: shift(3),
            exceptions: SparseRow::from_iter([
                (0, None),
                (2, Some(Action::Accept)),
            ]),
            num_terminals: 4,
        };

        assert_eq!(row.keys().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn compress_default_prefers_default_row_when_structurally_smaller() {
        let mut row = ActionRow::from_iter([
            (0, Action::Accept),
            (1, Action::Accept),
            (3, Action::Accept),
            (4, shift(8)),
        ]);

        row.compress_default(5);

        assert!(row.is_default_compressed());
        assert_eq!(row.get(&0), Some(&Action::Accept));
        assert_eq!(row.get(&2), None);
        assert_eq!(row.get(&4), Some(&shift(8)));
    }

    #[test]
    fn mutable_action_traversal_expands_default_rows_and_preserves_holes() {
        let mut row = ActionRow::Default {
            default: shift(3),
            exceptions: SparseRow::from_iter([
                (1, None),
                (2, Some(Action::Accept)),
            ]),
            num_terminals: 4,
        };

        row.for_each_value_mut(|action| *action = Action::Accept);

        assert!(!row.is_default_compressed());
        assert_eq!(row.get(&0), Some(&Action::Accept));
        assert_eq!(row.get(&1), None);
        assert_eq!(row.get(&2), Some(&Action::Accept));
        assert_eq!(row.get(&3), Some(&Action::Accept));
    }

    #[test]
    fn parallel_table_compression_matches_serial_rows() {
        let rows = (0..256)
            .map(|state| {
                ActionRow::from_iter([
                    (0, Action::Accept),
                    (1, Action::Accept),
                    (2, Action::Accept),
                    (4, shift(10 + state)),
                    (5, Action::Accept),
                ])
            })
            .collect::<Vec<_>>();
        let mut parallel = GLRTable {
            action: rows.clone(),
            goto: vec![SparseRow::default(); rows.len()],
            num_states: rows.len() as u32,
            num_terminals: 6,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            forwarded_shifts: Default::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut serial = parallel.clone();
        for row in &mut serial.action {
            row.compress_default(serial.num_terminals);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| parallel.compress_default_action_rows());

        assert_eq!(parallel.action, serial.action);
    }

    #[test]
    fn table_compression_preserves_lookup_equivalence() {
        let mut table = GLRTable {
            action: vec![ActionRow::from_iter([
                (0, Action::Accept),
                (1, Action::Accept),
                (2, Action::Accept),
                (4, shift(11)),
                (5, Action::Accept),
            ])],
            goto: vec![SparseRow::default()],
            num_states: 1,
            num_terminals: 6,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            forwarded_shifts: Default::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };

        let before = (0..table.num_terminals)
            .map(|terminal| table.action(0, terminal).cloned())
            .collect::<Vec<_>>();

        table.compress_default_action_rows();

        let after = (0..table.num_terminals)
            .map(|terminal| table.action(0, terminal).cloned())
            .collect::<Vec<_>>();

        assert_eq!(before, after);
        assert!(table.action[0].is_default_compressed());
    }
}
