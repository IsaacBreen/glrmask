use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::SmallVec;

use crate::ds::leveled_gss::Merge;

const INLINE_PAIR_CAPACITY: usize = 4;
type InlinePairs = SmallVec<[(u32, u32); INLINE_PAIR_CAPACITY]>;

/// Terminal exclusions carried by one parser path.
///
/// Up to four `(tokenizer_state, terminal)` pairs are stored directly. These
/// small correlated exclusion sets dominate runtime decoding. Larger values use
/// the persistent `Arc<BTreeMap<...>>` representation.
#[derive(Clone, Debug)]
pub struct TerminalsDisallowed(Repr);

#[derive(Clone, Debug)]
enum Repr {
    Empty,
    One { state: u32, terminal: u32 },
    Few(InlinePairs),
    Many(Arc<BTreeMap<u32, BTreeSet<u32>>>),
}

#[derive(Clone, Copy)]
pub enum TerminalSetRef<'a> {
    One(&'a u32),
    Few(&'a [(u32, u32)]),
    Many(&'a BTreeSet<u32>),
}

impl TerminalSetRef<'_> {
    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Few(values) => values.len(),
            Self::Many(values) => values.len(),
        }
    }

    pub fn contains(&self, terminal: &u32) -> bool {
        match self {
            Self::One(value) => *value == terminal,
            Self::Few(values) => values.iter().any(|(_, value)| value == terminal),
            Self::Many(values) => values.contains(terminal),
        }
    }

    pub fn to_btree_set(self) -> BTreeSet<u32> {
        self.iter().copied().collect()
    }

    pub fn iter(&self) -> TerminalSetIter<'_> {
        match self {
            Self::One(value) => TerminalSetIter::One(Some(value)),
            Self::Few(values) => TerminalSetIter::Few(values.iter()),
            Self::Many(values) => TerminalSetIter::Many(values.iter()),
        }
    }
}

impl<'a> IntoIterator for TerminalSetRef<'a> {
    type Item = &'a u32;
    type IntoIter = TerminalSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::One(value) => TerminalSetIter::One(Some(value)),
            Self::Few(values) => TerminalSetIter::Few(values.iter()),
            Self::Many(values) => TerminalSetIter::Many(values.iter()),
        }
    }
}

pub enum TerminalSetIter<'a> {
    One(Option<&'a u32>),
    Few(std::slice::Iter<'a, (u32, u32)>),
    Many(std::collections::btree_set::Iter<'a, u32>),
}

impl<'a> Iterator for TerminalSetIter<'a> {
    type Item = &'a u32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(value) => value.take(),
            Self::Few(values) => values.next().map(|(_, terminal)| terminal),
            Self::Many(values) => values.next(),
        }
    }
}

pub enum TerminalsDisallowedIter<'a> {
    Empty,
    One {
        state: Option<&'a u32>,
        terminal: &'a u32,
    },
    Few {
        pairs: &'a [(u32, u32)],
        index: usize,
    },
    Many(std::collections::btree_map::Iter<'a, u32, BTreeSet<u32>>),
}

impl<'a> Iterator for TerminalsDisallowedIter<'a> {
    type Item = (&'a u32, TerminalSetRef<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::One { state, terminal } => state
                .take()
                .map(|state| (state, TerminalSetRef::One(terminal))),
            Self::Few { pairs, index } => {
                let start = *index;
                let (state, _) = pairs.get(start)?;
                let mut end = start + 1;
                while end < pairs.len() && pairs[end].0 == *state {
                    end += 1;
                }
                *index = end;
                Some((&pairs[start].0, TerminalSetRef::Few(&pairs[start..end])))
            }
            Self::Many(values) => values
                .next()
                .map(|(state, terminals)| (state, TerminalSetRef::Many(terminals))),
        }
    }
}

impl TerminalsDisallowed {
    pub fn new() -> Self {
        Self(Repr::Empty)
    }

    fn from_inline_pairs(mut pairs: InlinePairs) -> Self {
        pairs.sort_unstable();
        pairs.dedup();
        match pairs.as_slice() {
            [] => Self::new(),
            [(state, terminal)] => Self(Repr::One {
                state: *state,
                terminal: *terminal,
            }),
            _ => Self(Repr::Few(pairs)),
        }
    }

    fn inline_pairs(&self) -> Option<InlinePairs> {
        match &self.0 {
            Repr::Empty => Some(InlinePairs::new()),
            Repr::One { state, terminal } => {
                let mut pairs = InlinePairs::new();
                pairs.push((*state, *terminal));
                Some(pairs)
            }
            Repr::Few(pairs) => Some(pairs.clone()),
            Repr::Many(_) => None,
        }
    }

    pub fn from_map(map: BTreeMap<u32, BTreeSet<u32>>) -> Self {
        let pair_count = map.values().map(BTreeSet::len).sum::<usize>();
        if pair_count <= INLINE_PAIR_CAPACITY {
            let mut pairs = InlinePairs::new();
            for (state, terminals) in &map {
                for terminal in terminals {
                    pairs.push((*state, *terminal));
                }
            }
            return Self::from_inline_pairs(pairs);
        }
        Self(Repr::Many(Arc::new(map)))
    }

    pub fn is_empty(&self) -> bool {
        matches!(self.0, Repr::Empty)
    }

    pub fn is_inline(&self) -> bool {
        !matches!(self.0, Repr::Many(_))
    }

    pub fn try_with_insert_inline(&self, state: u32, terminal: u32) -> Option<Self> {
        let mut pairs = self.inline_pairs()?;
        let pair = (state, terminal);
        match pairs.binary_search(&pair) {
            Ok(_) => Some(self.clone()),
            Err(index) if pairs.len() < INLINE_PAIR_CAPACITY => {
                pairs.insert(index, pair);
                Some(Self::from_inline_pairs(pairs))
            }
            Err(_) => None,
        }
    }

    pub fn try_merge_inline(&self, other: &Self) -> Option<Self> {
        let mut pairs = self.inline_pairs()?;
        for pair in other.inline_pairs()? {
            match pairs.binary_search(&pair) {
                Ok(_) => {}
                Err(index) if pairs.len() < INLINE_PAIR_CAPACITY => pairs.insert(index, pair),
                Err(_) => return None,
            }
        }
        Some(Self::from_inline_pairs(pairs))
    }

    pub fn try_remap_single_state_inline(
        &self,
        source_state: u32,
        end_states: &[u32],
    ) -> Option<Self> {
        let pairs = self.inline_pairs()?;
        let terminals = pairs
            .iter()
            .filter_map(|(state, terminal)| (*state == source_state).then_some(*terminal))
            .collect::<SmallVec<[u32; INLINE_PAIR_CAPACITY]>>();
        if terminals.is_empty() || end_states.is_empty() {
            return Some(Self::new());
        }
        let mut remapped = InlinePairs::new();
        for &end_state in end_states {
            for &terminal in &terminals {
                let pair = (end_state, terminal);
                if remapped.binary_search(&pair).is_err() {
                    if remapped.len() == INLINE_PAIR_CAPACITY {
                        return None;
                    }
                    let index = remapped.binary_search(&pair).unwrap_err();
                    remapped.insert(index, pair);
                }
            }
        }
        Some(Self::from_inline_pairs(remapped))
    }

    /// Number of tokenizer-state keys, matching `BTreeMap::len`.
    pub fn len(&self) -> usize {
        match &self.0 {
            Repr::Empty => 0,
            Repr::One { .. } => 1,
            Repr::Few(pairs) => {
                let mut count = 0usize;
                let mut prior = None;
                for &(state, _) in pairs {
                    if prior != Some(state) {
                        count += 1;
                        prior = Some(state);
                    }
                }
                count
            }
            Repr::Many(values) => values.len(),
        }
    }

    pub fn get(&self, state: &u32) -> Option<TerminalSetRef<'_>> {
        match &self.0 {
            Repr::Empty => None,
            Repr::One {
                state: existing,
                terminal,
            } => (*existing == *state).then_some(TerminalSetRef::One(terminal)),
            Repr::Few(pairs) => {
                let start = pairs.iter().position(|(candidate, _)| candidate == state)?;
                let mut end = start + 1;
                while end < pairs.len() && pairs[end].0 == *state {
                    end += 1;
                }
                Some(TerminalSetRef::Few(&pairs[start..end]))
            }
            Repr::Many(values) => values.get(state).map(TerminalSetRef::Many),
        }
    }

    pub fn iter(&self) -> TerminalsDisallowedIter<'_> {
        match &self.0 {
            Repr::Empty => TerminalsDisallowedIter::Empty,
            Repr::One { state, terminal } => TerminalsDisallowedIter::One {
                state: Some(state),
                terminal,
            },
            Repr::Few(pairs) => TerminalsDisallowedIter::Few {
                pairs,
                index: 0,
            },
            Repr::Many(values) => TerminalsDisallowedIter::Many(values.iter()),
        }
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        if self.is_empty() {
            return true;
        }
        self.iter().all(|(state, terminals)| {
            let Some(other_terminals) = other.get(state) else {
                return false;
            };
            terminals
                .iter()
                .all(|terminal| other_terminals.contains(terminal))
        })
    }

    /// Return a new value with an additional exclusion inserted.
    pub fn with_insert(&self, state: u32, terminal: u32) -> Self {
        if let Some(updated) = self.try_with_insert_inline(state, terminal) {
            return updated;
        }
        let mut map = self.to_map();
        map.entry(state).or_default().insert(terminal);
        Self::from_map(map)
    }

    fn to_map(&self) -> BTreeMap<u32, BTreeSet<u32>> {
        match &self.0 {
            Repr::Empty => BTreeMap::new(),
            Repr::One { state, terminal } => {
                BTreeMap::from([(*state, BTreeSet::from([*terminal]))])
            }
            Repr::Few(pairs) => {
                let mut map = BTreeMap::<u32, BTreeSet<u32>>::new();
                for &(state, terminal) in pairs {
                    map.entry(state).or_default().insert(terminal);
                }
                map
            }
            Repr::Many(values) => (**values).clone(),
        }
    }
}

impl Default for TerminalsDisallowed {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for TerminalsDisallowed {
    fn eq(&self, other: &Self) -> bool {
        if let (Repr::Many(left), Repr::Many(right)) = (&self.0, &other.0) {
            if Arc::ptr_eq(left, right) {
                return true;
            }
        }
        self.is_subset_of(other) && other.is_subset_of(self)
    }
}

impl Eq for TerminalsDisallowed {}

impl Hash for TerminalsDisallowed {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (tokenizer_state, terminals) in self.iter() {
            tokenizer_state.hash(state);
            let mut terminal_count = 0usize;
            for terminal in terminals.iter() {
                terminal.hash(state);
                terminal_count += 1;
            }
            terminal_count.hash(state);
        }
        self.len().hash(state);
    }
}

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        if self.is_subset_of(other) {
            return other.clone();
        }
        if other.is_subset_of(self) {
            return self.clone();
        }
        if let Some(merged) = self.try_merge_inline(other) {
            return merged;
        }

        let mut merged = self.to_map();
        for (state, terminals) in other.iter() {
            merged
                .entry(*state)
                .or_default()
                .extend(terminals.iter().copied());
        }
        Self::from_map(merged)
    }

    fn subsumes(&self, other: &Self) -> bool {
        other.is_subset_of(self)
    }
}
