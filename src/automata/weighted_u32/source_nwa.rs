//! Source-local token-deterministic weighted NWAs.
//!
//! Global terminal merging combines vocabulary partitions whose token domains
//! are disjoint. A token from source `s` only observes `s`'s local tokenizer
//! class coordinate, so eagerly expanding every weight into the global product
//! TSID space is unnecessary while merging and minimizing the terminal NWA.
//! This representation keeps those local coordinates separate until a later
//! consumer explicitly materializes a global `Weight`.

use std::collections::BTreeMap;

use smallvec::SmallVec;

use super::nwa::Label;
use crate::ds::weight::Weight;

pub type SourceId = u8;

/// A sparse sum of source-local weights. Components are sorted by source and
/// have pairwise-disjoint token domains by construction.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceWeight {
    components: SmallVec<[(SourceId, Weight); 2]>,
}

impl SourceWeight {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn single(source: SourceId, weight: Weight) -> Self {
        if weight.is_empty() {
            Self::empty()
        } else {
            Self {
                components: smallvec::smallvec![(source, weight)],
            }
        }
    }

    pub fn components(&self) -> &[(SourceId, Weight)] {
        &self.components
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        let mut left = 0usize;
        let mut right = 0usize;
        while left < self.components.len() && right < other.components.len() {
            match self.components[left].0.cmp(&other.components[right].0) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    if !self.components[left]
                        .1
                        .is_disjoint(&other.components[right].1)
                    {
                        return false;
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        true
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let mut output = SmallVec::<[(SourceId, Weight); 2]>::new();
        let mut left = 0usize;
        let mut right = 0usize;
        while left < self.components.len() && right < other.components.len() {
            match self.components[left].0.cmp(&other.components[right].0) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    let weight = self.components[left]
                        .1
                        .intersection(&other.components[right].1);
                    if !weight.is_empty() {
                        output.push((self.components[left].0, weight));
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        Self { components: output }
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut output = SmallVec::<[(SourceId, Weight); 2]>::new();
        let mut left = 0usize;
        let mut right = 0usize;
        while left < self.components.len() || right < other.components.len() {
            match (self.components.get(left), other.components.get(right)) {
                (Some((left_source, left_weight)), Some((right_source, right_weight))) => {
                    match left_source.cmp(right_source) {
                        std::cmp::Ordering::Less => {
                            output.push((*left_source, left_weight.clone()));
                            left += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            output.push((*right_source, right_weight.clone()));
                            right += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            let weight = left_weight.union(right_weight);
                            if !weight.is_empty() {
                                output.push((*left_source, weight));
                            }
                            left += 1;
                            right += 1;
                        }
                    }
                }
                (Some((source, weight)), None) => {
                    output.push((*source, weight.clone()));
                    left += 1;
                }
                (None, Some((source, weight))) => {
                    output.push((*source, weight.clone()));
                    right += 1;
                }
                (None, None) => break,
            }
        }
        Self { components: output }
    }

    pub fn union_all<'a>(weights: impl IntoIterator<Item = &'a SourceWeight>) -> Self {
        let mut by_source = BTreeMap::<SourceId, Vec<&Weight>>::new();
        for weight in weights {
            for (source, local_weight) in &weight.components {
                by_source.entry(*source).or_default().push(local_weight);
            }
        }
        let mut components = SmallVec::<[(SourceId, Weight); 2]>::new();
        for (source, local_weights) in by_source {
            let merged = Weight::union_all(local_weights);
            if !merged.is_empty() {
                components.push((source, merged));
            }
        }
        Self { components }
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.components.iter().all(|(source, weight)| {
            other
                .components
                .binary_search_by_key(source, |(other_source, _)| *other_source)
                .ok()
                .is_some_and(|index| weight.is_subset(&other.components[index].1))
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct SourceNwaState {
    pub transitions: BTreeMap<Label, Vec<(u32, SourceWeight)>>,
    pub final_weight: Option<SourceWeight>,
}

#[derive(Debug, Clone, Default)]
pub struct SourceNWA {
    states: Vec<SourceNwaState>,
    start_states: Vec<u32>,
}

impl SourceNWA {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_parts(states: Vec<SourceNwaState>, start_states: Vec<u32>) -> Self {
        Self { states, start_states }
    }

    pub fn states(&self) -> &[SourceNwaState] {
        &self.states
    }

    pub fn states_mut(&mut self) -> &mut [SourceNwaState] {
        &mut self.states
    }

    pub fn start_states(&self) -> &[u32] {
        &self.start_states
    }

    pub fn set_start_states(&mut self, start_states: Vec<u32>) {
        self.start_states = start_states;
    }

    pub fn add_state(&mut self) -> u32 {
        let state = self.states.len() as u32;
        self.states.push(SourceNwaState::default());
        state
    }

    pub fn set_final_weight(&mut self, state: u32, weight: SourceWeight) {
        self.states[state as usize].final_weight = (!weight.is_empty()).then_some(weight);
    }

    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: SourceWeight) {
        if !weight.is_empty() {
            self.states[from as usize]
                .transitions
                .entry(label)
                .or_default()
                .push((to, weight));
        }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn num_transitions(&self) -> usize {
        self.states
            .iter()
            .map(|state| state.transitions.values().map(Vec::len).sum::<usize>())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;

    fn weight(tokens: &[u32]) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(tokens.iter().copied().map(|token| token..=token)),
        )))
    }

    #[test]
    fn source_weight_operations_are_componentwise() {
        let left = SourceWeight::single(0, weight(&[1, 2]));
        let right = SourceWeight::single(1, weight(&[2, 3]));
        let same_source = SourceWeight::single(0, weight(&[2, 4]));

        assert!(left.is_disjoint(&right));
        assert!(!left.is_disjoint(&same_source));
        assert!(left.intersection(&right).is_empty());
        assert_eq!(left.intersection(&same_source), SourceWeight::single(0, weight(&[2])));
        assert_eq!(
            left.union(&right).components().len(),
            2,
        );
        assert_eq!(
            left.union(&same_source),
            SourceWeight::single(0, weight(&[1, 2, 4])),
        );
    }
}
