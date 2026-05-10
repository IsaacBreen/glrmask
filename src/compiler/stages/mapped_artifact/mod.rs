//! Mapped artifacts and the generic weight-level operations that preserve their ID maps.

mod compaction;
mod reconcile;

use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::constraint_possible_matches::RuntimePossibleMatchesByTerminal;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::weight::Weight;

pub(crate) use compaction::{CompactReport, InternedRangeCounts};

pub(crate) trait WeightRefs {
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight>;
}

impl WeightRefs for DWA {
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = Vec::new();
        for state in self.states_mut() {
            if let Some(final_weight) = state.final_weight.as_mut() {
                weights.push(final_weight);
            }
            for (_, weight) in state.transitions.values_mut() {
                weights.push(weight);
            }
        }
        weights
    }
}

impl WeightRefs for RuntimePossibleMatchesByTerminal {
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        self.values_mut().collect()
    }
}

impl<A, B> WeightRefs for (A, B)
where
    A: WeightRefs,
    B: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = self.0.weight_refs_mut();
        weights.extend(self.1.weight_refs_mut());
        weights
    }
}

impl<A, B, C> WeightRefs for (A, B, C)
where
    A: WeightRefs,
    B: WeightRefs,
    C: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = self.0.weight_refs_mut();
        weights.extend(self.1.weight_refs_mut());
        weights.extend(self.2.weight_refs_mut());
        weights
    }
}

impl<T> WeightRefs for [T]
where
    T: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = Vec::new();
        for item in self.iter_mut() {
            weights.extend(item.weight_refs_mut());
        }
        weights
    }
}

impl<T> WeightRefs for Vec<T>
where
    T: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        self.as_mut_slice().weight_refs_mut()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MappedArtifact<T: WeightRefs> {
    artifact: T,
    id_map: InternalIdMap,
}

impl<T: WeightRefs> MappedArtifact<T> {
    /// Invariant: `artifact` IDs are expressed in the internal spaces described by `id_map`.
    pub(crate) fn new(artifact: T, id_map: InternalIdMap) -> Self {
        Self { artifact, id_map }
    }

    pub(crate) fn artifact(&self) -> &T {
        &self.artifact
    }

    pub(crate) fn artifact_mut(&mut self) -> &mut T {
        &mut self.artifact
    }

    pub(crate) fn id_map(&self) -> &InternalIdMap {
        &self.id_map
    }

    pub(crate) fn parts_mut(&mut self) -> (&mut T, &mut InternalIdMap) {
        (&mut self.artifact, &mut self.id_map)
    }

    pub(crate) fn into_parts(self) -> (T, InternalIdMap) {
        (self.artifact, self.id_map)
    }

    pub(crate) fn split(self) -> (T, InternalIdMap) {
        self.into_parts()
    }

    pub(crate) fn into_artifact(self) -> T {
        self.artifact
    }

    pub(crate) fn compact_dimensions_with_stats(&mut self) -> CompactReport {
        self.compact_dimensions_with_layout_stats(true)
    }

    pub(crate) fn compact_dimensions_fast_with_stats(&mut self) -> CompactReport {
        self.compact_dimensions_with_layout_stats(false)
    }

    fn compact_dimensions_with_layout_stats(&mut self, allow_expensive_layout: bool) -> CompactReport {
        let (artifact, id_map) = self.parts_mut();
        let mut weights = artifact.weight_refs_mut();
        compaction::compact_weights_with_id_map(&mut weights, id_map, true, allow_expensive_layout)
    }

    pub(crate) fn interned_range_counts(&mut self) -> InternedRangeCounts {
        let weights = self.artifact.weight_refs_mut();
        let weight_refs: Vec<_> = weights.iter().map(|weight| &**weight).collect();
        count_interned_ranges_for_weights(weight_refs)
    }

    pub(crate) fn num_interned_ranges(&mut self) -> usize {
        self.interned_range_counts().total_ranges()
    }

    pub(crate) fn reconcile_with<U>(&mut self, other: &mut MappedArtifact<U>) -> InternalIdMap
    where
        U: WeightRefs,
    {
        let (left_artifact, left_id_map) = self.parts_mut();
        let (right_artifact, right_id_map) = other.parts_mut();
        let mut left_weights = left_artifact.weight_refs_mut();
        let mut right_weights = right_artifact.weight_refs_mut();
        reconcile::reconcile_weight_id_maps(
            &mut left_weights,
            left_id_map,
            &mut right_weights,
            right_id_map,
        );
        left_id_map.clone()
    }

}

impl<T, U> From<(MappedArtifact<T>, MappedArtifact<U>)> for MappedArtifact<(T, U)>
where
    T: WeightRefs,
    U: WeightRefs,
{
    fn from((mut left, mut right): (MappedArtifact<T>, MappedArtifact<U>)) -> Self {
        if same_internal_id_maps(left.id_map(), right.id_map()) {
            let (left_artifact, id_map) = left.into_parts();
            let right_artifact = right.into_artifact();
            return MappedArtifact::new((left_artifact, right_artifact), id_map);
        }

        let common_id_map = {
            let (left_artifact, left_id_map) = left.parts_mut();
            let (right_artifact, right_id_map) = right.parts_mut();
            let mut left_weights = left_artifact.weight_refs_mut();
            let mut right_weights = right_artifact.weight_refs_mut();
            reconcile::reconcile_weight_id_maps(
                &mut left_weights,
                left_id_map,
                &mut right_weights,
                right_id_map,
            );
            left_id_map.clone()
        };
        MappedArtifact::new((left.into_artifact(), right.into_artifact()), common_id_map)
    }
}

fn same_internal_id_maps(left: &InternalIdMap, right: &InternalIdMap) -> bool {
    left.tokenizer_states.original_to_internal == right.tokenizer_states.original_to_internal
        && left.vocab_tokens.original_to_internal == right.vocab_tokens.original_to_internal
}

impl<A, B> MappedArtifact<(A, B)>
where
    A: WeightRefs,
    B: WeightRefs,
{
    pub(crate) fn split_pair(self) -> (MappedArtifact<A>, MappedArtifact<B>) {
        let ((left, right), id_map) = self.into_parts();
        (
            MappedArtifact::new(left, id_map.clone()),
            MappedArtifact::new(right, id_map),
        )
    }
}

impl<T> MappedArtifact<Vec<T>>
where
    T: WeightRefs,
{
    pub(crate) fn reconcile_vec(inputs: Vec<MappedArtifact<T>>) -> MappedArtifact<Vec<T>> {
        assert!(!inputs.is_empty(), "MappedArtifact::reconcile_vec called with empty inputs");

        let mut iter = inputs.into_iter();
        let first = iter.next().unwrap();
        let (first_artifact, first_id_map) = first.into_parts();
        let mut acc = MappedArtifact::new(vec![first_artifact], first_id_map);

        for next in iter {
            let mut next = next;
            let common_id_map = acc.reconcile_with(&mut next);
            let (artifacts, id_map) = acc.parts_mut();
            artifacts.push(next.into_artifact());
            *id_map = common_id_map;
        }

        acc
    }

    pub(crate) fn split_vec(self) -> Vec<MappedArtifact<T>> {
        let (artifacts, id_map) = self.into_parts();
        artifacts
            .into_iter()
            .map(|artifact| MappedArtifact::new(artifact, id_map.clone()))
            .collect()
    }
}

impl InternedRangeCounts {
    pub(crate) fn total_ranges(self) -> usize {
        self.tsid_ranges + self.token_ranges
    }
}

pub(crate) fn count_interned_ranges_for_weights<'a>(
    weights: impl IntoIterator<Item = &'a Weight>,
) -> InternedRangeCounts {
    let weight_refs: Vec<&Weight> = weights.into_iter().collect();
    compaction::count_interned_ranges_for_weight_refs(&weight_refs)
}
