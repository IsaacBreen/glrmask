//! Mapped artifacts and the generic weight-level operations that preserve their ID maps.

mod compaction;
mod reconcile;

use std::collections::BTreeMap;
use std::ops::ControlFlow;

use crate::automata::weighted_u32::dwa::{DWA, DWAState};
use crate::automata::weighted_u32::nwa::{NWA, NWAState};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::equiv_types::InternalIdMap;
use crate::ds::weight::Weight;

pub use compaction::{CompactLayout, InternedRangeCounts};
#[cfg(feature = "internal-api")]
pub use reconcile::{remap_weights_with_maps, remap_weights_with_token_axis, remap_weights_with_axes, TokenAxisMap, refine_observation_classes, refine_vocab_observations, refine_sparse_memberships, local_to_refined_classes};

pub fn common_internal_id_map(inputs: &[&InternalIdMap]) -> InternalIdMap {
    reconcile::build_common_internal_id_map(inputs)
}

/// Select an existing exact refinement when possible; otherwise build the
/// common observation classes in the supplied source order.
pub fn joint_observation_id_map(inputs: &[&InternalIdMap]) -> InternalIdMap {
    assert!(!inputs.is_empty(), "a joint observation coordinate needs a source");
    if let Some(finer) = inputs.iter().copied().find(|candidate| {
        inputs.iter().all(|other| {
            same_internal_id_maps(candidate, other)
                || reconcile::internal_id_map_refines(candidate, other)
        })
    }) {
        finer.clone().into_dense()
    } else {
        reconcile::build_common_internal_id_map(inputs)
    }
}

/// One ordered traversal through weight-bearing leaves. Nested artifacts
/// never allocate intermediate reference lists; a presence query may stop
/// at the first leaf without allocating any list at all.
pub trait WeightRefs {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()>;
    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()>;

    fn weight_refs(&self) -> Vec<&Weight> {
        let mut out = Vec::new();
        let _ = self.visit_weights(&mut |weight| {
            out.push(weight);
            ControlFlow::Continue(())
        });
        out
    }

    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut out = Vec::new();
        let _ = self.visit_weights_mut(&mut |weight| {
            out.push(weight);
            ControlFlow::Continue(())
        });
        out
    }

    fn has_weights(&self) -> bool {
        self.visit_weights(&mut |_| ControlFlow::Break(())).is_break()
    }
}

impl WeightRefs for Weight {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        visit(self)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        visit(self)
    }
}

impl WeightRefs for DWAState {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        if let Some(weight) = &self.final_weight { visit(weight)?; }
        for (_, weight) in self.transitions.values() { visit(weight)?; }
        ControlFlow::Continue(())
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        if let Some(weight) = &mut self.final_weight { visit(weight)?; }
        for (_, weight) in self.transitions.values_mut() { visit(weight)?; }
        ControlFlow::Continue(())
    }
}

impl WeightRefs for NWAState {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        if let Some(weight) = &self.final_weight { visit(weight)?; }
        for (_, weight) in self.transitions.values().flatten().chain(&self.epsilons) { visit(weight)?; }
        ControlFlow::Continue(())
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        if let Some(weight) = &mut self.final_weight { visit(weight)?; }
        for (_, weight) in self.transitions.values_mut().flatten().chain(&mut self.epsilons) { visit(weight)?; }
        ControlFlow::Continue(())
    }
}

impl WeightRefs for DWA {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.states().visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.states_mut().visit_weights_mut(visit)
    }
}

impl WeightRefs for NWA {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.states().visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.states_mut().visit_weights_mut(visit)
    }
}

impl WeightRefs for TerminalAutomaton {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        match self {
            Self::Dwa(dwa) => dwa.visit_weights(visit),
            Self::TokenDeterministicNwa(nwa) | Self::EpsilonNwa(nwa) => nwa.visit_weights(visit),
        }
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        match self {
            Self::Dwa(dwa) => dwa.visit_weights_mut(visit),
            Self::TokenDeterministicNwa(nwa) | Self::EpsilonNwa(nwa) => nwa.visit_weights_mut(visit),
        }
    }
}

impl<K: Ord, T: WeightRefs> WeightRefs for BTreeMap<K, T> {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        for value in self.values() { value.visit_weights(visit)?; }
        ControlFlow::Continue(())
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        for value in self.values_mut() { value.visit_weights_mut(visit)?; }
        ControlFlow::Continue(())
    }
}

impl<A: WeightRefs, B: WeightRefs> WeightRefs for (A, B) {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.0.visit_weights(visit)?;
        self.1.visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.0.visit_weights_mut(visit)?;
        self.1.visit_weights_mut(visit)
    }
}

impl<A: WeightRefs, B: WeightRefs, C: WeightRefs> WeightRefs for (A, B, C) {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.0.visit_weights(visit)?;
        self.1.visit_weights(visit)?;
        self.2.visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.0.visit_weights_mut(visit)?;
        self.1.visit_weights_mut(visit)?;
        self.2.visit_weights_mut(visit)
    }
}

impl<T: WeightRefs> WeightRefs for [T] {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        for value in self { value.visit_weights(visit)?; }
        ControlFlow::Continue(())
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        for value in self { value.visit_weights_mut(visit)?; }
        ControlFlow::Continue(())
    }
}

impl<T: WeightRefs> WeightRefs for Vec<T> {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.as_slice().visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        self.as_mut_slice().visit_weights_mut(visit)
    }
}

impl<T: WeightRefs + ?Sized> WeightRefs for &mut T {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        (**self).visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        (**self).visit_weights_mut(visit)
    }
}

impl<T: WeightRefs + ?Sized> WeightRefs for Box<T> {
    fn visit_weights<'a>(&'a self, visit: &mut impl FnMut(&'a Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        (**self).visit_weights(visit)
    }

    fn visit_weights_mut<'a>(&'a mut self, visit: &mut impl FnMut(&'a mut Weight) -> ControlFlow<()>) -> ControlFlow<()> {
        (**self).visit_weights_mut(visit)
    }
}

#[derive(Debug, Clone)]
pub struct MappedArtifact<T: WeightRefs> {
    artifact: T,
    id_map: InternalIdMap,
}

impl<T: WeightRefs> MappedArtifact<T> {
    /// Invariant: `artifact` IDs are expressed in the internal spaces described by `id_map`.
    pub fn new(artifact: T, id_map: InternalIdMap) -> Self {
        Self { artifact, id_map }
    }

    pub fn artifact(&self) -> &T {
        &self.artifact
    }

    pub fn artifact_mut(&mut self) -> &mut T {
        &mut self.artifact
    }

    pub fn id_map(&self) -> &InternalIdMap {
        &self.id_map
    }

    pub fn parts_mut(&mut self) -> (&mut T, &mut InternalIdMap) {
        (&mut self.artifact, &mut self.id_map)
    }

    pub fn into_parts(self) -> (T, InternalIdMap) {
        (self.artifact, self.id_map)
    }

    pub fn into_artifact(self) -> T {
        self.artifact
    }

    pub fn compact_dimensions(&mut self, layout: CompactLayout) {
        let mut weights = self.artifact.weight_refs_mut();
        compaction::compact_weights_with_id_map(&mut weights, &mut self.id_map, layout);
    }

    pub fn interned_range_counts(&mut self) -> InternedRangeCounts {
        count_interned_ranges_for_weights(self.artifact.weight_refs())
    }

    /// Consume the local artifact while transporting its weights into a
    /// caller-owned common coordinate. The destination map is not cloned.
    pub fn into_common_artifact(mut self, common: &InternalIdMap) -> T {
        if !same_internal_id_maps(self.id_map(), common) {
            let (artifact, local) = self.parts_mut();
            let mut weights = artifact.weight_refs_mut();
            debug_assert!(weights.is_empty() || reconcile::internal_id_map_refines(common, local),
                "common coordinate must refine every weighted source");
            reconcile::remap_weights_into_existing_common(&mut weights, local, common);
        }
        self.into_artifact()
    }

    /// Remap this artifact into an already-computed exact common refinement.
    ///
    /// The supplied map must refine this artifact's current coordinates. This
    /// avoids rediscovering the same common partition when another concurrent
    /// stage has already certified it.
    pub fn remap_into_existing_common(mut self, common_id_map: &InternalIdMap) -> Self {
        if same_internal_id_maps(self.id_map(), common_id_map) {
            self.id_map = common_id_map.clone();
            return self;
        }
        debug_assert!(
            reconcile::internal_id_map_refines(common_id_map, self.id_map()),
            "precomputed common ID map must refine the artifact's local map",
        );
        let (artifact, id_map) = self.parts_mut();
        let mut weights = artifact.weight_refs_mut();
        reconcile::remap_weights_into_existing_common(&mut weights, id_map, common_id_map);
        *id_map = common_id_map.clone();
        self
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
            return MappedArtifact::new((left_artifact, right_artifact), id_map.into_dense());
        }

        let common_id_map = {
            let (left_artifact, left_id_map) = left.parts_mut();
            let (right_artifact, right_id_map) = right.parts_mut();
            let mut left_weights = left_artifact.weight_refs_mut();
            let mut right_weights = right_artifact.weight_refs_mut();
            reconcile::reconcile_weight_id_maps_into_common(
                &mut left_weights,
                left_id_map,
                &mut right_weights,
                right_id_map,
            )
        };
        MappedArtifact::new((left.into_artifact(), right.into_artifact()), common_id_map)
    }
}

fn same_internal_id_maps(left: &InternalIdMap, right: &InternalIdMap) -> bool {
    left.tokenizer_states.same_coordinate(&right.tokenizer_states)
        && left.vocab_tokens.same_coordinate(&right.vocab_tokens)
}

impl<T> MappedArtifact<Vec<T>>
where
    T: WeightRefs,
{
    /// Build the joint observation partition before touching any weights.
    /// Repeated pairwise merging used to remap the accumulated artifacts
    /// again whenever another family split an existing class.
    pub fn reconcile_vec(inputs: Vec<MappedArtifact<T>>) -> MappedArtifact<Vec<T>> {
        Self::reconcile_all(inputs, false)
    }

    /// Boundary observations have a deliberately partial raw-state domain.
    /// Keep states absent from every input absent from the common coordinate.
    pub fn reconcile_vec_preserving_unmapped_tokenizer(
        inputs: Vec<MappedArtifact<T>>,
    ) -> MappedArtifact<Vec<T>> {
        Self::reconcile_all(inputs, true)
    }

    fn reconcile_all(inputs: Vec<MappedArtifact<T>>, sparse: bool) -> MappedArtifact<Vec<T>> {
        assert!(!inputs.is_empty(), "cannot reconcile an empty artifact inventory");
        if inputs.len() == 1 {
            let (artifact, id_map) = inputs.into_iter().next().unwrap().into_parts();
            return MappedArtifact::new(vec![artifact], id_map.into_dense());
        }

        let common = {
            // A weightless ordinary artifact has no observation capable of
            // distinguishing classes. Preserve the historical empty-map
            // fast path rather than refining on irrelevant coordinates.
            let mut maps = inputs.iter().filter(|input|
                sparse || input.artifact().has_weights())
                .map(MappedArtifact::id_map).collect::<Vec<_>>();
            if maps.is_empty() {
                maps.push(inputs[0].id_map());
            }
            if !sparse {
                joint_observation_id_map(&maps)
            } else {
                reconcile::build_common_internal_id_map_preserving_unmapped_tokenizer(&maps)
            }
        };
        let artifacts = inputs.into_iter().map(|mut input| {
            if !same_internal_id_maps(input.id_map(), &common) {
                let (artifact, local) = input.parts_mut();
                let mut weights = artifact.weight_refs_mut();
                reconcile::remap_weights_into_existing_common(&mut weights, local, &common);
            }
            input.into_artifact()
        }).collect();
        MappedArtifact::new(artifacts, common)
    }

}

impl InternedRangeCounts {
    pub fn total_ranges(self) -> usize {
        self.tsid_ranges + self.token_ranges
    }
}

pub fn count_interned_ranges_for_weights<'a>(
    weights: impl IntoIterator<Item = &'a Weight>,
) -> InternedRangeCounts {
    let weight_refs: Vec<&Weight> = weights.into_iter().collect();
    compaction::count_interned_ranges_for_weight_refs(&weight_refs)
}

#[cfg(test)]
mod weight_visit_tests {
    use super::*;

    #[test]
    fn ordered_nested_weight_traversal_visits_each_leaf_once() {
        let mut graph = NWA::new(1, 1);
        graph.add_state();
        let state = &mut graph.states_mut()[0];
        state.final_weight = Some(Weight::all());
        state.transitions.insert(7, vec![(0, Weight::empty()), (0, Weight::all())]);
        state.epsilons.push((0, Weight::empty()));
        let mut nested = (BTreeMap::from([(3, vec![Weight::empty(), Weight::all()])]), Box::new(graph));
        let expected = [false, true, true, false, true, false];
        assert_eq!(nested.weight_refs().iter().map(|w| w.is_full()).collect::<Vec<_>>(), expected);
        let mut visited = 0;
        assert!(nested.visit_weights(&mut |_| {
            visited += 1;
            ControlFlow::Break(())
        }).is_break());
        assert_eq!(visited, 1);
        for (weight, full) in nested.weight_refs_mut().into_iter().zip(expected) {
            *weight = if full { Weight::empty() } else { Weight::all() };
        }
        assert_eq!(nested.weight_refs().iter().map(|w| w.is_full()).collect::<Vec<_>>(), expected.map(|full| !full));
    }
}
