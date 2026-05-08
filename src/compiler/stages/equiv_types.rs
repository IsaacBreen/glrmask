//! Shared equivalence-analysis types used across multiple compiler stages.
//!
//! `ManyToOneIdMap` and `InternalIdMap` are pure data types that represent
//! equivalence-class mappings.  The analysis that *produces* them lives in
//! `id_map_and_terminal_dwa::l2p::equivalence_analysis`.

#[derive(Debug, Clone)]
pub struct ManyToOneIdMap {
    pub original_to_internal: Vec<u32>,
    pub internal_to_originals: Vec<Vec<u32>>,
    pub representative_original_ids: Vec<u32>,
}

impl ManyToOneIdMap {
    /// Construct from a pre-computed original→internal mapping with explicit representatives.
    pub fn from_original_to_internal_with_representatives(
        original_to_internal: Vec<u32>,
        num_internal: u32,
        representative_original_ids: Vec<u32>,
    ) -> Self {
        let mut internal_to_originals = vec![Vec::new(); num_internal as usize];
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if (internal as usize) < internal_to_originals.len() {
                internal_to_originals[internal as usize].push(original as u32);
            }
        }
        Self {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        }
    }

    pub fn from_original_to_internal_allowing_unmapped(
        original_to_internal: Vec<u32>,
        num_internal: u32,
    ) -> Self {
        let mut internal_to_originals = vec![Vec::new(); num_internal as usize];
        let mut representative_original_ids = vec![u32::MAX; num_internal as usize];
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if internal == u32::MAX || (internal as usize) >= internal_to_originals.len() {
                continue;
            }
            let originals = &mut internal_to_originals[internal as usize];
            if originals.is_empty() {
                representative_original_ids[internal as usize] = original as u32;
            }
            originals.push(original as u32);
        }
        Self {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        }
    }

    pub fn num_internal_ids(&self) -> u32 {
        self.internal_to_originals.len() as u32
    }

    pub fn compose(&self, next: &ManyToOneIdMap) -> Self {
        let mut original_to_internal = vec![u32::MAX; self.original_to_internal.len()];
        for (original, &mid) in self.original_to_internal.iter().enumerate() {
            if mid == u32::MAX {
                continue;
            }
            original_to_internal[original] = next
                .original_to_internal
                .get(mid as usize)
                .copied()
                .unwrap_or(u32::MAX);
        }
        Self::from_original_to_internal_allowing_unmapped(
            original_to_internal,
            next.num_internal_ids(),
        )
    }

    pub fn representative_original_id_for_internal(&self, internal_id: u32) -> Option<u32> {
        self.representative_original_ids
            .get(internal_id as usize)
            .copied()
    }

    pub fn iter_representative_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.representative_original_ids.iter().copied()
    }
    pub fn internal_to_originals_vecs(&self) -> Vec<Vec<u32>> {
        self.internal_to_originals.clone()
    }
    /// Fill any unmapped original entries (`u32::MAX`) into a new
    /// internal class.  This is safe when the simplified DFA dropped
    /// states that had no active-terminal future: they get a class that
    /// contributes no allowed tokens.
    pub fn fill_unmapped_with_new_class(mut self) -> Self {
        if !self.original_to_internal.iter().any(|&id| id == u32::MAX) {
            return self;
        }
        let new_internal = self.internal_to_originals.len() as u32;
        let mut originals = Vec::new();
        for (original, internal) in self.original_to_internal.iter_mut().enumerate() {
            if *internal == u32::MAX {
                *internal = new_internal;
                originals.push(original as u32);
            }
        }
        let representative = originals.first().copied().unwrap_or(u32::MAX);
        self.internal_to_originals.push(originals);
        self.representative_original_ids.push(representative);
        self
    }
}

#[derive(Debug, Clone)]
pub struct InternalIdMap {
    pub tokenizer_states: ManyToOneIdMap,
    pub vocab_tokens: ManyToOneIdMap,
}

#[derive(Debug, Clone)]
pub(crate) struct MappedArtifact<T> {
    artifact: T,
    id_map: InternalIdMap,
}

impl<T> MappedArtifact<T> {
    /// Invariant: `artifact` IDs are expressed in the internal spaces described by `id_map`.
    pub(crate) fn new(artifact: T, id_map: InternalIdMap) -> Self {
        Self { artifact, id_map }
    }

    pub(crate) fn artifact(&self) -> &T {
        &self.artifact
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

    pub(crate) fn into_artifact(self) -> T {
        self.artifact
    }
}

impl<A, B> MappedArtifact<(A, B)> {
    pub(crate) fn split_pair(self) -> (MappedArtifact<A>, MappedArtifact<B>) {
        let ((left, right), id_map) = self.into_parts();
        (
            MappedArtifact::new(left, id_map.clone()),
            MappedArtifact::new(right, id_map),
        )
    }
}

impl<T> MappedArtifact<Vec<T>> {
    pub(crate) fn split_vec(self) -> Vec<MappedArtifact<T>> {
        let (artifacts, id_map) = self.into_parts();
        artifacts.into_iter().map(|artifact| MappedArtifact::new(artifact, id_map.clone())).collect()
    }
}

impl InternalIdMap {
    pub fn num_tsids(&self) -> u32 {
        self.tokenizer_states.num_internal_ids()
    }

    pub fn num_internal_tokens(&self) -> u32 {
        self.vocab_tokens.num_internal_ids()
    }

    pub fn max_internal_token_id(&self) -> u32 {
        self.num_internal_tokens().saturating_sub(1)
    }
}
