//! Shared equivalence-analysis types used across multiple compiler stages.
//!
//! `ManyToOneIdMap` and `InternalIdMap` are pure data types that represent
//! equivalence-class mappings.  The analysis that *produces* them lives in
//! `id_map_and_terminal_dwa::l2p::equivalence_analysis`.

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ManyToOneIdMap {
    pub original_to_internal: Vec<u32>,
    pub internal_to_originals: Vec<Vec<u32>>,
    pub representative_original_ids: Vec<u32>,
}

impl ManyToOneIdMap {
    pub(crate) fn empty() -> Self {
        Self {
            original_to_internal: Vec::new(),
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        }
    }

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

    pub fn from_singleton_original_to_internal_with_representatives(
        original_to_internal: Vec<u32>,
        representative_original_ids: Vec<u32>,
    ) -> Self {
        debug_assert!(representative_original_ids
            .iter()
            .enumerate()
            .all(|(internal, &original)| original_to_internal
                .get(original as usize)
                .copied()
                == Some(internal as u32)));
        let internal_to_originals = representative_original_ids
            .iter()
            .map(|&original| vec![original])
            .collect();
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

    /// Relabel internal ids into a new order given by a caller-supplied sort
    /// key derived from each class's representative original id. This is a pure
    /// bijective relabel: the original→class partition is unchanged, only the
    /// integer names of the internal classes move. The sort is stable so
    /// equal-key classes keep their prior relative order.
    pub fn reorder_internal_by_representative_key<K: Ord>(
        &mut self,
        mut key: impl FnMut(u32) -> K,
    ) {
        let n = self.representative_original_ids.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&old_internal| key(self.representative_original_ids[old_internal]));
        // perm[old_internal] = new_internal
        let mut perm = vec![0u32; n];
        for (new_internal, &old_internal) in order.iter().enumerate() {
            perm[old_internal] = new_internal as u32;
        }
        for internal in self.original_to_internal.iter_mut() {
            if (*internal as usize) < n {
                *internal = perm[*internal as usize];
            }
        }
        let mut new_internal_to_originals = vec![Vec::new(); n];
        let mut new_representative_original_ids = vec![0u32; n];
        for (new_internal, &old_internal) in order.iter().enumerate() {
            new_internal_to_originals[new_internal] =
                std::mem::take(&mut self.internal_to_originals[old_internal]);
            new_representative_original_ids[new_internal] =
                self.representative_original_ids[old_internal];
        }
        self.internal_to_originals = new_internal_to_originals;
        self.representative_original_ids = new_representative_original_ids;
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

    /// Split one original ID out into its own class while preserving all
    /// existing class IDs.  This is used for structured epsilon lexers: the
    /// global dispatch state has no scalar byte transitions and must never be
    /// equated with an ordinary dead-looking DFA state, because its actual
    /// behavior is the union of the deterministic component roots.
    pub fn isolate_original(&mut self, original: u32) {
        let Some(class) = self
            .original_to_internal
            .get(original as usize)
            .copied()
            .filter(|&class| class != u32::MAX)
        else {
            return;
        };
        let Some(originals) = self.internal_to_originals.get_mut(class as usize) else {
            return;
        };
        if originals.len() <= 1 {
            return;
        }
        let Some(index) = originals.iter().position(|&candidate| candidate == original) else {
            return;
        };
        originals.swap_remove(index);
        if self.representative_original_ids[class as usize] == original {
            self.representative_original_ids[class as usize] = originals[0];
        }

        let new_class = self.internal_to_originals.len() as u32;
        self.original_to_internal[original as usize] = new_class;
        self.internal_to_originals.push(vec![original]);
        self.representative_original_ids.push(original);
    }
}

/// A structurally total representative map over the raw scanner-state domain.
///
/// This is deliberately distinct from the `ManyToOneIdMap` values threaded as
/// `initial_state_map` through id-map analysis. An initial map is an analysis
/// coordinate or seed and can be local, composed, or otherwise unsuitable for
/// terminal interchangeability. A `GlobalScannerStateQuotient` instead covers
/// every raw lexer state and names one raw representative for every class.
///
/// The type itself encodes only total raw-state coverage and one representative
/// per class. Despite the historical `Quotient` name, it does not prove that
/// class membership is symmetric equivalence: a producer may establish only a
/// directional member-to-representative relation. Partition C, for example,
/// uses positional subsumption and proves each member safe to replace by its
/// chosen representative at every vocabulary position where that member can
/// occur; two non-representative members need not replace one another.
/// Consumers must rely on the producer's representative-substitution contract
/// for the observations they make rather than assuming pairwise equivalence.
#[derive(Debug, Clone)]
pub(crate) struct GlobalScannerStateQuotient {
    map: ManyToOneIdMap,
}

impl GlobalScannerStateQuotient {
    /// Wrap a structurally total raw-state representative map. Semantic safety
    /// for a particular consumer follows from the producer's replacement
    /// proof, which may be directional rather than an equivalence relation.
    pub(crate) fn from_total_raw_state_map(map: ManyToOneIdMap, raw_state_count: usize) -> Self {
        assert_eq!(
            map.original_to_internal.len(),
            raw_state_count,
            "global scanner-state quotient must cover every raw lexer state",
        );
        assert_eq!(
            map.internal_to_originals.len(),
            map.representative_original_ids.len(),
            "global scanner-state quotient must have one representative slot per class",
        );
        for (raw_state, &class) in map.original_to_internal.iter().enumerate() {
            assert!(
                class != u32::MAX && (class as usize) < map.representative_original_ids.len(),
                "global scanner-state quotient omitted raw lexer state {raw_state}",
            );
        }
        for (class, &representative) in map.representative_original_ids.iter().enumerate() {
            assert!(
                (representative as usize) < raw_state_count
                    && map.original_to_internal[representative as usize] == class as u32,
                "global scanner-state quotient representative must belong to its class",
            );
        }
        Self { map }
    }

    #[inline]
    pub(crate) fn as_many_to_one(&self) -> &ManyToOneIdMap {
        &self.map
    }

    #[inline]
    pub(crate) fn raw_state_count(&self) -> usize {
        self.map.original_to_internal.len()
    }

}

#[derive(Debug, Clone)]
pub struct InternalIdMap {
    pub tokenizer_states: ManyToOneIdMap,
    pub vocab_tokens: ManyToOneIdMap,
    /// Internal-token order for a temporary singleton vocabulary coordinate.
    /// Local L1 artifacts are compacted immediately, so retaining the prepared
    /// order by `Arc` avoids cloning two dense token maps per partition.
    pub(crate) deferred_vocab_singleton_original_ids: Option<Arc<[u32]>>,
}

pub(crate) use super::mapped_artifact::MappedArtifact;

impl InternalIdMap {
    pub fn num_tsids(&self) -> u32 {
        self.tokenizer_states.num_internal_ids()
    }

    pub fn num_internal_tokens(&self) -> u32 {
        self.deferred_vocab_singleton_original_ids
            .as_ref()
            .map_or_else(|| self.vocab_tokens.num_internal_ids(), |ids| ids.len() as u32)
    }

    pub fn max_internal_token_id(&self) -> u32 {
        self.num_internal_tokens().saturating_sub(1)
    }

    pub(crate) fn internal_token_for_original(&self, original: u32) -> Option<u32> {
        if let Some(original_ids) = self.deferred_vocab_singleton_original_ids.as_ref() {
            return original_ids
                .iter()
                .position(|&candidate| candidate == original)
                .map(|internal| internal as u32);
        }
        self.vocab_tokens
            .original_to_internal
            .get(original as usize)
            .copied()
            .filter(|&internal| internal != u32::MAX)
    }

    pub(crate) fn materialize_deferred_vocab_singletons(&mut self) {
        let Some(original_ids) = self.deferred_vocab_singleton_original_ids.take() else {
            return;
        };
        let original_count = original_ids
            .iter()
            .copied()
            .max()
            .map_or(0, |max_original| max_original as usize + 1);
        let mut original_to_internal = vec![u32::MAX; original_count];
        let mut internal_to_originals = Vec::with_capacity(original_ids.len());
        let mut representative_original_ids = Vec::with_capacity(original_ids.len());
        for (internal, &original) in original_ids.iter().enumerate() {
            original_to_internal[original as usize] = internal as u32;
            internal_to_originals.push(vec![original]);
            representative_original_ids.push(original);
        }
        self.vocab_tokens = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        };
    }
}
