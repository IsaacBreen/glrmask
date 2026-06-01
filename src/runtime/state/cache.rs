//! Mask-cache storage owned by [`ConstraintState`](super::ConstraintState).
//!
//! These buffers are not semantic state.  They cache a materialized Mask result
//! for the current generation and may be discarded without changing the accepted
//! language.

/// Cached fill_mask result, keyed on generation counter.
pub(crate) struct MaskCacheData {
    pub generation: u64,
    pub mask: Vec<u32>,
    /// The merged internal token dense bitmap used to compute this mask.
    /// Enables incremental updates when the state changes slightly.
    pub merged_dense: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct MaskScratch {
    pub merged_dense: Vec<u64>,
    pub chain_merged_dense: Vec<u64>,
}
