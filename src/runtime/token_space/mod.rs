//! Runtime token-space materialization.
//!
//! Compilation may quotient the original vocabulary into a smaller internal
//! token id space. Runtime Mask computes in that internal space and then
//! materializes back to original token ids here.

pub(crate) mod final_mask_mapping;
