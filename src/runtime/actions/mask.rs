//! Hot-path mask API.
//!
//! This module owns the public mask surface exposed by [`ConstraintState`].
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;

pub struct MaskView<'state, 'constraint> {
    state: &'state ConstraintState<'constraint>,
}

impl<'a> ConstraintState<'a> {
    /// Return the dedicated mask view for this live runtime state.
    pub fn mask_view(&self) -> MaskView<'_, 'a> {
        MaskView { state: self }
    }
}

impl MaskView<'_, '_> {
    /// Compute the allowed-token mask as a `Vec<u32>`.
    ///
    /// Token `i` is allowed iff `result[i / 32] & (1u32 << (i % 32)) != 0`.
    /// Allocate the buffer with [`crate::runtime::Constraint::mask_len`] words.
    pub fn mask(&self) -> Vec<u32> {
        let _ = self.state;
        unimplemented!()
    }

    /// Fill a pre-allocated mask buffer.
    ///
    /// `buf` must be at least `self.constraint.mask_len()` words long.
    /// Token `i` is allowed iff `buf[i / 32] & (1u32 << (i % 32)) != 0`.
    pub fn fill_mask(&self, buf: &mut [u32]) {
        let _ = self.state;
        let _ = buf;
        unimplemented!()
    }
}
