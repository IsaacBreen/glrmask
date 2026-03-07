//! Runtime commit path.
//!
//! This module owns token/byte commit entrypoints and the shared byte
//! processing engine used to advance a live `ConstraintState`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;

impl<'a> ConstraintState<'a> {
    /// Commit a token: advance the constraint state.
    ///
    /// Infallible. If `token_id` is not in the vocabulary, the method is a
    /// no-op and the parser state is left unchanged. The next call to
    /// [`mask`] / [`fill_mask`] will reflect whatever state the parser is in
    /// after any bytes that *were* successfully committed.
    pub fn commit(
        &mut self,
        token_id: u32,
    ) {
        unimplemented!()
    }

    /// Commit raw bytes, advancing tokenizer and parser state.
    ///
    /// Infallible. If the bytes produce no valid parse continuations the next
    /// mask will simply be empty.
    pub fn commit_bytes(&mut self, bytes: &[u8]) {
        unimplemented!()
    }

    /// Commit multiple tokens in sequence (batch convenience wrapper).
    ///
    /// Equivalent to calling [`commit`] for each token ID in order.
    pub fn commit_tokens(&mut self, tokens: &[u32]) {
        unimplemented!()
    }

    /// Core byte-processing engine shared by `commit` and `commit_bytes`.
    pub(crate) fn process_bytes_raw(&mut self, bytes: &[u8]) {
        unimplemented!()
    }
}
