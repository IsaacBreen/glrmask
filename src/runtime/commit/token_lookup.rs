//! Vocabulary-token lookup for Commit.
//!
//! Commit accepts original vocabulary token ids.  This helper maps an original
//! token id to the byte string that must be scanned by the tokenizer.

use crate::runtime::constraint::Constraint;

pub(super) fn token_bytes_for_id(constraint: &Constraint, token_id: u32) -> Option<&[u8]> {
    constraint
        .token_bytes_dense
        .get(token_id as usize)
        .and_then(|bytes| bytes.as_deref())
        .or_else(|| constraint.token_bytes.get(&token_id).map(Vec::as_slice))
}

