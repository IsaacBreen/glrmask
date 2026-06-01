//! Dense bitset vocabulary used by compiled runtime artifacts.
//!
//! A `DenseWords` value is a shared immutable dense bit-vector over final
//! runtime-internal token identifiers.  It is used in Parser-DWA weights,
//! CanMatch seed masks, and runtime cache entries.

use std::sync::Arc;

/// Shared dense bit-vector over final runtime-internal token ids.
pub(crate) type DenseWords = Arc<[u64]>;

/// Serde default for skipped/cache dense vectors.
pub(crate) fn empty_dense_words() -> DenseWords {
    Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice())
}
