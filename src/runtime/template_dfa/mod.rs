//! Runtime execution for compiled template DFAs.
//! This optional fast path must be extensionally equal to direct GLR advance whenever it applies.

pub(crate) mod advance;
