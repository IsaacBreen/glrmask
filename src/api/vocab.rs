//! Public vocabulary type.
//!
//! [`Vocab`] maps original token ids to their byte strings.  The constraint is
//! over bytes and grammar terminals, not over tokenizer text.  During compile
//! and runtime the implementation may introduce compact internal token ids; the
//! public vocabulary always remains the user-provided token-id space.

#[doc(inline)]
pub use crate::vocab::Vocab;
