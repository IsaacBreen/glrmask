//! Public compiled-constraint type.
//!
//! A [`Constraint`] is the immutable result of compiling a source grammar and a
//! vocabulary.  It owns the Terminal DWA, Parser DWA, tokenizer, parser table,
//! token-space maps, and final mask materialization data.  The details stay in
//! `runtime` and `compile`; users only need the following lifecycle:
//!
//! ```text
//! source grammar + Vocab  ──compile──▶  Constraint
//! Constraint::start()     ───────────▶  ConstraintState
//! ```
//!
//! The constructors live as inherent methods on this type for compatibility:
//! `from_json_schema`, `from_lark`, `from_ebnf`, and `from_glrm_grammar`.

#[doc(inline)]
pub use crate::runtime::Constraint;
