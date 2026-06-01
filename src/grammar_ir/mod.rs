//! Grammar intermediate representation.
//!
//! This namespace is the boundary between source-language importers and the
//! compiler.  Importers build a [`NamedGrammar`].  Transforms rewrite it.
//! Lowering converts it into [`flat::GrammarDef`].  Renderers serialize or print
//! it without changing its denotation.

pub mod ast;
pub mod expr_nfa;
pub mod flat;
pub mod glrm;
pub mod lower;
pub mod render;
pub mod transforms;

pub use ast::{CommaSepShape, GrammarExpr, NamedGrammar, NamedRule};
pub use lower::{expr_to_grammar_expr, lower};
