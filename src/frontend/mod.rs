//! Frontend parsers: convert user-facing grammar formats to internal IR.
//!
//! All frontends parse their input into a shared `GrammarExpr` AST,
//! then lower it to the internal `GrammarDef` used by the compiler.

pub mod ebnf;
pub mod grammar_expr;
pub mod json_schema;
pub mod lark;
