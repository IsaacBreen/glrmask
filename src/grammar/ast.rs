//! Compatibility shim for `crate::grammar::ast`.

pub use crate::grammar_ir::ast::*;
pub use crate::grammar_ir::lower::{expr_to_grammar_expr, lower};
pub(crate) use crate::grammar_ir::lower::separated_sequence::comma_sep_shape;
