mod interface;
mod tokenizer_combinators;
mod tests;
mod ebnf;
mod lark;
mod optimization;
mod ebnf_factoring;
mod extract_alternatives;

// JSON Schema conversion - new modular structure
pub mod json_schema_types;
pub mod json_schema_parser;
pub mod json_schema_convert;
pub mod json_schema_emit;

// Legacy JSON Schema module (uses old monolithic approach, will be deprecated)
pub mod json_schema;

// JSON Schema tests
#[cfg(test)]
mod test_json;

pub use ebnf::*;
pub use interface::{choice, display_productions, literal, optional, r#ref, repeat, sequence, CompiledGrammar, GrammarDefinition, GrammarExpr, IncrementalParser, ExprNullability, get_expr_nullability};
pub use tokenizer_combinators::*;
pub use json_schema::*;
