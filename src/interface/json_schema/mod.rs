//! JSON Schema to EBNF grammar conversion.
//!
//! This module provides functionality to convert JSON Schema definitions into
//! EBNF grammars that can be used for grammar-constrained decoding.
//!
//! # Architecture
//!
//! The conversion happens in multiple stages:
//!
//! 1. **Parsing** ([`parser`]): JSON Schema JSON → [`SchemaType`](types::SchemaType)
//!    - Parses the JSON Schema document
//!    - Resolves `$ref` references
//!    - Detects cycles
//!
//! 2. **Conversion** ([`convert`]): [`SchemaType`](types::SchemaType) → [`GrammarType`](types::GrammarType)
//!    - Transforms schema-centric representation to grammar-centric
//!    - Handles property ordering constraints
//!    - Manages additionalProperties placement
//!
//! 3. **Emission** ([`emit`]): [`GrammarType`](types::GrammarType) → [`GrammarExpr`](crate::interface::GrammarExpr)
//!    - Converts to final grammar expression
//!    - Generates primitive rules (string, number, boolean, null)
//!
//! # Legacy Module
//!
//! The [`legacy`] module contains the original monolithic implementation.
//! It is kept for backward compatibility but will be deprecated.
//! New code should use the staged pipeline above.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use sep1::interface::json_schema::json_schema_to_ebnf;
//!
//! let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
//! let ebnf = json_schema_to_ebnf(schema).unwrap();
//! ```

// Type definitions
pub mod types;

// Stage 1: Parse JSON Schema to SchemaType
pub mod parser;

// Stage 2: Convert SchemaType to GrammarType
pub mod convert;

// Stage 3: Emit GrammarType to GrammarExpr
pub mod emit;

// Legacy monolithic module (deprecated)
pub mod legacy;

// Tests
#[cfg(test)]
mod tests;

// Re-export common types and functions for convenience
pub use types::{SchemaType, GrammarType};
pub use parser::parse_json_schema;
pub use convert::SchemaToGrammar;
pub use emit::GrammarEmitter;

// Re-export legacy functions and types for backward compatibility
pub use legacy::{json_schema_to_ebnf, json_schema_to_grammar_exprs, JsonSchemaConverter};
