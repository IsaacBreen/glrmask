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
//! # Quick Start
//!
//! ```rust,ignore
//! use sep1::interface::json_schema::json_schema_to_ebnf;
//!
//! let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
//! let ebnf = json_schema_to_ebnf(schema).unwrap();
//! ```

use crate::interface::GrammarExpr;
use serde_json::Value;

// Type definitions
pub mod types;

// Stage 1: Parse JSON Schema to SchemaType
pub mod parser;

// Stage 2: Convert SchemaType to GrammarType
pub mod convert;

// Stage 3: Emit GrammarType to GrammarExpr
pub mod emit;

// Tests
#[cfg(test)]
mod tests;

// Re-export common types and functions for convenience
pub use types::{SchemaType, GrammarType};
pub use parser::parse_json_schema;
pub use convert::SchemaToGrammar;
pub use emit::GrammarEmitter;

// ============================================================================
// Public API Functions
// ============================================================================

/// Convert a JSON Schema string to EBNF string.
///
/// This is the main entry point for JSON Schema → EBNF conversion.
pub fn json_schema_to_ebnf(schema_json: &str) -> Result<String, String> {
    let rules = json_schema_to_grammar_exprs(schema_json)?;
    
    // Check if whitespace is disabled
    let no_whitespace = std::env::var("SEP1_NO_JSON_WHITESPACE")
        .map(|v| v == "1")
        .unwrap_or(false);
    
    // Convert rules to EBNF format
    let ignore_prefix = if no_whitespace { "" } else { "#![ignore(WS)]\n\n" };
    let mut ebnf = String::from(ignore_prefix);
    let prefix_len = ignore_prefix.len();
    
    let root_rule = "root".to_string();
    
    for (name, expr) in &rules {
        let ebnf_str = grammar_expr_to_ebnf(expr);
        if name == &root_rule {
            // Put root rule first (after ignore directive if present)
            ebnf = format!("{}{} ::= {} ;\n", ignore_prefix, name, ebnf_str) + &ebnf[prefix_len..];
        } else {
            ebnf.push_str(&format!("{} ::= {} ;\n", name, ebnf_str));
        }
    }
    
    Ok(ebnf)
}

/// Convert a JSON Schema to a Vec<(String, GrammarExpr)>.
pub fn json_schema_to_grammar_exprs(schema_json: &str) -> Result<Vec<(String, GrammarExpr)>, String> {
    // Parse JSON
    let schema: Value = serde_json::from_str(schema_json)
        .map_err(|e| format!("Failed to parse JSON schema: {}", e))?;
    
    // Use the converter
    let mut converter = JsonSchemaConverter::new(schema);
    let (rules, _root_rule) = converter.convert()?;
    
    Ok(rules)
}

/// JSON Schema to Grammar converter.
///
/// This is a wrapper around the modular conversion pipeline that maintains
/// backward compatibility with the original API.
pub struct JsonSchemaConverter {
    root_schema: Value,
}

impl JsonSchemaConverter {
    pub fn new(schema: Value) -> Self {
        Self { root_schema: schema }
    }
    
    /// Sanitize a name to be a valid EBNF rule name
    fn sanitize_rule_name(name: &str) -> String {
        let mut result = String::with_capacity(name.len());
        for c in name.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => result.push(c),
                '<' | '>' | '[' | ']' | '(' | ')' | '{' | '}' => result.push('_'),
                '/' | '-' | '.' | ':' | '#' | ' ' => result.push('_'),
                _ => result.push('_'),
            }
        }
        result
    }
    
    /// Convert a ref path like "#/$defs/Foo" to a rule name like "_def_Foo"
    fn ref_path_to_rule_name(ref_path: &str) -> String {
        // Extract the definition name from the path
        if let Some(name) = ref_path.strip_prefix("#/$defs/") {
            format!("_def_{}", Self::sanitize_rule_name(name))
        } else if let Some(name) = ref_path.strip_prefix("#/definitions/") {
            format!("_def_{}", Self::sanitize_rule_name(name))
        } else if let Some(name) = ref_path.strip_prefix("#/defs/") {
            format!("_def_{}", Self::sanitize_rule_name(name))
        } else if let Some(name) = ref_path.strip_prefix("#/refs/") {
            format!("_ref_{}", Self::sanitize_rule_name(name))
        } else {
            // Fallback: sanitize the whole path
            format!("_ref_{}", Self::sanitize_rule_name(ref_path))
        }
    }
    
    pub fn convert(self) -> Result<(Vec<(String, GrammarExpr)>, String), String> {
        // Parse schema
        let mut parser = parser::SchemaParser::new(self.root_schema);
        let root_schema_type = parser.parse_root()?;
        
        // Parse all definitions
        let definitions = parser.parse_definitions()?;
        
        // Convert root to grammar type
        let mut converter = convert::SchemaToGrammar::new();
        let root_grammar_type = converter.convert(&root_schema_type);
        
        // Convert all definitions to grammar types
        let mut def_rules: Vec<(String, types::GrammarType)> = Vec::new();
        for (ref_path, schema_type) in definitions {
            let rule_name = Self::ref_path_to_rule_name(&ref_path);
            let grammar_type = converter.convert(&schema_type);
            def_rules.push((rule_name, grammar_type));
        }
        
        // Get primitive needs
        let needs = converter.get_needs();
        
        // Emit to GrammarExpr
        let mut emitter = emit::GrammarEmitter::new();
        
        // Emit root rule
        let root_expr = emitter.emit(&root_grammar_type);
        emitter.add_rule("root".to_string(), root_expr);
        
        // Emit definition rules
        for (rule_name, grammar_type) in &def_rules {
            let expr = emitter.emit(grammar_type);
            emitter.add_rule(rule_name.clone(), expr);
        }
        
        // Add primitive rules as needed
        emitter.add_primitive_rules(
            needs.json_value,
            needs.json_object,
            needs.json_array,
            needs.json_kv,
        );
        
        let rules = emitter.into_rules();
        Ok((rules, "root".to_string()))
    }
}

/// Convert a GrammarExpr to EBNF string.
fn grammar_expr_to_ebnf(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Ref(name) => name.clone(),
        GrammarExpr::Literal(bytes) => {
            let s = String::from_utf8_lossy(bytes);
            // Escape special characters for EBNF literal
            let mut escaped = String::new();
            for c in s.chars() {
                match c {
                    '\\' => escaped.push_str("\\\\"),
                    '\'' => escaped.push_str("\\'"),
                    '\n' => escaped.push_str("\\n"),
                    '\r' => escaped.push_str("\\r"),
                    '\t' => escaped.push_str("\\t"),
                    c if c.is_control() => {
                        escaped.push_str(&format!("\\x{:02x}", c as u32));
                    }
                    _ => escaped.push(c),
                }
            }
            format!("'{}'", escaped)
        }
        GrammarExpr::Sequence(exprs) => {
            let parts: Vec<String> = exprs.iter().map(grammar_expr_to_ebnf).collect();
            parts.join(" ")
        }
        GrammarExpr::Choice(exprs) => {
            let parts: Vec<String> = exprs.iter().map(|e| grammar_expr_to_ebnf(e)).collect();
            format!("( {} )", parts.join(" | "))
        }
        GrammarExpr::Optional(e) => {
            format!("( {} )?", grammar_expr_to_ebnf(e))
        }
        GrammarExpr::Repeat(e) => {
            format!("( {} )*", grammar_expr_to_ebnf(e))
        }
        GrammarExpr::RepeatBounded { min, max, inner } => {
            let inner_str = grammar_expr_to_ebnf(inner);
            match max {
                Some(max_val) if *min == *max_val => {
                    // Exact count: {n}
                    format!("( {} ){{{}}}", inner_str, min)
                }
                Some(max_val) => {
                    // Range: {min,max}
                    format!("( {} ){{{},{}}}", inner_str, min, max_val)
                }
                None => {
                    // Unbounded: {min,}
                    format!("( {} ){{{},}}", inner_str, min)
                }
            }
        }
        GrammarExpr::CharClass(s) => s.clone(),
        GrammarExpr::AnyChar => ".".to_string(),
    }
}
