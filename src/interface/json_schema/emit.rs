//! Grammar Emission
//!
//! Converts GrammarType to GrammarExpr - the final stage of the pipeline.
//!
//! This module handles:
//! - Converting intermediate grammar representation to final GrammarExpr
//! - Generating primitive rule definitions (JSON_STRING, etc.)
//! - Rule deduplication and optimization

use super::types::*;
use crate::interface::GrammarExpr;

/// Emitter that converts GrammarType to GrammarExpr
pub struct GrammarEmitter {
    /// Generated rules: (name, expr)
    rules: Vec<(String, GrammarExpr)>,
    /// Track which primitives have been added
    primitives_added: bool,
}

impl GrammarEmitter {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            primitives_added: false,
        }
    }
    
    /// Emit a GrammarType as a GrammarExpr
    pub fn emit(&mut self, grammar: &GrammarType) -> GrammarExpr {
        match grammar {
            GrammarType::PrimitiveRef(p) => {
                GrammarExpr::Ref(p.rule_name().to_string())
            }
            
            GrammarType::Literal(bytes) => {
                GrammarExpr::Literal(bytes.clone())
            }
            
            GrammarType::Sequence(items) => {
                let exprs: Vec<GrammarExpr> = items.iter()
                    .map(|g| self.emit(g))
                    .collect();
                GrammarExpr::Sequence(exprs)
            }
            
            GrammarType::Choice(items) => {
                let exprs: Vec<GrammarExpr> = items.iter()
                    .map(|g| self.emit(g))
                    .collect();
                GrammarExpr::Choice(exprs)
            }
            
            GrammarType::Optional(inner) => {
                GrammarExpr::Optional(Box::new(self.emit(inner)))
            }
            
            GrammarType::Repeat(inner) => {
                GrammarExpr::Repeat(Box::new(self.emit(inner)))
            }
            
            GrammarType::RepeatOnePlus(inner) => {
                // a+ = a a*
                let inner_expr = self.emit(inner);
                GrammarExpr::Sequence(vec![
                    inner_expr.clone(),
                    GrammarExpr::Repeat(Box::new(inner_expr)),
                ])
            }
            
            GrammarType::RepeatBounded { min, max, inner } => {
                GrammarExpr::RepeatBounded {
                    min: *min,
                    max: *max,
                    inner: Box::new(self.emit(inner)),
                }
            }
            
            GrammarType::CharClass(pattern) => {
                GrammarExpr::CharClass(pattern.clone())
            }
            
            GrammarType::RuleRef(name) => {
                GrammarExpr::Ref(name.clone())
            }
            
            GrammarType::RuleDefinition(name, body) => {
                let expr = self.emit(body);
                self.rules.push((name.clone(), expr.clone()));
                GrammarExpr::Ref(name.clone())
            }
            
            GrammarType::JsonObject { open, content, close } => {
                GrammarExpr::Sequence(vec![
                    self.emit(open),
                    self.emit(content),
                    self.emit(close),
                ])
            }
            
            GrammarType::JsonArray { open, content, close } => {
                GrammarExpr::Sequence(vec![
                    self.emit(open),
                    self.emit(content),
                    self.emit(close),
                ])
            }
            
            GrammarType::JsonKeyValue { key, colon, value } => {
                GrammarExpr::Sequence(vec![
                    self.emit(key),
                    self.emit(colon),
                    self.emit(value),
                ])
            }
            
            GrammarType::Empty => {
                GrammarExpr::Sequence(vec![])
            }
        }
    }
    
    /// Add a rule
    pub fn add_rule(&mut self, name: String, expr: GrammarExpr) {
        self.rules.push((name, expr));
    }
    
    /// Add primitive JSON rules
    pub fn add_primitive_rules(&mut self, needs_value: bool, needs_object: bool, needs_array: bool, needs_kv: bool) {
        if self.primitives_added {
            return;
        }
        self.primitives_added = true;
        
        // Check if whitespace is disabled
        let no_whitespace = std::env::var("SEP1_NO_JSON_WHITESPACE")
            .map(|v| v == "1")
            .unwrap_or(false);
        
        if !no_whitespace {
            self.rules.push(("WS".to_string(), GrammarExpr::Repeat(Box::new(
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b" ".to_vec()),
                    GrammarExpr::Literal(b"\t".to_vec()),
                    GrammarExpr::Literal(b"\n".to_vec()),
                    GrammarExpr::Literal(b"\r".to_vec()),
                ])
            ))));
        }
        
        // JSON_STRING
        self.rules.push(("JSON_STRING".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            GrammarExpr::Ref("STRING_CHARS".to_string()),
            GrammarExpr::Literal(b"\"".to_vec()),
        ])));
        
        // STRING_CHARS = STRING_CHAR*
        self.rules.push(("STRING_CHARS".to_string(), GrammarExpr::Repeat(Box::new(
            GrammarExpr::Choice(vec![
                GrammarExpr::Ref("STRING_CHAR".to_string()),
                GrammarExpr::Ref("ESCAPE_SEQ".to_string()),
            ])
        ))));
        
        // STRING_CHAR - printable chars except " and \
        self.rules.push(("STRING_CHAR".to_string(), 
            GrammarExpr::CharClass("[^\"\\\\\\x00-\\x1f]".to_string())
        ));
        
        // ESCAPE_SEQ
        self.rules.push(("ESCAPE_SEQ".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\\".to_vec()),
            GrammarExpr::Choice(vec![
                GrammarExpr::CharClass("[\"\\\\/bfnrt]".to_string()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"u".to_vec()),
                    GrammarExpr::CharClass("[0-9a-fA-F]".to_string()),
                    GrammarExpr::CharClass("[0-9a-fA-F]".to_string()),
                    GrammarExpr::CharClass("[0-9a-fA-F]".to_string()),
                    GrammarExpr::CharClass("[0-9a-fA-F]".to_string()),
                ]),
            ]),
        ])));
        
        // JSON_NUMBER
        self.rules.push(("JSON_NUMBER".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
            GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"0".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::CharClass("[1-9]".to_string()),
                    GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass("[0-9]".to_string()))),
                ]),
            ]),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b".".to_vec()),
                GrammarExpr::CharClass("[0-9]".to_string()),
                GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass("[0-9]".to_string()))),
            ]))),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::CharClass("[eE]".to_string()),
                GrammarExpr::Optional(Box::new(GrammarExpr::CharClass("[+-]".to_string()))),
                GrammarExpr::CharClass("[0-9]".to_string()),
                GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass("[0-9]".to_string()))),
            ]))),
        ])));
        
        // JSON_INTEGER (subset of number without fraction/exponent)
        self.rules.push(("JSON_INTEGER".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
            GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"0".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::CharClass("[1-9]".to_string()),
                    GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass("[0-9]".to_string()))),
                ]),
            ]),
        ])));
        
        // JSON_BOOL
        self.rules.push(("JSON_BOOL".to_string(), GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"true".to_vec()),
            GrammarExpr::Literal(b"false".to_vec()),
        ])));
        
        // JSON_NULL
        self.rules.push(("JSON_NULL".to_string(), GrammarExpr::Literal(b"null".to_vec())));
        
        // Conditional rules based on needs
        // Note: _json_kv and _json_object both reference _json_value, so if either is needed,
        // we must emit _json_value
        let needs_value = needs_value || needs_kv || needs_object;
        
        if needs_value {
            self.rules.push(("_json_value".to_string(), GrammarExpr::Choice(vec![
                GrammarExpr::Ref("_json_object".to_string()),
                GrammarExpr::Ref("_json_array".to_string()),
                GrammarExpr::Ref("JSON_STRING".to_string()),
                GrammarExpr::Ref("JSON_NUMBER".to_string()),
                GrammarExpr::Ref("JSON_BOOL".to_string()),
                GrammarExpr::Ref("JSON_NULL".to_string()),
            ])));
        }
        
        if needs_object || needs_value {
            let comma_kv = GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b",".to_vec()),
                GrammarExpr::Ref("_json_kv".to_string()),
            ]);
            
            self.rules.push(("_json_object".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"{".to_vec()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref("_json_kv".to_string()),
                    GrammarExpr::Repeat(Box::new(comma_kv)),
                ]))),
                GrammarExpr::Literal(b"}".to_vec()),
            ])));
        }
        
        if needs_kv || needs_object || needs_value {
            self.rules.push(("_json_kv".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("JSON_STRING".to_string()),
                GrammarExpr::Literal(b":".to_vec()),
                GrammarExpr::Ref("_json_value".to_string()),
            ])));
        }
        
        if needs_array || needs_value {
            let comma_val = GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b",".to_vec()),
                GrammarExpr::Ref("_json_value".to_string()),
            ]);
            
            self.rules.push(("_json_array".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"[".to_vec()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref("_json_value".to_string()),
                    GrammarExpr::Repeat(Box::new(comma_val)),
                ]))),
                GrammarExpr::Literal(b"]".to_vec()),
            ])));
        }
    }
    
    /// Get the generated rules
    pub fn into_rules(self) -> Vec<(String, GrammarExpr)> {
        self.rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_emit_literal() {
        let mut emitter = GrammarEmitter::new();
        let g = GrammarType::lit("hello");
        let expr = emitter.emit(&g);
        assert!(matches!(expr, GrammarExpr::Literal(_)));
    }
    
    #[test]
    fn test_emit_sequence() {
        let mut emitter = GrammarEmitter::new();
        let g = GrammarType::seq(vec![
            GrammarType::lit("{"),
            GrammarType::lit("}"),
        ]);
        let expr = emitter.emit(&g);
        assert!(matches!(expr, GrammarExpr::Sequence(_)));
    }
}
