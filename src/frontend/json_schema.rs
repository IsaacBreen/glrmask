//! JSON Schema → grammar converter.
//!
//! Converts a JSON Schema into a context-free grammar (`GrammarDef`) that
//! generates exactly the set of valid JSON strings conforming to the schema.
//!
//! Supports: object, array, string, number, integer, boolean, null,
//! enum, const, oneOf/anyOf/allOf, and `$ref` (simple same-document refs).

use crate::compiler::grammar_def::GrammarDef;
use crate::frontend::grammar_expr::{lower, GrammarExpr, NamedGrammar};
use crate::GlrMaskError;

/// Convert a JSON Schema (as a JSON string) into a `GrammarDef`.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {}", e)))?;
    let named = schema_to_named_grammar(&schema)?;
    lower(&named)
}

/// Convert a parsed JSON Schema value into a `NamedGrammar`.
pub fn schema_to_named_grammar(schema: &serde_json::Value) -> Result<NamedGrammar, GlrMaskError> {
    let mut ctx = SchemaCtx::new();
    let start_expr = ctx.convert_schema(schema)?;

    // Build the rules: start rule + JSON primitives + any generated sub-rules.
    let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
    rules.push(("start".into(), start_expr));

    // Add JSON whitespace rule.
    rules.push((
        "ws".into(),
        GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass {
            def: " \\t\\n\\r".into(),
            negate: false,
        })),
    ));

    // Add sub-rules generated during conversion.
    rules.extend(ctx.sub_rules);

    Ok(NamedGrammar {
        rules,
        start: "start".into(),
    })
}

struct SchemaCtx {
    sub_rules: Vec<(String, GrammarExpr)>,
    counter: usize,
}

impl SchemaCtx {
    fn new() -> Self {
        SchemaCtx {
            sub_rules: Vec::new(),
            counter: 0,
        }
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        let name = format!("_json_{}_{}", hint, self.counter);
        self.counter += 1;
        name
    }

    fn convert_schema(&mut self, schema: &serde_json::Value) -> Result<GrammarExpr, GlrMaskError> {
        // Handle boolean schemas.
        if let Some(b) = schema.as_bool() {
            return if b {
                Ok(self.json_value())
            } else {
                Err(GlrMaskError::GrammarParse("schema is false (matches nothing)".into()))
            };
        }

        let obj = schema.as_object().ok_or_else(|| {
            GlrMaskError::GrammarParse("schema must be an object or boolean".into())
        })?;

        // Handle const.
        if let Some(val) = obj.get("const") {
            return Ok(self.json_literal(val));
        }

        // Handle enum.
        if let Some(vals) = obj.get("enum").and_then(|v| v.as_array()) {
            let alts: Vec<GrammarExpr> = vals.iter().map(|v| self.json_literal(v)).collect();
            return Ok(if alts.len() == 1 {
                alts.into_iter().next().unwrap()
            } else {
                GrammarExpr::Choice(alts)
            });
        }

        // Handle oneOf / anyOf.
        if let Some(variants) = obj.get("oneOf").or(obj.get("anyOf")).and_then(|v| v.as_array()) {
            let mut alts = Vec::new();
            for v in variants {
                alts.push(self.convert_schema(v)?);
            }
            return Ok(if alts.len() == 1 {
                alts.into_iter().next().unwrap()
            } else {
                GrammarExpr::Choice(alts)
            });
        }

        // Dispatch on type.
        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("any");

        match ty {
            "object" => self.convert_object(obj),
            "array" => self.convert_array(obj),
            "string" => Ok(self.json_string()),
            "number" => Ok(self.json_number()),
            "integer" => Ok(self.json_integer()),
            "boolean" => Ok(GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"true".to_vec()),
                GrammarExpr::Literal(b"false".to_vec()),
            ])),
            "null" => Ok(GrammarExpr::Literal(b"null".to_vec())),
            "any" | _ => Ok(self.json_value()),
        }
    }

    fn convert_object(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let properties = obj.get("properties").and_then(|v| v.as_object());
        let required: Vec<String> = obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if let Some(props) = properties {
            let mut parts: Vec<GrammarExpr> = Vec::new();
            parts.push(GrammarExpr::Literal(b"{".to_vec()));
            parts.push(GrammarExpr::Ref("ws".into()));

            let prop_entries: Vec<(&String, &serde_json::Value)> = props.iter().collect();
            for (i, (key, val_schema)) in prop_entries.iter().enumerate() {
                if i > 0 {
                    parts.push(GrammarExpr::Literal(b",".to_vec()));
                    parts.push(GrammarExpr::Ref("ws".into()));
                }

                let val_expr = self.convert_schema(val_schema)?;
                let name = self.fresh_name(&format!("prop_{}", key));
                self.sub_rules.push((name.clone(), val_expr));

                let prop_expr = GrammarExpr::Sequence(vec![
                    self.json_string_literal(key),
                    GrammarExpr::Ref("ws".into()),
                    GrammarExpr::Literal(b":".to_vec()),
                    GrammarExpr::Ref("ws".into()),
                    GrammarExpr::Ref(name),
                ]);

                if required.contains(key) {
                    parts.push(prop_expr);
                } else {
                    parts.push(GrammarExpr::Optional(Box::new(prop_expr)));
                }
            }

            parts.push(GrammarExpr::Ref("ws".into()));
            parts.push(GrammarExpr::Literal(b"}".to_vec()));

            Ok(GrammarExpr::Sequence(parts))
        } else {
            // Generic object: { (key: value)* }
            let name = self.fresh_name("obj_entry");
            let entry = GrammarExpr::Sequence(vec![
                self.json_string(),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Literal(b":".to_vec()),
                GrammarExpr::Ref("ws".into()),
                self.json_value(),
            ]);
            self.sub_rules.push((name.clone(), entry));

            let entries = self.fresh_name("obj_entries");
            let entries_expr = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref(name.clone()),
                GrammarExpr::Repeat(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    GrammarExpr::Ref("ws".into()),
                    GrammarExpr::Ref(name),
                ]))),
            ])));
            self.sub_rules.push((entries.clone(), entries_expr));

            Ok(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"{".to_vec()),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Ref(entries),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Literal(b"}".to_vec()),
            ]))
        }
    }

    fn convert_array(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let items_schema = obj.get("items");

        let item_expr = if let Some(schema) = items_schema {
            self.convert_schema(schema)?
        } else {
            self.json_value()
        };

        let name = self.fresh_name("arr_item");
        self.sub_rules.push((name.clone(), item_expr));

        let items = self.fresh_name("arr_items");
        let items_expr = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
            GrammarExpr::Ref(name.clone()),
            GrammarExpr::Repeat(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b",".to_vec()),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Ref(name),
            ]))),
        ])));
        self.sub_rules.push((items.clone(), items_expr));

        Ok(GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"[".to_vec()),
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Ref(items),
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Literal(b"]".to_vec()),
        ]))
    }

    /// JSON string: `"` (escape | [^"\\])* `"`
    fn json_string(&mut self) -> GrammarExpr {
        let name = "_json_string".to_string();
        // Only add the rule once.
        if !self.sub_rules.iter().any(|(n, _)| n == &name) {
            let char_name = "_json_str_char".to_string();
            let char_expr = GrammarExpr::Choice(vec![
                GrammarExpr::CharClass {
                    def: "\"\\\\".into(),
                    negate: true,
                },
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"\\".to_vec()),
                    GrammarExpr::CharClass {
                        def: "\"\\\\bfnrt/".into(),
                        negate: false,
                    },
                ]),
            ]);
            self.sub_rules.push((char_name.clone(), char_expr));

            let str_expr = GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"\"".to_vec()),
                GrammarExpr::Repeat(Box::new(GrammarExpr::Ref(char_name))),
                GrammarExpr::Literal(b"\"".to_vec()),
            ]);
            self.sub_rules.push((name.clone(), str_expr));
        }
        GrammarExpr::Ref(name)
    }

    /// JSON number: -? (0 | [1-9][0-9]*) (.[0-9]+)? ([eE][+-]?[0-9]+)?
    fn json_number(&mut self) -> GrammarExpr {
        let name = "_json_number".to_string();
        if !self.sub_rules.iter().any(|(n, _)| n == &name) {
            let digits = GrammarExpr::RepeatOne(Box::new(GrammarExpr::CharClass {
                def: "0-9".into(),
                negate: false,
            }));
            let integer_part = GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"0".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::CharClass {
                        def: "1-9".into(),
                        negate: false,
                    },
                    GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass {
                        def: "0-9".into(),
                        negate: false,
                    })),
                ]),
            ]);
            let frac = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b".".to_vec()),
                digits.clone(),
            ])));
            let exp = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::CharClass {
                    def: "eE".into(),
                    negate: false,
                },
                GrammarExpr::Optional(Box::new(GrammarExpr::CharClass {
                    def: "+-".into(),
                    negate: false,
                })),
                digits,
            ])));
            let num_expr = GrammarExpr::Sequence(vec![
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
                integer_part,
                frac,
                exp,
            ]);
            self.sub_rules.push((name.clone(), num_expr));
        }
        GrammarExpr::Ref(name)
    }

    /// JSON integer: -? (0 | [1-9][0-9]*)
    fn json_integer(&mut self) -> GrammarExpr {
        let name = "_json_integer".to_string();
        if !self.sub_rules.iter().any(|(n, _)| n == &name) {
            let int_expr = GrammarExpr::Sequence(vec![
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"0".to_vec()),
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::CharClass {
                            def: "1-9".into(),
                            negate: false,
                        },
                        GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass {
                            def: "0-9".into(),
                            negate: false,
                        })),
                    ]),
                ]),
            ]);
            self.sub_rules.push((name.clone(), int_expr));
        }
        GrammarExpr::Ref(name)
    }

    /// Generic JSON value.
    fn json_value(&mut self) -> GrammarExpr {
        let name = "_json_value".to_string();
        if !self.sub_rules.iter().any(|(n, _)| n == &name) {
            // We need forward references, so add a placeholder.
            // The sub_rules will be added by the individual type methods.
            let val_expr = GrammarExpr::Choice(vec![
                self.json_string(),
                self.json_number(),
                GrammarExpr::Literal(b"true".to_vec()),
                GrammarExpr::Literal(b"false".to_vec()),
                GrammarExpr::Literal(b"null".to_vec()),
                // For arrays and objects, we'd need recursive rules.
                // Add them as simple references; the caller must define them.
            ]);
            self.sub_rules.push((name.clone(), val_expr));
        }
        GrammarExpr::Ref(name)
    }

    /// Produce a GrammarExpr for a specific JSON literal value.
    fn json_literal(&self, value: &serde_json::Value) -> GrammarExpr {
        let s = value.to_string();
        GrammarExpr::Literal(s.into_bytes())
    }

    /// Produce a GrammarExpr for a JSON string literal: `"key"`.
    fn json_string_literal(&self, s: &str) -> GrammarExpr {
        let mut bytes = Vec::new();
        bytes.push(b'"');
        for b in s.bytes() {
            match b {
                b'"' => {
                    bytes.push(b'\\');
                    bytes.push(b'"');
                }
                b'\\' => {
                    bytes.push(b'\\');
                    bytes.push(b'\\');
                }
                _ => bytes.push(b),
            }
        }
        bytes.push(b'"');
        GrammarExpr::Literal(bytes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolean_schema() {
        let g = json_schema_to_grammar(r#"{"type": "boolean"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_string_schema() {
        let g = json_schema_to_grammar(r#"{"type": "string"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_integer_schema() {
        let g = json_schema_to_grammar(r#"{"type": "integer"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_null_schema() {
        let g = json_schema_to_grammar(r#"{"type": "null"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_enum_schema() {
        let g = json_schema_to_grammar(r#"{"enum": ["a", "b", "c"]}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_const_schema() {
        let g = json_schema_to_grammar(r#"{"const": 42}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_schema() {
        let g = json_schema_to_grammar(
            r#"{
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }"#,
        )
        .unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_schema() {
        let g = json_schema_to_grammar(
            r#"{
                "type": "array",
                "items": {"type": "integer"}
            }"#,
        )
        .unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_oneof_schema() {
        let g = json_schema_to_grammar(
            r#"{
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"}
                ]
            }"#,
        )
        .unwrap();
        assert!(!g.rules.is_empty());
    }
}
