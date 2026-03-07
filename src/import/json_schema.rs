












#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::HashMap;

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, lower};


pub fn json_schema_to_grammar(schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    unimplemented!()
}


pub fn schema_to_named_grammar(schema: &serde_json::Value) -> Result<NamedGrammar, GlrMaskError> {
    unimplemented!()
}





struct SchemaCtx {
    sub_rules: Vec<(String, GrammarExpr)>,
    counter: usize,
    
    defs: HashMap<String, serde_json::Value>,
}

impl SchemaCtx {
    fn new(root: &serde_json::Value) -> Self {
        unimplemented!()
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        unimplemented!()
    }

    
    
    

    fn convert_schema(&mut self, schema: &serde_json::Value) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    
    

    fn resolve_ref(&mut self, ref_str: &str) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    
    

    fn convert_all_of(
        &mut self,
        all: &[serde_json::Value],
        parent: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    
    

    fn convert_object(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    
    
    
    
    
    
    
    
    
    
    fn build_object_rule(
        &mut self,
        properties: &[(String, serde_json::Value)],
        required: &[String],
        additional: Option<&serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    
    

    fn convert_array(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    
    

    
    fn json_value(&mut self) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_array_generic(&mut self) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_object_generic(&mut self) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_string(&mut self) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_string_bounded(&mut self, min: usize, max: Option<usize>) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_string_pattern(&self, pattern: &str) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_number(&mut self) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_integer(&mut self) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_literal(&self, value: &serde_json::Value) -> GrammarExpr {
        unimplemented!()
    }

    
    fn json_string_literal(&self, s: &str) -> GrammarExpr {
        unimplemented!()
    }
}






fn choice_or_single(alts: Vec<GrammarExpr>) -> GrammarExpr {
    unimplemented!()
}


fn sanitize_rule_name(s: &str) -> String {
    unimplemented!()
}











fn build_optional_choice(optional_keys: &[String], kv_rules: &[(String, String)]) -> GrammarExpr {
    unimplemented!()
}






fn build_repetition(item_rule: &str, min: usize, max: Option<usize>) -> GrammarExpr {
    unimplemented!()
}





#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;

    
    
    

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
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_additional_properties_false() {
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": false
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_only_required_comma_free() {
        
        
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_all_optional_no_required() {
        
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "integer"}
            }
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_empty_additional_false() {
        
        let g = json_schema_to_grammar(r#"{"type": "object", "additionalProperties": false}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_schema() {
        let g = json_schema_to_grammar(r#"{"type": "array", "items": {"type": "integer"}}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_min_max_items() {
        let g = json_schema_to_grammar(r#"{"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_prefix_items() {
        let g = json_schema_to_grammar(r#"{
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_oneof_schema() {
        let g = json_schema_to_grammar(r#"{
            "oneOf": [{"type": "string"}, {"type": "integer"}]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_allof_schema() {
        let g = json_schema_to_grammar(r#"{
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"properties": {"b": {"type": "integer"}}}
            ]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_ref_schema() {
        let g = json_schema_to_grammar(r##"{
            "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}},
            "$ref": "#/$defs/Point"
        }"##).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_string_min_length() {
        let g = json_schema_to_grammar(r#"{"type": "string", "minLength": 3}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_string_min_max_length() {
        let g = json_schema_to_grammar(r#"{"type": "string", "minLength": 1, "maxLength": 5}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_type_array_of_types() {
        let g = json_schema_to_grammar(r#"{"type": ["string", "null"]}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    
    
    

    
    
    fn accepts_sequence(schema_json: &str, tokens: &[&[u8]]) -> bool {
        let entries: Vec<(u32, Vec<u8>)> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u32, t.to_vec()))
            .collect();
        let vocab = Vocab::new(entries, None);

        let c = match crate::Constraint::from_json_schema(schema_json, &vocab) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let mut state = c.start();
        for (i, _tok) in tokens.iter().enumerate() {
            let id = i as u32;
            let mask = state.mask_view().mask();
            let (wi, bi) = (id as usize / 32, id as usize % 32);
            let allowed = wi < mask.len() && (mask[wi] >> bi) & 1 != 0;
            if !allowed {
                return false;
            }
            state.commit_token(id);
        }
        state.is_finished()
    }

    #[test]
    fn test_accepts_boolean_true() {
        assert!(accepts_sequence(r#"{"type": "boolean"}"#, &[b"true"]));
    }

    #[test]
    fn test_accepts_boolean_false() {
        assert!(accepts_sequence(r#"{"type": "boolean"}"#, &[b"false"]));
    }

    #[test]
    fn test_accepts_null_value() {
        assert!(accepts_sequence(r#"{"type": "null"}"#, &[b"null"]));
    }

    #[test]
    fn test_accepts_enum_value() {
        assert!(accepts_sequence(r#"{"enum": ["yes", "no"]}"#, &[b"\"yes\""]));
    }

    #[test]
    fn test_accepts_const_value() {
        assert!(accepts_sequence(r#"{"const": true}"#, &[b"true"]));
    }

    #[test]
    fn test_object_required_only_accepts_valid() {
        let schema = r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#;
        let g = json_schema_to_grammar(schema).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_optional_no_trailing_comma() {
        let schema = r#"{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x"]}"#;
        let g = json_schema_to_grammar(schema).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_json_value_is_recursive() {
        let schema = r#"{"type":"array"}"#;
        let g = json_schema_to_grammar(schema).unwrap();
        assert!(!g.rules.is_empty());
    }
}
