//! Tests for JSON Schema to grammar conversion and mask generation.
//!
//! These tests specifically target edge cases and bugs discovered in the
//! JSON Schema conversion and constraint system.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use indoc::indoc;
    
    use crate::constraint::{GrammarConstraint, GrammarConstraintConfig};
    use crate::interface::GrammarDefinition;
    use crate::json_schema::json_schema_to_ebnf;
    use crate::tokenizer::{LLMTokenID, LLMTokenMap};
    use crate::datastructures::bitset::Bitset;

    /// Create a simple token map with all single-byte tokens (256 total).
    fn simple_byte_token_map() -> LLMTokenMap {
        let mut map = LLMTokenMap::new();
        for b in 0u8..=255 {
            map.insert(vec![b], LLMTokenID(b as usize));
        }
        map
    }
    
    /// Create a GPT-2-like token map with multi-byte tokens.
    /// This simulates the real GPT-2 tokenizer which has tokens like `{"`, ` "`, etc.
    fn gpt2_like_token_map() -> (LLMTokenMap, usize) {
        let mut map = LLMTokenMap::new();
        let mut next_id = 0;
        
        // Single byte tokens for all ASCII
        for b in 0u8..=127 {
            map.insert(vec![b], LLMTokenID(next_id));
            next_id += 1;
        }
        
        // Multi-byte tokens that GPT-2 commonly uses
        // NOTE: These should NOT include single-byte tokens as that would overwrite the ID mapping
        let multi_tokens = [
            // Space + character combinations
            b" \"".to_vec(),   // common in JSON
            b" '".to_vec(),
            b"  ".to_vec(),
            b"   ".to_vec(),
            b"\n\n".to_vec(),
            b"\t\t".to_vec(),
            // Quote combinations
            b"\"\"".to_vec(),
            b"\":".to_vec(),
            b"\",".to_vec(),
            b"\"n".to_vec(),
            b"\"na".to_vec(),
            b"\"name".to_vec(),
            // Brace combinations (this is the key one!)
            b"{\"".to_vec(),
            b"{}".to_vec(),
            b"[]".to_vec(),
            // Common words
            b"name".to_vec(),
            b"true".to_vec(),
            b"false".to_vec(),
            b"null".to_vec(),
            // Other common patterns
            b": ".to_vec(),
            b", ".to_vec(),
            b"\\n".to_vec(),
            b"\\t".to_vec(),
            b"\\\"".to_vec(),
        ];
        
        for bytes in multi_tokens {
            map.insert(bytes, LLMTokenID(next_id));
            next_id += 1;
        }
        
        (map, next_id)
    }
    
    /// Test a JSON schema by generating tokens for a valid JSON string and verifying
    /// that each prefix allows the next character.
    fn test_schema_with_inputs(schema_json: &str, valid_inputs: &[&str]) {
        test_schema_with_inputs_and_vocab(schema_json, valid_inputs, simple_byte_token_map(), 255);
    }
    
    /// Test a JSON schema with a custom vocabulary.
    fn test_schema_with_inputs_and_vocab(
        schema_json: &str, 
        valid_inputs: &[&str], 
        token_map: LLMTokenMap,
        max_token_id: usize
    ) {
        let ebnf = json_schema_to_ebnf(schema_json)
            .expect(&format!("Schema should convert: {}", schema_json));
        println!("Generated EBNF:\n{}", ebnf);
        
        let gd = GrammarDefinition::from_ebnf(&ebnf)
            .expect("Grammar should build");
        
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd),
            token_map,
            max_token_id,
            &GrammarConstraintConfig::default(),
        );
        
        for input in valid_inputs {
            println!("\nTesting input: {:?}", input);
            let mut state = constraint.init();
            
            for (i, ch) in input.bytes().enumerate() {
                let prefix = &input[..i];
                let mask = state.get_mask();
                let is_valid = mask.contains(ch as usize);
                
                assert!(is_valid,
                    "After {:?}, character {:?} (byte {}) should be valid but wasn't.\n\
                     Valid bytes: {:?}",
                    prefix,
                    ch as char,
                    ch,
                    mask.iter().take(30).collect::<Vec<_>>()
                );
                
                state.commit(LLMTokenID(ch as usize))
                    .expect(&format!("Failed to commit byte {} at position {}", ch, i));
            }
            println!("✓ Input {:?} accepted", input);
        }
    }
    
    /// Test using multi-byte tokens like GPT-2. This tests that tokens like `{"` work correctly.
    fn test_schema_with_multibyte_tokens(schema_json: &str, valid_inputs: &[&str]) {
        let (token_map, max_token_id) = gpt2_like_token_map();
        
        let ebnf = json_schema_to_ebnf(schema_json)
            .expect(&format!("Schema should convert: {}", schema_json));
        println!("Generated EBNF:\n{}", ebnf);
        
        let gd = GrammarDefinition::from_ebnf(&ebnf)
            .expect("Grammar should build");
        
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd),
            token_map.clone(),
            max_token_id,
            &GrammarConstraintConfig::default(),
        );
        
        // Find key token IDs
        let open_brace_id = token_map.get(&vec![b'{']).expect("should have {");
        let quote_id = token_map.get(&vec![b'"']).expect("should have quote");
        let open_brace_quote_id = token_map.get(&vec![b'{', b'"']).expect("should have {\"");
        
        println!("Token IDs: {{ = {:?}, \" = {:?}, {{\" = {:?}", 
            open_brace_id, quote_id, open_brace_quote_id);
        
        // Test initial state
        let state = constraint.init();
        let mask = state.get_mask();
        
        // Both `{` and `{"` should be valid at start
        let open_brace_valid = mask.contains(open_brace_id.0);
        let open_brace_quote_valid = mask.contains(open_brace_quote_id.0);
        
        println!("At start: {{ valid = {}, {{\" valid = {}", open_brace_valid, open_brace_quote_valid);
        println!("Valid tokens: {:?}", mask.iter().collect::<Vec<_>>());
        
        assert!(open_brace_valid, "'{{' should be valid at start");
        assert!(open_brace_quote_valid, "'{{\"' should be valid at start for object with properties");
        
        // Also test inputs byte-by-byte
        for input in valid_inputs {
            println!("\nTesting input byte-by-byte: {:?}", input);
            let mut state = constraint.init();
            
            for (i, ch) in input.bytes().enumerate() {
                let prefix = &input[..i];
                let mask = state.get_mask();
                // Note: we're only testing single-byte tokens here (first 128 IDs)
                let is_valid = mask.contains(ch as usize);
                
                assert!(is_valid,
                    "After {:?}, character {:?} (byte {}) should be valid but wasn't.\n\
                     Valid token IDs: {:?}",
                    prefix,
                    ch as char,
                    ch,
                    mask.iter().take(30).collect::<Vec<_>>()
                );
                
                state.commit(LLMTokenID(ch as usize))
                    .expect(&format!("Failed to commit byte {} at position {}", ch, i));
            }
            println!("✓ Input {:?} accepted", input);
        }
    }
    
    /// Test that WS (whitespace) is nullable - empty object should work.
    #[test]
    fn test_ebnf_ws_nullable() {
        let ebnf = indoc! {r#"
            root ::= '{' WS '}' ;
            WS ::= ( ' ' | '\t' | '\n' | '\r' )* ;
        "#};

        let gd = GrammarDefinition::from_ebnf(ebnf).expect("EBNF should parse");
        
        let token_map = simple_byte_token_map();
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd),
            token_map,
            255,
            &GrammarConstraintConfig::default(),
        );
        
        let mut state = constraint.init();
        
        // Commit '{'
        state.commit(LLMTokenID(b'{' as usize)).expect("should commit");
        
        // '}' should be immediately valid (WS is nullable)
        let mask = state.get_mask();
        let close_brace_valid = mask.contains(b'}' as usize);
        
        assert!(close_brace_valid, 
            "'}}' should be valid immediately after '{{' when WS is nullable. Valid: {:?}",
            mask.iter().take(20).collect::<Vec<_>>());
    }

    /// Test that object member after open brace allows both whitespace and quote.
    #[test]
    fn test_ebnf_object_member_after_brace() {
        let ebnf = indoc! {r#"
            root ::= '{' WS member_opt WS '}' ;
            member_opt ::= ( member ( ',' WS member )* )? ;
            member ::= '"name"' WS ':' WS 'value' ;
            WS ::= ( ' ' | '\t' | '\n' | '\r' )* ;
        "#};

        let gd = GrammarDefinition::from_ebnf(ebnf).expect("EBNF should parse");
        
        let token_map = simple_byte_token_map();
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd),
            token_map,
            255,
            &GrammarConstraintConfig::default(),
        );
        
        let mut state = constraint.init();
        state.commit(LLMTokenID(b'{' as usize)).expect("should commit");
        
        let mask = state.get_mask();
        
        assert!(mask.contains(b'"' as usize), 
            "'\"' should be valid after '{{' for starting a member");
        
        assert!(mask.contains(b'}' as usize),
            "'}}' should be valid after '{{' for empty object");
    }

    /// Test simple object schema without additionalProperties
    #[test]
    fn test_schema_simple_object() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        
        test_schema_with_inputs(schema, &[
            "{}",
            r#"{"name": "test"}"#,
            r#"{ "name" : "hello world" }"#,
        ]);
    }

    /// Test object with additionalProperties: true
    #[test]
    fn test_schema_additional_properties_true() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": true
        }"#;
        
        test_schema_with_inputs(schema, &[
            "{}",
            r#"{"name": "test"}"#,
            r#"{"foo": "bar"}"#,
            r#"{"name": "test", "extra": 123}"#,
            r#"{"x": null, "y": true, "z": [1, 2, 3]}"#,
        ]);
    }
    
    /// Test object with additionalProperties schema
    #[test]
    fn test_schema_additional_properties_schema() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": {"type": "string"}
        }"#;
        
        test_schema_with_inputs(schema, &[
            "{}",
            r#"{"name": "test"}"#,
            r#"{"foo": "bar"}"#,
            r#"{"name": "test", "extra": "value"}"#,
        ]);
    }

    /// Test dependencies schema (like package.json subset)
    #[test]
    fn test_schema_dependencies() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "dependencies": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["name", "version"]
        }"#;
        
        test_schema_with_inputs(schema, &[
            r#"{"name": "pkg", "version": "1.0.0"}"#,
            r#"{"name": "pkg", "version": "1.0.0", "dependencies": {}}"#,
            r#"{"name": "pkg", "version": "1.0.0", "dependencies": {"lodash": "^4.0.0"}}"#,
        ]);
    }
    
    /// Test nested objects
    #[test]
    fn test_schema_nested_objects() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"}
                    }
                }
            }
        }"#;
        
        test_schema_with_inputs(schema, &[
            "{}",
            r#"{"outer": {}}"#,
            r#"{"outer": {"inner": "value"}}"#,
        ]);
    }
    
    /// Test array schema
    #[test]
    fn test_schema_array() {
        let schema = r#"{
            "type": "array",
            "items": {"type": "string"}
        }"#;
        
        test_schema_with_inputs(schema, &[
            "[]",
            r#"["a"]"#,
            r#"["a", "b", "c"]"#,
        ]);
    }
    
    /// Test mixed types (anyOf)
    #[test]
    fn test_schema_anyof() {
        let schema = r#"{
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"}
            ]
        }"#;
        
        test_schema_with_inputs(schema, &[
            r#""hello""#,
            "42",
            "3.14",
            "true",
            "false",
        ]);
    }
    
    /// Test enum
    #[test]
    fn test_schema_enum() {
        let schema = r#"{
            "enum": ["red", "green", "blue"]
        }"#;
        
        test_schema_with_inputs(schema, &[
            r#""red""#,
            r#""green""#,
            r#""blue""#,
        ]);
    }
    
    /// Test const
    #[test]
    fn test_schema_const() {
        let schema = r#"{
            "const": "fixed_value"
        }"#;
        
        test_schema_with_inputs(schema, &[
            r#""fixed_value""#,
        ]);
    }
    
    /// Test that multi-byte tokens like `{"` work with simple object schema.
    #[test]
    fn test_multibyte_tokens_simple_object() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        
        test_schema_with_multibyte_tokens(schema, &[
            r#"{"name": "test"}"#,
        ]);
    }
    
    /// Test that multi-byte tokens work with additionalProperties: true.
    #[test]
    fn test_multibyte_tokens_additional_properties_true() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": true
        }"#;
        
        test_schema_with_multibyte_tokens(schema, &[
            r#"{"name": "test"}"#,
        ]);
    }
}
