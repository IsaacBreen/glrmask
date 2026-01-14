//! Tests for JSON Schema to grammar conversion and mask generation.
//!
//! These tests specifically target edge cases and bugs discovered in the
//! JSON Schema conversion and constraint system.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use indoc::indoc;
    
    use crate::constraint::{GrammarConstraint, GrammarConstraintConfig};
    use crate::interface::GrammarDefinition;
    use crate::interface::json_schema::json_schema_to_ebnf;
    use crate::dfa_u8::{LLMTokenID, LLMTokenMap};
    use crate::datastructures::bitset::Bitset;

    /// Create a simple token map with all single-byte tokens (256 total).
    fn simple_byte_token_map() -> LLMTokenMap {
        let mut map = LLMTokenMap::new();
        for b in 0u8..=255 {
            map.insert(vec![b], LLMTokenID(b as usize));
        }
        map
    }

    fn format_byte(b: u8) -> String {
        match b {
            b if b.is_ascii_graphic() => format!("'{}'", b as char),
            b' ' => "' '".to_string(),
            b'\n' => r"'\n'".to_string(),
            b'\r' => r"'\r'".to_string(),
            b'\t' => r"'\t'".to_string(),
            _ => format!("\\x{:02x}", b),
        }
    }

    fn format_bytes(bytes: &[usize]) -> String {
        if bytes.is_empty() {
            return "[]".to_string();
        }
        let mut s = String::from("[");
        for (i, &b) in bytes.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format_byte(b as u8));
        }
        s.push(']');
        s
    }
    
    /// Create a small token map with JSON punctuation and some multi-byte tokens.
    fn small_json_token_map() -> LLMTokenMap {
        let mut map = LLMTokenMap::new();
        map.insert(vec![b'{'], LLMTokenID(0));
        map.insert(vec![b'}'], LLMTokenID(1));
        map.insert(vec![b'"'], LLMTokenID(2));
        map.insert(vec![b':'], LLMTokenID(3));
        map.insert(vec![b','], LLMTokenID(4));
        map.insert(vec![b'n'], LLMTokenID(5));
        map.insert(vec![b'a'], LLMTokenID(6));
        map.insert(vec![b'm'], LLMTokenID(7));
        map.insert(vec![b'e'], LLMTokenID(8));
        map.insert(vec![b's'], LLMTokenID(9));
        map.insert(vec![b't'], LLMTokenID(10));
        map.insert(vec![b'r'], LLMTokenID(11));
        map.insert(vec![b'i'], LLMTokenID(12));
        map.insert(vec![b'g'], LLMTokenID(13));
        map.insert(vec![b'{', b'"'], LLMTokenID(14));  // Multi-byte
        map.insert(vec![b'"', b':'], LLMTokenID(15));  // Multi-byte
        map
    }
    
    /// Create a GPT-2 token map by loading the actual GPT-2 vocabulary.
    /// Falls back to a simulated vocab if the file doesn't exist.
    fn gpt2_like_token_map() -> Option<(LLMTokenMap, usize)> {
        // Try to load the actual GPT-2 vocab
        // We look for vocab files in several locations
        let paths = [
            "vocab.json",
            "gpt2_vocab.json",
            "benchmarking/gpt2_vocab.json",
            "python/.cache/py_benchmark_vocabs/gpt2_vocab.json",
        ];
        
        for path in paths {
            println!("Trying to load GPT-2 vocab from {}", path);
            let vocab_path = std::path::Path::new(path);
            if vocab_path.exists() {
                match load_gpt2_vocab_from_file(vocab_path) {
                    Ok((map, max_id)) => {
                        // Verify this is a real GPT-2 vocab (should have thousands of tokens)
                        // and contains basic tokens like `{`, `}`, `"`, etc.
                        if map.len() < 1000 {
                            eprintln!("Warning: {} has only {} tokens, not a real GPT-2 vocab", path, map.len());
                            continue;
                        }
                        if !map.contains_key(&vec![b'{']) {
                            eprintln!("Warning: {} missing '{{' token, not a real GPT-2 vocab", path);
                            continue;
                        }
                        return Some((map, max_id));
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load {}: {}", path, e);
                        continue;
                    }
                }
            }
        }
        
        // No valid GPT-2 vocab found
        None
    }
    
    /// Build the byte decoder map once.
    fn build_gpt2_byte_decoder() -> HashMap<char, u8> {
        let mut byte_decoder: std::collections::HashMap<char, u8> = std::collections::HashMap::new();
        
        // Printable ASCII chars (except space) map to themselves
        for b in b'!'..=b'~' {
            byte_decoder.insert(b as char, b);
        }
        // Extended printable chars
        for b in 0xa1u8..=0xac {
            byte_decoder.insert(b as char, b);
        }
        for b in 0xaeu8..=0xff {
            byte_decoder.insert(b as char, b);
        }
        
        // Non-printable bytes map to Unicode chars starting at U+0100
        let mut n: u32 = 0;
        for b in 0u8..=255 {
            if !byte_decoder.values().any(|&v| v == b) {
                byte_decoder.insert(char::from_u32(256 + n).unwrap(), b);
                n += 1;
            }
        }
        byte_decoder
    }

    /// Decode a GPT-2 BPE token string to its actual byte representation.
    /// GPT-2 uses a byte-level BPE where bytes 0-255 are mapped to specific Unicode characters.
    fn gpt2_bpe_decode(token_str: &str, byte_decoder: &HashMap<char, u8>) -> Vec<u8> {
        // Decode the token string
        token_str.chars().map(|c| {
            *byte_decoder.get(&c).unwrap_or(&(c as u8))
        }).collect()
    }
    
    /// Load the actual GPT-2 vocab from vocab.json
    fn load_gpt2_vocab_from_file(path: &std::path::Path) -> Result<(LLMTokenMap, usize), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read vocab file: {}", e))?;
        
        let vocab: HashMap<String, usize> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse vocab JSON: {}", e))?;
        
        let mut map = LLMTokenMap::new();
        let mut max_id = 0;
        
        // Build the decoder map once
        let byte_decoder = build_gpt2_byte_decoder();
        
        for (token_str, token_id) in vocab {
            // Decode the BPE-encoded token string to actual bytes
            let bytes = gpt2_bpe_decode(&token_str, &byte_decoder);
            map.insert(bytes, LLMTokenID(token_id));
            if token_id > max_id {
                max_id = token_id;
            }
        }
        
        Ok((map, max_id))
    }
    
    /// Create a simulated GPT-2-like token map (fallback when vocab.json not available)
    fn create_simulated_gpt2_vocab() -> (LLMTokenMap, usize) {
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
                     Valid bytes: {}",
                    prefix,
                    ch as char,
                    ch,
                    format_bytes(&mask.iter().take(30).collect::<Vec<_>>())
                );
                
                state.commit(LLMTokenID(ch as usize))
                    .expect(&format!("Failed to commit byte {} at position {}", ch, i));
            }
            println!("✓ Input {:?} accepted", input);
        }
    }
    
    /// Test using multi-byte tokens like GPT-2. This tests that tokens like `{"` work correctly.
    /// Requires a valid GPT-2 vocab to be available - will panic if not found.
    fn test_schema_with_multibyte_tokens(schema_json: &str, valid_inputs: &[&str]) {
        println!("Loading GPT-2-like vocab for multi-byte token test.");
        let (token_map, max_token_id) = gpt2_like_token_map()
            .expect("No valid GPT-2 vocab found! This test requires a real GPT-2 vocab with thousands of tokens. \
                     Try: wget -O benchmarking/gpt2_vocab.json https://huggingface.co/openai-community/gpt2/raw/main/vocab.json");
        println!("Using GPT-2-like vocab with {} tokens, max ID {}", token_map.len(), max_token_id);

        println!("Generating EBNF for schema.");
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
                
                // Look up the token ID for this single byte
                let token_id = token_map.get(&vec![ch])
                    .expect(&format!("Token map should have single byte {}", ch));
                let is_valid = mask.contains(token_id.0);
                
                assert!(is_valid,
                    "After {:?}, character {:?} (byte {}, token_id {}) should be valid but wasn't.\n\
                     Valid token IDs: {}",
                    prefix,
                    ch as char,
                    ch,
                    token_id.0,
                    format_bytes(&mask.iter().take(30).collect::<Vec<_>>())
                );
                
                state.commit(*token_id)
                    .expect(&format!("Failed to commit byte {} (token_id {}) at position {}", ch, token_id.0, i));
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

    /// Bug reproduction: With small vocab, only `{` and `{"` should be valid at empty prefix.
    /// 
    /// This test uses a minimal token map to clearly show the bug.
    #[test]
    fn test_small_vocab_only_brace_valid_at_start() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).expect("Schema should convert");
        println!("Generated EBNF:\n{}", ebnf);
        
        let gd = GrammarDefinition::from_ebnf(&ebnf).expect("Grammar should build");
        println!("GrammarDefinition: {}", gd);
        
        // Debug: Print how many groups the grammar has
        println!("\n=== Grammar terminal info ===");
        println!("regex_name_to_group_id: {:?}", gd.regex_name_to_group_id);
        println!("literal_to_group_id: {:?}", gd.literal_to_group_id);
        println!("Total groups: {}", gd.regex_name_to_group_id.len() + gd.literal_to_group_id.len());
        
        let token_map = small_json_token_map();
        let max_token_id = 15;  // Maximum token ID in small_json_token_map
        
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd),
            token_map.clone(),
            max_token_id,
            &GrammarConstraintConfig::default(),
        );
        
        // Debug: Print the internal-to-original mapping
        println!("\n=== DEBUG: Vocabulary mapping ===");
        let vocab = &constraint.parser_dwa_vocab;
        println!("Internal max token: {}", vocab.internal_max_llm_token);
        println!("Original max token: {}", vocab.max_original_llm_token_id);
        println!("internal_to_original:");
        for (int_id, originals) in &vocab.internal_to_original {
            let orig_ids: Vec<usize> = originals.iter_up_to(max_token_id).collect();
            print!("  internal {} -> originals {:?} = ", int_id, orig_ids);
            for &orig in &orig_ids {
                let token_str = match orig {
                    0 => "'{'" ,
                    1 => "'}'",
                    2 => "'\"'",
                    3 => "':'",
                    4 => "','",
                    5 => "'n'",
                    6 => "'a'",
                    7 => "'m'",
                    8 => "'e'",
                    9 => "'s'",
                    10 => "'t'",
                    11 => "'r'",
                    12 => "'i'",
                    13 => "'g'",
                    14 => "'{\"'",
                    15 => "'\":'" ,
                    _ => "?",
                };
                print!("{} ", token_str);
            }
            println!();
        }
        
        // Debug: Print possible_matches
        println!("\n=== possible_matches ===");
        for (tok_state, terminal_map) in &constraint.possible_matches {
            for (terminal_id, llm_tokens) in terminal_map {
                let token_ids: Vec<usize> = llm_tokens.iter_up_to(max_token_id).collect();
                if !token_ids.is_empty() {
                    println!("  State {} Terminal {} -> {} tokens: {:?}", 
                        tok_state.0, terminal_id.0, token_ids.len(), token_ids);
                }
            }
        }
        
        // Debug DWA
        println!("\n=== DWA info ===");
        let dwa = &constraint.parser_dwa;
        println!("DWA start state: {:?}", dwa.body.start_state);
        println!("DWA states count: {}", dwa.states.len());
        let dwa_start_state = &dwa.states[dwa.body.start_state];
        println!("Start state transitions:");
        for (label, target) in &dwa_start_state.transitions {
            println!("  Label {} -> target {:?}", label, target);
        }
        if let Some(fw) = &dwa_start_state.final_weight {
            let fw_tokens: Vec<usize> = fw.rsb().iter().take(20).collect();
            println!("Start state final_weight (first 20): {:?}", fw_tokens);
        } else {
            println!("Start state has no final_weight");
        }

        println!("Full DWA: {}", dwa);
        
        let state = constraint.init();
        
        // Debug: Print internal mask before conversion
        println!("\n=== Internal mask (before conversion to original) ===");
        let internal_mask = state.compute_internal_mask_debug();
        let internal_valid: Vec<usize> = internal_mask.iter_up_to(vocab.internal_max_llm_token + 1).collect();
        println!("Valid internal IDs: {:?}", internal_valid);
        
        // Show which original tokens each internal ID maps to
        for &int_id in &internal_valid {
            if let Some(originals) = vocab.internal_to_original.get(&int_id) {
                let orig_ids: Vec<usize> = originals.iter_up_to(max_token_id).collect();
                println!("  Internal {} -> original {:?}", int_id, orig_ids);
            }
        }
        
        let mask = state.get_mask();
        
        // Print all valid tokens
        println!("\nValid tokens at empty prefix:");
        let valid_tokens: Vec<usize> = mask.iter().collect();
        for &token_id in &valid_tokens {
            let token_str = match token_id {
                0 => "{",
                1 => "}",
                2 => "\"",
                3 => ":",
                4 => ",",
                5 => "n",
                6 => "a",
                7 => "m",
                8 => "e",
                9 => "s",
                10 => "t",
                11 => "r",
                12 => "i",
                13 => "g",
                14 => "{\"",
                15 => "\":",
                _ => "?",
            };
            println!("  {}: {} valid", token_id, token_str);
        }
        
        // At empty prefix, ONLY tokens starting with '{' should be valid:
        // - Token 0: `{` - YES
        // - Token 14: `{"` - YES
        // All others should be INVALID
        
        assert!(mask.contains(0), "Token 0 '{{' should be valid");
        assert!(mask.contains(14), "Token 14 '{{\"' should be valid");
        
        assert!(!mask.contains(1), "Token 1 '}}' should NOT be valid at empty prefix - BUG!");
        assert!(!mask.contains(2), "Token 2 '\"' should NOT be valid at empty prefix - BUG!");
        assert!(!mask.contains(3), "Token 3 ':' should NOT be valid at empty prefix - BUG!");
        assert!(!mask.contains(4), "Token 4 ',' should NOT be valid at empty prefix - BUG!");
        assert!(!mask.contains(15), "Token 15 '\":' should NOT be valid at empty prefix - BUG!");
        
        // Only 2 tokens should be valid
        let valid_count = valid_tokens.len();
        assert!(valid_count <= 2, 
            "Only 2 tokens starting with {{ should be valid at empty prefix, but got {}", valid_count);
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

    /// Test simple object schema with weight-heavy encoding
    /// This test verifies that weight-heavy mode (now the default) works correctly.
    /// Ignored: This test asserts weight-heavy is the default, fails when DISABLE_WEIGHT_HEAVY=1
    #[test]
    #[ignore]
    fn test_schema_simple_object_weight_heavy() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).expect("Schema should convert");
        println!("Generated EBNF:\n{}", ebnf);
        
        let gd = GrammarDefinition::from_ebnf(&ebnf).expect("Grammar should build");
        let token_map = simple_byte_token_map();
        let max_token_id = 255;
        
        // Create constraint (now defaults to weight-heavy)
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd.clone()),
            token_map.clone(),
            max_token_id,
            &GrammarConstraintConfig::default(),
        );
        assert!(constraint.is_weight_heavy(), "Should default to weight-heavy");
        println!("Weight-heavy: {} tsids, {} DWA states", 
            constraint.num_tsids, constraint.parser_dwa.states.len());
        
        // Test inputs work correctly in weight-heavy mode
        let inputs = &["{}",r#"{"name": "test"}"#, r#"{ "name" : "hello world" }"#];
        
        for input in inputs {
            println!("\nTesting input: {:?}", input);
            
            let mut state = constraint.init();
            
            for (i, ch) in input.bytes().enumerate() {
                let mask = state.get_mask();
                let is_valid = mask.contains(ch as usize);
                
                assert!(is_valid, 
                    "At position {}, char {:?} (byte {}) should be valid. mask: {:?}...",
                    i, ch as char, ch,
                    mask.iter().take(30).collect::<Vec<_>>()
                );
                
                state.commit(LLMTokenID(ch as usize)).expect("commit failed");
            }
            
            println!("✓ Input {:?} accepted in weight-heavy mode", input);
        }
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

    /// Simpler reproduction case with minimal vocab.
    #[test]
    fn test_schema_const2() {
        let schema = r#"{
            "const": "x"
        }"#;

        let mut token_map = LLMTokenMap::new();
        token_map.insert(vec![b'"'], LLMTokenID(b'"' as usize));
        token_map.insert(vec![b'x'], LLMTokenID(b'x' as usize));
        token_map.insert(vec![b' '], LLMTokenID(b' ' as usize));
        token_map.insert(vec![b'\n'], LLMTokenID(b'\n' as usize));
        token_map.insert(vec![b'\r'], LLMTokenID(b'\r' as usize));
        token_map.insert(vec![b'\t'], LLMTokenID(b'\t' as usize));

        test_schema_with_inputs_and_vocab(schema, &["\"x\""], token_map, 255);
    }
    
    /// Test that multi-byte tokens like `{"` work with simple object schema.
    /// Requires a valid GPT-2 vocab - will panic if not available.
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
    /// Requires a valid GPT-2 vocab - will panic if not available.
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
    
    /// Bug reproduction test: sep1 incorrectly allows `"` at empty prefix for object schema.
    /// 
    /// For an object schema like `{"type": "object", "properties": {"name": {"type": "string"}}}`,
    /// only tokens starting with `{` should be valid at the empty prefix (e.g., `{` and `{"`).
    /// The token `"` (which would start a string) should NOT be valid.
    /// 
    /// This test uses the actual GPT-2 vocabulary where:
    /// - Token 90: `{`
    /// - Token 1: `"`
    /// - Token 4895: `{"`
    #[test]
    fn test_object_schema_rejects_quote_at_empty_prefix() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        
        let (token_map, max_token_id) = gpt2_like_token_map()
            .expect("No valid GPT-2 vocab found! This test requires a real GPT-2 vocab with thousands of tokens. \
                     Try: wget -O benchmarking/gpt2_vocab.json https://huggingface.co/openai-community/gpt2/raw/main/vocab.json");
        
        let ebnf = json_schema_to_ebnf(schema)
            .expect("Schema should convert");
        println!("Generated EBNF:\n{}", ebnf);
        
        let gd = GrammarDefinition::from_ebnf(&ebnf)
            .expect("Grammar should build");
        
        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(gd),
            token_map.clone(),
            max_token_id,
            &GrammarConstraintConfig::default(),
        );
        
        // Get token IDs (these are the actual GPT-2 token IDs)
        let open_brace = b"{".to_vec();
        let quote = b"\"".to_vec();
        let open_brace_quote = b"{\"".to_vec();
        
        let open_brace_id = token_map.get(&open_brace);
        let quote_id = token_map.get(&quote);
        let open_brace_quote_id = token_map.get(&open_brace_quote);
        
        println!("Token IDs: {{ = {:?}, \" = {:?}, {{\" = {:?}", 
            open_brace_id, quote_id, open_brace_quote_id);
        
        // Get initial mask
        let state = constraint.init();
        let mask = state.get_mask();
        
        println!("Valid tokens at empty prefix: {:?}", mask.iter().take(100).collect::<Vec<_>>());
        
        // `{` should be valid (starts an object)
        if let Some(id) = open_brace_id {
            let is_valid = mask.contains(id.0);
            println!("Token '{{' (id={}): valid={}", id.0, is_valid);
            assert!(is_valid, "'{{' MUST be valid at empty prefix for object schema");
        }
        
        // `{"` should be valid (starts an object with a property key)
        if let Some(id) = open_brace_quote_id {
            let is_valid = mask.contains(id.0);
            println!("Token '{{\"' (id={}): valid={}", id.0, is_valid);
            assert!(is_valid, "'{{\"' MUST be valid at empty prefix for object schema with properties");
        }
        
        // `"` should NOT be valid (would start a string, not an object)
        // THIS IS THE BUG: sep1 currently allows this token when it shouldn't
        if let Some(id) = quote_id {
            let is_valid = mask.contains(id.0);
            println!("Token '\"' (id={}): valid={}", id.0, is_valid);
            assert!(!is_valid, 
                "'\"' MUST NOT be valid at empty prefix for object schema! \
                 An object must start with '{{', not '\"'. \
                 This is a known bug where sep1 incorrectly allows tokens \
                 that are prefixes of valid multi-byte tokens.");
        }
        
        // Count total valid tokens
        let total_valid: usize = mask.iter().count();
        println!("Total valid tokens: {}", total_valid);
        
        // Sanity check: there shouldn't be hundreds of valid tokens at the start of an object
        // In fact, with the actual GPT-2 vocab, only tokens that START with `{` should be valid
        assert!(total_valid < 100, 
            "Too many valid tokens ({}) at empty prefix for object schema. \
             Only tokens starting with '{{' should be valid.", total_valid);
    }
    
    // Tests moved from legacy module
    
    use crate::interface::json_schema::json_schema_to_grammar_exprs;
    
    #[test]
    fn test_conversion_simple_object() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).unwrap();
        assert!(ebnf.contains("root"));
        assert!(ebnf.contains("JSON_STRING"));
        assert!(ebnf.contains("JSON_INTEGER"));
    }

    #[test]
    fn test_conversion_any_of() {
        let schema = r#"{
            "anyOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        }"#;
        
        let rules = json_schema_to_grammar_exprs(schema).unwrap();
        // Should have root rule and alternatives
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_conversion_enum() {
        let schema = r#"{
            "enum": ["red", "green", "blue"]
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).unwrap();
        assert!(ebnf.contains("\"red\""));
        assert!(ebnf.contains("\"green\""));
        assert!(ebnf.contains("\"blue\""));
    }

    #[test]
    fn test_conversion_ref() {
        let schema = r##"{
            "$defs": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            },
            "type": "array",
            "items": {"$ref": "#/$defs/person"}
        }"##;
        
        let rules = json_schema_to_grammar_exprs(schema).unwrap();
        assert!(!rules.is_empty());
    }
}
