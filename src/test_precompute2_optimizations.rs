//! Precompute2 optimization equivalence tests.
//!
//! This suite verifies that applying optimization passes to the Precompute2
//! trie graph preserves its semantics by comparing the optimized result
//! against an unoptimized baseline using are_precompute2_trees_equivalent.
//!
//! We keep tests compact by:
//! - Defining grammars via inline EBNF strings.
//! - Using small, explicit LLM vocabularies.
//! - Factoring common logic into small helpers.

use crate::constraint::{
    are_precompute2_trees_equivalent, clone_trie2_graph, context_aware_merge_trie2, GrammarConstraint, Precomputed2,
};
use crate::interface::{CompiledGrammar, GrammarDefinition};
use crate::json_serialization::JSONConvertible;
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use reqwest::blocking;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

//
// -------------------------------
// Common helpers
// -------------------------------
//

fn compiled_from_ebnf_str(ebnf: &str) -> Result<CompiledGrammar, Box<dyn Error>> {
    let def = GrammarDefinition::from_ebnf(ebnf)?;
    Ok(CompiledGrammar::from_definition(Arc::new(def)))
}

fn make_llm_token_map(tokens: &[&str]) -> (LLMTokenMap, usize) {
    let mut map = LLMTokenMap::new();
    let mut max_id = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        map.insert(t.as_bytes().to_vec(), LLMTokenID(i));
        max_id = max_id.max(i);
    }
    (map, max_id)
}

fn assert_optimized_equivalent(original_precomputed2: &Precomputed2) {
    // Deep clone the original precomputed2 tree(s)
    let mut optimized_precomputed2: Precomputed2 = BTreeMap::new();
    for (sid, root_arc) in original_precomputed2.iter() {
        let (cloned_root, _map) = clone_trie2_graph(root_arc);
        optimized_precomputed2.insert(*sid, cloned_root);
    }

    // Apply optimization passes
    context_aware_merge_trie2(&mut optimized_precomputed2);

    // Compare the original and optimized trees for semantic equivalence
    assert_eq!(
        original_precomputed2.len(),
        optimized_precomputed2.len(),
        "Number of tokenizer states with trees differs."
    );

    for sid in original_precomputed2.keys() {
        let original_root = original_precomputed2.get(sid).unwrap();
        let optimized_root = optimized_precomputed2.get(sid).unwrap();
        assert!(
            are_precompute2_trees_equivalent(original_root, optimized_root),
            "Mismatch found for tokenizer state ID: {}",
            sid.0
        );
    }
}

fn run_equivalence_test(ebnf: &str, llm_tokens: &[&str]) -> Result<(), Box<dyn Error>> {
    let compiled = compiled_from_ebnf_str(ebnf)?;
    let (llm_token_map, max_original_llm_token_id) = make_llm_token_map(llm_tokens);
    let gc = GrammarConstraint::from_compiled_grammar(
        compiled,
        llm_token_map,
        LLMTokenID(0), // dummy EOF placeholder
        max_original_llm_token_id,
    );
    assert_optimized_equivalent(&gc.precomputed2);
    Ok(())
}

//
// -------------------------------
// Specialized JS test (large grammar + GPT-2 subset vocab)
// -------------------------------
//

fn load_or_download_gpt2_vocab(
    cache_dir: &Path,
    file_name: &str,
    url: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    let cache_path = cache_dir.join(file_name);

    let vocab_map: BTreeMap<String, u32> = if cache_path.exists() {
        println!("Loading GPT-2 vocab from cache: {:?}", cache_path);
        let file = File::open(cache_path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    } else {
        println!("Downloading GPT-2 vocab from: {}", url);
        let response = blocking::get(url)?.error_for_status()?;
        let content = response.text()?;

        let mut file = File::create(&cache_path)?;
        file.write_all(content.as_bytes())?;
        println!("Saved GPT-2 vocab to cache: {:?}", cache_path);

        serde_json::from_str(&content)?
    };
    Ok(vocab_map.into_keys().collect())
}

#[cfg(not(rustrover))]
#[test]
fn test_precompute2_optimizations_are_equivalent_for_js() -> Result<(), Box<dyn std::error::Error>> {
    // --- Setup Phase ---
    println!("--- Setting up for Precompute2 Optimization Equivalence Test (JS) ---");

    const FORCE_RECOMPUTE: bool = false;
    const SAVE_TO_CACHE: bool = false;
    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use crate::json_serialization;

    let cache_dir = Path::new(".cache/test_precompute2");
    fs::create_dir_all(cache_dir)?;
    let precomputed2_cache_path = cache_dir.join("precomputed2_js_gpt2_small.json.gz");

    // 1. Load and compile the JavaScript grammar.
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    println!("Compiling GrammarDefinition into CompiledGrammar...");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Successfully compiled GrammarDefinition.");

    // 2. Load a small, representative vocabulary.
    println!("\nLoading GPT-2 vocabulary...");
    let vocab_cache_dir = Path::new(".cache/test_vocabs");
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let vocab_file_name = "gpt2_vocab.json";
    let mut gpt2_raw_vocab = load_or_download_gpt2_vocab(vocab_cache_dir, vocab_file_name, vocab_url)?;
    // Keep a smaller subset to speed up the test
    gpt2_raw_vocab.retain(|s| s.len() < 5);
    println!("Using a subset of {} tokens for the test.", gpt2_raw_vocab.len());

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;
    for (i, token_str) in gpt2_raw_vocab.iter().enumerate() {
        let id_val = i;
        let processed_token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        let token_bytes = processed_token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(id_val));
        max_original_llm_token_id_val = max_original_llm_token_id_val.max(id_val);
    }

    // 3. Construct the GrammarConstraint to get the baseline precomputed2 tree.
    let original_precomputed2: Precomputed2;

    if !FORCE_RECOMPUTE && precomputed2_cache_path.exists() {
        println!("\nLoading Precomputed2 from cache: {:?}", precomputed2_cache_path);
        let file = File::open(&precomputed2_cache_path)?;
        let decompressor = GzDecoder::new(BufReader::new(file));
        original_precomputed2 = Precomputed2::from_json_reader(decompressor)?;
        println!("Successfully loaded Precomputed2 from cache.");
    } else {
        println!("\nConstructing GrammarConstraint (will generate Precomputed2)...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar.clone(),
            llm_token_map.clone(),
            LLMTokenID(0), // dummy_eof_placeholder
            max_original_llm_token_id_val,
        );
        println!("GrammarConstraint constructed successfully.");
        original_precomputed2 = grammar_constraint.precomputed2;

        if SAVE_TO_CACHE {
            println!("\nSerializing and saving Precomputed2 to cache: {:?}", precomputed2_cache_path);
            let file = File::create(&precomputed2_cache_path)?;
            let writer = BufWriter::new(file);
            let mut encoder = GzEncoder::new(writer, Compression::default());
            json_serialization::write_precomputed2_to_stream(&original_precomputed2, &mut encoder)?;
            println!("Successfully saved Precomputed2 to cache.");
        }
    }

    // 4. Assert that the optimized result is equivalent to the baseline.
    assert_optimized_equivalent(&original_precomputed2);

    println!("\nEquivalence test passed: All trees are semantically equivalent after optimization (JS).");

    Ok(())
}

//
// -------------------------------
// Compact tests using inline EBNF
// (based on grammars in test_constraint_basic.rs)
// -------------------------------
//

// Trivial: S -> A EOF; A -> 'a'; EOF -> '$'
#[test]
fn test_p2_opt_trivial_a_eof() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= A EOF;
        A ::= 'a';
        EOF ::= '$';
    "#;
    let llm = ["a", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Simple choice: S -> X EOF; X -> A B_OR_C | AB
#[test]
fn test_p2_opt_simple_choice_ab_or_ac() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= X EOF;
        X ::= A B_OR_C | AB;
        A ::= 'a';
        B_OR_C ::= 'b' | 'c';
        AB ::= 'ab';
        EOF ::= '$';
    "#;
    let llm = ["ab", "ac", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression grammar (E -> E + T | T; T -> T * F | F; F -> (E) | i)
#[test]
fn test_p2_opt_expression_full() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= E PLUS T | T;
        T ::= T TIMES F | F;
        F ::= LPAREN E RPAREN | I;

        PLUS ::= '+';
        TIMES ::= '*';
        LPAREN ::= '(';
        RPAREN ::= ')';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "+", "*", "(", ")", "(i", "+i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression (no '*'): E -> E + T | T; T -> F; F -> (E) | i
#[test]
fn test_p2_opt_expression_no_times() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= E PLUS T | T;
        T ::= F;
        F ::= LPAREN E RPAREN | I;

        PLUS ::= '+';
        LPAREN ::= '(';
        RPAREN ::= ')';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "+", "(", ")", "(i", "+i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression (no parens): E -> E + T | T; T -> T * F | F; F -> i
#[test]
fn test_p2_opt_expression_no_parens() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= E PLUS T | T;
        T ::= T TIMES F | F;
        F ::= I;

        PLUS ::= '+';
        TIMES ::= '*';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "+", "*", "+i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression (no '+' or '*'): E -> T; T -> F; F -> (E) | i
#[test]
fn test_p2_opt_expression_no_plus_times() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= T;
        T ::= F;
        F ::= LPAREN E RPAREN | I;

        LPAREN ::= '(';
        RPAREN ::= ')';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "(", ")", "(i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Direct recursion: E -> '(' E | 'i'; S -> E EOF
#[test]
fn test_p2_opt_expression_trivial_direct() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= LPAREN E | I;

        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "(", "(i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Same as above, but limited vocab: only "(i" (to exercise multi-grammar-token LLM token)
#[test]
fn test_p2_opt_expression_trivial_direct_limited_vocab() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= LPAREN E | I;

        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["(i"];
    run_equivalence_test(ebnf, &llm)
}

// Unbalanced parens variant (F -> '(' E | 'i')
#[test]
fn test_p2_opt_expression_unbalanced_parens() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= E EOF;
        E ::= T;
        T ::= F;
        F ::= LPAREN E | I;

        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "(", "(i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Indirect recursion simplified: S -> 'a' E | 'b'; E -> S
#[test]
fn test_p2_opt_indirect_recursion_simplified() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= A E | B;
        E ::= S;

        A ::= 'a';
        B ::= 'b';
    "#;
    let llm = ["a", "b"];
    run_equivalence_test(ebnf, &llm)
}

// Repetition: S ::= A*, A ::= 'a'
#[test]
fn test_p2_opt_repetition_a_star() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= A*;
        A ::= 'a';
    "#;
    let llm = ["a"];
    run_equivalence_test(ebnf, &llm)
}

// a+ token: S ::= A_PLUS; A_PLUS ::= 'a'+
#[test]
fn test_p2_opt_a_plus_terminal() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= A_PLUS;
        A_PLUS ::= 'a'+;
    "#;
    let llm = ["a", "aaa"];
    run_equivalence_test(ebnf, &llm)
}

// Ignore whitespace: WS ignored between tokens. S ::= A B; A='a'; B='b'; WS=' '
// LLM tokens include "a", " ", "b", "a b"
#[test]
fn test_p2_opt_ignore_whitespace() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        #![ignore(WS)]
        S ::= A B;
        A ::= 'a';
        B ::= 'b';
        WS ::= ' '+;
    "#;
    let llm = ["a", " ", "b", "a b"];
    run_equivalence_test(ebnf, &llm)
}

// 'x' SPACE+ '=' pattern (compact variant of x_eq)
#[test]
fn test_p2_opt_x_space_equals() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        S ::= X SPACE EQUALS;
        X ::= 'x';
        SPACE ::= ' '+;
        EQUALS ::= '=';
    "#;
    let llm = ["x", " ="];
    run_equivalence_test(ebnf, &llm)
}
