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

use crate::constraint::{GrammarConstraint, GrammarConstraintConfig, Precomputed2, Trie2GodWrapper};
use crate::interface::{CompiledGrammar, GrammarDefinition};
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use reqwest::blocking;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use bimap::BiBTreeMap;
use crate::constraint_extra::PrecomputeStats;
use crate::constraint_precompute2_utils::{are_precompute2_trees_equivalent, clone_trie2_graph, optimize_trie2_size};
use crate::json_serialization::JSONConvertible;
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

fn assert_trees_are_equivalent(
    original_precomputed2: &Precomputed2,
    original_god: &Trie2GodWrapper,
    optimized_precomputed2: &Precomputed2,
    optimized_god: &Trie2GodWrapper,
    original_to_internal_id_bimap: &BiBTreeMap<usize, usize>,
    llm_token_map: &LLMTokenMap,
) {
    println!("\n--- Final Stats Comparison ---");
    println!("\n--- Stats for Original Precompute2 Tree ---");
    let mut stats_original = PrecomputeStats::default();
    crate::constraint_extra::calculate_final_stats2(original_precomputed2, &mut stats_original, original_god);
    crate::constraint_extra::print_precompute_stats2(&stats_original, original_god);

    println!("\n--- Stats for Optimized Precompute2 Tree ---");
    let mut stats_optimized = PrecomputeStats::default();
    crate::constraint_extra::calculate_final_stats2(optimized_precomputed2, &mut stats_optimized, optimized_god);
    crate::constraint_extra::print_precompute_stats2(&stats_optimized, optimized_god);

    println!("\n--- Dumping Optimized Precompute2 Tree ---");
    GrammarConstraint::_dump_precomputed2(
        optimized_precomputed2,
        original_to_internal_id_bimap,
        llm_token_map,
        optimized_god,
    );
    println!("--- Finished Dumping Optimized Tree ---\n");

    // Compare the original and optimized trees for semantic equivalence
    assert_eq!(
        original_precomputed2.len(),
        optimized_precomputed2.len(),
        "Number of tokenizer states with trees differs."
    );

    for sid in original_precomputed2.keys() {
        let original_root = original_precomputed2.get(sid).unwrap();
        let optimized_root = optimized_precomputed2.get(sid).unwrap();
        if !are_precompute2_trees_equivalent(original_root, original_god, optimized_root, optimized_god) {
            // Detailed info is now printed inside are_precompute2_trees_equivalent
            panic!(
                "Optimized and original Precompute2 trees are not equivalent for tokenizer state ID: {}. See details above.",
                sid.0);
        }
    }
}

fn run_equivalence_test(ebnf: &str, llm_tokens: &[&str]) -> Result<(), Box<dyn Error>> {
    let compiled = compiled_from_ebnf_str(ebnf)?;
    let (llm_token_map, max_original_llm_token_id) = make_llm_token_map(llm_tokens);

    // Create with optimizations OFF to get the baseline
    let no_opt_config = GrammarConstraintConfig {
        optimize_trie2_prune_dead_paths: false,
        optimize_trie2_merge_nodes: false,
        optimize_trie2_factor_common_destinations: false,
        optimize_trie2_compress_edges: false,
        optimize_trie2_gc: true,
    };
    println!("\n--- Building baseline GrammarConstraint (optimizations OFF) ---");
    let gc_baseline = GrammarConstraint::from_compiled_grammar_with_config(
        compiled.clone(),
        llm_token_map.clone(),
        LLMTokenID(0), // dummy EOF placeholder
        max_original_llm_token_id,
        &no_opt_config,
    );

    // Create with optimizations ON
    let opt_config = GrammarConstraintConfig::default(); // Now defaults to ON
    println!("\n--- Building optimized GrammarConstraint (optimizations ON) ---");
    let gc_optimized = GrammarConstraint::from_compiled_grammar_with_config(
        compiled,
        llm_token_map.clone(),
        LLMTokenID(0), // dummy EOF placeholder
        max_original_llm_token_id,
        &opt_config,
    );

    assert_trees_are_equivalent(
        &gc_baseline.precomputed2,
        &gc_baseline.trie2_god,
        &gc_optimized.precomputed2,
        &gc_optimized.trie2_god,
        &gc_baseline.llm_vocab.original_to_internal_id_bimap,
        &llm_token_map,
    );
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

#[test]
fn test_precompute_optimizations_are_equivalent_for_js() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(rustrover) {
        println!("Skipping test_precompute2_optimizations_are_equivalent_for_js in rustrover mode.");
        return Ok(());
    }

    // --- Setup Phase ---
    println!("--- Setting up for Precompute2 Optimization Equivalence Test (JS) ---");

    // 1. Load and compile the JavaScript grammar.
    let grammar_path = "src/js_simplified2.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    println!("Compiling GrammarDefinition into CompiledGrammar...");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Successfully compiled GrammarDefinition.");
    println!("{}", compiled_grammar);

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

    // 3. Construct baseline GrammarConstraint (optimizations OFF)
    println!("\nConstructing baseline GrammarConstraint with optimizations OFF...");
    let no_opt_config = GrammarConstraintConfig {
        optimize_trie2_prune_dead_paths: false,
        optimize_trie2_merge_nodes: false,
        optimize_trie2_factor_common_destinations: false,
        optimize_trie2_compress_edges: false,
        optimize_trie2_gc: true,
    };
    let gc_baseline = GrammarConstraint::from_compiled_grammar_with_config(
        compiled_grammar.clone(),
        llm_token_map.clone(),
        LLMTokenID(0),
        max_original_llm_token_id_val,
        &no_opt_config,
    );

    // 4. Construct optimized GrammarConstraint (optimizations ON)
    println!("\nConstructing optimized GrammarConstraint with optimizations ON...");
    let opt_config = GrammarConstraintConfig::default();
    let gc_optimized = GrammarConstraint::from_compiled_grammar_with_config(
        compiled_grammar.clone(),
        llm_token_map.clone(),
        LLMTokenID(0),
        max_original_llm_token_id_val,
        &opt_config,
    );

    // 5. Assert equivalence
    assert_trees_are_equivalent(
        &gc_baseline.precomputed2,
        &gc_baseline.trie2_god,
        &gc_optimized.precomputed2,
        &gc_optimized.trie2_god,
        &gc_baseline.llm_vocab.original_to_internal_id_bimap,
        &llm_token_map,
    );

    println!("\nEquivalence test passed: All trees are semantically equivalent after optimization (JS).");

    Ok(())
}

//
// -------------------------------
// Compact tests using inline EBNF
// (based on grammars in test_constraint_basic.rs)
// -------------------------------
//

// Trivial: s -> A EOF; A -> 'a'; EOF -> '$'
#[test]
fn test_p_opt_trivial_a_eof() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= A EOF;
        A ::= 'a';
        EOF ::= '$';
    "#;
    let llm = ["a", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Simple choice: s -> x EOF; x -> A B_OR_C | AB
#[test]
fn test_p_opt_simple_choice_ab_or_ac() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= x EOF;
        x ::= A B_OR_C | AB;
        A ::= 'a';
        B_OR_C ::= 'b' | 'c';
        AB ::= 'ab';
        EOF ::= '$';
    "#;
    let llm = ["ab", "ac", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression grammar (e -> e + t | t; t -> t * f | f; f -> (e) | i)
#[test]
fn test_p_opt_expression_full() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= e PLUS t | t;
        t ::= t TIMES f | f;
        f ::= LPAREN e RPAREN | I;

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

// Expression (no '*'): e -> e + t | t; t -> f; f -> (e) | i
#[test]
fn test_p_opt_expression_no_times() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= e PLUS t | t;
        t ::= f;
        f ::= LPAREN e RPAREN | I;

        PLUS ::= '+';
        LPAREN ::= '(';
        RPAREN ::= ')';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "+", "(", ")", "(i", "+i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression (no parens): e -> e + t | t; t -> t * f | f; f -> i
#[test]
fn test_p_opt_expression_no_parens() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= e PLUS t | t;
        t ::= t TIMES f | f;
        f ::= I;

        PLUS ::= '+';
        TIMES ::= '*';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "+", "*", "+i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Expression (no '+' or '*'): e -> t; t -> f; f -> (e) | i
#[test]
fn test_p_opt_expression_no_plus_times() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= t;
        t ::= f;
        f ::= LPAREN e RPAREN | I;

        LPAREN ::= '(';
        RPAREN ::= ')';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "(", ")", "(i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Direct recursion: e -> '(' e | 'i'; s -> e EOF
#[test]
fn test_p_opt_expression_trivial_direct() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= LPAREN e | I;

        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "(", "(i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Same as above, but limited vocab: only "(i" (to exercise multi-grammar-token LLM token)
#[test]
fn test_p_opt_expression_trivial_direct_limited_vocab() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= LPAREN e | I;

        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["(i"];
    run_equivalence_test(ebnf, &llm)
}

// Unbalanced parens variant (f -> '(' e | 'i')
#[test]
fn test_p_opt_expression_unbalanced_parens() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= e EOF;
        e ::= t;
        t ::= f;
        f ::= LPAREN e | I;

        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#;
    let llm = ["i", "(", "(i", "$"];
    run_equivalence_test(ebnf, &llm)
}

// Indirect recursion simplified: s -> 'a' e | 'b'; e -> s
#[test]
fn test_p_opt_indirect_recursion_simplified() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= A e | B;
        e ::= s;

        A ::= 'a';
        B ::= 'b';
    "#;
    let llm = ["a", "b"];
    run_equivalence_test(ebnf, &llm)
}

// Repetition: s ::= A*, A ::= 'a'
#[test]
fn test_p_opt_repetition_a_star() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= A*;
        A ::= 'a';
    "#;
    let llm = ["a"];
    run_equivalence_test(ebnf, &llm)
}

// a+ token: s ::= A_PLUS; A_PLUS ::= 'a'+
#[test]
fn test_p_opt_a_plus_terminal() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= A_PLUS;
        A_PLUS ::= 'a'+;
    "#;
    let llm = ["a", "aaa"];
    run_equivalence_test(ebnf, &llm)
}

// Ignore whitespace: WS ignored between tokens. s ::= A B; A='a'; B='b'; WS=' '
// LLM tokens include "a", " ", "b", "a b"
#[test]
fn test_p_opt_ignore_whitespace() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        #![ignore(WS)]
        s ::= A B;
        A ::= 'a';
        B ::= 'b';
        WS ::= ' '+;
    "#;
    let llm = ["a", " ", "b", "a b"];
    run_equivalence_test(ebnf, &llm)
}

// 'x' SPACE+ '=' pattern (compact variant of x_eq)
#[test]
fn test_p_opt_x_space_equals() -> Result<(), Box<dyn Error>> {
    let ebnf = r#"
        s ::= X SPACE EQUALS;
        X ::= 'x';
        SPACE ::= ' '+;
        EQUALS ::= '=';
    "#;
    let llm = ["x", " ="];
    run_equivalence_test(ebnf, &llm)
}